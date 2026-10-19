import { test } from 'vitest';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { auditGltf } from '../scripts/audit-gltf.mjs';

const fixture = {
  asset: { version: '2.0' },
  accessors: [{ count: 4, componentType: 5126, type: 'VEC3' }, { count: 6, componentType: 5123, type: 'SCALAR' }],
  buffers: [{ byteLength: 60, uri: 'mesh.bin' }],
  meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 0 }, indices: 1, material: 0 }] }],
  materials: [{ alphaMode: 'MASK', pbrMetallicRoughness: { baseColorTexture: { index: 0, texCoord: 1 } }, normalTexture: { index: 0 } }],
  textures: [{ source: 0 }], images: [{ uri: 'color.png', mimeType: 'image/png' }],
  nodes: [{ children: [1, 2] }, { mesh: 0 }, { mesh: 0, scale: [-1, 1, 1] }],
  scenes: [{ nodes: [0] }], scene: 0,
};

test('audits unique accessor geometry separately from expanded instances and flags semantic blockers', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'audit-gltf-'));
  try {
    const path = join(dir, 'scene.gltf');
    await writeFile(path, JSON.stringify(fixture));
    await writeFile(join(dir, 'color.png'), Buffer.from([137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,64,0,0,0,32]));
    const result = await auditGltf(path);
    assert.equal(result.counts.uniqueAccessorVertices, 4);
    assert.equal(result.counts.primitiveTriangles, 2);
    assert.equal(result.scenes[0].expandedTriangles, 4);
    assert.deepEqual(result.scenes[0].negativeDeterminantMeshNodes, [2]);
    assert.deepEqual([result.images[0].width, result.images[0].height], [64, 32]);
    assert.deepEqual(result.blockers.map(x => x.code), ['GLTF_UNSUPPORTED_TEXTURE_COORDINATES', 'GLTF_UNSUPPORTED_REFLECTED_OR_SINGULAR_MESH_TRANSFORM']);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('reads only GLB JSON chunk and reports unsupported extension and alpha semantics', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'audit-glb-'));
  try {
    const json = { ...fixture, materials: [{ alphaMode: 'BLEND' }], images: [], textures: [], extensionsRequired: ['EXT_mesh_gpu_instancing'] };
    const body = Buffer.from(JSON.stringify(json));
    const padded = Buffer.alloc(Math.ceil(body.length / 4) * 4, 0x20); body.copy(padded);
    const glb = Buffer.alloc(20 + padded.length);
    glb.writeUInt32LE(0x46546c67, 0); glb.writeUInt32LE(2, 4); glb.writeUInt32LE(glb.length, 8);
    glb.writeUInt32LE(padded.length, 12); glb.writeUInt32LE(0x4e4f534a, 16); padded.copy(glb, 20);
    const path = join(dir, 'scene.glb'); await writeFile(path, glb);
    const result = await auditGltf(path);
    assert.equal(result.container, 'glb');
    assert.equal(result.containerBytes, glb.length);
    assert.equal(result.jsonBytes, padded.length);
    assert(result.blockers.some(x => x.code === 'GLTF_REQUIRED_EXTENSION'));
    assert(result.blockers.some(x => x.code === 'GLTF_UNSUPPORTED_ALPHA_MODE'));
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('deep scene traversal stays bounded and reports shared nodes only once per scene', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'audit-gltf-graph-'));
  try {
    const chain = Array.from({ length: 5000 }, (_, i) => ({ ...(i < 4999 ? { children: [i + 1] } : { mesh: 0 }) }));
    const json = { ...fixture, nodes: chain, scenes: [{ nodes: [0] }] };
    const path = join(dir, 'deep.gltf'); await writeFile(path, JSON.stringify(json));
    const deep = await auditGltf(path);
    assert.equal(deep.scenes[0].instances, 1);
    assert(deep.blockers.some(x => x.code === 'GLTF_ARRAY_LIMIT'));
    const shared = { ...fixture, nodes: [{ children: [1, 2] }, { children: [3] }, { children: [3] }, { mesh: 0 }], scenes: [{ nodes: [0] }, { nodes: [0] }] };
    await writeFile(path, JSON.stringify(shared));
    const report = await auditGltf(path);
    assert.deepEqual(report.scenes.map(x => x.instances), [1, 1]);
    assert.deepEqual(report.scenes.map(x => x.repeatedNodes), [[3], [3]]);
    assert.equal(report.blockers.filter(x => x.code === 'NODE_REPEATED_IN_SCENE').length, 2);
    await writeFile(path, JSON.stringify({ ...fixture, nodes: [{ children: [1] }, { children: [0] }], scenes: [] }));
    const cyclic = await auditGltf(path);
    assert(cyclic.blockers.some(x => x.code === 'NODE_NO_ROOTS'));
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('flags default vertex and index budgets without decoding buffers', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'audit-gltf-budget-'));
  try {
    const path = join(dir, 'large.gltf');
    const json = { ...fixture, accessors: [{ count: 1_000_001, componentType: 5126 }, { count: 3_000_003 }] };
    await writeFile(path, JSON.stringify(json));
    const result = await auditGltf(path);
    assert(result.blockers.some(x => x.code === 'GLTF_VERTEX_BUDGET'));
    assert(result.blockers.some(x => x.code === 'GLTF_INDICES_OR_BUDGET'));
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('checks COLOR_0 accessor declarations while explicitly leaving channel values unverified', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'audit-gltf-color-'));
  try {
    const path = join(dir, 'color.gltf');
    const make = color => ({ ...fixture,
      accessors: [...fixture.accessors, color],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 0, COLOR_0: 2 }, indices: 1 }] }],
      materials: [], textures: [], images: [], nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }] });
    for (const color of [
      { count: 4, type: 'VEC3', componentType: 5126 },
      { count: 4, type: 'VEC4', componentType: 5121, normalized: true },
      { count: 4, type: 'VEC3', componentType: 5123, normalized: true },
    ]) {
      await writeFile(path, JSON.stringify(make(color)));
      const result = await auditGltf(path);
      assert.equal(result.counts.color0Primitives, 1);
      assert.equal(result.counts.color0DeclaredVertices, 4);
      assert.equal(result.primitives[0].color0.valuesRead, false);
      assert.equal(result.primitives[0].color0.valuesValidated, false);
      assert(result.limitations[0].includes('COLOR_0'));
      assert(!result.blockers.some(x => x.code === 'GLTF_COLOR_LAYOUT'));
    }
    for (const color of [
      { count: 3, type: 'VEC4', componentType: 5126 },
      { count: 4, type: 'VEC2', componentType: 5126 },
      { count: 4, type: 'VEC4', componentType: 5121, normalized: false },
      { count: 4, type: 'VEC3', componentType: 5126, normalized: true },
      { count: 4, type: 'VEC3', componentType: 5126, normalized: 'false' },
      { count: 4, type: 'VEC3', componentType: 5126, normalized: null },
      { count: 4, type: 'VEC3', componentType: 5122, normalized: true },
    ]) {
      await writeFile(path, JSON.stringify(make(color)));
      const result = await auditGltf(path);
      assert.equal(result.assessment, 'blocked');
      assert(result.blockers.some(x => x.code === 'GLTF_COLOR_LAYOUT'));
    }
  } finally { await rm(dir, { recursive: true, force: true }); }
});
