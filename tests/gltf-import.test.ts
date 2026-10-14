import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { importGltf } from '../src/gltf-import';
import { GltfSceneInstance } from '../src/gltf-scene';
import type { GltfBinaryDocument } from '../src/gltf-binary';
import { transformPoint, translation } from '../src/math';

type FixtureName = 'SimpleSkin' | 'AnimatedMorphCube';
interface FixtureDocument extends GltfBinaryDocument {
  nodes: Array<{ children?: number[]; mesh?: number; skin?: number; translation?: number[]; scale?: number[] }>;
  meshes: Array<{ primitives: Array<{ attributes: Record<string, number>; targets?: Array<Record<string, number>> }> }>;
  skins?: Array<{ joints: number[] }>;
  animations?: Array<Record<string, unknown>>;
  extensionsRequired?: string[];
  materials?: Array<Record<string, unknown>>;
}

function fixture(name: FixtureName): { json: FixtureDocument; buffers: ArrayBuffer[] } {
  const directory = new URL(`../public/assets/gltf/${name}/glTF/`, import.meta.url);
  const json = JSON.parse(readFileSync(new URL(`${name}.gltf`, directory), 'utf8')) as FixtureDocument;
  const buffers = json.buffers!.map(buffer => Uint8Array.from(readFileSync(new URL(buffer.uri!, directory))).buffer);
  return { json, buffers };
}

function floatsAt(buffer: ArrayBuffer, offset: number, count: number): number[] {
  const view = new DataView(buffer);
  return Array.from({ length: count }, (_, i) => view.getFloat32(offset + i * 4, true));
}

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], Math.round(-Math.log10(tolerance)));
}

describe('Khronos glTF 2.0 reference assets', () => {
  it('imports SimpleSkin with its source indices, weighted joints, and bind matrices', () => {
    const { json, buffers } = fixture('SimpleSkin');
    const asset = importGltf(json, buffers);
    const primitive = asset.meshes[0].primitives[0];
    expect(asset.nodes).toHaveLength(3);
    expect(asset.meshes).toHaveLength(1);
    expect(asset.skins).toHaveLength(1);
    expect(asset.skins[0].joints).toEqual([1, 2]);
    expect(primitive.mesh.indices).toHaveLength(24); // Eight source triangles.
    expect(primitive.mesh.vertexCount).toBe(24); // Missing source normals require flat expansion.
    expect(Array.from(primitive.mesh.indices)).toEqual(Array.from({ length: 24 }, (_, i) => i));
    expect(primitive.hasSkin).toBe(true);
    expect(primitive.morphTargets).toHaveLength(0);
    expect(primitive.material.textures?.filter(Boolean)).toHaveLength(0);
    const vertices = new DataView(primitive.mesh.vertices);
    // Source vertex 9 is fully controlled by joint 1 and starts at (0.5, 2, 0).
    const sourceIndices = new DataView(buffers[0]);
    const expandedSlot = Array.from({ length: 24 }, (_, i) => sourceIndices.getUint16(i * 2, true)).indexOf(9);
    expect(expandedSlot).toBeGreaterThanOrEqual(0);
    const base = expandedSlot * 80;
    close([0, 1, 2].map(axis => vertices.getFloat32(base + axis * 4, true)), [0.5, 2, 0]);
    expect(vertices.getUint32(base + 32, true)).toBe(0);
    expect(vertices.getUint32(base + 36, true)).toBe(1);
    close([vertices.getFloat32(base + 48, true), vertices.getFloat32(base + 52, true)], [0, 1]);
    close(asset.skins[0].inverseBindMatrices[1].slice(12, 15), [0, -1, 0]);
    expect(asset.rig.clips).toHaveLength(1);
    expect(asset.rig.clips[0].tracks[0].times).toHaveLength(12);
    // At source key t=1, the child joint has rotated 90 degrees about Z.
    const pose = asset.rig.sample(1);
    close(transformPoint(pose.world[2], [0.5, 0, 0]), [0, 1.5, 0], 2e-3);
  });

  it('imports AnimatedMorphCube with both target streams and source weight keys', () => {
    const { json, buffers } = fixture('AnimatedMorphCube');
    const asset = importGltf(json, buffers);
    const primitive = asset.meshes[0].primitives[0];
    expect(asset.nodes).toHaveLength(1);
    expect(primitive.mesh.vertexCount).toBe(24);
    expect(primitive.mesh.indices).toHaveLength(36); // Twelve source triangles.
    expect(primitive.morphTargets).toHaveLength(2);
    expect(primitive.hasSkin).toBe(false);
    expect(primitive.material.textures?.filter(Boolean)).toHaveLength(0);
    for (let target = 0; target < 2; target++) {
      const sourceAccessor = json.accessors![target === 0 ? 4 : 7];
      const sourceView = json.bufferViews![sourceAccessor.bufferView!];
      const source = floatsAt(buffers[sourceView.buffer], (sourceView.byteOffset ?? 0) + (sourceAccessor.byteOffset ?? 0), 24 * 3);
      close(primitive.morphTargets[target].positions, source);
      expect(primitive.morphTargets[target].normals).toHaveLength(24 * 3);
      expect(primitive.morphTargets[target].tangents).toHaveLength(24 * 3);
    }
    const clip = asset.rig.clips[0];
    expect(clip.tracks[0].path).toBe('weights');
    expect(clip.tracks[0].times).toHaveLength(127);
    const track = clip.tracks[0];
    const key = 15;
    const sourceTime = floatsAt(buffers[0], json.bufferViews![10].byteOffset!, 127)[key];
    const sourceWeights = floatsAt(buffers[0], json.bufferViews![11].byteOffset! + key * 2 * 4, 2);
    close(asset.rig.sample(sourceTime).weights[0], sourceWeights);
    const nextTime = track.times[key + 1];
    const nextWeights = floatsAt(buffers[0], json.bufferViews![11].byteOffset! + (key + 1) * 2 * 4, 2);
    const halfway = asset.rig.sample((sourceTime + nextTime) / 2).weights[0];
    close(halfway, sourceWeights.map((value, i) => (value + nextWeights[i]) / 2));
  });

  it('rejects corrupt hierarchy, joint references, and required extensions explicitly', () => {
    const { json, buffers } = fixture('SimpleSkin');
    const cycle = structuredClone(json);
    cycle.nodes[2].children = [1];
    expect(() => importGltf(cycle, buffers)).toThrow(/cycle/i);
    const badJoint = structuredClone(json);
    badJoint.skins![0].joints[1] = 99;
    expect(() => importGltf(badJoint, buffers)).toThrow(/GLTF_INDEX.*skin.joint/);
    const extension = structuredClone(json);
    extension.extensionsRequired = ['KHR_draco_mesh_compression'];
    expect(() => importGltf(extension, buffers)).toThrow(/GLTF_REQUIRED_EXTENSION.*KHR_draco_mesh_compression/);
  });

  it('does not silently drop an unsupported material extension', () => {
    const { json, buffers } = fixture('AnimatedMorphCube');
    const altered = structuredClone(json);
    altered.materials![0].extensions = { KHR_materials_clearcoat: { clearcoatFactor: 1 } };
    expect(() => importGltf(altered, buffers)).toThrow(/GLTF_UNSUPPORTED_MATERIAL_EXTENSION.*KHR_materials_clearcoat/);
  });

  it('binds SimpleSkin to an instance palette and keeps independent animation states', () => {
    const { json, buffers } = fixture('SimpleSkin');
    const asset = importGltf(json, buffers);
    const first = new GltfSceneInstance(asset, { firstEntityId: 40 });
    const second = new GltfSceneInstance(asset, { firstEntityId: 50, transform: translation(10, 0, 0) });
    first.sample(0);
    second.sample(1);
    expect(first.batches).toHaveLength(1);
    expect(second.batches).toHaveLength(1);
    const a = first.batches[0].instances[0], b = second.batches[0].instances[0];
    expect(a.entityId).toBe(40);
    expect(b.entityId).toBe(50);
    expect(a.bones).toHaveLength(2);
    expect(b.bones).toHaveLength(2);
    // Mesh-space joint palette is jointWorld * inverseBind; joint 1 pivots at y=1.
    close(transformPoint(a.bones[1], [0.5, 2, 0]), [0.5, 2, 0]);
    close(transformPoint(b.bones[1], [0.5, 2, 0]), [-1, 1.5, 0], 2e-3);
    close(transformPoint(b.model, [0, 0, 0]), [10, 0, 0]);
    const oldA = a;
    first.sample(1);
    expect(first.batches[0].instances[0]).not.toBe(oldA); // Committed input replaces render snapshots.
    close(transformPoint(second.batches[0].instances[0].bones[1], [0.5, 2, 0]), [-1, 1.5, 0], 2e-3);
    const history = first.batches[0].instances[0].historyVersion;
    first.sample(1.5, 0, false, true);
    expect(first.batches[0].instances[0].historyVersion).toBe(history + 1);
  });

  it('reuses committed states for an asset without clips while preserving validation and history invalidation', () => {
    const { json, buffers } = fixture('AnimatedMorphCube');
    delete json.animations;
    const scene = new GltfSceneInstance(importGltf(json, buffers));
    const initial = scene.batches.map(batch => batch.instances[0]);
    scene.sample(10);
    scene.sample(-20, 0, true);
    expect(scene.batches.map(batch => batch.instances[0])).toEqual(initial);
    for (let i = 0; i < initial.length; i++) expect(scene.batches[i].instances[0]).toBe(initial[i]);
    expect(() => scene.sample(Number.NaN)).toThrow(/sample time/);
    expect(() => scene.sample(1, 1)).toThrow(/clip index/);
    expect(() => scene.sample(1, -1)).toThrow(/clip index/);
    for (let i = 0; i < initial.length; i++) expect(scene.batches[i].instances[0]).toBe(initial[i]);
    scene.sample(30, undefined, false, true);
    for (let i = 0; i < initial.length; i++) {
      expect(scene.batches[i].instances[0]).not.toBe(initial[i]);
      expect(scene.batches[i].instances[0].historyVersion).toBe(initial[i].historyVersion + 1);
    }
    const invalidated = scene.batches.map(batch => batch.instances[0]);
    scene.sample(40);
    for (let i = 0; i < invalidated.length; i++) expect(scene.batches[i].instances[0]).toBe(invalidated[i]);
  });

  it('does not treat an omitted clip index as a static pose when clips exist', () => {
    const { json, buffers } = fixture('SimpleSkin');
    const scene = new GltfSceneInstance(importGltf(json, buffers));
    const initial = scene.batches[0].instances[0];
    scene.sample(1);
    const animated = scene.batches[0].instances[0];
    expect(animated).not.toBe(initial);
    scene.sample(0);
    const returned = scene.batches[0].instances[0];
    expect(returned).not.toBe(animated);
    close(transformPoint(returned.bones[1], [0.5, 2, 0]), [0.5, 2, 0]);
  });

  it('cancels mesh-node translation in the skin palette before applying the model matrix', () => {
    const { json, buffers } = fixture('SimpleSkin');
    json.nodes[0].translation = [3, 0, 0];
    const scene = new GltfSceneInstance(importGltf(json, buffers));
    scene.sample(1);
    const state = scene.batches[0].instances[0];
    const local = transformPoint(state.bones[1], [0.5, 2, 0]);
    close(local, [-4, 1.5, 0], 2e-3);
    close(transformPoint(state.model, local), [-1, 1.5, 0], 2e-3);
  });

  it('rejects reflected and singular mesh transforms before binding render state', () => {
    const { json, buffers } = fixture('SimpleSkin');
    for (const scale of [[-1, 1, 1], [0, 1, 1]]) {
      const altered = structuredClone(json);
      altered.nodes[0].scale = scale;
      expect(() => new GltfSceneInstance(importGltf(altered, buffers))).toThrow(/singular|reflected/i);
    }
  });

  it('ignores tangent streams when source normals are absent', () => {
    const { json, buffers } = fixture('AnimatedMorphCube');
    const altered = structuredClone(json);
    const primitive = altered.meshes[0].primitives[0];
    delete primitive.attributes.NORMAL;
    for (const target of primitive.targets ?? []) delete target.NORMAL;
    const asset = importGltf(altered, buffers);
    expect(asset.meshes[0].primitives[0].mesh.tangents).toBeUndefined();
    expect(asset.meshes[0].primitives[0].morphTargets.every(target => target.tangents === undefined)).toBe(true);
    expect(asset.warnings.filter(warning => warning.includes('tangents are ignored'))).toHaveLength(1);
  });

  it('rejects node morph weights without a mesh', () => {
    expect(() => importGltf({ asset: { version: '2.0' }, nodes: [{ weights: [0] }] }, []))
      .toThrow(/GLTF_NODE_WEIGHTS_REQUIRES_MESH/);
  });

  it('budgets repeated animation accessor channels before decoding', () => {
    const bytes = new Float32Array([0, 1, 0, 0, 0, 1, 2, 3]).buffer;
    const json = { asset: { version: '2.0' }, buffers: [{ byteLength: 32 }],
      bufferViews: [{ buffer: 0, byteLength: 8 }, { buffer: 0, byteOffset: 8, byteLength: 24 }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      nodes: [{}, {}],
      animations: [{ samplers: [{ input: 0, output: 1 }], channels: [
        { sampler: 0, target: { node: 0, path: 'translation' } },
        { sampler: 0, target: { node: 1, path: 'translation' } },
      ] }],
    };
    expect(() => importGltf(json, [], { maxAnimationScalars: 15 })).toThrow(/GLTF_ANIMATION_SCALAR_BUDGET/);
    expect(() => importGltf(json, [], { maxAnimationTracks: 1 })).toThrow(/GLTF_ANIMATION_TRACK_BUDGET/);
    expect(importGltf(json, [bytes], { maxAnimationScalars: 16, maxAnimationTracks: 2 }).rig.clips[0].tracks)
      .toHaveLength(2);
    expect(() => importGltf(json, [bytes], { maxAnimationScalars: 16_000_001 }))
      .toThrow(/GLTF_ANIMATION_BUDGET_OPTION/);
  });
});
