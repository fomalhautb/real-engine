import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

type Json = Record<string, any>;
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
const node = (script: string, args: string[]): string => execFileSync(process.execPath, [resolve('scripts', script), ...args], {
  cwd: resolve('.'), windowsHide: true, timeout: 120_000, maxBuffer: 16 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'],
}).toString();
const COMPONENT_BYTES: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

/** Independent accessor element reader (bytes exactly as stored). */
function element(doc: Json, bin: Buffer, accessorIndex: number, i: number): Buffer {
  const accessor = doc.accessors[accessorIndex], view = doc.bufferViews[accessor.bufferView];
  const size = COMPONENT_BYTES[accessor.componentType] * COMPONENTS[accessor.type], stride = view.byteStride ?? size;
  expect(i).toBeLessThan(accessor.count);
  const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0) + i * stride;
  expect(start + size).toBeLessThanOrEqual((view.byteOffset ?? 0) + view.byteLength);
  return bin.subarray(start, start + size);
}
function indexValue(doc: Json, bin: Buffer, accessorIndex: number, i: number): number {
  const bytes = element(doc, bin, accessorIndex, i), type = doc.accessors[accessorIndex].componentType;
  return type === 5121 ? bytes.readUInt8(0) : type === 5123 ? bytes.readUInt16LE(0) : bytes.readUInt32LE(0);
}
function floats(doc: Json, bin: Buffer, accessorIndex: number, i: number): number[] {
  const bytes = element(doc, bin, accessorIndex, i);
  return Array.from({ length: bytes.length / 4 }, (_, c) => bytes.readFloatLE(c * 4));
}
const uint32s = (bytes: Buffer): number[] => Array.from({ length: bytes.length / 4 }, (_, i) => bytes.readUInt32LE(i * 4));

/**
 * mesh 0: a 20x20 grid (800 triangles + one exactly degenerate, 441 vertices + one
 * unreferenced) with interleaved POSITION/NORMAL, float TANGENT, normalized uint16
 * TEXCOORD_0, normalized uint8 VEC3 COLOR_0 (stride 4), float TEXCOORD_1 with min/max
 * and uint16 indices; plus a quad. mesh 1: a quad sharing the quad's POSITION/NORMAL.
 */
function scene(): { json: Json; binary: Buffer } {
  const chunks: Buffer[] = [], views: Json[] = [], accessors: Json[] = [];
  let length = 0;
  const view = (data: Buffer, extra: Json = {}): number => {
    const padding = (4 - length % 4) % 4;
    if (padding) { chunks.push(Buffer.alloc(padding, 0xcd)); length += padding; }
    views.push({ buffer: 0, byteOffset: length, byteLength: data.length, ...extra });
    chunks.push(data); length += data.length;
    return views.length - 1;
  };
  const accessor = (definition: Json): number => { accessors.push(definition); return accessors.length - 1; };
  const n = 20, gridVertices = (n + 1) * (n + 1) + 1;
  const interleaved = Buffer.alloc(gridVertices * 24), tangents = Buffer.alloc(gridVertices * 16);
  const uv0 = Buffer.alloc(gridVertices * 4), colors = Buffer.alloc(gridVertices * 4, 0xee), uv1 = Buffer.alloc(gridVertices * 8);
  for (let v = 0; v < gridVertices; v++) {
    const x = v % (n + 1), y = Math.min(Math.floor(v / (n + 1)), n);
    const position = v < gridVertices - 1 ? [x * 0.5, y * 0.25, ((x * y) % 5) * 0.01] : [99, 99, 99];
    position.forEach((value, c) => interleaved.writeFloatLE(value, v * 24 + c * 4));
    [0, 0, 1].forEach((value, c) => interleaved.writeFloatLE(value, v * 24 + 12 + c * 4));
    [1, 0, 0, v % 3 ? 1 : -1].forEach((value, c) => tangents.writeFloatLE(value, v * 16 + c * 4));
    uv0.writeUInt16LE(Math.round(x / n * 65535), v * 4); uv0.writeUInt16LE(Math.round(y / n * 65535), v * 4 + 2);
    [x * 12 % 256, y * 12 % 256, (x + y) * 6 % 256].forEach((value, c) => colors.writeUInt8(value, v * 4 + c));
    uv1.writeFloatLE(x * 0.125 - 1, v * 8); uv1.writeFloatLE(y * -0.5, v * 8 + 4);
  }
  const gridIndices: number[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const a = y * (n + 1) + x, b = a + 1, c = a + n + 1, d = c + 1;
    gridIndices.push(a, b, d, a, d, c);
  }
  gridIndices.splice(300, 0, 5, 5, 6); // an exactly degenerate triangle inside the list
  const quadInterleaved = Buffer.alloc(4 * 24);
  [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]].forEach((p, v) => {
    p.forEach((value, c) => quadInterleaved.writeFloatLE(value, v * 24 + c * 4));
    [0, 0, 1].forEach((value, c) => quadInterleaved.writeFloatLE(value, v * 24 + 12 + c * 4));
  });
  const quadUv = Buffer.alloc(4 * 8);
  [0, 0, 1, 0, 1, 1, 0, 1].forEach((value, i) => quadUv.writeFloatLE(value, i * 4));
  const indexData = (values: number[]) => { const data = Buffer.alloc(values.length * 2); values.forEach((value, i) => data.writeUInt16LE(value, i * 2)); return data; };
  const gridIndexBytes = indexData(gridIndices), quadIndexBytes = indexData([0, 1, 2, 0, 2, 3]), sharedIndexBytes = indexData([0, 2, 1, 0, 3, 2]);
  const indicesView = view(Buffer.concat([gridIndexBytes, quadIndexBytes, sharedIndexBytes]), { target: 34963 });
  const gridView = view(interleaved, { byteStride: 24, target: 34962 });
  const grid = { attributes: {
    POSITION: accessor({ bufferView: gridView, componentType: 5126, count: gridVertices, type: 'VEC3', min: [0, 0, 0], max: [99, 99, 99], name: 'positions' }),
    NORMAL: accessor({ bufferView: gridView, byteOffset: 12, componentType: 5126, count: gridVertices, type: 'VEC3' }),
    TANGENT: accessor({ bufferView: view(tangents), componentType: 5126, count: gridVertices, type: 'VEC4' }),
    TEXCOORD_0: accessor({ bufferView: view(uv0), componentType: 5123, normalized: true, count: gridVertices, type: 'VEC2' }),
    COLOR_0: accessor({ bufferView: view(colors, { byteStride: 4 }), componentType: 5121, normalized: true, count: gridVertices, type: 'VEC3' }),
    TEXCOORD_1: accessor({ bufferView: view(uv1), componentType: 5126, count: gridVertices, type: 'VEC2', min: [-1, -10], max: [11.5, 0], extras: { kept: true } }),
  }, indices: accessor({ bufferView: indicesView, componentType: 5123, count: gridIndices.length, type: 'SCALAR' }), material: 0, mode: 4 };
  const quadView = view(quadInterleaved, { byteStride: 24 });
  const quadPosition = accessor({ bufferView: quadView, componentType: 5126, count: 4, type: 'VEC3', min: [0, 0, 1], max: [1, 1, 1] });
  const quadNormal = accessor({ bufferView: quadView, byteOffset: 12, componentType: 5126, count: 4, type: 'VEC3' });
  const quad = { attributes: { POSITION: quadPosition, NORMAL: quadNormal,
    TEXCOORD_0: accessor({ bufferView: view(quadUv), componentType: 5126, count: 4, type: 'VEC2' }) },
  indices: accessor({ bufferView: indicesView, byteOffset: gridIndexBytes.length, componentType: 5123, count: 6, type: 'SCALAR' }),
  material: 1, extras: { primitive: 'quad' } };
  const shared = { attributes: { POSITION: quadPosition, NORMAL: quadNormal },
    indices: accessor({ bufferView: indicesView, byteOffset: gridIndexBytes.length + quadIndexBytes.length, componentType: 5123, count: 6, type: 'SCALAR' }) };
  const binary = Buffer.concat(chunks);
  const json = { asset: { version: '2.0', generator: 'split test', copyright: 'test fixture' }, extras: { note: 'kept' },
    scene: 0, scenes: [{ nodes: [0, 1], name: 'main' }],
    nodes: [{ mesh: 0, translation: [1, 2, 3], rotation: [0, 0.38268343, 0, 0.92387953], scale: [2, 2, 2], name: 'grid' },
      { mesh: 1, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1], children: [2] }, { mesh: 0, translation: [0, 1, 0] }],
    meshes: [{ name: 'grid', primitives: [grid, quad], extras: { mesh: 0 } }, { name: 'shared', primitives: [shared] }],
    materials: [{ name: 'leaf', pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0 }, doubleSided: true, extras: { k: 1 } },
      { name: 'plain', pbrMetallicRoughness: { baseColorFactor: [0.5, 0.25, 0, 1] } }],
    textures: [{ source: 0, sampler: 0 }], images: [{ uri: './textures/base%20color.png' }], samplers: [{ magFilter: 9729, minFilter: 9987 }],
    accessors, bufferViews: views, buffers: [{ uri: 'scene.bin', byteLength: binary.length }] };
  return { json, binary };
}

async function withRoot(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(resolve('.'), '.split-gltf-primitives-test-'));
  try { await body(root); } finally {
    if (dirname(root) !== resolve('.') || !basename(root).startsWith('.split-gltf-primitives-test-')) throw Error('refusing to remove');
    await rm(root, { recursive: true, force: true });
  }
}

async function writeScene(root: string, json: Json, binary: Buffer, name = 'scene'): Promise<string> {
  const folder = join(root, name);
  await mkdir(join(folder, 'textures'), { recursive: true });
  await writeFile(join(folder, 'scene.gltf'), JSON.stringify(json));
  await writeFile(join(folder, 'scene.bin'), binary);
  await writeFile(join(folder, 'textures', 'base color.png'), Buffer.from('not decoded by the splitter'));
  await writeFile(join(folder, 'credits_license.txt'), 'license text\n');
  return join(folder, 'scene.gltf');
}

async function listFiles(folder: string, prefix = ''): Promise<string[]> {
  const result: string[] = [];
  for (const entry of await readdir(folder, { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) result.push(...await listFiles(join(folder, entry.name), name)); else result.push(name);
  }
  return result.sort();
}

describe('split-gltf-primitives', () => {
  it('splits an oversized primitive exactly, keeps everything else and is accepted by the geometry builder', async () => {
    await withRoot(async root => {
      const { json, binary } = scene();
      const input = await writeScene(root, json, binary);
      const output = join(root, 'split');
      const args = ['--input', input, '--output', output, '--max-triangles', '120', '--max-vertices', '100',
        '--license', join(root, 'scene', 'credits_license.txt')];
      const report = JSON.parse(node('split-gltf-primitives.mjs', args).trim());
      expect(report.totals.splitPrimitives).toBe(1);
      const manifest = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'));
      expect(manifest.format).toBe('real-engine-gltf-primitive-split');
      expect(manifest.source).toMatchObject({ file: 'scene.gltf', sha256: sha256(Buffer.from(JSON.stringify(json))),
        buffers: [{ index: 0, uri: 'scene.bin', bytes: binary.length, sha256: sha256(binary) }] });
      expect(manifest.options).toEqual({ maxTriangles: 120, maxVertices: 100, selection: 'exceeding-limits', primitives: null });
      const out: Json = JSON.parse(await readFile(join(output, 'scene.split.gltf'), 'utf8'));
      const bin = await readFile(join(output, 'scene.split.bin'));
      expect(out.buffers).toEqual([{ uri: 'scene.split.bin', byteLength: bin.length }]);

      // Everything outside the mesh primitives and binary layout is unchanged.
      for (const key of ['asset', 'extras', 'scene', 'scenes', 'nodes', 'materials', 'textures', 'images', 'samplers']) expect(out[key]).toEqual(json[key]);
      expect(Object.keys(out)).toEqual(Object.keys(json));
      expect(out.meshes.map((mesh: Json) => mesh.name)).toEqual(['grid', 'shared']);
      expect(out.meshes[0].extras).toEqual({ mesh: 0 });

      const [gridRecord, quadRecord, sharedRecord] = manifest.sourcePrimitives;
      expect(gridRecord).toMatchObject({ meshIndex: 0, primitiveIndex: 0, split: true, exceedsLimits: true, triangles: 801, vertices: 442,
        referencedVertices: 441, unreferencedVertices: 1, coverage: { sourceTriangles: 801, coveredTriangles: 801, exactlyOnce: true } });
      expect(quadRecord).toMatchObject({ split: false, exceedsLimits: false });
      expect(sharedRecord).toMatchObject({ split: false, outputPrimitives: { first: 0, count: 1 } });
      const parts = gridRecord.parts as number;
      expect(parts).toBeGreaterThanOrEqual(7);
      expect(out.meshes[0].primitives).toHaveLength(parts + 1);
      expect(gridRecord.duplicatedVertices).toBeGreaterThan(0);
      expect(gridRecord.partVertices.total).toBe(441 + gridRecord.duplicatedVertices);

      // Split parts: limits, provenance order, exact attribute bytes, exact triangles, recomputed ranges.
      const source = json, sourceBin = binary, grid = source.meshes[0].primitives[0];
      const covered = new Uint8Array(801);
      const entries = manifest.primitives.filter((p: Json) => p.split);
      expect(entries).toHaveLength(parts);
      for (const [k, entry] of entries.entries()) {
        expect(entry).toMatchObject({ meshIndex: 0, primitiveIndex: k, source: { meshIndex: 0, primitiveIndex: 0 }, part: k, parts, materialIndex: 0 });
        expect(entry.triangleCount).toBeLessThanOrEqual(120);
        expect(entry.vertexCount).toBeLessThanOrEqual(100);
        const triangleBytes = await readFile(join(output, entry.provenance.triangles.file));
        const vertexBytes = await readFile(join(output, entry.provenance.vertices.file));
        expect(sha256(triangleBytes)).toBe(entry.provenance.triangles.sha256);
        expect(sha256(vertexBytes)).toBe(entry.provenance.vertices.sha256);
        const triangles = uint32s(triangleBytes), vertices = uint32s(vertexBytes);
        expect(triangles).toHaveLength(entry.triangleCount);
        expect(vertices).toHaveLength(entry.vertexCount);
        expect(triangles).toEqual([...triangles].sort((a, b) => a - b));
        expect(vertices).toEqual([...vertices].sort((a, b) => a - b));
        for (const t of triangles) covered[t]++;
        const primitive = out.meshes[0].primitives[k];
        expect(primitive.material).toBe(0);
        expect(primitive.mode).toBe(4);
        expect(Object.keys(primitive.attributes)).toEqual(Object.keys(grid.attributes));
        for (const name of Object.keys(grid.attributes)) {
          const produced = out.accessors[primitive.attributes[name]], original = source.accessors[grid.attributes[name]];
          expect(produced.componentType).toBe(original.componentType);
          expect(produced.type).toBe(original.type);
          expect(produced.normalized).toBe(original.normalized);
          expect(produced.count).toBe(vertices.length);
          const view = out.bufferViews[produced.bufferView];
          expect(view.byteOffset % 4).toBe(0);
          expect((view.byteStride ?? 4) % 4).toBe(0);
          for (const [j, vertex] of vertices.entries()) {
            expect(element(out, bin, primitive.attributes[name], j).equals(element(source, sourceBin, grid.attributes[name], vertex))).toBe(true);
          }
        }
        expect(out.bufferViews[out.accessors[primitive.attributes.COLOR_0].bufferView].byteStride).toBe(4);
        expect(out.accessors[primitive.attributes.TEXCOORD_1].extras).toEqual({ kept: true });
        for (const name of ['POSITION', 'TEXCOORD_1']) {
          const accessorIndex = primitive.attributes[name], values = vertices.map((_, j) => floats(out, bin, accessorIndex, j));
          // JSON has no negative zero, so compare with -0 folded into 0.
          const min = [0, 1, 2].slice(0, values[0].length).map(c => Math.min(...values.map(v => v[c])) + 0);
          const max = [0, 1, 2].slice(0, values[0].length).map(c => Math.max(...values.map(v => v[c])) + 0);
          expect(out.accessors[accessorIndex].min).toEqual(min);
          expect(out.accessors[accessorIndex].max).toEqual(max);
          if (name === 'POSITION') expect(entry.bounds).toEqual({ min, max });
        }
        expect(out.accessors[primitive.attributes.NORMAL].min).toBeUndefined();
        expect(out.accessors[primitive.indices].componentType).toBe(5123);
        expect(out.accessors[primitive.indices].count).toBe(triangles.length * 3);
        for (const [i, t] of triangles.entries()) for (let corner = 0; corner < 3; corner++) {
          expect(vertices[indexValue(out, bin, primitive.indices, i * 3 + corner)]).toBe(indexValue(source, sourceBin, grid.indices, t * 3 + corner));
        }
      }
      expect(Array.from(covered).every(hit => hit === 1)).toBe(true);

      // Unsplit primitives keep their element bytes, fields and accessor sharing.
      const quadOut = out.meshes[0].primitives[parts], sharedOut = out.meshes[1].primitives[0];
      const quad = source.meshes[0].primitives[1], shared = source.meshes[1].primitives[0];
      expect(quadOut.material).toBe(1);
      expect(quadOut.extras).toEqual({ primitive: 'quad' });
      expect(sharedOut.attributes.POSITION).toBe(quadOut.attributes.POSITION);
      expect(sharedOut.attributes.NORMAL).toBe(quadOut.attributes.NORMAL);
      for (const [produced, original] of [[quadOut, quad], [sharedOut, shared]]) {
        expect(Object.keys(produced)).toEqual(Object.keys(original));
        for (const name of Object.keys(original.attributes)) {
          const accessor = out.accessors[produced.attributes[name]];
          expect({ ...accessor, bufferView: 0 }).toEqual({ ...source.accessors[original.attributes[name]], bufferView: 0, byteOffset: undefined });
          for (let j = 0; j < accessor.count; j++) {
            expect(element(out, bin, produced.attributes[name], j).equals(element(source, sourceBin, original.attributes[name], j))).toBe(true);
          }
        }
        for (let i = 0; i < 6; i++) expect(indexValue(out, bin, produced.indices, i)).toBe(indexValue(source, sourceBin, original.indices, i));
      }
      expect(manifest.primitives[parts]).toMatchObject({ split: false, source: { meshIndex: 0, primitiveIndex: 1 }, provenance: null,
        triangleCount: 2, vertexCount: 4, bounds: { min: [0, 0, 1], max: [1, 1, 1] } });

      // Every file is listed with its hash; images and the license are carried along.
      const listed = manifest.files.map((file: Json) => file.file);
      expect(await listFiles(output)).toEqual([...listed, 'manifest.json'].sort());
      for (const file of manifest.files) {
        const bytes = await readFile(join(output, file.file));
        expect([bytes.length, sha256(bytes)]).toEqual([file.bytes, file.sha256]);
      }
      expect(await readFile(join(output, 'textures', 'base color.png'), 'utf8')).toBe('not decoded by the splitter');
      expect((await stat(join(output, 'textures', 'base color.png'))).ino).toBe((await stat(join(root, 'scene', 'textures', 'base color.png'))).ino);
      expect(manifest.images).toEqual([{ image: 0, uri: './textures/base%20color.png', file: 'textures/base color.png',
        bytes: 27, sha256: sha256(Buffer.from('not decoded by the splitter')), identicalToSource: true }]);
      expect(manifest.license).toMatchObject({ file: 'credits_license.txt', bytes: 13 });
      expect(manifest.verification).toMatchObject({ attributeBytesIdentical: true, trianglesIdentical: true, coverageExactlyOnce: true, triangles: 801 });

      // A second run is byte-identical.
      const again = join(root, 'split-again');
      node('split-gltf-primitives.mjs', [...args.slice(0, 3), again, ...args.slice(4)]);
      const files = await listFiles(output);
      expect(await listFiles(again)).toEqual(files);
      for (const file of files) expect((await readFile(join(again, file))).equals(await readFile(join(output, file)))).toBe(true);
      expect(() => node('split-gltf-primitives.mjs', args)).toThrow(/SPLIT_OUTPUT_EXISTS/);

      // The geometry builder accepts every part (the one degenerate triangle is removed by its explicit cleanup).
      node('build-gltf-geometry.mjs', ['--input', join(output, 'scene.split.gltf'), '--output', join(root, 'pages'), '--remove-zero-area', '--group-size', '4']);
      const geometry = JSON.parse(await readFile(join(root, 'pages', 'manifest.json'), 'utf8'));
      expect(geometry.primitives).toHaveLength(parts + 2);
      expect(geometry.primitives.every((p: Json) => p.route === 'paged')).toBe(true);
      expect(geometry.totals).toMatchObject({ sourceTriangles: 805, triangles: 804, removedTriangles: 1, pagedPrimitives: parts + 2 });
      expect(geometry.primitives.slice(0, parts).map((p: Json) => p.sourceTriangleCount)).toEqual(entries.map((p: Json) => p.triangleCount));
    });
  }, 120_000);

  it('cuts by vertex count alone and breaks equal centroid keys by source triangle index', async () => {
    await withRoot(async root => {
      const { json, binary } = scene();
      const input = await writeScene(root, json, binary);
      node('split-gltf-primitives.mjs', ['--input', input, '--output', join(root, 'by-vertices'), '--max-triangles', '100000', '--max-vertices', '150']);
      const byVertices = JSON.parse(await readFile(join(root, 'by-vertices', 'manifest.json'), 'utf8'));
      expect(byVertices.sourcePrimitives[0].parts).toBeGreaterThan(2);
      expect(byVertices.primitives.filter((p: Json) => p.split).every((p: Json) => p.vertexCount <= 150)).toBe(true);

      // Ten copies of one triangle: every centroid is equal, so parts are index ranges.
      const positions = Buffer.alloc(36), normals = Buffer.alloc(36), indices = Buffer.alloc(120);
      [0, 0, 0, 1, 0, 0, 0, 1, 0].forEach((value, i) => positions.writeFloatLE(value, i * 4));
      [0, 0, 1, 0, 0, 1, 0, 0, 1].forEach((value, i) => normals.writeFloatLE(value, i * 4));
      for (let t = 0; t < 10; t++) [0, 1, 2].forEach((value, c) => indices.writeUInt32LE(value, (t * 3 + c) * 4));
      const tie = { asset: { version: '2.0' }, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
        meshes: [{ primitives: [{ attributes: { POSITION: 1, NORMAL: 2 }, indices: 0 }] }],
        accessors: [{ bufferView: 0, componentType: 5125, count: 30, type: 'SCALAR' },
          { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] },
          { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' }],
        bufferViews: [{ buffer: 0, byteLength: 120 }, { buffer: 0, byteOffset: 120, byteLength: 36 }, { buffer: 0, byteOffset: 156, byteLength: 36 }],
        buffers: [{ uri: 'scene.bin', byteLength: 192 }] };
      const tieInput = await writeScene(root, tie, Buffer.concat([indices, positions, normals]), 'tie');
      node('split-gltf-primitives.mjs', ['--input', tieInput, '--output', join(root, 'tie-split'), '--max-triangles', '3', '--max-vertices', '3']);
      const manifest = JSON.parse(await readFile(join(root, 'tie-split', 'manifest.json'), 'utf8'));
      const ranges = [];
      for (const entry of manifest.primitives) ranges.push(uint32s(await readFile(join(root, 'tie-split', entry.provenance.triangles.file))));
      expect(ranges).toEqual([[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]);
      expect(manifest.primitives.map((p: Json) => p.vertexCount)).toEqual([3, 3, 3, 3]);
    });
  }, 120_000);

  it('splits only listed primitives and rejects unsupported or invalid inputs explicitly', async () => {
    await withRoot(async root => {
      const { json, binary } = scene();
      const input = await writeScene(root, json, binary);
      node('split-gltf-primitives.mjs', ['--input', input, '--output', join(root, 'listed'), '--max-triangles', '120', '--max-vertices', '100', '--primitives', '1:0']);
      const listed = JSON.parse(await readFile(join(root, 'listed', 'manifest.json'), 'utf8'));
      expect(listed.options).toMatchObject({ selection: 'listed', primitives: [{ meshIndex: 1, primitiveIndex: 0 }] });
      expect(listed.sourcePrimitives[0]).toMatchObject({ split: false, exceedsLimits: true });
      expect(listed.sourcePrimitives[2]).toMatchObject({ split: true, parts: 1, referencedVertices: 4, duplicatedVertices: 0 });

      const fails = async (name: string, document: Json, extra: string[], code: RegExp) => {
        const sceneInput = await writeScene(root, document, binary, name);
        expect(() => node('split-gltf-primitives.mjs', ['--input', sceneInput, '--output', join(root, name + '-out'),
          '--max-triangles', '120', '--max-vertices', '100', ...extra])).toThrow(code);
      };
      const copy = (): Json => structuredClone(json);
      expect(() => node('split-gltf-primitives.mjs', ['--input', input, '--output', join(root, 'x'), '--max-triangles', '120', '--max-vertices', '2'])).toThrow(/SPLIT_MAX_VERTICES/);
      expect(() => node('split-gltf-primitives.mjs', ['--input', input, '--output', join(root, 'x'), '--max-triangles', '0', '--max-vertices', '9'])).toThrow(/SPLIT_MAX_TRIANGLES/);
      const morph = copy(); morph.meshes[1].primitives[0].targets = [{ POSITION: 0 }];
      await fails('morph', morph, [], /SPLIT_MORPH_TARGETS_UNSUPPORTED/);
      const sparse = copy(); sparse.accessors[1].sparse = { count: 1, indices: { bufferView: 0, componentType: 5125 }, values: { bufferView: 1 } };
      await fails('sparse', sparse, [], /SPLIT_SPARSE_ACCESSOR_UNSUPPORTED/);
      const points = copy(); points.meshes[0].primitives[0].mode = 0;
      await fails('points', points, [], /SPLIT_TRIANGLES_WITH_INDICES_REQUIRED/);
      const extension = copy(); extension.extensionsUsed = ['EXT_meshopt_compression'];
      await fails('extension', extension, [], /SPLIT_EXTENSION_UNSUPPORTED/);
      const animated = copy(); animated.animations = [{ channels: [], samplers: [] }];
      await fails('animated', animated, [], /SPLIT_ANIMATIONS_UNSUPPORTED/);
      await fails('missing', copy(), ['--primitives', '3:0'], /SPLIT_PRIMITIVE_SELECTION/);
      const outside = copy(); outside.images[0].uri = '../outside.png';
      await fails('outside', outside, [], /SPLIT_IMAGE_URI/);
      const range = copy(); range.accessors[range.meshes[0].primitives[0].indices].count = 2404;
      await fails('range', range, [], /SPLIT_ACCESSOR_LAYOUT|SPLIT_INDEX/);
    });
  }, 120_000);
});
