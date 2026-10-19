import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { validateGeometryStreamDirectory } from '../src/geometry-stream-directory';

const script = resolve('scripts/build-gltf-geometry.mjs');
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

type FileRef = { file: string; sha256: string; bytes: number };
type BuiltPrimitive = {
  meshIndex: number; primitiveIndex: number; coverage: string; route: string;
  sourceTriangleCount: number; triangleCount: number; removedTriangleCount: number;
  directory?: string; streamDirectoryFile?: string; queryProxy?: FileRef;
  pages?: FileRef[]; provenance?: {
    vertices: FileRef; triangles: FileRef; removedTriangles: FileRef;
  };
};

function fixture(): { json: Buffer; binary: Buffer; opaqueIndices: number[]; blendIndices: number[] } {
  const views: { buffer: number; byteOffset: number; byteLength: number }[] = [];
  const accessors: Record<string, unknown>[] = [];
  const chunks: Buffer[] = [];
  let offset = 0;
  function add(values: number[], type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4',
    componentType: 5126 | 5123, extra: Record<string, unknown> = {}): number {
    const componentSize = componentType === 5126 ? 4 : 2;
    const components = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[type];
    const data = Buffer.alloc(values.length * componentSize);
    values.forEach((value, index) => {
      if (componentType === 5126) data.writeFloatLE(value, index * 4);
      else data.writeUInt16LE(value, index * 2);
    });
    const padding = (4 - offset % 4) % 4;
    if (padding) { chunks.push(Buffer.alloc(padding)); offset += padding; }
    const bufferView = views.length;
    views.push({ buffer: 0, byteOffset: offset, byteLength: data.byteLength });
    chunks.push(data);
    offset += data.byteLength;
    accessors.push({ bufferView, componentType, count: values.length / components, type, ...extra });
    return accessors.length - 1;
  }

  // Two adjacent quads share indices; vertex 6 repeats position 0 with another UV
  // so the source also has an authored UV seam. Triangle 4 is exactly degenerate.
  const opaquePositions = [
    0, 0, 0, 1, 0, 0, 2, 0, 0,
    0, 1, 0, 1, 1, 0, 2, 1, 0,
    0, 0, 0,
  ];
  const opaqueUv = [0, 0, .5, 0, 1, 0, 0, 1, .5, 1, 1, 1, .125, 0];
  const opaqueIndices = [6, 1, 4, 0, 4, 3, 1, 2, 5, 1, 5, 4, 6, 6, 1];
  const opaque = {
    POSITION: add(opaquePositions, 'VEC3', 5126, { min: [0, 0, 0], max: [2, 1, 0] }),
    NORMAL: add(Array(7).fill([0, 0, 1]).flat(), 'VEC3', 5126),
    TEXCOORD_0: add(opaqueUv, 'VEC2', 5126),
    TEXCOORD_1: add(Array(7).fill([.25, .75]).flat(), 'VEC2', 5126),
    TANGENT: add(Array(7).fill([1, 0, 0, 1]).flat(), 'VEC4', 5126),
    COLOR_0: add(Array(7).fill([.75, .5, .25, 1]).flat(), 'VEC4', 5126),
  };
  const opaqueIndexAccessor = add(opaqueIndices, 'SCALAR', 5123);

  const blendPositions = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
  const blendIndices = [0, 1, 2, 0, 2, 3];
  const blend = {
    POSITION: add(blendPositions, 'VEC3', 5126, { min: [0, 0, 1], max: [1, 1, 1] }),
    NORMAL: add(Array(4).fill([0, 0, 1]).flat(), 'VEC3', 5126),
    TEXCOORD_0: add([0, 0, 1, 0, 1, 1, 0, 1], 'VEC2', 5126),
    TANGENT: add(Array(4).fill([1, 0, 0, 1]).flat(), 'VEC4', 5126),
    COLOR_0: add(Array(4).fill([1, .5, 1, .5]).flat(), 'VEC4', 5126),
  };
  const blendIndexAccessor = add(blendIndices, 'SCALAR', 5123);
  const binary = Buffer.concat(chunks);
  const gltf = {
    asset: { version: '2.0', generator: 'original glTF geometry CLI test' },
    buffers: [{ uri: 'mesh.bin', byteLength: binary.byteLength }],
    bufferViews: views, accessors,
    images: [{ uri: 'textures/not-downloaded.png', name: 'image metadata only' }],
    textures: [{ source: 0 }],
    materials: [
      { name: 'stone', alphaMode: 'OPAQUE', pbrMetallicRoughness: { baseColorTexture: { index: 0 } } },
      { name: 'glass', alphaMode: 'BLEND', pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, .5] } },
    ],
    meshes: [{ primitives: [
      { attributes: opaque, indices: opaqueIndexAccessor, material: 0, mode: 4 },
      { attributes: blend, indices: blendIndexAccessor, material: 1, mode: 4 },
    ] }],
    nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0,
  };
  return { json: Buffer.from(`${JSON.stringify(gltf, null, 2)}\n`), binary, opaqueIndices, blendIndices };
}

function run(input: string, output: string, extra: string[] = []): void {
  execFileSync(process.execPath, [script, '--input', input, '--output', output, ...extra], {
    cwd: resolve('.'), windowsHide: true, timeout: 120_000, maxBuffer: 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

async function removeOwnTestRoot(root: string): Promise<void> {
  const absolute = resolve(root);
  if (dirname(absolute) !== resolve('.') || !basename(absolute).startsWith('.gltf-geometry-test-')) {
    throw new Error('refusing to remove a directory outside this test root');
  }
  await rm(absolute, { recursive: true, force: true });
}

async function files(root: string, prefix = ''): Promise<Map<string, Buffer>> {
  const result = new Map<string, Buffer>();
  for (const item of await readdir(join(root, prefix), { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory()) {
      for (const [key, value] of await files(root, name)) result.set(key, value);
    } else result.set(name, await readFile(join(root, name)));
  }
  return result;
}

async function expectFileRef(output: string, ref: FileRef): Promise<Buffer> {
  expect(ref.file).not.toMatch(/(^[/\\]|\.\.)/);
  const bytes = await readFile(join(output, ref.file));
  expect(bytes.byteLength).toBe(ref.bytes);
  expect(sha256(bytes)).toBe(ref.sha256);
  return bytes;
}

function u32(bytes: Buffer): number[] {
  expect(bytes.byteLength % 4).toBe(0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getUint32(index * 4, true));
}

function glbOf(source: ReturnType<typeof fixture>): Buffer {
  const document = JSON.parse(source.json.toString('utf8'));
  delete document.buffers[0].uri;
  const json = Buffer.from(JSON.stringify(document), 'utf8');
  const jsonPad = (4 - json.length % 4) % 4;
  const binPad = (4 - source.binary.length % 4) % 4;
  const jsonChunk = Buffer.concat([json, Buffer.alloc(jsonPad, 0x20)]);
  const binChunk = Buffer.concat([source.binary, Buffer.alloc(binPad)]);
  const container = Buffer.alloc(12 + 8 + jsonChunk.length + 8 + binChunk.length);
  container.writeUInt32LE(0x46546c67, 0);
  container.writeUInt32LE(2, 4);
  container.writeUInt32LE(container.length, 8);
  container.writeUInt32LE(jsonChunk.length, 12);
  container.writeUInt32LE(0x4e4f534a, 16);
  jsonChunk.copy(container, 20);
  const binHeader = 20 + jsonChunk.length;
  container.writeUInt32LE(binChunk.length, binHeader);
  container.writeUInt32LE(0x004e4942, binHeader + 4);
  binChunk.copy(container, binHeader + 8);
  return container;
}

describe('glTF geometry CLI', () => {
  it('rejects degeneracy by default, then preserves coverage and provenance after explicit removal', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.gltf-geometry-test-'));
    try {
      const input = join(temporary, 'scene.gltf');
      const source = fixture();
      await writeFile(input, source.json);
      await writeFile(join(temporary, 'mesh.bin'), source.binary);
      expect(() => run(input, join(temporary, 'rejected'))).toThrow();
      expect(await readdir(temporary)).toEqual(['mesh.bin', 'scene.gltf']);

      const first = join(temporary, 'first');
      const second = join(temporary, 'second');
      run(input, first, ['--remove-zero-area', '--query-error', '0']);
      run(input, second, ['--remove-zero-area', '--query-error', '0']);
      const left = await files(first);
      const right = await files(second);
      expect([...left.keys()].sort()).toEqual([...right.keys()].sort());
      for (const [name, bytes] of left) expect(bytes).toEqual(right.get(name));

      const manifest = JSON.parse((await readFile(join(first, 'manifest.json'))).toString('utf8')) as {
        format: string; version: number;
        source: FileRef & { buffers: FileRef[] };
        images: unknown[]; imageResourcesCopied: boolean; primitives: BuiltPrimitive[];
      };
      expect(manifest.format).toBe('real-engine-gltf-geometry');
      expect(manifest.version).toBe(1);
      expect(manifest.source).toMatchObject({ file: 'source/scene.gltf', bytes: source.json.byteLength,
        sha256: sha256(source.json) });
      expect(await expectFileRef(first, manifest.source)).toEqual(source.json);
      expect(manifest.source.buffers).toHaveLength(1);
      expect(manifest.source.buffers[0]).toMatchObject({ file: 'source/mesh.bin', bytes: source.binary.byteLength,
        sha256: sha256(source.binary) });
      expect(await expectFileRef(first, manifest.source.buffers[0])).toEqual(source.binary);
      expect(manifest.images).toEqual([{ uri: 'textures/not-downloaded.png', name: 'image metadata only' }]);
      expect(manifest.imageResourcesCopied).toBe(false);
      expect(left.has('source/textures/not-downloaded.png')).toBe(false);

      expect(manifest.primitives).toHaveLength(2);
      const [opaque, blend] = manifest.primitives;
      expect(manifest.primitives.map(p => [p.meshIndex, p.primitiveIndex, p.coverage, p.route])).toEqual([
        [0, 0, 'opaque', 'paged'], [0, 1, 'blend', 'source'],
      ]);
      expect(opaque).toMatchObject({ sourceTriangleCount: 5, triangleCount: 4, removedTriangleCount: 1 });
      expect(blend).toMatchObject({ sourceTriangleCount: 2, triangleCount: 2, removedTriangleCount: 0 });
      expect(opaque.directory).toBeTruthy();
      expect(opaque.streamDirectoryFile).toBe(`${opaque.directory}/stream-directory.json`);
      expect(opaque.queryProxy).toBeDefined();
      expect(opaque.pages?.length).toBeGreaterThan(0);
      expect(opaque.provenance).toBeDefined();
      const directory = JSON.parse((await readFile(join(first, opaque.streamDirectoryFile!))).toString('utf8'));
      expect(() => validateGeometryStreamDirectory(directory)).not.toThrow();
      await expectFileRef(first, opaque.queryProxy!);
      for (const page of opaque.pages!) await expectFileRef(first, page);
      const provenance = opaque.provenance!;
      const vertices = u32(await expectFileRef(first, provenance.vertices));
      const triangles = u32(await expectFileRef(first, provenance.triangles));
      const removed = u32(await expectFileRef(first, provenance.removedTriangles));
      expect(vertices).toEqual([6, 1, 4, 0, 3, 2, 5]);
      expect(triangles).toEqual([0, 1, 2, 3]);
      expect(removed).toEqual([4]);
      expect(blend.pages ?? []).toHaveLength(0);
      expect(blend.queryProxy).toBeUndefined();
      expect(blend.provenance).toBeUndefined();
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);

  it('never replaces an existing output and rejects a buffer URI outside the input directory', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.gltf-geometry-test-'));
    try {
      const source = fixture();
      const input = join(temporary, 'scene.gltf');
      await writeFile(input, source.json);
      await writeFile(join(temporary, 'mesh.bin'), source.binary);
      const existing = join(temporary, 'existing');
      await mkdir(existing);
      await writeFile(join(existing, 'sentinel.txt'), 'keep me');
      expect(() => run(input, existing, ['--remove-zero-area'])).toThrow();
      expect(await readFile(join(existing, 'sentinel.txt'), 'utf8')).toBe('keep me');

      const escape = JSON.parse(source.json.toString('utf8'));
      escape.buffers[0].uri = '../outside.bin';
      const unsafe = join(temporary, 'unsafe.gltf');
      await writeFile(unsafe, `${JSON.stringify(escape)}\n`);
      expect(() => run(unsafe, join(temporary, 'unsafe-output'), ['--remove-zero-area'])).toThrow();
      for (const [name, budget] of [['tiny', '1'], ['zero', '0'], ['over-hard', '4294967297']] as const) {
        const output = join(temporary, name);
        expect(() => run(input, output, ['--remove-zero-area', '--max-page-bytes', budget])).toThrow();
        expect((await readdir(temporary)).includes(name)).toBe(false);
      }
      expect((await readdir(temporary)).sort()).toEqual(['existing', 'mesh.bin', 'scene.gltf', 'unsafe.gltf']);
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);

  it('retains GLB and data URI source bytes and exposes each embedded BIN separately', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.gltf-geometry-test-'));
    try {
      const source = fixture();
      const container = glbOf(source);
      const inlineDoc = JSON.parse(source.json.toString('utf8'));
      inlineDoc.buffers[0].uri = `data:application/octet-stream;base64,${source.binary.toString('base64')}`;
      const inline = Buffer.from(`${JSON.stringify(inlineDoc)}\n`);
      for (const [name, bytes] of [['scene.glb', container], ['inline.gltf', inline]] as const) {
        const input = join(temporary, name);
        const output = join(temporary, `${name}-built`);
        await writeFile(input, bytes);
        run(input, output, ['--remove-zero-area']);
        const manifest = JSON.parse((await readFile(join(output, 'manifest.json'))).toString('utf8')) as {
          source: FileRef & { buffers: (FileRef & { embedded: boolean })[] };
          primitives: BuiltPrimitive[];
        };
        expect(manifest.source).toMatchObject({ file: `source/${name}`, bytes: bytes.length,
          sha256: sha256(bytes) });
        expect(await expectFileRef(output, manifest.source)).toEqual(bytes);
        expect(manifest.source.buffers).toHaveLength(1);
        expect(manifest.source.buffers[0]).toMatchObject({ file: 'embedded/buffer-0.bin', embedded: true,
          bytes: source.binary.length, sha256: sha256(source.binary) });
        expect(await expectFileRef(output, manifest.source.buffers[0])).toEqual(source.binary);
        expect(manifest.primitives.map(p => [p.coverage, p.route, p.sourceTriangleCount, p.triangleCount])).toEqual([
          ['opaque', 'paged', 5, 4], ['blend', 'source', 2, 2],
        ]);
        expect(u32(await expectFileRef(output, manifest.primitives[0].provenance!.vertices)))
          .toEqual([6, 1, 4, 0, 3, 2, 5]);
        expect(u32(await expectFileRef(output, manifest.primitives[0].provenance!.triangles)))
          .toEqual([0, 1, 2, 3]);
        expect(u32(await expectFileRef(output, manifest.primitives[0].provenance!.removedTriangles)))
          .toEqual([4]);
      }
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 120_000);
});
