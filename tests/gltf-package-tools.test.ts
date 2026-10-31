import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { decodeGeometryQueryProxy } from '../src/geometry-query-proxy';
import { validateGeometryStreamDirectory } from '../src/geometry-stream-directory';

const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
const node = (script: string, args: string[]): string => execFileSync(process.execPath, [resolve('scripts', script), ...args], {
  cwd: resolve('.'), windowsHide: true, timeout: 120_000, maxBuffer: 4 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'],
}).toString();

/** A 12x12 opaque grid (288 triangles) plus one exactly degenerate triangle, and a BLEND quad. */
function scene(): { json: Buffer; binary: Buffer } {
  const views: object[] = [], accessors: object[] = [], chunks: Buffer[] = [];
  let offset = 0;
  const add = (values: number[], type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4', componentType: 5126 | 5125, extra: object = {}): number => {
    const data = Buffer.alloc(values.length * 4);
    values.forEach((value, i) => componentType === 5126 ? data.writeFloatLE(value, i * 4) : data.writeUInt32LE(value, i * 4));
    views.push({ buffer: 0, byteOffset: offset, byteLength: data.length });
    chunks.push(data); offset += data.length;
    accessors.push({ bufferView: views.length - 1, componentType, count: values.length / { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[type], type, ...extra });
    return accessors.length - 1;
  };
  const n = 12, positions: number[] = [], uvs: number[] = [], indices: number[] = [];
  for (let y = 0; y <= n; y++) for (let x = 0; x <= n; x++) { positions.push(x * 0.5, y * 0.25, 0); uvs.push(x / n, y / n); }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const a = y * (n + 1) + x, b = a + 1, c = a + n + 1, d = c + 1;
    indices.push(a, b, d, a, d, c);
  }
  indices.push(0, 0, 1);
  const count = (n + 1) ** 2;
  const grid = { attributes: {
    POSITION: add(positions, 'VEC3', 5126, { min: [0, 0, 0], max: [n * 0.5, n * 0.25, 0] }),
    NORMAL: add(Array(count).fill([0, 0, 1]).flat(), 'VEC3', 5126),
    TEXCOORD_0: add(uvs, 'VEC2', 5126),
    TANGENT: add(Array(count).fill([1, 0, 0, 1]).flat(), 'VEC4', 5126),
  }, indices: add(indices, 'SCALAR', 5125), material: 0, mode: 4 };
  const quad = { attributes: {
    POSITION: add([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], 'VEC3', 5126, { min: [0, 0, 1], max: [1, 1, 1] }),
    NORMAL: add(Array(4).fill([0, 0, 1]).flat(), 'VEC3', 5126),
    TEXCOORD_0: add([0, 0, 1, 0, 1, 1, 0, 1], 'VEC2', 5126),
    TANGENT: add(Array(4).fill([1, 0, 0, 1]).flat(), 'VEC4', 5126),
  }, indices: add([0, 1, 2, 0, 2, 3], 'SCALAR', 5125), material: 1, mode: 4 };
  const binary = Buffer.concat(chunks);
  const gltf = { asset: { version: '2.0', generator: 'gltf package tools test' },
    buffers: [{ uri: 'mesh.bin', byteLength: binary.length }], bufferViews: views, accessors,
    materials: [{ name: 'opaque', alphaMode: 'OPAQUE' }, { name: 'glass', alphaMode: 'BLEND' }],
    meshes: [{ primitives: [grid, quad] }], nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0 };
  return { json: Buffer.from(JSON.stringify(gltf)), binary };
}

async function withRoot(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(resolve('.'), '.gltf-package-tools-test-'));
  try { await body(root); } finally {
    if (dirname(root) !== resolve('.') || !basename(root).startsWith('.gltf-package-tools-test-')) throw Error('refusing to remove');
    await rm(root, { recursive: true, force: true });
  }
}

describe('glTF package tools', () => {
  it('rebuilds exact query proxies from a hashed package and cross-checks UV demand against it', async () => {
    await withRoot(async root => {
      const { json, binary } = scene();
      const input = join(root, 'scene.gltf');
      await writeFile(input, json);
      await writeFile(join(root, 'mesh.bin'), binary);
      node('build-gltf-geometry.mjs', ['--input', input, '--output', join(root, 'pages'), '--remove-zero-area', '--group-size', '4']);
      const geometry = JSON.parse(await readFile(join(root, 'pages', 'manifest.json'), 'utf8'));
      const grid = geometry.primitives.find((p: { route: string }) => p.route === 'paged');
      expect(grid.sourceTriangleCount).toBe(289);
      expect(grid.triangleCount).toBe(288);

      node('build-gltf-query-proxies.mjs', ['--package', join(root, 'pages'), '--output', join(root, 'proxies'), '--query-error', '0']);
      const proxies = JSON.parse(await readFile(join(root, 'proxies', 'manifest.json'), 'utf8'));
      expect(proxies.format).toBe('real-engine-gltf-query-proxies');
      expect(proxies.sourcePackage).toEqual({ root: 'pages', manifestSha256: sha256(await readFile(join(root, 'pages', 'manifest.json'))),
        assetIdentity: geometry.assetIdentity, sourceSha256: sha256(json) });
      expect(proxies.primitives).toHaveLength(1);
      const entry = proxies.primitives[0];
      expect(entry.queryProxy.requestedError).toBe(0);
      expect(entry.queryProxy.triangleCount).toBe(288);
      const directoryBytes = await readFile(join(root, 'pages', grid.streamDirectoryFile));
      expect(entry.streamDirectorySha256).toBe(sha256(directoryBytes));
      const proxyBytes = await readFile(join(root, 'proxies', entry.queryProxy.file));
      expect(sha256(proxyBytes)).toBe(entry.queryProxy.sha256);
      const decoded = await decodeGeometryQueryProxy(validateGeometryStreamDirectory(JSON.parse(directoryBytes.toString())),
        new Uint8Array(proxyBytes), entry.queryProxy.sha256, { maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * 1024 * 1024 });
      expect(decoded.mesh.indices.length).toBe(288 * 3);
      expect(() => node('build-gltf-query-proxies.mjs', ['--package', join(root, 'pages'), '--output', join(root, 'proxies'), '--query-error', '0'])).toThrow();

      node('audit-gltf-uv-demand.mjs', ['--input', input, '--output', join(root, 'uv.json'), '--remove-zero-area',
        '--geometry-manifest', join(root, 'pages', 'manifest.json')]);
      const audit = JSON.parse(await readFile(join(root, 'uv.json'), 'utf8'));
      expect(audit.geometryCheck).toMatchObject({ assetIdentity: geometry.assetIdentity, primitives: 2, identical: true });
      expect(audit.primitives.map((p: { textureUvSpan: number | string }) => p.textureUvSpan))
        .toEqual(geometry.primitives.map((p: { textureUvSpan: number | string }) => p.textureUvSpan));
      expect(typeof audit.primitives[0].textureUvSpan).toBe('number');

      // A drifted manifest value is rejected rather than silently audited.
      const drifted = { ...geometry, primitives: geometry.primitives.map((p: object, i: number) => i ? p : { ...p, textureUvSpan: 1234 }) };
      await writeFile(join(root, 'drifted.json'), JSON.stringify(drifted));
      expect(() => node('audit-gltf-uv-demand.mjs', ['--input', input, '--output', join(root, 'uv-drift.json'), '--remove-zero-area',
        '--geometry-manifest', join(root, 'drifted.json')])).toThrow(/UV_AUDIT_GEOMETRY_MISMATCH/);
    });
  }, 120_000);
});
