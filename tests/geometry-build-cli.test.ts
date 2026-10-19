import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtemp, mkdir, readFile, readdir, rm, truncate, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { decodeGeometryPage } from '../src/geometry-pages';
import { decodeGeometryQueryProxy } from '../src/geometry-query-proxy';
import { validateGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';

const script = resolve('scripts/build-geometry.mjs');
const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

function gridObj(size = 16): string {
  const lines = ['mtllib swatches.mtl', 'o grid', 'vn 0 0 1'];
  for (let y = 0; y <= size; y++) {
    for (let x = 0; x <= size; x++) lines.push(`v ${x} ${y} 0`);
  }
  for (let y = 0; y < size; y++) {
    if (y === 0) lines.push('usemtl lower');
    if (y === size / 2) lines.push('usemtl upper');
    for (let x = 0; x < size; x++) {
      const a = y * (size + 1) + x + 1;
      lines.push(`f ${a}//1 ${a + 1}//1 ${a + size + 2}//1 ${a + size + 1}//1`);
    }
  }
  return `${lines.join('\n')}\n`;
}

function run(input: string, output: string): void {
  execFileSync(process.execPath, [script, '--input', input, '--output', output, '--coverage', 'opaque'], {
    cwd: resolve('.'), windowsHide: true, timeout: 60_000, maxBuffer: 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

async function removeOwnTestRoot(root: string): Promise<void> {
  const workspace = resolve('.');
  const absolute = resolve(root);
  if (dirname(absolute) !== workspace || !basename(absolute).startsWith('.geometry-build-test-')) {
    throw new Error('refusing to remove a directory outside the geometry CLI test root');
  }
  await rm(absolute, { recursive: true, force: true });
}

async function files(root: string, prefix = ''): Promise<Map<string, Uint8Array>> {
  const result = new Map<string, Uint8Array>();
  for (const item of await readdir(join(root, prefix), { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory()) {
      for (const [key, value] of await files(root, name)) result.set(key, value);
    } else {
      result.set(name, await readFile(join(root, name)));
    }
  }
  return result;
}

describe('offline geometry CLI', () => {
  it('builds independently named primitives, deterministic pages and a verifiable manifest', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.geometry-build-test-'));
    try {
      const input = join(temporary, 'grid.obj');
      const source = gridObj();
      await writeFile(input, source);
      const first = join(temporary, 'first');
      const second = join(temporary, 'second');
      run(input, first);
      run(input, second);

      const left = await files(first);
      const right = await files(second);
      expect([...left.keys()].sort()).toEqual([...right.keys()].sort());
      for (const [name, bytes] of left) expect(bytes).toEqual(right.get(name));

      const manifest = JSON.parse((await readFile(join(first, 'manifest.json'))).toString('utf8'));
      expect(manifest.version).toBe(1);
      expect(manifest.source.sha256).toBe(sha256(Buffer.from(source)));
      expect(manifest.source.bytes).toBe(Buffer.byteLength(source));
      expect(manifest.config.coverage).toBe('opaque');
      expect(manifest.materialLibraries).toEqual(['swatches.mtl']);
      expect(manifest.primitives).toHaveLength(2);
      expect(manifest.primitives.map((primitive: { name: string; materialName: string }) =>
        [primitive.name, primitive.materialName])).toEqual([['grid', 'lower'], ['grid', 'upper']]);
      for (const primitive of manifest.primitives) {
        expect(primitive.sourceSha256).toBe(manifest.source.sha256);
        expect(primitive.vertexCount).toBeGreaterThan(0);
        expect(primitive.triangleCount).toBe(256);
        expect(primitive.clusterCount).toBeGreaterThan(0);
        expect(primitive.pages.length).toBeGreaterThan(0);
        expect(primitive.clusterPages).toHaveLength(primitive.clusterCount);
        const hierarchy = JSON.parse((await readFile(join(first, primitive.hierarchyFile))).toString('utf8'));
        expect(hierarchy.purpose).toMatch(/offline-audit/);
        expect(hierarchy.source.vertexCount).toBe(primitive.vertexCount);
        expect(hierarchy.source.vertices).toBeUndefined();
        expect(hierarchy.source.indices).toBeUndefined();
        expect(hierarchy.clusters).toHaveLength(primitive.clusterCount);
        expect(hierarchy.groups).toHaveLength(primitive.groupCount);
        const directoryBytes = await readFile(join(first, primitive.streamDirectoryFile));
        expect(primitive.streamDirectoryFile).toBe(`primitive-${primitive.index}/stream-directory.json`);
        expect(directoryBytes.byteLength).toBe(primitive.streamDirectoryBytes);
        expect(sha256(directoryBytes)).toBe(primitive.streamDirectorySha256);
        expect(directoryBytes.at(-1)).toBe(10); // The manifest hashes the written newline too.
        const directory = JSON.parse(directoryBytes.toString('utf8')) as GeometryStreamDirectory;
        expect(() => validateGeometryStreamDirectory(directory)).not.toThrow();
        expect(directory.primitiveKey).toBe(primitive.primitiveKey);
        expect(directory.sourceVertexCount).toBe(primitive.vertexCount);
        expect(directory.sourceTriangleCount).toBe(primitive.triangleCount);
        expect(directory.clusters).toHaveLength(primitive.clusterCount);
        expect(directory.groups).toHaveLength(primitive.groupCount);
        expect(directory.pages).toHaveLength(primitive.pages.length);
        expect(directory.clusterPages).toEqual(primitive.clusterPages);
        expect(directory.rootPages).toEqual(primitive.rootPages);
        expect((directory as unknown as Record<string, unknown>).source).toBeUndefined();
        expect(directory.clusters.every(cluster => !('vertices' in cluster) && !('triangles' in cluster))).toBe(true);
        const proxyBytes = new Uint8Array(await readFile(join(first, primitive.queryProxy.file)));
        expect(proxyBytes.byteLength).toBe(primitive.queryProxy.bytes);
        expect(sha256(proxyBytes)).toBe(primitive.queryProxy.sha256);
        const proxy = await decodeGeometryQueryProxy(directory, proxyBytes, primitive.queryProxy.sha256);
        expect(proxy.directorySha256).toBe(primitive.queryProxy.directorySha256);
        expect(proxy.mesh.vertexCount).toBe(primitive.queryProxy.vertexCount);
        expect(proxy.mesh.indices.length / 3).toBe(primitive.queryProxy.triangleCount);
        expect(proxy.mesh.indices.length / 3).toBeLessThan(primitive.triangleCount);
        expect(proxy.approximateObjectSpaceError).toBe(primitive.queryProxy.approximateObjectSpaceError);
        expect(proxy.requestedError).toBe(manifest.config.queryProxy.maxError);
        expect(proxy.selectedClusters).toEqual([...directory.roots].sort((a, b) => a - b));
        for (const page of primitive.pages) {
          const bytes = await readFile(join(first, page.file));
          expect(bytes.byteLength).toBe(page.compressedBytes);
          expect(sha256(bytes)).toBe(page.sha256);
          expect(directory.pages[page.id]).toMatchObject({ id: page.id, clusterIds: page.clusterIds,
            encodedBytes: bytes.byteLength, decodedBytes: page.decodedBytes, sha256: sha256(bytes) });
          const decoded = await decodeGeometryPage(bytes);
          expect(decoded.vertexStride).toBe(primitive.vertexStride);
          expect(decoded.clusters.map(cluster => cluster.id)).toEqual(page.clusterIds);
          expect(decoded.clusters.every(cluster => cluster.sourceVertexIds.length > 0
            && cluster.rest.byteLength === cluster.sourceVertexIds.length * 80
            && cluster.localTriangles.length > 0)).toBe(true);
          expect(page.decodedBytes).toBeGreaterThan(0);
          expect(page.clusterIds.length).toBeGreaterThan(0);
        }
        expect(primitive.rootPages.every((id: number) => primitive.pages.some((page: { id: number }) => page.id === id)))
          .toBe(true);
      }
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 90_000);

  it('rejects missing input and never replaces an existing output directory', async () => {
    const temporary = await mkdtemp(join(resolve('.'), '.geometry-build-test-'));
    try {
      const output = join(temporary, 'existing');
      await mkdir(output);
      await writeFile(join(output, 'sentinel.txt'), 'keep me');
      const input = join(temporary, 'grid.obj');
      await writeFile(input, gridObj(2));
      expect(() => run(input, output)).toThrow();
      expect(await readFile(join(output, 'sentinel.txt'), 'utf8')).toBe('keep me');
      expect(() => run(join(temporary, 'missing.obj'), join(temporary, 'missing-output'))).toThrow();
      expect(() => execFileSync(process.execPath,
        [script, '--input', input, '--output', join(temporary, 'no-coverage')],
        { cwd: resolve('.'), windowsHide: true, timeout: 30_000,
          stdio: ['ignore', 'pipe', 'pipe'] })).toThrow();
      const oversized = join(temporary, 'oversized.obj');
      await writeFile(oversized, '');
      await truncate(oversized, 32 * 1024 * 1024 + 1);
      expect(() => run(oversized, join(temporary, 'oversized-output'))).toThrow();
      expect(() => execFileSync(process.execPath,
        [script, '--input', input, '--output', join(temporary, 'invalid-error'), '--coverage', 'opaque', '--query-error', '-1'],
        { cwd: resolve('.'), windowsHide: true, timeout: 30_000, stdio: ['ignore', 'pipe', 'pipe'] })).toThrow();
      expect((await readdir(temporary)).sort()).toEqual(['existing', 'grid.obj', 'oversized.obj']);
    } finally {
      await removeOwnTestRoot(temporary);
    }
  }, 30_000);
});
