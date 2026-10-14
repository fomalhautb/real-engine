import { describe, expect, it } from 'vitest';
import { MeshoptDecoder } from 'meshoptimizer/decoder';
import { MeshoptEncoder } from 'meshoptimizer/encoder';
import { decodeGeometryPage, encodeGeometryPages } from '../src/geometry-pages';
import type { GeometryHierarchy, GeometryCluster } from '../src/geometry-hierarchy-types';

const HEADER_BYTES = 40;
const REST_BYTES = 80;

function fixture(uvs = true, tangents = true): GeometryHierarchy {
  const vertexCount = 7;
  const vertices = new ArrayBuffer(vertexCount * REST_BYTES);
  const raw = new Uint8Array(vertices), view = new DataView(vertices);
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    raw.fill((vertex * 37 + 11) & 255, vertex * REST_BYTES, (vertex + 1) * REST_BYTES);
    view.setFloat32(vertex * REST_BYTES, vertex * 0.5 - 1, true);
    view.setFloat32(vertex * REST_BYTES + 4, vertex + 0.25, true);
    view.setFloat32(vertex * REST_BYTES + 8, -vertex, true);
  }
  const source = { vertexCount, vertices, indices: new Uint32Array([0, 1, 2]),
    bounds: { min: [-1, 0, -6] as [number, number, number], max: [2, 7, 0] as [number, number, number] },
    uvs: uvs ? new Float32Array(Array.from({ length: vertexCount * 2 }, (_, i) => i / 7)) : undefined,
    tangents: tangents ? new Float32Array(Array.from({ length: vertexCount * 4 }, (_, i) => i / 11)) : undefined };
  const bounds = { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] };
  const cluster = (ids: number[]): GeometryCluster => ({ level: 0, vertices: new Uint32Array(ids),
    triangles: new Uint8Array([0, 1, 2]), bounds, lodBounds: bounds, error: 0, refinedGroup: null, group: 0 });
  return { version: 1, builder: 'meshoptimizer-1.3.0', primitiveKey: 'fixture', source,
    options: {} as GeometryHierarchy['options'], clusters: [cluster([0, 1, 2]), cluster([2, 3, 4]), cluster([4, 5, 6])],
    groups: [], roots: [0, 2], sourceTriangleCount: 1 };
}

function hash(bytes: Uint8Array): number {
  let value = 0x811c9dc5;
  for (const byte of bytes) value = Math.imul(value ^ byte, 0x01000193) >>> 0;
  return value;
}

async function mutateDecodedPage(bytes: Uint8Array, mutation: (decoded: Uint8Array) => void): Promise<Uint8Array> {
  await Promise.all([MeshoptDecoder.ready, MeshoptEncoder.ready]);
  const input = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const decoded = new Uint8Array(input.getUint32(16, true));
  MeshoptDecoder.decodeVertexBuffer(decoded, decoded.length / 4, 4, bytes.subarray(HEADER_BYTES));
  mutation(decoded);
  const compressed = MeshoptEncoder.encodeVertexBuffer(decoded, decoded.length / 4, 4);
  const output = new Uint8Array(HEADER_BYTES + compressed.length);
  output.set(bytes.subarray(0, HEADER_BYTES));
  output.set(compressed, HEADER_BYTES);
  const header = new DataView(output.buffer);
  header.setUint32(20, compressed.length, true);
  header.setUint32(32, hash(decoded), true);
  return output;
}

describe('lossless geometry pages', () => {
  it('round-trips every rest byte and optional stream, with independently decodable pages', async () => {
    for (const [withUv, withTangent, stride] of [
      [false, false, 80], [true, false, 88], [false, true, 96], [true, true, 104],
    ] as const) {
      const hierarchy = fixture(withUv, withTangent);
      const result = await encodeGeometryPages(hierarchy, { maxPageBytes: 350 });
      expect(result.vertexStride).toBe(stride);
      expect(result.clusterPages).toEqual(stride === 104 ? [0, 1, 2] : expect.any(Array));
      expect(result.rootPages).toEqual([...new Set([result.clusterPages[0], result.clusterPages[2]])].sort());
      for (const page of result.pages) {
        expect(page.decodedBytes).toBeLessThanOrEqual(350);
        expect(page.bytes.byteLength).toBeGreaterThan(HEADER_BYTES);
        const decoded = await decodeGeometryPage(page.bytes, { maxPageBytes: 350 });
        expect(decoded.vertexStride).toBe(stride);
        expect(decoded.hasUvs).toBe(withUv);
        expect(decoded.hasTangents).toBe(withTangent);
        expect(decoded.clusters.map((cluster) => cluster.id)).toEqual(page.clusterIds);
        for (const cluster of decoded.clusters) {
          const sourceIds = hierarchy.clusters[cluster.id].vertices;
          expect(Array.from(cluster.sourceVertexIds)).toEqual(Array.from(sourceIds));
          expect(Array.from(cluster.localTriangles)).toEqual([0, 1, 2]);
          for (let local = 0; local < sourceIds.length; local++) {
            const original = new Uint8Array(hierarchy.source.vertices, sourceIds[local] * REST_BYTES, REST_BYTES);
            expect(Array.from(cluster.rest.subarray(local * REST_BYTES, (local + 1) * REST_BYTES)))
              .toEqual(Array.from(original));
          }
          expect(cluster.uvs ? Array.from(cluster.uvs) : undefined)
            .toEqual(withUv ? Array.from(sourceIds).flatMap((id) => Array.from(hierarchy.source.uvs!.subarray(id * 2, id * 2 + 2))) : undefined);
          expect(cluster.tangents ? Array.from(cluster.tangents) : undefined)
            .toEqual(withTangent ? Array.from(sourceIds).flatMap((id) => Array.from(hierarchy.source.tangents!.subarray(id * 4, id * 4 + 4))) : undefined);
        }
      }
    }
  });

  it('packs deterministically and maps terminal roots to unique resident pages', async () => {
    const hierarchy = fixture();
    const first = await encodeGeometryPages(hierarchy, { maxPageBytes: 350 });
    const second = await encodeGeometryPages(hierarchy, { maxPageBytes: 350 });
    expect(first.pages).toHaveLength(3);
    expect(first.clusterPages).toEqual([0, 1, 2]);
    expect(first.rootPages).toEqual([0, 2]);
    expect(first.pages.map(page => Array.from(page.bytes))).toEqual(second.pages.map(page => Array.from(page.bytes)));
    const standalone = await decodeGeometryPage(first.pages[2].bytes, { maxPageBytes: 350 });
    expect(standalone.clusters.map(cluster => cluster.id)).toEqual([2]);
  });

  it('rejects a cluster too large for a page and caps encoded plus decoded totals', async () => {
    await expect(encodeGeometryPages(fixture(), { maxPageBytes: 343 })).rejects.toThrow(/maxPageBytes/);
    await expect(encodeGeometryPages(fixture(), { maxPageBytes: 350, maxTotalBytes: 400 }))
      .rejects.toThrow(/maxTotalBytes/);
    await expect(encodeGeometryPages(fixture(), { maxPageBytes: 4 * 1024 * 1024 + 1 }))
      .rejects.toThrow(/maxPageBytes/);
    const bad = fixture();
    bad.clusters[0].triangles[2] = 3;
    await expect(encodeGeometryPages(bad)).rejects.toThrow(/local triangle/);
    const nan = fixture();
    new DataView(nan.source.vertices).setFloat32(0, Number.NaN, true);
    await expect(encodeGeometryPages(nan)).rejects.toThrow(/position/);
    const badNormal = fixture();
    new DataView(badNormal.source.vertices).setFloat32(4 * 4, Number.NaN, true);
    await expect(encodeGeometryPages(badNormal)).rejects.toThrow(/normal/);
    const badWeight = fixture();
    new DataView(badWeight.source.vertices).setFloat32(12 * 4, Infinity, true);
    await expect(encodeGeometryPages(badWeight)).rejects.toThrow(/weight/);
  });

  it('rejects untrusted lengths, corruption, duplicate IDs, invalid source IDs and NaN positions', async () => {
    const page = (await encodeGeometryPages(fixture(), { maxPageBytes: 700 })).pages[0].bytes;
    const truncated = page.subarray(0, page.length - 1);
    await expect(decodeGeometryPage(truncated, { maxPageBytes: 700 })).rejects.toThrow(/length/);
    const excessive = page.slice();
    new DataView(excessive.buffer).setUint32(16, 0xffff_ffff, true);
    await expect(decodeGeometryPage(excessive, { maxPageBytes: 700 })).rejects.toThrow(/payload lengths/);
    const corrupted = page.slice();
    corrupted[32] ^= 1; // Header checksum disagrees with the lossless decoded payload.
    await expect(decodeGeometryPage(corrupted, { maxPageBytes: 700 })).rejects.toThrow(/checksum/);
    // Raw payload's first cluster is 344 bytes; the second ID starts immediately after it.
    const duplicate = await mutateDecodedPage(page, raw => new DataView(raw.buffer).setUint32(344, 0, true));
    await expect(decodeGeometryPage(duplicate, { maxPageBytes: 700 })).rejects.toThrow(/duplicate/);
    const invalidVertex = await mutateDecodedPage(page, raw => new DataView(raw.buffer).setUint32(16, 7, true));
    await expect(decodeGeometryPage(invalidVertex, { maxPageBytes: 700 })).rejects.toThrow(/source vertex ID/);
    const nanPosition = await mutateDecodedPage(page, raw => new DataView(raw.buffer).setFloat32(28, Number.NaN, true));
    await expect(decodeGeometryPage(nanPosition, { maxPageBytes: 700 })).rejects.toThrow(/position/);
    const nanNormal = await mutateDecodedPage(page, raw => new DataView(raw.buffer).setFloat32(28 + 4 * 4, Number.NaN, true));
    await expect(decodeGeometryPage(nanNormal, { maxPageBytes: 700 })).rejects.toThrow(/normal/);
    const infiniteWeight = await mutateDecodedPage(page, raw => new DataView(raw.buffer).setFloat32(28 + 12 * 4, Infinity, true));
    await expect(decodeGeometryPage(infiniteWeight, { maxPageBytes: 700 })).rejects.toThrow(/weight/);
  });
});
