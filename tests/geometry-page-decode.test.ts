import { describe, expect, it } from 'vitest';
import { encodeGeometryPages } from '../src/geometry-pages';
import type { GeometryHierarchy } from '../src/geometry-hierarchy-types';
import { decodeVerifiedGeometryPage, geometryPageTransferBuffers } from '../src/geometry-page-decode';

function fixture(withUvs: boolean, withTangents: boolean): GeometryHierarchy {
  const vertices = new ArrayBuffer(3 * 80);
  const view = new DataView(vertices);
  for (let i = 0; i < 3; i++) view.setFloat32(i * 80, i, true);
  const bounds = { min: [0, 0, 0] as [number, number, number],
    max: [2, 1, 0] as [number, number, number] };
  return {
    version: 1, builder: 'meshoptimizer-1.3.0', primitiveKey: 'verified-page',
    source: { vertices, vertexCount: 3, indices: new Uint32Array([0, 1, 2]), bounds,
      uvs: withUvs ? new Float32Array([0, 0, 1, 0, 0, 1]) : undefined,
      tangents: withTangents ? new Float32Array(12).fill(1) : undefined },
    options: {} as GeometryHierarchy['options'],
    clusters: [{ level: 0, vertices: new Uint32Array([0, 1, 2]),
      triangles: new Uint8Array([0, 1, 2]), bounds, lodBounds: bounds, error: 0,
      refinedGroup: null, group: 0 }],
    groups: [], roots: [0], sourceTriangleCount: 1,
  };
}

async function encodedFixture(withUvs = true, withTangents = true) {
  const page = (await encodeGeometryPages(fixture(withUvs, withTangents), { maxPageBytes: 2048 })).pages[0];
  const bytes = new Uint8Array(page.bytes);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const sha256 = Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
  return { descriptor: { id: page.id, encodedBytes: bytes.byteLength,
    decodedBytes: page.decodedBytes, sha256 }, bytes };
}

describe('verified geometry page decode', () => {
  it('checks the real encoded payload and exposes all optional transfer streams', async () => {
    for (const [withUvs, withTangents] of [[false, false], [true, false], [false, true], [true, true]]) {
      const { descriptor, bytes } = await encodedFixture(withUvs, withTangents);
      const decoded = await decodeVerifiedGeometryPage(descriptor, bytes);
      expect(decoded.hasUvs).toBe(withUvs);
      expect(decoded.hasTangents).toBe(withTangents);
      const cluster = decoded.clusters[0];
      const arrays = [cluster.sourceVertexIds, cluster.rest, cluster.uvs, cluster.tangents,
        cluster.localTriangles].filter(array => array !== undefined);
      const transfers = geometryPageTransferBuffers(decoded);
      expect(transfers).toHaveLength(3 + Number(withUvs) + Number(withTangents));
      expect(new Set(transfers).size).toBe(transfers.length);
      expect(transfers.reduce((sum, buffer) => sum + buffer.byteLength, 0))
        .toBe(arrays.reduce((sum, array) => sum + array.byteLength, 0));
      for (const array of arrays) expect(transfers).toContain(array.buffer);
    }
  });

  it('rejects wrong wire length and a corrupted hash with the stream error prefixes', async () => {
    const { descriptor, bytes } = await encodedFixture();
    await expect(decodeVerifiedGeometryPage(descriptor, bytes.subarray(0, -1)))
      .rejects.toThrow(/GEOMETRY_STREAM_ENCODED_LENGTH/);
    const corrupted = bytes.slice();
    corrupted[corrupted.length - 1] ^= 1;
    await expect(decodeVerifiedGeometryPage(descriptor, corrupted)).rejects.toThrow(/GEOMETRY_STREAM_HASH/);
  });

  it('honors abort before work and across the digest boundary', async () => {
    const { descriptor, bytes } = await encodedFixture();
    const before = new AbortController();
    before.abort();
    await expect(decodeVerifiedGeometryPage(descriptor, bytes, before.signal)).rejects.toMatchObject({ name: 'AbortError' });
    const during = new AbortController();
    const pending = decodeVerifiedGeometryPage(descriptor, bytes, during.signal);
    during.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('deduplicates aliases and excludes SharedArrayBuffer-backed arrays', async () => {
    const { descriptor, bytes } = await encodedFixture();
    const decoded = await decodeVerifiedGeometryPage(descriptor, bytes);
    const cluster = decoded.clusters[0];
    cluster.localTriangles = cluster.rest.subarray(0, 3);
    expect(geometryPageTransferBuffers(decoded)).toHaveLength(4);
    if (typeof SharedArrayBuffer !== 'undefined') {
      cluster.rest = new Uint8Array(new SharedArrayBuffer(cluster.rest.byteLength));
      expect(geometryPageTransferBuffers(decoded)).toHaveLength(4);
      expect(geometryPageTransferBuffers(decoded)).not.toContain(cluster.rest.buffer);
    }
  });
});
