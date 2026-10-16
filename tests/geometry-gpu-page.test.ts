import { describe, expect, it } from 'vitest';
import { GEOMETRY_GPU_VERTEX_BYTES, packGeometryGpuPage } from '../src/geometry-gpu-page';
import type { DecodedGeometryPage } from '../src/geometry-pages';
import { validateGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';

function fixture(hasUvs = true, hasTangents = true): { directory: GeometryStreamDirectory; decoded: DecodedGeometryPage } {
  const stride = 80 + (hasUvs ? 8 : 0) + (hasTangents ? 16 : 0);
  const bounds = { min: [0, 0, -3] as [number, number, number], max: [3, 6, 0] as [number, number, number] };
  const clusterBytes = 16 + 3 * (4 + stride) + 4;
  const directory = validateGeometryStreamDirectory({
    version: 1, primitiveKey: 'gpu/fixture', sourceVertexCount: 4, sourceTriangleCount: 2, vertexStride: stride,
    clusters: [0, 1].map(() => ({ level: 0, group: 0, refinedGroup: null, vertexCount: 3,
      triangleCount: 1, bounds, lodBounds: bounds, error: 0 })),
    groups: [{ level: 0, children: [0, 1], parents: [], bounds, error: 0, terminal: true }],
    roots: [0, 1], pages: [{ id: 0, clusterIds: [0, 1], encodedBytes: 41,
      decodedBytes: clusterBytes * 2, sha256: '0'.repeat(64) }],
    clusterPages: [0, 0], rootPages: [0],
  });
  const clusters = [[0, 1, 2], [2, 3, 0]].map((ids, id) => {
    const rest = new Uint8Array(ids.length * 80), view = new DataView(rest.buffer);
    for (let local = 0; local < ids.length; local++) {
      const source = ids[local], base = local * 80;
      for (let word = 0; word < 20; word++) view.setFloat32(base + word * 4, source + word / 16, true);
      view.setFloat32(base, source, true);
      view.setFloat32(base + 4, source * 2, true);
      view.setFloat32(base + 8, -source, true);
      for (let joint = 0; joint < 4; joint++) view.setUint32(base + 32 + joint * 4, 0xff00_0100 + source * 4 + joint, true);
    }
    return { id, sourceVertexIds: new Uint32Array(ids), rest,
      uvs: hasUvs ? new Float32Array(ids.flatMap(source => [source / 8, source / 4])) : undefined,
      tangents: hasTangents ? new Float32Array(ids.flatMap(source => [source / 3, 0.25, -0.5, 1])) : undefined,
      localTriangles: new Uint8Array(id === 0 ? [2, 0, 1] : [1, 2, 0]) };
  });
  return { directory, decoded: { version: 1, vertexStride: stride, hasUvs, hasTangents,
    sourceVertexCount: 4, hierarchyClusterCount: 2, clusters } };
}

describe('physical geometry page packing', () => {
  it('preserves exact rest bytes and packs optional streams, flags and identity at fixed offsets', () => {
    expect(GEOMETRY_GPU_VERTEX_BYTES).toBe(128);
    for (const [uv, tangent, flags] of [[false, false, 0], [true, false, 1],
      [false, true, 2], [true, true, 3]] as const) {
      const { directory, decoded } = fixture(uv, tangent);
      const packed = packGeometryGpuPage(directory, 0, decoded);
      expect(packed.vertices.byteLength).toBe(6 * 128);
      const bytes = new Uint8Array(packed.vertices), view = new DataView(packed.vertices);
      for (let ordinal = 0; ordinal < 2; ordinal++) for (let local = 0; local < 3; local++) {
        const cluster = decoded.clusters[ordinal], offset = (ordinal * 3 + local) * 128;
        expect(bytes.slice(offset, offset + 80)).toEqual(cluster.rest.slice(local * 80, (local + 1) * 80));
        expect(view.getUint32(offset + 32, true)).toBe(0xff00_0100 + cluster.sourceVertexIds[local] * 4);
        expect(view.getFloat32(offset + 80, true)).toBe(uv ? cluster.uvs![local * 2] : 0);
        expect(view.getFloat32(offset + 84, true)).toBe(uv ? cluster.uvs![local * 2 + 1] : 0);
        expect(view.getUint32(offset + 88, true)).toBe(flags);
        expect(view.getUint32(offset + 92, true)).toBe(0);
        for (let axis = 0; axis < 4; axis++) {
          expect(view.getFloat32(offset + 96 + axis * 4, true)).toBe(tangent ? cluster.tangents![local * 4 + axis] : 0);
        }
        expect([112, 116, 120].map(at => view.getUint32(offset + at, true)))
          .toEqual([cluster.sourceVertexIds[local], ordinal, 0]);
        expect(view.getUint32(offset + 124, true)).toBe(ordinal === 0 ? 182926153 : 3132816440);
      }
      expect(Array.from(packed.indices)).toEqual([2, 0, 1, 4, 5, 3]);
      expect(packed.clusters).toEqual([
        { id: 0, vertexOffset: 0, vertexCount: 3, indexOffset: 0, indexCount: 3 },
        { id: 1, vertexOffset: 3, vertexCount: 3, indexOffset: 3, indexCount: 3 },
      ]);
      const again = packGeometryGpuPage(directory, 0, decoded);
      expect(new Uint8Array(again.vertices)).toEqual(bytes);
      expect(again.indices).toEqual(packed.indices);
    }
  });

  it('rejects spoofed identity, counts, malformed arrays, nonfinite values and out-of-bounds positions', () => {
    const make = () => fixture();
    {
      const { directory, decoded } = make();
      expect(() => packGeometryGpuPage(directory, -1, decoded)).toThrow(/PAGE_ID/);
      expect(() => packGeometryGpuPage(directory, 1, decoded)).toThrow(/PAGE_ID/);
    }
    for (const mutate of [
      (page: DecodedGeometryPage) => { page.version = 2 as 1; },
      (page: DecodedGeometryPage) => { page.vertexStride = 80; },
      (page: DecodedGeometryPage) => { page.hasUvs = false; },
      (page: DecodedGeometryPage) => { page.sourceVertexCount = 3; },
      (page: DecodedGeometryPage) => { page.hierarchyClusterCount = 3; },
      (page: DecodedGeometryPage) => { page.clusters.pop(); },
      (page: DecodedGeometryPage) => { page.clusters[0].id = 1; },
    ]) {
      const { directory, decoded } = make(); mutate(decoded);
      expect(() => packGeometryGpuPage(directory, 0, decoded)).toThrow(/IDENTITY|CLUSTER_LAYOUT/);
    }
    for (const mutate of [
      (page: DecodedGeometryPage) => { page.clusters[0].rest = new Uint8Array(80); },
      (page: DecodedGeometryPage) => { page.clusters[0].uvs = new Float32Array(2); },
      (page: DecodedGeometryPage) => { page.clusters[0].tangents = undefined; },
      (page: DecodedGeometryPage) => { page.clusters[0].localTriangles = new Uint8Array(2); },
      (page: DecodedGeometryPage) => { page.clusters[0].sourceVertexIds[1] = 0; },
      (page: DecodedGeometryPage) => { page.clusters[0].sourceVertexIds[1] = 4; },
      (page: DecodedGeometryPage) => { page.clusters[0].localTriangles[0] = 3; },
      (page: DecodedGeometryPage) => { new DataView(page.clusters[0].rest.buffer).setFloat32(0, NaN, true); },
      (page: DecodedGeometryPage) => { new DataView(page.clusters[0].rest.buffer).setFloat32(16, Infinity, true); },
      (page: DecodedGeometryPage) => { new DataView(page.clusters[0].rest.buffer).setFloat32(0, 4, true); },
      (page: DecodedGeometryPage) => { page.clusters[0].uvs![0] = NaN; },
      (page: DecodedGeometryPage) => { page.clusters[0].tangents![0] = Infinity; },
    ]) {
      const { directory, decoded } = make(); mutate(decoded);
      expect(() => packGeometryGpuPage(directory, 0, decoded)).toThrow(/GEOMETRY_GPU_PAGE_/);
    }
  });
});
