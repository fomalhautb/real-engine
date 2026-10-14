import { describe, expect, it } from 'vitest';
import { createCube } from '../src/geometry';
import { buildGeometryHierarchy } from '../src/geometry-hierarchy';
import type { GeometryHierarchy } from '../src/geometry-hierarchy-types';
import { packGeometryLod } from '../src/geometry-lod-data';

function fixture(): GeometryHierarchy {
  const vertices = new ArrayBuffer(4 * 80);
  const view = new DataView(vertices);
  const points = [[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 1, 0]];
  for (let id = 0; id < points.length; id++) {
    for (let axis = 0; axis < 3; axis++) view.setFloat32(id * 80 + axis * 4, points[id][axis], true);
  }
  const bounds = () => ({ min: [0, 0, 0] as [number, number, number], max: [1, 1, 0] as [number, number, number] });
  const firstTightBounds = bounds();
  firstTightBounds.max[1] = 0.5;
  return {
    version: 1,
    builder: 'meshoptimizer-1.3.0',
    primitiveKey: 'fixture/square',
    source: { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4, bounds: bounds() },
    options: {} as GeometryHierarchy['options'],
    clusters: [
      { level: 0, vertices: new Uint32Array([0, 1, 2]), triangles: new Uint8Array([0, 1, 2]),
        sourceTriangles: new Uint32Array([0]), bounds: firstTightBounds, lodBounds: bounds(), error: 0,
        refinedGroup: null, group: 0 },
      { level: 0, vertices: new Uint32Array([0, 2, 3]), triangles: new Uint8Array([0, 1, 2]),
        sourceTriangles: new Uint32Array([1]), bounds: bounds(), lodBounds: bounds(), error: 0,
        refinedGroup: null, group: 0 },
      { level: 1, vertices: new Uint32Array([0, 1, 2, 3]), triangles: new Uint8Array([0, 1, 2, 0, 2, 3]),
        bounds: bounds(), lodBounds: bounds(), error: 1, refinedGroup: 0, group: 1 },
    ],
    groups: [
      { level: 0, children: [0, 1], parents: [2], bounds: bounds(), localError: 0.5, error: 1, terminal: false },
      { level: 1, children: [2], parents: [], bounds: bounds(), localError: 0, error: 1, terminal: true },
    ],
    roots: [2],
    sourceTriangleCount: 2,
  };
}

describe('packGeometryLod', () => {
  it('accepts the offline builder output with its terminal root group', async () => {
    const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'builder/cube', mesh: createCube(), coverage: 'opaque' });
    const packed = packGeometryLod(hierarchy);
    expect(packed.clusterCount).toBe(hierarchy.clusters.length);
    expect(packed.groupCount).toBe(hierarchy.groups.length);
    expect(packed.sourceTriangleCount).toBe(hierarchy.sourceTriangleCount);
    expect(packed.indices.length).toBe(hierarchy.clusters.reduce((count, cluster) => count + cluster.triangles.length, 0));
  });

  it('packs the fixed GPU layout and stable cluster-local vertex identity without mutating its input', () => {
    const hierarchy = fixture();
    const sourceBefore = hierarchy.source.vertices.slice(0);
    const packed = packGeometryLod(hierarchy);
    expect(packed).toMatchObject({ sourceVertexCount: 4, sourceTriangleCount: 2,
      clusterVertexCount: 10, clusterCount: 3, groupCount: 2 });
    expect(packed.clusters.byteLength).toBe(3 * 64);
    expect(packed.groups.byteLength).toBe(2 * 32);
    expect(packed.vertices.length).toBe(10 * 4);
    expect([...packed.indices]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 8, 9]);

    const cluster = new DataView(packed.clusters);
    expect(cluster.getFloat32(0, true)).toBe(0);
    expect(cluster.getFloat32(16, true)).toBe(1);
    expect(cluster.getFloat32(20, true)).toBe(0.5);
    expect(cluster.getUint32(32, true)).toBe(0);
    expect(cluster.getUint32(36, true)).toBe(3);
    expect(cluster.getUint32(40, true)).toBe(0);
    expect(cluster.getUint32(44, true)).toBe(0xffff_ffff);
    expect(cluster.getUint32(48, true)).toBe(0);
    expect(cluster.getUint32(52, true)).toBe(3);
    expect(cluster.getUint32(56, true)).toBe(0);
    expect(cluster.getUint32(60, true)).toBe(0);
    expect(cluster.getUint32(2 * 64 + 32, true)).toBe(6);
    expect(cluster.getUint32(2 * 64 + 36, true)).toBe(6);
    expect(cluster.getUint32(2 * 64 + 40, true)).toBe(1);
    expect(cluster.getUint32(2 * 64 + 44, true)).toBe(0);
    expect(cluster.getUint32(2 * 64 + 48, true)).toBe(6);
    expect(cluster.getUint32(2 * 64 + 52, true)).toBe(4);
    expect(cluster.getUint32(2 * 64 + 56, true)).toBe(1);
    expect(cluster.getUint32(2 * 64 + 60, true)).toBe(2);

    const group = new DataView(packed.groups);
    expect(group.getFloat32(12, true)).toBe(1);
    expect(group.getFloat32(20, true)).toBe(1);
    expect(group.getFloat32(28, true)).toBe(0);
    expect(group.getFloat32(32 + 28, true)).toBe(1);
    expect([...packed.vertices.slice(6 * 4, 6 * 4 + 3)]).toEqual([0, 2, 1]);
    expect(packed.vertices[3]).toBe(packGeometryLod(hierarchy).vertices[3]);
    expect(packed.vertices[3]).not.toBe(packed.vertices[3 * 4 + 3]);
    hierarchy.primitiveKey = 'different-key';
    expect(packed.vertices[3]).not.toBe(packGeometryLod(hierarchy).vertices[3]);
    expect(hierarchy.source.vertices).toEqual(sourceBefore);
  });

  it('rejects broken DAG ownership, root coverage, and nonmonotonic error/bounds', () => {
    const duplicatedChild = fixture();
    duplicatedChild.groups[0].children.push(0);
    expect(() => packGeometryLod(duplicatedChild)).toThrow('GROUP_CHILD_RELATION');
    const wrongParent = fixture();
    wrongParent.clusters[2].refinedGroup = 1;
    expect(() => packGeometryLod(wrongParent)).toThrow('GROUP_PARENT_RELATION');
    const missingRoot = fixture();
    missingRoot.roots = [];
    expect(() => packGeometryLod(missingRoot)).toThrow('ROOT_COVERAGE');
    const zeroError = fixture();
    zeroError.groups[0].error = 0;
    zeroError.groups[0].localError = 0;
    expect(() => packGeometryLod(zeroError)).toThrow('GROUP_ERROR');
    const nonmonotonic = fixture();
    nonmonotonic.clusters[2].error = 0.5;
    expect(() => packGeometryLod(nonmonotonic)).toThrow('GROUP_PARENT_RELATION');
    const looseCluster = fixture();
    looseCluster.clusters[0].bounds.max[0] = 2;
    expect(() => packGeometryLod(looseCluster)).toThrow('INHERITED_BOUNDS');
    const missingInheritedBounds = fixture();
    missingInheritedBounds.clusters[2].lodBounds.max[0] = 0.5;
    expect(() => packGeometryLod(missingInheritedBounds)).toThrow('INHERITED_BOUNDS');
  });

  it('rejects invalid local indices, incomplete leaf coverage, and changed coarse geometry', () => {
    const index = fixture();
    index.clusters[0].triangles[2] = 3;
    expect(() => packGeometryLod(index)).toThrow('LOCAL_TRIANGLE_INDEX');
    const missingSource = fixture();
    missingSource.clusters[1].sourceTriangles![0] = 0;
    expect(() => packGeometryLod(missingSource)).toThrow('LEAF_SOURCE_COVERAGE');
    const changedBoundary = fixture();
    changedBoundary.clusters[2].triangles.set([0, 1, 2, 0, 1, 2]);
    expect(() => packGeometryLod(changedBoundary)).toThrow('GROUP_BOUNDARY');
    const growth = fixture();
    growth.clusters[2].triangles = new Uint8Array([0, 1, 2, 0, 2, 3, 0, 1, 3]);
    expect(() => packGeometryLod(growth)).toThrow('GROUP_TRIANGLE_GROWTH');
  });

  it('validates the complete owned source streams and cluster triangle limit', () => {
    const invalidRest = fixture();
    new DataView(invalidRest.source.vertices).setFloat32(16, Number.NaN, true);
    expect(() => packGeometryLod(invalidRest)).toThrow('SOURCE_VERTEX');
    const invalidUv = fixture();
    invalidUv.source.uvs = new Float32Array([0, 0]);
    expect(() => packGeometryLod(invalidUv)).toThrow('SOURCE_UVS');
    const invalidTangent = fixture();
    invalidTangent.source.tangents = new Float32Array(16);
    invalidTangent.source.tangents[5] = Infinity;
    expect(() => packGeometryLod(invalidTangent)).toThrow('SOURCE_TANGENTS');
    const oversizedCluster = fixture();
    oversizedCluster.clusters[2].triangles = new Uint8Array(513 * 3);
    expect(() => packGeometryLod(oversizedCluster)).toThrow('CLUSTER_LAYOUT');
  });

  it('enforces hard budgets before allocation', () => {
    const tooManyClusters = fixture();
    tooManyClusters.clusters = Array(65_537).fill(tooManyClusters.clusters[0]);
    expect(() => packGeometryLod(tooManyClusters)).toThrow('CLUSTER_BUDGET');
    const tooManyGroups = fixture();
    tooManyGroups.groups = Array(65_537).fill(tooManyGroups.groups[0]);
    expect(() => packGeometryLod(tooManyGroups)).toThrow('GROUP_BUDGET');
    const tooManyVertices = fixture();
    tooManyVertices.clusters = Array(15_626).fill({ ...tooManyVertices.clusters[0],
      vertices: new Uint32Array(256) });
    expect(() => packGeometryLod(tooManyVertices)).toThrow('PACKED_VERTEX_BUDGET');
    const tooManyTriangles = fixture();
    tooManyTriangles.clusters = Array(6_000).fill({ ...tooManyTriangles.clusters[0],
      triangles: new Uint8Array(512 * 3) });
    expect(() => packGeometryLod(tooManyTriangles)).toThrow('PACKED_TRIANGLE_BUDGET');
  });
});
