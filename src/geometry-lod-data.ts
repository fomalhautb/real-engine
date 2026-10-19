import type { GeometryHierarchy, GeometryCluster } from './geometry-hierarchy-types';
import type { Bounds3 } from './deformation';

export interface PackedGeometryLod {
  clusters: ArrayBuffer;
  groups: ArrayBuffer;
  /** Four u32 values per local cluster vertex: source ID, cluster ID, level, stable color seed. */
  vertices: Uint32Array;
  /** Global packed-vertex indices for every cluster triangle at every level. */
  indices: Uint32Array;
  sourceVertexCount: number;
  sourceTriangleCount: number;
  clusterVertexCount: number;
  clusterCount: number;
  groupCount: number;
}

const CLUSTER_BYTES = 64;
const GROUP_BYTES = 32;
const MAX_CLUSTERS = 65_536;
const MAX_GROUPS = 65_536;
const MAX_TRIANGLES = 3_000_000;
const MAX_VERTICES = 4_000_000;
const MAX_CLUSTER_TRIANGLES = 512;
const REST_BYTES = 80;
const ROOT_SENTINEL = 0xffff_ffff;

function fail(label: string): never { throw new RangeError(`GEOMETRY_LOD_${label}`); }

function integer(value: number, min: number, max: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail(label);
  return value;
}

function finite32(value: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) fail(label);
  return Math.fround(value);
}

function validatedBounds(bounds: Bounds3, label: string): Bounds3 {
  if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max)
    || bounds.min.length !== 3 || bounds.max.length !== 3) fail(label);
  const min = bounds.min.map((value) => finite32(value, label)) as [number, number, number];
  const max = bounds.max.map((value) => finite32(value, label)) as [number, number, number];
  for (let axis = 0; axis < 3; axis++) {
    if (min[axis] > max[axis] || min[axis] > bounds.min[axis] || max[axis] < bounds.max[axis]) fail(label);
  }
  return { min, max };
}

function contains(outer: Bounds3, inner: Bounds3): boolean {
  return [0, 1, 2].every((axis) => outer.min[axis] <= inner.min[axis] && outer.max[axis] >= inner.max[axis]);
}

function containsPoint(bounds: Bounds3, position: readonly number[]): boolean {
  return [0, 1, 2].every((axis) => bounds.min[axis] <= position[axis] && position[axis] <= bounds.max[axis]);
}

function sameOrientedTriangle(a: readonly number[], b: readonly number[]): boolean {
  return (a[0] === b[0] && a[1] === b[1] && a[2] === b[2])
    || (a[0] === b[1] && a[1] === b[2] && a[2] === b[0])
    || (a[0] === b[2] && a[1] === b[0] && a[2] === b[1]);
}

/** FNV-1a over UTF-8 primitive key, a separator, then little-endian cluster ID. */
function stableSeed(primitiveKey: string, clusterId: number): number {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(primitiveKey)) hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  for (let shift = 0; shift < 32; shift += 8) hash = Math.imul(hash ^ ((clusterId >>> shift) & 255), 0x01000193) >>> 0;
  return hash;
}

function boundary(ids: readonly number[], clusters: readonly GeometryCluster[], canonical: (id: number) => number): Map<string, number> {
  const edges = new Map<string, number>();
  for (const id of ids) {
    const cluster = clusters[id];
    for (let i = 0; i < cluster.triangles.length; i += 3) {
      const triangle = [0, 1, 2].map((corner) => canonical(cluster.vertices[cluster.triangles[i + corner]]));
      for (let edge = 0; edge < 3; edge++) {
        const a = triangle[edge], b = triangle[(edge + 1) % 3];
        if (a === b) continue;
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        const count = (edges.get(key) ?? 0) + (a < b ? 1 : -1);
        if (count === 0) edges.delete(key);
        else edges.set(key, count);
      }
    }
  }
  return edges;
}

function sameBoundary(a: Map<string, number>, b: Map<string, number>): boolean {
  return a.size === b.size && [...a].every(([edge, count]) => b.get(edge) === count);
}

/** Validate a builder result before allocating GPU upload buffers. */
export function packGeometryLod(hierarchy: GeometryHierarchy): PackedGeometryLod {
  if (!hierarchy || hierarchy.version !== 1 || hierarchy.builder !== 'meshoptimizer-1.3.0' || !hierarchy.source
    || typeof hierarchy.primitiveKey !== 'string' || hierarchy.primitiveKey.length === 0 || hierarchy.primitiveKey.length > 1024
    || !Array.isArray(hierarchy.clusters) || !Array.isArray(hierarchy.groups) || !Array.isArray(hierarchy.roots)) {
    fail('HIERARCHY');
  }
  const clusterCount = integer(hierarchy.clusters.length, 1, MAX_CLUSTERS, 'CLUSTER_BUDGET');
  const groupCount = integer(hierarchy.groups.length, 1, MAX_GROUPS, 'GROUP_BUDGET');
  const source = hierarchy.source;
  const sourceVertexCount = integer(source.vertexCount, 1, MAX_VERTICES, 'SOURCE_VERTEX_BUDGET');
  if (!(source.vertices instanceof ArrayBuffer) || source.vertices.byteLength !== sourceVertexCount * REST_BYTES
    || !(source.indices instanceof Uint32Array) || source.indices.length % 3 !== 0) fail('SOURCE_LAYOUT');
  if (source.uvs !== undefined && (!(source.uvs instanceof Float32Array) || source.uvs.length !== sourceVertexCount * 2
    || !source.uvs.every(Number.isFinite))) fail('SOURCE_UVS');
  if (source.tangents !== undefined && (!(source.tangents instanceof Float32Array) || source.tangents.length !== sourceVertexCount * 4
    || !source.tangents.every(Number.isFinite))) fail('SOURCE_TANGENTS');
  if (source.colors !== undefined && (!(source.colors instanceof Float32Array) || source.colors.length !== sourceVertexCount * 4
    || !source.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1))) fail('SOURCE_COLORS');
  const sourceTriangleCount = integer(source.indices.length / 3, 1, MAX_TRIANGLES, 'SOURCE_TRIANGLE_BUDGET');
  if (hierarchy.sourceTriangleCount !== sourceTriangleCount) fail('SOURCE_TRIANGLE_COUNT');

  let totalVertices = 0, totalTriangles = 0;
  for (const cluster of hierarchy.clusters) {
    if (!cluster || !(cluster.vertices instanceof Uint32Array) || !(cluster.triangles instanceof Uint8Array)
      || cluster.vertices.length < 3 || cluster.vertices.length > 256
      || cluster.triangles.length < 3 || cluster.triangles.length % 3 !== 0
      || cluster.triangles.length / 3 > MAX_CLUSTER_TRIANGLES) fail('CLUSTER_LAYOUT');
    totalVertices += cluster.vertices.length;
    totalTriangles += cluster.triangles.length / 3;
    if (totalVertices > MAX_VERTICES) fail('PACKED_VERTEX_BUDGET');
    if (totalTriangles > MAX_TRIANGLES) fail('PACKED_TRIANGLE_BUDGET');
  }

  const sourceView = new DataView(source.vertices);
  for (let vertex = 0; vertex < sourceVertexCount; vertex++) {
    for (let field = 0; field < 20; field++) {
      if (field >= 8 && field < 12) continue; // Joint IDs are u32, not f32.
      finite32(sourceView.getFloat32(vertex * REST_BYTES + field * 4, true), 'SOURCE_VERTEX');
    }
  }
  const positions = new Map<number, [number, number, number]>();
  const position = (id: number): [number, number, number] => {
    integer(id, 0, sourceVertexCount - 1, 'SOURCE_VERTEX_ID');
    let value = positions.get(id);
    if (!value) {
      value = [0, 1, 2].map((axis) => finite32(sourceView.getFloat32(id * REST_BYTES + axis * 4, true), 'SOURCE_POSITION')) as [number, number, number];
      positions.set(id, value);
    }
    return value;
  };
  const sourceBounds = validatedBounds(source.bounds, 'SOURCE_BOUNDS');
  for (const id of source.indices) if (!containsPoint(sourceBounds, position(id))) fail('SOURCE_BOUNDS');

  const clusterBounds: Bounds3[] = [], lodBounds: Bounds3[] = [];
  const leafTriangles = new Uint8Array(sourceTriangleCount);
  for (let id = 0; id < clusterCount; id++) {
    const cluster = hierarchy.clusters[id];
    integer(cluster.level, 0, 65_535, 'CLUSTER_LEVEL');
    integer(cluster.group, 0, groupCount - 1, 'CLUSTER_GROUP');
    if (cluster.refinedGroup !== null) integer(cluster.refinedGroup, 0, groupCount - 1, 'REFINED_GROUP');
    if (cluster.level === 0 ? cluster.refinedGroup !== null : cluster.refinedGroup === null) fail('REFINED_GROUP');
    const tight = validatedBounds(cluster.bounds, 'CLUSTER_BOUNDS');
    const inherited = validatedBounds(cluster.lodBounds, 'LOD_BOUNDS');
    if (!contains(inherited, tight) || !contains(sourceBounds, tight)) fail('INHERITED_BOUNDS');
    clusterBounds.push(tight); lodBounds.push(inherited);
    const error = finite32(cluster.error, 'CLUSTER_ERROR');
    if (error < 0) fail('CLUSTER_ERROR');
    const localIds = new Set<number>();
    for (const sourceId of cluster.vertices) {
      if (sourceId >= sourceVertexCount || localIds.has(sourceId)) fail('SOURCE_VERTEX_ID');
      localIds.add(sourceId);
      if (!containsPoint(tight, position(sourceId))) fail('CLUSTER_BOUNDS');
    }
    for (const local of cluster.triangles) if (local >= cluster.vertices.length) fail('LOCAL_TRIANGLE_INDEX');
    if (cluster.level === 0) {
      if (!(cluster.sourceTriangles instanceof Uint32Array)
        || cluster.sourceTriangles.length !== cluster.triangles.length / 3) fail('LEAF_SOURCE_TRIANGLES');
      for (let triangle = 0; triangle < cluster.sourceTriangles.length; triangle++) {
        const sourceId = cluster.sourceTriangles[triangle];
        if (sourceId >= sourceTriangleCount || leafTriangles[sourceId]) fail('LEAF_SOURCE_COVERAGE');
        leafTriangles[sourceId] = 1;
        const offset = triangle * 3;
        const actual = [0, 1, 2].map((corner) => cluster.vertices[cluster.triangles[offset + corner]]);
        const expected = Array.from(source.indices.subarray(sourceId * 3, sourceId * 3 + 3));
        if (!sameOrientedTriangle(actual, expected)) fail('LEAF_SOURCE_COVERAGE');
      }
    }
  }
  if (leafTriangles.some((covered) => covered !== 1)) fail('LEAF_SOURCE_COVERAGE');

  const groupBounds: Bounds3[] = [];
  const childOwner = new Int32Array(clusterCount).fill(-1);
  const parentOwner = new Int32Array(clusterCount).fill(-1);
  const terminalChildren = new Set<number>();
  const positionIds = new Map<string, number>();
  const canonical = (id: number): number => {
    const key = position(id).join(',');
    let canonicalId = positionIds.get(key);
    if (canonicalId === undefined) { canonicalId = positionIds.size; positionIds.set(key, canonicalId); }
    return canonicalId;
  };
  for (let id = 0; id < groupCount; id++) {
    const group = hierarchy.groups[id];
    if (!group || !Array.isArray(group.children) || !Array.isArray(group.parents)
      || group.children.length === 0 || typeof group.terminal !== 'boolean') fail('GROUP_LAYOUT');
    integer(group.level, 0, 65_535, 'GROUP_LEVEL');
    const bounds = validatedBounds(group.bounds, 'GROUP_BOUNDS');
    groupBounds.push(bounds);
    const error = finite32(group.error, 'GROUP_ERROR');
    const localError = finite32(group.localError, 'GROUP_LOCAL_ERROR');
    if (error < 0 || localError < 0 || error < localError
      || (!group.terminal && error <= 0)) fail('GROUP_ERROR');
    let childTriangleCount = 0, parentTriangleCount = 0;
    for (const child of group.children) {
      integer(child, 0, clusterCount - 1, 'GROUP_CHILD');
      if (childOwner[child] !== -1 || hierarchy.clusters[child].group !== id
        || hierarchy.clusters[child].level !== group.level
        || hierarchy.clusters[child].error > group.error
        || !contains(bounds, lodBounds[child])) fail('GROUP_CHILD_RELATION');
      childOwner[child] = id;
      childTriangleCount += hierarchy.clusters[child].triangles.length / 3;
      if (group.terminal) terminalChildren.add(child);
    }
    if (group.terminal ? group.parents.length !== 0 : group.parents.length === 0) fail('GROUP_TERMINAL');
    for (const parent of group.parents) {
      integer(parent, 0, clusterCount - 1, 'GROUP_PARENT');
      if (parentOwner[parent] !== -1 || hierarchy.clusters[parent].refinedGroup !== id
        || hierarchy.clusters[parent].level !== group.level + 1
        || hierarchy.clusters[parent].error < group.error
        || !contains(lodBounds[parent], bounds)) fail('GROUP_PARENT_RELATION');
      parentOwner[parent] = id;
      parentTriangleCount += hierarchy.clusters[parent].triangles.length / 3;
    }
    if (!group.terminal) {
      if (parentTriangleCount > childTriangleCount) fail('GROUP_TRIANGLE_GROWTH');
      if (!sameBoundary(boundary(group.children, hierarchy.clusters, canonical),
        boundary(group.parents, hierarchy.clusters, canonical))) fail('GROUP_BOUNDARY');
    }
  }
  for (let id = 0; id < clusterCount; id++) {
    if (childOwner[id] === -1 || (hierarchy.clusters[id].refinedGroup === null ? parentOwner[id] !== -1 : parentOwner[id] === -1)) {
      fail('DAG_RELATION');
    }
  }
  const roots = new Set<number>();
  for (const root of hierarchy.roots) {
    integer(root, 0, clusterCount - 1, 'ROOT');
    if (roots.has(root) || !terminalChildren.has(root)) fail('ROOT');
    roots.add(root);
  }
  if (roots.size !== terminalChildren.size) fail('ROOT_COVERAGE');

  const clusterBytes = new ArrayBuffer(clusterCount * CLUSTER_BYTES);
  const groupBytes = new ArrayBuffer(groupCount * GROUP_BYTES);
  const clusterView = new DataView(clusterBytes), groupView = new DataView(groupBytes);
  const vertices = new Uint32Array(totalVertices * 4), indices = new Uint32Array(totalTriangles * 3);
  let vertexBase = 0, indexOffset = 0;
  for (let id = 0; id < clusterCount; id++) {
    const cluster = hierarchy.clusters[id], bounds = clusterBounds[id];
    const byte = id * CLUSTER_BYTES;
    for (let axis = 0; axis < 3; axis++) {
      clusterView.setFloat32(byte + axis * 4, bounds.min[axis], true);
      clusterView.setFloat32(byte + 16 + axis * 4, bounds.max[axis], true);
    }
    clusterView.setUint32(byte + 32, indexOffset, true);
    clusterView.setUint32(byte + 36, cluster.triangles.length, true);
    clusterView.setUint32(byte + 40, cluster.group, true);
    clusterView.setUint32(byte + 44, cluster.refinedGroup ?? ROOT_SENTINEL, true);
    clusterView.setUint32(byte + 48, vertexBase, true);
    clusterView.setUint32(byte + 52, cluster.vertices.length, true);
    clusterView.setUint32(byte + 56, cluster.level, true);
    clusterView.setUint32(byte + 60, id, true);
    const seed = stableSeed(hierarchy.primitiveKey, id);
    for (let local = 0; local < cluster.vertices.length; local++) {
      vertices.set([cluster.vertices[local], id, cluster.level, seed], (vertexBase + local) * 4);
    }
    for (const local of cluster.triangles) indices[indexOffset++] = vertexBase + local;
    vertexBase += cluster.vertices.length;
  }
  for (let id = 0; id < groupCount; id++) {
    const group = hierarchy.groups[id], bounds = groupBounds[id], byte = id * GROUP_BYTES;
    for (let axis = 0; axis < 3; axis++) {
      groupView.setFloat32(byte + axis * 4, bounds.min[axis], true);
      groupView.setFloat32(byte + 16 + axis * 4, bounds.max[axis], true);
    }
    groupView.setFloat32(byte + 12, Math.fround(group.error), true);
    groupView.setFloat32(byte + 28, group.terminal ? 1 : 0, true);
  }
  return { clusters: clusterBytes, groups: groupBytes, vertices, indices, sourceVertexCount,
    sourceTriangleCount, clusterVertexCount: totalVertices, clusterCount, groupCount };
}
