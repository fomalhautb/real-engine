import { packGeometryLod } from './geometry-lod-data';
import type { GeometryHierarchy } from './geometry-hierarchy-types';
import type { EncodedGeometryPages } from './geometry-pages';
import type { GeometryStreamDirectory } from './geometry-stream-types';

const MAX_CLUSTERS = 65_536;
const MAX_GROUPS = 65_536;
const MAX_PAGES = 65_536;
const MAX_DECODED_PAGE_BYTES = 4 * 1024 * 1024;
const MAX_ENCODED_PAGE_BYTES = 8 * 1024 * 1024 + 40;
const MAX_TOTAL_PAGE_BYTES = 512 * 1024 * 1024;

function fail(reason: string): never { throw new RangeError(`GEOMETRY_STREAM_DIRECTORY_${reason}`); }
function integer(value: unknown, min: number, max: number, reason: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) fail(reason);
  return value;
}
function finite(value: unknown, reason: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) fail(reason);
  return value;
}
function bounds(value: unknown, reason: string) {
  if (!value || typeof value !== 'object') fail(reason);
  const source = value as { min?: unknown; max?: unknown };
  if (!Array.isArray(source.min) || !Array.isArray(source.max) || source.min.length !== 3 || source.max.length !== 3) fail(reason);
  const min = source.min.map(v => finite(v, reason)) as [number, number, number];
  const max = source.max.map(v => finite(v, reason)) as [number, number, number];
  if (min.some((v, axis) => v > max[axis])) fail(reason);
  return { min, max };
}
function contains(outer: { min: readonly number[]; max: readonly number[] }, inner: { min: readonly number[]; max: readonly number[] }): boolean {
  return [0, 1, 2].every(axis => outer.min[axis] <= inner.min[axis] && outer.max[axis] >= inner.max[axis]);
}
function ids(value: unknown, maximum: number, reason: string, nonempty = false): number[] {
  if (!Array.isArray(value) || (nonempty && !value.length) || value.length > maximum) fail(reason);
  const result: number[] = [];
  for (let index = 0; index < value.length; index++) {
    if (!Object.hasOwn(value, index)) fail(reason);
    result.push(integer(value[index], 0, maximum - 1, reason));
  }
  if (new Set(result).size !== result.length) fail(reason);
  return result;
}
function freezeBounds(value: { min: [number, number, number]; max: [number, number, number] }) {
  Object.freeze(value.min); Object.freeze(value.max); return Object.freeze(value);
}

/** Validate untrusted compact metadata and retain no caller-owned mutable arrays. */
export function validateGeometryStreamDirectory(input: GeometryStreamDirectory): GeometryStreamDirectory {
  if (!input || input.version !== 1 || typeof input.primitiveKey !== 'string'
    || !input.primitiveKey.length || input.primitiveKey.length > 1024) fail('HEADER');
  const sourceVertexCount = integer(input.sourceVertexCount, 1, 16_000_000, 'SOURCE_VERTICES');
  const sourceTriangleCount = integer(input.sourceTriangleCount, 1, 3_000_000, 'SOURCE_TRIANGLES');
  const vertexStride = integer(input.vertexStride, 80, 104, 'VERTEX_STRIDE');
  if (![80, 88, 96, 104].includes(vertexStride)) fail('VERTEX_STRIDE');
  if (!Array.isArray(input.clusters) || !input.clusters.length || input.clusters.length > MAX_CLUSTERS
    || !Array.isArray(input.groups) || !input.groups.length || input.groups.length > MAX_GROUPS
    || !Array.isArray(input.pages) || !input.pages.length || input.pages.length > MAX_PAGES) fail('COUNTS');
  const clusterCount = input.clusters.length, groupCount = input.groups.length, pageCount = input.pages.length;
  let groupReferences = 0, pageReferences = 0;
  for (const group of input.groups) {
    if (!group || !Array.isArray(group.children) || !Array.isArray(group.parents)) fail('GROUP');
    groupReferences += group.children.length + group.parents.length;
    if (groupReferences > 2 * clusterCount) fail('GROUP_REFERENCE_BUDGET');
  }
  for (const page of input.pages) {
    if (!page || !Array.isArray(page.clusterIds)) fail('PAGE_CLUSTERS');
    pageReferences += page.clusterIds.length;
    if (pageReferences > clusterCount) fail('PAGE_REFERENCE_BUDGET');
  }
  const clusters = input.clusters.map((entry) => {
    if (!entry || typeof entry !== 'object') fail('CLUSTER');
    const level = integer(entry.level, 0, 31, 'CLUSTER_LEVEL');
    const group = integer(entry.group, 0, groupCount - 1, 'CLUSTER_GROUP');
    const refinedGroup = entry.refinedGroup === null ? null : integer(entry.refinedGroup, 0, groupCount - 1, 'REFINED_GROUP');
    if ((level === 0) !== (refinedGroup === null)) fail('REFINED_GROUP');
    const vertexCount = integer(entry.vertexCount, 3, Math.min(256, sourceVertexCount), 'CLUSTER_VERTICES');
    const triangleCount = integer(entry.triangleCount, 1, 512, 'CLUSTER_TRIANGLES');
    const tight = bounds(entry.bounds, 'CLUSTER_BOUNDS'), lod = bounds(entry.lodBounds, 'LOD_BOUNDS');
    if (!contains(lod, tight)) fail('LOD_BOUNDS');
    const error = finite(entry.error, 'CLUSTER_ERROR');
    if (error < 0) fail('CLUSTER_ERROR');
    return { level, group, refinedGroup, vertexCount, triangleCount, bounds: tight, lodBounds: lod, error };
  });
  const groups = input.groups.map((entry) => {
    if (!entry || typeof entry !== 'object' || typeof entry.terminal !== 'boolean') fail('GROUP');
    const level = integer(entry.level, 0, 31, 'GROUP_LEVEL');
    const children = ids(entry.children, clusterCount, 'GROUP_CHILDREN', true);
    const parents = ids(entry.parents, clusterCount, 'GROUP_PARENTS');
    const groupBounds = bounds(entry.bounds, 'GROUP_BOUNDS');
    const error = finite(entry.error, 'GROUP_ERROR');
    if (error < 0 || (entry.terminal ? parents.length !== 0 : parents.length === 0 || error === 0)) fail('GROUP_TERMINAL');
    return { level, children, parents, bounds: groupBounds, error, terminal: entry.terminal };
  });
  const childOwners = new Int32Array(clusterCount).fill(-1), parentOwners = new Int32Array(clusterCount).fill(-1);
  const terminalChildren = new Set<number>();
  let leafTriangles = 0, totalTriangles = 0;
  for (const cluster of clusters) {
    totalTriangles += cluster.triangleCount;
    if (cluster.level === 0) leafTriangles += cluster.triangleCount;
    if (totalTriangles > 3_000_000) fail('OUTPUT_TRIANGLE_BUDGET');
  }
  if (leafTriangles !== sourceTriangleCount) fail('LEAF_TRIANGLE_COUNT');
  for (let groupId = 0; groupId < groupCount; groupId++) {
    const group = groups[groupId];
    const childTriangles = group.children.reduce((sum, id) => sum + clusters[id].triangleCount, 0);
    const parentTriangles = group.parents.reduce((sum, id) => sum + clusters[id].triangleCount, 0);
    if (!group.terminal && parentTriangles > childTriangles) fail('GROUP_TRIANGLE_GROWTH');
    for (const childId of group.children) {
      const child = clusters[childId];
      if (childOwners[childId] !== -1 || child.group !== groupId || child.level !== group.level
        || child.error > group.error || !contains(group.bounds, child.lodBounds)) fail('CHILD_RELATION');
      childOwners[childId] = groupId;
      if (group.terminal) terminalChildren.add(childId);
    }
    for (const parentId of group.parents) {
      const parent = clusters[parentId];
      if (parentOwners[parentId] !== -1 || parent.refinedGroup !== groupId || parent.level !== group.level + 1
        || parent.error < group.error || !contains(parent.lodBounds, group.bounds)) fail('PARENT_RELATION');
      parentOwners[parentId] = groupId;
    }
  }
  for (let id = 0; id < clusterCount; id++) {
    if (childOwners[id] < 0 || (clusters[id].refinedGroup === null ? parentOwners[id] !== -1 : parentOwners[id] < 0)) fail('DAG_RELATION');
  }
  const roots = ids(input.roots, clusterCount, 'ROOTS', true);
  if (roots.length !== terminalChildren.size || roots.some(id => !terminalChildren.has(id))) fail('ROOT_COVERAGE');
  let totalPageBytes = 0;
  const pages = input.pages.map((page, index) => {
    if (!page || typeof page !== 'object' || page.id !== index) fail('PAGE_ID');
    const clusterIds = ids(page.clusterIds, clusterCount, 'PAGE_CLUSTERS', true);
    const encodedBytes = integer(page.encodedBytes, 41, MAX_ENCODED_PAGE_BYTES, 'PAGE_ENCODED_BYTES');
    const decodedBytes = integer(page.decodedBytes, 1, MAX_DECODED_PAGE_BYTES, 'PAGE_DECODED_BYTES');
    totalPageBytes += encodedBytes + decodedBytes;
    if (totalPageBytes > MAX_TOTAL_PAGE_BYTES) fail('PAGE_TOTAL_BUDGET');
    if (decodedBytes % 4 || typeof page.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(page.sha256)) fail('PAGE_FORMAT');
    const expectedBytes = clusterIds.reduce((sum, clusterId) => {
      const cluster = clusters[clusterId], indexBytes = cluster.triangleCount * 3;
      return sum + 16 + cluster.vertexCount * (4 + vertexStride) + ((indexBytes + 3) & ~3);
    }, 0);
    if (expectedBytes !== decodedBytes) fail('PAGE_DECODED_BYTES');
    return { id: index, clusterIds, encodedBytes, decodedBytes, sha256: page.sha256 };
  });
  if (!Array.isArray(input.clusterPages) || input.clusterPages.length !== clusterCount) fail('CLUSTER_PAGES');
  const clusterPages = input.clusterPages.map(id => integer(id, 0, pageCount - 1, 'CLUSTER_PAGES'));
  if (clusterPages.length !== clusterCount) fail('CLUSTER_PAGES');
  const owned = new Int32Array(clusterCount).fill(-1);
  for (const page of pages) for (const id of page.clusterIds) {
    if (owned[id] !== -1 || clusterPages[id] !== page.id) fail('PAGE_OWNERSHIP');
    owned[id] = page.id;
  }
  if (owned.some(id => id < 0)) fail('PAGE_COVERAGE');
  const rootPages = ids(input.rootPages, pageCount, 'ROOT_PAGES', true);
  const expectedRoots = [...new Set(roots.map(id => clusterPages[id]))].sort((a, b) => a - b);
  if (rootPages.length !== expectedRoots.length || expectedRoots.some((id, i) => rootPages[i] !== id)) fail('ROOT_PAGES');
  for (const cluster of clusters) { freezeBounds(cluster.bounds); freezeBounds(cluster.lodBounds); Object.freeze(cluster); }
  for (const group of groups) { Object.freeze(group.children); Object.freeze(group.parents); freezeBounds(group.bounds); Object.freeze(group); }
  for (const page of pages) { Object.freeze(page.clusterIds); Object.freeze(page); }
  Object.freeze(clusters); Object.freeze(groups); Object.freeze(roots); Object.freeze(pages);
  Object.freeze(clusterPages); Object.freeze(rootPages);
  return Object.freeze({ version: 1, primitiveKey: input.primitiveKey, sourceVertexCount, sourceTriangleCount,
    vertexStride, clusters, groups, roots, pages, clusterPages, rootPages });
}

/** Convert an audited offline hierarchy and encoded pages into runtime-only metadata. */
export async function buildGeometryStreamDirectory(hierarchy: GeometryHierarchy, encoded: EncodedGeometryPages): Promise<GeometryStreamDirectory> {
  packGeometryLod(hierarchy);
  if (!encoded || encoded.version !== 1 || !Array.isArray(encoded.pages)
    || !encoded.pages.length || encoded.pages.length > MAX_PAGES) fail('ENCODING');
  let totalPageBytes = 0;
  for (const page of encoded.pages) {
    if (!page || !(page.bytes instanceof Uint8Array)) fail('PAGE_BYTES');
    const encodedBytes = integer(page.bytes.byteLength, 41, MAX_ENCODED_PAGE_BYTES, 'PAGE_ENCODED_BYTES');
    const decodedBytes = integer(page.decodedBytes, 1, MAX_DECODED_PAGE_BYTES, 'PAGE_DECODED_BYTES');
    totalPageBytes += encodedBytes + decodedBytes;
    if (totalPageBytes > MAX_TOTAL_PAGE_BYTES) fail('PAGE_TOTAL_BUDGET');
  }
  // Snapshot and validate all caller-owned topology before the first asynchronous digest.
  const snapshot = validateGeometryStreamDirectory({
    version: 1, primitiveKey: hierarchy.primitiveKey, sourceVertexCount: hierarchy.source.vertexCount,
    sourceTriangleCount: hierarchy.sourceTriangleCount, vertexStride: encoded.vertexStride,
    clusters: hierarchy.clusters.map(cluster => ({ level: cluster.level, group: cluster.group,
      refinedGroup: cluster.refinedGroup, vertexCount: cluster.vertices.length, triangleCount: cluster.triangles.length / 3,
      bounds: cluster.bounds, lodBounds: cluster.lodBounds, error: cluster.error })),
    groups: hierarchy.groups.map(group => ({ level: group.level, children: group.children, parents: group.parents,
      bounds: group.bounds, error: group.error, terminal: group.terminal })),
    roots: hierarchy.roots,
    pages: encoded.pages.map(page => ({ id: page.id, clusterIds: page.clusterIds, encodedBytes: page.bytes.byteLength,
      decodedBytes: page.decodedBytes, sha256: '0'.repeat(64) })),
    clusterPages: encoded.clusterPages, rootPages: encoded.rootPages,
  });
  const pages = [];
  for (const page of snapshot.pages) {
    const source = encoded.pages[page.id].bytes;
    if (!(source instanceof Uint8Array) || source.byteLength !== page.encodedBytes) fail('PAGE_BYTES_CHANGED');
    const bytes = new Uint8Array(source.byteLength);
    bytes.set(source);
    const digest = await crypto.subtle.digest('SHA-256', bytes.buffer);
    pages.push({ ...page, sha256: Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') });
  }
  return validateGeometryStreamDirectory({ ...snapshot, pages });
}
