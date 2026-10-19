import { MeshoptClusterizer } from 'meshoptimizer/clusterizer';
import { MeshoptSimplifier } from 'meshoptimizer/simplifier';
import type { Bounds3 } from './deformation';
import type { MeshData } from './geometry';
import type { GeometryBuildInput, GeometryBuildOptions, GeometryCluster, GeometryHierarchy } from './geometry-hierarchy-types';

export type { GeometryBuildInput, GeometryBuildOptions, GeometryCluster, GeometryGroup, GeometryHierarchy } from './geometry-hierarchy-types';

function integer(value: number, min: number, max: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError('GEOMETRY_' + label);
  return value;
}

function options(input: GeometryBuildOptions): Required<GeometryBuildOptions> {
  const result = {
    maxClusterVertices: integer(input.maxClusterVertices ?? 64, 3, 256, 'CLUSTER_VERTICES'),
    maxClusterTriangles: integer(input.maxClusterTriangles ?? 128, 1, 512, 'CLUSTER_TRIANGLES'),
    groupSize: integer(input.groupSize ?? 4, 2, 16, 'GROUP_SIZE'),
    simplifyRatio: input.simplifyRatio ?? 0.5,
    maxSimplificationError: input.maxSimplificationError ?? 1e30,
    maxLevels: integer(input.maxLevels ?? 16, 1, 32, 'LEVELS'),
    maxVertices: integer(input.maxVertices ?? 1_000_000, 3, 4_000_000, 'VERTEX_BUDGET'),
    maxTriangles: integer(input.maxTriangles ?? 1_000_000, 1, 4_000_000, 'TRIANGLE_BUDGET'),
    maxOutputTriangles: integer(input.maxOutputTriangles ?? 3_000_000, 1, 8_000_000, 'OUTPUT_TRIANGLE_BUDGET'),
    maxClusters: integer(input.maxClusters ?? 65_536, 1, 262_144, 'CLUSTER_BUDGET'),
    maxGroups: integer(input.maxGroups ?? 65_536, 1, 262_144, 'GROUP_BUDGET'),
  };
  if (!Number.isFinite(result.simplifyRatio) || result.simplifyRatio < 0.1 || result.simplifyRatio > 0.8) throw new RangeError('GEOMETRY_SIMPLIFY_RATIO');
  if (!Number.isFinite(result.maxSimplificationError) || result.maxSimplificationError < 0 || result.maxSimplificationError > 1e30) throw new RangeError('GEOMETRY_SIMPLIFICATION_ERROR');
  return result;
}

const emptyBounds = (): Bounds3 => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
function includePoint(bounds: Bounds3, positions: Float32Array, vertex: number): void {
  for (let axis = 0; axis < 3; axis++) {
    const value = positions[vertex * 3 + axis];
    bounds.min[axis] = Math.min(bounds.min[axis], value);
    bounds.max[axis] = Math.max(bounds.max[axis], value);
  }
}
function union(bounds: Bounds3, other: Bounds3): void {
  for (let axis = 0; axis < 3; axis++) {
    bounds.min[axis] = Math.min(bounds.min[axis], other.min[axis]);
    bounds.max[axis] = Math.max(bounds.max[axis], other.max[axis]);
  }
}

function readSource(mesh: MeshData, limits: Required<GeometryBuildOptions>) {
  if (!(mesh.vertices instanceof ArrayBuffer) || !Number.isSafeInteger(mesh.vertexCount)
    || mesh.vertexCount < 3 || mesh.vertexCount > limits.maxVertices || mesh.vertices.byteLength !== mesh.vertexCount * 80) throw new RangeError('GEOMETRY_VERTEX_BUDGET_OR_LAYOUT');
  if (!(mesh.indices instanceof Uint32Array) || !mesh.indices.length || mesh.indices.length % 3
    || mesh.indices.length / 3 > limits.maxTriangles || mesh.indices.length / 3 > limits.maxOutputTriangles) throw new RangeError('GEOMETRY_TRIANGLE_BUDGET_OR_LAYOUT');
  if (mesh.uvs && (!(mesh.uvs instanceof Float32Array) || mesh.uvs.length !== mesh.vertexCount * 2 || !mesh.uvs.every(Number.isFinite))) throw new Error('GEOMETRY_UVS');
  if (mesh.tangents && (!(mesh.tangents instanceof Float32Array) || mesh.tangents.length !== mesh.vertexCount * 4 || !mesh.tangents.every(Number.isFinite))) throw new Error('GEOMETRY_TANGENTS');
  if (mesh.colors && (!(mesh.colors instanceof Float32Array) || mesh.colors.length !== mesh.vertexCount * 4
    || !mesh.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1))) throw new Error('GEOMETRY_COLORS');
  const positions = new Float32Array(mesh.vertexCount * 3), view = new DataView(mesh.vertices), bounds = emptyBounds();
  const attributeStride = 3 + (mesh.uvs ? 2 : 0) + (mesh.tangents ? 4 : 0) + (mesh.colors ? 4 : 0);
  const attributes = new Float32Array(mesh.vertexCount * attributeStride);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    for (let field = 0; field < 20; field++) {
      if (field >= 8 && field < 12) continue; // These four words are uint32 joints.
      if (!Number.isFinite(view.getFloat32(vertex * 80 + field * 4, true))) throw new Error('GEOMETRY_NONFINITE_VERTEX');
    }
    for (let axis = 0; axis < 3; axis++) {
      positions[vertex * 3 + axis] = view.getFloat32(vertex * 80 + axis * 4, true);
      attributes[vertex * attributeStride + axis] = view.getFloat32(vertex * 80 + 16 + axis * 4, true);
    }
    if (mesh.uvs) attributes.set(mesh.uvs.subarray(vertex * 2, vertex * 2 + 2), vertex * attributeStride + 3);
    if (mesh.tangents) attributes.set(mesh.tangents.subarray(vertex * 4, vertex * 4 + 4), vertex * attributeStride + 3 + (mesh.uvs ? 2 : 0));
    if (mesh.colors) attributes.set(mesh.colors.subarray(vertex * 4, vertex * 4 + 4),
      vertex * attributeStride + 3 + (mesh.uvs ? 2 : 0) + (mesh.tangents ? 4 : 0));
  }
  for (const index of mesh.indices) {
    if (index >= mesh.vertexCount) throw new RangeError('GEOMETRY_INDEX');
    includePoint(bounds, positions, index);
  }
  const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]));
  if (!(extent > 0) || !Number.isFinite(extent)) throw new Error('GEOMETRY_DEGENERATE_EXTENT');
  // Operate in normalized local space to avoid overflowing WASM float math.
  // Source bytes and all exported bounds stay in the original mesh units.
  const normalized = new Float32Array(positions.length);
  // Unreferenced vertices cannot affect precision, clustering or simplification.
  for (const vertex of mesh.indices) for (let axis = 0; axis < 3; axis++) {
    normalized[vertex * 3 + axis] = (positions[vertex * 3 + axis] - bounds.min[axis]) / extent;
  }
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const a = mesh.indices[i] * 3, b = mesh.indices[i + 1] * 3, c = mesh.indices[i + 2] * 3;
    const ab = [0, 1, 2].map(k => normalized[b + k] - normalized[a + k]);
    const ac = [0, 1, 2].map(k => normalized[c + k] - normalized[a + k]);
    if (Math.hypot(ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]) === 0) throw new Error('GEOMETRY_DEGENERATE_TRIANGLE');
  }
  const source: MeshData = { vertices: mesh.vertices.slice(0), vertexCount: mesh.vertexCount,
    indices: new Uint32Array(mesh.indices), bounds,
    ...(mesh.uvs ? { uvs: new Float32Array(mesh.uvs) } : {}),
    ...(mesh.tangents ? { tangents: new Float32Array(mesh.tangents) } : {}),
    ...(mesh.colors ? { colors: new Float32Array(mesh.colors) } : {}) };
  return { source, positions, normalized, attributes, attributeStride, extent };
}

/** Position identity is exact in the original float32 coordinates, including UV wedges. */
function positionRemap(positions: Float32Array): Uint32Array {
  const unique = new Map<string, number>(), remap = new Uint32Array(positions.length / 3);
  for (let i = 0; i < remap.length; i++) {
    const key = `${positions[i * 3]},${positions[i * 3 + 1]},${positions[i * 3 + 2]}`;
    const prior = unique.get(key);
    remap[i] = prior ?? i;
    if (prior === undefined) unique.set(key, i);
  }
  return remap;
}

function expanded(cluster: GeometryCluster): Uint32Array {
  return Uint32Array.from(cluster.triangles, index => cluster.vertices[index]);
}
function mergedIndices(ids: readonly number[], clusters: GeometryCluster[]): Uint32Array {
  const length = ids.reduce((sum, id) => sum + clusters[id].triangles.length, 0);
  const indices = new Uint32Array(length);
  let offset = 0;
  for (const id of ids) {
    const cluster = clusters[id];
    for (const index of cluster.triangles) indices[offset++] = cluster.vertices[index];
  }
  return indices;
}

/** JS/WASM calls copy their input arrays. Compact a sparse group before calling
 * them, so a small group never recopies the entire million-vertex source mesh. */
function localSubset(indices: Uint32Array, positions: Float32Array) {
  const map = new Map<number, number>(), globalIds: number[] = [];
  const localIndices = new Uint32Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
    const global = indices[i];
    let local = map.get(global);
    if (local === undefined) { local = globalIds.length; map.set(global, local); globalIds.push(global); }
    localIndices[i] = local;
  }
  const localPositions = new Float32Array(globalIds.length * 3);
  for (let i = 0; i < globalIds.length; i++) {
    for (let axis = 0; axis < 3; axis++) localPositions[i * 3 + axis] = positions[globalIds[i] * 3 + axis];
  }
  return { indices: localIndices, positions: localPositions, globalIds };
}

/** Directed geometric boundary, after cancelling opposite interior half edges. */
function boundary(indices: Uint32Array, remap: Uint32Array): Map<string, number> {
  const edges = new Map<string, number>();
  for (let i = 0; i < indices.length; i += 3) for (let e = 0; e < 3; e++) {
    const a = remap[indices[i + e]], b = remap[indices[i + (e + 1) % 3]];
    if (a === b) continue;
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    const count = (edges.get(key) ?? 0) + (a < b ? 1 : -1);
    if (count) edges.set(key, count); else edges.delete(key);
  }
  return edges;
}
function sameBoundary(before: Uint32Array, after: Uint32Array, remap: Uint32Array): boolean {
  const a = boundary(before, remap), b = boundary(after, remap);
  return a.size === b.size && Array.from(a).every(([key, count]) => b.get(key) === count);
}

function triangleKey(a: number, b: number, c: number): string {
  return a <= b && a <= c ? `${a},${b},${c}` : b <= c ? `${b},${c},${a}` : `${c},${a},${b}`;
}

function morton(x: number, y: number, z: number): number {
  let result = 0;
  for (let bit = 0; bit < 10; bit++) result |= ((x >>> bit) & 1) << (bit * 3)
    | ((y >>> bit) & 1) << (bit * 3 + 1) | ((z >>> bit) & 1) << (bit * 3 + 2);
  return result;
}

/** Deterministic spatial grouping; does not merge material primitives. */
function partition(active: number[], clusters: GeometryCluster[], bounds: Bounds3, size: number): number[][] {
  const sorted = active.map(id => {
    const b = clusters[id].bounds;
    const q = b.min.map((value, axis) => {
      const length = bounds.max[axis] - bounds.min[axis];
      return length > 0 ? Math.max(0, Math.min(1023, Math.floor(((value + b.max[axis]) * 0.5 - bounds.min[axis]) / length * 1023))) : 0;
    });
    return { id, key: morton(q[0], q[1], q[2]) };
  }).sort((a, b) => a.key - b.key || a.id - b.id);
  const groups: number[][] = [];
  for (let i = 0; i < sorted.length; i += size) groups.push(sorted.slice(i, i + size).map(item => item.id));
  return groups;
}

/** Offline-only builder: original attributes, true reduced index buffers and a cluster DAG.
 * Based on the lock/regroup/refine contracts in meshoptimizer's MIT clusterlod example;
 * error remains an approximate appearance metric, not a certified Hausdorff bound. */
export async function buildGeometryHierarchy(input: GeometryBuildInput, requested: GeometryBuildOptions = {}): Promise<GeometryHierarchy> {
  const limits = options(requested);
  if (typeof input.primitiveKey !== 'string' || !input.primitiveKey.length || input.primitiveKey.length > 1024) throw new Error('GEOMETRY_PRIMITIVE_KEY');
  if (input.coverage !== 'opaque') throw new Error('GEOMETRY_UNSUPPORTED_COVERAGE: retain the full-detail path');
  if (input.deformation && (input.deformation.jointCount !== 0 || input.deformation.nodes.length !== 0)) throw new Error('GEOMETRY_UNSUPPORTED_DEFORMATION: retain the full-detail path');
  const { source, positions, normalized, attributes, attributeStride, extent } = readSource(input.mesh, limits);
  if (!MeshoptClusterizer.supported || !MeshoptSimplifier.supported) throw new Error('GEOMETRY_WASM_UNAVAILABLE');
  await Promise.all([MeshoptClusterizer.ready, MeshoptSimplifier.ready]);
  const result: GeometryHierarchy = { version: 1, builder: 'meshoptimizer-1.3.0', primitiveKey: input.primitiveKey,
    source, options: limits, clusters: [], groups: [], roots: [], sourceTriangleCount: source.indices.length / 3 };
  const remap = positionRemap(positions), terminalLocks = new Uint8Array(source.vertexCount);
  const locks = new Uint8Array(source.vertexCount), owners = new Int32Array(source.vertexCount);
  let totalTriangles = 0;
  const clusterize = (indices: Uint32Array, level: number, refinedGroup: number | null, lodBounds?: Bounds3, error = 0): number[] => {
    const subset = level ? localSubset(indices, normalized) : undefined;
    const buffers = MeshoptClusterizer.buildMeshlets(subset?.indices ?? indices, subset?.positions ?? normalized,
      3, limits.maxClusterVertices, limits.maxClusterTriangles);
    if (result.clusters.length + buffers.meshletCount > limits.maxClusters) throw new RangeError('GEOMETRY_CLUSTER_BUDGET');
    if (totalTriangles + indices.length / 3 > limits.maxOutputTriangles) throw new RangeError('GEOMETRY_OUTPUT_TRIANGLE_BUDGET');
    const ids: number[] = [];
    for (let i = 0; i < buffers.meshletCount; i++) {
      const built = MeshoptClusterizer.extractMeshlet(buffers, i), bounds = emptyBounds();
      const vertices = subset ? Uint32Array.from(built.vertices, index => subset.globalIds[index]) : new Uint32Array(built.vertices);
      const triangles = new Uint8Array(built.triangles);
      if (!vertices.length || !triangles.length || triangles.length % 3) throw new Error('GEOMETRY_INVALID_CLUSTER');
      for (const vertex of vertices) includePoint(bounds, positions, vertex);
      ids.push(result.clusters.length);
      result.clusters.push({ level, vertices, triangles, bounds,
        lodBounds: structuredClone(lodBounds ?? bounds), error, refinedGroup, group: -1 });
    }
    totalTriangles += indices.length / 3;
    return ids;
  };
  let active = clusterize(source.indices, 0, null);
  // Recover original triangle IDs even if clustering cyclically rotates triangles.
  const sourceIds = new Map<string, { ids: number[]; next: number }>();
  for (let i = 0; i < source.indices.length; i += 3) {
    const key = triangleKey(source.indices[i], source.indices[i + 1], source.indices[i + 2]);
    let entry = sourceIds.get(key);
    if (!entry) sourceIds.set(key, entry = { ids: [], next: 0 });
    entry.ids.push(i / 3);
  }
  for (const id of active) {
    const cluster = result.clusters[id], indices = expanded(cluster);
    cluster.sourceTriangles = new Uint32Array(indices.length / 3);
    for (let i = 0; i < indices.length; i += 3) {
      const entry = sourceIds.get(triangleKey(indices[i], indices[i + 1], indices[i + 2]));
      if (!entry || entry.next >= entry.ids.length) throw new Error('GEOMETRY_LEAF_COVERAGE');
      cluster.sourceTriangles[i / 3] = entry.ids[entry.next++];
    }
  }
  if (Array.from(sourceIds.values()).some(entry => entry.next !== entry.ids.length)) throw new Error('GEOMETRY_LEAF_COVERAGE');
  sourceIds.clear();
  for (let level = 0; active.length; level++) {
    const groups = partition(active, result.clusters, source.bounds, limits.groupSize), pending: number[] = [];
    if (result.groups.length + groups.length > limits.maxGroups) throw new RangeError('GEOMETRY_GROUP_BUDGET');
    // Same-position wedges on different group boundaries must all remain fixed.
    owners.fill(-1); locks.set(terminalLocks);
    for (let group = 0; group < groups.length; group++) for (const id of groups[group]) {
      for (const vertex of result.clusters[id].vertices) {
        const canonical = remap[vertex];
        if (owners[canonical] !== -1 && owners[canonical] !== group) locks[canonical] = 1;
        owners[canonical] = group;
      }
    }
    for (let i = 0; i < locks.length; i++) locks[i] = locks[remap[i]];
    for (const children of groups) {
      const id = result.groups.length, bounds = emptyBounds();
      let previousError = 0;
      for (const child of children) {
        const cluster = result.clusters[child];
        union(bounds, cluster.lodBounds); previousError = Math.max(previousError, cluster.error);
        cluster.group = id;
      }
      const group = { level, children, parents: [] as number[], bounds, localError: 0, error: previousError, terminal: true };
      result.groups.push(group);
      const indices = mergedIndices(children, result.clusters);
      const target = Math.max(1, Math.floor(indices.length / 3 * limits.simplifyRatio)) * 3;
      if (level < limits.maxLevels - 1 && target < indices.length) {
        const subset = localSubset(indices, normalized);
        const localAttributes = new Float32Array(subset.globalIds.length * attributeStride);
        const localLocks = new Uint8Array(subset.globalIds.length);
        for (let i = 0; i < subset.globalIds.length; i++) {
          const global = subset.globalIds[i];
          localAttributes.set(attributes.subarray(global * attributeStride, (global + 1) * attributeStride), i * attributeStride);
          localLocks[i] = locks[global];
        }
        const [localIndices, normalizedError] = MeshoptSimplifier.simplifyWithAttributes(subset.indices, subset.positions, 3,
          localAttributes, attributeStride, Array(attributeStride).fill(1), localLocks, target,
          Math.min(1e30, limits.maxSimplificationError / extent), ['LockBorder', 'Sparse', 'ErrorAbsolute']);
        const simplified = Uint32Array.from(localIndices, index => subset.globalIds[index]);
        const error = normalizedError * extent;
        if (!Number.isFinite(error) || error < 0) throw new Error('GEOMETRY_NONFINITE_SIMPLIFICATION');
        // Small reductions are not worth another level or page dependency. Keep
        // the original group when topology/attributes prevent useful reduction.
        if (simplified.length >= 3 && simplified.length <= indices.length * 0.9
          && sameBoundary(indices, simplified, remap)) {
          group.localError = error;
          // A positive float-scale floor keeps threshold=0 an exact LOD0 mode.
          group.error = previousError + Math.max(error, extent * 2 ** -24);
          group.terminal = false;
          group.parents = clusterize(simplified, level + 1, id, bounds, group.error);
          pending.push(...group.parents);
        }
      }
      if (group.terminal) {
        result.roots.push(...children);
        for (const child of children) for (const vertex of result.clusters[child].vertices) terminalLocks[remap[vertex]] = 1;
      }
    }
    active = pending;
  }
  return result;
}

/** CPU reference for future GPU traversal. This does not do frustum culling or residency. */
export function selectGeometryCut(hierarchy: GeometryHierarchy, view: {
  eye: readonly [number, number, number]; projectionScale: number; near: number; maxErrorPixels: number;
}): number[] {
  if (view.eye.length !== 3 || !view.eye.every(Number.isFinite) || !Number.isFinite(view.projectionScale)
    || view.projectionScale <= 0 || !Number.isFinite(view.near) || view.near <= 0
    || !Number.isFinite(view.maxErrorPixels) || view.maxErrorPixels < 0) throw new RangeError('GEOMETRY_LOD_VIEW');
  const coarsen = hierarchy.groups.map(group => {
    if (group.terminal) return false;
    let distanceSquared = 0;
    for (let axis = 0; axis < 3; axis++) {
      const delta = Math.max(group.bounds.min[axis] - view.eye[axis], 0, view.eye[axis] - group.bounds.max[axis]);
      distanceSquared += delta * delta;
    }
    const pixels = group.error * view.projectionScale / Math.max(view.near, Math.sqrt(distanceSquared));
    return pixels <= view.maxErrorPixels;
  });
  const selected: number[] = [];
  for (let id = 0; id < hierarchy.clusters.length; id++) {
    const cluster = hierarchy.clusters[id];
    if (!coarsen[cluster.group] && (cluster.refinedGroup === null || coarsen[cluster.refinedGroup])) selected.push(id);
  }
  return selected;
}
