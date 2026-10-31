import type { Bounds3 } from './deformation';
import { MAX_FIXED_TOPOLOGY_LEAVES, orderInstanceBounds } from './instance-bvh';

/**
 * CPU contract for the shared static world-query traversal.
 *
 * A budgeted subset of rigid, committed batches is copied into three pooled
 * storage buffers so one dispatch can traverse all of them: a two-level BVH
 * whose top level has one leaf per (batch, instance slot), and per-instance
 * bottom levels that reuse each batch's existing triangle BVH topology and
 * GPU-refit bounds. Only positions are pooled; every other attribute stays in
 * the batch's own buffers and is resolved by a per-batch material dispatch for
 * the rays that batch wins. Membership (Opaque, current face policy) is
 * re-evaluated on every trace so material changes never bypass coverage.
 */

/** Two vec4<u32>: [low.xyz bits, escape] [high.xyz bits, info]. */
export const SHARED_NODE_BYTES = 32;
/** Three local vertex indices in BLAS leaf order. */
export const SHARED_TRIANGLE_BYTES = 12;
/** Packed float32 xyz per pooled vertex. */
export const SHARED_POSITION_BYTES = 12;
/** [batch, instance slot, entity, generation, BLAS record base, triangle base, position base, node count (0 = inactive)]. */
export const SHARED_LEAF_WORDS = 8;
export const SHARED_MAX_LEAVES = MAX_FIXED_TOPOLOGY_LEAVES;
export const SHARED_FLAG_MEMBER = 1;
export const SHARED_FLAG_DOUBLE_SIDED = 2;
/** TLAS info word: valid bit, plus leaf ID + 1 for leaves (0 = internal). */
export const SHARED_TLAS_VALID = 0x8000_0000;
const INACTIVE = 0xffff_ffff;
/** BLAS leaf info packs a leaf-order triangle start above a 4-bit count (leaves hold 1..8 triangles). */
const MAX_LEAF_FIRST = 0x0fff_ffff;

export type SharedStaticExclusion = 'deformation' | 'mutable-input' | 'dispatch-limit' | 'leaf-limit'
  | 'binding-limit' | 'budget';

export interface SharedStaticCandidate {
  /** Registered batch index; order decides deterministic first-fit admission. */
  readonly index: number;
  readonly capacity: number;
  readonly vertexCount: number;
  readonly nodeCount: number;
  readonly triangleCount: number;
  /** Structural reason the batch can never be pooled, or null. */
  readonly ineligible: 'deformation' | 'mutable-input' | null;
}

export interface SharedStaticPoolBatch {
  readonly index: number;
  readonly capacity: number;
  readonly vertexCount: number;
  readonly nodeCount: number;
  readonly triangleCount: number;
  /** First leaf ID; slot i uses leafBase + i. */
  readonly leafBase: number;
  /** Absolute node record of slot 0's BLAS; slot i adds i * nodeCount. */
  readonly nodeBase: number;
  /** First pooled triangle; shared by every slot. */
  readonly triangleBase: number;
  /** First pooled vertex of slot 0; slot i adds i * vertexCount. */
  readonly positionBase: number;
}

export interface SharedStaticLimits {
  /** Bytes available to this subsystem, including fixed overhead. */
  readonly maxBytes: number;
  readonly maxBindingBytes: number;
  readonly maxBufferBytes: number;
  readonly maxWorkgroupsPerDimension: number;
  /** Registered batch flag words appended to the leaf buffer. */
  readonly batchSlots: number;
  /** Bytes allocated regardless of pool size (parameters, winners). */
  readonly fixedBytes: number;
}

export interface SharedStaticBytes {
  readonly nodes: number;
  readonly triangles: number;
  readonly positions: number;
  readonly leaves: number;
  readonly topology: number;
  readonly fixed: number;
  readonly total: number;
}

export interface SharedStaticPlan {
  readonly batches: readonly SharedStaticPoolBatch[];
  readonly excluded: readonly { readonly index: number; readonly reason: SharedStaticExclusion }[];
  readonly leafCount: number;
  readonly tlasNodeCount: number;
  readonly nodeRecords: number;
  readonly triangleCount: number;
  readonly vertexCount: number;
  readonly bytes: SharedStaticBytes;
}

function integer(value: number, label: string, low: number, high: number): number {
  if (!Number.isSafeInteger(value) || value < low || value > high) {
    throw new RangeError(`${label} must be an integer in [${low}, ${high}]`);
  }
  return value;
}

/** TLAS topology, Morton order and per-depth node lists share one u32 buffer. */
export function sharedTopologyWords(leafCount: number): number {
  return leafCount ? (2 * leafCount - 1) * 4 + leafCount + (2 * leafCount - 1) : 0;
}

function bytesFor(leafCount: number, blasRecords: number, triangles: number, vertices: number,
  batchSlots: number, fixed: number): SharedStaticBytes {
  const tlas = leafCount ? 2 * leafCount - 1 : 0;
  const nodes = (tlas + blasRecords) * SHARED_NODE_BYTES;
  const trianglesBytes = triangles * SHARED_TRIANGLE_BYTES;
  const positions = vertices * SHARED_POSITION_BYTES;
  const leaves = (leafCount * SHARED_LEAF_WORDS + batchSlots) * 4;
  const topology = sharedTopologyWords(leafCount) * 4;
  return { nodes, triangles: trianglesBytes, positions, leaves, topology, fixed,
    total: nodes + trianglesBytes + positions + leaves + topology + fixed };
}

/** First-fit in registration order. Over-budget batches are skipped, never truncated. */
export function planSharedStaticPool(candidates: readonly SharedStaticCandidate[], limits: SharedStaticLimits): SharedStaticPlan {
  integer(limits.maxBytes, 'maxBytes', 0, Number.MAX_SAFE_INTEGER);
  integer(limits.maxBindingBytes, 'maxBindingBytes', 4, Number.MAX_SAFE_INTEGER);
  integer(limits.maxBufferBytes, 'maxBufferBytes', 4, Number.MAX_SAFE_INTEGER);
  integer(limits.maxWorkgroupsPerDimension, 'maxWorkgroupsPerDimension', 1, Number.MAX_SAFE_INTEGER);
  integer(limits.batchSlots, 'batchSlots', 1, 4096);
  integer(limits.fixedBytes, 'fixedBytes', 0, Number.MAX_SAFE_INTEGER);
  const bindingLimit = Math.min(limits.maxBindingBytes, limits.maxBufferBytes);
  const batches: SharedStaticPoolBatch[] = [];
  const excluded: { index: number; reason: SharedStaticExclusion }[] = [];
  let leaves = 0, records = 0, triangles = 0, vertices = 0;
  let previous = -1;
  for (const candidate of candidates) {
    integer(candidate.index, 'candidate.index', previous + 1, limits.batchSlots - 1);
    previous = candidate.index;
    integer(candidate.capacity, 'candidate.capacity', 1, SHARED_MAX_LEAVES);
    integer(candidate.vertexCount, 'candidate.vertexCount', 1, 0xffff_ffff);
    integer(candidate.nodeCount, 'candidate.nodeCount', 1, 0xffff_ffff);
    integer(candidate.triangleCount, 'candidate.triangleCount', 1, MAX_LEAF_FIRST);
    if (candidate.ineligible) { excluded.push({ index: candidate.index, reason: candidate.ineligible }); continue; }
    const updateThreads = candidate.capacity * (candidate.vertexCount + candidate.nodeCount);
    if (Math.ceil(updateThreads / 64) > limits.maxWorkgroupsPerDimension) {
      excluded.push({ index: candidate.index, reason: 'dispatch-limit' }); continue;
    }
    const nextLeaves = leaves + candidate.capacity;
    if (nextLeaves > SHARED_MAX_LEAVES) { excluded.push({ index: candidate.index, reason: 'leaf-limit' }); continue; }
    const next = bytesFor(nextLeaves, records + candidate.capacity * candidate.nodeCount,
      triangles + candidate.triangleCount, vertices + candidate.capacity * candidate.vertexCount,
      limits.batchSlots, limits.fixedBytes);
    if ([next.nodes, next.triangles, next.positions, next.leaves, next.topology].some(size => size > bindingLimit)
      // Pooled indices are u32 words; records/vertices beyond them cannot be addressed.
      || (next.nodes / 4) > 0xffff_ffff || (next.positions / 4) > 0xffff_ffff) {
      excluded.push({ index: candidate.index, reason: 'binding-limit' }); continue;
    }
    if (next.total > limits.maxBytes) { excluded.push({ index: candidate.index, reason: 'budget' }); continue; }
    batches.push({ index: candidate.index, capacity: candidate.capacity, vertexCount: candidate.vertexCount,
      nodeCount: candidate.nodeCount, triangleCount: candidate.triangleCount,
      leafBase: leaves, nodeBase: records, triangleBase: triangles, positionBase: vertices });
    leaves = nextLeaves; records += candidate.capacity * candidate.nodeCount;
    triangles += candidate.triangleCount; vertices += candidate.capacity * candidate.vertexCount;
  }
  const tlasNodeCount = leaves ? 2 * leaves - 1 : 0;
  // BLAS records follow the TLAS records in the shared node buffer.
  const placed = batches.map(batch => ({ ...batch, nodeBase: batch.nodeBase + tlasNodeCount }));
  return { batches: placed, excluded, leafCount: leaves, tlasNodeCount, nodeRecords: tlasNodeCount + records,
    triangleCount: triangles, vertexCount: vertices,
    bytes: leaves ? bytesFor(leaves, records, triangles, vertices, limits.batchSlots, limits.fixedBytes)
      : { nodes: 0, triangles: 0, positions: 0, leaves: 0, topology: 0, fixed: 0, total: 0 } };
}

/** Pool triangle words for one batch: local vertex indices in BLAS leaf order. */
export function packLeafOrderedTriangles(indices: Uint32Array, leafOrder: Uint32Array, vertexCount: number): Uint32Array<ArrayBuffer> {
  if (!(indices instanceof Uint32Array) || indices.length % 3 !== 0 || !(leafOrder instanceof Uint32Array)
    || leafOrder.length * 3 !== indices.length) throw new RangeError('SHARED_STATIC_TRIANGLE_LAYOUT');
  const words = new Uint32Array(indices.length);
  for (let position = 0; position < leafOrder.length; position++) {
    const triangle = leafOrder[position];
    if (triangle >= leafOrder.length) throw new RangeError('SHARED_STATIC_TRIANGLE_LAYOUT');
    for (let corner = 0; corner < 3; corner++) {
      const vertex = indices[triangle * 3 + corner];
      if (vertex >= vertexCount) throw new RangeError('SHARED_STATIC_TRIANGLE_LAYOUT');
      words[position * 3 + corner] = vertex;
    }
  }
  return words;
}

export interface SharedLeafIdentity {
  readonly entityId: number;
  readonly generation: number;
}

/** Leaf records for every capacity slot of one pooled batch; inactive slots keep node count zero. */
export function packSharedLeaves(batch: SharedStaticPoolBatch, active: readonly SharedLeafIdentity[]): Uint32Array<ArrayBuffer> {
  if (active.length > batch.capacity) throw new RangeError('SHARED_STATIC_LEAF_CAPACITY');
  const words = new Uint32Array(batch.capacity * SHARED_LEAF_WORDS);
  for (let slot = 0; slot < batch.capacity; slot++) {
    const at = slot * SHARED_LEAF_WORDS;
    words[at] = batch.index; words[at + 1] = slot;
    words[at + 4] = batch.nodeBase + slot * batch.nodeCount;
    words[at + 5] = batch.triangleBase;
    words[at + 6] = batch.positionBase + slot * batch.vertexCount;
    if (slot < active.length) {
      words[at + 2] = active[slot].entityId >>> 0; words[at + 3] = active[slot].generation >>> 0;
      words[at + 7] = batch.nodeCount;
    }
  }
  return words;
}

/** Morton order over active leaves; inactive and unused slots stay 0xffffffff. Ordering only affects speed. */
export function orderSharedLeaves(leafCount: number, bounds: readonly (Bounds3 | null)[]): Uint32Array<ArrayBuffer> {
  integer(leafCount, 'leafCount', 1, SHARED_MAX_LEAVES);
  if (bounds.length !== leafCount) throw new RangeError('SHARED_STATIC_ORDER_LENGTH');
  const active: number[] = [], boxes: Bounds3[] = [];
  bounds.forEach((box, leaf) => { if (box) { active.push(leaf); boxes.push(box); } });
  const order = new Uint32Array(leafCount).fill(INACTIVE);
  if (!active.length) return order;
  const local = orderInstanceBounds(boxes, leafCount, SHARED_MAX_LEAVES);
  for (let i = 0; i < active.length; i++) order[i] = active[local[i]];
  return order;
}

export function packSharedFlags(member: boolean, doubleSided: boolean): number {
  return (member ? SHARED_FLAG_MEMBER : 0) | (doubleSided ? SHARED_FLAG_DOUBLE_SIDED : 0);
}

// ---------------------------------------------------------------------------
// CPU reference of the GPU traversal. It mirrors the WGSL control flow and the
// float32 operation order (each step rounded with Math.fround) so unit tests can
// check pool packing, escape links, tie-breaks, flags and budgets without a GPU.
// GPU compilers may fuse operations; GPU-vs-GPU parity tests cover that.
// ---------------------------------------------------------------------------

const f = Math.fround;
type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
const dot = (a: V3, b: V3): number => f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
const cross = (a: V3, b: V3): V3 => [f(f(a[1] * b[2]) - f(a[2] * b[1])), f(f(a[2] * b[0]) - f(a[0] * b[2])),
  f(f(a[0] * b[1]) - f(a[1] * b[0]))];
const length = (a: V3): number => f(Math.sqrt(dot(a, a)));

export interface CpuRay {
  origin: V3; direction: V3; tMin: number; tMax: number;
  ignore?: { entityId: number; generation: number };
}

export interface CpuTriangleHit {
  /** Registered batch index (0-based). */
  batch: number;
  instance: number;
  /** Leaf-order triangle position within the batch. */
  position: number;
  distance: number;
  u: number;
  v: number;
  front: boolean;
}

export interface CpuTraceResult {
  hit: CpuTriangleHit | null;
  workUnits: number;
  triangleTests: number;
  overflow: boolean;
}

/** Shared triangle test: returns null when rejected; distance interval and ties are the caller's job. */
export function cpuIntersectTriangle(ray: CpuRay, a: V3, b: V3, c: V3, doubleSided: boolean, reversed: boolean):
  { distance: number; u: number; v: number; determinant: number } | null {
  const edge1 = sub(b, a), edge2 = sub(c, a);
  const crossDirection = cross(ray.direction, edge2);
  const determinant = dot(edge1, crossDirection);
  const epsilon = Math.max(1e-30, f(f(length(edge1) * length(edge2)) * f(1e-7)));
  const rejectedFace = reversed ? determinant > 0 : determinant < 0;
  if (Math.abs(determinant) <= epsilon || (!doubleSided && rejectedFace)) return null;
  const delta = sub(ray.origin, a);
  const u = f(dot(delta, crossDirection) / determinant);
  if (u < 0 || u > 1) return null;
  const crossDelta = cross(delta, edge1);
  const v = f(dot(ray.direction, crossDelta) / determinant);
  if (v < 0 || f(u + v) > 1) return null;
  return { distance: f(dot(edge2, crossDelta) / determinant), u, v, determinant };
}

export function cpuIntersectBox(ray: CpuRay, low: V3, high: V3, closest: number): boolean {
  let near = ray.tMin, far = closest;
  for (let axis = 0; axis < 3; axis++) {
    const direction = ray.direction[axis];
    if (direction === 0) {
      if (ray.origin[axis] < low[axis] || ray.origin[axis] > high[axis]) return false;
    } else {
      const a = f(f(low[axis] - ray.origin[axis]) / direction);
      const b = f(f(high[axis] - ray.origin[axis]) / direction);
      near = Math.max(near, Math.min(a, b)); far = Math.min(far, Math.max(a, b));
      if (far < near) return false;
    }
  }
  return true;
}

export interface CpuSharedPool {
  /** 8 u32 per record, TLAS first. */
  nodes: Uint32Array;
  triangles: Uint32Array;
  positions: Float32Array;
  /** Leaf records followed by one flag word per registered batch. */
  leaves: Uint32Array;
  leafCount: number;
  tlasNodeCount: number;
}

/** Equal-distance ties keep the lowest (batch, instance slot), then the first triangle in BLAS order. */
export function traceSharedPoolCpu(pool: CpuSharedPool, ray: CpuRay, workLimit: number, reversed = false): CpuTraceResult {
  const floats = new Float32Array(pool.nodes.buffer, pool.nodes.byteOffset, pool.nodes.length);
  const box = (record: number): [V3, V3] => [[floats[record * 8], floats[record * 8 + 1], floats[record * 8 + 2]],
    [floats[record * 8 + 4], floats[record * 8 + 5], floats[record * 8 + 6]]];
  const vertex = (index: number): V3 => [pool.positions[index * 3], pool.positions[index * 3 + 1], pool.positions[index * 3 + 2]];
  const flagBase = pool.leafCount * SHARED_LEAF_WORDS;
  let work = 0, tests = 0, overflow = false, closest = ray.tMax;
  let best: CpuTriangleHit | null = null;
  if (ray.tMax < ray.tMin) return { hit: null, workUnits: 0, triangleTests: 0, overflow: false };
  let tlas = 0;
  outer: while (tlas < pool.tlasNodeCount) {
    if (work >= workLimit) { overflow = true; break; }
    work++;
    const [low, high] = box(tlas);
    const escape = pool.nodes[tlas * 8 + 3], info = pool.nodes[tlas * 8 + 7];
    if ((info & SHARED_TLAS_VALID) === 0 || !cpuIntersectBox(ray, low, high, closest)) { tlas = escape; continue; }
    tlas++;
    const leafRef = info & 0x7fff_ffff;
    if (!leafRef) continue;
    const leaf = (leafRef - 1) * SHARED_LEAF_WORDS;
    const batch = pool.leaves[leaf], flags = pool.leaves[flagBase + batch];
    if (!(flags & SHARED_FLAG_MEMBER)) continue;
    if (ray.ignore && ray.ignore.entityId === pool.leaves[leaf + 2] && ray.ignore.generation === pool.leaves[leaf + 3]) continue;
    const instance = pool.leaves[leaf + 1], nodeBase = pool.leaves[leaf + 4], triangleBase = pool.leaves[leaf + 5];
    const positionBase = pool.leaves[leaf + 6], nodeCount = pool.leaves[leaf + 7];
    const doubleSided = (flags & SHARED_FLAG_DOUBLE_SIDED) !== 0;
    let node = 0;
    while (node < nodeCount) {
      if (work >= workLimit) { overflow = true; break outer; }
      work++;
      const record = nodeBase + node;
      const [lo, hi] = box(record);
      if (!cpuIntersectBox(ray, lo, hi, closest)) { node = pool.nodes[record * 8 + 3]; continue; }
      node++;
      const leafInfo = pool.nodes[record * 8 + 7];
      if (!leafInfo) continue;
      const first = leafInfo >>> 4, count = leafInfo & 15;
      for (let i = 0; i < count; i++) {
        if (work >= workLimit) { overflow = true; break outer; }
        work++; tests++;
        const t = (triangleBase + first + i) * 3;
        const result = cpuIntersectTriangle(ray, vertex(positionBase + pool.triangles[t]),
          vertex(positionBase + pool.triangles[t + 1]), vertex(positionBase + pool.triangles[t + 2]), doubleSided, reversed);
        if (!result) continue;
        const { distance } = result;
        const tieRejected = distance === closest && best !== null
          && (batch > best.batch || (batch === best.batch && instance >= best.instance));
        if (distance < ray.tMin || distance > ray.tMax || distance > closest || tieRejected) continue;
        closest = distance;
        best = { batch, instance, position: first + i, distance, u: result.u, v: result.v, front: result.determinant > 0 };
      }
    }
  }
  return { hit: best, workUnits: work, triangleTests: tests, overflow };
}

export interface CpuReferenceTriangle {
  batch: number; instance: number; position: number; a: V3; b: V3; c: V3; doubleSided: boolean;
  entityId: number; generation: number;
}

/** Independent brute force over every triangle: minimum of (distance, batch, instance, BLAS position). */
export function nearestTriangleCpu(triangles: readonly CpuReferenceTriangle[], ray: CpuRay, reversed = false): CpuTriangleHit | null {
  let best: CpuTriangleHit | null = null;
  for (const triangle of triangles) {
    if (ray.ignore && ray.ignore.entityId === triangle.entityId && ray.ignore.generation === triangle.generation) continue;
    const result = cpuIntersectTriangle(ray, triangle.a, triangle.b, triangle.c, triangle.doubleSided, reversed);
    if (!result || result.distance < ray.tMin || result.distance > ray.tMax) continue;
    const better = !best || result.distance < best.distance || (result.distance === best.distance
      && (triangle.batch < best.batch || (triangle.batch === best.batch && (triangle.instance < best.instance
        || (triangle.instance === best.instance && triangle.position < best.position)))));
    if (better) best = { batch: triangle.batch, instance: triangle.instance, position: triangle.position,
      distance: result.distance, u: result.u, v: result.v, front: result.determinant > 0 };
  }
  return best;
}

/** CPU mirror of the GPU pool update and TLAS refit, from per-instance BLAS bounds. */
export function buildCpuSharedPool(plan: SharedStaticPlan, inputs: readonly {
  /** BvhTopology.nodes words ([left|first, right|count, escape, leaf]). */
  topology: Uint32Array;
  leafTriangles: Uint32Array;
  /** Per active slot: world positions (xyz per vertex) and BLAS bounds (low.xyz, high.xyz per node). */
  instances: readonly { positions: Float32Array; bounds: Float32Array; entityId: number; generation: number }[];
  flags: number;
}[], batchSlots: number, tlasTopology: { nodes: Uint32Array; levels: readonly Uint32Array[] }): CpuSharedPool {
  if (inputs.length !== plan.batches.length) throw new RangeError('SHARED_STATIC_CPU_POOL_INPUTS');
  const nodes = new Uint32Array(plan.nodeRecords * 8), floats = new Float32Array(nodes.buffer);
  const triangles = new Uint32Array(plan.triangleCount * 3), positions = new Float32Array(plan.vertexCount * 3);
  const leaves = new Uint32Array(plan.leafCount * SHARED_LEAF_WORDS + batchSlots);
  const leafBounds: (Bounds3 | null)[] = Array(plan.leafCount).fill(null);
  plan.batches.forEach((batch, b) => {
    const input = inputs[b];
    triangles.set(input.leafTriangles, batch.triangleBase * 3);
    leaves.set(packSharedLeaves(batch, input.instances), batch.leafBase * SHARED_LEAF_WORDS);
    leaves[plan.leafCount * SHARED_LEAF_WORDS + batch.index] = input.flags;
    input.instances.forEach((instance, slot) => {
      positions.set(instance.positions, (batch.positionBase + slot * batch.vertexCount) * 3);
      for (let node = 0; node < batch.nodeCount; node++) {
        const record = batch.nodeBase + slot * batch.nodeCount + node, t = node * 4;
        floats.set(instance.bounds.subarray(node * 6, node * 6 + 3), record * 8);
        floats.set(instance.bounds.subarray(node * 6 + 3, node * 6 + 6), record * 8 + 4);
        nodes[record * 8 + 3] = input.topology[t + 2];
        nodes[record * 8 + 7] = input.topology[t + 3] === 1 ? (input.topology[t] << 4) | input.topology[t + 1] : 0;
      }
      const root = instance.bounds;
      leafBounds[batch.leafBase + slot] = { min: [root[0], root[1], root[2]], max: [root[3], root[4], root[5]] };
    });
  });
  const order = orderSharedLeaves(plan.leafCount, leafBounds);
  for (let depth = tlasTopology.levels.length - 1; depth >= 0; depth--) {
    for (const node of tlasTopology.levels[depth]) {
      const t = node * 4;
      let low: V3 = [Infinity, Infinity, Infinity], high: V3 = [-Infinity, -Infinity, -Infinity], valid = false, ref = 0;
      const merge = (record: number) => {
        for (let axis = 0; axis < 3; axis++) {
          low[axis] = Math.min(low[axis], floats[record * 8 + axis]);
          high[axis] = Math.max(high[axis], floats[record * 8 + 4 + axis]);
        }
        valid = true;
      };
      if (tlasTopology.nodes[t + 3] === 1) {
        const leaf = order[tlasTopology.nodes[t]];
        if (leaf !== INACTIVE && leaves[leaf * SHARED_LEAF_WORDS + 7] !== 0) {
          merge(leaves[leaf * SHARED_LEAF_WORDS + 4]); ref = leaf + 1;
        }
      } else {
        for (const child of [tlasTopology.nodes[t], tlasTopology.nodes[t + 1]]) {
          if (nodes[child * 8 + 7] & SHARED_TLAS_VALID) merge(child);
        }
      }
      if (!valid) { low = [0, 0, 0]; high = [0, 0, 0]; }
      floats.set(low, node * 8); floats.set(high, node * 8 + 4);
      nodes[node * 8 + 3] = tlasTopology.nodes[t + 2];
      nodes[node * 8 + 7] = (valid ? SHARED_TLAS_VALID : 0) | ref;
    }
  }
  return { nodes, triangles, positions, leaves, leafCount: plan.leafCount, tlasNodeCount: plan.tlasNodeCount };
}
