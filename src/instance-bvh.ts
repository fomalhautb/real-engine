import type { Bounds3 } from './deformation';

/** Fixed-capacity preorder topology. A leaf's order index addresses the sorted instance slot buffer. */
export interface InstanceBvhTopology {
  /** Four u32s per node: internal [left, right, escape, 0], leaf [orderIndex, 1, escape, 1]. */
  nodes: Uint32Array<ArrayBuffer>;
  /** Node indices grouped by depth, root first, for bottom-up GPU refit. */
  levels: readonly Uint32Array<ArrayBuffer>[];
  nodeCount: number;
  /** Root depth is zero. */
  maxDepth: number;
}

const MAX_CAPACITY = 4096;
const INACTIVE_SLOT = 0xffff_ffff;

function validCapacity(capacity: number): void {
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > MAX_CAPACITY) {
    throw new RangeError(`capacity must be an integer in [1, ${MAX_CAPACITY}]`);
  }
}

/** Build a median split over fixed order-slot ranges, independent of instance positions. */
export function buildInstanceBvhTopology(capacity: number): InstanceBvhTopology {
  validCapacity(capacity);
  const nodeCount = capacity * 2 - 1;
  const nodes = new Uint32Array(nodeCount * 4);
  const levels: number[][] = [];
  let nextNode = 0;
  let maxDepth = 0;

  const build = (first: number, end: number, depth: number): number => {
    const node = nextNode++;
    (levels[depth] ??= []).push(node);
    maxDepth = Math.max(maxDepth, depth);
    const offset = node * 4;
    if (end - first === 1) {
      nodes[offset] = first;
      nodes[offset + 1] = 1;
      nodes[offset + 2] = nextNode;
      nodes[offset + 3] = 1;
      return node;
    }
    const middle = first + Math.floor((end - first) / 2);
    const left = build(first, middle, depth + 1);
    const right = build(middle, end, depth + 1);
    nodes[offset] = left;
    nodes[offset + 1] = right;
    nodes[offset + 2] = nextNode;
    return node;
  };

  build(0, capacity, 0);
  return { nodes, levels: levels.map(level => new Uint32Array(level)), nodeCount, maxDepth };
}

function validateBounds(bounds: unknown): asserts bounds is Bounds3 {
  if (!bounds || typeof bounds !== 'object') throw new RangeError('bounds must contain three-dimensional min/max');
  const { min, max } = bounds as Partial<Bounds3>;
  if (!Array.isArray(min) || !Array.isArray(max) || min.length !== 3 || max.length !== 3) {
    throw new RangeError('bounds must contain three-dimensional min/max');
  }
  for (let axis = 0; axis < 3; axis++) {
    if (typeof min[axis] !== 'number' || typeof max[axis] !== 'number'
      || !Number.isFinite(min[axis]) || !Number.isFinite(max[axis])
      || !Number.isFinite(Math.fround(min[axis])) || !Number.isFinite(Math.fround(max[axis]))
      || min[axis] > max[axis]) {
      throw new RangeError('bounds must be ordered and finite in float32');
    }
  }
}

function morton3(x: number, y: number, z: number): number {
  let code = 0;
  for (let bit = 0; bit < 10; bit++) {
    code |= ((x >>> bit) & 1) << (bit * 3);
    code |= ((y >>> bit) & 1) << (bit * 3 + 1);
    code |= ((z >>> bit) & 1) << (bit * 3 + 2);
  }
  return code >>> 0;
}

/** Sort conservative world bounds by 10-bit/axis center Morton code, with slot-ID ties. */
export function orderInstanceBounds(bounds: readonly Bounds3[], capacity: number): Uint32Array<ArrayBuffer> {
  validCapacity(capacity);
  if (!Array.isArray(bounds) || bounds.length > capacity) {
    throw new RangeError('bounds length exceeds capacity');
  }

  // Validate all input and the budget before allocating the output or sort metadata.
  const lower = [Infinity, Infinity, Infinity];
  const upper = [-Infinity, -Infinity, -Infinity];
  for (const bound of bounds) {
    validateBounds(bound);
    for (let axis = 0; axis < 3; axis++) {
      const center = bound.min[axis] / 2 + bound.max[axis] / 2;
      lower[axis] = Math.min(lower[axis], center);
      upper[axis] = Math.max(upper[axis], center);
    }
  }

  const quantize = (center: number, axis: number): number => {
    const span = upper[axis] - lower[axis];
    return span === 0 ? 0 : Math.min(1023, Math.floor((center - lower[axis]) / span * 1024));
  };
  const entries = bounds.map((bound, slot) => {
    const center = [0, 1, 2].map(axis => bound.min[axis] / 2 + bound.max[axis] / 2);
    return { slot, code: morton3(quantize(center[0], 0), quantize(center[1], 1), quantize(center[2], 2)) };
  });
  entries.sort((a, b) => a.code - b.code || a.slot - b.slot);
  const order = new Uint32Array(capacity);
  order.fill(INACTIVE_SLOT);
  for (let i = 0; i < entries.length; i++) order[i] = entries[i].slot;
  return order;
}
