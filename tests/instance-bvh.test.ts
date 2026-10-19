import { describe, expect, it } from 'vitest';
import type { Bounds3 } from '../src/deformation';
import { buildInstanceBvhTopology, orderInstanceBounds, type InstanceBvhTopology } from '../src/instance-bvh';

function box(x: number, y = 0, z = 0): Bounds3 {
  return { min: [x - 0.25, y - 0.25, z - 0.25], max: [x + 0.25, y + 0.25, z + 0.25] };
}

function verifyTopology(topology: InstanceBvhTopology, capacity: number): void {
  expect(topology.nodeCount).toBe(2 * capacity - 1);
  expect(topology.nodes).toHaveLength(topology.nodeCount * 4);
  expect(topology.nodes.buffer).toBeInstanceOf(ArrayBuffer);
  expect(topology.levels).toHaveLength(topology.maxDepth + 1);
  const visited = new Set<number>();
  const slots: number[] = [];
  const visit = (node: number, first: number, end: number, depth: number): number => {
    expect(visited.has(node)).toBe(false);
    visited.add(node);
    expect(topology.levels[depth]).toContain(node);
    const [a, b, escape, kind] = topology.nodes.subarray(node * 4, node * 4 + 4);
    if (end - first === 1) {
      expect([a, b, escape, kind]).toEqual([first, 1, node + 1, 1]);
      slots.push(a);
    } else {
      const middle = first + Math.floor((end - first) / 2);
      expect(kind).toBe(0);
      expect(a).toBe(node + 1);
      expect(b).toBe(visit(a, first, middle, depth + 1));
      expect(escape).toBe(visit(b, middle, end, depth + 1));
      expect(topology.nodes[node * 4 + 3]).toBe(0);
    }
    return escape;
  };
  expect(visit(0, 0, capacity, 0)).toBe(topology.nodeCount);
  expect(slots).toEqual(Array.from({ length: capacity }, (_, i) => i));
  expect([...visited].sort((a, b) => a - b)).toEqual(Array.from({ length: topology.nodeCount }, (_, i) => i));
  expect(topology.levels.flatMap(level => [...level]).sort((a, b) => a - b))
    .toEqual(Array.from({ length: topology.nodeCount }, (_, i) => i));
  expect(topology.maxDepth).toBeLessThanOrEqual(12);
}

describe('fixed-capacity instance BVH topology', () => {
  it('covers odd capacities and singleton leaves with exact preorder escape links', () => {
    for (const capacity of [1, 2, 3, 5, 9, 17]) verifyTopology(buildInstanceBvhTopology(capacity), capacity);
  });

  it('handles the 4096-slot limit within depth 12', () => {
    const topology = buildInstanceBvhTopology(4096);
    verifyTopology(topology, 4096);
    expect(topology.maxDepth).toBe(12);
  });

  it('rejects invalid capacities before constructing nodes', () => {
    for (const capacity of [0, -1, 1.5, NaN, Infinity, 4097]) {
      expect(() => buildInstanceBvhTopology(capacity)).toThrow(/capacity/);
    }
  });
});

describe('conservative instance-bound ordering', () => {
  it('returns a fixed empty order and validates capacity before reading bounds', () => {
    expect([...orderInstanceBounds([], 4)]).toEqual([0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]);
    expect(() => orderInstanceBounds([], 0)).toThrow(/capacity/);
    expect(() => orderInstanceBounds([box(0), box(1)], 1)).toThrow(/capacity/);
  });

  it('clusters centers on a single axis and breaks coincident Morton codes by slot ID', () => {
    const bounds = [box(8), box(2), box(2), box(0), box(10), box(5)];
    const snapshot = structuredClone(bounds);
    expect([...orderInstanceBounds(bounds, 8)]).toEqual([3, 1, 2, 5, 0, 4, 0xffff_ffff, 0xffff_ffff]);
    expect(bounds).toEqual(snapshot);
    expect([...orderInstanceBounds([box(1), box(1), box(1)], 3)]).toEqual([0, 1, 2]);
  });

  it('orders randomized bounded inputs deterministically and retains each active slot once', () => {
    let state = 0x1234_5678;
    const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 0x1_0000_0000; };
    const bounds = Array.from({ length: 257 }, () => box(
      Math.floor(random() * 1000), Math.floor(random() * 1000), Math.floor(random() * 1000)));
    const first = orderInstanceBounds(bounds, 300);
    const second = orderInstanceBounds(bounds, 300);
    expect(first).toEqual(second);
    expect([...first.subarray(0, bounds.length)].sort((a, b) => a - b))
      .toEqual(Array.from({ length: bounds.length }, (_, i) => i));
    expect([...first.subarray(bounds.length)]).toEqual(Array(43).fill(0xffff_ffff));
    expect(bounds.every(bound => bound.min.every(Number.isFinite) && bound.max.every(Number.isFinite))).toBe(true);
  });

  it('rejects malformed, non-float32 and inverted world bounds', () => {
    const invalid: unknown[] = [
      null,
      { min: [0, 0], max: [1, 1, 1] },
      { min: [0, 0, 0], max: [1, 1, 1, 1] },
      { min: [0, NaN, 0], max: [1, 1, 1] },
      { min: [0, 0, 0], max: [Infinity, 1, 1] },
      { min: [0, 0, 0], max: [Number.MAX_VALUE, 1, 1] },
      { min: [1, 0, 0], max: [0, 1, 1] },
    ];
    for (const bound of invalid) expect(() => orderInstanceBounds([bound as Bounds3], 1)).toThrow(/bounds/);
    expect(() => orderInstanceBounds(null as unknown as Bounds3[], 1)).toThrow(/bounds/);
  });
});
