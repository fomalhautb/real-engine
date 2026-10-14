import { describe, expect, it } from 'vitest';
import { buildBvhTopology, type BvhTopology } from '../src/bvh-topology';

function mesh(xs: readonly number[], order: readonly number[] = xs.map((_, i) => i)) {
  const vertices = new ArrayBuffer(xs.length * 3 * 80);
  const view = new DataView(vertices);
  for (let triangle = 0; triangle < xs.length; triangle++) {
    for (let corner = 0; corner < 3; corner++) {
      const offset = (triangle * 3 + corner) * 80;
      view.setFloat32(offset, xs[triangle], true);
      view.setFloat32(offset + 4, corner === 1 ? 1 : 0, true);
      view.setFloat32(offset + 8, corner === 2 ? 1 : 0, true);
    }
  }
  return { vertices, indices: new Uint32Array(order.flatMap(id => [id * 3, id * 3 + 1, id * 3 + 2])), vertexCount: xs.length * 3 };
}

function leaves(topology: BvhTopology): { first: number; count: number }[] {
  const result: { first: number; count: number }[] = [];
  const visit = (node: number, depth: number): number => {
    const offset = node * 4;
    const [a, b, escape, kind] = topology.nodes.subarray(offset, offset + 4);
    expect(topology.levels[depth]).toContain(node);
    if (kind === 1) {
      expect(b).toBeGreaterThan(0);
      expect(b).toBeLessThanOrEqual(8);
      expect(escape).toBe(node + 1);
      result.push({ first: a, count: b });
    } else {
      expect(kind).toBe(0);
      expect(a).toBe(node + 1);
      expect(b).toBe(visit(a, depth + 1));
      expect(escape).toBe(visit(b, depth + 1));
    }
    return escape;
  };
  expect(visit(0, 0)).toBe(topology.nodeCount);
  return result;
}

describe('CPU BVH topology', () => {
  it('covers each original triangle once with valid preorder escape links and dependency levels', () => {
    const topology = buildBvhTopology(mesh([8, 1, 7, 0, 5, 2, 6, 3, 4]), { leafSize: 2 });
    expect(topology.triangleCount).toBe(9);
    expect(topology.nodes.length).toBe(topology.nodeCount * 4);
    expect(topology.levels.length).toBe(topology.maxDepth + 1);
    expect(topology.maxDepth).toBeLessThanOrEqual(32);
    const ranges = leaves(topology);
    expect(ranges[0].first).toBe(0);
    for (let i = 1; i < ranges.length; i++) expect(ranges[i].first).toBe(ranges[i - 1].first + ranges[i - 1].count);
    expect(ranges.at(-1)!.first + ranges.at(-1)!.count).toBe(9);
    expect([...topology.triangles].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const allLevels = topology.levels.flatMap(level => [...level]);
    expect([...allLevels].sort((a, b) => a - b)).toEqual(Array.from({ length: topology.nodeCount }, (_, i) => i));
  });

  it('clusters on the longest centroid axis and breaks identical-centroid ties by triangle ID', () => {
    const topology = buildBvhTopology(mesh([4, 1, 7, 0, 6, 2, 5, 3]), { leafSize: 2 });
    for (const { first, count } of leaves(topology)) {
      const values = [...topology.triangles.subarray(first, first + count)].map(id => [4, 1, 7, 0, 6, 2, 5, 3][id]);
      expect(Math.max(...values) - Math.min(...values)).toBe(1);
    }
    const same = mesh([0, 0, 0, 0, 0]);
    const a = buildBvhTopology(same, { leafSize: 1 });
    const b = buildBvhTopology(same, { leafSize: 1 });
    expect([...a.triangles]).toEqual([0, 1, 2, 3, 4]);
    expect(a).toEqual(b);
    expect(leaves(a)).toHaveLength(5);
    expect(a.maxDepth).toBeLessThanOrEqual(32);
  });

  it('retains degenerate triangles as original IDs', () => {
    const input = mesh([0, 1]);
    input.indices.set([0, 0, 0], 0);
    const topology = buildBvhTopology(input, { leafSize: 1 });
    expect([...topology.triangles].sort()).toEqual([0, 1]);
    expect(leaves(topology)).toHaveLength(2);
  });

  it('rejects malformed geometry and budget violations before topology construction', () => {
    const input = mesh([0, 1, 2]);
    expect(() => buildBvhTopology(input, { maxTriangles: 2 })).toThrow(/maxTriangles/);
    expect(() => buildBvhTopology(input, { maxTriangles: 262_145 })).toThrow(/maxTriangles/);
    expect(() => buildBvhTopology(input, { leafSize: 0 })).toThrow(/leafSize/);
    expect(() => buildBvhTopology(input, { leafSize: 9 })).toThrow(/leafSize/);
    expect(() => buildBvhTopology({ ...input, indices: new Uint32Array() })).toThrow(/indices/);
    expect(() => buildBvhTopology({ ...input, indices: new Uint32Array([0, 1]) })).toThrow(/indices/);
    expect(() => buildBvhTopology({ ...input, vertexCount: 8 })).toThrow(/vertices/);
    expect(() => buildBvhTopology({ ...input, indices: new Uint32Array([0, 1, 9]) })).toThrow(/missing vertex/);
    const bad = mesh([0]);
    new DataView(bad.vertices).setFloat32(0, NaN, true);
    expect(() => buildBvhTopology(bad)).toThrow(/nonfinite/);
  });
});
