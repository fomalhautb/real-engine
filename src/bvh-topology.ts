import type { MeshData } from './geometry';

/** Four u32s per preorder node. Internal: [left, right, escape, 0]; leaf: [first triangle, count, escape, 1]. */
export interface BvhTopology {
  nodes: Uint32Array;
  /** Original triangle IDs, reordered so each leaf owns a contiguous range. */
  triangles: Uint32Array;
  /** Node indices grouped by depth, from root to leaves, for reverse-order GPU refit. */
  levels: readonly Uint32Array[];
  nodeCount: number;
  triangleCount: number;
  /** Root depth is zero. */
  maxDepth: number;
}

const VERTEX_STRIDE = 80;
const HARD_TRIANGLE_LIMIT = 262_144;

function option(value: number, low: number, high: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < low || value > high) {
    throw new RangeError(`${label} must be an integer in [${low}, ${high}]`);
  }
  return value;
}

/** Build bounded median-split topology only; bounds and intersection data are refit elsewhere. */
export function buildBvhTopology(
  mesh: Pick<MeshData, 'vertices' | 'indices' | 'vertexCount'>,
  options: { maxTriangles?: number; leafSize?: number } = {},
): BvhTopology {
  const maxTriangles = option(options.maxTriangles ?? 65_536, 1, HARD_TRIANGLE_LIMIT, 'maxTriangles');
  const leafSize = option(options.leafSize ?? 4, 1, 8, 'leafSize');
  if (!(mesh?.indices instanceof Uint32Array) || mesh.indices.length === 0 || mesh.indices.length % 3 !== 0) {
    throw new RangeError('indices must be a nonempty Uint32Array triangle list');
  }
  const triangleCount = mesh.indices.length / 3;
  if (triangleCount > maxTriangles) throw new RangeError('triangle count exceeds maxTriangles');
  if (!(mesh.vertices instanceof ArrayBuffer) || !Number.isSafeInteger(mesh.vertexCount) || mesh.vertexCount < 1
    || mesh.vertexCount > Number.MAX_SAFE_INTEGER / VERTEX_STRIDE
    || mesh.vertices.byteLength !== mesh.vertexCount * VERTEX_STRIDE) {
    throw new RangeError('vertices must contain exactly vertexCount interleaved 80-byte vertices');
  }

  const source = new DataView(mesh.vertices);
  const centroids = new Float64Array(triangleCount * 3);
  const triangles = new Uint32Array(triangleCount);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    triangles[triangle] = triangle;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = mesh.indices[triangle * 3 + corner];
      if (vertex >= mesh.vertexCount) throw new RangeError(`triangle ${triangle} references a missing vertex`);
      const offset = vertex * VERTEX_STRIDE;
      for (let axis = 0; axis < 3; axis++) {
        const position = source.getFloat32(offset + axis * 4, true);
        if (!Number.isFinite(position)) throw new RangeError(`triangle ${triangle} has a nonfinite position`);
        centroids[triangle * 3 + axis] += position / 3;
      }
    }
  }

  // The binary tree has at most 2*n-1 nodes; no triangle-sized allocation occurs before budget validation.
  const nodes = new Uint32Array((triangleCount * 2 - 1) * 4);
  const levels: number[][] = [];
  let nodeCount = 0;
  let maxDepth = 0;
  const build = (first: number, end: number, depth: number): number => {
    if (depth > 32) throw new RangeError('BVH depth exceeds 32');
    const node = nodeCount++;
    (levels[depth] ??= []).push(node);
    maxDepth = Math.max(maxDepth, depth);
    const offset = node * 4;
    if (end - first <= leafSize) {
      nodes[offset] = first;
      nodes[offset + 1] = end - first;
      nodes[offset + 2] = nodeCount;
      nodes[offset + 3] = 1;
      return node;
    }

    const minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
    for (let i = first; i < end; i++) {
      const centroid = triangles[i] * 3;
      for (let axis = 0; axis < 3; axis++) {
        minimum[axis] = Math.min(minimum[axis], centroids[centroid + axis]);
        maximum[axis] = Math.max(maximum[axis], centroids[centroid + axis]);
      }
    }
    let axis = 0;
    for (let candidate = 1; candidate < 3; candidate++) {
      if (maximum[candidate] - minimum[candidate] > maximum[axis] - minimum[axis]) axis = candidate;
    }
    triangles.subarray(first, end).sort((a, b) =>
      centroids[a * 3 + axis] - centroids[b * 3 + axis] || a - b);
    const middle = first + Math.floor((end - first) / 2);
    const left = build(first, middle, depth + 1);
    const right = build(middle, end, depth + 1);
    nodes[offset] = left;
    nodes[offset + 1] = right;
    nodes[offset + 2] = nodeCount;
    return node;
  };
  build(0, triangleCount, 0);
  return { nodes: nodes.subarray(0, nodeCount * 4), triangles,
    levels: levels.map(level => new Uint32Array(level)), nodeCount, triangleCount, maxDepth };
}
