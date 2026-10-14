import { describe, expect, it } from 'vitest';
import type { MeshData } from '../src/geometry';
import { buildGeometryHierarchy, selectGeometryCut } from '../src/geometry-hierarchy';
import type { GeometryBuildInput, GeometryBuildOptions, GeometryHierarchy } from '../src/geometry-hierarchy-types';

const BUILD_OPTIONS: GeometryBuildOptions = {
  maxClusterVertices: 64,
  maxClusterTriangles: 32,
  groupSize: 4,
  simplifyRatio: 0.5,
  maxSimplificationError: 2,
  maxLevels: 8,
  maxVertices: 1_000,
  maxTriangles: 2_000,
  maxOutputTriangles: 8_000,
  maxGroups: 500,
  maxClusters: 1_000,
};

/** Two curved grid halves share XYZ positions at x=n/2, but have distinct UVs there. */
function grid(n = 16): MeshData {
  const points: number[][] = [];
  const normals: number[][] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const halves: number[][][] = [];
  for (let half = 0; half < 2; half++) {
    const first = half === 0 ? 0 : n / 2;
    const last = half === 0 ? n / 2 : n;
    const rows: number[][] = [];
    for (let y = 0; y <= n; y++) {
      const row: number[] = [];
      for (let x = first; x <= last; x++) {
        const z = 0.15 * Math.sin(x * 0.41) * Math.cos(y * 0.37);
        const dx = 0.15 * 0.41 * Math.cos(x * 0.41) * Math.cos(y * 0.37);
        const dy = -0.15 * 0.37 * Math.sin(x * 0.41) * Math.sin(y * 0.37);
        const length = Math.hypot(dx, dy, 1);
        row.push(points.length);
        points.push([x, y, z]);
        normals.push([-dx / length, -dy / length, 1 / length]);
        uvs.push(x / n + (half === 1 ? 0.25 : 0), y / n);
        min[2] = Math.min(min[2], z);
        max[2] = Math.max(max[2], z);
      }
      rows.push(row);
    }
    halves.push(rows);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < last - first; x++) {
        const a = rows[y][x], b = rows[y][x + 1];
        const c = rows[y + 1][x], d = rows[y + 1][x + 1];
        indices.push(a, b, d, a, d, c);
      }
    }
  }
  expect(halves[0][0][n / 2]).not.toBe(halves[1][0][0]);
  const vertices = new ArrayBuffer(points.length * 80);
  const view = new DataView(vertices);
  for (let i = 0; i < points.length; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(offset + axis * 4, points[i][axis], true);
      view.setFloat32(offset + 16 + axis * 4, normals[i][axis], true);
    }
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 48, 1, true);
  }
  return {
    vertices,
    indices: new Uint32Array(indices),
    uvs: new Float32Array(uvs),
    vertexCount: points.length,
    bounds: { min: [0, 0, min[2]], max: [n, n, max[2]] },
  };
}

function input(mesh = grid()): GeometryBuildInput {
  return { primitiveKey: 'curved-grid/0', mesh, coverage: 'opaque' };
}

function positionKey(mesh: MeshData, vertex: number): string {
  expect(vertex).toBeGreaterThanOrEqual(0);
  expect(vertex).toBeLessThan(mesh.vertexCount);
  const view = new DataView(mesh.vertices);
  return [0, 4, 8].map(offset => view.getUint32(vertex * 80 + offset, true).toString(16)).join(',');
}

type Triangle = readonly [number, number, number];

function clusterTriangles(hierarchy: GeometryHierarchy, ids: readonly number[]): Triangle[] {
  const result: Triangle[] = [];
  for (const id of ids) {
    const cluster = hierarchy.clusters[id];
    expect(cluster).toBeDefined();
    expect(cluster.triangles.length % 3).toBe(0);
    for (let t = 0; t < cluster.triangles.length; t += 3) {
      const local = cluster.triangles;
      const a = local[t], b = local[t + 1], c = local[t + 2];
      expect(a).toBeLessThan(cluster.vertices.length);
      expect(b).toBeLessThan(cluster.vertices.length);
      expect(c).toBeLessThan(cluster.vertices.length);
      result.push([cluster.vertices[a], cluster.vertices[b], cluster.vertices[c]]);
    }
  }
  return result;
}

function sourceTriangles(mesh: MeshData): Triangle[] {
  const result: Triangle[] = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    result.push([mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]]);
  }
  return result;
}

function orientedTriangle(mesh: MeshData, [a, b, c]: Triangle): string {
  const keys = [positionKey(mesh, a), positionKey(mesh, b), positionKey(mesh, c)];
  return [
    `${keys[0]}/${keys[1]}/${keys[2]}`,
    `${keys[1]}/${keys[2]}/${keys[0]}`,
    `${keys[2]}/${keys[0]}/${keys[1]}`,
  ].sort()[0];
}

/** Cancel opposite oriented edges, retaining the exact directed boundary multiset. */
function directedBoundary(mesh: MeshData, triangles: readonly Triangle[]): [string, number][] {
  const edges = new Map<string, number>();
  for (const triangle of triangles) {
    const keys = triangle.map(vertex => positionKey(mesh, vertex));
    for (let corner = 0; corner < 3; corner++) {
      const a = keys[corner], b = keys[(corner + 1) % 3];
      const forward = `${a}>${b}`, reverse = `${b}>${a}`;
      const reverseCount = edges.get(reverse) ?? 0;
      if (reverseCount > 0) {
        if (reverseCount === 1) edges.delete(reverse);
        else edges.set(reverse, reverseCount - 1);
      } else edges.set(forward, (edges.get(forward) ?? 0) + 1);
    }
  }
  return [...edges].sort(([a], [b]) => a.localeCompare(b));
}

function contains(outer: MeshData['bounds'], inner: MeshData['bounds']): void {
  for (let axis = 0; axis < 3; axis++) {
    expect(outer.min[axis]).toBeLessThanOrEqual(inner.min[axis] + 1e-5);
    expect(outer.max[axis]).toBeGreaterThanOrEqual(inner.max[axis] - 1e-5);
  }
}

function point(mesh: MeshData, vertex: number): [number, number] {
  const view = new DataView(mesh.vertices);
  return [view.getFloat32(vertex * 80, true), view.getFloat32(vertex * 80 + 4, true)];
}

/** Every off-edge XY sample on this height field must have exactly one selected triangle. */
function expectSingleCoverage(mesh: MeshData, triangles: readonly Triangle[], n: number): void {
  const xy = Array.from({ length: mesh.vertexCount }, (_, vertex) => point(mesh, vertex));
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const px = x + 0.317, py = y + 0.593;
      let hits = 0;
      for (const [a, b, c] of triangles) {
        const [ax, ay] = xy[a], [bx, by] = xy[b], [cx, cy] = xy[c];
        const ab = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        const bc = (cx - bx) * (py - by) - (cy - by) * (px - bx);
        const ca = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
        if ((ab > 1e-8 && bc > 1e-8 && ca > 1e-8) ||
            (ab < -1e-8 && bc < -1e-8 && ca < -1e-8)) hits++;
      }
      expect(hits, `coverage at (${px}, ${py})`).toBe(1);
    }
  }
}

describe('offline geometry hierarchy', () => {
  it('covers each source triangle exactly once at LOD0 and produces actual simplification', async () => {
    const mesh = grid();
    const hierarchy = await buildGeometryHierarchy(input(mesh), BUILD_OPTIONS);
    expect(hierarchy.version).toBe(1);
    expect(hierarchy.sourceTriangleCount).toBe(mesh.indices.length / 3);
    const seen: number[] = [];
    for (const cluster of hierarchy.clusters.filter(cluster => cluster.level === 0)) {
      expect(cluster.refinedGroup).toBeNull();
      expect(cluster.sourceTriangles).toBeInstanceOf(Uint32Array);
      expect(cluster.sourceTriangles!.length).toBe(cluster.triangles.length / 3);
      seen.push(...cluster.sourceTriangles!);
      const triangles = clusterTriangles(hierarchy, [hierarchy.clusters.indexOf(cluster)]);
      for (let t = 0; t < triangles.length; t++) {
        const sourceId = cluster.sourceTriangles![t];
        expect(orientedTriangle(hierarchy.source, triangles[t])).toBe(
          orientedTriangle(mesh, [mesh.indices[sourceId * 3], mesh.indices[sourceId * 3 + 1], mesh.indices[sourceId * 3 + 2]]),
        );
      }
    }
    expect(seen.sort((a, b) => a - b)).toEqual(
      Array.from({ length: hierarchy.sourceTriangleCount }, (_, i) => i),
    );
    const roots = clusterTriangles(hierarchy, hierarchy.roots);
    expect(roots.length).toBeLessThan(hierarchy.sourceTriangleCount);
    expect(hierarchy.clusters.some(cluster => cluster.level > 0)).toBe(true);
    expect(directedBoundary(hierarchy.source, roots)).toEqual(directedBoundary(mesh, sourceTriangles(mesh)));
  });

  it('keeps group relationships, LOD bounds and approximate errors monotonic, with identical replacement boundaries', async () => {
    const hierarchy = await buildGeometryHierarchy(input(), BUILD_OPTIONS);
    const terminalChildren = new Set<number>();
    for (let groupId = 0; groupId < hierarchy.groups.length; groupId++) {
      const group = hierarchy.groups[groupId];
      expect(group.children.length).toBeGreaterThan(0);
      expect(Number.isFinite(group.localError)).toBe(true);
      expect(group.localError).toBeGreaterThanOrEqual(0);
      expect(group.error).toBeGreaterThanOrEqual(group.localError);
      for (const childId of group.children) {
        const child = hierarchy.clusters[childId];
        expect(child).toBeDefined();
        expect(child.group).toBe(groupId);
        expect(child.level).toBe(group.level);
        expect(group.error + 1e-6).toBeGreaterThanOrEqual(child.error);
        contains(group.bounds, child.lodBounds);
      }
      if (group.terminal) {
        expect(group.parents).toEqual([]);
        for (const childId of group.children) terminalChildren.add(childId);
      } else {
        expect(group.parents.length).toBeGreaterThan(0);
        for (const parentId of group.parents) {
          const parent = hierarchy.clusters[parentId];
          expect(parent).toBeDefined();
          expect(parent.refinedGroup).toBe(groupId);
          expect(parent.level).toBe(group.level + 1);
          expect(parent.error + 1e-6).toBeGreaterThanOrEqual(group.error);
          contains(parent.lodBounds, group.bounds);
        }
        const detailed = directedBoundary(hierarchy.source, clusterTriangles(hierarchy, group.children));
        const replacement = directedBoundary(hierarchy.source, clusterTriangles(hierarchy, group.parents));
        expect(replacement, `boundary of group ${groupId}`).toEqual(detailed);
      }
    }
    expect([...hierarchy.roots].sort((a, b) => a - b)).toEqual([...terminalChildren].sort((a, b) => a - b));
    for (const [clusterId, cluster] of hierarchy.clusters.entries()) {
      expect(hierarchy.groups[cluster.group]?.children).toContain(clusterId);
      if (cluster.level > 0) {
        expect(cluster.refinedGroup).not.toBeNull();
        expect(hierarchy.groups[cluster.refinedGroup!]?.parents).toContain(clusterId);
      }
    }
  });

  it('selects complete fine, coarse and intermediate cuts without seams or overlapping coverage', async () => {
    const n = 16;
    const hierarchy = await buildGeometryHierarchy(input(grid(n)), BUILD_OPTIONS);
    const outside = directedBoundary(hierarchy.source, sourceTriangles(hierarchy.source));
    const base = { eye: [8, 8, 6] as [number, number, number], projectionScale: 800, near: 0.1 };
    const fine = selectGeometryCut(hierarchy, { ...base, maxErrorPixels: 0 });
    const coarse = selectGeometryCut(hierarchy, { ...base, maxErrorPixels: 1e9 });
    expect(coarse.slice().sort((a, b) => a - b)).toEqual(hierarchy.roots.slice().sort((a, b) => a - b));
    expect(fine.length).toBeGreaterThan(coarse.length);
    for (const limit of [0, 0.1, 1, 4, 16, 64, 1e9]) {
      const cut = selectGeometryCut(hierarchy, { ...base, maxErrorPixels: limit });
      expect(new Set(cut).size).toBe(cut.length);
      for (const id of cut) {
        const cluster = hierarchy.clusters[id];
        const group = hierarchy.groups[cluster.group];
        const projected = (g: typeof group) => {
          const sq = g.bounds.min.reduce((sum, min, axis) => {
            const value = base.eye[axis];
            const delta = value < min ? min - value : Math.max(0, value - g.bounds.max[axis]);
            return sum + delta * delta;
          }, 0);
          return g.error * base.projectionScale / Math.max(Math.sqrt(sq), base.near);
        };
        expect(group.terminal || projected(group) > limit).toBe(true);
        expect(cluster.refinedGroup === null || projected(hierarchy.groups[cluster.refinedGroup]) <= limit).toBe(true);
      }
      const triangles = clusterTriangles(hierarchy, cut);
      expect(directedBoundary(hierarchy.source, triangles), `cut boundary at ${limit}`).toEqual(outside);
      expectSingleCoverage(hierarchy.source, triangles, n);
    }
  });

  it('ignores an unindexed, distant finite vertex when building bounds and selecting LOD', async () => {
    const mesh = grid(16);
    const extraVertices = new ArrayBuffer(mesh.vertices.byteLength + 80);
    new Uint8Array(extraVertices).set(new Uint8Array(mesh.vertices));
    const extraView = new DataView(extraVertices);
    const offset = mesh.vertices.byteLength;
    extraView.setFloat32(offset, 1e30, true);
    extraView.setFloat32(offset + 4, -1e30, true);
    extraView.setFloat32(offset + 8, 1e30, true);
    extraView.setFloat32(offset + 12, 1, true);
    extraView.setFloat32(offset + 24, 1, true);
    extraView.setFloat32(offset + 48, 1, true);
    const extraUvs = new Float32Array(mesh.uvs!.length + 2);
    extraUvs.set(mesh.uvs!);
    const withUnindexed: MeshData = { ...mesh, vertices: extraVertices, vertexCount: mesh.vertexCount + 1, uvs: extraUvs };
    const base = await buildGeometryHierarchy(input(mesh), BUILD_OPTIONS);
    const extended = await buildGeometryHierarchy(input(withUnindexed), BUILD_OPTIONS);
    expect(extended.source.bounds).toEqual(base.source.bounds);
    expect(extended.clusters).toEqual(base.clusters);
    expect(extended.groups).toEqual(base.groups);
    expect(extended.roots).toEqual(base.roots);
    for (const eye of [[8, 8, 2], [8, 8, 200]] as const) {
      for (const maxErrorPixels of [0, 1, 16, 1e9]) {
        const view = { eye, projectionScale: 800, near: 0.1, maxErrorPixels };
        expect(selectGeometryCut(extended, view)).toEqual(selectGeometryCut(base, view));
      }
    }
  });

  it('does not mutate input and produces deterministic output', async () => {
    const mesh = grid(8);
    const originalVertices = new Uint8Array(mesh.vertices).slice();
    const originalIndices = mesh.indices.slice();
    const originalUvs = mesh.uvs!.slice();
    const originalBounds = structuredClone(mesh.bounds);
    const first = await buildGeometryHierarchy(input(mesh), BUILD_OPTIONS);
    const second = await buildGeometryHierarchy(input(mesh), BUILD_OPTIONS);
    expect(new Uint8Array(mesh.vertices)).toEqual(originalVertices);
    expect(mesh.indices).toEqual(originalIndices);
    expect(mesh.uvs).toEqual(originalUvs);
    expect(mesh.bounds).toEqual(originalBounds);
    expect(second).toEqual(first);
    const ownedBounds = structuredClone(first.source.bounds);
    new Uint8Array(mesh.vertices)[0] ^= 0xff;
    mesh.indices[0] = mesh.indices[1];
    mesh.uvs![0] = 99;
    mesh.bounds.min[0] = -999;
    expect(new Uint8Array(first.source.vertices)).toEqual(originalVertices);
    expect(first.source.indices).toEqual(originalIndices);
    expect(first.source.uvs).toEqual(originalUvs);
    expect(first.source.bounds).toEqual(ownedBounds);
  });

  it('rejects unsupported coverage/deformation, malformed geometry and exceeded build budgets', async () => {
    const mesh = grid(8);
    await expect(buildGeometryHierarchy({ ...input(mesh), coverage: 'mask' }, BUILD_OPTIONS)).rejects.toThrow();
    await expect(buildGeometryHierarchy({ ...input(mesh), deformation: {
      jointCount: 0, nodes: [{ kind: 'wave', amplitude: 1, frequency: 1, phase: 0 }],
    } }, BUILD_OPTIONS)).rejects.toThrow();
    await expect(buildGeometryHierarchy(input({ ...mesh, indices: new Uint32Array([0, 1, mesh.vertexCount]) }), BUILD_OPTIONS)).rejects.toThrow();
    const nonfinite = mesh.vertices.slice(0);
    new DataView(nonfinite).setFloat32(0, NaN, true);
    await expect(buildGeometryHierarchy(input({ ...mesh, vertices: nonfinite }), BUILD_OPTIONS)).rejects.toThrow();
    const tight: GeometryBuildOptions[] = [
      { maxVertices: mesh.vertexCount - 1 },
      { maxTriangles: mesh.indices.length / 3 - 1 },
      { maxOutputTriangles: mesh.indices.length / 3 - 1 },
      { maxGroups: 1 },
      { maxClusters: 1 },
      { maxLevels: 0 },
      { maxVertices: 1_000_000_000 },
      { maxTriangles: 1_000_000_000 },
      { maxOutputTriangles: 1_000_000_000 },
      { maxGroups: 1_000_000_000 },
      { maxClusters: 1_000_000_000 },
      { maxLevels: 1_000_000_000 },
    ];
    for (const limits of tight) {
      await expect(buildGeometryHierarchy(input(mesh), { ...BUILD_OPTIONS, ...limits })).rejects.toThrow();
    }
  });
});
