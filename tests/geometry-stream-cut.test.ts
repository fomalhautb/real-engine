import { describe, expect, it } from 'vitest';
import { buildGeometryHierarchy, selectGeometryCut } from '../src/geometry-hierarchy';
import type { GeometryHierarchy } from '../src/geometry-hierarchy-types';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory, validateGeometryStreamDirectory } from '../src/geometry-stream-directory';
import { GeometryStreamCut } from '../src/geometry-stream-cut';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { identity, scaling, translation, multiply } from '../src/math';

const N = 16;
function curvedGrid() {
  const rows: number[][] = [], indices: number[] = [], vertices = new ArrayBuffer((N + 1) * (N + 1) * 80);
  const view = new DataView(vertices);
  let minZ = Infinity, maxZ = -Infinity;
  for (let y = 0; y <= N; y++) {
    const row: number[] = [];
    for (let x = 0; x <= N; x++) {
      const id = y * (N + 1) + x, z = 0.2 * Math.sin(x * 0.47) * Math.cos(y * 0.39);
      view.setFloat32(id * 80, x, true); view.setFloat32(id * 80 + 4, y, true);
      view.setFloat32(id * 80 + 8, z, true); view.setFloat32(id * 80 + 12, 1, true);
      view.setFloat32(id * 80 + 24, 1, true); view.setFloat32(id * 80 + 48, 1, true);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z); row.push(id);
    }
    rows.push(row);
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = rows[y][x], b = rows[y][x + 1], c = rows[y + 1][x], d = rows[y + 1][x + 1];
    indices.push(a, b, d, a, d, c);
  }
  return { vertices, vertexCount: (N + 1) ** 2, indices: new Uint32Array(indices),
    bounds: { min: [0, 0, minZ] as [number, number, number], max: [N, N, maxZ] as [number, number, number] } };
}

async function fixture() {
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'stream/curved', mesh: curvedGrid(), coverage: 'opaque' }, {
    maxClusterTriangles: 32, groupSize: 4, simplifyRatio: 0.5, maxSimplificationError: 2, maxLevels: 8,
  });
  const clusterBytes = hierarchy.clusters.map(cluster => 16 + cluster.vertices.length * 84
    + ((cluster.triangles.length + 3) & ~3));
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: Math.max(...clusterBytes) });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  return { hierarchy, encoded, directory, planner: new GeometryStreamCut(directory) };
}

function boundary(hierarchy: GeometryHierarchy, clusterIds: readonly number[]): string[] {
  const view = new DataView(hierarchy.source.vertices), edges = new Map<string, number>();
  const key = (vertex: number) => [0, 4, 8].map(offset => view.getUint32(vertex * 80 + offset, true)).join(':');
  for (const id of clusterIds) {
    const cluster = hierarchy.clusters[id];
    for (let i = 0; i < cluster.triangles.length; i += 3) {
      const triangle = Array.from(cluster.triangles.subarray(i, i + 3), local => key(cluster.vertices[local]));
      for (let edge = 0; edge < 3; edge++) {
        const a = triangle[edge], b = triangle[(edge + 1) % 3];
        const name = a < b ? `${a}|${b}` : `${b}|${a}`;
        const value = (edges.get(name) ?? 0) + (a < b ? 1 : -1);
        if (value) edges.set(name, value); else edges.delete(name);
      }
    }
  }
  return [...edges].sort(([a], [b]) => a.localeCompare(b)).map(([edge, count]) => `${edge}:${count}`);
}

function coverage(hierarchy: GeometryHierarchy, clusterIds: readonly number[]): void {
  const view = new DataView(hierarchy.source.vertices);
  const xy = (vertex: number) => [view.getFloat32(vertex * 80, true), view.getFloat32(vertex * 80 + 4, true)];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const px = x + 0.317, py = y + 0.593;
    let hits = 0;
    for (const id of clusterIds) {
      const cluster = hierarchy.clusters[id];
      for (let i = 0; i < cluster.triangles.length; i += 3) {
        const [a, b, c] = Array.from(cluster.triangles.subarray(i, i + 3), local => xy(cluster.vertices[local]));
        const ab = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
        const bc = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0]);
        const ca = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0]);
        if (ab > 1e-8 && bc > 1e-8 && ca > 1e-8 || ab < -1e-8 && bc < -1e-8 && ca < -1e-8) hits++;
      }
    }
    expect(hits, `coverage at ${x},${y}`).toBe(1);
  }
}

describe('compact streamed geometry cut', () => {
  it('rejects sparse ID lists before accepting a root or page mapping', () => {
    const bounds = { min: [0, 0, 0] as [number, number, number],
      max: [1, 1, 1] as [number, number, number] };
    const minimal: GeometryStreamDirectory = {
      version: 1, primitiveKey: 'sparse-ids', sourceVertexCount: 3, sourceTriangleCount: 1, vertexStride: 80,
      clusters: [{ level: 0, group: 0, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds, lodBounds: bounds, error: 0 }],
      groups: [{ level: 0, children: [0], parents: [], bounds, error: 0, terminal: true }],
      roots: [0], pages: [{ id: 0, clusterIds: [0], encodedBytes: 41, decodedBytes: 272,
        sha256: '0'.repeat(64) }],
      clusterPages: [0], rootPages: [0],
    };
    expect(() => validateGeometryStreamDirectory(minimal)).not.toThrow();
    expect(() => validateGeometryStreamDirectory({ ...minimal,
      roots: new Array<number>(1), rootPages: new Array<number>(1) })).toThrow('ROOTS');
    expect(() => validateGeometryStreamDirectory({ ...minimal,
      groups: [{ ...minimal.groups[0], children: new Array<number>(1) }] })).toThrow('GROUP_CHILDREN');
    expect(() => validateGeometryStreamDirectory({ ...minimal,
      pages: [{ ...minimal.pages[0], clusterIds: new Array<number>(1) }] })).toThrow('PAGE_CLUSTERS');
  });

  it('stores owned metadata only, hashes pages, and rejects broken relationships and budgets', async () => {
    const { hierarchy, encoded, directory } = await fixture();
    expect('source' in directory).toBe(false);
    expect(directory.clusters[0]).not.toHaveProperty('vertices');
    expect(directory.pages[0].sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(directory.pages.map(page => page.encodedBytes)).toEqual(encoded.pages.map(page => page.bytes.length));
    const original = directory.clusters[0].bounds.min[0];
    const broken = structuredClone(directory) as GeometryStreamDirectory;
    (broken.groups[0].children as number[]).push(broken.groups[0].children[0]);
    expect(() => validateGeometryStreamDirectory(broken)).toThrow('GROUP_CHILDREN');
    const missingRoot = structuredClone(directory) as GeometryStreamDirectory;
    (missingRoot.roots as number[]).pop();
    expect(() => validateGeometryStreamDirectory(missingRoot)).toThrow('ROOT_COVERAGE');
    const badPage = structuredClone(directory) as GeometryStreamDirectory;
    (badPage.pages[0] as { decodedBytes: number }).decodedBytes = 4 * 1024 * 1024 + 1;
    expect(() => validateGeometryStreamDirectory(badPage)).toThrow('PAGE_DECODED_BYTES');
    const badMapping = structuredClone(directory) as GeometryStreamDirectory;
    (badMapping.clusterPages as number[])[0] = (badMapping.clusterPages[0] + 1) % badMapping.pages.length;
    expect(() => validateGeometryStreamDirectory(badMapping)).toThrow('PAGE_OWNERSHIP');
    const tooManyReferences = structuredClone(directory) as GeometryStreamDirectory;
    (tooManyReferences.groups[0] as unknown as { children: number[] }).children = Array(directory.clusters.length * 2 + 1).fill(0);
    expect(() => validateGeometryStreamDirectory(tooManyReferences)).toThrow('GROUP_REFERENCE_BUDGET');
    const tooManyPageReferences = structuredClone(directory) as GeometryStreamDirectory;
    (tooManyPageReferences.pages[0] as unknown as { clusterIds: number[] }).clusterIds = Array(directory.clusters.length + 1).fill(0);
    expect(() => validateGeometryStreamDirectory(tooManyPageReferences)).toThrow('PAGE_REFERENCE_BUDGET');
    const encodedOverBudget = { ...encoded, pages: encoded.pages.map((page, i) => i === 0
      ? { ...page, bytes: new Uint8Array(8 * 1024 * 1024 + 41) } : page) };
    await expect(buildGeometryStreamDirectory(hierarchy, encodedOverBudget)).rejects.toThrow('PAGE_ENCODED_BYTES');
    hierarchy.clusters[0].bounds.min[0] -= 9;
    expect(directory.clusters[0].bounds.min[0]).toBe(original);
  });

  it('matches the normal cut when every page is resident and preserves complete coverage with a missing child page', async () => {
    const { hierarchy, directory, planner } = await fixture();
    expect(new Set(hierarchy.roots.map(id => hierarchy.clusters[id].level)).size).toBeGreaterThan(1);
    expect(hierarchy.groups.some(group => new Set(group.parents.map(id => hierarchy.clusters[id].group)).size > 1)).toBe(true);
    const all = new Set(directory.pages.map(page => page.id));
    const base = { eye: [8, 8, 6] as [number, number, number], projectionScale: 800, near: 0.1 };
    for (const maxErrorPixels of [0, 0.1, 1, 4, 16, 64, 1e9]) {
      const view = { ...base, maxErrorPixels };
      expect(planner.plan(all, view).selectedClusters).toEqual(selectGeometryCut(hierarchy, view));
    }
    const roots = new Set(directory.rootPages);
    const fine = { ...base, maxErrorPixels: 0 };
    const initial = planner.plan(roots, fine);
    expect(initial.selectedClusters).toEqual([...hierarchy.roots].sort((a, b) => a - b));
    expect(initial.requestedPages.length).toBeGreaterThan(0);
    const unavailable = initial.requestedPages[0].pageId;
    const partial = new Set(all); partial.delete(unavailable);
    const result = planner.plan(partial, fine);
    expect(result.requestedPages).toContainEqual({ pageId: unavailable, priority: expect.any(Number) });
    expect(result.fallbackGroups.length).toBeGreaterThan(0);
    expect(result.selectedClusters.every(id => partial.has(directory.clusterPages[id]))).toBe(true);
    expect(result.selectedPages).toEqual([...new Set(result.selectedClusters.map(id => directory.clusterPages[id]))].sort((a, b) => a - b));
    expect(boundary(hierarchy, result.selectedClusters)).toEqual(boundary(hierarchy, hierarchy.roots));
    coverage(hierarchy, result.selectedClusters);
    expect(planner.plan(partial, fine)).toEqual(result);
    expect(() => planner.plan(new Set<number>(), fine)).toThrow('ROOT_PAGE_MISSING');
  });

  it('accepts affine scale and translation, while rejecting perspective and invalid views', async () => {
    const { directory, planner } = await fixture();
    const all = new Set(directory.pages.map(page => page.id));
    const view = { eye: [8, 8, 6] as [number, number, number], projectionScale: 800, near: 0.1,
      maxErrorPixels: 1, model: multiply(translation(10, 0, 0), scaling(2, 1, 1)) };
    const result = planner.plan(all, view);
    expect(result.selectedClusters.length).toBeGreaterThan(0);
    expect(result.selectedPages.every(id => all.has(id))).toBe(true);
    expect(() => planner.plan(all, { ...view, projectionScale: 0 })).toThrow('VIEW');
    const perspective = new Float32Array(view.model); perspective[3] = 1;
    expect(() => planner.plan(all, { ...view, model: perspective })).toThrow('MODEL');
    expect(() => planner.plan(all, { ...view, eye: [1e100, 0, 0] })).toThrow('VIEW');
    expect(() => planner.plan(all, { ...view, projectionScale: 1e100 })).toThrow('VIEW');
    const overflowing = identity(); overflowing[0] = Infinity;
    expect(() => planner.plan(all, { ...view, model: overflowing })).toThrow('MODEL');
  });

  it('uses the same conservative identity bounds with and without an explicit model at a split threshold', async () => {
    const { directory, planner } = await fixture();
    const all = new Set(directory.pages.map(page => page.id));
    const roots = new Set(directory.roots);
    const groupId = directory.groups.findIndex(group => !group.terminal && group.parents.every(id => roots.has(id)));
    expect(groupId).toBeGreaterThanOrEqual(0);
    const group = directory.groups[groupId];
    const center = group.bounds.min.map((min, axis) => (min + group.bounds.max[axis]) * 0.5);
    const extent = group.bounds.min.map((min, axis) => (group.bounds.max[axis] - min) * 0.5);
    const eye: [number, number, number] = [group.bounds.max[0] + 10, center[1], center[2]];
    const expandedX = extent[0] + (Math.abs(center[0]) + extent[0] + 1) * 0.000002;
    const withoutAllowance = group.error * 800 / 10;
    const withAllowance = group.error * 800 / (eye[0] - center[0] - expandedX);
    expect(withAllowance).toBeGreaterThan(withoutAllowance);
    const view = { eye, projectionScale: 800, near: 0.1, maxErrorPixels: (withAllowance + withoutAllowance) * 0.5 };
    const omitted = planner.plan(all, view);
    const explicit = planner.plan(all, { ...view, model: identity() });
    expect(omitted).toEqual(explicit);
    expect(omitted.refinedGroups).toContain(groupId);
  });
});
