import { describe, expect, it } from 'vitest';
import type { MeshData } from '../src/geometry';
import { buildGeometryHierarchy } from '../src/geometry-hierarchy';
import { decodeGeometryPage } from '../src/geometry-pages';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import { GeometryStreamCut } from '../src/geometry-stream-cut';
import { geometryCutCoverage, geometryGroupBoundaryAudit, type GeometryClusterPayload } from '../src/geometry-stream-coverage';
import { oversubscribedGeometryBudget } from '../src/geometry-stream-oversubscription';
import { streamedGalleryMesh } from '../src/geometry-demo-streamed-scene';

const LIMITS = { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };

/** Open height-field grid of n x n quads: its LOD0 border has exactly 4n edges. */
function openGrid(n: number): MeshData {
  const side = n + 1, vertexCount = side * side, vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
  const height = (x: number, z: number) => 0.15 * Math.sin(x * 2.1) * Math.cos(z * 1.7) + 0.05 * Math.sin(x * z * 3.3);
  for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) {
    const x = i / n * 4 - 2, z = j / n * 4 - 2, offset = (i * side + j) * 80;
    view.setFloat32(offset, x, true); view.setFloat32(offset + 4, height(x, z), true); view.setFloat32(offset + 8, z, true);
    view.setFloat32(offset + 12, 1, true); view.setFloat32(offset + 20, 1, true); view.setFloat32(offset + 48, 1, true);
  }
  const indices = new Uint32Array(n * n * 6);
  let cursor = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const a = i * side + j, b = (i + 1) * side + j, c = (i + 1) * side + j + 1, d = i * side + j + 1;
    indices.set([a, d, b, b, d, c], cursor); cursor += 6;
  }
  return { vertices, indices, vertexCount, bounds: { min: [-2, -0.25, -2], max: [2, 0.25, 2] } };
}

async function build(mesh: MeshData, key: string) {
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: key, mesh, coverage: 'opaque' },
    { maxClusterTriangles: 96, groupSize: 4, simplifyRatio: 0.5, maxLevels: 12 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 16384 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const payloads = new Map<number, GeometryClusterPayload>();
  for (const page of encoded.pages) for (const cluster of (await decodeGeometryPage(page.bytes)).clusters) {
    payloads.set(cluster.id, { rest: cluster.rest, localTriangles: cluster.localTriangles });
  }
  return { directory, payloads };
}

/** Test-local net boundary of an arbitrary cluster set, keyed and signed by
 * position strings so different cluster sets compare directly. */
function netBoundary(payloads: ReadonlyMap<number, GeometryClusterPayload>, ids: Iterable<number>) {
  const edges = new Map<string, number>();
  for (const id of ids) {
    const { rest, localTriangles } = payloads.get(id)!, view = new DataView(rest.buffer, rest.byteOffset, rest.byteLength);
    const position = (local: number) => [0, 4, 8].map(offset => view.getFloat32(local * 80 + offset, true)).join(',');
    for (let t = 0; t < localTriangles.length; t += 3) for (let e = 0; e < 3; e++) {
      const a = position(localTriangles[t + e]), b = position(localTriangles[t + (e + 1) % 3]);
      if (a === b) continue;
      const key = a < b ? `${a}|${b}` : `${b}|${a}`, count = (edges.get(key) ?? 0) + (a < b ? 1 : -1);
      if (count) edges.set(key, count); else edges.delete(key);
    }
  }
  return edges;
}

describe('geometryGroupBoundaryAudit', () => {
  it('finds every group of a closed and an open source boundary-preserving', async () => {
    const closed = await build(streamedGalleryMesh(11, 64, 16), 'boundary/closed');
    const audit = geometryGroupBoundaryAudit(closed.directory, closed.payloads);
    expect(audit.checkedGroups).toBeGreaterThan(4);
    expect(audit).toMatchObject({ mismatchedGroups: 0, lod0BoundaryEdges: 0, clusters: closed.directory.clusters.length });
    const open = await build(openGrid(40), 'boundary/open');
    const openAudit = geometryGroupBoundaryAudit(open.directory, id => open.payloads.get(id));
    expect(openAudit.checkedGroups).toBeGreaterThan(4);
    expect(openAudit).toMatchObject({ mismatchedGroups: 0, lod0BoundaryEdges: 160 });
  });

  it('gives every coverage-exact cut the net boundary of LOD0', async () => {
    const open = await build(openGrid(40), 'boundary/open-cuts'), planner = new GeometryStreamCut(open.directory);
    const lod0 = open.directory.clusters.flatMap((cluster, id) => cluster.level === 0 ? [id] : []);
    const reference = netBoundary(open.payloads, lod0);
    expect(reference.size).toBe(160);
    const all = new Set(open.directory.pages.map(page => page.id));
    let coarse = 0;
    for (const maxErrorPixels of [0.5, 4, 64, 1e9]) for (const eye of [[0, 1, 5], [6, 2, 0], [0, 40, 0]] as [number, number, number][]) {
      const plan = planner.plan(all, { eye, near: 0.1, projectionScale: 800, maxErrorPixels });
      expect(geometryCutCoverage(open.directory, plan.selectedClusters)).toMatchObject({ uncovered: 0, overlapping: 0, partialGroups: 0 });
      coarse += plan.selectedClusters.filter(id => open.directory.clusters[id].level > 0).length;
      const net = netBoundary(open.payloads, plan.selectedClusters);
      expect(net.size).toBe(reference.size);
      for (const [key, count] of reference) expect(net.get(key)).toBe(count);
    }
    expect(coarse).toBeGreaterThan(0);
  });

  it('reports a group whose parents lose a triangle and rejects malformed payloads', async () => {
    const closed = await build(streamedGalleryMesh(5, 64, 16), 'boundary/corrupt');
    const group = closed.directory.groups.findIndex(entry => !entry.terminal);
    const victim = closed.directory.groups[group].parents[0], original = closed.payloads.get(victim)!;
    const corners = original.localTriangles.slice();
    corners[1] = corners[0]; corners[2] = corners[0];
    const corrupted = new Map(closed.payloads);
    corrupted.set(victim, { rest: original.rest, localTriangles: corners });
    const audit = geometryGroupBoundaryAudit(closed.directory, corrupted);
    expect(audit.mismatchedGroups).toBeGreaterThanOrEqual(1);
    expect(audit.mismatchExamples).toContain(group);
    const truncated = new Map(closed.payloads);
    truncated.set(victim, { rest: original.rest.subarray(80), localTriangles: original.localTriangles });
    expect(() => geometryGroupBoundaryAudit(closed.directory, truncated)).toThrow(/GEOMETRY_GROUP_BOUNDARY_PAYLOAD/);
    const missing = new Map(closed.payloads);
    missing.delete(victim);
    expect(() => geometryGroupBoundaryAudit(closed.directory, missing)).toThrow(/GEOMETRY_GROUP_BOUNDARY_PAYLOAD/);
  });
});

describe('oversubscribedGeometryBudget', () => {
  it('sizes the live budget from the unique-to-pool ratio and keeps roots inside it', async () => {
    const streams = await Promise.all([1, 2, 3].map(seed => build(streamedGalleryMesh(seed, 96, 24), 'budget/' + seed)));
    const inputs = streams.map((stream, index) => ({ directory: stream.directory, capacity: index + 1 }));
    const two = oversubscribedGeometryBudget(inputs, 2, LIMITS, 2), three = oversubscribedGeometryBudget(inputs, 3, LIMITS, 2);
    expect(two.uniqueToPool).toBeGreaterThanOrEqual(2);
    expect(two.uniqueToPool).toBeLessThan(2.001);
    expect(three.uniqueToPool).toBeGreaterThanOrEqual(3);
    expect(two.instances).toBe(6);
    for (const budget of [two, three]) {
      expect(budget.rootBytes).toBeLessThan(budget.maxResidentBytes);
      expect(budget.maxResidentBytes).toBeLessThan(budget.fullResidencyBytes);
      expect(budget.maxPeakBytes).toBeGreaterThan(budget.maxResidentBytes);
      expect(budget.storageShare).toBeGreaterThan(0);
      expect(budget.storageShare).toBeLessThanOrEqual(1);
    }
    expect(three.maxResidentBytes).toBeLessThan(two.maxResidentBytes);
    expect(() => oversubscribedGeometryBudget(inputs, 0.5, LIMITS, 2)).toThrow(/RATIO/);
    expect(() => oversubscribedGeometryBudget(inputs, 2, LIMITS, 0)).toThrow(/MIGRATIONS/);
    expect(() => oversubscribedGeometryBudget(inputs, 64, LIMITS, 2)).toThrow(/ROOTS_EXCEED_BUDGET/);
  });
});
