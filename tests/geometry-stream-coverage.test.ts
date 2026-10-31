import { describe, expect, it } from 'vitest';
import { buildGeometryHierarchy } from '../src/geometry-hierarchy';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import { GeometryStreamCut } from '../src/geometry-stream-cut';
import { geometryCutCoverage, geometryCutWatertightness } from '../src/geometry-stream-coverage';
import { streamedGalleryMesh } from '../src/geometry-demo-streamed-scene';

async function build(groupSize = 4) {
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'coverage/gallery', mesh: streamedGalleryMesh(7, 64, 16), coverage: 'opaque' },
    { maxClusterTriangles: 96, groupSize, simplifyRatio: 0.5, maxLevels: 12 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 16384 });
  return { hierarchy, directory: await buildGeometryStreamDirectory(hierarchy, encoded) };
}
async function fixture() { return (await build()).directory; }

describe('geometryCutCoverage', () => {
  it('accepts every CPU reference cut across thresholds, views and partial residency', async () => {
    const directory = await fixture(), planner = new GeometryStreamCut(directory);
    expect(directory.pages.length).toBeGreaterThan(directory.rootPages.length + 2);
    const all = new Set(directory.pages.map(page => page.id)), roots = new Set(directory.rootPages);
    let seed = 1;
    const random = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296;
    const partial = Array.from({ length: 6 }, () => new Set([...roots, ...[...all].filter(() => random() < 0.5)]));
    let fallbacks = 0;
    for (const resident of [all, roots, ...partial]) for (const maxErrorPixels of [0, 0.5, 2, 16, 1e9]) {
      for (const eye of [[0, 0.4, 6], [8, 3, -2], [0, 0, 60]] as [number, number, number][]) {
        const plan = planner.plan(resident, { eye, near: 0.1, projectionScale: 800, maxErrorPixels });
        fallbacks += plan.fallbackGroups.length;
        const coverage = geometryCutCoverage(directory, plan.selectedClusters);
        expect(coverage).toMatchObject({ uncovered: 0, overlapping: 0, partialGroups: 0, selectedClusters: plan.selectedClusters.length });
        expect(coverage.lod0Clusters).toBe(directory.clusters.filter(cluster => cluster.level === 0).length);
      }
    }
    // Partial residency must actually exercise ancestor fallback.
    expect(fallbacks).toBeGreaterThan(0);
  });

  it('reports holes, overlaps and split group replacements exactly', async () => {
    const directory = await fixture(), planner = new GeometryStreamCut(directory);
    const all = new Set(directory.pages.map(page => page.id));
    const plan = planner.plan(all, { eye: [0, 0.4, 3], near: 0.1, projectionScale: 800, maxErrorPixels: 4 });
    expect(plan.refinedGroups.length).toBeGreaterThan(0);
    const selected = new Set(plan.selectedClusters);
    // Hole: drop one drawn LOD0 cluster, if any, else any drawn cluster.
    const victim = plan.selectedClusters.find(id => directory.clusters[id].level === 0) ?? plan.selectedClusters[0];
    expect(geometryCutCoverage(directory, [...selected].filter(id => id !== victim)).uncovered).toBeGreaterThan(0);
    // Overlap: also draw the parents of a group already refined into its children.
    const refined = plan.refinedGroups.find(groupId => directory.groups[groupId].children.every(id => selected.has(id)))!;
    expect(refined).toBeDefined();
    const overlap = geometryCutCoverage(directory, [...selected, ...directory.groups[refined].parents.filter(id => !selected.has(id))]);
    expect(overlap.overlapping).toBeGreaterThan(0);
    expect(overlap.uncovered).toBe(0);
    // Split replacement: replace a refined group's children with only some of its parents.
    const split = plan.refinedGroups.find(groupId => directory.groups[groupId].parents.length > 1
      && directory.groups[groupId].children.every(id => selected.has(id)));
    if (split !== undefined) {
      const group = directory.groups[split];
      const cut = [...selected].filter(id => !group.children.includes(id)).concat(group.parents.slice(0, 1));
      const result = geometryCutCoverage(directory, cut);
      expect(result.partialGroups).toBeGreaterThan(0);
      expect(result.uncovered).toBeGreaterThan(0);
    }
  });

  it('reports the projected error refinement compares, for every drawn cluster of a cut', async () => {
    const directory = await fixture(), planner = new GeometryStreamCut(directory);
    const all = new Set(directory.pages.map(page => page.id)), roots = new Set(directory.rootPages);
    const view = { eye: [0, 0.4, 3] as [number, number, number], near: 0.1, projectionScale: 800, maxErrorPixels: 4 };
    const full = planner.plan(all, view);
    for (const groupId of full.refinedGroups) expect(planner.groupErrorPixels(groupId, view)).toBeGreaterThan(view.maxErrorPixels);
    // Fully resident: every drawn coarse cluster displays at most the threshold.
    for (const id of full.selectedClusters) {
      const refined = directory.clusters[id].refinedGroup;
      if (refined !== null) expect(planner.groupErrorPixels(refined, view)).toBeLessThanOrEqual(view.maxErrorPixels);
    }
    // Roots only: fallback clusters display more than the threshold.
    const fallback = planner.plan(roots, view);
    expect(fallback.fallbackGroups.length).toBeGreaterThan(0);
    expect(Math.max(...fallback.selectedClusters.map(id => directory.clusters[id].refinedGroup)
      .filter((group): group is number => group !== null).map(group => planner.groupErrorPixels(group, view)))).toBeGreaterThan(view.maxErrorPixels);
    expect(() => planner.groupErrorPixels(directory.groups.length, view)).toThrow('GEOMETRY_STREAM_GROUP');
    expect(() => planner.groupErrorPixels(0.5, view)).toThrow('GEOMETRY_STREAM_GROUP');
  });

  it('rejects invalid and duplicate cluster IDs', async () => {
    const directory = await fixture();
    expect(() => geometryCutCoverage(directory, [directory.clusters.length])).toThrow('GEOMETRY_CUT_COVERAGE_CLUSTER');
    expect(() => geometryCutCoverage(directory, [-1])).toThrow('GEOMETRY_CUT_COVERAGE_CLUSTER');
    expect(() => geometryCutCoverage(directory, [0.5])).toThrow('GEOMETRY_CUT_COVERAGE_CLUSTER');
    expect(() => geometryCutCoverage(directory, [directory.roots[0], directory.roots[0]])).toThrow('GEOMETRY_CUT_COVERAGE_DUPLICATE');
  });

  // Simplification may pinch a coarse level into a non-manifold edge (four
  // triangles on one edge, observed in the 56-triangle root of this torus); that
  // closes the surface but repeats directed edges. Cracks are boundary edges.
  it('proves every reference cut of a closed surface is crack-free and detects removed or duplicated surface', async () => {
    for (const groupSize of [4, 8]) {
      const { hierarchy, directory } = await build(groupSize), planner = new GeometryStreamCut(directory);
      const all = new Set(directory.pages.map(page => page.id)), roots = new Set(directory.rootPages);
      const half = new Set([...roots, ...directory.pages.filter((_, index) => index % 2 === 0).map(page => page.id)]);
      for (const resident of [all, roots, half]) for (const maxErrorPixels of [0, 1, 8, 1e9]) {
        const plan = planner.plan(resident, { eye: [0.5, 0.6, 4], near: 0.1, projectionScale: 900, maxErrorPixels });
        const result = geometryCutWatertightness(hierarchy, plan.selectedClusters);
        expect(result.boundaryEdges).toBe(0);
        if (resident === all && maxErrorPixels === 0) expect(result.repeatedEdges).toBe(0);
      }
      const leaves = hierarchy.clusters.map((cluster, id) => ({ cluster, id })).filter(entry => entry.cluster.level === 0).map(entry => entry.id);
      expect(geometryCutWatertightness(hierarchy, leaves)).toEqual({ triangles: 64 * 16 * 2, boundaryEdges: 0, repeatedEdges: 0 });
      expect(geometryCutWatertightness(hierarchy, leaves.slice(1)).boundaryEdges).toBeGreaterThan(0);
      expect(geometryCutWatertightness(hierarchy, [...leaves, leaves[0]]).repeatedEdges).toBeGreaterThan(0);
    }
  });
});
