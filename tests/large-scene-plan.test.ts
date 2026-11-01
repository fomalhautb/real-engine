import { describe, expect, it } from 'vitest';
import { DEFAULT_CAMERA_PROJECTION } from '../src/camera-projection';
import { GeometryStreamRuntimeBudget } from '../src/geometry-stream-runtime-budget';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { largeSceneProbeGrid, planLargeSceneGeometry, planLargeSceneLighting } from '../src/large-scene-plan';
import { translation } from '../src/math';

const limits = { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };
/** Two refinable branches over one terminal root group (same fixture as the runtime-budget tests). */
function branches(key: string): GeometryStreamDirectory {
  const left = { min: [-2, 0, 0] as [number, number, number], max: [-1, 1, 1] as [number, number, number] };
  const right = { min: [1, 0, 0] as [number, number, number], max: [2, 1, 1] as [number, number, number] };
  return { version: 1, primitiveKey: key, sourceVertexCount: 6, sourceTriangleCount: 2, vertexStride: 80,
    clusters: [0, 1, 2, 3, 4, 5].map(id => ({ level: Math.floor(id / 2), group: id < 4 ? id : 4,
      refinedGroup: id < 2 ? null : id - 2, vertexCount: 3, triangleCount: 1,
      bounds: id % 2 ? right : left, lodBounds: id % 2 ? right : left, error: id < 2 ? 0 : id < 4 ? 1 : 4 })),
    groups: [0, 1, 2, 3].map(id => ({ level: Math.floor(id / 2), children: [id], parents: [id + 2],
      bounds: id % 2 ? right : left, error: id < 2 ? 1 : 4, terminal: false })).concat([
      { level: 2, children: [4, 5], parents: [], bounds: { min: [-2, 0, 0], max: [2, 1, 1] }, error: 4, terminal: true },
    ]), roots: [4, 5], pages: [0, 1, 2, 3, 4, 5].map(id => ({ id, clusterIds: [id], encodedBytes: 41, decodedBytes: 272, sha256: 'a'.repeat(64) })),
    clusterPages: [0, 1, 2, 3, 4, 5], rootPages: [4, 5] };
}
const inputs = [{ directory: branches('near'), models: [translation(0, 0, -3)] }, { directory: branches('side'), models: [translation(14, 0, -6)] }];
const view = { eye: [0, 0.5, 0] as const, target: [0, 0.5, -1] as const, width: 1280, height: 720, projection: DEFAULT_CAMERA_PROJECTION, maxErrorPixels: 2 };
const sizing = new GeometryStreamRuntimeBudget(inputs.map(input => ({ directory: input.directory, capacity: 1 })),
  { maxResidentBytes: 1024 ** 3, maxPeakBytes: 1024 ** 3 }, limits);
const state = (index: number, slots: number) => sizing.state(index, slots).totalBytes;

describe('large-scene geometry plan', () => {
  it('sizes initial pools from the first view demand under one explicit budget', () => {
    const plan = planLargeSceneGeometry(inputs, view, { maxResidentBytes: 64 * 1024 ** 2 }, limits);
    // The near branch wants every page; the side branch (about 65° off-axis, outside the default 16:9 frustum) only its roots.
    expect(plan.report.rootBytes).toBe(state(0, 2) + state(1, 2));
    expect(plan.report.allPagesBytes).toBe(state(0, 6) + state(1, 6));
    expect(plan.report.initialDemandBytes).toBe(state(0, 6) + state(1, 2));
    expect(plan.report).toMatchObject({ streams: 2, initialBytes: plan.report.initialDemandBytes, budgetDeferredGroups: 0, demandedGroups: 4 });
    expect(plan.poolBytes).toEqual([sizing.state(0, 6).poolBytes, sizing.state(1, 2).poolBytes]);
    expect(plan.capacity).toEqual({ maxResidentBytes: 64 * 1024 ** 2, maxPeakBytes: plan.report.maxPeakBytes, intervalFrames: 2,
      maxConcurrentMigrations: 8, maxCopyBytes: 64 * 1024 ** 2, maxCopyPages: 4096 });
    expect(plan.report.maxPeakBytes).toBeGreaterThan(64 * 1024 ** 2 + state(0, 6));
  });

  it('reports groups the budget defers instead of hiding them, and rejects budgets below the roots', () => {
    const tight = planLargeSceneGeometry(inputs, view, { maxResidentBytes: state(0, 4) + state(1, 2) }, limits);
    expect(tight.report.initialBytes).toBe(state(0, 4) + state(1, 2));
    expect(tight.report.budgetDeferredGroups).toBe(2);
    expect(tight.report.initialDemandBytes).toBe(state(0, 6) + state(1, 2));
    expect(() => planLargeSceneGeometry(inputs, view, { maxResidentBytes: state(0, 2) + state(1, 2) - 1 }, limits))
      .toThrow('LARGE_SCENE_GEOMETRY_BUDGET');
    expect(() => planLargeSceneGeometry(inputs, view, { maxResidentBytes: 1024 ** 2, maxPeakBytes: 1024 }, limits)).toThrow('maxPeakBytes');
    expect(() => planLargeSceneGeometry(inputs, view, { maxResidentBytes: 1024 ** 2, maxCopyBytes: 16 }, limits)).toThrow('LARGE_SCENE_GEOMETRY_COPY_BUDGET');
    expect(() => planLargeSceneGeometry(inputs, { ...view, target: view.eye }, { maxResidentBytes: 1024 ** 2 }, limits)).toThrow('LARGE_SCENE_GEOMETRY_VIEW');
    expect(() => planLargeSceneGeometry([], view, { maxResidentBytes: 1024 ** 2 }, limits)).toThrow('LARGE_SCENE_GEOMETRY_INPUTS');
  });

  it('follows the projection: a wider field of view brings the side branch into demand', () => {
    const plan = planLargeSceneGeometry(inputs, { ...view, projection: { ...DEFAULT_CAMERA_PROJECTION, verticalFovRadians: (120 * Math.PI) / 180 } },
      { maxResidentBytes: 64 * 1024 ** 2 }, limits);
    expect(plan.report.demandedGroups).toBeGreaterThan(4);
    expect(plan.report.initialDemandBytes).toBeGreaterThan(state(0, 6) + state(1, 2));
  });
});

describe('large-scene lighting plan', () => {
  const bounds = { min: [-62.5, -7.6, -87.9] as const, max: [111.3, 43.5, 96.5] as const };
  it('sizes the probe grid from the bounds within the probe ceiling', () => {
    const grid = largeSceneProbeGrid(bounds);
    expect(grid.counts).toEqual([11, 4, 11]);
    expect(grid.counts.reduce((a, b) => a * b, 1)).toBeLessThanOrEqual(512);
    grid.counts.forEach((count, axis) => expect(grid.spacing[axis] * count).toBeCloseTo(bounds.max[axis] - bounds.min[axis], 9));
    expect(grid.origin[0]).toBeCloseTo(bounds.min[0] + grid.spacing[0] / 2, 9);
    expect(largeSceneProbeGrid({ min: [0, 0, 0], max: [1, 1, 1] }, 8).counts).toEqual([2, 2, 2]);
    expect(() => largeSceneProbeGrid(bounds, 513)).toThrow('LARGE_SCENE_GI_PROBES');
    expect(() => largeSceneProbeGrid({ min: [0, 0, 0], max: [1, 0, 1] })).toThrow('LARGE_SCENE_GI_BOUNDS');
  });

  it('returns explicit budgets and rejects requests the engine would coarsen', () => {
    const request = { gi: true, reflections: true, fog: true, airLighting: true } as const;
    const plan = planLargeSceneLighting(bounds, 1440, 1000, 259, 1591, request);
    expect(plan.report.probes).toMatchObject({ counts: [11, 4, 11], rays: 484 * 64 });
    expect(plan.options.diffuseGi?.update).toEqual({ mode: 'scheduled' });
    expect(plan.options.worldQueries).toEqual({ maxRays: 65536, maxBatches: 1591, maxTrianglesPerMesh: 262144, maxBytes: 512 * 1024 ** 2,
      maxTotalWorkUnits: 67_108_864 });
    expect(plan.report.reflections!.spacing).toBeGreaterThanOrEqual(4);
    expect(plan.report.reflections!.rays).toBeLessThanOrEqual(32768);
    expect(plan.options.reflections?.temporal).toEqual({});
    expect(plan.options.volumetricFog).toMatchObject({ pixelSize: plan.report.fog!.pixelSize, airLighting: { gi: true, outsideSkyVisibility: 0 } });
    expect(plan.report.fog!.froxels).toBeLessThanOrEqual(1_048_576);
    // The chosen values stay valid at the same size; a finer explicit request is rejected, not coarsened.
    expect(() => planLargeSceneLighting(bounds, 1440, 1000, 259, 1591, { ...request, reflectionSpacing: plan.report.reflections!.spacing,
      fogPixelSize: plan.report.fog!.pixelSize })).not.toThrow();
    expect(() => planLargeSceneLighting(bounds, 1440, 1000, 259, 1591, { ...request, reflectionSpacing: 2 })).toThrow('LARGE_SCENE_REFLECTION_BUDGET');
    expect(() => planLargeSceneLighting(bounds, 1440, 1000, 259, 1591, { ...request, fogPixelSize: 2 })).toThrow('LARGE_SCENE_FOG_BUDGET');
    expect(() => planLargeSceneLighting(bounds, 2880, 2000, 259, 1591, { ...request, fogPixelSize: plan.report.fog!.pixelSize })).toThrow('LARGE_SCENE_FOG_BUDGET');
    expect(() => planLargeSceneLighting(bounds, 1440, 1000, 259, 1591, { ...request, gi: false })).toThrow('air lighting requires gi and fog');
    const direct = planLargeSceneLighting(bounds, 1440, 1000, 259, 1591, { gi: false, reflections: false, fog: false, airLighting: false });
    expect(direct.options).toEqual({});
  });
});
