import { expect, test } from '@playwright/test';
import type { runWorldQuerySceneFixture } from './fixtures/world-query-scene';

type Result = Awaited<ReturnType<typeof runWorldQuerySceneFixture>>;
type Stage = Result['initial'];
type Hit = NonNullable<Stage['accelerated']['hits'][number]>;

function hit(value: Hit | null): Hit {
  expect(value).not.toBeNull();
  if (!value) throw new Error('Expected an authored cross-batch hit');
  return value;
}
function near(actual: readonly number[], expected: readonly number[], tolerance = 0.0005): void {
  expect(actual).toHaveLength(expected.length);
  for (let axis = 0; axis < expected.length; axis++) {
    expect(Math.abs(actual[axis] - expected[axis])).toBeLessThan(tolerance);
  }
}
function parity(stage: Stage, instanceAcceleration: 'bvh' | 'linear'): void {
  expect(stage.accelerated.hits).toEqual(stage.linear.hits);
  expect(stage.accelerated.frame).toBe(stage.linear.frame);
  expect(stage.accelerated.materialRevision).toBe(stage.linear.materialRevision);
  expect(stage.acceleratedDiagnostics.sceneAcceleration).toBe('batch-bvh');
  expect(stage.linearDiagnostics.sceneAcceleration).toBe('linear-batches');
  expect(stage.acceleratedDiagnostics.sceneBvhNodes).toBe(511);
  expect(stage.acceleratedDiagnostics.sceneCandidateBytes).toBeGreaterThan(0);
  for (const diagnostics of [stage.acceleratedDiagnostics, stage.linearDiagnostics]) {
    expect(diagnostics.instanceAcceleration).toBe(instanceAcceleration === 'bvh' ? 'batch-bvh' : 'linear-root-scan');
    expect(diagnostics.acceleratedBatches).toBe(instanceAcceleration === 'bvh' ? 1 : 0);
    expect(diagnostics.instanceBvhNodes).toBe(instanceAcceleration === 'bvh' ? 15 : 0);
  }
  expect(stage.acceleratedDiagnostics.lastReadbackQuery?.overflows).toBe(0);
  expect(stage.linearDiagnostics.lastReadbackQuery?.overflows).toBe(0);
}

for (const instanceAcceleration of ['linear', 'bvh'] as const) test(`scene BVH preserves cross-batch queries and GI with ${instanceAcceleration} instance traversal`, async ({ page }, testInfo) => {
  test.setTimeout(120000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') console.log('WORLD_QUERY_SCENE_BROWSER_ERROR ' + message.text());
  });
  await page.goto('/tests/browser/fixtures/world-query-scene.html');
  const result = await page.evaluate(async (mode) => {
    const module = await import('/tests/browser/fixtures/' + 'world-query-scene.ts') as {
      runWorldQuerySceneFixture: typeof runWorldQuerySceneFixture;
    };
    return module.runWorldQuerySceneFixture(mode);
  }, instanceAcceleration);
  expect(result.instanceAcceleration).toBe(instanceAcceleration);
  for (const stage of [result.initial, result.steady, result.staticFrame, result.moved, result.empty,
    result.restored, result.materialChanged]) parity(stage, instanceAcceleration);
  expect(result.pictureEqual).toBe(true);
  for (const count of result.visiblePixels) expect(count).toBeGreaterThan(50);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  }

  const initial = result.initial.accelerated.hits;
  expect(initial).toHaveLength(11);
  expect(initial.slice(0, 4).map(value => hit(value).entityId)).toEqual([1000, 1005, 1011, 5000]);
  near(hit(initial[2]).position, result.initialDeformedPosition);
  expect(hit(initial[4]).batch).toBe('scene-batch-17'); // Transparent Masked center exposes farther geometry.
  expect(hit(initial[4]).distance).toBeCloseTo(5, 4);
  expect(hit(initial[5]).batch).toBe('scene-batch-42'); // Opaque Masked border remains nearest.
  expect(hit(initial[5]).distance).toBeCloseTo(4, 4);
  expect(hit(initial[6]).batch).toBe('scene-batch-21'); // Same distance: earlier registered batch wins.
  expect(initial[7]).toBeNull();
  expect(initial[8]).toBeNull(); // Exact ID and generation ignored.
  expect(hit(initial[9]).entityId).toBe(1000); // Stale generation does not ignore.
  expect(hit(initial[10]).entityId).toBe(1063);

  expect(result.steady.accelerated.hits).toEqual(initial);
  expect(result.steady.accelerated.frame).toBe(result.initial.accelerated.frame);
  expect(result.steady.acceleratedDiagnostics.sceneCandidateBytes)
    .toBe(result.initial.acceleratedDiagnostics.sceneCandidateBytes);
  expect(result.steady.acceleratedDiagnostics.lastReadbackQuery?.workUnits)
    .toBe(result.initial.acceleratedDiagnostics.lastReadbackQuery?.workUnits);
  expect(result.staticFrame.accelerated.hits).toEqual(initial);
  expect(result.staticFrame.accelerated.frame).toBe(result.initial.accelerated.frame + 1);
  expect(result.staticFrame.acceleratedDiagnostics.lastReadbackQuery?.workUnits)
    .toBe(result.initial.acceleratedDiagnostics.lastReadbackQuery?.workUnits);

  const moved = result.moved.accelerated.hits;
  expect(hit(moved[0]).entityId).toBe(1011);
  near(hit(moved[0]).position, result.movedPosition);
  expect(moved[1]).toBeNull(); // The old deformation and batch bounds no longer contain the ray.
  expect(hit(moved[2]).entityId).toBe(1000);
  expect(hit(moved[3]).batch).toBe('scene-batch-17');
  expect(hit(moved[4]).batch).toBe('scene-batch-21');
  expect(result.empty.accelerated.hits.slice(0, 2)).toEqual([null, null]);
  expect(hit(result.empty.accelerated.hits[2]).entityId).toBe(1000);

  const restored = result.restored.accelerated.hits;
  expect(hit(restored[0]).entityId).toBe(1011);
  expect(hit(restored[0]).entityGeneration).toBe(2);
  expect(hit(restored[0]).historyVersion).toBe(1);
  expect(hit(restored[1]).entityGeneration).toBe(2);
  expect(restored[2]).toBeNull();
  expect(hit(restored[3]).entityId).toBe(1000);

  const changed = result.materialChanged.accelerated;
  expect(changed.frame).toBe(result.beforeMutationFrame); // Both material edits are visible without rendering.
  expect(changed.materialRevision).toBe(result.restored.accelerated.materialRevision + 2);
  near(hit(changed.hits[0]).baseColor, [0.9, 0.1, 0.3]);
  expect(hit(changed.hits[1]).batch).toBe('scene-batch-42');
  expect(hit(changed.hits[1]).distance).toBeCloseTo(4, 4);
  near(hit(changed.hits[1]).baseColor, [0.8, 0.15, 0.1]);

  const acceleratedWork = result.initial.acceleratedDiagnostics.lastReadbackQuery!;
  const linearWork = result.initial.linearDiagnostics.lastReadbackQuery!;
  expect(acceleratedWork.rays).toBe(11);
  expect(linearWork.rays).toBe(11);
  expect(acceleratedWork.workUnits).toBeLessThan(linearWork.workUnits * 0.5);
  expect(acceleratedWork.triangleTests).toBeLessThanOrEqual(linearWork.triangleTests);
  expect(acceleratedWork.maxWorkUnits).toBeGreaterThan(0);

  expect(result.acceleratedGi.invalidProbes).toBe(0);
  expect(result.linearGi.invalidProbes).toBe(0);
  expect(result.acceleratedGi.probes).toHaveLength(8);
  expect(result.linearGi.probes).toHaveLength(8);
  let energy = 0;
  for (let index = 0; index < 8; index++) {
    const a = result.acceleratedGi.probes[index], b = result.linearGi.probes[index];
    expect(a.valid).toBe(true); expect(b.valid).toBe(true);
    expect(a.primaryOverflows + a.shadowOverflows).toBe(0);
    expect(b.primaryOverflows + b.shadowOverflows).toBe(0);
    near(a.coefficients.flat(), b.coefficients.flat(), 0.0001);
    near(a.maxRadiance, b.maxRadiance, 0.0001);
    energy += a.coefficients.flat().reduce((sum, component) => sum + Math.abs(component), 0);
  }
  expect(energy).toBeGreaterThan(0.001);

  expect(result.overflow).toContain('WORLD_QUERY_WORK_BUDGET');
  expect(result.overflowDiagnostics.sceneAcceleration).toBe('batch-bvh');
  expect(result.overflowDiagnostics.lastReadbackQuery?.overflows).toBeGreaterThan(0);
  expect(result.errors).toEqual({ accelerated: [], linear: [] });
  expect(result.overflowErrors).toEqual([]);
  expect(result.acceleratedAfterDispose).toBe(0);
  expect(result.linearAfterDispose).toBe(0);
  expect(result.overflowAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);

  const screenshot = testInfo.outputPath(`world-query-scene-${instanceAcceleration}.png`);
  await page.locator('#scene-pair').screenshot({ path: screenshot });
  await testInfo.attach('Cross-batch scene BVH parity', { path: screenshot, contentType: 'image/png' });
  console.log('WORLD_QUERY_SCENE_DIAGNOSTIC ' + JSON.stringify({
    adapter: result.adapter, visiblePixels: result.visiblePixels,
    acceleratedWork, linearWork, sceneBvhNodes: result.initial.acceleratedDiagnostics.sceneBvhNodes,
    sceneCandidateBytes: result.initial.acceleratedDiagnostics.sceneCandidateBytes,
    energy, overflow: result.overflow, errors: result.errors,
  }));
});
