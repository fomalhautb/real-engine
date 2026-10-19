import { expect, test } from '@playwright/test';
import type { runWorldQueryInstancesFixture } from './fixtures/world-query-instances';

type Result = Awaited<ReturnType<typeof runWorldQueryInstancesFixture>>;
type Stage = Result['initial'];
type Hit = NonNullable<Stage['accelerated']['hits'][number]>;

function hit(value: Hit | null): Hit {
  expect(value).not.toBeNull();
  if (!value) throw new Error('Expected an authored triangle hit');
  return value;
}
function near(actual: readonly number[], expected: readonly number[], tolerance = 0.0005): void {
  expect(actual).toHaveLength(expected.length);
  for (let axis = 0; axis < expected.length; axis++) {
    expect(Math.abs(actual[axis] - expected[axis])).toBeLessThan(tolerance);
  }
}
function parity(stage: Stage): void {
  expect(stage.accelerated.hits).toEqual(stage.linear.hits);
  expect(stage.accelerated.frame).toBe(stage.linear.frame);
  expect(stage.accelerated.materialRevision).toBe(stage.linear.materialRevision);
  expect(stage.acceleratedDiagnostics.instanceAcceleration).toBe('batch-bvh');
  expect(stage.linearDiagnostics.instanceAcceleration).toBe('linear-root-scan');
  expect(stage.acceleratedDiagnostics.acceleratedBatches).toBe(1);
  expect(stage.acceleratedDiagnostics.instanceBvhNodes).toBeGreaterThan(0);
  expect(stage.linearDiagnostics.acceleratedBatches).toBe(0);
  expect(stage.linearDiagnostics.instanceBvhNodes).toBe(0);
  expect(stage.acceleratedDiagnostics.lastReadbackQuery?.overflows).toBe(0);
  expect(stage.linearDiagnostics.lastReadbackQuery?.overflows).toBe(0);
}

test('batch instance BVH matches linear queries and GI while reducing measured GPU work', async ({ page }, testInfo) => {
  test.setTimeout(120000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') console.log('WORLD_QUERY_INSTANCES_BROWSER_ERROR ' + message.text());
  });
  await page.goto('/tests/browser/fixtures/world-query-instances.html');
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'world-query-instances.ts') as {
      runWorldQueryInstancesFixture: typeof runWorldQueryInstancesFixture;
    };
    return module.runWorldQueryInstancesFixture();
  });
  for (const stage of [result.initial, result.moved, result.reordered, result.empty, result.restored]) parity(stage);
  expect(result.pictureEqual).toBe(true);
  expect(result.overlapTraversalOrder).toBe(true);
  expect(result.tieWitness.hits.map(value => hit(value).entityId)).toEqual([result.overlapFirst, result.overlapSecond]);
  for (const value of result.tieWitness.hits) expect(hit(value).distance === 0).toBe(true);
  for (const count of [...result.initialVisiblePixels, ...result.finalVisiblePixels]) expect(count).toBeGreaterThan(50);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  }

  const initial = result.initial.accelerated.hits;
  expect(initial).toHaveLength(11);
  expect(initial.slice(0, 7).map(value => hit(value).entityId)).toEqual([1000, 1001, 1017, 1063, 1127, 1200, 1255]);
  near(hit(initial[0]).position, result.knownDeformedPosition);
  expect(hit(initial[0]).entityGeneration).toBe(1);
  expect(hit(initial[0]).topologyVersion).toBe(1);
  expect(initial[7]).toBeNull(); // far miss
  expect(initial[8]).toBeNull(); // exact entity + generation ignored
  expect(hit(initial[9]).entityId).toBe(1000); // wrong generation must not ignore
  expect(hit(initial[10]).entityId).toBe(result.overlapFirst); // equal distance: first slot wins

  const moved = result.moved.accelerated.hits;
  expect(hit(moved[0]).entityId).toBe(result.movedId);
  near(hit(moved[0]).position, result.movedPosition);
  expect(moved[1]).toBeNull();
  expect(hit(moved[2]).entityId).toBe(1000);
  expect(hit(moved[3]).entityId).toBe(result.overlapFirst);

  const reordered = result.reordered.accelerated.hits;
  expect(hit(reordered[0]).entityId).toBe(result.movedId);
  expect(hit(reordered[1]).entityId).toBe(result.overlapSecond); // reversed slot order controls equal-distance tie
  expect(hit(reordered[2]).entityId).toBe(1000);
  expect(result.empty.accelerated.hits).toEqual([null, null, null]);

  const restored = result.restored.accelerated.hits;
  expect(hit(restored[0]).entityId).toBe(result.movedId);
  expect(hit(restored[0]).entityGeneration).toBe(2);
  expect(hit(restored[0]).historyVersion).toBe(1);
  expect(hit(restored[1]).entityId).toBe(result.movedId); // stale generation is not ignored
  expect(restored[2]).toBeNull(); // current generation is ignored
  expect(hit(restored[3]).entityId).toBe(result.overlapFirst);

  const acceleratedWork = result.initial.acceleratedDiagnostics.lastReadbackQuery!;
  const linearWork = result.initial.linearDiagnostics.lastReadbackQuery!;
  expect(acceleratedWork.rays).toBe(11);
  expect(linearWork.rays).toBe(11);
  expect(acceleratedWork.workUnits).toBeLessThan(linearWork.workUnits * 0.5);
  expect(acceleratedWork.triangleTests).toBeLessThanOrEqual(linearWork.triangleTests);
  expect(acceleratedWork.maxWorkUnits).toBeGreaterThan(0);
  expect(acceleratedWork.maxWorkUnits).toBeLessThanOrEqual(acceleratedWork.workUnits);

  expect(result.acceleratedGi.invalidProbes).toBe(0);
  expect(result.linearGi.invalidProbes).toBe(0);
  expect(result.acceleratedGi.probes).toHaveLength(8);
  expect(result.linearGi.probes).toHaveLength(8);
  let energy = 0;
  for (let i = 0; i < result.acceleratedGi.probes.length; i++) {
    const a = result.acceleratedGi.probes[i], b = result.linearGi.probes[i];
    expect(a.valid).toBe(true); expect(b.valid).toBe(true);
    expect(a.primaryOverflows + a.shadowOverflows).toBe(0);
    expect(b.primaryOverflows + b.shadowOverflows).toBe(0);
    const aSh = a.coefficients.flat(), bSh = b.coefficients.flat();
    near(aSh, bSh, 0.0001);
    near(a.maxRadiance, b.maxRadiance, 0.0001);
    energy += aSh.reduce((sum, value) => sum + Math.abs(value), 0);
  }
  expect(energy).toBeGreaterThan(0.001);

  expect(result.overflow).toContain('WORLD_QUERY_WORK_BUDGET');
  expect(result.overflowDiagnostics.lastReadbackQuery?.overflows).toBeGreaterThan(0);
  expect(result.errors).toEqual({ accelerated: [], linear: [] });
  expect(result.overflowErrors).toEqual([]);
  expect(result.acceleratedAfterDispose).toBe(0);
  expect(result.linearAfterDispose).toBe(0);
  expect(result.overflowAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);

  const screenshot = testInfo.outputPath('world-query-instances.png');
  await page.locator('#scene-pair').screenshot({ path: screenshot });
  await testInfo.attach('Deformed multi-instance query parity', { path: screenshot, contentType: 'image/png' });
  console.log('WORLD_QUERY_INSTANCES_DIAGNOSTIC ' + JSON.stringify({
    adapter: result.adapter, initialVisiblePixels: result.initialVisiblePixels,
    finalVisiblePixels: result.finalVisiblePixels, overlapTraversalOrder: result.overlapTraversalOrder,
    acceleratedWork, linearWork, instanceBvhNodes: result.initial.acceleratedDiagnostics.instanceBvhNodes,
    energy, overflow: result.overflow, errors: result.errors,
  }));
});
