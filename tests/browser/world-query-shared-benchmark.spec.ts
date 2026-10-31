import { expect, test } from '@playwright/test';
import type { runWorldQuerySharedBenchmark } from './fixtures/world-query-shared-benchmark';

// Timing needs a real adapter; SwiftShader proves functionality in world-query-shared.spec.ts.
test.skip(process.env.PLAYWRIGHT_GPU !== 'hardware', 'hardware-only paired timing');
// Unquantized timestamps (Chrome otherwise rounds them to 65.536 us) for this diagnostic only;
// the override applies to hardware runs so a software run keeps its SwiftShader arguments.
if (process.env.PLAYWRIGHT_GPU === 'hardware') {
  test.use({ launchOptions: { args: [...(process.env.PLAYWRIGHT_HIGH_PERFORMANCE_GPU === '1' ? ['--force-high-performance-gpu'] : []),
    '--enable-webgpu-developer-features'] } });
}

test('paired timing of shared static traversal and a dense distance-field experiment over many static batches (diagnostic)', async ({ page }) => {
  test.setTimeout(900_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const settings = {
    scenes: [{ name: 'rocks-405x2x32', batches: 405, triangles: 32, instances: 2 },
      { name: 'rocks-405x1x512', batches: 405, triangles: 512, instances: 1 }],
    rayCounts: [4096, 32768], rounds: Number(process.env.QUERY_BENCHMARK_ROUNDS ?? 12), warmupRounds: 2, repeats: 4,
    // 6.25 cm is expected to exceed the dense grid's binding budget; the refusal is part of the record.
    sdf: { voxels: [0.25, 0.125, 0.0625], bandVoxels: 4, thresholdVoxels: 0.75, maxSteps: 256, buildRepeats: 3, sweepThresholdVoxels: [0.25, 0.5, 1] },
  };
  const result = await page.evaluate(async value => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'world-query-shared-benchmark.ts') as {
      runWorldQuerySharedBenchmark: typeof runWorldQuerySharedBenchmark };
    return fixture.runWorldQuerySharedBenchmark(value);
  }, settings);
  // Logged before assertions: timings are diagnostic (contended unless stated otherwise).
  console.log('WORLD_QUERY_SHARED_BENCHMARK', JSON.stringify(result));
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(result.adapter.isFallbackAdapter).not.toBe(true);
  for (const scene of result.results) {
    for (const arm of scene.parity) {
      expect(arm.mismatches, `${scene.scene.name} ${arm.arm}`).toBe(0);
      expect(arm.overflows, `${scene.scene.name} ${arm.arm}`).toBe(0);
    }
    expect(scene.parity[0].hits).toBeGreaterThan(1000);
    const shared = scene.diagnostics.find(entry => entry.arm === 'B')!.sharedStatic!;
    expect(shared.pooledBatches).toBe(405);
    // Distance-field arms are measured, not gated on accuracy.
    expect(scene.sdf.map(arm => arm.arm)).toEqual(['SDF0.25', 'SDF0.125']);
    expect(scene.sdfSkipped.map(entry => entry.voxel)).toEqual([0.0625]);
    expect(scene.sdfSkipped[0].reason).toContain('UDF_GRID_BUDGET');
  }
  expect(result.errors).toEqual([]);
  expect(result.cleanup.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
