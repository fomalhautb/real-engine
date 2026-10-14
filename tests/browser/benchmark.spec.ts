import { expect, test } from '@playwright/test';

test('benchmark measures fixed frames, tags sparse GPU samples and disposes resources', async ({ page }) => {
  await page.goto('/benchmark.html');
  await page.waitForFunction(() => window.__REAL_BENCHMARK__);
  const result = await page.evaluate(async () => {
    const report = await window.__REAL_BENCHMARK__.run({ frames: 12, warmupFrames: 2, trees: 3, width: 320, height: 180 });
    const csv = window.__REAL_BENCHMARK__.csv();
    return { report, csv, cleanup: window.__REAL_BENCHMARK__.dispose() };
  });
  expect(result.report.summary.frameCount).toBe(12);
  expect(result.report.summary.frameIntervalMs!.p50).toBeGreaterThan(0);
  expect(result.report.summary.elapsedMs).toBeGreaterThan(0);
  expect(result.report.measuredDurationMs).toBeGreaterThan(result.report.summary.elapsedMs!);
  expect(result.report.summary.gpuSampleCount).toBeLessThanOrEqual(1);
  expect(result.report.resolutions).toEqual([{ frame: 0, width: 320, height: 180 }]);
  expect(result.report.summary.ownedGpuBytes!.growthBytes).toBe(0);
  expect(result.report.summary.workload!.physicsBodies.last).toBe(6);
  expect(result.csv.trim().split('\n')).toHaveLength(13);
  expect(result.report.verification.maxError).toBeLessThan(0.0002);
  expect(result.report.errors).toEqual([]);
  expect(result.cleanup.ownedGpuBytes).toBe(0);
});

test('duration benchmark measures real elapsed time and rejects conflicting limits', async ({ page }) => {
  await page.goto('/benchmark.html');
  await page.waitForFunction(() => window.__REAL_BENCHMARK__);
  const result = await page.evaluate(async () => {
    let rejected = '';
    try { await window.__REAL_BENCHMARK__.run({ frames: 1, durationSeconds: 1 }); }
    catch (error) { rejected = String(error); }
    const before = performance.now();
    const report = await window.__REAL_BENCHMARK__.run({ durationSeconds: 1, warmupFrames: 0, trees: 1, width: 160, height: 90 });
    const wallMs = performance.now() - before;
    const lines = window.__REAL_BENCHMARK__.csv().trim().split('\n');
    return { rejected, report, wallMs, lastCsv: lines.at(-1), cleanup: window.__REAL_BENCHMARK__.dispose() };
  });
  expect(result.rejected).toContain('choose frames or durationSeconds');
  expect(result.report.measuredDurationMs).toBeGreaterThanOrEqual(1000);
  expect(result.wallMs).toBeGreaterThanOrEqual(result.report.measuredDurationMs);
  expect(Number(result.lastCsv!.split(',')[1])).toBeCloseTo(result.report.measuredDurationMs, 3);
  expect(result.report.summary.frameCount).toBeGreaterThan(1);
  expect(result.report.errors).toEqual([]);
  expect(result.cleanup.ownedGpuBytes).toBe(0);
});
