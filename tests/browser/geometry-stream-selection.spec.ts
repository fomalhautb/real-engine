import { expect, test } from '@playwright/test';
import type { runGeometryPagePoolFixture } from './fixtures/geometry-page-pool';

test('GPU selects complete resident cuts, requests missing pages and draws physical indices indirectly', async ({ page }, testInfo) => {
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-page-pool.ts') as { runGeometryPagePoolFixture: typeof runGeometryPagePoolFixture };
    return fixture.runGeometryPagePoolFixture(true);
  });
  for (const capture of [result.coarse, result.detailed, result.recovered]) {
    expect(capture.maxError).toBe(0); expect(capture.differentBytes).toBe(0);
    expect(capture.gpuPlan!.overflow).toBe(false);
    expect(capture.gpuPlan!.selectedClusters).toHaveLength(capture.clusters);
    expect(capture.gpuPlan!.selectedClusters!.every(cluster => cluster.visible)).toBe(true);
    expect(capture.gpuPlan!.indices).toBe(capture.triangles * 3);
    expect(capture.foregroundPixels).toBeGreaterThan(20000);
  }
  expect(result.detailed.triangles).toBeGreaterThan(result.coarse.triangles);
  expect(result.recovered.triangles).toBeLessThan(result.detailed.triangles);
  expect(result.ownedAfterDispose).toBe(0); expect(result.selectionOwned).toBe(0);
  expect(result.selection).toMatchObject({ disposed: true, gpuBytes: 0, feedbackPending: false });
  expect(result.errors).toEqual([]); expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  const screenshot = testInfo.outputPath('gpu-stream-selection-cuts.png');
  await page.locator('#geometry-page-pool-visual').screenshot({ path: screenshot });
  await testInfo.attach('GPU resident cut selection', { path: screenshot, contentType: 'image/png' });
  console.log('GPU_STREAM_SELECTION_DIAGNOSTIC', JSON.stringify(result));
});
