import { expect, test } from '@playwright/test';
import type { runGeometryPagePoolFixture } from './fixtures/geometry-page-pool';

test('physical GPU page slots draw complete fallback cuts and retire only after submitted frames', async ({ page }, testInfo) => {
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-page-pool.ts') as { runGeometryPagePoolFixture: typeof runGeometryPagePoolFixture };
    return fixture.runGeometryPagePoolFixture();
  });
  expect(result.unpublished).toBe(true);
  expect(result.unpublishedFlags.every(flag => flag === 0)).toBe(true);
  expect(result.slotCount).toBeLessThan(result.pages);
  expect(result.maxUploadBytes).toBeLessThanOrEqual(8192);
  expect(result.maxUploadWrites).toBeLessThanOrEqual(8);
  expect(result.peak).toBeLessThanOrEqual(result.budget);
  expect(result.detailed.triangles).toBeGreaterThan(result.coarse.triangles);
  for (const capture of [result.coarse, result.detailed, result.recovered]) {
    expect(capture.maxError).toBe(0); expect(capture.differentBytes).toBe(0); expect(capture.foregroundPixels).toBeGreaterThan(20_000);
  }
  expect(result.pageByteMismatches).toBe(0);
  expect(result.heldBlocked).toMatchObject({ blocked: true, bytes: 0, evictedPages: [] });
  expect(result.submittedBlocked).toMatchObject({ blocked: true, bytes: 0, evictedPages: [] });
  expect(result.evicted).toHaveLength(1);
  expect(result.reused.residentPages).toContain(result.incoming);
  expect(result.reused.evictions).toBe(1);
  expect(result.publishedIds).toEqual(result.residentIds);
  expect(result.cleanup).toMatchObject({ disposed: true, allocatedGpuBytes: 0, stagingBytes: 0, framesInFlight: 0,
    pendingPages: [], residentPages: [], leasedPages: [], pinnedPages: [], error: null });
  expect(result.ownedAfterDispose).toBe(0); expect(result.errors).toEqual([]); expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  const path = testInfo.outputPath('geometry-page-pool-cuts.png');
  await page.locator('#geometry-page-pool-visual').screenshot({ path });
  await testInfo.attach('physical GPU page pool cuts', { path, contentType: 'image/png' });
  console.log('GEOMETRY_GPU_PAGE_POOL_DIAGNOSTIC', JSON.stringify(result));
});
