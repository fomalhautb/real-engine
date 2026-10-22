import { expect, test } from '@playwright/test';
import type { runGeometryStreamSelectionCases } from './fixtures/geometry-stream-selection-cases';

test('native GPU stream cuts agree with per-instance CPU plans across residency, transforms, LOD and culling', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-stream-selection-cases.ts') as {
      runGeometryStreamSelectionCases: typeof runGeometryStreamSelectionCases;
    };
    return fixture.runGeometryStreamSelectionCases();
  });
  expect(result.pages).toBeGreaterThan(1);
  expect(result.rootOnly.requests).toBeGreaterThan(0);
  expect(result.rootOnly.desiredGroups.length).toBeGreaterThan(0);
  expect(result.rootOnly.desiredGroups).toEqual(result.fine.desiredGroups);
  expect(result.swapped.desiredGroups).toEqual(result.fine.desiredGroups);
  expect(result.fine.requests).toBe(0);
  expect(result.middle.requests).toBe(0);
  expect(result.coarse.requests).toBe(0);
  expect(result.fine.clusters).toBeGreaterThan(result.middle.clusters);
  expect(result.middle.clusters).toBeGreaterThan(result.coarse.clusters);
  expect(result.frozen.clusters).toBe(result.middle.clusters);
  expect(result.frozen.indices).toBe(result.middle.indices);
  expect(result.frozen.desiredGroups).toEqual(result.middle.desiredGroups);
  expect(result.farCount).not.toBe(result.middle.clusters);
  expect(result.coarse.desiredGroups).toEqual([]);
  expect(result.culled.culledInstanceClusters).toBeGreaterThan(0);
  expect(result.culled.visibleClusters).toBeLessThan(result.culled.clusters);
  expect(result.zero).toMatchObject({ count: 0, clusters: 0, indices: 0, requests: 0, visibleClusters: 0 });
  expect(result.zero.desiredGroups).toEqual([]);
  expect(result.allOffscreen.visibleClusters).toBe(0);
  expect(result.allOffscreen.indices).toBe(0);
  expect(result.allOffscreen.desiredGroups).toEqual([]);
  expect(result.stableSmall).toHaveLength(12);
  expect(result.stableLarge).toHaveLength(12);
  expect(result.stableSmall[0].pairs).toBeLessThanOrEqual(512);
  expect(result.stableLarge[0].pairs).toBeGreaterThan(512);
  expect(new Set(result.stableSmall.map(entry => entry.orderedHash)).size).toBe(1);
  expect(new Set(result.stableLarge.map(entry => entry.orderedHash)).size).toBe(1);
  expect(result.stableSmall[0].orderedHash).not.toBeNull();
  expect(result.stableLarge[0].orderedHash).not.toBeNull();
  expect(result.boundaryBelow.pairs).toBeLessThanOrEqual(512);
  expect(result.boundaryZero.indices).toBe(0);
  expect(result.boundaryRegrow.orderedHash).toBe(result.stableLarge[0].orderedHash);
  expect(result.recoveryCases).toHaveLength(8);
  for (const entry of result.recoveryCases) expect(new Set(entry.hashes).size).toBe(1);
  expect(result.overflowCases).toHaveLength(4);
  expect(result.overflowCases.every(entry => entry.overflow && entry.indices === 0)).toBe(true);
  expect(result.singlePair).toMatchObject({ clusters: 1, capacity: 2, trianglesPerInstance: 12,
    restoredIndices: 36, poolOwned: 0, selectionOwned: 0 });
  expect(result.singlePair.rows).toHaveLength(25);
  for (const entry of result.singlePair.rows) {
    const visible = entry.mode !== 'reject' && entry.mode !== 'frustum';
    expect(entry.indices).toBe(visible ? entry.count * 36 : 0);
    expect(entry.firstIndices).toBe(visible && entry.mode !== 'recover' ? entry.count * 36 : 0);
    expect(entry.recoveredIndices).toBe(entry.mode === 'recover' ? entry.count * 36 : 0);
  }
  expect(result.singlePair.faults).toHaveLength(10);
  expect(result.singlePair.faults.every(entry => entry.overflow && entry.indices === 0)).toBe(true);
  expect(result.poolOwned).toBe(0);
  expect(result.selectionOwned).toBe(0);
  expect(result.pool).toMatchObject({ disposed: true, allocatedGpuBytes: 0, framesInFlight: 0 });
  expect(result.selection).toMatchObject({ disposed: true, gpuBytes: 0, feedbackPending: false });
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  console.log('GPU_STREAM_SELECTION_CASES', JSON.stringify(result));
});
