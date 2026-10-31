import { expect, test } from '@playwright/test';
import type { runWorldQueryWatertightFixture } from './fixtures/world-query-watertight';

test('rays exactly on shared edges and vertices hit on both static traversal paths', async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'world-query-watertight.ts') as {
      runWorldQueryWatertightFixture: typeof runWorldQueryWatertightFixture };
    return fixture.runWorldQueryWatertightFixture();
  });
  // Logged before assertions so a failing gate keeps its data.
  console.log('WORLD_QUERY_WATERTIGHT', JSON.stringify(result));
  expect(result.rays).toBeGreaterThan(5000);
  expect(result.kinds['exact-vertex']).toBeGreaterThan(400);
  expect(result.kinds['exact-edge']).toBeGreaterThan(2000);
  for (const arm of result.arms) {
    expect(arm.misses, arm.arm).toEqual({});
    expect(arm.hits, arm.arm).toBe(result.rays);
    expect(arm.wrongBatch, arm.arm).toBe(0);
    expect(arm.overflows, arm.arm).toBe(0);
  }
  expect(result.mismatches).toBe(0);
  expect(result.sharedStatic).toMatchObject({ pooledBatches: 8, members: 8 });
  expect(result.errors).toEqual([]);
  expect(result.cleanup.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
