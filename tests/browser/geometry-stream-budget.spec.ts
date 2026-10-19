import { expect, test } from '@playwright/test';
import type { runGeometryStreamBudget } from './fixtures/geometry-stream-budget';

test('small streamed working sets retain whole groups and settle without repeated uploads across views', async ({ page }, testInfo) => {
  test.setTimeout(120000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-stream-budget.ts') as { runGeometryStreamBudget: typeof runGeometryStreamBudget };
    return fixture.runGeometryStreamBudget();
  });
  expect(result.slots).toBeLessThan(result.pages);
  expect(result.checks).toBeGreaterThan(25);
  expect(result.steadyTriangles).toBeGreaterThan(result.firstTriangles);
  expect(result.firstAllocation.pages).not.toEqual(result.viewBAllocation.pages);
  expect(result.firstAllocation.budgetDeferredGroups.length).toBeGreaterThan(0);
  expect(result.finalPool.evictions).toBeGreaterThan(0);
  expect(result.transitions).toHaveLength(5);
  expect(result.transitions[4].pages).toEqual(result.transitions[0].pages);
  expect(result.transitions[4].evictions).toBeGreaterThanOrEqual(4);
  expect(result.peakCpu).toBeLessThanOrEqual(result.cpuBytes);
  expect(result.afterDispose).toBe(0); expect(result.errors).toEqual([]); expect(errors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  const screenshot = testInfo.outputPath('stream-working-set.png'); await page.locator('#stream-budget-visual').screenshot({ path: screenshot });
  await testInfo.attach('Bounded group residency in main renderer', { path: screenshot, contentType: 'image/png' });
  console.log('STREAM_WORKING_SET_DIAGNOSTIC', JSON.stringify(result));
});
