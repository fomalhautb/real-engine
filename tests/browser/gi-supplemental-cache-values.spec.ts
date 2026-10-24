import { expect, test } from '@playwright/test';
import type { runGiSupplementalCacheValuesFixture } from './fixtures/gi-supplemental-cache-values';

test('supplemental cache combines only matching fresh samples and expires carried values', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-supplemental-cache-values.ts') as {
      runGiSupplementalCacheValuesFixture: typeof runGiSupplementalCacheValuesFixture;
    };
    return fixture.runGiSupplementalCacheValuesFixture();
  });
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  }
  expect(result.ordinary).toHaveLength(8);
  expect(result.reset).toHaveLength(8);
  for (const [lane, value] of result.ordinary.entries()) {
    expect(value.identity, `identity ${lane}`).toEqual([100 + lane, 100 + lane, 1, lane + 1]);
    expect(value.versions, `versions ${lane}`).toEqual([10, 11, 1, 8]);
    expect(value.state[1], `frame ${lane}`).toBe(10);
  }
  const merged = result.ordinary[0];
  expect(merged.state[0]).toBe(1);
  for (const channel of merged.irradiance.slice(0, 3)) expect(channel).toBeCloseTo(2.8, 5);
  expect(merged.irradiance[3]).toBe(5);
  expect(merged.moments[0]).toBeCloseTo(4.16, 5);
  expect(merged.moments[1]).toBe(0);

  const carried = result.ordinary[1];
  expect(carried.state[0]).toBe(1);
  expect(carried.irradiance).toEqual([2, 2, 2, 4]);
  expect(carried.moments.slice(0, 2)).toEqual([2, 4]);
  for (const lane of [2, 3, 4, 5]) {
    expect(result.ordinary[lane].state[0], `expired or mismatched lane ${lane}`).toBe(0);
    expect(result.ordinary[lane].irradiance.slice(0, 3)).toEqual([0, 0, 0]);
    expect(result.ordinary[lane].moments[1]).toBe(0);
  }
  const failed = result.ordinary[6];
  expect(failed.state[0]).toBe(0);
  expect(failed.irradiance.slice(0, 3)).toEqual([0, 0, 0]);
  expect(failed.irradiance[3]).toBe(1);
  const noHistory = result.ordinary[7];
  expect(noHistory.state[0]).toBe(1);
  expect(noHistory.irradiance).toEqual([6, 6, 6, 1]);
  expect(noHistory.moments.slice(0, 2)).toEqual([0, 0]);

  expect(result.reset[1].state[0]).toBe(0);
  expect(result.reset[1].irradiance.slice(0, 3)).toEqual([0, 0, 0]);
  expect(result.reset[0].state[0]).toBe(1);
  expect(result.reset[0].irradiance).toEqual([6, 6, 6, 1]);
  expect(result.reset[0].moments.slice(0, 2)).toEqual([0, 0]);
  expect(result.errors).toEqual([]);
  expect(result.validationError).toBe('');
  expect(result.ownedBytes).toBe(0);
  expect(pageErrors).toEqual([]);
});
