import { expect, test, type Page } from '@playwright/test';
import { GI_DISTANCE_CELLS } from '../../src/gi-visibility';
import type { runGiUniformIrradianceFixture, runGiVisibilityLeakFixture,
  runGiVisibilityMomentsFixture } from './fixtures/gi-visibility';

type MomentsResult = Awaited<ReturnType<typeof runGiVisibilityMomentsFixture>>;
type LeakResult = Awaited<ReturnType<typeof runGiVisibilityLeakFixture>>;
type UniformResult = Awaited<ReturnType<typeof runGiUniformIrradianceFixture>>;

async function fixturePage(page: Page): Promise<string[]> {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return pageErrors;
}

test('closed box GPU distance moments match analytic ray exits and production four-tap seam sampling', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-visibility.ts') as {
      runGiVisibilityMomentsFixture: typeof runGiVisibilityMomentsFixture;
    };
    return fixture.runGiVisibilityMomentsFixture();
  }) as MomentsResult;
  expect(result.actual.probes).toHaveLength(8);
  expect(result.actual.invalidProbes).toBe(0);
  for (let probe = 0; probe < 8; probe++) {
    const measured = result.actual.probes[probe];
    expect(measured.valid, `probe ${probe}`).toBe(true);
    expect(measured.primaryOverflows, `probe ${probe} primary`).toBe(0);
    expect(measured.shadowOverflows, `probe ${probe} shadow`).toBe(0);
    expect(measured.moments).toHaveLength(GI_DISTANCE_CELLS);
    for (let cell = 0; cell < GI_DISTANCE_CELLS; cell++) {
      const actual = measured.moments[cell], expected = result.expected[probe][cell];
      expect(Math.abs(actual[0] - expected[0]), `probe ${probe} cell ${cell} mean`).toBeLessThanOrEqual(0.01);
      expect(Math.abs(actual[1] - expected[1]), `probe ${probe} cell ${cell} mean squared`).toBeLessThanOrEqual(0.03);
      expect(Math.abs(actual[2] - expected[2]), `probe ${probe} cell ${cell} weight`).toBeLessThanOrEqual(2e-4);
    }
  }
  console.log('GI_VISIBILITY_FOUR_TAP_GPU_DIAGNOSTIC ' + JSON.stringify(result.helper.samples));
  for (const [index, sample] of result.helper.samples.entries()) {
    expect(Math.abs(sample.actual[0] - sample.expected[0]), `sample ${index} mean`).toBeLessThanOrEqual(0.005);
    expect(Math.abs(sample.actual[1] - sample.expected[1]), `sample ${index} mean squared`).toBeLessThanOrEqual(0.015);
  }
  for (const [left, right] of result.helper.seamPairs) {
    const a = result.helper.samples[left].actual, b = result.helper.samples[right].actual;
    expect(Math.abs(a[0] - b[0]), `seam pair ${left}/${right} mean`).toBeLessThanOrEqual(0.01);
    expect(Math.abs(a[1] - b[1]), `seam pair ${left}/${right} mean squared`).toBeLessThanOrEqual(0.03);
  }
  expect(result.helper.uncaptured).toEqual([]);
  expect(result.helper.validationError).toBe('');
  expect(result.helper.ownedBytes).toBe(0);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('solid partition suppresses actual dark-side HDR from a bright ceiling', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-visibility.ts') as {
      runGiVisibilityLeakFixture: typeof runGiVisibilityLeakFixture;
    };
    return fixture.runGiVisibilityLeakFixture();
  }) as LeakResult;
  const noWall = (result.noWallHdr[0] + result.noWallHdr[1] + result.noWallHdr[2]) / 3;
  const withWall = (result.withWallHdr[0] + result.withWallHdr[1] + result.withWallHdr[2]) / 3;
  console.log('GI_VISIBILITY_LEAK_GPU_DIAGNOSTIC ' + JSON.stringify({
    noWallHdr: result.noWallHdr, withWallHdr: result.withWallHdr,
    floorRay: result.floorRay, wallRay: result.wallRay,
    noWallInvalid: result.noWallProbe.invalidProbes,
    withWallInvalid: result.withWallProbe.invalidProbes, errors: result.errors,
  }));
  expect(result.floorRay?.batch).toBe('receiver-floor');
  expect(result.wallRay?.batch).toBe('solid-partition');
  expect(result.noWallProbe.invalidProbes).toBe(0);
  expect(result.withWallProbe.invalidProbes).toBe(0);
  for (const set of [result.noWallProbe, result.withWallProbe]) {
    expect(set.probes).toHaveLength(8);
    expect(set.probes.every(probe => probe.valid && probe.primaryOverflows === 0 && probe.shadowOverflows === 0)).toBe(true);
  }
  expect(noWall).toBeGreaterThan(0.05);
  expect(withWall).toBeLessThan(0.015);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('production GI interpolation preserves uniform irradiance at center, probe, and fringe corners', async ({ page }) => {
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-visibility.ts') as {
      runGiUniformIrradianceFixture: typeof runGiUniformIrradianceFixture;
    };
    return fixture.runGiUniformIrradianceFixture();
  }) as UniformResult;
  console.log('GI_UNIFORM_IRRADIANCE_GPU_DIAGNOSTIC ' + JSON.stringify(result.samples));
  expect(result.samples).toHaveLength(9);
  for (const sample of result.samples) {
    for (let channel = 0; channel < 3; channel++) {
      expect(Math.abs(sample.actual[channel] - sample.expected[channel]),
        `${sample.label} channel ${channel}`).toBeLessThanOrEqual(0.002);
    }
  }
  expect(result.uncaptured).toEqual([]);
  expect(result.validationError).toBe('');
  expect(result.ownedBytes).toBe(0);
  expect(pageErrors).toEqual([]);
});
