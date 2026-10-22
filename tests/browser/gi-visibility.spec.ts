import { expect, test, type Page } from '@playwright/test';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from '../../src/gi-visibility';
import type { runGiUniformIrradianceFixture, runGiVisibilityLeakFixture,
  runGiVisibilityMomentsFixture, runGiVisibilityFilterFixture } from './fixtures/gi-visibility';

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
    expect(Math.abs(sample.actualVisibility - sample.expectedVisibility), `sample ${index} visibility`).toBeLessThanOrEqual(0.001);
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

test('GI visibility filters near and far directional confidence without inventing a fully clear footprint', async ({ page }) => {
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-visibility.ts') as {
      runGiVisibilityFilterFixture: typeof runGiVisibilityFilterFixture;
    };
    return fixture.runGiVisibilityFilterFixture();
  });
  const blocked = 0.0001 / (0.0001 + 4);
  for (const [name, field] of Object.entries(result)) {
    for (const sample of field.samples) {
      expect(sample.actualVisibility).toBeGreaterThanOrEqual(0);
      expect(sample.actualVisibility).toBeLessThanOrEqual(1);
      expect(Math.abs(sample.actualVisibility - sample.expectedVisibility), name).toBeLessThan(1e-5);
      if (name === 'clear') expect(sample.actualVisibility).toBeCloseTo(1, 6);
      if (name === 'blocked') expect(sample.actualVisibility).toBeCloseTo(blocked, 6);
    }
    for (const [a, b] of field.seamPairs) {
      if (name.startsWith('legacy')) continue;
      // The directions approach the fold from epsilon=1/4096 on either
      // side; they are not identical directions. A full-contrast bilinear
      // step has slope at most the map resolution in this UV interval.
      const maximumStep = 2 / 4096 * GI_DISTANCE_RESOLUTION * (1 - blocked);
      expect(Math.abs(field.samples[a].actualVisibility - field.samples[b].actualVisibility)).toBeLessThan(maximumStep + 1e-6);
    }
    expect(field.uncaptured).toEqual([]); expect(field.validationError).toBe(''); expect(field.ownedBytes).toBe(0);
  }
  const center = result.split.samples.find(s => s.uv[0] === 0.5 && s.uv[1] === 0.5)!;
  expect(center.actualVisibility).toBeCloseTo((1 + blocked) / 2, 6);
  // Interpolating the old moments instead gives mean 3 and visibility 1.
  expect(center.actual[0]).toBeCloseTo(3, 6);
  expect(center.actualVisibility).toBeLessThan(0.501);
  for (const field of [result.legacySplit, result.legacyReverse]) {
    expect(field.samples.find(s => s.uv[0] === 0.5 && s.uv[1] === 0.5)!.actualVisibility).toBeCloseTo(1, 6);
  }
  expect(result.splitReverse.samples.map(s => s.actualVisibility)).toEqual(result.split.samples.map(s => s.actualVisibility));
  expect(result.legacyReverse.samples.map(s => s.actualVisibility)).toEqual(result.legacySplit.samples.map(s => s.actualVisibility));
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
