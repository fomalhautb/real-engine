import { expect, test } from '@playwright/test';
import type { runGiFilterBoundaryFixture, runGiFilterBoundaryDenoiseFixture } from './fixtures/gi-filter-boundary';

async function ready(page: import('@playwright/test').Page) {
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
}

test('production final gather resolves the analytic black half-roof and exposes raw versus filtered error', async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await ready(page);
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-filter-boundary.ts') as {
      runGiFilterBoundaryFixture: typeof runGiFilterBoundaryFixture;
    };
    return fixture.runGiFilterBoundaryFixture();
  });
  await testInfo.attach('gi-filter-boundary-production.json', {
    body: JSON.stringify(result, null, 2), contentType: 'application/json',
  });
  console.log('GI_FILTER_BOUNDARY_PRODUCTION ' + JSON.stringify({ schedule: result.schedule,
    samples: result.samples.length, metrics: result.metrics, widths: result.widths,
    errors: result.errors, afterDispose: result.afterDispose }));
  expect(pageErrors).toEqual([]); expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  expect(result.size).toEqual({ width: 128, height: 128 });
  expect(result.schedule.firstSamplingFrame).toBe(3);
  expect(result.schedule.lastSamplingFrame).toBe(258);
  expect(result.schedule.samplesPerAnchor).toBe(4096);
  expect(result.diagnostics?.actualSpacing).toBe(4);
  expect(result.diagnostics?.pathSegmentsPerUpdate).toBe(1);
  expect(result.diagnostics?.options.receiver).toBe('raster');
  expect(result.samples.length).toBeGreaterThanOrEqual(48);
  for (const side of [-1, 1]) expect(result.samples.filter(p => p.x * side > 0).length).toBeGreaterThanOrEqual(16);
  expect(Math.min(...result.samples.map(p => p.expected))).toBeLessThan(0.2);
  expect(Math.max(...result.samples.map(p => p.expected))).toBeGreaterThan(0.8);
  expect(Math.min(...result.samples.map(p => p.raw))).toBeLessThan(0.25);
  expect(Math.max(...result.samples.map(p => p.raw))).toBeGreaterThan(0.75);
  expect(Math.max(...result.samples.map(p => p.filtered))
    - Math.min(...result.samples.map(p => p.filtered))).toBeGreaterThan(0.5);
  expect(result.constant.anchors.length).toBeGreaterThanOrEqual(48);
  for (const anchor of result.constant.anchors) for (let channel = 0; channel < 3; channel++) {
    const expected = Math.PI * result.sky[channel];
    expect(Math.abs(anchor.irradiance[channel] - expected), 'constant sky raw energy').toBeLessThan(2e-5);
    expect(Math.abs(anchor.filteredIrradiance[channel] - expected), 'constant sky filtered energy').toBeLessThan(2e-5);
  }
  for (const anchor of result.samples) {
    expect(anchor.historyUpdates).toBe(256);
    expect(anchor.state.slice(2)).toEqual([0, 0]);
    expect(anchor.farEdgeDistance).toBeGreaterThan(result.maxDistance);
    expect(anchor.finiteRangeTailFraction).toBeLessThan(6.3e-6);
    // A six-sigma Bernoulli sampling envelope, plus deterministic range and
    // float tolerances. Hash/stratified samples are not claimed to be a formal
    // independent-random confidence interval. No measured result sets this gate.
    const samplingEnvelope = 6 * Math.sqrt(anchor.expected * (1 - anchor.expected)
      / result.schedule.samplesPerAnchor);
    for (let channel = 0; channel < 3; channel++) {
      const raw = anchor.irradiance[channel] / (Math.PI * result.sky[channel]);
      const filtered = anchor.filteredIrradiance[channel] / (Math.PI * result.sky[channel]);
      expect(Number.isFinite(raw)).toBe(true); expect(Number.isFinite(filtered)).toBe(true);
      expect(Math.abs(raw - anchor.expected), 'raw half-plane transport').toBeLessThan(
        samplingEnvelope + anchor.finiteRangeTailFraction + 2e-4);
      expect(filtered).toBeGreaterThanOrEqual(0); expect(filtered).toBeLessThanOrEqual(1.00002);
      expect(Math.abs(filtered - anchor.filtered), 'spectral ratio').toBeLessThan(2e-5);
    }
  }
  expect(result.metrics.raw.rmse).toBeLessThan(2 / Math.sqrt(result.schedule.samplesPerAnchor));
  // Filter bias is intentionally reported separately, not hidden inside the
  // Monte Carlo tolerance or accepted because the whole surface became darker.
  // This baseline does not claim that the existing filter solves light leaks.
  for (const metric of Object.values(result.metrics.filtered)) expect(Number.isFinite(metric)).toBe(true);
});

test('production denoise quantifies analytic boundary diffusion while preserving constants and four-mm separation', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await ready(page);
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-filter-boundary.ts') as {
      runGiFilterBoundaryDenoiseFixture: typeof runGiFilterBoundaryDenoiseFixture;
    };
    return fixture.runGiFilterBoundaryDenoiseFixture();
  });
  await testInfo.attach('gi-filter-boundary-denoise.json', {
    body: JSON.stringify(result, null, 2), contentType: 'application/json',
  });
  console.log('GI_FILTER_BOUNDARY_DENOISE ' + JSON.stringify({ metrics: result.halfPlane.metrics,
    widths: result.halfPlane.widths, constant: result.constant.metrics.filtered,
    thinWall: result.thinWall.metrics.filtered, errors: result.errors }));
  expect(pageErrors).toEqual([]); expect(result.errors).toEqual([]);
  expect(result.validationError).toBe(''); expect(result.ownedBytes).toBe(0);
  expect(result.wallSeparation).toBe(0.004);
  for (const sample of result.constant.samples) for (const channel of sample.rgb) {
    expect(Math.abs(channel - sample.expected), 'constant field energy').toBeLessThan(2e-6);
  }
  for (const sample of result.thinWall.samples) for (const channel of sample.rgb) {
    expect(Math.abs(channel - sample.expected), '4mm parallel surfaces with identical identity').toBeLessThan(2e-6);
  }
  const half = result.halfPlane;
  expect(half.metrics.raw.maxAbsoluteError).toBeLessThan(1e-7);
  expect(Math.abs(half.metrics.filtered.meanSignedError), 'symmetric field total energy').toBeLessThan(2e-6);
  for (const sample of half.samples) {
    expect(Number.isFinite(sample.filtered)).toBe(true);
    expect(sample.filtered).toBeGreaterThanOrEqual(0); expect(sample.filtered).toBeLessThanOrEqual(1);
    const rowStart = Math.floor(sample.index / result.gridColumns) * result.gridColumns;
    const mirrored = half.samples[rowStart + result.gridColumns - 1 - sample.index % result.gridColumns];
    expect(Math.abs(sample.filtered + mirrored.filtered - 1), 'complementary half-plane symmetry').toBeLessThan(2e-6);
  }
  // Exact input removes stochastic noise: dark-side gain, bright-side loss and
  // width change quantify spatial diffusion without prescribing today's kernel.
  expect(half.widths.expected.width).not.toBeNull();
  expect(half.widths.filtered.width).not.toBeNull();
});
