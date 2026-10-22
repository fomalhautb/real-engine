import { expect, test } from '@playwright/test';
import type { runGiReconstructionFixture } from './fixtures/gi-reconstruction';

test('GI reconstruction retains occlusion confidence and recovers only compatible final-gather ring anchors', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-reconstruction.ts') as {
      runGiReconstructionFixture: typeof runGiReconstructionFixture;
    };
    return fixture.runGiReconstructionFixture();
  });
  console.log('GI_RECONSTRUCTION ' + JSON.stringify(result));
  expect(result.results).toHaveLength(19);
  for (const sample of result.results) for (let channel = 0; channel < 3; channel++) {
    const value = sample.actual[channel], expected = sample.expected![channel];
    expect(Number.isFinite(value), sample.label).toBe(true);
    expect(value, sample.label).toBeGreaterThanOrEqual(0);
    if (sample.upperFraction !== undefined) {
      expect(value, sample.label).toBeLessThan(expected * sample.upperFraction);
    } else {
      expect(Math.abs(value - expected), sample.label).toBeLessThan(2e-5);
    }
  }
  const boundary = result.results.find(sample => sample.label === 'zero-weight-visible-probes')!;
  const nearBoundary = result.results.find(sample => sample.label === 'near-zero-weight-visible-probes')!;
  for (let channel = 0; channel < 3; channel++) {
    expect(Math.abs(boundary.actual[channel] - nearBoundary.actual[channel]), 'cell-boundary continuity').toBeLessThan(1e-5);
  }
  const below = result.results.find(sample => sample.label === 'threshold-below')!;
  const above = result.results.find(sample => sample.label === 'threshold-above')!;
  for (let channel = 0; channel < 3; channel++) {
    expect(Math.abs(below.actual[channel] - above.actual[channel]), 'confidence-threshold continuity').toBeLessThan(5e-6);
  }
  expect(result.errors).toEqual([]); expect(result.validationError).toBe('');
  expect(result.ownedBytes).toBe(0); expect(pageErrors).toEqual([]);
});
