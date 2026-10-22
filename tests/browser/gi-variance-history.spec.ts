import { expect, test } from '@playwright/test';
import type { runGiVarianceHistoryFixture } from './fixtures/gi-variance-history';

function population(values: readonly number[], weights: readonly number[]) {
  const mean = values.reduce((sum, value, index) => sum + value * weights[index], 0);
  const variance = values.reduce((sum, value, index) => sum + weights[index] * (value - mean) ** 2, 0);
  return { mean, variance };
}

test('production final gather accumulates raw luminance population variance with bounded history', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-variance-history.ts') as {
      runGiVarianceHistoryFixture: typeof runGiVarianceHistoryFixture;
    };
    return fixture.runGiVarianceHistoryFixture();
  });
  expect(pageErrors).toEqual([]);
  expect(result.errors).toEqual([]);
  expect(result.validationError).toBe('');
  expect(result.ownedBytes).toBe(0);
  expect(result.stationary).toHaveLength(result.estimates.length);
  for (let index = 0; index < result.estimates.length; index++) {
    // First four estimates have uniform population weights. Later frames use
    // a saturated four-update history: each older weight decays by 3/4.
    const weights = result.estimates.slice(0, index + 1).map((_, sample) => {
      if (index < 4) return 1 / (index + 1);
      return sample < 4 ? 0.75 ** (index - 3) / 4 : 0.25 * 0.75 ** (index - sample);
    });
    const expected = population(result.estimates.slice(0, index + 1), weights);
    const actual = result.stationary[index];
    expect(actual.count).toBe(Math.min(index + 1, 4));
    expect(Math.abs(actual.mean - expected.mean), `frame ${index + 1} mean`).toBeLessThan(2e-5);
    expect(Math.abs(actual.variance - expected.variance), `frame ${index + 1} population variance`).toBeLessThan(2e-4);
  }
  expect(result.stationary[1].variance).toBeCloseTo(1, 5);
  expect(result.stationary[3].variance).toBeCloseTo(8.75, 4);
  for (const sample of result.discontinuities) {
    expect(sample.count, sample.label).toBe(1);
    expect(sample.mean, sample.label).toBeCloseTo(3, 5);
    expect(sample.variance, sample.label).toBe(0);
  }
  expect(result.discontinuities.map(sample => sample.label)).toEqual([
    'reset', 'source', 'entity', 'generation', 'representation-token',
    'history-version', 'topology-version', 'material-revision',
  ]);
  // The four accepted old anchors have different means. Their mixture has
  // within-history variance 1 plus variance among means; a new estimate 10
  // contributes one fifth of the resulting five-update population.
  const previous = population([0, 2, 4, 6], [0.25, 0.25, 0.25, 0.25]);
  const expectedMovingMean = previous.mean * 0.8 + 10 * 0.2;
  const expectedMovingVariance = 0.8 * (1 + previous.variance + (previous.mean - expectedMovingMean) ** 2)
    + 0.2 * (10 - expectedMovingMean) ** 2;
  expect(result.moved.count).toBe(5);
  expect(Number.isFinite(result.moved.variance)).toBe(true);
  expect(Math.abs(result.moved.mean - expectedMovingMean)).toBeLessThan(2e-5);
  expect(Math.abs(result.moved.variance - expectedMovingVariance)).toBeLessThan(2e-4);
  const compacted = result.compacted;
  expect(compacted.sourceSlot).toBe(1);
  expect(compacted.targetSlot).toBe(3);
  expect(compacted.mapRank).toBe(7);
  // Four prior observations [1,1,3,3] have mean 2 and variance 1. The
  // stationary map lookup must preserve them after the receiver changes slot.
  const expectedCompacted = population([1, 1, 3, 3, 10], [0.2, 0.2, 0.2, 0.2, 0.2]);
  expect(compacted.recovered.count).toBe(5);
  expect(Math.abs(compacted.recovered.mean - expectedCompacted.mean)).toBeLessThan(2e-5);
  expect(Math.abs(compacted.recovered.variance - expectedCompacted.variance)).toBeLessThan(2e-4);
  expect(compacted.withoutMap).toEqual({ mean: 10, count: 1, variance: 0 });
  expect(compacted.rejected.map(sample => sample.label)).toEqual([
    'reset', 'source', 'entity', 'generation', 'representation-token',
    'history-version', 'topology-version', 'material-revision', 'position', 'stale-frame',
  ]);
  for (const sample of compacted.rejected) {
    expect(sample.count, `compacted ${sample.label}`).toBe(1);
    expect(sample.mean, `compacted ${sample.label}`).toBeCloseTo(10, 5);
    expect(sample.variance, `compacted ${sample.label}`).toBe(0);
  }
});
