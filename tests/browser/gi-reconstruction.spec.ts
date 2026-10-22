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
  expect(result.results).toHaveLength(21);
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
  const by = (label: string) => result.results.find(sample => sample.label === label)!.diagnostic;
  for (const sample of result.results) {
    expect(Number.isFinite(sample.diagnostic.total), sample.label).toBe(true);
    expect(sample.diagnostic.rejections, sample.label).toHaveLength(16);
  }
  for (const sample of result.results.slice(0, 12)) {
    expect(sample.diagnostic.branch).toBe(0);
    expect(sample.diagnostic.total).toBe(0);
    expect(sample.diagnostic.rejections).toEqual(Array(16).fill(128));
  }
  const ring = by('matching-normal-outside-primary-cell');
  expect(ring.branch).toBe(2);
  expect([ring.primaryAccepted, ring.ringAccepted]).toEqual([0, 1]);
  expect(ring.total).toBeCloseTo(Math.exp(-2.5), 6);
  expect(ring.rejections).toEqual([4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4]);
  for (const [label, rejection] of [
    ['thin-parallel-wall-rejected', 24],
    ['different-geometric-plane-rejected', 16],
    ['stale-ring-anchor-rejected', 1],
    ['distant-ring-anchor-rejected', 32],
    ['different-entity-ring-anchor-rejected', 2],
  ] as const) {
    const diagnostic = by(label);
    expect(diagnostic.branch, label).toBe(3);
    expect(diagnostic.total, label).toBe(0);
    expect([diagnostic.primaryAccepted, diagnostic.ringAccepted], label).toEqual([0, 0]);
    expect(diagnostic.rejections.slice(0, 4), label).toEqual([4, 4, 4, 4]);
    expect(diagnostic.rejections[8], label).toBe(rejection);
    expect(diagnostic.rejections.filter((mask, index) => index !== 8 && mask !== 4), label).toEqual([]);
  }
  for (const label of ['covered-cell-does-not-use-ring', 'partial-last-cell-position']) {
    const diagnostic = by(label);
    expect(diagnostic.branch, label).toBe(1);
    expect(diagnostic.total, label).toBeCloseTo(1, 6);
    expect([diagnostic.primaryAccepted, diagnostic.ringAccepted], label).toEqual([4, 0]);
    expect(diagnostic.rejections, label).toEqual([0, 0, 0, 0, ...Array(12).fill(128)]);
  }
  const edge = by('edge-out-of-bounds-order');
  expect(edge.branch).toBe(3);
  expect(edge.total).toBe(0);
  expect([edge.primaryAccepted, edge.ringAccepted]).toEqual([0, 0]);
  expect(edge.rejections).toEqual([64, 64, 64, 4, 64, 64, 64, 64, 64, 64, 64, 4, 64, 64, 4, 4]);
  expect(result.errors).toEqual([]); expect(result.validationError).toBe('');
  expect(result.ownedBytes).toBe(0); expect(pageErrors).toEqual([]);
});
