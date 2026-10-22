import { expect, test } from '@playwright/test';
import type { runGiSupplementalFixture } from './fixtures/gi-supplemental';

test('supplemental raster receivers fill a thin surface gap and reject removed or recycled history', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-supplemental.ts') as { runGiSupplementalFixture: typeof runGiSupplementalFixture };
    return fixture.runGiSupplementalFixture();
  });
  await testInfo.attach('gi-supplemental.json', { body: JSON.stringify(result), contentType: 'application/json' });
  expect(errors).toEqual([]);
  const [baseline, repair] = result.runs;
  expect(baseline.stationary.reconstruction.points[0].branch).toBe('uncovered');
  expect(repair.stationary.reconstruction.points[0].branch).toBe('supplemental');
  expect(repair.stationary.reconstruction.points[0].supplementalAccepted).toBeGreaterThan(0);
  expect(repair.stationary.hdr[1]).toEqual(baseline.stationary.hdr[1]);
  for (const run of result.runs) { expect(run.errors).toEqual([]); expect(run.afterDispose).toBe(0); }
  for (const stage of [repair.stationary, repair.moving]) {
    expect(stage.reconstruction.points[0].surface?.entityId).toBe(2);
    expect(stage.reconstruction.points[0].branch).toBe('supplemental');
    expect(stage.diagnostic.anchorsPerUpdate).toBeLessThanOrEqual(512);
    const valid = stage.anchors!.anchors.filter(a => a.valid && a.identity[1] === 2);
    expect(valid.length).toBeGreaterThan(0);
    for (const anchor of valid) for (let c = 0; c < 3; c++) {
      expect(Math.abs(anchor.irradiance[c] - Math.PI * result.sky[c])).toBeLessThan(2e-5);
    }
  }
  expect(repair.stationary.anchors!.anchors.some(a => a.valid && a.historyUpdates === 16)).toBe(true);
  expect(repair.moving.anchors!.anchors.some(a => a.valid && a.historyUpdates > 1 && a.historyUpdates <= 8)).toBe(true);
  expect(repair.removed.reconstruction.points[0].branch).toBe('primary');
  expect(repair.removed.anchors!.anchors.filter(a => a.valid && a.identity[1] === 2)).toEqual([]);
  const recycled = repair.replaced.anchors!.anchors.filter(a => a.valid && a.identity[1] === 2);
  expect(recycled.length).toBeGreaterThan(0);
  expect(recycled.every(a => a.identity[2] === 2 && a.historyUpdates === 1)).toBe(true);
});

test('supplemental split queries preserve two area sources and multi-segment transport', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const { runGiPathSegmentsFixture } = await import('/tests/browser/fixtures/' + 'gi-path-segments.ts');
    return runGiPathSegmentsFixture(2, true) as Promise<Awaited<ReturnType<typeof import('./fixtures/gi-path-segments').runGiPathSegmentsFixture>>>;
  });
  await testInfo.attach('gi-supplemental-area.json', { body: JSON.stringify(result), contentType: 'application/json' });
  expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  expect(result.unlitMulti.diagnostics.activeEmitters).toBe(2);
  expect(result.unlitMulti.diagnostics.supplementalQueryPassesPerUpdate).toBe(5);
  expect(result.missSingle.supplementalAnchors.some(a => a.valid)).toBe(true);
  expect(result.supplementalAreaChecks).toHaveLength(9);
  for (const check of result.supplementalAreaChecks) {
    expect(check.reference[0] - Math.PI * result.sky[0]).toBeGreaterThan(0.02);
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(check.anchor.irradiance[c] - check.reference[c])).toBeLessThan(check.stage === 2 ? 0.055 : 0.035);
    }
  }
  for (let c = 0; c < 3; c++) {
    for (const stage of [result.missSingle, result.missMulti, result.morphedAway]) {
      expect(Math.abs(stage.anchor.filteredIrradiance[c] - Math.PI * result.sky[c])).toBeLessThan(0.003);
    }
    for (const stage of [result.unlitSingle, result.unlitMulti]) {
      expect(Math.abs(stage.anchor.filteredIrradiance[c] - result.reference[c])).toBeLessThan(0.035);
    }
    expect(Math.abs(result.doubled.anchor.filteredIrradiance[c] - result.doubledReference[c])).toBeLessThan(0.055);
  }
});
