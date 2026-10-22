import { expect, test } from '@playwright/test';
import type { runGiMultiSurfaceFixture } from './fixtures/gi-supplemental';

test('coverage compaction keeps several thin surfaces in one tile without stale removed identities', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'gi-supplemental.ts') as { runGiMultiSurfaceFixture: typeof runGiMultiSurfaceFixture };
    return fixture.runGiMultiSurfaceFixture();
  });
  await testInfo.attach('gi-multisurface.json', { body: JSON.stringify(result), contentType: 'application/json' });
  expect(errors).toEqual([]);
  const [baseline, repair] = result.runs;
  expect(baseline.stationary.reconstruction.points.map(p => p.branch)).toEqual(['uncovered', 'uncovered', 'uncovered']);
  for (const stage of [repair.stationary, repair.moving]) {
    expect(stage.reconstruction.points.map(p => p.surface?.entityId)).toEqual([2, 3, 4]);
    expect(stage.reconstruction.points.map(p => p.branch)).toEqual(['supplemental', 'supplemental', 'supplemental']);
    expect(stage.map!.tiles[0].indices.length).toBeGreaterThanOrEqual(3);
    expect(new Set(stage.map!.tiles[0].indices).size).toBe(stage.map!.tiles[0].indices.length);
    for (const anchor of stage.anchors!.anchors) {
      expect(anchor.valid).toBe(true);
      for (let c = 0; c < 3; c++) expect(Math.abs(anchor.irradiance[c] - Math.PI * result.sky[c])).toBeLessThan(2e-5);
    }
  }
  expect(repair.stationary.anchors!.anchors.every(a => a.historyUpdates === 16)).toBe(true);
  expect(repair.moving.anchors!.anchors.every(a => a.historyUpdates > 1 && a.historyUpdates <= 8)).toBe(true);
  expect(repair.removed.reconstruction.points[0].surface?.entityId).toBe(1);
  expect(repair.removed.anchors!.anchors.some(a => a.identity[1] === 2)).toBe(false);
  for (const run of result.runs) { expect(run.errors).toEqual([]); expect(run.afterDispose).toBe(0); }
});
