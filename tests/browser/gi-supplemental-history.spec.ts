import { expect, test } from '@playwright/test';
import type { runSupplementalHistoryFixture } from './fixtures/gi-supplemental-history';

test('bounded supplemental updates retain unscheduled surfaces and invalidate stale history', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'gi-supplemental-history.ts') as {
      runSupplementalHistoryFixture: typeof runSupplementalHistoryFixture;
    };
    return fixture.runSupplementalHistoryFixture();
  });
  await testInfo.attach('supplemental-history.json', { body: JSON.stringify(result), contentType: 'application/json' });
  console.log('SUPPLEMENTAL_HISTORY_ADAPTER ' + JSON.stringify(result.runs[1].info.adapterInfo));
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.runs[1].info.adapterInfo.vendor).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(errors).toEqual([]);
  const [baseline, retained, expiring] = result.runs;
  expect(baseline.frames.at(-1)!.reconstruction!.points.filter(p => p.branch === 'uncovered').length).toBeGreaterThan(0);
  for (const stage of retained.frames.slice(7)) {
    expect(stage.reconstruction!.points.map(p => p.branch)).toEqual(Array(7).fill('supplemental'));
    expect(stage.anchors.anchors).toHaveLength(7);
    expect(stage.map.tiles[0].indices.every(index => index >= 512)).toBe(true);
    expect(stage.diagnostic.queryRaysPerUpdate).toBe(baseline.frames[stage.frame - 1].diagnostic.queryRaysPerUpdate);
    expect(stage.primary).toEqual(baseline.frames[stage.frame - 1].primary);
    for (const anchor of stage.anchors.anchors) {
      expect(anchor.valid).toBe(true); expect(anchor.sampleAgeFrames).toBeLessThanOrEqual(8);
      for (let c = 0; c < 3; c++) expect(Math.abs(anchor.irradiance[c] - Math.PI * result.sky[c])).toBeLessThan(2e-5);
      expect(anchor.luminanceVariance).toBeLessThan(1e-10);
    }
  }
  // Carried records age without fabricating new samples or shrinking variance.
  let carried = 0;
  for (let i = 8; i < retained.frames.length; i++) {
    const old = new Map(retained.frames[i - 1].anchors.anchors.map(a => [a.index, a]));
    for (const anchor of retained.frames[i].anchors.anchors) if (anchor.sampleAgeFrames > 0) {
      const prior = old.get(anchor.index)!; carried++;
      expect(anchor.sampleAgeFrames).toBe(prior.sampleAgeFrames + 1);
      expect(anchor.historyUpdates).toBe(prior.historyUpdates);
      expect(anchor.irradiance).toEqual(prior.irradiance);
      expect(anchor.luminanceVariance).toBe(prior.luminanceVariance);
    }
  }
  expect(carried).toBeGreaterThan(0);
  expect(retained.frames.at(-1)!.diagnostic.ownedGpuBytes - baseline.frames.at(-1)!.diagnostic.ownedGpuBytes).toBe(2048 * 340);
  for (const stage of retained.moving) {
    expect(stage.anchors.anchors.every(a => a.sampleAgeFrames <= 8 && a.historyUpdates <= 8)).toBe(true);
    for (const anchor of stage.anchors.anchors) for (let c = 0; c < 3; c++)
      expect(Math.abs(anchor.irradiance[c] - Math.PI * result.sky[c])).toBeLessThan(2e-5);
  }
  for (const stage of [retained.reset, retained.removed, retained.recycled, retained.lightChanged, retained.materialChanged]) {
    expect(stage.anchors.anchors.every(a => a.sampleAgeFrames === 0 && a.historyUpdates === 1)).toBe(true);
    expect(stage.anchors.anchors.length).toBeLessThan(7);
  }
  expect(retained.removed.reconstruction!.points[0].surface?.entityId).toBe(1);
  expect(retained.removed.anchors.anchors.some(a => a.identity[1] === 66)).toBe(false);
  expect(retained.restored.anchors.anchors.find(a => a.identity[1] === 66)?.identity[2]).toBe(2);
  expect(expiring.frames.slice(7).some(s => s.anchors.anchors.length < 7)).toBe(true);
  expect(expiring.frames.every(s => s.anchors.anchors.every(a => a.sampleAgeFrames <= 1))).toBe(true);
  for (const run of result.runs) { expect(run.errors).toEqual([]); expect(run.afterDispose).toBe(0); }
});
