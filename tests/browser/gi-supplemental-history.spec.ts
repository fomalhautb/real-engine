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

test('progressive supplemental directions advance per retained update without changing sky energy or GPU allocation', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'gi-supplemental-history.ts') as {
      runSupplementalHistoryFixture: typeof runSupplementalHistoryFixture;
    };
    return fixture.runSupplementalHistoryFixture({ progressiveComparison: true });
  });
  await testInfo.attach('supplemental-progressive-history.json', { body: JSON.stringify(result), contentType: 'application/json' });
  expect(errors).toEqual([]);
  expect(result.runs).toHaveLength(2);
  const [hashed, progressive] = result.runs;
  console.log('SUPPLEMENTAL_PROGRESSIVE_ADAPTER ' + JSON.stringify(progressive.info.adapterInfo));
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(progressive.info.adapterInfo.vendor).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(hashed.sampling).toBe('frame-hash'); expect(progressive.sampling).toBe('progressive-halton');
  expect(hashed.historyFrames).toBe(8); expect(progressive.historyFrames).toBe(8);
  const options = progressive.frames[0].diagnostic.options;
  expect(options.raysPerAnchor).toBe(2); expect(options.pathSegments).toBe(2);
  expect(options.maxHistory).toBe(2); expect(options.supplementalSamplingSeed).toBe(0xffff_ffff);
  expect(progressive.frames[0].diagnostic.pathSegmentsPerUpdate).toBe(2);
  expect(progressive.frames[0].diagnostic.supplementalDirectionSampling?.method).toBe('random-shifted-halton-prime-dimensions');
  expect(progressive.frames[0].diagnostic.supplementalDirectionSampling?.dimensionBases).toEqual([2, 3, 5, 7, 11, 13, 17, 19]);
  expect(hashed.frames[0].diagnostic.supplementalDirectionSampling).toBeNull();

  const expectSky = (stage: typeof progressive.frames[number]) => {
    for (const anchor of stage.anchors.anchors) {
      expect(anchor.valid).toBe(true);
      for (let c = 0; c < 3; c++) expect(Math.abs(anchor.irradiance[c] - Math.PI * result.sky[c])).toBeLessThan(2e-5);
      expect(anchor.luminanceVariance).toBeLessThan(1e-10);
    }
  };
  for (let i = 0; i < progressive.frames.length; i++) {
    const stage = progressive.frames[i], control = hashed.frames[i];
    expect(stage.frame).toBe(control.frame);
    expect(stage.ownedGpuBytes).toBe(control.ownedGpuBytes);
    expect(stage.diagnostic.ownedGpuBytes).toBe(control.diagnostic.ownedGpuBytes);
    expect(stage.diagnostic.queryRaysPerUpdate).toBe(control.diagnostic.queryRaysPerUpdate);
    expect(stage.primary).toEqual(control.primary);
    if (i >= 7) {
      expect(stage.reconstruction!.points.map(p => p.branch)).toEqual(Array(7).fill('supplemental'));
      expect(stage.anchors.anchors).toHaveLength(7);
      expectSky(stage); expectSky(control);
    }
  }

  // The cached count advances once per selected surface update, despite two rays and two path segments.
  expect(progressive.frames[0].anchors.anchors.length).toBeGreaterThan(0);
  expect(progressive.frames[0].anchors.anchors.every(a => a.progressiveSampling?.updates === 1)).toBe(true);
  let carried = 0, advanced = 0, maximumUpdates = 0;
  for (let i = 8; i < progressive.frames.length; i++) {
    const priorByIndex = new Map(progressive.frames[i - 1].anchors.anchors.map(a => [a.index, a]));
    for (const anchor of progressive.frames[i].anchors.anchors) {
      const prior = priorByIndex.get(anchor.index)!;
      expect(anchor.identity).toEqual(prior.identity);
      expect(anchor.position).toEqual(prior.position);
      expect(anchor.progressiveSampling?.key).toBe(prior.progressiveSampling?.key);
      expect(anchor.progressiveSampling?.key).toBeGreaterThanOrEqual(0);
      expect(anchor.progressiveSampling?.key).toBeLessThanOrEqual(0xff_ffff);
      expect(anchor.historyUpdates).toBeLessThanOrEqual(2);
      const updates = anchor.progressiveSampling!.updates;
      maximumUpdates = Math.max(maximumUpdates, updates);
      if (anchor.sampleAgeFrames > 0) {
        carried++;
        expect(anchor.sampleAgeFrames).toBe(prior.sampleAgeFrames + 1);
        expect(updates).toBe(prior.progressiveSampling!.updates);
        expect(anchor.historyUpdates).toBe(prior.historyUpdates);
        expect(anchor.irradiance).toEqual(prior.irradiance);
      } else {
        advanced++;
        expect(updates).toBe(prior.progressiveSampling!.updates + 1);
      }
    }
  }
  expect(carried).toBeGreaterThan(0); expect(advanced).toBeGreaterThan(0);
  expect(maximumUpdates).toBeGreaterThan(options.maxHistory);
  expect(progressive.frames.at(-1)!.anchors.anchors.some(a => a.historyUpdates === 2)).toBe(true);

  for (let i = 0; i < progressive.moving.length; i++) {
    const stage = progressive.moving[i];
    expect(stage.ownedGpuBytes).toBe(hashed.moving[i].ownedGpuBytes);
    expect(stage.anchors.anchors.every(a => a.valid && a.sampleAgeFrames <= 8 &&
      (a.progressiveSampling?.updates ?? 0) >= 1)).toBe(true);
    expectSky(stage);
  }
  expectSky(progressive.reset); expectSky(progressive.removed);
  expectSky(progressive.recycled); expectSky(progressive.restored);
  for (const stage of [progressive.reset, progressive.removed, progressive.recycled,
    progressive.lightChanged, progressive.materialChanged]) {
    expect(stage.anchors.anchors.length).toBeLessThan(7);
    expect(stage.anchors.anchors.every(a => a.sampleAgeFrames === 0 && a.historyUpdates === 1 &&
      a.progressiveSampling?.updates === 1)).toBe(true);
  }
  expect(progressive.removed.reconstruction!.points[0].surface?.entityId).toBe(1);
  expect(progressive.removed.anchors.anchors.some(a => a.identity[1] === 66)).toBe(false);
  expect(progressive.recycled.anchors.anchors.some(a => a.identity[1] === 66 && a.identity[2] === 1)).toBe(false);
  expect(progressive.restored.anchors.anchors.find(a => a.identity[1] === 66)?.identity[2]).toBe(2);
  for (const [control, stage] of [
    [hashed.reset, progressive.reset], [hashed.removed, progressive.removed],
    [hashed.recycled, progressive.recycled], [hashed.restored, progressive.restored],
    [hashed.lightChanged, progressive.lightChanged], [hashed.materialChanged, progressive.materialChanged],
  ]) {
    expect(stage.ownedGpuBytes).toBe(control.ownedGpuBytes);
    expect(stage.diagnostic.ownedGpuBytes).toBe(control.diagnostic.ownedGpuBytes);
  }
  for (const run of result.runs) { expect(run.errors).toEqual([]); expect(run.afterDispose).toBe(0); }
});
