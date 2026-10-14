import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runGiFloorReferenceFixture } from './fixtures/gi-floor-reference';

type Fixture = Awaited<ReturnType<typeof runGiFloorReferenceFixture>>;

type Compact = ReturnType<typeof compactResult>;

function compactResult(result: Fixture) {
  return {
    grid: result.grid, camera: result.camera, canvas: result.canvas,
    reference: result.reference,
    samples: result.samples.map((sample, index) => ({
      index, world: sample.world, pixel: sample.pixel,
      cameraHit: sample.cameraHit ? {
        batch: sample.cameraHit.batch, entityId: sample.cameraHit.entityId,
        normal: sample.cameraHit.normal, baseColor: sample.cameraHit.baseColor,
      } : null,
      twelveBounce: sample.reference?.truncatedTwelveBounce ?? null,
      lampOffTwelveBounce: sample.lampOffReference?.truncatedTwelveBounce ?? null,
      allHitCategories: sample.reference?.allHitCategories ?? null,
      allSourceContributions: sample.reference?.allSourceContributions ?? null,
      emitterHitCountsByBounce: sample.reference?.emitterHitCountsByBounce ?? null,
    })),
    cases: Object.fromEntries(Object.entries(result.cases).map(([name, value]) =>
      [name, { frame: value.frame, hdr: value.hdr,
        historyUpdates: value.gi?.historyUpdates }])) as Record<string, {
      frame: number; hdr: number[][]; historyUpdates: number | undefined;
    }>,
    lampAblation: result.lampAblation,
    probes: {
      single: { frame: result.probes.single.frame, invalid: result.probes.single.invalid },
      multi: { frame: result.probes.multi.frame, invalid: result.probes.multi.invalid },
      sunOffMulti: { frame: result.probes.sunOffMulti.frame, invalid: result.probes.sunOffMulti.invalid },
      lampOffMulti: { frame: result.probes.lampOffMulti.frame, invalid: result.probes.lampOffMulti.invalid },
    },
    errors: result.errors, afterDispose: result.afterDispose,
    pngs: { multi: result.cases.multi.png, lampOffMulti: result.cases.lampOffMulti.png },
  };
}

function metrics(result: Compact) {
  const points = result.reference.selectedIndices.map(index => {
    const reference = result.samples[index].twelveBounce!.mean;
    const measured = result.cases.multi.hdr[index].slice(0, 3);
    const direct = result.cases.off.hdr[index].slice(0, 3);
    const relativeError = reference.map((value, channel) =>
      (measured[channel] - direct[channel] - value) / Math.max(value, 0.001));
    return { index, measured, direct, reference, relativeError,
      emitterHitCountsByBounce: result.samples[index].emitterHitCountsByBounce,
      emitterContribution: result.samples[index].allSourceContributions?.['emissive-strip'] ?? [0, 0, 0] };
  });
  const channelErrors = points.flatMap(point => point.relativeError);
  const lampReference = result.lampAblation.referenceCurrentMinusLampOff
    .find(point => point.index === 3)?.twelveBounce[0];
  const lampMeasured = result.lampAblation.measuredCurrentMinusLampOff[3][0];
  return { grid: result.grid, points, denominatorFloor: 0.001,
    meanAbsoluteRelativeError: channelErrors.reduce((sum, value) => sum + Math.abs(value), 0)
      / channelErrors.length,
    index3LampRed: { measured: lampMeasured, reference: lampReference,
      absoluteError: lampReference === undefined ? null : Math.abs(lampMeasured - lampReference) } };
}

test('emissive area sampling improves the floor against a 12-bounce reference', async ({ page }, testInfo) => {
  test.setTimeout(900_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });

  const runs: { name: string; result: Compact; metrics: ReturnType<typeof metrics> }[] = [];
  for (const [name, emissiveSamples] of [['area0', 0], ['area32', null]] as const) {
    const result = await page.evaluate(async override => {
      const module = await import('/tests/browser/fixtures/' + 'gi-floor-reference.ts') as {
        runGiFloorReferenceFixture: typeof runGiFloorReferenceFixture;
      };
      return module.runGiFloorReferenceFixture(false, undefined, undefined,
        override === 0 ? 0 : undefined);
    }, emissiveSamples) as Fixture;
    const compact = compactResult(result);
    const measured = metrics(compact);
    runs.push({ name, result: compact, metrics: measured });
    const jsonPath = testInfo.outputPath(`gi-floor-${name}.json`);
    await writeFile(jsonPath, JSON.stringify({ ...compact, pngs: undefined, metrics: measured }, null, 2));
    await testInfo.attach(`gi-floor-${name}.json`, { path: jsonPath, contentType: 'application/json' });
    for (const [caseName, png] of Object.entries(compact.pngs)) {
      const path = testInfo.outputPath(`gi-floor-${name}-${caseName}.png`);
      await writeFile(path, Buffer.from(png.split(',')[1], 'base64'));
      await testInfo.attach(`gi-floor-${name}-${caseName}.png`, { path, contentType: 'image/png' });
    }
  }

  const [baseline, refined] = runs;
  const summary = { baseline: baseline.metrics, refined: refined.metrics,
    improvementRatio: refined.metrics.meanAbsoluteRelativeError
      / baseline.metrics.meanAbsoluteRelativeError };
  const summaryPath = testInfo.outputPath('gi-floor-area-quality-summary.json');
  await writeFile(summaryPath, JSON.stringify(summary, null, 2));
  await testInfo.attach('gi-floor-area-quality-summary.json', {
    path: summaryPath, contentType: 'application/json',
  });
  console.log('GI_FLOOR_AREA_QUALITY ' + JSON.stringify(summary));

  expect(pageErrors).toEqual([]);
  for (const run of runs) {
    expect(run.result.errors).toEqual([]);
    expect(run.result.afterDispose).toBe(0);
    expect(run.result.reference.pathsPerPoint).toBe(4096);
    expect(run.result.reference.maxBounces).toBe(12);
    expect(run.result.reference.selectedIndices).toHaveLength(6);
    expect(run.result.probes.multi.invalid).toBe(0);
    expect(run.result.probes.lampOffMulti.invalid).toBe(0);
    expect(run.result.cases.multi.historyUpdates).toBe(12);
    expect(run.result.cases.lampOffMulti.historyUpdates).toBe(12);
    for (const point of run.metrics.points) {
      expect(run.result.samples[point.index].cameraHit?.batch).toBe('room-neutral');
      expect(run.result.samples[point.index].cameraHit?.entityId).toBe(1);
      for (const value of [...point.reference, ...point.measured, ...point.direct,
        ...point.relativeError]) expect(Number.isFinite(value)).toBe(true);
    }
  }
  expect(baseline.result.reference.selectedIndices).toEqual(refined.result.reference.selectedIndices);
  expect(baseline.result.grid.counts).toEqual([6, 6, 6]);
  expect(refined.result.grid.counts).toEqual([6, 6, 6]);
  expect(baseline.result.grid.raysPerProbe).toBe(128);
  expect(refined.result.grid.raysPerProbe).toBe(128);
  expect(baseline.result.grid.emissiveSamplesPerProbe).toBe(0);
  expect(refined.result.grid.emissiveSamplesPerProbe).toBe(32);
  expect({ ...baseline.result.grid, emissiveSamplesPerProbe: undefined })
    .toEqual({ ...refined.result.grid, emissiveSamplesPerProbe: undefined });

  expect(refined.metrics.meanAbsoluteRelativeError).toBeLessThan(0.065);
  expect(refined.metrics.meanAbsoluteRelativeError)
    .toBeLessThan(baseline.metrics.meanAbsoluteRelativeError * 0.8);
  for (const point of refined.metrics.points) for (const error of point.relativeError) {
    expect(Math.abs(error), `floor point ${point.index}`).toBeLessThan(0.2);
  }
  expect(refined.metrics.index3LampRed.absoluteError).not.toBeNull();
  expect(baseline.metrics.index3LampRed.absoluteError).not.toBeNull();
  expect(refined.metrics.index3LampRed.absoluteError!).toBeLessThan(0.003);
  expect(refined.metrics.index3LampRed.absoluteError!)
    .toBeLessThan(baseline.metrics.index3LampRed.absoluteError! * 0.25);
});
