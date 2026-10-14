import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { GiBlueWallQuality, runGiBlueWallReferenceFixture } from './fixtures/gi-blue-wall-reference';

type Result = Awaited<ReturnType<typeof runGiBlueWallReferenceFixture>>;
const qualities: GiBlueWallQuality[] = ['baseline', 'dense-rays', 'dense-grid', 'dense-grid-fast'];

function summarize(result: Result) {
  const points = result.reference.selectedIndices.map(index => {
    const reference = result.samples[index].oneBounce!.outgoingDiffuse;
    const measured = result.cases.single.hdr[index].slice(0, 3);
    return { index, pixel: result.samples[index].pixel, reference, measured,
      relativeError: measured.map((value, channel) =>
        (value - reference[channel]) / Math.max(reference[channel], 0.001)) };
  });
  const errors = points.flatMap(point => point.relativeError);
  const mean = errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null;
  const stdDev = errors.length && mean !== null ? Math.sqrt(errors.reduce((sum, value) =>
    sum + (value - mean) ** 2, 0) / errors.length) : null;
  return { quality: result.quality, grid: result.grid, denominatorFloor: 0.001,
    points, meanSignedRelativeError: mean, relativeErrorStdDev: stdDev,
    meanAbsoluteRelativeError: errors.length
      ? errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length : null };
}

test('blue-wall probe density and ray count comparison', async ({ page }, testInfo) => {
  test.setTimeout(900_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });

  const summaries = [];
  for (const quality of qualities) {
    const result = await page.evaluate(async requestedQuality => {
      const module = await import('/tests/browser/fixtures/' + 'gi-blue-wall-reference.ts') as {
        runGiBlueWallReferenceFixture: typeof runGiBlueWallReferenceFixture;
      };
      return module.runGiBlueWallReferenceFixture(requestedQuality);
    }, quality) as Result;
    const summary = summarize(result);
    summaries.push(summary);
    const jsonPath = testInfo.outputPath(`gi-blue-wall-${quality}.json`);
    const compact = { ...result, comparisonStatistics: summary,
      cases: Object.fromEntries(Object.entries(result.cases)
        .map(([name, value]) => [name, { ...value, png: undefined }])) };
    await writeFile(jsonPath, JSON.stringify(compact, null, 2));
    await testInfo.attach(`gi-blue-wall-${quality}.json`, {
      path: jsonPath, contentType: 'application/json',
    });
    for (const [caseName, value] of Object.entries(result.cases)) {
      if (!value.png) continue;
      const path = testInfo.outputPath(`gi-blue-wall-${quality}-${caseName}.png`);
      await writeFile(path, Buffer.from(value.png.split(',')[1], 'base64'));
      await testInfo.attach(`gi-blue-wall-${quality}-${caseName}.png`, { path, contentType: 'image/png' });
    }
    expect(result.quality).toBe(quality);
    expect(result.errors).toEqual([]);
    expect(result.afterDispose).toBe(0);
    expect(result.reference.selectedIndices.length).toBeGreaterThanOrEqual(3);
    expect(result.probe.invalidProbes).toBe(0);
    for (const point of summary.points) {
      for (const value of [...point.reference, ...point.measured, ...point.relativeError]) {
        expect(Number.isFinite(value)).toBe(true);
      }
    }
  }
  const summaryPath = testInfo.outputPath('gi-blue-wall-quality-summary.json');
  await writeFile(summaryPath, JSON.stringify(summaries, null, 2));
  await testInfo.attach('gi-blue-wall-quality-summary.json', {
    path: summaryPath, contentType: 'application/json',
  });
  console.log('GI_BLUE_WALL_QUALITY_COMPARISON ' + JSON.stringify(summaries));
  const baseline = summaries.find(summary => summary.quality === 'baseline')!;
  const refined = summaries.find(summary => summary.quality === 'dense-grid')!;
  if (baseline.meanAbsoluteRelativeError === null || refined.meanAbsoluteRelativeError === null) {
    throw new Error('GI_BLUE_WALL_REFERENCE_MISSING');
  }
  // A fixed, authored one-bounce scene: more local probes must improve the
  // independently traced reference, rather than just producing finite values.
  expect(refined.meanAbsoluteRelativeError).toBeLessThan(0.3);
  expect(refined.meanAbsoluteRelativeError).toBeLessThan(baseline.meanAbsoluteRelativeError * 0.65);
  for (const point of refined.points) for (const error of point.relativeError) {
    expect(Math.abs(error)).toBeLessThan(0.5);
  }
  expect(pageErrors).toEqual([]);
});
