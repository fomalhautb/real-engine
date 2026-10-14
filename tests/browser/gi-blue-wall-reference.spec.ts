import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runGiBlueWallReferenceFixture } from './fixtures/gi-blue-wall-reference';

type Result = Awaited<ReturnType<typeof runGiBlueWallReferenceFixture>>;

test('blue inner wall GI has an independent one-bounce radiance diagnostic', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });

  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-blue-wall-reference.ts') as {
      runGiBlueWallReferenceFixture: typeof runGiBlueWallReferenceFixture;
    };
    return module.runGiBlueWallReferenceFixture();
  }) as Result;

  const comparisons = result.samples.map((sample, index) => ({
    index, pixel: sample.pixel, hit: sample.cameraHit?.batch ?? null,
    reference: sample.oneBounce?.outgoingDiffuse ?? null,
    off: result.cases.off.hdr[index].slice(0, 3),
    single: result.cases.single.hdr[index].slice(0, 3),
    multi: result.cases.multi.hdr[index].slice(0, 3),
    sunOff: result.cases.sunOff.hdr[index].slice(0, 3),
    noWhiteForms: result.cases.noWhiteForms.hdr[index].slice(0, 3),
  }));
  const referenceErrors = result.reference.selectedIndices.map(index => {
    const sample = result.samples[index];
    const predicted = sample.oneBounce!.outgoingDiffuse;
    const measured = result.cases.single.hdr[index].slice(0, 3);
    return { index, predicted, measured,
      relative: measured.map((value, channel) =>
        (value - predicted[channel]) / Math.max(predicted[channel], 0.001)) };
  });
  const errors = referenceErrors.flatMap(entry => entry.relative);
  const meanRelativeError = errors.length
    ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null;
  const relativeErrorStdDev = errors.length && meanRelativeError !== null
    ? Math.sqrt(errors.reduce((sum, value) =>
      sum + (value - meanRelativeError) ** 2, 0) / errors.length) : null;
  const comparisonStatistics = { denominatorFloor: 0.001, referenceErrors,
    meanRelativeError, relativeErrorStdDev,
    meanAbsoluteRelativeError: errors.length
      ? errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length : null };
  // Save raw measurements even if a structural assertion below fails. These
  // errors are diagnostics until a reference tolerance is calibrated.
  const compact = { ...result, comparisonStatistics,
    cases: Object.fromEntries(Object.entries(result.cases)
      .map(([name, value]) => [name, { ...value, png: undefined }])) };
  const jsonPath = testInfo.outputPath('gi-blue-wall-reference.json');
  await writeFile(jsonPath, JSON.stringify(compact, null, 2));
  await testInfo.attach('gi-blue-wall-reference.json', { path: jsonPath, contentType: 'application/json' });
  for (const [name, value] of Object.entries(result.cases)) {
    if (value.png) {
      const pngPath = testInfo.outputPath(`gi-blue-wall-${name}.png`);
      await writeFile(pngPath, Buffer.from(value.png.split(',')[1], 'base64'));
      await testInfo.attach(`gi-blue-wall-${name}.png`, { path: pngPath, contentType: 'image/png' });
    }
  }
  console.log('GI_BLUE_WALL_REFERENCE ' + JSON.stringify({
    camera: result.camera, sunDirection: result.sunDirection, sunColor: result.sunColor,
    canvas: result.canvas, reference: result.reference, comparisons,
    referenceSamples: result.samples.filter(sample => sample.oneBounce)
      .map(sample => sample.oneBounce),
    historyUpdates: result.cases.multi.gi?.historyUpdates, comparisonStatistics,
  }));

  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(result.reference.selectedIndices.length).toBeGreaterThanOrEqual(3);
  expect(result.reference.selectedIndices.length).toBeLessThanOrEqual(6);
  expect(result.probe.invalidProbes).toBe(0);
  expect(result.cases.single.gi?.multiBounce).toBe(false);
  expect(result.cases.multi.gi?.multiBounce).toBe(true);
  expect(result.cases.multi.gi?.historyUpdates).toBe(12);
  for (const index of result.reference.selectedIndices) {
    const sample = result.samples[index];
    expect(sample.cameraHit?.batch).toBe('blue-wall');
    expect(sample.cameraHit?.entityId).toBe(4);
    expect(sample.cameraHit!.normal[0]).toBeLessThan(-0.9);
    expect(sample.cameraHit!.baseColor[2]).toBeCloseTo(0.72, 2);
    expect(sample.oneBounce).not.toBeNull();
    expect(sample.oneBounce!.samples).toBe(1024);
    for (const value of [
      ...sample.oneBounce!.meanIncoming, ...sample.oneBounce!.irradiance,
      ...sample.oneBounce!.outgoingDiffuse,
      ...sample.oneBounce!.fullBrdfOutgoingDiffuse,
    ]) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
    // The blue inner face points away from the sun. Its direct sun term is zero.
    for (const value of result.cases.off.hdr[index].slice(0, 3)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(0.01);
    }
  }
  for (const sceneCase of Object.values(result.cases)) {
    expect(sceneCase.hdr).toHaveLength(result.samples.length);
    for (const pixel of sceneCase.hdr) for (const value of pixel) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  }
});
