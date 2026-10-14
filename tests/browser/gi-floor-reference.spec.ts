import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runGiFloorReferenceFixture } from './fixtures/gi-floor-reference';

type Result = Awaited<ReturnType<typeof runGiFloorReferenceFixture>>;

test('floor color blotches have a bounded independent diffuse transport reference', async ({ page }, testInfo) => {
  test.setTimeout(900_000);
  const renderReferenceImage = process.env.GI_FLOOR_REFERENCE_IMAGE === '1';
  const raysPerProbeOverride = process.env.GI_FLOOR_RAYS === '256' ? 256 as const : undefined;
  const gridMode = process.env.GI_FLOOR_GRID === '8' ? '8' as const : undefined;
  const emissiveSamplesOverride = process.env.GI_FLOOR_EMISSIVE_SAMPLES === '0'
    ? 0 as const : undefined;
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });

  const result = await page.evaluate(async requested => {
    const module = await import('/tests/browser/fixtures/' + 'gi-floor-reference.ts') as {
      runGiFloorReferenceFixture: typeof runGiFloorReferenceFixture;
    };
    return module.runGiFloorReferenceFixture(requested.renderReferenceImage,
      requested.raysPerProbeOverride, requested.gridMode, requested.emissiveSamplesOverride);
  }, { renderReferenceImage, raysPerProbeOverride, gridMode, emissiveSamplesOverride }) as Result;

  const comparisons = result.samples.map((sample, index) => {
    const off = result.cases.off.hdr[index].slice(0, 3);
    const single = result.cases.single.hdr[index].slice(0, 3);
    const multi = result.cases.multi.hdr[index].slice(0, 3);
    const sunOffMulti = result.cases.sunOffMulti.hdr[index].slice(0, 3);
    const lampOffMulti = result.cases.lampOffMulti.hdr[index].slice(0, 3);
    const singleIndirect = single.map((value, channel) => value - off[channel]);
    const multiIndirect = multi.map((value, channel) => value - off[channel]);
    const oneBounce = sample.reference?.oneBounce.mean ?? null;
    const truncatedFourBounce = sample.reference?.truncatedFourBounce.mean ?? null;
    const truncatedEightBounce = sample.reference?.truncatedEightBounce.mean ?? null;
    const truncatedTwelveBounce = sample.reference?.truncatedTwelveBounce.mean ?? null;
    const lampOffTwelveBounce = sample.lampOffReference?.truncatedTwelveBounce.mean ?? null;
    return { index, world: sample.world, pixel: sample.pixel,
      cameraHit: sample.cameraHit?.batch ?? null, off, single, multi, sunOffMulti,
      lampOffMulti, currentMinusLampOff: multi.map((value, channel) =>
        value - lampOffMulti[channel]),
      singleIndirect, multiIndirect, oneBounce, truncatedFourBounce,
      truncatedEightBounce, truncatedTwelveBounce, lampOffTwelveBounce,
      referenceTwelveBounceLampContribution: lampOffTwelveBounce?.map((value, channel) =>
        truncatedTwelveBounce![channel] - value) ?? null,
      tailFourToEight: truncatedEightBounce?.map((value, channel) =>
        value - truncatedFourBounce![channel]) ?? null,
      tailEightToTwelve: truncatedTwelveBounce?.map((value, channel) =>
        value - truncatedEightBounce![channel]) ?? null,
      firstHitCategories: sample.reference?.firstHitCategories ?? null,
      firstHitContributions: sample.reference?.firstHitContributions ?? null,
      allHitCategories: sample.reference?.allHitCategories ?? null,
      allSourceContributions: sample.reference?.allSourceContributions ?? null,
      emitterHitCountsByBounce: sample.reference?.emitterHitCountsByBounce ?? null,
      emitterContributionsByBounce: sample.reference?.emitterContributionsByBounce ?? null,
      oneBounceRelativeError: oneBounce?.map((value, channel) =>
        (singleIndirect[channel] - value) / Math.max(value, 0.001)) ?? null,
      fourBounceRelativeError: truncatedFourBounce?.map((value, channel) =>
        (multiIndirect[channel] - value) / Math.max(value, 0.001)) ?? null,
      eightBounceRelativeError: truncatedEightBounce?.map((value, channel) =>
        (multiIndirect[channel] - value) / Math.max(value, 0.001)) ?? null,
      twelveBounceRelativeError: truncatedTwelveBounce?.map((value, channel) =>
        (multiIndirect[channel] - value) / Math.max(value, 0.001)) ?? null };
  });
  const means = (key: 'oneBounceRelativeError' | 'fourBounceRelativeError'
    | 'eightBounceRelativeError' | 'twelveBounceRelativeError') => {
    const values = comparisons.flatMap(comparison => comparison[key] ?? []);
    return { sampleChannels: values.length,
      meanSignedRelativeError: values.reduce((sum, value) => sum + value, 0) / values.length,
      meanAbsoluteRelativeError: values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length };
  };
  const summary = { camera: result.camera, sunDirection: result.sunDirection,
    sunColor: result.sunColor, grid: result.grid, reference: result.reference,
    comparisons, oneBounceError: means('oneBounceRelativeError'),
    fourBounceError: means('fourBounceRelativeError'),
    eightBounceError: means('eightBounceRelativeError'),
    twelveBounceError: means('twelveBounceRelativeError'),
    directOnlyPeak: Math.max(...result.cases.off.hdr.flatMap(pixel => pixel.slice(0, 3))),
    probeState: { single: result.probes.single, multi: { frame: result.probes.multi.frame,
      invalid: result.probes.multi.invalid }, sunOffMulti: result.probes.sunOffMulti,
      lampOffMulti: { frame: result.probes.lampOffMulti.frame,
        invalid: result.probes.lampOffMulti.invalid } },
    lampAblation: result.lampAblation,
    referenceImage: result.referenceImage ? {
      ...result.referenceImage, actualGiCropPng: undefined, referencePng: undefined } : null };
  const compact = { ...result, summary,
    referenceImage: result.referenceImage ? {
      ...result.referenceImage, actualGiCropPng: undefined, referencePng: undefined } : null,
    cases: Object.fromEntries(Object.entries(result.cases)
    .map(([name, value]) => [name, { ...value, png: undefined }])) };
  const jsonPath = testInfo.outputPath('gi-floor-reference.json');
  await writeFile(jsonPath, JSON.stringify(compact, null, 2));
  await testInfo.attach('gi-floor-reference.json', { path: jsonPath, contentType: 'application/json' });
  for (const [name, value] of Object.entries(result.cases)) {
    const pngPath = testInfo.outputPath(`gi-floor-${name}.png`);
    await writeFile(pngPath, Buffer.from(value.png.split(',')[1], 'base64'));
    await testInfo.attach(`gi-floor-${name}.png`, { path: pngPath, contentType: 'image/png' });
  }
  if (result.referenceImage) {
    for (const [name, png] of [
      ['actual-crop', result.referenceImage.actualGiCropPng],
      ['independent-eight-bounce-floor-crop', result.referenceImage.referencePng],
    ] as const) {
      const path = testInfo.outputPath(`gi-floor-${name}.png`);
      await writeFile(path, Buffer.from(png.split(',')[1], 'base64'));
      await testInfo.attach(`gi-floor-${name}.png`, { path, contentType: 'image/png' });
    }
  }
  console.log('GI_FLOOR_REFERENCE ' + JSON.stringify(summary));

  expect(pageErrors).toEqual([]);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(result.samples).toHaveLength(20);
  expect(result.reference.selectedIndices).toHaveLength(6);
  expect(result.probes.single.invalid).toBe(0);
  expect(result.probes.multi.invalid).toBe(0);
  expect(result.probes.sunOffMulti.invalid).toBe(0);
  expect(result.probes.lampOffMulti.invalid).toBe(0);
  if (result.referenceImage) {
    expect(result.referenceImage.width).toBe(155);
    expect(result.referenceImage.height).toBe(75);
    expect(result.referenceImage.floorPixels).toBeGreaterThan(0);
    expect(result.referenceImage.maxWorldQueryBatch).toBeLessThanOrEqual(27648);
    expect(result.referenceImage.samplesPerPixel).toBe(128);
    expect(result.referenceImage.tilePixels).toBe(64);
  }
  expect(result.cases.single.gi?.multiBounce).toBe(false);
  expect(result.cases.multi.gi?.historyUpdates).toBe(12);
  expect(result.cases.sunOffMulti.gi?.historyUpdates).toBe(12);
  expect(result.cases.lampOffMulti.gi?.historyUpdates).toBe(12);
  expect(result.lampAblation.stripHit?.batch).toBe('emissive-strip');
  expect(result.lampAblation.stripHit?.shading).toBe('unlit');
  for (const value of [...result.lampAblation.stripHit!.baseColor,
    ...result.lampAblation.stripHit!.emissive]) expect(value).toBe(0);
  for (const sample of result.samples) {
    expect(sample.pixel.x).toBeGreaterThanOrEqual(0);
    expect(sample.pixel.x).toBeLessThan(result.canvas.width);
    expect(sample.pixel.y).toBeGreaterThanOrEqual(0);
    expect(sample.pixel.y).toBeLessThan(result.canvas.height);
    expect(sample.cameraHit?.batch).toBe('room-neutral');
    expect(sample.cameraHit?.entityId).toBe(1);
    expect(sample.cameraHit!.normal[1]).toBeGreaterThan(0.9);
  }
  for (const index of result.reference.selectedIndices) {
    const reference = result.samples[index].reference!;
    const lampOffReference = result.samples[index].lampOffReference!;
    expect(reference.pathCount).toBe(4096);
    expect(reference.maxBounces).toBe(12);
    expect(lampOffReference.pathCount).toBe(4096);
    expect(lampOffReference.maxBounces).toBe(12);
    for (const pointReference of [reference, lampOffReference]) {
      expect(pointReference.emitterHitCountsByBounce).toHaveLength(12);
      expect(pointReference.emitterContributionsByBounce).toHaveLength(12);
      const emitterCount = pointReference.emitterHitCountsByBounce.reduce((sum, value) => sum + value, 0);
      expect(emitterCount).toBe(pointReference.allHitCategories['emissive-strip'] ?? 0);
      for (const value of Object.values(pointReference.allSourceContributions).flat()) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
      }
      for (let channel = 0; channel < 3; channel++) {
        const total = Object.values(pointReference.allSourceContributions)
          .reduce((sum, rgb) => sum + rgb[channel], 0);
        expect(Math.abs(total - pointReference.truncatedTwelveBounce.mean[channel]))
          .toBeLessThan(0.0001);
      }
    }
    for (const estimate of [reference.oneBounce, reference.truncatedFourBounce,
      reference.truncatedEightBounce, reference.truncatedTwelveBounce,
      lampOffReference.oneBounce, lampOffReference.truncatedFourBounce,
      lampOffReference.truncatedEightBounce, lampOffReference.truncatedTwelveBounce]) {
      for (const value of [...estimate.mean, ...estimate.sampleVariance,
        ...estimate.standardError, ...estimate.approximate95PercentInterval.flat()]) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
      }
    }
  }
  for (const sceneCase of Object.values(result.cases)) {
    expect(sceneCase.hdr).toHaveLength(20);
    for (const pixel of sceneCase.hdr) for (const value of pixel) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  }
});
