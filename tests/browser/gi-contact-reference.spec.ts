import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runGiContactReferenceFixture } from './fixtures/gi-contact-reference';

type Result = Awaited<ReturnType<typeof runGiContactReferenceFixture>>;

test('right block near-contact GI has a bias-stable independent path reference', async ({ page }, testInfo) => {
  test.setTimeout(360_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-contact-reference.ts') as {
      runGiContactReferenceFixture: typeof runGiContactReferenceFixture;
    };
    return module.runGiContactReferenceFixture();
  }) as Result;
  const profile = result.samples.map((sample, index) => {
    const validPixel = sample.pixelReference !== null;
    return { distance: sample.distance, requestedWorld: sample.requestedWorld,
      referenceWorld: sample.referenceWorld, pixel: sample.pixel,
      cameraHit: sample.cameraHit?.batch ?? null, pixelDistanceFromFront: sample.pixelDistanceFromFront,
      pixelWorldOffsetMillimeters: sample.pixelWorldOffsetMillimeters,
      validPixel,
      physicalReferenceOne: sample.reference.oneBounce.mean,
      physicalReferenceTwelve: sample.reference.twelveBounce.mean,
      pixelReferenceWorld: sample.pixelReference?.world ?? null,
      pixelReferenceOne: sample.pixelReference?.oneBounce.mean ?? null,
      pixelReferenceTwelve: sample.pixelReference?.twelveBounce.mean ?? null,
      pixelReference95: sample.pixelReference?.twelveBounce.approximate95PercentInterval ?? null,
      measuredSingleIndirect: validPixel ? result.cases.single.hdr[index].slice(0, 3)
        .map((value, c) => value - result.cases.off.hdr[index][c]) : null,
      measuredMultiIndirect: validPixel ? result.cases.multi.hdr[index].slice(0, 3)
        .map((value, c) => value - result.cases.off.hdr[index][c]) : null,
      blockFirstHitCount: sample.reference.firstHits['white-forms'] ?? 0,
      blockFirstHitContribution: sample.reference.firstHitSources['white-forms'] ?? [0, 0, 0],
    };
  });
  const report = { ...result, profile,
    cases: Object.fromEntries(Object.entries(result.cases)
      .map(([name, value]) => [name, { ...value, png: undefined }])) };
  const jsonPath = testInfo.outputPath('gi-contact-reference.json');
  await writeFile(jsonPath, JSON.stringify(report, null, 2));
  await testInfo.attach('gi-contact-reference.json', { path: jsonPath, contentType: 'application/json' });
  const png = result.cases.multi.png;
  if (png) {
    const pngPath = testInfo.outputPath('gi-contact-current-multi.png');
    await writeFile(pngPath, Buffer.from(png.split(',')[1], 'base64'));
    await testInfo.attach('gi-contact-current-multi.png', { path: pngPath, contentType: 'image/png' });
  }
  console.log('GI_CONTACT_REFERENCE ' + JSON.stringify({ profile,
    biasChecks: result.biasChecks.map(check => ({ bias: check.bias,
      oneBounce: check.estimates.map(estimate => estimate.oneBounce) })) }));

  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(result.grid.counts).toEqual([6, 6, 6]);
  expect(result.grid.raysPerProbe).toBe(128);
  expect(result.referenceConfig.primaryBias).toBe(1e-4);
  expect(result.referenceConfig.maxLiveRayBatch).toBeLessThanOrEqual(8192);
  expect(result.referenceConfig.pathsPerPoint).toBeGreaterThanOrEqual(4096);
  expect(result.referenceConfig.bounces).toBe(12);
  expect(result.samples.map(sample => sample.distance)).toEqual([0.01, 0.02, 0.05, 0.1, 0.2, 0.4]);
  expect(result.cases.single.gi?.multiBounce).toBe(false);
  expect(result.cases.multi.gi?.multiBounce).toBe(true);
  expect(profile.some(sample => sample.validPixel)).toBe(true);
  expect(profile[0].validPixel).toBe(false); // At 1 cm the nearest camera pixel hits the block.
  expect(profile.some(sample => sample.blockFirstHitCount > 0 &&
    sample.blockFirstHitContribution.some(value => value > 0))).toBe(true);
  for (const sample of result.samples) {
    expect(sample.referenceWorld[0]).toBeCloseTo(sample.requestedWorld[0], 4);
    expect(sample.referenceWorld[1]).toBeCloseTo(0, 4);
    expect(sample.referenceWorld[2]).toBeCloseTo(sample.requestedWorld[2], 4);
    if (sample.pixelReference) {
      expect(sample.cameraHit?.batch).toBe('room-neutral');
      expect(sample.cameraHit?.entityId).toBe(1);
      expect(sample.pixelDistanceFromFront).toBeGreaterThan(0);
      for (let c = 0; c < 3; c++) {
        expect(sample.pixelReference.world[c]).toBeCloseTo(sample.cameraHit!.position[c], 6);
      }
      expect(sample.pixelWorldOffsetMillimeters).toBeGreaterThanOrEqual(0);
    } else {
      expect(profile.find(entry => entry.distance === sample.distance)?.measuredMultiIndirect).toBeNull();
    }
    expect(sample.reference.firstHits['room-neutral'] ?? 0).toBe(0);
    for (const estimate of [sample.reference.oneBounce, sample.reference.twelveBounce]) {
      for (const value of [...estimate.mean, ...estimate.sampleVariance,
        ...estimate.standardError, ...estimate.approximate95PercentInterval.flat()]) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeGreaterThanOrEqual(0);
      }
    }
  }
  for (let point = 0; point < 2; point++) {
    const low = result.biasChecks[0].estimates[point].oneBounce;
    const high = result.biasChecks[1].estimates[point].oneBounce;
    for (let c = 0; c < 3; c++) {
      const difference = Math.abs(low.mean[c] - high.mean[c]);
      // Both offsets trace identical directions and path IDs. Their paired
      // difference tests the numerical origin offset, not Monte Carlo noise.
      expect(difference).toBeLessThanOrEqual(0.0002 +
        0.01 * Math.max(low.mean[c], high.mean[c]));
    }
  }
  for (const sceneCase of Object.values(result.cases)) {
    expect(sceneCase.hdr).toHaveLength(result.samples.length);
    for (const pixel of sceneCase.hdr) for (const value of pixel) expect(Number.isFinite(value)).toBe(true);
  }
});
