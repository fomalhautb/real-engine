import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runFinalGatherMotionFixture } from './fixtures/gi-final-gather-motion';

for (const pathSegments of [1, 2]) test(`native final gather ${pathSegments} segments history reduces blue-wall motion noise against per-frame reset`, async ({ page }, testInfo) => {
  test.skip(process.env.PLAYWRIGHT_GPU !== 'hardware',
    'Dense motion quality comparison requires native GPU execution.');
  test.setTimeout(600_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 960, height: 540 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async (pathSegments: number) => {
    const module = await import('/tests/browser/fixtures/' + 'gi-final-gather-motion.ts') as {
      runFinalGatherMotionFixture: typeof runFinalGatherMotionFixture;
    };
    return module.runFinalGatherMotionFixture(pathSegments);
  }, pathSegments);
  const report = { pathSegments: result.pathSegments, carried: { ...result.carried, png: undefined },
    baseline: { ...result.baseline, png: undefined } };
  const jsonPath = testInfo.outputPath('gi-final-gather-motion.json');
  await writeFile(jsonPath, JSON.stringify(report, null, 2));
  await testInfo.attach('gi-final-gather-motion.json', { path: jsonPath, contentType: 'application/json' });
  for (const run of [result.carried, result.baseline]) {
    const path = testInfo.outputPath(`gi-final-gather-motion-${run.mode}.png`);
    await writeFile(path, Buffer.from(run.png.split(',')[1], 'base64'));
    await testInfo.attach(`gi-final-gather-motion-${run.mode}.png`, { path, contentType: 'image/png' });
  }
  console.log('FINAL_GATHER_MOTION ' + JSON.stringify(report));
  expect(pageErrors).toEqual([]);
  for (const run of [result.carried, result.baseline]) {
    expect(run.errors).toEqual([]);
    expect(run.afterDispose).toBe(0);
    expect(run.ownedGpuBytesLast).toBe(run.ownedGpuBytesFirst);
    expect(run.motionFrames).toBe(48);
    expect(run.validPairs).toBeGreaterThan(100);
    expect(Number.isFinite(run.changeRms)).toBe(true);
    expect(run.changeRms).toBeGreaterThan(0);
  }
  expect(result.carried.validPairs).toBe(result.baseline.validPairs);
  expect(result.carried.validSamples).toBe(result.baseline.validSamples);
  expect(result.carried.sampleHash).toBe(result.baseline.sampleHash);
  expect(result.carried.changeRms).toBeLessThan(result.baseline.changeRms * 0.5);
  for (let channel = 0; channel < 3; channel++) {
    const carried = result.carried.meanRgb[channel], baseline = result.baseline.meanRgb[channel];
    expect(Math.abs(carried - baseline)).toBeLessThan(0.05 * Math.max(baseline, 0.001));
  }
});
