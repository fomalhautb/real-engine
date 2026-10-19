import { test, expect } from '@playwright/test';
import type { runFogQualityFixture } from './fixtures/volumetric-fog-quality';

test.use({ deviceScaleFactor: 1 });
function rms(actual: number[][], expected: number[][]) {
  return Math.sqrt(actual.reduce((sum, pixel, i) => sum + expected[i].reduce((s, value, c) => s + (pixel[c] - value) ** 2, 0), 0) / (expected.length * 3));
}
test('temporal fog improves static and moving slat-shadow integration against analytic intervals', async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-quality.ts') as { runFogQualityFixture: typeof runFogQualityFixture };
    return fixture.runFogQualityFixture();
  });
  const staticCentered = rms(result.centered, result.staticReference), staticTemporal = rms(result.accumulated, result.staticReference);
  const motionCentered = rms(result.moving.flatMap(x => x.centered), result.moving.flatMap(x => x.reference));
  const motionTemporal = rms(result.moving.flatMap(x => x.temporal), result.moving.flatMap(x => x.reference));
  const residualDelta = (mode: 'centered' | 'temporal') => result.moving.slice(1).flatMap((frame, i) => frame[mode].map((pixel, p) =>
    frame.reference[p].map((value, c) => pixel[c] - value - (result.moving[i][mode][p][c] - result.moving[i].reference[p][c]))));
  const centeredDelta = residualDelta('centered'), temporalDelta = residualDelta('temporal');
  const motionCenteredVariation = rms(centeredDelta, centeredDelta.map(() => [0, 0, 0]));
  const motionTemporalVariation = rms(temporalDelta, temporalDelta.map(() => [0, 0, 0]));
  console.log('FOG_TEMPORAL_QUALITY ' + JSON.stringify({ staticCentered, staticTemporal, motionCentered, motionTemporal,
    motionCenteredVariation, motionTemporalVariation, adapter: result.info.adapterInfo }));
  expect(staticCentered).toBeGreaterThan(0.001);
  expect(staticTemporal).toBeLessThan(staticCentered * 0.6);
  expect(motionTemporal).toBeLessThan(motionCentered * 0.8);
  expect(motionTemporalVariation).toBeLessThan(motionCenteredVariation * 0.8);
  expect(errors).toEqual([]); expect(result.errors).toEqual([]); expect(result.released).toBe(0);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.info.adapterInfo.vendor).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
});
