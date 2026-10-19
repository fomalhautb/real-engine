import { test, expect } from '@playwright/test';
import type { runFogShadowFixture } from './fixtures/volumetric-fog-shadow';

test.use({ deviceScaleFactor: 1 });
test('local media add optical coefficients and share masked Morph/skin sun shadows with GI enabled', async ({ page }) => {
  test.setTimeout(180_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'volumetric-fog-shadow.ts') as { runFogShadowFixture: typeof runFogShadowFixture };
    return module.runFogShadowFixture();
  });
  let maximumLocalError = 0, maximumHeightError = 0;
  for (let i = 0; i < result.pixels.length; i++) {
    for (let c = 0; c < 3; c++) {
      maximumLocalError = Math.max(maximumLocalError, Math.abs(result.local[i][c] - result.reference[i][c]));
      maximumHeightError = Math.max(maximumHeightError, Math.abs(result.heightFog[i][c] - result.heightReference[i][c]));
      expect(result.lit[i][c] - result.blocked[i][c]).toBeGreaterThan(0.04);
      expect(result.blocked[i][c]).toBeLessThan(result.vacuum[i][c]);
      for (const recovered of [result.morphed, result.skinned, result.deleted]) expect(Math.abs(recovered[i][c] - result.lit[i][c])).toBeLessThan(0.001);
    }
    expect(result.surface[i]).toEqual(result.vacuum[i]);
  }
  expect(maximumLocalError).toBeLessThan(0.012);
  expect(maximumHeightError).toBeLessThan(0.001);
  // A camera-only medium does not silently feed fogged surface radiance back into GI.
  expect(result.mediaGi.probes).toEqual(result.firstGi.probes);
  expect(result.diagnostics.passes).toEqual(expect.arrayContaining(['fog-inject', 'fog-integrate', 'fog-compose']));
  expect(result.errors).toEqual([]); expect(errors).toEqual([]); expect(result.releasedBytes).toBe(0);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.info.adapterInfo.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  console.log('FOG_SHADOW_DIAGNOSTIC ' + JSON.stringify({ maximumLocalError, maximumHeightError, lit: result.lit[4], blocked: result.blocked[4], fog: result.diagnostics.volumetricFog, adapter: result.info.adapterInfo }));
});
