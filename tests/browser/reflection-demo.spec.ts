import { expect, test } from '@playwright/test';

test.use({ deviceScaleFactor: 1 });

test('reflection gallery exposes offscreen geometry, material and media controls', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  // Bound the software path without presenting it as native performance evidence.
  if (process.env.PLAYWRIGHT_GPU !== 'hardware') await page.setViewportSize({ width: 800, height: 600 });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/reflection.html?benchmark=1');
  await page.waitForFunction(() => (window.__REFLECTION_DEMO__?.snapshot().stats?.frames ?? 0) > 0
    || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const initial = await page.evaluate(() => window.__REFLECTION_DEMO__!.snapshot());
  expect(initial.controls.paused).toBe(true);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(initial.adapter?.adapterInfo.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(initial.adapter?.adapterInfo.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(initial.reflections?.grid?.anchorCount).toBeLessThanOrEqual(16_384);
  expect(initial.reflections?.grid?.rayCount).toBeLessThanOrEqual(65_536);
  expect(initial.reflections?.ownedGpuBytes).toBeLessThanOrEqual(32 * 1024 * 1024);
  expect(initial.errors).toEqual([]);
  const on = await page.locator('#viewport').screenshot({ path: testInfo.outputPath('reflection-on.png') });
  await page.locator('#reflections-enabled').uncheck();
  const off = await page.locator('#viewport').screenshot({ path: testInfo.outputPath('reflection-off.png') });
  expect(on.equals(off)).toBe(false);
  await page.locator('#reflections-enabled').check();
  await page.locator('#fog-enabled').uncheck();
  await page.locator('#viewport').screenshot({ path: testInfo.outputPath('reflection-no-fog.png') });
  await page.locator('#roughness').fill('0.345');
  await page.locator('#roughness').dispatchEvent('input');
  const rough = await page.evaluate(() => window.__REFLECTION_DEMO__!.snapshot());
  expect(rough.controls.roughness).toBeCloseTo(0.345);
  await page.locator('#viewport').screenshot({ path: testInfo.outputPath('reflection-rough.png') });
  await page.evaluate(() => window.__REFLECTION_DEMO__!.step(0.5));
  const stepped = await page.evaluate(() => window.__REFLECTION_DEMO__!.snapshot());
  expect(stepped.controls.phase).toBeCloseTo(initial.controls.phase + 0.5);
  expect(stepped.errors).toEqual([]);
  expect(errors).toEqual([]);
  await testInfo.attach('reflection-gallery.json', { body: JSON.stringify({ initial, rough, stepped }, null, 2), contentType: 'application/json' });
  await page.evaluate(() => window.__REFLECTION_DEMO__!.dispose());
});
