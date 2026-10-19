import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runTextureResidencyFixture } from './fixtures/texture-residency';

test.use({ deviceScaleFactor: 1 });
test('bounded texture uploads publish every shared material atomically and preserve pending replacements', async ({ page }, info) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'texture-residency.ts') as {
      runTextureResidencyFixture: typeof runTextureResidencyFixture };
    return module.runTextureResidencyFixture();
  });
  await writeFile(info.outputPath('texture-residency.json'), JSON.stringify(result, null, 2));
  for (const sample of [result.initial, result.beforeCancel, result.upgraded, result.holes, result.restored]) {
    await writeFile(info.outputPath(sample.name + '.png'), Buffer.from(sample.png.split(',')[1], 'base64'));
  }
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  const near = (actual: readonly number[], expected: readonly number[], label: string) => {
    expected.forEach((value, c) => expect(Math.abs(actual[c] - value), `${label}/${c}`).toBeLessThan(0.004));
  };
  const coarse = [[8, 32, 192], [16, 16, 192]].map(rgb => rgb.map(c => c / 255));
  const fine = [[96, 32, 16], [16, 16, 192]].map(rgb => rgb.map(c => c / 255));
  for (const sample of [result.initial, result.beforeCancel, result.afterCancel, ...result.partial, result.restored, result.upgraded]) {
    for (let side = 0; side < 2; side++) {
      const expected = sample.name === 'fine' ? fine[side] : coarse[side];
      near(sample.hdr[side], expected, sample.name + ' HDR');
      expect(sample.hits[side], sample.name).not.toBeNull();
      near(sample.hits[side]!.baseColor, expected, sample.name + ' query');
      expect(sample.depth[side].surface).toBeLessThan(1); expect(sample.depth[side].shadow).toBeLessThan(1);
    }
    expect(sample.variants).toBe(result.initial.variants);
  }
  expect(result.cancelBytes).toBe(32);
  expect(result.ticks.every(bytes => bytes > 0 && bytes <= 32)).toBe(true);
  expect(result.ticks.reduce((sum, bytes) => sum + bytes, 0)).toBe(340);
  expect(result.partial.length).toBeGreaterThan(1);
  expect(result.partialRejections).toBe(result.partial.length);
  for (const sample of result.partial) expect(sample.revision).toBe(result.initial.revision);
  expect(result.beforeCancel.images.imageBytes).toBe(360);
  expect(result.afterCancel.images.imageBytes).toBe(20);
  expect(result.replacement).toEqual({ materials: 3, slots: 3 });
  expect(result.revisionAfter).toBe(result.revisionBefore + 1);
  expect(result.upgraded.revision).toBe(result.revisionAfter + 1); // Pending setMaterial publishes once.
  expect(result.upgraded.images).toEqual({ materials: 2, images: 1, imageBytes: 340 });
  expect(result.upgraded.uploads).toEqual({ active: 0, pending: 0, pendingImageBytes: 0 });
  expect(result.unused).toEqual({ materials: 0, slots: 0 });
  expect(result.budgetError).toContain('MATERIAL_TEXTURE_BUDGET'); expect(result.budgetUnchanged).toBe(true);
  expect(result.maskTicks.every(bytes => bytes > 0 && bytes <= 8)).toBe(true);
  expect(result.maskTicks.reduce((sum, bytes) => sum + bytes, 0)).toBe(20);
  expect(result.maskReplacement).toEqual({ materials: 2, slots: 2 });
  expect(result.holes.hits).toEqual([null, null]);
  expect(result.holes.depth).toEqual([{ surface: 1, shadow: 1 }, { surface: 1, shadow: 1 }]);
  expect(result.restoredReplacement).toEqual({ materials: 2, slots: 2 });
  expect(result.restored.images).toEqual({ materials: 2, images: 1, imageBytes: 20 });
  expect(result.afterDispose).toBe(0); expect(result.errors).toEqual([]); expect(errors).toEqual([]);
});
