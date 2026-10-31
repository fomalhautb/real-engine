import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

test('large-scene page loads without an asset set and explains how to provide one', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/large-scene.html');
  await page.waitForFunction(() => window.__LARGE_SCENE__?.snapshot().status === 'idle');
  await expect(page.locator('#status')).toContainText('NO ASSET SET');
  await expect(page.locator('#error')).toBeHidden();
  expect(errors).toEqual([]);
});

test('large-scene page rejects a non-local asset set path explicitly', async ({ page }) => {
  await page.goto('/large-scene.html?set=https://example.com/asset-set.json');
  await page.waitForFunction(() => window.__LARGE_SCENE__?.snapshot().status === 'error');
  await expect(page.locator('#error')).toContainText('GLTF_PACKAGE_ROOT: assetSet');
});

// Local large assets only (CI has none): LARGE_SCENE_ASSET_SET is the server
// path of an asset-set.json written by scripts/write-asset-set.mjs, e.g.
// /artifacts/bistro-exterior-asset-set-mac-20261031/asset-set.json.
const assetSet = process.env.LARGE_SCENE_ASSET_SET;
test('large-scene walk renders a local asset set along its camera path and disposes everything', async ({ page }, testInfo) => {
  test.skip(!assetSet, 'LARGE_SCENE_ASSET_SET is not set (local large assets only)');
  test.setTimeout(900_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const frames = Number(process.env.LARGE_SCENE_FRAMES ?? 240);
  await page.goto(`/large-scene.html?set=${encodeURIComponent(assetSet!)}&frames=${frames}`);
  await page.waitForFunction(() => ['done', 'error'].includes(window.__LARGE_SCENE__?.snapshot().status ?? ''), undefined, { timeout: 840_000 });
  const result = await page.evaluate(() => window.__LARGE_SCENE__!.snapshot());
  await page.screenshot({ path: testInfo.outputPath('large-scene-final.png') });
  for (const debug of ['material', 'clusters', 'levels'] as const) {
    // Fixed first keyframe; 90 frames let streamed pages and textures settle.
    await page.evaluate(view => { const demo = window.__LARGE_SCENE__!; demo.setDebug(view); demo.showKeyframe(0); demo.run(90, false); }, debug);
    await page.waitForFunction(() => ['done', 'error'].includes(window.__LARGE_SCENE__?.snapshot().status ?? ''), undefined, { timeout: 300_000 });
    await page.screenshot({ path: testInfo.outputPath(`large-scene-keyframe0-${debug}.png`) });
  }
  const views = await page.evaluate(() => window.__LARGE_SCENE__!.snapshot());
  expect(views.failure).toBeNull();
  const afterDispose = await page.evaluate(() => { const r = window.__LARGE_SCENE__!.renderer!; window.__LARGE_SCENE__!.dispose(); return r.stats.ownedGpuBytes; });
  await writeFile(testInfo.outputPath('large-scene-report.json'), JSON.stringify({ ...result, afterDispose, errors }, null, 2));
  console.log('LARGE_SCENE ' + JSON.stringify({ status: result.status, failure: result.failure, frames: result.renderedFrames,
    adapter: result.info?.adapterInfo, scene: result.scene && { ...result.scene, bounds: undefined }, ownedGpuBytes: result.stats?.ownedGpuBytes,
    cpuSubmitMs: result.stats?.cpuSubmitMs, afterDispose }));
  expect(result.failure).toBeNull();
  expect(result.status).toBe('done');
  expect(result.renderedFrames).toBe(frames);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.info!.adapterInfo.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.info!.adapterInfo.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(errors).toEqual([]);
  expect(afterDispose).toBe(0);
});
