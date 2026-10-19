import { writeFile } from 'node:fs/promises';
import { test, expect } from '@playwright/test';

test('fog colonnade exposes real HDR media, repeatable controls and bounded resources', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  // Software adapters prove controls/history and resource correctness at a
  // bounded resolution; the full-size visual capture remains on native GPUs.
  if (process.env.PLAYWRIGHT_GPU !== 'hardware') await page.setViewportSize({ width: 640, height: 600 });
  await page.goto('/fog.html?benchmark=1');
  await page.waitForFunction(() => window.__FOG_DEMO__?.snapshot().stats?.frames || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const capture = () => page.evaluate(() => {
    window.__FOG_DEMO__!.resetHistory();
    // The reset frame also settles object history. Accumulate an identical
    // sequence for every frozen-view capture instead of comparing unlike ages.
    for (let frame = 0; frame < 33; frame++) window.__FOG_DEMO__!.render();
    const canvas = document.getElementById('viewport') as HTMLCanvasElement;
    return { png: canvas.toDataURL('image/png'), snapshot: window.__FOG_DEMO__!.snapshot() };
  });
  const on = await capture();
  await writeFile(testInfo.outputPath('fog-colonnade-on.png'), Buffer.from(on.png.split(',')[1], 'base64'));
  await page.locator('#fog-enabled').uncheck();
  const off = await capture();
  await writeFile(testInfo.outputPath('fog-colonnade-off.png'), Buffer.from(off.png.split(',')[1], 'base64'));
  expect(on.png).not.toEqual(off.png);
  expect(on.snapshot.fog!.grid!.bytes).toBeLessThanOrEqual(128 * 1024 * 1024);
  expect(on.snapshot.stats!.ownedGpuBytes).toBe(off.snapshot.stats!.ownedGpuBytes);
  await page.locator('#fog-enabled').check();
  const restored = await capture();
  expect(restored.png).toEqual(on.png);
  await page.locator('#temporal-enabled').uncheck();
  const centered = await capture();
  await writeFile(testInfo.outputPath('fog-colonnade-centered.png'), Buffer.from(centered.png.split(',')[1], 'base64'));
  expect(centered.png).not.toEqual(on.png);
  expect(centered.snapshot.fog!.grid).toEqual(on.snapshot.fog!.grid);
  await page.locator('#temporal-enabled').check();
  await page.locator('#step').click();
  const moved = await capture();
  expect(moved.snapshot.camera.phase).toBeGreaterThan(on.snapshot.camera.phase);
  expect(moved.snapshot.errors).toEqual([]); expect(errors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(on.snapshot.adapter!.adapterInfo.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  console.log('FOG_DEMO_DIAGNOSTIC ' + JSON.stringify({ stats: moved.snapshot.stats, fog: moved.snapshot.fog, adapter: moved.snapshot.adapter!.adapterInfo }));
  await page.evaluate(() => window.__FOG_DEMO__!.dispose());
});
