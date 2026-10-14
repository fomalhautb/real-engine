import { expect, test } from '@playwright/test';

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status !== testInfo.expectedStatus && !page.isClosed()) {
    const diagnostic = await page.evaluate(() => ({
      error: document.getElementById('error')?.textContent,
      snapshot: window.__REAL_ENGINE__?.snapshot(),
    })).catch(error => ({ error: String(error) }));
    console.log('ENGINE_FAILURE_DIAGNOSTIC ' + JSON.stringify(diagnostic));
  }
});

test('shared deformation, lifetime, physics controls, resize, and GPU validation', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=12');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.waitForFunction(() => (window.__REAL_ENGINE__?.snapshot().stats?.frames ?? 0) >= 5 || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const first = await page.evaluate(() => window.__REAL_ENGINE__!.snapshot());
  expect(first.stats!.drawCalls).toBe(14);
  expect(first.physics.bodies).toBe(6);
  expect(first.errors).toEqual([]);
  expect(first.renderGraph!.passes).toEqual(['deformation', 'shadow', 'surface', 'resolve']);
  expect(first.renderGraph!.deformationVariants).toBe(5);
  expect(first.renderGraph!.bounds.length).toBe(23);
  expect(first.renderGraph!.materialVariants).toBe(4);
  expect(first.renderGraph!.materials.images).toBe(2);
  const result = await page.evaluate(async () => {
    const engine = window.__REAL_ENGINE__!;
    engine.pause(true);
    engine.step(1 / 60);
    return engine.verify();
  });
  expect(result.vertices).toBeGreaterThan(100);
  expect(result.passed).toBe(true);
  expect(result.maxNormalError).toBeLessThan(0.0002);
  expect(result.boundsViolations).toBe(0);
  await page.waitForFunction(() => window.__REAL_ENGINE__!.snapshot().stats!.deformationDispatches === 0);
  const cached = await page.evaluate(() => window.__REAL_ENGINE__!.snapshot());
  expect(cached.stats!.uploadedBytes).toBe(256); // Only the camera/light uniform changes.
  expect(cached.sceneChanges!.lastFrame).toEqual({ create: 0, update: 0, remove: 0 });
  expect((await page.evaluate(() => window.__REAL_ENGINE__!.verify())).passed).toBe(true);
  await page.getByRole('button', { name: '添加刚体', exact: true }).click();
  expect(await page.evaluate(() => window.__REAL_ENGINE__!.snapshot().physics.bodies)).toBe(7);
  await page.getByRole('button', { name: '移除刚体', exact: true }).click();
  expect(await page.evaluate(() => window.__REAL_ENGINE__!.snapshot().physics.bodies)).toBe(6);
  const reset = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.reset();
    return window.__REAL_ENGINE__!.verify();
  });
  expect(reset.passed).toBe(true);
  const hit = await page.evaluate(() => window.__REAL_ENGINE__!.raycast([0, 10, -2.5], [0, -1, 0], 20));
  expect(typeof hit?.entityId).toBe('number');
  expect(hit!.normal[1]).toBeGreaterThan(0.99);
  await page.setViewportSize({ width: 1100, height: 820 });
  await page.waitForFunction(() => window.__REAL_ENGINE__!.snapshot().stats!.width < 1000);
  const afterResize = await page.evaluate(() => window.__REAL_ENGINE__!.verify());
  expect(afterResize.passed).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.waitForFunction(() => window.__REAL_ENGINE__!.snapshot().stats!.width > 1000);
  const final = await page.evaluate(() => window.__REAL_ENGINE__!.snapshot());
  expect(final.errors).toEqual([]);
  expect(final.stats!.ownedGpuBytes).toBe(first.stats!.ownedGpuBytes);
  expect(pageErrors).toEqual([]);
  const viewportPng = await page.locator('#viewport').screenshot();
  const visiblePixels = await page.evaluate(async (base64) => {
    const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const probe = document.createElement('canvas');
    probe.width = bitmap.width;
    probe.height = bitmap.height;
    const context = probe.getContext('2d', { willReadFrequently: true })!;
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
    let blue = 0;
    let green = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
      // Relative color checks reject the dark backdrop and blue-gray ground.
      if (b > r * 1.3 && b > g * 1.15 && b > 45) blue++;
      if (g > r * 1.15 && g > b * 1.15 && g > 40) green++;
    }
    return { blue, green };
  }, viewportPng.toString('base64'));
  expect(visiblePixels.blue, 'visible blue character pixels').toBeGreaterThan(50);
  expect(visiblePixels.green, 'visible green tree pixels').toBeGreaterThan(50);
  console.log('ENGINE_RENDER_DIAGNOSTIC ' + JSON.stringify({ result, visiblePixels }));
  await page.screenshot({ path: testInfo.outputPath('foundation.png'), fullPage: true });
  await testInfo.attach('diagnostics', { body: JSON.stringify({ result, snapshot: final, visiblePixels }, null, 2), contentType: 'application/json' });
  await page.evaluate(() => window.__REAL_ENGINE__!.dispose());
  expect(pageErrors).toEqual([]);
});

test('WebGPU initialization failures are visible instead of a blank canvas', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('/');
  await expect(page.locator('#error')).toContainText('WEBGPU_UNAVAILABLE');
  await expect(page.locator('#status')).toHaveText('RUNTIME ERROR');
});
