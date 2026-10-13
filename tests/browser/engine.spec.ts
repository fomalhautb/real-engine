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
  expect(first.stats!.drawCalls).toBe(8);
  expect(first.physics.bodies).toBe(6);
  expect(first.errors).toEqual([]);
  const result = await page.evaluate(async () => {
    const engine = window.__REAL_ENGINE__!;
    engine.pause(true);
    engine.step(1 / 60);
    return engine.verify();
  });
  expect(result.vertices).toBeGreaterThan(100);
  expect(result.passed).toBe(true);
  await page.getByRole('button', { name: '添加刚体', exact: true }).click();
  expect(await page.evaluate(() => window.__REAL_ENGINE__!.snapshot().physics.bodies)).toBe(7);
  await page.getByRole('button', { name: '移除刚体', exact: true }).click();
  expect(await page.evaluate(() => window.__REAL_ENGINE__!.snapshot().physics.bodies)).toBe(6);
  const reset = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.reset();
    return window.__REAL_ENGINE__!.verify();
  });
  expect(reset.passed).toBe(true);
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
  await page.screenshot({ path: testInfo.outputPath('foundation.png'), fullPage: true });
  await testInfo.attach('diagnostics', { body: JSON.stringify({ result, snapshot: final }, null, 2), contentType: 'application/json' });
  await page.evaluate(() => window.__REAL_ENGINE__!.dispose());
  expect(pageErrors).toEqual([]);
});

test('WebGPU initialization failures are visible instead of a blank canvas', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
  await page.goto('/');
  await expect(page.locator('#error')).toContainText('WEBGPU_UNAVAILABLE');
  await expect(page.locator('#status')).toHaveText('RUNTIME ERROR');
});
