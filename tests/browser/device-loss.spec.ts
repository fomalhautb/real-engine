import { expect, test } from '@playwright/test';
import type { runDeviceLossFixture } from './fixtures/device-loss';

test('explicit WebGPU device loss stops rendering and releases resources', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('DEVICE_LOSS_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'device-loss.ts') as {
      runDeviceLossFixture: typeof runDeviceLossFixture;
    };
    return module.runDeviceLossFixture();
  });
  console.log('DEVICE_LOSS_GPU_DIAGNOSTIC ' + JSON.stringify(result));

  expect(result.normal.beforeDispose).toBe(1);
  expect(result.normal.lossReason).toBe('destroyed');
  expect(result.normal.errors).toEqual([]);
  expect(result.normal.ownedGpuBytesAfterDispose).toBe(0);

  expect(result.loss.framesBeforeLoss).toBe(1);
  expect(result.loss.bytesBeforeLoss).toBeGreaterThan(0);
  expect(result.loss.lossReason).toBe('destroyed');
  expect(result.loss.errorsAfterLoss.some(error => error.includes('DEVICE_LOST'))).toBe(true);
  expect(result.loss.framesAfterAttempt).toBe(result.loss.framesBeforeLoss);
  expect(result.loss.queryRejection).toContain('DEVICE_LOST');
  expect(result.loss.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
