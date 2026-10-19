import { expect, test } from '@playwright/test';
import type { runGeometryWorkerDeviceLossAfterInit, runGeometryWorkerDeviceLossDuringInit } from './fixtures/geometry-worker-device-loss';

test('streamed Worker pool stops on explicit device loss after Renderer initialization', async ({ page }) => {
  test.setTimeout(120000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-worker-device-loss.ts') as {
      runGeometryWorkerDeviceLossAfterInit: typeof runGeometryWorkerDeviceLossAfterInit;
    };
    return fixture.runGeometryWorkerDeviceLossAfterInit();
  });
  expect(result.beforeLoss).toMatchObject({ disposed: false, activeRequests: 0, waitingRequests: 0, inFlightBytes: 0 });
  expect(result.beforeLoss!.workersCreated).toBeGreaterThan(0);
  expect(result.beforeLoss!.completed).toBeGreaterThan(0);
  expect(result.lossReason).toBe('destroyed');
  expect(result.afterLoss).toMatchObject({ disposed: true, workers: 0, activeRequests: 0,
    waitingRequests: 0, inFlightBytes: 0 });
  expect(result.bytesBeforeLoss).toBeGreaterThan(0);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('device loss during an unregistered streamed root fetch aborts provider and rejects Renderer creation', async ({ page }) => {
  test.setTimeout(120000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-worker-device-loss.ts') as {
      runGeometryWorkerDeviceLossDuringInit: typeof runGeometryWorkerDeviceLossDuringInit;
    };
    return fixture.runGeometryWorkerDeviceLossDuringInit();
  });
  expect(result.providerEntered).toBe(true);
  expect(result.providerSignalAborted).toBe(true);
  expect(result.rejection).not.toBe('');
  expect(result.lossReason).toBe('destroyed');
  expect(result.contextUnconfigured).toBe(true);
  expect(pageErrors).toEqual([]);
});
