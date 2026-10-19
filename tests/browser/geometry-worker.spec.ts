import { expect, test } from '@playwright/test';
import type { runGeometryWorkerFixture } from './fixtures/geometry-worker';

test('real decode Workers transfer buffers, cancel active work and recover without retaining resources', async ({ page }) => {
  const errors: string[] = [], workers: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('worker', worker => workers.push(worker.url()));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async workerUrl => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-worker.ts') as { runGeometryWorkerFixture: typeof runGeometryWorkerFixture };
    return fixture.runGeometryWorkerFixture(workerUrl);
  }, process.env.PLAYWRIGHT_GEOMETRY_WORKER_URL);
  expect(result.detachedInput).toBe(true); expect(result.providerBufferPreserved).toBe(true);
  expect(result.queuedError).toContain('AbortError'); expect(result.cancelledError).toContain('AbortError');
  expect(result.afterCancel).toMatchObject({ cancelled: 2, workers: 0, activeRequests: 0, waitingRequests: 0, inFlightBytes: 0 });
  expect(result.afterRecovery).toMatchObject({ completed: 1, failed: 0, workersCreated: 2, workersTerminated: 1 });
  expect(result.afterRecovery.peakInFlightBytes).toBe(result.budget);
  expect(result.mismatches).toBe(0); expect(result.comparedBytes).toBeGreaterThan(0);
  expect(result.disposalErrors.every(error => error.includes('GEOMETRY_DECODE_DISPOSED'))).toBe(true);
  expect(result.afterDispose).toMatchObject({ disposed: true, workers: 0, activeRequests: 0, waitingRequests: 0,
    inFlightBytes: 0, cleanupFailures: 0, workersCreated: 2, workersTerminated: 2 });
  // Cancellation can terminate the first Worker before DevTools attaches to it.
  // Pool counters above verify both lifetimes; the recovered Worker must be observable.
  const observedWorkers = workers.filter(url => url.includes('geometry-page.worker'));
  expect(observedWorkers.length).toBeGreaterThanOrEqual(1);
  expect(observedWorkers.length).toBeLessThanOrEqual(2);
  expect(errors).toEqual([]);
  console.log('GEOMETRY_WORKER_DIAGNOSTIC', JSON.stringify(result));
});
