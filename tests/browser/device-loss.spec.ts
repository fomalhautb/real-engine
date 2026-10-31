import { expect, test, type Page } from '@playwright/test';
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

type LossControl = { devices: GPUDevice[]; adapterRequests: number; failAdapters: number };
declare global { interface Window { __DEVICE_LOSS_TEST__?: LossControl } }

/** Test-local fault injection: record every device the page creates and optionally answer
 * adapter requests with null, as a browser may while its GPU process restarts. A real driver
 * reset cannot be triggered from a page; GPUDevice.destroy() is the injected loss. */
async function instrument(page: Page): Promise<string[]> {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('DEVICE_RECOVERY_BROWSER_ERROR ' + message.text()); });
  await page.addInitScript(() => {
    const control: LossControl = { devices: [], adapterRequests: 0, failAdapters: 0 };
    window.__DEVICE_LOSS_TEST__ = control;
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (this: GPUAdapter, descriptor?: GPUDeviceDescriptor) {
      const device = await requestDevice.call(this, descriptor);
      control.devices.push(device);
      return device;
    };
    const requestAdapter = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = async function (this: GPU, options?: GPURequestAdapterOptions) {
      control.adapterRequests++;
      if (control.failAdapters > 0) { control.failAdapters--; return null; }
      return requestAdapter.call(this, options);
    };
  });
  await page.goto('/?trees=4');
  await page.waitForFunction(() => (window.__REAL_ENGINE__?.snapshot().stats?.frames ?? 0) >= 5 || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  return pageErrors;
}
const loseCurrentDevice = (page: Page) => page.evaluate(() => {
  const control = window.__DEVICE_LOSS_TEST__!;
  control.devices.at(-1)!.destroy();
  return control.devices.length;
});

test('main page rebuilds its Renderer on a new device after loss and keeps the retained scene', async ({ page }) => {
  const pageErrors = await instrument(page);
  const before = await page.evaluate(() => {
    const engine = window.__REAL_ENGINE__!;
    if (!engine.spawn()) throw new Error('spawn failed');
    return engine.snapshot();
  });
  expect(before.physics.bodies).toBe(7);
  expect(before.deviceRecovery!.state).toBe('running');
  const devicesAtLoss = await loseCurrentDevice(page);
  await page.waitForFunction(() => {
    const snapshot = window.__REAL_ENGINE__!.snapshot();
    return (snapshot.deviceRecovery!.recoveries === 1 && (snapshot.stats?.frames ?? 0) >= 5) || !document.getElementById('error')!.hidden;
  });
  await expect(page.locator('#error')).toBeHidden();
  const after = await page.evaluate(async () => {
    const control = window.__DEVICE_LOSS_TEST__!, engine = window.__REAL_ENGINE__!;
    const oldLoss = await control.devices.at(-2)!.lost;
    return { snapshot: engine.snapshot(), verification: await engine.verify(), oldReason: oldLoss.reason,
      devices: control.devices.length, status: document.getElementById('status')!.textContent };
  });
  console.log('DEVICE_RECOVERY_DIAGNOSTIC ' + JSON.stringify({ before: before.deviceRecovery, after: after.snapshot.deviceRecovery,
    adapter: after.snapshot.environment?.adapterInfo, bytes: [before.stats!.ownedGpuBytes, after.snapshot.stats!.ownedGpuBytes] }));
  const recovery = after.snapshot.deviceRecovery!;
  expect(recovery.state).toBe('running');
  expect(recovery.losses).toBe(1);
  expect(recovery.events.map(event => event.kind)).toEqual(['started', 'lost', 'attempt', 'recovered']);
  expect(recovery.events[1].reason).toBe('destroyed');
  expect(after.oldReason).toBe('destroyed');
  expect(after.devices).toBe(devicesAtLoss + 1);
  // Simulation state survives; GPU state is rebuilt from the same retained scene inputs.
  expect(after.snapshot.physics.bodies).toBe(7);
  expect(after.snapshot.errors).toEqual([]);
  expect(after.snapshot.stats!.ownedGpuBytes).toBe(before.stats!.ownedGpuBytes);
  expect(after.snapshot.environment!.adapterInfo.vendor).toBe(before.environment!.adapterInfo.vendor);
  expect(after.verification.passed).toBe(true);
  expect(after.verification.boundsViolations).toBe(0);

  // A second loss is recovered the same way; the bounded window still has room.
  await loseCurrentDevice(page);
  await page.waitForFunction(() => {
    const snapshot = window.__REAL_ENGINE__!.snapshot();
    return (snapshot.deviceRecovery!.recoveries === 2 && (snapshot.stats?.frames ?? 0) >= 3) || !document.getElementById('error')!.hidden;
  });
  await expect(page.locator('#error')).toBeHidden();
  const second = await page.evaluate(async () => ({ snapshot: window.__REAL_ENGINE__!.snapshot(),
    verification: await window.__REAL_ENGINE__!.verify() }));
  expect(second.snapshot.errors).toEqual([]);
  expect(second.verification.passed).toBe(true);
  const disposed = await page.evaluate(async () => {
    const control = window.__DEVICE_LOSS_TEST__!, last = control.devices.at(-1)!, count = control.devices.length;
    window.__REAL_ENGINE__!.dispose();
    const lost = await last.lost;
    await new Promise(resolve => setTimeout(resolve, 300));
    return { reason: lost.reason, created: control.devices.length - count, state: window.__REAL_ENGINE__!.snapshot().deviceRecovery!.state };
  });
  // Intentional disposal destroys the device without being treated as a loss to recover from.
  expect(disposed).toEqual({ reason: 'destroyed', created: 0, state: 'disposed' });
  expect(pageErrors).toEqual([]);
});

test('device recovery retries a missing adapter within its bounds and reports exhaustion', async ({ page }) => {
  const pageErrors = await instrument(page);
  const firstDevice = await page.evaluate(() => {
    window.__DEVICE_LOSS_TEST__!.failAdapters = 2;
    return window.__DEVICE_LOSS_TEST__!.devices.length;
  });
  await loseCurrentDevice(page);
  await page.waitForFunction(() => {
    const snapshot = window.__REAL_ENGINE__!.snapshot();
    return (snapshot.deviceRecovery!.recoveries === 1 && (snapshot.stats?.frames ?? 0) >= 3) || !document.getElementById('error')!.hidden;
  }, undefined, { timeout: 20_000 });
  await expect(page.locator('#error')).toBeHidden();
  const transient = await page.evaluate(() => window.__REAL_ENGINE__!.snapshot());
  const attempts = transient.deviceRecovery!.events.filter(event => event.loss === 1);
  expect(attempts.map(event => event.kind)).toEqual(['lost', 'attempt', 'attempt-failed', 'attempt', 'attempt-failed', 'attempt', 'recovered']);
  expect(attempts.filter(event => event.kind === 'attempt-failed').every(event => event.error!.startsWith('ADAPTER_UNAVAILABLE'))).toBe(true);
  expect(attempts.at(-1)!.durationMs!).toBeGreaterThanOrEqual(250 + 1000);
  expect(transient.errors).toEqual([]);

  const exhausted = await page.evaluate(() => {
    window.__DEVICE_LOSS_TEST__!.failAdapters = 1_000;
    return window.__DEVICE_LOSS_TEST__!.adapterRequests;
  });
  await loseCurrentDevice(page);
  await expect(page.locator('#error')).toBeVisible({ timeout: 20_000 });
  const failed = await page.evaluate(() => ({ snapshot: window.__REAL_ENGINE__!.snapshot(),
    error: document.getElementById('error')!.textContent, status: document.getElementById('status')!.textContent,
    adapterRequests: window.__DEVICE_LOSS_TEST__!.adapterRequests, devices: window.__DEVICE_LOSS_TEST__!.devices.length }));
  console.log('DEVICE_RECOVERY_FAILURE_DIAGNOSTIC ' + JSON.stringify(failed));
  expect(failed.error).toContain('DEVICE_RECOVERY_FAILED: 3 attempts after destroyed device loss');
  expect(failed.error).toContain('last error: ADAPTER_UNAVAILABLE');
  expect(failed.status).toBe('RUNTIME ERROR');
  expect(failed.snapshot.deviceRecovery!.state).toBe('failed');
  expect(failed.snapshot.stats).toBeNull(); // the lost Renderer was released, nothing renders
  expect(failed.adapterRequests - exhausted).toBe(3);
  expect(failed.devices).toBe(firstDevice + 1);
  expect(pageErrors).toEqual([]);
});
