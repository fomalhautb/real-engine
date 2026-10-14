import { expect, test, type Page } from '@playwright/test';
import { evaluateGiIrradiance } from '../../src/diffuse-gi';
import type { runGiMultiBounceFixture } from './fixtures/gi-multibounce';

type Result = Awaited<ReturnType<typeof runGiMultiBounceFixture>>;
type Snapshot = Result['first'];
const emission = [0.2, 0.1, 0.05] as const;
const axes = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
] as const;

async function fixturePage(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return errors;
}

function meanIrradiance(snapshot: Snapshot): [number, number, number] {
  const sum: [number, number, number] = [0, 0, 0];
  expect(snapshot.probe.probes).toHaveLength(8);
  expect(snapshot.probe.invalidProbes).toBe(0);
  for (const probe of snapshot.probe.probes) {
    expect(probe.valid).toBe(true);
    expect(probe.primaryOverflows).toBe(0);
    expect(probe.shadowOverflows).toBe(0);
    expect(probe.maxRadiance).toHaveLength(3);
    for (const value of [...probe.coefficients.flat(), ...probe.moments.flat(), ...probe.maxRadiance]) {
      expect(Number.isFinite(value)).toBe(true);
    }
    for (const value of probe.maxRadiance) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
    for (const axis of axes) {
      const value = evaluateGiIrradiance(probe.coefficients.flat(), axis);
      for (let channel = 0; channel < 3; channel++) sum[channel] += value[channel];
    }
  }
  return sum.map(value => value / (8 * axes.length)) as [number, number, number];
}

function nearChannels(actual: readonly number[], expected: readonly number[], relative = 0.12): void {
  for (let channel = 0; channel < 3; channel++) {
    expect(Math.abs(actual[channel] - expected[channel]), `channel ${channel}`)
      .toBeLessThan(Math.max(0.005, expected[channel] * relative));
  }
}

function coefficientDelta(a: Snapshot, b: Snapshot): number {
  let largest = 0;
  for (let probe = 0; probe < 8; probe++) {
    const left = a.probe.probes[probe].coefficients.flat();
    const right = b.probe.probes[probe].coefficients.flat();
    for (let coefficient = 0; coefficient < left.length; coefficient++) {
      largest = Math.max(largest, Math.abs(left[coefficient] - right[coefficient]));
    }
  }
  return largest;
}

test('closed PBR cavity converges toward analytic diffuse transport and resets history on dependency edits', async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-multibounce.ts') as {
      runGiMultiBounceFixture: typeof runGiMultiBounceFixture;
    };
    return module.runGiMultiBounceFixture();
  }) as Result;

  const single = emission.map(value => Math.PI * value);
  const stable = emission.map(value => Math.PI * value / (1 - 0.96 * 0.5));
  const first = meanIrradiance(result.first);
  const second = meanIrradiance(result.second);
  const third = meanIrradiance(result.third);
  const settled = meanIrradiance(result.settled);
  console.log('GI_MULTIBOUNCE_GPU_DIAGNOSTIC ' + JSON.stringify({
    first, second, third, settled, expectedSingle: single, expectedStable: stable,
    historyUpdates: result.settled.diagnostics?.historyUpdates,
    resetReason: result.readded.diagnostics?.historyResetReason,
    ownedBytes: result.ownedBytes,
  }));

  // Every ray meets a wall. With no sun or sky, the first field is emission.
  // A Lambertian cavity with albedo .5 and dielectric factor .96 tends to
  // L = emission / (1 - .96 * .5), hence hemispherical E = pi * L.
  nearChannels(first, single, 0.06);
  nearChannels(settled, stable, 0.18);
  for (const probe of result.first.probe.probes) nearChannels(probe.maxRadiance, emission, 0.06);
  for (const probe of result.settled.probe.probes) {
    nearChannels(probe.maxRadiance, emission.map(value => value / (1 - 0.48)), 0.2);
  }
  expect(second[0]).toBeGreaterThan(first[0] * 1.25);
  expect(third[0]).toBeGreaterThan(second[0] * 1.08);
  expect(settled[0]).toBeGreaterThan(first[0] * 1.5);
  expect(settled[0]).toBeLessThan(first[0] * 2.2);
  expect(result.first.diagnostics?.historyUpdates).toBe(1);
  expect(result.first.diagnostics?.lastHistoryFrame).toBe(0);
  expect(result.settled.diagnostics?.historyUpdates).toBe(16);
  expect(result.settled.diagnostics?.lastHistoryFrame).toBe(result.settled.probe.frame - 1);

  // The field is world-space. A camera move must not reset the cache.
  nearChannels(meanIrradiance(result.cameraMoved), stable, 0.18);
  expect(coefficientDelta(result.cameraMoved, result.settled)).toBeLessThan(0.01);
  expect(result.cameraMoved.diagnostics?.historyUpdates)
    .toBeGreaterThan(result.settled.diagnostics?.historyUpdates ?? 0);
  nearChannels(meanIrradiance(result.modelMoved), stable, 0.2);
  expect(result.modelMoved.diagnostics?.historyUpdates)
    .toBeGreaterThan(result.cameraMoved.diagnostics?.historyUpdates ?? 0);
  expect(result.modelMoved.diagnostics?.historyResetReason)
    .toBe(result.cameraMoved.diagnostics?.historyResetReason);

  for (const snapshot of [result.historyChanged, result.generationChanged,
    result.sunChanged, result.explicitReset, result.emissionRestored,
    result.singleBounce, result.multiRestored, result.reenabled, result.readded]) {
    nearChannels(meanIrradiance(snapshot), single, 0.08);
  }
  nearChannels(meanIrradiance(result.emissionRemoved), [0, 0, 0], 0);
  nearChannels(meanIrradiance(result.removed), [0, 0, 0], 0);
  for (const [snapshot, reason] of [
    [result.historyChanged, 'scene-discontinuity'],
    [result.generationChanged, 'scene-discontinuity'],
    [result.sunChanged, 'sun-changed'],
    [result.explicitReset, 'explicit'],
    [result.emissionRemoved, 'material-changed'],
    [result.emissionRestored, 'material-changed'],
    [result.singleBounce, 'mode-changed'],
    [result.multiRestored, 'mode-changed'],
    [result.reenabled, 'enabled-changed'],
    [result.removed, 'scene-discontinuity'],
    [result.readded, 'scene-discontinuity'],
  ] as const) {
    expect(snapshot.diagnostics?.historyResetReason).toBe(reason);
    expect(snapshot.diagnostics?.historyUpdates).toBe(1);
  }
  expect(meanIrradiance(result.multiWarmed)[0]).toBeGreaterThan(first[0] * 1.2);
  expect(result.singleBounce.diagnostics?.multiBounce).toBe(false);
  expect(result.multiRestored.diagnostics?.multiBounce).toBe(true);
  expect(result.disabled?.enabled).toBe(false);
  expect(result.disabled?.historyUpdates).toBe(0);
  expect(result.reenabled.diagnostics?.enabled).toBe(true);
  expect(result.ownedBytes.length).toBeGreaterThan(30);
  expect(new Set(result.ownedBytes).size).toBe(1);
  expect(result.afterDispose).toBe(0);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test('demo saves single and multi bounce PNGs from the same fixed camera', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1024, height: 768 });
  try {
    await page.goto('/gi.html');
    await page.waitForFunction(() => window.__GI_DEMO__?.renderer && window.__GI_DEMO__.renderer.stats.frames > 0
      || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    await page.evaluate(() => window.__GI_DEMO__!.pause(true));
    await page.keyboard.press('h');
    await expect(page.locator('#gi-app')).toHaveClass(/ui-hidden/);
    const capture = async (name: string) => {
      await page.evaluate(async () => {
        const renderer = window.__GI_DEMO__!.renderer!;
        await renderer.probeHdr([{ x: Math.floor(renderer.stats.width / 2),
          y: Math.floor(renderer.stats.height / 2) }]);
      });
      const path = testInfo.outputPath(name + '.png');
      await page.screenshot({ path });
      await testInfo.attach(name, { path, contentType: 'image/png' });
    };
    const single = await page.evaluate(() => {
      const demo = window.__GI_DEMO__!;
      demo.setMultiBounce(false);
      return demo.snapshot();
    });
    await capture('diffuse-gi-single-bounce');
    const multiple = await page.evaluate(() => {
      const demo = window.__GI_DEMO__!;
      const camera = demo.snapshot().camera;
      demo.setMultiBounce(true); // One fixed-camera render in the control handler.
      for (let frame = 1; frame < 12; frame++) {
        demo.renderer!.render(camera.eye, camera.target as [number, number, number]);
      }
      return demo.snapshot();
    });
    await capture('diffuse-gi-multi-bounce');
    await page.keyboard.press('h');
    await capture('diffuse-gi-multi-bounce-ui');
    expect(single.camera).toEqual(multiple.camera);
    expect(single.controls.multiBounce).toBe(false);
    expect(multiple.controls.multiBounce).toBe(true);
    expect(single.diffuseGi?.multiBounce).toBe(false);
    expect(multiple.diffuseGi?.multiBounce).toBe(true);
    expect(multiple.diffuseGi?.historyUpdates).toBeGreaterThan(0);
    expect(multiple.stats!.frames).toBe(single.stats!.frames + 12);
    expect(single.errors).toEqual([]);
    expect(multiple.errors).toEqual([]);
    expect(pageErrors).toEqual([]);
  } finally {
    await page.evaluate(() => window.__GI_DEMO__?.dispose()).catch(() => {});
  }
});
