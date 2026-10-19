import { expect, test } from '@playwright/test';
import { composeWithSurface, integrateUniformMedium } from '../../src/participating-media';
import type { runVolumetricFogTemporalFixture, runVolumetricFogTemporalShadowFixture } from './fixtures/volumetric-fog-temporal';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runVolumetricFogTemporalFixture>>;
type Capture = Result['first'];

function close(actual: readonly number[], expected: readonly number[], tolerance: number, label: string): void {
  for (let channel = 0; channel < expected.length; channel++) {
    expect(Math.abs(actual[channel] - expected[channel]), `${label} channel ${channel}`).toBeLessThan(tolerance);
  }
}

function analytic(sample: Capture, index: number, result: Result): [number, number, number] {
  const pixel = result.pixels[index], width = sample.size.width, height = sample.size.height;
  const tangent = Math.tan(Math.PI / 6.4);
  const x = (2 * (pixel.x + 0.5) / width - 1) * tangent * width / height;
  const y = (1 - 2 * (pixel.y + 0.5) / height) * tangent;
  const rayLength = Math.hypot(x, y, 1);
  const segment = integrateUniformMedium(result.medium, 6 * rayLength,
    result.sun.color, -1 / rayLength);
  return composeWithSurface(segment, sample.surface[index].slice(0, 3) as [number, number, number]);
}

function fresh(sample: Capture, raw: number[][], label: string, reason: string): void {
  expect(sample.history.map(cell => cell.count), `${label} first-frame counts`).toEqual(sample.history.map(() => 1));
  expect(sample.diagnostics?.history.lastResetReason, label).toBe(reason);
  expect(sample.diagnostics?.history.valid, label).toBe(true);
  expect(sample.diagnostics?.history.framesSinceReset, label).toBe(1);
  sample.hdr.forEach((pixel, index) => close(pixel, raw[index], 0.003, `${label} HDR`));
}

for (const inputMode of ['committed', 'mutable'] as const) test(`temporal fog grows bounded ${inputMode} world-space history and rejects changed scene inputs without stale HDR`, async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async mode => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-temporal.ts') as {
      runVolumetricFogTemporalFixture: typeof runVolumetricFogTemporalFixture;
    };
    return fixture.runVolumetricFogTemporalFixture(mode);
  }, inputMode);

  expect(result.first.size).toEqual({ width: 64, height: 64 });
  expect(result.first.diagnostics?.grid).toMatchObject({ columns: 16, rows: 16, slices: 32 });
  expect(result.first.diagnostics?.grid?.bytes).toBeLessThanOrEqual(1024 * 1024);
  expect(result.first.diagnostics?.history).toMatchObject({ configured: true, maxFrames: 8 });
  expect(result.first.history.map(cell => cell.count)).toEqual([1, 1]);
  expect(result.countGrowth[0]).toEqual([1, 1]);
  for (const counts of result.countGrowth) for (const count of counts) {
    expect(count).toBeGreaterThanOrEqual(1);
    expect(count).toBeLessThanOrEqual(8);
  }
  expect(result.countGrowth.at(-1)![0]).toBe(8);
  expect(result.stable.diagnostics?.history.framesSinceReset).toBe(11);
  expect(result.translated.history[0].count).toBeGreaterThan(1);
  for (let index = 0; index < result.pixels.length; index++) {
    close(result.first.hdr[index], analytic(result.first, index, result), 0.003, 'first analytic HDR');
    close(result.stable.hdr[index], analytic(result.stable, index, result), 0.003, 'accumulated analytic HDR');
  }

  fresh(result.cameraCut, result.cameraCutRaw, 'camera cut', 'camera-cut');
  fresh(result.lightChanged, result.lightChangedRaw, 'light change', 'lighting-change');
  fresh(result.mediaChanged, result.mediaChangedRaw, 'media change', 'media-change');
  fresh(result.geometryChanged, result.geometryChangedRaw, 'geometry change', 'scene-change');
  fresh(result.materialChanged, result.materialChangedRaw, 'material change', 'material-change');
  expect(result.directionalStable.history.map(cell => cell.count)).toEqual([8, 8]);
  expect(result.directionalMoved.history.map(cell => cell.count)).toEqual([1, 1]);
  // Only the GPU's per-cell direction test rejected these samples: the CPU
  // did not mark the translation as a camera cut or reset the full history.
  expect(result.directionalMoved.diagnostics!.history.framesSinceReset).toBeGreaterThan(12);
  result.directionalMoved.hdr.forEach((pixel, i) => close(pixel, result.directionalRaw[i], 0.003, 'directional source'));

  expect(result.resized.size).toEqual({ width: 80, height: 64 });
  expect(result.resized.history[0].count).toBe(1);
  expect(result.resized.diagnostics?.grid?.columns).toBe(20);
  expect(result.resized.diagnostics?.history.framesSinceReset).toBe(1);
  expect(['resize', 'camera-cut']).toContain(result.resized.diagnostics?.history.lastResetReason);
  expect(result.bytesBeforeResize).toBeGreaterThan(0);
  expect(result.resized.diagnostics?.ownedGpuBytes).toBeGreaterThan(0);
  result.disabled.hdr.forEach((pixel, index) => close(pixel, result.disabled.surface[index], 0.003, 'disabled fog'));
  expect(result.disabled.diagnostics?.enabled).toBe(false);
  expect(result.disabled.diagnostics?.history.valid).toBe(false);
  expect(result.enabledAgain.history[0].count).toBe(1);
  expect(result.enabledAgain.diagnostics?.history.lastResetReason).toBe('fog-toggle');
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('accumulated fog rejects an offscreen mask and moving Morph or skin air shadow on the first changed frame', async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-temporal.ts') as {
      runVolumetricFogTemporalShadowFixture: typeof runVolumetricFogTemporalShadowFixture;
    };
    return fixture.runVolumetricFogTemporalShadowFixture();
  });

  expect(result.clear.diagnostics?.history.framesSinceReset).toBeGreaterThanOrEqual(12);
  expect(result.clear.history.every(cell => cell.count === 8)).toBe(true);
  for (const [label, capture] of Object.entries({ opaque: result.opaque, morphed: result.morphed,
    restoredAfterMorph: result.restoredAfterMorph, skinned: result.skinned,
    restoredAfterSkin: result.restoredAfterSkin, deleted: result.deleted })) {
    expect(capture.history.every(cell => cell.count === 1), `${label} new history`).toBe(true);
    expect(capture.diagnostics?.history.framesSinceReset, label).toBe(1);
  }
  expect(result.opaque.diagnostics?.history.lastResetReason).toBe('material-change');
  for (const capture of [result.morphed, result.restoredAfterMorph,
    result.skinned, result.restoredAfterSkin, result.deleted]) {
    expect(capture.diagnostics?.history.lastResetReason).toBe('scene-change');
  }
  for (let index = 0; index < result.pixels.length; index++) {
    for (let channel = 0; channel < 3; channel++) {
      const clear = result.clear.hdr[index][channel];
      const shadowed = result.opaque.hdr[index][channel];
      const noSun = result.zeroSun.hdr[index][channel];
      expect(clear - shadowed, `sun removed from air ${index}:${channel}`).toBeGreaterThan(0.04);
      expect(Math.abs(shadowed - noSun), `blocked air near zero-sun reference ${index}:${channel}`).toBeLessThan(0.02);
      for (const [label, capture] of [['morphed', result.morphed], ['skinned', result.skinned],
        ['deleted', result.deleted]] as const) {
        expect(Math.abs(capture.hdr[index][channel] - clear), `${label} relights air ${index}:${channel}`).toBeLessThan(0.003);
      }
      for (const [label, capture] of [['restoredAfterMorph', result.restoredAfterMorph],
        ['restoredAfterSkin', result.restoredAfterSkin]] as const) {
        expect(Math.abs(capture.hdr[index][channel] - shadowed), `${label} shadows air ${index}:${channel}`).toBeLessThan(0.003);
      }
    }
  }
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
