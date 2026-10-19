import { expect, test } from '@playwright/test';
import { composeWithSurface, integrateUniformMedium, type MediumCoefficients } from '../../src/participating-media';
import type { runVolumetricFogFixture } from './fixtures/volumetric-fog';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runVolumetricFogFixture>>;
type Sample = Result['near'];

function near(actual: readonly number[], expected: readonly number[], tolerance: number, label: string): void {
  expect(actual.length, label).toBeGreaterThanOrEqual(expected.length);
  for (let channel = 0; channel < expected.length; channel++) {
    expect(Math.abs(actual[channel] - expected[channel]), `${label} channel ${channel}`)
      .toBeLessThan(tolerance);
  }
}

/** Perspective geometry calculated independently from the GPU fog path. */
function ray(point: { x: number; y: number }, width: number, height: number) {
  const tangent = Math.tan(Math.PI / 6.4);
  const x = (2 * (point.x + 0.5) / width - 1) * tangent * width / height;
  const y = (1 - 2 * (point.y + 0.5) / height) * tangent;
  const length = Math.hypot(x, y, 1);
  return { length, cosSun: -1 / length };
}

function expectedFog(sample: Sample, pointIndex: number, result: Result,
  coefficients: MediumCoefficients, sun: readonly [number, number, number]) {
  const viewDepth = sample.planeZ === null ? result.maxViewDepth
    : Math.min(result.maxViewDepth, result.eye[2] - sample.planeZ);
  const direction = ray(result.points[pointIndex], result.initialSize.width, result.initialSize.height);
  const segment = integrateUniformMedium(coefficients, viewDepth * direction.length, sun, direction.cosSun);
  return composeWithSurface(segment, sample.surface[pointIndex].slice(0, 3) as [number, number, number]);
}

test('bounded fog matches analytic absorption, emission, and anisotropic sun scattering on rendered pixels', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog.ts') as {
      runVolumetricFogFixture: typeof runVolumetricFogFixture;
    };
    return fixture.runVolumetricFogFixture();
  });

  expect(result.initialSize).toEqual({ width: 64, height: 64 });
  for (let index = 0; index < result.points.length; index++) {
    near(result.disabled.fog[index], result.disabled.surface[index], 0.003, 'disabled');
    near(result.empty.fog[index], result.empty.surface[index], 0.003, 'vacuum');
    near(result.empty.fog[index], result.disabled.fog[index], 0.003, 'vacuum versus disabled');
    near(result.toggledOff.fog[index], result.toggledOff.surface[index], 0.003, 'toggled off');
    near(result.replacedEmpty.fog[index], result.replacedEmpty.surface[index], 0.003, 'cleared media');
    for (const sample of [result.nearClip, result.near, result.far, result.beyondLimit, result.sky]) {
      near(sample.fog[index], expectedFog(sample, index, result, result.absorption, [0, 0, 0]),
        0.003, `absorption z=${sample.planeZ}`);
    }
    near(result.emitted.fog[index], expectedFog(result.emitted, index, result, result.emission, [0, 0, 0]),
      0.003, 'zero-extinction emission');
    near(result.scattered.fog[index], expectedFog(result.scattered, index, result, result.scattering, result.sun.color),
      0.003, 'anisotropic sun scattering');
  }
  expect(result.nearClip.surface[0][0]).toBeGreaterThan(1.5);
  expect(result.beyondLimit.surface[0][0]).toBeGreaterThan(1.5);
  expect(result.near.fog[0][0]).toBeGreaterThan(result.far.fog[0][0]);
  // Both rays stop at maxViewDepth; their difference is only the attenuated surface.
  const limitedDistance = result.maxViewDepth * ray(result.points[0], 64, 64).length;
  const transmission = integrateUniformMedium(result.absorption, limitedDistance, [0, 0, 0]).transmittance;
  for (let channel = 0; channel < 3; channel++) {
    const difference = result.beyondLimit.fog[0][channel] - result.sky.fog[0][channel];
    const expected = transmission * (result.beyondLimit.surface[0][channel] - result.sky.surface[0][channel]);
    expect(Math.abs(difference - expected), `sky cutoff channel ${channel}`).toBeLessThan(0.025);
  }

  expect(result.initialFog).not.toBeNull();
  expect(result.initialFog!.grid).not.toBeNull();
  expect(result.initialFog!.grid!.pixelSize).toBe(1);
  expect(result.initialFog!.grid!.froxels).toBe(64 * 64 * 64);
  expect(result.initialFog!.grid!.froxels).toBeLessThanOrEqual(result.maxFroxels);
  expect(result.initialFog!.grid!.bytes).toBeLessThanOrEqual(result.maxBytes);
  expect(result.initialFog!.ownedGpuBytes).toBe(result.initialFog!.grid!.bytes);
  expect(result.resizedSize).toEqual({ width: 80, height: 64 });
  expect(result.resizedFog!.grid!.pixelSize).toBeGreaterThan(1);
  expect(result.resizedFog!.grid!.froxels).toBeLessThanOrEqual(result.maxFroxels);
  expect(result.resizedFog!.grid!.bytes).toBeLessThanOrEqual(result.maxBytes);
  expect(result.resizedFog!.ownedGpuBytes).toBe(result.resizedFog!.grid!.bytes);
  expect(result.initialOwnedBytes).toBeGreaterThan(0);
  expect(result.resizedOwnedBytes).toBeGreaterThan(0);
  expect(result.frames).toBe(12);
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
  console.log('VOLUMETRIC_FOG_GPU_DIAGNOSTIC ' + JSON.stringify({
    near: result.near.fog, far: result.far.fog, sky: result.sky.fog,
    scattered: result.scattered.fog, initial: result.initialFog, resized: result.resizedFog,
  }));
});
