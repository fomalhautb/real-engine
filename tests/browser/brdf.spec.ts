import { expect, test } from '@playwright/test';
import type { runBrdfFixture } from './fixtures/brdf';

type Result = Awaited<ReturnType<typeof runBrdfFixture>>;
test.use({ deviceScaleFactor: 1 });

function direct(lit: readonly number[], ambient: readonly number[]): number[] {
  return lit.slice(0, 3).map((value, channel) => value - ambient[channel]);
}

/** GGX and Schlick at the single normal-incidence point v=l=n; G=1 and nl=1. */
function expectedNormalIncidence(base: number, metalness: number, roughness: number): number {
  const f0 = metalness ? base : 0.04;
  const diffuse = (1 - f0) * (1 - metalness) * base / Math.PI;
  const specular = f0 / (4 * Math.PI * roughness ** 4);
  return diffuse + specular;
}

function near(actual: number, expected: number, label: string, absolute = 0.00005): void {
  expect(Number.isFinite(actual), label).toBe(true);
  expect(Math.abs(actual - expected), `${label}: actual ${actual}, expected ${expected}`)
    .toBeLessThan(Math.max(absolute, Math.abs(expected) * 0.002));
}

test('PBR HDR radiance obeys normal-incidence GGX and reciprocal direct response', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') console.log('BRDF_BROWSER_ERROR ' + message.text());
  });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'brdf.ts') as { runBrdfFixture: typeof runBrdfFixture };
    return module.runBrdfFixture();
  });
  console.log('BRDF_GPU_DIAGNOSTIC ' + JSON.stringify({
    normal: result.normalIncidence, smooth: result.smooth, reciprocal: result.reciprocal,
    frames: result.frames, errors: result.errors,
  }));

  expect(result.normalIncidence).toHaveLength(4);
  for (const measurement of result.normalIncidence) {
    expect(measurement.ambient).toHaveLength(4);
    expect(measurement.lit).toHaveLength(4);
    const measuredDirect = direct(measurement.lit, measurement.ambient);
    for (let channel = 0; channel < 3; channel++) {
      near(measuredDirect[channel],
        expectedNormalIncidence(result.baseColor[channel], measurement.metalness, measurement.roughness),
        `normal incidence roughness=${measurement.roughness} metalness=${measurement.metalness} channel=${channel}`);
      expect(measurement.ambient[channel]).toBeGreaterThanOrEqual(0);
      expect(measurement.lit[channel]).toBeGreaterThanOrEqual(0);
    }
    near(measurement.ambient[3], 1, 'ambient alpha', 0.001);
    near(measurement.lit[3], 1, 'lit alpha', 0.001);
  }

  // The declared minimum roughness has a strong, finite highlight that remains
  // below the rgba16float ceiling. It is checked against the same closed form.
  const smoothDirect = direct(result.smooth.lit, result.smooth.ambient);
  for (let channel = 0; channel < 3; channel++) {
    const expected = expectedNormalIncidence(result.baseColor[channel], 0, 0.04);
    expect(expected).toBeGreaterThan(1000);
    expect(expected).toBeLessThan(65504);
    near(smoothDirect[channel], expected, `smooth channel=${channel}`);
    expect(result.smooth.lit[channel]).toBeGreaterThan(1000);
    expect(result.smooth.lit[channel]).toBeLessThan(65504);
  }

  const normalAmbient = result.normalIncidence.find(item => item.metalness === 0 && item.roughness === 0.5)!.ambient;
  expect(result.reciprocal).toHaveLength(2);
  for (const [pair, measurement] of result.reciprocal.entries()) {
    for (let channel = 0; channel < 3; channel++) {
      // Ambient is independent of camera and sun directions, so the measured
      // zero-sun baseline can be subtracted from both orientations of this pair.
      near(measurement.ambient[channel], normalAmbient[channel], `ambient angle pair=${pair} channel=${channel}`, 0.00005);
      const forward = (measurement.forward[channel] - measurement.ambient[channel]) / measurement.light[2];
      const reverse = (measurement.reverse[channel] - measurement.ambient[channel]) / measurement.view[2];
      expect(forward).toBeGreaterThanOrEqual(0);
      expect(reverse).toBeGreaterThanOrEqual(0);
      expect(Math.abs(forward - reverse), `reciprocity pair=${pair} channel=${channel}`)
        .toBeLessThan(Math.max(0.0002, Math.max(forward, reverse) * 0.004));
    }
  }

  expect(result.frames).toBe(16);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
