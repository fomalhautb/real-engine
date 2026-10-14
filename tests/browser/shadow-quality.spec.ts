import { expect, test } from '@playwright/test';
import type { runShadowQualityFixture, ShadowSample } from './fixtures/shadow-quality';

type Result = Awaited<ReturnType<typeof runShadowQualityFixture>>;
type Vec3 = readonly [number, number, number];
test.use({ deviceScaleFactor: 1 });

function dot(a: Vec3, b: Vec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function unit(v: Vec3): [number, number, number] {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}

/** Ray/plane intersection from the fixed camera and authored plane equation,
 * independent of the renderer's shadow comparison implementation. */
function worldAtPixel(result: Result, sample: ShadowSample, slope: number): Vec3 {
  const tanHalfFov = Math.tan(Math.PI / 6.4);
  const rayX = (2 * (sample.x + 0.5) / result.width - 1) * tanHalfFov * result.width / result.height;
  const rayY = (1 - 2 * (sample.y + 0.5) / result.height) * tanHalfFov;
  const distance = result.eye[2] / (1 + slope * rayX);
  return [rayX * distance, rayY * distance, result.eye[2] - distance];
}

/** Independent GGX/Schlick direct-light value with visibility fixed to one. */
function unshadowedDirect(result: Result, sample: ShadowSample, slope: number): number[] {
  const world = worldAtPixel(result, sample, slope);
  const n = unit([-slope, 0, 1]);
  const v = unit([result.eye[0] - world[0], result.eye[1] - world[1], result.eye[2] - world[2]]);
  const l = unit(result.sunDirection);
  const h = unit([l[0] + v[0], l[1] + v[1], l[2] + v[2]]);
  const nv = Math.max(dot(n, v), 0.001);
  const nl = Math.max(dot(n, l), 0);
  const nh = Math.max(dot(n, h), 0);
  const vh = Math.max(dot(v, h), 0);
  const alpha = result.roughness ** 2;
  const a2 = alpha ** 2;
  const denominator = (1 - nh * nh) + nh * nh * a2;
  const distribution = a2 / Math.max(Math.PI * denominator ** 2, 1e-20);
  const k = (result.roughness + 1) ** 2 / 8;
  const geometry = (nv / (nv * (1 - k) + k)) * (nl / (nl * (1 - k) + k));
  const fresnel = 0.04 + 0.96 * (1 - vh) ** 5;
  const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
  return result.baseColor.map((base, channel) =>
    ((1 - fresnel) * base / Math.PI + specular) * result.sunColor[channel] * nl);
}

function direct(sample: ShadowSample, channel: number): number {
  return sample.lit[channel] - sample.baseline[channel];
}

test('shadow PCF preserves direct light across low and steep unoccluded receivers and retains near contact', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'shadow-quality.ts') as {
      runShadowQualityFixture: typeof runShadowQualityFixture;
    };
    return fixture.runShadowQualityFixture();
  }) as Result;

  const diagnostic: Record<string, { min: number; max: number; mean: number }> = {};
  for (const receiver of [result.low, result.high]) {
    const ratios = receiver.samples.flatMap(sample => {
      const expected = unshadowedDirect(result, sample, receiver.slope);
      return expected.map((value, channel) => direct(sample, channel) / value);
    });
    diagnostic[String(receiver.slope)] = { min: Math.min(...ratios), max: Math.max(...ratios),
      mean: ratios.reduce((a, b) => a + b, 0) / ratios.length };
  }
  console.log('SHADOW_QUALITY_GPU_DIAGNOSTIC ' + JSON.stringify({ diagnostic,
    contact: result.contact, wallContact: result.wallContact, frames: result.frames, errors: result.errors }));
  for (const receiver of [result.low, result.high]) {
    expect(receiver.samples).toHaveLength(35);
    for (const sample of receiver.samples) {
      const expected = unshadowedDirect(result, sample, receiver.slope);
      for (let channel = 0; channel < 3; channel++) {
        // A zero-sun frame establishes that the pixel is on the intended
        // receiver, and removes legacy ambient from the direct-light check.
        expect(Math.abs(sample.baseline[channel] - result.baseColor[channel] * 0.045),
          `baseline ${receiver.slope} (${sample.x},${sample.y}) channel ${channel}`).toBeLessThan(0.002);
        const measured = direct(sample, channel);
        expect(expected[channel]).toBeGreaterThan(0.05);
        expect(Number.isFinite(measured)).toBe(true);
        expect(Math.abs(measured - expected[channel]),
          `self-shadow ${receiver.slope} (${sample.x},${sample.y}) channel ${channel}: ${measured} vs ${expected[channel]}`)
          .toBeLessThan(Math.max(0.006, expected[channel] * (receiver.slope === 0 ? 0.07 : 0.09)));
      }
      expect(sample.baseline[3]).toBeCloseTo(1, 3);
      expect(sample.lit[3]).toBeCloseTo(1, 3);
    }
  }

  const center = result.low.samples.find(sample => sample.x === 128 && sample.y === 128)!;
  const clear = result.low.samples.find(sample => sample.x === 80 && sample.y === 128)!;
  for (let channel = 0; channel < 3; channel++) {
    const shaded = result.contact[0][channel] - center.baseline[channel];
    const unshaded = direct(center, channel);
    expect(shaded, `near caster must retain contact shadow, channel ${channel}`)
      .toBeLessThan(unshaded * 0.55);
    const control = result.contact[1][channel] - clear.baseline[channel];
    expect(Math.abs(control - direct(clear, channel)),
      `outside caster remains lit, channel ${channel}`).toBeLessThan(direct(clear, channel) * 0.08);
  }
  expect(result.wallContact.points).toHaveLength(3);
  for (let point = 0; point < result.wallContact.points.length; point++) {
    for (let channel = 0; channel < 3; channel++) {
      const openBaseline = result.wallContact.openBaseline[point][channel];
      const sealedBaseline = result.wallContact.sealedBaseline[point][channel];
      const openDirect = result.wallContact.openLit[point][channel] - openBaseline;
      const sealedDirect = result.wallContact.sealedLit[point][channel] - sealedBaseline;
      // Same visible floor pixel in both frames. A wall accidentally covering
      // the camera ray would change this zero-sun baseline.
      expect(Math.abs(sealedBaseline - openBaseline),
        `wall/floor receiver identity point=${point} channel=${channel}`).toBeLessThan(0.002);
      expect(openDirect, `wall/floor unobstructed reference point=${point} channel=${channel}`)
        .toBeGreaterThan(0.2);
      expect(sealedDirect / openDirect,
        `closed wall/floor sun leak point=${point} channel=${channel}`)
        .toBeLessThan([0.65, 0.3, 0.15][point]);
    }
  }
  expect(result.frames).toBe(9);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
