import { expect, test } from '@playwright/test';
import { sampleReflectionGgx } from '../../src/reflections';
import type { runReflectionSamplingFixture } from './fixtures/reflections-sampling';

type Result = Awaited<ReturnType<typeof runReflectionSamplingFixture>>;

function reverseBits(index: number): number {
  let bits = index >>> 0;
  bits = ((bits >>> 1) & 0x55555555) | ((bits & 0x55555555) << 1);
  bits = ((bits >>> 2) & 0x33333333) | ((bits & 0x33333333) << 2);
  bits = ((bits >>> 4) & 0x0f0f0f0f) | ((bits & 0x0f0f0f0f) << 4);
  bits = ((bits >>> 8) & 0x00ff00ff) | ((bits & 0x00ff00ff) << 8);
  return (((bits >>> 16) | (bits << 16)) >>> 0);
}

test('production reflection sampling matches independently evaluated grazing BRDF divided by GGX PDF', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-sampling.ts') as {
      runReflectionSamplingFixture: typeof runReflectionSamplingFixture;
    };
    return fixture.runReflectionSamplingFixture();
  }) as Result;
  await testInfo.attach('reflection-sampling-numerical.json', {
    body: JSON.stringify(result, null, 2), contentType: 'application/json',
  });
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapterInfo.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapterInfo.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  expect(result.cases.map(item => item.nv)).toEqual([0.0001, 0.005, 0.1, 1]);
  for (const { nv, eye, samples } of result.cases) {
    expect(samples).toHaveLength(16);
    let validCount = 0;
    for (let index = 0; index < 16; index++) {
      const actual = samples[index];
      const cpu = sampleReflectionGgx([0, 0, 1], eye as [number, number, number], 0.5,
        reverseBits(index) / 0x100000000, (index + 0.5) / 16);
      const nl = cpu.direction[2];
      const vh = eye[0] * cpu.halfVector[0] + eye[1] * cpu.halfVector[1] + eye[2] * cpu.halfVector[2];
      const shouldBeValid = nv > 0 && nl > 0 && vh > 0 && cpu.pdf > 0;
      const label = `NoV ${nv}, sample ${index}`;
      expect(actual.direction.every(Number.isFinite), `${label} direction finite`).toBe(true);
      expect(actual.weight.every(Number.isFinite), `${label} weight finite`).toBe(true);
      if (!shouldBeValid) {
        expect(actual.weight[3], `${label} invalid flag`).toBe(0);
        expect(actual.weight.slice(0, 3), `${label} invalid weight`).toEqual([0, 0, 0]);
        continue;
      }
      validCount++;
      expect(actual.weight[3], `${label} valid flag`).toBe(1);
      expect(Math.abs(Math.hypot(...actual.direction.slice(0, 3)) - 1),
        `${label} unit direction`).toBeLessThanOrEqual(1e-5);
      // WGSL transcendental results need not match CPU binary64. The measured
      // SwiftShader direction error is 7.31e-5, while its weight error is under
      // 0.022%. Keep a bounded direction check and the independent BRDF/PDF
      // check below; do not weaken energy, hemisphere, or normalization checks.
      // https://www.w3.org/TR/WGSL/#floating-point-accuracy
      for (let component = 0; component < 3; component++) {
        expect(Math.abs(actual.direction[component] - cpu.direction[component]),
          `${label} direction ${component}`).toBeLessThanOrEqual(1e-4);
      }
      // Independent Cook-Torrance evaluation. The CPU sampler supplies only
      // the sampled direction and its half-vector PDF, never its BRDF weight.
      const nh = cpu.halfVector[2];
      const alpha2 = 0.5 ** 4;
      const denominator = 1 - nh * nh + nh * nh * alpha2;
      const distribution = alpha2 / (Math.PI * denominator * denominator);
      const k = (0.5 + 1) ** 2 / 8;
      const shadingNv = Math.max(nv, 0.001);
      const geometry = shadingNv / (shadingNv * (1 - k) + k)
        * nl / (nl * (1 - k) + k);
      const specularBrdf = distribution * geometry / Math.max(4 * shadingNv * nl, 0.001);
      const expectedWeight = specularBrdf * nl / cpu.pdf;
      for (let component = 0; component < 3; component++) {
        expect(Math.abs(actual.weight[component] - expectedWeight),
          `${label} BRDF/PDF weight ${component}: actual ${actual.weight[component]} expected ${expectedWeight}`)
          .toBeLessThanOrEqual(Math.max(2e-6, Math.abs(expectedWeight) * 0.005));
      }
    }
    expect(validCount, `NoV ${nv} has measurable valid rays`).toBeGreaterThan(0);
  }
  expect(result.uncaptured).toEqual([]);
  expect(result.validationError).toBe('');
  expect(result.ownedBytes).toBe(0);
  expect(pageErrors).toEqual([]);
});
