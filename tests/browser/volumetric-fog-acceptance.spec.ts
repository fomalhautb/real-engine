import { expect, test } from '@playwright/test';
import { composeMediumSegments, integrateUniformMedium, type MediumSegment } from '../../src/participating-media';
import type { runFogAnalyticAcceptanceFixture } from './fixtures/volumetric-fog-acceptance';
import { FOG_ANALYTIC_CASES } from './fixtures/volumetric-fog-acceptance-cases';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runFogAnalyticAcceptanceFixture>>;
type Vec3 = [number, number, number];

/** Independent world-space reference along the pixel ray to the receiver at planar depth 6. */
function reference(entry: (typeof FOG_ANALYTIC_CASES)[number], pixel: { x: number; y: number }, result: Result): MediumSegment {
  const tangent = Math.tan(Math.PI / 6.4);
  const ray: Vec3 = [(2 * (pixel.x + 0.5) / result.size - 1) * tangent, (1 - 2 * (pixel.y + 0.5) / result.size) * tangent, -1];
  const length = Math.hypot(...ray), view = ray.map(value => value / length) as Vec3;
  const sunLength = Math.hypot(...entry.sun.direction);
  const cosine = view.reduce((sum, value, axis) => sum + value * entry.sun.direction[axis] / sunLength, 0);
  const global = entry.global!;
  const distance = 6 * length;
  if (!global.heightFalloff) return integrateUniformMedium(global.coefficients, distance, entry.sun.color, cosine);
  const steps = 8192, delta = distance / steps;
  let sum: MediumSegment = { transmittance: 1, scattering: [0, 0, 0] };
  for (let step = 0; step < steps; step++) {
    const y = result.eye[1] + view[1] * (step + 0.5) * delta;
    const density = Math.exp(-global.heightFalloff * Math.max(y - (global.baseHeight ?? 0), 0));
    const c = global.coefficients;
    sum = composeMediumSegments(sum, integrateUniformMedium({ ...c, extinction: c.extinction * density,
      emission: c.emission.map(value => value * density) as unknown as Vec3 }, delta, entry.sun.color, cosine));
  }
  return sum;
}

test('F01: split L_fog and T match analytic media; zero density is exact and extremes stay finite', async ({ page }) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-acceptance.ts') as {
      runFogAnalyticAcceptanceFixture: typeof runFogAnalyticAcceptanceFixture };
    return fixture.runFogAnalyticAcceptanceFixture();
  });

  const summary: Record<string, { transmittance: number; relative: number; composite: number;
    accumulatedTransmittance: number; accumulatedRelative: number }> = {};
  for (const entry of FOG_ANALYTIC_CASES) {
    const measured = result.cases.find(item => item.label === entry.label)!;
    expect(measured.history?.lastResetReason, entry.label).toBe('temporal-toggle');
    for (const [label, values] of [['raw', measured.rawLattice], ['accumulated', measured.accumulatedLattice]] as const) {
      for (const value of values) {
        expect(value.every(Number.isFinite), `${entry.label} ${label} finite`).toBe(true);
        for (let c = 0; c < 3; c++) {
          expect(value[c], `${entry.label} ${label} nonnegative`).toBeGreaterThanOrEqual(0);
          expect(value[c], `${entry.label} ${label} bounded`).toBeLessThanOrEqual(65504);
        }
        expect(value[3], `${entry.label} ${label} T`).toBeGreaterThanOrEqual(0);
        expect(value[3], `${entry.label} ${label} T`).toBeLessThanOrEqual(1);
      }
    }
    let worstT = 0, worstRelative = 0, worstComposite = 0, accumulatedT = 0, accumulatedRelative = 0;
    result.roi.forEach((pixel, index) => {
      const surface = measured.surface[index];
      const split = measured.raw[index];
      // The composite is exactly L_fog + T * L_surface of the same split.
      for (let c = 0; c < 3; c++) {
        const error = Math.abs(measured.composite[index][c] - (split[c] + split[3] * surface[c]));
        worstComposite = Math.max(worstComposite, error);
        expect(error, `${entry.label} composite ${index}:${c}`).toBeLessThan(0.002 + 0.002 * measured.composite[index][c]);
      }
      if (entry.reference === 'none') return;
      const expected = reference(entry, pixel, result);
      for (const [label, value] of [['raw', split], ['accumulated', measured.accumulated[index]]] as const) {
        // F01 scenes gate both sequences; the non-uniform extra scene gates the history-off sequence
        // and records the jittered accumulation residual for the report.
        const gated = entry.reference === 'f01' || label === 'raw';
        const errorT = Math.abs(value[3] - expected.transmittance);
        if (label === 'raw') worstT = Math.max(worstT, errorT); else accumulatedT = Math.max(accumulatedT, errorT);
        if (gated) expect(errorT, `${entry.label} ${label} T ${index}`).toBeLessThanOrEqual(0.001);
        for (let c = 0; c < 3; c++) {
          if (expected.scattering[c] <= 0.01) {
            if (gated) expect(Math.abs(value[c] - expected.scattering[c]), `${entry.label} ${label} weak L ${index}:${c}`).toBeLessThan(0.0005);
            continue;
          }
          const relative = Math.abs(value[c] - expected.scattering[c]) / expected.scattering[c];
          if (label === 'raw') worstRelative = Math.max(worstRelative, relative); else accumulatedRelative = Math.max(accumulatedRelative, relative);
          if (gated) expect(relative, `${entry.label} ${label} L ${index}:${c}`).toBeLessThanOrEqual(0.01);
        }
      }
      if (entry.label === 'zero-density') {
        for (const value of [split, measured.accumulated[index]]) {
          expect(value.slice(0, 3).map(Math.abs)).toEqual([0, 0, 0]);
          expect(value[3]).toBe(1);
        }
        expect(measured.composite[index].slice(0, 3)).toEqual(surface.slice(0, 3));
      }
    });
    summary[entry.label] = { transmittance: worstT, relative: worstRelative, composite: worstComposite,
      accumulatedTransmittance: accumulatedT, accumulatedRelative };
  }
  // Opposite anisotropy signs separate under backward scattering: a phase sign error would swap them.
  const forward = result.cases.find(item => item.label === 'uniform-forward-g')!.raw[0];
  const backward = result.cases.find(item => item.label === 'uniform-backward-g')!.raw[0];
  expect(backward[0]).toBeGreaterThan(5 * forward[0]);
  // Overflow is clamped, never infinite.
  const peaked = result.cases.find(item => item.label === 'thin-and-peaked')!;
  expect(Math.max(...peaked.raw.map(value => value[0]))).toBe(65504);

  for (const value of result.disabledSplit) expect([...value.slice(0, 3).map(Math.abs), value[3]]).toEqual([0, 0, 0, 1]);
  result.disabled.hdr.forEach((value, index) => expect(value).toEqual(result.disabled.surface[index]));
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  console.log('FOG_F01 ' + JSON.stringify({ summary, adapter: result.adapter }));
});
