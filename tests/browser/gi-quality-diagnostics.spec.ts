import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { JOINT_LEAK_SHADOW_FOCUS } from '../../src/gi-joint-scene';
import type { runL01Diagnostics, runL04Diagnostics, runL04DirectLeak } from './fixtures/gi-quality-diagnostics';

type Fixture = typeof import('./fixtures/gi-quality-diagnostics');
async function fixturePage(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return errors;
}
async function call<K extends 'runL01Diagnostics' | 'runL04Diagnostics' | 'runL04DirectLeak'>(page: Page, name: K,
  input: Parameters<Fixture[K]>[0]): Promise<Awaited<ReturnType<Fixture[K]>>> {
  return await page.evaluate(async ({ name, input }) => {
    const module = await import('/tests/browser/fixtures/' + 'gi-quality-diagnostics.ts') as Fixture;
    return (module[name] as (value: unknown) => Promise<unknown>)(input);
  }, { name, input }) as Awaited<ReturnType<Fixture[K]>>;
}

// GPU indirect radiance = HDR(GI on) - HDR(GI off) = albedo * replayed irradiance. Both
// HDR values are rgba16float, so the tolerance is one half-float ULP of each plus 1%.
const ulp = (value: number) => 2 ** (Math.floor(Math.log2(Math.max(Math.abs(value), 2 ** -14))) - 10);
function replayRatios(receivers: number, albedo: number[][], on: Record<'multi' | 'single', number[][]>, off: number[][],
  replay: Record<'multi' | 'single', Record<string, number[][]>>, include: (index: number) => boolean = () => true) {
  let worst = 0, compared = 0;
  for (let i = 0; i < receivers; i++) {
    if (!include(i)) continue;
    compared++;
    for (const mode of ['multi', 'single'] as const) for (let c = 0; c < 3; c++) {
      const gpu = on[mode][i][c] - off[i][c], cpu = albedo[i][c] * replay[mode].production[i][c];
      const tolerance = ulp(on[mode][i][c]) + ulp(off[i][c]) + 0.01 * Math.abs(cpu);
      const ratio = Math.abs(gpu - cpu) / tolerance;
      if (process.env.GI_REPLAY_DEBUG && ratio > 1) console.log('GI_REPLAY_MISMATCH ' + JSON.stringify({ i, mode, c, gpu, cpu, ratio }));
      worst = Math.max(worst, ratio);
    }
  }
  return { worst, compared };
}

for (const refinement of ['none', 'grid-axis-neighbor'] as const) {
  test(`CPU probe-field replay reproduces GPU world GI at the frozen joint ROIs (refinement ${refinement})`, async ({ page }) => {
    test.setTimeout(600_000);
    const pageErrors = await fixturePage(page);
    const run = await call(page, 'runL01Diagnostics', { width: 320, height: 180, roiGrid: 3, warmup: 12, singleBounceWarmup: 4,
      grid: { raysPerProbe: 32, emissiveSamplesPerProbe: 4, visibilityRefinement: refinement }, oracleDirections: 64,
      emitterDensity: 4, references: [] });
    expect(run.errors).toEqual([]);
    expect(run.afterDispose).toBe(0);
    const { worst } = replayRatios(run.receivers.length, run.albedo, run.hdr, run.hdr.off, run.replay);
    console.log('GI_REPLAY_VALIDATION ' + JSON.stringify({ refinement, receivers: run.receivers.length, worstToleranceRatio: worst }));
    expect(run.receivers.length).toBeGreaterThan(30);
    expect(worst).toBeLessThanOrEqual(1);
    expect(pageErrors).toEqual([]);
  });
}

test('CPU replay reproduces GPU world GI with grid-axis-neighbor refinement inside the leak boxes', async ({ page }) => {
  test.setTimeout(600_000);
  const pageErrors = await fixturePage(page);
  // The closed white boxes converge slowly under feedback; HDR and the field are read
  // a few updates apart, so the multi-bounce field must be stationary (0.77^60 ~ 1e-7).
  const run = await call(page, 'runL04Diagnostics', { width: 320, height: 180, warmup: 60, samples: [16, 9],
    grid: { raysPerProbe: 32, visibilityRefinement: 'grid-axis-neighbor' } });
  expect(run.errors).toEqual([]);
  expect(run.afterDispose).toBe(0);
  const views = ['blackBox', 'thinBox', 'slitBox'] as const;
  const flat = (values: Record<string, number[][]>) => views.flatMap(view => values[view]);
  // Raster and ray hits can land on different faces within a pixel of an interior
  // edge; compare receivers at least 2 cm from every interior edge of their box.
  const centers = { black: -2.75, thin: 0.25, slit: 2.75 } as const;
  const interior = (i: number) => {
    const box = run.receiverBoxes[i];
    if (!box) return false;
    const p = run.receivers[i].position, local = [p[0] - centers[box], p[1] - 0.6, p[2]];
    const n = run.receivers[i].geometricNormal;
    return [0, 1, 2].every(axis => Math.abs(n[axis]) > 0.5 || Math.abs(local[axis]) < 0.48);
  };
  // Below the visibility weight threshold the support curve scales the result by
  // support^2 ~ visibility^6, which amplifies f32 and raster-position differences that
  // are first order elsewhere (SwiftShader 2026-10-31: the 3 compared receivers in this
  // regime reached 1.04-1.13x the tolerance; Apple 0.42x). They are counted, not compared.
  const supported = (i: number) => run.corners[i].support >= run.grid.visibilityWeightThreshold;
  const lowSupport = run.receivers.filter((_, i) => interior(i) && !supported(i)).length;
  const { worst, compared } = replayRatios(run.receivers.length, run.albedo, { multi: flat(run.hdr.multi), single: flat(run.hdr.single) },
    flat(run.hdr.off), run.replay, i => interior(i) && supported(i));
  console.log('GI_REPLAY_VALIDATION_LEAK ' + JSON.stringify({ receivers: run.receivers.length, compared, lowSupport,
    worstToleranceRatio: worst }));
  if (process.env.GI_REPLAY_DUMP) writeFileSync(process.env.GI_REPLAY_DUMP, JSON.stringify(run));
  expect(lowSupport).toBeLessThanOrEqual(8);
  expect(compared).toBeGreaterThan(100);
  expect(worst).toBeLessThanOrEqual(1);
  expect(pageErrors).toEqual([]);
});

test('joint GI L01/L04 error decomposition capture on hardware', async ({ page }, testInfo) => {
  test.skip(process.env.PLAYWRIGHT_GPU !== 'hardware' || !process.env.GI_QUALITY_OUT,
    'Diagnostic capture: set PLAYWRIGHT_GPU=hardware and GI_QUALITY_OUT.');
  test.setTimeout(3_600_000);
  const pageErrors = await fixturePage(page);
  const out = process.env.GI_QUALITY_OUT ?? testInfo.outputPath('gi-quality');
  mkdirSync(out, { recursive: true });
  // GI_QUALITY_GRID: JSON partial DiffuseGiOptions for controlled experiments.
  const grid = JSON.parse(process.env.GI_QUALITY_GRID ?? '{}');
  const leakGrid = JSON.parse(process.env.GI_QUALITY_LEAK_GRID ?? process.env.GI_QUALITY_GRID ?? '{}');
  const parts = (process.env.GI_QUALITY_PARTS ?? 'l01,l04,direct').split(',');
  // GI_QUALITY_DUMP=1 adds both probe fields and every probe ray for offline iteration.
  const dumpFields = process.env.GI_QUALITY_DUMP === '1';
  const references = JSON.parse(process.env.GI_QUALITY_REFERENCES ?? JSON.stringify([
    { paths: 512, bounces: 8, seed: 0 }, { paths: 512, bounces: 8, seed: 1 }, { paths: 2048, bounces: 24, seed: 2 }]));
  if (parts.includes('l01')) {
    const l01: Awaited<ReturnType<typeof runL01Diagnostics>> = await call(page, 'runL01Diagnostics', { width: 640, height: 360,
      roiGrid: 4, warmup: 240, singleBounceWarmup: 60, grid, oracleDirections: 4096, emitterDensity: 40, references, dumpFields });
    expect(l01.errors).toEqual([]); expect(l01.afterDispose).toBe(0);
    writeFileSync(join(out, 'l01-diagnostics.json'), JSON.stringify(l01));
  }
  if (parts.includes('l04')) {
    const l04: Awaited<ReturnType<typeof runL04Diagnostics>> = await call(page, 'runL04Diagnostics', { width: 640, height: 360,
      warmup: 120, samples: [16, 9], grid: leakGrid, dumpFields });
    expect(l04.errors).toEqual([]); expect(l04.afterDispose).toBe(0);
    writeFileSync(join(out, 'l04-diagnostics.json'), JSON.stringify(l04));
  }
  if (parts.includes('direct')) {
    const bounds = JOINT_LEAK_SHADOW_FOCUS;
    const direct: Awaited<ReturnType<typeof runL04DirectLeak>> = await call(page, 'runL04DirectLeak', { width: 640, height: 360,
      samples: [16, 9], shadows: [{ name: 'legacy-1024' }, { name: 'legacy-4096', options: { resolution: 4096 } },
        { name: 'focus-1024', options: { focusBounds: bounds } }, { name: 'focus-4096', options: { focusBounds: bounds, resolution: 4096 } }] });
    writeFileSync(join(out, 'l04-direct.json'), JSON.stringify(direct));
  }
  expect(pageErrors).toEqual([]);
});
