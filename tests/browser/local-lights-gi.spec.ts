import { expect, test, type Page } from '@playwright/test';
import type { LocalLightDescriptor } from '../../src/local-lights';
import type {
  LocalLightGatherPlan, LocalLightGiProbePlan, runLocalLightGatherFixture, runLocalLightGiProbeFixture,
} from './fixtures/local-lights-gi';

type Vec3 = [number, number, number];
test.use({ deviceScaleFactor: 1 });

// --- Independent oracle --------------------------------------------------------------------------
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
function fibonacci(index: number, count: number): Vec3 {
  const y = 1 - 2 * (index + 0.5) / count, radius = Math.sqrt(Math.max(0, 1 - y * y)), phi = index * GOLDEN_ANGLE;
  return [Math.cos(phi) * radius, y, Math.sin(phi) * radius];
}
function basis([x, y, z]: Vec3): number[] {
  return [0.28209479, 0.48860251 * y, 0.48860251 * z, 0.48860251 * x, 1.09254843 * x * y, 1.09254843 * y * z,
    0.31539157 * (3 * z * z - 1), 1.09254843 * x * z, 0.546274215 * (x * x - y * y)];
}
function rangeWindow(distance: number, range: number): number {
  const ratio = (distance * distance) / (range * range);
  return Math.min(1, Math.max(0, 1 - ratio * ratio)) ** 2;
}
/** Point light E⊥ × max(n·l, 0) on a +Z floor. */
function floorIrradiance(light: LocalLightDescriptor, hit: Vec3): Vec3 {
  const delta = sub(light.position, hit), d2 = dot(delta, delta), d = Math.sqrt(d2);
  if (d >= light.range || delta[2] <= 0) return [0, 0, 0];
  const scale = light.intensity * rangeWindow(d, light.range) / Math.max(d2, 1e-4) * (delta[2] / d);
  return light.color.map(c => c * scale) as Vec3;
}

const giLight: LocalLightDescriptor = { kind: 'point', position: [2, 0.5, 0.8], color: [1, 0.9, 0.7], intensity: 4, range: 2.5 };
const giPlan: LocalLightGiProbePlan = {
  // Plate chosen so every GI ray stays > 5 cm (in the plate plane) from the shadow boundary and > 10 cm from the plate edges.
  floorHalf: 6, floorColor: [0.7, 0.6, 0.5, 1], plate: { center: [2.4, 0.3, 0.3], half: 0.5 },
  grid: { origin: [-1, -1, 0.25], spacing: [2, 2, 0.75], counts: [2, 2, 2], raysPerProbe: 128, maxDistance: 20,
    skyRadiance: [0, 0, 0], multiBounce: false },
  light: giLight,
  // The camera looks straight down at floor x in [-1.53, -0.47]; every visible point is > 2.5 m from the light.
  eye: [-1, 0, 1], target: [-1, 0.001, 0],
  pixels: [{ x: 48, y: 48 }, { x: 20, y: 20 }, { x: 76, y: 20 }, { x: 20, y: 76 }, { x: 76, y: 76 }],
};

type RayCase = { probe: number; ray: number; kind: 'sky' | 'floor' | 'plate'; hit?: Vec3; shadowed: boolean;
  floorMargin: number; plateMargin: number; shadowMargin: number };
/** Analytic first hits of every GI ray against the floor and the double-sided black plate. */
function giRays(plan: LocalLightGiProbePlan, plate: boolean): RayCase[] {
  const { origin, spacing, counts } = plan.grid, rays = plan.grid.raysPerProbe!, cases: RayCase[] = [];
  const px = plan.plate.center, half = plan.plate.half;
  for (let probe = 0; probe < counts[0] * counts[1] * counts[2]; probe++) {
    const coordinate = [probe % counts[0], Math.floor(probe / counts[0]) % counts[1], Math.floor(probe / (counts[0] * counts[1]))];
    const p = origin.map((value, axis) => value + coordinate[axis] * spacing[axis]) as Vec3;
    for (let ray = 0; ray < rays; ray++) {
      const d = fibonacci(ray, rays);
      let best = Infinity, kind: RayCase['kind'] = 'sky', hit: Vec3 | undefined;
      let floorMargin = Infinity, plateMargin = Infinity;
      if (d[2] < 0) {
        const t = -p[2] / d[2], h: Vec3 = [p[0] + d[0] * t, p[1] + d[1] * t, 0];
        floorMargin = Math.min(plan.floorHalf - Math.abs(h[0]), plan.floorHalf - Math.abs(h[1]));
        if (t <= plan.grid.maxDistance! && floorMargin > 0) { best = t; kind = 'floor'; hit = h; }
      }
      if (plate && d[2] !== 0) {
        const t = (px[2] - p[2]) / d[2];
        if (t > 0) {
          const h = [p[0] + d[0] * t, p[1] + d[1] * t];
          plateMargin = half - Math.max(Math.abs(h[0] - px[0]), Math.abs(h[1] - px[1]));
          if (plateMargin > 0 && t < best) { best = t; kind = 'plate'; hit = [h[0], h[1], px[2]]; }
          plateMargin = Math.abs(plateMargin);
        }
      }
      let shadowed = false, shadowMargin = Infinity;
      if (kind === 'floor' && plate) {
        const l = plan.light.position, s = (px[2] - hit![2]) / (l[2] - hit![2]);
        const crossing = [hit![0] + (l[0] - hit![0]) * s, hit![1] + (l[1] - hit![1]) * s];
        const inside = half - Math.max(Math.abs(crossing[0] - px[0]), Math.abs(crossing[1] - px[1]));
        shadowed = inside > 0; shadowMargin = Math.abs(inside);
      }
      cases.push({ probe, ray, kind, hit, shadowed, floorMargin: Math.abs(floorMargin), plateMargin, shadowMargin });
    }
  }
  return cases;
}
function expectedProbes(plan: LocalLightGiProbePlan, plate: boolean): number[][][] {
  const rays = plan.grid.raysPerProbe!, counts = plan.grid.counts;
  const result = Array.from({ length: counts[0] * counts[1] * counts[2] }, () => Array.from({ length: 9 }, () => [0, 0, 0]));
  for (const item of giRays(plan, plate)) {
    if (item.kind !== 'floor' || item.shadowed) continue;
    const e = floorIrradiance(plan.light, item.hit!), y = basis(fibonacci(item.ray, rays));
    for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) {
      result[item.probe][k][c] += plan.floorColor[c] * (0.96 / Math.PI) * e[c] * y[k] * (4 * Math.PI / rays);
    }
  }
  return result;
}
function compareProbes(actual: { coefficients: number[][]; valid: boolean }[], expected: number[][][], label: string) {
  let worst = 0;
  expected.forEach((probe, index) => {
    expect(actual[index].valid, `${label} probe ${index} valid`).toBe(true);
    const scale = Math.max(1e-6, ...probe.flat().map(Math.abs));
    for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) {
      const error = Math.abs(actual[index].coefficients[k][c] - probe[k][c]);
      expect(error, `${label} probe ${index} coefficient ${k}:${c} actual ${actual[index].coefficients[k][c]} expected ${probe[k][c]}`)
        .toBeLessThanOrEqual(2e-3 * scale + 1e-6);
      worst = Math.max(worst, error / scale);
    }
  });
  return worst;
}

async function openPage(page: Page) {
  page.on('console', message => { if (message.type() === 'error') console.log('LOCAL_LIGHTS_GI_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
}
function adapterChecks(adapter: { vendor: string; isFallbackAdapter: boolean | null }) {
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
}

test('an off-screen point light reaches world GI probes with exact SH and atlas shadows', async ({ page }) => {
  test.setTimeout(180_000);
  await openPage(page);
  const result = await page.evaluate(async plan => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'local-lights-gi.ts') as { runLocalLightGiProbeFixture: typeof runLocalLightGiProbeFixture };
    return module.runLocalLightGiProbeFixture(plan);
  }, giPlan);
  expect(result.errors).toEqual([]);
  adapterChecks(result.adapter);

  // The plan stays away from discontinuities: floor edges, plate edges and the plate's shadow boundary.
  const open = giRays(giPlan, false), plated = giRays(giPlan, true);
  expect(Math.min(...open.filter(item => item.kind === 'floor').map(item => item.floorMargin))).toBeGreaterThan(0.01);
  expect(Math.min(...plated.map(item => item.plateMargin))).toBeGreaterThan(0.005);
  expect(Math.min(...plated.map(item => item.shadowMargin))).toBeGreaterThan(0.02);
  const shadowedRays = plated.filter(item => item.shadowed).length, plateRays = plated.filter(item => item.kind === 'plate').length;
  expect(shadowedRays).toBeGreaterThan(3);
  expect(plateRays).toBeGreaterThan(0);
  // Rays from every probe hit floor points lit by the light, including grazing angles up to the range.
  const lit = open.filter(item => item.kind === 'floor' && floorIrradiance(giLight, item.hit!)[0] > 0);
  expect(new Set(lit.map(item => item.probe)).size).toBe(8);

  for (const probe of result.dark.probes) for (const coefficient of probe.coefficients) expect(coefficient).toEqual([0, 0, 0]);
  const unshadowedError = compareProbes(result.unshadowed.probes, expectedProbes(giPlan, false), 'unshadowed');
  const openError = compareProbes(result.shadowedOpen.probes, expectedProbes(giPlan, false), 'shadowed without occluder');
  const plateError = compareProbes(result.shadowedPlate.probes, expectedProbes(giPlan, true), 'atlas shadow from plate');
  expect(result.shadowedRestored.probes).toEqual(result.shadowedOpen.probes);
  // The plate changes probes only through the shadow atlas and its own black hits.
  expect(result.shadowedPlate.probes).not.toEqual(result.shadowedOpen.probes);
  // Full GI updates treat a local light change as a global lighting change (renderer.markDiffuseGiDirty()).
  expect(result.unshadowed.resetReason).toBe('external-change');

  // Off-screen: without GI the visible image ignores the light bit for bit; with GI it brightens.
  expect(result.giOffLit.hdr).toEqual(result.giOffDark.hdr);
  // Single bounce: probe SH (light × floor albedo) feeds visible floor diffuse (× albedo again). Every
  // per-probe weight and the ringing clamp are channel-shared, so the added colour is light × albedo²
  // up to two rgba16float roundings.
  const chroma = [0, 1, 2].map(c => giLight.color[c] * giPlan.floorColor[c] ** 2);
  for (let i = 0; i < giPlan.pixels.length; i++) {
    expect(result.dark.hdr[i].slice(0, 3)).toEqual([0, 0, 0]);
    const added = result.unshadowed.hdr[i];
    for (let c = 0; c < 3; c++) {
      expect(added[c], `pixel ${i} channel ${c}`).toBeGreaterThan(1e-4);
      expect(Math.abs(added[c] / added[0] - chroma[c] / chroma[0]), `pixel ${i} chroma ${c}`).toBeLessThan(2e-3);
    }
  }
  expect(result.lightDiagnostics).toMatchObject({ cameraCulling: false, shadowedLights: 1, renderedShadowViews: 6 });

  // Scheduled GI: the in-place intensity change reports one range box; every probe is within reach of it.
  const base = result.scheduledBase.schedule, first = result.scheduledFirst;
  expect(result.scheduledBase.resetReason).toBe('update-policy-changed');
  expect(base.pending).toBe(0);
  const scheduledBaseError = compareProbes(result.scheduledBase.probes, expectedProbes(giPlan, false), 'scheduled baseline');
  expect(first.resetReason).toBe('update-policy-changed');
  expect(first.external - base.external).toBe(1);
  expect(first.totals.regionMarks - base.totals.regionMarks).toBe(1);
  expect(first.totals.markedProbes - base.totals.markedProbes).toBe(8);
  expect(first.totals.globalMarks).toBe(base.totals.globalMarks);
  expect(first.totals.fullRefreshes).toBe(base.totals.fullRefreshes);
  expect(first.dirtyLastUpdate).toBeGreaterThan(0);
  const doubled = { ...giPlan, light: { ...giLight, intensity: 2 * giLight.intensity } };
  expect(result.scheduledDoubled.schedule.pending).toBe(0);
  expect(result.scheduledDoubled.resetReason).toBe('update-policy-changed');
  const scheduledError = compareProbes(result.scheduledDoubled.probes, expectedProbes(doubled, false), 'scheduled after doubling');
  expect(result.afterDispose).toBe(0);
  console.log('LOCAL_LIGHTS_GI_DIAGNOSTIC ' + JSON.stringify({ adapter: result.adapter, unshadowedError, openError, plateError,
    scheduledBaseError, scheduledError, scheduledFrames: 1 + result.scheduledDoubled.frames,
    shadowedRays, plateRays, litRays: lit.length, hdr: { dark: result.dark.hdr, lit: result.unshadowed.hdr } }));
});

const gatherLight: LocalLightDescriptor = { kind: 'point', position: [0, 0, 0], color: [1, 1, 1], intensity: 3, range: 10 };
const gatherPlan: LocalLightGatherPlan = {
  radius: 2, subdivisions: 4, color: [0.8, 0.7, 0.6, 1], light: gatherLight,
  grid: { origin: [-0.6, -0.6, -0.6], spacing: [1.2, 1.2, 1.2], counts: [2, 2, 2], raysPerProbe: 64, maxDistance: 10,
    skyRadiance: [0, 0, 0], multiBounce: false,
    finalGather: { pixelSpacing: 4, maxAnchors: 1024, raysPerAnchor: 8, pathSegments: 1 } },
  eye: [0, 0.3, 0.8], target: [0.2, -0.2, -2], size: 64, anchors: Array.from({ length: 256 }, (_, i) => i),
};

test('Final Gather and GI capture see a uniform local-light radiance field inside a closed sphere', async ({ page }) => {
  test.setTimeout(240_000);
  await openPage(page);
  const result = await page.evaluate(async plan => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'local-lights-gi.ts') as { runLocalLightGatherFixture: typeof runLocalLightGatherFixture };
    return module.runLocalLightGatherFixture(plan);
  }, gatherPlan);
  expect(result.errors).toEqual([]);
  adapterChecks(result.adapter);
  const { minPlane, radius } = result.geometry;
  // Every gathered hit lies on a facet with plane distance P in [minPlane, R] at distance d in [P, R]:
  // E n·l = I W(d) P / d^3, so the uniform outgoing radiance L = ρ 0.96/π E n·l is bounded on both sides.
  const bounds = (intensity: number) => {
    const low = intensity * rangeWindow(radius, gatherLight.range) * minPlane / radius ** 3;
    const high = intensity * rangeWindow(minPlane, gatherLight.range) / minPlane ** 2;
    return [0, 1, 2].map(c => [gatherPlan.color[c] * 0.96 / Math.PI * low, gatherPlan.color[c] * 0.96 / Math.PI * high]);
  };
  // The tessellated sphere keeps the bound tight: a missing or mis-scaled light term cannot pass.
  expect(bounds(3)[0][1] / bounds(3)[0][0]).toBeLessThan(1.004);
  const meanBasis = Array.from({ length: 9 }, (_, k) => {
    const count = gatherPlan.grid.raysPerProbe!;
    let sum = 0;
    for (let i = 0; i < count; i++) sum += basis(fibonacci(i, count))[k];
    return sum / count;
  });
  const check = (capture: typeof result.unshadowed, intensity: number, label: string) => {
    const b = bounds(intensity);
    let anchors = 0;
    for (const anchor of capture.anchors) {
      expect(anchor.valid, `${label} anchor valid`).toBe(true);
      anchors++;
      for (let c = 0; c < 3; c++) {
        for (const value of [anchor.irradiance[c], anchor.filtered[c]]) {
          expect(value, `${label} gather ${c}`).toBeGreaterThanOrEqual(Math.PI * b[c][0] * (1 - 1e-3));
          expect(value, `${label} gather ${c}`).toBeLessThanOrEqual(Math.PI * b[c][1] * (1 + 1e-3));
        }
      }
    }
    for (const probe of capture.probes) {
      expect(probe.valid).toBe(true);
      for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) {
        // c_k = 4π mean(L Y_k): a DC term plus the sampling pattern's own mean of Y_k.
        const lowValue = 4 * Math.PI * Math.min(b[c][0] * meanBasis[k], b[c][1] * meanBasis[k]);
        const highValue = 4 * Math.PI * Math.max(b[c][0] * meanBasis[k], b[c][1] * meanBasis[k]);
        const slack = 4 * Math.PI * (b[c][1] - b[c][0]) * 1.1 + 1e-3 * 4 * Math.PI * b[c][1] * 0.2821;
        expect(probe.coefficients[k][c], `${label} SH ${k}:${c}`).toBeGreaterThanOrEqual(lowValue - slack);
        expect(probe.coefficients[k][c], `${label} SH ${k}:${c}`).toBeLessThanOrEqual(highValue + slack);
      }
    }
    return anchors;
  };
  for (const anchor of result.dark.anchors) expect(anchor.irradiance).toEqual([0, 0, 0]);
  const anchorCount = check(result.unshadowed, 3, 'unshadowed');
  check(result.unshadowedFirst, 3, 'first frame after adding the light');
  check(result.shadowed, 3, 'cube shadow on curved receivers');
  check(result.shadowedFirst, 3, 'first frame after enabling shadows');
  check(result.doubled, 6, 'doubled intensity');
  expect(anchorCount).toBe(256);
  expect(result.unshadowedFirst.giReset).toBe('external-change');
  expect(result.shadowedFirst.giReset).toBe('external-change');
  expect(result.unshadowedFirst.gatherHistory).toBeLessThanOrEqual(1);
  expect(result.afterDispose).toBe(0);
  const mean = (capture: typeof result.unshadowed) => capture.anchors.reduce((sum, anchor) => sum + anchor.irradiance[0], 0) / capture.anchors.length;
  console.log('LOCAL_LIGHTS_GATHER_DIAGNOSTIC ' + JSON.stringify({ adapter: result.adapter, geometry: result.geometry,
    bounds: bounds(3).map(item => item.map(value => value * Math.PI)), meanIrradiance: { unshadowed: mean(result.unshadowed),
      shadowed: mean(result.shadowed), doubled: mean(result.doubled) }, giReset: result.unshadowedFirst.giReset,
    gatherReset: result.unshadowedFirst.gatherReset, gatherHistory: result.unshadowedFirst.gatherHistory,
    dc: result.unshadowed.probes[0].coefficients[0] }));
});
