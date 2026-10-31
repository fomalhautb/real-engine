import { expect, test } from '@playwright/test';
import type { LocalLightDescriptor } from '../../src/local-lights';
import type { LocalLightAirPlan, LocalLightFogPlan, runLocalLightAirFixture, runLocalLightFogFixture } from './fixtures/local-lights-fog';
import { dot, henyeyGreenstein, irradiance, normalize, type Vec3 } from './local-light-oracle';

test.use({ deviceScaleFactor: 1 });
type Result = Awaited<ReturnType<typeof runLocalLightFogFixture>>;

const GLOBAL = { extinction: 0.15, albedo: [0.9, 0.8, 0.7] as Vec3, anisotropy: 0.6, baseHeight: 0.5, heightFalloff: 0.4 };
const BOX = { min: [-1.5, -1.5, -5.05] as Vec3, max: [1.5, 0.33, -2.95] as Vec3, extinction: 0.4, albedo: [0.5, 0.7, 0.9] as Vec3, anisotropy: -0.3 };
// Off-screen point light above the view; a black plate (also off-screen) casts a shadow shaft down into the fog.
const pointLight: LocalLightDescriptor = { kind: 'point', position: [0, 3, -2.5], color: [1, 0.9, 0.8], intensity: 20, range: 4 };
// Spot inside the view volume aimed back at the camera: in-scattering towards the eye is the HG forward peak.
const spotLight: LocalLightDescriptor = { kind: 'spot', position: [0.3, -0.2, -6.5], direction: [-0.05, 0.1, 1], innerConeAngle: 0.2,
  outerConeAngle: 0.5, color: [0.8, 0.9, 1], intensity: 6, range: 6 };
const plan: LocalLightFogPlan = {
  size: 64, eye: [0, 0, 0], target: [0, 0, -1],
  fog: { pixelSize: 8, slices: 16, maxViewDepth: 8, temporal: { maxHistoryFrames: 8 },
    global: { coefficients: { extinction: GLOBAL.extinction, albedo: GLOBAL.albedo, emission: [0, 0, 0], anisotropy: GLOBAL.anisotropy },
      baseHeight: GLOBAL.baseHeight, heightFalloff: GLOBAL.heightFalloff },
    volumes: [{ id: 1, shape: { kind: 'box', min: BOX.min, max: BOX.max },
      coefficients: { extinction: BOX.extinction, albedo: BOX.albedo, emission: [0, 0, 0], anisotropy: BOX.anisotropy } }] },
  pointLight, spotLight, plate: { x: 0, y: 2.75, z: -2.71, half: 0.25 },
  pixels: Array.from({ length: 64 }, (_, i) => ({ x: (i % 8) * 8 + 2 + (i % 3), y: Math.floor(i / 8) * 8 + 6 - (i % 5) })),
  grid: { columns: 8, rows: 8, slices: 16 },
};
type Grid = { C: number; R: number; S: number };
const ENGINE: Grid = { C: 8, R: 8, S: 16 }, FINE: Grid = { C: 64, R: 64, S: 1024 };
const { C, R, S } = ENGINE, D = 8, TAN = Math.tan(Math.PI / 6.4);

// --- Independent reference: media, radiometry, HG phase and the plate's geometric shadow ------------
type Light = LocalLightDescriptor & { shadowed?: boolean };
/** Unnormalised view ray whose camera-forward component is 1 (camera at the origin looking down -Z). */
const viewRay = (u: number, v: number): Vec3 => [(2 * u - 1) * TAN, (1 - 2 * v) * TAN, -1];
/** Quadratic depth slices to the 8 m froxel depth. */
const sliceDepth = (z: number, grid: Grid = ENGINE) => D * (z / grid.S) ** 2;
function inBox(p: Vec3) { return p.every((value, axis) => value >= BOX.min[axis] && value <= BOX.max[axis]); }
function plateCrossing(p: Vec3, light: LocalLightDescriptor): number {
  const { x, y, z, half } = plan.plate;
  if (p[1] >= y || light.position[1] <= y) return -Infinity;
  const s = (y - p[1]) / (light.position[1] - p[1]);
  return half - Math.max(Math.abs(p[0] + (light.position[0] - p[0]) * s - x), Math.abs(p[2] + (light.position[2] - p[2]) * s - z));
}
/** σ_t and in-scattered radiance per metre at p for view direction ray. */
function source(p: Vec3, ray: Vec3, lights: readonly Light[]): [number, number, number, number] {
  const density = Math.exp(-GLOBAL.heightFalloff * Math.max(p[1] - GLOBAL.baseHeight, 0)), box = inBox(p);
  const direction = normalize(ray), out: [number, number, number, number] = [0, 0, 0, density * GLOBAL.extinction + (box ? BOX.extinction : 0)];
  for (const light of lights) {
    const { l, e } = irradiance(light, p);
    if (light.shadowed && plateCrossing(p, light) > 0) continue;
    const cosine = dot(direction, l);
    for (let c = 0; c < 3; c++) {
      const scattering = density * GLOBAL.extinction * GLOBAL.albedo[c] * henyeyGreenstein(cosine, GLOBAL.anisotropy)
        + (box ? BOX.extinction * BOX.albedo[c] * henyeyGreenstein(cosine, BOX.anisotropy) : 0);
      out[c] += scattering * e[c];
    }
  }
  return out;
}
function cellCentre(x: number, y: number, z: number, grid: Grid = ENGINE) {
  const ray = viewRay((x + 0.5) / grid.C, (y + 0.5) / grid.R), depth = (sliceDepth(z, grid) + sliceDepth(z + 1, grid)) / 2;
  return { ray, p: ray.map(value => value * depth) as Vec3 };
}
function expectedSources(lights: readonly Light[]) {
  const list: [number, number, number, number][] = [];
  for (let z = 0; z < S; z++) for (let y = 0; y < R; y++) for (let x = 0; x < C; x++) {
    const { ray, p } = cellCentre(x, y, z);
    list.push(source(p, ray, lights));
  }
  return list;
}
/** Piecewise-constant Beer-Lambert segment, front-to-back. */
function over(front: number[], cell: number[], distance: number): number[] {
  const tau = cell[3] * distance, transmission = Math.exp(-tau);
  const integral = cell[3] > 0 ? (1 - transmission) / cell[3] : distance;
  return [0, 1, 2].map(c => front[c] + front[3] * cell[c] * integral).concat(front[3] * transmission);
}
/** The froxel quadrature on the CPU: per-column slices to the 8 m depth, bilinear across columns.
 * Returns [scattered rgb, transmittance]. */
function froxelMedium(sourceAt: (x: number, y: number, z: number) => number[], pixel: { x: number; y: number }, grid: Grid): number[] {
  const gx = (pixel.x + 0.5) / plan.size * grid.C - 0.5, gy = (pixel.y + 0.5) / plan.size * grid.R - 0.5;
  const bx = Math.floor(gx), by = Math.floor(gy), fx = gx - bx, fy = gy - by;
  const column = (x: number, y: number) => {
    const cx = Math.min(Math.max(x, 0), grid.C - 1), cy = Math.min(Math.max(y, 0), grid.R - 1);
    const length = Math.hypot(...viewRay((cx + 0.5) / grid.C, (cy + 0.5) / grid.R));
    let value = [0, 0, 0, 1];
    for (let z = 0; z < grid.S; z++) value = over(value, sourceAt(cx, cy, z), (sliceDepth(z + 1, grid) - sliceDepth(z, grid)) * length);
    return value;
  };
  const mix = (a: number[], b: number[], t: number) => a.map((value, i) => value + (b[i] - value) * t);
  return mix(mix(column(bx, by), column(bx + 1, by), fx), mix(column(bx, by + 1), column(bx + 1, by + 1), fx), fy);
}
const composite = (medium: readonly number[], surface: readonly number[]) => [0, 1, 2].map(c => medium[c] + medium[3] * surface[c]);
/** Converged single-scattering integral along the exact pixel ray (midpoint rule, 4000 steps to 8 m depth). */
function continuousMedium(pixel: { x: number; y: number }, lights: readonly Light[]): number[] {
  const ray = viewRay((pixel.x + 0.5) / plan.size, (pixel.y + 0.5) / plan.size), length = Math.hypot(...ray), steps = 4000;
  let value = [0, 0, 0, 1];
  for (let i = 0; i < steps; i++) {
    const depth = (i + 0.5) * D / steps;
    value = over(value, source(ray.map(component => component * depth) as Vec3, ray, lights), D / steps * length);
  }
  return value;
}

test('local lights scatter in froxel fog with HG phase, atlas shadow and history invalidation', async ({ page }) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result: Result = await page.evaluate(async input => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'local-lights-fog.ts') as { runLocalLightFogFixture: typeof runLocalLightFogFixture };
    return fixture.runLocalLightFogFixture(input);
  }, plan);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(result.diagnostics.grid).toMatchObject({ columns: C, rows: R, slices: S });
  // Plan guards: cell centres stay > 0.02 rad (≈5 atlas texels) from the plate's shadow edge and > 1 mm from box faces.
  let shadowCells = 0, litCells = 0, spotCells = 0;
  for (let z = 0; z < S; z++) for (let y = 0; y < R; y++) for (let x = 0; x < C; x++) {
    const { p, ray } = cellCentre(x, y, z);
    for (let axis = 0; axis < 3; axis++) expect(Math.min(Math.abs(p[axis] - BOX.min[axis]), Math.abs(p[axis] - BOX.max[axis]))).toBeGreaterThan(1e-3);
    if (irradiance(spotLight, p).e[0] > 0 && dot(normalize(ray), irradiance(spotLight, p).l) > 0.9) spotCells++;
    if (irradiance(pointLight, p).e[0] <= 0) continue;
    const margin = plateCrossing(p, pointLight);
    if (margin === -Infinity) { litCells++; continue; }
    const s = (plan.plate.y - p[1]) / (pointLight.position[1] - p[1]);
    const reach = Math.hypot(...[0, 1, 2].map(axis => (pointLight.position[axis] - p[axis]) * (1 - s)));
    expect(Math.abs(margin) / reach).toBeGreaterThan(0.02);
    if (margin > 0) shadowCells++; else litCells++;
  }
  expect(shadowCells).toBeGreaterThan(50);
  expect(litCells).toBeGreaterThan(200);
  expect(spotCells).toBeGreaterThan(20);

  // Light changes reset the fog history explicitly; a frame without changes does not.
  expect(result.lit.fog).toMatchObject({ resetReason: 'local-light-change', framesSinceReset: 1,
    localLightScattering: 'froxel-block-light-list-shadowed-single-scattering' });
  expect(result.settled).toMatchObject({ resetReason: 'local-light-change', framesSinceReset: 2 });
  expect(result.pointOff.fog).toMatchObject({ resetReason: 'local-light-change', framesSinceReset: 1 });
  expect(result.spotDouble.fog).toMatchObject({ resetReason: 'local-light-change', framesSinceReset: 1 });
  expect(result.shadowed.fog.framesSinceReset).toBe(1);
  console.log('LOCAL_LIGHTS_FOG_STATES ' + JSON.stringify([result.dark.fog, result.lit.fog, result.settled, result.shadowed.fog, result.pointOff.fog, result.spotDouble.fog]));
  const compare = (capture: Result['dark'], lights: readonly Light[], label: string) => {
    const expected = expectedSources(lights);
    let worst = 0, zeros = 0, scattering = 0;
    capture.sources.forEach(cell => {
      const index = (cell.z * R + cell.y) * C + cell.x, reference = expected[index];
      expect(Math.abs(cell.source[3] - reference[3]), `${label} cell ${index} extinction`).toBeLessThanOrEqual(1e-5 * reference[3]);
      for (let c = 0; c < 3; c++) {
        if (reference[c] === 0) { expect(cell.source[c], `${label} cell ${index} exact zero ${c}`).toBe(0); continue; }
        const error = Math.abs(cell.source[c] - reference[c]);
        expect(error, `${label} cell ${index}:${c} actual ${cell.source[c]} expected ${reference[c]}`).toBeLessThanOrEqual(1e-3 * reference[c] + 1e-9);
        worst = Math.max(worst, error / reference[c]);
      }
      if (reference[0] === 0) zeros++; else scattering++;
    });
    let hdrWorst = 0, engineGap = 0, fineGap = 0;
    const fineSource = (x: number, y: number, z: number) => { const { ray, p } = cellCentre(x, y, z, FINE); return source(p, ray, lights); };
    plan.pixels.forEach((pixel, index) => {
      const engine = froxelMedium((x, y, z) => expected[(z * R + y) * C + x], pixel, ENGINE), froxel = composite(engine, capture.surface[index]);
      for (let c = 0; c < 3; c++) {
        // rgba16float rounds the composite by ≤ 2^-11; the f32 chain adds far less.
        const error = Math.abs(capture.hdr[index][c] - froxel[c]);
        expect(error, `${label} pixel ${pixel.x},${pixel.y}:${c} actual ${capture.hdr[index][c]} froxel ${froxel[c]}`)
          .toBeLessThanOrEqual(2e-3 * Math.abs(froxel[c]) + 1e-5);
        hdrWorst = Math.max(hdrWorst, error / Math.max(Math.abs(froxel[c]), 1e-5));
      }
      if (!lights.length || index % 4) return;
      // The same quadrature refined to 64×64×1024 converges to the exact integral; the 8×8×16 engine grid is
      // its documented coarse discretization (reported, not asserted).
      const continuous = continuousMedium(pixel, lights), fine = froxelMedium(fineSource, pixel, FINE);
      for (let c = 0; c < 3; c++) {
        if (continuous[c] < 1e-4) continue;
        const fineError = Math.abs(fine[c] - continuous[c]) / continuous[c];
        expect(fineError, `${label} pixel ${pixel.x},${pixel.y}:${c} fine quadrature ${fine[c]} exact ${continuous[c]}`).toBeLessThan(0.03);
        fineGap = Math.max(fineGap, fineError);
        engineGap = Math.max(engineGap, Math.abs(engine[c] - continuous[c]) / continuous[c]);
      }
    });
    return { label, worst, zeros, scattering, hdrWorst, fineGap, engineGap };
  };
  const shadowedPoint: Light = { ...pointLight, shadowed: true };
  const summary = [
    compare(result.dark, [], 'dark'),
    compare(result.lit, [pointLight, spotLight], 'lit'),
    compare(result.shadowedAgain, [shadowedPoint, { ...spotLight, intensity: spotLight.intensity * 1.5 }], 'shadowed again'),
    compare(result.shadowed, [shadowedPoint, spotLight], 'shadowed'),
    compare(result.pointOff, [{ ...spotLight, intensity: spotLight.intensity * 1.5 }], 'point off'),
    compare(result.spotDouble, [{ ...spotLight, intensity: spotLight.intensity * 2 }], 'spot doubled'),
  ];
  for (const item of summary.slice(1)) expect(item.scattering).toBeGreaterThan(150);
  expect(result.diagnostics.localLights).toMatchObject({ cameraCulling: false, lights: 2, enabledLights: 1 });
  expect(result.afterDispose).toBe(0);
  console.log('LOCAL_LIGHTS_FOG_DIAGNOSTIC ' + JSON.stringify({ adapter: result.adapter, summary, shadowCells, litCells, spotCells,
    shadowedReset: result.shadowed.fog.resetReason, darkReset: result.dark.fog, timings: result.diagnostics.timings,
    counts: [...new Set(result.lit.sources.map(cell => cell.count))] }));
});

test('local light fog scattering adds to GI air lighting in one fog module, beside reflected fog', async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  // Empty scene: every GI ray reaches the sky, so the air term cannot depend on the local light.
  const airPlan: LocalLightAirPlan = {
    size: 32, eye: [0, 0, 3], target: [0, 0, 0],
    medium: { extinction: 0.3, albedo: [0.8, 0.8, 0.8], emission: [0, 0, 0], anisotropy: 0 },
    grid: { origin: [-4.5, -4.5, -2.5], spacing: [3, 3, 3], counts: [4, 4, 4], raysPerProbe: 64, maxDistance: 20, skyRadiance: [0.2, 0.5, 1] },
    light: { kind: 'point', position: [0.3, 0.2, -1], color: [1, 0.9, 0.8], intensity: 6, range: 4, castShadows: true },
    pixels: [{ x: 16, y: 16 }, { x: 18, y: 14 }, { x: 12, y: 20 }, { x: 24, y: 8 }],
  };
  const result = await page.evaluate(async input => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'local-lights-fog.ts') as { runLocalLightAirFixture: typeof runLocalLightAirFixture };
    return fixture.runLocalLightAirFixture(input);
  }, airPlan);
  const { air, plain } = result;
  expect(air.errors).toEqual([]);
  expect(plain.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(air.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  expect(air.diagnostics.fog).toMatchObject({ airLighting: { configured: true },
    localLightScattering: 'froxel-block-light-list-shadowed-single-scattering' });
  expect(plain.diagnostics.fog!.airLighting).toBeNull();
  for (const variant of [air, plain]) {
    expect(variant.diagnostics.localLights).toMatchObject({ enabledLights: 1, renderedShadowViews: 6, cameraCulling: false });
    expect(variant.diagnostics.reflections).toMatchObject({ localLights: true });
    expect(variant.afterDispose).toBe(0);
  }
  let worst = 0;
  airPlan.pixels.forEach((pixel, index) => {
    const label = `pixel ${pixel.x},${pixel.y}`;
    // No light and no air: exactly dark. With air lighting the sky term is present before the light.
    expect(plain.dark[index].slice(0, 3), label).toEqual([0, 0, 0]);
    for (let c = 0; c < 3; c++) {
      expect(air.dark[index][c], `${label}:${c} air term`).toBeGreaterThan(1e-3);
      const withAir = air.lit[index][c] - air.dark[index][c], alone = plain.lit[index][c] - plain.dark[index][c];
      expect(alone, `${label}:${c} local light`).toBeGreaterThan(1e-3);
      // Additive: half-float rounding of each probe (<= 2^-11 relative) bounds the difference of differences.
      const error = Math.abs(withAir - alone), bound = 2e-3 * alone + 1e-3 * (air.lit[index][c] + air.dark[index][c]) + 1e-5;
      expect(error, `${label}:${c} local light with ${withAir} alone ${alone}`).toBeLessThanOrEqual(bound);
      worst = Math.max(worst, error / bound);
    }
    expect(Math.abs(air.lit[index][3] - plain.lit[index][3]), `${label} transmittance`).toBeLessThanOrEqual(1e-4);
  });
  console.log('LOCAL_LIGHTS_AIR_DIAGNOSTIC ' + JSON.stringify({ adapter: air.adapter, worst,
    air: { dark: air.dark, lit: air.lit }, plain: { dark: plain.dark, lit: plain.lit } }));
});
