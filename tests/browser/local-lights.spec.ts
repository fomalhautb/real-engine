import { expect, test } from '@playwright/test';
import type { LocalLightDescriptor } from '../../src/local-lights';
import type {
  LocalLightPixel, LocalLightSurfacePlan, LocalLightSurfaceStep, runLocalLightContractFixture, runLocalLightSurfaceFixture,
} from './fixtures/local-lights';

type Vec3 = [number, number, number];
type Vec4 = [number, number, number, number];
type SurfaceResult = Awaited<ReturnType<typeof runLocalLightSurfaceFixture>>;
test.use({ deviceScaleFactor: 1 });

const SIZE = 129;
const EYE: Vec3 = [0, 0, 6];
const TAN = Math.tan(Math.PI / 6.4);
const RECEIVER: Vec4 = [0.6, 0.3, 0.15, 1];
const UNLIT: Vec4 = [0.2, 0.5, 0.9, 1];
const UNLIT_REGION = { min: [2.4, -3.1] as [number, number], max: [3.1, -2.4] as [number, number] };
const OPTIONS = { maxLights: 8, maxShadowedLights: 2, shadowResolution: 512, maxShadowViews: 12 };

// --- Independent oracle (does not import the engine) ---------------------------------------
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const normalize = (v: readonly number[]): Vec3 => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };

/** Pixel centre on the z = 0 receiver for the renderer's fixed π/3.2 vertical field of view. */
function pixelWorld(pixel: LocalLightPixel): Vec3 {
  const x = (2 * (pixel.x + 0.5) / SIZE - 1) * TAN, y = (1 - 2 * (pixel.y + 0.5) / SIZE) * TAN;
  return [EYE[0] + x * EYE[2], EYE[1] + y * EYE[2], 0];
}
function pixelAt(x: number, y: number): LocalLightPixel {
  return { x: Math.round(((x / EYE[2]) / TAN + 1) / 2 * SIZE - 0.5), y: Math.round((1 - (y / EYE[2]) / TAN) / 2 * SIZE - 0.5) };
}
/** E⊥ = colour × intensity × saturate(1 − (d/r)⁴)² × cone / max(d², 0.01²). */
function irradiance(light: LocalLightDescriptor, world: Vec3): { l: Vec3; e: Vec3 } {
  const delta = sub(light.position, world), d2 = dot(delta, delta), l = normalize(delta);
  if (d2 >= light.range * light.range) return { l, e: [0, 0, 0] };
  const ratio = d2 / (light.range * light.range), window = Math.min(1, Math.max(0, 1 - ratio * ratio)) ** 2;
  let factor = window / Math.max(d2, 1e-4);
  if (light.kind === 'spot') {
    const axis = normalize(light.direction!), cosOuter = Math.cos(light.outerConeAngle!);
    const t = Math.min(1, Math.max(0, (-dot(l, axis) - cosOuter) / (Math.cos(light.innerConeAngle!) - cosOuter)));
    factor *= t * t;
  }
  return { l, e: light.color.map(c => c * light.intensity * factor) as Vec3 };
}
/** GGX/Smith-Schlick/Schlick-Fresnel direct response × n·l, the documented engine BRDF. */
function brdf(base: readonly number[], metallic: number, roughness: number, n: Vec3, v: Vec3, l: Vec3): Vec3 {
  const half = [l[0] + v[0], l[1] + v[1], l[2] + v[2]], hl = Math.max(Math.hypot(half[0], half[1], half[2]), 1e-5);
  const h = half.map(value => value / hl);
  const nv = Math.max(dot(n, v), 0.001), nl = dot(n, l) > 0 ? Math.max(dot(n, l), 0) : 0;
  const nh = Math.max(dot(n, h), 0), vh = Math.max(dot(v, h), 0);
  const a2 = roughness ** 4, denominator = (1 - nh * nh) + nh * nh * a2;
  const distribution = a2 / Math.max(Math.PI * denominator * denominator, 1e-20);
  const k = (roughness + 1) ** 2 / 8;
  const geometry = (nv / (nv * (1 - k) + k)) * (nl / (nl * (1 - k) + k));
  return [0, 1, 2].map(c => {
    const f0 = 0.04 + (base[c] - 0.04) * metallic, fresnel = f0 + (1 - f0) * (1 - vh) ** 5;
    const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
    const diffuse = (1 - fresnel) * (1 - metallic) * base[c] / Math.PI;
    return (diffuse + specular) * nl;
  }) as Vec3;
}
function expectedDirect(lights: readonly LocalLightDescriptor[], receiver: { metalness: number; roughness: number },
  pixel: LocalLightPixel): Vec3 {
  const world = pixelWorld(pixel), v = normalize(sub(EYE, world)), n: Vec3 = [0, 0, 1];
  const sum: Vec3 = [0, 0, 0];
  for (const light of lights) {
    const { l, e } = irradiance(light, world);
    const response = brdf(RECEIVER, receiver.metalness, Math.min(1, Math.max(0.04, receiver.roughness)), n, v, l);
    for (let c = 0; c < 3; c++) sum[c] += response[c] * e[c];
  }
  return sum;
}
/** Each rgba16float value carries ≤ 2^-11 relative rounding; allow twice that on lit and baseline. */
function tolerance(lit: number, ambient: number): number { return 1e-4 + 1e-3 * (Math.abs(lit) + Math.abs(ambient)); }
const inUnlit = (world: Vec3) => world[0] > UNLIT_REGION.min[0] - 0.1 && world[0] < UNLIT_REGION.max[0] + 0.1
  && world[1] > UNLIT_REGION.min[1] - 0.1 && world[1] < UNLIT_REGION.max[1] + 0.1;

// --- Plan --------------------------------------------------------------------------------------
const axisPixels = [4, 19, 34, 49, 64, 79, 94, 109, 124];
// A coarse 9×9 grid plus a dense line through the spot aim point covering the inner cone, ramp and outside.
const spotLine: LocalLightPixel[] = Array.from({ length: 24 }, (_, i) => ({ x: 64 + i * 2, y: 66 }));
const grid: LocalLightPixel[] = [...axisPixels.flatMap(y => axisPixels.map(x => ({ x, y }))), ...spotLine]
  .filter(p => !inUnlit(pixelWorld(p)));
const unlitPixel = pixelAt((UNLIT_REGION.min[0] + UNLIT_REGION.max[0]) / 2, (UNLIT_REGION.min[1] + UNLIT_REGION.max[1]) / 2);
const withUnlit = (pixels: LocalLightPixel[]) => [...pixels, unlitPixel];
const point = (position: Vec3, color: Vec3, intensity: number, range: number, castShadows = false): LocalLightDescriptor =>
  ({ kind: 'point', position, color, intensity, range, castShadows });
const A = point([0, 0, 1], [1, 0.8, 0.6], 2, 6);
const B = point([0.7, -0.4, 2.2], [0.9, 1, 0.7], 5, 3);
const C = point([-1, 0.5, 1.5], [1, 1, 1], 3, 8);
const D: LocalLightDescriptor = { kind: 'spot', position: [0, 0, 3], direction: [0.6, 0, -3], innerConeAngle: 0.15,
  outerConeAngle: 0.4, color: [1, 0.9, 0.8], intensity: 6, range: 9 };
const S = point([0, 0, 3], [1, 1, 1], 4, 6, true);
const T: LocalLightDescriptor = { kind: 'spot', position: [0, 0, 3], direction: [0, 0, -1], innerConeAngle: 0.8,
  outerConeAngle: 0.9, color: [1, 1, 1], intensity: 4, range: 6, castShadows: true };
const targets = (list: [number, number][]) => list.map(([x, y]) => pixelAt(x, y));
const shadowCases = {
  // Occluder ±0.25 m at z = 2.5 under a light at z = 3 shadows |x|,|y| < 1.5 m; the camera sees it only for |x|,|y| < 0.43 m.
  center: { shadowed: targets([[0.8, 0], [0, -1.1], [-0.9, 0.9], [1.2, -1.2]]), lit: targets([[2.2, 0], [0, 2.3], [-2.2, -1.9], [1.8, 1.7]]) },
  // Bone translation (-0.6, 0, 0) moves the skinned shadow to x in [-5.1, -2.1].
  left: { shadowed: targets([[-2.4, 0], [-2.9, 0.8]]), lit: targets([[0.8, 0], [0, -1.1], [2.2, 0]]) },
  // Light at z = 1, occluder translated to (1.25, 0, 0.5): shadow x in [2, 3], |y| < 0.5 through the +X cube face;
  // lit points exercise the -Z, +Y, -X and -Y faces and both sides of the +Y/-Z seam.
  face: { shadowed: targets([[2.5, 0], [2.3, 0.3]]), lit: targets([[0, 1.05], [0, 0.95], [0.6, 0], [-1.5, 0], [0, -1.5]]) },
};
const shadowPixels = (key: keyof typeof shadowCases) => withUnlit([...shadowCases[key].shadowed, ...shadowCases[key].lit]);
const probe = (label: string, pixels = withUnlit(grid)): LocalLightSurfaceStep => ({ kind: 'probe', label, pixels });
const steps: LocalLightSurfaceStep[] = [
  { kind: 'receiver', metalness: 0, roughness: 1 }, probe('ambient-rough'),
  { kind: 'add', key: 'A', light: A }, probe('A-rough'), { kind: 'remove', key: 'A' },
  { kind: 'receiver', metalness: 0, roughness: 0.5 }, probe('ambient-mid'),
  { kind: 'add', key: 'B', light: B }, probe('B-mid'), { kind: 'remove', key: 'B' },
  { kind: 'receiver', metalness: 1, roughness: 0.35 }, probe('ambient-metal'),
  { kind: 'add', key: 'C', light: C }, probe('C-metal'), { kind: 'remove', key: 'C' },
  { kind: 'receiver', metalness: 0, roughness: 0.6 }, probe('ambient-spot'),
  { kind: 'add', key: 'D', light: D }, probe('D-spot'),
  { kind: 'add', key: 'A', light: A }, probe('AD-sum'),
  { kind: 'enable', key: 'A', enabled: false }, probe('A-off'),
  { kind: 'enable', key: 'A', enabled: true }, probe('A-on'),
  { kind: 'update', key: 'A', patch: { intensity: 4 } }, probe('A-double'),
  { kind: 'remove', key: 'A' }, { kind: 'remove', key: 'D' }, probe('removed'),
  { kind: 'receiver', metalness: 0, roughness: 1 },
  { kind: 'masked', alpha: 1 }, probe('masked-ambient', shadowPixels('center')),
  { kind: 'add', key: 'S', light: S }, probe('masked-covered', shadowPixels('center')),
  { kind: 'masked', alpha: 0 }, probe('masked-clear', shadowPixels('center')),
  { kind: 'masked', alpha: 1 }, { kind: 'remove', key: 'S' }, { kind: 'add', key: 'T', light: T },
  probe('spot-covered', shadowPixels('center')), { kind: 'remove', key: 'T' }, { kind: 'masked', alpha: null },
  probe('left-ambient', shadowPixels('left')), { kind: 'add', key: 'S', light: S },
  { kind: 'skinned', translation: [-0.6, 0, 0] }, probe('skin-left', shadowPixels('left')),
  { kind: 'skinned', translation: [0, 0, 0] }, probe('skin-center', shadowPixels('left')),
  { kind: 'update', key: 'S', patch: { position: [0, 0, 1] } }, { kind: 'skinned', translation: [1.25, 0, -2] },
  probe('skin-face', shadowPixels('face')),
  { kind: 'skinned', translation: null }, { kind: 'enable', key: 'S', enabled: false }, probe('skin-face-ambient', shadowPixels('face')),
];
const plan: LocalLightSurfacePlan = { size: SIZE, eye: EYE, options: OPTIONS, receiverColor: RECEIVER, unlitColor: UNLIT,
  unlitRegion: UNLIT_REGION, occluderHalfSize: 0.25, occluderHeight: 2.5, steps };

function compare(result: SurfaceResult, label: string, ambientLabel: string, lights: readonly LocalLightDescriptor[],
  pixels: readonly LocalLightPixel[], options: { shadowed?: readonly LocalLightPixel[] } = {}) {
  const measured = result.probes[label], ambient = result.probes[ambientLabel];
  expect(measured, label).toBeTruthy();
  let maxError = 0, maxRelative = 0, lit = 0, zero = 0;
  pixels.forEach((pixel, index) => {
    const shadowed = options.shadowed?.some(p => p.x === pixel.x && p.y === pixel.y);
    const expected = shadowed ? [0, 0, 0] : expectedDirect(lights, measured.receiver, pixel);
    for (let c = 0; c < 3; c++) {
      const value = measured.hdr[index][c], base = ambient.hdr[index][c], direct = value - base;
      const error = Math.abs(direct - expected[c]);
      expect(error, `${label} pixel ${pixel.x},${pixel.y} channel ${c}: direct ${direct}, expected ${expected[c]}`)
        .toBeLessThanOrEqual(tolerance(value, base));
      maxError = Math.max(maxError, error);
      if (expected[c] > 0.01) maxRelative = Math.max(maxRelative, error / expected[c]);
    }
    if (Math.max(...expected) > 1e-6) lit++; else zero++;
  });
  return { label, pixels: pixels.length, lit, zero, maxError, maxRelative };
}

test('local lights match analytic HDR direct lighting, shadows, toggles and budgets', async ({ page }) => {
  test.setTimeout(180_000);
  page.on('console', message => { if (message.type() === 'error') console.log('LOCAL_LIGHTS_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async input => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'local-lights.ts') as {
      runLocalLightSurfaceFixture: typeof runLocalLightSurfaceFixture; runLocalLightContractFixture: typeof runLocalLightContractFixture };
    return { surface: await module.runLocalLightSurfaceFixture(input), contract: await module.runLocalLightContractFixture() };
  }, plan);
  const surface = result.surface, adapter = surface.adapter;
  expect(surface.errors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);

  // Oracle coverage: inverse square + window (including exact zeros beyond range), oblique Lambert/GGX and every cone regime.
  const coneRegimes = grid.map(pixel => {
    const world = pixelWorld(pixel), l = normalize(sub(D.position, world));
    const angle = Math.acos(-dot(l, normalize(D.direction!)));
    return angle < D.innerConeAngle! ? 'inner' : angle < D.outerConeAngle! ? 'ramp' : 'outside';
  });
  for (const regime of ['inner', 'ramp', 'outside']) expect(coneRegimes.filter(item => item === regime).length, regime).toBeGreaterThan(1);
  expect(grid.filter(pixel => Math.hypot(...sub(B.position, pixelWorld(pixel))) >= B.range).length).toBeGreaterThan(4);

  const summary = [
    compare(surface, 'A-rough', 'ambient-rough', [A], grid),
    compare(surface, 'B-mid', 'ambient-mid', [B], grid),
    compare(surface, 'C-metal', 'ambient-metal', [C], grid),
    compare(surface, 'D-spot', 'ambient-spot', [D], grid),
    compare(surface, 'AD-sum', 'ambient-spot', [A, D], grid),
    compare(surface, 'A-double', 'ambient-spot', [{ ...A, intensity: 4 }, D], grid),
    compare(surface, 'masked-covered', 'masked-ambient', [S], shadowPixels('center').slice(0, -1), { shadowed: shadowCases.center.shadowed }),
    compare(surface, 'masked-clear', 'masked-ambient', [S], shadowPixels('center').slice(0, -1)),
    compare(surface, 'spot-covered', 'masked-ambient', [T], shadowPixels('center').slice(0, -1), { shadowed: shadowCases.center.shadowed }),
    compare(surface, 'skin-left', 'left-ambient', [S], shadowPixels('left').slice(0, -1), { shadowed: shadowCases.left.shadowed }),
    compare(surface, 'skin-center', 'left-ambient', [S], shadowPixels('left').slice(0, -1),
      { shadowed: shadowCases.left.lit.slice(0, 2) }),
    compare(surface, 'skin-face', 'skin-face-ambient', [{ ...S, position: [0, 0, 1] }], shadowPixels('face').slice(0, -1),
      { shadowed: shadowCases.face.shadowed }),
  ];
  // Beyond-range pixels of B are bitwise identical to the baseline (an exact, bounded influence volume).
  grid.forEach((pixel, index) => {
    if (Math.hypot(...sub(B.position, pixelWorld(pixel))) >= B.range) {
      expect(surface.probes['B-mid'].hdr[index]).toEqual(surface.probes['ambient-mid'].hdr[index]);
    }
  });
  // Toggle response is immediate and exact: disabling restores the previous image, re-enabling repeats it.
  expect(surface.probes['A-off'].hdr).toEqual(surface.probes['D-spot'].hdr);
  expect(surface.probes['A-on'].hdr).toEqual(surface.probes['AD-sum'].hdr);
  expect(surface.probes.removed.hdr).toEqual(surface.probes['ambient-spot'].hdr);
  // Unlit materials ignore every local light.
  const unlitValues = Object.values(surface.probes).map(item => item.hdr.at(-1)!);
  for (const value of unlitValues) {
    expect(value).toEqual(unlitValues[0]);
  }
  for (let c = 0; c < 3; c++) expect(Math.abs(unlitValues[0][c] - UNLIT[c])).toBeLessThanOrEqual(UNLIT[c] * 1e-3);

  // Explicit versions and budgets in diagnostics; no camera culling of lights or shadow views.
  const covered = surface.probes['masked-covered'].diagnostics as Record<string, unknown>;
  expect(covered).toMatchObject({ lights: 1, enabledLights: 1, shadowedLights: 1, reservedShadowViews: 6,
    renderedShadowViews: 6, cameraCulling: false, shadowAtlasBytes: 4 * 512 * 512 * 12 });
  expect((surface.probes['spot-covered'].diagnostics as Record<string, unknown>).renderedShadowViews).toBe(1);
  expect(surface.probes['A-double'].lights.find(light => light.intensity === 4)!.version).toBe(4);
  const atlasBytes = 4 * 512 * 512 * 12, uniformBytes = 4 * (4 + 8 * 20 + 12 * 16), viewBytes = 12 * 256;
  // Two extra timestamp slots in the resolve and readback buffers are part of the same ledger.
  expect(surface.ownedBytes - surface.referenceBytes).toBe(atlasBytes + uniformBytes + viewBytes + (surface.timestamps ? 2 * 2 * 8 : 0));
  expect(surface.beforeDispose).toBe(atlasBytes + uniformBytes + viewBytes);
  expect(surface.afterDispose).toEqual({ localLights: 0, renderer: 0 });
  expect(surface.disposedApi).toContain('RENDERER_DISPOSED');
  expect(result.contract.errors).toEqual([]);
  expect(result.contract.failures.disabled).toContain('LOCAL_LIGHTS_DISABLED');
  expect(result.contract.failures.atlasBudget).toContain('LOCAL_LIGHT_SHADOW_BUDGET');
  expect(result.contract.failures.shadowBudget).toContain('LOCAL_LIGHT_SHADOW_BUDGET');
  expect(result.contract.failures.lightBudget).toContain('LOCAL_LIGHT_BUDGET');
  expect(result.contract.failures.invalid).toContain('INVALID_LOCAL_LIGHT: direction');
  console.log('LOCAL_LIGHTS_SURFACE_DIAGNOSTIC ' + JSON.stringify({ adapter, summary, frames: surface.frames,
    ownedBytes: surface.ownedBytes - surface.referenceBytes, timestamps: surface.timestamps, gpuShadowMs: surface.probes['skin-face'].gpuShadowMs }));
});
