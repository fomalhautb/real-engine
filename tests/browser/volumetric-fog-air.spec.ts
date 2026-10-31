import { expect, test } from '@playwright/test';
import { fogAirPhaseRadiance } from '../../src/volumetric-fog';
import type { runFogAirAcceptanceFixture } from './fixtures/volumetric-fog-air';
import {
  AIR_COLUMN, AIR_EYE, AIR_MEDIUM, AIR_MIRROR, AIR_OPEN_DEPTH, AIR_RECEIVER_DEPTH, AIR_RED, AIR_ROI, AIR_ROOM_FIELD_DEPTH, AIR_ROOM_GRID,
  AIR_ROOM_WALLS, AIR_SIZE, AIR_SKY, type AirRectangle, type Vec3,
} from './fixtures/volumetric-fog-air-cases';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runFogAirAcceptanceFixture>>;
type Probe = Result['open']['field'][number];
const Y00 = 0.28209479;
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a: readonly number[]): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

/** Unit view ray and ray length per unit planar depth for one pixel centre. */
function pixelRay(pixel: { x: number; y: number }, size = AIR_SIZE) {
  const tangent = Math.tan(Math.PI / 6.4);
  const ray = [(2 * (pixel.x + 0.5) / size - 1) * tangent, (1 - 2 * (pixel.y + 0.5) / size) * tangent, -1];
  const length = Math.hypot(...ray);
  return { view: normalize(ray), length };
}

/** Per-probe phase radiance along a view direction, as the field interpolates it (convex bounds). */
function phaseBounds(probes: readonly Probe[], view: Vec3, g: number) {
  const values = probes.map(probe => fogAirPhaseRadiance(probe.coefficients, view, g)
    .map((value, c) => Math.min(Math.max(value, 0), probe.maxRadiance[c])));
  return [0, 1, 2].map(c => ({ min: Math.min(...values.map(value => value[c])), max: Math.max(...values.map(value => value[c])) }));
}

/** CPU sky visibility of one probe along the GI's deterministic Fibonacci rays through black walls. */
function skyFraction(origin: Vec3, walls: readonly AirRectangle[], count: number, maxDistance: number): number {
  let escaped = 0;
  for (let i = 0; i < count; i++) {
    const y = 1 - 2 * (i + 0.5) / count, radius = Math.sqrt(Math.max(0, 1 - y * y)), phi = i * 2.399963229728653;
    const direction: Vec3 = [Math.cos(phi) * radius, y, Math.sin(phi) * radius];
    const blocked = walls.some(wall => {
      const normal = cross(wall.u, wall.v), denominator = dot(direction, normal);
      if (Math.abs(denominator) < 1e-12) return false;
      const t = dot(sub(wall.corner, origin), normal) / denominator;
      if (t <= 0.0001 || t > maxDistance) return false;
      const local = sub([origin[0] + t * direction[0], origin[1] + t * direction[1], origin[2] + t * direction[2]], wall.corner);
      const a = dot(local, wall.u) / dot(wall.u, wall.u), b = dot(local, wall.v) / dot(wall.v, wall.v);
      return a >= 0 && a <= 1 && b >= 0 && b <= 1;
    });
    if (!blocked) escaped++;
  }
  return escaped / count;
}

function probePosition(index: number, grid: { origin: Vec3; spacing: Vec3; counts: Vec3 }): Vec3 {
  const x = index % grid.counts[0], y = Math.floor(index / grid.counts[0]) % grid.counts[1];
  const z = Math.floor(index / (grid.counts[0] * grid.counts[1]));
  return [grid.origin[0] + x * grid.spacing[0], grid.origin[1] + y * grid.spacing[1], grid.origin[2] + z * grid.spacing[2]];
}

/** L_fog lies in albedo x [min, max] probe phase radiance x the opacity of the covered ray. */
function expectConvex(label: string, value: readonly number[], probes: readonly Probe[], pixel: { x: number; y: number }) {
  const { view, length } = pixelRay(pixel);
  const bounds = phaseBounds(probes, view, AIR_MEDIUM.anisotropy ?? 0);
  const core = 1 - Math.exp(-AIR_MEDIUM.extinction * AIR_ROOM_FIELD_DEPTH * length);
  const full = 1 - Math.exp(-AIR_MEDIUM.extinction * AIR_RECEIVER_DEPTH * length);
  for (let c = 0; c < 3; c++) {
    const albedo = AIR_MEDIUM.albedo[c];
    expect(value[c], `${label} lower ${c}`).toBeGreaterThanOrEqual(albedo * bounds[c].min * core * 0.98 - 0.0005);
    expect(value[c], `${label} upper ${c}`).toBeLessThanOrEqual(albedo * bounds[c].max * full * 1.02 + 0.0005);
  }
}

test('F04: air light from GI incident radiance and sky visibility, off-screen colour, room leaks and the mirror path', async ({ page }) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result: Result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-air.ts') as {
      runFogAirAcceptanceFixture: typeof runFogAirAcceptanceFixture };
    return fixture.runFogAirAcceptanceFixture();
  });
  const report: Record<string, unknown> = {};
  console.log('FOG_F04_RAW ' + JSON.stringify({ open: { isotropic: result.open.isotropic[0], forward: result.open.forward[0],
    off: result.open.off[0], band0: result.open.field[0].coefficients.slice(0, 3), errors: result.open.errors },
  closed: result.room.closed.roi[0], red: result.room.offscreenRed.roi[0], court: result.room.courtyard.column,
  roomErrors: result.room.errors, mirror: [result.mirror.airOn[0], result.mirror.airOff[0], result.mirror.noFog[0]],
  mirrorErrors: result.mirror.errors }));

  // Open sky: every probe holds the same sky-only SH; its band 0 is the sky radiance.
  const { open } = result;
  expect(open.field.every(probe => probe.valid)).toBe(true);
  for (const probe of open.field) {
    for (let c = 0; c < 27; c++) expect(Math.abs(probe.coefficients[c] - open.field[0].coefficients[c])).toBeLessThan(1e-5);
  }
  for (let c = 0; c < 3; c++) expect(Math.abs(open.field[0].coefficients[c] * Y00 - AIR_SKY[c]) / AIR_SKY[c]).toBeLessThan(1e-4);
  let openWorst = 0, openWorstT = 0, phaseResidual = 0;
  AIR_ROI.forEach((pixel, index) => {
    const { view, length } = pixelRay(pixel);
    const transmittance = Math.exp(-AIR_MEDIUM.extinction * AIR_OPEN_DEPTH * length);
    for (const [label, value, g] of [['isotropic', open.isotropic[index], 0], ['forward', open.forward[index], 0.6]] as const) {
      const radiance = fogAirPhaseRadiance(open.field[0].coefficients, view, g);
      openWorstT = Math.max(openWorstT, Math.abs(value[3] - transmittance));
      expect(Math.abs(value[3] - transmittance), `${label} T ${index}`).toBeLessThanOrEqual(0.001);
      for (let c = 0; c < 3; c++) {
        phaseResidual = Math.max(phaseResidual, Math.abs(radiance[c] - AIR_SKY[c]) / AIR_SKY[c]);
        // A constant sky is isotropic under every normalized phase function: L = albedo * sky * (1 - T).
        const expected = AIR_MEDIUM.albedo[c] * AIR_SKY[c] * (1 - transmittance);
        const relative = Math.abs(value[c] - expected) / expected;
        openWorst = Math.max(openWorst, relative);
        expect(relative, `${label} L ${index}:${c}`).toBeLessThanOrEqual(0.01);
      }
    }
    // Disabled air adds nothing; it never changes transmittance.
    for (let c = 0; c < 3; c++) expect(Math.abs(open.off[index][c]), `off ${index}:${c}`).toBeLessThan(0.0001);
    expect(open.off[index][3]).toBeCloseTo(open.forward[index][3], 6);
    // Sun and air sources add linearly.
    for (let c = 0; c < 3; c++) {
      const sum = open.sunOnly[index][c] + open.forward[index][c];
      expect(Math.abs(open.sunAndAir[index][c] - sum), `sun+air ${index}:${c}`).toBeLessThanOrEqual(0.003 + 0.002 * sum);
    }
  });
  expect(phaseResidual).toBeLessThan(0.002);
  report.open = { worstRelative: openWorst, worstT: openWorstT, phaseResidual };

  // Air light never enters history: GI invalidation and GI on/off neither reset nor lag the fog.
  expect(open.beforeInvalidation).toMatchObject({ valid: true, lastResetReason: 'temporal-toggle' });
  const frames = open.beforeInvalidation.framesSinceReset;
  expect(open.historyAfterInvalidation).toMatchObject({ valid: true, framesSinceReset: frames + 1, lastResetReason: 'temporal-toggle' });
  expect(open.historyGiDisabled).toMatchObject({ valid: true, framesSinceReset: frames + 2, lastResetReason: 'temporal-toggle' });
  expect(open.historyGiEnabled).toMatchObject({ valid: true, framesSinceReset: frames + 3, lastResetReason: 'temporal-toggle' });
  AIR_ROI.forEach((_, index) => {
    for (let c = 0; c < 3; c++) {
      const reference = open.forward[index][c];
      expect(Math.abs(open.accumulated[index][c] - reference) / reference, `accumulated ${index}:${c}`).toBeLessThanOrEqual(0.005);
      expect(Math.abs(open.afterInvalidation[index][c] - reference) / reference, `after GI invalidation ${index}:${c}`).toBeLessThanOrEqual(0.005);
      expect(Math.abs(open.giDisabledFirstFrame[index][c]), `GI disabled first frame ${index}:${c}`).toBeLessThan(0.0001);
      expect(Math.abs(open.giEnabledFirstFrame[index][c] - reference) / reference, `GI enabled first frame ${index}:${c}`).toBeLessThanOrEqual(0.005);
    }
  });
  expect(open.diagnostics?.airLighting).toMatchObject({ gi: true, outsideSkyVisibility: 1, configured: true });

  // Closed room of black walls: the sky must not leak in.
  const { room } = result;
  expect(room.closed.probes.every(probe => probe.valid)).toBe(true);
  for (const probe of room.closed.probes) for (let c = 0; c < 3; c++) expect(probe.coefficients[c]).toBeLessThan(1e-6);
  let leak = 0;
  AIR_ROI.forEach((pixel, index) => {
    // Relative to an unoccluded sky over the same receiver depth.
    const opacity = 1 - Math.exp(-AIR_MEDIUM.extinction * AIR_RECEIVER_DEPTH * pixelRay(pixel).length);
    for (let c = 0; c < 3; c++) leak = Math.max(leak, room.closed.roi[index][c] / (AIR_MEDIUM.albedo[c] * AIR_SKY[c] * opacity));
  });
  expect(leak).toBeLessThanOrEqual(0.02);

  // Off-screen red wall behind the camera: never on screen, but it colours the air in front.
  for (const value of room.offscreenRed.surface) for (let c = 0; c < 3; c++) expect(value[c]).toBeLessThan(0.0001);
  let redRatio = 0;
  AIR_ROI.forEach((pixel, index) => {
    const value = room.offscreenRed.roi[index];
    expect(value[0], `red signal ${index}`).toBeGreaterThan(0.01);
    // Every radiance in the closed room is the wall's: the air keeps its chromaticity.
    expect(Math.abs(value[1] / value[0] - AIR_RED[1] / AIR_RED[0]), `red chroma g ${index}`).toBeLessThanOrEqual(0.1 * AIR_RED[1] / AIR_RED[0]);
    expect(Math.abs(value[2] / value[0] - AIR_RED[2] / AIR_RED[0]), `red chroma b ${index}`).toBeLessThanOrEqual(0.1 * AIR_RED[2] / AIR_RED[0]);
    redRatio = Math.max(redRatio, Math.abs(value[1] / value[0] - AIR_RED[1] / AIR_RED[0]) / (AIR_RED[1] / AIR_RED[0]));
    expectConvex(`red ROI ${index}`, value, room.offscreenRed.probes, pixel);
  });

  // Courtyard: sky through the open ceiling. Probe band 0 is the sky times its geometric visibility.
  const walls = AIR_ROOM_WALLS.filter(wall => wall.name !== 'ceiling');
  let visibilityError = 0;
  room.courtyard.probes.forEach((probe, index) => {
    const expected = skyFraction(probePosition(index, AIR_ROOM_GRID), walls, AIR_ROOM_GRID.raysPerProbe, AIR_ROOM_GRID.maxDistance);
    for (let c = 0; c < 3; c++) {
      const measured = probe.coefficients[c] * Y00 / AIR_SKY[c];
      visibilityError = Math.max(visibilityError, Math.abs(measured - expected));
      expect(Math.abs(measured - expected), `probe ${index} sky visibility ${c}`).toBeLessThanOrEqual(2 / AIR_ROOM_GRID.raysPerProbe);
    }
  });
  AIR_ROI.forEach((pixel, index) => expectConvex(`courtyard ROI ${index}`, room.courtyard.roi[index], room.courtyard.probes, pixel));
  AIR_COLUMN.forEach((pixel, index) => {
    expectConvex(`courtyard column ${index}`, room.courtyard.column[index], room.courtyard.probes, pixel);
    // Air light falls off away from the opening.
    if (index) for (let c = 0; c < 3; c++) expect(room.courtyard.column[index][c]).toBeLessThan(room.courtyard.column[index - 1][c]);
  });
  report.room = { leak, redChromaError: redRatio, visibilityError,
    courtyardColumn: room.courtyard.column.map(value => value.slice(0, 3)) };

  // Mirror: fog behind the camera is outside the camera grid but on the reflected rays.
  const { mirror } = result;
  for (const value of mirror.mainView) {
    for (let c = 0; c < 3; c++) expect(Math.abs(value[c])).toBeLessThan(0.0001);
    expect(value[3]).toBe(1);
  }
  const sigma = AIR_MIRROR.medium.extinction;
  const mirrorReport = mirror.airOn.map((anchor, index) => {
    expect(anchor.valid && mirror.airOff[index].valid && mirror.noFog[index].valid).toBe(true);
    const incident = normalize(sub(anchor.position, AIR_MIRROR.eye));
    const direction = normalize(sub(incident, anchor.normal.map(value => 2 * dot(incident, anchor.normal) * value)));
    const thickness = (AIR_MIRROR.box.max[2] - AIR_MIRROR.box.min[2]) / direction[2];
    const step = (AIR_MIRROR.emitterZ - anchor.position[2]) / direction[2] / AIR_MIRROR.fogSteps;
    const bounds = phaseBounds(mirror.field, direction, 0);
    const result = [0, 1, 2].map(c => {
      const noFog = mirror.noFog[index].radiance[c];
      expect(Math.abs(noFog - AIR_MIRROR.emitter[c]) / AIR_MIRROR.emitter[c], `mirror emitter ${index}:${c}`).toBeLessThanOrEqual(0.05);
      const transmittance = mirror.airOff[index].radiance[c] / noFog;
      expect(transmittance).toBeGreaterThanOrEqual(Math.exp(-sigma * (thickness + 2 * step)) - 0.01);
      expect(transmittance).toBeLessThanOrEqual(Math.exp(-sigma * (thickness - 2 * step)) + 0.01);
      const air = mirror.airOn[index].radiance[c] - mirror.airOff[index].radiance[c];
      const albedo = AIR_MIRROR.medium.albedo[c];
      expect(air, `mirror air ${index}:${c}`).toBeGreaterThanOrEqual(albedo * bounds[c].min * (1 - Math.exp(-sigma * (thickness - 2 * step))) * 0.98 - 0.002);
      expect(air, `mirror air ${index}:${c}`).toBeLessThanOrEqual(albedo * bounds[c].max * (1 - Math.exp(-sigma * (thickness + 2 * step))) * 1.02 + 0.002);
      expect(air).toBeGreaterThan(0.005);
      return { air, transmittance };
    });
    return result;
  });
  report.mirror = mirrorReport;

  for (const part of [open, room, mirror]) {
    expect(part.errors).toEqual([]);
    expect(part.ownedBytesAfterDispose).toBe(0);
  }
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(open.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  console.log('FOG_F04_AIR ' + JSON.stringify({ ...report, adapter: open.adapter, eye: AIR_EYE }));
});
