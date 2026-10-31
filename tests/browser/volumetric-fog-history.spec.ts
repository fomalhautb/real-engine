import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';
import type { runFogHistoryAcceptanceFixture } from './fixtures/volumetric-fog-history';
import type { runLocalLightFogResponseFixture } from './fixtures/local-lights-fog-response';
import {
  HISTORY_BOX, HISTORY_BOX_PIXEL, HISTORY_GHOST_PIXEL, HISTORY_LATTICE, HISTORY_MAX_FRAMES, HISTORY_MEDIUM, HISTORY_MOVED_BOX_PIXEL,
  HISTORY_PIXELS, HISTORY_SHAFT_PIXEL, HISTORY_SUN,
} from './fixtures/volumetric-fog-history-cases';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runFogHistoryAcceptanceFixture>>;
type Run = Result['local'];
const SURFACE = [2, 1, 0.5];
const luminance = (v: readonly number[]) => 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
/** Final radiance of a receiver pixel from the split view: L_fog + T * L_surface. */
const composite = (v: readonly number[]) => [0, 1, 2].map(c => v[c] + v[3] * SURFACE[c]);

function rayLength(pixel: { x: number; y: number }, size = 64) {
  const tangent = Math.tan(Math.PI / 6.4);
  return Math.hypot((2 * (pixel.x + 0.5) / size - 1) * tangent, (1 - 2 * (pixel.y + 0.5) / size) * tangent, 1);
}

/** Isotropic sun single scattering over the fully lit shaft crossing, planar depth 2..5 (high-sample closed form). */
function shaftReference(pixel: { x: number; y: number }) {
  const length = rayLength(pixel), sigma = HISTORY_MEDIUM.extinction;
  return [0, 1, 2].map(c => HISTORY_MEDIUM.albedo[c] * HISTORY_SUN.color[c] / (4 * Math.PI)
    * (Math.exp(-sigma * 2 * length) - Math.exp(-sigma * 5 * length)));
}

/** Emission of the box (planar depth 2.5..3.5) behind unlit global fog. */
function boxReference(pixel: { x: number; y: number }) {
  const length = rayLength(pixel), total = HISTORY_MEDIUM.extinction + HISTORY_BOX.coefficients.extinction;
  return [0, 1, 2].map(c => Math.exp(-HISTORY_MEDIUM.extinction * 2.5 * length) * HISTORY_BOX.coefficients.emission[c] / total
    * (1 - Math.exp(-total * length)));
}

function common(run: Run, report: Record<string, unknown>) {
  const label = run.occluderRejection;
  const shaft = HISTORY_PIXELS.indexOf(HISTORY_SHAFT_PIXEL), ghost = HISTORY_PIXELS.length;
  // F02: scattering and transmittance separately against the closed-form shaft.
  let shaftError = 0, transmittanceError = 0;
  for (const index of [0, 1, 2]) {
    const pixel = HISTORY_PIXELS[index], value = run.clear.split[index], expected = shaftReference(pixel);
    transmittanceError = Math.max(transmittanceError, Math.abs(value[3] - Math.exp(-HISTORY_MEDIUM.extinction * 6 * rayLength(pixel))));
    for (let c = 0; c < 3; c++) shaftError = Math.max(shaftError, Math.abs(value[c] - expected[c]) / expected[c]);
  }
  expect(transmittanceError, `${label} shaft T`).toBeLessThanOrEqual(0.001);
  expect(shaftError, `${label} shaft L`).toBeLessThanOrEqual(0.15);
  const open = run.clear.split[shaft][0];
  const sideLeak = Math.max(...[3, 4].map(index => run.clear.split[index][0])) / open;
  const thinWallLeak = Math.max(...run.closedLattice.map(value => value[0])) / open;
  expect(sideLeak, `${label} shadowed side`).toBeLessThanOrEqual(0.01);
  expect(thinWallLeak, `${label} closed thin roof`).toBeLessThanOrEqual(0.01);

  // F02 skinned character: its shadow column darkens the ghost pixel; neither first frame keeps a ghost.
  const lit = run.clear.split[ghost][0], shadowed = run.entered.split[ghost][0], step = lit - shadowed;
  expect(step, `${label} character shadow step`).toBeGreaterThan(0.05 * lit);
  const enterGhost = Math.abs(run.enteredFirst.split[ghost][0] - shadowed) / step;
  const leaveGhost = Math.abs(run.leftFirst.split[ghost][0] - run.left.split[ghost][0]) / step;
  expect(enterGhost, `${label} enter ghost`).toBeLessThanOrEqual(0.5);
  expect(leaveGhost, `${label} leave ghost`).toBeLessThanOrEqual(0.5);

  // F03 door: the first closed frame is dark (no stale shaft); opening reaches 90 % within 100 ms at 60 Hz.
  const closeResidual = run.closeCurve[0].split[shaft][0] / open;
  const openCurve = run.openCurve.map(entry => entry.split[shaft][0] / open);
  const t90 = openCurve.findIndex(value => value >= 0.9) + 1;
  expect(closeResidual, `${label} closed first frame`).toBeLessThanOrEqual(0.02);
  expect(t90, `${label} door T90 frames`).toBeGreaterThan(0);
  expect(t90, `${label} door T90 frames`).toBeLessThanOrEqual(6);
  expect(run.errors).toEqual([]);
  expect(run.ownedBytesAfterDispose).toBe(0);
  report[label] = { shaftError, transmittanceError, sideLeak, thinWallLeak, enterGhost, leaveGhost, closeResidual, t90,
    openCurve, closeCurve: run.closeCurve.map(entry => entry.split[shaft][0] / open),
    cells: { entered: run.enteredFirst.cells, left: run.leftFirst.cells, closed: run.closeCurve[0].cells, opened: run.openCurve[0].cells },
    reset: { entered: run.enteredFirst.history.lastResetReason, door: run.closeCurve[0].history.lastResetReason },
    regions: run.enteredFirst.regions, ownedBytes: run.ownedBytes };
}

test('F02/F03: local fog history rejection around moving casters, door response, media edits, teleport and 120-frame stability', async ({ page }) => {
  test.setTimeout(300_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const evidenceDir = process.env.FOG_EVIDENCE_DIR;
  const result: Result = await page.evaluate(async evidence => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-history.ts') as {
      runFogHistoryAcceptanceFixture: typeof runFogHistoryAcceptanceFixture };
    return fixture.runFogHistoryAcceptanceFixture(evidence);
  }, !!evidenceDir);
  const report: Record<string, unknown> = {};
  console.log('FOG_HISTORY_RAW ' + JSON.stringify({ local: { clear: result.local.clear.split, entered: result.local.entered.split,
    enteredFirst: result.local.enteredFirst, closed0: result.local.closeCurve[0].split[0], open0: result.local.openCurve[0].split[0] },
  global: { enteredFirst: result.global.enteredFirst.cells, reason: result.global.enteredFirst.history }, errors: result.local.errors }).slice(0, 4000));
  common(result.local, report);
  common(result.global, report);
  const { local, global } = result;

  // Local rejection: cells under the changed casters restart; far cells keep their history; no global reset.
  const toggled = local.clear.history.framesSinceReset;
  expect(local.enteredFirst.history).toMatchObject({ lastResetReason: 'temporal-toggle', framesSinceReset: toggled + 1 });
  for (const entry of [local.enteredFirst, local.leftFirst, local.closeCurve[0], local.openCurve[0]]) {
    expect(entry.cells[2].count, 'far cell keeps history').toBe(HISTORY_MAX_FRAMES);
    expect(entry.cells[2].supportRadius, 'far cell support').toBeLessThan(0.5);
  }
  expect(local.enteredFirst.cells[1].count, 'cell under the character restarts').toBe(1);
  expect(local.leftFirst.cells[1].count, 'cell under the old position restarts').toBe(1);
  expect(local.closeCurve[0].cells[0].count, 'shaft cell restarts when the door closes').toBe(1);
  expect(local.openCurve[0].cells[0].count, 'shaft cell restarts when the door opens').toBe(1);
  // Global mode: any caster change resets every cell.
  expect(global.enteredFirst.history.lastResetReason).toBe('scene-change');
  expect(global.enteredFirst.cells[2].count).toBe(1);
  expect(global.closeCurve[0].history.lastResetReason).toBe('scene-change');
  expect(local.ownedBytes - global.ownedBytes, 'local rejection region uniform').toBe(1040);

  // 120 frames of a fixed scene and camera: final luminance fluctuation of non-near-black pixels.
  const stability = local.extra!.stability;
  let fluctuation = 0;
  HISTORY_LATTICE.forEach((_, index) => {
    const series = stability.map(frame => luminance(composite(frame[index])));
    const mean = series.reduce((a, b) => a + b, 0) / series.length;
    if (mean > 0.01) fluctuation = Math.max(fluctuation, (Math.max(...series) - Math.min(...series)) / mean);
  });
  expect(fluctuation).toBeLessThanOrEqual(0.03);

  // Teleport: no explicit cut; the first frame and the time to restabilize within 3 %.
  const converged = local.extra!.teleportConverged.map(value => luminance(composite(value)));
  const teleportErrors = local.extra!.teleportCurve.map(frame => Math.max(...frame.map((value, index) =>
    Math.abs(luminance(composite(value)) - converged[index]) / converged[index])));
  const restabilized = teleportErrors.findIndex((error, index) => teleportErrors.slice(index).every(value => value <= 0.03)) + 1;
  expect(restabilized).toBeGreaterThan(0);
  expect(restabilized).toBeLessThanOrEqual(60);

  // Fog volume add / move / remove: explicit media resets; first frames carry no stale fog.
  const box = boxReference(HISTORY_BOX_PIXEL), movedBox = boxReference(HISTORY_MOVED_BOX_PIXEL);
  for (const [entry, reason] of [[local.extra!.added, 'media-change'], [local.extra!.moved, 'media-change'], [local.extra!.removed, 'media-change']] as const) {
    expect(entry.history).toMatchObject({ lastResetReason: reason, framesSinceReset: 1 });
  }
  const mediaErrors = {
    added: Math.abs(local.extra!.added.split[0][0] - local.extra!.beforeBox.split[0][0] - box[0]) / box[0],
    addedConverged: Math.abs(local.extra!.addedConverged.split[0][0] - local.extra!.beforeBox.split[0][0] - box[0]) / box[0],
    movedNew: Math.abs(local.extra!.moved.split[1][0] - local.extra!.beforeBox.split[1][0] - movedBox[0]) / movedBox[0],
    movedOld: Math.abs(local.extra!.moved.split[0][0] - local.extra!.beforeBox.split[0][0]) / box[0],
    removed: Math.max(...[0, 1].map(index => Math.abs(local.extra!.removed.split[index][0] - local.extra!.beforeBox.split[index][0]))) / box[0],
  };
  expect(mediaErrors.added).toBeLessThanOrEqual(0.2);
  expect(mediaErrors.addedConverged).toBeLessThanOrEqual(0.1);
  expect(mediaErrors.movedNew).toBeLessThanOrEqual(0.2);
  expect(mediaErrors.movedOld).toBeLessThanOrEqual(0.02);
  expect(mediaErrors.removed).toBeLessThanOrEqual(0.02);

  // Fog on/off at the same exposure.
  local.extra!.fogOff.forEach(value => expect(value.slice(0, 3)).toEqual(SURFACE));
  const fogOnShaft = local.extra!.fogOn[HISTORY_LATTICE.findIndex(pixel => pixel.x === 44 && pixel.y === 28)];
  expect(fogOnShaft[0]).not.toBe(SURFACE[0]);
  if (evidenceDir) {
    for (const [name, url] of [['fog-on', local.extra!.fogOnImage], ['fog-off', local.extra!.fogOffImage]] as const) {
      if (typeof url === 'string' && url.startsWith('data:image/png;base64,')) {
        writeFileSync(join(evidenceDir, `history-${name}.png`), Buffer.from(url.slice(22), 'base64'));
      }
    }
  }
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(local.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  console.log('FOG_HISTORY ' + JSON.stringify({ ...report, fluctuation, teleport: { firstFrame: teleportErrors[0], restabilized,
    reason: local.extra!.teleportHistory.lastResetReason }, mediaErrors, adapter: local.adapter }));
});

test('F03: moving spotlight and short flash response curves', async ({ page }) => {
  // Local-light direct changes (benchmarks.md: T90 <= 100 ms at 60 Hz, saved ghost area and brightness curves,
  // flashes judged on their own frames, not only the settled end state). Red scattering of the 8x8 lattice.
  test.setTimeout(300_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'local-lights-fog-response.ts') as {
      runLocalLightFogResponseFixture: typeof runLocalLightFogResponseFixture };
    return fixture.runLocalLightFogResponseFixture();
  });
  type Frame = { values: number[]; reason: string | null; framesSinceReset: number };
  const at = (x: number, y: number) => HISTORY_LATTICE.findIndex(pixel => pixel.x === x && pixel.y === y);
  const rows = [20, 28, 36, 44], beamA = rows.map(y => at(12, y)), beamB = rows.map(y => at(52, y));
  const { litA, litB, litA2, litA3, shadowedA } = result;
  const peakA = Math.max(...litA.values), peakB = Math.max(...litB.values);
  // The beams are disjoint: each set is well lit by its own beam and exactly dark under the other.
  for (const index of beamA) { expect(litA.values[index]).toBeGreaterThan(0.25 * peakA); expect(litB.values[index]).toBeLessThanOrEqual(1e-6); }
  for (const index of beamB) { expect(litB.values[index]).toBeGreaterThan(0.25 * peakB); expect(litA.values[index]).toBeLessThanOrEqual(1e-6); }
  /** Frames until every pixel of the set completed 90 % of its change (1 = the first changed frame). */
  const t90 = (curve: Frame[], set: number[], before: Frame, after: Frame) => curve.findIndex(frame => set.every(index =>
    Math.abs(frame.values[index] - after.values[index]) <= 0.1 * Math.abs(before.values[index] - after.values[index]))) + 1;
  /** Pixels lit above 2 % of the peak where the settled state is dark (below 1 %). */
  const ghostArea = (frame: Frame, settled: Frame, peak: number) =>
    frame.values.filter((value, index) => settled.values[index] < 0.01 * peak && value > 0.02 * peak).length;
  const ratios = (curve: Frame[], set: number[], reference: Frame) => curve.map(frame => set.map(index => frame.values[index] / reference.values[index]));

  // Jump A -> B: the light version resets history; the old beam is gone and the new one complete on the first frame.
  expect(result.jumpCurve[0]).toMatchObject({ reason: 'local-light-change', framesSinceReset: 1 });
  const jumpResidual = Math.max(...beamA.map(index => result.jumpCurve[0].values[index] / litA.values[index]));
  const jumpT90 = t90(result.jumpCurve, beamB, litA, litB);
  const jumpGhostArea = Math.max(...result.jumpCurve.map(frame => ghostArea(frame, litB, peakB)));
  expect(jumpResidual, 'old beam on the first moved frame').toBeLessThanOrEqual(0.02);
  expect(jumpT90, 'new beam T90 frames').toBeGreaterThan(0);
  expect(jumpT90, 'new beam T90 frames').toBeLessThanOrEqual(6);
  expect(jumpGhostArea, 'jump ghost area').toBe(0);

  // Sweep at 5 cm per frame: each frame matches a temporal-off renderer at the same light position (no trail).
  const sweepGhost = Math.max(...result.sweep.map(frame => Math.max(...frame.on.map((value, index) => Math.abs(value - frame.off[index]))))) / peakA;
  const sweepGhostArea = Math.max(...result.sweep.map(frame => frame.on.filter((value, index) => Math.abs(value - frame.off[index]) > 0.02 * peakA).length));
  for (const frame of result.sweep) expect(frame).toMatchObject({ reason: 'local-light-change', framesSinceReset: 1 });
  expect(sweepGhost, 'sweep trail').toBeLessThanOrEqual(0.02);
  expect(sweepGhostArea, 'sweep ghost area').toBe(0);
  const settledRatio = Math.max(...beamA.map(index => Math.abs(litA2.values[index] / litA.values[index] - 1)));
  expect(settledRatio, 'back at A after the sweep').toBeLessThanOrEqual(0.02);

  // Short flashes from a dark field: full on their own frames, no residue afterwards.
  expect(Math.max(...result.dark.values)).toBeLessThanOrEqual(1e-6);
  const flashFrames = [result.flash1, ...result.flash2];
  const flashResponse = Math.min(...flashFrames.flatMap(frame => beamA.map(index => frame.values[index] / litA2.values[index])));
  const after = [...result.afterFlash1, ...result.afterFlash2];
  const flashResidual = Math.max(...after.flatMap(frame => frame.values)) / peakA;
  const flashGhostArea = Math.max(...after.map(frame => ghostArea(frame, result.dark, peakA)));
  expect(result.flash1).toMatchObject({ reason: 'local-light-change', framesSinceReset: 1 });
  expect(result.afterFlash1[0]).toMatchObject({ reason: 'local-light-change', framesSinceReset: 1 });
  expect(flashResponse, 'flash frames reach 90 % of the settled beam').toBeGreaterThanOrEqual(0.9);
  expect(flashResidual, 'after-flash residue').toBeLessThanOrEqual(0.02);
  expect(flashGhostArea, 'after-flash ghost area').toBe(0);

  // A caster enters the shadowed beam under local rejection: shadowed local lights force a global reset.
  expect(result.enterCurve[0]).toMatchObject({ reason: 'scene-change', framesSinceReset: 1 });
  expect(result.leaveCurve[0]).toMatchObject({ reason: 'scene-change', framesSinceReset: 1 });
  for (const index of beamA) expect(litA3.values[index] - shadowedA.values[index]).toBeGreaterThan(0.5 * litA3.values[index]);
  const enterResidual = Math.max(...beamA.map(index => Math.abs(result.enterCurve[0].values[index] - shadowedA.values[index])
    / (litA3.values[index] - shadowedA.values[index])));
  const enterT90 = t90(result.enterCurve, beamA, litA3, shadowedA), leaveT90 = t90(result.leaveCurve, beamA, shadowedA, result.settledAgain);
  expect(enterResidual, 'stale light on the first shadowed frame').toBeLessThanOrEqual(0.02);
  expect(enterT90, 'shadow T90 frames').toBeGreaterThan(0);
  expect(enterT90, 'shadow T90 frames').toBeLessThanOrEqual(6);
  expect(leaveT90, 'unshadow T90 frames').toBeGreaterThan(0);
  expect(leaveT90, 'unshadow T90 frames').toBeLessThanOrEqual(6);

  expect(result.localLights).toMatchObject({ lights: 1, shadowedLights: 1, cameraCulling: false });
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(result.twinAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  console.log('FOG_LOCAL_LIGHT_RESPONSE ' + JSON.stringify({
    jump: { residual: jumpResidual, t90: jumpT90, ghostArea: jumpGhostArea,
      oldBeam: ratios(result.jumpCurve, beamA, litA), newBeam: ratios(result.jumpCurve, beamB, litB) },
    sweep: { ghost: sweepGhost, ghostArea: sweepGhostArea, frames: result.sweep.length, settledRatio },
    flash: { response: flashResponse, residual: flashResidual, ghostArea: flashGhostArea,
      flashFrames: flashFrames.map(frame => beamA.map(index => frame.values[index] / litA2.values[index])),
      after: after.map(frame => Math.max(...frame.values) / peakA) },
    caster: { enterResidual, enterT90, leaveT90, enter: ratios(result.enterCurve, beamA, litA3), leave: ratios(result.leaveCurve, beamA, litA3) },
    peaks: { a: peakA, b: peakB }, adapter: result.adapter }));
});
