import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import type { LocalLightDescriptor } from '../../../src/local-lights';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity, normalMatrix, translation, type Mat4 } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import {
  HISTORY_EYE, HISTORY_LATTICE, HISTORY_MAX_FRAMES, HISTORY_MEDIUM, HISTORY_SIZE, HISTORY_TARGET, type Vec3,
} from './volumetric-fog-history-cases';

/** Downward shadowed spot above planar depth 3. It jumps and sweeps between x = -0.9 (A) and x = +0.9 (B);
 * the two beams never overlap, so a pixel lit by one is exactly dark under the other. */
const SPOT: LocalLightDescriptor = { kind: 'spot', position: [-0.9, 2.5, 3], direction: [0, -1, 0],
  innerConeAngle: 0.08, outerConeAngle: 0.15, color: [1, 1, 1], intensity: 100, range: 6, castShadows: true };
const B_POSITION: Vec3 = [0.9, 2.5, 3];
const SWEEP_FRAMES = 36;
/** A 0.4 m caster that enters beam A at y = 1 and otherwise parks outside every beam and the view. */
const OCCLUDER = { inside: [-0.9, 1, 3] as Vec3, parked: [3.5, 1, 3] as Vec3, half: 0.2 };
/** Zero-colour sun from the side: its light-space footprint of the caster misses the spot's shadow column, so
 * only the local-light rule (shadowed local lights force a global reset on caster changes) can clear it. */
const SUN = { direction: [0.97, 0.243, 0] as Vec3, color: [0, 0, 0] as Vec3 };

function quad(corner: Vec3, u: Vec3, v: Vec3): MeshData {
  const points = [corner, corner.map((c, i) => c + u[i]), corner.map((c, i) => c + u[i] + v[i]), corner.map((c, i) => c + v[i])];
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]], length = Math.hypot(...n);
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  points.forEach((point, index) => {
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(index * 80 + axis * 4, point[axis], true);
      view.setFloat32(index * 80 + 16 + axis * 4, n[axis] / length, true);
    }
    view.setFloat32(index * 80 + 12, 1, true);
    view.setFloat32(index * 80 + 48, 1, true);
  });
  const min = [0, 1, 2].map(axis => Math.min(...points.map(point => point[axis]))) as Vec3;
  const max = [0, 1, 2].map(axis => Math.max(...points.map(point => point[axis]))) as Vec3;
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), bounds: { min, max } };
}

function instance(entityId: number, model: Mat4 = identity()): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

const caster: MaterialBinding = { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, baseColor: m.constant([0.2, 0.2, 0.2, 1]) } };

async function rig(temporal: boolean) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:${temporal ? 0 : HISTORY_SIZE}px;top:0;width:${HISTORY_SIZE}px;height:${HISTORY_SIZE}px`;
  document.body.appendChild(canvas);
  const plain = (name: string, mesh: MeshData, entityId: number, material: MaterialBinding): SceneBatch => ({ name, mesh, capacity: 1,
    inputMode: 'committed', instances: [instance(entityId)], material, deformation: { jointCount: 0, nodes: [] }, morphTargets: [] });
  const receiver = plain('response-receiver', quad([-4, -4, 0], [8, 0, 0], [0, 8, 0]), 1,
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.constant([2, 1, 0.5, 1]) } });
  const h = OCCLUDER.half;
  const occluder = plain('response-occluder', quad([-h, 0, -h], [2 * h, 0, 0], [0, 0, 2 * h]), 2, caster);
  const place = (position: Vec3) => { occluder.instances = [instance(2, translation(...position))]; };
  place(OCCLUDER.parked);
  try {
    const renderer = await Renderer.create(canvas, [receiver, occluder], { antiAliasing: 'none',
      volumetricFog: { pixelSize: 2, slices: 64, maxViewDepth: 8, maxFroxels: 32 * 32 * 64, maxBytes: 8 * 1024 * 1024,
        ...(temporal ? { temporal: { maxHistoryFrames: HISTORY_MAX_FRAMES, occluderRejection: 'local' as const } } : {}) },
      localLights: { maxLights: 2, maxShadowedLights: 1, maxShadowViews: 1, shadowResolution: 512 } });
    renderer.setSun(SUN);
    renderer.setVolumetricFogMedia({ global: { coefficients: HISTORY_MEDIUM } });
    renderer.setVolumetricFogDebugView('scattering-transmittance');
    const light = renderer.addLocalLight(SPOT);
    /** One rendered frame: red in-scattering of every lattice pixel and the history state. */
    const frame = async () => {
      renderer.render(HISTORY_EYE, HISTORY_TARGET);
      const values = (await renderer.probeHdr(HISTORY_LATTICE)).map(value => value[0]);
      const history = renderer.diagnostics().volumetricFog!.history;
      return { values, reason: history.lastResetReason, framesSinceReset: history.framesSinceReset };
    };
    const frames = async (count: number) => { const curve = []; for (let i = 0; i < count; i++) curve.push(await frame()); return curve; };
    const settle = async (count = HISTORY_MAX_FRAMES) => (await frames(count)).at(-1)!;
    return { canvas, renderer, light, place, frame, frames, settle };
  } catch (error) { canvas.remove(); throw error; }
}

/** F03 for local lights: a spotlight jump and sweep, one- and two-frame flashes, and a caster entering its beam. */
export async function runLocalLightFogResponseFixture() {
  const main = await rig(true);
  let twin: Awaited<ReturnType<typeof rig>> | undefined;
  try {
    const { renderer, light } = main;
    const litA = await main.settle(24);
    // Jump A -> B in one frame: the old beam must vanish and the new one appear without a history tail.
    renderer.updateLocalLight(light, { position: B_POSITION });
    const jumpCurve = await main.frames(12);
    const litB = await main.settle();
    // Sweep B -> A over 36 frames (5 cm per frame), against a temporal-off twin driven identically.
    twin = await rig(false);
    twin.renderer.updateLocalLight(twin.light, { position: B_POSITION });
    const sweep = [];
    for (let k = 1; k <= SWEEP_FRAMES; k++) {
      const s = k / SWEEP_FRAMES;
      const position = B_POSITION.map((value, axis) => value + (SPOT.position[axis] - value) * s) as Vec3;
      renderer.updateLocalLight(light, { position });
      twin.renderer.updateLocalLight(twin.light, { position });
      const on = await main.frame(), off = await twin.frame();
      sweep.push({ on: on.values, off: off.values, reason: on.reason, framesSinceReset: on.framesSinceReset });
    }
    const twinErrors = [...twin.renderer.errors];
    twin.renderer.dispose();
    const twinAfterDispose = twin.renderer.stats.ownedGpuBytes;
    twin.canvas.remove(); twin = undefined;
    const litA2 = await main.settle();
    // Short flashes from a converged dark state: one frame, then two frames.
    renderer.setLocalLightEnabled(light, false);
    const dark = await main.settle();
    renderer.setLocalLightEnabled(light, true);
    const flash1 = await main.frame();
    renderer.setLocalLightEnabled(light, false);
    const afterFlash1 = await main.frames(8);
    renderer.setLocalLightEnabled(light, true);
    const flash2 = await main.frames(2);
    renderer.setLocalLightEnabled(light, false);
    const afterFlash2 = await main.frames(8);
    // A caster enters and leaves the shadowed beam under local occluder rejection.
    renderer.setLocalLightEnabled(light, true);
    const litA3 = await main.settle();
    main.place(OCCLUDER.inside);
    const enterCurve = await main.frames(8);
    const shadowedA = await main.settle();
    main.place(OCCLUDER.parked);
    const leaveCurve = await main.frames(8);
    const settledAgain = await main.settle();
    const localLights = renderer.diagnostics().localLights;
    const errors = [...renderer.errors, ...twinErrors], adapter = renderer.info.adapterInfo;
    renderer.dispose();
    return { litA, jumpCurve, litB, sweep, litA2, dark, flash1, afterFlash1, flash2, afterFlash2, litA3, enterCurve, shadowedA,
      leaveCurve, settledAgain, localLights, errors, adapter, afterDispose: renderer.stats.ownedGpuBytes, twinAfterDispose };
  } finally {
    main.renderer.dispose(); main.canvas.remove();
    if (twin) { twin.renderer.dispose(); twin.canvas.remove(); }
  }
}
