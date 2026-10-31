import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createCharacter, createCube } from '../../../src/geometry';
import { multiply, quaternionMatrix, rotationX, scaling, translation, type Mat4 } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { PhysicsWorld } from '../../../src/physics';
import { Renderer } from '../../../src/renderer';
import type { WorldRay } from '../../../src/world-query';
import { instance, quad } from './reflections-temporal';

export const DYNAMIC_SIZE = 64;
export const DYNAMIC_EYE: [number, number, number] = [0, 0, 3];
const TARGET: [number, number, number] = [0, 0, 0];
export const DYNAMIC_IDS = Object.freeze({ mirror: 1, character: 2, side: 3, probe: 4, boxes: [10, 11, 12] as const });
const BOX_HALF: [number, number, number] = [0.3, 0.3, 0.3];
const SKY: [number, number, number] = [0.05, 0.05, 0.08];
const CHARACTER_COLOR: [number, number, number] = [2.2, 1.3, 0.3];
const BOX_COLOR: [number, number, number] = [0.3, 1.8, 2.2];
const SIDE_COLOR: [number, number, number] = [2.5, 0.4, 1.5];
const unlit = { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const, doubleSided: true, baseColor: m.instanceColor };

function emitter(entityId: number, model: Mat4, color: readonly number[], historyVersion = 0): InstanceState {
  return { ...instance(entityId, model, 1, 0, [color[0], color[1], color[2], 1]), historyVersion };
}
/** Arms swing about the shoulder pivots; root and head stay rigid. */
function characterBones(phase: number): Mat4[] {
  const shoulder = (x: number, angle: number) => multiply(translation(x, 1.95, 0), multiply(rotationX(angle), translation(-x, -1.95, 0)));
  return [translation(0, 0, 0), translation(0, 0, 0), shoulder(-0.55, 0.9 * Math.sin(phase)), shoulder(0.55, -0.9 * Math.sin(phase))];
}
function characterState(x: number, phase: number): InstanceState {
  const model = translation(x, -2, 6.2);
  return { ...emitter(DYNAMIC_IDS.character, model, CHARACTER_COLOR), bones: characterBones(phase) };
}
function boxModel(position: readonly number[], rotation: readonly number[]): Mat4 {
  return multiply(translation(position[0], position[1], position[2]),
    multiply(quaternionMatrix(rotation[0], rotation[1], rotation[2], rotation[3]), scaling(BOX_HALF[0] * 2, BOX_HALF[1] * 2, BOX_HALF[2] * 2)));
}
const SIDE_CENTER: [number, number, number] = [2.2, 0, 1.8];
function sideState(phase: number): InstanceState {
  return emitter(DYNAMIC_IDS.side, multiply(translation(SIDE_CENTER[0], 0.35 * Math.sin(phase), SIDE_CENTER[2]), scaling(0.5, 0.5, 0.5)), SIDE_COLOR);
}
/** The exact mirror direction about the receiver's reported shading normal. */
function mirrorRay(position: readonly number[], normal: readonly number[]): WorldRay {
  const view = position.map((value, axis) => value - DYNAMIC_EYE[axis]), length = Math.hypot(...view);
  const w = view.map(value => value / length), dot = w[0] * normal[0] + w[1] * normal[1] + w[2] * normal[2];
  const direction = w.map((value, axis) => value - 2 * dot * normal[axis]) as [number, number, number];
  return { origin: position.map((value, axis) => value + normal[axis] * 0.0001) as [number, number, number], direction, tMax: 20 };
}
/** Conservative frustum test of a bounding sphere for the fixed 1:1 camera at DYNAMIC_EYE. */
function intersectsFrustum(center: readonly number[], radius: number): boolean {
  const depth = DYNAMIC_EYE[2] - center[2], slope = Math.tan(Math.PI / 6.4), pad = radius * Math.hypot(1, slope);
  return depth + radius > 0.1 && Math.abs(center[0] - DYNAMIC_EYE[0]) <= slope * depth + pad
    && Math.abs(center[1] - DYNAMIC_EYE[1]) <= slope * depth + pad;
}

type Stage = { label: string; roughness: number; reason: string; frames: { valid: boolean[]; position: number[][]; radiance: number[][]; raw: number[][];
  count: number[]; mode: string[]; rejected: number[]; clamped: boolean[]; reason: string | undefined }[] };

/** D02: a skinned character and Rapier rigid bodies behind the camera plus a
 * moving object beside the mirror, none inside the camera frustum; then
 * response curves for discrete changes seen in smooth and rough mirrors. */
export async function runReflectionsDynamicFixture(options: { motionFrames: number; probeEvery: number }) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${DYNAMIC_SIZE}px;height:${DYNAMIC_SIZE}px`;
  document.body.appendChild(canvas);
  const physics = await PhysicsWorld.create();
  const starts: [number, number, number][] = [[0.6, 1.6, 5.4], [1.2, 2.4, 5.6], [1.8, 3.2, 5.3]];
  const mirrorState = (roughness: number) => instance(DYNAMIC_IDS.mirror, scaling(3, 3, 1), roughness);
  const mirror: SceneBatch = { name: 'dynamic-mirror', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [mirrorState(0.045)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } } };
  const character: SceneBatch = { name: 'offscreen-skinned-character', mesh: createCharacter(), capacity: 1, inputMode: 'committed',
    instances: [characterState(-1.1, 0)], deformation: { jointCount: 4, nodes: [{ kind: 'skin' }] }, morphTargets: [],
    material: { definition: unlit } };
  const boxes: SceneBatch = { name: 'offscreen-rigid-bodies', mesh: createCube(), capacity: DYNAMIC_IDS.boxes.length, inputMode: 'committed',
    instances: DYNAMIC_IDS.boxes.map((id, index) => emitter(id, boxModel(starts[index], [0, 0, 0, 1]), BOX_COLOR)),
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: { definition: unlit } };
  const side: SceneBatch = { name: 'beside-mirror-object', mesh: createCube(), capacity: 1, inputMode: 'committed',
    instances: [sideState(0)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: { definition: unlit } };
  // The probe emitter's color is instance tint times one runtime material parameter.
  const probe: SceneBatch = { name: 'response-probe-emitter', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...unlit, baseColor: m.multiply(m.instanceColor, m.parameter(0)) }, parameters: new Float32Array([1, 1, 1, 1]) } };
  let renderer: Renderer | undefined;
  try {
    DYNAMIC_IDS.boxes.forEach((id, index) => physics.createBox(id, starts[index], BOX_HALF));
    renderer = await Renderer.create(canvas, [mirror, character, boxes, side, probe], { antiAliasing: 'none',
      worldQueries: { maxRays: DYNAMIC_SIZE * DYNAMIC_SIZE * 4 },
      reflections: { pixelSpacing: 1, maxAnchors: DYNAMIC_SIZE * DYNAMIC_SIZE, samplesPerAnchor: 4, maxDistance: 20,
        maxBytes: 16 * 1024 * 1024, skyRadiance: SKY, temporal: { minHistoryFrames: 8, maxHistoryFrames: 32 } } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const grid: number[] = [];
    for (let y = 1; y < DYNAMIC_SIZE; y += 2) for (let x = 1; x < DYNAMIC_SIZE; x += 2) grid.push(y * DYNAMIC_SIZE + x);

    // Phase 1: continuous motion. Every probed anchor's dominant reflected hit is
    // compared with an exact mirror ray traced against the same deformation snapshot.
    const motion = [];
    let phase = 0, characterX = -1.1;
    for (let frame = 0; frame < options.motionFrames; frame++) {
      physics.step(1 / 30);
      const poses = DYNAMIC_IDS.boxes.map(id => physics.pose(id, 1)!);
      boxes.instances = DYNAMIC_IDS.boxes.map((id, index) => emitter(id, boxModel(poses[index].position, poses[index].rotation), BOX_COLOR));
      phase += 0.12; characterX += 0.035;
      character.instances = [characterState(characterX, phase)];
      side.instances = [sideState(phase)];
      active.render(DYNAMIC_EYE, TARGET);
      if (frame % options.probeEvery !== options.probeEvery - 1) continue;
      const result = await active.probeReflections(grid);
      const traced = await active.traceRays(result.anchors.map(anchor => mirrorRay(anchor.position, anchor.normal)));
      motion.push({ frame: result.frame, anchors: result.anchors.map((anchor, i) => ({
        index: anchor.index, valid: anchor.valid, entity: anchor.reflectedIdentity[0] ? anchor.reflectedIdentity[1] : 0,
        expected: traced.hits[i]?.entityId ?? 0, radiance: anchor.radiance, raw: anchor.rawRadiance, count: anchor.historyCount,
        mode: anchor.historyMode, rejected: anchor.rejectedHistory, clamped: anchor.historyClamped })),
        boxPositions: poses.map(pose => [...pose.position]),
        inFrustum: { character: intersectsFrustum([characterX, -0.5, 6.2], 1.8),
          boxes: poses.map(pose => intersectsFrustum(pose.position, Math.hypot(...BOX_HALF))),
          side: intersectsFrustum(SIDE_CENTER, 0.6) },
        history: active.diagnostics().reflections?.temporal });
    }

    // Phase 2: response curves at anchors that reflect the probe emitter's interior.
    const center: number[] = [];
    const middle = DYNAMIC_SIZE / 2;
    for (let y = middle - 2; y < middle + 2; y++) for (let x = middle - 2; x < middle + 2; x++) center.push(y * DYNAMIC_SIZE + x);
    character.instances = []; boxes.instances = []; side.instances = [];
    const stages: Stage[] = [];
    const record = async (label: string, roughness: number, reason: string, frames: number) => {
      const stage: Stage = { label, roughness, reason, frames: [] };
      stages.push(stage);
      for (let i = 0; i < frames; i++) {
        active.render(DYNAMIC_EYE, TARGET);
        const result = await active.probeReflections(center);
        stage.frames.push({ valid: result.anchors.map(anchor => anchor.valid), position: result.anchors.map(anchor => anchor.position),
          radiance: result.anchors.map(anchor => anchor.radiance), raw: result.anchors.map(anchor => anchor.rawRadiance),
          count: result.anchors.map(anchor => anchor.historyCount), mode: result.anchors.map(anchor => anchor.historyMode),
          rejected: result.anchors.map(anchor => anchor.rejectedHistory), clamped: result.anchors.map(anchor => anchor.historyClamped),
          reason: active.diagnostics().reflections?.temporal.lastResetReason });
      }
    };
    const away = multiply(translation(9, 0, 5), scaling(0.6, 0.6, 1)), here = multiply(translation(0, 0, 5), scaling(0.6, 0.6, 1));
    const white = [3, 3, 3], red = [3, 0.3, 0.3];
    for (const roughness of [0.045, 0.3]) {
      mirror.instances = [mirrorState(roughness)];
      probe.instances = [emitter(DYNAMIC_IDS.probe, away, white)];
      active.setMaterialParameters(probe.name, new Float32Array([1, 1, 1, 1]));
      await record('settle', roughness, 'previous stage', 40);
      // Same history version: the emitter's motion vector spans the 9 m jump.
      probe.instances = [emitter(DYNAMIC_IDS.probe, here, white)];
      await record('teleport-in', roughness, 'reflected object moved into the mirror lobe', 40);
      probe.instances = [emitter(DYNAMIC_IDS.probe, here, red)];
      await record('instance-color', roughness, 'reflected object instance tint changed', 32);
      active.setMaterialParameters(probe.name, new Float32Array([0.1, 1, 1, 1]));
      await record('material-parameter', roughness, 'reflected material parameter changed', 32);
      active.setMaterialParameters(probe.name, new Float32Array([1, 1, 1, 1]));
      await record('material-restore', roughness, 'reflected material parameter restored', 32);
      // Only the history version and the instance tint change: the tint alone is
      // not a reset trigger, so the response comes from the version contract.
      probe.instances = [emitter(DYNAMIC_IDS.probe, here, white, roughness === 0.3 ? 2 : 1)];
      await record('history-version', roughness, 'reflected object history version bumped', 32);
      probe.instances = [emitter(DYNAMIC_IDS.probe, away, white, roughness === 0.3 ? 2 : 1)];
      await record('teleport-away', roughness, 'reflected object left the mirror lobe', 32);
    }
    // Receiver roughness change with fixed reflected content.
    probe.instances = [emitter(DYNAMIC_IDS.probe, here, white, 3)];
    mirror.instances = [mirrorState(0.3)];
    await record('settle', 0.3, 'previous stage', 40);
    mirror.instances = [mirrorState(0.045)];
    await record('receiver-roughness', 0.045, 'receiver roughness 0.3 -> 0.045', 32);
    const errors = [...active.errors];
    const diagnostics = active.diagnostics().reflections;
    active.dispose();
    return { adapter: active.info.adapterInfo, size: DYNAMIC_SIZE, sky: SKY, ids: DYNAMIC_IDS,
      colors: { character: CHARACTER_COLOR, boxes: BOX_COLOR, side: SIDE_COLOR }, motion, stages, diagnostics, errors,
      ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); physics.dispose(); canvas.remove(); }
}
