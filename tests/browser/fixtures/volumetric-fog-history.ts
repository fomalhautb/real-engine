import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import { createCube, type MeshData } from '../../../src/geometry';
import { identity, multiply, normalMatrix, scaling, translation, type Mat4 } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import {
  HISTORY_BOX, HISTORY_BOX_PIXEL, HISTORY_CELLS, HISTORY_CHARACTER, HISTORY_DOOR_OPEN_OFFSET, HISTORY_EYE, HISTORY_GHOST_PIXEL,
  HISTORY_LATTICE, HISTORY_MAX_FRAMES, HISTORY_MEDIUM, HISTORY_MOVED_BOX_PIXEL, HISTORY_PIXELS, HISTORY_ROOF, HISTORY_ROOF_Y,
  HISTORY_SIZE, HISTORY_STABILITY_FRAMES, HISTORY_SUN, HISTORY_TARGET, HISTORY_TELEPORT_EYE, HISTORY_WINDOW,
  type HistoryQuad, type Vec3,
} from './volumetric-fog-history-cases';

type Pixel = { x: number; y: number };

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

function roofQuad(piece: HistoryQuad): MeshData {
  return quad([piece.min[0], HISTORY_ROOF_Y, piece.min[1]], [piece.max[0] - piece.min[0], 0, 0], [0, 0, piece.max[1] - piece.min[1]]);
}

function instance(entityId: number, model: Mat4 = identity(), bones: Float32Array[] = []): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones, deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

const caster: MaterialBinding = { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, baseColor: m.constant([0.2, 0.2, 0.2, 1]) } };

function scene() {
  const plain = (name: string, mesh: MeshData, entityId: number, material: MaterialBinding): SceneBatch => ({ name, mesh, capacity: 1,
    inputMode: 'committed', instances: [instance(entityId)], material, deformation: { jointCount: 0, nodes: [] }, morphTargets: [] });
  const receiver = plain('history-receiver', quad([-4, -4, 0], [8, 0, 0], [0, 8, 0]), 1,
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.constant([2, 1, 0.5, 1]) } });
  const roof = HISTORY_ROOF.map((piece, index) => plain('history-' + piece.name, roofQuad(piece), 10 + index, caster));
  const door = plain('history-door', roofQuad(HISTORY_WINDOW), 20, caster);
  // Open: slid onto the right roof piece (just above it), where it changes no lighting.
  const doorAt = (open: boolean) => instance(20, translation(open ? HISTORY_DOOR_OPEN_OFFSET : 0, open ? 0.01 : 0, 0));
  const cube = createCube(), rest = new DataView(cube.vertices);
  for (let vertex = 0; vertex < cube.vertexCount; vertex++) {
    for (let joint = 0; joint < 4; joint++) rest.setUint32(vertex * 80 + 32 + joint * 4, 0, true);
  }
  const characterModel = multiply(translation(0, HISTORY_CHARACTER.y, HISTORY_CHARACTER.z), scaling(HISTORY_CHARACTER.scale, HISTORY_CHARACTER.scale, HISTORY_CHARACTER.scale));
  const characterAt = (x: number) => instance(30, characterModel, [translation(x / HISTORY_CHARACTER.scale, 0, 0)]);
  const character: SceneBatch = { name: 'history-skinned-character', mesh: cube, capacity: 1, inputMode: 'committed',
    instances: [characterAt(HISTORY_CHARACTER.path[0])], material: caster,
    deformation: { jointCount: 1, nodes: [{ kind: 'skin' }] }, morphTargets: [] };
  return { batches: [receiver, ...roof, door, character], door, doorAt, character, characterAt };
}

async function run(occluderRejection: 'local' | 'global', evidence: boolean) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${HISTORY_SIZE}px;height:${HISTORY_SIZE}px`;
  document.body.appendChild(canvas);
  const { batches, door, doorAt, character, characterAt } = scene();
  door.instances = [doorAt(true)];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches, { antiAliasing: 'none',
      volumetricFog: { pixelSize: 2, slices: 64, maxViewDepth: 8, maxFroxels: 32 * 32 * 64, maxBytes: 8 * 1024 * 1024,
        temporal: { maxHistoryFrames: HISTORY_MAX_FRAMES, occluderRejection } } });
    const active = renderer;
    let eye = HISTORY_EYE;
    active.setSun(HISTORY_SUN);
    active.setVolumetricFogMedia({ global: { coefficients: HISTORY_MEDIUM } });
    active.setVolumetricFogDebugView('scattering-transmittance');
    const cells = [HISTORY_CELLS.shaft, HISTORY_CELLS.under, HISTORY_CELLS.far];
    const frame = async (pixels: readonly Pixel[] = HISTORY_PIXELS) => {
      active.render(eye, HISTORY_TARGET);
      const [split, history] = await Promise.all([active.probeHdr(pixels), active.probeVolumetricFogHistory(cells)]);
      const diagnostics = active.diagnostics().volumetricFog!;
      return { split, cells: history.map(cell => ({ count: cell.count, supportRadius: cell.supportRadius })),
        history: { ...diagnostics.history }, regions: diagnostics.history.occluderRegions };
    };
    const settle = async (frames: number, pixels?: readonly Pixel[]) => {
      let last = await frame(pixels);
      for (let i = 1; i < frames; i++) last = await frame(pixels);
      return last;
    };
    const characterPixels = [...HISTORY_PIXELS, HISTORY_GHOST_PIXEL];
    // Without history, the raw sequence of the same scene (one jittered frame per sample).
    active.setVolumetricFogTemporalEnabled(false);
    active.render(eye, HISTORY_TARGET);
    const raw = await active.probeHdr(characterPixels);
    active.setVolumetricFogTemporalEnabled(true);
    const clear = await settle(24, characterPixels);

    // F02: a skinned character enters the shaft, converges, then leaves; first frames show no ghost.
    character.instances = [characterAt(HISTORY_CHARACTER.path[1])];
    const enteredFirst = await frame(characterPixels);
    const entered = await settle(HISTORY_MAX_FRAMES, characterPixels);
    character.instances = [characterAt(HISTORY_CHARACTER.path[2])];
    const leftFirst = await frame(characterPixels);
    const left = await settle(HISTORY_MAX_FRAMES, characterPixels);

    // F03: door closes (thin-wall leak), then opens; brightness curves from the first changed frame.
    door.instances = [doorAt(false)];
    const closeCurve = [];
    for (let i = 0; i < 12; i++) closeCurve.push(await frame());
    const closedLattice = (await frame(HISTORY_LATTICE)).split;
    door.instances = [doorAt(true)];
    const openCurve = [];
    for (let i = 0; i < 12; i++) openCurve.push(await frame());
    const openLattice = (await settle(HISTORY_MAX_FRAMES, HISTORY_LATTICE)).split;

    const localExtras = async () => {
      // 120 frames of a fixed scene and camera.
      const stability = [];
      for (let i = 0; i < HISTORY_STABILITY_FRAMES; i++) stability.push((await frame(HISTORY_LATTICE)).split);
      // Teleport: world-space history stays valid where it reprojects; the rest restarts on that frame.
      eye = HISTORY_TELEPORT_EYE;
      const teleportCurve = [];
      for (let i = 0; i < 60; i++) teleportCurve.push((await frame(HISTORY_LATTICE)).split);
      const teleportConverged = (await settle(HISTORY_MAX_FRAMES, HISTORY_LATTICE)).split;
      const teleportHistory = { ...active.diagnostics().volumetricFog!.history };
      eye = HISTORY_EYE;
      await settle(HISTORY_MAX_FRAMES);
      // Fog volume add / move / remove: explicit media resets, no stale fog on the first frame.
      const boxPixels = [HISTORY_BOX_PIXEL, HISTORY_MOVED_BOX_PIXEL];
      const media = (offset: Vec3 | null) => active.setVolumetricFogMedia({ global: { coefficients: HISTORY_MEDIUM },
        volumes: offset ? [{ id: 5, shape: { kind: 'box', min: HISTORY_BOX.min.map((v, i) => v + offset[i]) as Vec3,
          max: HISTORY_BOX.max.map((v, i) => v + offset[i]) as Vec3 }, coefficients: HISTORY_BOX.coefficients }] : [] });
      const beforeBox = await settle(HISTORY_MAX_FRAMES, boxPixels);
      media([0, 0, 0]);
      const added = await frame(boxPixels);
      const addedConverged = await settle(HISTORY_MAX_FRAMES, boxPixels);
      media(HISTORY_BOX.moved);
      const moved = await frame(boxPixels);
      media(null);
      const removed = await frame(boxPixels);
      // Fog on/off evidence in the composite view, same exposure (linear HDR probes; PNG when requested).
      active.setVolumetricFogDebugView('composite');
      const image = () => { active.render(eye, HISTORY_TARGET); try { return evidence ? canvas.toDataURL('image/png') : null; } catch { return null; } };
      await settle(HISTORY_MAX_FRAMES);
      const fogOnImage = image();
      const fogOn = await active.probeHdr(HISTORY_LATTICE);
      active.setVolumetricFogEnabled(false);
      const fogOffImage = image();
      const fogOff = await active.probeHdr(HISTORY_LATTICE);
      return { stability, teleportCurve, teleportConverged, teleportHistory, beforeBox, added, addedConverged, moved, removed,
        fogOn, fogOff, fogOnImage, fogOffImage };
    };
    const extra = occluderRejection === 'local' ? await localExtras() : undefined;
    const errors = [...active.errors], adapter = active.info.adapterInfo, ownedBytes = active.stats.ownedGpuBytes;
    const fog = active.diagnostics().volumetricFog;
    active.dispose();
    return { occluderRejection, raw, clear, enteredFirst, entered, leftFirst, left, closeCurve, closedLattice, openCurve, openLattice,
      extra, ownedBytes, fog, errors, adapter, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

export async function runFogHistoryAcceptanceFixture(evidence = false) {
  return { local: await run('local', evidence), global: await run('global', false) };
}
