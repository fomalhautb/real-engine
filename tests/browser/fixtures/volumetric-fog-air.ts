import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import {
  AIR_EYE, AIR_COLUMN, AIR_MAX_VIEW_DEPTH, AIR_MEDIUM, AIR_MEDIUM_FORWARD, AIR_MIRROR, AIR_OPEN_GRID, AIR_RED, AIR_ROI,
  AIR_ROOM_GRID, AIR_ROOM_WALLS, AIR_SIZE, AIR_TARGET, type AirRectangle, type Vec3,
} from './volumetric-fog-air-cases';

function rectangle({ corner, u, v }: Pick<AirRectangle, 'corner' | 'u' | 'v'>): MeshData {
  const points = [corner, corner.map((c, i) => c + u[i]), corner.map((c, i) => c + u[i] + v[i]), corner.map((c, i) => c + v[i])];
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const length = Math.hypot(...n);
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

function state(entityId: number, metalness = 0, roughness = 1): InstanceState {
  return { entityId, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness, roughness, historyVersion: 0 };
}

function unlit(color: Vec3): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.constant([...color, 1]) } };
}

function batch(name: string, mesh: MeshData, entityId: number, material: MaterialBinding, instance = state(entityId)): SceneBatch {
  return { name, mesh, capacity: 1, inputMode: 'committed', instances: [instance], material,
    queryIdentity: { assetId: 'fog-air/' + name, primitiveId: entityId, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

function canvas(size: number): HTMLCanvasElement {
  const element = document.createElement('canvas');
  element.style.cssText = `position:fixed;left:0;top:0;width:${size}px;height:${size}px`;
  document.body.appendChild(element);
  return element;
}

function fogOptions(outsideSkyVisibility: number) {
  return { pixelSize: 2, slices: 64, maxViewDepth: AIR_MAX_VIEW_DEPTH, maxFroxels: 32 * 32 * 64, maxBytes: 8 * 1024 * 1024,
    temporal: { maxHistoryFrames: 8 }, airLighting: { outsideSkyVisibility } };
}

function splitCapture(renderer: Renderer, eye: Vec3, target: Vec3) {
  return async (pixels: readonly { x: number; y: number }[]) => {
    renderer.setVolumetricFogDebugView('scattering-transmittance');
    renderer.render(eye, target);
    const result = await renderer.probeHdr(pixels);
    renderer.setVolumetricFogDebugView('composite');
    return result;
  };
}

async function probes(renderer: Renderer) {
  const result = await renderer.probeDiffuseGi();
  return result.probes.map(probe => ({ coefficients: probe.coefficients.flatMap(value => value.slice(0, 3)),
    maxRadiance: probe.maxRadiance.slice(0, 3), valid: probe.valid }));
}

/** F04 open sky without geometry: every probe sees only the sky, so the air term has a closed form. */
async function openSky() {
  const element = canvas(AIR_SIZE);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, [], { antiAliasing: 'none', worldQueries: { maxRays: 4096 },
      diffuseGi: AIR_OPEN_GRID, volumetricFog: fogOptions(1) });
    const active = renderer, split = splitCapture(active, AIR_EYE, AIR_TARGET);
    const history = () => ({ ...active.diagnostics().volumetricFog!.history });
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    active.setVolumetricFogMedia({ global: { coefficients: AIR_MEDIUM } });
    active.setVolumetricFogTemporalEnabled(false);
    const isotropic = await split(AIR_ROI);
    const field = await probes(active);
    active.setVolumetricFogMedia({ global: { coefficients: AIR_MEDIUM_FORWARD } });
    const forward = await split(AIR_ROI);
    active.setVolumetricFogAirLighting({ gi: false, outsideSkyVisibility: 0 });
    const off = await split(AIR_ROI);
    // Sun and air sources add: the air pass adds its term after the sun source is resolved.
    active.setSun({ direction: [0, 0.6, 0.8], color: [8, 8, 8] });
    const sunOnly = await split(AIR_ROI);
    active.setVolumetricFogAirLighting({ gi: true, outsideSkyVisibility: 1 });
    const sunAndAir = await split(AIR_ROI);

    // GI-side changes never reset fog history, and the air term follows them on the first frame.
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    active.setVolumetricFogTemporalEnabled(true);
    for (let frame = 0; frame < 12; frame++) active.render(AIR_EYE, AIR_TARGET);
    const accumulated = await split(AIR_ROI);
    const beforeInvalidation = history();
    active.invalidateDiffuseGiHistory();
    const afterInvalidation = await split(AIR_ROI);
    const historyAfterInvalidation = history();
    active.setDiffuseGiEnabled(false);
    const giDisabledFirstFrame = await split(AIR_ROI);
    const historyGiDisabled = history();
    active.setDiffuseGiEnabled(true);
    const giEnabledFirstFrame = await split(AIR_ROI);
    const historyGiEnabled = history();
    const diagnostics = active.diagnostics().volumetricFog;
    const errors = [...active.errors], adapter = active.info.adapterInfo;
    active.dispose();
    return { isotropic, forward, off, sunOnly, sunAndAir, field, accumulated, afterInvalidation, giDisabledFirstFrame,
      giEnabledFirstFrame, beforeInvalidation, historyAfterInvalidation, historyGiDisabled, historyGiEnabled,
      diagnostics, errors, adapter, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

/** F04 room: closed (leak), an off-screen red wall behind the camera, and an open ceiling (courtyard). */
async function room() {
  const element = canvas(AIR_SIZE);
  const walls = AIR_ROOM_WALLS.map((wall, index) => batch('air-room-' + wall.name, rectangle(wall), 10 + index, unlit([0, 0, 0])));
  const named = (name: string) => walls[AIR_ROOM_WALLS.findIndex(wall => wall.name === name)];
  const red = batch('air-room-back-red', rectangle(AIR_ROOM_WALLS.find(wall => wall.name === 'back')!), 30, unlit(AIR_RED));
  red.instances = [];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, [...walls, red], { antiAliasing: 'none', worldQueries: { maxRays: 4096 },
      diffuseGi: AIR_ROOM_GRID, volumetricFog: fogOptions(0) });
    const active = renderer, split = splitCapture(active, AIR_EYE, AIR_TARGET);
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    active.setVolumetricFogMedia({ global: { coefficients: AIR_MEDIUM } });
    active.setVolumetricFogTemporalEnabled(false);
    const capture = async () => ({ roi: await split(AIR_ROI), column: await split(AIR_COLUMN), probes: await probes(active),
      surface: await active.probeSurfaceHdr(AIR_ROI) });
    const closed = await capture();
    const back = named('back'), backState = back.instances[0];
    back.instances = [];
    red.instances = [state(30)];
    const offscreenRed = await capture();
    red.instances = [];
    back.instances = [backState];
    const ceiling = named('ceiling');
    ceiling.instances = [];
    const courtyard = await capture();
    const errors = [...active.errors];
    active.dispose();
    return { closed, offscreenRed, courtyard, errors, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

/** F04 mirror: local fog behind the camera is reached only by reflection rays, which use the same air term. */
async function mirror() {
  const element = canvas(AIR_MIRROR.size);
  const mirrorBatch = batch('air-mirror', rectangle({ corner: [-3, -3, 0], u: [6, 0, 0], v: [0, 6, 0] }), 41,
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } }, state(41, 1, 0.045));
  const emitter = batch('air-mirror-emitter', rectangle({ corner: [-10, -10, AIR_MIRROR.emitterZ], u: [20, 0, 0], v: [0, 20, 0] }), 42,
    unlit(AIR_MIRROR.emitter));
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, [mirrorBatch, emitter], { antiAliasing: 'none', worldQueries: { maxRays: 4096 },
      diffuseGi: AIR_MIRROR.grid,
      reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4, maxDistance: 20, maxBytes: 16 * 1024 * 1024,
        skyRadiance: AIR_MIRROR.sky, fogSteps: AIR_MIRROR.fogSteps },
      volumetricFog: { pixelSize: 4, slices: 32, maxViewDepth: 10, maxFroxels: 8 * 8 * 32, maxBytes: 1024 * 1024, airLighting: {} } });
    const active = renderer, target: Vec3 = [0, 0, 0];
    const anchors = AIR_MIRROR.pixels.map(pixel => pixel.y * AIR_MIRROR.size + pixel.x);
    const reflected = async () => {
      for (let frame = 0; frame < 2; frame++) active.render(AIR_MIRROR.eye, target);
      return (await active.probeReflections(anchors)).anchors.map(anchor => ({ radiance: anchor.radiance, valid: anchor.valid,
        normal: anchor.normal, position: anchor.position }));
    };
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    active.setVolumetricFogMedia({ volumes: [{ id: 3, shape: { kind: 'box', min: AIR_MIRROR.box.min, max: AIR_MIRROR.box.max },
      coefficients: AIR_MIRROR.medium }] });
    const airOn = await reflected();
    const field = await probes(active);
    const mainView = await splitCapture(active, AIR_MIRROR.eye, target)(AIR_MIRROR.pixels);
    active.setVolumetricFogAirLighting({ gi: false, outsideSkyVisibility: 0 });
    const airOff = await reflected();
    active.setVolumetricFogMedia({ volumes: [] });
    const noFog = await reflected();
    const errors = [...active.errors];
    active.dispose();
    return { airOn, airOff, noFog, field, mainView, errors, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

export async function runFogAirAcceptanceFixture() {
  return { open: await openSky(), room: await room(), mirror: await mirror() };
}
