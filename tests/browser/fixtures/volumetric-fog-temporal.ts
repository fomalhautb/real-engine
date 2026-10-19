import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createCube, type MeshData } from '../../../src/geometry';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import type { MediumCoefficients } from '../../../src/participating-media';
import { Renderer } from '../../../src/renderer';

const EYE: [number, number, number] = [0, 0, 6];
const TARGET: [number, number, number] = [0, 0, 0];
const PIXELS = [{ x: 32, y: 32 }, { x: 25, y: 29 }];
const CELLS = [{ x: 8, y: 8, z: 12 }, { x: 7, y: 7, z: 20 }];
// Parallel to the receiver: this test's analytic medium is uniformly lit even
// inside the last cell straddling its depth. Air-shadow transitions have a
// separate accumulated roof test and the independent slat integral reference.
const SUN = { direction: [0, 1, 0] as [number, number, number],
  color: [10, 6, 3] as [number, number, number] };
const MEDIUM: MediumCoefficients = { extinction: 0.2, albedo: [0.5, 0.4, 0.3],
  emission: [0.04, 0.02, 0.01], anisotropy: 0 };

function plane(): MeshData {
  const positions = [[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  for (let index = 0; index < 4; index++) {
    const base = index * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(base + axis * 4, positions[index][axis], true);
    view.setFloat32(base + 12, 1, true);
    view.setFloat32(base + 24, 1, true);
    view.setFloat32(base + 48, 1, true);
  }
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-4, -4, 0], max: [4, 4, 0] } };
}

function instance(z: number, historyVersion: number): InstanceState {
  return { entityId: 1201, model: translation(0, 0, z), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5,
    historyVersion };
}

/** Real WebGPU render and explicitly bounded diagnostic readback. */
export async function runVolumetricFogTemporalFixture(inputMode: 'committed' | 'mutable' = 'committed') {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px';
  document.body.appendChild(canvas);
  const batch: SceneBatch = { name: 'temporal-fog-plane', mesh: plane(), capacity: 1,
    inputMode, instances: [instance(0, 0)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.parameter(0) }, parameters: new Float32Array([2, 1, 0.5, 1]) } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], { antiAliasing: 'none',
      volumetricFog: { pixelSize: 4, slices: 32, maxViewDepth: 12,
        maxFroxels: 20 * 16 * 32, maxBytes: 1024 * 1024,
        temporal: { maxHistoryFrames: 8 } } });
    const active = renderer;
    active.setSun(SUN);
    active.setVolumetricFogMedia({ global: { coefficients: MEDIUM } });
    const capture = async (eye = EYE, target = TARGET, cells = CELLS) => {
      active.render(eye, target);
      const [history, hdr, surface] = await Promise.all([
        active.probeVolumetricFogHistory(cells), active.probeHdr(PIXELS), active.probeSurfaceHdr(PIXELS),
      ]);
      return { history, hdr, surface, diagnostics: active.diagnostics().volumetricFog,
        size: { width: active.stats.width, height: active.stats.height } };
    };
    const rawForSameView = async (eye = EYE, target = TARGET) => {
      active.setVolumetricFogTemporalEnabled(false);
      active.render(eye, target);
      const hdr = await active.probeHdr(PIXELS);
      active.setVolumetricFogTemporalEnabled(true);
      return hdr;
    };
    const warm = async (eye = EYE, target = TARGET) => {
      active.render(eye, target);
      active.render(eye, target);
    };

    // Commit once before measuring resets; the first stationary scene update may be dirty.
    await warm();
    active.setVolumetricFogMedia({ global: { coefficients: MEDIUM } });
    const first = await capture();
    const countGrowth = [first.history.map(cell => cell.count)];
    let stable = first;
    for (let frame = 0; frame < 10; frame++) {
      stable = await capture();
      countGrowth.push(stable.history.map(cell => cell.count));
    }
    const shiftedEye: [number, number, number] = [0.03, 0, 6];
    const shiftedTarget: [number, number, number] = [0.03, 0, 0];
    const translated = await capture(shiftedEye, shiftedTarget);

    active.invalidateHistory();
    const cameraCut = await capture(shiftedEye, shiftedTarget);
    const cameraCutRaw = await rawForSameView(shiftedEye, shiftedTarget);
    await warm();

    active.setSun({ direction: SUN.direction, color: [15, 7, 4] });
    const lightChanged = await capture();
    const lightChangedRaw = await rawForSameView();
    await warm();

    const changedMedium: MediumCoefficients = { ...MEDIUM, emission: [0.08, 0.03, 0.015] };
    active.setVolumetricFogMedia({ global: { coefficients: changedMedium } });
    const mediaChanged = await capture();
    const mediaChangedRaw = await rawForSameView();
    await warm();

    if (inputMode === 'mutable') batch.instances[0].model[14] = -0.5;
    else batch.instances[0] = instance(-0.5, 1);
    const geometryChanged = await capture();
    const geometryChangedRaw = await rawForSameView();
    await warm();

    active.setMaterialParameters(batch.name, new Float32Array([1.5, 0.7, 0.3, 1]));
    const materialChanged = await capture();
    const materialChangedRaw = await rawForSameView();
    await warm();

    active.setVolumetricFogMedia({ global: { coefficients: { ...MEDIUM, anisotropy: 0.8 } } });
    active.setSun({ direction: [1, 0, 0], color: SUN.color });
    for (let frame = 0; frame < 12; frame++) active.render(EYE, TARGET);
    const directionalStable = await capture();
    const directionalEye: [number, number, number] = [0.08, 0, 6];
    const directionalTarget: [number, number, number] = [0.08, 0, 0];
    const directionalMoved = await capture(directionalEye, directionalTarget);
    const directionalRaw = await rawForSameView(directionalEye, directionalTarget);

    const bytesBeforeResize = active.stats.ownedGpuBytes;
    canvas.style.width = '80px';
    const resized = await capture(EYE, TARGET, [{ x: 10, y: 8, z: 12 }]);
    active.setVolumetricFogEnabled(false);
    active.render(EYE, TARGET);
    const disabled = { hdr: await active.probeHdr(PIXELS), surface: await active.probeSurfaceHdr(PIXELS),
      diagnostics: active.diagnostics().volumetricFog };
    active.setVolumetricFogEnabled(true);
    const enabledAgain = await capture(EYE, TARGET, [{ x: 10, y: 8, z: 12 }]);
    const errors = [...active.errors], frames = active.stats.frames;
    active.dispose();
    return { pixels: PIXELS, cells: CELLS, eye: EYE, sun: SUN, medium: MEDIUM,
      maxViewDepth: 12, first, stable, countGrowth, translated, cameraCut, cameraCutRaw,
      lightChanged, lightChangedRaw, mediaChanged, mediaChangedRaw,
      geometryChanged, geometryChangedRaw, materialChanged, materialChangedRaw,
      directionalStable, directionalMoved, directionalRaw,
      resized, disabled, enabledAgain, bytesBeforeResize, errors, frames,
      ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Offscreen masked and deformed roof tests the current frame's sun shadow in the air itself. */
export async function runVolumetricFogTemporalShadowFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px';
  document.body.appendChild(canvas);
  const mesh = createCube(), rest = new DataView(mesh.vertices);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    for (let joint = 0; joint < 4; joint++) rest.setUint32(vertex * 80 + 32 + joint * 4, 0, true);
  }
  const positions = new Float32Array(mesh.vertexCount * 3);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) positions[vertex * 3] = 2;
  const model = multiply(translation(0, 3, 3), scaling(20, 0.2, 20));
  const roof = (morph: number, bone: Float32Array): InstanceState => ({ entityId: 1,
    model, normalMatrix: normalMatrix(model), bones: [bone], deformScalars: [morph],
    color: [1, 1, 1, 1], roughness: 1, metalness: 0, historyVersion: 0 });
  const batch: SceneBatch = { name: 'temporal-offscreen-roof', mesh, capacity: 1,
    inputMode: 'committed', instances: [roof(0, identity())],
    deformation: { jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' }] },
    morphTargets: [{ positions, normals: new Float32Array(mesh.vertexCount * 3) }],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
      alphaMode: 'mask', alphaCutoff: 0.5, baseColor: m.parameter(0) },
      parameters: new Float32Array([0.5, 0.5, 0.5, 0]) } };
  const pixels = [16, 32, 48].flatMap(y => [16, 32, 48].map(x => ({ x, y })));
  const cells = [{ x: 8, y: 8, z: 12 }, { x: 5, y: 8, z: 12 }];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], { antiAliasing: 'none',
      volumetricFog: { pixelSize: 4, slices: 32, maxViewDepth: 6,
        maxFroxels: 16 * 16 * 32, maxBytes: 1024 * 1024,
        temporal: { maxHistoryFrames: 8 } } });
    const active = renderer;
    active.setVolumetricFogMedia({ global: { coefficients: {
      extinction: 0.2, albedo: [0.9, 0.7, 0.5], emission: [0, 0, 0], anisotropy: 0,
    } } });
    active.setSun({ direction: [0, 1, 0.1], color: [6, 4, 2] });
    const capture = async () => {
      active.render([0, 0, 6], [0, 0, 0]);
      const [hdr, history] = await Promise.all([
        active.probeHdr(pixels), active.probeVolumetricFogHistory(cells),
      ]);
      return { hdr, history, diagnostics: active.diagnostics().volumetricFog };
    };
    let clear = await capture();
    for (let frame = 0; frame < 12; frame++) clear = await capture();
    active.setMaterialParameters(batch.name, new Float32Array([0.5, 0.5, 0.5, 1]));
    const opaque = await capture();
    batch.instances[0] = roof(1, identity());
    const morphed = await capture();
    batch.instances[0] = roof(0, identity());
    const restoredAfterMorph = await capture();
    batch.instances[0] = roof(0, translation(2, 0, 0));
    const skinned = await capture();
    batch.instances[0] = roof(0, identity());
    const restoredAfterSkin = await capture();
    batch.instances = [];
    const deleted = await capture();
    batch.instances = [roof(0, identity())];
    active.setSun({ direction: [0, 1, 0.1], color: [0, 0, 0] });
    const zeroSun = await capture();
    const errors = [...active.errors];
    active.dispose();
    return { pixels, cells, clear, opaque, morphed, restoredAfterMorph,
      skinned, restoredAfterSkin, deleted, zeroSun, errors,
      ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
