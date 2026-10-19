import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

const EYE: [number, number, number] = [0, 0, 3];
const TARGET: [number, number, number] = [0, 0, 0];
const PIXELS = [{ x: 16, y: 16 }, { x: 15, y: 16 }, { x: 17, y: 16 }];
const ANCHORS = PIXELS.map(pixel => pixel.y * 32 + pixel.x);
const EMITTER_COLOR: [number, number, number] = [2, 1, 0.5];
// Its bit pattern is a NaN as f32: identity must remain an integer on every path.
const MIRROR_ID = 0xffc00001;

function quad(): MeshData {
  const points = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  points.forEach((point, index) => {
    const offset = index * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, point[axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setFloat32(offset + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function instance(entityId: number, model: Float32Array, roughness: number): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 1, roughness, historyVersion: 0 };
}

/** Mirror is camera visible; the unlit plane is behind the camera and visible only to reflection rays. */
export async function runReflectionsFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:32px;height:32px';
  document.body.appendChild(canvas);
  const mirrorModel = scaling(3, 3, 1);
  const mirror: SceneBatch = { name: 'reflection-mirror', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [instance(MIRROR_ID, mirrorModel, 0.045)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } } };
  const emitter = (x: number) => instance(2, multiply(translation(x, 0, 5), scaling(20, 20, 1)), 1);
  const light: SceneBatch = { name: 'offscreen-reflection-emitter', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [emitter(0)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.parameter(0) }, parameters: new Float32Array([...EMITTER_COLOR, 1]) } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [mirror, light], { antiAliasing: 'none',
      worldQueries: { maxRays: 4096 },
      reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4,
        maxDistance: 20, maxBytes: 16 * 1024 * 1024, skyRadiance: [0, 0, 0] },
      volumetricFog: { pixelSize: 4, slices: 32, maxViewDepth: 10,
        maxFroxels: 8 * 8 * 32, maxBytes: 1024 * 1024 } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async (probe = true) => {
      active.render(EYE, TARGET);
      const [surface, reflection] = await Promise.all([
        active.probeSurfaceHdr(PIXELS), probe ? active.probeReflections(ANCHORS) : Promise.resolve(null),
      ]);
      return { surface, reflection, diagnostics: active.diagnostics().reflections };
    };
    const baseline = await capture();
    active.setReflectionsEnabled(false);
    const disabled = await capture(false);
    active.setReflectionsEnabled(true);
    const restored = await capture();

    light.instances[0] = emitter(30);
    const movedAway = await capture();
    light.instances[0] = emitter(0);
    const movedBack = await capture();
    active.setMaterialParameters(light.name, new Float32Array([0.5, 1.5, 2, 1]));
    const recolored = await capture();
    light.instances = [];
    const deleted = await capture();
    light.instances = [emitter(0)];
    active.setMaterialParameters(light.name, new Float32Array([...EMITTER_COLOR, 1]));
    const recreated = await capture();

    mirror.instances[0] = instance(MIRROR_ID, mirrorModel, 0.45);
    const rough = await capture();
    mirror.instances[0] = instance(MIRROR_ID, mirrorModel, 0.045);
    const smoothAgain = await capture();
    active.setVolumetricFogMedia({ global: { coefficients: {
      extinction: 0.2, albedo: [0, 0, 0], emission: [0, 0, 0],
    } } });
    const absorbed = await capture();
    active.setReflectionsEnabled(false);
    const absorbedDisabled = await capture(false);
    const errors = [...active.errors], frames = active.stats.frames;
    active.dispose();
    return { adapter: active.info.adapterInfo, eye: EYE, pixels: PIXELS, anchorIndices: ANCHORS, emitterColor: EMITTER_COLOR,
      baseline, disabled, restored, movedAway, movedBack, recolored, deleted,
      recreated, rough, smoothAgain, absorbed, absorbedDisabled,
      errors, frames, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Small combined GI/final-gather/fog binding case with one deforming offscreen emitter. */
export async function runReflectionsCombinedFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:32px;height:32px';
  document.body.appendChild(canvas);
  const mirrorModel = scaling(3, 3, 1);
  const mirror: SceneBatch = { name: 'combined-reflection-mirror', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [instance(11, mirrorModel, 0.045)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      baseColor: m.constant([1, 1, 1, 1]) } } };
  const mesh = quad(), positions = new Float32Array(mesh.vertexCount * 3);
  mesh.colors = new Float32Array(mesh.vertexCount * 4);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    positions[vertex * 3] = 3;
    new DataView(mesh.vertices).setUint32(vertex * 80 + 32, 0, true);
    mesh.colors.set([1, 0.5, 0.25, 1], vertex * 4);
  }
  const emitterModel = multiply(translation(0, 0, 5), scaling(20, 20, 1));
  const emitter = (morph: number, bone: Float32Array, phase: number): InstanceState => ({
    entityId: 12, model: emitterModel, normalMatrix: normalMatrix(emitterModel),
    bones: [bone], deformScalars: [morph, phase], color: [1, 1, 1, 1],
    metalness: 0, roughness: 1, historyVersion: 0,
  });
  const light: SceneBatch = { name: 'combined-offscreen-emitter', mesh, capacity: 1,
    inputMode: 'committed', instances: [emitter(0, identity(), 0)],
    deformation: { jointCount: 1, nodes: [
      { kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' },
      { kind: 'wave', amplitude: 3, frequency: 0, phase: 1 },
    ] },
    morphTargets: [{ positions, normals: new Float32Array(mesh.vertexCount * 3) }],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      alphaMode: 'mask', alphaCutoff: 0.5, baseColor: m.multiply(m.parameter(0), m.vertexColor) },
      parameters: new Float32Array([...EMITTER_COLOR, 0]) } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [mirror, light], { antiAliasing: 'none',
      worldQueries: { maxRays: 4096 },
      diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance: 10, skyRadiance: [0, 0, 0],
        finalGather: { pixelSpacing: 4, maxAnchors: 64, raysPerAnchor: 16, maxHistory: 8 } },
      reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4,
        maxDistance: 20, maxBytes: 16 * 1024 * 1024, skyRadiance: [0, 0, 0] },
      volumetricFog: { pixelSize: 4, slices: 32, maxViewDepth: 10,
        maxFroxels: 8 * 8 * 32, maxBytes: 1024 * 1024 },
    });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async (label: string) => {
      active.render(EYE, TARGET);
      const [on, probe] = await Promise.all([
        active.probeSurfaceHdr([PIXELS[0]]), active.probeReflections([ANCHORS[0]]),
      ]);
      const diagnostics = active.diagnostics();
      active.setReflectionsEnabled(false);
      active.render(EYE, TARGET);
      const off = await active.probeSurfaceHdr([PIXELS[0]]);
      active.setReflectionsEnabled(true);
      return { label, on: on[0], off: off[0], anchor: probe.anchors[0],
        reflection: diagnostics.reflections, gi: diagnostics.diffuseGi };
    };
    const samples = [];
    samples.push(await capture('transparent'));
    active.setMaterialParameters(light.name, new Float32Array([...EMITTER_COLOR, 1]));
    samples.push(await capture('opaque'));
    light.instances[0] = emitter(1, identity(), 0);
    samples.push(await capture('morph-away'));
    light.instances[0] = emitter(0, identity(), 0);
    samples.push(await capture('morph-restored'));
    light.instances[0] = emitter(0, translation(3, 0, 0), 0);
    samples.push(await capture('skin-away'));
    light.instances[0] = emitter(0, identity(), 0);
    samples.push(await capture('skin-restored'));
    light.instances[0] = emitter(0, identity(), Math.PI / 2);
    samples.push(await capture('wave-away'));
    light.instances[0] = emitter(0, identity(), 0);
    samples.push(await capture('wave-restored'));
    active.setMaterialParameters(light.name, new Float32Array([...EMITTER_COLOR, 0]));
    samples.push(await capture('transparent-again'));
    active.setMaterialParameters(light.name, new Float32Array([...EMITTER_COLOR, 1]));
    samples.push(await capture('opaque-again'));
    const errors = [...active.errors];
    active.dispose();
    return { adapter: active.info.adapterInfo, eye: EYE, emitterColor: [2, 0.5, 0.125] as const, samples, errors,
      ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
