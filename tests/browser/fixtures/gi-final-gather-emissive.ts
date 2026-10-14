import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const eye: Vec3 = [0, 3, 2];
const target: Vec3 = [0, 0, 0];
const floorBase: Vec3 = [0.6, 0.4, 0.2];
const emission: Vec3 = [8, 4, 2];
const size: [number, number] = [0.12, 0.08];
const emitterName = 'gather-area-emitter';

function state(id: number, height: number, width: number, depth: number, version = 0): InstanceState {
  const model = multiply(translation(0, height, 0), scaling(width, 1, depth));
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0,
    roughness: 1, historyVersion: version };
}

function batch(name: string, id: number, instances: InstanceState[], material: MaterialBinding): SceneBatch {
  return { name, mesh: createPlane(), capacity: 1, inputMode: 'committed', instances,
    queryIdentity: { assetId: 'final-gather-emissive/' + name, primitiveId: id, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material };
}

function emitterMaterial(shading: 'unlit' | 'pbr', masked = false): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading, doubleSided: true,
    alphaMode: masked ? 'mask' : 'opaque', alphaCutoff: 0.5,
    baseColor: m.constant(shading === 'pbr' ? [0.9, 0.8, 0.7, 1] : [0, 0, 0, masked ? 0 : 1]),
    metallic: 0, roughness: 1, emissive: m.constant([...emission, 1]) } };
}

export async function runFinalGatherEmissiveFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  const floor = batch('gather-receiver', 1, [state(1, 0, 100, 100)],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'pbr', doubleSided: true,
      baseColor: m.constant([...floorBase, 1]), metallic: 0, roughness: 1 } });
  const emitter = batch(emitterName, 2, [state(2, 1, ...size)], emitterMaterial('unlit'));
  emitter.emissiveSampling = 'triangle-area';
  const blocker = batch('gather-black-blocker', 3, [],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.constant([0, 0, 0, 1]), emissive: m.constant([0, 0, 0, 1]) } });
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [floor, emitter, blocker], {
      worldQueries: { maxRays: 8192 },
      diffuseGi: { origin: [-0.5, 0.25, -0.5], spacing: [1, 0.5, 1], counts: [2, 2, 2],
        raysPerProbe: 64, maxDistance: 12, skyRadiance: [0, 0, 0], multiBounce: false,
        emissiveSamplesPerProbe: 16,
        finalGather: { pixelSpacing: 4, maxAnchors: 4096, raysPerAnchor: 2,
          maxHistory: 64, normalBias: 0.0001 } },
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const capture = async () => {
      for (let frame = 0; frame < 256; frame++) renderer!.render(eye, target);
      return (await renderer!.probeHdr([{ x: 128, y: 128 }]))[0];
    };
    const initial = await capture();
    await renderer.setMaterial(emitterName, emitterMaterial('pbr'));
    const reflective = await capture();
    await renderer.setMaterial(emitterName, emitterMaterial('unlit', true));
    const masked = await capture();
    await renderer.setMaterial(emitterName, emitterMaterial('unlit'));
    blocker.instances = [state(3, 0.5, 0.2, 0.2, 1)];
    const blocked = await capture();
    blocker.instances = [];
    const restored = await capture();
    const diagnostics = renderer.diagnostics().diffuseGi?.finalGather;
    const errors = [...renderer.errors];
    renderer.dispose();
    return { eye, target, floorBase, emission, size, initial, reflective, masked, blocked,
      restored, diagnostics, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
