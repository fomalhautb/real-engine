import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const eye: Vec3 = [0, 3, 2], target: Vec3 = [0, 0, 0];
const base = [0.6, 0.4, 0.2] as const;
const sky: Vec3 = [0.1, 0.2, 0.3];
function state(id: number, y: number, size: number, generation = 1): InstanceState {
  const model = multiply(translation(0, y, 0), scaling(size, 1, size));
  return { entityId: id, entityGeneration: generation, model, normalMatrix: normalMatrix(model), bones: [],
    deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}
function batch(name: string, id: number, instance: InstanceState): SceneBatch {
  return { name, mesh: createPlane(), capacity: 1, inputMode: 'committed', instances: [instance],
    queryIdentity: { assetId: 'final-gather/' + name, primitiveId: id, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

export async function runFinalGatherSkyFixture(receiverMode: 'world-query' | 'raster' = 'world-query') {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  const receiver = batch('gather-floor', 1, state(1, 0, 100));
  receiver.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION,
    baseColor: m.constant([...base, 1]), metallic: 0, roughness: 1 } };
  const blocker = batch('gather-offscreen-ceiling', 2, state(2, 5, 100));
  blocker.instances = [];
  blocker.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
    doubleSided: true, baseColor: m.constant([0, 0, 0, 1]) } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [receiver, blocker], {
      worldQueries: { maxRays: 8192 },
      diffuseGi: { origin: [-1, 0.3, -1], spacing: [2, 1, 2], counts: [2, 2, 2],
        maxDistance: 12, skyRadiance: sky, finalGather: { pixelSpacing: 4, maxAnchors: 4096, receiver: receiverMode } },
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const point = { x: 128, y: 128 };
    const capture = async (frames: number) => {
      for (let i = 0; i < frames; i++) renderer!.render(eye, target);
      return { hdr: (await renderer!.probeHdr([point]))[0],
        gi: renderer!.diagnostics().diffuseGi, bytes: renderer!.stats.ownedGpuBytes };
    };
    const initial = await capture(2), stable = await capture(80);
    blocker.instances = [state(2, 5, 100)];
    const blocked = await capture(160);
    blocker.instances = [];
    const restoredImmediate = await capture(1);
    renderer.setDiffuseGiEnabled(false);
    const disabled = await capture(1);
    renderer.setDiffuseGiEnabled(true);
    const reenabled = await capture(2);
    canvas.style.width = '320px'; canvas.style.height = '192px';
    const resized = await capture(2);
    const errors = [...renderer.errors]; renderer.dispose();
    return { base, sky, initial, stable, blocked, restoredImmediate, disabled, reenabled,
      resized, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
