import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, rotationX, scaling, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
function state(id: number, width: number, height: number, z: number): InstanceState {
  const model = multiply(translation(0, 0, z), multiply(rotationX(Math.PI / 2), scaling(width, 1, height)));
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}
function batch(name: string, instance: InstanceState): SceneBatch {
  return { name, mesh: createPlane(), capacity: 1, inputMode: 'committed', instances: [instance],
    queryIdentity: { assetId: name, primitiveId: 0, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

/** A one-pixel-wide foreground strip falls between all primary grid columns.
 * Its outward hemisphere sees a constant sky, giving independent E = pi*L. */
export async function runGiSupplementalFixture() {
  const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:128px;height:128px'; document.body.appendChild(canvas);
  const sky: Vec3 = [0.1, 0.2, 0.3], pixels = [{ x: 64, y: 64 }, { x: 44, y: 64 }];
  const eye: Vec3 = [0, 0, 3], target: Vec3 = [0, 0, 0];
  const runs = [];
  try {
    for (const repair of [false, true]) {
      const wall = batch('supplemental-wall', state(1, 2, 2, 0));
      const strip = batch('supplemental-strip', state(2, 0.035, 1.6, 0.02));
      const renderer = await Renderer.create(canvas, [wall, strip], {
        worldQueries: { maxRays: 2048 },
        diffuseGi: { origin: [-2, -2, -2], spacing: [4, 4, 4], counts: [2, 2, 2], skyRadiance: sky,
          maxDistance: 12, multiBounce: false, finalGather: { receiver: 'raster', pixelSpacing: 8,
            maxAnchors: 512, maxSupplementalAnchors: repair ? 256 : 0, maxSupplementalWorkUnits: repair ? 1_048_576 : 0,
            raysPerAnchor: 2, maxHistory: 16, pathSegments: 2 } },
      });
      try {
        renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
        const capture = async () => {
          const reconstruction = await renderer.probeDiffuseGiReconstruction(pixels);
          const diagnostic = renderer.diagnostics().diffuseGi!.finalGather!;
          const indices = Array.from({ length: diagnostic.supplementalAnchorsPerUpdate }, (_, i) => diagnostic.primaryAnchorsPerUpdate + i);
          const anchors = indices.length ? await renderer.probeDiffuseGiFinalGather(indices) : null;
          return { reconstruction, anchors, diagnostic, hdr: await renderer.probeHdr(pixels) };
        };
        for (let i = 0; i < 16; i++) renderer.render(eye, target);
        const stationary = await capture();
        const movedEye: Vec3 = [0.002, 0, 3], movedTarget: Vec3 = [0.002, 0, 0];
        renderer.render(movedEye, movedTarget);
        const moving = await capture();
        strip.instances.length = 0;
        renderer.render(movedEye, movedTarget);
        const removed = await capture();
        strip.instances.push({ ...state(2, 0.035, 1.6, 0.02), entityGeneration: 2 });
        renderer.render(movedEye, movedTarget);
        const replaced = await capture();
        const errors = [...renderer.errors]; renderer.dispose();
        runs.push({ repair, stationary, moving, removed, replaced, errors, afterDispose: renderer.stats.ownedGpuBytes });
      } finally { renderer.dispose(); }
    }
    return { sky, pixels, runs };
  } finally { canvas.remove(); }
}
