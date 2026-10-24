import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, rotationX, scaling, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';

type Vec3 = [number, number, number];
function plane(id: number, x: number, width: number, z: number): InstanceState {
  const model = multiply(translation(x, 0, z), multiply(rotationX(Math.PI / 2), scaling(width, 1, 6)));
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}
function batch(name: string, instances: InstanceState[]): SceneBatch {
  return { name, mesh: createPlane(), capacity: instances.length, instances, inputMode: 'committed',
    queryIdentity: { assetId: name, primitiveId: 0, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

/** Seven distinct thin surfaces in every tile compete for one update per tile.
 * Every visible strip has an unobstructed outward hemisphere: E = pi * sky. */
export async function runSupplementalHistoryFixture(options: { progressiveComparison?: boolean } = {}) {
  const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:128px;height:128px'; document.body.append(canvas);
  const sky: Vec3 = [0.1, 0.2, 0.3], eye: Vec3 = [0, 0, 3], target: Vec3 = [0, 0, 0];
  const worldPerPixel = 2 * 2.98 * Math.tan(Math.PI / 6.4) / 128;
  const pixels = [64, 65, 66, 67, 69, 70, 71].map(x => ({ x, y: 64 }));
  const runs = [];
  const configurations: { historyFrames: number; sampling: 'frame-hash' | 'progressive-halton' }[] =
    options.progressiveComparison
      ? [{ historyFrames: 8, sampling: 'frame-hash' }, { historyFrames: 8, sampling: 'progressive-halton' }]
      : [0, 8, 1].map(historyFrames => ({ historyFrames, sampling: 'frame-hash' }));
  try {
    for (const { historyFrames, sampling } of configurations) {
      const progressiveComparison = options.progressiveComparison === true;
      const wall = batch('retained-wall', [plane(1, 0, 6, 0)]);
      const strips = batch('retained-strips', Array.from({ length: 128 }, (_, x) => x).filter(x => x % 8 !== 4)
        .map(x => plane(x + 2, (x + 0.5 - 64) * worldPerPixel, worldPerPixel * 0.8, 0.02)));
      const renderer = await Renderer.create(canvas, [wall, strips], {
        worldQueries: { maxRays: progressiveComparison ? 4096 : 2048, maxInstancesPerBatch: 128 },
        diffuseGi: { origin: [-2, -2, -2], spacing: [4, 4, 4], counts: [2, 2, 2], skyRadiance: sky,
          maxDistance: 12, multiBounce: progressiveComparison, finalGather: { receiver: 'raster', pixelSpacing: 8,
            maxAnchors: 512, maxSupplementalAnchors: 256, maxSupplementalWorkUnits: 1_048_576,
            supplementalHistoryFrames: historyFrames, raysPerAnchor: progressiveComparison ? 2 : 1,
            pathSegments: progressiveComparison ? 2 : 1, maxHistory: progressiveComparison ? 2 : 16,
            ...(progressiveComparison ? { supplementalSampling: sampling,
              supplementalSamplingSeed: sampling === 'progressive-halton' ? 0xffff_ffff : 0 } : {}) } },
      });
      try {
        renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
        const capture = async (reconstruction = false) => {
          const diagnostic = renderer.diagnostics().diffuseGi!.finalGather!;
          const map = await renderer.probeDiffuseGiFinalGatherCoverageTiles([8 + 8 * 16]);
          const anchors = await renderer.probeDiffuseGiFinalGather(map.tiles[0].indices);
          const primary = (await renderer.probeDiffuseGiFinalGather([8 + 8 * 16])).anchors[0];
          return { frame: renderer.stats.frames, ownedGpuBytes: renderer.stats.ownedGpuBytes, diagnostic, map, anchors, primary,
            reconstruction: reconstruction ? await renderer.probeDiffuseGiReconstruction(pixels) : null };
        };
        const frames = [];
        for (let i = 0; i < 24; i++) { renderer.render(eye, target); frames.push(await capture(i >= 7)); }
        const moving = [];
        for (let i = 1; i <= 8; i++) {
          renderer.render([0.0002 * i, 0, 3], [0.0002 * i, 0, 0]); moving.push(await capture(true));
        }
        renderer.invalidateDiffuseGiHistory();
        renderer.render(eye, target); const reset = await capture(true);
        for (let i = 0; i < 8; i++) renderer.render(eye, target);
        const removedInstance = strips.instances.find(instance => instance.entityId === 66)!;
        strips.instances = strips.instances.filter(instance => instance !== removedInstance);
        renderer.render(eye, target); const removed = await capture(true);
        strips.instances.push({ ...removedInstance, entityGeneration: 2, historyVersion: 1 });
        renderer.render(eye, target); const recycled = await capture(true);
        for (let i = 0; i < 8; i++) renderer.render(eye, target);
        const restored = await capture(true);
        renderer.setSun({ direction: [0, 0, 1], color: [1, 1, 1] });
        renderer.render(eye, target); const lightChanged = await capture(true);
        for (let i = 0; i < 8; i++) renderer.render(eye, target);
        await renderer.setMaterial('retained-strips', { definition: { ...DEFAULT_MATERIAL_DEFINITION,
          baseColor: m.constant([0.3, 0.6, 0.9, 1]) } });
        renderer.render(eye, target); const materialChanged = await capture(true);
        const errors = [...renderer.errors], info = renderer.info; renderer.dispose();
        runs.push({ historyFrames, sampling, frames, moving, reset, removed, recycled, restored, lightChanged, materialChanged,
          info, errors, afterDispose: renderer.stats.ownedGpuBytes });
      } finally { renderer.dispose(); }
    }
    return { sky, pixels, runs };
  } finally { canvas.remove(); }
}
