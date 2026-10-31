import type { SceneBatch } from '../../../src/contracts';
import { createCube, createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation, type Mat4 } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { FogVolume } from '../../../src/volumetric-fog';

export interface FogBudgetConfig {
  readonly label: string; readonly pixelSize: number; readonly slices: number; readonly maxViewDepth: number;
  readonly volumes: number; readonly air: boolean; readonly rejection: 'global' | 'local';
  /** 'default': the renderer's own froxel/byte budget selects the grid; otherwise up to the 4,194,304-froxel hard cap. */
  readonly budget?: 'default' | 'unbounded';
}
type Vec3 = [number, number, number];
const EYE: Vec3 = [0, 1.7, 10];
const TARGET: Vec3 = [0, 1, 0];

function batch(name: string, mesh: SceneBatch['mesh'], models: Mat4[], color: [number, number, number, number]): SceneBatch {
  return { name, mesh, capacity: models.length, inputMode: 'committed',
    instances: models.map((model, index) => ({ entityId: 100 + index + (name.length << 8), model, normalMatrix: normalMatrix(model),
      bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 })),
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant(color) } },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] } as SceneBatch;
}

/** K local volumes that all contain the point (0, 1, 0): the overlap count at the cluster centre is K. */
function volumes(count: number): FogVolume[] {
  return Array.from({ length: count }, (_, i) => {
    const angle = i * 2.399963229728653, radius = 0.4 + 0.6 * ((i * 7) % 11) / 11;
    const center: Vec3 = [Math.cos(angle) * radius, 1 + 0.3 * Math.sin(i), Math.sin(angle) * radius];
    const coefficients = { extinction: 0.05 + 0.01 * (i % 5), albedo: [0.8, 0.75, 0.7] as Vec3, emission: [0, 0, 0] as Vec3, anisotropy: 0.2 };
    return i % 2
      ? { id: i + 1, shape: { kind: 'sphere' as const, center, radius: 1.6 }, coefficients }
      : { id: i + 1, shape: { kind: 'box' as const, min: center.map(v => v - 1.4) as Vec3, max: center.map(v => v + 1.4) as Vec3 }, coefficients };
  });
}

/** F05 sweep point: one renderer, warm-up, then timestamp samples deduplicated by their sample frame. */
async function measure(config: FogBudgetConfig, samples: number) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:1920px;height:1080px';
  document.body.appendChild(canvas);
  const floor = batch('budget-floor', createPlane(), [scaling(40, 1, 40)], [0.6, 0.6, 0.6, 1]);
  const blocks = batch('budget-blocks', createCube(), [-3, 0, 3].map(x => multiply(translation(x, 1, -2), scaling(1, 2, 1))), [0.5, 0.4, 0.3, 1]);
  const roof = batch('budget-roof', createCube(), [multiply(translation(0, 4, 0), scaling(8, 0.1, 8))], [0.3, 0.3, 0.3, 1]);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [floor, blocks, roof], { antiAliasing: 'none',
      ...(config.air ? { worldQueries: { maxRays: 1024 }, diffuseGi: { origin: [-6, 0.5, -6] as Vec3, spacing: [4, 2, 4] as Vec3,
        counts: [4, 2, 4] as Vec3, raysPerProbe: 32, maxDistance: 20, skyRadiance: [0.3, 0.4, 0.5] as Vec3 } } : {}),
      volumetricFog: { pixelSize: config.pixelSize, slices: config.slices, maxViewDepth: config.maxViewDepth,
        ...(config.budget === 'default' ? {} : { maxFroxels: 4_194_304, maxBytes: 1024 * 1024 * 1024 }),
        temporal: { maxHistoryFrames: 16, occluderRejection: config.rejection }, ...(config.air ? { airLighting: { outsideSkyVisibility: 1 } } : {}) } });
    const active = renderer;
    active.setSun({ direction: [0.3, 0.8, 0.5], color: [6, 5.5, 5] });
    active.setVolumetricFogMedia({ global: { coefficients: { extinction: 0.05, albedo: [0.8, 0.8, 0.8], emission: [0, 0, 0], anisotropy: 0.3 },
      baseHeight: 0, heightFalloff: 0.3 }, volumes: volumes(config.volumes) });
    const center = [{ x: 960, y: 540 }];
    for (let frame = 0; frame < 15; frame++) { active.render(EYE, TARGET); await active.probeHdr(center); }
    const fog: number[] = [];
    let last: number | null = null;
    // The renderer resolves timestamps every 30th frame; a readback every 10th frame bounds queued work.
    for (let frame = 0; frame < samples * 30 + 60 && fog.length < samples; frame++) {
      active.render(EYE, TARGET);
      if (frame % 10 === 9) await active.probeHdr(center);
      const { gpuMs, gpuSampleFrame } = active.stats;
      if (gpuMs && gpuSampleFrame !== last) { fog.push(gpuMs.volumetricFog); last = gpuSampleFrame; }
    }
    const diagnostics = active.diagnostics().volumetricFog!;
    const result = { ...config, grid: diagnostics.grid, ownedBytes: diagnostics.ownedGpuBytes, fog, errors: [...active.errors],
      width: active.stats.width, height: active.stats.height, adapter: active.info.adapterInfo };
    active.dispose();
    return result;
  } finally { renderer?.dispose(); canvas.remove(); }
}

export async function runFogBudgetFixture(configs: readonly FogBudgetConfig[], samples = 12) {
  const results = [];
  for (const config of configs) results.push(await measure(config, samples));
  return results;
}
