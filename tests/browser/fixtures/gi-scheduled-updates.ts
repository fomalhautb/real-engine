import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createCube } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { DiffuseGiUpdateOptions } from '../../../src/gi-probe-schedule';
import type { DiffuseGiProbeResult } from '../../../src/gpu-diffuse-gi';

type Vec3 = [number, number, number];
const eye: Vec3 = [0, 1.2, 3.2];
const target: Vec3 = [0, 0.6, 0];
// 4 x 2 x 4 probes in a 4 m room with an open roof and front.
const grid = { origin: [-1.5, 0.45, -1.5] as Vec3, spacing: [1, 1, 1] as Vec3, counts: [4, 2, 4] as Vec3,
  raysPerProbe: 64, maxDistance: 10, normalBias: 0.03, skyRadiance: [0.05, 0.06, 0.08] as Vec3,
  emissiveSamplesPerProbe: 8 };
export const SCHEDULED_GI_POLICY: DiffuseGiUpdateOptions = { mode: 'scheduled', maxProbesPerUpdate: 8,
  maxStalenessUpdates: 8, convergenceUpdates: 2 };

function box(entityId: number, center: Vec3, size: Vec3, color: [number, number, number, number] = [1, 1, 1, 1]): InstanceState {
  const model = multiply(translation(...center), scaling(...size));
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [], color,
    metalness: 0, roughness: 1, historyVersion: 0 };
}
function batch(name: string, instances: InstanceState[], material: SceneBatch['material']): SceneBatch {
  return { name, mesh: createCube(), capacity: Math.max(1, instances.length), instances, inputMode: 'committed',
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material,
    queryIdentity: { assetId: 'gi-scheduled/' + name, primitiveId: 0, topologyVersion: 1 } };
}
const pbr = (baseColor = m.instanceColor) => ({ definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
  baseColor, metallic: 0, roughness: 1 } });

function scene() {
  const room = batch('room', [box(1, [0, -0.05, 0], [4.4, 0.1, 4.4], [0.8, 0.8, 0.8, 1]),
    box(2, [-2.15, 1, 0], [0.1, 2, 4.4], [0.85, 0.1, 0.08, 1]),
    box(3, [0, 1, -2.15], [4.4, 2, 0.1], [0.8, 0.8, 0.8, 1]),
    box(4, [2.15, 1, 0], [0.1, 2, 4.4], [0.1, 0.2, 0.8, 1])], pbr());
  const block = batch('block', [box(50, [0.6, 0.35, 0.4], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9, 1])], pbr());
  const panel = batch('panel', [box(60, [-1, 1.2, -2.05], [0.8, 0.4, 0.05])], { definition: {
    ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.constant([0, 0, 0, 1]),
    emissive: m.parameter(0) }, parameters: new Float32Array([3, 2, 1, 1]) });
  panel.emissiveSampling = 'triangle-area';
  return { batches: [room, block, panel], block, panel };
}

type Snapshot = { frame: number; probe: DiffuseGiProbeResult; schedule: unknown; queriesPerUpdate: number; update: string };

/** Runs one scripted sequence; probe readback is an explicit diagnostic. */
export async function runScheduledGiFixture(input: { update?: DiffuseGiUpdateOptions; multiBounce: boolean;
  traceAges?: boolean }) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:160px;height:120px;';
  document.body.appendChild(canvas);
  const { batches, block, panel } = scene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches, {
      worldQueries: { maxRays: 4096, maxTrianglesPerMesh: 12, maxInstancesPerBatch: 4, maxTraceParameterSetsPerEncoder: 4 },
      diffuseGi: { ...grid, multiBounce: input.multiBounce, update: input.update },
    });
    const r = renderer;
    r.setSun({ direction: [0.3, 0.9, 0.35], color: [3, 2.9, 2.7] });
    const ownedBytes: number[] = [];
    const ages: { frame: number; maxAge: number; captured: number[] }[] = [];
    const render = async (frames = 1) => {
      for (let i = 0; i < frames; i++) {
        r.render(eye, target);
        ownedBytes.push(r.stats.ownedGpuBytes);
        if (input.traceAges) {
          const probe = await r.probeDiffuseGi();
          ages.push({ frame: probe.frame, maxAge: Math.max(...probe.probes.map(item => probe.frame - item.captureFrame)),
            captured: probe.probes.flatMap((item, index) => item.captureFrame === probe.frame ? [index] : []) });
        }
      }
    };
    const snapshot = async (): Promise<Snapshot> => {
      const gi = r.diagnostics().diffuseGi!;
      return { frame: r.stats.frames, probe: await r.probeDiffuseGi(), schedule: gi.schedule,
        queriesPerUpdate: gi.queriesPerUpdate, update: gi.update };
    };
    await render(1);
    const initial = await snapshot();
    await render(47);
    const settled = await snapshot();
    // Local transform: only the moved block's neighbourhood is marked dirty.
    block.instances = [box(50, [-0.6, 0.35, -0.4], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9, 1])];
    await render(1);
    const moved = await snapshot();
    await render(47);
    const movedSettled = await snapshot();
    // Global light change: every probe dirty, nothing rejected.
    r.setSun({ direction: [-0.4, 0.85, 0.2], color: [2, 2.2, 2.6] });
    await render(1);
    const sunChanged = await snapshot();
    await render(47);
    const sunSettled = await snapshot();
    // Emissive material edit through the shared material parameters.
    r.setMaterialParameters(panel.name, new Float32Array([0.5, 2.5, 4, 1]));
    await render(1);
    const emissionChanged = await snapshot();
    await render(47);
    const emissionSettled = await snapshot();
    // External region (e.g. a local light) and an explicit hard reset.
    r.markDiffuseGiDirty({ min: [1.2, 0.2, 1.2], max: [1.6, 0.6, 1.6] });
    await render(1);
    const externalMarked = await snapshot();
    r.invalidateDiffuseGiHistory();
    await render(1);
    const hardReset = await snapshot();
    const errors = [...r.errors];
    r.dispose();
    return { initial, settled, moved, movedSettled, sunChanged, sunSettled, emissionChanged, emissionSettled,
      externalMarked, hardReset, ages, ownedBytes, errors, afterDispose: r.stats.ownedGpuBytes,
      adapter: r.info.adapterInfo };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Runtime policy switches reuse every allocation and force one full refresh. */
export async function runScheduledGiSwitchFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:160px;height:120px;';
  document.body.appendChild(canvas);
  const { batches } = scene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches, {
      worldQueries: { maxRays: 4096, maxTrianglesPerMesh: 12, maxInstancesPerBatch: 4, maxTraceParameterSetsPerEncoder: 4 },
      diffuseGi: { ...grid, multiBounce: true },
    });
    const r = renderer;
    r.setSun({ direction: [0.3, 0.9, 0.35], color: [3, 2.9, 2.7] });
    const frames: { update: string; capturedProbes: number; fullRefresh: boolean; reason: string | null; bytes: number;
      historyResetReason: string }[] = [];
    const step = () => {
      r.render(eye, target);
      const gi = r.diagnostics().diffuseGi!;
      frames.push({ update: gi.update, capturedProbes: gi.schedule.capturedProbesLastUpdate,
        fullRefresh: gi.schedule.lastPlan?.fullRefresh ?? false, reason: gi.schedule.lastPlan?.fullRefreshReason ?? null,
        bytes: r.stats.ownedGpuBytes, historyResetReason: gi.historyResetReason });
    };
    step(); step();
    r.setDiffuseGiUpdate(SCHEDULED_GI_POLICY);
    step(); step(); step();
    const invalid: string[] = [];
    for (const update of [{ mode: 'scheduled', maxProbesPerUpdate: 3, maxStalenessUpdates: 8 },
      { mode: 'scheduled', maxProbesPerUpdate: 33 }, { mode: 'full', maxProbesPerUpdate: 4 }] as DiffuseGiUpdateOptions[]) {
      try { r.setDiffuseGiUpdate(update); invalid.push('accepted'); } catch (error) { invalid.push(String(error)); }
    }
    step();
    r.setDiffuseGiUpdate(undefined);
    step(); step();
    const errors = [...r.errors];
    r.dispose();
    return { frames, invalid, errors, afterDispose: r.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Probe coefficient trace for convergence curves (explicit diagnostic readback). */
export async function runScheduledGiConvergence(input: { update?: DiffuseGiUpdateOptions; multiBounce: boolean;
  frames: number; every: number; sunChangeFrame?: number }) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:160px;height:120px;';
  document.body.appendChild(canvas);
  const { batches } = scene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches, {
      worldQueries: { maxRays: 4096, maxTrianglesPerMesh: 12, maxInstancesPerBatch: 4, maxTraceParameterSetsPerEncoder: 4 },
      diffuseGi: { ...grid, multiBounce: input.multiBounce, update: input.update },
    });
    const r = renderer;
    r.setSun({ direction: [0.3, 0.9, 0.35], color: [3, 2.9, 2.7] });
    const trace: { frame: number; coefficients: number[][] }[] = [];
    for (let frame = 1; frame <= input.frames; frame++) {
      if (frame === input.sunChangeFrame) r.setSun({ direction: [-0.4, 0.85, 0.2], color: [2, 2.2, 2.6] });
      r.render(eye, target);
      if (frame % input.every === 0 || frame === 1) {
        const probe = await r.probeDiffuseGi();
        trace.push({ frame: probe.frame, coefficients: probe.probes.map(item => item.coefficients.flat()) });
      }
    }
    const errors = [...r.errors];
    r.dispose();
    return { trace, errors };
  } finally { renderer?.dispose(); canvas.remove(); }
}
