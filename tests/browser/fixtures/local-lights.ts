import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import type { GpuLocalLights } from '../../../src/gpu-local-lights';
import type { LocalLightDescriptor, LocalLightHandle, LocalLightOptions, ResolvedLocalLight } from '../../../src/local-lights';
import { DEFAULT_MATERIAL_DEFINITION } from '../../../src/material';
import { identity, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
type Vec4 = [number, number, number, number];
export type LocalLightPixel = { x: number; y: number };

/** Plain-data steps chosen by the spec; the fixture only executes them in the production renderer. */
export type LocalLightSurfaceStep =
  | { kind: 'receiver'; metalness: number; roughness: number }
  | { kind: 'add'; key: string; light: LocalLightDescriptor }
  | { kind: 'update'; key: string; patch: Partial<LocalLightDescriptor> }
  | { kind: 'enable'; key: string; enabled: boolean }
  | { kind: 'remove'; key: string }
  | { kind: 'masked'; alpha: number | null }
  | { kind: 'skinned'; translation: Vec3 | null }
  | { kind: 'probe'; label: string; pixels: LocalLightPixel[] };

export interface LocalLightSurfacePlan {
  size: number;
  eye: Vec3;
  options: LocalLightOptions;
  receiverColor: Vec4;
  unlitColor: Vec4;
  unlitRegion: { min: [number, number]; max: [number, number] };
  occluderHalfSize: number;
  occluderHeight: number;
  steps: LocalLightSurfaceStep[];
}

/** Horizontal quad facing +Z; every vertex is bound to joint 0 with weight 1. */
function quad(min: [number, number], max: [number, number], z: number): MeshData {
  const points = [[min[0], min[1], z], [max[0], min[1], z], [max[0], max[1], z], [min[0], max[1], z]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  points.forEach((point, id) => {
    for (let axis = 0; axis < 3; axis++) view.setFloat32(id * 80 + axis * 4, point[axis], true);
    view.setFloat32(id * 80 + 12, 1, true);
    view.setFloat32(id * 80 + 24, 1, true);
    view.setFloat32(id * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), bounds: { min: [min[0], min[1], z], max: [max[0], max[1], z] } };
}

function state(entityId: number, color: Vec4, metalness = 0, roughness = 1, bones: Float32Array[] = []): InstanceState {
  return { entityId, model: identity(), normalMatrix: identity(), bones, deformScalars: [], color,
    metalness, roughness, historyVersion: 0 };
}

function batch(name: string, mesh: MeshData, instances: InstanceState[], jointCount = 0): SceneBatch {
  return { name, mesh, capacity: 1, inputMode: 'committed', instances,
    deformation: { jointCount, nodes: jointCount ? [{ kind: 'skin' }] : [] }, morphTargets: [] };
}

function scene(plan: LocalLightSurfacePlan) {
  const half = plan.occluderHalfSize, z = plan.occluderHeight;
  const receiver = batch('local-light-receiver', quad([-4, -4], [4, 4], 0), [state(1, plan.receiverColor)]);
  const unlit = batch('local-light-unlit', quad(plan.unlitRegion.min, plan.unlitRegion.max, 0.02), [state(2, plan.unlitColor)]);
  unlit.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' } };
  const masked = batch('local-light-masked-occluder', quad([-half, -half], [half, half], z), []);
  masked.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, alphaMode: 'mask' } };
  const skinned = batch('local-light-skinned-occluder', quad([-half, -half], [half, half], z), [], 1);
  return { batches: [receiver, unlit, masked, skinned], receiver, masked, skinned };
}

function canvas(size: number): HTMLCanvasElement {
  const element = document.createElement('canvas');
  element.style.cssText = `position:fixed;left:0;top:0;width:${size}px;height:${size}px;`;
  document.body.appendChild(element);
  return element;
}

export async function runLocalLightSurfaceFixture(plan: LocalLightSurfacePlan) {
  const element = canvas(plan.size);
  let reference: Renderer | undefined, renderer: Renderer | undefined;
  try {
    const target: Vec3 = [0, 0, 0];
    // Same scene without local lights: the ledger difference must be exactly the local light allocation.
    reference = await Renderer.create(element, scene(plan).batches);
    reference.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    reference.render(plan.eye, target);
    await reference.probeHdr([{ x: 0, y: 0 }]);
    const referenceBytes = reference.stats.ownedGpuBytes;
    reference.dispose(); reference = undefined;

    const { batches, receiver, masked, skinned } = scene(plan);
    renderer = await Renderer.create(element, batches, { localLights: plan.options });
    renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    renderer.render(plan.eye, target);
    await renderer.probeHdr([{ x: 0, y: 0 }]);
    if (renderer.stats.width !== plan.size || renderer.stats.height !== plan.size) {
      throw new Error(`LOCAL_LIGHT_FIXTURE_SIZE: ${renderer.stats.width}x${renderer.stats.height}`);
    }
    const ownedBytes = renderer.stats.ownedGpuBytes;
    const adapter = renderer.info.adapterInfo, timestamps = renderer.info.timing.startsWith('GPU timestamp-query');
    const initialDiagnostics = renderer.diagnostics().localLights;
    const handles = new Map<string, LocalLightHandle>();
    const probes: Record<string, { hdr: number[][]; lights: ResolvedLocalLight[]; diagnostics: unknown;
      receiver: { metalness: number; roughness: number }; gpuShadowMs: number | null }> = {};
    let material = { metalness: 0, roughness: 1 };
    for (const step of plan.steps) {
      if (step.kind === 'receiver') {
        material = { metalness: step.metalness, roughness: step.roughness };
        receiver.instances = [state(1, plan.receiverColor, step.metalness, step.roughness)];
      } else if (step.kind === 'add') {
        handles.set(step.key, renderer.addLocalLight(step.light));
      } else if (step.kind === 'update') {
        renderer.updateLocalLight(handles.get(step.key)!, step.patch);
      } else if (step.kind === 'enable') {
        renderer.setLocalLightEnabled(handles.get(step.key)!, step.enabled);
      } else if (step.kind === 'remove') {
        renderer.removeLocalLight(handles.get(step.key)!);
        handles.delete(step.key);
      } else if (step.kind === 'masked') {
        masked.instances = step.alpha === null ? [] : [state(3, [1, 1, 1, step.alpha])];
      } else if (step.kind === 'skinned') {
        skinned.instances = step.translation === null ? []
          : [state(4, [0.8, 0.8, 0.8, 1], 0, 1, [translation(...step.translation)])];
      } else {
        renderer.render(plan.eye, target);
        const hdr = await renderer.probeHdr(step.pixels);
        probes[step.label] = { hdr, lights: renderer.localLightList(), diagnostics: renderer.diagnostics().localLights,
          receiver: { ...material }, gpuShadowMs: renderer.stats.gpuMs?.localLightShadows ?? null };
      }
    }
    const errors = [...renderer.errors];
    const owner = renderer as unknown as { localLights?: GpuLocalLights };
    const lights = owner.localLights!;
    const beforeDispose = lights.diagnostics().ownedGpuBytes;
    renderer.dispose();
    const afterDispose = { localLights: lights.diagnostics().ownedGpuBytes, renderer: renderer.stats.ownedGpuBytes };
    let disposedApi = '';
    try { renderer.addLocalLight({ kind: 'point', position: [0, 0, 1], color: [1, 1, 1], intensity: 1, range: 1 }); }
    catch (error) { disposedApi = String(error); }
    return { referenceBytes, ownedBytes, beforeDispose, afterDispose, disposedApi, initialDiagnostics, probes, errors,
      frames: renderer.stats.frames, adapter, timestamps };
  } finally {
    reference?.dispose(); renderer?.dispose(); element.remove();
  }
}

/** Disabled renderers reject the API explicitly; option budgets fail before any GPU allocation. */
export async function runLocalLightContractFixture() {
  const element = canvas(17);
  const failures: Record<string, string> = {};
  const attempt = async (name: string, action: () => unknown) => {
    try { await action(); failures[name] = ''; } catch (error) { failures[name] = String(error); }
  };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, []);
    await attempt('disabled', () => renderer!.addLocalLight({ kind: 'point', position: [0, 0, 1], color: [1, 1, 1], intensity: 1, range: 1 }));
    renderer.dispose(); renderer = undefined;
    await attempt('atlasBudget', () => Renderer.create(element, [], { localLights: { maxShadowedLights: 8, shadowResolution: 2048 } }));
    renderer = await Renderer.create(element, [], { localLights: { maxLights: 2, maxShadowedLights: 1 } });
    renderer.addLocalLight({ kind: 'point', position: [0, 0, 1], color: [1, 1, 1], intensity: 1, range: 1, castShadows: true });
    await attempt('shadowBudget', () => renderer!.addLocalLight({ kind: 'spot', position: [0, 0, 1], direction: [0, 0, -1],
      color: [1, 1, 1], intensity: 1, range: 1, castShadows: true }));
    renderer.addLocalLight({ kind: 'point', position: [0, 0, 2], color: [1, 1, 1], intensity: 1, range: 1 });
    await attempt('lightBudget', () => renderer!.addLocalLight({ kind: 'point', position: [0, 0, 3], color: [1, 1, 1], intensity: 1, range: 1 }));
    await attempt('invalid', () => renderer!.addLocalLight({ kind: 'spot', position: [0, 0, 3], direction: [0, 0, 0],
      color: [1, 1, 1], intensity: 1, range: 1 }));
    renderer.render([0, 0, 5], [0, 0, 0]);
    await renderer.probeHdr([{ x: 0, y: 0 }]);
    const errors = [...renderer.errors];
    return { failures, errors };
  } finally { renderer?.dispose(); element.remove(); }
}
