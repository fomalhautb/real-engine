import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { identity } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

function triangle(): MeshData {
  const vertices = new ArrayBuffer(3 * 80), view = new DataView(vertices);
  const positions = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]];
  for (let vertex = 0; vertex < 3; vertex++) {
    const at = vertex * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(at + axis * 4, positions[vertex][axis], true);
    view.setFloat32(at + 12, 1, true);
    view.setFloat32(at + 24, 1, true);
    view.setFloat32(at + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2]), vertexCount: 3,
    bounds: { min: [-0.5, -0.5, 0], max: [0.5, 0.5, 0] } };
}

async function streamedBatch(provider: NonNullable<SceneBatch['geometryStream']>['provider']): Promise<SceneBatch> {
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'browser/worker-device-loss-triangle',
    mesh: triangle(), coverage: 'opaque' });
  const encoded = await encodeGeometryPages(hierarchy);
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const state: InstanceState = { entityId: 7001, model: identity(), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
  return { name: 'worker-device-loss-triangle', mesh: hierarchy.source, capacity: 1, inputMode: 'committed',
    instances: [state], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    geometryStream: { directory, provider: (page, signal) => provider(page, signal) } };
}

function canvas(): HTMLCanvasElement {
  const result = document.createElement('canvas');
  result.width = 64; result.height = 64;
  document.body.appendChild(result);
  return result;
}

function deviceOf(renderer: Renderer): GPUDevice {
  // Explicit test-only fault injection; Renderer intentionally keeps its device private.
  return (renderer as unknown as { device: GPUDevice }).device;
}

export async function runGeometryWorkerDeviceLossAfterInit() {
  const mesh = triangle();
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'browser/worker-device-loss-ready', mesh, coverage: 'opaque' });
  const encoded = await encodeGeometryPages(hierarchy);
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const state: InstanceState = { entityId: 7002, model: identity(), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
  const batch: SceneBatch = { name: 'worker-device-loss-ready', mesh: hierarchy.source, capacity: 1, inputMode: 'committed',
    instances: [state], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    geometryStream: { directory, provider: async (page, signal) => {
      signal.throwIfAborted(); return encoded.pages[page.id].bytes;
    } } };
  const surface = canvas();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(surface, [batch], { geometryDecoding: { workers: 1 } });
    const beforeLoss = renderer.diagnostics().geometryDecoding;
    const bytesBeforeLoss = renderer.stats.ownedGpuBytes;
    const device = deviceOf(renderer);
    device.destroy();
    const loss = await device.lost;
    await Promise.resolve();
    const afterLoss = renderer.diagnostics().geometryDecoding;
    renderer.dispose();
    return { beforeLoss, afterLoss, lossReason: loss.reason, bytesBeforeLoss,
      ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); surface.remove(); }
}

export async function runGeometryWorkerDeviceLossDuringInit() {
  const surface = canvas();
  const context = surface.getContext('webgpu');
  if (!context) throw new Error('WEBGPU_CONTEXT_UNAVAILABLE');
  let providerEntered = false, providerSignalAborted = false, device: GPUDevice | undefined;
  const provider: NonNullable<SceneBatch['geometryStream']>['provider'] = async (_page, signal) => {
    providerEntered = true;
    device = context.getConfiguration()?.device;
    if (!device) throw new Error('WEBGPU_CONFIGURATION_MISSING');
    return new Promise<Uint8Array<ArrayBuffer>>((_resolve, reject) => {
      const abort = () => { providerSignalAborted = signal.aborted; reject(new DOMException('Aborted', 'AbortError')); };
      signal.addEventListener('abort', abort, { once: true });
      if (signal.aborted) abort();
      device!.destroy();
    });
  };
  let renderer: Renderer | undefined, rejection = '';
  try {
    const batch = await streamedBatch(provider);
    try { renderer = await Renderer.create(surface, [batch], { geometryDecoding: { workers: 1 } }); }
    catch (error) { rejection = String(error); }
    const lossReason = device ? (await device.lost).reason : null;
    return { providerEntered, providerSignalAborted, rejection, lossReason,
      contextUnconfigured: context.getConfiguration() === null };
  } finally { renderer?.dispose(); surface.remove(); }
}
