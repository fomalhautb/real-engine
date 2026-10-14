import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

function triangle(): MeshData {
  const vertices = new ArrayBuffer(3 * 80);
  const view = new DataView(vertices);
  const positions = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]];
  for (let i = 0; i < 3; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, positions[i][axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setUint32(offset + 32, 0, true);
    view.setFloat32(offset + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2]), vertexCount: 3,
    bounds: { min: [-0.5, -0.5, 0], max: [0.5, 0.5, 0] } };
}

function scene(): SceneBatch[] {
  const state: InstanceState = { entityId: 1101, model: identity(), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
  return [{ name: 'device-loss-triangle', mesh: triangle(), capacity: 1, inputMode: 'committed',
    instances: [state], deformation: { jointCount: 0, nodes: [] }, morphTargets: [] }];
}

function canvas(): HTMLCanvasElement {
  const result = document.createElement('canvas');
  result.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px;';
  document.body.appendChild(result);
  return result;
}

function deviceOf(renderer: Renderer): GPUDevice {
  // Fault injection is test-local; the renderer does not expose a public device-loss hook.
  return (renderer as unknown as { device: GPUDevice }).device;
}

async function rejection(action: () => Promise<unknown>): Promise<string> {
  try { await action(); return ''; } catch (error) { return String(error); }
}

export async function runDeviceLossFixture() {
  const eye: [number, number, number] = [0, 0, 4];
  const target: [number, number, number] = [0, 0, 0];

  const normalCanvas = canvas();
  let normal: Renderer | undefined;
  let normalResult;
  try {
    normal = await Renderer.create(normalCanvas, scene());
    normal.render(eye, target);
    const beforeDispose = normal.stats.frames;
    const normalDevice = deviceOf(normal);
    normal.dispose();
    normal.dispose();
    const lost = await normalDevice.lost;
    await Promise.resolve();
    normalResult = { beforeDispose, lossReason: lost.reason, errors: [...normal.errors],
      ownedGpuBytesAfterDispose: normal.stats.ownedGpuBytes };
  } finally { normal?.dispose(); normalCanvas.remove(); }

  const lossCanvas = canvas();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(lossCanvas, scene(), { worldQueries: {
      maxRays: 2, maxTrianglesPerMesh: 1, maxInstancesPerBatch: 1, maxNodeVisits: 16,
    } });
    renderer.render(eye, target);
    const framesBeforeLoss = renderer.stats.frames;
    const bytesBeforeLoss = renderer.stats.ownedGpuBytes;
    const device = deviceOf(renderer);
    device.destroy();
    const lost = await device.lost;
    await Promise.resolve();
    const errorsAfterLoss = [...renderer.errors];
    renderer.render(eye, target);
    const framesAfterAttempt = renderer.stats.frames;
    const queryRejection = await rejection(() => renderer!.traceRays([
      { origin: [0, 0, 2], direction: [0, 0, -1] },
    ]));
    renderer.dispose();
    renderer.dispose();
    return { normal: normalResult, loss: { framesBeforeLoss, framesAfterAttempt, bytesBeforeLoss,
      lossReason: lost.reason, errorsAfterLoss, queryRejection,
      ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes } };
  } finally { renderer?.dispose(); lossCanvas.remove(); }
}
