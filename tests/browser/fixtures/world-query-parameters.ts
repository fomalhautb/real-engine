import type { SceneBatch } from '../../../src/contracts';
import { GpuWorldQuery } from '../../../src/gpu-world-query';
import { MaterialResources } from '../../../src/gpu-material';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity } from '../../../src/math';
import { decodeWorldHits, packWorldRays, WORLD_HIT_BYTES, type WorldQueryOptions } from '../../../src/world-query';

const FRAME = 17;
const REVISION = 23;
const ENTITY = 712;

function triangle(): SceneBatch {
  const vertices = new ArrayBuffer(3 * 80);
  const positions = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]];
  const view = new DataView(vertices);
  positions.forEach((position, i) => {
    const base = i * 80;
    position.forEach((value, axis) => view.setFloat32(base + axis * 4, value, true));
    view.setFloat32(base + 12, 1, true);
    view.setFloat32(base + 24, 1, true);
    view.setFloat32(base + 48, 1, true);
  });
  const pixels = new Uint8Array(4 * 4 * 4);
  for (let i = 0; i < 16; i++) pixels[i * 4 + 3] = 255;
  pixels[0] = 255;
  return {
    name: 'parameter-triangle',
    mesh: { vertices, vertexCount: 3, indices: new Uint32Array([0, 1, 2]),
      uvs: new Float32Array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125]),
      bounds: { min: [-1, -1, 0], max: [1, 1, 0] } },
    capacity: 1,
    instances: [{ entityId: ENTITY, entityGeneration: 2, historyVersion: 5,
      model: identity(), normalMatrix: identity(),
      bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0.2, roughness: 0.7 }],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'parameter-triangle', primitiveId: 0, topologyVersion: 9 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.texture(0) },
      textures: [{ width: 4, height: 4, colorSpace: 'linear', pixels }],
      samplers: [{ minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest',
        addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }] },
  };
}

function gpuScene(device: GPUDevice): { output: GPUBuffer; instances: GPUBuffer } {
  const output = device.createBuffer({ size: 3 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let instanceBuffer: GPUBuffer | undefined;
  try {
  instanceBuffer = device.createBuffer({ size: 224, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const data = new Float32Array(3 * 16);
  [[-1, -1, 0], [1, -1, 0], [0, 1, 0]].forEach((point, vertex) => {
    const at = vertex * 16;
    data.set([...point, 1], at);
    data.set([0, 0, 1, 0], at + 4);
    data.set([...point, 1], at + 8);
  });
  device.queue.writeBuffer(output, 0, data);
  const state = new Float32Array(56);
  for (const offset of [0, 16, 32]) state.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], offset);
  state.set([1, 1, 1, 1, 0.2, 0.7, 0, 0], 48);
  device.queue.writeBuffer(instanceBuffer, 0, state);
  return { output, instances: instanceBuffer };
  } catch (error) { output.destroy(); instanceBuffer?.destroy(); throw error; }
}

async function rejection(action: () => unknown): Promise<string> {
  try { await action(); return ''; } catch (error) { return String(error); }
}

async function setup(device: GPUDevice, options: WorldQueryOptions) {
  let ownedBytes = 0;
  const account = (delta: number) => { ownedBytes += delta; };
  const materials = new MaterialResources(device, account);
  let query: GpuWorldQuery | undefined;
  let buffers: ReturnType<typeof gpuScene> | undefined;
  try {
    const scene = triangle();
    const material = materials.create(scene.material);
    query = await GpuWorldQuery.create(device, materials.layout, account, {
      maxRays: 4, maxTrianglesPerMesh: 1, maxInstancesPerBatch: 1, ...options,
    });
    buffers = gpuScene(device);
    account(buffers.output.size + buffers.instances.size);
    const pipeline = await query.createPipeline(material.program);
    query.addBatch({ scene, ...buffers, getMaterial: () => material, getPipeline: () => pipeline });
    query.updateSnapshot(FRAME, REVISION);
    const readyQuery = query, readyBuffers = buffers;
    return { query: readyQuery, scene, getOwnedBytes: () => ownedBytes,
      dispose: () => { readyQuery.dispose(); materials.dispose();
        readyBuffers.output.destroy(); readyBuffers.instances.destroy();
        account(-readyBuffers.output.size - readyBuffers.instances.size); } };
  } catch (error) {
    query?.dispose(); materials.dispose();
    if (buffers) { buffers.output.destroy(); buffers.instances.destroy();
      account(-buffers.output.size - buffers.instances.size); }
    throw error;
  }
}

/** Each ray source is encoded as a GPU copy so later traces cannot alias queue.writeBuffer uploads. */
function encodeRays(device: GPUDevice, encoder: GPUCommandEncoder, query: GpuWorldQuery, count: number): GPUBuffer {
  const ray = { origin: [0, 0, 5] as [number, number, number], direction: [0, 0, -1] as [number, number, number] };
  const packed = packWorldRays(Array.from({ length: count }, () => ray), 4);
  const staging = device.createBuffer({ size: Math.max(4, packed.byteLength),
    usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
  new Uint8Array(staging.getMappedRange()).set(new Uint8Array(packed));
  staging.unmap();
  if (count) encoder.copyBufferToBuffer(staging, 0, query.rayBuffer, 0, packed.byteLength);
  return staging;
}

export async function runWorldQueryParametersFixture() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const validation: string[] = [];
  device.addEventListener('uncapturederror', event => validation.push(event.error.message));
  let first: Awaited<ReturnType<typeof setup>> | undefined;
  const staging: GPUBuffer[] = [];
  let readback: GPUBuffer | undefined;
  let budgetReadback: GPUBuffer | undefined;
  let second: Awaited<ReturnType<typeof setup>> | undefined;
  const cleanup = { firstOwnedBytesAfterDispose: -1, secondOwnedBytesAfterDispose: -1 };
  try {
    const readyFirst = await setup(device, { maxTraceParameterSetsPerEncoder: 3 });
    first = readyFirst;
    const uniformAlignment = device.limits.minUniformBufferOffsetAlignment;
    const strides = { trace: readyFirst.query.diagnostics().traceUniformStride,
      reset: readyFirst.query.diagnostics().resetUniformStride, uniformAlignment };
    const encoder = device.createCommandEncoder();
    readyFirst.query.encodeRefit(encoder);
    const plans = [{ count: 3, lod: 0 }, { count: 1, lod: 1 }, { count: 2, lod: 2 }, { count: 3, lod: 0 }];
    readback = device.createBuffer({ size: 4 * 4 * WORLD_HIT_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const uploads: number[] = [];
    plans.forEach(({ count, lod }, index) => {
      staging.push(encodeRays(device, encoder, readyFirst.query, count));
      uploads.push(readyFirst.query.encodeTrace(encoder, count, lod, REVISION));
      encoder.copyBufferToBuffer(readyFirst.query.hitBuffer, 0, readback!, index * 4 * WORLD_HIT_BYTES,
        count * WORLD_HIT_BYTES);
    });
    const zeroUpload = readyFirst.query.encodeTrace(encoder, 0, 3, REVISION);
    const invalidCount = await rejection(() => readyFirst.query.encodeTrace(encoder, 5, 0, REVISION));
    const invalidLod = await rejection(() => readyFirst.query.encodeTrace(encoder, 1, -1, REVISION));
    const snapshotChange = await rejection(() => readyFirst.query.encodeTrace(encoder, 1, 0, REVISION + 1));
    // This low-level fixture can temporarily restore an older frame; Renderer frames are monotonic.
    readyFirst.query.updateSnapshot(FRAME + 1, REVISION);
    const frameChange = await rejection(() => readyFirst.query.encodeTrace(encoder, 1, 0, REVISION));
    readyFirst.query.updateSnapshot(FRAME, REVISION);
    const previousInstances = readyFirst.scene.instances;
    readyFirst.scene.instances = [];
    readyFirst.query.updateSnapshot(FRAME, REVISION);
    const sourceCountChange = await rejection(() => readyFirst.query.encodeTrace(encoder, 1, 0, REVISION));
    readyFirst.scene.instances = previousInstances;
    readyFirst.query.updateSnapshot(FRAME, REVISION);
    const reuseAfterSnapshotReject = readyFirst.query.encodeTrace(encoder, 3, 0, REVISION);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const results = plans.map(({ count }, index) => decodeWorldHits(
      mapped.slice(index * 4 * WORLD_HIT_BYTES, index * 4 * WORLD_HIT_BYTES + count * WORLD_HIT_BYTES),
      ['parameter-triangle'], count, FRAME, REVISION));
    readback.unmap();

    second = await setup(device, { maxTraceParameterSetsPerEncoder: 2 });
    const budgetEncoder = device.createCommandEncoder();
    second.query.encodeRefit(budgetEncoder);
    staging.push(encodeRays(device, budgetEncoder, second.query, 1));
    budgetReadback = device.createBuffer({ size: 2 * WORLD_HIT_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const budgetUploads = [second.query.encodeTrace(budgetEncoder, 1, 0, REVISION)];
    budgetEncoder.copyBufferToBuffer(second.query.hitBuffer, 0, budgetReadback, 0, WORLD_HIT_BYTES);
    budgetUploads.push(second.query.encodeTrace(budgetEncoder, 1, 1, REVISION));
    const budgetError = await rejection(() => second!.query.encodeTrace(budgetEncoder, 1, 2, REVISION));
    const reuseAfterReject = second.query.encodeTrace(budgetEncoder, 1, 0, REVISION);
    budgetEncoder.copyBufferToBuffer(second.query.hitBuffer, 0, budgetReadback, WORLD_HIT_BYTES, WORLD_HIT_BYTES);
    device.queue.submit([budgetEncoder.finish()]);
    await budgetReadback.mapAsync(GPUMapMode.READ);
    const budgetMapped = budgetReadback.getMappedRange();
    const preserved = [0, 1].map(index => decodeWorldHits(
      budgetMapped.slice(index * WORLD_HIT_BYTES, (index + 1) * WORLD_HIT_BYTES),
      ['parameter-triangle'], 1, FRAME, REVISION));
    budgetReadback.unmap();
    const freshEncoder = device.createCommandEncoder();
    staging.push(encodeRays(device, freshEncoder, second.query, 1));
    const freshUpload = second.query.encodeTrace(freshEncoder, 1, 2, REVISION);
    device.queue.submit([freshEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return { results, uploads, zeroUpload, invalidCount, invalidLod, snapshotChange,
      frameChange, sourceCountChange, reuseAfterSnapshotReject,
      budgetUploads, budgetError, reuseAfterReject, preserved, freshUpload, strides, cleanup, validation };
  } finally {
    readback?.destroy();
    budgetReadback?.destroy();
    for (const buffer of staging) buffer.destroy();
    first?.dispose(); second?.dispose();
    cleanup.firstOwnedBytesAfterDispose = first?.getOwnedBytes() ?? 0;
    cleanup.secondOwnedBytesAfterDispose = second?.getOwnedBytes() ?? 0;
    device.destroy();
    if (cleanup.firstOwnedBytesAfterDispose !== 0 || cleanup.secondOwnedBytesAfterDispose !== 0) {
      throw new Error('WORLD_QUERY_FIXTURE_ACCOUNT_LEAK');
    }
  }
}
