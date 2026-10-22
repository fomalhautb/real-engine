import type { SceneBatch } from '../../../src/contracts';
import { MaterialResources } from '../../../src/gpu-material';
import { GpuWorldQuery, type WorldQueryFacePolicy } from '../../../src/gpu-world-query';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity } from '../../../src/math';
import { decodeWorldHits, packWorldRays, WORLD_HIT_BYTES, type WorldRay } from '../../../src/world-query';

const frame = 17, revision = 23;

function quad(name: string, entityId: number, x: number, z: number, backwards = false,
  doubleSided = false, mask = false): SceneBatch {
  const vertices = new ArrayBuffer(4 * 80), data = new DataView(vertices);
  const positions = [[x - 1, -1, z], [x + 1, -1, z], [x + 1, 1, z], [x - 1, 1, z]];
  positions.forEach((p, vertex) => {
    p.forEach((v, axis) => data.setFloat32(vertex * 80 + axis * 4, v, true));
    data.setFloat32(vertex * 80 + 12, 1, true);
    data.setFloat32(vertex * 80 + 24, backwards ? -1 : 1, true);
    data.setFloat32(vertex * 80 + 48, 1, true);
  });
  const definition = { ...DEFAULT_MATERIAL_DEFINITION, doubleSided,
    ...(mask ? { alphaMode: 'mask' as const, baseColor: m.texture(0), alphaCutoff: 128 / 255 } : {}) };
  return { name, capacity: 1,
    mesh: { vertices, vertexCount: 4, indices: new Uint32Array(backwards ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]),
      uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), bounds: { min: [x - 1, -1, z], max: [x + 1, 1, z] } },
    instances: [{ entityId, entityGeneration: 2, historyVersion: 5, model: identity(), normalMatrix: identity(),
      bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1 }],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: name, primitiveId: 0, topologyVersion: 9 },
    material: { definition, ...(mask ? {
      textures: [{ width: 2, height: 2, colorSpace: 'linear' as const,
        pixels: new Uint8Array([255, 255, 255, 0, 255, 255, 255, 127, 255, 255, 255, 128, 255, 255, 255, 255]) }],
      samplers: [{ minFilter: 'nearest' as const, magFilter: 'nearest' as const, mipmapFilter: 'nearest' as const,
        addressModeU: 'clamp-to-edge' as const, addressModeV: 'clamp-to-edge' as const }],
    } : {}) },
  };
}

function rays() {
  const labels: string[] = [], values: WorldRay[] = [];
  const add = (label: string, x: number, y: number, front: boolean, tMax = 3.9999) => {
    labels.push(label);
    values.push({ origin: [x, y, front ? 2 : -2], direction: [0, 0, front ? -1 : 1], tMin: .0001, tMax });
  };
  for (const [name, x] of [['single', 0], ['reversed-winding', 4], ['double', 8]] as const) {
    add(`${name}-front`, x, .1, true); add(`${name}-back`, x, .1, false);
  }
  for (const [prefix, center] of [['mask', 12], ['mask-double', 20]] as const) {
    for (const [name, x, y] of [['hole', -.5, -.5], ['below', .5, -.5], ['equal', -.5, .5], ['solid', .5, .5]] as const) {
      add(`${prefix}-${name}-front`, center + x, y, true); add(`${prefix}-${name}-back`, center + x, y, false);
    }
  }
  add('layers-probe', 16, .1, false); add('layers-receiver', 16, .1, true);
  add('short-interval', 0, .1, true, 1);
  add('excluded-endpoint', 0, .1, true, 1.9999);
  labels.push('parallel'); values.push({ origin: [0, 0, .5], direction: [1, 0, 0], tMin: .0001, tMax: .5 });
  return { labels, values };
}

function rejection(action: () => unknown): string {
  try { action(); return ''; } catch (error) { return String(error); }
}

/** Direct production query fixture; it never changes the Renderer query API. */
export async function runWorldQueryFacePolicyFixture(sceneBvh = false) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let ownedBytes = 0;
  const account = (delta: number) => { ownedBytes += delta; };
  const buffers: GPUBuffer[] = [];
  let materials: MaterialResources | undefined, query: GpuWorldQuery | undefined;
  const allocate = (size: number, usage: GPUBufferUsageFlags, source?: ArrayBuffer) => {
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: !!source });
    buffers.push(buffer); account(size);
    if (source) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(source)); buffer.unmap(); }
    return buffer;
  };
  const cleanup = { ownedBytesAfterDispose: -1 };
  try {
    materials = new MaterialResources(device, account);
    query = await GpuWorldQuery.create(device, materials.layout, account, { maxRays: 32, maxBatches: 8,
      maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1, maxTraceParameterSetsPerEncoder: 2,
      sceneAcceleration: sceneBvh ? 'bvh' : 'linear' });
    const scenes = [quad('single', 100, 0, 0), quad('reversed-winding', 101, 4, 0, true),
      quad('double', 102, 8, 0, false, true), quad('mask', 103, 12, 0, false, false, true),
      quad('layer-near-probe', 104, 16, -.75), quad('layer-near-receiver', 105, 16, .25),
      quad('mask-double', 106, 20, 0, false, true, true)];
    const names: string[] = [];
    for (let index = 0; index < 8; index++) {
      const scene = scenes[index] ?? { ...quad(`empty-${index}`, 106 + index, 20 + index * 4, 0), instances: [] };
      const mesh = scene.mesh!, source = new DataView(mesh.vertices), world = new Float32Array(4 * 16);
      for (let vertex = 0; vertex < 4; vertex++) {
        const position = [0, 1, 2].map(axis => source.getFloat32(vertex * 80 + axis * 4, true));
        world.set([...position, 1], vertex * 16);
        world.set([0, 0, source.getFloat32(vertex * 80 + 24, true), 0], vertex * 16 + 4);
        world.set([...position, 1], vertex * 16 + 8);
      }
      const instanceData = new Float32Array(56);
      for (const offset of [0, 16, 32]) instanceData.set(identity(), offset);
      instanceData.set([1, 1, 1, 1, 0, 1, 0, 0], 48);
      const output = allocate(world.byteLength, GPUBufferUsage.STORAGE, world.buffer);
      const instances = allocate(instanceData.byteLength, GPUBufferUsage.STORAGE, instanceData.buffer);
      const material = materials.create(scene.material), pipeline = await query.createPipeline(material.program);
      query.addBatch({ scene, output, instances, getMaterial: () => material, getPipeline: () => pipeline });
      names.push(scene.name);
    }
    query.updateSnapshot(frame, revision);
    const samples = rays(), count = samples.values.length, packed = packWorldRays(samples.values, 32);
    const staging = allocate(packed.byteLength, GPUBufferUsage.COPY_SRC, packed);
    const readback = allocate(4 * count * WORLD_HIT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const encoder = device.createCommandEncoder();
    query.encodeRefit(encoder); encoder.copyBufferToBuffer(staging, 0, query.rayBuffer, 0, packed.byteLength);
    const policies = ['material', 'reverse-material-support', 'material', 'reverse-material-support'] as const;
    const uploads = policies.map((policy, index) => {
      const uploaded = query!.encodeTrace(encoder, count, 0, revision, undefined, undefined, policy);
      encoder.copyBufferToBuffer(query!.hitBuffer, 0, readback, index * count * WORLD_HIT_BYTES, count * WORLD_HIT_BYTES);
      return uploaded;
    });
    const invalidPolicy = rejection(() => query!.encodeTrace(encoder, 0, 0, revision, undefined, undefined,
      'invalid' as WorldQueryFacePolicy));
    const slotBudget = rejection(() => query!.encodeTrace(encoder, count, 1, revision));
    const reuseAfterRejection = query.encodeTrace(encoder, count, 0, revision);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    let results;
    try {
      const mapped = readback.getMappedRange();
      results = policies.map((policy, index) => ({ policy, result: decodeWorldHits(
        mapped.slice(index * count * WORLD_HIT_BYTES, (index + 1) * count * WORLD_HIT_BYTES), names, count, frame, revision) }));
    } finally { readback.unmap(); }
    // Its own fresh encoder must still use authored faces after reverse traces.
    const publicDefault = await query.traceRays(samples.values, {}, revision);
    await device.queue.onSubmittedWorkDone();
    return { labels: samples.labels, results, publicDefault, uploads, invalidPolicy, slotBudget, reuseAfterRejection,
      diagnostics: query.diagnostics(), cleanup, errors };
  } finally {
    query?.dispose(); materials?.dispose();
    for (const buffer of buffers) { buffer.destroy(); account(-buffer.size); }
    cleanup.ownedBytesAfterDispose = ownedBytes;
    device.destroy();
    if (ownedBytes !== 0) throw new Error('WORLD_QUERY_FACE_POLICY_RESOURCE_LEAK');
  }
}
