import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { MaterialResources, type GpuMaterial } from '../../../src/gpu-material';
import { GpuWorldQuery, planWorldQueryTraceBudget, type WorldQueryFacePolicy, type WorldQueryWorkPartition } from '../../../src/gpu-world-query';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity, translation } from '../../../src/math';
import { decodeWorldHits, packWorldRays, WORLD_HIT_BYTES, type WorldQueryOptions, type WorldRay } from '../../../src/world-query';

/**
 * Direct production-query parity: identical world-space deformation outputs are
 * registered with several query configurations on one device. Every
 * configuration traces the same GPU-resident rays; raw 160-byte hit records are
 * compared word by word against the per-batch linear reference (work counters
 * excluded, since traversal order legitimately changes them).
 */
type Vec3 = [number, number, number];
type Kind = 'opaque' | 'double' | 'mask' | 'mask-double';
interface BatchSpec {
  name: string; mesh: MeshData; capacity: number; kind: Kind; inputMode: 'committed' | 'mutable'; morph?: boolean;
  slots: { offset: Vec3; entityId: number; generation: number }[];
}
interface Arm { name: string; options: WorldQueryOptions }
type TraceConfig = { label: string; policy: WorldQueryFacePolicy; maxWorkUnits?: number; partition?: WorldQueryWorkPartition };

const MAX_RAYS = 1536;
const ARMS: Arm[] = [
  { name: 'linear-reference', options: { staticTraversal: 'per-batch', sceneAcceleration: 'linear', instanceAcceleration: 'linear' } },
  { name: 'per-batch', options: { staticTraversal: 'per-batch' } },
  { name: 'per-batch-single-pass', options: { staticTraversal: 'per-batch', dispatchEncoding: 'single-pass' } },
  { name: 'shared', options: { staticTraversal: 'shared' } },
  { name: 'shared-single-pass', options: { staticTraversal: 'shared', dispatchEncoding: 'single-pass' } },
  { name: 'shared-linear-scene', options: { staticTraversal: 'shared', sceneAcceleration: 'linear', instanceAcceleration: 'linear' } },
];

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mesh(positions: Vec3[], normals: Vec3[], indices: number[], uvs: number[]): MeshData {
  const vertices = new ArrayBuffer(positions.length * 80), view = new DataView(vertices);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  positions.forEach((position, vertex) => {
    position.forEach((value, axis) => {
      view.setFloat32(vertex * 80 + axis * 4, value, true);
      min[axis] = Math.min(min[axis], value); max[axis] = Math.max(max[axis], value);
    });
    view.setFloat32(vertex * 80 + 12, 1, true);
    normals[vertex].forEach((value, axis) => view.setFloat32(vertex * 80 + 16 + axis * 4, value, true));
    view.setFloat32(vertex * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: positions.length, indices: new Uint32Array(indices), uvs: new Float32Array(uvs), bounds: { min, max } };
}
/** 2x2 quad at z = 0 centered on the origin; reversed winding faces -z. */
function quad(reversed = false): MeshData {
  return mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], Array(4).fill([0, 0, reversed ? -1 : 1]),
    reversed ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3], [0, 0, 1, 0, 1, 1, 0, 1]);
}
function soup(seed: number, triangles: number): MeshData {
  const next = random(seed), positions: Vec3[] = [], normals: Vec3[] = [], uvs: number[] = [];
  for (let i = 0; i < triangles; i++) {
    const center: Vec3 = [(next() - 0.5) * 2, (next() - 0.5) * 2, (next() - 0.5) * 2];
    const corners = [0, 1, 2].map(() => center.map(value => value + (next() - 0.5) * 1.2) as Vec3);
    const e1 = corners[1].map((value, axis) => value - corners[0][axis]), e2 = corners[2].map((value, axis) => value - corners[0][axis]);
    const n: Vec3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const length = Math.hypot(...n) || 1;
    for (const corner of corners) { positions.push(corner); normals.push(n.map(value => value / length) as Vec3); uvs.push(next(), next()); }
  }
  return mesh(positions, normals, positions.map((_, i) => i), uvs);
}

function specs(): BatchSpec[] {
  const row = (count: number, y: number, z: number, base: number, stride = 5): BatchSpec['slots'] =>
    Array.from({ length: count }, (_, slot) => ({ offset: [(slot - (count - 1) / 2) * stride, y, z], entityId: base + slot, generation: 1 }));
  const single = (x: number, entityId: number, z = 0): BatchSpec['slots'] => [{ offset: [x, 0, z], entityId, generation: 1 }];
  return [
    { name: 'soup-0', mesh: soup(1, 32), capacity: 8, kind: 'opaque', inputMode: 'committed', slots: row(6, 4, 0, 100) },
    { name: 'soup-1', mesh: soup(2, 24), capacity: 4, kind: 'double', inputMode: 'committed', slots: row(4, -4, 0.5, 200, 7) },
    { name: 'quad-single', mesh: quad(), capacity: 1, kind: 'opaque', inputMode: 'committed', slots: single(0, 300) },
    { name: 'quad-reversed', mesh: quad(true), capacity: 1, kind: 'opaque', inputMode: 'committed', slots: single(4, 301) },
    { name: 'quad-double', mesh: quad(), capacity: 1, kind: 'double', inputMode: 'committed', slots: single(8, 302) },
    { name: 'mask', mesh: quad(), capacity: 1, kind: 'mask', inputMode: 'committed', slots: single(12, 303) },
    { name: 'mask-double', mesh: quad(), capacity: 1, kind: 'mask-double', inputMode: 'committed', slots: single(20, 304) },
    // In-batch exact duplicates (slots 0, 1) and a cross-batch duplicate in the next batch.
    { name: 'dup-a', mesh: quad(), capacity: 3, kind: 'double', inputMode: 'committed', slots: [
      { offset: [16, 0, 0], entityId: 400, generation: 1 }, { offset: [16, 0, 0], entityId: 401, generation: 1 },
      { offset: [16, 0, -0.5], entityId: 402, generation: 1 }] },
    { name: 'dup-b', mesh: quad(), capacity: 1, kind: 'double', inputMode: 'committed', slots: single(16, 403) },
    { name: 'mutable', mesh: quad(), capacity: 1, kind: 'opaque', inputMode: 'mutable', slots: single(-4, 500) },
    { name: 'deformed', mesh: quad(), capacity: 1, kind: 'opaque', inputMode: 'committed', morph: true, slots: single(-8, 501) },
    { name: 'empty', mesh: quad(), capacity: 2, kind: 'opaque', inputMode: 'committed', slots: [] },
    { name: 'switch', mesh: quad(), capacity: 1, kind: 'opaque', inputMode: 'committed', slots: single(-12, 502) },
    { name: 'soup-2', mesh: soup(3, 48), capacity: 8, kind: 'opaque', inputMode: 'committed', slots: row(8, -8, -0.5, 600, 4.5) },
    // Behind the Masked quad, so its holes reveal a pooled hit.
    { name: 'behind-mask', mesh: quad(), capacity: 1, kind: 'opaque', inputMode: 'committed', slots: single(12, 700, -1) },
  ];
}

function binding(kind: Kind, index: number): MaterialBinding {
  const mask = kind === 'mask' || kind === 'mask-double';
  const definition = { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: kind === 'double' || kind === 'mask-double',
    ...(mask ? { alphaMode: 'mask' as const, baseColor: m.texture(0), alphaCutoff: 128 / 255 } : { baseColor: m.parameter(0) }) };
  return mask ? { definition,
    textures: [{ width: 2, height: 2, colorSpace: 'linear' as const,
      pixels: new Uint8Array([255, 255, 255, 0, 255, 255, 255, 127, 255, 255, 255, 128, 255, 255, 255, 255]) }],
    samplers: [{ minFilter: 'nearest' as const, magFilter: 'nearest' as const, mipmapFilter: 'nearest' as const,
      addressModeU: 'clamp-to-edge' as const, addressModeV: 'clamp-to-edge' as const }] }
    : { definition, parameters: new Float32Array([0.1 + (index % 7) * 0.12, 0.2 + (index % 5) * 0.15, 0.9 - (index % 3) * 0.2, 1]) };
}

function instance(slot: BatchSpec['slots'][number]): InstanceState {
  const model = translation(...slot.offset);
  return { entityId: slot.entityId, entityGeneration: slot.generation, historyVersion: slot.generation - 1,
    model, normalMatrix: identity(), bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0.25, roughness: 0.75 };
}

/** World-space deformation output for every active slot: position, normal, previous, tangent. */
function output(spec: BatchSpec): Float32Array<ArrayBuffer> {
  const vertexCount = spec.mesh.vertexCount, view = new DataView(spec.mesh.vertices);
  const data = new Float32Array(spec.capacity * vertexCount * 16);
  spec.slots.forEach((slot, index) => {
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const at = (index * vertexCount + vertex) * 16;
      const position = [0, 1, 2].map(axis => view.getFloat32(vertex * 80 + axis * 4, true) + slot.offset[axis]);
      const normal = [0, 1, 2].map(axis => view.getFloat32(vertex * 80 + 16 + axis * 4, true));
      data.set([...position, 1], at); data.set([...normal, 0], at + 4);
      data.set([position[0] - 0.125, position[1], position[2], 1], at + 8); data.set([1, 0, 0, 1], at + 12);
    }
  });
  return data;
}

function rays(): { labels: string[]; values: WorldRay[] } {
  const labels: string[] = [], values: WorldRay[] = [];
  const add = (label: string, ray: WorldRay) => { labels.push(label); values.push(ray); };
  const axis = (label: string, x: number, y: number, front: boolean, ignore?: WorldRay['ignore']) =>
    add(label, { origin: [x, y, front ? 3 : -3], direction: [0, 0, front ? -1 : 1], tMin: 0.0001, tMax: 50, ...(ignore ? { ignore } : {}) });
  for (const [name, x] of [['single', 0], ['reversed', 4], ['double', 8], ['switch', -12], ['mutable', -4], ['deformed', -8]] as const) {
    axis(`${name}-front`, x + 0.1, 0.1, true); axis(`${name}-back`, x + 0.1, 0.1, false);
    // Lower-left texel of the Masked switch material is a hole.
    axis(`${name}-hole-front`, x - 0.5, -0.5, true);
  }
  for (const [prefix, center] of [['mask', 12], ['mask-double', 20]] as const) {
    for (const [cell, x, y] of [['hole', -0.5, -0.5], ['below', 0.5, -0.5], ['equal', -0.5, 0.5], ['solid', 0.5, 0.5]] as const) {
      axis(`${prefix}-${cell}-front`, center + x, y, true); axis(`${prefix}-${cell}-back`, center + x, y, false);
    }
  }
  // (16.25, 0.25) is off the diagonal; (16.5, 0.5) is on it (both BLAS triangles tie too).
  for (const [x, y] of [[16.25, 0.25], [16.5, 0.5]]) {
    axis(`tie-front-${x}`, x, y, true); axis(`tie-back-${x}`, x, y, false);
    axis(`tie-ignore-first-${x}`, x, y, true, { entityId: 400, generation: 1 });
    axis(`tie-ignore-stale-${x}`, x, y, true, { entityId: 400, generation: 2 });
  }
  const next = random(77);
  while (values.length < MAX_RAYS - 32) {
    const origin: Vec3 = [(next() - 0.5) * 44, (next() - 0.5) * 22, (next() - 0.5) * 12];
    const target: Vec3 = [(next() - 0.5) * 40, (next() - 0.5) * 18, (next() - 0.5) * 3];
    const delta = target.map((value, i) => value - origin[i]) as Vec3;
    const index = values.length;
    add(`random-${index}`, { origin, direction: delta, tMin: index % 3 ? 0.0001 : 0, tMax: index % 4 ? 80 : Math.hypot(...delta) * 0.9,
      ...(index % 9 === 0 ? { ignore: { entityId: [100, 203, 602, 300, 401][index % 5], generation: index % 2 ? 1 : 2 } } : {}) });
  }
  while (values.length % 16) add(`pad-${values.length}`, { origin: [100, 100, 100], direction: [0, 0, 1], tMin: 0, tMax: 1 });
  return { labels, values };
}

type Words = Uint32Array<ArrayBuffer>;
const RECORD_WORDS = WORLD_HIT_BYTES / 4;
/** Words 36/37 are work units and triangle tests; 38 is the overflow flag. */
function compare(reference: Words, actual: Words, count: number) {
  const result = { rays: count, referenceHits: 0, equal: 0, mismatches: 0, overflowReference: 0, overflowActual: 0,
    overflowBoth: 0, examples: [] as { ray: number; word: number; reference: number; actual: number }[],
    words: {} as Record<number, { count: number; maxUlps: number }> };
  for (let ray = 0; ray < count; ray++) {
    const at = ray * RECORD_WORDS, refOverflow = reference[at + 38] !== 0, actualOverflow = actual[at + 38] !== 0;
    if (refOverflow || actualOverflow) {
      if (refOverflow && actualOverflow) result.overflowBoth++;
      else if (refOverflow) result.overflowReference++;
      else result.overflowActual++;
      continue;
    }
    if (reference[at] !== 0) result.referenceHits++;
    let word = -1;
    for (let w = 0; w < RECORD_WORDS; w++) {
      if (w >= 36 && w <= 38 || reference[at + w] === actual[at + w]) continue;
      if (word < 0) word = w;
      // Float words: distance in units in the last place (same-sign f32 bit patterns are ordered).
      const ulps = FLOAT_WORDS.has(w) ? Math.abs(ordered(reference[at + w]) - ordered(actual[at + w])) : Infinity;
      const entry = result.words[w] ??= { count: 0, maxUlps: 0 };
      entry.count++; entry.maxUlps = Math.max(entry.maxUlps, ulps);
    }
    if (word < 0) { result.equal++; continue; }
    result.mismatches++;
    if (result.examples.length < 8) result.examples.push({ ray, word, reference: reference[at + word], actual: actual[at + word] });
  }
  return result;
}
/** Record layout: identity u32x4, then f32 metric/position/normal/previous/baseColor/emissive, u32 versions, f32 uvInfo, u32 work. */
const FLOAT_WORDS = new Set([...Array.from({ length: 24 }, (_, i) => 4 + i), 32, 33, 34, 35]);
function ordered(bits: number): number {
  return bits & 0x8000_0000 ? -(bits & 0x7fff_ffff) : bits;
}
function work(records: Words, count: number, quota: number[]) {
  let units = 0, tests = 0, max = 0, overflows = 0, quotaViolations = 0;
  for (let ray = 0; ray < count; ray++) {
    const at = ray * RECORD_WORDS, value = records[at + 36];
    units += value; tests += records[at + 37]; max = Math.max(max, value);
    if (records[at + 38]) { overflows++; if (value !== quota[ray]) quotaViolations++; } else if (value > quota[ray]) quotaViolations++;
  }
  return { units, tests, max, overflows, quotaViolations };
}

export async function runWorldQuerySharedParityFixture() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let ownedBytes = 0;
  const account = (delta: number) => { ownedBytes += delta; };
  const buffers: GPUBuffer[] = [];
  const allocate = (size: number, usage: GPUBufferUsageFlags, source?: ArrayBuffer) => {
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: !!source });
    buffers.push(buffer); account(size);
    if (source) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(source)); buffer.unmap(); }
    return buffer;
  };
  const materials = new MaterialResources(device, account);
  const queries: { arm: Arm; query: GpuWorldQuery; pipelines: Map<string, GPUComputePipeline> }[] = [];
  const cleanup = { ownedBytesAfterDispose: -1 };
  try {
    const batchSpecs = specs();
    const scenes: SceneBatch[] = batchSpecs.map(spec => ({ name: spec.name, mesh: spec.mesh, capacity: spec.capacity,
      inputMode: spec.inputMode, instances: spec.slots.map(instance),
      deformation: { jointCount: 0, nodes: [] },
      morphTargets: spec.morph ? [{ positions: new Float32Array(spec.mesh.vertexCount * 3), normals: new Float32Array(spec.mesh.vertexCount * 3) }] : [],
      queryIdentity: { assetId: spec.name, primitiveId: 0, topologyVersion: 3 } }));
    const outputs = batchSpecs.map(spec => allocate(spec.capacity * spec.mesh.vertexCount * 64,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, output(spec).buffer));
    const instanceBuffers = batchSpecs.map(spec => {
      const data = new Float32Array(spec.capacity * 56);
      for (let slot = 0; slot < spec.capacity; slot++) {
        for (const offset of [0, 16, 32]) data.set(identity(), slot * 56 + offset);
        data.set([1, 1, 1, 1, 0.25, 0.75, 0, 0], slot * 56 + 48);
      }
      return allocate(data.byteLength, GPUBufferUsage.STORAGE, data.buffer);
    });
    const current: GpuMaterial[] = batchSpecs.map((spec, index) => materials.create(binding(spec.kind, index)));
    const dirty = batchSpecs.map(() => true);
    for (const arm of ARMS) {
      const query = await GpuWorldQuery.create(device, materials.layout, account, { maxRays: MAX_RAYS, maxBatches: 16,
        maxTrianglesPerMesh: 64, maxInstancesPerBatch: 8, maxTraceParameterSetsPerEncoder: 8, ...arm.options });
      const entry = { arm, query, pipelines: new Map<string, GPUComputePipeline>() };
      queries.push(entry);
      for (const material of current) if (!entry.pipelines.has(material.program.key)) {
        entry.pipelines.set(material.program.key, await query.createPipeline(material.program));
      }
      scenes.forEach((scene, index) => query.addBatch({ scene, output: outputs[index], instances: instanceBuffers[index],
        getMaterial: () => current[index], getPipeline: () => entry.pipelines.get(current[index].program.key)!,
        getDirty: () => dirty[index] }));
    }
    const switchMaterial = async (index: number, kind: Kind) => {
      current[index] = materials.create(binding(kind, index));
      for (const entry of queries) if (!entry.pipelines.has(current[index].program.key)) {
        entry.pipelines.set(current[index].program.key, await entry.query.createPipeline(current[index].program));
      }
    };
    const samples = rays(), count = samples.values.length;
    const packed = packWorldRays(samples.values, MAX_RAYS);
    const staging = allocate(packed.byteLength, GPUBufferUsage.COPY_SRC, packed);
    const configs: TraceConfig[] = [
      { label: 'material', policy: 'material' },
      { label: 'reverse-support', policy: 'reverse-material-support' },
      // Quotas below the linear reference's per-ray maximum: some rays overflow on every path.
      { label: 'tight', policy: 'material', maxWorkUnits: count * 96 },
      { label: 'partition', policy: 'material', partition: { raysPerBlock: 16, primaryRaysPerBlock: 12, supplementalMaxWorkUnits: (count / 16) * 4 * 64 } },
    ];
    const quotas = configs.map(config => {
      const plan = planWorldQueryTraceBudget(count, queries[0].query.options, config.maxWorkUnits, config.partition);
      return samples.values.map((_, ray) => plan.raysPerBlock && ray % plan.raysPerBlock >= plan.primaryRaysPerBlock
        ? plan.supplementalWorkPerRay : plan.primaryWorkPerRay);
    });
    const readback = allocate(configs.length * count * WORLD_HIT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const names = scenes.map(scene => scene.name);
    let frame = 0;
    const stages: Record<string, unknown>[] = [];
    const run = async (label: string, snapshot: boolean) => {
      if (snapshot) frame++;
      const records: Words[][] = [];
      const diagnostics: Record<string, unknown>[] = [];
      for (const { query } of queries) {
        if (snapshot) { query.validateSources(); query.updateSnapshot(frame, 0); }
        const encoder = device.createCommandEncoder();
        if (snapshot) query.encodeRefit(encoder);
        encoder.copyBufferToBuffer(staging, 0, query.rayBuffer, 0, packed.byteLength);
        configs.forEach((config, index) => {
          query.encodeTrace(encoder, count, 0, 0, config.maxWorkUnits, config.partition, config.policy);
          encoder.copyBufferToBuffer(query.hitBuffer, 0, readback, index * count * WORLD_HIT_BYTES, count * WORLD_HIT_BYTES);
        });
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        try {
          const mapped = readback.getMappedRange();
          records.push(configs.map((_, index) => new Uint32Array(mapped.slice(index * count * WORLD_HIT_BYTES, (index + 1) * count * WORLD_HIT_BYTES))));
        } finally { readback.unmap(); }
        const value = query.diagnostics();
        diagnostics.push({ sharedStatic: value.sharedStatic, sceneAcceleration: value.sceneAcceleration,
          instanceAcceleration: value.instanceAcceleration, staticTraversal: value.staticTraversal, dispatchEncoding: value.dispatchEncoding });
      }
      const reference = records[0];
      // Decoded public hits for the default configuration (throws on any overflow).
      const decoded = queries.map((_, arm) => decodeWorldHits(records[arm][0].buffer.slice(0), names, count, frame, 0));
      const labelled = Object.fromEntries(samples.labels.filter(name => !name.startsWith('random') && !name.startsWith('pad'))
        .map(name => [name, (() => { const hit = decoded[0].hits[samples.labels.indexOf(name)];
          return hit && { batch: hit.batch, entityId: hit.entityId, entityGeneration: hit.entityGeneration, distance: hit.distance,
            opacity: hit.opacity, triangleIndex: hit.triangleIndex }; })()]));
      stages.push({ label, frame,
        arms: queries.map(({ arm }, index) => ({ arm: arm.name, diagnostics: diagnostics[index],
          decodedEqual: JSON.stringify(decoded[index]) === JSON.stringify(decoded[0]),
          configs: configs.map((config, c) => ({ config: config.label, ...compare(reference[c], records[index][c], count),
            work: work(records[index][c], count, quotas[c]) })) })),
        labelled });
      dirty.fill(false);
    };
    const moveSlot = (index: number, slot: number, update: Partial<BatchSpec['slots'][number]>) => {
      const spec = batchSpecs[index];
      spec.slots[slot] = { ...spec.slots[slot], ...update };
    };
    const commit = (index: number) => {
      const spec = batchSpecs[index];
      scenes[index].instances = spec.slots.map(instance);
      device.queue.writeBuffer(outputs[index], 0, output(spec));
      dirty[index] = true;
    };

    await run('initial', true);
    await run('steady', true);
    moveSlot(0, 2, { offset: [batchSpecs[0].slots[2].offset[0] + 0.75, 4, 0.5] }); commit(0);
    await run('moved', true);
    batchSpecs[0].slots.pop(); commit(0);
    batchSpecs[7].slots.pop(); commit(7);
    const removed = batchSpecs[13].slots.splice(0); commit(13);
    await run('deleted', true);
    batchSpecs[13].slots.push(...removed.map(slot => ({ ...slot, generation: 2 }))); commit(13);
    batchSpecs[0].slots.push({ offset: [12.5, 4, 0], entityId: 105, generation: 2 }); commit(0);
    await run('restored', true);
    // Masked without a new snapshot: membership follows the material at trace time.
    await switchMaterial(12, 'mask');
    await run('switch-masked-unsnapshotted', false);
    await run('switch-masked', true);
    await switchMaterial(12, 'double');
    await run('switch-opaque-double', true);
    await device.queue.onSubmittedWorkDone();
    return { labels: samples.labels.length, stages, errors, cleanup };
  } finally {
    for (const { query } of queries) query.dispose();
    materials.dispose();
    for (const buffer of buffers) { buffer.destroy(); account(-buffer.size); }
    cleanup.ownedBytesAfterDispose = ownedBytes;
    device.destroy();
  }
}
