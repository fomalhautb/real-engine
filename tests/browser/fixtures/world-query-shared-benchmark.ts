import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { MaterialResources, type GpuMaterial } from '../../../src/gpu-material';
import { createTimestampQuerySet } from '../../../src/gpu-timestamp-span';
import { GpuWorldQuery } from '../../../src/gpu-world-query';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity, translation } from '../../../src/math';
import { packWorldRays, WORLD_HIT_BYTES, type WorldQueryOptions, type WorldRay } from '../../../src/world-query';
import { DenseUdf, type DenseUdfSettings } from './query-sdf';

/**
 * Paired (ABBA-interleaved) GPU timing of one world query over many static
 * Opaque batches: per-batch passes (A), per-batch dispatches in one pass (A1),
 * shared static traversal with per-batch passes (B) and in one pass (B1).
 * Every arm traces the same GPU-resident rays over the same deformation outputs;
 * hit records are checked word for word before any timing is reported.
 * Timestamp intervals bracket K back-to-back queries with empty passes; every
 * timed pass writes both ends (partial writes lose the device on Apple Metal).
 * Optional dense unsigned-distance-field arms (query-sdf.ts) trace the same rays
 * in the same interleaved rounds; their hits are compared with arm A, not gated.
 */
type Vec3 = [number, number, number];
export interface SharedBenchmarkScene { name: string; batches: number; triangles: number; instances: number }
export interface SharedBenchmarkSettings { scenes: SharedBenchmarkScene[]; rayCounts: number[]; rounds: number; warmupRounds: number; repeats: number;
  sdf?: { voxels: number[]; bandVoxels: number; thresholdVoxels: number; maxSteps: number; buildRepeats: number; sweepThresholdVoxels: number[] } }
const ARMS: { name: string; options: WorldQueryOptions }[] = [
  { name: 'A', options: { staticTraversal: 'per-batch', dispatchEncoding: 'pass-per-batch' } },
  { name: 'B', options: { staticTraversal: 'shared', dispatchEncoding: 'pass-per-batch' } },
  { name: 'A1', options: { staticTraversal: 'per-batch', dispatchEncoding: 'single-pass' } },
  { name: 'B1', options: { staticTraversal: 'shared', dispatchEncoding: 'single-pass' } },
];
// Palindromic order: every timed arm appears early and late in each round.
const palindrome = (arms: number) => [...Array(arms).keys(), ...[...Array(arms).keys()].reverse()];

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
/** A clustered soup (roughly a 1.5 m rock): local vertices, per-face normals. */
function soup(seed: number, triangles: number): MeshData {
  const next = random(seed), vertices = new ArrayBuffer(triangles * 3 * 80), view = new DataView(vertices);
  const uvs = new Float32Array(triangles * 6);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let t = 0; t < triangles; t++) {
    const center = [0, 1, 2].map(() => (next() - 0.5) * 1.5);
    const corners = [0, 1, 2].map(() => center.map(value => value + (next() - 0.5) * 0.6));
    const e1 = corners[1].map((v, a) => v - corners[0][a]), e2 = corners[2].map((v, a) => v - corners[0][a]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const length = Math.hypot(...n) || 1;
    corners.forEach((corner, c) => {
      const at = (t * 3 + c) * 80;
      corner.forEach((value, axis) => { view.setFloat32(at + axis * 4, value, true); min[axis] = Math.min(min[axis], value); max[axis] = Math.max(max[axis], value); });
      view.setFloat32(at + 12, 1, true);
      n.forEach((value, axis) => view.setFloat32(at + 16 + axis * 4, value / length, true));
      view.setFloat32(at + 48, 1, true);
      uvs[(t * 3 + c) * 2] = next(); uvs[(t * 3 + c) * 2 + 1] = next();
    });
  }
  return { vertices, vertexCount: triangles * 3, indices: Uint32Array.from({ length: triangles * 3 }, (_, i) => i), uvs, bounds: { min, max } };
}
function output(meshData: MeshData, offsets: Vec3[], capacity: number): Float32Array<ArrayBuffer> {
  const view = new DataView(meshData.vertices), data = new Float32Array(capacity * meshData.vertexCount * 16);
  offsets.forEach((offset, slot) => {
    for (let vertex = 0; vertex < meshData.vertexCount; vertex++) {
      const at = (slot * meshData.vertexCount + vertex) * 16;
      const position = [0, 1, 2].map(axis => view.getFloat32(vertex * 80 + axis * 4, true) + offset[axis]);
      data.set([...position, 1], at);
      data.set([0, 1, 2].map(axis => view.getFloat32(vertex * 80 + 16 + axis * 4, true)).concat(0), at + 4);
      data.set([...position, 1], at + 8); data.set([1, 0, 0, 1], at + 12);
    }
  });
  return data;
}
function percentile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q, low = Math.floor(position), high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
const stats = (values: number[]) => ({ n: values.length, median: percentile(values, 0.5), p25: percentile(values, 0.25),
  p75: percentile(values, 0.75), min: Math.min(...values), max: Math.max(...values) });

export async function runWorldQuerySharedBenchmark(settings: SharedBenchmarkSettings) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  if (!adapter.features.has('timestamp-query')) throw new Error('BENCHMARK_TIMESTAMP_QUERY_UNAVAILABLE');
  const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
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
  const release = (owned: GPUBuffer[]) => { for (const buffer of owned) { buffer.destroy(); account(-buffer.size); buffers.splice(buffers.indexOf(buffer), 1); } };
  // Slots 0-1 time a single span (refits, builds, a one-query trace); with K queries the
  // first query carries 0-1 and the last 2-3, and the interval is slot 3 minus slot 0.
  const querySet = createTimestampQuerySet(device, 4, 'shared query benchmark timestamps');
  const span: GPUComputePassTimestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  const lastSpan: GPUComputePassTimestampWrites = { querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
  const resolve = allocate(32, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
  const stamps = allocate(32, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const maxRays = Math.max(...settings.rayCounts);
  const results = [];
  const cleanup = { ownedBytesAfterDispose: -1 };
  try {
    for (const sceneSettings of settings.scenes) {
      const owned: GPUBuffer[] = [];
      const local = (size: number, usage: GPUBufferUsageFlags, source?: ArrayBuffer) => { const buffer = allocate(size, usage, source); owned.push(buffer); return buffer; };
      const materials = new MaterialResources(device, account);
      const queries: { name: string; query: GpuWorldQuery }[] = [];
      const sdfArms: { name: string; udf: DenseUdf; settings: DenseUdfSettings; build: number[]; region: number[]; evaluations: Record<string, number> }[] = [];
      try {
        const next = random(sceneSettings.batches * 131 + sceneSettings.triangles);
        const meshes = Array.from({ length: 8 }, (_, i) => soup(1000 + i * 17 + sceneSettings.triangles, sceneSettings.triangles));
        const capacity = sceneSettings.instances;
        const offsets: Vec3[][] = [];
        const scenes: SceneBatch[] = Array.from({ length: sceneSettings.batches }, (_, index) => {
          const slots = Array.from({ length: capacity }, () => [(next() - 0.5) * 48, (next() - 0.5) * 48, (next() - 0.5) * 12] as Vec3);
          offsets.push(slots);
          const instances: InstanceState[] = slots.map((offset, slot) => ({ entityId: 10_000 + index * 16 + slot, entityGeneration: 1,
            historyVersion: 0, model: translation(...offset), normalMatrix: identity(), bones: [], deformScalars: [],
            color: [1, 1, 1, 1], metalness: 0, roughness: 1 }));
          return { name: `rock-${index}`, mesh: meshes[index % meshes.length], capacity, inputMode: 'committed', instances,
            deformation: { jointCount: 0, nodes: [] }, morphTargets: [], queryIdentity: { assetId: `rock-${index % 8}`, primitiveId: index, topologyVersion: 1 } };
        });
        const outputs = scenes.map((scene, index) => local(capacity * scene.mesh!.vertexCount * 64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          output(scene.mesh!, offsets[index], capacity).buffer));
        const instanceData = new Float32Array(capacity * 56);
        for (let slot = 0; slot < capacity; slot++) {
          for (const offset of [0, 16, 32]) instanceData.set(identity(), slot * 56 + offset);
          instanceData.set([1, 1, 1, 1, 0, 1, 0, 0], slot * 56 + 48);
        }
        const instanceBuffer = local(instanceData.byteLength, GPUBufferUsage.STORAGE, instanceData.buffer);
        const materialList: GpuMaterial[] = scenes.map((_, index) => materials.create({ definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, baseColor: m.parameter(0) },
          parameters: new Float32Array([(index % 9) / 9, (index % 5) / 5, 0.5, 1]) } as MaterialBinding));
        const dirty = scenes.map(() => true);
        for (const arm of ARMS) {
          const query = await GpuWorldQuery.create(device, materials.layout, account, { maxRays, maxBatches: sceneSettings.batches,
            maxInstancesPerBatch: capacity, maxTrianglesPerMesh: sceneSettings.triangles, maxBytes: 512 * 1024 * 1024,
            maxSharedStaticBytes: 256 * 1024 * 1024, maxTotalWorkUnits: 67_108_864, ...arm.options });
          const pipeline = await query.createPipeline(materialList[0].program);
          scenes.forEach((scene, index) => query.addBatch({ scene, output: outputs[index], instances: instanceBuffer,
            getMaterial: () => materialList[index], getPipeline: () => pipeline, getDirty: () => dirty[index] }));
          queries.push({ name: arm.name, query });
        }
        // GI-like rays: random origins in the populated volume, uniform directions, bounded length.
        const rayNext = random(4242);
        const rays: WorldRay[] = Array.from({ length: maxRays }, () => {
          const z = rayNext() * 2 - 1, phi = rayNext() * Math.PI * 2, r = Math.sqrt(1 - z * z);
          return { origin: [(rayNext() - 0.5) * 48, (rayNext() - 0.5) * 48, (rayNext() - 0.5) * 12],
            direction: [r * Math.cos(phi), r * Math.sin(phi), z], tMin: 0.001, tMax: 16 };
        });
        const packed = packWorldRays(rays, maxRays);
        const staging = local(packed.byteLength, GPUBufferUsage.COPY_SRC, packed);
        // Distance-field input: world-space triangles with the same float32 values as the deformed outputs.
        const perInstance = sceneSettings.triangles, instanceTriangles = (batch: number, slot: number) => {
          const view = new DataView(scenes[batch].mesh!.vertices), data = new Float32Array(perInstance * 12);
          for (let vertex = 0; vertex < perInstance * 3; vertex++) for (let axis = 0; axis < 3; axis++) {
            data[vertex * 4 + axis] = view.getFloat32(vertex * 80 + axis * 4, true) + offsets[batch][slot][axis];
          }
          return data;
        };
        const instanceBounds = (batch: number, offset: Vec3) => ({ min: scenes[batch].mesh!.bounds!.min.map((v, a) => v + offset[a]) as Vec3,
          max: scenes[batch].mesh!.bounds!.max.map((v, a) => v + offset[a]) as Vec3 });
        const sdfSkipped: { voxel: number; reason: string }[] = [];
        const sdfRays = settings.sdf ? local(packed.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, packed) : null;
        const timeSpan = async (encode: (pass: GPUComputePassEncoder) => void) => {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass({ label: 'udf experiment span', timestampWrites: span }); encode(pass); pass.end();
          encoder.resolveQuerySet(querySet, 0, 2, resolve, 0); encoder.copyBufferToBuffer(resolve, 0, stamps, 0, 16);
          device.queue.submit([encoder.finish()]);
          await stamps.mapAsync(GPUMapMode.READ);
          const [begin, end] = new BigUint64Array(stamps.getMappedRange().slice(0));
          stamps.unmap();
          return Number(end - begin) / 1e6;
        };
        const allTriangles = new Float32Array(sceneSettings.batches * capacity * perInstance * 12);
        let sceneMin: Vec3 = [Infinity, Infinity, Infinity], sceneMax: Vec3 = [-Infinity, -Infinity, -Infinity];
        if (settings.sdf) {
          for (let batch = 0; batch < sceneSettings.batches; batch++) for (let slot = 0; slot < capacity; slot++) {
            allTriangles.set(instanceTriangles(batch, slot), (batch * capacity + slot) * perInstance * 12);
            const box = instanceBounds(batch, offsets[batch][slot]);
            sceneMin = sceneMin.map((v, a) => Math.min(v, box.min[a])) as Vec3; sceneMax = sceneMax.map((v, a) => Math.max(v, box.max[a])) as Vec3;
          }
          for (const voxel of settings.sdf.voxels) {
            const udfSettings = { voxel, bandVoxels: settings.sdf.bandVoxels, thresholdVoxels: settings.sdf.thresholdVoxels, maxSteps: settings.sdf.maxSteps };
            try {
              const udf = await DenseUdf.create(device, { min: sceneMin, max: sceneMax }, udfSettings, allTriangles, maxRays);
              account(udf.ownedBytes);
              const build: number[] = [];
              for (let repeat = 0; repeat < settings.sdf.buildRepeats; repeat++) {
                udf.beginEncoder();
                build.push(await timeSpan(pass => { udf.encodeClear(pass, udf.all); udf.encodeSplat(pass, 0, allTriangles.length / 12, udf.all); }));
              }
              sdfArms.push({ name: `SDF${voxel}`, udf, settings: udfSettings, build, region: [],
                evaluations: { build: udf.evaluations(allTriangles, 0, allTriangles.length / 12, udf.all) } });
            } catch (error) { sdfSkipped.push({ voxel, reason: String(error) }); }
          }
        }
        const timeRefit = async (query: GpuWorldQuery, frame: number) => {
          query.validateSources();
          const uploaded = query.updateSnapshot(frame, 0);
          const encoder = device.createCommandEncoder();
          query.encodeRefit(encoder, span);
          encoder.resolveQuerySet(querySet, 0, 2, resolve, 0); encoder.copyBufferToBuffer(resolve, 0, stamps, 0, 16);
          device.queue.submit([encoder.finish()]);
          await stamps.mapAsync(GPUMapMode.READ);
          const [begin, end] = new BigUint64Array(stamps.getMappedRange().slice(0));
          stamps.unmap();
          return { ms: Number(end - begin) / 1e6, uploaded };
        };
        // Initial snapshot (pool creation and first fill for the shared arms), then a one-batch edit, then a static frame.
        const refit: Record<string, Record<string, unknown>> = {};
        for (const { name, query } of queries) refit[name] = { initial: await timeRefit(query, 1) };
        dirty.fill(false);
        const edited = 7, before = instanceBounds(edited, [...offsets[edited][0]] as Vec3);
        offsets[edited][0][0] += 0.5;
        device.queue.writeBuffer(outputs[edited], 0, output(scenes[edited].mesh!, offsets[edited], capacity));
        scenes[edited].instances = scenes[edited].instances.map((state, slot) => slot ? state : { ...state, model: translation(...offsets[edited][0]) });
        dirty[edited] = true;
        for (const { name, query } of queries) refit[name].oneBatch = await timeRefit(query, 2);
        // Distance-field region update for the same edit: clear the old and new band, re-splat overlapping batches.
        const after = instanceBounds(edited, offsets[edited][0]);
        const changed = { min: before.min.map((v, a) => Math.min(v, after.min[a])) as Vec3, max: before.max.map((v, a) => Math.max(v, after.max[a])) as Vec3 };
        if (sdfArms.length) allTriangles.set(instanceTriangles(edited, 0), edited * capacity * perInstance * 12);
        for (const arm of sdfArms) {
          arm.udf.writeTriangles(edited * capacity * perInstance, instanceTriangles(edited, 0));
          const region = arm.udf.region(changed), reach = arm.udf.band * 2;
          const overlapping = scenes.map((_, batch) => batch).filter(batch => offsets[batch].some(offset => {
            const box = instanceBounds(batch, offset);
            return box.min.every((v, a) => v <= changed.max[a] + reach) && box.max.every((v, a) => v >= changed.min[a] - reach);
          }));
          for (let repeat = 0; repeat < settings.sdf!.buildRepeats; repeat++) {
            arm.udf.beginEncoder();
            arm.region.push(await timeSpan(pass => {
              arm.udf.encodeClear(pass, region);
              for (const batch of overlapping) arm.udf.encodeSplat(pass, batch * capacity * perInstance, capacity * perInstance, region);
            }));
          }
          arm.evaluations.region = overlapping.reduce((sum, batch) => sum + arm.udf.evaluations(allTriangles, batch * capacity * perInstance, capacity * perInstance, region), 0);
          arm.evaluations.regionCells = (region.high[0] - region.low[0]) * (region.high[1] - region.low[1]) * (region.high[2] - region.low[2]);
          arm.evaluations.regionBatches = overlapping.length;
        }
        dirty.fill(false);
        for (const { name, query } of queries) refit[name].static = await timeRefit(query, 3);
        for (const { query } of queries) {
          const encoder = device.createCommandEncoder();
          encoder.copyBufferToBuffer(staging, 0, query.rayBuffer, 0, packed.byteLength);
          device.queue.submit([encoder.finish()]);
        }
        // Parity and hardware-independent work at full ray count.
        const readback = local(maxRays * WORLD_HIT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const records: Uint32Array[] = [];
        for (const { query } of queries) {
          const encoder = device.createCommandEncoder();
          query.encodeTrace(encoder, maxRays, 0, 0);
          encoder.copyBufferToBuffer(query.hitBuffer, 0, readback, 0, maxRays * WORLD_HIT_BYTES);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          records.push(new Uint32Array(readback.getMappedRange().slice(0)));
          readback.unmap();
        }
        const words = WORLD_HIT_BYTES / 4;
        const parity = queries.map(({ name }, arm) => {
          let mismatches = 0, hits = 0, overflows = 0, units = 0, tests = 0;
          for (let ray = 0; ray < maxRays; ray++) {
            const at = ray * words;
            if (records[arm][at]) hits++;
            if (records[arm][at + 38]) overflows++;
            units += records[arm][at + 36]; tests += records[arm][at + 37];
            for (let w = 0; w < words; w++) if ((w < 36 || w > 38) && records[arm][at + w] !== records[0][at + w]) { mismatches++; break; }
          }
          return { arm: name, mismatches, hits, overflows, workUnitsPerRay: units / maxRays, triangleTestsPerRay: tests / maxRays };
        });
        // Timestamps sit on the traced passes themselves: on apple/metal-3, separate bracket passes
        // before and after the work (empty, or writing the arm's hit buffer) measured 0-1 quantum
        // for four queries of every arm, so they do not bracket the traced work there.
        type Encode = (encoder: GPUCommandEncoder, count: number, timing?: GPUComputePassTimestampWrites) => void;
        const timed: { name: string; encode: Encode }[] = [
          ...queries.map(({ name, query }) => ({ name, encode: ((encoder, count, timing) => {
            query.encodeTrace(encoder, count, 0, 0, undefined, undefined, 'material', timing); }) as Encode })),
          ...sdfArms.map(({ name, udf }) => ({ name, encode: ((encoder, count, timing) => {
            const pass = encoder.beginComputePass({ label: 'udf experiment trace', timestampWrites: timing }); udf.encodeTrace(pass, sdfRays!, count); pass.end();
          }) as Encode }))];
        const order = palindrome(timed.length);
        const timing = [];
        for (const count of settings.rayCounts) {
          const samples: number[][] = timed.map(() => []), cpu: number[][] = timed.map(() => []);
          const roundMeans: number[][] = timed.map(() => []);
          for (let round = 0; round < settings.warmupRounds + settings.rounds; round++) {
            const roundSamples: number[][] = timed.map(() => []);
            for (const arm of order) {
              for (const { udf } of sdfArms) udf.beginEncoder();
              const encoder = device.createCommandEncoder();
              const started = performance.now();
              for (let repeat = 0; repeat < settings.repeats; repeat++) {
                timed[arm].encode(encoder, count, repeat === 0 ? span : repeat === settings.repeats - 1 ? lastSpan : undefined);
              }
              const encoded = performance.now() - started;
              encoder.resolveQuerySet(querySet, 0, 4, resolve, 0); encoder.copyBufferToBuffer(resolve, 0, stamps, 0, 32);
              device.queue.submit([encoder.finish()]);
              await stamps.mapAsync(GPUMapMode.READ);
              const [first, firstEnd, , lastEnd] = new BigUint64Array(stamps.getMappedRange().slice(0));
              stamps.unmap();
              if (round < settings.warmupRounds) continue;
              const ms = Number((settings.repeats === 1 ? firstEnd : lastEnd) - first) / 1e6 / settings.repeats;
              samples[arm].push(ms); roundSamples[arm].push(ms); cpu[arm].push(encoded / settings.repeats);
            }
            if (round >= settings.warmupRounds) roundSamples.forEach((values, arm) => roundMeans[arm].push(values.reduce((a, b) => a + b, 0) / values.length));
          }
          const ratio = (a: number, b: number) => stats(roundMeans[a].map((value, round) => value / roundMeans[b][round]));
          const sdfRatios = Object.fromEntries(sdfArms.flatMap((arm, i) => [[`${arm.name}/B1`, ratio(ARMS.length + i, 3)], [`${arm.name}/A`, ratio(ARMS.length + i, 0)]]));
          timing.push({ rays: count, repeats: settings.repeats, perQueryGpuMs: Object.fromEntries(timed.map((arm, i) => [arm.name, stats(samples[i])])),
            perQueryCpuEncodeMs: Object.fromEntries(timed.map((arm, i) => [arm.name, stats(cpu[i])])),
            pairedRoundRatios: { 'B/A': ratio(1, 0), 'A1/A': ratio(2, 0), 'B1/A': ratio(3, 0), 'B1/A1': ratio(3, 2), 'B1/B': ratio(3, 1), ...sdfRatios } });
        }
        // Distance-field hits against arm A (exact BVH) on the same rays.
        const reference = new Float32Array(records[0].buffer, records[0].byteOffset, records[0].length);
        const sdf = [];
        for (const arm of sdfArms) {
          const output = local(maxRays * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          const traceAt = async (threshold: number) => {
            arm.udf.beginEncoder();
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass({ label: 'udf experiment readback trace' }); arm.udf.encodeTrace(pass, sdfRays!, maxRays, threshold); pass.end();
            encoder.copyBufferToBuffer(arm.udf.resultBuffer, 0, output, 0, maxRays * 16);
            device.queue.submit([encoder.finish()]);
            await output.mapAsync(GPUMapMode.READ);
            const traced = new Float32Array(output.getMappedRange().slice(0));
            output.unmap();
            return traced;
          };
          let values = await traceAt(arm.udf.threshold);
          const classify = (freeOnly: boolean) => {
            let bothHit = 0, bothMiss = 0, sdfOnly = 0, bvhOnly = 0, unresolved = 0, considered = 0;
            const absolute: number[] = [], signed: number[] = [];
            for (let ray = 0; ray < maxRays; ray++) {
              if (freeOnly && !(values[ray * 4 + 2] >= 2 * arm.udf.threshold)) continue;
              considered++;
              const bvh = records[0][ray * (WORLD_HIT_BYTES / 4)] !== 0, distance = values[ray * 4];
              if (values[ray * 4 + 3] > 0) unresolved++;
              if (bvh && distance >= 0) {
                bothHit++;
                const delta = distance - reference[ray * (WORLD_HIT_BYTES / 4) + 4];
                absolute.push(Math.abs(delta)); signed.push(delta);
              } else if (bvh) bvhOnly++; else if (distance >= 0) sdfOnly++; else bothMiss++;
            }
            return { rays: considered, bothHit, bothMiss, sdfOnly, bvhOnly, unresolved,
              absDistanceError: absolute.length ? { ...stats(absolute), p90: percentile(absolute, 0.9), p99: percentile(absolute, 0.99) } : null,
              meanSignedError: signed.length ? signed.reduce((a, b) => a + b, 0) / signed.length : null };
          };
          const steps = Array.from({ length: maxRays }, (_, ray) => values[ray * 4 + 1]);
          const entry = { arm: arm.name, settings: arm.settings, dims: arm.udf.dims, cellBytes: arm.udf.cellBytes, ownedBytes: arm.udf.ownedBytes,
            band: arm.udf.band, threshold: arm.udf.threshold, buildGpuMs: arm.build, regionUpdateGpuMs: arm.region, evaluations: arm.evaluations,
            stepsPerRay: { mean: steps.reduce((a, b) => a + b, 0) / maxRays, p99: percentile(steps, 0.99), max: Math.max(...steps) },
            all: classify(false), freeSpaceOrigins: classify(true), sweep: [] as { thresholdVoxels: number; all: ReturnType<typeof classify>; freeSpaceOrigins: ReturnType<typeof classify> }[] };
          // Accuracy only (untimed): a lower hit distance trades false hits for tunnelling misses.
          for (const thresholdVoxels of settings.sdf!.sweepThresholdVoxels) {
            values = await traceAt(thresholdVoxels * arm.settings.voxel);
            entry.sweep.push({ thresholdVoxels, all: classify(false), freeSpaceOrigins: classify(true) });
          }
          sdf.push(entry);
        }
        const diagnostics = queries.map(({ name, query }) => {
          const value = query.diagnostics();
          return { arm: name, ownedGpuBytes: value.ownedGpuBytes, sharedStatic: value.sharedStatic, sceneAcceleration: value.sceneAcceleration,
            instanceAcceleration: value.instanceAcceleration };
        });
        results.push({ scene: sceneSettings, triangles: sceneSettings.batches * sceneSettings.instances * sceneSettings.triangles,
          refit, parity, timing, diagnostics, sdf, sdfSkipped });
      } finally {
        for (const arm of sdfArms.splice(0)) { arm.udf.dispose(); account(-arm.udf.ownedBytes); }
        for (const { query } of queries) query.dispose();
        materials.dispose();
        release(owned);
      }
    }
    await device.queue.onSubmittedWorkDone();
    const info = adapter.info as GPUAdapterInfo & { isFallbackAdapter?: boolean };
    return { adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description,
      isFallbackAdapter: info.isFallbackAdapter ?? null }, settings, results, errors, cleanup };
  } finally {
    release([...buffers]);
    querySet.destroy();
    cleanup.ownedBytesAfterDispose = ownedBytes;
    device.destroy();
  }
}
