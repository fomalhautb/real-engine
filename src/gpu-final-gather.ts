import commonSource from './shaders/diffuse-gi-common.wgsl?raw';
import interpolateSource from './shaders/diffuse-gi-interpolate.wgsl?raw';
import updateSource from './shaders/final-gather-update.wgsl?raw';
import sampleSource from './shaders/final-gather-sample.wgsl?raw';
import surfaceFrameSource from './shaders/surface-frame.wgsl?raw';
import type { SceneBatch } from './contracts';
import { DIFFUSE_GI_PROBE_BYTES } from './diffuse-gi';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from './gi-visibility';
import { planFinalGatherGrid, resolveFinalGatherOptions, type FinalGatherGridPlan, type ResolvedFinalGatherOptions } from './final-gather';
import type { GpuWorldQuery } from './gpu-world-query';
import { WORLD_HIT_BYTES, WORLD_RAY_BYTES } from './world-query';
import { DEFORMED_VERTEX_BYTES } from './geometry';

export const finalGatherSampleSource = sampleSource;
export const FINAL_GATHER_ANCHOR_BYTES = 96;
const PARAMETER_BYTES = 224;
const AREA_SAMPLE_BYTES = 48;
const AREA_PARAMETER_BYTES = 32;
const MAX_PROBE_ANCHORS = 1_024;
const READBACK_BYTES = MAX_PROBE_ANCHORS * FINAL_GATHER_ANCHOR_BYTES * 2;
type EntryPoint = 'generateCamera' | 'generateHemisphere' | 'generateShadows' | 'integrate'
  | 'clearArea' | 'generateArea' | 'integrateArea' | 'accumulate' | 'denoise';
export interface FinalGatherUpdate {
  frame: number;
  width: number;
  height: number;
  eye: readonly number[];
  viewProjection: ArrayLike<number>;
  previousViewProjection?: ArrayLike<number>;
  reset: boolean;
  enabled: boolean;
  multiBounce: boolean;
  materialRevision: number;
}
export interface FinalGatherProbeResult {
  frame: number;
  anchors: {
    index: number;
    position: [number, number, number];
    normal: [number, number, number];
    identity: [number, number, number, number];
    versions: [number, number, number, number];
    /** Raw irradiance, before the separate spatial denoise pass. */
    irradiance: [number, number, number];
    /** Actual GPU effective history count, not the CPU nominal update counter. */
    historyUpdates: number;
    valid: boolean;
    state: [number, number, number, number];
    filteredIrradiance: [number, number, number];
  }[];
}
interface AreaBatch {
  scene: SceneBatch;
  mesh: SceneBatch['mesh'];
  vertices: ArrayBuffer;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
  name: string;
  identity: string;
  capacity: number;
  sourceId: number;
  count: number;
  parameters: GPUBuffer;
  identities: GPUBuffer;
  metadata: Uint32Array<ArrayBuffer>;
  group: GPUBindGroup;
}

function uint(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) throw new Error('FINAL_GATHER_' + label);
  return value;
}
function topologyIdentity(scene: SceneBatch): string {
  const identity = scene.queryIdentity;
  return identity ? JSON.stringify([identity.assetId, identity.primitiveId, identity.topologyVersion]) : '';
}
/** Bounded 4x4 inversion, independent of the affine-only simulation helpers. */
function inverseProjection(matrix: ArrayLike<number>): Float32Array<ArrayBuffer> {
  if (matrix.length !== 16 || !Array.from(matrix).every(Number.isFinite)) throw new Error('FINAL_GATHER_VIEW_PROJECTION');
  const rows = Array.from({ length: 4 }, (_, row) => Array.from({ length: 8 }, (_, column) =>
    column < 4 ? matrix[column * 4 + row] : Number(column - 4 === row)));
  for (let column = 0; column < 4; column++) {
    let pivot = column;
    for (let row = column + 1; row < 4; row++) if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    if (rows[pivot][column] === 0) throw new Error('FINAL_GATHER_VIEW_PROJECTION');
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const divisor = rows[column][column];
    for (let i = 0; i < 8; i++) rows[column][i] /= divisor;
    for (let row = 0; row < 4; row++) {
      if (row === column) continue;
      const multiplier = rows[row][column];
      for (let i = 0; i < 8; i++) rows[row][i] -= multiplier * rows[column][i];
    }
  }
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column++) for (let row = 0; row < 4; row++) output[column * 4 + row] = rows[row][column + 4];
  if (!output.every(Number.isFinite)) throw new Error('FINAL_GATHER_VIEW_PROJECTION');
  // A finite near and far plane are required by the camera-ray unprojection.
  for (const depth of [0, 1]) for (const x of [-1, 1]) for (const y of [-1, 1]) {
    const w = output[3] * x + output[7] * y + output[11] * depth + output[15];
    if (!Number.isFinite(w) || Math.abs(w) < 1e-12) throw new Error('FINAL_GATHER_FINITE_CAMERA_RANGE');
  }
  return output;
}

/** Exact first-segment diffuse transport at a bounded screen grid. The world
 * irradiance field supplies secondary bounces at the actual first hit. Camera
 * motion reprojects raw history through validated surface identities/geometry. */
export class GpuFinalGather {
  readonly options: ResolvedFinalGatherOptions;
  private readonly buffers: GPUBuffer[] = [];
  private readonly batches: AreaBatch[] = [];
  private bytes = 0;
  private disposed = false;
  private encoding = false;
  private readbackPending = false;
  private updated = false;
  private enabled = false;
  private frame = 0;
  private encodedFrame = 0;
  private materialRevision = 0;
  private outputIndex = 1;
  private sourceBlocks = 0;
  private activeEmitters = 0;
  private areaRayCount = 0;
  private grid: FinalGatherGridPlan = { spacing: 1, columns: 0, rows: 0, anchorCount: 0, rayCount: 0 };
  private historyUpdates = 0;
  private historyResetReason = 'initial';
  private cameraMoving = false;
  private previous?: { frame: number; width: number; height: number; spacing: number; camera: number[];
    material: number; enabled: boolean; multiBounce: boolean; sources: string };
  private parameters!: GPUBuffer;
  private cameraHits!: GPUBuffer;
  private primaryHits!: GPUBuffer;
  private anchors!: [GPUBuffer, GPUBuffer];
  private filteredAnchors!: GPUBuffer;
  private readback!: GPUBuffer;
  private areaSamples!: GPUBuffer;
  private pipelines!: Record<EntryPoint, GPUComputePipeline>;
  private groups!: Record<Exclude<EntryPoint, 'integrate' | 'generateArea' | 'integrateArea' | 'accumulate' | 'denoise'>, GPUBindGroup>;
  private integrateGroups!: GPUBindGroup[][];
  private areaIntegrateGroups!: GPUBindGroup[];
  private accumulateGroups!: GPUBindGroup[];
  private denoiseGroups!: GPUBindGroup[];

  get surfaceParameters(): GPUBuffer { return this.parameters; }
  get surfaceBuffer(): GPUBuffer { return this.filteredAnchors; }
  /** A stable filtered output is separate from both raw temporal history fields. */
  get outputBuffers(): [GPUBuffer, GPUBuffer] { return [this.filteredAnchors, this.filteredAnchors]; }
  get currentIndex(): number { return this.outputIndex; }

  static async create(device: GPUDevice, queries: GpuWorldQuery, giParameterBuffer: GPUBuffer,
    giProbeBuffers: [GPUBuffer, GPUBuffer], options: ResolvedFinalGatherOptions,
    account: (bytes: number) => void, maxBytesRemaining: number): Promise<GpuFinalGather> {
    const resolved = resolveFinalGatherOptions(options);
    const capacity = Math.min(resolved.maxAnchors, Math.floor(queries.options.maxRays / resolved.raysPerAnchor),
      Math.floor(queries.options.maxTotalWorkUnits / resolved.raysPerAnchor));
    if (capacity < 1) throw new Error('FINAL_GATHER_QUERY_RAY_BUDGET');
    if (queries.options.maxTraceParameterSetsPerEncoder < 4) throw new Error('FINAL_GATHER_QUERY_PARAMETER_BUDGET');
    if (device.limits.maxStorageBuffersPerShaderStage < 6) throw new Error('FINAL_GATHER_STORAGE_BINDING_LIMIT');
    if (!Number.isSafeInteger(maxBytesRemaining) || maxBytesRemaining < 1) throw new Error('FINAL_GATHER_MEMORY_BUDGET');
    if (giParameterBuffer.size < 112 || !(giParameterBuffer.usage & GPUBufferUsage.UNIFORM)
      || giProbeBuffers[0].size !== giProbeBuffers[1].size || giProbeBuffers[0].size < DIFFUSE_GI_PROBE_BYTES
      || giProbeBuffers[0].size % DIFFUSE_GI_PROBE_BYTES !== 0
      || giProbeBuffers.some(buffer => !(buffer.usage & GPUBufferUsage.STORAGE))
      || queries.rayBuffer.size < queries.options.maxRays * WORLD_RAY_BYTES
      || queries.hitBuffer.size < queries.options.maxRays * WORLD_HIT_BYTES) throw new Error('FINAL_GATHER_BORROWED_BUFFER_LAYOUT');
    const result = new GpuFinalGather(device, queries, giParameterBuffer, giProbeBuffers, resolved,
      capacity, account, Math.min(resolved.maxBytes, maxBytesRemaining));
    try { await result.initialize(); return result; }
    catch (error) { result.dispose(); throw error; }
  }
  private constructor(private readonly device: GPUDevice, private readonly queries: GpuWorldQuery,
    private readonly giParameterBuffer: GPUBuffer, private readonly giProbeBuffers: [GPUBuffer, GPUBuffer],
    options: ResolvedFinalGatherOptions, private readonly anchorCapacity: number,
    private readonly account: (bytes: number) => void, private readonly maxBytes: number) { this.options = options; }

  private assertAlive(): void { if (this.disposed) throw new Error('FINAL_GATHER_DISPOSED'); }
  private preflight(sizes: readonly number[]): void {
    if (sizes.some(size => !Number.isSafeInteger(size) || size < 4
      || size > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize))) {
      throw new Error('FINAL_GATHER_BUFFER_LIMIT');
    }
    if (this.bytes + sizes.reduce((a, b) => a + b, 0) > this.maxBytes) throw new Error('FINAL_GATHER_MEMORY_BUDGET');
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: Uint32Array<ArrayBuffer>): GPUBuffer {
    this.preflight([size]);
    const buffer = this.device.createBuffer({ label, size, usage, mappedAtCreation: !!data });
    this.buffers.push(buffer); this.bytes += size; this.account(size);
    if (data) { new Uint32Array(buffer.getMappedRange()).set(data); buffer.unmap(); }
    return buffer;
  }
  private bind(pipeline: GPUComputePipeline, entries: readonly [number, GPUBuffer][]): GPUBindGroup {
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  }
  private async initialize(): Promise<void> {
    const cap = this.anchorCapacity, rayCapacity = cap * this.options.raysPerAnchor;
    this.preflight([PARAMETER_BYTES, cap * WORLD_HIT_BYTES, rayCapacity * WORLD_HIT_BYTES,
      cap * FINAL_GATHER_ANCHOR_BYTES, cap * FINAL_GATHER_ANCHOR_BYTES, cap * FINAL_GATHER_ANCHOR_BYTES,
      this.queries.options.maxRays * AREA_SAMPLE_BYTES, READBACK_BYTES]);
    if (Math.ceil(this.queries.options.maxRays / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error('FINAL_GATHER_DISPATCH_LIMIT');
    }
    this.parameters = this.buffer('surface final gather parameters', PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.cameraHits = this.buffer('surface final gather exact camera hits', cap * WORLD_HIT_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.primaryHits = this.buffer('surface final gather first-segment hits', rayCapacity * WORLD_HIT_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.anchors = [this.buffer('surface final gather A', cap * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
      this.buffer('surface final gather B', cap * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)];
    this.filteredAnchors = this.buffer('surface final gather filtered output', cap * FINAL_GATHER_ANCHOR_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.readback = this.buffer('surface final gather explicit anchor diagnostics', READBACK_BYTES,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.areaSamples = this.buffer('surface final gather emission samples', this.queries.options.maxRays * AREA_SAMPLE_BYTES, GPUBufferUsage.STORAGE);
    const common = commonSource.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`);
    const module = this.device.createShaderModule({ label: 'surface final gather',
      code: surfaceFrameSource + '\n' + updateSource.replace('// GI_COMMON', common).replace('// GI_INTERPOLATE', interpolateSource) });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `final gather:${error.lineNum}: ${error.message}`).join('\n'));
    const names: EntryPoint[] = ['generateCamera', 'generateHemisphere', 'generateShadows', 'integrate',
      'clearArea', 'generateArea', 'integrateArea', 'accumulate', 'denoise'];
    const results = await Promise.all(names.map(entryPoint => this.device.createComputePipelineAsync({
      label: 'surface final gather ' + entryPoint, layout: 'auto', compute: { module, entryPoint } })));
    this.pipelines = Object.fromEntries(names.map((name, i) => [name, results[i]])) as Record<EntryPoint, GPUComputePipeline>;
    const p = this.pipelines, q = this.queries;
    this.groups = {
      generateCamera: this.bind(p.generateCamera, [[0, this.parameters], [2, q.rayBuffer]]),
      generateHemisphere: this.bind(p.generateHemisphere, [[0, this.parameters], [1, this.giParameterBuffer], [2, q.rayBuffer], [4, this.cameraHits]]),
      generateShadows: this.bind(p.generateShadows, [[0, this.parameters], [1, this.giParameterBuffer], [2, q.rayBuffer], [5, this.primaryHits]]),
      clearArea: this.bind(p.clearArea, [[0, this.parameters], [2, q.rayBuffer], [13, this.areaSamples]]),
    };
    this.integrateGroups = this.anchors.map(output => this.giProbeBuffers.map(probes => this.bind(p.integrate,
      [[0, this.parameters], [1, this.giParameterBuffer], [3, q.hitBuffer], [4, this.cameraHits],
        [5, this.primaryHits], [6, output], [8, probes]])));
    this.areaIntegrateGroups = this.anchors.map(output => this.bind(p.integrateArea,
      [[0, this.parameters], [3, q.hitBuffer], [6, output], [13, this.areaSamples]]));
    this.accumulateGroups = this.anchors.map((output, index) => this.bind(p.accumulate,
      [[0, this.parameters], [4, this.cameraHits], [6, output], [7, this.anchors[1 - index]]]));
    this.denoiseGroups = this.anchors.map(input => this.bind(p.denoise,
      [[0, this.parameters], [6, this.filteredAnchors], [7, input]]));
  }

  registerBatch(scene: SceneBatch, output: GPUBuffer, sourceId: number): void {
    this.assertAlive();
    if (this.updated) throw new Error('FINAL_GATHER_REGISTRATION_CLOSED');
    if (scene.emissiveSampling === undefined) return;
    if (scene.emissiveSampling !== 'triangle-area') throw new Error('FINAL_GATHER_EMISSIVE_SAMPLING');
    uint(sourceId, 'SOURCE_ID');
    if (sourceId === 0 || this.batches.some(batch => batch.scene === scene || batch.sourceId === sourceId)) throw new Error('FINAL_GATHER_DUPLICATE_SOURCE');
    const mesh = scene.mesh;
    if (!(mesh.indices instanceof Uint32Array) || mesh.indices.length < 3 || mesh.indices.length % 3 !== 0
      || mesh.indices.length / 3 > this.queries.options.maxTrianglesPerMesh
      || !Number.isSafeInteger(mesh.vertexCount) || mesh.vertexCount < 1 || !(mesh.vertices instanceof ArrayBuffer)
      || mesh.vertices.byteLength !== mesh.vertexCount * 80) throw new Error('FINAL_GATHER_MESH_LAYOUT');
    if (!Number.isSafeInteger(scene.capacity) || scene.capacity < 1 || scene.capacity > this.queries.options.maxInstancesPerBatch
      || scene.instances.length > scene.capacity || this.batches.length >= 256) throw new Error('FINAL_GATHER_INSTANCE_BUDGET');
    for (const index of mesh.indices) if (index >= mesh.vertexCount) throw new Error('FINAL_GATHER_MESH_INDEX');
    if (output.size < mesh.vertexCount * scene.capacity * DEFORMED_VERTEX_BYTES || !(output.usage & GPUBufferUsage.STORAGE)
      || output.size > this.device.limits.maxStorageBufferBindingSize) throw new Error('FINAL_GATHER_DEFORMATION_BUFFER');
    this.preflight([mesh.indices.byteLength, scene.capacity * 16, AREA_PARAMETER_BYTES]);
    const start = this.buffers.length, before = this.bytes;
    try {
      const indices = this.buffer('final gather emitter immutable indices', mesh.indices.byteLength, GPUBufferUsage.STORAGE, mesh.indices.slice());
      const identities = this.buffer('final gather emitter identities', scene.capacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const parameters = this.buffer('final gather emitter parameters', AREA_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const group = this.bind(this.pipelines.generateArea, [[0, this.parameters], [1, this.giParameterBuffer],
        [2, this.queries.rayBuffer], [4, this.cameraHits], [9, parameters], [10, output], [11, indices],
        [12, identities], [13, this.areaSamples]]);
      this.batches.push({ scene, mesh, vertices: mesh.vertices, indices: mesh.indices, vertexCount: mesh.vertexCount,
        indexCount: mesh.indices.length, name: scene.name, identity: topologyIdentity(scene), capacity: scene.capacity,
        sourceId, count: 0, parameters, identities, metadata: new Uint32Array(scene.capacity * 4), group });
    } catch (error) {
      for (const buffer of this.buffers.splice(start)) buffer.destroy();
      this.account(before - this.bytes); this.bytes = before;
      throw error;
    }
  }

  update(input: FinalGatherUpdate): number {
    this.assertAlive();
    uint(input.frame, 'FRAME'); uint(input.materialRevision, 'MATERIAL_REVISION');
    if (input.frame === 0 || input.eye.length !== 3 || !input.eye.every(Number.isFinite)
      || typeof input.enabled !== 'boolean' || typeof input.reset !== 'boolean' || typeof input.multiBounce !== 'boolean') {
      throw new Error('FINAL_GATHER_UPDATE');
    }
    if (!input.eye.every(value => Number.isFinite(Math.fround(value)))) throw new Error('FINAL_GATHER_EYE');
    const inverse = inverseProjection(input.viewProjection);
    let sourceBlocks = 0, activeEmitters = 0;
    this.batches.forEach((batch, index) => {
      const { scene } = batch;
      if (scene.mesh !== batch.mesh || scene.mesh.vertices !== batch.vertices || scene.mesh.indices !== batch.indices
        || scene.mesh.vertexCount !== batch.vertexCount || scene.mesh.indices.length !== batch.indexCount
        || scene.name !== batch.name || scene.capacity !== batch.capacity || topologyIdentity(scene) !== batch.identity
        || scene.emissiveSampling !== 'triangle-area') throw new Error('FINAL_GATHER_TOPOLOGY_CHANGED');
      if (scene.instances.length > batch.capacity) throw new Error('FINAL_GATHER_INSTANCE_BUDGET');
      const alive = new Set<number>();
      for (const state of scene.instances) {
        uint(state.entityId, 'ENTITY_ID'); uint(state.entityGeneration ?? 0, 'GENERATION');
        if (alive.has(state.entityId)) throw new Error('FINAL_GATHER_DUPLICATE_ENTITY');
        alive.add(state.entityId);
      }
      if (scene.instances.length) { sourceBlocks = index + 1; activeEmitters++; }
    });
    const countLimit = Math.max(this.options.raysPerAnchor, sourceBlocks, 1);
    const anchorLimit = Math.min(this.anchorCapacity, Math.floor(this.queries.options.maxRays / countLimit),
      Math.floor(this.queries.options.maxTotalWorkUnits / countLimit));
    if (anchorLimit < 1) throw new Error('FINAL_GATHER_EMISSIVE_RAY_BUDGET');
    const grid = planFinalGatherGrid(input.width, input.height, { ...this.options, maxAnchors: anchorLimit },
      this.queries.options.maxRays, this.queries.options.maxTotalWorkUnits);
    const camera = [...Array.from(input.viewProjection, Math.fround), ...input.eye.map(Math.fround)];
    const sources = JSON.stringify(this.batches.map(batch => batch.scene.instances.map(state =>
      [state.entityId, state.entityGeneration ?? 0, state.historyVersion])));
    const old = this.previous;
    let previousProjection = new Float32Array(camera.slice(0, 16));
    if (input.previousViewProjection !== undefined) {
      const values = Array.from(input.previousViewProjection);
      if (values.length !== 16 || !values.every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value)))) {
        throw new Error('FINAL_GATHER_PREVIOUS_VIEW_PROJECTION');
      }
      previousProjection = new Float32Array(values);
      if (old && previousProjection.some((value, i) => value !== old.camera[i])) {
        throw new Error('FINAL_GATHER_PREVIOUS_CAMERA_MISMATCH');
      }
    }
    const cameraMoving = !!old && old.camera.some((value, i) => value !== camera[i]);
    const resetReason = input.reset ? 'scene-lighting-discontinuity' : !old ? 'initial'
      : old.frame + 1 !== input.frame ? 'frame-discontinuity'
      : old.frame !== this.encodedFrame ? 'unrendered-history'
      : old.width !== input.width || old.height !== input.height || old.spacing !== grid.spacing ? 'grid-changed'
      : input.previousViewProjection === undefined ? 'camera-cut'
      : old.material !== input.materialRevision ? 'material-changed'
      : old.enabled !== input.enabled || old.multiBounce !== input.multiBounce ? 'mode-changed'
      : old.sources !== sources ? 'emitter-identities-changed' : null;
    if (resetReason) { this.historyUpdates = 0; this.historyResetReason = resetReason; }
    this.previous = { frame: input.frame, width: input.width, height: input.height, spacing: grid.spacing,
      camera, material: input.materialRevision, enabled: input.enabled, multiBounce: input.multiBounce, sources };
    this.grid = grid; this.frame = input.frame; this.materialRevision = input.materialRevision;
    this.cameraMoving = cameraMoving;
    this.enabled = input.enabled; this.sourceBlocks = sourceBlocks; this.activeEmitters = activeEmitters;
    this.areaRayCount = input.enabled ? sourceBlocks * grid.anchorCount : 0;
    const data = new Float32Array(PARAMETER_BYTES / 4), integers = new Uint32Array(data.buffer);
    data.set(inverse); data.set(previousProjection, 16); data.set(input.eye, 32);
    integers.set([input.width, input.height, grid.columns, grid.rows], 36);
    integers.set([grid.spacing, grid.anchorCount, this.options.raysPerAnchor, this.options.maxHistory], 40);
    integers.set([input.frame, Number(input.enabled), Number(resetReason !== null), Number(input.multiBounce)], 44);
    data.set([this.options.normalBias, 0, Math.max(0.001, this.options.normalBias * 4), 0.25], 48);
    integers.set([sourceBlocks, this.areaRayCount, input.materialRevision, Number(cameraMoving)], 52);
    this.device.queue.writeBuffer(this.parameters, 0, data);
    let uploaded = data.byteLength;
    if (input.enabled) this.batches.forEach((batch, index) => {
      batch.count = batch.scene.instances.length;
      batch.scene.instances.forEach((state, i) => batch.metadata.set([state.entityId, state.entityGeneration ?? 0, 0, 0], i * 4));
      if (batch.count) {
        this.device.queue.writeBuffer(batch.identities, 0, batch.metadata, 0, batch.count * 4);
        uploaded += batch.count * 16;
      }
      this.device.queue.writeBuffer(batch.parameters, 0, new Uint32Array([batch.vertexCount, batch.indexCount / 3,
        batch.count, batch.sourceId, this.areaRayCount, grid.anchorCount, 0, index * grid.anchorCount]));
      uploaded += AREA_PARAMETER_BYTES;
    });
    this.updated = true;
    return uploaded;
  }

  encode(encoder: GPUCommandEncoder, materialRevision: number, giIndex: number,
    timing?: GPUComputePassTimestampWrites): number {
    this.assertAlive();
    if (!this.updated) throw new Error('FINAL_GATHER_NO_UPDATE');
    if (materialRevision !== this.materialRevision || (giIndex !== 0 && giIndex !== 1)) throw new Error('FINAL_GATHER_SNAPSHOT_CHANGED');
    if (this.encodedFrame === this.frame) throw new Error('FINAL_GATHER_FRAME_ALREADY_ENCODED');
    if (this.encoding) throw new Error('FINAL_GATHER_ENCODING_IN_PROGRESS');
    this.encoding = true;
    try {
      if (!this.enabled) {
        encoder.beginComputePass({ label: 'disabled surface final gather', timestampWrites: timing }).end();
        this.encodedFrame = this.frame; return 0;
      }
      const output = 1 - this.outputIndex, grid = this.grid;
      const dispatch = (name: EntryPoint, count: number, group: GPUBindGroup,
        timestampWrites?: GPUComputePassTimestampWrites): void => {
        if (!count) return;
        const pass = encoder.beginComputePass({ label: 'surface final gather ' + name, timestampWrites });
        pass.setPipeline(this.pipelines[name]); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
      };
      dispatch('generateCamera', grid.anchorCount, this.groups.generateCamera,
        timing && { querySet: timing.querySet, beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex });
      let uploaded = this.queries.encodeTrace(encoder, grid.anchorCount, 0, materialRevision);
      encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.cameraHits, 0, grid.anchorCount * WORLD_HIT_BYTES);
      dispatch('generateHemisphere', grid.rayCount, this.groups.generateHemisphere);
      uploaded += this.queries.encodeTrace(encoder, grid.rayCount, 0, materialRevision);
      encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.primaryHits, 0, grid.rayCount * WORLD_HIT_BYTES);
      dispatch('generateShadows', grid.rayCount, this.groups.generateShadows);
      uploaded += this.queries.encodeTrace(encoder, grid.rayCount, 0, materialRevision);
      dispatch('integrate', grid.anchorCount, this.integrateGroups[output][giIndex]);
      if (this.areaRayCount) {
        dispatch('clearArea', this.areaRayCount, this.groups.clearArea);
        for (const batch of this.batches) if (batch.count) dispatch('generateArea', grid.anchorCount, batch.group);
        uploaded += this.queries.encodeTrace(encoder, this.areaRayCount, 0, materialRevision);
        dispatch('integrateArea', grid.anchorCount, this.areaIntegrateGroups[output]);
      }
      dispatch('accumulate', grid.anchorCount, this.accumulateGroups[output]);
      dispatch('denoise', grid.anchorCount, this.denoiseGroups[output],
        timing && { querySet: timing.querySet, endOfPassWriteIndex: timing.endOfPassWriteIndex });
      this.outputIndex = output; this.encodedFrame = this.frame;
      this.historyUpdates = Math.min(this.options.maxHistory, this.historyUpdates + 1);
      return uploaded;
    } finally { this.encoding = false; }
  }

  /** Explicit, bounded diagnostics only. Call after submitting the rendered
   * frame; no per-frame geometry or anchor readback occurs in the runtime. */
  async probe(indices: readonly number[]): Promise<FinalGatherProbeResult> {
    this.assertAlive();
    if (!this.encodedFrame) throw new Error('FINAL_GATHER_NO_FRAME');
    if (!this.enabled) throw new Error('FINAL_GATHER_DISABLED');
    if (this.encoding || this.encodedFrame !== this.frame) throw new Error('FINAL_GATHER_FRAME_NOT_ENCODED');
    if (this.readbackPending) throw new Error('FINAL_GATHER_READBACK_PENDING');
    if (!Array.isArray(indices) || indices.length > MAX_PROBE_ANCHORS) throw new Error('FINAL_GATHER_PROBE_INDEX_BUDGET');
    if (indices.some(index => !Number.isSafeInteger(index) || index < 0 || index >= this.grid.anchorCount)
      || new Set(indices).size !== indices.length) throw new Error('FINAL_GATHER_PROBE_INDICES');
    const frame = this.encodedFrame, requested = [...indices];
    if (!requested.length) return { frame, anchors: [] };
    this.readbackPending = true;
    try {
      const encoder = this.device.createCommandEncoder({ label: 'surface final gather explicit diagnostics' });
      for (let i = 0; i < requested.length; i++) {
        const sourceOffset = requested[i] * FINAL_GATHER_ANCHOR_BYTES;
        const destinationOffset = i * FINAL_GATHER_ANCHOR_BYTES * 2;
        encoder.copyBufferToBuffer(this.anchors[this.outputIndex], sourceOffset, this.readback,
          destinationOffset, FINAL_GATHER_ANCHOR_BYTES);
        encoder.copyBufferToBuffer(this.filteredAnchors, sourceOffset, this.readback,
          destinationOffset + FINAL_GATHER_ANCHOR_BYTES, FINAL_GATHER_ANCHOR_BYTES);
      }
      this.device.queue.submit([encoder.finish()]);
      await this.readback.mapAsync(GPUMapMode.READ, 0, requested.length * FINAL_GATHER_ANCHOR_BYTES * 2);
      this.assertAlive();
      const data = new Float32Array(this.readback.getMappedRange(0, requested.length * FINAL_GATHER_ANCHOR_BYTES * 2));
      const integers = new Uint32Array(data.buffer, data.byteOffset, data.length);
      const vector3 = (base: number): [number, number, number] => [data[base], data[base + 1], data[base + 2]];
      const vector4u = (base: number): [number, number, number, number] =>
        [integers[base], integers[base + 1], integers[base + 2], integers[base + 3]];
      const anchors: FinalGatherProbeResult['anchors'] = requested.map((index, i) => {
        const base = i * 48, filtered = base + 24;
        const position = vector3(base), normal = vector3(base + 4), identity = vector4u(base + 8);
        const versions = vector4u(base + 12), irradiance = vector3(base + 16), historyUpdates = data[base + 19];
        const state = vector4u(base + 20), filteredState = vector4u(filtered + 20);
        const filteredIrradiance = vector3(filtered + 16);
        if (state[1] !== frame || filteredState[1] !== frame || versions[0] !== frame) throw new Error('FINAL_GATHER_FRAME_MISMATCH');
        const metadataMatches = Array.from({ length: 16 }, (_, j) => j).every(j =>
          j < 8 ? data[base + j] === data[filtered + j] : integers[base + j] === integers[filtered + j]);
        if ((state[0] !== 0 && state[0] !== 1) || filteredState.some((value, j) => value !== state[j])
          || !metadataMatches || data[filtered + 19] !== historyUpdates
          || ![...position, ...normal, ...irradiance, ...filteredIrradiance, historyUpdates].every(Number.isFinite)
          || historyUpdates < 1 || historyUpdates > this.options.maxHistory
          || [...irradiance, ...filteredIrradiance].some(value => value < 0 || value > 65504)) {
          throw new Error('FINAL_GATHER_INVALID_ANCHOR');
        }
        return { index, position, normal, identity, versions, irradiance, historyUpdates,
          valid: state[0] === 1, state, filteredIrradiance };
      });
      return { frame, anchors };
    } catch (error) {
      if (this.disposed) throw new Error('FINAL_GATHER_DISPOSED');
      throw error;
    } finally {
      if (this.readback.mapState === 'mapped') this.readback.unmap();
      this.readbackPending = false;
    }
  }

  diagnostics() {
    const anchors = this.enabled ? this.grid.anchorCount : 0;
    const hemisphere = this.enabled ? this.grid.rayCount : 0;
    const area = this.enabled ? this.areaRayCount : 0;
    return { enabled: this.enabled, frame: this.encodedFrame, options: this.options, ownedGpuBytes: this.bytes,
      representation: 'exact-camera-hit-screen-grid-irradiance', update: 'full-grid-every-frame',
      actualSpacing: this.grid.spacing, columns: this.grid.columns, rows: this.grid.rows,
      anchorCapacity: this.anchorCapacity, anchorsPerUpdate: anchors, hemisphereRaysPerUpdate: hemisphere,
      areaRaysPerUpdate: area, areaRayCapacity: this.queries.options.maxRays,
      activeEmitters: this.activeEmitters, registeredEmitters: this.batches.length,
      inactiveEmitterBlockRaysPerUpdate: this.enabled ? (this.sourceBlocks - this.activeEmitters) * anchors : 0,
      queryRaysPerUpdate: anchors + hemisphere * 2 + area,
      queryPassesPerUpdate: this.enabled ? 3 + Number(area > 0) : 0,
      maxWorkUnitsPerUpdate: this.enabled ? this.queries.options.maxTotalWorkUnits * (3 + Number(area > 0)) : 0,
      historyUpdates: this.historyUpdates, nominalHistoryUpdates: this.historyUpdates,
      historyUpdatesMeaning: 'nominal-completed-updates-since-global-reset', historyResetReason: this.historyResetReason,
      temporalAccumulation: 'surface-validated-raw-camera-reprojection', cameraReprojection: true,
      cameraMoving: this.cameraMoving, cameraMotionHistoryCap: Math.min(this.options.maxHistory, 8),
      cameraReprojectionCandidates: 4, cameraReprojectionMaxWorldDistance: 0.10,
      cameraReprojectionPlaneTolerance: 0.001, cameraReprojectionNormalThreshold: 0.98,
      cameraReprojectionInput: 'previous-rendered-surface-position-and-exact-previous-view-projection',
      dynamicOcclusionHistory: 'conservative-global-reset-on-geometry-change',
      diagnosticMaxAnchors: MAX_PROBE_ANCHORS, diagnosticReadbackBytes: READBACK_BYTES,
      diagnosticReadbackPending: this.readbackPending,
      spatialDenoise: 'Gaussian-irradiance-only-with-surface-identity-plane-normal-rejection',
      spatialDenoiseRadius: 2, spatialDenoiseMaxNeighbors: 25, spatialDenoiseMaxWorldDistance: 0.10,
      spatialDenoiseNormalThreshold: 0.98, spatialDenoisePlaneTolerance: 0.001,
      filteredOutputFeedsHistory: false,
      spatialFilter: 'four-anchor-bilinear-plane-normal-distance-rejection', maxSpatialFilterDistance: 0.25,
      uncoveredFallback: 'completed-world-GI', firstSegment: 'full-configured-GI-range',
      normalBias: this.options.normalBias, emissionSampling: 'one-area-sample-per-source-per-anchor-per-update',
      feedbackToWorldField: false, automaticAdaptiveContactAnchors: false };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
