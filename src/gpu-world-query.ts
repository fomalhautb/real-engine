import refitSource from './shaders/query-refit.wgsl?raw';
import traceSource from './shaders/query-trace.wgsl?raw';
import surfaceFrameSource from './shaders/surface-frame.wgsl?raw';
import { buildBvhTopology } from './bvh-topology';
import { residentMesh, type RenderBatch, type StreamedSceneBatch } from './contracts';
import type { MeshData } from './geometry';
import type { GpuMaterial } from './gpu-material';
import type { MaterialProgram } from './material';
import { decodeWorldHits, packWorldRays, resolveWorldQueryOptions, WORLD_HIT_BYTES, WORLD_RAY_BYTES,
  type WorldHitGeometry, type WorldQueryOptions, type WorldQueryResult, type WorldRay } from './world-query';

type QueryProxy = StreamedSceneBatch['geometryStream']['queryProxy'];

interface QuerySource {
  scene: RenderBatch;
  output: GPUBuffer;
  instances: GPUBuffer;
  getMaterial(): GpuMaterial;
  getPipeline(): GPUComputePipeline;
  getDirty?(): boolean;
}
type QueryBatch = {
  source: QuerySource; mesh: MeshData; name: string; capacity: number; identity: string;
  proxy?: QueryProxy; proxyIdentity: string; geometry: WorldHitGeometry;
  topologyVersion: number; nodeCount: number; offsets: number[]; count: number; dirty: boolean;
  emissiveSampling: RenderBatch['emissiveSampling'];
  vertexCount: number; vertices: ArrayBuffer; indices: Uint32Array; uvs?: Float32Array; tangents?: Float32Array;
  identities: GPUBuffer; metadata: Uint32Array<ArrayBuffer>; params: GPUBuffer; trace: GPUBindGroup[];
  levels: { params: GPUBuffer; bindGroup: GPUBindGroup; count: number; offset: number }[];
};

const resetSource = `
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
  identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
  previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
  uvInfo: vec4f, work: vec4u,
}
@group(0) @binding(0) var<uniform> params: vec4u;
@group(0) @binding(1) var<storage, read> rays: array<Ray>;
@group(0) @binding(2) var<storage, read_write> hits: array<Hit>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.x) { return; }
  var hit: Hit;
  hit.metric.x = rays[id.x].direction.w;
  hit.versions = vec4u(params.y, 0u, 0u, params.z);
  hits[id.x] = hit;
}`;

function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) throw new Error('WORLD_QUERY_' + label);
  return value;
}
function identity(scene: RenderBatch): string {
  const source = scene.queryIdentity;
  if (!source) return '';
  if (typeof source.assetId !== 'string' || !source.assetId.length || source.assetId.length > 1024) throw new Error('WORLD_QUERY_ASSET_ID');
  return JSON.stringify([source.assetId, u32(source.primitiveId, 'PRIMITIVE_ID'), u32(source.topologyVersion, 'TOPOLOGY_VERSION')]);
}
function queryProxy(scene: RenderBatch): QueryProxy | undefined {
  return scene.mesh ? undefined : scene.geometryStream.queryProxy;
}
function proxyIdentity(proxy: QueryProxy | undefined): string {
  return proxy ? JSON.stringify([proxy.version, proxy.primitiveKey, proxy.directorySha256,
    proxy.topologyVersion, proxy.requestedError, proxy.approximateObjectSpaceError]) : '';
}
function geometryMetadata(proxy: QueryProxy | undefined): WorldHitGeometry {
  return proxy ? { kind: 'fixed-query-proxy', primitiveKey: proxy.primitiveKey,
    directorySha256: proxy.directorySha256, topologyVersion: proxy.topologyVersion,
    approximateObjectSpaceError: proxy.approximateObjectSpaceError,
    errorMetric: 'accumulated-simplification' } : { kind: 'source' };
}

/** Camera-independent, refitted triangle queries over the renderer's shared
 * world-space deformation stream. Buffers and GPU encoding can be consumed by
 * lighting without CPU geometry readback; traceRays is a diagnostic adapter. */
export class GpuWorldQuery {
  readonly options: Required<WorldQueryOptions>;
  readonly rayBuffer!: GPUBuffer;
  readonly hitBuffer!: GPUBuffer;
  private readonly buffers: GPUBuffer[] = [];
  private readonly batches: QueryBatch[] = [];
  private bytes = 0;
  private frame = 0;
  private sceneHistoryRevision = 0;
  private disposed = false;
  private pending = false;
  /** A recorded dispatch never observes a later parameter overwrite. The same
   * frame/material snapshot may use bounded distinct count/LOD parameter sets. */
  private readonly traceParameters = new WeakMap<GPUCommandEncoder, {
    snapshot: string; slots: Map<string, number>;
  }>();
  private readonly traceUniformStride: number;
  private readonly resetUniformStride: number;
  private readonly refitLayout: GPUBindGroupLayout;
  private readonly traceLayout: GPUBindGroupLayout;
  private resetParams!: GPUBuffer;
  private readBuffer!: GPUBuffer;
  private resetPipeline!: GPUComputePipeline;
  private resetGroups!: GPUBindGroup[];
  private refitPipeline!: GPUComputePipeline;

  static async create(device: GPUDevice, materialLayout: GPUBindGroupLayout,
    account: (bytes: number) => void, options: WorldQueryOptions): Promise<GpuWorldQuery> {
    const result = new GpuWorldQuery(device, materialLayout, account, options);
    try { await result.initialize(); return result; }
    catch (error) { result.dispose(); throw error; }
  }
  private constructor(private readonly device: GPUDevice, private readonly materialLayout: GPUBindGroupLayout,
    private readonly account: (bytes: number) => void, options: WorldQueryOptions) {
    this.options = Object.freeze(resolveWorldQueryOptions(options));
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    this.traceUniformStride = Math.ceil(80 / alignment) * alignment;
    this.resetUniformStride = Math.ceil(16 / alignment) * alignment;
    this.refitLayout = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 3 ? 'storage' : 'read-only-storage' },
    })) });
    this.traceLayout = device.createBindGroupLayout({ entries: Array.from({ length: 8 }, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 6 ? 'storage' : 'read-only-storage' },
    })) });
  }
  private preflight(sizes: readonly number[]): void {
    if (sizes.some(size => !Number.isSafeInteger(size) || size < 4 || size > this.device.limits.maxStorageBufferBindingSize
      || size > this.device.limits.maxBufferSize)) throw new Error('WORLD_QUERY_BUFFER_LIMIT');
    if (this.bytes + sizes.reduce((sum, size) => sum + size, 0) > this.options.maxBytes) throw new Error('WORLD_QUERY_MEMORY_BUDGET');
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer {
    this.preflight([size]);
    const buffer = this.device.createBuffer({ label, size, usage, mappedAtCreation: !!data });
    // Register before initialization so an exception is still cleaned up.
    this.buffers.push(buffer); this.bytes += size; this.account(size);
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data)); buffer.unmap(); }
    return buffer;
  }
  private async module(code: string, label: string): Promise<GPUShaderModule> {
    const module = this.device.createShaderModule({ code, label });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `${label}:${error.lineNum}: ${error.message}`).join('\n'));
    return module;
  }
  private async initialize(): Promise<void> {
    const sizes = [this.options.maxRays * WORLD_RAY_BYTES, this.options.maxRays * WORLD_HIT_BYTES,
      this.options.maxRays * WORLD_HIT_BYTES,
      this.resetUniformStride * this.options.maxTraceParameterSetsPerEncoder];
    this.preflight(sizes);
    // readonly public references; ownership remains entirely with this module.
    Object.assign(this, {
      rayBuffer: this.buffer('world rays', sizes[0], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      hitBuffer: this.buffer('world hits', sizes[1], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
    });
    this.readBuffer = this.buffer('world hit diagnostics', sizes[2], GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.resetParams = this.buffer('world query reset parameter slots', sizes[3], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.refitPipeline = await this.device.createComputePipelineAsync({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.refitLayout] }),
      compute: { module: await this.module(refitSource, 'world BVH refit'), entryPoint: 'main' },
    });
    this.resetPipeline = await this.device.createComputePipelineAsync({ layout: 'auto',
      compute: { module: await this.module(resetSource, 'world ray reset'), entryPoint: 'main' } });
    this.resetGroups = Array.from({ length: this.options.maxTraceParameterSetsPerEncoder }, (_, slot) =>
      this.device.createBindGroup({ layout: this.resetPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.resetParams, offset: slot * this.resetUniformStride, size: 16 } },
      { binding: 1, resource: { buffer: this.rayBuffer } },
      { binding: 2, resource: { buffer: this.hitBuffer } },
    ] }));
  }
  async createPipeline(program: MaterialProgram): Promise<GPUComputePipeline> {
    if (this.disposed) throw new Error('WORLD_QUERY_DISPOSED');
    const code = surfaceFrameSource + '\n' + traceSource.replace('// MATERIAL_PROGRAM',
      `const MATERIAL_DOUBLE_SIDED = ${program.doubleSided};\nconst MATERIAL_UNLIT = ${program.shading === 'unlit'};\nconst MATERIAL_NORMAL_MAPPED = ${program.normalMapped};\n${program.wgsl}`);
    return this.device.createComputePipelineAsync({ label: 'shared material world hits',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.traceLayout, this.materialLayout] }),
      compute: { module: await this.module(code, 'world material query'), entryPoint: 'main' } });
  }
  addBatch(source: QuerySource): void {
    if (this.disposed || this.frame) throw new Error('WORLD_QUERY_REGISTRATION_CLOSED');
    const { scene } = source, mesh = residentMesh(scene), proxy = queryProxy(scene);
    if (scene.emissiveSampling !== undefined && scene.emissiveSampling !== 'triangle-area') throw new Error('WORLD_QUERY_EMISSIVE_SAMPLING');
    if (this.batches.length >= 256) throw new Error('WORLD_QUERY_BATCH_BUDGET');
    if (!Number.isSafeInteger(scene.capacity) || scene.capacity < 1 || scene.capacity > this.options.maxInstancesPerBatch) throw new Error('WORLD_QUERY_INSTANCE_BUDGET');
    const sourceIdentity = identity(scene);
    const topology = buildBvhTopology(mesh, { maxTriangles: this.options.maxTrianglesPerMesh });
    const offsets = [topology.nodes.length, topology.nodes.length + topology.triangles.length];
    offsets.push(offsets[1] + mesh.indices.length);
    let length = offsets[2] + mesh.vertexCount * 2;
    const levelOffsets = topology.levels.map(level => { const start = length; length += level.length; return start; });
    const boundsSize = topology.nodeCount * scene.capacity * 32;
    const parameterBytes = this.traceUniformStride * this.options.maxTraceParameterSetsPerEncoder;
    this.preflight([length * 4, boundsSize, scene.capacity * 16, parameterBytes, ...topology.levels.map(() => 48)]);
    for (const level of topology.levels) {
      if (Math.ceil(level.length * scene.capacity / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('WORLD_QUERY_REFIT_DISPATCH_LIMIT');
    }
    const packed = new Uint32Array(length);
    packed.set(topology.nodes); packed.set(topology.triangles, offsets[0]); packed.set(mesh.indices, offsets[1]);
    if (mesh.uvs) new Float32Array(packed.buffer).set(mesh.uvs, offsets[2]);
    topology.levels.forEach((level, i) => packed.set(level, levelOffsets[i]));
    const data = this.buffer('world BVH topology and UVs', packed.byteLength, GPUBufferUsage.STORAGE, packed.buffer);
    const bounds = this.buffer('world BVH instance bounds', boundsSize, GPUBufferUsage.STORAGE);
    const identities = this.buffer('world entity identities', scene.capacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const params = this.buffer('world batch parameter slots', parameterBytes, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const traceBuffers = [params, source.output, data, bounds, source.instances, this.rayBuffer, this.hitBuffer, identities];
    const trace = Array.from({ length: this.options.maxTraceParameterSetsPerEncoder }, (_, slot) =>
      this.device.createBindGroup({ layout: this.traceLayout,
      entries: traceBuffers.map((buffer, binding) => ({ binding, resource: binding === 0
        ? { buffer, offset: slot * this.traceUniformStride, size: 80 } : { buffer } })) }));
    const levels = topology.levels.map((level, i) => {
      const params = this.buffer('world refit depth parameters', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      return { params, count: level.length, offset: levelOffsets[i],
        bindGroup: this.device.createBindGroup({ layout: this.refitLayout,
          entries: [params, source.output, data, bounds].map((buffer, binding) => ({ binding, resource: { buffer } })) }) };
    });
    this.batches.push({ source, mesh, name: scene.name, capacity: scene.capacity, identity: sourceIdentity,
      proxy, proxyIdentity: proxyIdentity(proxy), geometry: geometryMetadata(proxy),
      emissiveSampling: scene.emissiveSampling,
      topologyVersion: scene.queryIdentity?.topologyVersion ?? 0, nodeCount: topology.nodeCount, offsets, count: 0, dirty: true,
      vertexCount: mesh.vertexCount, vertices: mesh.vertices, indices: mesh.indices, uvs: mesh.uvs, tangents: mesh.tangents,
      identities, metadata: new Uint32Array(scene.capacity * 4), params, trace, levels });
  }
  private validateTopology(): void {
    if (this.disposed) throw new Error('WORLD_QUERY_DISPOSED');
    for (const batch of this.batches) {
      const scene = batch.source.scene;
      const mesh = residentMesh(scene), proxy = queryProxy(scene);
      if (mesh !== batch.mesh || mesh.vertexCount !== batch.vertexCount || mesh.vertices !== batch.vertices
        || mesh.indices !== batch.indices || mesh.uvs !== batch.uvs || mesh.tangents !== batch.tangents || scene.capacity !== batch.capacity
        || proxy !== batch.proxy || proxyIdentity(proxy) !== batch.proxyIdentity
        || scene.name !== batch.name || identity(scene) !== batch.identity || scene.emissiveSampling !== batch.emissiveSampling) {
        throw new Error('WORLD_QUERY_TOPOLOGY_CHANGED: recreate the renderer for a new immutable mesh/topology');
      }
    }
  }
  validateSources(): void {
    this.validateTopology();
    for (const batch of this.batches) {
      const scene = batch.source.scene;
      if (scene.instances.length > batch.capacity) throw new Error('WORLD_QUERY_INSTANCE_BUDGET');
      const alive = new Set<number>();
      for (const state of scene.instances) {
        u32(state.entityId, 'ENTITY_ID'); u32(state.entityGeneration ?? 0, 'GENERATION'); u32(state.historyVersion, 'HISTORY_VERSION');
        if (alive.has(state.entityId)) throw new Error('WORLD_QUERY_DUPLICATE_ENTITY');
        alive.add(state.entityId);
      }
    }
  }
  updateSnapshot(frame: number, _materialRevision: number): number {
    this.frame = u32(frame, 'FRAME');
    let uploadedBytes = 0;
    let historyChanged = false;
    for (const batch of this.batches) {
      batch.dirty = batch.source.getDirty?.() ?? true;
      historyChanged ||= batch.count !== batch.source.scene.instances.length;
      batch.count = batch.source.scene.instances.length;
      if (!batch.dirty || !batch.count) continue;
      batch.source.scene.instances.forEach((state, i) => {
        const offset = i * 4, generation = state.entityGeneration ?? 0;
        historyChanged ||= batch.metadata[offset] !== state.entityId || batch.metadata[offset + 1] !== generation
          || batch.metadata[offset + 2] !== state.historyVersion;
        batch.metadata.set([state.entityId, generation, state.historyVersion, 0], offset);
      });
      this.device.queue.writeBuffer(batch.identities, 0, batch.metadata, 0, batch.count * 4);
      uploadedBytes += batch.count * 16 + batch.levels.length * 48;
      for (const level of batch.levels) this.device.queue.writeBuffer(level.params, 0, new Uint32Array([
        batch.nodeCount, batch.mesh.vertexCount, batch.count, 0,
        ...batch.offsets, level.offset, level.count, 0, 0, 0,
      ]));
    }
    if (historyChanged) this.sceneHistoryRevision = (this.sceneHistoryRevision + 1) >>> 0;
    return uploadedBytes;
  }
  /** Discrete identity/history changes; ordinary current/previous pose motion does not advance it. */
  get historyRevision(): number { return this.sceneHistoryRevision; }
  encodeRefit(encoder: GPUCommandEncoder, timing?: GPUComputePassTimestampWrites): void {
    let remaining = this.batches.reduce((sum, batch) => sum + (batch.count && batch.dirty ? batch.levels.length : 0), 0);
    let first = true;
    if (!remaining && timing) {
      encoder.beginComputePass({ label: 'cached world BVH refit', timestampWrites: timing }).end();
    }
    for (const batch of this.batches) {
      if (!batch.count || !batch.dirty) continue;
      for (let depth = batch.levels.length - 1; depth >= 0; depth--) {
        const level = batch.levels[depth];
        const timestampWrites = timing && (first || remaining === 1) ? {
          querySet: timing.querySet,
          beginningOfPassWriteIndex: first ? timing.beginningOfPassWriteIndex : undefined,
          endOfPassWriteIndex: remaining === 1 ? timing.endOfPassWriteIndex : undefined,
        } : undefined;
        const pass = encoder.beginComputePass({ label: `world BVH refit ${batch.name} depth ${depth}`, timestampWrites });
        pass.setPipeline(this.refitPipeline); pass.setBindGroup(0, level.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(level.count * batch.count / 64)); pass.end();
        first = false; remaining--;
      }
    }
  }
  /** Encode GPU-resident rays -> hits. One encoder may vary ray count and texture
   * LOD while retaining the same frame, material and geometry snapshot. Every
   * distinct parameter set gets a slot; identical sets reuse their slot. Rewrite
   * rays with encoded GPU operations, not queue writes that alias recorded work.
   * Submit it before encoding another encoder with different parameters: GPU
   * snapshots and uniform buffers assume serial encode/submit. Callers own ray
   * generation and inspect hit.work.z for an explicit overflow. Returns the
   * number of uniform bytes actually queued for upload. */
  encodeTrace(encoder: GPUCommandEncoder, count: number, textureLod: number, materialRevision: number): number {
    if (this.disposed || !this.frame) throw new Error('WORLD_QUERY_NO_RENDERED_FRAME');
    this.validateTopology();
    if (!Number.isSafeInteger(count) || count < 0 || count > this.options.maxRays) throw new Error('WORLD_QUERY_RAY_BUDGET');
    if (!Number.isFinite(textureLod) || textureLod < 0 || textureLod > 32) throw new Error('WORLD_QUERY_TEXTURE_LOD');
    u32(materialRevision, 'MATERIAL_REVISION');
    if (!count) return 0;
    const workPerRay = Math.min(this.options.maxNodeVisits, Math.floor(this.options.maxTotalWorkUnits / count));
    if (workPerRay < 1) throw new Error('WORLD_QUERY_WORK_BUDGET: total budget cannot cover every ray');
    const snapshot = JSON.stringify([this.frame, materialRevision,
      this.batches.map((batch, index) => [index + 1, batch.nodeCount, batch.mesh.vertexCount, batch.count,
        ...batch.offsets, batch.topologyVersion])]);
    let state = this.traceParameters.get(encoder);
    if (state && state.snapshot !== snapshot) throw new Error('WORLD_QUERY_ENCODER_SNAPSHOT_CHANGED');
    const key = JSON.stringify([count, Math.fround(textureLod), workPerRay]);
    let slot = state?.slots.get(key);
    const upload = slot === undefined;
    if (upload) {
      const next = state?.slots.size ?? 0;
      if (next >= this.options.maxTraceParameterSetsPerEncoder) throw new Error('WORLD_QUERY_ENCODER_PARAMETER_BUDGET');
      if (!state) { state = { snapshot, slots: new Map() }; this.traceParameters.set(encoder, state); }
      slot = next;
      state.slots.set(key, slot);
    }
    const parameterSlot = slot!;
    let uploadedBytes = 0;
    if (upload) {
      this.device.queue.writeBuffer(this.resetParams, parameterSlot * this.resetUniformStride,
        new Uint32Array([count, this.frame, materialRevision, 0]));
      uploadedBytes += 16;
    }
    const reset = encoder.beginComputePass({ label: 'world ray reset' });
    reset.setPipeline(this.resetPipeline); reset.setBindGroup(0, this.resetGroups[parameterSlot]);
    reset.dispatchWorkgroups(Math.ceil(count / 64)); reset.end();
    for (let index = 0; index < this.batches.length; index++) {
      const batch = this.batches[index];
      if (!batch.count) continue;
      if (upload) {
        const params = new ArrayBuffer(80);
        new Uint32Array(params).set([batch.nodeCount, batch.mesh.vertexCount, batch.count, index + 1,
          ...batch.offsets, 0, count, workPerRay, this.frame, batch.topologyVersion, materialRevision,
          batch.emissiveSampling === 'triangle-area' ? 1 : 0]);
        new Float32Array(params)[16] = textureLod;
        this.device.queue.writeBuffer(batch.params, parameterSlot * this.traceUniformStride, params);
        uploadedBytes += 80;
      }
      // Distinct passes provide ordering for closest-hit updates between batches.
      const pass = encoder.beginComputePass({ label: 'world rays ' + batch.name });
      pass.setPipeline(batch.source.getPipeline()); pass.setBindGroup(0, batch.trace[parameterSlot]);
      pass.setBindGroup(1, batch.source.getMaterial().bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    }
    return uploadedBytes;
  }
  async traceRays(rays: readonly WorldRay[], options: { textureLod?: number } = {}, materialRevision = 0): Promise<WorldQueryResult> {
    if (this.disposed || !this.frame) throw new Error('WORLD_QUERY_NO_RENDERED_FRAME');
    this.validateTopology();
    if (this.pending) throw new Error('WORLD_QUERY_READBACK_PENDING');
    const packed = packWorldRays(rays, this.options.maxRays), count = rays.length, textureLod = options.textureLod ?? 0;
    if (!Number.isFinite(textureLod) || textureLod < 0 || textureLod > 32) throw new Error('WORLD_QUERY_TEXTURE_LOD');
    const frame = this.frame, names = this.batches.map(batch => batch.name);
    if (!count) return { frame, materialRevision, hits: [] };
    this.pending = true;
    try {
      this.device.queue.writeBuffer(this.rayBuffer, 0, packed);
      const encoder = this.device.createCommandEncoder({ label: 'world query diagnostics' });
      this.encodeTrace(encoder, count, textureLod, materialRevision);
      encoder.copyBufferToBuffer(this.hitBuffer, 0, this.readBuffer, 0, count * WORLD_HIT_BYTES);
      this.device.queue.submit([encoder.finish()]);
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      try { return decodeWorldHits(this.readBuffer.getMappedRange().slice(0, count * WORLD_HIT_BYTES), names,
        count, frame, materialRevision, this.batches.map(batch => batch.geometry)); }
      finally { this.readBuffer.unmap(); }
    } finally { this.pending = false; }
  }
  diagnostics() {
    return { frame: this.frame, batches: this.batches.length, ownedGpuBytes: this.bytes,
      topologyNodes: this.batches.reduce((sum, batch) => sum + batch.nodeCount, 0),
      geometryRepresentations: this.batches.map(batch => ({ batch: batch.name, geometry: { ...batch.geometry } })),
      instances: this.batches.reduce((sum, batch) => sum + batch.count, 0), options: { ...this.options },
      traceParameters: 'aligned-immutable-slots-per-encoder', traceUniformStride: this.traceUniformStride,
      resetUniformStride: this.resetUniformStride,
      instanceAcceleration: 'linear-root-scan', refit: 'changed-batches', refitTiming: 'provided-by-caller', externalTraceTiming: false };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
