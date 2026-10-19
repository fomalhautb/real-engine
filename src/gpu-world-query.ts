import refitSource from './shaders/query-refit.wgsl?raw';
import traceSource from './shaders/query-trace.wgsl?raw';
import surfaceFrameSource from './shaders/surface-frame.wgsl?raw';
import { buildBvhTopology } from './bvh-topology';
import { buildInstanceBvhTopology, orderInstanceBounds } from './instance-bvh';
import { GpuQuerySceneBvh } from './gpu-query-scene-bvh';
import type { Bounds3 } from './deformation';
import { residentMesh, type RenderBatch, type StreamedSceneBatch } from './contracts';
import { validateMeshColors, type MeshData } from './geometry';
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
  /** Ordering only: GPU refit always uses actual deformed triangle bounds. */
  getInstanceBounds?(): readonly Bounds3[];
}
type RefitLevel = { params: GPUBuffer; bindGroup: GPUBindGroup; count: number; offset: number };
type QueryBatch = {
  source: QuerySource; mesh: MeshData; name: string; capacity: number; identity: string;
  proxy?: QueryProxy; proxyIdentity: string; geometry: WorldHitGeometry;
  topologyVersion: number; nodeCount: number; offsets: number[]; count: number; dirty: boolean;
  emissiveSampling: RenderBatch['emissiveSampling'];
  vertexCount: number; vertices: ArrayBuffer; indices: Uint32Array; uvs?: Float32Array; tangents?: Float32Array;
  colors?: Float32Array; colorOffset: number;
  identities: GPUBuffer; metadata: Uint32Array<ArrayBuffer>; params: GPUBuffer; trace: GPUBindGroup[];
  levels: RefitLevel[];
  data: GPUBuffer; instanceTree: readonly [number, number, number, number]; instanceLevels: RefitLevel[];
  bounds: GPUBuffer; root?: RefitLevel; sortBounds: Bounds3 | null;
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
  private instanceRefitPipeline!: GPUComputePipeline;
  private batchRefitPipeline!: GPUComputePipeline;
  private sceneBvh?: GpuQuerySceneBvh;
  private candidateBuffer!: GPUBuffer;
  private sceneDirty = true;
  private lastReadbackQuery: { frame: number; rays: number; workUnits: number; triangleTests: number;
    maxWorkUnits: number; overflows: number } | null = null;

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
    this.traceUniformStride = Math.ceil(96 / alignment) * alignment;
    this.resetUniformStride = Math.ceil(16 / alignment) * alignment;
    this.refitLayout = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 3 ? 'storage' : 'read-only-storage' },
    })) });
    this.traceLayout = device.createBindGroupLayout({ entries: Array.from({ length: 9 }, (_, binding) => ({
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
    if (this.options.sceneAcceleration === 'bvh') {
      this.sceneBvh = await GpuQuerySceneBvh.create(this.device,
        (label, size, usage, data) => this.buffer(label, size, usage, data), this.rayBuffer, this.hitBuffer,
        this.options.maxRays, this.options.maxTraceParameterSetsPerEncoder);
      this.candidateBuffer = this.sceneBvh.maskBuffer;
    } else this.candidateBuffer = this.buffer('unused world batch candidates', 4, GPUBufferUsage.STORAGE);
    const refitModule = await this.module(refitSource, 'world BVH refit');
    const refitPipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.refitLayout] });
    this.refitPipeline = await this.device.createComputePipelineAsync({ layout: refitPipelineLayout,
      compute: { module: refitModule, entryPoint: 'main' } });
    this.instanceRefitPipeline = await this.device.createComputePipelineAsync({ layout: refitPipelineLayout,
      compute: { module: refitModule, entryPoint: 'refitInstances' } });
    this.batchRefitPipeline = await this.device.createComputePipelineAsync({ layout: refitPipelineLayout,
      compute: { module: refitModule, entryPoint: 'refitBatch' } });
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
    validateMeshColors(mesh.colors, mesh.vertexCount);
    const topology = buildBvhTopology(mesh, { maxTriangles: this.options.maxTrianglesPerMesh });
    const offsets = [topology.nodes.length, topology.nodes.length + topology.triangles.length];
    offsets.push(offsets[1] + mesh.indices.length);
    let length = offsets[2] + mesh.vertexCount * 2;
    const colorOffset = mesh.colors ? length : 0;
    if (mesh.colors) length += mesh.vertexCount * 4;
    const levelOffsets = topology.levels.map(level => { const start = length; length += level.length; return start; });
    const instanceTopology = this.options.instanceAcceleration === 'bvh' && scene.capacity >= 8
      ? buildInstanceBvhTopology(scene.capacity) : undefined;
    const instanceNodesOffset = length;
    length += instanceTopology?.nodes.length ?? 0;
    const instanceOrderOffset = length;
    length += instanceTopology ? scene.capacity : 0;
    const instanceLevelOffsets = instanceTopology?.levels.map(level => { const start = length; length += level.length; return start; }) ?? [];
    const instanceBoundsOffset = topology.nodeCount * scene.capacity;
    const instanceTree = [instanceNodesOffset, instanceOrderOffset, instanceBoundsOffset,
      instanceTopology?.nodeCount ?? 0] as const;
    const rootOffset = instanceBoundsOffset + instanceTree[3];
    const boundsSize = (rootOffset + (this.sceneBvh ? 1 : 0)) * 32;
    const parameterBytes = this.traceUniformStride * this.options.maxTraceParameterSetsPerEncoder;
    this.preflight([length * 4, boundsSize, scene.capacity * 16, parameterBytes,
      ...topology.levels.map(() => 48), ...instanceLevelOffsets.map(() => 48), ...(this.sceneBvh ? [48] : [])]);
    for (const level of topology.levels) {
      if (Math.ceil(level.length * scene.capacity / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('WORLD_QUERY_REFIT_DISPATCH_LIMIT');
    }
    const packed = new Uint32Array(length);
    packed.set(topology.nodes); packed.set(topology.triangles, offsets[0]); packed.set(mesh.indices, offsets[1]);
    if (mesh.uvs) new Float32Array(packed.buffer).set(mesh.uvs, offsets[2]);
    if (mesh.colors) new Float32Array(packed.buffer).set(mesh.colors, colorOffset);
    topology.levels.forEach((level, i) => packed.set(level, levelOffsets[i]));
    if (instanceTopology) {
      packed.set(instanceTopology.nodes, instanceNodesOffset);
      packed.fill(0xffff_ffff, instanceOrderOffset, instanceOrderOffset + scene.capacity);
      instanceTopology.levels.forEach((level, i) => packed.set(level, instanceLevelOffsets[i]));
    }
    const data = this.buffer('world BVH topology and material attributes', packed.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, packed.buffer);
    const bounds = this.buffer('world BVH instance bounds', boundsSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const identities = this.buffer('world entity identities', scene.capacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const params = this.buffer('world batch parameter slots', parameterBytes, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const traceBuffers = [params, source.output, data, bounds, source.instances, this.rayBuffer, this.hitBuffer, identities, this.candidateBuffer];
    const trace = Array.from({ length: this.options.maxTraceParameterSetsPerEncoder }, (_, slot) =>
      this.device.createBindGroup({ layout: this.traceLayout,
      entries: traceBuffers.map((buffer, binding) => ({ binding, resource: binding === 0
        ? { buffer, offset: slot * this.traceUniformStride, size: 96 } : { buffer } })) }));
    const createLevels = (input: readonly Uint32Array[], starts: readonly number[]): RefitLevel[] => input.map((level, i) => {
      const params = this.buffer('world refit depth parameters', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      return { params, count: level.length, offset: starts[i],
        bindGroup: this.device.createBindGroup({ layout: this.refitLayout,
          entries: [params, source.output, data, bounds].map((buffer, binding) => ({ binding, resource: { buffer } })) }) };
    });
    const levels = createLevels(topology.levels, levelOffsets);
    const instanceLevels = createLevels(instanceTopology?.levels ?? [], instanceLevelOffsets);
    const root = this.sceneBvh ? createLevels([new Uint32Array([0])], [rootOffset])[0] : undefined;
    this.batches.push({ source, mesh, name: scene.name, capacity: scene.capacity, identity: sourceIdentity,
      proxy, proxyIdentity: proxyIdentity(proxy), geometry: geometryMetadata(proxy),
      emissiveSampling: scene.emissiveSampling,
      topologyVersion: scene.queryIdentity?.topologyVersion ?? 0, nodeCount: topology.nodeCount, offsets, count: 0, dirty: true,
      vertexCount: mesh.vertexCount, vertices: mesh.vertices, indices: mesh.indices, uvs: mesh.uvs, tangents: mesh.tangents,
      colors: mesh.colors, colorOffset,
      identities, metadata: new Uint32Array(scene.capacity * 4), params, trace, levels, data, instanceTree, instanceLevels,
      bounds, root, sortBounds: null });
  }
  private validateTopology(): void {
    if (this.disposed) throw new Error('WORLD_QUERY_DISPOSED');
    for (const batch of this.batches) {
      const scene = batch.source.scene;
      const mesh = residentMesh(scene), proxy = queryProxy(scene);
      if (mesh !== batch.mesh || mesh.vertexCount !== batch.vertexCount || mesh.vertices !== batch.vertices
        || mesh.indices !== batch.indices || mesh.uvs !== batch.uvs || mesh.tangents !== batch.tangents || mesh.colors !== batch.colors || scene.capacity !== batch.capacity
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
    const firstSnapshot = this.frame === 0;
    this.frame = u32(frame, 'FRAME');
    let uploadedBytes = 0;
    let historyChanged = false;
    this.sceneDirty = false;
    for (const batch of this.batches) {
      const countChanged = batch.count !== batch.source.scene.instances.length;
      batch.dirty = firstSnapshot || countChanged || (batch.source.getDirty?.() ?? true);
      historyChanged ||= countChanged;
      batch.count = batch.source.scene.instances.length;
      if (!batch.dirty) continue;
      if (this.sceneEnabled) {
        this.sceneDirty = true;
        this.device.queue.writeBuffer(batch.root!.params, 0, new Uint32Array([
          batch.nodeCount, 0, batch.count, batch.root!.offset,
          batch.instanceTree[2], batch.instanceTree[3], 0, 0, 0, 0, 0, 0,
        ]));
        uploadedBytes += 48;
      }
      if (!batch.count) { batch.sortBounds = null; continue; }
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
      if (batch.instanceTree[3] || this.sceneEnabled) {
        // CPU centers only determine traversal quality. Bounds used for culling
        // come from GPU BLAS roots, including skin, Morph and wave deformation.
        const sortBounds = batch.source.getInstanceBounds?.() ?? batch.source.scene.instances.map(state => {
          const center: [number, number, number] = [state.model[12], state.model[13], state.model[14]];
          return { min: center, max: center };
        });
        if (sortBounds.length !== batch.count) throw new Error('WORLD_QUERY_INSTANCE_BOUNDS_COUNT');
        if (this.sceneEnabled) {
          const min: [number, number, number] = [Infinity, Infinity, Infinity];
          const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
          for (const bounds of sortBounds) for (let axis = 0; axis < 3; axis++) {
            const low = bounds.min[axis], high = bounds.max[axis];
            if (!Number.isFinite(Math.fround(low)) || !Number.isFinite(Math.fround(high)) || low > high) {
              throw new Error('WORLD_QUERY_INSTANCE_BOUNDS');
            }
            min[axis] = Math.min(min[axis], low); max[axis] = Math.max(max[axis], high);
          }
          batch.sortBounds = { min, max };
        }
        if (!batch.instanceTree[3]) continue;
        const order = orderInstanceBounds(sortBounds, batch.capacity);
        this.device.queue.writeBuffer(batch.data, batch.instanceTree[1] * 4, order);
        uploadedBytes += order.byteLength + batch.instanceLevels.length * 48;
        for (const level of batch.instanceLevels) this.device.queue.writeBuffer(level.params, 0, new Uint32Array([
          batch.nodeCount, 0, batch.count, 0,
          ...batch.instanceTree.slice(0, 3), level.offset, level.count, 0, 0, 0,
        ]));
      }
    }
    if (this.sceneDirty) uploadedBytes += this.sceneBvh!.updateOrder(this.batches.map(batch => batch.sortBounds));
    if (historyChanged) this.sceneHistoryRevision = (this.sceneHistoryRevision + 1) >>> 0;
    return uploadedBytes;
  }
  /** Discrete identity/history changes; ordinary current/previous pose motion does not advance it. */
  get historyRevision(): number { return this.sceneHistoryRevision; }
  private get sceneEnabled(): boolean { return !!this.sceneBvh && this.batches.length >= 8; }
  encodeRefit(encoder: GPUCommandEncoder, timing?: GPUComputePassTimestampWrites): void {
    let remaining = this.batches.reduce((sum, batch) => sum + (batch.dirty
      ? (batch.count ? batch.levels.length + batch.instanceLevels.length : 0) + (this.sceneEnabled ? 1 : 0) : 0),
    this.sceneDirty ? this.sceneBvh!.levelCount : 0);
    let first = true;
    if (!remaining && timing) {
      encoder.beginComputePass({ label: 'cached world BVH refit', timestampWrites: timing }).end();
    }
    const timestampForPass = (): GPUComputePassTimestampWrites | undefined => {
      const writes = timing && (first || remaining === 1) ? {
        querySet: timing.querySet,
        beginningOfPassWriteIndex: first ? timing.beginningOfPassWriteIndex : undefined,
        endOfPassWriteIndex: remaining === 1 ? timing.endOfPassWriteIndex : undefined,
      } : undefined;
      first = false; remaining--;
      return writes;
    };
    for (const [index, batch] of this.batches.entries()) {
      if (!batch.dirty) continue;
      if (batch.count) for (const instanceLevel of [false, true]) {
        const levels = instanceLevel ? batch.instanceLevels : batch.levels;
        for (let depth = levels.length - 1; depth >= 0; depth--) {
          const level = levels[depth];
          const timestampWrites = timestampForPass();
          const pass = encoder.beginComputePass({ label: `world ${instanceLevel ? 'instance' : 'triangle'} BVH refit ${batch.name} depth ${depth}`, timestampWrites });
          pass.setPipeline(instanceLevel ? this.instanceRefitPipeline : this.refitPipeline); pass.setBindGroup(0, level.bindGroup);
          pass.dispatchWorkgroups(Math.ceil(level.count * (instanceLevel ? 1 : batch.count) / 64)); pass.end();
        }
      }
      if (this.sceneEnabled) {
        const pass = encoder.beginComputePass({ label: `world batch root ${batch.name}`, timestampWrites: timestampForPass() });
        pass.setPipeline(this.batchRefitPipeline); pass.setBindGroup(0, batch.root!.bindGroup);
        pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(batch.bounds, batch.root!.offset * 32, this.sceneBvh!.boundsBuffer, index * 32, 32);
      }
    }
    if (this.sceneDirty) this.sceneBvh!.encodeRefit(encoder, timestampForPass);
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
    if (this.sceneEnabled) {
      uploadedBytes += this.sceneBvh!.encodeCandidates(encoder, parameterSlot, count, workPerRay, upload);
    }
    for (let index = 0; index < this.batches.length; index++) {
      const batch = this.batches[index];
      if (!batch.count) continue;
      if (upload) {
        const params = new ArrayBuffer(96);
        new Uint32Array(params).set([batch.nodeCount, batch.mesh.vertexCount, batch.count, index + 1,
          ...batch.offsets, batch.colorOffset, count, workPerRay, this.frame, batch.topologyVersion, materialRevision,
          batch.emissiveSampling === 'triangle-area' ? 1 : 0, this.sceneEnabled ? 1 : 0]);
        new Float32Array(params)[16] = textureLod;
        new Uint32Array(params).set(batch.instanceTree, 20);
        this.device.queue.writeBuffer(batch.params, parameterSlot * this.traceUniformStride, params);
        uploadedBytes += 96;
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
    if (!count) {
      this.lastReadbackQuery = { frame, rays: 0, workUnits: 0, triangleTests: 0, maxWorkUnits: 0, overflows: 0 };
      return { frame, materialRevision, hits: [] };
    }
    this.pending = true;
    try {
      this.device.queue.writeBuffer(this.rayBuffer, 0, packed);
      const encoder = this.device.createCommandEncoder({ label: 'world query diagnostics' });
      this.encodeTrace(encoder, count, textureLod, materialRevision);
      encoder.copyBufferToBuffer(this.hitBuffer, 0, this.readBuffer, 0, count * WORLD_HIT_BYTES);
      this.device.queue.submit([encoder.finish()]);
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      try {
        const data = this.readBuffer.getMappedRange().slice(0, count * WORLD_HIT_BYTES);
        const view = new DataView(data);
        const work = { frame, rays: count, workUnits: 0, triangleTests: 0, maxWorkUnits: 0, overflows: 0 };
        for (let ray = 0; ray < count; ray++) {
          const offset = ray * WORLD_HIT_BYTES + 144, units = view.getUint32(offset, true);
          work.workUnits += units; work.triangleTests += view.getUint32(offset + 4, true);
          work.maxWorkUnits = Math.max(work.maxWorkUnits, units);
          work.overflows += view.getUint32(offset + 8, true) ? 1 : 0;
        }
        this.lastReadbackQuery = work;
        return decodeWorldHits(data, names, count, frame, materialRevision, this.batches.map(batch => batch.geometry));
      }
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
      instanceAcceleration: this.batches.some(batch => batch.instanceTree[3]) ? 'batch-bvh' : 'linear-root-scan',
      instanceBvhNodes: this.batches.reduce((sum, batch) => sum + batch.instanceTree[3], 0),
      acceleratedBatches: this.batches.filter(batch => batch.instanceTree[3]).length,
      sceneAcceleration: this.sceneEnabled ? 'batch-bvh' : 'linear-batches',
      sceneBvhNodes: this.sceneEnabled ? this.sceneBvh!.nodeCount : 0,
      sceneCandidateBytes: this.candidateBuffer.size,
      lastReadbackQuery: this.lastReadbackQuery && { ...this.lastReadbackQuery },
      refit: 'changed-batches', refitTiming: 'provided-by-caller', externalTraceTiming: false };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
