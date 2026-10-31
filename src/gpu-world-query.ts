import refitSource from './shaders/query-refit.wgsl?raw';
import traceSource from './shaders/query-trace.wgsl?raw';
import surfaceFrameSource from './shaders/surface-frame.wgsl?raw';
import { buildBvhTopology } from './bvh-topology';
import { buildInstanceBvhTopology, orderInstanceBounds } from './instance-bvh';
import { GpuQuerySceneBvh } from './gpu-query-scene-bvh';
import { GpuQuerySharedStatic, GpuQuerySharedStaticPipelines } from './gpu-query-shared-static';
import { orderSharedLeaves, packLeafOrderedTriangles, packSharedFlags, packSharedLeaves, planSharedStaticPool,
  SHARED_FLAG_MEMBER, type SharedStaticExclusion, type SharedStaticPlan } from './query-shared-static';
import type { Bounds3 } from './deformation';
import { residentMesh, type RenderBatch, type StreamedSceneBatch } from './contracts';
import { validateMeshColors, type MeshData } from './geometry';
import type { GpuMaterial } from './gpu-material';
import type { MaterialProgram } from './material';
import { timestampSpanEnd, timestampSpanStart } from './gpu-timestamp-span';
import { decodeWorldHits, packWorldRays, resolveWorldQueryOptions, WORLD_HIT_BYTES, WORLD_RAY_BYTES,
  type WorldHitGeometry, type WorldQueryOptions, type WorldQueryResult, type WorldRay } from './world-query';

type QueryProxy = StreamedSceneBatch['geometryStream']['queryProxy'];

/** Support captures may reverse single-sided eligibility without changing the
 * material's coverage or the authored face policy used by transport queries. */
export type WorldQueryFacePolicy = 'material' | 'reverse-material-support';
export const WORLD_QUERY_TRACE_PARAMETER_BYTES = 112;

/** Every source-major block contains a primary prefix and a supplemental suffix.
 * The primary budget remains the encodeTrace maxWorkUnits argument. */
export interface WorldQueryWorkPartition {
  readonly raysPerBlock: number;
  readonly primaryRaysPerBlock: number;
  readonly supplementalMaxWorkUnits: number;
}

export interface WorldQueryTraceBudgetPlan {
  readonly raysPerBlock: number;
  readonly primaryRaysPerBlock: number;
  readonly primaryWorkPerRay: number;
  readonly supplementalWorkPerRay: number;
  readonly primaryRayCount: number;
  readonly supplementalRayCount: number;
  readonly allocatedWorkUnits: number;
  readonly maxTotalWorkUnits: number;
}

/** Integer quotas bound aggregate traversal work independently of dispatch order. */
export function planWorldQueryTraceBudget(count: number,
  options: Pick<Required<WorldQueryOptions>, 'maxRays' | 'maxNodeVisits' | 'maxTotalWorkUnits'>,
  maxWorkUnits = options.maxTotalWorkUnits, partition?: WorldQueryWorkPartition): WorldQueryTraceBudgetPlan {
  if (!Number.isSafeInteger(count) || count < 0 || count > options.maxRays) throw new Error('WORLD_QUERY_RAY_BUDGET');
  const validBudget = (value: number): boolean => Number.isSafeInteger(value) && value >= 1 && value <= options.maxTotalWorkUnits;
  if (!validBudget(maxWorkUnits)) throw new Error('WORLD_QUERY_TRACE_WORK_BUDGET');
  let raysPerBlock = 0, primaryRaysPerBlock = 0, primaryRayCount = count, supplementalRayCount = 0;
  let supplementalBudget = 0;
  if (partition !== undefined) {
    if (!partition || typeof partition !== 'object') throw new Error('WORLD_QUERY_WORK_PARTITION');
    raysPerBlock = partition.raysPerBlock; primaryRaysPerBlock = partition.primaryRaysPerBlock;
    if (!Number.isSafeInteger(raysPerBlock) || raysPerBlock < 2 || raysPerBlock > count
      || count % raysPerBlock !== 0 || !Number.isSafeInteger(primaryRaysPerBlock)
      || primaryRaysPerBlock < 1 || primaryRaysPerBlock >= raysPerBlock) throw new Error('WORLD_QUERY_WORK_PARTITION');
    supplementalBudget = partition.supplementalMaxWorkUnits;
    if (!validBudget(supplementalBudget)) throw new Error('WORLD_QUERY_TRACE_WORK_BUDGET');
    primaryRayCount = (count / raysPerBlock) * primaryRaysPerBlock;
    supplementalRayCount = count - primaryRayCount;
  }
  const primaryWorkPerRay = primaryRayCount ? Math.min(options.maxNodeVisits, Math.floor(maxWorkUnits / primaryRayCount)) : 0;
  const supplementalWorkPerRay = supplementalRayCount ? Math.min(options.maxNodeVisits, Math.floor(supplementalBudget / supplementalRayCount)) : 0;
  if ((primaryRayCount && primaryWorkPerRay < 1) || (supplementalRayCount && supplementalWorkPerRay < 1)) {
    throw new Error('WORLD_QUERY_WORK_BUDGET: total budget cannot cover every ray');
  }
  return Object.freeze({ raysPerBlock, primaryRaysPerBlock, primaryWorkPerRay, supplementalWorkPerRay,
    primaryRayCount, supplementalRayCount,
    allocatedWorkUnits: primaryRayCount * primaryWorkPerRay + supplementalRayCount * supplementalWorkPerRay,
    maxTotalWorkUnits: maxWorkUnits + supplementalBudget });
}

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
  /** Rigid committed inputs can be pooled for shared static traversal. */
  rigid: boolean; committed: boolean;
  /** Leaf-ordered triangle words, retained only until the shared pool is built. */
  leafTriangles?: Uint32Array<ArrayBuffer>;
  /** Pool slot and per-parameter-slot resolve bindings when admitted to the shared pool. */
  pool?: { index: number; resolve: GPUBindGroup[] };
  /** Per-instance ordering boxes retained for the shared TLAS. */
  leafBounds: readonly Bounds3[];
  /** Whether the scene candidate tree currently contains this batch. */
  inSceneOrder: boolean;
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
function sameMembers(current: Uint32Array, previous: Uint32Array | undefined): boolean {
  return !!previous && previous.length === current.length
    && current.every((flags, index) => (flags & SHARED_FLAG_MEMBER) === (previous[index] & SHARED_FLAG_MEMBER));
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
  private sharedPipelines?: GpuQuerySharedStaticPipelines;
  private sharedPool?: GpuQuerySharedStatic;
  private sharedPlan?: SharedStaticPlan;
  /** Trace pipeline -> material resolve pipeline compiled from the same module. */
  private readonly resolvePipelines = new WeakMap<GPUComputePipeline, GPUComputePipeline>();
  /** Flags last written to the GPU, and those that built the scene candidate order. */
  private sharedUploadedFlags?: Uint32Array<ArrayBuffer>;
  private sharedOrderFlags?: Uint32Array<ArrayBuffer>;
  private sharedUpdates: { poolIndex: number; count: number }[] = [];
  private sharedLastUpdate: { frame: number; batches: number; dispatches: number; uploadedBytes: number } | null = null;
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
    if (WORLD_QUERY_TRACE_PARAMETER_BYTES > device.limits.maxUniformBufferBindingSize) {
      throw new Error('WORLD_QUERY_UNIFORM_BINDING_LIMIT');
    }
    this.traceUniformStride = Math.ceil(WORLD_QUERY_TRACE_PARAMETER_BYTES / alignment) * alignment;
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
      rayBuffer: this.buffer('world rays', sizes[0], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
      hitBuffer: this.buffer('world hits', sizes[1], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
    });
    this.readBuffer = this.buffer('world hit diagnostics', sizes[2], GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    this.resetParams = this.buffer('world query reset parameter slots', sizes[3], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    if (this.options.staticTraversal === 'shared') this.sharedPipelines = await GpuQuerySharedStaticPipelines.create(this.device);
    if (this.options.sceneAcceleration === 'bvh') {
      this.sceneBvh = await GpuQuerySceneBvh.create(this.device,
        (label, size, usage, data) => this.buffer(label, size, usage, data), this.rayBuffer, this.hitBuffer,
        this.options.maxRays, this.options.maxTraceParameterSetsPerEncoder, this.options.maxBatches);
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
    const module = await this.module(code, 'world material query');
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.traceLayout, this.materialLayout] });
    const trace = await this.device.createComputePipelineAsync({ label: 'shared material world hits',
      layout, compute: { module, entryPoint: 'main' } });
    if (this.sharedPipelines) {
      // Same module and layout: a pooled winner gets the per-batch path's exact shading code.
      this.resolvePipelines.set(trace, await this.device.createComputePipelineAsync({
        label: 'shared static material resolve', layout, compute: { module, entryPoint: 'resolveShared' } }));
    }
    return trace;
  }
  addBatch(source: QuerySource): void {
    if (this.disposed || this.frame) throw new Error('WORLD_QUERY_REGISTRATION_CLOSED');
    const { scene } = source, mesh = residentMesh(scene), proxy = queryProxy(scene);
    if (scene.emissiveSampling !== undefined && scene.emissiveSampling !== 'triangle-area') throw new Error('WORLD_QUERY_EMISSIVE_SAMPLING');
    if (this.batches.length >= this.options.maxBatches) throw new Error('WORLD_QUERY_BATCH_BUDGET');
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
        ? { buffer, offset: slot * this.traceUniformStride, size: WORLD_QUERY_TRACE_PARAMETER_BYTES } : { buffer } })) }));
    const createLevels = (input: readonly Uint32Array[], starts: readonly number[]): RefitLevel[] => input.map((level, i) => {
      const params = this.buffer('world refit depth parameters', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      return { params, count: level.length, offset: starts[i],
        bindGroup: this.device.createBindGroup({ layout: this.refitLayout,
          entries: [params, source.output, data, bounds].map((buffer, binding) => ({ binding, resource: { buffer } })) }) };
    });
    const levels = createLevels(topology.levels, levelOffsets);
    const instanceLevels = createLevels(instanceTopology?.levels ?? [], instanceLevelOffsets);
    const root = this.sceneBvh ? createLevels([new Uint32Array([0])], [rootOffset])[0] : undefined;
    // Only rigid committed inputs keep a static deformation output between edits.
    const rigid = scene.deformation.nodes.length === 0 && scene.morphTargets.length === 0;
    const committed = scene.inputMode === 'committed';
    this.batches.push({ source, mesh, name: scene.name, capacity: scene.capacity, identity: sourceIdentity,
      proxy, proxyIdentity: proxyIdentity(proxy), geometry: geometryMetadata(proxy),
      emissiveSampling: scene.emissiveSampling,
      topologyVersion: scene.queryIdentity?.topologyVersion ?? 0, nodeCount: topology.nodeCount, offsets, count: 0, dirty: true,
      vertexCount: mesh.vertexCount, vertices: mesh.vertices, indices: mesh.indices, uvs: mesh.uvs, tangents: mesh.tangents,
      colors: mesh.colors, colorOffset,
      identities, metadata: new Uint32Array(scene.capacity * 4), params, trace, levels, data, instanceTree, instanceLevels,
      bounds, root, sortBounds: null, rigid, committed,
      leafTriangles: this.sharedPipelines && rigid && committed
        ? packLeafOrderedTriangles(mesh.indices, topology.triangles, mesh.vertexCount) : undefined,
      leafBounds: [], inSceneOrder: false });
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
    let uploadedBytes = 0, sharedBytes = 0;
    let historyChanged = false;
    this.sceneDirty = false;
    if (firstSnapshot && this.sharedPipelines) this.buildSharedPool();
    const pool = this.sharedPool;
    this.sharedUpdates = [];
    for (const batch of this.batches) {
      const countChanged = batch.count !== batch.source.scene.instances.length;
      batch.dirty = firstSnapshot || countChanged || (batch.source.getDirty?.() ?? true);
      historyChanged ||= countChanged;
      batch.count = batch.source.scene.instances.length;
      if (!batch.dirty) continue;
      if (batch.pool && pool) {
        // Changed pooled batch: refresh its leaves now, copy positions/bounds after its refit.
        sharedBytes += pool.writeLeaves(batch.pool.index, packSharedLeaves(this.sharedPlan!.batches[batch.pool.index],
          batch.source.scene.instances.map(state => ({ entityId: state.entityId, generation: state.entityGeneration ?? 0 }))));
        sharedBytes += pool.writeUpdate(batch.pool.index, batch.count);
        this.sharedUpdates.push({ poolIndex: batch.pool.index, count: batch.count });
        if (!batch.count) batch.leafBounds = [];
      }
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
      if (batch.instanceTree[3] || this.sceneEnabled || batch.pool) {
        // CPU centers only determine traversal quality. Bounds used for culling
        // come from GPU BLAS roots, including skin, Morph and wave deformation.
        const sortBounds = batch.source.getInstanceBounds?.() ?? batch.source.scene.instances.map(state => {
          const center: [number, number, number] = [state.model[12], state.model[13], state.model[14]];
          return { min: center, max: center };
        });
        if (sortBounds.length !== batch.count) throw new Error('WORLD_QUERY_INSTANCE_BOUNDS_COUNT');
        if (batch.pool) batch.leafBounds = sortBounds;
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
    if (pool) {
      sharedBytes += this.uploadSharedFlags();
      // Pooled members leave the candidate tree; a membership change rebuilds its order.
      if (this.sceneEnabled && !sameMembers(this.sharedUploadedFlags!, this.sharedOrderFlags)) this.sceneDirty = true;
      if (this.sharedUpdates.length) sharedBytes += pool.writeOrder(this.sharedLeafOrder());
      if (this.sharedUpdates.length) this.sharedLastUpdate = { frame: this.frame, batches: this.sharedUpdates.length,
        dispatches: 0, uploadedBytes: sharedBytes };
    }
    if (this.sceneDirty) {
      const flags = this.sharedUploadedFlags;
      this.sharedOrderFlags = flags;
      const order = this.batches.map((batch, index) => flags && (flags[index] & SHARED_FLAG_MEMBER) ? null : batch.sortBounds);
      this.batches.forEach((batch, index) => { batch.inSceneOrder = order[index] !== null; });
      uploadedBytes += this.sceneBvh!.updateOrder(order);
    }
    if (historyChanged) this.sceneHistoryRevision = (this.sceneHistoryRevision + 1) >>> 0;
    return uploadedBytes + sharedBytes;
  }
  /** Admits the deterministic first-fit subset once registration has closed. */
  private buildSharedPool(): void {
    if (!this.batches.length) return;
    const slots = this.options.maxTraceParameterSetsPerEncoder;
    const alignment = this.device.limits.minUniformBufferOffsetAlignment;
    const stride = (bytes: number) => Math.ceil(bytes / alignment) * alignment;
    // Winners, traversal/update parameter slots and up to 17 TLAS depth parameters.
    const fixedBytes = this.options.maxRays * 4 + stride(32) * (slots + this.batches.length) + 17 * 16;
    const plan = planSharedStaticPool(this.batches.map((batch, index) => ({ index, capacity: batch.capacity,
      vertexCount: batch.vertexCount, nodeCount: batch.nodeCount, triangleCount: batch.indices.length / 3,
      ineligible: !batch.rigid ? 'deformation' as const : !batch.committed ? 'mutable-input' as const : null })), {
      maxBytes: Math.max(0, Math.min(this.options.maxSharedStaticBytes, this.options.maxBytes - this.bytes)),
      maxBindingBytes: this.device.limits.maxStorageBufferBindingSize, maxBufferBytes: this.device.limits.maxBufferSize,
      maxWorkgroupsPerDimension: this.device.limits.maxComputeWorkgroupsPerDimension,
      batchSlots: this.batches.length, fixedBytes });
    this.sharedPlan = plan;
    if (plan.batches.length) {
      const admitted = plan.batches.map(entry => this.batches[entry.index]);
      const pool = new GpuQuerySharedStatic(this.device, this.sharedPipelines!, plan,
        (label, size, usage, data) => this.buffer(label, size, usage, data), this.rayBuffer, this.hitBuffer,
        this.options.maxRays, slots, this.batches.length, admitted.map(batch => ({ output: batch.source.output,
          data: batch.data, bounds: batch.bounds, triangles: batch.leafTriangles! })));
      this.sharedPool = pool;
      admitted.forEach((batch, index) => {
        const buffers = [batch.params, batch.source.output, batch.data, batch.bounds, batch.source.instances,
          this.rayBuffer, this.hitBuffer, batch.identities, pool.winners];
        batch.pool = { index, resolve: Array.from({ length: slots }, (_, slot) => this.device.createBindGroup({
          layout: this.traceLayout, entries: buffers.map((buffer, binding) => ({ binding, resource: binding === 0
            ? { buffer, offset: slot * this.traceUniformStride, size: WORLD_QUERY_TRACE_PARAMETER_BYTES } : { buffer } })) })) };
      });
    }
    for (const batch of this.batches) batch.leafTriangles = undefined;
  }
  /** Current Opaque membership and face flags of pooled batches; non-pooled words stay zero. */
  private sharedMembership(): Uint32Array<ArrayBuffer> {
    const flags = new Uint32Array(this.batches.length);
    this.batches.forEach((batch, index) => {
      if (!batch.pool) return;
      const program = batch.source.getMaterial().program;
      flags[index] = packSharedFlags(program.alphaMode === 'opaque', program.doubleSided);
    });
    return flags;
  }
  private uploadSharedFlags(): number {
    const flags = this.sharedMembership();
    if (this.sharedUploadedFlags && flags.every((value, index) => value === this.sharedUploadedFlags![index])) return 0;
    this.sharedUploadedFlags = flags;
    return this.sharedPool!.writeFlags(flags);
  }
  private sharedLeafOrder(): Uint32Array<ArrayBuffer> {
    const plan = this.sharedPlan!, bounds: (Bounds3 | null)[] = Array(plan.leafCount).fill(null);
    for (const batch of this.batches) {
      if (!batch.pool) continue;
      const entry = plan.batches[batch.pool.index];
      batch.leafBounds.forEach((box, slot) => { if (slot < batch.count) bounds[entry.leafBase + slot] = box; });
    }
    return orderSharedLeaves(plan.leafCount, bounds);
  }
  /** Discrete identity/history changes; ordinary current/previous pose motion does not advance it. */
  get historyRevision(): number { return this.sceneHistoryRevision; }
  private get sceneEnabled(): boolean { return !!this.sceneBvh && this.batches.length >= 8; }
  encodeRefit(encoder: GPUCommandEncoder, timing?: GPUComputePassTimestampWrites): void {
    const sharedUpdates = this.sharedPool ? this.sharedUpdates : [];
    let remaining = this.batches.reduce((sum, batch) => sum + (batch.dirty
      ? (batch.count ? batch.levels.length + batch.instanceLevels.length : 0) + (this.sceneEnabled ? 1 : 0) : 0),
    (this.sceneDirty ? this.sceneBvh!.levelCount : 0) + (sharedUpdates.length ? 1 : 0));
    let first = true;
    if (!remaining && timing) {
      encoder.beginComputePass({ label: 'cached world BVH refit', timestampWrites: timing }).end();
    }
    const timestampForPass = (): GPUComputePassTimestampWrites | undefined => {
      const writes = first && remaining === 1 ? timing
        : first ? timestampSpanStart(timing) : remaining === 1 ? timestampSpanEnd(timing) : undefined;
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
    if (sharedUpdates.length) {
      // Changed pooled batches only; a static scene pays nothing after its first frame.
      const pass = encoder.beginComputePass({ label: 'world shared static pool update', timestampWrites: timestampForPass() });
      const dispatches = this.sharedPool!.encodeUpdate(pass, sharedUpdates);
      pass.end();
      if (this.sharedLastUpdate) this.sharedLastUpdate = { ...this.sharedLastUpdate, dispatches };
    }
    if (this.sceneDirty) this.sceneBvh!.encodeRefit(encoder, timestampForPass);
  }
  /** Encode GPU-resident rays -> hits. One encoder may vary ray count and texture
   * LOD while retaining the same frame, material and geometry snapshot. Every
   * distinct parameter set gets a slot; identical sets reuse their slot. Rewrite
   * rays with encoded GPU operations, not queue writes that alias recorded work.
   * Submit it before encoding another encoder with different parameters: GPU
   * snapshots and uniform buffers assume serial encode/submit. Callers own ray
   * generation and inspect hit.work.z for an explicit overflow. An optional
   * maxWorkUnits limits this trace. An explicit partition adds a separate bounded
   * supplemental budget without changing primary quotas or ray/hit indices.
   * Reversed material support changes only single-sided eligibility; its hits
   * are not receiver-origin nearest shading hits. An optional timestamp pair spans
   * this trace's own passes (first pass beginning to last pass end; both ends are
   * written on every timed pass). Returns queued uniform bytes. */
  encodeTrace(encoder: GPUCommandEncoder, count: number, textureLod: number, materialRevision: number,
    maxWorkUnits = this.options.maxTotalWorkUnits, partition?: WorldQueryWorkPartition,
    facePolicy: WorldQueryFacePolicy = 'material', timing?: GPUComputePassTimestampWrites): number {
    if (this.disposed || !this.frame) throw new Error('WORLD_QUERY_NO_RENDERED_FRAME');
    if (facePolicy !== 'material' && facePolicy !== 'reverse-material-support') throw new Error('WORLD_QUERY_FACE_POLICY');
    this.validateTopology();
    if (!Number.isSafeInteger(count) || count < 0 || count > this.options.maxRays) throw new Error('WORLD_QUERY_RAY_BUDGET');
    if (!Number.isFinite(textureLod) || textureLod < 0 || textureLod > 32) throw new Error('WORLD_QUERY_TEXTURE_LOD');
    u32(materialRevision, 'MATERIAL_REVISION');
    const budget = planWorldQueryTraceBudget(count, this.options, maxWorkUnits, partition);
    if (!count) return 0;
    const workPerRay = budget.primaryWorkPerRay;
    // Pooled membership follows the current material, so a Masked switch can
    // never bypass coverage; it is part of the snapshot recorded work assumes.
    const pool = this.sharedPool, flags = pool ? this.sharedMembership() : undefined;
    const members = this.batches.map((batch, index) => !!flags && (flags[index] & SHARED_FLAG_MEMBER) !== 0);
    const snapshot = JSON.stringify([this.frame, materialRevision,
      this.batches.map((batch, index) => [index + 1, batch.nodeCount, batch.mesh.vertexCount, batch.count,
        ...batch.offsets, batch.topologyVersion]), flags ? Array.from(flags) : null]);
    let state = this.traceParameters.get(encoder);
    if (state && state.snapshot !== snapshot) throw new Error('WORLD_QUERY_ENCODER_SNAPSHOT_CHANGED');
    const key = JSON.stringify([count, Math.fround(textureLod), workPerRay,
      budget.raysPerBlock, budget.primaryRaysPerBlock, budget.supplementalWorkPerRay, facePolicy]);
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
    // Resolve every pipeline before recording so a failure never leaves a pass open.
    const active = this.batches.map((batch, index) => ({ batch, index, member: members[index] }))
      .filter(entry => entry.batch.count > 0);
    const resolves = new Map<number, GPUComputePipeline>();
    for (const { batch, index, member } of active) {
      if (!member) continue;
      const resolve = this.resolvePipelines.get(batch.source.getPipeline());
      if (!resolve) throw new Error('WORLD_QUERY_SHARED_RESOLVE_PIPELINE');
      resolves.set(index, resolve);
    }
    let uploadedBytes = 0;
    if (upload) {
      this.device.queue.writeBuffer(this.resetParams, parameterSlot * this.resetUniformStride,
        new Uint32Array([count, this.frame, materialRevision, 0]));
      uploadedBytes += 16;
    }
    // A material change since the last snapshot: the encoder snapshot above already
    // isolates recorded work, and the next snapshot rebuilds the candidate order.
    if (pool) uploadedBytes += this.uploadSharedFlags();
    const single = this.options.dispatchEncoding === 'single-pass';
    let shared: GPUComputePassEncoder | undefined;
    // WebGPU orders dispatches and their storage writes within one pass, so the
    // single-pass encoding keeps closest-hit ordering without a pass per batch.
    const begin = (label: string, timestampWrites?: GPUComputePassTimestampWrites): GPUComputePassEncoder => single
      ? (shared ??= encoder.beginComputePass({ label: 'world query trace', timestampWrites: timing }))
      : encoder.beginComputePass({ label, timestampWrites });
    const end = (pass: GPUComputePassEncoder): void => { if (!single) pass.end(); };
    // Pass-per-batch timing spans the reset pass to the last batch pass.
    const last = active.length ? active[active.length - 1].index : -1;
    const reset = begin('world ray reset', last < 0 ? timing : timestampSpanStart(timing));
    reset.setPipeline(this.resetPipeline); reset.setBindGroup(0, this.resetGroups[parameterSlot]);
    reset.dispatchWorkgroups(Math.ceil(count / 64)); end(reset);
    // Pooled members are not in the candidate tree; skip it when nothing else uses it.
    // Without a pool the per-batch path keeps its unconditional candidate pass.
    const candidates = this.sceneEnabled && (!pool || active.some(entry => !entry.member && entry.batch.inSceneOrder));
    if (candidates) {
      uploadedBytes += this.sceneBvh!.encodeCandidates(encoder, parameterSlot, count, workPerRay, upload,
        partition ? budget : undefined, single ? begin('') : undefined);
    }
    if (pool && active.some(entry => entry.member)) {
      const pass = begin('world shared static traversal');
      uploadedBytes += pool.encodeTraverse(pass, parameterSlot, count, workPerRay, upload, budget,
        facePolicy === 'reverse-material-support');
      end(pass);
    }
    for (const { batch, index, member } of active) {
      if (upload) {
        const params = new ArrayBuffer(WORLD_QUERY_TRACE_PARAMETER_BYTES);
        new Uint32Array(params).set([batch.nodeCount, batch.mesh.vertexCount, batch.count, index + 1,
          ...batch.offsets, batch.colorOffset, count, workPerRay, this.frame, batch.topologyVersion, materialRevision,
          batch.emissiveSampling === 'triangle-area' ? 1 : 0,
          // Candidate masks cover only batches in the tree; others scan their instances.
          this.sceneEnabled && batch.inSceneOrder ? 1 : 0, this.sceneBvh?.maskWords ?? 0]);
        new Float32Array(params)[16] = textureLod;
        new Uint32Array(params).set([budget.raysPerBlock, budget.primaryRaysPerBlock, budget.supplementalWorkPerRay], 17);
        new Uint32Array(params).set(batch.instanceTree, 20);
        new Uint32Array(params)[24] = facePolicy === 'reverse-material-support' ? 1 : 0;
        this.device.queue.writeBuffer(batch.params, parameterSlot * this.traceUniformStride, params);
        uploadedBytes += WORLD_QUERY_TRACE_PARAMETER_BYTES;
      }
      // Distinct passes (or ordered dispatches in one pass) order closest-hit
      // updates between batches. Members only resolve rays they won.
      const pass = begin((member ? 'world shared static resolve ' : 'world rays ') + batch.name,
        index === last ? timestampSpanEnd(timing) : undefined);
      pass.setPipeline(member ? resolves.get(index)! : batch.source.getPipeline());
      pass.setBindGroup(0, member ? batch.pool!.resolve[parameterSlot] : batch.trace[parameterSlot]);
      pass.setBindGroup(1, batch.source.getMaterial().bindGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); end(pass);
    }
    shared?.end();
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
      traceParameterBytes: WORLD_QUERY_TRACE_PARAMETER_BYTES,
      resetUniformStride: this.resetUniformStride,
      instanceAcceleration: this.batches.some(batch => batch.instanceTree[3]) ? 'batch-bvh' : 'linear-root-scan',
      instanceBvhNodes: this.batches.reduce((sum, batch) => sum + batch.instanceTree[3], 0),
      acceleratedBatches: this.batches.filter(batch => batch.instanceTree[3]).length,
      sceneAcceleration: this.sceneEnabled ? 'batch-bvh' : 'linear-batches',
      sceneBvhNodes: this.sceneEnabled ? this.sceneBvh!.nodeCount : 0,
      sceneCandidateWordsPerRay: this.sceneBvh?.maskWords ?? 0,
      sceneCandidateBytes: this.candidateBuffer.size,
      staticTraversal: this.options.staticTraversal, dispatchEncoding: this.options.dispatchEncoding,
      sharedStatic: this.sharedStaticDiagnostics(),
      lastReadbackQuery: this.lastReadbackQuery && { ...this.lastReadbackQuery },
      refit: 'changed-batches', refitTiming: 'provided-by-caller', externalTraceTiming: false };
  }
  /** Admission is fixed at the first snapshot; membership is the last uploaded material state. */
  private sharedStaticDiagnostics() {
    const plan = this.sharedPlan;
    if (!plan) return null;
    const excluded: Record<SharedStaticExclusion, number> = { deformation: 0, 'mutable-input': 0,
      'dispatch-limit': 0, 'leaf-limit': 0, 'binding-limit': 0, budget: 0 };
    for (const entry of plan.excluded) excluded[entry.reason]++;
    const flags = this.sharedUploadedFlags;
    return { pooledBatches: plan.batches.length, excludedBatches: plan.excluded.length, excluded,
      members: flags ? flags.reduce((sum, value) => sum + (value & SHARED_FLAG_MEMBER), 0) : 0,
      leaves: plan.leafCount, tlasNodes: plan.tlasNodeCount, blasRecords: plan.nodeRecords - plan.tlasNodeCount,
      triangles: plan.triangleCount, vertices: plan.vertexCount, bytes: { ...plan.bytes },
      lastUpdate: this.sharedLastUpdate && { ...this.sharedLastUpdate } };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
