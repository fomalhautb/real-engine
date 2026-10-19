import commonSource from './shaders/diffuse-gi-common.wgsl?raw';
import areaSource from './shaders/gi-emissive.wgsl?raw';
import { residentMesh, type RenderBatch, type StreamedSceneBatch } from './contracts';
import { DIFFUSE_GI_PROBE_BYTES } from './diffuse-gi';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from './gi-visibility';
import type { GpuWorldQuery } from './gpu-world-query';
import { WORLD_HIT_BYTES, WORLD_RAY_BYTES } from './world-query';
import { DEFORMED_VERTEX_BYTES, type MeshData } from './geometry';

type QueryProxy = StreamedSceneBatch['geometryStream']['queryProxy'];

const SAMPLE_BYTES = 48;
const DEFAULT_MAX_BYTES = 64 * 1024 * 1024;
const HARD_MAX_BYTES = 128 * 1024 * 1024;
const PARAMETER_BYTES = 32;
interface EmissiveBatch {
  scene: RenderBatch;
  mesh: MeshData;
  proxy?: QueryProxy;
  proxyIdentity: string;
  vertices: ArrayBuffer;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
  capacity: number;
  name: string;
  identity: string;
  sourceId: number;
  count: number;
  parameters: GPUBuffer;
  identities: GPUBuffer;
  metadata: Uint32Array<ArrayBuffer>;
  group: GPUBindGroup;
}

function uint(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) throw new Error('GI_EMISSIVE_' + label);
  return value;
}
function topologyIdentity(scene: RenderBatch): string {
  const identity = scene.queryIdentity;
  return identity ? JSON.stringify([identity.assetId, identity.primitiveId, identity.topologyVersion]) : '';
}
function queryProxy(scene: RenderBatch): QueryProxy | undefined {
  return scene.mesh ? undefined : scene.geometryStream.queryProxy;
}
function proxyIdentity(proxy: QueryProxy | undefined): string {
  return proxy ? JSON.stringify([proxy.version, proxy.primitiveKey, proxy.directorySha256,
    proxy.topologyVersion, proxy.requestedError, proxy.approximateObjectSpaceError]) : '';
}

/** Bounded area importance sampling of opted-in deformed scene triangles.
 * Materials, Mask coverage, sidedness and occlusion are evaluated by the same
 * world queries as the sphere capture. Borrowed buffers are never destroyed. */
export class GpuGiEmissiveSampler {
  private readonly buffers: GPUBuffer[] = [];
  private readonly batches: EmissiveBatch[] = [];
  private bytes = 0;
  private disposed = false;
  private updated = false;
  private currentRayCount = 0;
  private samples!: GPUBuffer;
  private parameters!: GPUBuffer;
  private clearPipeline!: GPUComputePipeline;
  private generatePipeline!: GPUComputePipeline;
  private integratePipeline!: GPUComputePipeline;
  private clearGroup!: GPUBindGroup;
  private integrateGroups!: GPUBindGroup[];

  get enabled(): boolean { return !this.disposed && this.samplesPerProbe > 0 && this.batches.length > 0; }
  get emitterCount(): number { return this.batches.length; }
  /** Active blocks keep their stable indices; only inactive trailing blocks
   * are omitted. Interior holes use empty rays, without full-grid padding. */
  get queryRayCount(): number { return this.currentRayCount; }

  static async create(device: GPUDevice, queries: GpuWorldQuery, giParameterBuffer: GPUBuffer,
    outputProbeBuffers: [GPUBuffer, GPUBuffer], rayCount: number, samplesPerProbe: number,
    account: (bytes: number) => void, maxBytes = DEFAULT_MAX_BYTES): Promise<GpuGiEmissiveSampler> {
    const probeCount = outputProbeBuffers[0].size / DIFFUSE_GI_PROBE_BYTES;
    if (!Number.isSafeInteger(probeCount) || probeCount < 1 || probeCount > 512
      || outputProbeBuffers[1].size !== outputProbeBuffers[0].size) throw new Error('GI_EMISSIVE_PROBE_LAYOUT');
    if (!Number.isSafeInteger(rayCount) || rayCount < 1 || rayCount > 65_536 || rayCount % probeCount !== 0
      || rayCount / probeCount > 256 || rayCount > queries.options.maxRays
      || rayCount > queries.options.maxTotalWorkUnits) throw new Error('GI_EMISSIVE_RAY_BUDGET');
    if (!Number.isSafeInteger(samplesPerProbe) || samplesPerProbe < 0 || samplesPerProbe > 256
      || probeCount * samplesPerProbe > rayCount) throw new Error('GI_EMISSIVE_SAMPLE_BUDGET');
    if (device.limits.maxStorageBuffersPerShaderStage < 5) throw new Error('GI_EMISSIVE_STORAGE_BINDING_LIMIT');
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > HARD_MAX_BYTES) throw new Error('GI_EMISSIVE_MEMORY_BUDGET');
    if (giParameterBuffer.size < 112 || !(giParameterBuffer.usage & GPUBufferUsage.UNIFORM)
      || outputProbeBuffers.some(buffer => !(buffer.usage & GPUBufferUsage.STORAGE))
      || queries.rayBuffer.size < rayCount * WORLD_RAY_BYTES || queries.hitBuffer.size < rayCount * WORLD_HIT_BYTES) {
      throw new Error('GI_EMISSIVE_BORROWED_BUFFER_LAYOUT');
    }
    const result = new GpuGiEmissiveSampler(device, queries, giParameterBuffer,
      outputProbeBuffers, rayCount, samplesPerProbe, probeCount, account, maxBytes);
    try { await result.initialize(); return result; }
    catch (error) { result.dispose(); throw error; }
  }
  private constructor(private readonly device: GPUDevice, private readonly queries: GpuWorldQuery,
    private readonly giParameterBuffer: GPUBuffer, private readonly outputProbeBuffers: [GPUBuffer, GPUBuffer],
    private readonly rayCount: number, private readonly samplesPerProbe: number, private readonly probeCount: number,
    private readonly account: (bytes: number) => void, private readonly maxBytes: number) {}

  private assertAlive(): void { if (this.disposed) throw new Error('GI_EMISSIVE_DISPOSED'); }
  private preflight(sizes: readonly number[]): void {
    if (sizes.some(size => !Number.isSafeInteger(size) || size < 4
      || size > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize))) {
      throw new Error('GI_EMISSIVE_BUFFER_LIMIT');
    }
    if (this.bytes + sizes.reduce((total, size) => total + size, 0) > this.maxBytes) throw new Error('GI_EMISSIVE_MEMORY_BUDGET');
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: Uint32Array<ArrayBuffer>): GPUBuffer {
    this.preflight([size]);
    const buffer = this.device.createBuffer({ label, size, usage, mappedAtCreation: !!data });
    this.buffers.push(buffer); this.bytes += size; this.account(size);
    if (data) { new Uint32Array(buffer.getMappedRange()).set(data); buffer.unmap(); }
    return buffer;
  }
  private async initialize(): Promise<void> {
    this.preflight([this.rayCount * SAMPLE_BYTES, PARAMETER_BYTES]);
    if (Math.ceil(this.rayCount / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error('GI_EMISSIVE_DISPATCH_LIMIT');
    }
    this.samples = this.buffer('GI emissive area sample metadata', this.rayCount * SAMPLE_BYTES, GPUBufferUsage.STORAGE);
    this.parameters = this.buffer('GI emissive global sample parameters', PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const constants = `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`;
    const common = commonSource.replace('// GI_DISTANCE_CONSTANTS', constants);
    const module = this.device.createShaderModule({ label: 'GI emissive area sampling', code: areaSource.replace('// GI_COMMON', common) });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `GI emissive:${error.lineNum}: ${error.message}`).join('\n'));
    [this.clearPipeline, this.generatePipeline, this.integratePipeline] = await Promise.all(
      ['clear', 'generate', 'integrate'].map(entryPoint => this.device.createComputePipelineAsync({
        label: 'GI emissive ' + entryPoint, layout: 'auto', compute: { module, entryPoint },
      })));
    this.clearGroup = this.device.createBindGroup({ layout: this.clearPipeline.getBindGroupLayout(0), entries: [
      { binding: 1, resource: { buffer: this.parameters } }, { binding: 5, resource: { buffer: this.queries.rayBuffer } },
      { binding: 6, resource: { buffer: this.samples } },
    ] });
    this.integrateGroups = this.outputProbeBuffers.map(buffer => this.device.createBindGroup({
      layout: this.integratePipeline.getBindGroupLayout(0), entries: [
        { binding: 1, resource: { buffer: this.parameters } }, { binding: 6, resource: { buffer: this.samples } },
        { binding: 7, resource: { buffer: this.queries.hitBuffer } }, { binding: 8, resource: { buffer } },
      ],
    }));
  }
  addBatch(scene: RenderBatch, output: GPUBuffer, sourceId: number): void {
    this.assertAlive();
    if (this.updated) throw new Error('GI_EMISSIVE_REGISTRATION_CLOSED');
    if (scene.emissiveSampling === undefined) return;
    if (scene.emissiveSampling !== 'triangle-area') throw new Error('GI_EMISSIVE_SAMPLING_MODE');
    if (this.samplesPerProbe === 0) return;
    uint(sourceId, 'SOURCE_ID');
    if (sourceId === 0 || this.batches.some(batch => batch.sourceId === sourceId || batch.scene === scene)) {
      throw new Error('GI_EMISSIVE_DUPLICATE_SOURCE');
    }
    const mesh = residentMesh(scene), proxy = queryProxy(scene);
    if (!(mesh.indices instanceof Uint32Array) || mesh.indices.length === 0 || mesh.indices.length % 3 !== 0
      || mesh.indices.length / 3 > this.queries.options.maxTrianglesPerMesh
      || !Number.isSafeInteger(mesh.vertexCount) || mesh.vertexCount < 1 || !(mesh.vertices instanceof ArrayBuffer)
      || mesh.vertices.byteLength !== mesh.vertexCount * 80) throw new Error('GI_EMISSIVE_MESH_LAYOUT');
    if (!Number.isSafeInteger(scene.capacity) || scene.capacity < 1 || scene.capacity > this.queries.options.maxInstancesPerBatch
      || scene.instances.length > scene.capacity) throw new Error('GI_EMISSIVE_INSTANCE_BUDGET');
    if (this.batches.length >= 256
      || (this.batches.length + 1) * this.probeCount * this.samplesPerProbe > this.rayCount) {
      throw new Error('GI_EMISSIVE_SOURCE_SAMPLE_BUDGET');
    }
    const requiredOutput = mesh.vertexCount * scene.capacity * DEFORMED_VERTEX_BYTES;
    if (!Number.isSafeInteger(requiredOutput) || output.size < requiredOutput || !(output.usage & GPUBufferUsage.STORAGE)
      || output.size > this.device.limits.maxStorageBufferBindingSize) throw new Error('GI_EMISSIVE_DEFORMATION_BUFFER');
    this.preflight([mesh.indices.byteLength, scene.capacity * 16, PARAMETER_BYTES]);
    for (const index of mesh.indices) if (index >= mesh.vertexCount) throw new Error('GI_EMISSIVE_MESH_INDEX');
    const start = this.buffers.length, beforeBytes = this.bytes;
    try {
      const indices = this.buffer('GI emissive immutable triangle indices', mesh.indices.byteLength,
        GPUBufferUsage.STORAGE, mesh.indices.slice());
      const identities = this.buffer('GI emissive current entity identities', scene.capacity * 16,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const parameters = this.buffer('GI emissive source sample parameters', PARAMETER_BYTES,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const group = this.device.createBindGroup({ layout: this.generatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.giParameterBuffer } }, { binding: 1, resource: { buffer: parameters } },
        { binding: 2, resource: { buffer: output } }, { binding: 3, resource: { buffer: indices } },
        { binding: 4, resource: { buffer: identities } }, { binding: 5, resource: { buffer: this.queries.rayBuffer } },
        { binding: 6, resource: { buffer: this.samples } },
      ] });
      this.batches.push({ scene, mesh, proxy, proxyIdentity: proxyIdentity(proxy),
        vertices: mesh.vertices, indices: mesh.indices, vertexCount: mesh.vertexCount,
        indexCount: mesh.indices.length, capacity: scene.capacity, name: scene.name, identity: topologyIdentity(scene),
        sourceId, count: 0, parameters, identities, metadata: new Uint32Array(scene.capacity * 4), group });
    } catch (error) {
      for (const buffer of this.buffers.splice(start)) buffer.destroy();
      this.account(beforeBytes - this.bytes); this.bytes = beforeBytes;
      throw error;
    }
  }
  update(): number {
    this.assertAlive();
    // Validate the complete snapshot before publishing any queue writes.
    for (const batch of this.batches) {
      const { scene } = batch;
      const mesh = residentMesh(scene), proxy = queryProxy(scene);
      if (mesh !== batch.mesh || mesh.vertices !== batch.vertices || mesh.indices !== batch.indices
        || mesh.vertexCount !== batch.vertexCount || mesh.indices.length !== batch.indexCount
        || proxy !== batch.proxy || proxyIdentity(proxy) !== batch.proxyIdentity
        || scene.capacity !== batch.capacity || scene.name !== batch.name || topologyIdentity(scene) !== batch.identity
        || scene.emissiveSampling !== 'triangle-area') throw new Error('GI_EMISSIVE_TOPOLOGY_CHANGED');
      if (scene.instances.length > batch.capacity) throw new Error('GI_EMISSIVE_INSTANCE_BUDGET');
      const alive = new Set<number>();
      for (const state of scene.instances) {
        uint(state.entityId, 'ENTITY_ID'); uint(state.entityGeneration ?? 0, 'GENERATION');
        if (alive.has(state.entityId)) throw new Error('GI_EMISSIVE_DUPLICATE_ENTITY');
        alive.add(state.entityId);
      }
    }
    this.updated = true;
    let blockCount = 0;
    this.batches.forEach((batch, index) => {
      batch.count = batch.scene.instances.length;
      if (batch.count) blockCount = index + 1;
    });
    this.currentRayCount = blockCount * this.probeCount * this.samplesPerProbe;
    this.device.queue.writeBuffer(this.parameters, 0, new Uint32Array([
      blockCount, 0, 0, 0, this.currentRayCount, this.probeCount, this.samplesPerProbe, 0,
    ]));
    let uploaded = PARAMETER_BYTES;
    this.batches.forEach((batch, index) => {
      batch.count = batch.scene.instances.length;
      batch.scene.instances.forEach((state, i) => batch.metadata.set([state.entityId, state.entityGeneration ?? 0, 0, 0], i * 4));
      if (batch.count) {
        this.device.queue.writeBuffer(batch.identities, 0, batch.metadata, 0, batch.count * 4);
        uploaded += batch.count * 16;
      }
      this.device.queue.writeBuffer(batch.parameters, 0, new Uint32Array([
        batch.vertexCount, batch.indexCount / 3, batch.count, batch.sourceId,
        this.currentRayCount, this.probeCount, this.samplesPerProbe, index * this.probeCount * this.samplesPerProbe,
      ]));
      uploaded += PARAMETER_BYTES;
    });
    return uploaded;
  }
  encode(encoder: GPUCommandEncoder, outputIndex: number, materialRevision: number): number {
    this.assertAlive();
    if (outputIndex !== 0 && outputIndex !== 1) throw new Error('GI_EMISSIVE_OUTPUT_INDEX');
    uint(materialRevision, 'MATERIAL_REVISION');
    if (!this.enabled) return 0;
    if (!this.updated) throw new Error('GI_EMISSIVE_SNAPSHOT_MISSING');
    if (!this.currentRayCount) return 0;
    const dispatch = (label: string, pipeline: GPUComputePipeline, group: GPUBindGroup, count: number) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    };
    dispatch('clear GI emissive sample blocks', this.clearPipeline, this.clearGroup, this.currentRayCount);
    for (const batch of this.batches) if (batch.count) {
      dispatch('sample deformed GI emissive triangles', this.generatePipeline, batch.group,
        this.probeCount * this.samplesPerProbe);
    }
    const uploaded = this.queries.encodeTrace(encoder, this.currentRayCount, 0, materialRevision);
    dispatch('integrate GI emissive solid angle', this.integratePipeline, this.integrateGroups[outputIndex], this.probeCount);
    return uploaded;
  }
  diagnostics() {
    return { enabled: this.enabled, emitterCount: this.emitterCount,
      activeEmitterCount: this.batches.filter(batch => batch.count > 0).length,
      samplesPerProbePerEmitter: this.samplesPerProbe,
      sampleCapacity: this.batches.length * this.probeCount * this.samplesPerProbe,
      samplesPerUpdate: this.batches.filter(batch => batch.count > 0).length * this.probeCount * this.samplesPerProbe,
      activeSamplesPerUpdate: this.batches.filter(batch => batch.count > 0).length * this.probeCount * this.samplesPerProbe,
      queryRaysPerUpdate: this.currentRayCount, queryRayCapacity: this.rayCount,
      inactiveBlockRaysPerUpdate: this.currentRayCount
        - this.batches.filter(batch => batch.count > 0).length * this.probeCount * this.samplesPerProbe,
      maxWorkUnitsPerUpdate: this.currentRayCount ? this.queries.options.maxTotalWorkUnits : 0,
      ownedGpuBytes: this.bytes, maxBytes: this.maxBytes, representation: 'triangle-area-to-radiance-SH9',
      coverage: 'shared-world-query', geometry: 'shared-current-deformation',
      geometryRepresentations: this.batches.map(batch => ({ batch: batch.name,
        geometry: batch.proxy ? { kind: 'fixed-query-proxy' as const, primitiveKey: batch.proxy.primitiveKey,
          directorySha256: batch.proxy.directorySha256, topologyVersion: batch.proxy.topologyVersion,
          approximateObjectSpaceError: batch.proxy.approximateObjectSpaceError,
          errorMetric: 'accumulated-simplification' as const } : { kind: 'source' as const } })),
      queryGeometryRepresentations: this.queries.diagnostics().geometryRepresentations,
      sampling: 'fixed-probe-source-hashed-triangles-rotated-Hammersley',
      nearEmitterVariance: 'uniform-triangle-area-estimator', temporalAccumulation: false };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
