import source from './shaders/geometry-lod.wgsl?raw';
import { packGeometryLod, type PackedGeometryLod } from './geometry-lod-data';
import type { SceneBatch } from './contracts';
import type { Mat4 } from './math';

export interface GeometryLodSettings {
  maxErrorPixels?: number;
  frustumCulling?: boolean;
  /** Freeze the selection camera; drawing still uses the live camera and poses. */
  frozen?: boolean;
  debug?: 'material' | 'clusters' | 'levels';
}
export interface GeometryLodProbe {
  frame: number;
  indices: number;
  triangles: number;
  selectedClusters: { clusterId: number; instanceIndex: number; level: number }[];
  overflow: boolean;
}
export interface GeometryLodBinding {
  vertices: GPUBuffer;
  indices: GPUBuffer;
  arguments: GPUBuffer;
}
type Entry = {
  scene: SceneBatch; packed: PackedGeometryLod; binding: GeometryLodBinding;
  params: GPUBuffer; paramsData: Uint32Array<ArrayBuffer>; selected: GPUBuffer; compute: GPUBindGroup;
  enabled: () => boolean; count: number; source: SceneBatch['geometryLod'];
  capacity: number;
  vertexSource: ArrayBuffer; indexSource: Uint32Array; uvSource?: Float32Array; tangentSource?: Float32Array;
};
const BUDGET = 128 * 1024 * 1024;
const MAX_PAIRS = 262_144;

/** Fully resident GPU cut selection and index compaction. No page streaming or occlusion yet. */
export class GpuGeometryLod {
  readonly settings: Required<GeometryLodSettings> = { maxErrorPixels: 1, frustumCulling: true, frozen: false, debug: 'material' };
  private readonly entries = new Map<string, Entry>();
  private readonly buffers: GPUBuffer[] = [];
  private bytes = 0;
  private frame = 0;
  private disposed = false;
  private probing = false;
  private readback?: GPUBuffer;
  private hasView = false;
  private readonly viewData = new Float32Array(24);
  private readonly view: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private select!: GPUComputePipeline;
  private finalize!: GPUComputePipeline;

  static async create(device: GPUDevice, track: (bytes: number) => void): Promise<GpuGeometryLod> {
    const result = new GpuGeometryLod(device, track);
    try {
      const module = device.createShaderModule({ label: 'resident geometry LOD', code: source });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(e => `GEOMETRY_LOD_SHADER:${e.lineNum}: ${e.message}`).join('\n'));
      const layout = device.createPipelineLayout({ bindGroupLayouts: [result.layout] });
      result.select = await device.createComputePipelineAsync({ label: 'select LOD and compact indices', layout,
        compute: { module, entryPoint: 'selectClusters' } });
      result.finalize = await device.createComputePipelineAsync({ label: 'validate indirect LOD arguments', layout,
        compute: { module, entryPoint: 'finalize' } });
      return result;
    } catch (error) { result.dispose(); throw error; }
  }
  private constructor(private device: GPUDevice, private track: (bytes: number) => void) {
    this.view = this.buffer('LOD selection camera', 96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.layout = device.createBindGroupLayout({ entries: Array.from({ length: 9 }, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 || binding === 5 ? 'uniform' : binding >= 6 ? 'storage' : 'read-only-storage' },
    })) });
  }
  private buffer(label: string, bytes: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer {
    const size = Math.max(4, bytes);
    if (size > this.device.limits.maxBufferSize || (usage & GPUBufferUsage.STORAGE) && size > this.device.limits.maxStorageBufferBindingSize
      || this.bytes + size > BUDGET) throw new Error('GEOMETRY_LOD_RESOURCE_BUDGET');
    const buffer = this.device.createBuffer({ label, size, usage, mappedAtCreation: !!data });
    this.buffers.push(buffer); this.bytes += size; this.track(size);
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data)); buffer.unmap(); }
    return buffer;
  }
  setSettings(next: GeometryLodSettings): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (next.maxErrorPixels !== undefined && (!Number.isFinite(next.maxErrorPixels) || next.maxErrorPixels < 0 || next.maxErrorPixels > 10_000)
      || next.frustumCulling !== undefined && typeof next.frustumCulling !== 'boolean'
      || next.frozen !== undefined && typeof next.frozen !== 'boolean'
      || next.debug !== undefined && !['material', 'clusters', 'levels'].includes(next.debug)) throw new Error('GEOMETRY_LOD_SETTINGS');
    if (next.maxErrorPixels !== undefined) this.settings.maxErrorPixels = next.maxErrorPixels;
    if (next.frustumCulling !== undefined) this.settings.frustumCulling = next.frustumCulling;
    if (next.frozen !== undefined) this.settings.frozen = next.frozen;
    if (next.debug !== undefined) this.settings.debug = next.debug;
  }
  addBatch(scene: SceneBatch, instances: GPUBuffer, enabled: () => boolean): GeometryLodBinding {
    const hierarchy = scene.geometryLod;
    if (!hierarchy || hierarchy.source !== scene.mesh || scene.deformation.jointCount !== 0 || scene.deformation.nodes.length || scene.morphTargets.length) {
      throw new Error('GEOMETRY_LOD_REQUIRES_STATIC_SOURCE: use hierarchy.source and the full-detail path for deformation');
    }
    if (!Number.isSafeInteger(scene.capacity) || scene.capacity < 1 || scene.capacity > 256) throw new Error('GEOMETRY_LOD_INSTANCE_BUDGET');
    const packed = packGeometryLod(hierarchy);
    const pairs = packed.clusterCount * scene.capacity;
    const maxIndices = packed.sourceTriangleCount * 3 * scene.capacity;
    if (pairs > MAX_PAIRS || Math.ceil(pairs / 64) > this.device.limits.maxComputeWorkgroupsPerDimension
      || packed.clusterVertexCount * scene.capacity > 0xffffffff || maxIndices > 0xffffffff) throw new Error('GEOMETRY_LOD_WORK_BUDGET');
    const sizes = [packed.clusters.byteLength, packed.groups.byteLength, packed.vertices.byteLength, packed.indices.byteLength, maxIndices * 4, pairs * 4, 32, 32];
    if (sizes.some(size => size > this.device.limits.maxStorageBufferBindingSize || size > this.device.limits.maxBufferSize)
      || sizes.reduce((a, b) => a + b, this.bytes) > BUDGET) throw new Error('GEOMETRY_LOD_RESOURCE_BUDGET');
    const clusters = this.buffer('LOD clusters', packed.clusters.byteLength, GPUBufferUsage.STORAGE, packed.clusters);
    const groups = this.buffer('LOD groups', packed.groups.byteLength, GPUBufferUsage.STORAGE, packed.groups);
    const vertices = this.buffer('LOD vertex identities', packed.vertices.byteLength, GPUBufferUsage.STORAGE, new Uint32Array(packed.vertices).buffer);
    const input = this.buffer('LOD source indices', packed.indices.byteLength, GPUBufferUsage.STORAGE, new Uint32Array(packed.indices).buffer);
    const indices = this.buffer('GPU selected indices', maxIndices * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX);
    const args = this.buffer('LOD indirect arguments and counters', 32, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const selected = this.buffer('LOD selected cluster flags', pairs * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const paramsData = new Uint32Array([packed.clusterCount, packed.clusterVertexCount, packed.sourceVertexCount, maxIndices, 0, 0, 0, 0]);
    const params = this.buffer('LOD batch parameters', 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, paramsData.buffer);
    const compute = this.device.createBindGroup({ layout: this.layout,
      entries: [this.view, clusters, groups, input, instances, params, indices, args, selected].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const binding = { vertices, indices, arguments: args };
    this.entries.set(scene.name, { scene, source: hierarchy, packed, binding, params, paramsData, selected, compute, enabled, count: 0, capacity: scene.capacity,
      vertexSource: scene.mesh.vertices, indexSource: scene.mesh.indices, uvSource: scene.mesh.uvs, tangentSource: scene.mesh.tangents });
    return binding;
  }
  validateSources(): void {
    for (const entry of this.entries.values()) {
      const scene = entry.scene;
      if (scene.geometryLod !== entry.source || scene.mesh !== entry.source!.source || scene.mesh.vertices !== entry.vertexSource
        || scene.mesh.indices !== entry.indexSource || scene.mesh.uvs !== entry.uvSource || scene.mesh.tangents !== entry.tangentSource
        || scene.deformation.jointCount !== 0 || scene.deformation.nodes.length || scene.morphTargets.length
        || scene.capacity !== entry.capacity || scene.instances.length > entry.capacity) throw new Error('GEOMETRY_LOD_ASSET_CHANGED');
    }
  }
  /** Called before the frame is submitted; returns actual uniform upload bytes. */
  update(frame: number, vp: Mat4, eye: readonly number[], projectionScale: number): number {
    if (![...vp, ...eye, projectionScale].every(v => Number.isFinite(v) && Number.isFinite(Math.fround(v))) || eye.length !== 3 || projectionScale <= 0) {
      throw new Error('GEOMETRY_LOD_VIEW');
    }
    let uploaded = 96;
    if (!this.settings.frozen || !this.hasView) {
      this.viewData.set(vp); this.viewData.set(eye, 16); this.viewData[20] = projectionScale; this.hasView = true;
    }
    this.viewData[21] = 0.1; this.viewData[22] = this.settings.maxErrorPixels; this.viewData[23] = Number(this.settings.frustumCulling);
    this.device.queue.writeBuffer(this.view, 0, this.viewData);
    for (const entry of this.entries.values()) {
      const scene = entry.scene;
      const count = entry.enabled() ? scene.instances.length : 0;
      if (count !== entry.count) {
        entry.paramsData[4] = count; this.device.queue.writeBuffer(entry.params, 0, entry.paramsData); uploaded += 32; entry.count = count;
      }
    }
    this.frame = frame;
    return uploaded;
  }
  encode(encoder: GPUCommandEncoder, timestamp?: GPUComputePassTimestampWrites): void {
    for (const entry of this.entries.values()) encoder.clearBuffer(entry.binding.arguments);
    const select = encoder.beginComputePass({ label: 'GPU LOD selection and index compaction', timestampWrites: timestamp ? {
      querySet: timestamp.querySet, beginningOfPassWriteIndex: timestamp.beginningOfPassWriteIndex,
    } : undefined });
    select.setPipeline(this.select);
    for (const entry of this.entries.values()) {
      if (!entry.count) continue;
      select.setBindGroup(0, entry.compute); select.dispatchWorkgroups(Math.ceil(entry.packed.clusterCount * entry.count / 64));
    }
    select.end();
    const finalize = encoder.beginComputePass({ label: 'LOD indirect argument guard', timestampWrites: timestamp ? {
      querySet: timestamp.querySet, endOfPassWriteIndex: timestamp.endOfPassWriteIndex,
    } : undefined });
    finalize.setPipeline(this.finalize);
    for (const entry of this.entries.values()) {
      if (!entry.count) continue;
      finalize.setBindGroup(0, entry.compute); finalize.dispatchWorkgroups(1);
    }
    finalize.end();
  }
  async probe(name: string): Promise<GeometryLodProbe> {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (this.frame === 0) throw new Error('GEOMETRY_LOD_NO_FRAME');
    if (this.probing) throw new Error('GEOMETRY_LOD_PROBE_BUSY');
    const entry = this.entries.get(name);
    if (!entry) throw new Error('GEOMETRY_LOD_BATCH_NOT_FOUND');
    const frame = this.frame, count = entry.count, flags = entry.packed.clusterCount * count;
    const size = 32 + flags * 4;
    const read = this.device.createBuffer({ label: 'explicit LOD diagnostic readback', size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.probing = true; this.readback = read; this.track(size);
    try {
      const encoder = this.device.createCommandEncoder({ label: 'explicit LOD diagnostic' });
      encoder.copyBufferToBuffer(entry.binding.arguments, 0, read, 0, 32);
      if (flags) encoder.copyBufferToBuffer(entry.selected, 0, read, 32, flags * 4);
      this.device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const data = new Uint32Array(read.getMappedRange()), selectedClusters: GeometryLodProbe['selectedClusters'] = [];
      const clusterView = new DataView(entry.packed.clusters);
      for (let index = 0; index < flags; index++) if (data[8 + index]) {
        const clusterId = index % entry.packed.clusterCount;
        selectedClusters.push({ clusterId, instanceIndex: Math.floor(index / entry.packed.clusterCount), level: clusterView.getUint32(clusterId * 64 + 56, true) });
      }
      const result = { frame, indices: data[0], triangles: data[0] / 3, selectedClusters, overflow: data[6] !== 0 };
      if (!result.overflow && selectedClusters.length !== data[5]) throw new Error('GEOMETRY_LOD_COUNTER_MISMATCH');
      return result;
    } finally {
      read.destroy();
      if (this.readback === read) { this.readback = undefined; this.track(-size); }
      this.probing = false;
    }
  }
  diagnostics() {
    return { settings: { ...this.settings }, frame: this.frame, residentBytes: this.bytes, budgetBytes: BUDGET,
      fullyResident: true, streaming: false, occlusionCulling: false, maxClusterInstances: MAX_PAIRS,
      batches: Array.from(this.entries, ([name, entry]) => ({ name, active: entry.enabled(), sourceTriangles: entry.packed.sourceTriangleCount,
        clusters: entry.packed.clusterCount, instances: entry.count, primitiveKey: entry.source!.primitiveKey })) };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.readback) { this.track(-this.readback.size); this.readback.destroy(); this.readback = undefined; }
    for (const buffer of this.buffers) buffer.destroy();
    this.track(-this.bytes); this.bytes = 0; this.entries.clear();
  }
}
