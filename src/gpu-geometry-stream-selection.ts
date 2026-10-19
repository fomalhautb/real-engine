import source from './shaders/geometry-stream-select.wgsl?raw';
import { packGeometryStreamSelection } from './geometry-stream-selection-data';
import type { GpuGeometryPagePool } from './gpu-geometry-page-pool';
import type { GeometryLodSettings } from './gpu-geometry-lod';
import type { Mat4 } from './math';
import type { GeometryGroupDemand } from './geometry-stream-working-set';
import { occlusionTestSource, type GpuOcclusion } from './gpu-occlusion';

/** Includes persistent buffers and the maximum one-at-a-time feedback buffer. */
export interface GeometryStreamSelectionOptions { maxGpuBytes?: number }
export interface GeometryStreamSelectionFeedback {
  frame: number;
  requestedPages: { pageId: number; priority: number }[];
  /** Visible group demand independent of residency; used only for bounded allocation. */
  desiredGroups: GeometryGroupDemand[];
  indices: number;
  overflow: boolean;
  /** Previous-frame candidates, current-frame rejects, and recovered draws. */
  occlusion: { deferredClusters: number; culledClusters: number; recoveredClusters: number };
  /** Present only for explicit diagnostics, never required for page scheduling. */
  selectedClusters?: { clusterId: number; instanceIndex: number; level: number; visible: boolean }[];
}

/** GPU-resident complete group cut, request generation, culling and physical
 * index compaction. Caller holds a pool frame lease over all resident pages
 * before encoding; no CPU geometry or selection readback drives drawing. */
export class GpuGeometryStreamSelection {
  readonly indices: GPUBuffer;
  readonly arguments: GPUBuffer;
  readonly physicalVertexCount: number;
  readonly settings: Required<GeometryLodSettings> = { maxErrorPixels: 1, frustumCulling: true, frozen: false, debug: 'material' };
  private readonly packed;
  private readonly buffers: GPUBuffer[] = [];
  private readonly view: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly selected: GPUBuffer;
  private readonly requests: GPUBuffer;
  private readonly selectGroup: GPUBindGroup;
  private readonly compactGroup: GPUBindGroup;
  private readonly selectLayout: GPUBindGroupLayout;
  private readonly compactLayout: GPUBindGroupLayout;
  private readonly parameterData: Uint32Array<ArrayBuffer>;
  private readonly parameterStride: number;
  private readonly viewData = new Float32Array(24);
  private pipelines!: Record<'demand' | 'initialize' | 'refine' | 'compact' | 'finalize', GPUComputePipeline>
    & Partial<Record<'recover' | 'finalizeRecovery', GPUComputePipeline>>;
  private hasView = false;
  private count = 0;
  private frame = 0;
  private encodedFrame = 0;
  private recoveredFrame = 0;
  private bytes = 0;
  private disposed = false;
  private reading?: GPUBuffer;
  private error: string | null = null;
  private readonly feedbackBudget: number;
  private readonly maxGpuBytes: number;

  static async create(device: GPUDevice, pool: GpuGeometryPagePool, instances: GPUBuffer, capacity: number,
    track: (bytes: number) => void = () => {}, options: GeometryStreamSelectionOptions = {}, occlusion?: GpuOcclusion) {
    let result: GpuGeometryStreamSelection | undefined;
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    try {
      result = new GpuGeometryStreamSelection(device, pool, instances, capacity, track, options, occlusion);
      const stub = 'fn geometryOccluded(center:vec3f,extent:vec3f)->bool { return false; }';
      const module = device.createShaderModule({ label: 'resident-page GPU selection',
        code: (occlusion ? occlusionTestSource : stub) + '\n' + source });
      const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(e => `GEOMETRY_STREAM_SHADER:${e.lineNum}: ${e.message}`).join('\n'));
      const pipelines = {} as GpuGeometryStreamSelection['pipelines'];
      const entries = ['demand', 'initialize', 'refine', 'compact', 'finalize',
        ...(occlusion ? ['recover', 'finalizeRecovery'] as const : [])] as const;
      for (const entryPoint of entries) {
        const layout = ['demand', 'initialize', 'refine'].includes(entryPoint) ? result.selectLayout : result.compactLayout;
        const occlusionPass = !!occlusion && (entryPoint === 'compact' || entryPoint === 'recover');
        pipelines[entryPoint] = await device.createComputePipelineAsync({ label: 'stream geometry ' + entryPoint,
          layout: device.createPipelineLayout({ bindGroupLayouts: occlusionPass ? [layout, occlusion.testLayout] : [layout] }),
          compute: { module, entryPoint } });
      }
      result.pipelines = pipelines;
    } catch (error) {
      try { result?.dispose(); } catch { /* preserve initialization failure */ }
      await Promise.allSettled([device.popErrorScope(), device.popErrorScope()]); throw error;
    }
    try {
      const errors = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
      if (errors.some(Boolean)) throw new Error('GEOMETRY_STREAM_INITIALIZATION: ' + errors.find(Boolean)!.message);
      result.live(); return result;
    } catch (error) { try { result.dispose(); } catch { /* preserve initialization failure */ } throw error; }
  }

  private constructor(private readonly device: GPUDevice, readonly pool: GpuGeometryPagePool, instances: GPUBuffer,
    readonly capacity: number, private readonly track: (bytes: number) => void,
    options: GeometryStreamSelectionOptions, private readonly occlusion?: GpuOcclusion) {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 256) throw new Error('GEOMETRY_STREAM_INSTANCE_BUDGET');
    this.packed = packGeometryStreamSelection(pool.directory);
    const directory = this.packed.directory, pairs = directory.clusters.length * capacity;
    this.physicalVertexCount = pool.slotCount * pool.verticesPerSlot;
    const maxIndices = directory.sourceTriangleCount * 3 * capacity;
    if (pairs > 262144 || Math.ceil(pairs / 64) > device.limits.maxComputeWorkgroupsPerDimension
      || directory.groups.length * capacity > 262144 || this.physicalVertexCount * capacity > 0xffffffff
      || maxIndices > 0xffffffff) throw new Error('GEOMETRY_STREAM_WORK_BUDGET');
    this.parameterStride = Math.ceil(48 / device.limits.minUniformBufferOffsetAlignment) * device.limits.minUniformBufferOffsetAlignment;
    this.parameterData = new Uint32Array(this.parameterStride / 4 * (this.packed.levels.length + 1));
    for (let slot = 0; slot <= this.packed.levels.length; slot++) {
      const level = this.packed.levels[slot - 1];
      this.parameterData.set([directory.clusters.length, this.physicalVertexCount, 0, maxIndices,
        level?.firstGroup ?? 0, level?.groupCount ?? directory.groups.length, this.packed.clusterOffset, this.packed.groupOffset,
        this.packed.edgeOffset, directory.pages.length, level?.level ?? 0, 0], slot * this.parameterStride / 4);
    }
    const sizes = [96, this.parameterData.byteLength, this.packed.words.byteLength, pairs * 4,
      (directory.pages.length + directory.groups.length) * 4, maxIndices * 4, 64];
    const budget = options.maxGpuBytes ?? 128 * 1024 * 1024;
    this.feedbackBudget = 64 + sizes[4] + pairs * 4;
    this.maxGpuBytes = budget;
    if (!Number.isSafeInteger(budget) || budget < 1 || budget > 256 * 1024 * 1024
      || sizes.some(size => size > device.limits.maxBufferSize || size > device.limits.maxStorageBufferBindingSize)
      || this.feedbackBudget > device.limits.maxBufferSize
      || sizes.reduce((a, b) => a + b, 0) + this.feedbackBudget > budget) throw new Error('GEOMETRY_STREAM_RESOURCE_BUDGET');
    try {
      this.view = this.buffer('stream selection view', sizes[0], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.params = this.buffer('immutable per-level stream parameters', sizes[1], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const metadata = this.buffer('stream topology metadata', sizes[2], GPUBufferUsage.STORAGE, this.packed.words.buffer);
      this.selected = this.buffer('stream cut and visible flags', sizes[3], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      this.requests = this.buffer('stream page requests and group demand', sizes[4], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      this.indices = this.buffer('stream selected physical indices', sizes[5], GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_SRC);
      this.arguments = this.buffer('stream indirect draw arguments', sizes[6], GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const layouts = (bindings: number[]) => device.createBindGroupLayout({ entries: bindings.map(binding => ({ binding,
        visibility: GPUShaderStage.COMPUTE, buffer: { type: binding < 2 ? 'uniform' : [5, 6, 9, 10].includes(binding) ? 'storage' : 'read-only-storage',
          ...(binding === 1 ? { hasDynamicOffset: true, minBindingSize: 48 } : {}) } })) });
      this.selectLayout = layouts([0, 1, 2, 3, 4, 5, 6]);
      this.compactLayout = layouts([0, 1, 2, 3, 4, 5, 7, 8, 9, 10]);
      const buffers = [this.view, this.params, metadata, instances, pool.pageResidency, this.selected,
        this.requests, pool.clusterLocations, pool.indices, this.indices, this.arguments];
      const group = (layout: GPUBindGroupLayout, bindings: number[]) => device.createBindGroup({ layout,
        entries: bindings.map(binding => ({ binding, resource: { buffer: buffers[binding], ...(binding === 1 ? { size: 48 } : {}) } })) });
      this.selectGroup = group(this.selectLayout, [0, 1, 2, 3, 4, 5, 6]);
      this.compactGroup = group(this.compactLayout, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]);
    } catch (error) { try { this.dispose(); } catch { /* preserve allocation failure */ } throw error; }
    void device.lost.then(() => {
      this.error = 'GEOMETRY_STREAM_DEVICE_LOST';
      try { this.dispose(); } catch (error) { this.error += '; cleanup: ' + String(error); }
    });
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size, usage, mappedAtCreation: !!data });
    this.buffers.push(buffer); this.bytes += size; this.track(size);
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data)); buffer.unmap(); } return buffer;
  }
  private live() { if (this.error) throw new Error(this.error); if (this.disposed) throw new Error('GEOMETRY_STREAM_SELECTION_DISPOSED'); }
  setSettings(next: GeometryLodSettings) {
    this.live();
    if (next.maxErrorPixels !== undefined && (!Number.isFinite(next.maxErrorPixels) || next.maxErrorPixels < 0 || next.maxErrorPixels > 10000)
      || next.frustumCulling !== undefined && typeof next.frustumCulling !== 'boolean'
      || next.frozen !== undefined && typeof next.frozen !== 'boolean'
      || next.debug !== undefined && !['material', 'clusters', 'levels'].includes(next.debug)) throw new Error('GEOMETRY_LOD_SETTINGS');
    for (const key of ['maxErrorPixels', 'frustumCulling', 'frozen', 'debug'] as const) {
      if (next[key] !== undefined) Object.assign(this.settings, { [key]: next[key] });
    }
  }
  update(frame: number, count: number, vp: Mat4, eye: readonly number[], projectionScale: number) {
    this.live();
    if (!Number.isSafeInteger(frame) || frame <= this.frame || !Number.isSafeInteger(count) || count < 0 || count > this.capacity) throw new Error('GEOMETRY_STREAM_FRAME');
    if (vp.length !== 16 || eye.length !== 3 || ![...vp, ...eye, projectionScale].every(v => Number.isFinite(v) && Number.isFinite(Math.fround(v)))
      || projectionScale <= 0) throw new Error('GEOMETRY_STREAM_VIEW');
    if (!this.settings.frozen || !this.hasView) { this.viewData.set(vp); this.viewData.set(eye, 16); this.viewData[20] = projectionScale; this.hasView = true; }
    this.viewData[21] = 0.1; this.viewData[22] = this.settings.maxErrorPixels; this.viewData[23] = Number(this.settings.frustumCulling);
    for (let slot = 0; slot <= this.packed.levels.length; slot++) this.parameterData[slot * this.parameterStride / 4 + 2] = count;
    this.device.queue.writeBuffer(this.view, 0, this.viewData); this.device.queue.writeBuffer(this.params, 0, this.parameterData);
    this.frame = frame; this.count = count; return 96 + this.parameterData.byteLength;
  }
  encode(encoder: GPUCommandEncoder) {
    this.live();
    if (!this.frame || this.encodedFrame === this.frame) throw new Error('GEOMETRY_STREAM_SELECTION_FRAME');
    if (!this.pool.diagnostics().rootsReady) throw new Error('GEOMETRY_STREAM_ROOTS_NOT_READY');
    encoder.clearBuffer(this.arguments); encoder.clearBuffer(this.requests);
    const dispatch = (entry: keyof GpuGeometryStreamSelection['pipelines'], work: number, offset = 0) => {
      if (!work) return;
      const pass = encoder.beginComputePass({ label: 'stream geometry ' + entry }); pass.setPipeline(this.pipelines[entry]!);
      pass.setBindGroup(0, ['demand', 'initialize', 'refine'].includes(entry) ? this.selectGroup : this.compactGroup, [offset]);
      if (entry === 'compact' && this.occlusion) pass.setBindGroup(1, this.occlusion.previousGroup);
      pass.dispatchWorkgroups(entry === 'finalize' ? 1 : Math.ceil(work / 64)); pass.end();
    };
    dispatch('demand', this.packed.directory.groups.length * this.count);
    dispatch('initialize', this.packed.directory.clusters.length * this.count);
    this.packed.levels.forEach((level, index) => dispatch('refine', level.groupCount * this.count, (index + 1) * this.parameterStride));
    dispatch('compact', this.packed.directory.clusters.length * this.count); dispatch('finalize', this.count);
    this.encodedFrame = this.frame; this.recoveredFrame = this.occlusion ? 0 : this.frame;
  }
  /** Recheck deferred candidates against the current frame's completed Hi-Z. */
  encodeRecovery(encoder: GPUCommandEncoder): void {
    this.live();
    if (!this.occlusion) return;
    if (!this.frame || this.encodedFrame !== this.frame || this.recoveredFrame === this.frame)
      throw new Error('GEOMETRY_STREAM_RECOVERY_FRAME');
    const work = this.packed.directory.clusters.length * this.count;
    if (work) {
      const pass = encoder.beginComputePass({ label: 'stream geometry recover' });
      pass.setPipeline(this.pipelines.recover!); pass.setBindGroup(0, this.compactGroup, [0]);
      pass.setBindGroup(1, this.occlusion.currentGroup);
      pass.dispatchWorkgroups(Math.ceil(work / 64)); pass.end();
      const finalize = encoder.beginComputePass({ label: 'stream geometry finalizeRecovery' });
      finalize.setPipeline(this.pipelines.finalizeRecovery!); finalize.setBindGroup(0, this.compactGroup, [0]);
      finalize.dispatchWorkgroups(1); finalize.end();
    }
    this.recoveredFrame = this.frame;
  }
  /** Bounded async page IDs/counters. Include cut flags only for explicit diagnostics. */
  async feedback(includeSelected = false): Promise<GeometryStreamSelectionFeedback> {
    this.live(); if (!this.encodedFrame) throw new Error('GEOMETRY_STREAM_NO_FRAME');
    if (this.occlusion && this.recoveredFrame !== this.encodedFrame) throw new Error('GEOMETRY_STREAM_RECOVERY_PENDING');
    if (this.reading) throw new Error('GEOMETRY_STREAM_FEEDBACK_BUSY');
    const frame = this.encodedFrame, directory = this.packed.directory, pairs = directory.clusters.length * this.count;
    const requestBytes = (directory.pages.length + directory.groups.length) * 4;
    const bytes = 64 + requestBytes + (includeSelected ? pairs * 4 : 0);
    const read = this.device.createBuffer({ label: 'bounded stream requests and counters', size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.reading = read;
    try {
      this.track(bytes);
      const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.arguments, 0, read, 0, 64);
      encoder.copyBufferToBuffer(this.requests, 0, read, 64, requestBytes);
      if (includeSelected && pairs) encoder.copyBufferToBuffer(this.selected, 0, read, 64 + requestBytes, pairs * 4);
      this.device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(read.getMappedRange()), requestedPages: GeometryStreamSelectionFeedback['requestedPages'] = [];
      for (let id = 0; id < directory.pages.length; id++) if (values[16 + id]) requestedPages.push({ pageId: id, priority: values[16 + id] - 1 });
      requestedPages.sort((a, b) => b.priority - a.priority || a.pageId - b.pageId);
      const floats = new Float32Array(values.buffer), desiredGroups: GeometryGroupDemand[] = [];
      for (let ordinal = 0; ordinal < directory.groups.length; ordinal++) {
        const projectedError = floats[16 + directory.pages.length + ordinal];
        if (projectedError > 0) desiredGroups.push({ groupId: this.packed.groupIds[ordinal], projectedError });
      }
      const selectedClusters = includeSelected ? [] as NonNullable<GeometryStreamSelectionFeedback['selectedClusters']> : undefined;
      if (selectedClusters) for (let i = 0; i < pairs; i++) {
        const flag = values[16 + requestBytes / 4 + i], clusterId = i % directory.clusters.length;
        if (flag) selectedClusters.push({ clusterId, instanceIndex: Math.floor(i / directory.clusters.length), level: directory.clusters[clusterId].level, visible: flag === 2 });
      }
      return { frame, requestedPages, desiredGroups, indices: values[0] + values[8],
        overflow: values[6] !== 0 || values[14] !== 0,
        occlusion: { deferredClusters: values[7], culledClusters: values[15], recoveredClusters: values[13] },
        selectedClusters };
    } finally { read.destroy(); if (this.reading === read) { this.reading = undefined; this.track(-bytes); } }
  }
  diagnostics() { return { frame: this.encodedFrame, instances: this.count, gpuBytes: this.bytes + (this.reading?.size ?? 0), disposed: this.disposed,
    error: this.error, maxGpuBytes: this.maxGpuBytes, maxFeedbackBytes: this.feedbackBudget,
    physicalVertexCount: this.physicalVertexCount, levels: this.packed.levels.length, feedbackPending: !!this.reading,
    occlusion: { enabled: !!this.occlusion, stages: this.occlusion ? 2 : 1,
      recoveredFrame: this.recoveredFrame }, settings: { ...this.settings } }; }
  dispose() {
    if (this.disposed) return; this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy(); this.buffers.length = 0;
    let released = this.bytes; this.bytes = 0;
    if (this.reading) { released += this.reading.size; this.reading.destroy(); this.reading = undefined; }
    this.track(-released);
  }
}
