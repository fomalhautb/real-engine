import commonSource from './shaders/diffuse-gi-common.wgsl?raw';
import updateSource from './shaders/diffuse-gi-update.wgsl?raw';
import sampleSource from './shaders/diffuse-gi-sample.wgsl?raw';
import interpolateSource from './shaders/diffuse-gi-interpolate.wgsl?raw';
import { DIFFUSE_GI_PROBE_BYTES, DIFFUSE_GI_PROBE_FLOATS, DIFFUSE_GI_STATE_FLOAT_OFFSET,
  giRayDirection, resolveDiffuseGiOptions, type DiffuseGiOptions } from './diffuse-gi';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION, buildGiDistanceWeights } from './gi-visibility';
import { WORLD_HIT_BYTES } from './world-query';
import type { GpuWorldQuery } from './gpu-world-query';

const constants = `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`;
const giCommon = commonSource.replace('// GI_DISTANCE_CONSTANTS', constants);
const giSource = (source: string) => source.replace('// GI_COMMON', giCommon).replace('// GI_INTERPOLATE', interpolateSource);
export const diffuseGiSampleSource = giSource(sampleSource);
export interface DiffuseGiProbeResult {
  frame: number;
  invalidProbes: number;
  probes: { coefficients: number[][]; moments: number[][]; maxRadiance: number[]; valid: boolean; primaryOverflows: number; shadowOverflows: number }[];
}

/** Fixed world-space probes with optional cross-frame diffuse transport. Each
 * complete update reads the preceding field and writes a separate buffer. */
export class GpuDiffuseGi {
  readonly options: ReturnType<typeof resolveDiffuseGiOptions>;
  readonly surfaceLayout!: GPUBindGroupLayout;
  private readonly buffers: GPUBuffer[] = [];
  private parameters!: GPUBuffer;
  private primary!: GPUBuffer;
  private distanceWeights!: GPUBuffer;
  private probeBuffers!: [GPUBuffer, GPUBuffer];
  private surfaceGroups!: GPUBindGroup[];
  private currentIndex = 1;
  private readback!: GPUBuffer;
  private computeLayout!: GPUBindGroupLayout;
  private computeGroups!: GPUBindGroup[];
  private generate!: GPUComputePipeline;
  private generateShadows!: GPUComputePipeline;
  private integrate!: GPUComputePipeline;
  private integrateVisibility!: GPUComputePipeline;
  private bytes = 0;
  private frame = 0;
  private pending = false;
  private disposed = false;
  private enabled = true;
  private multiBounce: boolean;
  private historyUpdates = 0;
  private lastHistoryFrame = 0;
  private historyResetReason = 'initial';
  private resetRequested: string | null = 'initial';
  private previousDependencies?: { material: number; scene: number; sun: number[] };

  get surfaceGroup(): GPUBindGroup { return this.surfaceGroups[this.currentIndex]; }

  static async create(device: GPUDevice, queries: GpuWorldQuery, options: DiffuseGiOptions,
    account: (bytes: number) => void): Promise<GpuDiffuseGi> {
    const resolved = resolveDiffuseGiOptions(options);
    if (device.limits.maxStorageBuffersPerShaderStage < 6) throw new Error('DIFFUSE_GI_STORAGE_BINDING_LIMIT');
    if (resolved.rayCount > queries.options.maxRays) throw new Error('DIFFUSE_GI_QUERY_RAY_BUDGET');
    if (resolved.rayCount > queries.options.maxTotalWorkUnits) throw new Error('DIFFUSE_GI_QUERY_WORK_BUDGET');
    const sizes = [112, resolved.rayCount * WORLD_HIT_BYTES, resolved.probeCount * DIFFUSE_GI_PROBE_BYTES,
      resolved.probeCount * DIFFUSE_GI_PROBE_BYTES,
      resolved.probeCount * DIFFUSE_GI_PROBE_BYTES,
      GI_DISTANCE_CELLS * resolved.raysPerProbe * Float32Array.BYTES_PER_ELEMENT];
    if (sizes.some(size => size > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize))
      || sizes.reduce((a, b) => a + b, 0) > resolved.maxBytes) throw new Error('DIFFUSE_GI_MEMORY_BUDGET');
    const result = new GpuDiffuseGi(device, queries, resolved, account);
    try { await result.initialize(); return result; }
    catch (error) { result.dispose(); throw error; }
  }
  private constructor(private readonly device: GPUDevice, private readonly queries: GpuWorldQuery,
    options: ReturnType<typeof resolveDiffuseGiOptions>, private readonly account: (bytes: number) => void) {
    this.options = options;
    this.multiBounce = options.multiBounce;
  }
  private initializeResources(): void {
    const { device, queries, options } = this;
    this.parameters = this.buffer('diffuse GI parameters', 112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.primary = this.buffer('diffuse GI primary hits', options.rayCount * WORLD_HIT_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const probeBytes = options.probeCount * DIFFUSE_GI_PROBE_BYTES;
    this.probeBuffers = [this.buffer('diffuse GI field A', probeBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
      this.buffer('diffuse GI field B', probeBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)];
    this.readback = this.buffer('diffuse GI diagnostics', probeBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const weights = buildGiDistanceWeights(Array.from({ length: options.raysPerProbe }, (_, index) =>
      giRayDirection(index, options.raysPerProbe)));
    this.distanceWeights = this.buffer('diffuse GI octahedral capture weights', weights.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.distanceWeights, 0, weights);
    this.computeLayout = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 1 || binding === 4 ? 'storage' : 'read-only-storage' },
    })) });
    this.computeGroups = this.probeBuffers.map((output, index) => device.createBindGroup({ layout: this.computeLayout,
      entries: [this.parameters, queries.rayBuffer, this.primary, queries.hitBuffer, output, this.probeBuffers[1 - index], this.distanceWeights]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    const surfaceLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ] });
    this.surfaceGroups = this.probeBuffers.map(buffer => device.createBindGroup({ layout: surfaceLayout, entries: [
      { binding: 0, resource: { buffer: this.parameters } }, { binding: 1, resource: { buffer } },
    ] }));
    Object.assign(this, { surfaceLayout });
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size, usage });
    this.buffers.push(buffer); this.bytes += size; this.account(size);
    return buffer;
  }
  private async initialize(): Promise<void> {
    // Allocate inside create()'s try/finally ownership boundary.
    this.initializeResources();
    const module = this.device.createShaderModule({ label: 'diffuse GI', code: giSource(updateSource) });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `diffuse GI:${error.lineNum}: ${error.message}`).join('\n'));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    [this.generate, this.generateShadows, this.integrate, this.integrateVisibility] = await Promise.all(
      ['generate', 'generateShadows', 'integrate', 'integrateVisibility'].map(entryPoint =>
      this.device.createComputePipelineAsync({ label: 'diffuse GI ' + entryPoint, layout, compute: { module, entryPoint } })));
  }
  setEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('DIFFUSE_GI_INVALID_ENABLED');
    if (this.enabled !== enabled) this.resetRequested = 'enabled-changed';
    this.enabled = enabled;
  }
  setMultiBounce(enabled: boolean): void {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('DIFFUSE_GI_INVALID_MULTI_BOUNCE');
    if (this.multiBounce !== enabled) this.resetRequested = 'mode-changed';
    this.multiBounce = enabled;
  }
  invalidateHistory(): void {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    this.resetRequested = 'explicit';
  }
  update(frame: number, direction: readonly number[], color: readonly number[], materialRevision: number, sceneRevision: number): number {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    const sun = [...direction, ...color], previous = this.previousDependencies;
    const reset = this.resetRequested ?? (previous && (
      previous.material !== materialRevision ? 'material-changed' : previous.scene !== sceneRevision ? 'scene-discontinuity'
        : sun.some((value, i) => value !== previous.sun[i]) ? 'sun-changed' : null));
    if (reset) {
      this.historyUpdates = 0;
      this.historyResetReason = reset;
    }
    this.resetRequested = null;
    this.previousDependencies = { material: materialRevision, scene: sceneRevision, sun };
    this.lastHistoryFrame = this.enabled && this.multiBounce && this.historyUpdates > 0 ? this.frame : 0;
    const data = new Float32Array(28), integers = new Uint32Array(data.buffer), o = this.options;
    data.set([...o.origin, this.enabled ? 1 : 0]); data.set([...o.spacing, o.maxDistance], 4);
    integers.set([...o.counts, o.raysPerProbe], 8);
    data.set([...direction, o.normalBias], 12); data.set(color, 16); data.set(o.skyRadiance, 20);
    data[23] = o.visibilityWeightThreshold;
    integers[24] = frame;
    integers[25] = this.lastHistoryFrame;
    integers[26] = this.lastHistoryFrame ? 1 : 0;
    this.device.queue.writeBuffer(this.parameters, 0, data);
    return data.byteLength;
  }
  encode(encoder: GPUCommandEncoder, frame: number, materialRevision: number, timing?: GPUComputePassTimestampWrites): number {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    if (!this.enabled) { encoder.beginComputePass({ label: 'disabled diffuse GI', timestampWrites: timing }).end(); return 0; }
    const count = this.options.rayCount;
    const outputIndex = 1 - this.currentIndex;
    const dispatch = (label: string, pipeline: GPUComputePipeline, count: number, timestampWrites?: GPUComputePassTimestampWrites) => {
      const pass = encoder.beginComputePass({ label, timestampWrites });
      pass.setPipeline(pipeline); pass.setBindGroup(0, this.computeGroups[outputIndex]);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    };
    dispatch('generate world-space GI rays', this.generate, count, timing && { querySet: timing.querySet, beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex });
    const uploaded = this.queries.encodeTrace(encoder, count, 0, materialRevision);
    encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.primary, 0, count * WORLD_HIT_BYTES);
    dispatch('generate GI sun visibility rays', this.generateShadows, count);
    this.queries.encodeTrace(encoder, count, 0, materialRevision);
    dispatch('integrate world-space GI radiance', this.integrate, this.options.probeCount);
    dispatch('integrate GI directional visibility', this.integrateVisibility, this.options.probeCount * GI_DISTANCE_CELLS,
      timing && { querySet: timing.querySet, endOfPassWriteIndex: timing.endOfPassWriteIndex });
    this.frame = frame;
    this.currentIndex = outputIndex;
    this.historyUpdates = this.multiBounce ? this.historyUpdates + 1 : 1;
    return uploaded;
  }
  async probe(): Promise<DiffuseGiProbeResult> {
    if (this.disposed || !this.frame) throw new Error('DIFFUSE_GI_NO_FRAME');
    if (this.pending) throw new Error('DIFFUSE_GI_READBACK_PENDING');
    this.pending = true;
    const frame = this.frame;
    try {
      const encoder = this.device.createCommandEncoder({ label: 'diffuse GI diagnostics' });
      const probesBuffer = this.probeBuffers[this.currentIndex];
      encoder.copyBufferToBuffer(probesBuffer, 0, this.readback, 0, probesBuffer.size);
      this.device.queue.submit([encoder.finish()]);
      await this.readback.mapAsync(GPUMapMode.READ);
      try {
        const data = new Float32Array(this.readback.getMappedRange()), u32 = new Uint32Array(data.buffer);
        const probes: DiffuseGiProbeResult['probes'] = [];
        for (let i = 0; i < this.options.probeCount; i++) {
          const base = i * DIFFUSE_GI_PROBE_FLOATS, state = base + DIFFUSE_GI_STATE_FLOAT_OFFSET;
          if (u32[state + 1] !== frame) throw new Error('DIFFUSE_GI_FRAME_MISMATCH');
          const coefficients = Array.from({ length: 9 }, (_, j) => Array.from(data.slice(base + j * 4, base + j * 4 + 3)));
          const moments = Array.from({ length: GI_DISTANCE_CELLS }, (_, j) => Array.from(data.slice(base + 36 + j * 4, base + 39 + j * 4)));
          const maxRadiance = [data[base + 3], data[base + 7], data[base + 11]];
          if (![...coefficients.flat(), ...moments.flat(), ...maxRadiance].every(Number.isFinite)) throw new Error('DIFFUSE_GI_NONFINITE');
          probes.push({ coefficients, moments, maxRadiance, valid: u32[state] === 1,
            primaryOverflows: u32[state + 2], shadowOverflows: u32[state + 3] });
        }
        return { frame, probes, invalidProbes: probes.filter(probe => !probe.valid).length };
      } finally { this.readback.unmap(); }
    } finally { this.pending = false; }
  }
  diagnostics() {
    return { enabled: this.enabled, frame: this.frame, options: this.options, ownedGpuBytes: this.bytes,
      multiBounce: this.multiBounce, historyUpdates: this.historyUpdates, lastHistoryFrame: this.lastHistoryFrame,
      historyResetReason: this.historyResetReason, transport: this.multiBounce ? 'cross-frame-diffuse-feedback' : 'single-diffuse-bounce',
      capture: 'diffuse-and-emissive', update: 'full-grid-every-frame', representation: 'radiance-SH9',
      visibility: 'octahedral-distance-moments', visibilityResolution: GI_DISTANCE_RESOLUTION,
      visibilityCapture: 'normalized-cosine-power-32', distanceCaptureSamplesPerUpdate: GI_DISTANCE_CELLS * this.options.rayCount,
      placement: 'caller-provided-free-space-grid',
      queriesPerUpdate: this.options.rayCount * 2, maxWorkUnitsPerUpdate: this.queries.options.maxTotalWorkUnits * 2,
      temporalAccumulation: false, automaticRelocation: false, reflections: false };
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
