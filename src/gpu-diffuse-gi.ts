import commonSource from './shaders/diffuse-gi-common.wgsl?raw';
import updateSource from './shaders/diffuse-gi-update.wgsl?raw';
import sampleSource from './shaders/diffuse-gi-sample.wgsl?raw';
import { DIFFUSE_GI_PROBE_BYTES, resolveDiffuseGiOptions, type DiffuseGiOptions } from './diffuse-gi';
import { WORLD_HIT_BYTES } from './world-query';
import type { GpuWorldQuery } from './gpu-world-query';

export const diffuseGiSampleSource = sampleSource.replace('// GI_COMMON', commonSource);
export interface DiffuseGiProbeResult {
  frame: number;
  invalidProbes: number;
  probes: { coefficients: number[][]; moments: number[][]; valid: boolean; primaryOverflows: number; shadowOverflows: number }[];
}

/** Fixed world-space radiance probes. One diffuse surface bounce, refreshed in
 * full each rendered frame; no screen input, temporal history or readback in lighting. */
export class GpuDiffuseGi {
  readonly options: ReturnType<typeof resolveDiffuseGiOptions>;
  readonly surfaceLayout!: GPUBindGroupLayout;
  readonly surfaceGroup!: GPUBindGroup;
  private readonly buffers: GPUBuffer[] = [];
  private parameters!: GPUBuffer;
  private primary!: GPUBuffer;
  private probes!: GPUBuffer;
  private readback!: GPUBuffer;
  private computeLayout!: GPUBindGroupLayout;
  private computeGroup!: GPUBindGroup;
  private generate!: GPUComputePipeline;
  private generateShadows!: GPUComputePipeline;
  private integrate!: GPUComputePipeline;
  private bytes = 0;
  private frame = 0;
  private pending = false;
  private disposed = false;
  private enabled = true;

  static async create(device: GPUDevice, queries: GpuWorldQuery, options: DiffuseGiOptions,
    account: (bytes: number) => void): Promise<GpuDiffuseGi> {
    const resolved = resolveDiffuseGiOptions(options);
    if (resolved.rayCount > queries.options.maxRays) throw new Error('DIFFUSE_GI_QUERY_RAY_BUDGET');
    if (resolved.rayCount > queries.options.maxTotalWorkUnits) throw new Error('DIFFUSE_GI_QUERY_WORK_BUDGET');
    const sizes = [112, resolved.rayCount * WORLD_HIT_BYTES, resolved.probeCount * DIFFUSE_GI_PROBE_BYTES,
      resolved.probeCount * DIFFUSE_GI_PROBE_BYTES];
    if (sizes.some(size => size > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize))
      || sizes.reduce((a, b) => a + b, 0) > resolved.maxBytes) throw new Error('DIFFUSE_GI_MEMORY_BUDGET');
    const result = new GpuDiffuseGi(device, queries, resolved, account);
    try { await result.initialize(); return result; }
    catch (error) { result.dispose(); throw error; }
  }
  private constructor(private readonly device: GPUDevice, private readonly queries: GpuWorldQuery,
    options: ReturnType<typeof resolveDiffuseGiOptions>, private readonly account: (bytes: number) => void) {
    this.options = options;
  }
  private initializeResources(): void {
    const { device, queries, options } = this;
    this.parameters = this.buffer('diffuse GI parameters', 112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.primary = this.buffer('diffuse GI primary hits', options.rayCount * WORLD_HIT_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.probes = this.buffer('diffuse GI radiance and visibility', options.probeCount * DIFFUSE_GI_PROBE_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.readback = this.buffer('diffuse GI diagnostics', this.probes.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    this.computeLayout = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 1 || binding === 4 ? 'storage' : 'read-only-storage' },
    })) });
    this.computeGroup = device.createBindGroup({ layout: this.computeLayout,
      entries: [this.parameters, queries.rayBuffer, this.primary, queries.hitBuffer, this.probes].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const surfaceLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ] });
    const surfaceGroup = device.createBindGroup({ layout: surfaceLayout, entries: [
      { binding: 0, resource: { buffer: this.parameters } }, { binding: 1, resource: { buffer: this.probes } },
    ] });
    Object.assign(this, { surfaceLayout, surfaceGroup });
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size, usage });
    this.buffers.push(buffer); this.bytes += size; this.account(size);
    return buffer;
  }
  private async initialize(): Promise<void> {
    // Allocate inside create()'s try/finally ownership boundary.
    this.initializeResources();
    const module = this.device.createShaderModule({ label: 'diffuse GI', code: updateSource.replace('// GI_COMMON', commonSource) });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `diffuse GI:${error.lineNum}: ${error.message}`).join('\n'));
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    [this.generate, this.generateShadows, this.integrate] = await Promise.all(['generate', 'generateShadows', 'integrate'].map(entryPoint =>
      this.device.createComputePipelineAsync({ label: 'diffuse GI ' + entryPoint, layout, compute: { module, entryPoint } })));
  }
  setEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('DIFFUSE_GI_INVALID_ENABLED');
    this.enabled = enabled;
  }
  update(frame: number, direction: readonly number[], color: readonly number[]): number {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    const data = new Float32Array(28), integers = new Uint32Array(data.buffer), o = this.options;
    data.set([...o.origin, this.enabled ? 1 : 0]); data.set([...o.spacing, o.maxDistance], 4);
    integers.set([...o.counts, o.raysPerProbe], 8);
    data.set([...direction, o.normalBias], 12); data.set(color, 16); data.set(o.skyRadiance, 20);
    integers[24] = frame;
    this.device.queue.writeBuffer(this.parameters, 0, data);
    return data.byteLength;
  }
  encode(encoder: GPUCommandEncoder, frame: number, materialRevision: number, timing?: GPUComputePassTimestampWrites): number {
    if (this.disposed) throw new Error('DIFFUSE_GI_DISPOSED');
    if (!this.enabled) { encoder.beginComputePass({ label: 'disabled diffuse GI', timestampWrites: timing }).end(); return 0; }
    const count = this.options.rayCount;
    const dispatch = (label: string, pipeline: GPUComputePipeline, count: number, timestampWrites?: GPUComputePassTimestampWrites) => {
      const pass = encoder.beginComputePass({ label, timestampWrites });
      pass.setPipeline(pipeline); pass.setBindGroup(0, this.computeGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    };
    dispatch('generate world-space GI rays', this.generate, count, timing && { querySet: timing.querySet, beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex });
    const uploaded = this.queries.encodeTrace(encoder, count, 0, materialRevision);
    encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.primary, 0, count * WORLD_HIT_BYTES);
    dispatch('generate GI sun visibility rays', this.generateShadows, count);
    this.queries.encodeTrace(encoder, count, 0, materialRevision);
    dispatch('integrate world-space GI radiance', this.integrate, this.options.probeCount,
      timing && { querySet: timing.querySet, endOfPassWriteIndex: timing.endOfPassWriteIndex });
    this.frame = frame;
    return uploaded;
  }
  async probe(): Promise<DiffuseGiProbeResult> {
    if (this.disposed || !this.frame) throw new Error('DIFFUSE_GI_NO_FRAME');
    if (this.pending) throw new Error('DIFFUSE_GI_READBACK_PENDING');
    this.pending = true;
    const frame = this.frame;
    try {
      const encoder = this.device.createCommandEncoder({ label: 'diffuse GI diagnostics' });
      encoder.copyBufferToBuffer(this.probes, 0, this.readback, 0, this.probes.size);
      this.device.queue.submit([encoder.finish()]);
      await this.readback.mapAsync(GPUMapMode.READ);
      try {
        const data = new Float32Array(this.readback.getMappedRange()), u32 = new Uint32Array(data.buffer);
        const probes: DiffuseGiProbeResult['probes'] = [];
        for (let i = 0; i < this.options.probeCount; i++) {
          const base = i * 64;
          if (u32[base + 61] !== frame) throw new Error('DIFFUSE_GI_FRAME_MISMATCH');
          const coefficients = Array.from({ length: 9 }, (_, j) => Array.from(data.slice(base + j * 4, base + j * 4 + 3)));
          const moments = Array.from({ length: 6 }, (_, j) => Array.from(data.slice(base + 36 + j * 4, base + 39 + j * 4)));
          if (![...coefficients.flat(), ...moments.flat()].every(Number.isFinite)) throw new Error('DIFFUSE_GI_NONFINITE');
          probes.push({ coefficients, moments, valid: u32[base + 60] === 1,
            primaryOverflows: u32[base + 62], shadowOverflows: u32[base + 63] });
        }
        return { frame, probes, invalidProbes: probes.filter(probe => !probe.valid).length };
      } finally { this.readback.unmap(); }
    } finally { this.pending = false; }
  }
  diagnostics() {
    return { enabled: this.enabled, frame: this.frame, options: this.options, ownedGpuBytes: this.bytes,
      bounceCount: 1, capture: 'diffuse-and-emissive', update: 'full-grid-every-frame', representation: 'radiance-SH9',
      visibility: 'six-direction-distance-moments', placement: 'caller-provided-free-space-grid',
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
