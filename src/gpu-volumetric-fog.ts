import common from './shaders/volumetric-fog-common.wgsl?raw';
import computeSource from './shaders/volumetric-fog.wgsl?raw';
import composeSource from './shaders/volumetric-fog-compose.wgsl?raw';
import temporalSource from './shaders/volumetric-fog-temporal.wgsl?raw';
import { lookAt, type Mat4 } from './math';
import { resolveVolumetricFogOptions, selectFogGrid, type VolumetricFogOptions, type ResolvedVolumetricFogOptions } from './volumetric-fog';

type Grid = ReturnType<typeof selectFogGrid>;
type FogMedia = Pick<VolumetricFogOptions, 'global' | 'volumes'>;
export type FogHistoryCell = { x: number; y: number; z: number };
type HistoryView = { source: GPUBuffer; count: GPUBuffer; compute: GPUBindGroup; temporal: GPUBindGroup; compose: GPUBindGroup };
const FIXED_BYTES = 384 + 64 * 80;
const LIGHT_COMPONENTS = [20, 21, 22, 23, 24, 25, 26, ...Array.from({ length: 16 }, (_, i) => i + 48)];

function radicalInverse(value: number, base: number): number {
  let factor = 1 / base, result = 0;
  while (value > 0) { result += (value % base) * factor; value = Math.floor(value / base); factor /= base; }
  return result;
}

/** View cache only. Media remain world-space definitions independent of surface LOD. */
export class GpuVolumetricFog {
  /** Leaves f32 headroom for 65 overlapping media, HG phase and long oblique rays. */
  static validateSunColor(color: readonly number[]): void {
    if (color.length !== 3 || color.some(value => !Number.isFinite(value) || value < 0 || value > 1e20)) throw new RangeError('FOG_SUN_RADIANCE_LIMIT');
  }
  private options: ResolvedVolumetricFogOptions;
  private readonly uniform: GPUBuffer;
  private readonly volumes: GPUBuffer;
  private readonly data = new Float32Array(96);
  private readonly volumeData = new Float32Array(64 * 20);
  private computeLayout!: GPUBindGroupLayout;
  private composeLayout!: GPUBindGroupLayout;
  private injectPipeline!: GPUComputePipeline;
  private integratePipeline!: GPUComputePipeline;
  private composePipeline!: GPURenderPipeline;
  private temporalLayout?: GPUBindGroupLayout;
  private temporalPipeline?: GPUComputePipeline;
  private historyViews: HistoryView[] = [];
  private historyIndex = 1;
  private historyValid = false;
  private temporalEnabled: boolean;
  private historyFrames = 0;
  private resetReason = 'initial';
  private resetPending = true;
  private previousView?: Float32Array;
  private previousMaterialRevision?: number;
  private submittedView?: Float32Array;
  private submittedMaterialRevision?: number;
  private computeGroup?: GPUBindGroup;
  private composeGroup?: GPUBindGroup;
  private sources?: GPUBuffer;
  private integrated?: GPUBuffer;
  private output?: GPUTexture;
  private grid?: Grid;
  private dimensions?: [number, number];
  private enabled = true;
  private revision = 1;
  private disposed = false;
  private volumeUploadPending = true;
  private maxAnisotropy = 0;

  static async create(device: GPUDevice, options: VolumetricFogOptions, account: (bytes: number) => void): Promise<GpuVolumetricFog> {
    const resolved = resolveVolumetricFogOptions(options);
    selectFogGrid(1, 1, resolved);
    if (device.limits.maxStorageBuffersPerShaderStage < (resolved.temporal ? 5 : 3)) throw new Error('FOG_STORAGE_BINDING_LIMIT');
    const fog = new GpuVolumetricFog(device, resolved, account);
    try { await fog.initialize(); return fog; } catch (error) { fog.dispose(); throw error; }
  }

  private constructor(private device: GPUDevice, options: ResolvedVolumetricFogOptions, private account: (bytes: number) => void) {
    this.options = options;
    this.temporalEnabled = !!options.temporal;
    this.uniform = device.createBuffer({ label: 'fog frame', size: 384, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    try {
      this.volumes = device.createBuffer({ label: 'fog local media', size: 64 * 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    } catch (error) { this.uniform.destroy(); throw error; }
    account(FIXED_BYTES);
    this.packVolumes();
  }

  private async initialize(): Promise<void> {
    const shader = async (code: string, label: string) => {
      const module = this.device.createShaderModule({ code: common + '\n' + code, label });
      const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(error => `${label}:${error.lineNum}: ${error.message}`).join('\n'));
      return module;
    };
    this.computeLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ...[2, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } })),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'comparison' } },
      ...(this.options.temporal ? [{ binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' as const } }] : []),
    ] });
    this.composeLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ...[2, 3].map(binding => ({ binding, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' as const } })),
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ] });
    const resolvedSource = this.options.temporal
      ? '@group(0) @binding(6) var<storage, read> historySource: array<vec4f>;\nfn resolvedSource(index:u32)->vec4f { if(frame.reserved.w > 0.5){return historySource[index];}return sources[index];}'
      : 'fn resolvedSource(index:u32)->vec4f {return sources[index];}';
    const compute = await shader(computeSource.replace('// FOG_RESOLVED_SOURCE', resolvedSource), 'fog source and integration');
    const computeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    this.injectPipeline = await this.device.createComputePipelineAsync({ label: 'fog density and sun', layout: computeLayout, compute: { module: compute, entryPoint: 'inject' } });
    this.integratePipeline = await this.device.createComputePipelineAsync({ label: 'fog front-to-back integral', layout: computeLayout, compute: { module: compute, entryPoint: 'integrate' } });
    const compose = await shader(composeSource, 'fog HDR composition');
    this.composePipeline = await this.device.createRenderPipelineAsync({ label: 'fog depth-truncated HDR',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.composeLayout] }),
      vertex: { module: compose, entryPoint: 'vertexMain' }, fragment: { module: compose, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }] },
    });
    if (this.options.temporal) {
      this.temporalLayout = this.device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: binding >= 4 ? 'storage' as const : 'read-only-storage' as const } })),
      ] });
      const temporal = await shader(temporalSource, 'fog world-space history');
      this.temporalPipeline = await this.device.createComputePipelineAsync({ label: 'fog world-space reprojection',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.temporalLayout] }), compute: { module: temporal, entryPoint: 'temporalMain' } });
    }
  }

  /** Validated before Renderer retires its old attachments. Device limits never silently reduce quality. */
  planResize(width: number, height: number): Grid {
    if (this.disposed) throw new Error('FOG_DISPOSED');
    const grid = selectFogGrid(width, height, this.options);
    if (width > this.device.limits.maxTextureDimension2D || height > this.device.limits.maxTextureDimension2D
      || grid.froxels * 16 > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize)
      || Math.ceil(grid.columns / 4) > this.device.limits.maxComputeWorkgroupsPerDimension
      || Math.ceil(grid.rows / 4) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('FOG_DEVICE_BUDGET');
    return grid;
  }

  resize(width: number, height: number, depth: GPUTexture, hdr: GPUTexture, shadow: GPUTexture): void {
    const grid = this.planResize(width, height);
    this.releaseView();
    let sources: GPUBuffer | undefined, integrated: GPUBuffer | undefined, output: GPUTexture | undefined;
    const histories: { source: GPUBuffer; count: GPUBuffer }[] = [], historyBuffers: GPUBuffer[] = [];
    try {
      sources = this.device.createBuffer({ label: 'fog source and extinction', size: grid.froxels * 16, usage: GPUBufferUsage.STORAGE });
      integrated = this.device.createBuffer({ label: 'fog cumulative radiance and transmission', size: grid.froxels * 16, usage: GPUBufferUsage.STORAGE });
      output = this.device.createTexture({ label: 'fog-composited linear HDR', size: [width, height], format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
      if (this.options.temporal) for (let index = 0; index < 2; index++) {
        const source = this.device.createBuffer({ label: 'fog history source ' + index, size: grid.froxels * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        historyBuffers.push(source);
        const count = this.device.createBuffer({ label: 'fog history count ' + index, size: grid.froxels * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        historyBuffers.push(count); histories.push({ source, count });
      }
      const computeGroup = (history?: GPUBuffer) => this.device.createBindGroup({ layout: this.computeLayout, entries: [
        { binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: this.volumes } },
        { binding: 2, resource: { buffer: sources! } }, { binding: 3, resource: { buffer: integrated! } },
        { binding: 4, resource: shadow.createView() },
        { binding: 5, resource: this.device.createSampler({ compare: 'less-equal', minFilter: 'linear', magFilter: 'linear' }) },
        ...(history ? [{ binding: 6, resource: { buffer: history } }] : []),
      ] });
      const composeGroup = (source: GPUBuffer) => this.device.createBindGroup({ layout: this.composeLayout, entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 2, resource: { buffer: source } }, { binding: 3, resource: { buffer: integrated! } },
        { binding: 4, resource: depth.createView() }, { binding: 5, resource: hdr.createView() },
      ] });
      const historyViews = histories.map((history, index): HistoryView => ({ ...history,
        compute: computeGroup(history.source), compose: composeGroup(history.source),
        temporal: this.device.createBindGroup({ layout: this.temporalLayout!, entries: [
          { binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: sources! } },
          { binding: 2, resource: { buffer: histories[1 - index].source } }, { binding: 3, resource: { buffer: histories[1 - index].count } },
          { binding: 4, resource: { buffer: history.source } }, { binding: 5, resource: { buffer: history.count } },
        ] }),
      }));
      const rawCompute = this.options.temporal ? historyViews[0].compute : computeGroup();
      const rawCompose = composeGroup(sources);
      this.sources = sources; this.integrated = integrated; this.output = output;
      this.computeGroup = rawCompute; this.composeGroup = rawCompose; this.historyViews = historyViews;
      this.grid = grid; this.dimensions = [width, height];
      this.account(grid.bytes - FIXED_BYTES);
      this.revision++;
      this.invalidateHistory('resize');
    } catch (error) { sources?.destroy(); integrated?.destroy(); output?.destroy(); for (const buffer of historyBuffers) buffer.destroy(); throw error; }
  }

  get colorTexture(): GPUTexture | undefined { return this.output; }

  setEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('FOG_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('INVALID_FOG_ENABLED');
    if (enabled !== this.enabled) { this.enabled = enabled; this.revision++; this.invalidateHistory('fog-toggle'); }
  }

  get temporalConfigured(): boolean { return !!this.options.temporal; }

  setTemporalEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('FOG_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('INVALID_FOG_ENABLED');
    if (enabled && !this.options.temporal) throw new Error('FOG_TEMPORAL_NOT_CONFIGURED');
    if (enabled !== this.temporalEnabled) { this.temporalEnabled = enabled; this.revision++; this.invalidateHistory('temporal-toggle'); }
  }

  invalidateHistory(reason = 'camera-cut'): void {
    this.historyValid = false; this.historyFrames = 0; this.resetReason = reason; this.resetPending = true;
  }

  /** Atomic replacement; omitted global/volumes means no global medium/no local media. */
  setMedia(media: FogMedia): void {
    if (this.disposed) throw new Error('FOG_DISPOSED');
    const resolved = resolveVolumetricFogOptions({ ...this.options, global: media.global, volumes: media.volumes ?? [] });
    this.options = resolved; this.packVolumes(); this.revision++; this.invalidateHistory('media-change');
  }

  private packVolumes(): void {
    this.volumeData.fill(0);
    this.maxAnisotropy = Math.abs(this.options.global?.coefficients.anisotropy ?? 0);
    this.options.volumes.forEach((volume, index) => {
      const offset = index * 20, shape = volume.shape;
      this.maxAnisotropy = Math.max(this.maxAnisotropy, Math.abs(volume.coefficients.anisotropy));
      if (shape.kind === 'sphere') {
        this.volumeData.set([...shape.center, 1, shape.radius, shape.radius, shape.radius, 0], offset);
      } else {
        this.volumeData.set([0, 1, 2].map(axis => (shape.min[axis] + shape.max[axis]) * 0.5), offset);
        this.volumeData.set([0, 1, 2].map(axis => (shape.max[axis] - shape.min[axis]) * 0.5), offset + 4);
      }
      this.volumeData.set([volume.coefficients.extinction, volume.coefficients.anisotropy, 0, 0], offset + 8);
      this.volumeData.set(volume.coefficients.albedo, offset + 12);
      this.volumeData.set(volume.coefficients.emission, offset + 16);
    });
    this.volumeUploadPending = true;
  }

  update(eye: readonly number[], target: readonly number[], fovY: number, near: number, far: number,
    sunDirection: readonly number[], sunColor: readonly number[], lightMatrix: Mat4,
    viewProjection: Mat4, state: { cameraCut: boolean; sceneChanged: boolean; materialRevision: number }): number {
    if (this.disposed || !this.grid || !this.dimensions) throw new Error('FOG_NO_VIEW');
    GpuVolumetricFog.validateSunColor(sunColor);
    const [width, height] = this.dimensions, grid = this.grid, data = this.data;
    const view = lookAt(eye, target, [0, 1, 0]);
    const tangent = Math.tan(fovY / 2), horizontal = tangent * width / height;
    data.fill(0);
    new Uint32Array(data.buffer).set([grid.columns, grid.rows, grid.slices, this.options.volumes.length]);
    data.set([...eye, this.options.maxViewDepth], 4);
    data.set([view[0] * horizontal, view[4] * horizontal, view[8] * horizontal, near], 8);
    data.set([view[1] * tangent, view[5] * tangent, view[9] * tangent, far], 12);
    data.set([-view[2], -view[6], -view[10], this.enabled ? 1 : 0], 16);
    data.set([...sunDirection, 0.00001], 20);
    data.set(sunColor, 24);
    if (this.options.global) {
      const global = this.options.global;
      data.set([global.coefficients.extinction, global.coefficients.anisotropy, global.baseHeight, global.heightFalloff], 28);
      data.set(global.coefficients.albedo, 32); data.set(global.coefficients.emission, 36);
    }
    data.set([width, height, this.options.global ? 1 : 0, 0], 40);
    data.set(lightMatrix, 48);
    if (this.options.temporal) {
      if (!this.resetPending && state.cameraCut) this.invalidateHistory('camera-cut');
      if (!this.resetPending && state.sceneChanged) this.invalidateHistory('scene-change');
      if (!this.resetPending && state.materialRevision !== this.previousMaterialRevision) this.invalidateHistory('material-change');
      if (!this.resetPending && this.previousView && LIGHT_COMPONENTS
        .some(index => data[index] !== this.previousView![index])) this.invalidateHistory('lighting-change');
      const previous = this.previousView ?? data;
      data.set(previous.subarray(4, 8), 64); data.set(previous.subarray(16, 20), 68);
      data.set(previous.subarray(72, 88), 72);
      const active = this.temporalEnabled && this.enabled;
      if (active && this.historyFrames > 0) {
        const sequence = 1 + (this.historyFrames - 1) % 1024;
        data.set([radicalInverse(sequence, 2) - 0.5, radicalInverse(sequence, 3) - 0.5, radicalInverse(sequence, 5) - 0.5], 44);
      }
      data[47] = active ? 1 : 0;
      data.set([this.historyValid && active ? 1 : 0, this.options.temporal.maxHistoryFrames, this.maxAnisotropy, 0], 88);
      this.submittedView = data.slice(); this.submittedView.set(viewProjection, 72); this.submittedMaterialRevision = state.materialRevision;
    }
    this.device.queue.writeBuffer(this.uniform, 0, data);
    let bytes = 384;
    if (this.volumeUploadPending) { this.device.queue.writeBuffer(this.volumes, 0, this.volumeData); bytes += this.volumeData.byteLength; this.volumeUploadPending = false; }
    return bytes;
  }

  encodeInject(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
    const pass = encoder.beginComputePass({ label: 'fog density and lighting', timestampWrites });
    if (this.enabled && this.grid) {
      pass.setPipeline(this.injectPipeline); pass.setBindGroup(0, this.computeGroup!);
      pass.dispatchWorkgroups(Math.ceil(this.grid.columns / 4), Math.ceil(this.grid.rows / 4), Math.ceil(this.grid.slices / 4));
    }
    pass.end();
  }
  encodeIntegrate(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
    const pass = encoder.beginComputePass({ label: 'fog integration', timestampWrites });
    if (this.enabled && this.grid) {
      pass.setPipeline(this.integratePipeline); pass.setBindGroup(0, this.temporalEnabled ? this.historyViews[1 - this.historyIndex].compute : this.computeGroup!);
      pass.dispatchWorkgroups(Math.ceil(this.grid.columns / 8), Math.ceil(this.grid.rows / 8));
    }
    pass.end();
  }
  encodeTemporal(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
    const pass = encoder.beginComputePass({ label: 'fog world-space history', timestampWrites });
    if (this.enabled && this.temporalEnabled && this.grid) {
      pass.setPipeline(this.temporalPipeline!); pass.setBindGroup(0, this.historyViews[1 - this.historyIndex].temporal);
      pass.dispatchWorkgroups(Math.ceil(this.grid.columns / 4), Math.ceil(this.grid.rows / 4), Math.ceil(this.grid.slices / 4));
    }
    pass.end();
  }
  encodeCompose(encoder: GPUCommandEncoder, timestampWrites?: GPURenderPassTimestampWrites): void {
    const pass = encoder.beginRenderPass({ label: 'fog HDR composition', timestampWrites,
      colorAttachments: [{ view: this.output!.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.composePipeline);
    pass.setBindGroup(0, this.temporalEnabled && this.enabled ? this.historyViews[1 - this.historyIndex].compose : this.composeGroup!);
    pass.draw(3); pass.end();
  }

  afterSubmit(): void {
    this.previousView = this.submittedView; this.previousMaterialRevision = this.submittedMaterialRevision;
    if (this.enabled && this.temporalEnabled) {
      this.historyIndex = 1 - this.historyIndex; this.historyValid = true;
      this.historyFrames++; this.resetPending = false;
    }
  }

  /** Explicit bounded diagnostic; production never reads volume history back to the CPU. */
  async probeHistory(cells: readonly FogHistoryCell[]) {
    if (this.disposed || !this.historyValid || !this.grid) throw new Error('FOG_HISTORY_UNAVAILABLE');
    const grid = this.grid, history = this.historyViews[this.historyIndex];
    if (cells.length > 1024 || cells.some(cell => ![cell.x, cell.y, cell.z].every(Number.isSafeInteger)
      || cell.x < 0 || cell.x >= grid.columns || cell.y < 0 || cell.y >= grid.rows || cell.z < 0 || cell.z >= grid.slices)) throw new Error('INVALID_FOG_HISTORY_CELLS');
    if (!cells.length) return [];
    const read = this.device.createBuffer({ label: 'diagnostic fog history', size: cells.length * 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder();
      cells.forEach((cell, i) => {
        const index = (cell.z * grid.rows + cell.y) * grid.columns + cell.x;
        encoder.copyBufferToBuffer(history.source, index * 16, read, i * 32, 16);
        encoder.copyBufferToBuffer(history.count, index * 4, read, i * 32 + 16, 4);
      });
      this.device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const mapped = read.getMappedRange(), floats = new Float32Array(mapped), integers = new Uint32Array(mapped);
      return cells.map((cell, i) => ({ ...cell, source: Array.from(floats.subarray(i * 8, i * 8 + 4)), count: integers[i * 8 + 4] }));
    } finally { read.destroy(); }
  }
  diagnostics() {
    return { enabled: this.enabled, revision: this.revision, grid: this.grid ? { ...this.grid } : null,
      maxViewDepth: this.options.maxViewDepth, volumes: this.options.volumes.length, global: !!this.options.global,
      depthDistribution: 'quadratic' as const, lighting: 'sun-shadow-single-scattering' as const, temporalHistory: this.temporalEnabled,
      history: { configured: !!this.options.temporal, valid: this.historyValid, framesSinceReset: this.historyFrames,
        maxFrames: this.options.temporal ? this.options.temporal.maxHistoryFrames : 0, lastResetReason: this.resetReason },
      shadowOutsideCoverage: 'unshadowed' as const, ownedGpuBytes: this.disposed ? 0 : this.grid?.bytes ?? FIXED_BYTES };
  }
  private releaseView(): void {
    this.sources?.destroy(); this.integrated?.destroy(); this.output?.destroy();
    for (const history of this.historyViews) { history.source.destroy(); history.count.destroy(); }
    this.historyViews = []; this.historyIndex = 1; this.previousView = undefined; this.invalidateHistory('resize');
    if (this.grid && this.dimensions) this.account(-(this.grid.bytes - FIXED_BYTES));
    this.sources = undefined; this.integrated = undefined; this.output = undefined; this.grid = undefined; this.dimensions = undefined;
    this.computeGroup = undefined; this.composeGroup = undefined;
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true; this.releaseView(); this.uniform.destroy(); this.volumes.destroy(); this.account(-FIXED_BYTES);
  }
}
