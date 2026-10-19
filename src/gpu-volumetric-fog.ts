import common from './shaders/volumetric-fog-common.wgsl?raw';
import computeSource from './shaders/volumetric-fog.wgsl?raw';
import composeSource from './shaders/volumetric-fog-compose.wgsl?raw';
import { lookAt, type Mat4 } from './math';
import { resolveVolumetricFogOptions, selectFogGrid, type VolumetricFogOptions, type ResolvedVolumetricFogOptions } from './volumetric-fog';

type Grid = ReturnType<typeof selectFogGrid>;
type FogMedia = Pick<VolumetricFogOptions, 'global' | 'volumes'>;

/** View cache only. Media remain world-space definitions independent of surface LOD. */
export class GpuVolumetricFog {
  /** Leaves f32 headroom for 65 overlapping media, HG phase and long oblique rays. */
  static validateSunColor(color: readonly number[]): void {
    if (color.length !== 3 || color.some(value => !Number.isFinite(value) || value < 0 || value > 1e20)) throw new RangeError('FOG_SUN_RADIANCE_LIMIT');
  }
  private options: ResolvedVolumetricFogOptions;
  private readonly uniform: GPUBuffer;
  private readonly volumes: GPUBuffer;
  private readonly data = new Float32Array(64);
  private readonly volumeData = new Float32Array(64 * 20);
  private computeLayout!: GPUBindGroupLayout;
  private composeLayout!: GPUBindGroupLayout;
  private injectPipeline!: GPUComputePipeline;
  private integratePipeline!: GPUComputePipeline;
  private composePipeline!: GPURenderPipeline;
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

  static async create(device: GPUDevice, options: VolumetricFogOptions, account: (bytes: number) => void): Promise<GpuVolumetricFog> {
    const resolved = resolveVolumetricFogOptions(options);
    selectFogGrid(1, 1, resolved);
    if (device.limits.maxStorageBuffersPerShaderStage < 3) throw new Error('FOG_STORAGE_BINDING_LIMIT');
    const fog = new GpuVolumetricFog(device, resolved, account);
    try { await fog.initialize(); return fog; } catch (error) { fog.dispose(); throw error; }
  }

  private constructor(private device: GPUDevice, options: ResolvedVolumetricFogOptions, private account: (bytes: number) => void) {
    this.options = options;
    this.uniform = device.createBuffer({ label: 'fog frame', size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    try {
      this.volumes = device.createBuffer({ label: 'fog local media', size: 64 * 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    } catch (error) { this.uniform.destroy(); throw error; }
    account(256 + 64 * 80);
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
    ] });
    this.composeLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ...[2, 3].map(binding => ({ binding, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' as const } })),
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ] });
    const compute = await shader(computeSource, 'fog source and integration');
    const computeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    this.injectPipeline = await this.device.createComputePipelineAsync({ label: 'fog density and sun', layout: computeLayout, compute: { module: compute, entryPoint: 'inject' } });
    this.integratePipeline = await this.device.createComputePipelineAsync({ label: 'fog front-to-back integral', layout: computeLayout, compute: { module: compute, entryPoint: 'integrate' } });
    const compose = await shader(composeSource, 'fog HDR composition');
    this.composePipeline = await this.device.createRenderPipelineAsync({ label: 'fog depth-truncated HDR',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.composeLayout] }),
      vertex: { module: compose, entryPoint: 'vertexMain' }, fragment: { module: compose, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }] },
    });
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
    try {
      sources = this.device.createBuffer({ label: 'fog source and extinction', size: grid.froxels * 16, usage: GPUBufferUsage.STORAGE });
      integrated = this.device.createBuffer({ label: 'fog cumulative radiance and transmission', size: grid.froxels * 16, usage: GPUBufferUsage.STORAGE });
      output = this.device.createTexture({ label: 'fog-composited linear HDR', size: [width, height], format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
      const computeGroup = this.device.createBindGroup({ layout: this.computeLayout, entries: [
        { binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: { buffer: this.volumes } },
        { binding: 2, resource: { buffer: sources } }, { binding: 3, resource: { buffer: integrated } },
        { binding: 4, resource: shadow.createView() },
        { binding: 5, resource: this.device.createSampler({ compare: 'less-equal', minFilter: 'linear', magFilter: 'linear' }) },
      ] });
      const composeGroup = this.device.createBindGroup({ layout: this.composeLayout, entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 2, resource: { buffer: sources } }, { binding: 3, resource: { buffer: integrated } },
        { binding: 4, resource: depth.createView() }, { binding: 5, resource: hdr.createView() },
      ] });
      this.sources = sources; this.integrated = integrated; this.output = output;
      this.computeGroup = computeGroup; this.composeGroup = composeGroup;
      this.grid = grid; this.dimensions = [width, height];
      this.account(grid.froxels * 32 + width * height * 8);
      this.revision++;
    } catch (error) { sources?.destroy(); integrated?.destroy(); output?.destroy(); throw error; }
  }

  get colorTexture(): GPUTexture | undefined { return this.output; }

  setEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('FOG_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('INVALID_FOG_ENABLED');
    if (enabled !== this.enabled) { this.enabled = enabled; this.revision++; }
  }

  /** Atomic replacement; omitted global/volumes means no global medium/no local media. */
  setMedia(media: FogMedia): void {
    if (this.disposed) throw new Error('FOG_DISPOSED');
    const resolved = resolveVolumetricFogOptions({ ...this.options, global: media.global, volumes: media.volumes ?? [] });
    this.options = resolved; this.packVolumes(); this.revision++;
  }

  private packVolumes(): void {
    this.volumeData.fill(0);
    this.options.volumes.forEach((volume, index) => {
      const offset = index * 20, shape = volume.shape;
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
    sunDirection: readonly number[], sunColor: readonly number[], lightMatrix: Mat4): number {
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
    this.device.queue.writeBuffer(this.uniform, 0, data);
    let bytes = 256;
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
      pass.setPipeline(this.integratePipeline); pass.setBindGroup(0, this.computeGroup!);
      pass.dispatchWorkgroups(Math.ceil(this.grid.columns / 8), Math.ceil(this.grid.rows / 8));
    }
    pass.end();
  }
  encodeCompose(encoder: GPUCommandEncoder, timestampWrites?: GPURenderPassTimestampWrites): void {
    const pass = encoder.beginRenderPass({ label: 'fog HDR composition', timestampWrites,
      colorAttachments: [{ view: this.output!.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.composePipeline); pass.setBindGroup(0, this.composeGroup!); pass.draw(3); pass.end();
  }
  diagnostics() {
    return { enabled: this.enabled, revision: this.revision, grid: this.grid ? { ...this.grid } : null,
      maxViewDepth: this.options.maxViewDepth, volumes: this.options.volumes.length, global: !!this.options.global,
      depthDistribution: 'quadratic' as const, lighting: 'sun-shadow-single-scattering' as const, temporalHistory: false,
      shadowOutsideCoverage: 'unshadowed' as const, ownedGpuBytes: this.disposed ? 0 : this.grid?.bytes ?? 5376 };
  }
  private releaseView(): void {
    this.sources?.destroy(); this.integrated?.destroy(); this.output?.destroy();
    if (this.grid && this.dimensions) this.account(-(this.grid.froxels * 32 + this.dimensions[0] * this.dimensions[1] * 8));
    this.sources = undefined; this.integrated = undefined; this.output = undefined; this.grid = undefined; this.dimensions = undefined;
    this.computeGroup = undefined; this.composeGroup = undefined;
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true; this.releaseView(); this.uniform.destroy(); this.volumes.destroy(); this.account(-5376);
  }
}
