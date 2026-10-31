import testSource from './shaders/occlusion-test.wgsl?raw';
import baseSource from './shaders/occlusion-pyramid.wgsl?raw';
import reduceSource from './shaders/occlusion-reduce.wgsl?raw';
import { planOcclusionPyramid, resolveOcclusionOptions, type OcclusionOptions, type OcclusionPyramidPlan } from './occlusion';
import type { Mat4 } from './math';
import { timestampSpanEnd, timestampSpanStart } from './gpu-timestamp-span';

export const occlusionTestSource = testSource;
/** Previous depth predicts the first draw; current depth alone makes final exclusions.
 * Owns two bounded max-depth pyramids. No CPU visibility/geometry readback. */
export class GpuOcclusion {
  readonly options;
  readonly testLayout: GPUBindGroupLayout;
  private readonly parameters: GPUBuffer[] = [];
  private readonly textures: GPUTexture[] = [];
  private basePipeline!: GPUComputePipeline;
  private reducePipeline!: GPUComputePipeline;
  private testGroups: GPUBindGroup[][] = [];
  private baseGroups: GPUBindGroup[] = [];
  private reduceGroups: GPUBindGroup[][] = [];
  private grid?: OcclusionPyramidPlan;
  private bytes = 0;
  private enabled = true;
  private activeForView = true;
  private disposed = false;
  private historyValid = false;
  private writeIndex = 0;
  private previousIndex = 1;
  private depth?: GPUTexture;
  private frame = 0;
  private encodedFrame = 0;
  private width = 0;
  private height = 0;
  private readonly currentData = new Float32Array(24);
  private readonly historyData = new Float32Array(24);
  get previousGroup(): GPUBindGroup { return this.testGroups[0][this.previousIndex]; }
  get currentGroup(): GPUBindGroup { return this.testGroups[1][this.writeIndex]; }

  static async create(device: GPUDevice, options: OcclusionOptions, track: (bytes: number) => void) {
    const result = new GpuOcclusion(device, options, track);
    try {
      if (result.options.maxBytes < 208) throw new Error('OCCLUSION_BYTE_BUDGET');
      for (const size of [96, 96, 16]) {
        const buffer = device.createBuffer({ label: 'occlusion immutable view parameters', size,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        result.parameters.push(buffer); result.bytes += size; track(size);
      }
      const module = async (code: string) => {
        const shader = device.createShaderModule({ code });
        const errors = (await shader.getCompilationInfo()).messages.filter(item => item.type === 'error');
        if (errors.length) throw new Error('OCCLUSION_SHADER: ' + errors.map(item => item.message).join('\n'));
        return shader;
      };
      result.basePipeline = await device.createComputePipelineAsync({ layout: 'auto',
        compute: { module: await module(baseSource), entryPoint: 'base' } });
      const reduceLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
      ] });
      result.reducePipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [reduceLayout] }),
        compute: { module: await module(reduceSource), entryPoint: 'reduce' } });
      return result;
    } catch (error) { result.dispose(); throw error; }
  }
  private constructor(private device: GPUDevice, options: OcclusionOptions, private track: (bytes: number) => void) {
    this.options = resolveOcclusionOptions(options);
    this.testLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: 96 } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
    ] });
  }
  setEnabled(enabled: boolean) {
    if (this.disposed) throw new Error('OCCLUSION_DISPOSED');
    if (typeof enabled !== 'boolean') throw new Error('OCCLUSION_ENABLED');
    if (enabled !== this.enabled) this.historyValid = false;
    this.enabled = enabled;
  }
  private releaseTextures() {
    for (const texture of this.textures) texture.destroy(); this.textures.length = 0;
    if (this.grid) { const released = this.grid.textureBytes * 2; this.bytes -= released; this.track(-released); }
    this.grid = undefined; this.testGroups = []; this.baseGroups = []; this.reduceGroups = [];
    this.historyValid = false;
  }
  /** Validate a proposed view before the renderer retires its current attachments. */
  planResize(width: number, height: number): OcclusionPyramidPlan {
    if (this.disposed) throw new Error('OCCLUSION_DISPOSED');
    const plan = planOcclusionPyramid(width, height, this.options);
    if (plan.width > this.device.limits.maxTextureDimension2D || plan.height > this.device.limits.maxTextureDimension2D
      || Math.ceil(Math.max(plan.width, plan.height) / 8) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('OCCLUSION_DEVICE_LIMIT');
    return plan;
  }
  update(frame: number, width: number, height: number, viewProjection: Mat4, depth: GPUTexture, allowOcclusion = true): number {
    if (this.disposed) throw new Error('OCCLUSION_DISPOSED');
    if (typeof allowOcclusion !== 'boolean' || !Number.isSafeInteger(frame) || frame <= this.frame || viewProjection.length !== 16
      || ![...viewProjection].every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value)))) throw new Error('OCCLUSION_VIEW');
    const plan = this.planResize(width, height);
    if (!this.grid || width !== this.width || height !== this.height || depth !== this.depth) {
      this.releaseTextures();
      try {
        for (let index = 0; index < 2; index++) {
          const texture = this.device.createTexture({ label: 'occlusion max-depth pyramid', size: [plan.width, plan.height],
            mipLevelCount: plan.mipLevelCount, format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
          this.textures.push(texture); this.bytes += plan.textureBytes; this.track(plan.textureBytes);
        }
      } catch (error) {
        const released = this.textures.length * plan.textureBytes;
        for (const texture of this.textures) texture.destroy(); this.textures.length = 0;
        this.bytes -= released; this.track(-released); throw error;
      }
      this.grid = plan;
      try {
        this.testGroups = [0, 1].map(role => this.textures.map(texture => this.device.createBindGroup({ layout: this.testLayout,
          entries: [{ binding: 0, resource: { buffer: this.parameters[role] } }, { binding: 1, resource: texture.createView() }] })));
        this.baseGroups = this.textures.map(texture => this.device.createBindGroup({ layout: this.basePipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: depth.createView() }, { binding: 1, resource: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
            { binding: 2, resource: { buffer: this.parameters[2] } }] }));
        this.reduceGroups = this.textures.map(texture => Array.from({ length: plan.mipLevelCount - 1 }, (_, level) =>
          this.device.createBindGroup({ layout: this.reducePipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: texture.createView({ baseMipLevel: level, mipLevelCount: 1 }) },
            { binding: 1, resource: texture.createView({ baseMipLevel: level + 1, mipLevelCount: 1 }) },
          ] })));
        this.width = width; this.height = height; this.depth = depth;
      } catch (error) { this.releaseTextures(); throw error; }
    }
    this.previousIndex = this.writeIndex; this.writeIndex = 1 - this.writeIndex;
    this.activeForView = this.enabled && allowOcclusion;
    this.historyData[20] = Number(this.activeForView && this.historyValid);
    this.currentData.set(viewProjection);
    new Uint32Array(this.currentData.buffer).set([width, height, plan.width, plan.height], 16);
    this.currentData.set([Number(this.activeForView), this.options.depthBias, plan.pixelSize, plan.mipLevelCount - 1], 20);
    this.device.queue.writeBuffer(this.parameters[0], 0, this.historyData);
    this.device.queue.writeBuffer(this.parameters[1], 0, this.currentData);
    this.device.queue.writeBuffer(this.parameters[2], 0, new Uint32Array([width, height, plan.pixelSize, 0]));
    this.frame = frame; return 208;
  }
  encodePyramid(encoder: GPUCommandEncoder, timestamp?: GPUComputePassTimestampWrites) {
    if (this.disposed || !this.grid || this.frame === this.encodedFrame) throw new Error('OCCLUSION_FRAME');
    const levels = this.activeForView ? this.grid.mipLevelCount : 1;
    for (let level = 0; level < levels; level++) {
      const pass = encoder.beginComputePass({ label: 'occlusion maximum depth ' + level, timestampWrites: levels === 1 ? timestamp
        : level === 0 ? timestampSpanStart(timestamp) : level === levels - 1 ? timestampSpanEnd(timestamp) : undefined });
      if (this.activeForView) {
        pass.setPipeline(level === 0 ? this.basePipeline : this.reducePipeline);
        pass.setBindGroup(0, level === 0 ? this.baseGroups[this.writeIndex] : this.reduceGroups[this.writeIndex][level - 1]);
        pass.dispatchWorkgroups(Math.ceil(Math.max(1, this.grid.width >> level) / 8), Math.ceil(Math.max(1, this.grid.height >> level) / 8));
      }
      pass.end();
    }
    this.encodedFrame = this.frame; this.historyData.set(this.currentData); this.historyValid = this.activeForView;
  }
  diagnostics() { return { enabled: this.enabled, activeForView: this.activeForView, historyValid: this.historyValid, grid: this.grid ? { ...this.grid } : null,
    ownedGpuBytes: this.bytes, frame: this.encodedFrame, algorithm: 'two-phase-current-depth-recovery' as const,
    geometryReadback: false, depthConvention: 'forward-max' as const }; }
  dispose() {
    if (this.disposed) return; this.disposed = true;
    this.releaseTextures(); for (const buffer of this.parameters) buffer.destroy(); this.parameters.length = 0;
    this.track(-this.bytes); this.bytes = 0;
  }
}
