type TextureAdapterInfo = Pick<GPUAdapterInfo, 'architecture' | 'description'>;

const preparationWGSL = `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(positions[index], 0, 1);
}
@fragment fn fragmentMain() -> @location(0) vec4f {
  return textureSampleLevel(image, imageSampler, vec2f(0.5), 0);
}
`;

/** SwiftShader can leave freshly uploaded compressed images undecoded in compute.
 * Its graphics path calls DescriptorSet::PrepareForSampling; the isolated compute
 * path does not. A single draw after upload prepares the complete mip view before
 * any future query. This is an upload cost, never a per-frame or hardware cost.
 * https://github.com/google/swiftshader/blob/master/src/Device/Renderer.cpp
 * https://github.com/google/swiftshader/blob/master/src/Pipeline/ComputeProgram.cpp
 */
export class CompressedTexturePreparation {
  private readonly mode: 'none' | 'swiftshader-render';
  private resources?: { target: GPUTexture; layout: GPUBindGroupLayout;
    pipeline: GPURenderPipeline; sampler: GPUSampler };
  private imagesPrepared = 0;
  private disposed = false;

  constructor(private readonly device: GPUDevice, private readonly account: (bytes: number) => void,
    adapterInfo: TextureAdapterInfo | undefined = device.adapterInfo) {
    this.mode = /swiftshader/i.test((adapterInfo?.architecture ?? '') + ' ' + (adapterInfo?.description ?? ''))
      ? 'swiftshader-render' : 'none';
  }

  /** Call only after all mip uploads to a new compressed image, before publishing it. */
  prepare(texture: GPUTexture): void {
    if (this.disposed) throw new Error('TEXTURE_PREPARATION_DISPOSED');
    if (this.mode === 'none') return;
    if (!this.resources) {
      const layout = this.device.createBindGroupLayout({ label: 'compressed image preparation', entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ] });
      const module = this.device.createShaderModule({ label: 'compressed image preparation', code: preparationWGSL });
      const pipeline = this.device.createRenderPipeline({
        label: 'SwiftShader compressed image preparation',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: 'vertexMain' },
        fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] },
      });
      const sampler = this.device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest' });
      const target = this.device.createTexture({ label: 'compressed image preparation target',
        size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
      try { this.account(4); }
      catch (error) { target.destroy(); throw error; }
      this.resources = { target, layout, pipeline, sampler };
    }
    const { target, layout, pipeline, sampler } = this.resources;
    const group = this.device.createBindGroup({ layout, entries: [
      { binding: 0, resource: texture.createView() }, { binding: 1, resource: sampler },
    ] });
    const encoder = this.device.createCommandEncoder({ label: 'prepare compressed image for compute' });
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
    // Queue ordering places this after writeTexture and before the consumer's work.
    // No CPU wait/readback and no reliance on main-view visibility or shadow draws.
    this.device.queue.submit([encoder.finish()]);
    this.imagesPrepared++;
  }

  diagnostics() {
    return { mode: this.mode, imagesPrepared: this.imagesPrepared, ownedGpuBytes: this.resources ? 4 : 0 };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.resources) {
      this.resources.target.destroy(); this.account(-4); this.resources = undefined;
    }
  }
}
