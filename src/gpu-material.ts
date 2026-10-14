import { compileMaterial, DEFAULT_MATERIAL_DEFINITION, type MaterialProgram } from './material';
import { buildMipChain, textureByteSize, type TextureAsset } from './texture-assets';
import type { MaterialBinding } from './contracts';

type ImageResource = { texture: GPUTexture; bytes: number; refs: number; asset: TextureAsset };
function materialSampler(source: GPUSamplerDescriptor): GPUSamplerDescriptor {
  if (!source || typeof source !== 'object' || source.compare !== undefined) throw new Error('MATERIAL_SAMPLER_DESCRIPTOR');
  const descriptor: GPUSamplerDescriptor = { addressModeU: 'repeat', addressModeV: 'repeat',
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', ...source };
  for (const mode of [descriptor.addressModeU, descriptor.addressModeV, descriptor.addressModeW ?? 'clamp-to-edge']) {
    if (!['repeat', 'mirror-repeat', 'clamp-to-edge'].includes(mode!)) throw new Error('MATERIAL_SAMPLER_ADDRESS');
  }
  for (const mode of [descriptor.magFilter, descriptor.minFilter, descriptor.mipmapFilter]) {
    if (mode !== 'linear' && mode !== 'nearest') throw new Error('MATERIAL_SAMPLER_FILTER');
  }
  const min = descriptor.lodMinClamp ?? 0, max = descriptor.lodMaxClamp ?? 32, anisotropy = descriptor.maxAnisotropy ?? 1;
  if (![min, max].every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value))) || min < 0 || max < min) throw new Error('MATERIAL_SAMPLER_LOD');
  if (!Number.isInteger(anisotropy) || anisotropy < 1 || anisotropy > 16 ||
    (anisotropy > 1 && [descriptor.magFilter, descriptor.minFilter, descriptor.mipmapFilter].some(mode => mode !== 'linear'))) throw new Error('MATERIAL_SAMPLER_ANISOTROPY');
  return descriptor;
}
export interface GpuMaterial {
  readonly program: MaterialProgram;
  readonly bindGroup: GPUBindGroup;
  readonly uniform: GPUBuffer;
  readonly images: ImageResource[];
  disposed: boolean;
}

/** Owns ref-counted images and material parameters, independently of pipelines.
 * Image object identity is immutable asset identity; use a new asset to replace pixels. */
export class MaterialResources {
  readonly layout: GPUBindGroupLayout;
  private readonly sampler: GPUSampler;
  private readonly images = new Map<TextureAsset, ImageResource>();
  private readonly materials = new Set<GpuMaterial>();
  private readonly fallback: GPUTexture;
  private disposed = false;
  private imageBytes = 0;

  constructor(private readonly device: GPUDevice, private readonly account: (bytes: number) => void,
    private readonly imageBudgetBytes = 256 * 1024 * 1024) {
    if (!Number.isSafeInteger(imageBudgetBytes) || imageBudgetBytes < 0) throw new Error('INVALID_TEXTURE_BUDGET');
    this.layout = device.createBindGroupLayout({ label: 'shared material inputs', entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: 256 } },
      ...[1, 2, 3, 4].map(binding => ({ binding, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, texture: { sampleType: 'float' as const } })),
      ...[5, 6, 7, 8].map(binding => ({ binding, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, sampler: { type: 'filtering' as const } })),
    ] });
    this.sampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    this.fallback = device.createTexture({ label: 'unused material texture', size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: this.fallback }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1]);
    account(4);
  }

  create(binding?: MaterialBinding): GpuMaterial {
    if (this.disposed) throw new Error('MATERIAL_RESOURCES_DISPOSED');
    const program = compileMaterial(binding?.definition ?? DEFAULT_MATERIAL_DEFINITION);
    const parameters = this.parameters(program, binding?.parameters);
    if ((binding?.samplers?.length ?? 0) > 4) throw new Error('MATERIAL_SAMPLER_COUNT');
    const samplers = Array.from({ length: 4 }, (_, slot) => {
      const descriptor = binding?.samplers?.[slot];
      return descriptor === undefined ? undefined : materialSampler(descriptor);
    });
    const retained: ImageResource[] = [];
    let uniform: GPUBuffer | undefined;
    try {
      const views = Array.from({ length: 4 }, () => this.fallback.createView());
      for (const slot of program.textureSlots) {
        const asset = binding?.textures?.[slot];
        if (!asset) throw new Error('MATERIAL_TEXTURE_MISSING: slot ' + slot);
        const image = this.retainImage(asset); retained.push(image); views[slot] = image.texture.createView();
      }
      uniform = this.device.createBuffer({ label: 'material vec4 parameters', size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.account(256); this.device.queue.writeBuffer(uniform, 0, parameters);
      const bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
        { binding: 0, resource: { buffer: uniform } },
        ...views.map((resource, index) => ({ binding: index + 1, resource })),
        ...[0, 1, 2, 3].map(slot => ({ binding: slot + 5,
          resource: samplers[slot] ? this.device.createSampler(samplers[slot]) : this.sampler })),
      ] });
      const result: GpuMaterial = { program, bindGroup, uniform, images: retained, disposed: false };
      this.materials.add(result);
      return result;
    } catch (error) {
      if (uniform) { uniform.destroy(); this.account(-256); }
      for (const image of retained) this.releaseImage(image);
      throw error;
    }
  }

  updateParameters(material: GpuMaterial, values: Float32Array): void {
    if (material.disposed || this.disposed) throw new Error('MATERIAL_DISPOSED');
    this.device.queue.writeBuffer(material.uniform, 0, this.parameters(material.program, values));
  }

  private parameters(program: MaterialProgram, values?: Float32Array): Float32Array<ArrayBuffer> {
    if ((values?.length ?? 0) < program.parameterCount * 4 || (values?.length ?? 0) > 64) throw new Error('MATERIAL_PARAMETER_COUNT');
    const data = new Float32Array(64);
    if (values) {
      if (!Array.from(values).every(Number.isFinite)) throw new Error('MATERIAL_PARAMETER_NONFINITE');
      data.set(values);
    }
    return data;
  }

  private retainImage(asset: TextureAsset): ImageResource {
    const cached = this.images.get(asset);
    if (cached) { cached.refs++; return cached; }
    if (asset.width > this.device.limits.maxTextureDimension2D || asset.height > this.device.limits.maxTextureDimension2D) throw new Error('MATERIAL_TEXTURE_DEVICE_LIMIT');
    const bytes = textureByteSize(asset);
    if (this.imageBytes + bytes > this.imageBudgetBytes) throw new Error('MATERIAL_TEXTURE_BUDGET: replacement must fit alongside currently referenced images');
    const levels = buildMipChain(asset);
    const texture = this.device.createTexture({ label: 'material image (' + asset.colorSpace + ')',
      size: [asset.width, asset.height], mipLevelCount: levels.length,
      format: asset.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    try {
      for (let mipLevel = 0; mipLevel < levels.length; mipLevel++) {
        const level = levels[mipLevel];
        this.device.queue.writeTexture({ texture, mipLevel }, new Uint8Array(level.pixels), { bytesPerRow: level.width * 4 }, [level.width, level.height]);
      }
    } catch (error) { texture.destroy(); throw error; }
    const image = { texture, bytes, refs: 1, asset };
    this.images.set(asset, image); this.imageBytes += bytes; this.account(bytes);
    return image;
  }

  private releaseImage(image: ImageResource): void {
    if (--image.refs !== 0) return;
    image.texture.destroy(); this.imageBytes -= image.bytes; this.account(-image.bytes); this.images.delete(image.asset);
  }

  release(material: GpuMaterial): void {
    if (material.disposed) return;
    material.disposed = true; material.uniform.destroy(); this.account(-256);
    for (const image of material.images) this.releaseImage(image);
    this.materials.delete(material);
  }

  hasProgram(key: string): boolean { return Array.from(this.materials).some(material => material.program.key === key); }

  diagnostics() { return { materials: this.materials.size, images: this.images.size, imageBytes: this.imageBytes }; }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const material of Array.from(this.materials)) this.release(material);
    this.fallback.destroy(); this.account(-4);
  }
}
