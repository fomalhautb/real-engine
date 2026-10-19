import { compileMaterial, DEFAULT_MATERIAL_DEFINITION, type MaterialProgram } from './material';
import { buildMipChain } from './texture-assets';
import { textureStorageLayout, type MaterialTextureAsset, type PreparedTextureAsset, type TextureStorageLayout } from './texture-storage';
import type { MaterialBinding } from './contracts';
import { CompressedTexturePreparation } from './gpu-texture-preparation';
import { GpuTextureUpload } from './gpu-texture-upload';

type ImageResource = { texture: GPUTexture; bytes: number; refs: number;
  asset: MaterialTextureAsset; layout: TextureStorageLayout };
const MATERIAL_UNIFORM_BYTES = 320;
const MAX_MATERIAL_RESOURCES = 4096;
type MaterialInputs = { parameters: Float32Array<ArrayBuffer>;
  textures: (MaterialTextureAsset | undefined)[]; samplers: (GPUSamplerDescriptor | undefined)[] };
export interface TextureReplacementResult { materials: number; slots: number }
export interface PreparedTextureUpload {
  readonly asset: PreparedTextureAsset;
  readonly complete: boolean;
  readonly disposed: boolean;
  /** Native block bytes queued this call; never exceeds maxBytes. */
  upload(maxBytes: number): number;
  diagnostics(): { complete: boolean; remainingBytes: number; nextRowBytes: number; allocationBytes: number };
  /** Cancel or release the upload's pin. Published material references remain valid. */
  dispose(): void;
}
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
  private readonly images = new Map<MaterialTextureAsset, ImageResource>();
  private readonly materials = new Set<GpuMaterial>();
  private readonly inputs = new Map<GpuMaterial, MaterialInputs>();
  private readonly uploads = new Set<PreparedTextureUpload>();
  private readonly pendingUploads = new Map<MaterialTextureAsset, PreparedTextureUpload>();
  private readonly fallback: GPUTexture;
  private disposed = false;
  private imageBytes = 0;
  private readonly texturePreparation: CompressedTexturePreparation;

  constructor(private readonly device: GPUDevice, private readonly account: (bytes: number) => void,
    private readonly imageBudgetBytes = 256 * 1024 * 1024,
    adapterInfo: Pick<GPUAdapterInfo, 'architecture' | 'description'> | undefined = device.adapterInfo) {
    if (!Number.isSafeInteger(imageBudgetBytes) || imageBudgetBytes < 0) throw new Error('INVALID_TEXTURE_BUDGET');
    this.texturePreparation = new CompressedTexturePreparation(device, account, adapterInfo);
    this.layout = device.createBindGroupLayout({ label: 'shared material inputs', entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: MATERIAL_UNIFORM_BYTES } },
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
    if (this.materials.size >= MAX_MATERIAL_RESOURCES) throw new Error('MATERIAL_RESOURCE_LIMIT');
    const program = compileMaterial(binding?.definition ?? DEFAULT_MATERIAL_DEFINITION);
    const parameters = this.parameters(program, binding?.parameters);
    if ((binding?.samplers?.length ?? 0) > 4) throw new Error('MATERIAL_SAMPLER_COUNT');
    const samplers = Array.from({ length: 4 }, (_, slot) => {
      const descriptor = binding?.samplers?.[slot];
      return descriptor === undefined ? undefined : materialSampler(descriptor);
    });
    const textures = Array.from({ length: 4 }, (_, slot) => program.textureSlots.includes(slot) ? binding?.textures?.[slot] : undefined);
    const inputs = { parameters, samplers, textures };
    const material = this.allocate(program, inputs);
    this.materials.add(material); this.inputs.set(material, inputs);
    return material;
  }

  /** Allocate complete immutable bindings without publishing or compiling a pipeline. */
  private allocate(program: MaterialProgram, inputs: MaterialInputs): GpuMaterial {
    const { parameters, samplers, textures } = inputs;
    const retained: ImageResource[] = [];
    let uniform: GPUBuffer | undefined;
    try {
      const views = Array.from({ length: 4 }, () => this.fallback.createView());
      const textureInfo = new Float32Array(16);
      for (let slot = 0; slot < 4; slot++) {
        textureInfo[slot * 4 + 1] = 1; textureInfo[slot * 4 + 2] = 1;
      }
      for (const slot of program.textureSlots) {
        const asset = textures[slot];
        if (!asset) throw new Error('MATERIAL_TEXTURE_MISSING: slot ' + slot);
        const image = this.retainImage(asset); retained.push(image); views[slot] = image.texture.createView();
        const layout = image.layout;
        textureInfo[slot * 4] = layout.baseMipLevel;
        // Preserve source-space LOD even when odd dimensions round down in a tail.
        textureInfo[slot * 4 + 1] = layout.sourceWidth * 2 ** -layout.baseMipLevel / layout.width;
        textureInfo[slot * 4 + 2] = layout.sourceHeight * 2 ** -layout.baseMipLevel / layout.height;
      }
      const data = new Float32Array(MATERIAL_UNIFORM_BYTES / 4);
      data.set(parameters); data.set(textureInfo, 64);
      uniform = this.device.createBuffer({ label: 'material parameters and texture residency', size: MATERIAL_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.account(MATERIAL_UNIFORM_BYTES); this.device.queue.writeBuffer(uniform, 0, data);
      const bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
        { binding: 0, resource: { buffer: uniform } },
        ...views.map((resource, index) => ({ binding: index + 1, resource })),
        ...[0, 1, 2, 3].map(slot => {
          const descriptor = samplers[slot], baseMip = textureInfo[slot * 4];
          return { binding: slot + 5, resource: descriptor ? this.device.createSampler({ ...descriptor,
            lodMinClamp: Math.max(0, (descriptor.lodMinClamp ?? 0) - baseMip),
            lodMaxClamp: Math.max(0, (descriptor.lodMaxClamp ?? 32) - baseMip),
          }) : this.sampler };
        }),
      ] });
      const result: GpuMaterial = { program, bindGroup, uniform, images: retained, disposed: false };
      return result;
    } catch (error) {
      if (uniform) { uniform.destroy(); this.account(-MATERIAL_UNIFORM_BYTES); }
      for (const image of retained) this.releaseImage(image);
      throw error;
    }
  }

  updateParameters(material: GpuMaterial, values: Float32Array): void {
    if (material.disposed || this.disposed) throw new Error('MATERIAL_DISPOSED');
    const inputs = this.inputs.get(material);
    if (!inputs) throw new Error('MATERIAL_NOT_OWNED');
    const parameters = this.parameters(material.program, values);
    this.device.queue.writeBuffer(material.uniform, 0, parameters);
    inputs.parameters = parameters;
  }

  /** Replace every currently retained use, including materials awaiting pipeline
   * compilation. Prepare all bindings first; any failure preserves all old uses.
   * No shader compilation, async gap, or material-handle change. The caller must
   * invalidate dependent lighting/history once when materials > 0. */
  replaceTexture(source: MaterialTextureAsset, replacement: MaterialTextureAsset): TextureReplacementResult {
    if (this.disposed) throw new Error('MATERIAL_RESOURCES_DISPOSED');
    if (source === replacement || !this.images.has(source)) return { materials: 0, slots: 0 };
    const prepared: { target: GpuMaterial; next: GpuMaterial; inputs: MaterialInputs }[] = [];
    let slots = 0;
    try {
      // Both traversal and the staging peak are bounded by MAX_MATERIAL_RESOURCES.
      for (const target of this.materials) {
        const previous = this.inputs.get(target)!;
        if (!previous.textures.includes(source)) continue;
        const textures = previous.textures.map(asset => {
          if (asset !== source) return asset;
          slots++; return replacement;
        });
        const inputs = { parameters: previous.parameters.slice(), samplers: previous.samplers, textures };
        prepared.push({ target, next: this.allocate(target.program, inputs), inputs });
      }
    } catch (error) {
      for (const item of prepared) this.destroyMaterialResources(item.next);
      throw error;
    }
    // GPU allocation/upload and all potentially invalid binding construction have
    // completed. JavaScript cannot enter render/trace between these synchronous swaps.
    for (const { target, next, inputs } of prepared) {
      const previous = { uniform: target.uniform, images: target.images };
      Object.assign(target, { bindGroup: next.bindGroup, uniform: next.uniform, images: next.images });
      this.inputs.set(target, inputs);
      this.destroyMaterialResources(previous);
    }
    return { materials: prepared.length, slots };
  }

  /** Reserve the final allocation, then upload a prepared tail over bounded ticks.
   * Partial contents cannot be bound by create/replaceTexture. After complete,
   * the handle pins the image until disposed; publication is a separate atomic
   * replaceTexture call. All pending allocations count against the image budget. */
  stageTexture(asset: PreparedTextureAsset): PreparedTextureUpload {
    if (this.disposed) throw new Error('MATERIAL_RESOURCES_DISPOSED');
    if (this.uploads.size >= 64) throw new Error('MATERIAL_TEXTURE_UPLOAD_LIMIT');
    if (!asset || asset.kind !== 'prepared') throw new Error('MATERIAL_TEXTURE_PREPARED_REQUIRED');
    if (this.pendingUploads.has(asset)) throw new Error('MATERIAL_TEXTURE_UPLOAD_IN_PROGRESS');
    let image = this.images.get(asset), uploader: GpuTextureUpload | undefined;
    let complete = !!image, disposed = false;
    if (image) image.refs++;
    else {
      const layout = this.checkedImageLayout(asset);
      const texture = this.device.createTexture({ label: 'staged material image (' + asset.colorSpace + ')',
        size: [layout.width, layout.height], mipLevelCount: layout.levels.length, format: layout.gpuFormat,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      try {
        uploader = new GpuTextureUpload(this.device, texture, layout, asset.levels.map(level => level.data));
        this.account(layout.byteLength);
      } catch (error) { texture.destroy(); throw error; }
      image = { texture, bytes: layout.byteLength, refs: 1, asset, layout };
      this.imageBytes += image.bytes;
    }
    const owned = image;
    const handle: PreparedTextureUpload = {
      asset,
      get complete() { return complete; }, get disposed() { return disposed; },
      upload: maxBytes => {
        if (disposed || this.disposed) throw new Error('MATERIAL_TEXTURE_UPLOAD_DISPOSED');
        if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error('MATERIAL_TEXTURE_UPLOAD_BUDGET');
        if (complete) return 0;
        const bytes = uploader!.upload(maxBytes);
        if (uploader!.done) {
          if (owned.layout.requiredFeature) this.texturePreparation.prepare(owned.texture);
          this.pendingUploads.delete(asset); this.images.set(asset, owned); complete = true;
        }
        return bytes;
      },
      diagnostics: () => ({ complete, remainingBytes: uploader?.remainingBytes ?? 0,
        nextRowBytes: uploader?.nextRowBytes ?? 0, allocationBytes: owned.bytes }),
      dispose: () => {
        if (disposed) return;
        disposed = true;
        this.pendingUploads.delete(asset); this.uploads.delete(handle); this.releaseImage(owned);
      },
    };
    this.uploads.add(handle);
    if (!complete) this.pendingUploads.set(asset, handle);
    return handle;
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

  private checkedImageLayout(asset: MaterialTextureAsset): TextureStorageLayout {
    const layout = textureStorageLayout(asset);
    if (layout.width > this.device.limits.maxTextureDimension2D || layout.height > this.device.limits.maxTextureDimension2D) throw new Error('MATERIAL_TEXTURE_DEVICE_LIMIT');
    if (layout.requiredFeature && !this.device.features.has(layout.requiredFeature)) {
      throw new Error('MATERIAL_TEXTURE_FEATURE: ' + layout.requiredFeature);
    }
    const bytes = layout.byteLength;
    if (this.imageBytes + bytes > this.imageBudgetBytes) throw new Error('MATERIAL_TEXTURE_BUDGET: replacement must fit alongside currently referenced images');
    return layout;
  }

  private retainImage(asset: MaterialTextureAsset): ImageResource {
    if (this.pendingUploads.has(asset)) throw new Error('MATERIAL_TEXTURE_PENDING_UPLOAD');
    const cached = this.images.get(asset);
    if (cached) { cached.refs++; return cached; }
    const layout = this.checkedImageLayout(asset), bytes = layout.byteLength;
    const levels = 'kind' in asset ? asset.levels.map(level => level.data)
      : buildMipChain(asset).map(level => level.pixels);
    const texture = this.device.createTexture({ label: 'material image (' + asset.colorSpace + ')',
      size: [layout.width, layout.height], mipLevelCount: levels.length,
      format: layout.gpuFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    try {
      for (let mipLevel = 0; mipLevel < levels.length; mipLevel++) {
        const level = layout.levels[mipLevel];
        this.device.queue.writeTexture({ texture, mipLevel }, levels[mipLevel],
          { bytesPerRow: level.bytesPerRow, rowsPerImage: level.rowsPerImage }, [level.copyWidth, level.copyHeight]);
      }
      if (layout.requiredFeature) this.texturePreparation.prepare(texture);
    } catch (error) { texture.destroy(); throw error; }
    const image = { texture, bytes, refs: 1, asset, layout };
    this.images.set(asset, image); this.imageBytes += bytes; this.account(bytes);
    return image;
  }

  private releaseImage(image: ImageResource): void {
    if (--image.refs !== 0) return;
    image.texture.destroy(); this.imageBytes -= image.bytes; this.account(-image.bytes); this.images.delete(image.asset);
  }

  release(material: GpuMaterial): void {
    if (material.disposed) return;
    if (!this.materials.has(material)) throw new Error('MATERIAL_NOT_OWNED');
    material.disposed = true;
    this.destroyMaterialResources(material);
    this.materials.delete(material);
    this.inputs.delete(material);
  }

  private destroyMaterialResources(material: Pick<GpuMaterial, 'uniform' | 'images'>): void {
    material.uniform.destroy(); this.account(-MATERIAL_UNIFORM_BYTES);
    for (const image of material.images) this.releaseImage(image);
  }

  hasProgram(key: string): boolean { return Array.from(this.materials).some(material => material.program.key === key); }

  diagnostics() { return { materials: this.materials.size, images: this.images.size, imageBytes: this.imageBytes }; }

  preparationDiagnostics() { return this.texturePreparation.diagnostics(); }

  uploadDiagnostics() {
    return { active: this.uploads.size, pending: this.pendingUploads.size,
      pendingImageBytes: Array.from(this.pendingUploads.values()).reduce((sum, item) => sum + item.diagnostics().allocationBytes, 0) };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const material of Array.from(this.materials)) this.release(material);
    for (const upload of Array.from(this.uploads)) upload.dispose();
    this.texturePreparation.dispose();
    this.fallback.destroy(); this.account(-4);
  }
}
