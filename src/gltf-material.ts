import type { MaterialBinding } from './contracts';
import { m, type Vec4Expr } from './material';
import type { MaterialTextureAsset } from './texture-storage';
import { textureStorageLayout } from './texture-storage';

type Json = Record<string, unknown>;

export interface GltfMaterialOptions {
  /** Decoded or prepared image assets, indexed by glTF image index. */
  images?: ReadonlyMap<number, MaterialTextureAsset>;
  /** Resolves once per image/color-space pair. Must return the requested color space;
   * when present, images is not consulted. Returned assets remain caller-owned. */
  resolveImage?: (imageIndex: number, colorSpace: 'srgb' | 'linear') => MaterialTextureAsset | undefined;
}

export interface GltfMaterialLibrary {
  readonly materialCount: number;
  get(index: number | null): MaterialBinding;
}

function object(value: unknown, label: string): Json {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('GLTF_OBJECT: ' + label);
  return value as Json;
}
function array(value: unknown, label: string, limit = 65536): unknown[] {
  if (!Array.isArray(value) || value.length > limit) throw new Error('GLTF_ARRAY_LIMIT: ' + label);
  return value;
}
function index(value: unknown, length: number, label: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value >= length) throw new Error('GLTF_INDEX: ' + label);
  return value;
}
function number(value: unknown, fallback: number, label: string, min = -Infinity, max = Infinity): number {
  if (value === undefined) return fallback;
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)) || value < min || value > max) throw new Error('GLTF_NUMBER: ' + label);
  return value;
}
function vector(value: unknown, defaults: readonly number[], label: string): number[] {
  if (value === undefined) return [...defaults];
  const values = array(value, label, defaults.length);
  if (values.length !== defaults.length) throw new Error('GLTF_VECTOR: ' + label);
  return values.map(v => number(v, 0, label));
}

/** Parses materials independently of scene geometry and binary buffers. Each material is validated on first access.
 * Keep input metadata and image payloads immutable for the lifetime of the library and its returned bindings. */
export function createGltfMaterialLibrary(json: unknown, options: GltfMaterialOptions = {}): GltfMaterialLibrary {
  const doc = object(json, 'root'), header = object(doc.asset, 'asset');
  if (header.version !== '2.0' || (header.minVersion !== undefined && header.minVersion !== '2.0')) throw new Error('GLTF_VERSION: requires 2.0');
  const materials = array(doc.materials ?? [], 'materials');
  const textures = array(doc.textures ?? [], 'textures');
  const images = array(doc.images ?? [], 'images');
  const samplers = array(doc.samplers ?? [], 'samplers');
  const bindings = new Map<number | null, MaterialBinding>();
  const textureViews = new Map<string, MaterialTextureAsset>();
  const image = (imageIndex: number, colorSpace: 'srgb' | 'linear'): MaterialTextureAsset => {
    const key = imageIndex + ':' + colorSpace;
    const cached = textureViews.get(key);
    if (cached) return cached;
    const original = options.resolveImage ? options.resolveImage(imageIndex, colorSpace) : options.images?.get(imageIndex);
    if (!original) throw new Error('GLTF_IMAGE_NOT_DECODED: ' + imageIndex);
    const prepared = 'kind' in original && original.kind === 'prepared';
    if ((prepared || options.resolveImage) && original.colorSpace !== colorSpace) {
      throw new Error('GLTF_IMAGE_COLOR_SPACE: ' + imageIndex);
    }
    const view = original.colorSpace === colorSpace ? original : { ...original, colorSpace };
    textureStorageLayout(view);
    textureViews.set(key, view);
    return view;
  };
  const parse = (source: Json): MaterialBinding => {
    const pbr = object(source.pbrMetallicRoughness ?? {}, 'pbrMetallicRoughness');
    if (source.occlusionTexture !== undefined) throw new Error('GLTF_UNSUPPORTED_MATERIAL: occlusion texture');
    const extensions = object(source.extensions ?? {}, 'material.extensions');
    for (const name of Object.keys(extensions)) if (name !== 'KHR_materials_unlit') throw new Error('GLTF_UNSUPPORTED_MATERIAL_EXTENSION: ' + name);
    const alphaMode = source.alphaMode ?? 'OPAQUE';
    if (alphaMode !== 'OPAQUE' && alphaMode !== 'MASK') throw new Error('GLTF_UNSUPPORTED_ALPHA_MODE: ' + String(alphaMode));
    if (source.doubleSided !== undefined && typeof source.doubleSided !== 'boolean') throw new Error('GLTF_DOUBLE_SIDED');
    const baseColor = vector(pbr.baseColorFactor, [1, 1, 1, 1], 'baseColorFactor');
    const emissive = vector(source.emissiveFactor, [0, 0, 0], 'emissiveFactor');
    if ([...baseColor, ...emissive].some(v => v < 0 || v > 1)) throw new Error('GLTF_MATERIAL_FACTOR_RANGE');
    const slots: (MaterialTextureAsset | undefined)[] = [];
    const samplerSlots: NonNullable<MaterialBinding['samplers']>[number][] = [];
    const withTexture = (input: Vec4Expr, infoValue: unknown, slot: number, colorSpace: 'srgb' | 'linear') => {
      if (infoValue === undefined) return input;
      const info = object(infoValue, 'textureInfo');
      if ((info.texCoord ?? 0) !== 0 || info.extensions !== undefined) throw new Error('GLTF_UNSUPPORTED_TEXTURE_COORDINATES');
      const texture = object(textures[index(info.index, textures.length, 'textureInfo.index')], 'texture');
      if (texture.extensions !== undefined) throw new Error('GLTF_UNSUPPORTED_TEXTURE_EXTENSION');
      const imageIndex = index(texture.source, images.length, 'texture.source');
      slots[slot] = image(imageIndex, colorSpace);
      if (texture.sampler !== undefined) {
        const sampler = object(samplers[index(texture.sampler, samplers.length, 'texture.sampler')], 'sampler');
        const wrap = (value: unknown): GPUAddressMode => {
          switch (value ?? 10497) { case 10497: return 'repeat'; case 33071: return 'clamp-to-edge'; case 33648: return 'mirror-repeat'; default: throw new Error('GLTF_SAMPLER_WRAP'); }
        };
        const min = sampler.minFilter ?? 9987, mag = sampler.magFilter ?? 9729;
        if (![9728, 9729, 9984, 9985, 9986, 9987].includes(min as number) || (mag !== 9728 && mag !== 9729)) throw new Error('GLTF_SAMPLER_FILTER');
        samplerSlots[slot] = { addressModeU: wrap(sampler.wrapS), addressModeV: wrap(sampler.wrapT),
          magFilter: mag === 9728 ? 'nearest' : 'linear', minFilter: [9728, 9984, 9986].includes(min as number) ? 'nearest' : 'linear',
          mipmapFilter: [9986, 9987].includes(min as number) ? 'linear' : 'nearest',
          ...([9728, 9729].includes(min as number) ? { lodMaxClamp: 0 } : {}) };
      }
      return m.multiply(input, m.texture(slot));
    };
    const color = m.multiply(withTexture(m.parameter(0), pbr.baseColorTexture, 0, 'srgb'), m.vertexColor);
    const mr = withTexture(m.parameter(1), pbr.metallicRoughnessTexture, 1, 'linear');
    const emission = withTexture(m.parameter(2), source.emissiveTexture, 2, 'srgb');
    const normalInfo = source.normalTexture === undefined ? undefined : object(source.normalTexture, 'normalTexture');
    const normal = normalInfo === undefined ? undefined : withTexture(m.constant([1, 1, 1, 1]), normalInfo, 3, 'linear');
    const normalScale = normalInfo === undefined ? undefined : number(normalInfo.scale, 1, 'normalTexture.scale');
    return { definition: { shading: extensions.KHR_materials_unlit !== undefined ? 'unlit' : 'pbr',
      alphaMode: alphaMode === 'MASK' ? 'mask' : 'opaque', doubleSided: source.doubleSided === true,
      baseColor: color, metallic: m.component(mr, 2), roughness: m.component(mr, 1), emissive: emission,
      alphaCutoff: m.component(m.parameter(1), 3),
      ...(normal === undefined ? {} : { normal, normalScale: m.component(m.parameter(3), 0) }) },
      parameters: new Float32Array([...baseColor, 0, number(pbr.roughnessFactor, 1, 'roughnessFactor', 0, 1),
        number(pbr.metallicFactor, 1, 'metallicFactor', 0, 1), number(source.alphaCutoff, 0.5, 'alphaCutoff', 0), ...emissive, 0,
        ...(normalScale === undefined ? [] : [normalScale, 0, 0, 0])]),
      textures: slots, samplers: samplerSlots };
  };
  return { materialCount: materials.length, get(materialIndex: number | null): MaterialBinding {
    if (materialIndex !== null) index(materialIndex, materials.length, 'material');
    const cached = bindings.get(materialIndex);
    if (cached) return cached;
    const binding = parse(materialIndex === null ? {} : object(materials[materialIndex], 'material' + materialIndex));
    bindings.set(materialIndex, binding);
    return binding;
  } };
}
