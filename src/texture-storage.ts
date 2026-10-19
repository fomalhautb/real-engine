import { MAX_TEXTURE_DIMENSION, validateTextureAsset, type TextureAsset } from './texture-assets';

export const TEXTURE_COMPRESSION_FEATURES = [
  'texture-compression-bc',
  'texture-compression-etc2',
  'texture-compression-astc',
] as const satisfies readonly GPUFeatureName[];

export type PreparedTextureFormat = 'rgba8' | 'bc7' | 'etc2-rgba8' | 'astc-4x4';

/** Already encoded immutable mip payloads. width/height describe the full source
 * image; baseMipLevel is the first resident source mip, followed by every level
 * through 1x1. Sampler clamps and explicit query LOD remain in source-mip units.
 * Use a new asset object and setMaterial to replace residency. This contract
 * does not schedule IO, encode images, or stream mip levels automatically. */
export interface PreparedTextureAsset {
  kind: 'prepared';
  width: number;
  height: number;
  colorSpace: 'srgb' | 'linear';
  format: PreparedTextureFormat;
  baseMipLevel: number;
  levels: readonly { width: number; height: number; data: Uint8Array }[];
}

export type MaterialTextureAsset = TextureAsset | PreparedTextureAsset;

export interface TextureStorageLevel {
  width: number;
  height: number;
  copyWidth: number;
  copyHeight: number;
  bytesPerRow: number;
  rowsPerImage: number;
  byteLength: number;
}

export interface TextureStorageLayout {
  gpuFormat: GPUTextureFormat;
  requiredFeature?: GPUFeatureName;
  sourceWidth: number;
  sourceHeight: number;
  width: number;
  height: number;
  baseMipLevel: number;
  byteLength: number;
  levels: readonly TextureStorageLevel[];
}

function dimensions(width: number, height: number): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height)
    || width < 1 || height < 1 || width > MAX_TEXTURE_DIMENSION || height > MAX_TEXTURE_DIMENSION) {
    throw new RangeError(`TEXTURE_STORAGE: dimensions must be integers from 1 to ${MAX_TEXTURE_DIMENSION}`);
  }
}

function mipSize(value: number, level: number): number {
  return Math.max(1, Math.floor(value / 2 ** level));
}

function mipCount(width: number, height: number): number {
  return 1 + Math.floor(Math.log2(Math.max(width, height)));
}

function levelLayout(width: number, height: number, blockSize: number, blockBytes: number): TextureStorageLevel {
  const blocksAcross = Math.ceil(width / blockSize);
  const blocksDown = Math.ceil(height / blockSize);
  const bytesPerRow = blocksAcross * blockBytes;
  return {
    width, height,
    copyWidth: blocksAcross * blockSize,
    copyHeight: blocksDown * blockSize,
    bytesPerRow,
    rowsPerImage: blocksDown,
    byteLength: bytesPerRow * blocksDown,
  };
}

function formatInfo(format: PreparedTextureFormat, colorSpace: PreparedTextureAsset['colorSpace']): {
  gpuFormat: GPUTextureFormat; requiredFeature?: GPUFeatureName; blockSize: number; blockBytes: number;
} {
  switch (format) {
    case 'rgba8': return { gpuFormat: colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm', blockSize: 1, blockBytes: 4 };
    case 'bc7': return { gpuFormat: colorSpace === 'srgb' ? 'bc7-rgba-unorm-srgb' : 'bc7-rgba-unorm', requiredFeature: 'texture-compression-bc', blockSize: 4, blockBytes: 16 };
    case 'etc2-rgba8': return { gpuFormat: colorSpace === 'srgb' ? 'etc2-rgba8unorm-srgb' : 'etc2-rgba8unorm', requiredFeature: 'texture-compression-etc2', blockSize: 4, blockBytes: 16 };
    case 'astc-4x4': return { gpuFormat: colorSpace === 'srgb' ? 'astc-4x4-unorm-srgb' : 'astc-4x4-unorm', requiredFeature: 'texture-compression-astc', blockSize: 4, blockBytes: 16 };
    default: throw new TypeError('TEXTURE_STORAGE: unsupported format');
  }
}

/** Validates storage and upload extents without decoding, copying, or generating image data. */
export function textureStorageLayout(asset: MaterialTextureAsset): TextureStorageLayout {
  if (!asset || typeof asset !== 'object') throw new TypeError('TEXTURE_STORAGE: expected a texture asset');
  if (!('kind' in asset)) {
    validateTextureAsset(asset as TextureAsset);
    const legacy = asset as TextureAsset;
    const levels: TextureStorageLevel[] = [];
    const count = mipCount(legacy.width, legacy.height);
    let byteLength = 0;
    for (let mip = 0; mip < count; mip++) {
      const level = levelLayout(mipSize(legacy.width, mip), mipSize(legacy.height, mip), 1, 4);
      byteLength += level.byteLength;
      levels.push(level);
    }
    return {
      gpuFormat: legacy.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm',
      sourceWidth: legacy.width, sourceHeight: legacy.height,
      width: legacy.width, height: legacy.height, baseMipLevel: 0, byteLength, levels,
    };
  }

  if (asset.kind !== 'prepared') throw new TypeError('TEXTURE_STORAGE: unknown kind');
  const prepared = asset as PreparedTextureAsset;
  dimensions(prepared.width, prepared.height);
  if (prepared.colorSpace !== 'srgb' && prepared.colorSpace !== 'linear') {
    throw new TypeError('TEXTURE_STORAGE: colorSpace must be srgb or linear');
  }
  const info = formatInfo(prepared.format, prepared.colorSpace);
  const count = mipCount(prepared.width, prepared.height);
  if (!Number.isSafeInteger(prepared.baseMipLevel) || prepared.baseMipLevel < 0 || prepared.baseMipLevel >= count) {
    throw new RangeError('TEXTURE_STORAGE: invalid baseMipLevel');
  }
  const width = mipSize(prepared.width, prepared.baseMipLevel);
  const height = mipSize(prepared.height, prepared.baseMipLevel);
  if (info.blockSize === 4 && (width % 4 !== 0 || height % 4 !== 0)) {
    throw new RangeError('TEXTURE_STORAGE: compressed resident base must be divisible by four');
  }
  if (!Array.isArray(prepared.levels) || prepared.levels.length !== count - prepared.baseMipLevel) {
    throw new RangeError('TEXTURE_STORAGE: levels must form a complete resident mip chain');
  }
  const levels: TextureStorageLevel[] = [];
  let byteLength = 0;
  for (let index = 0; index < prepared.levels.length; index++) {
    const mip = prepared.baseMipLevel + index;
    const expectedWidth = mipSize(prepared.width, mip);
    const expectedHeight = mipSize(prepared.height, mip);
    const input = prepared.levels[index];
    if (!input || input.width !== expectedWidth || input.height !== expectedHeight) {
      throw new RangeError(`TEXTURE_STORAGE: invalid dimensions for mip ${mip}`);
    }
    const level = levelLayout(expectedWidth, expectedHeight, info.blockSize, info.blockBytes);
    if (!(input.data instanceof Uint8Array) || input.data.byteLength !== level.byteLength) {
      throw new TypeError(`TEXTURE_STORAGE: mip ${mip} requires ${level.byteLength} bytes`);
    }
    byteLength += level.byteLength;
    levels.push(level);
  }
  return {
    gpuFormat: info.gpuFormat, requiredFeature: info.requiredFeature,
    sourceWidth: prepared.width, sourceHeight: prepared.height,
    width, height, baseMipLevel: prepared.baseMipLevel, byteLength, levels,
  };
}
