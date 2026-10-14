/** CPU-side RGBA8 texture data. Assets are treated as immutable after upload. */
export interface TextureAsset {
  width: number;
  height: number;
  pixels: Uint8Array;
  colorSpace: 'srgb' | 'linear';
}

export interface TextureMipLevel {
  width: number;
  height: number;
  pixels: Uint8Array;
}

export const MAX_TEXTURE_DIMENSION = 8192;
const CHANNELS = 4;

function checkedByteLength(width: number, height: number): number {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
      width < 1 || height < 1 || width > MAX_TEXTURE_DIMENSION || height > MAX_TEXTURE_DIMENSION) {
    throw new RangeError(`TEXTURE_ASSET: dimensions must be integers from 1 to ${MAX_TEXTURE_DIMENSION}`);
  }
  const byteLength = width * height * CHANNELS;
  if (!Number.isSafeInteger(byteLength)) throw new RangeError('TEXTURE_ASSET: byte count exceeds safe integer range');
  return byteLength;
}

export function validateTextureAsset(asset: TextureAsset): void {
  if (!asset || typeof asset !== 'object') throw new TypeError('TEXTURE_ASSET: expected an asset');
  const expected = checkedByteLength(asset.width, asset.height);
  if (!(asset.pixels instanceof Uint8Array) || asset.pixels.byteLength !== expected) {
    throw new TypeError(`TEXTURE_ASSET: expected ${expected} RGBA8 bytes`);
  }
  if (asset.colorSpace !== 'srgb' && asset.colorSpace !== 'linear') {
    throw new TypeError('TEXTURE_ASSET: colorSpace must be srgb or linear');
  }
}

function srgbToLinear(value: number): number {
  const normalized = value / 255;
  return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(value: number): number {
  const encoded = value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
  return Math.round(Math.max(0, Math.min(1, encoded)) * 255);
}

/** Generates full RGBA8 mip levels using source-area coverage for odd sizes.
 * RGB is filtered in linear light for sRGB assets; alpha always stays linear. */
export function buildMipChain(asset: TextureAsset): TextureMipLevel[] {
  validateTextureAsset(asset);
  const levels: TextureMipLevel[] = [{
    width: asset.width,
    height: asset.height,
    pixels: new Uint8Array(asset.pixels),
  }];

  while (levels.at(-1)!.width > 1 || levels.at(-1)!.height > 1) {
    const source = levels.at(-1)!;
    const width = Math.max(1, Math.floor(source.width / 2));
    const height = Math.max(1, Math.floor(source.height / 2));
    const pixels = new Uint8Array(checkedByteLength(width, height));
    const xScale = source.width / width;
    const yScale = source.height / height;
    for (let y = 0; y < height; y++) {
      const yStart = y * yScale;
      const yEnd = Math.min(source.height, (y + 1) * yScale);
      for (let x = 0; x < width; x++) {
        const xStart = x * xScale;
        const xEnd = Math.min(source.width, (x + 1) * xScale);
        const sums = [0, 0, 0, 0];
        for (let sy = Math.floor(yStart); sy < Math.min(source.height, Math.ceil(yEnd)); sy++) {
          const yWeight = Math.min(sy + 1, yEnd) - Math.max(sy, yStart);
          if (yWeight <= 0) continue;
          for (let sx = Math.floor(xStart); sx < Math.min(source.width, Math.ceil(xEnd)); sx++) {
            const weight = yWeight * (Math.min(sx + 1, xEnd) - Math.max(sx, xStart));
            if (weight <= 0) continue;
            const sourceOffset = (sy * source.width + sx) * CHANNELS;
            for (let channel = 0; channel < 3; channel++) {
              const value = source.pixels[sourceOffset + channel];
              sums[channel] += weight * (asset.colorSpace === 'srgb' ? srgbToLinear(value) : value / 255);
            }
            sums[3] += weight * source.pixels[sourceOffset + 3] / 255;
          }
        }
        const area = (xEnd - xStart) * (yEnd - yStart);
        const offset = (y * width + x) * CHANNELS;
        for (let channel = 0; channel < 3; channel++) {
          const value = sums[channel] / area;
          pixels[offset + channel] = asset.colorSpace === 'srgb' ? linearToSrgb(value) : Math.round(value * 255);
        }
        pixels[offset + 3] = Math.round(sums[3] / area * 255);
      }
    }
    levels.push({ width, height, pixels });
  }
  return levels;
}

/** Total RGBA8 storage for a supplied chain or for an asset's full mip layout. */
export function textureByteSize(texture: TextureAsset | readonly TextureMipLevel[]): number {
  if (Array.isArray(texture)) {
    let bytes = 0;
    for (const level of texture) {
      const expected = checkedByteLength(level.width, level.height);
      if (!(level.pixels instanceof Uint8Array) || level.pixels.byteLength !== expected) {
        throw new TypeError(`TEXTURE_ASSET: expected ${expected} RGBA8 bytes`);
      }
      bytes += expected;
    }
    return bytes;
  }
  const asset = texture as TextureAsset;
  validateTextureAsset(asset);
  let width = asset.width;
  let height = asset.height;
  let bytes = 0;
  for (;;) {
    bytes += checkedByteLength(width, height);
    if (width === 1 && height === 1) break;
    width = Math.max(1, Math.floor(width / 2));
    height = Math.max(1, Math.floor(height / 2));
  }
  return bytes;
}

/** Decode through browser canvas APIs. Canvas readback can lose RGB values at
 * fully transparent pixels because 2D canvases may internally premultiply. */
export async function decodeTexture(blob: Blob, colorSpace: TextureAsset['colorSpace'], maxDecodedBytes = Number.MAX_SAFE_INTEGER): Promise<TextureAsset> {
  if (colorSpace !== 'srgb' && colorSpace !== 'linear') {
    throw new TypeError('TEXTURE_ASSET: colorSpace must be srgb or linear');
  }
  if (!(blob instanceof Blob)) throw new TypeError('TEXTURE_ASSET: expected a Blob');
  if (!Number.isSafeInteger(maxDecodedBytes) || maxDecodedBytes < 0) throw new RangeError('TEXTURE_DECODE_BUDGET: invalid limit');
  if (typeof createImageBitmap !== 'function') throw new Error('TEXTURE_ASSET: createImageBitmap unavailable');

  const bitmap = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  try {
    const { width, height } = bitmap;
    const decodedBytes = checkedByteLength(width, height);
    if (decodedBytes > maxDecodedBytes) throw new RangeError('TEXTURE_DECODE_BUDGET: decoded image exceeds byte limit');
    let context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
    if (typeof OffscreenCanvas !== 'undefined') {
      context = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
    } else if (typeof document !== 'undefined') {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      context = canvas.getContext('2d', { willReadFrequently: true });
    }
    if (!context) throw new Error('TEXTURE_ASSET: 2D canvas unavailable');
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, width, height).data;
    const pixels = new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength);
    const asset: TextureAsset = { width, height, pixels, colorSpace };
    validateTextureAsset(asset);
    return asset;
  } finally {
    bitmap.close();
  }
}
