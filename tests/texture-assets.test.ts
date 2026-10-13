import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  buildMipChain, decodeTexture, textureByteSize, validateTextureAsset,
  type TextureAsset,
} from '../src/texture-assets';

function texture(width: number, height: number, pixels: number[], colorSpace: TextureAsset['colorSpace'] = 'linear'): TextureAsset {
  return { width, height, pixels: new Uint8Array(pixels), colorSpace };
}

describe('CPU texture assets', () => {
  it('filters sRGB in linear light and alpha linearly', () => {
    const levels = buildMipChain(texture(2, 1, [0, 0, 0, 0, 255, 255, 255, 255], 'srgb'));
    expect(levels.map(({ width, height }) => [width, height])).toEqual([[2, 1], [1, 1]]);
    expect([...levels[1].pixels]).toEqual([188, 188, 188, 128]);
  });

  it('filters linear RGB without gamma conversion', () => {
    const levels = buildMipChain(texture(2, 1, [0, 0, 0, 255, 255, 255, 255, 255]));
    expect([...levels[1].pixels]).toEqual([128, 128, 128, 255]);
  });

  it('covers every pixel of odd dimensions with fractional edge weights', () => {
    const pixels: number[] = [];
    for (const red of [0, 40, 80, 120, 160]) {
      for (let x = 0; x < 3; x++) pixels.push(red, 0, 0, 255);
    }
    const levels = buildMipChain(texture(3, 5, pixels));
    expect(levels.map(({ width, height }) => [width, height])).toEqual([[3, 5], [1, 2], [1, 1]]);
    expect(levels[1].pixels[0]).toBe(32);
    expect(levels[1].pixels[4]).toBe(128);
    expect(levels[2].pixels[0]).toBe(80);
    expect(textureByteSize(levels)).toBe(3 * 5 * 4 + 1 * 2 * 4 + 4);
  });

  it('weights horizontal odd-size edges as well as vertical ones', () => {
    const pixels: number[] = [];
    for (let y = 0; y < 3; y++) {
      for (const red of [0, 40, 80, 120, 160]) pixels.push(red, 0, 0, 255);
    }
    const levels = buildMipChain(texture(5, 3, pixels));
    expect(levels.map(({ width, height }) => [width, height])).toEqual([[5, 3], [2, 1], [1, 1]]);
    expect(levels[1].pixels[0]).toBe(32);
    expect(levels[1].pixels[4]).toBe(128);
    expect(levels[2].pixels[0]).toBe(80);
  });

  it('preserves uniform edge texels when floating mip bounds round past the image', () => {
    const rgba = [37, 101, 187, 203];
    for (const colorSpace of ['linear', 'srgb'] as const) {
      for (const [width, height] of [[27, 1], [1, 27], [27, 27]]) {
        const pixels = Array.from({ length: width * height }, () => rgba).flat();
        const levels = buildMipChain(texture(width, height, pixels, colorSpace));
        for (const level of levels) {
          for (let offset = 0; offset < level.pixels.length; offset += 4) {
            expect([...level.pixels.slice(offset, offset + 4)],
              `${colorSpace} ${width}x${height} mip ${level.width}x${level.height} texel ${offset / 4}`).toEqual(rgba);
          }
        }
      }
    }
  });

  it('terminates at 1x1, copies the source, and reports full mip bytes', () => {
    const asset = texture(1, 1, [12, 34, 56, 78]);
    const levels = buildMipChain(asset);
    expect(levels).toHaveLength(1);
    expect(levels[0].pixels).not.toBe(asset.pixels);
    levels[0].pixels[0] = 99;
    expect(asset.pixels[0]).toBe(12);
    expect(textureByteSize(asset)).toBe(4);
    expect(textureByteSize(texture(3, 5, new Array(60).fill(0)))).toBe(72);
  });

  it('rejects invalid sizes, byte counts, and color spaces', () => {
    for (const [width, height] of [[0, 1], [1, 0], [8193, 1], [1.5, 2], [NaN, 1]]) {
      expect(() => validateTextureAsset(texture(width, height, []))).toThrow(/dimensions/);
    }
    expect(() => validateTextureAsset(texture(2, 1, [0, 0, 0, 0]))).toThrow(/RGBA8 bytes/);
    expect(() => validateTextureAsset({ ...texture(1, 1, [0, 0, 0, 0]), colorSpace: 'invalid' as 'srgb' })).toThrow(/colorSpace/);
  });
});

describe('browser texture decoding', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('closes the bitmap after copying decoded pixels', async () => {
    const close = vi.fn();
    const drawImage = vi.fn();
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 1, height: 1, close }));
    vi.stubGlobal('OffscreenCanvas', class {
      getContext() { return { drawImage, getImageData: () => ({ data: new Uint8ClampedArray([1, 2, 3, 4]) }) }; }
    });
    const asset = await decodeTexture(new Blob(['test']), 'linear');
    expect(asset).toEqual(texture(1, 1, [1, 2, 3, 4]));
    expect(drawImage).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
  });

  it('rejects oversized bitmaps before canvas allocation and still closes them', async () => {
    const close = vi.fn();
    const canvas = vi.fn();
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 8193, height: 1, close }));
    vi.stubGlobal('OffscreenCanvas', canvas);
    await expect(decodeTexture(new Blob(), 'srgb')).rejects.toThrow(/dimensions/);
    expect(canvas).not.toHaveBeenCalled();
    expect(close).toHaveBeenCalledOnce();
  });
});
