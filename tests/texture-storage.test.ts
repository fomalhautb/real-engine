import { describe, expect, it } from 'vitest';
import {
  TEXTURE_COMPRESSION_FEATURES, textureStorageLayout,
  type PreparedTextureAsset, type PreparedTextureFormat,
} from '../src/texture-storage';
import type { TextureAsset } from '../src/texture-assets';

function prepared(width: number, height: number, format: PreparedTextureFormat = 'rgba8', baseMipLevel = 0): PreparedTextureAsset {
  const levels: { width: number; height: number; data: Uint8Array }[] = [];
  const blocks = format === 'rgba8' ? 1 : 4;
  const bytes = format === 'rgba8' ? 4 : 16;
  for (let mip = baseMipLevel; ; mip++) {
    const w = Math.max(1, Math.floor(width / 2 ** mip));
    const h = Math.max(1, Math.floor(height / 2 ** mip));
    levels.push({ width: w, height: h, data: new Uint8Array(Math.ceil(w / blocks) * Math.ceil(h / blocks) * bytes) });
    if (w === 1 && h === 1) break;
  }
  return { kind: 'prepared', width, height, format, colorSpace: 'linear', baseMipLevel, levels };
}

describe('texture storage layout', () => {
  it('describes a non-power-of-two RGBA8 chain without touching the source bytes', () => {
    const asset = prepared(7, 3);
    asset.levels[0].data.fill(37);
    const original = asset.levels.map(level => level.data);
    const layout = textureStorageLayout(asset);
    expect(layout).toEqual({
      gpuFormat: 'rgba8unorm', requiredFeature: undefined,
      sourceWidth: 7, sourceHeight: 3, width: 7, height: 3, baseMipLevel: 0, byteLength: 100,
      levels: [
        { width: 7, height: 3, copyWidth: 7, copyHeight: 3, bytesPerRow: 28, rowsPerImage: 3, byteLength: 84 },
        { width: 3, height: 1, copyWidth: 3, copyHeight: 1, bytesPerRow: 12, rowsPerImage: 1, byteLength: 12 },
        { width: 1, height: 1, copyWidth: 1, copyHeight: 1, bytesPerRow: 4, rowsPerImage: 1, byteLength: 4 },
      ],
    });
    expect(asset.levels.map(level => level.data)).toEqual(original);
    expect(asset.levels[0].data.every(value => value === 37)).toBe(true);
    expect('data' in layout.levels[0]).toBe(false);
  });

  it.each([
    ['bc7', 'texture-compression-bc', 'bc7-rgba-unorm-srgb'],
    ['etc2-rgba8', 'texture-compression-etc2', 'etc2-rgba8unorm-srgb'],
    ['astc-4x4', 'texture-compression-astc', 'astc-4x4-unorm-srgb'],
  ] as const)('accounts for all padded %s tail mips', (format, feature, gpuFormat) => {
    const asset = { ...prepared(8, 8, format), colorSpace: 'srgb' as const };
    const layout = textureStorageLayout(asset);
    expect(layout.gpuFormat).toBe(gpuFormat);
    expect(layout.requiredFeature).toBe(feature);
    expect(layout.byteLength).toBe(112);
    expect(layout.levels.map(level => [level.width, level.height, level.copyWidth,
      level.copyHeight, level.bytesPerRow, level.rowsPerImage, level.byteLength])).toEqual([
      [8, 8, 8, 8, 32, 2, 64],
      [4, 4, 4, 4, 16, 1, 16],
      [2, 2, 4, 4, 16, 1, 16],
      [1, 1, 4, 4, 16, 1, 16],
    ]);
  });

  it('reports only resident mips after a compressed base-mip drop', () => {
    const layout = textureStorageLayout(prepared(16, 8, 'bc7', 1));
    expect(layout).toMatchObject({ sourceWidth: 16, sourceHeight: 8,
      width: 8, height: 4, baseMipLevel: 1, byteLength: 80 });
    expect(layout.levels.map(level => level.byteLength)).toEqual([32, 16, 16, 16]);
    expect(TEXTURE_COMPRESSION_FEATURES).toEqual([
      'texture-compression-bc', 'texture-compression-etc2', 'texture-compression-astc',
    ]);
  });

  it('keeps legacy assets on the complete RGBA8 mip layout', () => {
    const pixels = new Uint8Array(7 * 3 * 4).fill(91);
    const asset: TextureAsset = { width: 7, height: 3, pixels, colorSpace: 'srgb' };
    const layout = textureStorageLayout(asset);
    expect(layout).toMatchObject({ gpuFormat: 'rgba8unorm-srgb', sourceWidth: 7, sourceHeight: 3,
      width: 7, height: 3, baseMipLevel: 0, byteLength: 100 });
    expect(layout.levels.map(level => level.byteLength)).toEqual([84, 12, 4]);
    expect(asset.pixels).toBe(pixels);
    expect(pixels.every(value => value === 91)).toBe(true);
  });

  it('rejects missing, extra, incorrectly sized, or incorrectly encoded prepared levels', () => {
    const source = prepared(8, 8, 'bc7');
    expect(() => textureStorageLayout({ ...source, levels: source.levels.slice(1) })).toThrow(/complete resident mip chain/);
    expect(() => textureStorageLayout({ ...source, levels: [...source.levels, source.levels.at(-1)!] })).toThrow(/complete resident mip chain/);
    expect(() => textureStorageLayout({ ...source, levels: source.levels.map((level, i) => i === 1
      ? { ...level, width: 3 } : level) })).toThrow(/dimensions for mip 1/);
    expect(() => textureStorageLayout({ ...source, levels: source.levels.map((level, i) => i === 2
      ? { ...level, data: new Uint8Array(15) } : level) })).toThrow(/mip 2 requires 16 bytes/);
    expect(() => textureStorageLayout({ ...source, levels: source.levels.map((level, i) => i === 2
      ? { ...level, data: null as unknown as Uint8Array } : level) })).toThrow(/mip 2 requires 16 bytes/);
  });

  it('rejects malformed dimensions, base mip, format, color space, and kind', () => {
    const source = prepared(8, 8, 'bc7');
    expect(() => textureStorageLayout({ ...source, width: 8193 })).toThrow(/dimensions/);
    expect(() => textureStorageLayout({ ...source, height: NaN })).toThrow(/dimensions/);
    expect(() => textureStorageLayout({ ...source, baseMipLevel: 4 })).toThrow(/baseMipLevel/);
    expect(() => textureStorageLayout({ ...source, baseMipLevel: 1.5 })).toThrow(/baseMipLevel/);
    expect(() => textureStorageLayout(prepared(8, 8, 'bc7', 2))).toThrow(/divisible by four/);
    expect(() => textureStorageLayout({ ...source, format: 'unknown' as PreparedTextureFormat })).toThrow(/format/);
    expect(() => textureStorageLayout({ ...source, colorSpace: 'unknown' as 'srgb' })).toThrow(/colorSpace/);
    expect(() => textureStorageLayout({ ...source, kind: 'future' as 'prepared' })).toThrow(/kind/);
    expect(() => textureStorageLayout({ width: 1, height: 1, pixels: new Uint8Array(4),
      colorSpace: 'linear', kind: undefined } as unknown as TextureAsset)).toThrow(/kind/);
  });

  it('applies legacy RGBA8 validation and accepts a one-pixel chain', () => {
    expect(() => textureStorageLayout({ width: 2, height: 1, pixels: new Uint8Array(4),
      colorSpace: 'linear' })).toThrow(/RGBA8 bytes/);
    expect(textureStorageLayout(prepared(1, 1)).levels).toEqual([
      { width: 1, height: 1, copyWidth: 1, copyHeight: 1, bytesPerRow: 4, rowsPerImage: 1, byteLength: 4 },
    ]);
  });
});
