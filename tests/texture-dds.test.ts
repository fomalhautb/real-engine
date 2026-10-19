import { describe, expect, it } from 'vitest';
import { parseTextureDds } from '../src/texture-dds';
import { textureStorageLayout } from '../src/texture-storage';

// Build headers byte by byte from the DDS structure offsets, independently of the parser.
function dds(width: number, height: number, dxgiFormat: number, options: {
  mipCount?: number; flags?: number; pitch?: number; alphaMode?: number; payload?: Uint8Array;
} = {}): Uint8Array {
  const compressed = dxgiFormat === 98 || dxgiFormat === 99;
  const count = options.mipCount ?? (1 + Math.floor(Math.log2(Math.max(width, height))));
  const levels: number[] = [];
  let w = width;
  let h = height;
  for (let i = 0; i < count; i++) {
    levels.push(compressed ? Math.ceil(w / 4) * Math.ceil(h / 4) * 16 : w * h * 4);
    w = Math.max(1, Math.floor(w / 2));
    h = Math.max(1, Math.floor(h / 2));
  }
  const payload = options.payload ?? Uint8Array.from({ length: levels.reduce((a, b) => a + b, 0) }, (_, i) => i % 251);
  const bytes = new Uint8Array(148 + payload.length);
  const view = new DataView(bytes.buffer);
  const put = (offset: number, value: number) => view.setUint32(offset, value, true);
  put(0, 0x20534444); // 'DDS '
  put(4, 124);
  put(8, options.flags ?? (0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | (compressed ? 0x80000 : 0x8)));
  put(12, height);
  put(16, width);
  put(20, options.pitch ?? (compressed ? levels[0] ?? 0 : width * 4));
  put(28, count);
  put(76, 32);
  put(80, 0x4);
  put(84, 0x30315844); // 'DX10'
  put(108, 0x1000 | (count > 1 ? 0x8 | 0x400000 : 0));
  put(128, dxgiFormat);
  put(132, 3); // D3D10_RESOURCE_DIMENSION_TEXTURE2D
  put(140, 1);
  put(144, options.alphaMode ?? 0);
  bytes.set(payload, 148);
  return bytes;
}

function changed(bytes: Uint8Array, offset: number, value: number): Uint8Array {
  const copy = bytes.slice();
  new DataView(copy.buffer).setUint32(offset, value, true);
  return copy;
}

describe('DX10 DDS texture parser', () => {
  it.each([
    [28, 'rgba8', 'linear'], [29, 'rgba8', 'srgb'],
    [98, 'bc7', 'linear'], [99, 'bc7', 'srgb'],
  ] as const)('maps DXGI format %i to %s / %s and copies every mip', (dxgi, format, colorSpace) => {
    const source = dds(8, 8, dxgi, { alphaMode: 4 });
    const asset = parseTextureDds(source);
    expect(asset).toMatchObject({ kind: 'prepared', width: 8, height: 8,
      baseMipLevel: 0, format, colorSpace });
    expect(asset.levels.map(level => [level.width, level.height, level.data.length])).toEqual(
      format === 'bc7'
        ? [[8, 8, 64], [4, 4, 16], [2, 2, 16], [1, 1, 16]]
        : [[8, 8, 256], [4, 4, 64], [2, 2, 16], [1, 1, 4]],
    );
    expect(Uint8Array.from(asset.levels.flatMap(level => [...level.data]))).toEqual(source.subarray(148));
    expect(textureStorageLayout(asset).byteLength).toBe(source.length - 148);
    for (const level of asset.levels) expect(level.data.buffer).not.toBe(source.buffer);
    for (let i = 1; i < asset.levels.length; i++) {
      expect(asset.levels[i].data.buffer).not.toBe(asset.levels[i - 1].data.buffer);
    }
    source.fill(0, 148);
    expect(asset.levels[0].data[1]).toBe(1);
  });

  it('accepts a non-power-of-two RGBA chain and a sub-view input', () => {
    const source = dds(7, 3, 29);
    const padded = new Uint8Array(source.length + 19).fill(222);
    padded.set(source, 7);
    const asset = parseTextureDds(padded.subarray(7, 7 + source.length));
    expect(asset.levels.map(level => [level.width, level.height, level.data.length])).toEqual([
      [7, 3, 84], [3, 1, 12], [1, 1, 4],
    ]);
    expect(asset.levels[1].data[0]).toBe(source[148 + 84]);
  });

  it('uses whole BC blocks for tail mips', () => {
    const asset = parseTextureDds(dds(12, 8, 98));
    expect(asset.levels.map(level => [level.width, level.height, level.data.length])).toEqual([
      [12, 8, 96], [6, 4, 32], [3, 2, 16], [1, 1, 16],
    ]);
  });

  it('rejects BC7 bases that the texture storage contract cannot upload', () => {
    expect(() => parseTextureDds(dds(9, 8, 98))).toThrow(/divisible by four/);
    expect(() => parseTextureDds(dds(8, 5, 99))).toThrow(/divisible by four/);
  });

  it('accepts a one-pixel texture and the supported alpha modes', () => {
    for (const alphaMode of [0, 1, 3, 4]) {
      expect(parseTextureDds(dds(1, 1, 28, { alphaMode })).levels[0].data.length).toBe(4);
    }
  });

  it('rejects malformed headers and unsupported resource kinds', () => {
    const valid = dds(8, 8, 98);
    for (const [offset, value] of [
      [0, 0], [4, 123], [76, 31], [80, 0], [84, 0],
      [112, 0x200], [112, 0x400], [112, 0x200000],
      [132, 2], [132, 4], [136, 4], [140, 0], [140, 2],
      [24, 2], [144, 2], [144, 5],
    ] as const) {
      expect(() => parseTextureDds(changed(valid, offset, value)), `offset ${offset}`).toThrow();
    }
    expect(() => parseTextureDds(changed(valid, 8, 0x20000 | 0x80000 | 0x800000))).toThrow();
    expect(() => parseTextureDds(changed(valid, 128, 71))).toThrow();
  });

  it('rejects invalid dimensions, empty or short chains, and mismatched layouts', () => {
    const valid = dds(8, 8, 98);
    for (const [offset, value] of [[12, 0], [16, 0], [12, 8193], [16, 8193],
      [28, 0], [28, 3], [28, 5], [20, 0], [20, 63], [20, 65]] as const) {
      expect(() => parseTextureDds(changed(valid, offset, value)), `offset ${offset}`).toThrow();
    }
    expect(() => parseTextureDds(changed(dds(7, 3, 28), 20, 32))).toThrow();
    expect(() => parseTextureDds(changed(valid, 8, 0x80000 | 0x8))).toThrow();
    expect(() => parseTextureDds(changed(valid, 8, 0x8))).toThrow();
    expect(() => parseTextureDds(changed(dds(7, 3, 28), 8, 0x80000))).toThrow();
    expect(() => parseTextureDds(dds(8, 8, 98, { flags: 0, pitch: 64 }))).toThrow();
  });

  it('rejects truncated headers, truncated mips, and trailing bytes', () => {
    const valid = dds(8, 8, 98);
    for (const count of [0, 4, 127, 128, 147, valid.length - 1]) {
      expect(() => parseTextureDds(valid.subarray(0, count))).toThrow();
    }
    const extra = new Uint8Array(valid.length + 1);
    extra.set(valid);
    expect(() => parseTextureDds(extra)).toThrow();
    expect(() => parseTextureDds(dds(8, 8, 98, { payload: new Uint8Array(0) }))).toThrow();
  });
});
