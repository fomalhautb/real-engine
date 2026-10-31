import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { convertNormalMap, decodeBc1Color, decodeBcnBlock, decodeLegacyDds } from '../src/texture-bcn';
import { decodePng } from '../src/texture-image-decode';

const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

function randomBytes(count: number, seed: number): Uint8Array {
  const bytes = new Uint8Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i++) { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; bytes[i] = state >>> 24; }
  return bytes;
}

/** Legacy FourCC DDS (or a DX10 header with `dxgi`) around raw block data. */
function dds(width: number, height: number, fourCC: string, data: Uint8Array, options: { dxgi?: number; mips?: number; caps2?: number } = {}): Uint8Array {
  const headerBytes = options.dxgi === undefined ? 128 : 148;
  const bytes = new Uint8Array(headerBytes + data.length);
  const view = new DataView(bytes.buffer);
  bytes.set([0x44, 0x44, 0x53, 0x20]);
  view.setUint32(4, 124, true); view.setUint32(8, 0x81007, true); view.setUint32(12, height, true); view.setUint32(16, width, true);
  view.setUint32(20, data.length, true); view.setUint32(28, options.mips ?? 1, true); view.setUint32(76, 32, true); view.setUint32(80, 4, true);
  const code = options.dxgi === undefined ? fourCC : 'DX10';
  for (let i = 0; i < 4; i++) bytes[84 + i] = code.charCodeAt(i);
  view.setUint32(108, 0x1000, true); view.setUint32(112, options.caps2 ?? 0, true);
  if (options.dxgi !== undefined) { view.setUint32(128, options.dxgi, true); view.setUint32(132, 3, true); view.setUint32(140, 1, true); }
  bytes.set(data, headerBytes);
  return bytes;
}

/** Uncompressed 32-bit DDS with the given channel masks. */
function rawDds(width: number, height: number, masks: [number, number, number, number], alphaFlag: boolean, data: Uint8Array): Uint8Array {
  const bytes = new Uint8Array(128 + data.length), view = new DataView(bytes.buffer);
  bytes.set([0x44, 0x44, 0x53, 0x20]);
  view.setUint32(4, 124, true); view.setUint32(12, height, true); view.setUint32(16, width, true); view.setUint32(76, 32, true);
  view.setUint32(80, 0x40 | (alphaFlag ? 1 : 0), true); view.setUint32(88, 32, true);
  masks.forEach((mask, i) => view.setUint32(92 + i * 4, mask >>> 0, true));
  bytes.set(data, 128);
  return bytes;
}

function bc1Block(c0: number, c1: number, indices: number[]): Uint8Array {
  let bits = 0;
  indices.forEach((index, i) => { bits |= index << (2 * i); });
  return new Uint8Array([c0 & 255, c0 >> 8, c1 & 255, c1 >> 8, bits & 255, (bits >>> 8) & 255, (bits >>> 16) & 255, bits >>> 24]);
}

function bc4Block(a0: number, a1: number, indices: number[]): Uint8Array {
  const block = new Uint8Array(8);
  block[0] = a0; block[1] = a1;
  let bits = 0n;
  indices.forEach((index, i) => { bits |= BigInt(index) << BigInt(3 * i); });
  for (let k = 0; k < 6; k++) block[2 + k] = Number((bits >> BigInt(8 * k)) & 255n);
  return block;
}

const pixel = (out: Uint8Array, i: number) => Array.from(out.subarray(i * 4, i * 4 + 4));
const INDICES = Array.from({ length: 16 }, (_, i) => i & 3);
const ALPHA_INDICES = Array.from({ length: 16 }, (_, i) => i & 7);

describe('legacy BCn decoding', () => {
  it('expands 5:6:5 endpoints to the nearest 8-bit value, not by bit replication', () => {
    const out = new Uint8Array(64);
    let replicationDiffers = 0;
    for (let v = 1; v < 32; v++) {
      decodeBc1Color(bc1Block((v << 11) | v, 0, new Array(16).fill(0)), 0, out, 0);
      const expected = Math.round(v * 255 / 31);
      expect(pixel(out, 0)).toEqual([expected, 0, expected, 255]);
      if (expected !== ((v << 3) | (v >> 2))) replicationDiffers++;
    }
    for (let v = 1; v < 64; v++) {
      decodeBc1Color(bc1Block(v << 5, 0, new Array(16).fill(0)), 0, out, 0);
      const expected = Math.round(v * 255 / 63);
      expect(pixel(out, 0)).toEqual([0, expected, 0, 255]);
      if (expected !== ((v << 2) | (v >> 4))) replicationDiffers++;
    }
    expect(replicationDiffers).toBeGreaterThan(0);
  });

  it('builds the ideal rounded BC1 palette in four-color and punch-through modes', () => {
    const out = new Uint8Array(64);
    // Red to blue: 2/3 and 1/3 of 255 are 170 and 85 exactly after rounding.
    decodeBc1Color(bc1Block(0xf800, 0x001f, INDICES), 0, out, 0);
    expect([0, 1, 2, 3].map(i => pixel(out, i))).toEqual([[255, 0, 0, 255], [0, 0, 255, 255], [170, 0, 85, 255], [85, 0, 170, 255]]);
    // c0 <= c1: half point (127.5 ties round down) and transparent black.
    decodeBc1Color(bc1Block(0x001f, 0xf800, INDICES), 0, out, 0);
    expect([0, 1, 2, 3].map(i => pixel(out, i))).toEqual([[0, 0, 255, 255], [255, 0, 0, 255], [127, 0, 127, 255], [0, 0, 0, 0]]);
    // BC2/BC3 color blocks never use punch-through.
    decodeBc1Color(bc1Block(0x001f, 0xf800, INDICES), 0, out, 0, true);
    expect([2, 3].map(i => pixel(out, i))).toEqual([[85, 0, 170, 255], [170, 0, 85, 255]]);
    // Green 6-bit: 63 and 0 -> 255, 0, 170, 85.
    decodeBc1Color(bc1Block(0x07e0, 0x0000, INDICES), 0, out, 0);
    expect([0, 1, 2, 3].map(i => pixel(out, i)[1])).toEqual([255, 0, 170, 85]);
  });

  it('decodes BC2 explicit alpha, BC3 alpha in both modes, BC4 and BC5', () => {
    const out = new Uint8Array(64);
    const alpha = new Uint8Array(8);
    for (let i = 0; i < 8; i++) alpha[i] = (2 * i) | ((2 * i + 1) << 4);
    decodeBcnBlock('bc2', new Uint8Array([...alpha, ...bc1Block(0xffff, 0, new Array(16).fill(0))]), 0, out);
    expect(Array.from({ length: 16 }, (_, i) => out[i * 4 + 3])).toEqual(Array.from({ length: 16 }, (_, i) => i * 17));
    decodeBcnBlock('bc3', new Uint8Array([...bc4Block(255, 0, ALPHA_INDICES), ...bc1Block(0xffff, 0, new Array(16).fill(0))]), 0, out);
    expect(Array.from({ length: 8 }, (_, i) => out[i * 4 + 3])).toEqual([255, 0, 219, 182, 146, 109, 73, 36]);
    decodeBcnBlock('bc3', new Uint8Array([...bc4Block(0, 255, ALPHA_INDICES), ...bc1Block(0xffff, 0, new Array(16).fill(0))]), 0, out);
    expect(Array.from({ length: 8 }, (_, i) => out[i * 4 + 3])).toEqual([0, 255, 51, 102, 153, 204, 0, 255]);
    decodeBcnBlock('bc4', bc4Block(200, 100, ALPHA_INDICES), 0, out);
    expect([0, 1, 2].map(i => pixel(out, i))).toEqual([[200, 200, 200, 255], [100, 100, 100, 255], [186, 186, 186, 255]]);
    decodeBcnBlock('bc5', new Uint8Array([...bc4Block(10, 250, ALPHA_INDICES), ...bc4Block(250, 10, ALPHA_INDICES)]), 0, out);
    expect([0, 1, 7].map(i => pixel(out, i))).toEqual([[10, 250, 0, 255], [250, 10, 0, 255], [255, 44, 0, 255]]);
  });

  it('parses legacy headers, crops partial blocks and rejects what it does not decode', () => {
    const blocks = new Uint8Array(4 * 8);
    for (let b = 0; b < 4; b++) blocks.set(bc1Block(0xf800, 0x001f, new Array(16).fill(b & 1)), b * 8);
    const image = decodeLegacyDds(dds(6, 5, 'DXT1', blocks, { mips: 3 }));
    expect([image.width, image.height, image.format, image.fourCC, image.mipCount, image.channels]).toEqual([6, 5, 'bc1', 'DXT1', 3, 4]);
    expect(image.pixels.length).toBe(6 * 5 * 4);
    expect(pixel(image.pixels, 0)).toEqual([255, 0, 0, 255]);
    expect(pixel(image.pixels, 4)).toEqual([0, 0, 255, 255]);
    expect(pixel(image.pixels, 6 * 4)).toEqual([255, 0, 0, 255]);
    expect(decodeLegacyDds(dds(4, 4, 'ATI2', new Uint8Array(16))).channels).toBe(2);
    expect(decodeLegacyDds(dds(4, 4, 'ATI1', new Uint8Array(8))).format).toBe('bc4');
    const bgra = rawDds(2, 1, [0xff0000, 0xff00, 0xff, 0xff000000], true, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
    expect(Array.from(decodeLegacyDds(bgra).pixels)).toEqual([3, 2, 1, 4, 7, 6, 5, 8]);
    const rgbx = rawDds(1, 1, [0xff, 0xff00, 0xff0000, 0], false, new Uint8Array([9, 8, 7, 6]));
    expect(Array.from(decodeLegacyDds(rgbx).pixels)).toEqual([9, 8, 7, 255]);
    expect(() => decodeLegacyDds(new Uint8Array(200))).toThrow('DDS_MAGIC');
    expect(() => decodeLegacyDds(dds(4, 4, 'DXT1', new Uint8Array(16), { dxgi: 71 }))).toThrow('DDS_DX10_USE_PARSE_TEXTURE_DDS');
    expect(() => decodeLegacyDds(dds(4, 4, 'ETC2', new Uint8Array(8)))).toThrow('DDS_FOURCC_UNSUPPORTED:ETC2');
    expect(() => decodeLegacyDds(dds(8, 8, 'DXT5', new Uint8Array(48)))).toThrow('DDS_TRUNCATED');
    expect(() => decodeLegacyDds(dds(4, 4, 'DXT1', new Uint8Array(8), { caps2: 0x200 }))).toThrow('DDS_CUBE_OR_VOLUME_UNSUPPORTED');
    expect(() => decodeLegacyDds(dds(16, 16, 'DXT1', new Uint8Array(128)), 8)).toThrow('DDS_DIMENSIONS');
    expect(() => decodeLegacyDds(rawDds(1, 1, [0xf800, 0x7e0, 0x1f, 0], false, new Uint8Array(4)))).toThrow('DDS_RGB_MASKS_UNSUPPORTED');
  });

  it('decodes seeded random blocks to pinned bytes', () => {
    // Pinned on macOS, where the color formats match ImageIO bit for bit (see
    // the live check below); other platforms verify the same bytes.
    const hashes = Object.fromEntries(([['DXT1', 8], ['DXT3', 16], ['DXT5', 16], ['ATI1', 8], ['ATI2', 16]] as const)
      .map(([fourCC, blockBytes], seed) => [fourCC, sha256(decodeLegacyDds(dds(64, 64, fourCC, randomBytes(256 * blockBytes, 77 + seed))).pixels)]));
    expect(hashes).toEqual({
      DXT1: 'c7268eca328ad4c3a9ca66032eaa29533bb0781abbfd3704669ac244171e43d9',
      DXT3: 'b11b51c96d244c4cb68afe2c82d947cc9b979dfdd0bb4a1b23cb2dc9a258311a',
      DXT5: '57dfc6e5b3c511428aed6c06934af804745a89cd34e3f6cf2128ce02b674b5d2',
      ATI1: '359f54ff04ee09feaa183cabade3f9478150ade27e1d2f6e353c991d1a6eca39',
      ATI2: 'b34b23b8622238f414dff472d892efb86f693b4a3b18cdc54f320d85d9a1107e',
    });
  });

  it.skipIf(process.platform !== 'darwin' || !existsSync('/usr/bin/sips'))('matches macOS ImageIO: colors exactly, alpha and BC4/BC5 within its truncation', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'real-engine-bcn-'));
    try {
      const cases = [['DXT1', 8, undefined], ['DXT3', 16, undefined], ['DXT5', 16, undefined], ['ATI1', 8, 80], ['ATI2', 16, 83]] as const;
      for (const [index, [fourCC, blockBytes, dxgi]] of cases.entries()) {
        const data = randomBytes(256 * blockBytes, 900 + index);
        // ImageIO reads BC4/BC5 only behind a DX10 header; the blocks are identical.
        await writeFile(join(directory, `${fourCC}.dds`), dds(64, 64, fourCC, data, { dxgi }));
        execFileSync('/usr/bin/sips', ['-s', 'format', 'png', join(directory, `${fourCC}.dds`), '--out', join(directory, `${fourCC}.png`)], { stdio: 'pipe' });
        const reference = decodePng(new Uint8Array(await readFile(join(directory, `${fourCC}.png`))), bytes => inflateSync(bytes));
        const mine = decodeLegacyDds(dds(64, 64, fourCC, data));
        const expand = reference.source.channels === 1;
        let above = 0;
        for (let p = 0; p < 64 * 64; p++) for (let c = 0; c < 4; c++) {
          const theirs = expand ? (c === 3 ? 255 : reference.pixels[p * 4]) : reference.pixels[p * 4 + c];
          const delta = mine.pixels[p * 4 + c] - theirs;
          const exact = fourCC === 'DXT1' || fourCC === 'DXT3' || (fourCC === 'DXT5' && c < 3);
          if (exact) expect(delta, `${fourCC} pixel ${p} channel ${c}`).toBe(0);
          else { expect(delta === 0 || delta === 1, `${fourCC} pixel ${p} channel ${c}: ${delta}`).toBe(true); above += delta; }
        }
        if (fourCC === 'DXT5' || fourCC === 'ATI1' || fourCC === 'ATI2') expect(above, fourCC).toBeGreaterThan(0);
      }
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
});

describe('normal map conversion', () => {
  it('flips the DirectX green axis and rebuilds Z from two channels', () => {
    const pixels = new Uint8Array([128, 128, 0, 7, 255, 128, 0, 9, 128, 0, 0, 255, 200, 30, 50, 60]);
    convertNormalMap(pixels, { flipGreen: true, reconstructZ: true });
    const z = (x: number, y: number) => {
      const nx = x / 127.5 - 1, ny = y / 127.5 - 1;
      return Math.round((Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny)) * 0.5 + 0.5) * 255);
    };
    expect(Array.from(pixels)).toEqual([128, 127, 255, 7, 255, 127, z(255, 127), 9, 128, 255, z(128, 255), 255, 200, 225, z(200, 225), 60]);
    expect(z(255, 127)).toBe(128);
    const keep = new Uint8Array([10, 20, 30, 40]);
    convertNormalMap(keep, { flipGreen: true, reconstructZ: false });
    expect(Array.from(keep)).toEqual([10, 235, 30, 40]);
  });
});
