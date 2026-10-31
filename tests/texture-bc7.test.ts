import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import {
  BC7_TABLES, bc7ModeHistogram, decodeBc7Block, decodeBc7Image, encodeBc7Block, encodeBc7Image, encodeBc7Rows,
} from '../src/texture-bc7';
import { decodePng } from '../src/texture-image-decode';

const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state; };
}

/** Seeded random 128-bit blocks, mode = index % 8 (modes 0..7 equally often). */
function randomBlocks(count: number, seed = 12345): Uint8Array {
  const random = lcg(seed), blocks = new Uint8Array(count * 16);
  for (let b = 0; b < count; b++) {
    const mode = b % 8;
    for (let i = 0; i < 16; i++) blocks[b * 16 + i] = random() & 255;
    blocks[b * 16] = ((blocks[b * 16] & ~((1 << (mode + 1)) - 1)) | (1 << mode)) & 255;
  }
  return blocks;
}

/** DX10 DDS header for a single-level BC7 UNORM texture. */
function bc7Dds(width: number, height: number, blocks: Uint8Array): Buffer {
  const header = Buffer.alloc(148);
  header.write('DDS ', 0, 'ascii');
  header.writeUInt32LE(124, 4);
  header.writeUInt32LE(0x81007, 8);
  header.writeUInt32LE(height, 12);
  header.writeUInt32LE(width, 16);
  header.writeUInt32LE(blocks.length, 20);
  header.writeUInt32LE(1, 28);
  header.writeUInt32LE(32, 76);
  header.writeUInt32LE(0x4, 80);
  header.write('DX10', 84, 'ascii');
  header.writeUInt32LE(0x1000, 108);
  header.writeUInt32LE(98, 128);
  header.writeUInt32LE(3, 132);
  header.writeUInt32LE(1, 140);
  return Buffer.concat([header, blocks]);
}

/** Varied 4x4 test blocks: solid, gradients, noise, edges, data-like channels, with and without alpha. */
function corpus(count: number, seed = 7): Uint8Array[] {
  const random = lcg(seed), blocks: Uint8Array[] = [];
  for (let n = 0; n < count; n++) {
    const kind = n % 8, pixels = new Uint8Array(64), base = [random() & 255, random() & 255, random() & 255, random() & 255];
    const alpha = n % 3 === 0;
    for (let i = 0; i < 16; i++) {
      const x = i & 3, y = i >> 2, r = random();
      for (let c = 0; c < 4; c++) {
        let value: number;
        if (kind === 0) value = base[c];
        else if (kind === 1) value = base[c] + (x * 13 + y * 7) * (c + 1);
        else if (kind === 2) value = base[c] + ((r >> (c * 6)) & 63) - 32;
        else if (kind === 3) value = (x + y > 3 ? base[c] : 255 - base[c]);
        else if (kind === 4) value = c === 0 ? base[0] : c === 1 ? x * 60 + (r & 7) : y * 70 + ((r >> 3) & 15);
        else if (kind === 5) value = [base[c], 255 - base[c], (base[c] * 3) & 255][(x + 2 * y) % 3];
        else if (kind === 6) value = (r >> (c * 8)) & 255;
        else value = base[c] + (x === y ? 90 : 0);
        pixels[i * 4 + c] = Math.max(0, Math.min(255, value));
      }
      pixels[i * 4 + 3] = alpha ? (kind === 0 ? base[3] : Math.max(0, Math.min(255, base[3] + x * 20 - y * 15))) : 255;
    }
    blocks.push(pixels);
  }
  return blocks;
}

function errorOf(source: Uint8Array, decoded: Uint8Array): { sse: number; alphaSse: number; max: number } {
  let sse = 0, alphaSse = 0, max = 0;
  for (let i = 0; i < 64; i++) {
    const d = decoded[i] - source[i];
    if ((i & 3) === 3) alphaSse += d * d; else sse += d * d;
    max = Math.max(max, Math.abs(d));
  }
  return { sse, alphaSse, max };
}

describe('BC7 tables', () => {
  it('assigns every subset at least one pixel, pixel 0 to subset 0 and each anchor to its own subset', () => {
    for (const ns of [2, 3]) {
      for (let p = 0; p < 64; p++) {
        const subsets = BC7_TABLES.subsets[ns].subarray(p * 16, p * 16 + 16);
        expect(subsets[0]).toBe(0);
        for (let s = 0; s < ns; s++) expect(subsets.includes(s)).toBe(true);
        for (let s = 1; s < ns; s++) expect(subsets[BC7_TABLES.anchors[ns][p * 3 + s]]).toBe(s);
      }
    }
    expect(BC7_TABLES.weights[2]).toEqual([0, 21, 43, 64]);
    expect(BC7_TABLES.weights[3]).toEqual([0, 9, 18, 27, 37, 46, 55, 64]);
    expect(BC7_TABLES.weights[4]).toEqual([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]);
  });
});

describe('BC7 decoder', () => {
  it('reproduces the ImageIO-verified decode of 16,384 seeded random blocks covering modes 0-7', () => {
    // The decoded hash was cross-checked bit for bit against macOS 26.6.2 ImageIO
    // (sips DDS -> PNG) on 2026-10-30; the darwin-only test below repeats that check live.
    const blocks = randomBlocks(128 * 128);
    expect(bc7ModeHistogram(blocks)).toEqual([2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 0]);
    expect(sha256(blocks)).toBe('b05d6aeccfe6fba6a515b90492b563f5d92d54553325613dd4b22dcb9fa7df4c');
    expect(sha256(decodeBc7Image(blocks, 512, 512))).toBe('42dec328f142b87e6884cc1cfcd34b446b7cccc9402288ce0539f8ffc1372327');
  });

  it('decodes the reserved mode-8 encoding to transparent black', () => {
    const out = new Uint8Array(64).fill(7);
    decodeBc7Block(new Uint8Array(16), 0, out);
    expect([...out]).toEqual(new Array(64).fill(0));
  });

  it('crops partial edge blocks and rejects mismatched block lengths', () => {
    const blocks = randomBlocks(4, 99);
    const full = decodeBc7Image(blocks, 8, 8), cropped = decodeBc7Image(blocks, 6, 5);
    for (let y = 0; y < 5; y++) for (let x = 0; x < 6; x++) {
      expect([...cropped.subarray((y * 6 + x) * 4, (y * 6 + x) * 4 + 4)]).toEqual([...full.subarray((y * 8 + x) * 4, (y * 8 + x) * 4 + 4)]);
    }
    expect(() => decodeBc7Image(blocks, 12, 8)).toThrow(RangeError);
  });

  it.skipIf(process.platform !== 'darwin' || !existsSync('/usr/bin/sips'))('matches macOS ImageIO bit for bit (independent decoder)', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'real-engine-bc7-'));
    try {
      const blocks = randomBlocks(64 * 64, 2026);
      await writeFile(join(directory, 'random.dds'), bc7Dds(256, 256, blocks));
      execFileSync('/usr/bin/sips', ['-s', 'format', 'png', join(directory, 'random.dds'), '--out', join(directory, 'random.png')], { stdio: 'pipe' });
      const reference = decodePng(new Uint8Array(await readFile(join(directory, 'random.png'))), data => inflateSync(data));
      expect(reference.width).toBe(256);
      expect(Buffer.from(decodeBc7Image(blocks, 256, 256)).equals(Buffer.from(reference.pixels))).toBe(true);
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
});

describe('BC7 encoder', () => {
  const blocks = corpus(512);

  for (const quality of [0, 1, 2]) {
    it(`returns the exact decoded error, keeps opaque alpha and is deterministic at quality ${quality}`, () => {
      const out = new Uint8Array(16), again = new Uint8Array(16), decoded = new Uint8Array(64);
      for (const pixels of blocks) {
        const opaque = pixels.every((value, i) => (i & 3) !== 3 || value === 255);
        const reported = encodeBc7Block(pixels, out, 0, { quality });
        encodeBc7Block(pixels, again, 0, { quality });
        expect([...again]).toEqual([...out]);
        decodeBc7Block(out, 0, decoded);
        const error = errorOf(pixels, decoded);
        if (opaque) expect(error.alphaSse).toBe(0);
        expect(reported).toBe(error.sse + (opaque ? 1e6 : 1) * error.alphaSse);
      }
    });
  }

  it('reaches every mode across opaque and translucent content at quality 2 and never gets worse with quality', () => {
    const out = new Uint8Array(16), histogram = new Array(9).fill(0);
    let previous = blocks.map(() => Infinity);
    for (const quality of [0, 1, 2]) {
      const errors = blocks.map(pixels => encodeBc7Block(pixels, out, 0, { quality }));
      errors.forEach((error, i) => expect(error).toBeLessThanOrEqual(previous[i]));
      previous = errors;
    }
    for (const pixels of blocks) { encodeBc7Block(pixels, out, 0, { quality: 2 }); histogram[bc7ModeHistogram(out).indexOf(1)]++; }
    for (let mode = 0; mode < 8; mode++) expect(histogram[mode], `mode ${mode}`).toBeGreaterThan(0);
    expect(histogram[8]).toBe(0);
  });

  it('encodes solid colors within one code value', () => {
    const random = lcg(31), out = new Uint8Array(16), decoded = new Uint8Array(64);
    for (let n = 0; n < 256; n++) {
      const color = [random() & 255, random() & 255, random() & 255, n % 2 ? 255 : random() & 255];
      const pixels = new Uint8Array(64);
      for (let i = 0; i < 64; i++) pixels[i] = color[i & 3];
      encodeBc7Block(pixels, out, 0, { quality: 1 });
      decodeBc7Block(out, 0, decoded);
      expect(errorOf(pixels, decoded).max).toBeLessThanOrEqual(1);
    }
  });

  it('stops early only when the error is within the requested bound', () => {
    const out = new Uint8Array(16), full = new Uint8Array(16);
    for (const pixels of blocks.slice(0, 128)) {
      const bounded = encodeBc7Block(pixels, out, 0, { quality: 2, earlyOutError: 96 });
      const exhaustive = encodeBc7Block(pixels, full, 0, { quality: 2 });
      expect(bounded === exhaustive || bounded <= 96).toBe(true);
    }
  });

  it('produces identical bytes for any block-row split, with edge replication for partial blocks', () => {
    const random = lcg(5), width = 22, height = 18, pixels = new Uint8Array(width * height * 4);
    for (let i = 0; i < pixels.length; i++) pixels[i] = (i & 3) === 3 ? 255 : (random() >> 8) & 255;
    const whole = encodeBc7Image(pixels, width, height, { quality: 1 });
    const rows = Math.ceil(height / 4), parts = [encodeBc7Rows(pixels, width, height, 0, 2, { quality: 1 }),
      encodeBc7Rows(pixels, width, height, 2, 3, { quality: 1 }), encodeBc7Rows(pixels, width, height, 3, rows, { quality: 1 })];
    expect(Buffer.concat(parts).equals(Buffer.from(whole))).toBe(true);
    const decoded = decodeBc7Image(whole, width, height);
    for (let i = 3; i < decoded.length; i += 4) expect(decoded[i]).toBe(255);
    expect(() => encodeBc7Image(pixels.subarray(4), width, height)).toThrow(RangeError);
  });
});
