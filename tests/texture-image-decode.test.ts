import { deflateSync, inflateSync, crc32 } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { decodePng, decodeTga } from '../src/texture-image-decode';

const inflate = (data: Uint8Array): Uint8Array => inflateSync(data);

function chunk(type: string, data: Uint8Array): Buffer {
  const header = Buffer.alloc(8), body = Buffer.concat([Buffer.from(type, 'ascii'), data]), crc = Buffer.alloc(4);
  header.writeUInt32BE(data.length, 0);
  header.write(type, 4, 'ascii');
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([header, data, crc]);
}

/** Minimal independent PNG writer: raw scanlines, each with an explicit filter type. */
function png(width: number, height: number, bitDepth: number, colorType: number, rows: Uint8Array[], filters: number[],
  extra: Buffer[] = [], interlace = 0): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = bitDepth; ihdr[9] = colorType; ihdr[12] = interlace;
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType]!;
  const bpp = Math.max(1, (channels * bitDepth) >> 3);
  const raw: number[] = [];
  let previous: Uint8Array = new Uint8Array(rows[0].length);
  rows.forEach((row, y) => {
    const filter = filters[y % filters.length];
    raw.push(filter);
    for (let i = 0; i < row.length; i++) {
      const a = i >= bpp ? row[i - bpp] : 0, b = previous[i], c = i >= bpp ? previous[i - bpp] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const predictor = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][filter];
      raw.push((row[i] - predictor) & 255);
    }
    previous = row;
  });
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), ...extra,
    chunk('IDAT', deflateSync(Buffer.from(raw))), chunk('IEND', new Uint8Array(0))]);
}

describe('PNG decoder', () => {
  it('decodes 8-bit RGBA with every filter type', () => {
    const rows = Array.from({ length: 5 }, (_, y) => Uint8Array.from({ length: 3 * 4 }, (_, i) => (y * 53 + i * 29 + 7) & 255));
    const decoded = decodePng(png(3, 5, 8, 6, rows, [0, 1, 2, 3, 4]), inflate);
    expect(decoded.width).toBe(3);
    expect(decoded.height).toBe(5);
    expect([...decoded.pixels]).toEqual(rows.flatMap(row => [...row]));
    expect(decoded.source).toEqual({ format: 'png', bitDepth: 8, channels: 4, hasAlpha: true });
  });

  it('expands RGB, gray, gray-alpha, palette with tRNS and 16-bit samples to RGBA8', () => {
    const rgb = decodePng(png(2, 1, 8, 2, [Uint8Array.of(10, 20, 30, 40, 50, 60)], [4]), inflate);
    expect([...rgb.pixels]).toEqual([10, 20, 30, 255, 40, 50, 60, 255]);
    const gray = decodePng(png(3, 1, 8, 0, [Uint8Array.of(0, 128, 255)], [1]), inflate);
    expect([...gray.pixels]).toEqual([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
    const grayAlpha = decodePng(png(1, 1, 8, 4, [Uint8Array.of(77, 9)], [0]), inflate);
    expect([...grayAlpha.pixels]).toEqual([77, 77, 77, 9]);
    const palette = decodePng(png(4, 1, 2, 3, [Uint8Array.of(0b00011011)], [0],
      [chunk('PLTE', Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)), chunk('tRNS', Uint8Array.of(0, 128))]), inflate);
    expect([...palette.pixels]).toEqual([1, 2, 3, 0, 4, 5, 6, 128, 7, 8, 9, 255, 10, 11, 12, 255]);
    // 16-bit: round(v * 255 / 65535); 0x8080 -> 128, 0xffff -> 255, 0x00ff -> 1.
    const wide = decodePng(png(1, 1, 16, 6, [Uint8Array.of(0x80, 0x80, 0xff, 0xff, 0x00, 0xff, 0x00, 0x00)], [0]), inflate);
    expect([...wide.pixels]).toEqual([128, 255, 1, 0]);
    expect(wide.source.bitDepth).toBe(16);
    const gray1 = decodePng(png(8, 1, 1, 0, [Uint8Array.of(0b10100101)], [0]), inflate);
    expect([...gray1.pixels].filter((_, i) => i % 4 === 0)).toEqual([255, 0, 255, 0, 0, 255, 0, 255]);
  });

  it('rejects interlaced, corrupt and oversized images explicitly', () => {
    const rows = [Uint8Array.of(1, 2, 3, 4)];
    expect(() => decodePng(png(1, 1, 8, 6, rows, [0], [], 1), inflate)).toThrow('PNG_INTERLACED_UNSUPPORTED');
    const corrupt = png(1, 1, 8, 6, rows, [0]);
    corrupt[20] ^= 1;
    expect(() => decodePng(corrupt, inflate)).toThrow(/PNG_CHUNK_CRC/);
    expect(() => decodePng(png(4, 4, 8, 6, Array.from({ length: 4 }, () => new Uint8Array(16)), [0]), inflate, { maxDimension: 2 })).toThrow('IMAGE_DIMENSIONS');
    expect(() => decodePng(Buffer.from('not a png at all, definitely not a png at all'), inflate)).toThrow('PNG_SIGNATURE');
  });
});

function tga(type: number, width: number, height: number, depth: number, descriptor: number, payload: number[]): Uint8Array {
  const header = new Uint8Array(18);
  header[2] = type; header[12] = width; header[14] = height; header[16] = depth; header[17] = descriptor;
  return Uint8Array.from([...header, ...payload]);
}

describe('TGA decoder', () => {
  it('decodes bottom-up BGR and top-down BGRA truecolor', () => {
    // Bottom-left origin: the first stored row is the bottom image row.
    const bottomUp = decodeTga(tga(2, 1, 2, 24, 0, [3, 2, 1, 6, 5, 4]));
    expect([...bottomUp.pixels]).toEqual([4, 5, 6, 255, 1, 2, 3, 255]);
    const topDown = decodeTga(tga(2, 2, 1, 32, 0x28, [3, 2, 1, 9, 6, 5, 4, 10]));
    expect([...topDown.pixels]).toEqual([1, 2, 3, 9, 4, 5, 6, 10]);
    expect(topDown.source.hasAlpha).toBe(true);
  });

  it('decodes RLE and grayscale, and rejects overruns and unsupported types', () => {
    const rle = decodeTga(tga(10, 3, 1, 24, 0x20, [0x81, 3, 2, 1, 0x00, 9, 8, 7]));
    expect([...rle.pixels]).toEqual([1, 2, 3, 255, 1, 2, 3, 255, 7, 8, 9, 255]);
    const gray = decodeTga(tga(3, 2, 1, 8, 0x20, [10, 200]));
    expect([...gray.pixels]).toEqual([10, 10, 10, 255, 200, 200, 200, 255]);
    expect(() => decodeTga(tga(10, 1, 1, 24, 0, [0x83, 1, 2, 3]))).toThrow('TGA_RLE_OVERRUN');
    expect(() => decodeTga(tga(1, 1, 1, 8, 0, [0]))).toThrow('TGA_TYPE_UNSUPPORTED');
    expect(() => decodeTga(tga(2, 2, 2, 24, 0, [1, 2, 3]))).toThrow('TGA_TRUNCATED');
  });
});
