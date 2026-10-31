/**
 * Strict offline image decoders producing RGBA8 for the texture builder.
 * No color management is applied: embedded gAMA/sRGB/iCCP data is ignored and
 * the caller states the sRGB/linear meaning, matching the previous
 * `--ignore-srgb` policy. Inflate is injected (node:zlib in Node builds) so the
 * module stays free of platform imports. Unsupported layouts fail explicitly.
 */

export interface DecodedRgba8 {
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint8Array;
  /** Source sample precision and layout, for build records. */
  readonly source: { readonly format: string; readonly bitDepth: number; readonly channels: number; readonly hasAlpha: boolean };
}

export interface ImageDecodeLimits {
  readonly maxDimension?: number;
  readonly maxPixels?: number;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();
function crc32(bytes: Uint8Array, start: number, end: number): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function fail(code: string): never { throw new RangeError(code); }

function checkSize(width: number, height: number, limits: ImageDecodeLimits): void {
  const maxDimension = limits.maxDimension ?? 8192, maxPixels = limits.maxPixels ?? 8192 * 8192;
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1
    || width > maxDimension || height > maxDimension || width * height > maxPixels) fail('IMAGE_DIMENSIONS');
}

/** Decodes a non-interlaced PNG (all bit depths and color types) to RGBA8. */
export function decodePng(bytes: Uint8Array, inflate: (data: Uint8Array) => Uint8Array, limits: ImageDecodeLimits = {}): DecodedRgba8 {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.byteLength < 45 || signature.some((value, i) => bytes[i] !== value)) fail('PNG_SIGNATURE');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 8, width = 0, height = 0, bitDepth = 0, colorType = -1, interlace = 0;
  let palette: Uint8Array | null = null, transparency: Uint8Array | null = null, ended = false;
  const idat: Uint8Array[] = [];
  let idatBytes = 0;
  while (offset + 12 <= bytes.byteLength) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]);
    const dataStart = offset + 8, dataEnd = dataStart + length;
    if (length > 0x7fffffff || dataEnd + 4 > bytes.byteLength) fail('PNG_TRUNCATED_CHUNK');
    if (crc32(bytes, offset + 4, dataEnd) !== view.getUint32(dataEnd)) fail('PNG_CHUNK_CRC:' + type);
    const data = bytes.subarray(dataStart, dataEnd);
    if (type === 'IHDR') {
      if (offset !== 8 || length !== 13) fail('PNG_IHDR');
      width = view.getUint32(dataStart); height = view.getUint32(dataStart + 4);
      bitDepth = data[8]; colorType = data[9]; interlace = data[12];
      if (data[10] !== 0 || data[11] !== 0) fail('PNG_COMPRESSION_OR_FILTER_METHOD');
    } else if (type === 'PLTE') palette = data;
    else if (type === 'tRNS') transparency = data;
    else if (type === 'IDAT') { idat.push(data); idatBytes += length; }
    else if (type === 'IEND') { ended = true; break; }
    else if (!(bytes[offset + 4] & 0x20)) fail('PNG_UNKNOWN_CRITICAL_CHUNK:' + type);
    offset = dataEnd + 4;
  }
  if (!ended || !idat.length || colorType < 0) fail('PNG_MISSING_CHUNKS');
  checkSize(width, height, limits);
  const allowed: Record<number, number[]> = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] };
  if (!allowed[colorType]?.includes(bitDepth)) fail('PNG_BIT_DEPTH_OR_COLOR_TYPE');
  if (interlace !== 0) fail('PNG_INTERLACED_UNSUPPORTED');
  if (colorType === 3 && (!palette || palette.length % 3 !== 0 || palette.length === 0)) fail('PNG_PALETTE');
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType as 0 | 2 | 3 | 4 | 6];
  const bitsPerPixel = channels * bitDepth, stride = Math.ceil(width * bitsPerPixel / 8), bpp = Math.max(1, bitsPerPixel >> 3);
  const compressed = new Uint8Array(idatBytes);
  let cursor = 0;
  for (const part of idat) { compressed.set(part, cursor); cursor += part.length; }
  const raw = inflate(compressed);
  if (raw.byteLength !== (stride + 1) * height) fail('PNG_DATA_LENGTH');
  // Unfilter in place into a contiguous scanline buffer.
  const lines = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)], src = y * (stride + 1) + 1, dst = y * stride, prev = dst - stride;
    if (filter > 4) fail('PNG_FILTER');
    for (let x = 0; x < stride; x++) {
      const value = raw[src + x];
      const a = x >= bpp ? lines[dst + x - bpp] : 0, b = y ? lines[prev + x] : 0, c = x >= bpp && y ? lines[prev + x - bpp] : 0;
      let predicted = 0;
      if (filter === 1) predicted = a;
      else if (filter === 2) predicted = b;
      else if (filter === 3) predicted = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        predicted = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      lines[dst + x] = (value + predicted) & 0xff;
    }
  }
  const pixels = new Uint8Array(width * height * 4);
  const sample = (y: number, index: number): number => {
    // index counts samples within the row.
    if (bitDepth === 8) return lines[y * stride + index];
    if (bitDepth === 16) {
      const v = (lines[y * stride + index * 2] << 8) | lines[y * stride + index * 2 + 1];
      return Math.floor((v * 255 + 32767) / 65535);
    }
    const bitOffset = index * bitDepth, byte = lines[y * stride + (bitOffset >> 3)];
    return (byte >> (8 - bitDepth - (bitOffset & 7))) & ((1 << bitDepth) - 1);
  };
  const raw16 = (y: number, index: number): number => bitDepth === 16
    ? (lines[y * stride + index * 2] << 8) | lines[y * stride + index * 2 + 1] : sample(y, index);
  const key = transparency && (colorType === 0 || colorType === 2) && transparency.length >= (colorType === 0 ? 2 : 6)
    ? Array.from({ length: colorType === 0 ? 1 : 3 }, (_, i) => (transparency![i * 2] << 8) | transparency![i * 2 + 1]) : null;
  const scale = bitDepth < 8 && colorType === 0 ? 255 / ((1 << bitDepth) - 1) : 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      if (colorType === 3) {
        const entry = sample(y, x);
        if (entry * 3 + 2 >= palette!.length) fail('PNG_PALETTE_INDEX');
        pixels[o] = palette![entry * 3]; pixels[o + 1] = palette![entry * 3 + 1]; pixels[o + 2] = palette![entry * 3 + 2];
        pixels[o + 3] = transparency && entry < transparency.length ? transparency[entry] : 255;
      } else if (colorType === 0 || colorType === 4) {
        const gray = Math.round(sample(y, x * (channels)) * scale);
        pixels[o] = pixels[o + 1] = pixels[o + 2] = gray;
        pixels[o + 3] = colorType === 4 ? sample(y, x * 2 + 1) : key && raw16(y, x) === key[0] ? 0 : 255;
      } else {
        for (let c = 0; c < 3; c++) pixels[o + c] = sample(y, x * channels + c);
        pixels[o + 3] = colorType === 6 ? sample(y, x * 4 + 3)
          : key && raw16(y, x * 3) === key[0] && raw16(y, x * 3 + 1) === key[1] && raw16(y, x * 3 + 2) === key[2] ? 0 : 255;
      }
    }
  }
  return { width, height, pixels, source: { format: 'png', bitDepth, channels, hasAlpha: colorType === 4 || colorType === 6 || Boolean(transparency) } };
}

/** Decodes uncompressed or RLE truecolor/grayscale TGA (types 2, 3, 10, 11) to RGBA8. */
export function decodeTga(bytes: Uint8Array, limits: ImageDecodeLimits = {}): DecodedRgba8 {
  if (bytes.byteLength < 18) fail('TGA_HEADER');
  const idLength = bytes[0], colorMapType = bytes[1], type = bytes[2];
  const width = bytes[12] | (bytes[13] << 8), height = bytes[14] | (bytes[15] << 8), depth = bytes[16], descriptor = bytes[17];
  if (colorMapType !== 0 || ![2, 3, 10, 11].includes(type)) fail('TGA_TYPE_UNSUPPORTED');
  const gray = type === 3 || type === 11, rle = type >= 9;
  if ((gray && depth !== 8) || (!gray && depth !== 24 && depth !== 32)) fail('TGA_DEPTH_UNSUPPORTED');
  checkSize(width, height, limits);
  const bytesPerPixel = depth >> 3, count = width * height;
  const flat = new Uint8Array(count * bytesPerPixel);
  let offset = 18 + idLength;
  if (!rle) {
    if (offset + flat.length > bytes.byteLength) fail('TGA_TRUNCATED');
    flat.set(bytes.subarray(offset, offset + flat.length));
  } else {
    let written = 0;
    while (written < flat.length) {
      if (offset >= bytes.byteLength) fail('TGA_TRUNCATED');
      const header = bytes[offset++], run = (header & 0x7f) + 1;
      if (written + run * bytesPerPixel > flat.length) fail('TGA_RLE_OVERRUN');
      if (header & 0x80) {
        if (offset + bytesPerPixel > bytes.byteLength) fail('TGA_TRUNCATED');
        for (let i = 0; i < run; i++) flat.set(bytes.subarray(offset, offset + bytesPerPixel), written + i * bytesPerPixel);
        offset += bytesPerPixel;
      } else {
        if (offset + run * bytesPerPixel > bytes.byteLength) fail('TGA_TRUNCATED');
        flat.set(bytes.subarray(offset, offset + run * bytesPerPixel), written);
        offset += run * bytesPerPixel;
      }
      written += run * bytesPerPixel;
    }
  }
  const topDown = (descriptor & 0x20) !== 0, rightToLeft = (descriptor & 0x10) !== 0;
  const alphaBits = descriptor & 0x0f;
  const pixels = new Uint8Array(count * 4);
  for (let y = 0; y < height; y++) {
    const sy = topDown ? y : height - 1 - y;
    for (let x = 0; x < width; x++) {
      const sx = rightToLeft ? width - 1 - x : x, s = (sy * width + sx) * bytesPerPixel, o = (y * width + x) * 4;
      if (gray) { pixels[o] = pixels[o + 1] = pixels[o + 2] = flat[s]; pixels[o + 3] = 255; }
      else {
        pixels[o] = flat[s + 2]; pixels[o + 1] = flat[s + 1]; pixels[o + 2] = flat[s];
        pixels[o + 3] = depth === 32 && alphaBits ? flat[s + 3] : 255;
      }
    }
  }
  return { width, height, pixels, source: { format: rle ? 'tga-rle' : 'tga', bitDepth: 8, channels: gray ? 1 : bytesPerPixel, hasAlpha: depth === 32 && alphaBits > 0 } };
}
