/**
 * Offline decoders for legacy DX9-style DDS textures (FourCC DXT1/DXT3/DXT5,
 * ATI1/BC4U, ATI2/BC5U and uncompressed 32-bit BGRA/RGBA). Used by the asset
 * builder to re-encode third-party sources into the engine's runtime formats;
 * the runtime never decodes BCn on the CPU. Only the top level is decoded:
 * mips are rebuilt with buildMipChain semantics like every other source.
 * Self-contained (no imports, erasable TypeScript) for Node worker threads.
 */

export type LegacyDdsFormat = 'bc1' | 'bc2' | 'bc3' | 'bc4' | 'bc5' | 'bgra8' | 'rgba8';

export interface LegacyDdsImage {
  readonly width: number;
  readonly height: number;
  /** Decoded top level, RGBA8 row-major. BC4 -> (r, r, r, 255); BC5 -> (r, g, 0, 255). */
  readonly pixels: Uint8Array;
  readonly format: LegacyDdsFormat;
  readonly fourCC: string | null;
  readonly mipCount: number;
  /** Source channels carrying data: 1 for BC4, 2 for BC5, 3 or 4 otherwise. */
  readonly channels: number;
}

function fail(code: string): never { throw new RangeError(code); }

const CHANNEL_MAX = [31, 63, 31];
const palette = new Int32Array(16), alphaPalette = new Int32Array(8), raw = new Int32Array(6);

/** round(((wa * a + wb * b) / (wa + wb)) * 255 / max), half up, in exact integers. */
function interpolate(a: number, b: number, wa: number, wb: number, max: number): number {
  const d = (wa + wb) * max;
  return Math.floor(((wa * a + wb * b) * 510 + d - 1) / (2 * d));
}

/**
 * BC1 color block (8 bytes) into 16 RGBA pixels. Every palette entry is the
 * ideal value the format defines (the raw 5:6:5 endpoints v / max and their
 * 2/3, 1/3 or 1/2 blends) rounded to the nearest 8-bit value, ties down; this
 * is not bit replication, which differs for some endpoints. macOS ImageIO
 * decodes BC1-BC3 color identically; GPUs may differ by 1 per channel.
 * When `forceFourColor` (BC2/BC3 color blocks) the punch-through mode is off.
 */
export function decodeBc1Color(source: Uint8Array, offset: number, out: Uint8Array, outOffset: number, forceFourColor = false): void {
  const c0 = source[offset] | (source[offset + 1] << 8), c1 = source[offset + 2] | (source[offset + 3] << 8);
  raw[0] = c0 >> 11; raw[1] = (c0 >> 5) & 63; raw[2] = c0 & 31; raw[3] = c1 >> 11; raw[4] = (c1 >> 5) & 63; raw[5] = c1 & 31;
  for (let c = 0; c < 3; c++) {
    const max = CHANNEL_MAX[c], a = raw[c], b = raw[3 + c];
    palette[c] = interpolate(a, a, 1, 0, max); palette[4 + c] = interpolate(b, b, 1, 0, max);
    if (c0 > c1 || forceFourColor) { palette[8 + c] = interpolate(a, b, 2, 1, max); palette[12 + c] = interpolate(a, b, 1, 2, max); }
    else { palette[8 + c] = interpolate(a, b, 1, 1, max); palette[12 + c] = 0; }
  }
  palette[3] = 255; palette[7] = 255; palette[11] = 255; palette[15] = c0 > c1 || forceFourColor ? 255 : 0;
  const bits = (source[offset + 4] | (source[offset + 5] << 8) | (source[offset + 6] << 16) | (source[offset + 7] << 24)) >>> 0;
  for (let i = 0; i < 16; i++) {
    const index = (bits >>> (2 * i)) & 3, o = outOffset + i * 4;
    out[o] = palette[index * 4]; out[o + 1] = palette[index * 4 + 1]; out[o + 2] = palette[index * 4 + 2]; out[o + 3] = palette[index * 4 + 3];
  }
}

/** BC4 unsigned block (8 bytes) into channel `channel` of 16 RGBA pixels; interpolants rounded to nearest. */
export function decodeBc4Channel(source: Uint8Array, offset: number, out: Uint8Array, outOffset: number, channel: number): void {
  const a0 = source[offset], a1 = source[offset + 1];
  alphaPalette[0] = a0; alphaPalette[1] = a1;
  if (a0 > a1) for (let i = 1; i < 7; i++) alphaPalette[i + 1] = Math.floor(((7 - i) * a0 + i * a1 + 3) / 7);
  else {
    for (let i = 1; i < 5; i++) alphaPalette[i + 1] = Math.floor(((5 - i) * a0 + i * a1 + 2) / 5);
    alphaPalette[6] = 0; alphaPalette[7] = 255;
  }
  // 48 index bits, little endian, 3 bits per pixel.
  const low = (source[offset + 2] | (source[offset + 3] << 8) | (source[offset + 4] << 16)) >>> 0;
  const high = (source[offset + 5] | (source[offset + 6] << 8) | (source[offset + 7] << 16)) >>> 0;
  for (let i = 0; i < 16; i++) {
    const index = i < 8 ? (low >>> (3 * i)) & 7 : (high >>> (3 * (i - 8))) & 7;
    out[outOffset + i * 4 + channel] = alphaPalette[index];
  }
}

/** BC2 explicit 4-bit alpha (8 bytes) into the alpha channel of 16 pixels. */
function decodeBc2Alpha(source: Uint8Array, offset: number, out: Uint8Array, outOffset: number): void {
  for (let i = 0; i < 16; i++) {
    const nibble = (source[offset + (i >> 1)] >> ((i & 1) * 4)) & 15;
    out[outOffset + i * 4 + 3] = nibble * 17;
  }
}

/** Decodes one 4x4 block of the given compressed format into 64 RGBA bytes. */
export function decodeBcnBlock(format: 'bc1' | 'bc2' | 'bc3' | 'bc4' | 'bc5', source: Uint8Array, offset: number, out: Uint8Array): void {
  if (format === 'bc1') decodeBc1Color(source, offset, out, 0);
  else if (format === 'bc2') { decodeBc1Color(source, offset + 8, out, 0, true); decodeBc2Alpha(source, offset, out, 0); }
  else if (format === 'bc3') { decodeBc1Color(source, offset + 8, out, 0, true); decodeBc4Channel(source, offset, out, 0, 3); }
  else if (format === 'bc4') {
    decodeBc4Channel(source, offset, out, 0, 0);
    for (let i = 0; i < 16; i++) { out[i * 4 + 1] = out[i * 4 + 2] = out[i * 4]; out[i * 4 + 3] = 255; }
  } else {
    decodeBc4Channel(source, offset, out, 0, 0);
    decodeBc4Channel(source, offset + 8, out, 0, 1);
    for (let i = 0; i < 16; i++) { out[i * 4 + 2] = 0; out[i * 4 + 3] = 255; }
  }
}

const FOURCC: Record<string, LegacyDdsFormat> = { DXT1: 'bc1', DXT2: 'bc2', DXT3: 'bc2', DXT4: 'bc3', DXT5: 'bc3',
  ATI1: 'bc4', BC4U: 'bc4', ATI2: 'bc5', BC5U: 'bc5' };

/** Parses a legacy (non-DX10) 2D DDS and decodes its top level to RGBA8. */
export function decodeLegacyDds(bytes: Uint8Array, maxDimension = 8192): LegacyDdsImage {
  if (bytes.byteLength < 128 || bytes[0] !== 0x44 || bytes[1] !== 0x44 || bytes[2] !== 0x53 || bytes[3] !== 0x20) fail('DDS_MAGIC');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== 124 || view.getUint32(76, true) !== 32) fail('DDS_HEADER_SIZE');
  const height = view.getUint32(12, true), width = view.getUint32(16, true);
  const mipCount = Math.max(1, view.getUint32(28, true));
  const pfFlags = view.getUint32(80, true), caps2 = view.getUint32(112, true);
  if (width < 1 || height < 1 || width > maxDimension || height > maxDimension) fail('DDS_DIMENSIONS');
  if (caps2 & 0x200 || caps2 & 0x200000) fail('DDS_CUBE_OR_VOLUME_UNSUPPORTED');
  let format: LegacyDdsFormat, fourCC: string | null = null;
  if (pfFlags & 0x4) {
    fourCC = String.fromCharCode(bytes[84], bytes[85], bytes[86], bytes[87]);
    if (fourCC === 'DX10') fail('DDS_DX10_USE_PARSE_TEXTURE_DDS');
    const known = FOURCC[fourCC];
    if (!known) fail('DDS_FOURCC_UNSUPPORTED:' + fourCC);
    format = known;
  } else if ((pfFlags & 0x40) && view.getUint32(88, true) === 32) {
    const r = view.getUint32(92, true), g = view.getUint32(96, true), b = view.getUint32(100, true), a = view.getUint32(104, true);
    if (r === 0xff0000 && g === 0xff00 && b === 0xff && (a === 0xff000000 || a === 0)) format = 'bgra8';
    else if (r === 0xff && g === 0xff00 && b === 0xff0000 && (a === 0xff000000 || a === 0)) format = 'rgba8';
    else fail('DDS_RGB_MASKS_UNSUPPORTED');
  } else fail('DDS_PIXEL_FORMAT_UNSUPPORTED');
  const pixels = new Uint8Array(width * height * 4);
  const data = bytes.subarray(128);
  if (format === 'bgra8' || format === 'rgba8') {
    if (data.byteLength < width * height * 4) fail('DDS_TRUNCATED');
    const hasAlpha = view.getUint32(104, true) !== 0 && (pfFlags & 0x1) !== 0;
    for (let i = 0; i < width * height; i++) {
      const s = i * 4;
      pixels[s] = data[s + (format === 'bgra8' ? 2 : 0)]; pixels[s + 1] = data[s + 1]; pixels[s + 2] = data[s + (format === 'bgra8' ? 0 : 2)];
      pixels[s + 3] = hasAlpha ? data[s + 3] : 255;
    }
    return { width, height, pixels, format, fourCC, mipCount, channels: hasAlpha ? 4 : 3 };
  }
  const blockBytes = format === 'bc1' || format === 'bc4' ? 8 : 16;
  const bw = Math.ceil(width / 4), bh = Math.ceil(height / 4);
  if (data.byteLength < bw * bh * blockBytes) fail('DDS_TRUNCATED');
  const tile = new Uint8Array(64);
  for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) {
    decodeBcnBlock(format, data, (by * bw + bx) * blockBytes, tile);
    for (let y = 0; y < 4 && by * 4 + y < height; y++) for (let x = 0; x < 4 && bx * 4 + x < width; x++) {
      pixels.set(tile.subarray((y * 4 + x) * 4, (y * 4 + x) * 4 + 4), ((by * 4 + y) * width + bx * 4 + x) * 4);
    }
  }
  return { width, height, pixels, format, fourCC, mipCount, channels: format === 'bc4' ? 1 : format === 'bc5' ? 2 : 4 };
}

/**
 * Tangent-space normal map policy for builders: optionally flips green
 * (DirectX -> glTF/OpenGL convention) and, for two-channel sources, rebuilds
 * Z = sqrt(1 - x^2 - y^2) from X/Y decoded as v / 127.5 - 1, stored as
 * round((z * 0.5 + 0.5) * 255). In place; alpha (unused by glTF) is kept.
 */
export function convertNormalMap(pixels: Uint8Array, options: { flipGreen: boolean; reconstructZ: boolean }): void {
  for (let i = 0; i < pixels.length; i += 4) {
    if (options.flipGreen) pixels[i + 1] = 255 - pixels[i + 1];
    if (options.reconstructZ) {
      const x = pixels[i] / 127.5 - 1, y = pixels[i + 1] / 127.5 - 1;
      const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
      pixels[i + 2] = Math.round((z * 0.5 + 0.5) * 255);
    }
  }
}
