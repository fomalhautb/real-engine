/** Diagnostic 4x4 UNORM blocks, not a general texture encoder. All expected
 * channels are encoded-space linear values; no sRGB transfer is requested.
 *
 * Field references (Khronos Data Format Specification source):
 * BC7 mode 6: https://github.com/KhronosGroup/DataFormat/blob/main/bptc.txt
 *   `table-bptcmodebits`: bits 0..6 mode, 7..62 paired 7-bit RGBA endpoints,
 *   63..64 per-endpoint P bits, 65..127 indices. Blocks are little-endian.
 * ETC2 RGBA8: https://github.com/KhronosGroup/DataFormat/blob/main/etc2.txt
 *   `Table-etc2eac-dataformat` and `Equation-etc2eac-eqn-base`: first 8 bytes
 *   alpha base/multiplier/table/indices; next 8 bytes individual-mode RGB.
 *   Table 0 with zero RGB pixel indices adds +2 to expanded 4-bit base RGB.
 * ASTC: https://github.com/KhronosGroup/DataFormat/blob/main/astc.txt
 *   `astc-2d-void-extent-block-layout-overview`: bits 0..8 = 0x1fc,
 *   bit 9 = 0 for LDR, bits 10..11 = 1, all-one extent bits 12..47;
 *   bits 48..63 complete the extent; bits 64..127 hold R,G,B,A as
 *   little-endian UNORM16.
 */

export type Rgba8 = readonly [number, number, number, number];
/** Matches PreparedTextureAsset.format; callers select colorSpace: 'linear'. */
export type CompressedBlockFormat = 'bc7' | 'etc2-rgba8' | 'astc-4x4';
export interface CompressedSolidBlockCase {
  name: string;
  format: CompressedBlockFormat;
  bytes: Uint8Array<ArrayBuffer>;
  /** Expected sampled RGBA, normalized to [0, 1], before material factors. */
  expected: readonly [number, number, number, number];
  /** Maximum practical per-channel absolute error for GPU readback. */
  tolerance: number;
}

function channel(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 255) throw new RangeError(label);
  return value;
}

function normalized(rgba: Rgba8): readonly [number, number, number, number] {
  return [rgba[0] / 255, rgba[1] / 255, rgba[2] / 255, rgba[3] / 255];
}

function bits(block: Uint8Array, offset: number, count: number, value: number): void {
  for (let bit = 0; bit < count; bit++) {
    if ((value >>> bit) & 1) block[(offset + bit) >>> 3] |= 1 << ((offset + bit) & 7);
  }
}

/** BC7 mode 6 with equal endpoints, zero indices, and a shared zero P-bit.
 * This intentionally supports only even 8-bit channels, decoded exactly. */
export function bc7Mode6SolidBlock(rgba: Rgba8): Uint8Array<ArrayBuffer> {
  for (const value of rgba) {
    channel(value, 'BC7 channel');
    if (value & 1) throw new RangeError('BC7 mode-6 zero-P endpoint requires even channels');
  }
  const block = new Uint8Array(16);
  bits(block, 6, 1, 1); // Unary mode identifier 0000001, LSB first.
  let offset = 7;
  for (const value of rgba) {
    bits(block, offset, 7, value >>> 1);
    bits(block, offset + 7, 7, value >>> 1);
    offset += 14;
  }
  // Both endpoint P-bits and all indices remain zero. Identical endpoints
  // make the interpolation index and anchor convention immaterial.
  return block;
}

/** ETC2 RGBA8 individual mode with constant RGB (138,70,36) and exact alpha.
 * RGB base nibbles are (8,4,2) in both subblocks; table 0 and all-zero
 * pixel indices add +2. Alpha multiplier zero makes every texel the base. */
export function etc2Rgba8SolidBlock(alpha: number): Uint8Array<ArrayBuffer> {
  channel(alpha, 'ETC2 alpha');
  const block = new Uint8Array(16);
  block[0] = alpha;
  block.set([0x88, 0x44, 0x22, 0, 0, 0, 0, 0], 8);
  return block;
}

/** ASTC 4x4 LDR void-extent block, with the extent ignored (all-one bits). */
export function astc4x4LdrSolidBlock(rgba: Rgba8): Uint8Array<ArrayBuffer> {
  const block = new Uint8Array(16);
  block.set([0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
  const view = new DataView(block.buffer);
  rgba.forEach((value, index) => {
    channel(value, 'ASTC channel');
    view.setUint16(8 + index * 2, value * 257, true);
  });
  return block;
}

const etcRgb = [138, 70, 36] as const;
export const compressedSolidBlockCases: readonly CompressedSolidBlockCase[] = [
  { name: 'bc7-opaque', format: 'bc7',
    bytes: bc7Mode6SolidBlock([128, 64, 32, 254]), expected: normalized([128, 64, 32, 254]), tolerance: 2 / 255 },
  { name: 'bc7-mask-discard', format: 'bc7',
    bytes: bc7Mode6SolidBlock([128, 0, 0, 64]), expected: normalized([128, 0, 0, 64]), tolerance: 2 / 255 },
  { name: 'etc2-opaque', format: 'etc2-rgba8',
    bytes: etc2Rgba8SolidBlock(255), expected: normalized([...etcRgb, 255]), tolerance: 2 / 255 },
  { name: 'etc2-mask-discard', format: 'etc2-rgba8',
    bytes: etc2Rgba8SolidBlock(64), expected: normalized([...etcRgb, 64]), tolerance: 2 / 255 },
  { name: 'astc-opaque', format: 'astc-4x4',
    bytes: astc4x4LdrSolidBlock([64, 128, 192, 255]),
    expected: normalized([64, 128, 192, 255]), tolerance: 2 / 255 },
  { name: 'astc-mask-discard', format: 'astc-4x4',
    bytes: astc4x4LdrSolidBlock([64, 128, 192, 64]),
    expected: normalized([64, 128, 192, 64]), tolerance: 2 / 255 },
];
