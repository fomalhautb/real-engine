/**
 * Offline RGBA8 mip generation with exactly the semantics of
 * `buildMipChain` in texture-assets.ts (source-area box filter, sRGB RGB in
 * linear light, alpha linear and independent, round to nearest). Even-sized
 * levels use a 2x2 fast path with the same operation order, so results are
 * bit-identical (covered by tests). Self-contained so Node worker threads can
 * import it with type stripping.
 */

export interface MipLevelRgba8 { readonly width: number; readonly height: number; readonly pixels: Uint8Array }

function srgbToLinear(value: number): number {
  const normalized = value / 255;
  return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
}
function linearToSrgb(value: number): number {
  const encoded = value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
  return Math.round(Math.max(0, Math.min(1, encoded)) * 255);
}
const SRGB_TO_LINEAR = Float64Array.from({ length: 256 }, (_, v) => srgbToLinear(v));

function downsampleGeneral(source: MipLevelRgba8, srgb: boolean): MipLevelRgba8 {
  const width = Math.max(1, Math.floor(source.width / 2)), height = Math.max(1, Math.floor(source.height / 2));
  const pixels = new Uint8Array(width * height * 4);
  const xScale = source.width / width, yScale = source.height / height;
  const sums = [0, 0, 0, 0];
  for (let y = 0; y < height; y++) {
    const yStart = y * yScale, yEnd = Math.min(source.height, (y + 1) * yScale);
    for (let x = 0; x < width; x++) {
      const xStart = x * xScale, xEnd = Math.min(source.width, (x + 1) * xScale);
      sums[0] = 0; sums[1] = 0; sums[2] = 0; sums[3] = 0;
      for (let sy = Math.floor(yStart); sy < Math.min(source.height, Math.ceil(yEnd)); sy++) {
        const yWeight = Math.min(sy + 1, yEnd) - Math.max(sy, yStart);
        if (yWeight <= 0) continue;
        for (let sx = Math.floor(xStart); sx < Math.min(source.width, Math.ceil(xEnd)); sx++) {
          const w = yWeight * (Math.min(sx + 1, xEnd) - Math.max(sx, xStart));
          if (w <= 0) continue;
          const o = (sy * source.width + sx) * 4;
          for (let c = 0; c < 3; c++) {
            const v = source.pixels[o + c];
            sums[c] += w * (srgb ? SRGB_TO_LINEAR[v] : v / 255);
          }
          sums[3] += w * source.pixels[o + 3] / 255;
        }
      }
      const area = (xEnd - xStart) * (yEnd - yStart), o = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        const value = sums[c] / area;
        pixels[o + c] = srgb ? linearToSrgb(value) : Math.round(value * 255);
      }
      pixels[o + 3] = Math.round(sums[3] / area * 255);
    }
  }
  return { width, height, pixels };
}

/** Even width and height: every destination texel covers exactly 2x2 source texels with weight 1. */
function downsampleEven(source: MipLevelRgba8, srgb: boolean): MipLevelRgba8 {
  const width = source.width / 2, height = source.height / 2, sw = source.width, src = source.pixels;
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = ((2 * y) * sw + 2 * x) * 4, b = a + 4, c = a + sw * 4, d = c + 4, o = (y * width + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        // Same accumulation order as the general path: (y0,x0), (y0,x1), (y1,x0), (y1,x1); weights are exactly 1.
        let sum = 0;
        if (srgb) { sum += 1 * SRGB_TO_LINEAR[src[a + ch]]; sum += 1 * SRGB_TO_LINEAR[src[b + ch]]; sum += 1 * SRGB_TO_LINEAR[src[c + ch]]; sum += 1 * SRGB_TO_LINEAR[src[d + ch]]; }
        else { sum += 1 * (src[a + ch] / 255); sum += 1 * (src[b + ch] / 255); sum += 1 * (src[c + ch] / 255); sum += 1 * (src[d + ch] / 255); }
        const value = sum / 4;
        pixels[o + ch] = srgb ? linearToSrgb(value) : Math.round(value * 255);
      }
      let alpha = 0;
      alpha += 1 * src[a + 3] / 255; alpha += 1 * src[b + 3] / 255; alpha += 1 * src[c + 3] / 255; alpha += 1 * src[d + 3] / 255;
      pixels[o + 3] = Math.round(alpha / 4 * 255);
    }
  }
  return { width, height, pixels };
}

/** Full chain from the given level 0 down to 1x1. The first level is the input object itself. */
export function buildRgba8MipChain(width: number, height: number, pixels: Uint8Array, colorSpace: 'srgb' | 'linear'): MipLevelRgba8[] {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 || width > 8192 || height > 8192
    || pixels.byteLength !== width * height * 4) throw new RangeError('TEXTURE_MIP_CHAIN: invalid RGBA8 level');
  const srgb = colorSpace === 'srgb';
  const levels: MipLevelRgba8[] = [{ width, height, pixels }];
  while (levels.at(-1)!.width > 1 || levels.at(-1)!.height > 1) {
    const source = levels.at(-1)!;
    levels.push(source.width % 2 === 0 && source.height % 2 === 0 ? downsampleEven(source, srgb) : downsampleGeneral(source, srgb));
  }
  return levels;
}
