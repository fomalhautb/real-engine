import { describe, expect, it } from 'vitest';
import { buildMipChain } from '../src/texture-assets';
import { buildRgba8MipChain } from '../src/texture-mip-chain';

function image(width: number, height: number, seed: number): Uint8Array {
  const pixels = new Uint8Array(width * height * 4);
  let state = seed;
  for (let i = 0; i < pixels.length; i++) { state = (Math.imul(state, 1103515245) + 12345) >>> 0; pixels[i] = state >>> 24; }
  return pixels;
}

describe('offline RGBA8 mip chain', () => {
  // Even sizes use the fast 2x2 path and odd sizes the general source-area path;
  // both must equal the engine's buildMipChain byte for byte.
  for (const [width, height] of [[64, 64], [32, 8], [1, 16], [37, 21], [12, 7], [5, 1], [96, 40]]) {
    for (const colorSpace of ['srgb', 'linear'] as const) {
      it(`matches buildMipChain for ${width}x${height} ${colorSpace}`, () => {
        const pixels = image(width, height, width * 131 + height);
        const reference = buildMipChain({ width, height, pixels, colorSpace });
        const levels = buildRgba8MipChain(width, height, pixels, colorSpace);
        expect(levels.map(level => [level.width, level.height])).toEqual(reference.map(level => [level.width, level.height]));
        levels.forEach((level, index) => expect(Buffer.from(level.pixels).equals(Buffer.from(reference[index].pixels)), `level ${index}`).toBe(true));
      });
    }
  }

  it('rejects invalid levels', () => {
    expect(() => buildRgba8MipChain(0, 4, new Uint8Array(0), 'srgb')).toThrow(RangeError);
    expect(() => buildRgba8MipChain(4, 4, new Uint8Array(63), 'linear')).toThrow(RangeError);
    expect(() => buildRgba8MipChain(8193, 1, new Uint8Array(8193 * 4), 'linear')).toThrow(RangeError);
  });
});
