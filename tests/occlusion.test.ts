import { describe, expect, it } from 'vitest';
import { planOcclusionPyramid, resolveOcclusionOptions, type OcclusionOptions } from '../src/occlusion';

const MIB = 1024 * 1024;

describe('occlusion CPU contracts', () => {
  it('resolves and freezes bounded defaults and endpoint values', () => {
    const defaults = resolveOcclusionOptions();
    expect(defaults).toEqual({ pixelSize: 2, depthBias: 0.00002, maxBytes: 64 * MIB });
    expect(Object.isFrozen(defaults)).toBe(true);
    for (const pixelSize of [1, 2, 4, 8]) {
      expect(resolveOcclusionOptions({ pixelSize }).pixelSize).toBe(pixelSize);
    }
    expect(resolveOcclusionOptions({ depthBias: 0, maxBytes: 1 }).depthBias).toBe(0);
    expect(resolveOcclusionOptions({ depthBias: 0.01, maxBytes: 256 * MIB }).maxBytes).toBe(256 * MIB);
  });

  it('rejects malformed options and viewport dimensions', () => {
    for (const bad of [0, 3, 16, 2.5, NaN, Infinity, null, '2']) {
      expect(() => resolveOcclusionOptions({ pixelSize: bad } as OcclusionOptions))
        .toThrow('OCCLUSION_PIXEL_SIZE');
    }
    for (const bad of [-0.00001, 0.01001, NaN, Infinity, null, '0.001']) {
      expect(() => resolveOcclusionOptions({ depthBias: bad } as OcclusionOptions))
        .toThrow('OCCLUSION_DEPTH_BIAS');
    }
    for (const bad of [0, 1.5, 256 * MIB + 1, NaN, Infinity, null, '1024']) {
      expect(() => resolveOcclusionOptions({ maxBytes: bad } as OcclusionOptions))
        .toThrow('OCCLUSION_MAX_BYTES');
    }
    for (const bad of [null, [], 1, 'settings']) {
      expect(() => resolveOcclusionOptions(bad as unknown as OcclusionOptions))
        .toThrow('OCCLUSION_OPTIONS');
    }
    for (const bad of [0, -1, 1.5, 16_385, NaN, Infinity, null]) {
      expect(() => planOcclusionPyramid(bad as number, 1)).toThrow('OCCLUSION_WIDTH');
      expect(() => planOcclusionPyramid(1, bad as number)).toThrow('OCCLUSION_HEIGHT');
    }
  });

  it('pads non-power-of-two viewport edges and accounts for every mip through 1x1', () => {
    // 5x3 at two-pixel resolution is 3x2, padded to 4x2: 8+2+1 texels.
    expect(planOcclusionPyramid(5, 3)).toEqual({
      width: 4, height: 2, pixelSize: 2, mipLevelCount: 3,
      textureBytes: 44, bytes: 296,
    });
    expect(planOcclusionPyramid(1, 5)).toEqual({
      width: 1, height: 4, pixelSize: 2, mipLevelCount: 3,
      textureBytes: 28, bytes: 264,
    });
    expect(planOcclusionPyramid(1, 1)).toEqual({
      width: 1, height: 1, pixelSize: 2, mipLevelCount: 1,
      textureBytes: 4, bytes: 216,
    });
    expect(Object.isFrozen(planOcclusionPyramid(1, 1))).toBe(true);
  });

  it('uses the exact two-texture plus 208-byte budget boundary', () => {
    expect(planOcclusionPyramid(5, 3, { maxBytes: 296 }).bytes).toBe(296);
    expect(() => planOcclusionPyramid(5, 3, { maxBytes: 295 })).toThrow('OCCLUSION_BYTE_BUDGET');
    expect(() => planOcclusionPyramid(1, 1, { maxBytes: 215 })).toThrow('OCCLUSION_BYTE_BUDGET');
    expect(() => planOcclusionPyramid(16_384, 16_384,
      { pixelSize: 1, maxBytes: 256 * MIB })).toThrow('OCCLUSION_BYTE_BUDGET');
    expect(planOcclusionPyramid(16_384, 16_384,
      { pixelSize: 8 }).bytes).toBeLessThanOrEqual(64 * MIB);
  });

  it('matches an independent rectangular mip-size sum and max-depth reduction', () => {
    for (const [viewportWidth, viewportHeight, pixelSize] of [
      [1, 1, 1], [2, 17, 2], [129, 5, 4], [1919, 1081, 8],
    ]) {
      const plan = planOcclusionPyramid(viewportWidth, viewportHeight, { pixelSize });
      const expectedMipCount = 1 + Math.log2(Math.max(plan.width, plan.height));
      const texels = Array.from({ length: expectedMipCount }, (_, level) =>
        Math.max(1, plan.width / (2 ** level)) * Math.max(1, plan.height / (2 ** level)))
        .reduce((sum, count) => sum + count, 0);
      expect(plan.mipLevelCount).toBe(expectedMipCount);
      expect(plan.textureBytes).toBe(4 * texels);
      expect(plan.bytes).toBe(8 * texels + 208);
    }
    // Offline max-depth reference for a 4x2 base: each level takes the
    // maximum of its in-bounds 2x2 children, including narrow edge mips.
    let width = 4, height = 2, depth = [0.1, 0.8, 0.3, 0.2, 0.4, 0.6, 0.9, 0.5];
    const levels = [depth];
    while (width > 1 || height > 1) {
      const nextWidth = Math.max(1, width / 2), nextHeight = Math.max(1, height / 2);
      const next = Array.from({ length: nextWidth * nextHeight }, (_, index) => {
        const x = index % nextWidth, y = Math.floor(index / nextWidth);
        const values: number[] = [];
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
          if (2 * x + dx < width && 2 * y + dy < height) {
            values.push(depth[(2 * y + dy) * width + 2 * x + dx]);
          }
        }
        return Math.max(...values);
      });
      levels.push(next);
      depth = next;
      width = nextWidth;
      height = nextHeight;
    }
    expect(levels).toEqual([
      [0.1, 0.8, 0.3, 0.2, 0.4, 0.6, 0.9, 0.5],
      [0.8, 0.9],
      [0.9],
    ]);
  });
});
