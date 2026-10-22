import { describe, expect, it } from 'vitest';
import { finalGatherRasterRayBasis, invertFinalGatherProjection, planFinalGatherGrid, resolveFinalGatherOptions } from '../src/final-gather';
import { lookAt, multiply, perspective } from '../src/math';

describe('final gather CPU contract', () => {
  it('resolves immutable defaults and independent normal bias', () => {
    const defaults = resolveFinalGatherOptions();
    expect(defaults).toEqual({ pixelSpacing: 8, maxAnchors: 16_384, raysPerAnchor: 2,
      pathSegments: 1, maxHistory: 64, normalBias: 0.0001, maxBytes: 16 * 1024 * 1024,
      receiver: 'world-query', maxReceiverBytes: 128 * 1024 * 1024 });
    expect(Object.isFrozen(defaults)).toBe(true);
    expect(resolveFinalGatherOptions({ pixelSpacing: 1, normalBias: 0.01 }).normalBias).toBe(0.01);
    expect(resolveFinalGatherOptions({ pixelSpacing: 128, normalBias: 0 }).normalBias).toBe(0);
  });

  it('validates each option at its hard boundary', () => {
    expect(resolveFinalGatherOptions({ maxAnchors: 65_536, raysPerAnchor: 16,
      maxHistory: 256, maxBytes: 64 * 1024 * 1024 }).maxAnchors).toBe(65_536);
    for (const [key, bad] of [
      ['pixelSpacing', 0], ['pixelSpacing', 129], ['maxAnchors', 0], ['maxAnchors', 65_537],
      ['raysPerAnchor', 0], ['raysPerAnchor', 17], ['maxHistory', 0], ['maxHistory', 257],
      ['pathSegments', 0], ['pathSegments', 5], ['pathSegments', 1.5],
      ['normalBias', -1], ['normalBias', 0.011], ['maxBytes', 0], ['maxBytes', 64 * 1024 * 1024 + 1],
      ['maxReceiverBytes', 0], ['maxReceiverBytes', 512 * 1024 * 1024 + 1],
    ] as const) {
      expect(() => resolveFinalGatherOptions({ [key]: bad })).toThrow();
      expect(() => resolveFinalGatherOptions({ [key]: NaN })).toThrow();
      expect(() => resolveFinalGatherOptions({ [key]: null })).toThrow();
      expect(() => resolveFinalGatherOptions({ [key]: '1' })).toThrow();
    }
    expect(() => resolveFinalGatherOptions(null as unknown as object)).toThrow('FINAL_GATHER_OPTIONS');
    expect(() => resolveFinalGatherOptions({ receiver: 'invalid' as 'raster' })).toThrow('FINAL_GATHER_RECEIVER');
  });

  it('reconstructs raster world positions from clip.w across near and far depths at a distant camera', () => {
    const camera = multiply(perspective(Math.PI / 3, 16 / 9, 0.1, 2000),
      lookAt([4096, 123, -8192], [4096.4, 122.8, -8191], [0, 1, 0]));
    const basis = finalGatherRasterRayBasis(invertFinalGatherProjection(camera));
    for (const depth of [0.2, 1, 10, 500, 1500]) for (const xy of [[0, 0], [-0.8, 0.7], [0.9, -0.6]]) {
      const point = basis.origin.map((origin, axis) => origin + depth *
        (basis.dx[axis] * xy[0] + basis.dy[axis] * xy[1] + basis.dz[axis]));
      const clip = Array.from({ length: 4 }, (_, row) => camera[row] * point[0] + camera[4 + row] * point[1]
        + camera[8 + row] * point[2] + camera[12 + row]);
      expect(clip[3]).toBeCloseTo(depth, 4);
      expect(clip[0] / clip[3]).toBeCloseTo(xy[0], 4);
      expect(clip[1] / clip[3]).toBeCloseTo(xy[1], 4);
    }
  });

  it('keeps requested resolution when the 320 by 180 grid exactly fits ray capacity', () => {
    expect(planFinalGatherGrid(320, 180, { raysPerAnchor: 2 }, 1840, 1840))
      .toEqual({ spacing: 8, columns: 40, rows: 23, anchorCount: 920, rayCount: 1840 });
  });

  it('increases integer spacing to the finest grid satisfying all budgets', () => {
    const options = { pixelSpacing: 1, raysPerAnchor: 4, maxAnchors: 200 };
    const plan = planFinalGatherGrid(32_768, 20_000, options, 400, 400);
    expect(plan.anchorCount).toBeLessThanOrEqual(100);
    expect(plan.rayCount).toBeLessThanOrEqual(400);
    expect(plan.rayCount).toBe(plan.anchorCount * 4);
    expect(Math.ceil(32_768 / (plan.spacing - 1)) * Math.ceil(20_000 / (plan.spacing - 1)))
      .toBeGreaterThan(100);
    expect(planFinalGatherGrid(1, 1, { pixelSpacing: 128 }, 2, 2))
      .toEqual({ spacing: 128, columns: 1, rows: 1, anchorCount: 1, rayCount: 2 });
  });

  it('rejects a budget unable to fund one complete anchor and invalid dimensions', () => {
    expect(() => planFinalGatherGrid(320, 180, { raysPerAnchor: 4 }, 3, 100))
      .toThrow('FINAL_GATHER_QUERY_RAY_BUDGET');
    expect(() => planFinalGatherGrid(320, 180, { raysPerAnchor: 4 }, 100, 3))
      .toThrow('FINAL_GATHER_QUERY_RAY_BUDGET');
    for (const size of [0, 0.5, 32_769, NaN, Infinity]) {
      expect(() => planFinalGatherGrid(size, 1, {}, 1, 1)).toThrow();
    }
  });
});
