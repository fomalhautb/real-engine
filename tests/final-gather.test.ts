import { describe, expect, it } from 'vitest';
import { planFinalGatherGrid, resolveFinalGatherOptions } from '../src/final-gather';

describe('final gather CPU contract', () => {
  it('resolves immutable defaults and independent normal bias', () => {
    const defaults = resolveFinalGatherOptions();
    expect(defaults).toEqual({ pixelSpacing: 8, maxAnchors: 16_384, raysPerAnchor: 2,
      maxHistory: 64, normalBias: 0.0001, maxBytes: 16 * 1024 * 1024 });
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
      ['normalBias', -1], ['normalBias', 0.011], ['maxBytes', 0], ['maxBytes', 64 * 1024 * 1024 + 1],
    ] as const) {
      expect(() => resolveFinalGatherOptions({ [key]: bad })).toThrow();
      expect(() => resolveFinalGatherOptions({ [key]: NaN })).toThrow();
      expect(() => resolveFinalGatherOptions({ [key]: null })).toThrow();
      expect(() => resolveFinalGatherOptions({ [key]: '1' })).toThrow();
    }
    expect(() => resolveFinalGatherOptions(null as unknown as object)).toThrow('FINAL_GATHER_OPTIONS');
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
