import { describe, expect, it } from 'vitest';
import { finalGatherCoverageWindow, finalGatherRasterRayBasis, invertFinalGatherProjection,
  planFinalGatherGrid, planFinalGatherSupplemental, resolveFinalGatherOptions } from '../src/final-gather';
import { lookAt, multiply, perspective } from '../src/math';
import { decodeFinalGatherDiagnostics } from '../src/final-gather-diagnostics';

describe('final gather CPU contract', () => {
  it('resolves immutable defaults and independent normal bias', () => {
    const defaults = resolveFinalGatherOptions();
    expect(defaults).toEqual({ pixelSpacing: 8, maxAnchors: 16_384, raysPerAnchor: 2,
      pathSegments: 1, maxHistory: 64, normalBias: 0.0001, maxBytes: 16 * 1024 * 1024,
      receiver: 'world-query', maxReceiverBytes: 128 * 1024 * 1024, maxSupplementalAnchors: 0,
      maxSupplementalWorkUnits: 0, supplementalHistoryFrames: 0,
      supplementalSampling: 'frame-hash', supplementalSamplingSeed: 0 });
    expect(Object.isFrozen(defaults)).toBe(true);
    expect(resolveFinalGatherOptions({ pixelSpacing: 1, normalBias: 0.01 }).normalBias).toBe(0.01);
    expect(resolveFinalGatherOptions({ pixelSpacing: 128, normalBias: 0 }).normalBias).toBe(0);
  });

  it('fits optional coverage receivers in spare capacity without an unbounded tile scan', () => {
    expect(planFinalGatherSupplemental(1280, 720, 7936, 8192))
      .toEqual({ spacing: 11, columns: 117, rows: 66, anchorCount: 7936, tileCount: 7722, permutationStride: 521 });
    const disabled = { spacing: 0, columns: 0, rows: 0, anchorCount: 0, tileCount: 0, permutationStride: 1 };
    expect(planFinalGatherSupplemental(1280, 720, 0, 8192)).toEqual(disabled);
    expect(planFinalGatherSupplemental(1280, 720, 7936, 0)).toEqual(disabled);
    expect(planFinalGatherSupplemental(32768, 32768, 65536, 16384)).toEqual(disabled);
    expect(planFinalGatherSupplemental(1, 1, 1, 1))
      .toEqual({ spacing: 1, columns: 1, rows: 1, anchorCount: 1, tileCount: 1, permutationStride: 521 });
    expect(() => resolveFinalGatherOptions({ maxSupplementalAnchors: 1 })).toThrow(/REQUIRES_RASTER/);
    expect(() => resolveFinalGatherOptions({ receiver: 'raster', maxSupplementalAnchors: 1 })).toThrow(/WORK_REQUIRED/);
    for (const requested of [-1, 16385, 0.5, NaN]) {
      expect(() => planFinalGatherSupplemental(128, 128, 100, requested)).toThrow();
      expect(() => resolveFinalGatherOptions({ receiver: 'raster', maxSupplementalAnchors: requested })).toThrow();
    }
  });

  it('uses all spare slots independently of tile count and permutes each tile exactly once', () => {
    const spare = planFinalGatherSupplemental(1280, 720, 7936, 8192);
    expect(spare.anchorCount).toBeGreaterThan(spare.tileCount);
    expect(spare.anchorCount).toBe(Math.min(7936, 8192));
    for (const count of [1, 2, 521, 1042, 5210, 15_630, 523, 1046, 16_384]) {
      const plan = planFinalGatherSupplemental(count, 1, count, count);
      expect(plan.tileCount).toBe(count);
      expect(plan.permutationStride).toBe(count % 521 ? 521 : 523);
      const visited = new Set(Array.from({ length: count }, (_, index) => (index * plan.permutationStride) % count));
      expect(visited.size).toBe(count);
      expect([...visited].every(tile => tile >= 0 && tile < count)).toBe(true);
    }
  });

  it('bounds optional supplemental history and requires active raster coverage', () => {
    const coverage = { receiver: 'raster' as const, maxSupplementalAnchors: 1,
      maxSupplementalWorkUnits: 1 };
    expect(resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: 64 }).supplementalHistoryFrames).toBe(64);
    expect(resolveFinalGatherOptions({ ...coverage }).supplementalHistoryFrames).toBe(0);
    expect(() => resolveFinalGatherOptions({ supplementalHistoryFrames: 1 }))
      .toThrow('FINAL_GATHER_SUPPLEMENTAL_HISTORY_REQUIRES_COVERAGE');
    for (const invalid of [-1, 0.5, 65, NaN, Infinity, null, '1', true]) {
      expect(() => resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: invalid as number }))
        .toThrow('FINAL_GATHER_SUPPLEMENTAL_HISTORY_FRAMES');
    }
    expect(() => resolveFinalGatherOptions({ ...coverage, receiver: 'world-query', supplementalHistoryFrames: 1 }))
      .toThrow('FINAL_GATHER_SUPPLEMENTAL_REQUIRES_RASTER');
    expect(() => resolveFinalGatherOptions({ ...coverage, maxSupplementalWorkUnits: 0, supplementalHistoryFrames: 1 }))
      .toThrow('FINAL_GATHER_SUPPLEMENTAL_WORK_REQUIRED');
    expect(resolveFinalGatherOptions({ maxBytes: 128 * 1024 * 1024 }).maxBytes).toBe(128 * 1024 * 1024);
  });

  it('opts into progressive supplemental sampling with its own history and uint32 seed', () => {
    const coverage = { receiver: 'raster' as const, maxSupplementalAnchors: 1,
      maxSupplementalWorkUnits: 1 };
    expect(resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: 64,
      maxHistory: 1, supplementalSampling: 'progressive-halton', supplementalSamplingSeed: 0xffff_ffff }))
      .toMatchObject({ supplementalHistoryFrames: 64, maxHistory: 1,
        supplementalSampling: 'progressive-halton', supplementalSamplingSeed: 0xffff_ffff });
    expect(resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: 1,
      supplementalSampling: 'progressive-halton' }).supplementalSamplingSeed).toBe(0);
    expect(resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: 1,
      supplementalSampling: 'frame-hash' }).supplementalSampling).toBe('frame-hash');
    expect(() => resolveFinalGatherOptions({ ...coverage, supplementalSampling: 'progressive-halton' }))
      .toThrow('FINAL_GATHER_PROGRESSIVE_REQUIRES_HISTORY');
    expect(() => resolveFinalGatherOptions({ ...coverage, supplementalSampling: 'progressive-halton',
      supplementalHistoryFrames: 0 })).toThrow('FINAL_GATHER_PROGRESSIVE_REQUIRES_HISTORY');
    for (const invalid of [null, 0, true, '', 'halton', 'FRAME-HASH']) {
      expect(() => resolveFinalGatherOptions({ supplementalSampling: invalid as 'frame-hash' }))
        .toThrow('FINAL_GATHER_SUPPLEMENTAL_SAMPLING');
    }
    for (const invalid of [-1, 0.5, 0x1_0000_0000, NaN, Infinity, null, '1', true]) {
      expect(() => resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: 1,
        supplementalSampling: 'progressive-halton', supplementalSamplingSeed: invalid as number }))
        .toThrow('FINAL_GATHER_SUPPLEMENTAL_SAMPLING_SEED');
    }
    expect(() => resolveFinalGatherOptions({ supplementalSamplingSeed: 1 }))
      .toThrow('FINAL_GATHER_SAMPLING_SEED_REQUIRES_PROGRESSIVE');
    expect(() => resolveFinalGatherOptions({ ...coverage, supplementalHistoryFrames: 1,
      supplementalSampling: 'frame-hash', supplementalSamplingSeed: 1 }))
      .toThrow('FINAL_GATHER_SAMPLING_SEED_REQUIRES_PROGRESSIVE');
    const grid = planFinalGatherGrid(320, 180, {}, 1840, 1840);
    expect(planFinalGatherGrid(320, 180, { ...coverage, supplementalHistoryFrames: 1,
      supplementalSampling: 'progressive-halton', supplementalSamplingSeed: 7 }, 1840, 1840)).toEqual(grid);
  });

  it('rotates a bounded candidate update window across a fixed queue', () => {
    expect(finalGatherCoverageWindow(0, 0, 0)).toEqual({ start: 0, count: 0 });
    expect(finalGatherCoverageWindow(0, 16_384, 0xffff_ffff)).toEqual({ start: 0, count: 0 });
    expect(finalGatherCoverageWindow(7, 12, 1)).toEqual({ start: 5, count: 7 });
    expect(finalGatherCoverageWindow(10, 4, 2)).toEqual({ start: 8, count: 4 });
    expect(finalGatherCoverageWindow(13, 5, 0xffff_ffff)).toEqual({
      start: ((0xffff_ffff % 13) * 5) % 13, count: 5 });
    expect(finalGatherCoverageWindow(131_072, 16_384, 0xffff_ffff).start)
      .toBe(((0xffff_ffff % 131_072) * 16_384) % 131_072);
    const total = 17, budget = 5, seen = new Set<number>();
    // A fixed queue is covered in ceil(total / budget) consecutive frames.
    for (let frame = 0; frame < Math.ceil(total / budget); frame++) {
      const window = finalGatherCoverageWindow(total, budget, frame);
      expect(window.count).toBeLessThanOrEqual(budget);
      for (let ordinal = 0; ordinal < total; ordinal++) {
        if ((ordinal + total - window.start) % total < window.count) seen.add(ordinal);
      }
    }
    expect(seen.size).toBe(total);
    expect(finalGatherCoverageWindow(17, 0, 12)).toEqual({ start: 0, count: 0 });
    for (const [totalCandidates, updateBudget, frame] of [
      [-1, 1, 0], [131_073, 1, 0], [1.5, 1, 0], [1, -1, 0], [1, 16_385, 0],
      [1, 1.5, 0], [1, 1, -1], [1, 1, 0x1_0000_0000], [1, 1, 0.5],
      [null, 1, 0], [1, null, 0], [1, 1, null],
    ] as const) {
      expect(() => finalGatherCoverageWindow(totalCandidates as number, updateBudget as number, frame as number)).toThrow();
    }
  });

  it('decodes up to eight supplemental ranks in all nine neighboring tiles', () => {
    const words = new Uint32Array(8), floats = new Float32Array(words.buffer), pixel = [{ x: 0, y: 0 }];
    words.fill(0x80808080, 4); floats.set([1, 0.2, 0.3], 1);
    for (const count of [1, 8, 9, 10, 71, 72]) {
      words[0] = 6 | (count << 24);
      expect(decodeFinalGatherDiagnostics(words, pixel)[0]).toMatchObject({ branch: 'supplemental', supplementalAccepted: count });
    }
    for (const header of [6, 6 | (73 << 24), 6 | (255 << 24), 1 | (72 << 24)]) {
      words[0] = header;
      expect(() => decodeFinalGatherDiagnostics(words, pixel)).toThrow('FINAL_GATHER_DIAGNOSTIC_HEADER');
    }
  });

  it('validates each option at its hard boundary', () => {
    expect(resolveFinalGatherOptions({ maxAnchors: 65_536, raysPerAnchor: 16,
      maxHistory: 256, maxBytes: 128 * 1024 * 1024 }).maxAnchors).toBe(65_536);
    for (const [key, bad] of [
      ['pixelSpacing', 0], ['pixelSpacing', 129], ['maxAnchors', 0], ['maxAnchors', 65_537],
      ['raysPerAnchor', 0], ['raysPerAnchor', 17], ['maxHistory', 0], ['maxHistory', 257],
      ['pathSegments', 0], ['pathSegments', 5], ['pathSegments', 1.5],
      ['normalBias', -1], ['normalBias', 0.011], ['maxBytes', 0], ['maxBytes', 128 * 1024 * 1024 + 1],
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
