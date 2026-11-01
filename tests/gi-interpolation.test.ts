import { describe, expect, it } from 'vitest';
import { GI_DISTANCE_CELLS } from '../src/gi-visibility';
import { giReplayAxisVisibility, giReplayNeighborVisibility, replayGiSurfaceIrradiance,
  type GiReplayGrid, type GiReplayProbe } from '../src/gi-interpolation';

const OPEN: [number, number] = [50, 2500];
const OPEN_AXIAL = Array.from({ length: 6 }, () => OPEN).flat();

/** Constant incoming radiance: E(n) = pi * radiance for every normal. */
function probe(radiance: number, moment: [number, number] = OPEN, axial: number[] = OPEN_AXIAL): GiReplayProbe {
  const c0 = radiance * 4 * Math.PI * 0.28209479;
  return { coefficients: Array.from({ length: 9 }, (_, i) => i ? [0, 0, 0] : [c0, c0, c0]),
    moments: Array.from({ length: GI_DISTANCE_CELLS }, () => [moment[0], moment[1], 1, 0]),
    maxRadiance: [10 * radiance, 10 * radiance, 10 * radiance], valid: true, axial };
}

const base: Omit<GiReplayGrid, 'visibilityRefinement'> = { origin: [0, 0, 0], spacing: [1, 1, 1], counts: [2, 2, 2],
  normalBias: 0, visibilityWeightThreshold: 0.2, visibilityFilter: 'moment-bilinear' };
const refined: GiReplayGrid = { ...base, visibilityRefinement: 'grid-axis-neighbor' };
const index = (x: number, y: number, z: number) => x + 2 * (y + 2 * z);

describe('grid-axis-neighbor probe interpolation replay', () => {
  it('matches the default interpolation when no axial or mutual test rejects a corner', () => {
    const field = Array.from({ length: 8 }, (_, i) => probe(1 + i));
    const position = [0.3, 0.6, 0.45], normal = [0, 1, 0];
    const legacy = replayGiSurfaceIrradiance(field, base, position, normal, normal);
    const result = replayGiSurfaceIrradiance(field, refined, position, normal, normal);
    for (let c = 0; c < 3; c++) expect(result.irradiance[c]).toBeCloseTo(legacy.irradiance[c], 12);
    expect(result.total).toBeCloseTo(legacy.total, 12);
  });

  it('rejects corners behind a thin wall that the octahedral moments miss', () => {
    // Wall at x = 0.5: x = 0 probes see it at +0.5 along +x, x = 1 probes at 0.5 along -x.
    const field = Array.from({ length: 8 }, (_, i) => {
      const x = i & 1, axial = [...OPEN_AXIAL];
      axial.splice(x ? 2 : 0, 2, 0.5, 0.25);
      return probe(x ? 10 : 1, OPEN, axial);
    });
    const position = [0.2, 0.4, 0.6], normal = [0, 1, 0];
    const legacy = replayGiSurfaceIrradiance(field, base, position, normal, normal);
    const result = replayGiSurfaceIrradiance(field, refined, position, normal, normal);
    expect(legacy.irradiance[0]).toBeGreaterThan(1.5 * Math.PI);
    expect(result.irradiance[0]).toBeCloseTo(Math.PI, 6);
    for (const corner of result.corners) {
      if (corner.coordinate[0] === 1) expect(corner.chebyshev).toBeLessThan(0.002);
      else expect(corner.chebyshev).toBe(1);
    }
    expect(giReplayAxisVisibility(field[index(1, 0, 0)], [-0.4, 0, 0])).toBe(1);
    expect(giReplayAxisVisibility(field[index(1, 0, 0)], [-0.8, 0, 0])).toBeLessThan(0.002);
  });

  it('drops corners that lack mutual neighbour visibility with the strongest corner', () => {
    // The x = y = z = 0 probe sees a closed 0.5 m shell (the receiver is inside it); the
    // other probes' maps miss the shell, so only the mutual test can reject them.
    const field = Array.from({ length: 8 }, (_, i) => i ? probe(10) : probe(1, [0.5, 0.25]));
    const position = [0.2, 0.2, 0.2], facing = [-1, -1, -1].map(value => value / Math.sqrt(3));
    const legacy = replayGiSurfaceIrradiance(field, base, position, facing, facing);
    const result = replayGiSurfaceIrradiance(field, refined, position, facing, facing);
    expect(legacy.irradiance[0]).toBeGreaterThan(1.2 * Math.PI);
    expect(result.irradiance[0]).toBeCloseTo(Math.PI, 6);
    const anchor = result.corners.find(corner => corner.index === 0)!;
    expect(anchor.visibility).toBe(1);
    for (const corner of result.corners) if (corner.index) expect(corner.visibility).toBeLessThan(0.001);
    expect(giReplayNeighborVisibility(field[0], [0, 0, 0], base.spacing, base.visibilityFilter)).toBe(1);
    expect(giReplayNeighborVisibility(field[0], [1, 0, 0], base.spacing, base.visibilityFilter)).toBeLessThan(0.001);
  });

  it('requires axial moments for every refined probe', () => {
    const field = Array.from({ length: 8 }, () => ({ ...probe(1), axial: undefined }));
    expect(() => replayGiSurfaceIrradiance(field, refined, [0.5, 0.5, 0.5], [0, 1, 0], [0, 1, 0])).toThrow('GI_REPLAY_AXIAL');
    expect(() => replayGiSurfaceIrradiance(field, base, [0.5, 0.5, 0.5], [0, 1, 0], [0, 1, 0])).not.toThrow();
  });
});
