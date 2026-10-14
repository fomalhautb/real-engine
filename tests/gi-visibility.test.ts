import { describe, expect, it } from 'vitest';
import {
  GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION, buildGiDistanceWeights,
  octaDecode, octaEncode, wrapGiDistanceTexel,
} from '../src/gi-visibility';

type Vec3 = readonly [number, number, number];

function fibonacciDirections(count: number): Vec3[] {
  return Array.from({ length: count }, (_, index) => {
    const y = 1 - 2 * (index + 0.5) / count;
    const radius = Math.sqrt(1 - y * y);
    const angle = index * 2.399963229728653;
    return [Math.cos(angle) * radius, y, Math.sin(angle) * radius];
  });
}

function field(direction: Vec3): number {
  return 2 + 0.3 * direction[0] + 0.5 * direction[1] + 0.7 * direction[2];
}

function samplePaddedMap(values: Float32Array, u: number, v: number): number {
  const x = u * GI_DISTANCE_RESOLUTION - 0.5;
  const y = v * GI_DISTANCE_RESOLUTION - 0.5;
  const left = Math.floor(x), bottom = Math.floor(y);
  const fx = x - left, fy = y - bottom;
  const at = (dx: number, dy: number) => values[wrapGiDistanceTexel(left + dx, bottom + dy)];
  return (1 - fy) * ((1 - fx) * at(0, 0) + fx * at(1, 0))
    + fy * ((1 - fx) * at(0, 1) + fx * at(1, 1));
}

describe('octahedral GI visibility', () => {
  it('round-trips axes and deterministic directions in both hemispheres', () => {
    const axes: Vec3[] = [
      [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ];
    let state = 0x1234abcd;
    const randomDirections: Vec3[] = Array.from({ length: 200 }, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const z = 2 * state / 0x1_0000_0000 - 1;
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const angle = 2 * Math.PI * state / 0x1_0000_0000;
      const radius = Math.sqrt(1 - z * z);
      return [radius * Math.cos(angle), radius * Math.sin(angle), z];
    });
    for (const direction of [...axes, ...randomDirections]) {
      const uv = octaEncode(direction);
      expect(uv[0]).toBeGreaterThanOrEqual(0);
      expect(uv[0]).toBeLessThanOrEqual(1);
      expect(uv[1]).toBeGreaterThanOrEqual(0);
      expect(uv[1]).toBeLessThanOrEqual(1);
      const decoded = octaDecode(uv);
      expect(Math.hypot(...decoded)).toBeCloseTo(1, 12);
      expect(Math.hypot(...decoded.map((value, axis) => value - direction[axis]))).toBeLessThan(1e-12);
    }
    expect(octaEncode([0, 0, -1])).toEqual([1, 1]);
    expect(octaDecode([0.5, 0.5])).toEqual([0, 0, 1]);
  });

  it('reflects edge padding and keeps a bilinear directional field continuous at octahedral seams', () => {
    expect(GI_DISTANCE_RESOLUTION).toBe(8);
    expect(GI_DISTANCE_CELLS).toBe(64);
    expect(wrapGiDistanceTexel(-1, 2)).toBe(5 * 8);
    expect(wrapGiDistanceTexel(8, 2)).toBe(5 * 8 + 7);
    expect(wrapGiDistanceTexel(3, -1)).toBe(4);
    expect(wrapGiDistanceTexel(3, 8)).toBe(7 * 8 + 4);
    expect(wrapGiDistanceTexel(-1, -1)).toBe(63);
    expect(wrapGiDistanceTexel(8, 8)).toBe(0);

    const texels = Float32Array.from({ length: GI_DISTANCE_CELLS }, (_, cell) => {
      const x = cell % 8, y = Math.floor(cell / 8);
      return field(octaDecode([(x + 0.5) / 8, (y + 0.5) / 8]));
    });
    for (const t of [0.13, 0.29, 0.41]) {
      const left = samplePaddedMap(texels, 0, t);
      const leftFolded = samplePaddedMap(texels, 0, 1 - t);
      const right = samplePaddedMap(texels, 1, t);
      const rightFolded = samplePaddedMap(texels, 1, 1 - t);
      const bottom = samplePaddedMap(texels, t, 0);
      const bottomFolded = samplePaddedMap(texels, 1 - t, 0);
      expect(left).toBeCloseTo(leftFolded, 6);
      expect(right).toBeCloseTo(rightFolded, 6);
      expect(bottom).toBeCloseTo(bottomFolded, 6);
      // The 8x8 discretization approximates this smooth field; seam pairs
      // above must match exactly, while interpolation has finite grid error.
      expect(Math.abs(left - field(octaDecode([0, t])))).toBeLessThan(0.12);
      expect(Math.abs(right - field(octaDecode([1, t])))).toBeLessThan(0.12);
      expect(Math.abs(bottom - field(octaDecode([t, 0])))).toBeLessThan(0.12);
    }
  });

  it('produces finite nonnegative normalized rows for bounded Fibonacci ray sets', () => {
    for (const count of [16, 64, 256]) {
      const directions = fibonacciDirections(count);
      const weights = buildGiDistanceWeights(directions);
      expect(weights).toBeInstanceOf(Float32Array);
      expect(weights.length).toBe(GI_DISTANCE_CELLS * count);
      expect(buildGiDistanceWeights(directions)).toEqual(weights);
      for (let cell = 0; cell < GI_DISTANCE_CELLS; cell++) {
        let sum = 0;
        for (let ray = 0; ray < count; ray++) {
          const weight = weights[cell * count + ray];
          expect(Number.isFinite(weight)).toBe(true);
          expect(weight).toBeGreaterThanOrEqual(0);
          sum += weight;
        }
        expect(sum).toBeCloseTo(1, 5);
      }
    }
    // Even uneven but valid input has a well-defined nearest-ray fallback.
    const sameDirection = Array.from({ length: 16 }, (): Vec3 => [0, 0, 1]);
    const fallback = buildGiDistanceWeights(sameDirection);
    for (let cell = 0; cell < GI_DISTANCE_CELLS; cell++) {
      const row = fallback.subarray(cell * 16, (cell + 1) * 16);
      expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 5);
    }
  });

  it('normalizes a narrow lobe before Float32 storage can underflow it', () => {
    const x = 3, y = 3;
    const normal = octaDecode([(x + 0.5) / 8, (y + 0.5) / 8]);
    const tangentLength = Math.hypot(normal[0], normal[1]);
    const tangent: Vec3 = [-normal[1] / tangentLength, normal[0] / tangentLength, 0];
    const cosine = 0.001;
    const tangentScale = Math.sqrt(1 - cosine * cosine);
    const almostPerpendicular: Vec3 = [
      tangentScale * tangent[0] + cosine * normal[0],
      tangentScale * tangent[1] + cosine * normal[1],
      tangentScale * tangent[2] + cosine * normal[2],
    ];
    expect(normal.reduce((sum, value, axis) => sum + value * almostPerpendicular[axis], 0))
      .toBeCloseTo(cosine, 12);
    expect(cosine ** 32).toBeLessThan(2 ** -149); // underflows if stored as Float32 first

    const weights = buildGiDistanceWeights(Array.from({ length: 16 }, () => almostPerpendicular));
    const row = weights.subarray((y * 8 + x) * 16, (y * 8 + x + 1) * 16);
    expect(row.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 6);
    for (const weight of row) expect(weight).toBeCloseTo(1 / 16, 6);
  });

  it('rejects malformed directions, UVs, texel coordinates, and ray counts', () => {
    expect(() => octaEncode([0, 0, 0])).toThrow('GI_VISIBILITY_DIRECTION');
    expect(() => octaEncode([2, 0, 0])).toThrow('GI_VISIBILITY_DIRECTION');
    expect(() => octaEncode([NaN, 0, 1])).toThrow('GI_VISIBILITY_DIRECTION');
    expect(() => octaDecode([-0.001, 0.5])).toThrow('GI_VISIBILITY_UV');
    expect(() => octaDecode([0.5, Infinity])).toThrow('GI_VISIBILITY_UV');
    expect(() => octaDecode(new Array(2) as [number, number])).toThrow('GI_VISIBILITY_UV');
    expect(() => wrapGiDistanceTexel(-2, 0)).toThrow('GI_VISIBILITY_TEXEL');
    expect(() => wrapGiDistanceTexel(0, 9)).toThrow('GI_VISIBILITY_TEXEL');
    expect(() => wrapGiDistanceTexel(1.5, 0)).toThrow('GI_VISIBILITY_TEXEL');
    expect(() => buildGiDistanceWeights(fibonacciDirections(15))).toThrow('GI_VISIBILITY_RAYS');
    expect(() => buildGiDistanceWeights(fibonacciDirections(257))).toThrow('GI_VISIBILITY_RAYS');
    expect(() => buildGiDistanceWeights([...fibonacciDirections(15), [0, 0, Infinity]]))
      .toThrow('GI_VISIBILITY_DIRECTION');
    expect(() => buildGiDistanceWeights([...fibonacciDirections(15), [0, 0, 0]]))
      .toThrow('GI_VISIBILITY_DIRECTION');
    expect(() => buildGiDistanceWeights(new Array(16) as Vec3[]))
      .toThrow('GI_VISIBILITY_DIRECTION');
  });
});
