import { describe, expect, it } from 'vitest';
import {
  DIFFUSE_GI_PROBE_BYTES, DIFFUSE_GI_PROBE_FLOATS, DIFFUSE_GI_STATE_FLOAT_OFFSET,
  evaluateGiIrradiance, giRayDirection, giShBasis,
  resolveDiffuseGiOptions, type DiffuseGiOptions,
} from '../src/diffuse-gi';
import { GI_DISTANCE_CELLS } from '../src/gi-visibility';

const grid: DiffuseGiOptions = { origin: [0, 0, 0], spacing: [0.5, 0.5, 0.5], counts: [2, 2, 2] };

describe('diffuse GI probe contract', () => {
  it('resolves bounded defaults and owns immutable copies of caller arrays', () => {
    const origin: [number, number, number] = [1, 2, 3];
    const spacing: [number, number, number] = [0.5, 1, 2];
    const counts: [number, number, number] = [2, 3, 4];
    const skyRadiance: [number, number, number] = [0.1, 0.2, 0.3];
    const value = resolveDiffuseGiOptions({ origin, spacing, counts, skyRadiance });
    origin[0] = 100; spacing[1] = 100; counts[2] = 30; skyRadiance[0] = 10;
    expect(value.origin).toEqual([1, 2, 3]);
    expect(value.spacing).toEqual([0.5, 1, 2]);
    expect(value.counts).toEqual([2, 3, 4]);
    expect(value.skyRadiance).toEqual([0.1, 0.2, 0.3]);
    expect(value.probeCount).toBe(24);
    expect(value.raysPerProbe).toBe(64);
    expect(value.rayCount).toBe(1536);
    expect(value.maxDistance).toBe(30);
    expect(value.normalBias).toBe(0.03);
    expect(value.multiBounce).toBe(false);
    expect(value.visibilityWeightThreshold).toBe(0.2);
    expect(value.maxBytes).toBe(32 * 1024 * 1024);
    expect(DIFFUSE_GI_PROBE_BYTES).toBe((9 + GI_DISTANCE_CELLS + 1) * 16);
    expect(DIFFUSE_GI_PROBE_FLOATS).toBe(DIFFUSE_GI_PROBE_BYTES / 4);
    expect(DIFFUSE_GI_STATE_FLOAT_OFFSET).toBe((9 + GI_DISTANCE_CELLS) * 4);
    expect([value, value.origin, value.spacing, value.counts, value.skyRadiance].every(Object.isFrozen)).toBe(true);
  });

  it('keeps millimetre grids valid by bounding the default normal bias', () => {
    const value = resolveDiffuseGiOptions({ ...grid, spacing: [0.001, 0.002, 0.003] });
    expect(value.normalBias).toBeCloseTo(0.00025, 12);
    expect(() => resolveDiffuseGiOptions({ ...grid, spacing: [0.001, 0.002, 0.003], normalBias: 0.03 })).toThrow();
  });

  it('accepts only boolean multi-bounce modes and freezes the resolved choice', () => {
    const enabled = resolveDiffuseGiOptions({ ...grid, multiBounce: true });
    const disabled = resolveDiffuseGiOptions({ ...grid, multiBounce: false });
    expect(enabled.multiBounce).toBe(true);
    expect(disabled.multiBounce).toBe(false);
    expect(Object.isFrozen(enabled)).toBe(true);
    for (const invalid of [null, 0, 1, 'true', NaN]) {
      expect(() => resolveDiffuseGiOptions({ ...grid, multiBounce: invalid as unknown as boolean }))
        .toThrow('DIFFUSE_GI_MULTI_BOUNCE');
    }
  });

  it('bounds the visibility confidence curve and permits an explicit disabled curve', () => {
    for (const threshold of [0, 0.2, 1]) {
      expect(resolveDiffuseGiOptions({ ...grid, visibilityWeightThreshold: threshold }).visibilityWeightThreshold).toBe(threshold);
    }
    for (const invalid of [-0.001, 1.001, NaN, Infinity, null, '0.2']) {
      expect(() => resolveDiffuseGiOptions({ ...grid, visibilityWeightThreshold: invalid as unknown as number }))
        .toThrow('DIFFUSE_GI_VISIBILITY_WEIGHT_THRESHOLD');
    }
  });

  it('rejects invalid grids, float32-collapsed neighbours, and excess work or storage', () => {
    expect(() => resolveDiffuseGiOptions({ ...grid, origin: [1_000_001, 0, 0] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, origin: [1_000_000, 0, 0], spacing: [0.001, 1, 1] }))
      .toThrow('DIFFUSE_GI_FLOAT32_GRID');
    expect(() => resolveDiffuseGiOptions({ ...grid, origin: [1_000_000, 0, 0], spacing: [0.04, 1, 1], counts: [3, 2, 2] }))
      .toThrow('DIFFUSE_GI_FLOAT32_GRID');
    expect(() => resolveDiffuseGiOptions({ ...grid, spacing: [0, 1, 1] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, spacing: [1001, 1, 1] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, counts: [1, 2, 2] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, counts: [33, 2, 2] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, counts: [8, 8, 9] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, counts: [8, 8, 8], raysPerProbe: 256 })).toThrow();
    expect(resolveDiffuseGiOptions({ ...grid, counts: [8, 8, 8], raysPerProbe: 128 }).rayCount).toBe(65_536);
    expect(() => resolveDiffuseGiOptions({ ...grid, raysPerProbe: 15 })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, raysPerProbe: 257 })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, raysPerProbe: 16.5 })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, maxDistance: 0.009 })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, maxDistance: Infinity })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, skyRadiance: [-1, 0, 0] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, skyRadiance: [0, 0, 1025] })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, skyRadiance: new Array(3) as [number, number, number] })).toThrow();
    const minimumBytes = 8 * DIFFUSE_GI_PROBE_BYTES;
    expect(() => resolveDiffuseGiOptions({ ...grid, maxBytes: minimumBytes - 1 }))
      .toThrow('DIFFUSE_GI_MEMORY_BUDGET');
    expect(resolveDiffuseGiOptions({ ...grid, maxBytes: minimumBytes }).maxBytes).toBe(minimumBytes);
    expect(() => resolveDiffuseGiOptions({ ...grid, maxBytes: 128 * 1024 * 1024 + 1 })).toThrow();
    expect(() => resolveDiffuseGiOptions({ ...grid, maxBytes: minimumBytes + 0.5 })).toThrow();
  });
});

describe('diffuse GI numerical reference', () => {
  it('generates deterministic unit Fibonacci rays with symmetric vertical distribution', () => {
    const directions = Array.from({ length: 64 }, (_, index) => giRayDirection(index, 64));
    expect(directions[0][1]).toBeCloseTo(1 - 1 / 64, 12);
    expect(directions[0][2]).toBeCloseTo(0, 12);
    expect(directions[0][0]).toBeGreaterThan(0);
    expect(directions.at(-1)![1]).toBeCloseTo(-1 + 1 / 64, 12);
    expect(directions.reduce((sum, ray) => sum + ray[1], 0)).toBeCloseTo(0, 12);
    for (const ray of directions) expect(Math.hypot(...ray)).toBeCloseTo(1, 12);
    expect(giRayDirection(17, 64)).toEqual(giRayDirection(17, 64));
    expect(() => giRayDirection(-1, 64)).toThrow();
    expect(() => giRayDirection(64, 64)).toThrow();
    expect(() => giRayDirection(0, 0)).toThrow();
  });

  it('returns the specified nine real SH terms in basis-major order', () => {
    const basis = giShBasis([0, 0, 1]);
    expect(basis).toHaveLength(9);
    expect(basis[0]).toBeCloseTo(0.28209479, 8);
    expect(basis[1]).toBe(0);
    expect(basis[2]).toBeCloseTo(0.48860251, 8);
    expect(basis[3]).toBe(0);
    expect(basis[6]).toBeCloseTo(0.63078314, 8);
    expect(giShBasis([2, 0, 0])).toEqual(giShBasis([1, 0, 0]));
    expect(() => giShBasis([0, 0, 0])).toThrow();
  });

  it('convolves constant radiance to pi times radiance for every normal', () => {
    const radiance = [0.5, 2, 8];
    const coeffs = new Float32Array(27);
    for (let channel = 0; channel < 3; channel++) coeffs[channel] = 4 * Math.PI * 0.28209479 * radiance[channel];
    for (const normal of [[0, 1, 0], [0, 0, -1], [1, 2, 3]] as const) {
      const actual = evaluateGiIrradiance(coeffs, normal);
      for (let channel = 0; channel < 3; channel++) {
        expect(actual[channel]).toBeCloseTo(Math.PI * radiance[channel], 4);
      }
    }
  });

  it('keeps first-order directional irradiance and clamps negative lobes', () => {
    const ambient = 0.5, vertical = 0.2;
    const coeffs = new Float32Array(27);
    coeffs[0] = 4 * Math.PI * 0.28209479 * ambient;
    coeffs[3] = 4 * Math.PI / 3 * 0.48860251 * vertical; // coefficient 1, red channel
    expect(evaluateGiIrradiance(coeffs, [0, 1, 0])[0])
      .toBeCloseTo(Math.PI * ambient + 2 * Math.PI / 3 * vertical, 5);
    expect(evaluateGiIrradiance(coeffs, [0, -1, 0])[0])
      .toBeCloseTo(Math.PI * ambient - 2 * Math.PI / 3 * vertical, 5);
    expect(evaluateGiIrradiance(coeffs, [1, 0, 0])[0]).toBeCloseTo(Math.PI * ambient, 5);
    coeffs[0] = -10;
    expect(evaluateGiIrradiance(coeffs, [1, 0, 0])[0]).toBe(0);
    expect(() => evaluateGiIrradiance(new Float32Array(26), [0, 1, 0])).toThrow();
    coeffs[0] = NaN;
    expect(() => evaluateGiIrradiance(coeffs, [0, 1, 0])).toThrow();
  });
});
