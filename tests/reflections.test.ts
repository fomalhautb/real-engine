import { describe, expect, it } from 'vitest';
import { planReflectionGrid, resolveReflectionsOptions, sampleReflectionGgx,
  type ReflectionsOptions } from '../src/reflections';

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const length = (v: readonly number[]) => Math.hypot(...v);

describe('reflection CPU contracts', () => {
  it('resolves bounded defaults and snapshots the sky color', () => {
    const defaults = resolveReflectionsOptions();
    expect(defaults).toEqual({ pixelSpacing: 4, maxAnchors: 16_384, samplesPerAnchor: 4,
      maxDistance: 50, maxGatherDistance: 0.5, planeTolerance: 0.001,
      normalBias: 0.0001, fogSteps: 32, maxBytes: 32 * 1024 * 1024,
      skyRadiance: [0, 0, 0], textureLod: 0 });
    expect(Object.isFrozen(defaults)).toBe(true);
    expect(Object.isFrozen(defaults.skyRadiance)).toBe(true);
    const sky: [number, number, number] = [0.25, 2, 1024];
    const resolved = resolveReflectionsOptions({ skyRadiance: sky, maxDistance: 0.01,
      maxGatherDistance: 0.001, planeTolerance: 0.000001,
      normalBias: 0, fogSteps: 128, textureLod: 16 });
    sky[0] = 100;
    expect(resolved.skyRadiance).toEqual([0.25, 2, 1024]);
    expect(Object.isFrozen(resolved.skyRadiance)).toBe(true);
    expect(resolved.maxGatherDistance).toBe(0.001);
    expect(resolved.planeTolerance).toBe(0.000001);
    expect(resolved.fogSteps).toBe(128);
  });

  it('validates every option, including finite RGB and exact integer boundaries', () => {
    expect(resolveReflectionsOptions({ pixelSpacing: 128, maxAnchors: 65_536,
      samplesPerAnchor: 16, maxDistance: 1000, maxGatherDistance: 100,
      planeTolerance: 0.1, normalBias: 0.01, fogSteps: 1,
      maxBytes: 128 * 1024 * 1024, skyRadiance: [1024, 0, 0], textureLod: 16 }).pixelSpacing).toBe(128);
    for (const [key, bad] of [
      ['pixelSpacing', 0], ['pixelSpacing', 129], ['pixelSpacing', 1.5],
      ['maxAnchors', 0], ['maxAnchors', 65_537], ['samplesPerAnchor', 0], ['samplesPerAnchor', 17],
      ['maxDistance', 0.009], ['maxDistance', 1001], ['normalBias', -0.1], ['normalBias', 0.011],
      ['maxGatherDistance', 0.0009], ['maxGatherDistance', 100.001],
      ['planeTolerance', 0.0000009], ['planeTolerance', 0.100001],
      ['fogSteps', 0], ['fogSteps', 129], ['fogSteps', 1.5],
      ['maxBytes', 0], ['maxBytes', 128 * 1024 * 1024 + 1], ['maxBytes', 1.5],
      ['textureLod', -1], ['textureLod', 16.1],
    ] as const) {
      for (const value of [bad, NaN, Infinity, null, '1']) {
        expect(() => resolveReflectionsOptions({ [key]: value } as ReflectionsOptions)).toThrow();
      }
    }
    for (const sky of [[0, 0], [0, 0, 0, 1], [-1, 0, 0], [0, 1025, 0],
      [NaN, 0, 0], [0, Infinity, 0], null, 'blue']) {
      expect(() => resolveReflectionsOptions({ skyRadiance: sky } as unknown as ReflectionsOptions)).toThrow();
    }
    for (const options of [null, 1, [], 'fog']) {
      expect(() => resolveReflectionsOptions(options as unknown as ReflectionsOptions)).toThrow('REFLECTIONS_OPTIONS');
    }
  });

  it('keeps the requested spacing at exact anchor, ray, and work budgets', () => {
    expect(planReflectionGrid(320, 180, {}, 14_400, 14_400)).toEqual({
      spacing: 4, columns: 80, rows: 45, anchorCount: 3600, rayCount: 14_400,
    });
    expect(planReflectionGrid(1, 1, { pixelSpacing: 128 }, 4, 4)).toEqual({
      spacing: 128, columns: 1, rows: 1, anchorCount: 1, rayCount: 4,
    });
  });

  it('finds the smallest legal integer spacing under each shared budget', () => {
    expect(planReflectionGrid(320, 180, {}, 14_399, 14_400).spacing).toBe(5);
    expect(planReflectionGrid(320, 180, {}, 14_400, 14_399).spacing).toBe(5);
    const options = resolveReflectionsOptions({ pixelSpacing: 1, samplesPerAnchor: 16, maxAnchors: 73,
      maxGatherDistance: 2, planeTolerance: 0.01, fogSteps: 64 });
    const plan = planReflectionGrid(16_384, 16_384, options, 16 * 90, 16 * 80);
    expect(plan.anchorCount).toBeLessThanOrEqual(73);
    expect(plan.rayCount).toBeLessThanOrEqual(16 * 80);
    expect(Math.ceil(16_384 / (plan.spacing - 1)) ** 2).toBeGreaterThan(73);
    expect(() => planReflectionGrid(16, 8, { samplesPerAnchor: 4 }, 3, 100))
      .toThrow('REFLECTIONS_QUERY_RAY_BUDGET');
    expect(() => planReflectionGrid(16, 8, { samplesPerAnchor: 4 }, 100, 3))
      .toThrow('REFLECTIONS_QUERY_RAY_BUDGET');
    for (const bad of [0, 0.5, 16_385, NaN, Infinity]) {
      expect(() => planReflectionGrid(bad, 1, {}, 4, 4)).toThrow();
      expect(() => planReflectionGrid(1, bad, {}, 4, 4)).toThrow();
    }
    for (const bad of [0, 1.5, 0x1_0000_0000, NaN, Infinity]) {
      expect(() => planReflectionGrid(1, 1, {}, bad, 4)).toThrow();
      expect(() => planReflectionGrid(1, 1, {}, 4, bad)).toThrow();
    }
  });

  it('samples normalized GGX half-vectors and reflected rays with the NDF solid-angle PDF', () => {
    const roughness = 0.5, alpha2 = roughness ** 4;
    for (const [u1, u2] of [[0.0, 0], [0.25, 0.5], [0.6, 0.8], [0.99, 0.97]]) {
      const sample = sampleReflectionGgx([0, 0, 1], [0, 0, 1], roughness, u1, u2);
      const noH = sample.halfVector[2], voH = noH;
      const expectedCos = Math.sqrt((1 - u2) / (1 + (alpha2 - 1) * u2));
      expect(noH).toBeCloseTo(expectedCos, 12);
      expect(length(sample.halfVector)).toBeCloseTo(1, 12);
      expect(length(sample.direction)).toBeCloseTo(1, 12);
      expect(sample.direction[2]).toBeCloseTo(2 * noH * noH - 1, 12);
      if (sample.direction[2] > 0) {
        const d = alpha2 / (Math.PI * ((1 - noH * noH) + noH * noH * alpha2) ** 2);
        expect(sample.pdf).toBeCloseTo(d * noH / (4 * voH), 12);
        const noL = sample.direction[2], k = (roughness + 1) ** 2 / 8;
        const gL = noL / (noL * (1 - k) + k);
        expect(sample.weight).toBeCloseTo(gL, 12); // NoV=1, Gv=1, VoH=NoH.
      } else {
        expect(sample.pdf).toBe(0);
        expect(sample.weight).toBe(0);
      }
    }
    // Independent NDF normalization integral over the half-vector hemisphere.
    const n = 20_000;
    let integral = 0;
    for (let i = 0; i < n; i++) {
      const noH = (i + 0.5) / n;
      const d = alpha2 / (Math.PI * ((1 - noH * noH) + noH * noH * alpha2) ** 2);
      integral += 2 * Math.PI * d * noH / n;
    }
    expect(integral).toBeCloseTo(1, 5);
  });

  it('matches the surface BRDF at grazing view and light angles', () => {
    const normal: [number, number, number] = [0, 0, 1];
    const cases = [
      { view: [Math.sqrt(1 - 1e-8), 0, 1e-4] as [number, number, number],
        roughness: 0.5, u1: 0, u2: 0, grazing: 'view' },
      { view: [0, 0, 1] as [number, number, number],
        roughness: 1, u1: 0.25, u2: (1 - 1e-5) / 2, grazing: 'light' },
    ];
    for (const { view, roughness, u1, u2, grazing } of cases) {
      const sample = sampleReflectionGgx(normal, view, roughness, u1, u2);
      const noV = dot(normal, view), noL = dot(normal, sample.direction);
      const noH = dot(normal, sample.halfVector), voH = dot(view, sample.halfVector);
      const shadingNoV = Math.max(noV, 0.001);
      const alphaSquared = roughness ** 4;
      const distribution = alphaSquared / Math.max(
        Math.PI * ((1 - noH * noH) + noH * noH * alphaSquared) ** 2, 1e-20);
      const k = (roughness + 1) ** 2 / 8;
      const geometry = shadingNoV / (shadingNoV * (1 - k) + k)
        * noL / (noL * (1 - k) + k);
      // Reference is the complete shared direct specular BRDF (F=1) times
      // NoL, divided by the independently evaluated NDF solid-angle PDF.
      const specularBrdf = distribution * geometry / Math.max(4 * shadingNoV * noL, 0.001);
      const pdf = distribution * noH / (4 * voH);
      const expectedWeight = specularBrdf * noL / pdf;
      expect(Math.abs(sample.pdf / pdf - 1)).toBeLessThan(1e-12);
      expect(Math.abs(sample.weight / expectedWeight - 1)).toBeLessThan(1e-10);
      expect(4 * shadingNoV * noL).toBeLessThan(0.001);
      if (grazing === 'view') expect(noV).toBeCloseTo(1e-4, 12);
      else expect(noL).toBeCloseTo(1e-5, 12);
    }
  });

  it('matches the GPU azimuth/polar convention for an indexed four-sample stratum', () => {
    // sampleIndex=1: reverseBits(1)/2^32 = 1/2, (1 + 1/2)/4 = 3/8.
    const roughness = 0.45, polar = 3 / 8;
    const sample = sampleReflectionGgx([0, 1, 0], [0, 1, 0], roughness, 0.5, polar);
    const alpha2 = roughness ** 4;
    const cosine = Math.sqrt((1 - polar) / (1 + (alpha2 - 1) * polar));
    const sine = Math.sqrt(1 - cosine * cosine);
    // GPU's Y-up basis chooses helper X, tangent +Z; phi=pi points -Z.
    expect(sample.halfVector[0]).toBeCloseTo(0, 12);
    expect(sample.halfVector[1]).toBeCloseTo(cosine, 12);
    expect(sample.halfVector[2]).toBeCloseTo(-sine, 12);
    expect(sample.direction[1]).toBeCloseTo(2 * cosine * cosine - 1, 12);
  });

  it('handles tilted frames, rejected hemisphere rays, the mirror limit and malformed inputs', () => {
    const n: [number, number, number] = [1, 2, 3], v: [number, number, number] = [1, 1, 3];
    const sample = sampleReflectionGgx(n, v, 0.45, 0.3, 0.3);
    expect(length(sample.halfVector)).toBeCloseTo(1, 12);
    expect(length(sample.direction)).toBeCloseTo(1, 12);
    expect(dot(n, sample.halfVector)).toBeGreaterThan(0);
    expect(sample.pdf).toBeGreaterThan(0);
    expect(sample.weight).toBeGreaterThan(0);
    const rejected = sampleReflectionGgx([0, 0, 1], [0, 0, 1], 1, 0.25, 0.9);
    expect(rejected.direction[2]).toBeLessThan(0);
    expect(rejected.pdf).toBe(0);
    expect(rejected.weight).toBe(0);
    const mirror = sampleReflectionGgx([0, 0, 1], [0, 0, 1], 0, 0.2, 0.7);
    expect(mirror.direction).toEqual([0, 0, 1]);
    expect(mirror.pdf).toBe(0); // A Dirac mirror has no finite solid-angle density.
    for (const input of [
      [[0, 0, 0], [0, 0, 1], 0.5, 0.2, 0.3],
      [[0, 0, 1], [0, 0, -1], 0.5, 0.2, 0.3],
      [[0, NaN, 1], [0, 0, 1], 0.5, 0.2, 0.3],
      [[1.5e308, 1.5e308, 1.5e308], [0, 0, 1], 0.5, 0.2, 0.3],
      [[0, 0, 1], [0, 0, 1], -0.01, 0.2, 0.3],
      [[0, 0, 1], [0, 0, 1], Infinity, 0.2, 0.3],
      [[0, 0, 1], [0, 0, 1], 0.5, 1, 0.3],
      [[0, 0, 1], [0, 0, 1], 0.5, 0.2, NaN],
    ] as const) expect(() => sampleReflectionGgx(input[0], input[1], input[2], input[3], input[4])).toThrow();
  });
});
