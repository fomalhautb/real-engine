import { describe, expect, it } from 'vitest';
import { giShBasis } from '../src/diffuse-gi';
import { henyeyGreenstein } from '../src/participating-media';
import { FOG_OCCLUDER_REGION_BYTES, FOG_OCCLUDER_REGION_CAPACITY, decodeFogHistoryState, fogAirPhaseRadiance,
  fogFixedBytes, fogOccluderRegions, resolveFogAirLighting, resolveVolumetricFogOptions, selectFogGrid } from '../src/volumetric-fog';

type Vec3 = [number, number, number];

/** Midpoint quadrature of an incident radiance against HG, for directions omega and view v. */
function integratePhase(radiance: (omega: Vec3) => Vec3, view: Vec3, g: number, steps = 360): Vec3 {
  const result: Vec3 = [0, 0, 0];
  const dTheta = Math.PI / steps, dPhi = 2 * Math.PI / (2 * steps);
  for (let i = 0; i < steps; i++) {
    const theta = (i + 0.5) * dTheta, sine = Math.sin(theta);
    for (let j = 0; j < 2 * steps; j++) {
      const phi = (j + 0.5) * dPhi;
      const omega: Vec3 = [sine * Math.cos(phi), sine * Math.sin(phi), Math.cos(theta)];
      // Incident light arrives from omega; the camera looks along v, so the
      // scattering cosine between propagation (-omega) and outgoing (-v) is omega.v.
      const weight = henyeyGreenstein(omega[0] * view[0] + omega[1] * view[1] + omega[2] * view[2], g) * sine * dTheta * dPhi;
      const value = radiance(omega);
      for (let c = 0; c < 3; c++) result[c] += value[c] * weight;
    }
  }
  return result;
}

describe('fog air lighting', () => {
  it('resolves bounded air-lighting controls and leaves unconfigured fog unchanged', () => {
    expect('airLighting' in resolveVolumetricFogOptions()).toBe(false);
    expect('airLighting' in resolveVolumetricFogOptions({ airLighting: false })).toBe(false);
    const air = resolveVolumetricFogOptions({ airLighting: {} }).airLighting!;
    expect(air).toEqual({ gi: true, outsideSkyVisibility: 0 });
    expect(Object.isFrozen(air)).toBe(true);
    expect(resolveVolumetricFogOptions({ airLighting: { gi: false, outsideSkyVisibility: 0.25 } }).airLighting)
      .toEqual({ gi: false, outsideSkyVisibility: 0.25 });
    expect(resolveFogAirLighting({ outsideSkyVisibility: 1 }, { gi: false, outsideSkyVisibility: 0 }))
      .toEqual({ gi: false, outsideSkyVisibility: 1 });
    for (const invalid of [{ gi: 1 }, { outsideSkyVisibility: -0.01 }, { outsideSkyVisibility: 1.01 },
      { outsideSkyVisibility: Number.NaN }]) {
      expect(() => resolveVolumetricFogOptions({ airLighting: invalid as never })).toThrow();
    }
    expect(() => resolveVolumetricFogOptions({ airLighting: [] as never })).toThrow(/airLighting/);
    expect(() => resolveFogAirLighting(null as never)).toThrow(/airLighting/);
  });

  it('matches numerical HG integration of band-limited incident radiance in every anisotropy', () => {
    // Arbitrary L2 radiance per channel, including negative higher-band coefficients.
    const coefficients = [
      2.0, 1.0, 0.5, 0.3, -0.2, 0.1, -0.4, 0.25, 0.05, 0.6, 0.1, -0.3,
      0.2, 0.05, 0.1, -0.1, 0.3, 0.2, 0.15, -0.05, 0.12, 0.08, 0.02, -0.06, -0.2, 0.1, 0.04,
    ];
    const radiance = (omega: Vec3): Vec3 => {
      const basis = giShBasis(omega), value: Vec3 = [0, 0, 0];
      for (let k = 0; k < 9; k++) for (let c = 0; c < 3; c++) value[c] += coefficients[k * 3 + c] * basis[k];
      return value;
    };
    const views: Vec3[] = [[0, 0, 1], [0.6, -0.48, 0.64], [-1, 0, 0]];
    for (const g of [0, 0.3, -0.5, 0.6]) for (const view of views) {
      const expected = integratePhase(radiance, view, g);
      const actual = fogAirPhaseRadiance(coefficients, view, g);
      for (let c = 0; c < 3; c++) expect(Math.abs(actual[c] - expected[c]), `g ${g} channel ${c}`).toBeLessThan(2e-4);
    }
  });

  it('keeps a uniform sky isotropic under any phase and rejects malformed inputs', () => {
    const sky = [0.4, 0.8, 1.6];
    const coefficients = new Array(27).fill(0);
    for (let c = 0; c < 3; c++) coefficients[c] = sky[c] / 0.28209479;
    for (const g of [-0.9, 0, 0.9]) {
      const value = fogAirPhaseRadiance(coefficients, [0.3, 0.4, -Math.sqrt(0.75)], g);
      value.forEach((channel, c) => expect(channel).toBeCloseTo(sky[c], 6));
    }
    expect(() => fogAirPhaseRadiance(new Array(26).fill(0), [0, 0, 1], 0)).toThrow(/27/);
    expect(() => fogAirPhaseRadiance(new Array(27).fill(Number.NaN), [0, 0, 1], 0)).toThrow(/finite/);
    expect(() => fogAirPhaseRadiance(coefficients, [0, 0, 1], 0.96)).toThrow(/anisotropy/);
  });
});

describe('fog local history rejection', () => {
  // Column-major affine light matrix: NDC x = (x - 1) / 10, y = z / 5, depth = 0.5 - y / 20.
  const light = [0.1, 0, 0, 0, 0, 0, -0.05, 0, 0, 0.2, 0, 0, -0.1, 0, 0.5, 1];

  it('resolves the rejection mode and accounts for the changed-caster region uniform', () => {
    expect(resolveVolumetricFogOptions({ temporal: { occluderRejection: 'local' } }).temporal)
      .toEqual({ maxHistoryFrames: 8, occluderRejection: 'local' });
    expect(() => resolveVolumetricFogOptions({ temporal: { occluderRejection: 'cells' as never } })).toThrow(/occluderRejection/);
    const global = resolveVolumetricFogOptions({ pixelSize: 8, slices: 8, temporal: {} });
    const local = resolveVolumetricFogOptions({ pixelSize: 8, slices: 8, temporal: { occluderRejection: 'local' } });
    expect(fogFixedBytes(resolveVolumetricFogOptions())).toBe(5504);
    expect(fogFixedBytes(global)).toBe(5504);
    expect(fogFixedBytes(local)).toBe(5504 + FOG_OCCLUDER_REGION_BYTES);
    expect(FOG_OCCLUDER_REGION_BYTES).toBe(1040);
    expect(selectFogGrid(64, 64, local).bytes - selectFogGrid(64, 64, global).bytes).toBe(1040);
  });

  it('projects pre- and post-change caster bounds to padded light rectangles behind their nearest depth', () => {
    const regions = fogOccluderRegions([
      { min: [1, 2, 0], max: [3, 4, 5] },
      { min: [-1, 0, -5], max: [1, 1, 0] },
    ], light, 1000);
    expect(regions.count).toBe(2);
    expect(regions.merged).toBe(0);
    expect(regions.data.length).toBe(FOG_OCCLUDER_REGION_BYTES / 4);
    expect(Array.from(regions.data.subarray(0, 4))).toEqual([2, 0, 0, 0]);
    const margin = 2 * 2 / 1000;
    const first = Array.from(regions.data.subarray(4, 9));
    const expected = [0 - margin, 0 - margin, 0.2 + margin, 1 + margin, 0.5 - 4 / 20 - 0.0001];
    first.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 6));
    const second = Array.from(regions.data.subarray(12, 17));
    [-0.2 - margin, -1 - margin, 0 + margin, 0 + margin, 0.5 - 1 / 20 - 0.0001]
      .forEach((value, i) => expect(second[i]).toBeCloseTo(value, 6));
    expect(fogOccluderRegions([], light, 1024).count).toBe(0);
  });

  it('merges regions past capacity into one conservative union instead of dropping them', () => {
    const bounds = Array.from({ length: FOG_OCCLUDER_REGION_CAPACITY + 3 }, (_, i) => ({
      min: [i, 0, 0], max: [i + 0.5, 0.5 + i * 0.1, 0.5] }));
    const regions = fogOccluderRegions(bounds, light, 1024, 0, 0);
    expect(regions.count).toBe(FOG_OCCLUDER_REGION_CAPACITY);
    expect(regions.merged).toBe(3);
    const base = 4 + (FOG_OCCLUDER_REGION_CAPACITY - 1) * 8;
    const last = Array.from(regions.data.subarray(base, base + 5));
    const firstMerged = FOG_OCCLUDER_REGION_CAPACITY - 1, lastMerged = FOG_OCCLUDER_REGION_CAPACITY + 2;
    [(firstMerged - 1) / 10, 0, (lastMerged + 0.5 - 1) / 10, 0.1, 0.5 - (0.5 + lastMerged * 0.1) / 20]
      .forEach((value, i) => expect(last[i]).toBeCloseTo(value, 5));
  });

  it('validates region inputs and decodes packed local history state', () => {
    expect(() => fogOccluderRegions([{ min: [0, 0, 0], max: [1, 1, 1] }], light.slice(0, 15), 1024)).toThrow(/16/);
    expect(() => fogOccluderRegions([{ min: [0, 0, 0], max: [1, 1, Infinity] }], light, 1024)).toThrow(/finite/);
    expect(() => fogOccluderRegions([{ min: [0, 0], max: [1, 1, 1] }], light, 1024)).toThrow(/min and max/);
    expect(() => fogOccluderRegions([], light, 0)).toThrow(/shadowResolution/);
    expect(decodeFogHistoryState(7, false)).toEqual({ count: 7, supportRadius: 0 });
    // 1.5 m is binary16 0x3e00; 60000 m (the saturated sentinel) is 0x7b53.
    expect(decodeFogHistoryState((0x3e00 << 16 | 7) >>> 0, true)).toEqual({ count: 7, supportRadius: 1.5 });
    expect(decodeFogHistoryState((0x7b53 << 16 | 1) >>> 0, true)).toEqual({ count: 1, supportRadius: 60000 });
    expect(() => decodeFogHistoryState(-1, true)).toThrow(/u32/);
  });
});
