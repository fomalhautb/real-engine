import { describe, expect, it } from 'vitest';
import {
  composeMediumSegments, composeWithSurface, henyeyGreenstein,
  integrateUniformMedium, snapshotMediumCoefficients,
} from '../src/participating-media';

const vacuum = { extinction: 0, albedo: [0, 0, 0] as const, emission: [0, 0, 0] as const };

describe('uniform participating-medium reference', () => {
  it('snapshots validated coefficients instead of retaining mutable author input', () => {
    const albedo: [number, number, number] = [0.25, 0.5, 0.75];
    const emission: [number, number, number] = [1, 2, 3];
    const input = { extinction: 0.2, albedo, emission };
    const snapshot = snapshotMediumCoefficients(input);
    albedo[0] = 1;
    emission[0] = 99;
    input.extinction = 3;
    expect(snapshot).toEqual({ extinction: 0.2, albedo: [0.25, 0.5, 0.75], emission: [1, 2, 3], anisotropy: 0 });
    expect(Object.isFrozen(snapshot)).toBe(true);
    expect(Object.isFrozen(snapshot.albedo)).toBe(true);
    expect(Object.isFrozen(snapshot.emission)).toBe(true);
  });

  it('returns vacuum identity and independent zero-extinction emission', () => {
    const empty = integrateUniformMedium(vacuum, 12, [3, 4, 5]);
    expect(empty).toEqual({ transmittance: 1, scattering: [0, 0, 0] });
    expect(composeWithSurface(empty, [2, 5, 7])).toEqual([2, 5, 7]);
    const glow = integrateUniformMedium({ extinction: 0, albedo: [1, 1, 1], emission: [2, 0, 0.5] }, 4, [100, 100, 100]);
    expect(glow).toEqual({ transmittance: 1, scattering: [8, 0, 2] });
    expect(integrateUniformMedium({ extinction: 5, albedo: [1, 1, 1], emission: [2, 2, 2] }, 0, [9, 9, 9]))
      .toEqual({ transmittance: 1, scattering: [0, 0, 0] });
  });

  it('matches Beer–Lambert and the analytic constant-source integral', () => {
    const distance = 3;
    const extinction = 0.4;
    const phase = 1 / (4 * Math.PI);
    const result = integrateUniformMedium({ extinction, albedo: [0.5, 0.25, 0], emission: [0, 0.1, 0.3] },
      distance, [10, 20, 30]);
    const transmittance = Math.exp(-extinction * distance);
    const pathIntegral = (1 - transmittance) / extinction;
    expect(result.transmittance).toBeCloseTo(transmittance, 14);
    expect(result.scattering[0]).toBeCloseTo(extinction * 0.5 * phase * 10 * pathIntegral, 13);
    expect(result.scattering[1]).toBeCloseTo((extinction * 0.25 * phase * 20 + 0.1) * pathIntegral, 13);
    expect(result.scattering[2]).toBeCloseTo(0.3 * pathIntegral, 13);
  });

  it('remains accurate for tiny optical depth and when splitting a path', () => {
    const sigma = 1e-6, distance = 1e-3;
    const tiny = integrateUniformMedium({ extinction: sigma, albedo: [0, 0, 0], emission: [1, 0, 0] },
      distance, [0, 0, 0]);
    const series = distance - sigma * distance * distance / 2;
    expect(Math.abs(tiny.scattering[0] - series)).toBeLessThan(1e-15);
    const medium = { extinction: 0.3, albedo: [0.2, 0.4, 0.6] as const, emission: [0.1, 0.2, 0.3] as const };
    const whole = integrateUniformMedium(medium, 7, [4, 5, 6]);
    const split = composeMediumSegments(
      integrateUniformMedium(medium, 2, [4, 5, 6]),
      integrateUniformMedium(medium, 5, [4, 5, 6]),
    );
    expect(split.transmittance).toBeCloseTo(whole.transmittance, 14);
    for (let channel = 0; channel < 3; channel++) {
      expect(split.scattering[channel]).toBeCloseTo(whole.scattering[channel], 13);
    }
  });

  it('composes front to back in order before the surface', () => {
    const redFront = { transmittance: 0.25, scattering: [2, 0, 0] as const };
    const blueBack = { transmittance: 0.5, scattering: [0, 0, 4] as const };
    const frontFirst = composeMediumSegments(redFront, blueBack);
    const backFirst = composeMediumSegments(blueBack, redFront);
    expect(frontFirst).toEqual({ transmittance: 0.125, scattering: [2, 0, 1] });
    expect(backFirst).toEqual({ transmittance: 0.125, scattering: [1, 0, 4] });
    expect(composeWithSurface(frontFirst, [8, 2, 16])).toEqual([3, 0.25, 3]);
  });

  it('uses a normalized HG phase with the incoming propagation convention', () => {
    expect(henyeyGreenstein(-1, 0)).toBeCloseTo(1 / (4 * Math.PI), 14);
    expect(henyeyGreenstein(1, 0)).toBeCloseTo(1 / (4 * Math.PI), 14);
    expect(henyeyGreenstein(1, 0.7)).toBeGreaterThan(henyeyGreenstein(-1, 0.7));
    const intervals = 20_000;
    let integral = 0;
    for (let index = 0; index <= intervals; index++) {
      const cosine = -1 + index * 2 / intervals;
      integral += henyeyGreenstein(cosine, 0.7) * (index === 0 || index === intervals ? 0.5 : 1);
    }
    integral *= (2 / intervals) * 2 * Math.PI;
    expect(Math.abs(integral - 1)).toBeLessThan(1e-5);
  });

  it('rejects invalid inputs and nonrepresentable integrated radiance', () => {
    expect(() => snapshotMediumCoefficients({ extinction: -1, albedo: [0, 0, 0], emission: [0, 0, 0] }))
      .toThrow(/extinction/);
    expect(() => snapshotMediumCoefficients({ extinction: 1, albedo: [1.01, 0, 0], emission: [0, 0, 0] }))
      .toThrow(/albedo/);
    expect(() => snapshotMediumCoefficients({ extinction: 1, albedo: [0, 0, 0], emission: [0, 0, -1] }))
      .toThrow(/emission/);
    expect(() => snapshotMediumCoefficients({ extinction: 1, albedo: [0, 0, 0], emission: [0, 0, 0], anisotropy: 0.96 }))
      .toThrow(/anisotropy/);
    expect(() => henyeyGreenstein(1.01, 0)).toThrow(/cosTheta/);
    expect(() => integrateUniformMedium(vacuum, Infinity, [0, 0, 0])).toThrow(/distance/);
    expect(() => integrateUniformMedium(vacuum, 1, [NaN, 0, 0])).toThrow(/incidentRadiance/);
    expect(() => integrateUniformMedium(vacuum, 1, [-1, 0, 0])).toThrow(/incidentRadiance/);
    expect(() => composeMediumSegments({ transmittance: 2, scattering: [0, 0, 0] }, { transmittance: 1, scattering: [0, 0, 0] }))
      .toThrow(/transmittance/);
    const maxFiniteFloat32 = Math.fround(3.4028235e38);
    expect(() => integrateUniformMedium({ extinction: 0, albedo: [0, 0, 0], emission: [maxFiniteFloat32, 0, 0] },
      2, [0, 0, 0])).toThrow(/scattering/);
  });
});
