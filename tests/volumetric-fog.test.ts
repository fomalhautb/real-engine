import { describe, expect, it } from 'vitest';
import { fogHeightDensity, resolveVolumetricFogOptions, selectFogGrid,
  type FogVolume, type VolumetricFogOptions } from '../src/volumetric-fog';

const medium = { extinction: 0.25, albedo: [0.2, 0.4, 0.6] as const,
  emission: [1, 2, 3] as const, anisotropy: 0.3 };

describe('volumetric fog configuration', () => {
  it('sets bounded defaults and chooses the smallest valid 1080p grid', () => {
    const options = resolveVolumetricFogOptions();
    expect(options).toEqual({ pixelSize: 8, slices: 64, maxViewDepth: 80,
      maxFroxels: 1_048_576, maxBytes: 64 * 1024 * 1024, volumes: [] });
    expect(Object.isFrozen(options)).toBe(true);
    expect(Object.isFrozen(options.volumes)).toBe(true);
    expect(selectFogGrid(1920, 1080, options)).toEqual({ columns: 160, rows: 90,
      slices: 64, pixelSize: 12, froxels: 921_600, bytes: 46_085_376 });
    expect(selectFogGrid(1280, 720, options).pixelSize).toBe(8);
  });

  it('snapshots nested medium and volume data, including array coordinates', () => {
    const min: [number, number, number] = [-2, 0, -1];
    const max: [number, number, number] = [2, 4, 1];
    const center: [number, number, number] = [5, 6, 7];
    const albedo: [number, number, number] = [0.2, 0.4, 0.6];
    const input: VolumetricFogOptions = { global: { coefficients: { ...medium, albedo }, baseHeight: 2,
      heightFalloff: 0.4 }, volumes: [
      { id: 1, shape: { kind: 'box', min, max }, coefficients: medium },
      { id: 0xffff_ffff, shape: { kind: 'sphere', center, radius: 3 }, coefficients: medium },
    ] };
    const resolved = resolveVolumetricFogOptions(input);
    min[0] = -99; max[0] = 99; center[0] = 99; albedo[0] = 1;
    expect(resolved.global?.coefficients.albedo).toEqual([0.2, 0.4, 0.6]);
    expect(resolved.volumes[0].shape).toEqual({ kind: 'box', min: [-2, 0, -1], max: [2, 4, 1] });
    expect(resolved.volumes[1].shape).toEqual({ kind: 'sphere', center: [5, 6, 7], radius: 3 });
    expect(Object.isFrozen(resolved.global)).toBe(true);
    expect(Object.isFrozen(resolved.global?.coefficients.albedo)).toBe(true);
    expect(Object.isFrozen(resolved.volumes)).toBe(true);
    expect(Object.isFrozen(resolved.volumes[0].shape)).toBe(true);
    expect(Object.isFrozen(resolved.volumes[0].shape.kind === 'box' && resolved.volumes[0].shape.min)).toBe(true);
  });

  it('uses unit density below the base and exact exponential height falloff above it', () => {
    const { global } = resolveVolumetricFogOptions({ global: { coefficients: medium, baseHeight: 3, heightFalloff: 0.5 } });
    expect(fogHeightDensity(-100, global!)).toBe(1);
    expect(fogHeightDensity(3, global!)).toBe(1);
    expect(fogHeightDensity(7, global!)).toBeCloseTo(Math.exp(-2), 14);
    expect(fogHeightDensity(1e6, global!)).toBe(0);
    expect(() => fogHeightDensity(NaN, global!)).toThrow(/height/);
  });

  it('grows the pixel size minimally for froxel and byte limits', () => {
    const froxelLimited = resolveVolumetricFogOptions({ pixelSize: 1, slices: 8, maxFroxels: 96 });
    expect(selectFogGrid(33, 17, froxelLimited)).toEqual({ columns: 4, rows: 2,
      slices: 8, pixelSize: 9, froxels: 64, bytes: 256 + 64 * 80 + 33 * 17 * 8 + 64 * 32 });
    const byteLimited = resolveVolumetricFogOptions({ pixelSize: 1, slices: 8, maxBytes: 7000 });
    expect(selectFogGrid(10, 10, byteLimited)).toEqual({ columns: 1, rows: 1,
      slices: 8, pixelSize: 10, froxels: 8, bytes: 6432 });
    expect(() => selectFogGrid(10, 10, resolveVolumetricFogOptions({ slices: 8, maxBytes: 6431 })))
      .toThrow(/cannot fit/);
  });

  it('rejects invalid option, medium, and local-shape budgets', () => {
    const invalidOptions: VolumetricFogOptions[] = [
      { pixelSize: 0 }, { pixelSize: 129 }, { pixelSize: 1.5 },
      { slices: 7 }, { slices: 129 }, { maxViewDepth: 0.09 }, { maxViewDepth: 1001 },
      { maxFroxels: 63 }, { maxFroxels: 4_194_305 }, { maxBytes: 0 }, { maxBytes: 1.5 },
      { global: { coefficients: { ...medium, extinction: 10_001 } } },
      { global: { coefficients: { ...medium, emission: [0, 10_001, 0] } } },
      { global: { coefficients: medium, baseHeight: 1_000_001 } },
      { global: { coefficients: medium, heightFalloff: -0.1 } },
      { global: { coefficients: medium, heightFalloff: 101 } },
    ];
    for (const options of invalidOptions) expect(() => resolveVolumetricFogOptions(options)).toThrow();
    const volume = (id: number, shape: FogVolume['shape']): FogVolume => ({ id, shape, coefficients: medium });
    const box = { kind: 'box', min: [0, 0, 0], max: [1, 1, 1] } as const;
    for (const volumes of [
      [volume(1, box), volume(1, box)],
      [volume(-1, box)],
      [volume(0x1_0000_0000, box)],
      [volume(1, { kind: 'box', min: [0, 0, 0], max: [0, 1, 1] })],
      [volume(1, { kind: 'box', min: [0, 0, 0], max: [1_000_001, 1, 1] })],
      [volume(1, { kind: 'sphere', center: [0, 0, 0], radius: 0 })],
      [volume(1, { kind: 'sphere', center: [0, 0, 0], radius: 1_000_001 })],
      Array.from({ length: 65 }, (_, id) => volume(id, box)),
    ]) expect(() => resolveVolumetricFogOptions({ volumes })).toThrow();
    expect(() => selectFogGrid(0, 10, resolveVolumetricFogOptions())).toThrow(/width/);
    expect(() => selectFogGrid(10, Infinity, resolveVolumetricFogOptions())).toThrow(/height/);
    expect(() => selectFogGrid(Number.MAX_SAFE_INTEGER, 2, resolveVolumetricFogOptions())).toThrow(/HDR/);
  });
});
