import { describe, expect, it } from 'vitest';
import { DEFAULT_CAMERA_PROJECTION, projectionScale, resolveCameraProjection } from '../src/camera-projection';

describe('camera projection', () => {
  it('keeps the original fixed camera as the default', () => {
    expect(resolveCameraProjection(undefined)).toBe(DEFAULT_CAMERA_PROJECTION);
    expect(DEFAULT_CAMERA_PROJECTION).toEqual({ verticalFovRadians: Math.PI / 3.2, near: 0.1, far: 150 });
    expect(resolveCameraProjection({})).toEqual(DEFAULT_CAMERA_PROJECTION);
    // The renderer previously used height / (2 tan(π/6.4)) as the LOD and texture focal scale.
    expect(projectionScale(DEFAULT_CAMERA_PROJECTION, 1000)).toBeCloseTo(1000 / (2 * Math.tan(Math.PI / 6.4)), 12);
  });

  it('accepts bounded custom values and freezes them', () => {
    const projection = resolveCameraProjection({ verticalFovRadians: 0.753942, far: 260 });
    expect(projection).toEqual({ verticalFovRadians: 0.753942, near: 0.1, far: 260 });
    expect(Object.isFrozen(projection)).toBe(true);
    expect(projectionScale(projection, 720)).toBeCloseTo(720 / (2 * Math.tan(0.376971)), 12);
  });

  it('rejects out-of-range or non-numeric values explicitly', () => {
    for (const verticalFovRadians of [0, Math.PI / 200, Math.PI, Number.NaN, Infinity]) {
      expect(() => resolveCameraProjection({ verticalFovRadians })).toThrow('INVALID_PROJECTION: verticalFovRadians');
    }
    for (const near of [0, 0.001, 11, -1, Number.NaN]) expect(() => resolveCameraProjection({ near })).toThrow('INVALID_PROJECTION: near');
    for (const far of [0.1, 0.05, 100_001, Infinity]) expect(() => resolveCameraProjection({ far })).toThrow('INVALID_PROJECTION: far');
    expect(() => resolveCameraProjection({ near: 0.01, far: 20_000 })).toThrow('INVALID_PROJECTION: far');
    expect(() => resolveCameraProjection({ verticalFovRadians: '1' as unknown as number })).toThrow('INVALID_PROJECTION');
    expect(() => resolveCameraProjection(null as unknown as undefined)).toThrow('INVALID_PROJECTION');
  });
});
