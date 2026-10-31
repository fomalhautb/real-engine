import { describe, expect, it } from 'vitest';
import { reflectionHash, reflectionHistoryLimit, reflectionSampleCoordinates, reflectionSampleRotation,
  reflectionVirtualPoint, resolveReflectionsOptions } from '../src/reflections';

describe('reflection temporal contracts', () => {
  it('resolves bounded history options and keeps the deterministic path when omitted', () => {
    expect(resolveReflectionsOptions().temporal).toBeUndefined();
    expect(resolveReflectionsOptions({ temporal: {} }).temporal).toEqual({ maxHistoryFrames: 32, minHistoryFrames: 8,
      clampSigma: 1.5, parallaxRoughness: 0.35 });
    expect(resolveReflectionsOptions({ temporal: { maxHistoryFrames: 4 } }).temporal?.minHistoryFrames).toBe(4);
    const resolved = resolveReflectionsOptions({ temporal: { maxHistoryFrames: 128, minHistoryFrames: 1, clampSigma: 8, parallaxRoughness: 1 } });
    expect(Object.isFrozen(resolved.temporal)).toBe(true);
    expect(resolveReflectionsOptions(resolved).temporal).toEqual(resolved.temporal);
    for (const temporal of [{ maxHistoryFrames: 1 }, { maxHistoryFrames: 129 }, { maxHistoryFrames: 8, minHistoryFrames: 9 },
      { minHistoryFrames: 0 }, { clampSigma: 0.2 }, { clampSigma: Number.NaN }, { parallaxRoughness: -0.1 }, { maxHistoryFrames: 2.5 }]) {
      expect(() => resolveReflectionsOptions({ temporal })).toThrow(RangeError);
    }
    expect(() => resolveReflectionsOptions({ temporal: null as never })).toThrow('REFLECTIONS_TEMPORAL');
  });

  it('ramps the history cap from mirror-like to rough reflections', () => {
    const temporal = resolveReflectionsOptions({ temporal: { minHistoryFrames: 8, maxHistoryFrames: 32 } }).temporal!;
    expect(reflectionHistoryLimit(0, temporal)).toBe(8);
    expect(reflectionHistoryLimit(0.1, temporal)).toBe(8);
    expect(reflectionHistoryLimit(0.3, temporal)).toBe(20);
    expect(reflectionHistoryLimit(0.5, temporal)).toBe(32);
    expect(reflectionHistoryLimit(1, temporal)).toBe(32);
    let previous = 0;
    for (let roughness = 0; roughness <= 1; roughness += 0.01) {
      const limit = reflectionHistoryLimit(Math.min(roughness, 1), temporal);
      expect(limit).toBeGreaterThanOrEqual(previous);
      previous = limit;
    }
    expect(() => reflectionHistoryLimit(1.5, temporal)).toThrow('REFLECTIONS_ROUGHNESS');
  });

  it('rotates the stratified GGX set per anchor and frame with exact integer wrapping', () => {
    expect(reflectionHash(0)).toBe(0);
    expect(reflectionHash(1)).not.toBe(reflectionHash(2));
    expect(reflectionSampleCoordinates(1, 4)).toEqual([0.5, 0.375]);
    expect(reflectionSampleCoordinates(2, 4)).toEqual([0.25, 0.625]);
    const a = reflectionSampleRotation(7, 3), b = reflectionSampleRotation(7, 4), c = reflectionSampleRotation(8, 3);
    expect(a).toEqual(reflectionSampleRotation(7, 3));
    expect(a).not.toEqual(b);
    expect(a).not.toEqual(c);
    expect(reflectionSampleRotation(4096 + 7, 3)).toEqual(a);
    // Consecutive R2 steps differ by the fixed-point increment modulo 2^32.
    expect((c[0] - a[0]) >>> 0).toBe(3242174889);
    expect((c[1] - a[1]) >>> 0).toBe(2447445414);
    const strata = new Set<number>();
    for (let frame = 0; frame < 64; frame++) {
      for (let index = 0; index < 4; index++) {
        const [u1, u2] = reflectionSampleCoordinates(index, 4, reflectionSampleRotation(frame, 11));
        expect(u1).toBeGreaterThanOrEqual(0); expect(u1).toBeLessThan(1);
        expect(u2).toBeGreaterThanOrEqual(0); expect(u2).toBeLessThan(1);
        strata.add(Math.floor(u2 * 16) * 16 + Math.floor(u1 * 16));
      }
    }
    // 256 rotated samples cover most of a 16x16 stratification of the unit square.
    expect(strata.size).toBeGreaterThan(150);
    expect(() => reflectionSampleRotation(-1, 0)).toThrow('REFLECTIONS_FRAME');
    expect(() => reflectionSampleCoordinates(4, 4)).toThrow('REFLECTIONS_SAMPLE_INDEX');
  });

  it('places a static reflected hit at a camera-independent virtual image point', () => {
    const normal: [number, number, number] = [0, 0, 1];
    const hit: [number, number, number] = [0.3, -0.2, 5];
    const virtual = reflectionVirtualPoint(hit, [0, 0, 0], normal);
    expect(virtual).toEqual([0.3, -0.2, -5]);
    for (const eye of [[0, 0, 3], [1, 0.5, 2], [-2, 1, 4]]) {
      // The receiver point P is where eye→V crosses the mirror. Continuing the
      // view ray past P by the reflected hit distance reaches V.
      const t = eye[2] / (eye[2] - virtual[2]);
      const p = eye.map((value, axis) => value + (virtual[axis] - value) * t);
      const hitDistance = Math.hypot(...hit.map((value, axis) => value - p[axis]));
      const view = p.map((value, axis) => value - eye[axis]), length = Math.hypot(...view);
      const continued = p.map((value, axis) => value + view[axis] / length * hitDistance);
      continued.forEach((value, axis) => expect(value).toBeCloseTo(virtual[axis], 12));
    }
    // A moving receiver plane moves the image twice as far along its normal.
    expect(reflectionVirtualPoint(hit, [0, 0, 0.5], normal)[2]).toBeCloseTo(-4, 12);
    expect(() => reflectionVirtualPoint(hit, [0, 0, 0], [0, 0, 0])).toThrow('REFLECTIONS_NORMAL');
  });
});
