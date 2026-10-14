import { describe, expect, it } from 'vitest';
import { identity, lookAt, multiply, transformPoint } from '../src/math';
import { resolveSunShadowOptions, sunShadowProjection, type SunShadowBounds } from '../src/sun-shadow';

const corners = ({ min, max }: SunShadowBounds): [number, number, number][] => {
  const result: [number, number, number][] = [];
  for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
    result.push([x, y, z]);
  }
  return result;
};

describe('sun shadow projection', () => {
  it('preserves the original unfocused projection exactly', () => {
    const direction: [number, number, number] = [-18, 30, 19];
    const length = Math.hypot(...direction), d = direction.map(value => value / length);
    const target: [number, number, number] = [0, 0, -3];
    const eye = target.map((value, axis) => value + d[axis] * 40);
    const ortho = identity();
    ortho[0] = 1 / 25; ortho[5] = 1 / 25; ortho[10] = -1 / 70; ortho[14] = -0.1 / 70;
    const oldMatrix = multiply(ortho, lookAt(eye, target, [0, 1, 0]));
    const projected = sunShadowProjection(direction, resolveSunShadowOptions());
    expect([...projected.matrix]).toEqual([...oldMatrix]);
    expect(projected.width).toBe(50);
    expect(projected.depth).toBe(70);
    expect(projected.texelWorldSize).toBe(50 / 1024);
  });

  it.each([
    [[-18, 30, 19], { min: [-9, -2, -12], max: [11, 14, 8] }],
    [[0, 1, 0], { min: [-3, -5, -7], max: [4, 8, 12] }],
    [[0, -1, 0], { min: [-4, -4, -4], max: [4, 4, 4] }],
    [[1, 0, 0], { min: [2, 3, 4], max: [2, 3, 4] }],
    [[0.2, 0.8, -0.6], { min: [999_990, -999_999, 999_980], max: [1_000_000, -999_990, 1_000_000] }],
  ] as const)('contains all eight corners for light %j', (direction, focusBounds) => {
    const projected = sunShadowProjection(direction, resolveSunShadowOptions({ focusBounds, padding: 0 }));
    expect(projected.width).toBeGreaterThan(0);
    expect(projected.depth).toBeGreaterThan(0);
    expect(projected.texelWorldSize).toBeCloseTo(projected.width / 1024, 12);
    expect([...projected.matrix].every(Number.isFinite)).toBe(true);
    for (const corner of corners(focusBounds)) {
      const [x, y, z] = transformPoint(projected.matrix, corner);
      expect(x).toBeGreaterThanOrEqual(-1);
      expect(x).toBeLessThanOrEqual(1);
      expect(y).toBeGreaterThanOrEqual(-1);
      expect(y).toBeLessThanOrEqual(1);
      expect(z).toBeGreaterThanOrEqual(0);
      expect(z).toBeLessThanOrEqual(1);
    }
  });

  it('keeps the matrix stable when the focus box moves less than a shadow texel', () => {
    const bounds: SunShadowBounds = { min: [-4, -2, -5], max: [4, 2, 5] };
    const first = sunShadowProjection([0, 1, 0], resolveSunShadowOptions({ focusBounds: bounds }));
    // For a vertical sun, world X is an XY shadow-plane axis. A binary fraction
    // is used here so the box span is exactly unchanged after translation.
    const delta = 1 / 1024;
    expect(delta).toBeLessThan(first.texelWorldSize / 2);
    const shifted: SunShadowBounds = {
      min: [bounds.min[0] + delta, bounds.min[1], bounds.min[2]],
      max: [bounds.max[0] + delta, bounds.max[1], bounds.max[2]],
    };
    const second = sunShadowProjection([0, 1, 0], resolveSunShadowOptions({ focusBounds: shifted }));
    expect([...second.matrix]).toEqual([...first.matrix]);
  });

  it('resolves immutable bounds and enforces texture limits and byte budgets', () => {
    const bounds: { min: [number, number, number]; max: [number, number, number] } = {
      min: [-1, -2, -3], max: [4, 5, 6],
    };
    const resolved = resolveSunShadowOptions({ focusBounds: bounds, resolution: 2048, maxBytes: 16 * 1024 * 1024 });
    expect(resolved.bytes).toBe(16 * 1024 * 1024);
    expect(Object.isFrozen(resolved)).toBe(true);
    expect(Object.isFrozen(resolved.focusBounds?.min)).toBe(true);
    bounds.min[0] = -100;
    expect(resolved.focusBounds?.min[0]).toBe(-1);
    expect(() => resolveSunShadowOptions({ resolution: 2048 }, 1024)).toThrow('INVALID_SUN_SHADOW_RESOLUTION');
    expect(() => resolveSunShadowOptions({ resolution: 2048, maxBytes: resolved.bytes - 1 })).toThrow('SUN_SHADOW_BUDGET_EXCEEDED');
  });

  it.each([
    { resolution: 255 }, { resolution: 300 }, { resolution: 16384 }, { resolution: Infinity },
    { padding: -1 }, { padding: NaN }, { padding: 1e40 },
    { maxBytes: -1 }, { maxBytes: 1.5 },
    { focusBounds: { min: [2, 0, 0], max: [1, 1, 1] } },
    { focusBounds: { min: [0, NaN, 0], max: [1, 1, 1] } },
    { focusBounds: { min: [0, 0, 0], max: [1e40, 1, 1] } },
  ] as const)('rejects invalid or unbounded options %j', options => {
    expect(() => resolveSunShadowOptions(options)).toThrow();
  });

  it('rejects invalid directions before producing a matrix', () => {
    const resolved = resolveSunShadowOptions({ focusBounds: { min: [-1, -1, -1], max: [1, 1, 1] } });
    for (const direction of [[0, 0, 0], [NaN, 1, 0], [1e40, 0, 0]]) {
      expect(() => sunShadowProjection(direction as [number, number, number], resolved)).toThrow('INVALID_SUN_DIRECTION');
    }
  });
});
