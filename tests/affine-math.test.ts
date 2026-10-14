import { describe, expect, it } from 'vitest';
import { inverseAffine, normalMatrix } from '../src/math';

// Column-major affine transform: nonuniform scale, XY/XZ/YZ shear, and translation.
const affine = new Float32Array([
  2, 0, 0, 0,
  1, 3, 0, 0,
  0.5, 0.25, 4, 0,
  5, -2, 7, 1,
]);

function apply(matrix: ArrayLike<number>, vector: readonly [number, number, number], w: 0 | 1): [number, number, number] {
  return [0, 1, 2].map(row => matrix[row] * vector[0] + matrix[4 + row] * vector[1]
    + matrix[8 + row] * vector[2] + matrix[12 + row] * w) as [number, number, number];
}

function close(actual: readonly number[], expected: readonly number[]): void {
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 6);
}

describe('affine transform contracts', () => {
  it('round-trips points through an inverse with shear and nonuniform scale', () => {
    const inverse = inverseAffine(affine);
    close(apply(affine, [1, -2, 3], 1), [6.5, -7.25, 19]);
    for (const point of [[1, -2, 3], [0, 0, 0], [-4, 5, 0.25]] as const) {
      close(apply(inverse, apply(affine, point, 1), 1), point);
    }
  });

  it('keeps the transformed normal perpendicular to both transformed surface tangents', () => {
    const normal = apply(normalMatrix(affine), [-1, 1, 0], 0);
    close(normal, [-0.5, 0.5, 0.03125]);
    const tangentA = apply(affine, [1, 1, 0], 0);
    const tangentB = apply(affine, [0, 0, 1], 0);
    close(tangentA, [3, 3, 0]);
    close(tangentB, [0.5, 0.25, 4]);
    const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);
    expect(dot(normal, tangentA)).toBeCloseTo(0, 6);
    expect(dot(normal, tangentB)).toBeCloseTo(0, 6);
  });

  it('rejects singular affine transforms', () => {
    const singular = new Float32Array(affine);
    singular[0] = 0;
    expect(() => inverseAffine(singular)).toThrow(/nonsingular/);
    expect(() => normalMatrix(singular)).toThrow(/nonsingular/);
  });
});
