import { describe, expect, it } from 'vitest';
import { compileDeformation, evaluateDeformation, validateDeformationVertex, validateMorphTargets, type DeformationVertex, type MorphTarget } from '../src/deformation';
import { identity, rotationZ, scaling, type Mat4 } from '../src/math';
import { snapshotMeshTangents, validateMeshTangents, validateTangent } from '../src/tangent-frame';

type V3 = [number, number, number];
const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, component, axis) => sum + component * b[axis], 0);
const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v: V3): V3 => v.map(component => component / Math.hypot(...v)) as V3;
const close = (actual: ArrayLike<number>, expected: readonly number[], digits = 6) => {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((component, axis) => expect(actual[axis]).toBeCloseTo(component, digits));
};
const base: DeformationVertex = { position: [0.4, 0.3, -0.2], normal: [0, 1, 0], tangent: [1, 0, 0, -1], joints: [0, 0, 0, 0], weights: [1, 0, 0, 0] };

describe('shared tangent deformation', () => {
  it('matches numerical surface derivatives through Morph, blended affine reflection and wave deformation', () => {
    const palette: Mat4[] = [
      new Float32Array([-2, 0, 0, 0, 0.5, 1.2, 0, 0, 0.3, 0.2, 0.8, 0, 0.1, 0.3, -0.2, 1]),
      new Float32Array([-1, 0.1, 0, 0, 0.2, 2, 0.3, 0, -0.2, 0.1, 1.1, 0, -0.3, 0.2, 0.1, 1]),
    ];
    const program = compileDeformation({ jointCount: 2, nodes: [
      { kind: 'morph', target: 0, weight: 0 }, { kind: 'skin', normalMode: 'affine' },
      { kind: 'wave', amplitude: 0.7, frequency: 0.9, phase: 1 },
    ] });
    const du: V3 = [1.4, 0.2, 0.1], dv: V3 = [0.3, -0.1, 1.2];
    const targetNormal = unit(cross(dv, du));
    const vertex: DeformationVertex = { ...base, joints: [0, 1, 0, 0], weights: [0.35, 0.65, 0, 0] };
    const u = vertex.position[0], v = vertex.position[2], phase = 0.4;
    const targets: MorphTarget[] = [{
      positions: new Float32Array([0.4 * u + 0.3 * v, 0.2 * u - 0.1 * v, 0.1 * u + 0.2 * v]),
      normals: new Float32Array(targetNormal.map((component, axis) => component - base.normal[axis])),
      tangents: new Float32Array([0.4, 0.2, 0.1]),
    }];
    const result = evaluateDeformation(program, vertex, 0, targets, palette, [1, phase]);
    // Independent parametric surface: derive the entire final surface, not the tangent/normal algorithm.
    const surface = (s: number, t: number, offset = 0): V3 => {
      const point: V3 = [1.4 * s + 0.3 * t, 0.3 + 0.2 * s - 0.1 * t, 0.1 * s + 1.2 * t];
      point.forEach((_, axis) => { point[axis] += offset * targetNormal[axis]; });
      const transformed: V3 = [0, 0, 0];
      for (let bone = 0; bone < 2; bone++) for (let row = 0; row < 3; row++) {
        transformed[row] += [0.35, 0.65][bone] * (palette[bone][row] * point[0]
          + palette[bone][row + 4] * point[1] + palette[bone][row + 8] * point[2] + palette[bone][row + 12]);
      }
      transformed[0] += Math.fround(0.7) * Math.sin(transformed[1] * Math.fround(0.9) + phase);
      return transformed;
    };
    const derivative = (axis: number): V3 => {
      const epsilon = 1e-5, plus = [u, v, 0], minus = [u, v, 0];
      plus[axis] += epsilon; minus[axis] -= epsilon;
      const a = surface(plus[0], plus[1], plus[2]), b = surface(minus[0], minus[1], minus[2]);
      return a.map((component, componentAxis) => (component - b[componentAxis]) / (2 * epsilon)) as V3;
    };
    const tangentU = derivative(0), tangentV = derivative(1), outward = derivative(2);
    let normal = unit(cross(tangentV, tangentU));
    if (dot(normal, outward) < 0) normal = normal.map(component => -component) as V3;
    close(result.position, surface(u, v));
    close(result.normal, normal);
    close(result.tangent!.slice(0, 3), unit(tangentU));
    expect(dot(result.normal, result.tangent!.slice(0, 3))).toBeCloseTo(0, 7);
    expect(result.tangent![3]).toBe(1); // Initial -1 orientation multiplied by the reflected skin blend.
    const bitangent = cross(result.normal, result.tangent!.slice(0, 3)).map(component => component * result.tangent![3]);
    expect(dot(bitangent, tangentV)).toBeGreaterThan(0);
  });

  it('uses the blended linear transform for rigid skin tangents and composes reflection handedness', () => {
    const rigid = compileDeformation({ jointCount: 2, nodes: [{ kind: 'skin' }] });
    const vertex: DeformationVertex = { ...base, normal: [0, 0, 1], tangent: [1, 0, 0, 1], joints: [0, 1, 0, 0], weights: [0.5, 0.5, 0, 0] };
    close(evaluateDeformation(rigid, vertex, 0, [], [identity(), rotationZ(Math.PI / 2)], []).tangent!, [Math.SQRT1_2, Math.SQRT1_2, 0, 1]);
    const twice = compileDeformation({ jointCount: 1, nodes: [{ kind: 'skin', normalMode: 'affine' }, { kind: 'skin', normalMode: 'affine' }] });
    close(evaluateDeformation(twice, base, 0, [], [scaling(-2, 1, 1)], []).tangent!, [1, 0, 0, -1]);
  });

  it('keeps signed Morph tangent deltas in node order and preserves the authored orientation', () => {
    const graph = compileDeformation({ jointCount: 1, nodes: [{ kind: 'skin' }, { kind: 'morph', target: 0, weight: 0 }, { kind: 'morph', target: 1, weight: 1 }] });
    const target = (tangent: V3): MorphTarget => ({ positions: new Float32Array(3), normals: new Float32Array(3), tangents: new Float32Array(tangent) });
    const result = evaluateDeformation(graph, { ...base, normal: [0, 0, 1] }, 0, [target([0, 1, 0]), target([1, 0, 0])], [rotationZ(Math.PI / 2)], [-0.5, 2]);
    close(result.tangent!, [2 / Math.sqrt(4.25), 0.5 / Math.sqrt(4.25), 0, -1]);
  });

  it('distinguishes missing tangents and supplies stable finite orthogonal frames for collapsed and overflowing vectors', () => {
    const empty = compileDeformation({ jointCount: 0, nodes: [] });
    const { tangent: _unused, ...missing } = base;
    expect(Object.hasOwn(evaluateDeformation(empty, missing, 0, [], [], []), 'tangent')).toBe(false);
    for (const tangent of [[0, 0, 0, -1], [0, 1, 0, 1], [1e-10, 1, 0, 1]] as const) {
      const frame = evaluateDeformation(empty, { ...base, tangent }, 0, [], [], []);
      close(frame.tangent!, [1, 0, 0, tangent[3]]);
      expect(dot(frame.normal, frame.tangent!.slice(0, 3))).toBeCloseTo(0, 7);
    }
    const morph = compileDeformation({ jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] });
    const overflowing: MorphTarget = { positions: new Float32Array(3), normals: new Float32Array(3), tangents: new Float32Array([3e38, 0, 0]) };
    close(evaluateDeformation(morph, base, 0, [overflowing], [], [8]).tangent!, [1, 0, 0, -1]);
  });
});

describe('authored tangent asset contracts', () => {
  it('snapshots packed tangents and encodes missing data as zero w without aliasing caller storage', () => {
    const source = new Float32Array([1, 0, 0, 1, 0, 0, 1, -1]);
    const snapshot = snapshotMeshTangents(source, 2);
    source[0] = 99;
    close(snapshot, [1, 0, 0, 1, 0, 0, 1, -1]);
    close(snapshotMeshTangents(undefined, 2), Array(8).fill(0));
  });

  it('rejects invalid authored orientation, f32 overflow and malformed Morph tangent arrays before upload', () => {
    const graph = compileDeformation({ jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] });
    for (const tangent of [[1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 2], [Infinity, 0, 0, 1], [1e39, 0, 0, -1]]) {
      expect(() => validateTangent(tangent)).toThrow();
      expect(() => validateDeformationVertex(graph, { ...base, tangent: tangent as unknown as readonly [number, number, number, number] })).toThrow();
    }
    expect(() => validateMeshTangents(new Float32Array(3), 1)).toThrow(/packed/);
    expect(() => validateMeshTangents(new Float32Array([1, 0, 0, 0]), 1)).toThrow(/w/);
    const target: MorphTarget = { positions: new Float32Array(3), normals: new Float32Array(3), tangents: new Float32Array(2) };
    expect(() => validateMorphTargets(graph, 1, [target])).toThrow(/packed xyz/);
    target.tangents = new Float32Array([NaN, 0, 0]);
    expect(() => validateMorphTargets(graph, 1, [target])).toThrow(/finite/);
    expect(() => evaluateDeformation(graph, base, 0, [target], [], [1])).toThrow(/finite/);
  });
});
