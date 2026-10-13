import { describe, expect, it } from 'vitest';
import {
  compileDeformation, conservativeDeformationBounds, evaluateDeformation,
  validateDeformationState, validateDeformationVertex, validateMorphTargets,
  type Bounds3, type DeformationGraph, type DeformationVertex, type MorphTarget,
} from '../src/deformation';
import { identity, multiply, rotationX, rotationY, rotationZ, scaling, transformPoint, translation } from '../src/math';

const vertex: DeformationVertex = { position: [1, 2, 3], normal: [0, 1, 0], joints: [0, 0, 0, 0], weights: [1, 0, 0, 0] };
const target = (position: readonly number[], normal = [0, 0, 0]): MorphTarget => ({ positions: new Float32Array(position), normals: new Float32Array(normal) });
const close = (actual: readonly number[], expected: readonly number[], digits = 6) => {
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], digits);
};

describe('deformation compiler and reference', () => {
  it('snapshots immutable ordered graphs and keys static constants independently of runtime scalar values', () => {
    const nodes = [{ kind: 'wave' as const, amplitude: 0.25, frequency: 3, phase: 2 }];
    const compiled = compileDeformation({ jointCount: 0, nodes });
    expect(Object.isFrozen(compiled)).toBe(true);
    expect(Object.isFrozen(compiled.nodes)).toBe(true);
    expect(Object.isFrozen(compiled.nodes[0])).toBe(true);
    nodes[0].amplitude = 12;
    expect(compiled.nodes[0]).toMatchObject({ amplitude: 0.25 });
    expect(compiled.scalarCount).toBe(3);
    expect(compiled.morphTargetCount).toBe(0);
    expect(compiled.key).toBe(compileDeformation({ jointCount: 0, nodes: [{ kind: 'wave', amplitude: 0.25, frequency: 3, phase: 2 }] }).key);
    expect(compiled.key).not.toBe(compileDeformation({ jointCount: 0, nodes }).key);
  });

  it('preserves operator order and allows Morph after skin in the resulting coordinate space', () => {
    const morphFirst = compileDeformation({ jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' }] });
    const skinFirst = compileDeformation({ jointCount: 1, nodes: [{ kind: 'skin' }, { kind: 'morph', target: 0, weight: 0 }] });
    const v = { ...vertex, position: [1, 0, 0] as const }, targets = [target([1, 0, 0])], palette = [rotationZ(Math.PI / 2)];
    close(evaluateDeformation(morphFirst, v, 0, targets, palette, [1]).position, [0, 2, 0]);
    close(evaluateDeformation(skinFirst, v, 0, targets, palette, [1]).position, [1, 1, 0]);
    expect(morphFirst.key).not.toBe(skinFirst.key);
    expect(morphFirst.wgslBody.indexOf('let delta')).toBeLessThan(morphFirst.wgslBody.indexOf('let weights'));
    expect(skinFirst.wgslBody.indexOf('let weights')).toBeLessThan(skinFirst.wgslBody.indexOf('let delta'));
  });

  it('accumulates multiple signed Morph position/normal deltas before normalizing', () => {
    const program = compileDeformation({ jointCount: 0, nodes: [
      { kind: 'morph', target: 0, weight: 2 }, { kind: 'morph', target: 1, weight: 0 },
    ] });
    const targets = [target([1, 0, 0], [2, 0, 0]), target([0, 1, 2], [1, 0, -1])];
    const result = evaluateDeformation(program, vertex, 0, targets, [], [-1, 99, 1]);
    close(result.position, [2, 1, 1]);
    close(result.normal, [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)]);
    expect(program.scalarCount).toBe(3);
    expect(program.morphTargetCount).toBe(2);
  });

  it('addresses arbitrary joint palettes above four using normalized four-influence blending', () => {
    const program = compileDeformation({ jointCount: 8, nodes: [{ kind: 'skin' }] });
    const palette = Array.from({ length: 8 }, (_, joint) => translation(joint, joint * 2, 0));
    const v: DeformationVertex = { ...vertex, joints: [4, 5, 6, 7], weights: [0.1, 0.2, 0.3, 0.4] };
    const result = evaluateDeformation(program, v, 0, [], palette, []);
    close(result.position, [7, 14, 3]);
    close(result.normal, [0, 1, 0]);
    const quantized = { ...v, weights: [0.100001, 0.2, 0.3, 0.4] as const };
    const result2 = evaluateDeformation(program, quantized, 0, [], palette, []);
    const shift = (0.100001 * 4 + 0.2 * 5 + 0.3 * 6 + 0.4 * 7) / 1.000001;
    close(result2.position, [1 + shift, 2 + shift * 2, 3]);
  });

  it('wave inverse-transpose normals remain perpendicular to the displaced surface', () => {
    const program = compileDeformation({ jointCount: 0, nodes: [{ kind: 'wave', amplitude: 0.8, frequency: 1.7, phase: 0 }] });
    const v: DeformationVertex = { ...vertex, position: [0, 0.3, 0], normal: [1, 0, 0] };
    const result = evaluateDeformation(program, v, 0, [], [], [0.6]);
    const epsilon = 0.0001;
    const plus = evaluateDeformation(program, { ...v, position: [0, 0.3 + epsilon, 0] }, 0, [], [], [0.6]).position;
    const minus = evaluateDeformation(program, { ...v, position: [0, 0.3 - epsilon, 0] }, 0, [], [], [0.6]).position;
    const tangent = plus.map((p, i) => (p - minus[i]) / (2 * epsilon));
    expect(result.normal.reduce((sum, component, i) => sum + component * tangent[i], 0)).toBeCloseTo(0, 7);
    expect(Math.hypot(...result.normal)).toBeCloseTo(1, 7);
    const later = evaluateDeformation(program, v, 0, [], [], [1.8]);
    expect(later.position[0]).not.toBeCloseTo(result.position[0], 3);
  });

  it('supports rigid-only empty graphs and supplies a finite fallback for degenerate normals', () => {
    const program = compileDeformation({ jointCount: 0, nodes: [] });
    close(evaluateDeformation(program, vertex, 0, [], [], []).position, vertex.position);
    close(evaluateDeformation(program, { ...vertex, normal: [0, 0, 0] }, 0, [], [], []).normal, [0, 1, 0]);
  });
});

describe('deformation bounds and input contracts', () => {
  it('contains deformed vertices across signed Morphs, arbitrary blends, node order and affine model transforms', () => {
    const vertices: DeformationVertex[] = [];
    for (let x = -2; x <= 2; x++) for (let y = -2; y <= 2; y++) for (let z = -2; z <= 2; z++) {
      vertices.push({ position: [x, y, z], normal: [0, 1, 0], joints: [0, 2, 4, 5], weights: [0.1, 0.15, 0.25, 0.5] });
    }
    const targets: MorphTarget[] = [0, 1].map((i) => ({
      positions: Float32Array.from(vertices.flatMap((v) => [Math.sin(v.position[1] + i), v.position[0] * 0.2, v.position[2] * -0.3])),
      normals: new Float32Array(vertices.length * 3),
    }));
    const palette = Array.from({ length: 6 }, (_, i) => multiply(translation(i - 3, Math.sin(i), 1 - i), multiply(rotationY(i * 0.3), rotationX(i * 0.2))));
    const model = multiply(translation(12, -4, 7), multiply(rotationZ(0.7), scaling(2, -0.7, 1.3)));
    const baseBounds: Bounds3 = { min: [-2, -2, -2], max: [2, 2, 2] };
    const graphs: DeformationGraph[] = [
      { jointCount: 6, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'morph', target: 1, weight: 1 }, { kind: 'skin' }, { kind: 'wave', amplitude: -0.8, frequency: 2, phase: 2 }] },
      { jointCount: 6, nodes: [{ kind: 'wave', amplitude: -0.8, frequency: 2, phase: 2 }, { kind: 'skin' }, { kind: 'morph', target: 1, weight: 1 }, { kind: 'skin' }, { kind: 'morph', target: 0, weight: 0 }] },
    ];
    for (const graph of graphs) {
      const program = compileDeformation(graph);
      const envelopes = validateMorphTargets(program, vertices.length, targets);
      for (const weight of [-1.7, 0, 1.7]) {
        const scalars = [weight, 0.6, 0.37];
        const bounds = conservativeDeformationBounds(program, baseBounds, envelopes, palette, scalars, model);
        for (let i = 0; i < vertices.length; i++) {
          const p = transformPoint(model, evaluateDeformation(program, vertices[i], i, targets, palette, scalars).position);
          for (let axis = 0; axis < 3; axis++) {
            expect(p[axis]).toBeGreaterThanOrEqual(bounds.min[axis]);
            expect(p[axis]).toBeLessThanOrEqual(bounds.max[axis]);
          }
        }
      }
    }
  });

  it('extracts signed delta envelopes without assuming zero-centered Morph displacement', () => {
    const program = compileDeformation({ jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] });
    const envelopes = validateMorphTargets(program, 2, [target([2, -4, 7, 3, -2, 8], [0, 0, 0, 0, 0, 0])]);
    expect(envelopes).toEqual([{ min: [2, -4, 7], max: [3, -2, 8] }]);
    const bounds = conservativeDeformationBounds(program, { min: [0, 0, 0], max: [1, 1, 1] }, envelopes, [], [-2]);
    close(bounds.min, [-6, 4, -16], 3);
    close(bounds.max, [-3, 9, -13], 3);
  });

  it.each([
    { jointCount: -1, nodes: [] },
    { jointCount: 1025, nodes: [] },
    { jointCount: 1.5, nodes: [] },
    { jointCount: 0, nodes: [{ kind: 'skin' }] },
    { jointCount: 1, nodes: Array.from({ length: 33 }, () => ({ kind: 'skin' })) },
    { jointCount: 0, nodes: [{ kind: 'morph', target: -1, weight: 0 }] },
    { jointCount: 0, nodes: [{ kind: 'morph', target: 32, weight: 0 }] },
    { jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 128 }] },
    { jointCount: 0, nodes: [{ kind: 'wave', amplitude: Infinity, frequency: 1, phase: 0 }] },
    { jointCount: 0, nodes: [{ kind: 'wave', amplitude: 1, frequency: NaN, phase: 0 }] },
    { jointCount: 0, nodes: [{ kind: 'unsupported' }] },
  ])('rejects malformed or unbounded programs: %j', (graph) => {
    expect(() => compileDeformation(graph as DeformationGraph)).toThrow();
  });

  it('rejects unsupported bone transforms, invalid skin influences, missing scalars and malformed Morph assets', () => {
    const program = compileDeformation({ jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' }] });
    for (const palette of [[], [scaling(2, 1, 1)], [scaling(-1, 1, 1)]]) {
      expect(() => validateDeformationState(program, palette, [1])).toThrow();
    }
    expect(() => validateDeformationState(program, [identity()], [])).toThrow(/scalar/);
    expect(() => validateDeformationState(program, [identity()], [NaN])).toThrow(/finite/);
    for (const invalid of [
      { ...vertex, joints: [1, 0, 0, 0] as const },
      { ...vertex, weights: [1.5, -0.5, 0, 0] as const },
      { ...vertex, weights: [0, 0, 0, 0] as const },
      { ...vertex, weights: [2, 0, 0, 0] as const },
      { ...vertex, position: [NaN, 0, 0] as const },
    ]) expect(() => validateDeformationVertex(program, invalid)).toThrow();
    expect(() => validateMorphTargets(program, 1, [])).toThrow(/short/);
    expect(() => validateMorphTargets(program, 2, [target([0, 0, 0])])).toThrow(/packed/);
    expect(() => validateMorphTargets(program, 1, [target([0, NaN, 0])])).toThrow(/finite/);
    expect(() => conservativeDeformationBounds(program, { min: [1, 0, 0], max: [0, 0, 0] }, [], [identity()], [1])).toThrow(/inverted/);
  });
});
