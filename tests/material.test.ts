import { describe, expect, it, vi } from 'vitest';
import {
  compileMaterial, DEFAULT_MATERIAL_DEFINITION, DEFAULT_MATERIAL_PROGRAM, evaluateMaterial, m,
  MATERIAL_LIMITS, type MaterialDefinition, type MaterialEvaluationInputs, type ScalarExpr, type Vec4Expr,
} from '../src/material';

const inputs: MaterialEvaluationInputs = { instanceColor: [0.2, 0.4, 0.8, 0.6], instanceMetalRough: [0.3, 0.7] };
const definition = (overrides: Partial<MaterialDefinition> = {}): MaterialDefinition => ({ ...DEFAULT_MATERIAL_DEFINITION, ...overrides });
const compile = (overrides: Partial<MaterialDefinition> = {}) => compileMaterial(definition(overrides));
function close(actual: readonly number[], expected: readonly number[], digits = 6): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], digits));
}

describe('shared material expression compiler', () => {
  it('evaluates the default per-instance PBR material without texture or parameter dependencies', () => {
    const result = evaluateMaterial(DEFAULT_MATERIAL_PROGRAM, inputs);
    close(result.baseColor, [0.2, 0.4, 0.8]);
    expect(result.opacity).toBeCloseTo(0.6);
    expect(result.metallic).toBeCloseTo(0.3);
    expect(result.roughness).toBeCloseTo(0.7);
    expect(result.emissive).toEqual([0, 0, 0]);
    expect(result.covered).toBe(true);
    expect(DEFAULT_MATERIAL_PROGRAM.parameterCount).toBe(0);
    expect(DEFAULT_MATERIAL_PROGRAM.textureSlots).toEqual([]);
    expect(DEFAULT_MATERIAL_PROGRAM.wgsl).toContain('fn evaluateMaterial(uv: vec2f, lod: f32');
    expect(DEFAULT_MATERIAL_PROGRAM.coverageWgsl).toContain('return true;');
  });

  it('combines texture, parameter, multiply, add, mix and component expressions with known numeric values', () => {
    const program = compile({
      baseColor: m.add(m.multiply(m.texture(2), m.parameter(1)), m.constant([0.1, 0, 0, 0])),
      emissive: m.mix(m.constant([2, 4, 6, 8]), m.constant([10, 12, 14, 16]), m.component(m.parameter(0), 2)),
      metallic: m.component(m.parameter(0), 0), roughness: m.component(m.parameter(0), 1),
    });
    const sampler = vi.fn(() => [0.8, 0.6, 0.4, 0.2] as const);
    const result = evaluateMaterial(program, {
      ...inputs, parameters: new Float32Array([0.25, 0.5, 0.25, 0, 0.5, 1, 2, 0.75]),
      textures: sampler, uv: [0.25, 0.75], lod: 2,
    });
    close(result.baseColor, [0.5, 0.6, 0.8]);
    expect(result.opacity).toBeCloseTo(0.15);
    close(result.emissive, [4, 6, 8]);
    expect(result.metallic).toBe(0.25);
    expect(result.roughness).toBe(0.5);
    expect(program.textureSlots).toEqual([2]);
    expect(program.parameterCount).toBe(2);
    expect(sampler).toHaveBeenCalledExactlyOnceWith(2, [0.25, 0.75], 2);
  });

  it('clamps PBR reflectance while preserving Unlit and emissive HDR values', () => {
    for (const shading of ['pbr', 'unlit'] as const) {
      const program = compile({ shading, baseColor: m.constant([-1, 2, 3, 2]), metallic: -0.5,
        roughness: 0, emissive: m.constant([-1, 5, -2, 0]), alphaCutoff: 4 });
      const result = evaluateMaterial(program, inputs);
      expect(result.baseColor).toEqual(shading === 'pbr' ? [0, 1, 1] : [0, 2, 3]);
      expect(program.wgsl).toMatch(shading === 'pbr'
        ? /clamp\(v\d+\.rgb, vec3f\(0\.0\), vec3f\(1\.0\)\)/
        : /max\(v\d+\.rgb, vec3f\(0\.0\)\)/);
      expect(result.opacity).toBe(1);
      expect(result.metallic).toBe(0);
      expect(result.roughness).toBe(Math.fround(0.04));
      expect(result.emissive).toEqual([0, 5, 0]);
      expect(result.alphaCutoff).toBe(4);
    }
    expect(evaluateMaterial(compile({ metallic: 4, roughness: 8 }), inputs)).toMatchObject({ metallic: 1, roughness: 1 });
  });

  it('uses inclusive mask cutoff and clamps coverage exactly like the surface output', () => {
    const program = compile({ alphaMode: 'mask', baseColor: m.parameter(0), alphaCutoff: m.component(m.parameter(1), 0) });
    const sample = (opacity: number, cutoff: number) => evaluateMaterial(program, {
      ...inputs, parameters: new Float32Array([1, 1, 1, opacity, cutoff, 0, 0, 0]),
    });
    expect(sample(0.5, 0.5).covered).toBe(true);
    expect(sample(0.4999, 0.5).covered).toBe(false);
    expect(sample(-2, -1)).toMatchObject({ opacity: 0, alphaCutoff: 0, covered: true });
    expect(sample(4, 3)).toMatchObject({ opacity: 1, alphaCutoff: 3, covered: false });
    expect(sample(1, 1).covered).toBe(true);
    expect(sample(1, 1.01).covered).toBe(false);
    expect(program.coverageWgsl).toMatch(/clamp\(v\d+\.a, 0\.0, 1\.0\) >= max\(v\d+, 0\.0\)/);
    expect(evaluateMaterial(compile({ baseColor: m.constant([1, 1, 1, 0]) }), inputs).covered).toBe(true);
  });

  it('prunes BSDF and emissive-only parameters and textures from the coverage entry', () => {
    const program = compile({ alphaMode: 'mask', baseColor: m.texture(0), emissive: m.texture(2),
      metallic: m.component(m.texture(1), 0), roughness: m.component(m.parameter(15), 1),
      alphaCutoff: m.component(m.parameter(1), 3),
    });
    expect(program.textureSlots).toEqual([0, 1, 2]);
    expect(program.parameterCount).toBe(16);
    expect(program.coverageTextureSlots).toEqual([0]);
    expect(program.coverageParameterCount).toBe(2);
    expect(program.coverageWgsl).toContain('materialTexture0(uv, lod)');
    expect(program.coverageWgsl).not.toContain('materialTexture1');
    expect(program.coverageWgsl).not.toContain('materialTexture2');
    expect(program.coverageWgsl).not.toContain('values[15u]');
    const opaque = compile({ baseColor: m.texture(3), alphaCutoff: m.component(m.parameter(15), 0) });
    expect(opaque.coverageParameterCount).toBe(0);
    expect(opaque.coverageTextureSlots).toEqual([]);
    expect(opaque.coverageWgsl).not.toContain('materialTexture');
  });

  it('reuses repeated and structurally identical nodes without exponential key or source growth', () => {
    const program = compile({ alphaMode: 'mask', baseColor: m.multiply(m.texture(2), m.texture(2)), emissive: m.texture(2) });
    const surfaceSource = program.wgsl.split('fn evaluateCoverage')[0];
    expect(surfaceSource.match(/materialTexture2\(uv, lod\)/g)).toHaveLength(1);
    const sampler = vi.fn(() => [0.5, 0.25, 0.75, 1] as const);
    close(evaluateMaterial(program, { ...inputs, textures: sampler }).baseColor, [0.25, 0.0625, 0.5625]);
    expect(sampler).toHaveBeenCalledTimes(1);
    let shared: Vec4Expr = m.constant([1, 1, 1, 1]);
    for (let i = 0; i < 28; i++) shared = m.add(shared, shared);
    const bounded = compile({ baseColor: shared });
    expect(bounded.key.length).toBeLessThan(5000);
    expect(bounded.wgsl.length).toBeLessThan(6000);
  });

  it('keys static semantics while parameter and texture content updates reuse the program', () => {
    const def = definition({ baseColor: m.multiply(m.texture(0), m.parameter(0)) });
    const program = compileMaterial(def), key = program.key;
    const parameters = new Float32Array([1, 1, 1, 1]);
    const textures = () => [1, 0.5, 0.25, 1] as const;
    const before = evaluateMaterial(program, { ...inputs, parameters, textures });
    parameters[0] = 0.25;
    const after = evaluateMaterial(program, { ...inputs, parameters, textures });
    expect(before.baseColor[0]).toBe(1);
    expect(after.baseColor[0]).toBe(0.25);
    expect(program.key).toBe(key);
    expect(compileMaterial(def).key).toBe(key);
    expect(compileMaterial({ ...def, doubleSided: true }).key).not.toBe(key);
    expect(compileMaterial({ ...def, alphaMode: 'mask' }).key).not.toBe(key);
    expect(compileMaterial({ ...def, shading: 'unlit' }).key).not.toBe(key);
    expect(compileMaterial({ ...def, baseColor: m.texture(1) }).key).not.toBe(key);
  });

  it('snapshots mutable caller data and returns immutable program metadata', () => {
    const color: [number, number, number, number] = [1, 0.5, 0.25, 1];
    const baseColor = { kind: 'constant' as const, value: color };
    const mutable = { ...definition(), baseColor };
    const program = compileMaterial(mutable), source = program.wgsl;
    color[0] = 99;
    mutable.baseColor = { kind: 'constant', value: [0, 0, 0, 0] };
    expect(evaluateMaterial(program, inputs).baseColor).toEqual([1, 0.5, 0.25]);
    expect(program.wgsl).toBe(source);
    expect(Object.isFrozen(program)).toBe(true);
    expect(Object.isFrozen(program.textureSlots)).toBe(true);
    expect(Object.isFrozen(program.coverageTextureSlots)).toBe(true);
  });

  it('canonicalizes negative zero and f32-equivalent constants', () => {
    expect(compile({ baseColor: m.constant([-0, 0.1, 0, 1]) }).key)
      .toBe(compile({ baseColor: m.constant([0, Math.fround(0.1), 0, 1]) }).key);
    expect(compile({ metallic: 1e-30 }).wgsl).toMatch(/e-30f/);
  });

  it('rejects cycles crossing both vector and scalar expressions', () => {
    const cycle = { kind: 'add', a: m.instanceColor, b: m.instanceColor } as { kind: 'add'; a: Vec4Expr; b: Vec4Expr };
    cycle.a = cycle;
    expect(() => compile({ baseColor: cycle })).toThrow(/cycle/);
    const mixed = { kind: 'mix', a: m.instanceColor, b: m.instanceColor, factor: 0 } as { kind: 'mix'; a: Vec4Expr; b: Vec4Expr; factor: ScalarExpr };
    mixed.factor = m.component(mixed, 3);
    expect(() => compile({ baseColor: mixed })).toThrow(/cycle/);
  });

  it('rejects graph depth overflow including reused subtrees seen first at shallower depth', () => {
    let deep = m.instanceColor as Vec4Expr;
    for (let i = 0; i < MATERIAL_LIMITS.depth; i++) deep = m.multiply(deep, m.instanceColor);
    expect(() => compile({ baseColor: deep })).toThrow(/depth/);
    let reused: Vec4Expr = m.instanceColor;
    for (let i = 0; i < 20; i++) reused = m.add(reused, m.instanceColor);
    let nested = reused;
    for (let i = 0; i < 12; i++) nested = m.add(nested, m.instanceColor);
    expect(() => compile({ baseColor: reused, emissive: nested })).toThrow(/depth/);
  });

  it('rejects excessive graph width while allowing shared references within the budget', () => {
    const tree = (depth: number): Vec4Expr => depth === 0 ? m.constant([1, 1, 1, 1]) : m.add(tree(depth - 1), tree(depth - 1));
    expect(() => compile({ baseColor: tree(7) })).toThrow(/128 expression nodes/);
    const shared = m.constant([1, 1, 1, 1]);
    expect(() => compile({ baseColor: m.add(shared, shared) })).not.toThrow();
  });

  it.each([NaN, Infinity, -Infinity, 1e40])('rejects non-f32 finite values %s', (bad) => {
    expect(() => compile({ metallic: bad })).toThrow(/finite f32/);
    expect(() => compile({ baseColor: m.constant([1, bad, 1, 1]) })).toThrow(/finite f32/);
  });

  it('rejects malformed arity, types, sparse constants and invalid indices', () => {
    for (const baseColor of [
      { kind: 'constant', value: [1, 2, 3] }, { kind: 'constant', value: new Array(4) },
      { kind: 'add', a: m.instanceColor }, { kind: 'multiply', a: 1, b: m.instanceColor },
      { kind: 'mix', a: m.instanceColor, b: m.instanceColor, factor: m.instanceColor },
      { kind: 'surprise' }, { kind: 'parameter', slot: 16 }, { kind: 'parameter', slot: -1 },
      { kind: 'parameter', slot: 0.5 }, { kind: 'texture', slot: 4 },
    ]) expect(() => compile({ baseColor: baseColor as Vec4Expr })).toThrow();
    expect(() => compile({ metallic: { kind: 'component', value: m.instanceColor, channel: 4 } as unknown as ScalarExpr })).toThrow(/channel/);
    expect(() => compile({ metallic: m.instanceColor as unknown as ScalarExpr })).toThrow(/wrong value type/);
    expect(() => compile({ shading: 'metal' as 'pbr' })).toThrow(/shading/);
    expect(() => compile({ alphaMode: 'blend' as 'opaque' })).toThrow(/alphaMode/);
    expect(() => compile({ doubleSided: 1 as unknown as boolean })).toThrow(/doubleSided/);
  });

  it('rejects missing and invalid runtime inputs at the responsible slot', () => {
    const program = compile({ baseColor: m.multiply(m.parameter(1), m.texture(2)) });
    expect(() => evaluateMaterial(program, inputs)).toThrow(/8 packed parameter/);
    expect(() => evaluateMaterial(program, { ...inputs, parameters: new Float32Array(8) })).toThrow(/texture sampler slot 2/);
    const parameters = new Float32Array(8); parameters[6] = NaN;
    expect(() => evaluateMaterial(program, { ...inputs, parameters })).toThrow(/parameters\[6\]/);
    expect(() => evaluateMaterial(compile({ baseColor: m.texture(0) }), { ...inputs, textures: () => [1, Infinity, 1, 1] })).toThrow(/texture\[0\]\[1\]/);
    expect(() => evaluateMaterial(DEFAULT_MATERIAL_PROGRAM, { ...inputs, lod: NaN })).toThrow(/lod/);
    expect(() => evaluateMaterial({ ...DEFAULT_MATERIAL_PROGRAM }, inputs)).toThrow(/returned by compileMaterial/);
  });
});
