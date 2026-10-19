import { describe, expect, it } from 'vitest';
import { compileMaterial, DEFAULT_MATERIAL_DEFINITION, evaluateMaterial, m, type MaterialDefinition } from '../src/material';

const inputs = { instanceColor: [1, 1, 1, 1] as const, instanceMetalRough: [0, 1] as const };
const compile = (changes: Partial<MaterialDefinition>) => compileMaterial({ ...DEFAULT_MATERIAL_DEFINITION, ...changes });

describe('tangent-space material normal', () => {
  it('defaults to neutral and decodes an encoded neutral texel', () => {
    const plain = compile({});
    expect(plain.normalMapped).toBe(false);
    expect(plain.key).toMatch(/^material-v4:/);
    expect(evaluateMaterial(plain, inputs).tangentNormal).toEqual([0, 0, 1]);
    const mapped = compile({ normal: m.texture(3) });
    expect(mapped.normalMapped).toBe(true);
    expect(evaluateMaterial(mapped, { ...inputs, textures: () => [0.5, 0.5, 1, 1] }).tangentNormal).toEqual([0, 0, 1]);
    expect(mapped.wgsl).toContain('tangentNormal: vec3f');
  });

  it('normalizes tilted vectors and applies zero or signed XY scale without changing Z orientation', () => {
    const program = compile({ normal: m.constant([0.75, 0.5, 0.75, 1]),
      normalScale: m.component(m.parameter(3), 0) });
    const sample = (scale: number) => evaluateMaterial(program, { ...inputs,
      parameters: new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, scale, 0, 0, 0]) }).tangentNormal;
    expect(sample(1)[0]).toBeCloseTo(Math.SQRT1_2);
    expect(sample(1)[2]).toBeCloseTo(Math.SQRT1_2);
    expect(sample(0)).toEqual([0, 0, 1]);
    expect(sample(-1)[0]).toBeCloseTo(-Math.SQRT1_2);
    expect(sample(-1)[2]).toBeCloseTo(Math.SQRT1_2);
    const backFacing = compile({ normal: m.constant([0.5, 0.5, 0, 1]) });
    expect(evaluateMaterial(backFacing, inputs).tangentNormal).toEqual([0, 0, -1]);
  });

  it('returns a finite neutral fallback for zero length and overflowed arithmetic', () => {
    const zero = compile({ normal: m.constant([0.5, 0.5, 0.5, 1]) });
    expect(evaluateMaterial(zero, inputs).tangentNormal).toEqual([0, 0, 1]);
    const overflow = compile({ normal: m.multiply(m.constant([3e38, 3e38, 3e38, 1]),
      m.constant([3e38, 3e38, 3e38, 1])) });
    expect(evaluateMaterial(overflow, inputs).tangentNormal).toEqual([0, 0, 1]);
    expect(overflow.wgsl).toContain('all(abs(tangentRaw) < vec3f(1e18))');
  });

  it('includes normal inputs in surface resources but not coverage', () => {
    const program = compile({ alphaMode: 'mask', baseColor: m.texture(0),
      normal: m.texture(3), normalScale: m.component(m.parameter(15), 0) });
    expect(program.textureSlots).toEqual([0, 3]);
    expect(program.parameterCount).toBe(16);
    expect(program.coverageTextureSlots).toEqual([0]);
    expect(program.coverageParameterCount).toBe(0);
    expect(program.coverageWgsl).not.toContain('materialTexture3');
    expect(program.coverageWgsl).not.toContain('values[15u]');
    expect(compile({ normal: m.texture(2) }).key).not.toBe(compile({ normal: m.texture(3) }).key);
    expect(() => compile({ normalScale: Number.NaN })).toThrow(/finite f32/);
  });
});
