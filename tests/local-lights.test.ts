import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  LOCAL_LIGHT_HEADER_FLOATS, LOCAL_LIGHT_LIMITS, LOCAL_LIGHT_RECORD_FLOATS, LOCAL_SHADOW_BORDER_TEXELS,
  LocalLightSet, localLightChangeRegions, localLightConeFactor, localLightIrradiance, localLightRangeWindow, localLightShadowFrustum,
  localLightShadowViewCount, resolveLocalLightOptions, validateLocalLight, type LocalLightDescriptor, type LocalLightVec3,
} from '../src/local-lights';
import { FOG_LOCAL_LIGHTS_HOOK, GpuLocalLights, localLightsShaderSource } from '../src/gpu-local-lights';
import { createTimestampQuerySet } from '../src/gpu-timestamp-span';

const point = (overrides: Partial<LocalLightDescriptor> = {}): LocalLightDescriptor => ({
  kind: 'point', position: [0, 2, 0], color: [1, 0.5, 0.25], intensity: 4, range: 10, ...overrides });
const spot = (overrides: Partial<LocalLightDescriptor> = {}): LocalLightDescriptor => ({
  kind: 'spot', position: [0, 3, 0], direction: [0, -2, 0], innerConeAngle: 0.2, outerConeAngle: 0.5,
  color: [1, 1, 1], intensity: 10, range: 8, ...overrides });

function clip(matrix: ArrayLike<number>, p: readonly number[]): [number, number, number, number] {
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let row = 0; row < 4; row++) {
    out[row] = matrix[row] * p[0] + matrix[4 + row] * p[1] + matrix[8 + row] * p[2] + matrix[12 + row];
  }
  return out;
}
function normalize(v: readonly number[]): LocalLightVec3 {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}
/** The shader's face rule in localLightShadowView. */
function pointFace(delta: readonly number[]): number {
  const a = delta.map(Math.abs);
  if (a[0] >= a[1] && a[0] >= a[2]) return delta[0] < 0 ? 1 : 0;
  if (a[1] >= a[2]) return 2 + (delta[1] < 0 ? 1 : 0);
  return 4 + (delta[2] < 0 ? 1 : 0);
}
function determinant3(m: ArrayLike<number>): number {
  return m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
}

describe('local light options and budgets', () => {
  it('resolves documented defaults with exact byte budgets', () => {
    const options = resolveLocalLightOptions();
    expect(options).toMatchObject({ maxLights: 64, maxShadowedLights: 4, shadowResolution: 512, maxShadowViews: 24,
      maxShadowBytes: 32 * 1024 * 1024, shadowNormalBias: 1.5, shadowDepthBias: 1 });
    expect(options.atlasBytes).toBe(4 * 512 * 512 * 24);
    expect(options.uniformBytes).toBe(4 * (4 + 64 * 20 + 24 * 16));
    expect(options.viewUniformBytes).toBe(24 * 256);
    expect(Object.isFrozen(options)).toBe(true);
    const unshadowed = resolveLocalLightOptions({ maxShadowedLights: 0 });
    expect(unshadowed.maxShadowViews).toBe(1);
    expect(unshadowed.atlasBytes).toBe(4);
    expect(resolveLocalLightOptions({ maxShadowedLights: 8, shadowResolution: 256 }).maxShadowViews).toBe(48);
    // Eight shadowed point lights at the default resolution exceed the default 32 MiB atlas budget explicitly.
    expect(() => resolveLocalLightOptions({ maxShadowedLights: 8 })).toThrow('LOCAL_LIGHT_SHADOW_BUDGET');
  });

  it.each([
    [{ maxLights: 0 }, 'INVALID_LOCAL_LIGHT_OPTIONS'], [{ maxLights: 257 }, 'INVALID_LOCAL_LIGHT_OPTIONS'],
    [{ maxLights: 1.5 }, 'INVALID_LOCAL_LIGHT_OPTIONS'], [{ maxShadowedLights: 9 }, 'INVALID_LOCAL_LIGHT_OPTIONS'],
    [{ maxLights: 2, maxShadowedLights: 3 }, 'INVALID_LOCAL_LIGHT_OPTIONS'],
    [{ shadowResolution: 384 }, 'INVALID_LOCAL_LIGHT_SHADOW_RESOLUTION'],
    [{ shadowResolution: 64 }, 'INVALID_LOCAL_LIGHT_SHADOW_RESOLUTION'],
    [{ shadowResolution: 4096 }, 'INVALID_LOCAL_LIGHT_SHADOW_RESOLUTION'],
    [{ maxShadowViews: 0 }, 'INVALID_LOCAL_LIGHT_OPTIONS'], [{ maxShadowViews: 49 }, 'INVALID_LOCAL_LIGHT_OPTIONS'],
    [{ shadowNormalBias: -1 }, 'INVALID_LOCAL_LIGHT_OPTIONS'], [{ shadowDepthBias: Number.NaN }, 'INVALID_LOCAL_LIGHT_OPTIONS'],
    [{ shadowDepthBias: 17 }, 'INVALID_LOCAL_LIGHT_OPTIONS'], [{ maxShadowBytes: 0 }, 'INVALID_LOCAL_LIGHT_OPTIONS'],
    [{ shadowResolution: 2048, maxShadowViews: 3 }, 'LOCAL_LIGHT_SHADOW_BUDGET'],
  ] as const)('rejects %j with %s before allocation', (options, code) => {
    expect(() => resolveLocalLightOptions(options)).toThrow(code);
  });

  it('checks device limits explicitly instead of silently reducing quality', () => {
    expect(() => resolveLocalLightOptions({ shadowResolution: 2048, maxShadowBytes: 1 << 30 },
      { maxTextureDimension2D: 1024 })).toThrow('INVALID_LOCAL_LIGHT_SHADOW_RESOLUTION');
    expect(() => resolveLocalLightOptions({ maxShadowViews: 24 }, { maxTextureArrayLayers: 16 }))
      .toThrow('LOCAL_LIGHT_SHADOW_BUDGET');
    expect(() => resolveLocalLightOptions({ maxLights: 256, maxShadowedLights: 8, shadowResolution: 128 },
      { maxUniformBufferBindingSize: 16384 }))
      .toThrow('LOCAL_LIGHT_BUDGET');
    // The largest legal block still fits the WebGPU default uniform binding limit.
    expect(resolveLocalLightOptions({ maxLights: 256, maxShadowedLights: 8, shadowResolution: 128 }).uniformBytes)
      .toBe(4 * (4 + 256 * 20 + 48 * 16));
  });
});

describe('local light descriptors', () => {
  it('normalizes spot axes and applies documented defaults', () => {
    const resolved = validateLocalLight(spot({ innerConeAngle: undefined, outerConeAngle: undefined }));
    expect(resolved.direction).toEqual([0, -1, 0]);
    expect(resolved.outerConeAngle).toBeCloseTo(Math.PI / 4, 15);
    expect(resolved.innerConeAngle).toBe(0);
    expect(resolved.castShadows).toBe(false);
    expect(resolved.enabled).toBe(true);
    expect(validateLocalLight(point()).direction).toEqual([0, -1, 0]);
  });

  it.each([
    [{ kind: 'area' }, 'kind'], [{ position: [0, 0] }, 'position'], [{ position: [0, 2e6, 0] }, 'position'],
    [{ position: [0, Number.NaN, 0] }, 'position'], [{ color: [-1, 0, 0] }, 'color'], [{ color: [1e39, 0, 0] }, 'color'],
    [{ intensity: -1 }, 'intensity'], [{ intensity: Number.POSITIVE_INFINITY }, 'intensity'],
    [{ color: [1e7, 1, 1], intensity: 1e6 }, 'radiant intensity'], [{ range: 0.001 }, 'range'], [{ range: 20_000 }, 'range'],
    [{ castShadows: 1 as unknown as boolean }, 'flag'], [{ direction: [0, -1, 0] }, 'point lights have no cone'],
    [{ outerConeAngle: 0.4 }, 'point lights have no cone'],
  ] as const)('rejects point light change %j (%s)', (overrides, detail) => {
    expect(() => validateLocalLight(point(overrides as Partial<LocalLightDescriptor>))).toThrow(`INVALID_LOCAL_LIGHT: ${detail}`);
  });

  it.each([
    [{ direction: undefined }, 'direction'], [{ direction: [0, 0, 0] }, 'direction'],
    [{ outerConeAngle: 0 }, 'cone angles'], [{ outerConeAngle: LOCAL_LIGHT_LIMITS.maxSpotAngle + 0.01 }, 'cone angles'],
    [{ innerConeAngle: 0.6, outerConeAngle: 0.5 }, 'cone angles'], [{ innerConeAngle: -0.1 }, 'cone angles'],
  ] as const)('rejects spot light change %j (%s)', (overrides, detail) => {
    expect(() => validateLocalLight(spot(overrides as Partial<LocalLightDescriptor>))).toThrow(`INVALID_LOCAL_LIGHT: ${detail}`);
  });
});

describe('local light radiometry', () => {
  it('uses a smooth range window that is exactly zero at the range', () => {
    expect(localLightRangeWindow(0, 5)).toBe(1);
    expect(localLightRangeWindow(5, 5)).toBe(0);
    expect(localLightRangeWindow(6, 5)).toBe(0);
    expect(localLightRangeWindow(2.5, 5)).toBeCloseTo((1 - 1 / 16) ** 2, 15);
    let previous = 1;
    for (let d = 0.1; d < 5; d += 0.1) { const value = localLightRangeWindow(d, 5); expect(value).toBeLessThanOrEqual(previous); previous = value; }
  });

  it('applies a squared cosine ramp between the inner and outer cone', () => {
    expect(localLightConeFactor(Math.cos(0.1), 0.2, 0.5)).toBe(1);
    expect(localLightConeFactor(Math.cos(0.6), 0.2, 0.5)).toBe(0);
    const cosine = Math.cos(0.35);
    const t = (cosine - Math.cos(0.5)) / (Math.cos(0.2) - Math.cos(0.5));
    expect(localLightConeFactor(cosine, 0.2, 0.5)).toBeCloseTo(t * t, 14);
    // A hard cone (inner == outer) is bounded instead of dividing by zero.
    expect(localLightConeFactor(Math.cos(0.4), 0.5, 0.5)).toBe(1);
  });

  it('matches inverse-square irradiance in the sun irradiance units', () => {
    const light = validateLocalLight(point({ position: [0, 0, 0], color: [1, 1, 1], intensity: 3.5, range: 1000 }));
    const oneMetre = localLightIrradiance(light, [0, 1, 0]);
    expect(oneMetre.distance).toBe(1);
    expect(oneMetre.direction.map(value => value + 0)).toEqual([0, -1, 0]);
    expect(oneMetre.irradiance[0]).toBeCloseTo(3.5 * localLightRangeWindow(1, 1000), 12);
    const twoMetres = localLightIrradiance(light, [0, 2, 0]);
    expect(twoMetres.irradiance[0] / oneMetre.irradiance[0])
      .toBeCloseTo(0.25 * localLightRangeWindow(2, 1000) / localLightRangeWindow(1, 1000), 12);
    // Bounded inside the 1 cm clamp radius.
    const clamped = localLightIrradiance(light, [0, 0.001, 0]).irradiance[0];
    expect(Math.abs(clamped / (3.5 / 1e-4 * localLightRangeWindow(0.001, 1000)) - 1)).toBeLessThan(1e-12);
    expect(localLightIrradiance(light, [0, 1000, 0]).irradiance).toEqual([0, 0, 0]);
  });

  it('attenuates spot lights by the cone around the propagation axis', () => {
    const light = validateLocalLight(spot({ position: [0, 2, 0], direction: [0, -1, 0], innerConeAngle: 0.2, outerConeAngle: 0.5 }));
    const axis = localLightIrradiance(light, [0, 0, 0]);
    expect(axis.irradiance[0]).toBeCloseTo(10 * localLightRangeWindow(2, 8) / 4, 12);
    const angle = 0.35, onRing: LocalLightVec3 = [2 * Math.tan(angle), 0, 0];
    const ring = localLightIrradiance(light, onRing);
    const distance = Math.hypot(onRing[0], 2);
    expect(ring.irradiance[0]).toBeCloseTo(10 * localLightRangeWindow(distance, 8) / distance ** 2
      * localLightConeFactor(Math.cos(angle), 0.2, 0.5), 12);
    expect(localLightIrradiance(light, [0, 4, 0]).irradiance).toEqual([0, 0, 0]);
  });
});

describe('local light shadow frustums', () => {
  const resolution = 512;
  const border = 1 - 2 * LOCAL_SHADOW_BORDER_TEXELS / resolution;

  it('fits the spot cone inside the PCF border with WebGPU depth', () => {
    const light = validateLocalLight(spot({ position: [1, 3, -2], direction: [0.3, -1, 0.2], outerConeAngle: 0.6, range: 8 }));
    const frustum = localLightShadowFrustum(light, resolution);
    expect(frustum.matrices).toHaveLength(1);
    expect(frustum.far).toBe(8);
    expect(frustum.near).toBeCloseTo(8 / 500, 15);
    expect(frustum.tanHalfFov).toBeCloseTo(Math.tan(0.6) / border, 12);
    const matrix = frustum.matrices[0], axis = light.direction;
    const along = (t: number) => light.position.map((value, i) => value + axis[i] * t);
    const center = clip(matrix, along(4));
    expect(Math.abs(center[0] / center[3])).toBeLessThan(1e-6);
    expect(Math.abs(center[1] / center[3])).toBeLessThan(1e-6);
    expect(center[3]).toBeCloseTo(4, 5);
    // Depth contract used by the shader: reference = far (d - near) / (d (far - near)).
    for (const d of [0.5, 1, 4, 8]) {
      const p = clip(matrix, along(d));
      expect(p[2] / p[3]).toBeCloseTo(frustum.far * (d - frustum.near) / (d * (frustum.far - frustum.near)), 5);
    }
    // Every direction on the outer cone stays inside the inner, PCF-safe square.
    const tangent = normalize([axis[2], 0, -axis[0]]);
    const bitangent = [axis[1] * tangent[2] - axis[2] * tangent[1], axis[2] * tangent[0] - axis[0] * tangent[2],
      axis[0] * tangent[1] - axis[1] * tangent[0]];
    for (let step = 0; step < 32; step++) {
      const phi = step * Math.PI / 16;
      const direction = axis.map((value, i) => value * Math.cos(0.6)
        + Math.sin(0.6) * (tangent[i] * Math.cos(phi) + bitangent[i] * Math.sin(phi)));
      const p = clip(matrix, light.position.map((value, i) => value + direction[i] * 5));
      expect(Math.max(Math.abs(p[0] / p[3]), Math.abs(p[1] / p[3]))).toBeLessThanOrEqual(border + 1e-5);
    }
    expect(determinant3(matrix)).toBeLessThan(0); // projection z flip only; no handedness mirror in x/y
  });

  it('covers every direction with the shader face rule for point lights', () => {
    const light = validateLocalLight(point({ position: [-3, 1.5, 4], range: 6 }));
    const frustum = localLightShadowFrustum(light, resolution);
    expect(frustum.matrices).toHaveLength(6);
    expect(frustum.tanHalfFov).toBeCloseTo(1 / border, 12);
    let seed = 7;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32 * 2 - 1; };
    for (let sample = 0; sample < 2000; sample++) {
      const direction = normalize([random(), random(), random()]);
      const distance = 0.05 + (sample % 50) / 50 * 5.9;
      const world = light.position.map((value, i) => value + direction[i] * distance);
      const face = pointFace(direction);
      const p = clip(frustum.matrices[face], world);
      expect(p[3]).toBeGreaterThan(0);
      expect(Math.max(Math.abs(p[0] / p[3]), Math.abs(p[1] / p[3]))).toBeLessThanOrEqual(border + 1e-5);
      const depth = p[2] / p[3];
      expect(depth).toBeGreaterThanOrEqual(-1e-6);
      expect(depth).toBeLessThanOrEqual(1 + 1e-6);
    }
    // All faces keep the same handedness, so the shared back-face culling selects the same faces.
    const signs = frustum.matrices.map(matrix => Math.sign(determinant3(matrix)));
    expect(new Set(signs).size).toBe(1);
  });
});

describe('local light set', () => {
  it('reuses slots with new generations and rejects stale handles', () => {
    const set = new LocalLightSet({ maxLights: 2, maxShadowedLights: 0 });
    const a = set.add(point()), b = set.add(spot());
    expect([a, b]).toEqual([{ id: 0, generation: 0 }, { id: 1, generation: 0 }]);
    expect(() => set.add(point())).toThrow('LOCAL_LIGHT_BUDGET');
    expect(set.revision).toBe(2);
    set.remove(a);
    expect(set.revision).toBe(3);
    expect(() => set.get(a)).toThrow('LOCAL_LIGHT_NOT_FOUND');
    expect(() => set.update(a, { intensity: 1 })).toThrow('LOCAL_LIGHT_NOT_FOUND');
    expect(() => set.remove(a)).toThrow('LOCAL_LIGHT_NOT_FOUND');
    const c = set.add(point({ position: [1, 1, 1] }));
    expect(c).toEqual({ id: 0, generation: 1 });
    expect(set.get(c).version).toBe(1);
    expect(set.count).toBe(2);
    expect(() => set.get({ id: 5, generation: 0 })).toThrow('LOCAL_LIGHT_NOT_FOUND');
  });

  it('versions only effective changes and merges partial patches', () => {
    const set = new LocalLightSet();
    const handle = set.add(spot());
    const before = set.revision;
    expect(set.update(handle, { intensity: 10, position: [0, 3, 0] }).version).toBe(1);
    expect(set.revision).toBe(before);
    const moved = set.update(handle, { position: [1, 3, 0] });
    expect(moved.version).toBe(2);
    expect(moved.outerConeAngle).toBe(0.5);
    expect(set.revision).toBe(before + 1);
    expect(set.setEnabled(handle, false).version).toBe(3);
    expect(set.setEnabled(handle, false).version).toBe(3);
    expect(() => set.setEnabled(handle, 'yes' as unknown as boolean)).toThrow('INVALID_LOCAL_LIGHT');
    const asPoint = set.update(handle, { kind: 'point' });
    expect(asPoint.kind).toBe('point');
    expect(asPoint.outerConeAngle).toBe(0);
    expect(() => set.update(handle, { kind: 'spot' })).toThrow('INVALID_LOCAL_LIGHT: direction');
    expect(() => set.update(handle, { outerConeAngle: 0.3 })).toThrow('point lights have no cone');
    expect(set.update(handle, { kind: 'spot', direction: [1, 0, 0] }).outerConeAngle).toBeCloseTo(Math.PI / 4, 15);
  });

  it('reports the range boxes of added, changed, moved, disabled and removed lights', () => {
    const set = new LocalLightSet();
    const boxAt = (center: LocalLightVec3, range: number) => ({ min: center.map(value => value - range), max: center.map(value => value + range) });
    const a = set.add(point()), b = set.add(spot());
    const empty = set.list();
    expect(localLightChangeRegions([], empty)).toEqual([boxAt([0, 2, 0], 10), boxAt([0, 3, 0], 8)]);
    expect(localLightChangeRegions(empty, set.list())).toEqual([]);
    // In place: one box, not the same box twice.
    let before = set.list();
    set.update(a, { intensity: 8 });
    expect(localLightChangeRegions(before, set.list())).toEqual([boxAt([0, 2, 0], 10)]);
    before = set.list();
    set.update(b, { position: [4, 3, 0] });
    expect(localLightChangeRegions(before, set.list())).toEqual([boxAt([0, 3, 0], 8), boxAt([4, 3, 0], 8)]);
    before = set.list();
    set.setEnabled(a, false);
    expect(localLightChangeRegions(before, set.list())).toEqual([boxAt([0, 2, 0], 10)]);
    // A disabled light shades nothing: editing it changes no region until it is enabled again.
    before = set.list();
    set.update(a, { intensity: 2 });
    expect(localLightChangeRegions(before, set.list())).toEqual([]);
    before = set.list();
    set.setEnabled(a, true);
    expect(localLightChangeRegions(before, set.list())).toEqual([boxAt([0, 2, 0], 10)]);
    // A reused slot has a new generation, so its successor reports even in the same slot; one box when they coincide.
    before = set.list();
    set.remove(b);
    const c = set.add(spot({ position: [4, 3, 0] }));
    expect(c.id).toBe(b.id);
    expect(localLightChangeRegions(before, set.list())).toEqual([boxAt([4, 3, 0], 8)]);
    before = set.list();
    set.remove(c);
    set.add(point({ position: [-4, 1, 0], range: 2 }));
    expect(localLightChangeRegions(before, set.list())).toEqual([boxAt([4, 3, 0], 8), boxAt([-4, 1, 0], 2)]);
  });

  it('budgets shadowed lights and atlas views, including disabled reservations', () => {
    const set = new LocalLightSet({ maxShadowedLights: 2, maxShadowViews: 7 });
    const a = set.add(point({ castShadows: true, enabled: false }));
    expect(() => set.add(point({ castShadows: true }))).toThrow('LOCAL_LIGHT_SHADOW_BUDGET');
    const b = set.add(spot({ castShadows: true }));
    expect(() => set.add(spot({ castShadows: true }))).toThrow('LOCAL_LIGHT_SHADOW_BUDGET');
    expect(set.add(spot())).toEqual({ id: 2, generation: 0 });
    expect(set.shadowAssignments()).toEqual([
      { handle: a, firstView: 0, viewCount: 6, enabled: false },
      { handle: b, firstView: 6, viewCount: 1, enabled: true },
    ]);
    set.update(a, { castShadows: false });
    expect(set.shadowAssignments()).toEqual([{ handle: b, firstView: 0, viewCount: 1, enabled: true }]);
    expect(localLightShadowViewCount('point')).toBe(6);
    expect(localLightShadowViewCount('spot')).toBe(1);
  });

  it('packs enabled lights, cone terms and only active shadow views', () => {
    const set = new LocalLightSet({ maxLights: 4, maxShadowedLights: 2, shadowResolution: 256 });
    const options = set.options;
    const hidden = set.add(point({ enabled: false, castShadows: true }));
    const cone = set.add(spot({ castShadows: true, color: [0.5, 1, 2], intensity: 3 }));
    set.add(point({ position: [4, 1, -2], range: 5 }));
    const target = new Float32Array(options.uniformBytes / 4);
    expect(() => set.pack(new Float32Array(8))).toThrow('LOCAL_LIGHT_PACK_TARGET');
    const packed = set.pack(target);
    const words = new Uint32Array(target.buffer);
    expect(packed.lightCount).toBe(2);
    expect([...words.subarray(0, 4)]).toEqual([2, 1, set.revision, 256]);
    const record = (index: number) => [...target.subarray(LOCAL_LIGHT_HEADER_FLOATS + index * LOCAL_LIGHT_RECORD_FLOATS,
      LOCAL_LIGHT_HEADER_FLOATS + (index + 1) * LOCAL_LIGHT_RECORD_FLOATS)];
    const spotRecord = record(0), pointRecord = record(1);
    const spotLight = set.get(cone), frustum = localLightShadowFrustum(spotLight, 256);
    expect(spotRecord.slice(0, 4)).toEqual([0, 3, 0, 8]);
    expect(spotRecord.slice(4, 8)).toEqual([0, -1, 0, 1]);
    expect(spotRecord.slice(8, 12)).toEqual([1.5, 3, 6, 6].map(Math.fround));
    expect(spotRecord[12]).toBeCloseTo(Math.cos(0.5), 6);
    expect(spotRecord[13]).toBeCloseTo(1 / (Math.cos(0.2) - Math.cos(0.5)), 4);
    expect(spotRecord.slice(14, 16)).toEqual([frustum.near, frustum.far].map(Math.fround));
    expect(spotRecord.slice(16, 20)).toEqual([frustum.tanHalfFov, 1.5, 1, 0].map(Math.fround));
    expect(pointRecord.slice(0, 4)).toEqual([4, 1, -2, 5]);
    expect(pointRecord[7]).toBe(0);
    expect(pointRecord[11]).toBe(-1);
    expect(pointRecord.slice(12, 14)).toEqual([-2, 0]);
    // The disabled point light keeps views 0-5; the enabled spot renders only its own view 6.
    expect(packed.views.map(view => view.layer)).toEqual([6]);
    const viewBase = LOCAL_LIGHT_HEADER_FLOATS + options.maxLights * LOCAL_LIGHT_RECORD_FLOATS;
    expect([...target.subarray(viewBase + 6 * 16, viewBase + 7 * 16)]).toEqual([...frustum.matrices[0]]);
    expect(target.subarray(LOCAL_LIGHT_HEADER_FLOATS + 2 * LOCAL_LIGHT_RECORD_FLOATS, viewBase).every(value => value === 0)).toBe(true);
    set.setEnabled(hidden, true);
    expect(set.pack(target).views.map(view => view.layer)).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });
});

type FakeResource = { label?: string; destroyed: boolean; destroy(): void };
class FakeDevice {
  readonly limits = { maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxUniformBufferBindingSize: 65536,
    maxBufferSize: 256 * 1024 * 1024 };
  readonly buffers: FakeResource[] = [];
  readonly textures: FakeResource[] = [];
  readonly writes: { buffer: FakeResource; bytes: number }[] = [];
  failTexture = false;
  createBuffer = ({ label }: { label: string }) => {
    const buffer: FakeResource = { label, destroyed: false, destroy() { this.destroyed = true; } };
    this.buffers.push(buffer); return buffer as unknown as GPUBuffer;
  };
  createTexture = ({ label, size }: { label: string; size: number[] }) => {
    if (this.failTexture) throw new Error('FAKE_TEXTURE_FAILURE');
    const texture = { label, destroyed: false, depthOrArrayLayers: size[2], createView: () => ({}), destroy() { this.destroyed = true; } };
    this.textures.push(texture); return texture as unknown as GPUTexture;
  };
  createBindGroup = () => ({});
  createBindGroupLayout = () => ({});
  createSampler = () => ({});
  queue = { writeBuffer: (buffer: FakeResource, _offset: number, data: ArrayBufferView) => { this.writes.push({ buffer, bytes: data.byteLength }); } };
}

describe('GPU local light resources', () => {
  beforeEach(() => {
    vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
    vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2 });
    vi.stubGlobal('GPUShaderStage', { COMPUTE: 1, FRAGMENT: 2 });
  });
  afterEach(() => vi.unstubAllGlobals());

  it('accounts every owned byte, uploads only on revision changes and releases on dispose', () => {
    const device = new FakeDevice();
    let ledger = 0;
    const lights = GpuLocalLights.create(device as unknown as GPUDevice, { maxLights: 8, maxShadowedLights: 2, shadowResolution: 128 },
      {} as GPUBindGroupLayout, bytes => { ledger += bytes; });
    const options = lights.options;
    expect(ledger).toBe(options.uniformBytes + options.viewUniformBytes + options.atlasBytes);
    expect(lights.diagnostics()).toMatchObject({ ownedGpuBytes: ledger, cameraCulling: false, lights: 0, renderedShadowViews: 0 });
    expect(lights.update()).toBe(options.uniformBytes + options.viewUniformBytes);
    expect(lights.update()).toBe(0);
    const handle = lights.set.add(point({ castShadows: true }));
    const spotHandle = lights.set.add(spot({ castShadows: true }));
    expect(lights.update()).toBeGreaterThan(0);
    expect(lights.shadowEncodings().map(encoding => encoding.layer)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    const passes: string[] = [], writes: (GPUComputePassTimestampWrites | undefined)[] = [];
    const encoder = {
      beginRenderPass: (descriptor: GPURenderPassDescriptor) => { passes.push(descriptor.label!); writes.push(descriptor.timestampWrites); return { setBindGroup() {}, end() {} }; },
      beginComputePass: (descriptor: GPUComputePassDescriptor) => { passes.push(descriptor.label!); writes.push(descriptor.timestampWrites); return { end() {} }; },
    } as unknown as GPUCommandEncoder;
    let draws = 0;
    const querySet = createTimestampQuerySet({ createQuerySet: () => ({}) } as unknown as GPUDevice, 8);
    lights.encodeShadows(encoder, () => { draws++; }, { querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 });
    expect(draws).toBe(7);
    // The span sits on the view passes; every timed pass writes a whole pair, unowned ends to the scratch slot 8.
    expect(passes).toEqual([0, 1, 2, 3, 4, 5, 6].map(layer => 'local light shadow view ' + layer));
    expect(writes[0]).toEqual({ querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 8 });
    expect(writes.at(-1)).toEqual({ querySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 5 });
    expect(writes.slice(1, -1).every(write => write === undefined)).toBe(true);
    lights.set.setEnabled(handle, false);
    lights.update();
    expect(lights.shadowEncodings().map(encoding => encoding.layer)).toEqual([6]);
    const span = { querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 };
    passes.length = 0; writes.length = 0;
    lights.encodeShadows(encoder, () => { draws++; }, span);
    // One view owns both ends.
    expect(passes).toEqual(['local light shadow view 6']);
    expect(writes).toEqual([span]);
    expect(lights.diagnostics()).toMatchObject({ lights: 2, enabledLights: 1, shadowedLights: 2, reservedShadowViews: 7 });
    // Without views one empty pass still writes the whole span.
    lights.set.setEnabled(spotHandle, false);
    lights.update();
    passes.length = 0; writes.length = 0;
    lights.encodeShadows(encoder, () => { draws++; }, span);
    expect(passes).toEqual(['local light shadows (no views)']);
    expect(writes).toEqual([span]);
    lights.dispose();
    lights.dispose();
    expect(ledger).toBe(0);
    expect(lights.diagnostics().ownedGpuBytes).toBe(0);
    expect([...device.buffers, ...device.textures].every(resource => resource.destroyed)).toBe(true);
    expect(() => lights.update()).toThrow('LOCAL_LIGHTS_DISPOSED');
  });

  it('releases partial allocations when the atlas cannot be created', () => {
    const device = new FakeDevice();
    device.failTexture = true;
    let ledger = 0;
    expect(() => GpuLocalLights.create(device as unknown as GPUDevice, {}, {} as GPUBindGroupLayout, bytes => { ledger += bytes; }))
      .toThrow('FAKE_TEXTURE_FAILURE');
    expect(ledger).toBe(0);
    expect(device.buffers.length).toBe(2);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
  });

  it('rejects device limits before allocation', () => {
    const device = new FakeDevice();
    device.limits.maxTextureArrayLayers = 8;
    expect(() => GpuLocalLights.create(device as unknown as GPUDevice, {}, {} as GPUBindGroupLayout, () => {}))
      .toThrow('LOCAL_LIGHT_SHADOW_BUDGET');
    expect(device.buffers.length).toBe(0);
  });

  it('assembles one shared WGSL program per consumer without unresolved placeholders', () => {
    const options = resolveLocalLightOptions({ maxLights: 40, maxShadowedLights: 1, maxShadowViews: 6 });
    const surface = localLightsShaderSource(options, { group: 0, firstBinding: 3, direct: true });
    expect(surface).toContain('@group(0) @binding(3) var<uniform> localLights: LocalLightBlock;');
    expect(surface).toContain('@group(0) @binding(5) var localShadowSampler: sampler_comparison;');
    expect(surface).toContain('array<LocalLight, 40>');
    expect(surface).toContain('array<mat4x4f, 6>');
    expect(surface).toContain('const LOCAL_LIGHT_MASK_WORDS: u32 = 2u;');
    expect(surface).toContain('fn localLightsDirect(');
    expect(surface).not.toContain('fogLocalLightsScattering');
    for (const placeholder of ['LOCAL_LIGHT_CAPACITY', 'LOCAL_SHADOW_VIEW_CAPACITY', '// LOCAL_LIGHT_BINDINGS']) {
      expect(surface).not.toContain(placeholder);
    }
    const froxel = localLightsShaderSource(options, { group: 1, fog: 'froxel-block' });
    expect(froxel).toContain('var<workgroup> fogLocalMask');
    expect(froxel).toContain('fn fogCullLocalLights(');
    expect(froxel).not.toContain('fn localLightsDirect(');
    const reflected = localLightsShaderSource(options, { group: 2, fog: 'all-lights', direct: true });
    expect(reflected).toContain('@group(2) @binding(0) var<uniform> localLights');
    expect(reflected).toContain('fn fogLocalCandidateWord(word: u32) -> u32 {');
    expect(reflected).not.toContain('var<workgroup>');
    expect(FOG_LOCAL_LIGHTS_HOOK).toContain('fogLocalLightsScattering(world, direction)');
  });
});
