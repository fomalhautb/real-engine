/** CPU contracts and numerical reference for a bounded diffuse probe grid. */
import { GI_DISTANCE_CELLS } from './gi-visibility';
export interface DiffuseGiOptions {
  origin: readonly [number, number, number];
  spacing: readonly [number, number, number];
  counts: readonly [number, number, number];
  raysPerProbe?: number;
  maxDistance?: number;
  normalBias?: number;
  skyRadiance?: readonly [number, number, number];
  /** Feed the previous completed probe field into this frame's diffuse capture. */
  multiBounce?: boolean;
  /** Smoothly suppress low-confidence visibility weights; 0 disables the curve. */
  visibilityWeightThreshold?: number;
  maxBytes?: number;
}

export interface ResolvedDiffuseGiOptions {
  readonly origin: readonly [number, number, number];
  readonly spacing: readonly [number, number, number];
  readonly counts: readonly [number, number, number];
  readonly raysPerProbe: number;
  readonly maxDistance: number;
  readonly normalBias: number;
  readonly skyRadiance: readonly [number, number, number];
  readonly multiBounce: boolean;
  readonly visibilityWeightThreshold: number;
  readonly maxBytes: number;
  readonly probeCount: number;
  readonly rayCount: number;
}

/** 9 RGB SH vec4s, one depth-moment vec4 per octahedral cell, and 1 metadata vec4. */
export const DIFFUSE_GI_PROBE_BYTES = (9 + GI_DISTANCE_CELLS + 1) * 16;
export const DIFFUSE_GI_PROBE_FLOATS = DIFFUSE_GI_PROBE_BYTES / 4;
export const DIFFUSE_GI_STATE_FLOAT_OFFSET = (9 + GI_DISTANCE_CELLS) * 4;
const MAX_BYTES = 128 * 1024 * 1024;
const DEFAULT_BYTES = 32 * 1024 * 1024;
const GOLDEN_ANGLE = 2.399963229728653;
const SH_BAND_SCALE = [Math.PI, 2 * Math.PI / 3, Math.PI / 4] as const;

function triple(value: readonly number[], name: string): readonly [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3
    || !Number.isFinite(value[0]) || !Number.isFinite(value[1]) || !Number.isFinite(value[2])) {
    throw new RangeError(`DIFFUSE_GI_${name}`);
  }
  return Object.freeze([value[0], value[1], value[2]] as [number, number, number]);
}

function unit(value: readonly number[], name: string): readonly [number, number, number] {
  const v = triple(value, name);
  const length = Math.hypot(...v);
  if (!Number.isFinite(length) || length <= 0) throw new RangeError(`DIFFUSE_GI_${name}`);
  return [v[0] / length, v[1] / length, v[2] / length];
}

export function resolveDiffuseGiOptions(options: DiffuseGiOptions): ResolvedDiffuseGiOptions {
  if (!options || typeof options !== 'object') throw new RangeError('DIFFUSE_GI_OPTIONS');
  const origin = triple(options.origin, 'ORIGIN');
  const spacing = triple(options.spacing, 'SPACING');
  const counts = triple(options.counts, 'COUNTS');
  const skyRadiance = triple(options.skyRadiance ?? [0, 0, 0], 'SKY_RADIANCE');
  if (origin.some(x => Math.abs(x) > 1e6) || spacing.some(x => x < 0.001 || x > 1000)
    || counts.some(x => !Number.isInteger(x) || x < 2 || x > 32)
    || skyRadiance.some(x => x < 0 || x > 1024)) throw new RangeError('DIFFUSE_GI_GRID');

  const probeCount = counts[0] * counts[1] * counts[2];
  const raysPerProbe = options.raysPerProbe ?? 64;
  const rayCount = probeCount * raysPerProbe;
  if (probeCount > 512 || !Number.isInteger(raysPerProbe) || raysPerProbe < 16 || raysPerProbe > 256
    || rayCount > 65_536) throw new RangeError('DIFFUSE_GI_WORK_BUDGET');

  // The GPU grid is float32. Check every adjacent coordinate, including the far edge,
  // using float32 arithmetic so large origins cannot collapse distinct probes.
  for (let axis = 0; axis < 3; axis++) {
    const base = Math.fround(origin[axis]);
    const step = Math.fround(spacing[axis]);
    let previous = base;
    for (let index = 1; index < counts[axis]; index++) {
      const current = Math.fround(base + Math.fround(index * step));
      if (!Number.isFinite(current) || current <= previous) throw new RangeError('DIFFUSE_GI_FLOAT32_GRID');
      previous = current;
    }
  }

  const maxDistance = options.maxDistance ?? 30;
  if (options.multiBounce !== undefined && typeof options.multiBounce !== 'boolean') {
    throw new RangeError('DIFFUSE_GI_MULTI_BOUNCE');
  }
  const multiBounce = options.multiBounce ?? false;
  const visibilityWeightThreshold = options.visibilityWeightThreshold ?? 0.2;
  if (options.visibilityWeightThreshold === null || !Number.isFinite(visibilityWeightThreshold)
    || visibilityWeightThreshold < 0 || visibilityWeightThreshold > 1) {
    throw new RangeError('DIFFUSE_GI_VISIBILITY_WEIGHT_THRESHOLD');
  }
  const normalBiasLimit = Math.min(...spacing) * 0.25;
  // A millimetre-scale grid must remain legal when the caller omits the bias.
  const normalBias = options.normalBias ?? Math.min(0.03, normalBiasLimit);
  const maxBytes = options.maxBytes ?? DEFAULT_BYTES;
  if (!Number.isFinite(maxDistance) || maxDistance < 0.01 || maxDistance > 1000
    || !Number.isFinite(normalBias) || normalBias < 0 || normalBias > normalBiasLimit
    || !Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_BYTES) {
    throw new RangeError('DIFFUSE_GI_OPTIONS');
  }
  if (probeCount * DIFFUSE_GI_PROBE_BYTES > maxBytes) throw new RangeError('DIFFUSE_GI_MEMORY_BUDGET');
  return Object.freeze({ origin, spacing, counts, raysPerProbe, maxDistance, normalBias,
    skyRadiance, multiBounce, visibilityWeightThreshold, maxBytes, probeCount, rayCount });
}

/** Deterministic Fibonacci-sphere direction; rayIndex is zero based. */
export function giRayDirection(rayIndex: number, raysPerProbe: number): [number, number, number] {
  if (!Number.isInteger(raysPerProbe) || raysPerProbe < 16 || raysPerProbe > 256
    || !Number.isInteger(rayIndex) || rayIndex < 0 || rayIndex >= raysPerProbe) {
    throw new RangeError('DIFFUSE_GI_RAY_INDEX');
  }
  const y = 1 - 2 * (rayIndex + 0.5) / raysPerProbe;
  const radius = Math.sqrt(Math.max(0, 1 - y * y));
  const phi = rayIndex * GOLDEN_ANGLE;
  return [Math.cos(phi) * radius, y, Math.sin(phi) * radius];
}

/** Real SH, l <= 2, in the fixed basis order used by the GPU probe record. */
export function giShBasis(dir: readonly [number, number, number]): number[] {
  const [x, y, z] = unit(dir, 'DIRECTION');
  return [0.28209479, 0.48860251 * y, 0.48860251 * z, 0.48860251 * x,
    1.09254843 * x * y, 1.09254843 * y * z, 0.31539157 * (3 * z * z - 1),
    1.09254843 * x * z, 0.546274215 * (x * x - y * y)];
}

/** Coefficients are 9 consecutive RGB triples in giShBasis order. */
export function evaluateGiIrradiance(coeffs9xRGB: ArrayLike<number>,
  normal: readonly [number, number, number]): [number, number, number] {
  if (!coeffs9xRGB || coeffs9xRGB.length !== 27) throw new RangeError('DIFFUSE_GI_COEFFICIENTS');
  const basis = giShBasis(normal);
  const irradiance: [number, number, number] = [0, 0, 0];
  for (let coefficient = 0; coefficient < 9; coefficient++) {
    const band = coefficient === 0 ? 0 : coefficient <= 3 ? 1 : 2;
    const scale = basis[coefficient] * SH_BAND_SCALE[band];
    for (let channel = 0; channel < 3; channel++) {
      const value = coeffs9xRGB[coefficient * 3 + channel];
      if (!Number.isFinite(value)) throw new RangeError('DIFFUSE_GI_COEFFICIENTS');
      irradiance[channel] += value * scale;
    }
  }
  if (!irradiance.every(Number.isFinite)) throw new RangeError('DIFFUSE_GI_IRRADIANCE');
  return [Math.max(0, irradiance[0]), Math.max(0, irradiance[1]), Math.max(0, irradiance[2])];
}
