/** CPU contracts and numerical reference for a bounded diffuse probe grid. */
import { GI_DISTANCE_CELLS } from './gi-visibility';
import { resolveFinalGatherOptions, type FinalGatherOptions, type ResolvedFinalGatherOptions } from './final-gather';
import type { WorldQueryOptions } from './world-query';
import { resolveDiffuseGiUpdateOptions, type DiffuseGiUpdateOptions, type ResolvedDiffuseGiUpdateOptions } from './gi-probe-schedule';
export type GiVisibilityDirection = 'probe-to-receiver' | 'receiver-to-probe';
export type GiVisibilityFilter = 'moment-bilinear' | 'per-tap-chebyshev-bilinear';
export interface DiffuseGiOptions {
  origin: readonly [number, number, number];
  spacing: readonly [number, number, number];
  counts: readonly [number, number, number];
  raysPerProbe?: number;
  maxDistance?: number;
  /** Receiver offset for spatial probe interpolation, in world units. */
  normalBias?: number;
  /** Geometric-normal offset at a captured hit before tracing toward the sun.
   * Independent of probe interpolation; defaults to 0.0001 world units. */
  shadowNormalBias?: number;
  skyRadiance?: readonly [number, number, number];
  /** Feed the previous completed probe field into this frame's diffuse capture. */
  multiBounce?: boolean;
  /** Bounded area samples for emissive geometry at each probe; zero disables them. */
  emissiveSamplesPerProbe?: number;
  /** Smoothly suppress low-confidence visibility weights and unsupported probe
   * interpolation; 0 disables the curve. Moment confidence is not transmittance. */
  visibilityWeightThreshold?: number;
  /** Direction of material-sidedness used by the cache support estimate. */
  visibilityDirection?: GiVisibilityDirection;
  /** Default: interpolate moments before evaluating confidence. Per-tap
   * confidence filtering is opt-in and does not add capture rays or storage. */
  visibilityFilter?: GiVisibilityFilter;
  /** Additional traversal allowance for receiver-directed support. Zero (default)
   * disables that capture and rejects switching to it at runtime. */
  maxVisibilityWorkUnits?: number;
  maxBytes?: number;
  /** Optional bounded diffuse path segments at visible surfaces, ahead of the world probe cache. */
  finalGather?: FinalGatherOptions;
  /** Probe capture policy. Default 'full' recaptures the grid on every update;
   * 'scheduled' captures a bounded, change-driven subset with explicit staleness. */
  update?: DiffuseGiUpdateOptions;
}

export interface ResolvedDiffuseGiOptions {
  readonly origin: readonly [number, number, number];
  readonly spacing: readonly [number, number, number];
  readonly counts: readonly [number, number, number];
  readonly raysPerProbe: number;
  readonly maxDistance: number;
  readonly normalBias: number;
  readonly shadowNormalBias: number;
  readonly skyRadiance: readonly [number, number, number];
  readonly multiBounce: boolean;
  readonly emissiveSamplesPerProbe: number;
  readonly visibilityWeightThreshold: number;
  readonly visibilityDirection: GiVisibilityDirection;
  readonly visibilityFilter: GiVisibilityFilter;
  readonly maxVisibilityWorkUnits: number;
  readonly maxBytes: number;
  readonly probeCount: number;
  readonly rayCount: number;
  readonly finalGather?: ResolvedFinalGatherOptions;
  readonly update: ResolvedDiffuseGiUpdateOptions;
}

/** 9 RGB SH vec4s, one depth-moment vec4 per octahedral cell, and 1 metadata vec4. */
export const DIFFUSE_GI_PROBE_BYTES = (9 + GI_DISTANCE_CELLS + 1) * 16;
export const DIFFUSE_GI_PROBE_FLOATS = DIFFUSE_GI_PROBE_BYTES / 4;
export const DIFFUSE_GI_STATE_FLOAT_OFFSET = (9 + GI_DISTANCE_CELLS) * 4;
/** Eight vec4 lanes shared by capture, surface, reflection and area-sample shaders. */
export const DIFFUSE_GI_PARAMETER_BYTES = 128;
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
  const emissiveSamplesPerProbe = options.emissiveSamplesPerProbe === undefined ? 0 : options.emissiveSamplesPerProbe;
  if (!Number.isSafeInteger(emissiveSamplesPerProbe) || emissiveSamplesPerProbe < 0 || emissiveSamplesPerProbe > 64) {
    throw new RangeError('DIFFUSE_GI_EMISSIVE_SAMPLES_PER_PROBE');
  }
  const visibilityWeightThreshold = options.visibilityWeightThreshold ?? 0.2;
  if (options.visibilityWeightThreshold === null || !Number.isFinite(visibilityWeightThreshold)
    || visibilityWeightThreshold < 0 || visibilityWeightThreshold > 1) {
    throw new RangeError('DIFFUSE_GI_VISIBILITY_WEIGHT_THRESHOLD');
  }
  const visibilityDirection = options.visibilityDirection === undefined ? 'probe-to-receiver' : options.visibilityDirection;
  if (visibilityDirection !== 'probe-to-receiver' && visibilityDirection !== 'receiver-to-probe') {
    throw new RangeError('DIFFUSE_GI_VISIBILITY_DIRECTION');
  }
  const visibilityFilter = options.visibilityFilter === undefined ? 'moment-bilinear' : options.visibilityFilter;
  if (visibilityFilter !== 'moment-bilinear' && visibilityFilter !== 'per-tap-chebyshev-bilinear') {
    throw new RangeError('DIFFUSE_GI_VISIBILITY_FILTER');
  }
  const maxVisibilityWorkUnits = options.maxVisibilityWorkUnits === undefined ? 0 : options.maxVisibilityWorkUnits;
  if (!Number.isSafeInteger(maxVisibilityWorkUnits) || maxVisibilityWorkUnits < 0 || maxVisibilityWorkUnits > 67_108_864
    || (maxVisibilityWorkUnits > 0 && maxVisibilityWorkUnits < rayCount)
    || (visibilityDirection === 'receiver-to-probe' && maxVisibilityWorkUnits === 0)) {
    throw new RangeError('DIFFUSE_GI_VISIBILITY_WORK_BUDGET');
  }
  const normalBiasLimit = Math.min(...spacing) * 0.25;
  // A millimetre-scale grid must remain legal when the caller omits the bias.
  const normalBias = options.normalBias ?? Math.min(0.03, normalBiasLimit);
  const shadowNormalBias = options.shadowNormalBias === undefined ? 0.0001 : options.shadowNormalBias;
  if (!Number.isFinite(shadowNormalBias) || typeof shadowNormalBias !== 'number'
    || shadowNormalBias < 0 || shadowNormalBias > normalBiasLimit) {
    throw new RangeError('DIFFUSE_GI_SHADOW_NORMAL_BIAS');
  }
  const maxBytes = options.maxBytes ?? DEFAULT_BYTES;
  if (!Number.isFinite(maxDistance) || maxDistance < 0.01 || maxDistance > 1000
    || !Number.isFinite(normalBias) || normalBias < 0 || normalBias > normalBiasLimit
    || !Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_BYTES) {
    throw new RangeError('DIFFUSE_GI_OPTIONS');
  }
  if (probeCount * DIFFUSE_GI_PROBE_BYTES > maxBytes) throw new RangeError('DIFFUSE_GI_MEMORY_BUDGET');
  const finalGather = options.finalGather === undefined ? undefined : resolveFinalGatherOptions(options.finalGather);
  const update = resolveDiffuseGiUpdateOptions(options.update, probeCount);
  return Object.freeze({ origin, spacing, counts, raysPerProbe, maxDistance, normalBias, shadowNormalBias,
    skyRadiance, multiBounce, emissiveSamplesPerProbe, visibilityWeightThreshold, visibilityDirection, visibilityFilter,
    maxVisibilityWorkUnits, maxBytes, probeCount, rayCount, finalGather, update });
}

/** Conservative slots for all configured GI modes, including a later support
 * direction switch. The ordinary primary/shadow traces reuse one slot. */
export function diffuseGiQueryParameterSlots(options: ResolvedDiffuseGiOptions): number {
  return (options.finalGather ? 6 : options.emissiveSamplesPerProbe ? 2 : 1)
    + Number(options.maxVisibilityWorkUnits > 0);
}

export function validateDiffuseGiQueryBudget(options: ResolvedDiffuseGiOptions,
  queries: Pick<Required<WorldQueryOptions>, 'maxRays' | 'maxTotalWorkUnits' | 'maxTraceParameterSetsPerEncoder'>): void {
  if (options.rayCount > queries.maxRays) throw new Error('DIFFUSE_GI_QUERY_RAY_BUDGET');
  if (options.rayCount > queries.maxTotalWorkUnits) throw new Error('DIFFUSE_GI_QUERY_WORK_BUDGET');
  if (options.maxVisibilityWorkUnits > queries.maxTotalWorkUnits) throw new Error('DIFFUSE_GI_VISIBILITY_WORK_BUDGET');
  if (queries.maxTraceParameterSetsPerEncoder < diffuseGiQueryParameterSlots(options)) {
    throw new Error(options.finalGather ? 'FINAL_GATHER_QUERY_PARAMETER_BUDGET' : 'DIFFUSE_GI_QUERY_PARAMETER_BUDGET');
  }
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
