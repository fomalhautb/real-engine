import { snapshotMediumCoefficients, type MediumCoefficients, type MediumSnapshot } from './participating-media';
import { giShBasis } from './diffuse-gi';

type Point3 = readonly [number, number, number];

export interface FogHeightMedium {
  readonly coefficients: MediumCoefficients;
  readonly baseHeight?: number;
  readonly heightFalloff?: number;
}

export interface FogVolume {
  readonly id: number;
  readonly shape: { readonly kind: 'box'; readonly min: Point3; readonly max: Point3 }
    | { readonly kind: 'sphere'; readonly center: Point3; readonly radius: number };
  readonly coefficients: MediumCoefficients;
}

/** How moving or deforming shadow casters invalidate world-space fog history.
 * `global` (default) resets every cell. `local` tracks, per cell, a world-space
 * ball containing every retained sample and rejects only cells whose ball meets a
 * changed caster's directional-shadow region (pre- and post-change bounds). Changes
 * that cannot be localized (missing bounds, non-affine sun matrix) still reset globally. */
export type FogOccluderRejection = 'global' | 'local';

export interface FogTemporalOptions {
  readonly maxHistoryFrames?: number;
  readonly occluderRejection?: FogOccluderRejection;
}

/** Low-frequency light in the air. The world GI probe field stores incoming
 * radiance (SH9, whose sky miss radiance already carries geometric occlusion);
 * fog convolves it with each medium's HG phase. Requires `RendererOptions.diffuseGi`. */
export interface FogAirLightingOptions {
  /** Inject the GI field. Default true; can be toggled at runtime. */
  readonly gi?: boolean;
  /** Author-declared sky visibility in [0, 1] for media beyond the GI field's
   * support, lit by the GI field's own sky radiance. Default 0: no unverified skylight. */
  readonly outsideSkyVisibility?: number;
}

export interface VolumetricFogOptions {
  readonly pixelSize?: number;
  readonly slices?: number;
  readonly maxViewDepth?: number;
  readonly maxFroxels?: number;
  readonly maxBytes?: number;
  readonly temporal?: false | FogTemporalOptions;
  readonly airLighting?: false | FogAirLightingOptions;
  readonly global?: FogHeightMedium;
  readonly volumes?: readonly FogVolume[];
}

export interface ResolvedFogHeightMedium {
  readonly coefficients: MediumSnapshot;
  readonly baseHeight: number;
  readonly heightFalloff: number;
}

export interface ResolvedFogVolume {
  readonly id: number;
  readonly shape: FogVolume['shape'];
  readonly coefficients: MediumSnapshot;
}

export interface ResolvedFogTemporalOptions {
  readonly maxHistoryFrames: number;
  readonly occluderRejection: FogOccluderRejection;
}

export interface ResolvedFogAirLightingOptions {
  readonly gi: boolean;
  readonly outsideSkyVisibility: number;
}

export interface ResolvedVolumetricFogOptions {
  readonly pixelSize: number;
  readonly slices: number;
  readonly maxViewDepth: number;
  readonly maxFroxels: number;
  readonly maxBytes: number;
  readonly temporal: false | Readonly<ResolvedFogTemporalOptions>;
  /** Present only when air lighting was configured. */
  readonly airLighting?: Readonly<ResolvedFogAirLightingOptions>;
  readonly global?: ResolvedFogHeightMedium;
  readonly volumes: readonly ResolvedFogVolume[];
}

export interface FogGrid {
  readonly columns: number;
  readonly rows: number;
  readonly slices: number;
  readonly pixelSize: number;
  readonly froxels: number;
  readonly bytes: number;
}

const DEFAULTS = {
  pixelSize: 8, slices: 64, maxViewDepth: 80, maxFroxels: 1_048_576,
  maxBytes: 64 * 1024 * 1024,
} as const;
const MAX_COORDINATE = 1_000_000;
const MAX_VOLUMES = 64;
const VOLUME_BYTES = MAX_VOLUMES * 80;
const UNIFORM_BYTES = 384;
const BASE_FROXEL_BYTES = 32;
const TEMPORAL_FROXEL_BYTES = 2 * 16 + 2 * 4;
const MAX_UINT32 = 0xffff_ffff;
/** Changed-caster light-space regions consumed by local history rejection. */
export const FOG_OCCLUDER_REGION_CAPACITY = 32;
/** One header vec4 plus two vec4 per region (light-space rectangle, nearest depth). */
export const FOG_OCCLUDER_REGION_BYTES = 16 + FOG_OCCLUDER_REGION_CAPACITY * 32;

function integer(value: number, min: number, max: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${label} must be an integer in [${min}, ${max}]`);
  }
  return value;
}

function finite(value: number, min: number, max: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw new RangeError(`${label} must be finite and in [${min}, ${max}]`);
  }
  return value;
}

function point(input: Point3, label: string): Point3 {
  if (!Array.isArray(input) || input.length !== 3) throw new RangeError(`${label} must contain three coordinates`);
  return Object.freeze([
    finite(input[0], -MAX_COORDINATE, MAX_COORDINATE, `${label}[0]`),
    finite(input[1], -MAX_COORDINATE, MAX_COORDINATE, `${label}[1]`),
    finite(input[2], -MAX_COORDINATE, MAX_COORDINATE, `${label}[2]`),
  ] as [number, number, number]);
}

function coefficients(input: MediumCoefficients): MediumSnapshot {
  const snapshot = snapshotMediumCoefficients(input);
  if (snapshot.extinction > 10_000 || snapshot.emission.some(channel => channel > 10_000)) {
    throw new RangeError('fog extinction and emission must be at most 10000');
  }
  return snapshot;
}

/** Validate and copy author inputs so later mutation cannot change a rendered frame. */
export function resolveVolumetricFogOptions(options: VolumetricFogOptions = {}): ResolvedVolumetricFogOptions {
  if (!options || typeof options !== 'object') throw new TypeError('fog options must be an object');
  const pixelSize = integer(options.pixelSize ?? DEFAULTS.pixelSize, 1, 128, 'pixelSize');
  const slices = integer(options.slices ?? DEFAULTS.slices, 8, 128, 'slices');
  const maxViewDepth = finite(options.maxViewDepth ?? DEFAULTS.maxViewDepth, 0.1, 1000, 'maxViewDepth');
  const maxFroxels = integer(options.maxFroxels ?? DEFAULTS.maxFroxels, slices, 4_194_304, 'maxFroxels');
  const maxBytes = integer(options.maxBytes ?? DEFAULTS.maxBytes, 1, Number.MAX_SAFE_INTEGER, 'maxBytes');
  let temporal: ResolvedVolumetricFogOptions['temporal'] = false;
  if (options.temporal !== undefined && options.temporal !== false) {
    if (!options.temporal || typeof options.temporal !== 'object' || Array.isArray(options.temporal)) {
      throw new TypeError('temporal fog must be false or an options object');
    }
    const rejection = options.temporal.occluderRejection ?? 'global';
    if (rejection !== 'global' && rejection !== 'local') throw new RangeError('occluderRejection must be global or local');
    temporal = Object.freeze({ maxHistoryFrames: integer(options.temporal.maxHistoryFrames ?? 8,
      2, 32, 'maxHistoryFrames'), occluderRejection: rejection });
  }
  let airLighting: ResolvedVolumetricFogOptions['airLighting'];
  if (options.airLighting !== undefined && options.airLighting !== false) {
    if (!options.airLighting || typeof options.airLighting !== 'object' || Array.isArray(options.airLighting)) {
      throw new TypeError('airLighting must be false or an options object');
    }
    airLighting = resolveFogAirLighting(options.airLighting);
  }

  let global: ResolvedFogHeightMedium | undefined;
  if (options.global !== undefined) {
    if (!options.global || typeof options.global !== 'object') throw new TypeError('global fog must be an object');
    global = Object.freeze({ coefficients: coefficients(options.global.coefficients),
      baseHeight: finite(options.global.baseHeight ?? 0, -MAX_COORDINATE, MAX_COORDINATE, 'baseHeight'),
      heightFalloff: finite(options.global.heightFalloff ?? 0, 0, 100, 'heightFalloff') });
  }

  if (options.volumes !== undefined && (!Array.isArray(options.volumes) || options.volumes.length > MAX_VOLUMES)) {
    throw new RangeError(`volumes must be an array of at most ${MAX_VOLUMES}`);
  }
  const ids = new Set<number>();
  const volumes = (options.volumes ?? []).map((volume, index): ResolvedFogVolume => {
    if (!volume || typeof volume !== 'object') throw new TypeError(`volumes[${index}] must be an object`);
    const id = integer(volume.id, 0, MAX_UINT32, `volumes[${index}].id`);
    if (ids.has(id)) throw new RangeError(`duplicate fog volume id ${id}`);
    ids.add(id);
    if (!volume.shape || typeof volume.shape !== 'object') throw new TypeError(`volumes[${index}].shape`);
    let shape: FogVolume['shape'];
    if (volume.shape.kind === 'box') {
      const min = point(volume.shape.min, `volumes[${index}].min`);
      const max = point(volume.shape.max, `volumes[${index}].max`);
      if (min.some((value, axis) => value >= max[axis])) throw new RangeError('box min must be strictly below max');
      shape = Object.freeze({ kind: 'box', min, max });
    } else if (volume.shape.kind === 'sphere') {
      const center = point(volume.shape.center, `volumes[${index}].center`);
      const radius = finite(volume.shape.radius, 0, MAX_COORDINATE, `volumes[${index}].radius`);
      if (radius <= 0 || Math.fround(radius) === 0) throw new RangeError('sphere radius must be positive in float32');
      shape = Object.freeze({ kind: 'sphere', center, radius });
    } else throw new RangeError(`volumes[${index}].shape kind`);
    return Object.freeze({ id, shape, coefficients: coefficients(volume.coefficients) });
  });
  return Object.freeze({ pixelSize, slices, maxViewDepth, maxFroxels, maxBytes, temporal,
    ...(airLighting ? { airLighting } : {}), ...(global ? { global } : {}), volumes: Object.freeze(volumes) });
}

/** Validate runtime air-lighting controls; omitted fields keep `base` (or defaults). */
export function resolveFogAirLighting(input: FogAirLightingOptions,
  base: ResolvedFogAirLightingOptions = { gi: true, outsideSkyVisibility: 0 }): Readonly<ResolvedFogAirLightingOptions> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new TypeError('airLighting must be an object');
  const gi = input.gi ?? base.gi;
  if (typeof gi !== 'boolean') throw new TypeError('airLighting.gi must be boolean');
  const outsideSkyVisibility = finite(input.outsideSkyVisibility ?? base.outsideSkyVisibility, 0, 1,
    'airLighting.outsideSkyVisibility');
  return Object.freeze({ gi, outsideSkyVisibility });
}

/**
 * CPU reference for the GPU air term: HG convolved with band-limited incoming
 * radiance. `coefficients` holds 9 RGB triples in giShBasis order; `direction`
 * is the view ray (camera towards the sample). HG zonal coefficients are g^l, so
 * the result is exact for stored L2 radiance: sum_l g^l sum_m L_lm Y_lm(v).
 * Returns radiance per unit scattering coefficient, before any clamping.
 */
export function fogAirPhaseRadiance(coefficients: ArrayLike<number>, direction: readonly [number, number, number],
  anisotropy: number): [number, number, number] {
  if (!coefficients || coefficients.length !== 27) throw new RangeError('fog air coefficients must contain 27 values');
  if (!Number.isFinite(anisotropy) || anisotropy < -0.95 || anisotropy > 0.95) throw new RangeError('anisotropy must be in [-0.95, 0.95]');
  const basis = giShBasis(direction);
  const result: [number, number, number] = [0, 0, 0];
  for (let index = 0; index < 9; index++) {
    const band = index === 0 ? 0 : index < 4 ? 1 : 2;
    const scale = basis[index] * anisotropy ** band;
    for (let channel = 0; channel < 3; channel++) {
      const value = coefficients[index * 3 + channel];
      if (!Number.isFinite(value)) throw new RangeError('fog air coefficients must be finite');
      result[channel] += value * scale;
    }
  }
  return result;
}

export interface FogOccluderBounds { readonly min: readonly number[]; readonly max: readonly number[] }

export interface FogOccluderRegions {
  /** FOG_OCCLUDER_REGION_BYTES / 4 floats: header (count, merged, 0, 0) then regions. */
  readonly data: Float32Array;
  readonly count: number;
  /** Regions folded into the last slot by a conservative union after the capacity filled. */
  readonly merged: number;
}

/**
 * Light-space footprints of changed casters for local fog history rejection.
 * Each world AABB (pre-change and post-change bounds are separate entries) becomes
 * a light-NDC rectangle expanded by `marginTexels` shadow texels plus its nearest
 * light depth. A directional shadow test can only change for points whose light-space
 * position lies in that rectangle at or behind that depth, so a cell outside every
 * region keeps its history. Overflow is merged into the last slot, never dropped.
 */
export function fogOccluderRegions(bounds: readonly FogOccluderBounds[], lightMatrix: ArrayLike<number>,
  shadowResolution: number, marginTexels = 2, depthBias = 0.0001): FogOccluderRegions {
  if (!lightMatrix || lightMatrix.length !== 16) throw new RangeError('lightMatrix must contain 16 values');
  integer(shadowResolution, 1, 65_536, 'shadowResolution');
  finite(marginTexels, 0, 1024, 'marginTexels');
  finite(depthBias, 0, 1, 'depthBias');
  const data = new Float32Array(FOG_OCCLUDER_REGION_BYTES / 4);
  const margin = marginTexels * 2 / shadowResolution;
  const m = lightMatrix;
  let count = 0, merged = 0;
  for (const box of bounds) {
    if (!box || box.min?.length !== 3 || box.max?.length !== 3) throw new RangeError('occluder bounds must contain min and max');
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, z0 = Infinity;
    for (let corner = 0; corner < 8; corner++) {
      const x = corner & 1 ? box.max[0] : box.min[0], y = corner & 2 ? box.max[1] : box.min[1];
      const z = corner & 4 ? box.max[2] : box.min[2];
      if (![x, y, z].every(Number.isFinite)) throw new RangeError('occluder bounds must be finite');
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      const px = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
      const py = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
      const pz = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
      x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py); z0 = Math.min(z0, pz);
    }
    if (![x0, x1, y0, y1, z0].every(Number.isFinite)) throw new RangeError('occluder light-space bounds must be finite');
    const region = [x0 - margin, y0 - margin, x1 + margin, y1 + margin, z0 - depthBias];
    let slot = count;
    if (count === FOG_OCCLUDER_REGION_CAPACITY) {
      slot = count - 1; merged++;
      const base = 4 + slot * 8;
      region[0] = Math.min(region[0], data[base]); region[1] = Math.min(region[1], data[base + 1]);
      region[2] = Math.max(region[2], data[base + 2]); region[3] = Math.max(region[3], data[base + 3]);
      region[4] = Math.min(region[4], data[base + 4]);
    } else count++;
    const base = 4 + slot * 8;
    data.set(region.slice(0, 4), base);
    data[base + 4] = region[4];
  }
  data[0] = count; data[1] = merged;
  return { data, count, merged };
}

function halfToFloat(bits: number): number {
  const exponent = (bits >> 10) & 0x1f, mantissa = bits & 0x3ff, sign = bits & 0x8000 ? -1 : 1;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** Saturated support radius in metres: such a cell history is rejected by any caster change. */
export const FOG_HISTORY_UNBOUNDED_SUPPORT = 60000;

/**
 * Per-cell history state. Global rejection stores the plain frame count. Local
 * rejection packs the count into the low 16 bits and an f16 world-space support
 * radius (metres, rounded up) into the high 16 bits: every retained sample lies
 * within that radius of the unjittered cell centre.
 */
export function decodeFogHistoryState(state: number, local: boolean): { count: number; supportRadius: number } {
  if (!Number.isInteger(state) || state < 0 || state > 0xffffffff) throw new RangeError('fog history state must be a u32');
  if (!local) return { count: state, supportRadius: 0 };
  return { count: state & 0xffff, supportRadius: halfToFloat(state >>> 16) };
}

/** Persistent fog-owned bytes independent of the view: frame uniform, local media
 * and, for local history rejection, the changed-caster region uniform. */
export function fogFixedBytes(options: Pick<ResolvedVolumetricFogOptions, 'temporal'>): number {
  return UNIFORM_BYTES + VOLUME_BYTES
    + (options.temporal && options.temporal.occluderRejection === 'local' ? FOG_OCCLUDER_REGION_BYTES : 0);
}

/** Density is one at and below baseHeight, with exponential falloff above it. */
export function fogHeightDensity(y: number, medium: ResolvedFogHeightMedium): number {
  if (!Number.isFinite(y)) throw new RangeError('height must be finite');
  return Math.exp(-medium.heightFalloff * Math.max(y - medium.baseHeight, 0));
}

/** Choose the smallest integer pixel size at or above the requested size that fits both budgets. */
export function selectFogGrid(width: number, height: number, options: ResolvedVolumetricFogOptions): FogGrid {
  integer(width, 1, Number.MAX_SAFE_INTEGER, 'width');
  integer(height, 1, Number.MAX_SAFE_INTEGER, 'height');
  const hdrBytes = width * height * 8;
  if (!Number.isSafeInteger(hdrBytes)) throw new RangeError('HDR dimensions exceed safe byte accounting');
  const fixedBytes = fogFixedBytes(options) + hdrBytes;
  const froxelBytes = BASE_FROXEL_BYTES + (options.temporal ? TEMPORAL_FROXEL_BYTES : 0);
  if (fixedBytes + options.slices * froxelBytes > options.maxBytes) throw new RangeError('fog grid cannot fit 1x1 slices');

  const fits = (pixelSize: number): boolean => {
    const columns = Math.ceil(width / pixelSize), rows = Math.ceil(height / pixelSize);
    const froxels = columns * rows * options.slices;
    return froxels <= options.maxFroxels && fixedBytes + froxels * froxelBytes <= options.maxBytes;
  };
  let low = options.pixelSize, high = Math.max(options.pixelSize, width, height);
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (fits(middle)) high = middle;
    else low = middle + 1;
  }
  const columns = Math.ceil(width / low), rows = Math.ceil(height / low);
  const froxels = columns * rows * options.slices;
  const bytes = fixedBytes + froxels * froxelBytes;
  if (!fits(low) || !Number.isSafeInteger(bytes)) throw new RangeError('fog grid exceeds budget');
  return { columns, rows, slices: options.slices, pixelSize: low, froxels, bytes };
}
