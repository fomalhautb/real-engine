import { snapshotMediumCoefficients, type MediumCoefficients, type MediumSnapshot } from './participating-media';

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

export interface VolumetricFogOptions {
  readonly pixelSize?: number;
  readonly slices?: number;
  readonly maxViewDepth?: number;
  readonly maxFroxels?: number;
  readonly maxBytes?: number;
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

export interface ResolvedVolumetricFogOptions {
  readonly pixelSize: number;
  readonly slices: number;
  readonly maxViewDepth: number;
  readonly maxFroxels: number;
  readonly maxBytes: number;
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
const UNIFORM_BYTES = 256;
const MAX_UINT32 = 0xffff_ffff;

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
  return Object.freeze({ pixelSize, slices, maxViewDepth, maxFroxels, maxBytes,
    ...(global ? { global } : {}), volumes: Object.freeze(volumes) });
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
  const fixedBytes = UNIFORM_BYTES + VOLUME_BYTES + hdrBytes;
  if (fixedBytes + options.slices * 32 > options.maxBytes) throw new RangeError('fog grid cannot fit 1x1 slices');

  const fits = (pixelSize: number): boolean => {
    const columns = Math.ceil(width / pixelSize), rows = Math.ceil(height / pixelSize);
    const froxels = columns * rows * options.slices;
    return froxels <= options.maxFroxels && fixedBytes + froxels * 32 <= options.maxBytes;
  };
  let low = options.pixelSize, high = Math.max(options.pixelSize, width, height);
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (fits(middle)) high = middle;
    else low = middle + 1;
  }
  const columns = Math.ceil(width / low), rows = Math.ceil(height / low);
  const froxels = columns * rows * options.slices;
  const bytes = fixedBytes + froxels * 32;
  if (!fits(low) || !Number.isSafeInteger(bytes)) throw new RangeError('fog grid exceeds budget');
  return { columns, rows, slices: options.slices, pixelSize: low, froxels, bytes };
}
