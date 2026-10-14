import { identity, lookAt, multiply, type Mat4 } from './math';

type Vec3 = readonly [number, number, number];

export interface SunShadowBounds {
  readonly min: Vec3;
  readonly max: Vec3;
}

export interface SunShadowOptions {
  readonly resolution?: number;
  readonly focusBounds?: SunShadowBounds;
  readonly padding?: number;
  readonly maxBytes?: number;
}

export interface ResolvedSunShadowOptions {
  readonly resolution: number;
  readonly focusBounds?: SunShadowBounds;
  readonly padding: number;
  readonly maxBytes: number;
  readonly bytes: number;
}

export interface SunShadowProjection {
  readonly matrix: Mat4;
  readonly texelWorldSize: number;
  readonly width: number;
  readonly depth: number;
}

const DEFAULT_RESOLUTION = 1024;
const MAX_RESOLUTION = 8192;
const DEFAULT_PADDING = 0.25;
const DEFAULT_MAX_BYTES = 4 * MAX_RESOLUTION * MAX_RESOLUTION;
// Bounding world coordinates keeps f32 transforms and texel snapping meaningful.
const MAX_COORDINATE = 1_000_000;

function finiteF32(value: number): boolean {
  return Number.isFinite(value) && Number.isFinite(Math.fround(value));
}

function validateBounds(bounds: SunShadowBounds): SunShadowBounds {
  if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max) ||
      bounds.min.length !== 3 || bounds.max.length !== 3) throw new RangeError('INVALID_SUN_SHADOW_BOUNDS');
  for (let axis = 0; axis < 3; axis++) {
    const min = bounds.min[axis], max = bounds.max[axis];
    if (!finiteF32(min) || !finiteF32(max) || Math.abs(min) > MAX_COORDINATE ||
        Math.abs(max) > MAX_COORDINATE || min > max) throw new RangeError('INVALID_SUN_SHADOW_BOUNDS');
  }
  return Object.freeze({
    min: Object.freeze([...bounds.min]) as unknown as Vec3,
    max: Object.freeze([...bounds.max]) as unknown as Vec3,
  });
}

/** Resolve and budget a depth32float directional shadow before creating its texture. */
export function resolveSunShadowOptions(
  options: SunShadowOptions = {}, maxTextureDimension2D = MAX_RESOLUTION,
): ResolvedSunShadowOptions {
  const resolution = options.resolution ?? DEFAULT_RESOLUTION;
  if (!Number.isSafeInteger(maxTextureDimension2D) || maxTextureDimension2D < 1 ||
      !Number.isSafeInteger(resolution) || resolution < 256 || resolution > MAX_RESOLUTION ||
      (resolution & (resolution - 1)) !== 0 || resolution > maxTextureDimension2D) {
    throw new RangeError('INVALID_SUN_SHADOW_RESOLUTION');
  }
  const padding = options.padding ?? DEFAULT_PADDING;
  if (!finiteF32(padding) || padding < 0 || padding > MAX_COORDINATE) {
    throw new RangeError('INVALID_SUN_SHADOW_PADDING');
  }
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new RangeError('INVALID_SUN_SHADOW_BUDGET');
  const bytes = 4 * resolution * resolution;
  if (bytes > maxBytes) throw new RangeError('SUN_SHADOW_BUDGET_EXCEEDED');
  const focusBounds = options.focusBounds === undefined ? undefined : validateBounds(options.focusBounds);
  return Object.freeze({ resolution, focusBounds, padding, maxBytes, bytes });
}

/** World to light clip coordinates, with WebGPU depth in [0, 1]. */
export function sunShadowProjection(direction: Vec3, options: ResolvedSunShadowOptions): SunShadowProjection {
  if (!Array.isArray(direction) || direction.length !== 3 || !direction.every(finiteF32) ||
      Math.hypot(...direction) < 1e-6) throw new RangeError('INVALID_SUN_DIRECTION');
  const length = Math.hypot(...direction);
  const d: Vec3 = [direction[0] / length, direction[1] / length, direction[2] / length];
  const up: Vec3 = Math.abs(d[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];

  if (!options.focusBounds) {
    // Preserve the original projection bit for bit when no focus box is given.
    const target: Vec3 = [0, 0, -3];
    const eye = target.map((value, axis) => value + d[axis] * 40) as [number, number, number];
    const ortho = identity();
    ortho[0] = 1 / 25; ortho[5] = 1 / 25; ortho[10] = -1 / 70; ortho[14] = -0.1 / 70;
    return { matrix: multiply(ortho, lookAt(eye, target, up)), texelWorldSize: 50 / options.resolution,
      width: 50, depth: 70 };
  }

  // A rotation-only view fixes the light basis in world space. Translation of
  // the focus box cannot rotate it or alter the texel grid's origin.
  const view = lookAt(d, [0, 0, 0], up);
  view[12] = 0; view[13] = 0; view[14] = 0;
  const { min, max } = options.focusBounds;
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) {
    for (let axis = 0; axis < 3; axis++) {
      const value = view[axis] * x + view[4 + axis] * y + view[8 + axis] * z;
      low[axis] = Math.min(low[axis], value);
      high[axis] = Math.max(high[axis], value);
    }
  }
  // f32 multiplication/translation can round near the clip edge. Reserve a
  // small numerical margin even when callers request zero physical padding.
  const numericalGuard = Math.max(0.001, Math.max(...min.map(Math.abs), ...max.map(Math.abs)) * 1e-6);
  const span = Math.max(high[0] - low[0], high[1] - low[1]);
  // Solving width = span + 2*margin + one texel leaves half a texel per side
  // for nearest-grid snapping, so that snapping can never cut off a corner.
  const width = (span + 2 * (options.padding + numericalGuard)) / (1 - 1 / options.resolution);
  const texelWorldSize = width / options.resolution;
  const centerX = Math.round(((low[0] + high[0]) / 2) / texelWorldSize) * texelWorldSize;
  const centerY = Math.round(((low[1] + high[1]) / 2) / texelWorldSize) * texelWorldSize;
  const depth = high[2] - low[2] + 2 * (options.padding + numericalGuard);
  const ortho = identity();
  ortho[0] = 2 / width;
  ortho[5] = 2 / width;
  ortho[10] = -1 / depth;
  ortho[12] = -2 * centerX / width;
  ortho[13] = -2 * centerY / width;
  ortho[14] = (high[2] + options.padding + numericalGuard) / depth;
  const matrix = multiply(ortho, view);
  if (!matrix.every(Number.isFinite)) throw new RangeError('SUN_SHADOW_PROJECTION_OUT_OF_RANGE');
  return { matrix, texelWorldSize, width, depth };
}
