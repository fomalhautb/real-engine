/**
 * Typed perspective projection shared by the renderer's main view, geometry
 * LOD error scale, texture demand, volumetric fog and reflections. Values are
 * validated and bounded once; defaults reproduce the original fixed camera
 * (vertical FOV π/3.2, near 0.1, far 150).
 */
export interface CameraProjectionOptions {
  /** Vertical field of view in radians, [1°, 170°]. Default π/3.2. */
  verticalFovRadians?: number;
  /** Near clip distance in scene units, [0.01, 10]. Default 0.1. */
  near?: number;
  /** Far clip distance in scene units, (near, 100000], at most 1e6 × near. Default 150. */
  far?: number;
}

export interface CameraProjection {
  readonly verticalFovRadians: number;
  readonly near: number;
  readonly far: number;
}

export const DEFAULT_CAMERA_PROJECTION: CameraProjection = Object.freeze({ verticalFovRadians: Math.PI / 3.2, near: 0.1, far: 150 });

const MIN_FOV = Math.PI / 180, MAX_FOV = (170 * Math.PI) / 180;

export function resolveCameraProjection(options: CameraProjectionOptions | undefined): CameraProjection {
  if (options === undefined) return DEFAULT_CAMERA_PROJECTION;
  if (typeof options !== 'object' || options === null) throw new Error('INVALID_PROJECTION');
  const verticalFovRadians = options.verticalFovRadians ?? DEFAULT_CAMERA_PROJECTION.verticalFovRadians;
  const near = options.near ?? DEFAULT_CAMERA_PROJECTION.near;
  const far = options.far ?? DEFAULT_CAMERA_PROJECTION.far;
  if (typeof verticalFovRadians !== 'number' || !Number.isFinite(verticalFovRadians) || verticalFovRadians < MIN_FOV || verticalFovRadians > MAX_FOV) {
    throw new Error('INVALID_PROJECTION: verticalFovRadians must be in [1°, 170°]');
  }
  if (typeof near !== 'number' || !Number.isFinite(near) || near < 0.01 || near > 10) throw new Error('INVALID_PROJECTION: near must be in [0.01, 10]');
  if (typeof far !== 'number' || !Number.isFinite(far) || far <= near || far > 100_000 || far > near * 1e6) {
    throw new Error('INVALID_PROJECTION: far must be in (near, min(100000, 1e6 × near)]');
  }
  return Object.freeze({ verticalFovRadians, near, far });
}

/** Pixels per scene unit at unit view distance for a viewport `height` pixels tall. */
export function projectionScale(projection: CameraProjection, height: number): number {
  return height / (2 * Math.tan(projection.verticalFovRadians / 2));
}
