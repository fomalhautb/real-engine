import type { MeshData } from './geometry';

/** Fixed inputs of the G01 record (one high-poly mesh, slow zoom, then orbits).
 * Pure functions with no renderer types: the browser fixture, unit tests and
 * any later capture script share the exact mesh and camera trajectory. */
export type Vec3 = [number, number, number];

export const G01_SEED = 20261031;
/** Camera steps are fixed simulation steps, independent of wall-clock frame pacing. */
export const G01_STEP_SECONDS = 1 / 60;
/** The Renderer's fixed projection (vertical FOV pi/3.2, near 0.1, far 150). */
export const G01_VERTICAL_FOV = Math.PI / 3.2;
export const G01_NEAR = 0.1;

const MAX_GRID_VERTICES = 1 << 20;
/** Relative detail amplitude per octave (frequencies double per octave). */
export const G01_DETAIL_SPECTRUM: readonly number[] = Object.freeze([0.04, 0.02, 0.01, 0.005]);

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

interface Harmonic { u: number; v: number; amplitude: number; phase: number }

/** Seeded shape parameters. Integer frequencies keep both seams smooth; the
 * index grid wraps, so the surface is closed for any parameters. */
export function g01ShapeParameters(seed: number, detailSpectrum: readonly number[] = G01_DETAIL_SPECTRUM) {
  const random = mulberry32(seed);
  const pick = (low: number, high: number) => low + (high - low) * random();
  const integer = (low: number, high: number) => Math.floor(pick(low, high + 1));
  const sign = () => (random() < 0.5 ? -1 : 1);
  const major = pick(1.25, 1.4), tube = pick(0.4, 0.46), lobes = integer(3, 4), lobeDepth = pick(0.08, 0.14);
  const lift = pick(0.12, 0.25), liftFrequency = integer(2, 3), twist = integer(0, 1);
  const ridges: Harmonic[] = Array.from({ length: 3 }, () => ({ u: integer(3, 9), v: integer(1, 3) * sign(),
    amplitude: pick(0.03, 0.07), phase: pick(0, Math.PI * 2) }));
  // Detail octaves, two harmonics each, doubling in frequency. Amplitudes are
  // relative to the tube radius.
  const detail: Harmonic[] = [];
  for (let octave = 0; octave < detailSpectrum.length; octave++) for (let k = 0; k < 2; k++) {
    const scale = 2 ** octave;
    detail.push({ u: integer(12 * scale, 20 * scale), v: integer(3 * scale, 5 * scale) * sign(),
      amplitude: detailSpectrum[octave] * pick(0.75, 1), phase: pick(0, Math.PI * 2) });
  }
  return { major, tube, lobes, lobeDepth, lift, liftFrequency, twist, ridges, detail };
}

/** Closed, watertight torus-topology surface with seeded integer harmonics and
 * multi-octave detail. Triangles: 2 * around * across. Normals are analytic
 * finite differences of the same function, so they are exact per seed. */
export function g01Mesh(seed = G01_SEED, around = 1024, across = 256, detailSpectrum: readonly number[] = G01_DETAIL_SPECTRUM): MeshData {
  if (!Number.isSafeInteger(around) || !Number.isSafeInteger(across) || around < 16 || across < 8
    || around * across > MAX_GRID_VERTICES) throw new RangeError('G01_MESH_RESOLUTION');
  if (!Array.isArray(detailSpectrum) || detailSpectrum.length > 6
    || !detailSpectrum.every(value => Number.isFinite(value) && value >= 0 && value <= 0.05)) throw new RangeError('G01_MESH_DETAIL');
  const shape = g01ShapeParameters(seed, detailSpectrum);
  const point = (u: number, v: number): Vec3 => {
    let radius = 1;
    for (const r of shape.ridges) radius += r.amplitude * Math.cos(r.u * u + r.v * v + r.phase);
    for (const d of shape.detail) radius += d.amplitude * Math.cos(d.u * u + d.v * v + d.phase);
    const t = shape.tube * radius, w = v + shape.twist * u;
    const radial = shape.major * (1 + shape.lobeDepth * Math.cos(shape.lobes * u)) + t * Math.cos(w);
    return [radial * Math.cos(u), shape.lift * Math.sin(shape.liftFrequency * u) + t * Math.sin(w), radial * Math.sin(u)];
  };
  const vertexCount = around * across, vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
  const indices = new Uint32Array(vertexCount * 6);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  const h = 1e-5;
  for (let i = 0; i < around; i++) for (let j = 0; j < across; j++) {
    const u = i * Math.PI * 2 / around, v = j * Math.PI * 2 / across, p = point(u, v);
    const pu = point(u + h, v), mu = point(u - h, v), pv = point(u, v + h), mv = point(u, v - h);
    const du = [pu[0] - mu[0], pu[1] - mu[1], pu[2] - mu[2]], dv = [pv[0] - mv[0], pv[1] - mv[1], pv[2] - mv[2]];
    const n = [dv[1] * du[2] - dv[2] * du[1], dv[2] * du[0] - dv[0] * du[2], dv[0] * du[1] - dv[1] * du[0]];
    const length = Math.hypot(n[0], n[1], n[2]);
    if (!(length > 0)) throw new Error('G01_MESH_DEGENERATE_NORMAL');
    const offset = (i * across + j) * 80;
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(offset + axis * 4, p[axis], true);
      view.setFloat32(offset + 16 + axis * 4, n[axis] / length, true);
    }
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 48, 1, true);
    // Bounds of the stored float32 positions, not of the float64 function.
    for (let axis = 0; axis < 3; axis++) {
      const stored = view.getFloat32(offset + axis * 4, true);
      min[axis] = Math.min(min[axis], stored); max[axis] = Math.max(max[axis], stored);
    }
  }
  let cursor = 0;
  for (let i = 0; i < around; i++) for (let j = 0; j < across; j++) {
    const a = i * across + j, b = ((i + 1) % around) * across + j;
    const c = ((i + 1) % around) * across + (j + 1) % across, d = i * across + (j + 1) % across;
    indices.set([a, d, b, b, d, c], cursor); cursor += 6;
  }
  return { vertices, indices, vertexCount, bounds: { min, max } };
}

export type G01SegmentKind = 'zoom' | 'orbit' | 'dolly';
export interface G01Segment {
  readonly name: string;
  readonly kind: G01SegmentKind;
  readonly seconds: number;
  /** Eye distance from the target at the segment's start and end, in mesh units. */
  readonly fromDistance: number;
  readonly toDistance: number;
  /** Azimuth about +Y and elevation above the XZ plane, radians. */
  readonly fromAzimuth: number;
  readonly toAzimuth: number;
  readonly fromElevation: number;
  readonly toElevation: number;
}
export interface G01Pose {
  readonly step: number;
  readonly seconds: number;
  readonly segment: string;
  readonly kind: G01SegmentKind;
  readonly distance: number;
  readonly azimuth: number;
  readonly elevation: number;
  readonly eye: Vec3;
  readonly target: Vec3;
}
export interface G01TrajectoryOptions {
  /** Defaults reproduce the frozen G01 trajectory; tests may shorten it. */
  readonly zoomSeconds?: number;
  readonly orbitSeconds?: number;
  readonly dollySeconds?: number;
  readonly farDistance?: number;
  readonly nearDistance?: number;
  readonly midDistance?: number;
  readonly outerDistance?: number;
}

const DEFAULT_TRAJECTORY: Required<G01TrajectoryOptions> = Object.freeze({
  zoomSeconds: 10, orbitSeconds: 6, dollySeconds: 2, farDistance: 30, nearDistance: 2.6, midDistance: 6.5, outerDistance: 16,
});

/** The fixed G01 camera: a slow exponential zoom from far to near at a fixed
 * azimuth, then a full orbit at three distances joined by eased dollies. The
 * target is the mesh origin; up is +Y; elevations stay well below the pole. */
export class G01Trajectory {
  readonly segments: readonly G01Segment[];
  readonly options: Readonly<Required<G01TrajectoryOptions>>;
  readonly steps: number;
  private readonly starts: number[];

  constructor(input: G01TrajectoryOptions = {}) {
    const options = { ...DEFAULT_TRAJECTORY, ...input };
    for (const [key, value] of Object.entries(options)) {
      if (!Number.isFinite(value) || value <= 0 || value > 1000) throw new RangeError('G01_TRAJECTORY_' + key);
    }
    if (!(options.nearDistance < options.midDistance && options.midDistance < options.outerDistance
      && options.nearDistance < options.farDistance)) throw new RangeError('G01_TRAJECTORY_DISTANCES');
    this.options = Object.freeze(options);
    const azimuth = 0.35, turn = Math.PI * 2, low = 0.36, mid = 0.52, high = 0.7;
    const segment = (name: string, kind: G01SegmentKind, seconds: number, fromDistance: number, toDistance: number,
      fromAzimuth: number, toAzimuth: number, fromElevation: number, toElevation: number): G01Segment =>
      Object.freeze({ name, kind, seconds, fromDistance, toDistance, fromAzimuth, toAzimuth, fromElevation, toElevation });
    this.segments = Object.freeze([
      segment('zoom-in', 'zoom', options.zoomSeconds, options.farDistance, options.nearDistance, azimuth, azimuth, mid, mid),
      segment('orbit-near', 'orbit', options.orbitSeconds, options.nearDistance, options.nearDistance, azimuth, azimuth + turn, mid, mid),
      segment('dolly-mid', 'dolly', options.dollySeconds, options.nearDistance, options.midDistance, azimuth + turn, azimuth + turn, mid, low),
      segment('orbit-mid', 'orbit', options.orbitSeconds, options.midDistance, options.midDistance, azimuth + turn, azimuth, low, low),
      segment('dolly-outer', 'dolly', options.dollySeconds, options.midDistance, options.outerDistance, azimuth, azimuth, low, high),
      segment('orbit-outer', 'orbit', options.orbitSeconds, options.outerDistance, options.outerDistance, azimuth, azimuth + turn, high, high),
    ]);
    // Each segment owns its steps [start, start + n); the final pose closes the path.
    this.starts = [];
    let steps = 0;
    for (const entry of this.segments) {
      this.starts.push(steps);
      steps += Math.round(entry.seconds / G01_STEP_SECONDS);
    }
    this.steps = steps + 1;
  }

  /** Pose at a fixed simulation step, 0 .. steps - 1. */
  pose(step: number): G01Pose {
    if (!Number.isSafeInteger(step) || step < 0 || step >= this.steps) throw new RangeError('G01_TRAJECTORY_STEP');
    let index = this.segments.length - 1;
    while (index > 0 && this.starts[index] > step) index--;
    const entry = this.segments[index], count = Math.round(entry.seconds / G01_STEP_SECONDS);
    const s = Math.min(1, (step - this.starts[index]) / count);
    // Zoom: constant relative speed (exponential distance). Orbit: constant
    // angular speed. Dolly: smoothstep in log distance and elevation.
    const blend = entry.kind === 'dolly' ? s * s * (3 - 2 * s) : s;
    const distance = entry.fromDistance * (entry.toDistance / entry.fromDistance) ** blend;
    const azimuth = entry.fromAzimuth + (entry.toAzimuth - entry.fromAzimuth) * blend;
    const elevation = entry.fromElevation + (entry.toElevation - entry.fromElevation) * blend;
    const eye: Vec3 = [distance * Math.cos(elevation) * Math.cos(azimuth), distance * Math.sin(elevation),
      distance * Math.cos(elevation) * Math.sin(azimuth)];
    return Object.freeze({ step, seconds: step * G01_STEP_SECONDS, segment: entry.name, kind: entry.kind,
      distance, azimuth, elevation, eye, target: [0, 0, 0] as Vec3 });
  }
}

/** Pixels per unit length at unit view depth for a viewport of this height. */
export function g01ProjectionScale(height: number): number {
  if (!Number.isFinite(height) || height <= 0) throw new RangeError('G01_VIEWPORT');
  return height / (2 * Math.tan(G01_VERTICAL_FOV / 2));
}
