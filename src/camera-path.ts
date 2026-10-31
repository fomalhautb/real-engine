/**
 * Camera paths for large-scene walkthroughs (format real-engine-camera-path,
 * written by scripts/apply-fbx-material-map.mjs from an FBX camera animation
 * or authored by hand). Positions follow a uniform Catmull-Rom curve through
 * the samples; forward and up are normalized linear blends. Pure functions,
 * no renderer types.
 */

export type Vec3 = [number, number, number];

export interface CameraPathSample {
  time: number;
  position: Vec3;
  forward: Vec3;
  up: Vec3;
  key?: boolean;
}

export interface CameraPath {
  name: string;
  verticalFovRadians: number | null;
  samples: CameraPathSample[];
  duration: number;
}

export interface CameraPose { eye: Vec3; target: Vec3; up: Vec3 }

const MAX_SAMPLES = 100_000;

function vector(value: unknown, label: string): Vec3 {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) throw new Error('CAMERA_PATH_VECTOR: ' + label);
  return [value[0], value[1], value[2]];
}

function normalize(v: Vec3, label: string): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]);
  if (!(length > 1e-9)) throw new Error('CAMERA_PATH_DIRECTION: ' + label);
  return [v[0] / length, v[1] / length, v[2] / length];
}

/** Validates a parsed camera-path document: >= 2 samples, strictly increasing times, unit directions. */
export function parseCameraPath(input: unknown): CameraPath {
  const doc = input as Record<string, unknown> | null;
  if (!doc || doc.format !== 'real-engine-camera-path' || !Array.isArray(doc.samples)) throw new Error('CAMERA_PATH_FORMAT');
  if (doc.samples.length < 2 || doc.samples.length > MAX_SAMPLES) throw new Error('CAMERA_PATH_SAMPLE_COUNT');
  const samples = doc.samples.map((raw: unknown, i: number): CameraPathSample => {
    const sample = raw as Record<string, unknown>;
    if (!sample || !Number.isFinite(sample.time)) throw new Error('CAMERA_PATH_TIME: ' + i);
    return { time: sample.time as number, position: vector(sample.position, `samples[${i}].position`),
      forward: normalize(vector(sample.forward, `samples[${i}].forward`), `samples[${i}].forward`),
      up: normalize(vector(sample.up, `samples[${i}].up`), `samples[${i}].up`), ...(sample.key === true ? { key: true } : {}) };
  });
  for (let i = 1; i < samples.length; i++) if (!(samples[i].time > samples[i - 1].time)) throw new Error('CAMERA_PATH_TIME_ORDER: ' + i);
  const fov = doc.verticalFovRadians;
  if (fov !== undefined && (typeof fov !== 'number' || !(fov > 0.05 && fov < 3))) throw new Error('CAMERA_PATH_FOV');
  return { name: typeof doc.name === 'string' ? doc.name : 'camera-path', verticalFovRadians: typeof fov === 'number' ? fov : null,
    samples, duration: samples[samples.length - 1].time - samples[0].time };
}

function blend(a: Vec3, b: Vec3, t: number, label: string): Vec3 {
  const v: Vec3 = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  // Opposite unit vectors blend through zero; fall back to the nearer sample.
  return Math.hypot(v[0], v[1], v[2]) > 1e-6 ? normalize(v, label) : (t < 0.5 ? a : b);
}

/** Camera pose at `time` seconds from the first sample, clamped to the path (wrap with `loop`). */
export function sampleCameraPath(path: CameraPath, time: number, loop = true): CameraPose {
  if (!Number.isFinite(time)) throw new Error('CAMERA_PATH_SAMPLE_TIME');
  const s = path.samples, start = s[0].time;
  let t = loop ? ((time % path.duration) + path.duration) % path.duration : Math.min(Math.max(time, 0), path.duration);
  t += start;
  let lo = 0, hi = s.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s[mid].time <= t) lo = mid; else hi = mid; }
  const a = s[lo], b = s[hi], u = Math.min(1, Math.max(0, (t - a.time) / (b.time - a.time)));
  // Missing neighbours at the ends are reflected (2 * p1 - p2), so evenly
  // spaced collinear samples stay on their line.
  const p1 = a.position, p2 = b.position;
  const p0 = lo > 0 ? s[lo - 1].position : [0, 1, 2].map(c => 2 * p1[c] - p2[c]) as Vec3;
  const p3 = hi < s.length - 1 ? s[hi + 1].position : [0, 1, 2].map(c => 2 * p2[c] - p1[c]) as Vec3;
  const u2 = u * u, u3 = u2 * u;
  const eye = [0, 1, 2].map(c => 0.5 * ((2 * p1[c]) + (-p0[c] + p2[c]) * u + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * u2
    + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * u3)) as Vec3;
  const forward = blend(a.forward, b.forward, u, 'forward'), up = blend(a.up, b.up, u, 'up');
  return { eye, target: [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]], up };
}

/** Sample indices offered as fixed views: samples marked `key`, else up to `max` evenly spaced. */
export function cameraPathKeyframes(path: CameraPath, max = 8): number[] {
  if (!Number.isSafeInteger(max) || max < 1 || max > 64) throw new Error('CAMERA_PATH_KEYFRAME_COUNT');
  const marked = path.samples.flatMap((sample, i) => sample.key ? [i] : []);
  if (marked.length) return marked.slice(0, max);
  const count = Math.min(max, path.samples.length);
  return Array.from({ length: count }, (_, i) => Math.round(i * (path.samples.length - 1) / Math.max(1, count - 1)));
}
