import { multiply, quaternionMatrix } from './math';

/** Engine-owned animation data; no renderer or asset-library object types cross this boundary. */
export interface AnimationNode {
  parent: number | null;
  translation: readonly [number, number, number];
  rotation: readonly [number, number, number, number];
  scale: readonly [number, number, number];
  matrix?: Float32Array;
  weights: readonly number[];
}

export interface AnimationTrack {
  node: number;
  path: 'translation' | 'rotation' | 'scale' | 'weights';
  interpolation: 'STEP' | 'LINEAR' | 'CUBICSPLINE';
  times: Float32Array;
  values: Float32Array;
  components: number;
}

export interface AnimationClip {
  name: string;
  tracks: readonly AnimationTrack[];
  duration: number;
}

export interface AnimationPose {
  local: Float32Array[];
  world: Float32Array[];
  weights: number[][];
}

const MAX_NODES = 4096;
const MAX_TRACKS = 65536;
const MAX_KEY_VALUES = 16_000_000;

function finite(value: number, label: string): void {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
}

function vector(values: readonly number[], count: number, label: string): number[] {
  if (values.length !== count) throw new RangeError(`${label} must have ${count} components`);
  for (const value of values) finite(value, label);
  return [...values];
}

function normalizeQuaternion(values: ArrayLike<number>): [number, number, number, number] {
  const length = Math.hypot(values[0], values[1], values[2], values[3]);
  if (!Number.isFinite(length) || length === 0) throw new RangeError('rotation quaternion must be finite and nonzero');
  return [values[0] / length, values[1] / length, values[2] / length, values[3] / length];
}

function cloneNode(node: AnimationNode, index: number, nodeCount: number): AnimationNode {
  const label = `node ${index}`;
  if (node.parent !== null && (!Number.isInteger(node.parent) || node.parent < 0 || node.parent >= nodeCount || node.parent === index)) {
    throw new RangeError(`${label} has an invalid parent`);
  }
  const translation = vector(node.translation, 3, `${label} translation`) as [number, number, number];
  const rotation = vector(node.rotation, 4, `${label} rotation`) as [number, number, number, number];
  normalizeQuaternion(rotation);
  const scale = vector(node.scale, 3, `${label} scale`) as [number, number, number];
  const weights = [...node.weights];
  for (const weight of weights) finite(weight, `${label} weight`);
  let matrix: Float32Array | undefined;
  if (node.matrix !== undefined) {
    if (!(node.matrix instanceof Float32Array) || node.matrix.length !== 16) throw new RangeError(`${label} matrix must be a 4x4 Float32Array`);
    matrix = new Float32Array(node.matrix);
    for (const value of matrix) finite(value, `${label} matrix`);
    if (matrix[3] !== 0 || matrix[7] !== 0 || matrix[11] !== 0 || matrix[15] !== 1) {
      throw new RangeError(`${label} matrix must be affine`);
    }
    const a = matrix;
    const determinant = a[0] * (a[5] * a[10] - a[9] * a[6])
      - a[4] * (a[1] * a[10] - a[9] * a[2])
      + a[8] * (a[1] * a[6] - a[5] * a[2]);
    if (!Number.isFinite(determinant) || determinant === 0) throw new RangeError(`${label} matrix must be nonsingular`);
  }
  return { parent: node.parent, translation, rotation, scale, matrix, weights };
}

function cloneTrack(track: AnimationTrack, nodes: readonly AnimationNode[], label: string): AnimationTrack {
  if (!Number.isInteger(track.node) || track.node < 0 || track.node >= nodes.length) throw new RangeError(`${label} has an invalid node`);
  if (!['translation', 'rotation', 'scale', 'weights'].includes(track.path)) throw new RangeError(`${label} has an invalid path`);
  if (!['STEP', 'LINEAR', 'CUBICSPLINE'].includes(track.interpolation)) throw new RangeError(`${label} has an invalid interpolation`);
  if (!(track.times instanceof Float32Array) || !(track.values instanceof Float32Array)) throw new TypeError(`${label} times and values must be Float32Array`);
  if (track.times.length === 0) throw new RangeError(`${label} must contain at least one key`);
  if (!Number.isInteger(track.components) || track.components < 1) throw new RangeError(`${label} has an invalid component count`);
  const expectedComponents = track.path === 'weights' ? nodes[track.node].weights.length : track.path === 'rotation' ? 4 : 3;
  if (track.components !== expectedComponents) throw new RangeError(`${label} component count does not match its node path`);
  if (nodes[track.node].matrix && track.path !== 'weights') throw new RangeError(`${label} cannot animate TRS on a matrix node`);
  const stride = track.components * (track.interpolation === 'CUBICSPLINE' ? 3 : 1);
  if (track.values.length !== track.times.length * stride) throw new RangeError(`${label} has an invalid value count`);
  const times = new Float32Array(track.times);
  const values = new Float32Array(track.values);
  for (let i = 0; i < times.length; i++) {
    finite(times[i], `${label} time`);
    if (times[i] < 0 || (i > 0 && times[i] <= times[i - 1])) throw new RangeError(`${label} times must be nonnegative and strictly increasing`);
  }
  for (const value of values) finite(value, `${label} value`);
  if (track.path === 'rotation') {
    for (let key = 0; key < times.length; key++) {
      const start = key * stride + (track.interpolation === 'CUBICSPLINE' ? 4 : 0);
      normalizeQuaternion(values.subarray(start, start + 4));
    }
  }
  return { node: track.node, path: track.path, interpolation: track.interpolation, times, values, components: track.components };
}

function cloneClip(clip: AnimationClip): AnimationClip {
  return {
    name: clip.name,
    duration: clip.duration,
    tracks: clip.tracks.map((track) => ({ ...track, times: new Float32Array(track.times), values: new Float32Array(track.values) })),
  };
}

function localMatrix(node: AnimationNode, translation: readonly number[], rotation: readonly number[], scale: readonly number[]): Float32Array {
  if (node.matrix) return new Float32Array(node.matrix);
  const result = quaternionMatrix(rotation[0], rotation[1], rotation[2], rotation[3]);
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) result[col * 4 + row] *= scale[col];
  }
  result[12] = translation[0]; result[13] = translation[1]; result[14] = translation[2];
  return result;
}

function slerp(a: readonly number[], b: readonly number[], fraction: number): number[] {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const rhs = dot < 0 ? b.map((value) => -value) : [...b];
  if (dot < 0) dot = -dot;
  dot = Math.min(1, dot);
  if (dot > 0.9995) return normalizeQuaternion(a.map((value, i) => value + fraction * (rhs[i] - value)));
  const angle = Math.acos(dot);
  const denominator = Math.sin(angle);
  const before = Math.sin((1 - fraction) * angle) / denominator;
  const after = Math.sin(fraction * angle) / denominator;
  return normalizeQuaternion(a.map((value, i) => before * value + after * rhs[i]));
}

function keyValue(track: AnimationTrack, key: number): number[] {
  const offset = key * track.components * (track.interpolation === 'CUBICSPLINE' ? 3 : 1)
    + (track.interpolation === 'CUBICSPLINE' ? track.components : 0);
  return Array.from(track.values.subarray(offset, offset + track.components));
}

function sampleTrack(track: AnimationTrack, time: number): number[] {
  const { times, components, values } = track;
  if (time <= times[0]) return keyValue(track, 0);
  const last = times.length - 1;
  if (time >= times[last]) return keyValue(track, last);
  let low = 0, high = last;
  while (high - low > 1) {
    const mid = (low + high) >>> 1;
    if (times[mid] <= time) low = mid;
    else high = mid;
  }
  if (track.interpolation === 'STEP') return keyValue(track, low);
  const dt = times[high] - times[low];
  const t = (time - times[low]) / dt;
  const a = keyValue(track, low), b = keyValue(track, high);
  if (track.interpolation === 'LINEAR') {
    if (track.path === 'rotation') return slerp(normalizeQuaternion(a), normalizeQuaternion(b), t);
    return a.map((value, i) => value + (b[i] - value) * t);
  }
  // glTF cubic keys are [in tangent, value, out tangent]; tangents are derivatives per second.
  const stride = components * 3;
  const t2 = t * t, t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  const result = a.map((value, i) =>
    h00 * value + h10 * dt * values[low * stride + 2 * components + i]
    + h01 * b[i] + h11 * dt * values[high * stride + i]);
  return track.path === 'rotation' ? normalizeQuaternion(result) : result;
}

/** Evaluates independent node TRS and morph channels into column-major local/world matrices. */
export class AnimationRig {
  private readonly nodes: readonly AnimationNode[];
  private readonly ownedClips: readonly AnimationClip[];
  private readonly order: readonly number[];

  constructor(nodes: readonly AnimationNode[], clips: readonly AnimationClip[]) {
    if (nodes.length > MAX_NODES) throw new RangeError(`animation rig exceeds ${MAX_NODES} nodes`);
    this.nodes = nodes.map((node, index) => cloneNode(node, index, nodes.length));
    const children: number[][] = Array.from({ length: nodes.length }, () => []);
    const order: number[] = [];
    for (let i = 0; i < this.nodes.length; i++) {
      const parent = this.nodes[i].parent;
      if (parent === null) order.push(i);
      else children[parent].push(i);
    }
    for (let i = 0; i < order.length; i++) order.push(...children[order[i]]);
    if (order.length !== nodes.length) throw new RangeError('animation node hierarchy contains a cycle');
    this.order = order;

    let trackCount = 0, valueCount = 0;
    this.ownedClips = clips.map((clip, clipIndex) => {
      if (typeof clip.name !== 'string') throw new TypeError(`clip ${clipIndex} name must be a string`);
      finite(clip.duration, `clip ${clipIndex} duration`);
      if (clip.duration < 0) throw new RangeError(`clip ${clipIndex} duration must be nonnegative`);
      trackCount += clip.tracks.length;
      if (trackCount > MAX_TRACKS) throw new RangeError(`animation rig exceeds ${MAX_TRACKS} tracks`);
      const channels = new Set<string>();
      const tracks = clip.tracks.map((track, trackIndex) => {
        const label = `clip ${clipIndex} track ${trackIndex}`;
        const channel = `${track.node}:${track.path}`;
        if (channels.has(channel)) throw new RangeError(`${label} duplicates a channel`);
        channels.add(channel);
        valueCount += track.values.length;
        if (valueCount > MAX_KEY_VALUES) throw new RangeError(`animation rig exceeds ${MAX_KEY_VALUES} key values`);
        const copy = cloneTrack(track, this.nodes, label);
        if (copy.times[copy.times.length - 1] > clip.duration) throw new RangeError(`${label} extends beyond clip duration`);
        return copy;
      });
      return { name: clip.name, duration: clip.duration, tracks };
    });
  }

  /** Fresh copies keep callers from changing validated keys after construction. */
  get clips(): readonly AnimationClip[] { return this.ownedClips.map(cloneClip); }

  sample(time: number, clipIndex = 0, loop = false): AnimationPose {
    finite(time, 'sample time');
    if (!Number.isInteger(clipIndex) || clipIndex < 0 || (this.ownedClips.length > 0 && clipIndex >= this.ownedClips.length)
      || (this.ownedClips.length === 0 && clipIndex !== 0)) throw new RangeError('invalid animation clip index');
    const clip = this.ownedClips[clipIndex];
    const sampleTime = clip && loop && clip.duration > 0 ? ((time % clip.duration) + clip.duration) % clip.duration : time;
    const translations = this.nodes.map((node) => [...node.translation]);
    const rotations = this.nodes.map((node) => [...node.rotation]);
    const scales = this.nodes.map((node) => [...node.scale]);
    const weights = this.nodes.map((node) => [...node.weights]);
    for (const track of clip?.tracks ?? []) {
      const value = sampleTrack(track, sampleTime);
      if (track.path === 'translation') translations[track.node] = value;
      else if (track.path === 'rotation') rotations[track.node] = value;
      else if (track.path === 'scale') scales[track.node] = value;
      else weights[track.node] = value;
    }
    const local = this.nodes.map((node, index) => localMatrix(node, translations[index], rotations[index], scales[index]));
    const world: Float32Array[] = new Array(this.nodes.length);
    for (const index of this.order) {
      const parent = this.nodes[index].parent;
      world[index] = parent === null ? new Float32Array(local[index]) : multiply(world[parent], local[index]);
    }
    return { local, world, weights };
  }
}
