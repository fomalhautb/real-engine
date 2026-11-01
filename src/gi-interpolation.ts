/** CPU replay of the GPU probe interpolation in diffuse-gi-interpolate.wgsl.
 * Diagnostic and test use only: it evaluates a read-back probe field at
 * arbitrary receivers so error terms (visibility weighting, probe placement,
 * SH truncation) can be separated without changing the production frame. */
import { GI_DISTANCE_RESOLUTION, wrapGiDistanceTexel } from './gi-visibility';
import type { GiVisibilityFilter, GiVisibilityRefinement } from './diffuse-gi';

type Vec3 = readonly [number, number, number];

export interface GiReplayProbe {
  readonly coefficients: readonly (readonly number[])[];
  readonly moments: readonly (readonly number[])[];
  readonly maxRadiance: readonly number[];
  readonly valid: boolean;
  /** Axial-depth moments (giAxialMoments order); required by 'grid-axis-neighbor'. */
  readonly axial?: readonly number[];
}

export interface GiReplayGrid {
  readonly origin: Vec3;
  readonly spacing: Vec3;
  readonly counts: Vec3;
  readonly normalBias: number;
  readonly visibilityWeightThreshold: number;
  readonly visibilityFilter: GiVisibilityFilter;
  readonly visibilityRefinement?: GiVisibilityRefinement;
}

export interface GiReplayCorner {
  readonly index: number;
  readonly coordinate: [number, number, number];
  readonly position: [number, number, number];
  readonly spatialWeight: number;
  readonly distance: number;
  /** Probe-to-receiver unit direction. */
  readonly direction: [number, number, number];
  /** Visibility evaluated by the production filter. */
  readonly chebyshev: number;
  /** Visibility actually used (production or an override). */
  readonly visibility: number;
  readonly facing: number;
  readonly weight: number;
  readonly irradiance: [number, number, number];
}

export interface GiReplayResult {
  readonly irradiance: [number, number, number];
  readonly outside: boolean;
  readonly corners: readonly GiReplayCorner[];
  readonly total: number;
  readonly support: number;
  readonly supportWeight: number;
}

export interface GiReplayOverrides {
  /** Replace the per-corner visibility; receives the production value. */
  readonly visibility?: (corner: { index: number; position: readonly number[]; biased: readonly number[];
    direction: readonly number[]; distance: number; chebyshev: number }) => number;
  /** Replace the facing term (default: production `max(0.05, (1 - dot(d, ng)) / 2)`). */
  readonly facing?: (direction: readonly number[], geometricNormal: readonly number[]) => number;
  /** Receiver offset along the geometric normal (default: grid.normalBias). */
  readonly normalBias?: number;
}

const SH_CONVOLUTION = [Math.PI, 2.094395102, 2.094395102, 2.094395102,
  0.785398163, 0.785398163, 0.785398163, 0.785398163, 0.785398163] as const;

/** Real SH basis, l <= 2, in the GPU probe order (no normalization of the input). */
export function giReplayBasis(d: readonly number[]): number[] {
  return [0.28209479, 0.48860251 * d[1], 0.48860251 * d[2], 0.48860251 * d[0],
    1.09254843 * d[0] * d[1], 1.09254843 * d[1] * d[2], 0.31539157 * (3 * d[2] * d[2] - 1),
    1.09254843 * d[0] * d[2], 0.546274215 * (d[0] * d[0] - d[1] * d[1])];
}

/** Clamped cosine convolution of one probe's SH, bounded by pi times its sampled maximum. */
export function giReplayShIrradiance(probe: Pick<GiReplayProbe, 'coefficients' | 'maxRadiance'>,
  normal: readonly number[]): [number, number, number] {
  const basis = giReplayBasis(normal);
  const result: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 9; c++) for (let channel = 0; channel < 3; channel++) {
    result[channel] += probe.coefficients[c][channel] * basis[c] * SH_CONVOLUTION[c];
  }
  for (let channel = 0; channel < 3; channel++) {
    result[channel] = Math.min(Math.max(result[channel], 0), probe.maxRadiance[channel] * Math.PI);
  }
  return result;
}

/** Same fold as the GPU giOctaEncode (sign of zero is positive). */
function octaEncode(d: readonly number[]): [number, number] {
  const l1 = Math.max(Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]), 1e-8);
  let u = d[0] / l1, v = d[1] / l1;
  if (d[2] / l1 < 0) {
    const su = u >= 0 ? 1 : -1, sv = v >= 0 ? 1 : -1;
    const nu = (1 - Math.abs(v)) * su, nv = (1 - Math.abs(u)) * sv;
    u = nu; v = nv;
  }
  return [u * 0.5 + 0.5, v * 0.5 + 0.5];
}

function momentTaps(probe: GiReplayProbe, direction: readonly number[]) {
  const uv = octaEncode(direction);
  const px = uv[0] * GI_DISTANCE_RESOLUTION - 0.5, py = uv[1] * GI_DISTANCE_RESOLUTION - 0.5;
  const x = Math.floor(px), y = Math.floor(py), fx = px - x, fy = py - y;
  const tap = (dx: number, dy: number) => probe.moments[wrapGiDistanceTexel(x + dx, y + dy)];
  return { taps: [tap(0, 0), tap(1, 0), tap(0, 1), tap(1, 1)], fx, fy };
}

export function giReplayMomentVisibility(mean: number, meanSquared: number, distance: number): number {
  const variance = Math.max(meanSquared - mean * mean, 0.0001);
  const beyond = Math.max(0, distance - mean);
  return variance / (variance + beyond * beyond);
}

/** Production visibility for one probe and probe-to-receiver direction. */
export function giReplayVisibility(probe: GiReplayProbe, direction: readonly number[], distance: number,
  filter: GiVisibilityFilter): number {
  const { taps, fx, fy } = momentTaps(probe, direction);
  const mix = (a: number, b: number, t: number) => a + (b - a) * t;
  if (filter === 'per-tap-chebyshev-bilinear') {
    const v = taps.map(tap => giReplayMomentVisibility(tap[0], tap[1], distance));
    return mix(mix(v[0], v[1], fx), mix(v[2], v[3], fx), fy);
  }
  const m0 = mix(mix(taps[0][0], taps[1][0], fx), mix(taps[2][0], taps[3][0], fx), fy);
  const m1 = mix(mix(taps[0][1], taps[1][1], fx), mix(taps[2][1], taps[3][1], fx), fy);
  return giReplayMomentVisibility(m0, m1, distance);
}

/** Axial test of diffuse-gi-interpolate.wgsl giAxisVisibility. */
export function giReplayAxisVisibility(probe: Pick<GiReplayProbe, 'axial'>, delta: readonly number[]): number {
  const axial = probe.axial;
  if (!axial || axial.length !== 12) throw new RangeError('GI_REPLAY_AXIAL');
  let result = 1;
  for (let axis = 0; axis < 3; axis++) {
    const lane = 2 * (2 * axis + (delta[axis] >= 0 ? 0 : 1));
    result = Math.min(result, giReplayMomentVisibility(axial[lane], axial[lane + 1], Math.abs(delta[axis])));
  }
  return result;
}

/** One-sided visibility from a probe toward a grid offset (refineVisibility). */
export function giReplayNeighborVisibility(probe: GiReplayProbe, offset: readonly number[], spacing: readonly number[],
  filter: GiVisibilityFilter): number {
  if (offset[0] === 0 && offset[1] === 0 && offset[2] === 0) return 1;
  const delta = [0, 1, 2].map(axis => offset[axis] * spacing[axis]);
  const distance = Math.hypot(delta[0], delta[1], delta[2]);
  return giReplayVisibility(probe, delta.map(value => value / distance), distance, filter);
}

export function giReplayProbePosition(grid: Pick<GiReplayGrid, 'origin' | 'spacing' | 'counts'>, index: number): [number, number, number] {
  const [cx, cy] = grid.counts;
  const coordinate = [index % cx, Math.floor(index / cx) % cy, Math.floor(index / (cx * cy))];
  return [0, 1, 2].map(axis => grid.origin[axis] + coordinate[axis] * grid.spacing[axis]) as [number, number, number];
}

/** Mirror of giSampleSurfaceIrradianceSince for a complete, valid field. */
export function replayGiSurfaceIrradiance(field: readonly GiReplayProbe[], grid: GiReplayGrid, position: readonly number[],
  normal: readonly number[], geometricNormal: readonly number[], overrides: GiReplayOverrides = {}): GiReplayResult {
  const counts = grid.counts, threshold = grid.visibilityWeightThreshold;
  if (field.length !== counts[0] * counts[1] * counts[2]) throw new RangeError('GI_REPLAY_FIELD');
  const bias = overrides.normalBias ?? grid.normalBias;
  const biased = [0, 1, 2].map(axis => position[axis] + geometricNormal[axis] * bias);
  const coordinate = [0, 1, 2].map(axis => (biased[axis] - grid.origin[axis]) / grid.spacing[axis]);
  if (coordinate.some((value, axis) => value < -0.5 || value > counts[axis] - 1 + 0.5)) {
    return { irradiance: [0, 0, 0], outside: true, corners: [], total: 0, support: 0, supportWeight: 0 };
  }
  const clamped = coordinate.map((value, axis) => Math.min(Math.max(value, 0), counts[axis] - 1));
  const lower = clamped.map((value, axis) => Math.min(Math.floor(value), counts[axis] - 2));
  const fraction = clamped.map((value, axis) => value - lower[axis]);
  if ((grid.visibilityRefinement ?? 'none') === 'grid-axis-neighbor') {
    return replayRefined(field, grid, biased, normal, geometricNormal, lower, fraction, overrides);
  }
  const corners: GiReplayCorner[] = [];
  const sum = [0, 0, 0];
  let total = 0, visibleSupport = 0, peakSpatialWeight = 0;
  for (let corner = 0; corner < 8; corner++) {
    const offset = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
    const at = [0, 1, 2].map(axis => lower[axis] + offset[axis]) as [number, number, number];
    const index = at[0] + counts[0] * (at[1] + counts[1] * at[2]);
    const spatialWeight = [0, 1, 2].reduce((product, axis) => product * (offset[axis] ? fraction[axis] : 1 - fraction[axis]), 1);
    peakSpatialWeight = Math.max(peakSpatialWeight, spatialWeight);
    const probe = field[index];
    if (!probe.valid) continue;
    const probePosition = [0, 1, 2].map(axis => grid.origin[axis] + at[axis] * grid.spacing[axis]) as [number, number, number];
    const delta = [0, 1, 2].map(axis => biased[axis] - probePosition[axis]);
    const distance = Math.hypot(delta[0], delta[1], delta[2]);
    const direction = delta.map(value => value / Math.max(distance, 0.00001)) as [number, number, number];
    const chebyshev = giReplayVisibility(probe, direction, distance, grid.visibilityFilter);
    const visibility = overrides.visibility
      ? overrides.visibility({ index, position: probePosition, biased, direction, distance, chebyshev }) : chebyshev;
    const facing = overrides.facing ? overrides.facing(direction, geometricNormal)
      : Math.max(0.05, (1 - (direction[0] * geometricNormal[0] + direction[1] * geometricNormal[1]
        + direction[2] * geometricNormal[2])) * 0.5);
    let confidence = facing * facing * visibility * visibility * visibility;
    if (confidence < threshold) {
      const ratio = confidence / threshold;
      confidence *= ratio * ratio;
    }
    const weight = spatialWeight * confidence;
    if (weight > 0) visibleSupport = Math.max(visibleSupport, spatialWeight * visibility * visibility * visibility);
    const irradiance = giReplayShIrradiance(probe, normal);
    for (let channel = 0; channel < 3; channel++) sum[channel] += irradiance[channel] * weight;
    total += weight;
    corners.push({ index, coordinate: at, position: probePosition, spatialWeight, distance, direction, chebyshev,
      visibility, facing, weight, irradiance });
  }
  const support = peakSpatialWeight > 0 ? visibleSupport / peakSpatialWeight : 0;
  let supportWeight = 1;
  if (support < threshold) {
    const ratio = support / threshold;
    supportWeight = ratio * ratio;
  }
  const scale = supportWeight / Math.max(total, 1e-12);
  return { irradiance: sum.map(value => value * scale) as [number, number, number], outside: false, corners, total,
    support, supportWeight };
}

function confidenceOf(facing: number, visibility: number, threshold: number): number {
  let confidence = facing * facing * visibility * visibility * visibility;
  if (confidence < threshold) {
    const ratio = confidence / threshold;
    confidence *= ratio * ratio;
  }
  return confidence;
}

/** Mirror of giSampleRefinedIrradiance ('grid-axis-neighbor'). Corner visibility is
 * min(direction test, axial test), then multiplied by the mutual neighbour
 * visibility with the strongest (first maximum weight) corner. */
function replayRefined(field: readonly GiReplayProbe[], grid: GiReplayGrid, biased: readonly number[], normal: readonly number[],
  geometricNormal: readonly number[], lower: readonly number[], fraction: readonly number[], overrides: GiReplayOverrides): GiReplayResult {
  const counts = grid.counts, threshold = grid.visibilityWeightThreshold;
  type Entry = { corner: number; offset: number[]; index: number; position: [number, number, number]; spatialWeight: number;
    distance: number; direction: [number, number, number]; chebyshev: number; visibility: number; facing: number };
  const entries: Entry[] = [];
  let peakSpatialWeight = 0, anchor: Entry | undefined, anchorWeight = -1;
  for (let corner = 0; corner < 8; corner++) {
    const offset = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
    const at = [0, 1, 2].map(axis => lower[axis] + offset[axis]);
    const index = at[0] + counts[0] * (at[1] + counts[1] * at[2]);
    const spatialWeight = [0, 1, 2].reduce((product, axis) => product * (offset[axis] ? fraction[axis] : 1 - fraction[axis]), 1);
    peakSpatialWeight = Math.max(peakSpatialWeight, spatialWeight);
    const probe = field[index];
    if (!probe.valid) continue;
    const position = [0, 1, 2].map(axis => grid.origin[axis] + at[axis] * grid.spacing[axis]) as [number, number, number];
    const delta = [0, 1, 2].map(axis => biased[axis] - position[axis]);
    const distance = Math.hypot(delta[0], delta[1], delta[2]);
    const direction = delta.map(value => value / Math.max(distance, 0.00001)) as [number, number, number];
    const chebyshev = Math.min(giReplayVisibility(probe, direction, distance, grid.visibilityFilter), giReplayAxisVisibility(probe, delta));
    const visibility = overrides.visibility
      ? overrides.visibility({ index, position, biased, direction, distance, chebyshev }) : chebyshev;
    const facing = overrides.facing ? overrides.facing(direction, geometricNormal)
      : Math.max(0.05, (1 - (direction[0] * geometricNormal[0] + direction[1] * geometricNormal[1]
        + direction[2] * geometricNormal[2])) * 0.5);
    const entry = { corner, offset, index, position, spatialWeight, distance, direction, chebyshev, visibility, facing };
    entries.push(entry);
    const weight = spatialWeight * confidenceOf(facing, visibility, threshold);
    if (weight > anchorWeight) { anchorWeight = weight; anchor = entry; }
  }
  const corners: GiReplayCorner[] = [];
  const sum = [0, 0, 0];
  let total = 0, visibleSupport = 0;
  for (const entry of entries) {
    let visibility = entry.visibility;
    if (entry !== anchor) {
      const offset = [0, 1, 2].map(axis => entry.offset[axis] - anchor!.offset[axis]);
      visibility *= Math.min(giReplayNeighborVisibility(field[anchor!.index], offset, grid.spacing, grid.visibilityFilter),
        giReplayNeighborVisibility(field[entry.index], offset.map(value => -value), grid.spacing, grid.visibilityFilter));
    }
    const weight = entry.spatialWeight * confidenceOf(entry.facing, visibility, threshold);
    if (weight > 0) visibleSupport = Math.max(visibleSupport, entry.spatialWeight * visibility * visibility * visibility);
    const irradiance = giReplayShIrradiance(field[entry.index], normal);
    for (let channel = 0; channel < 3; channel++) sum[channel] += irradiance[channel] * weight;
    total += weight;
    corners.push({ index: entry.index, coordinate: [lower[0] + entry.offset[0], lower[1] + entry.offset[1], lower[2] + entry.offset[2]],
      position: entry.position, spatialWeight: entry.spatialWeight, distance: entry.distance, direction: entry.direction,
      chebyshev: entry.chebyshev, visibility, facing: entry.facing, weight, irradiance });
  }
  const support = peakSpatialWeight > 0 ? visibleSupport / peakSpatialWeight : 0;
  let supportWeight = 1;
  if (support < threshold) {
    const ratio = support / threshold;
    supportWeight = ratio * ratio;
  }
  const scale = supportWeight / Math.max(total, 1e-12);
  return { irradiance: sum.map(value => value * scale) as [number, number, number], outside: false, corners, total,
    support, supportWeight };
}
