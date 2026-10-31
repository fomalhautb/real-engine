/** CPU contracts and bounded screen grid planning for world-space reflection rays. */

/** Opt-in reprojected reflection history. Omitted means the deterministic single-frame path. */
export interface ReflectionTemporalOptions {
  /** History cap for rough reflections (roughness >= 0.5); 2..128. */
  readonly maxHistoryFrames?: number;
  /** History cap for mirror-like reflections (roughness <= 0.1); 1..maxHistoryFrames. */
  readonly minHistoryFrames?: number;
  /** Width, in neighborhood standard deviations, of the motion clamp box; 0.25..8. */
  readonly clampSigma?: number;
  /** Below this roughness a moving view reprojects by reflected-hit parallax first
   * and falls back to surface motion; at or above it only surface motion is
   * used. 0 disables the parallax candidate; 0..1. */
  readonly parallaxRoughness?: number;
}

export interface ResolvedReflectionTemporalOptions {
  readonly maxHistoryFrames: number;
  readonly minHistoryFrames: number;
  readonly clampSigma: number;
  readonly parallaxRoughness: number;
}

/** Opt-in short screen-space supplement over owned copies of the previous
 * frame's scene-linear HDR and depth (counted in maxBytes). A world-query sample
 * is replaced, with confidence, only where the march finds the same surface at
 * the same reflected distance; every other case keeps the world-query result. */
export interface ReflectionScreenTraceOptions {
  /** Longest world-hit distance that may use screen detail, meters; 0.05..20. */
  readonly maxDistance?: number;
  /** Fixed linear march steps per sample before a four-step refinement; 4..32. */
  readonly steps?: number;
  /** Depth-buffer thickness accepted behind a crossing, meters; 0.001..2. */
  readonly thickness?: number;
  /** Required agreement with the world-hit distance, and the largest accepted
   * reflected-hit motion, meters; 0.001..1. */
  readonly distanceTolerance?: number;
  /** Confidence fade width at the screen border, as a fraction of NDC; 0.01..0.5. */
  readonly edgeFade?: number;
}

export interface ResolvedReflectionScreenTraceOptions {
  readonly maxDistance: number;
  readonly steps: number;
  readonly thickness: number;
  readonly distanceTolerance: number;
  readonly edgeFade: number;
}

/** Owned previous-frame copies: rgba16float HDR plus depth32float. */
export const REFLECTION_SCREEN_BYTES_PER_PIXEL = 12;

export interface ReflectionsOptions {
  readonly pixelSpacing?: number;
  readonly maxAnchors?: number;
  readonly samplesPerAnchor?: number;
  readonly maxDistance?: number;
  /** World-space neighborhood radius for gathering nearby reflection anchors, in meters. */
  readonly maxGatherDistance?: number;
  /** Maximum separation along the receiving plane for a gathered anchor, in meters. */
  readonly planeTolerance?: number;
  readonly normalBias?: number;
  /** Bounded integration steps for participating media along a reflected segment. */
  readonly fogSteps?: number;
  readonly maxBytes?: number;
  readonly skyRadiance?: readonly [number, number, number];
  readonly textureLod?: number;
  /** Reprojected history with per-frame rotated GGX samples; memory is counted in maxBytes. */
  readonly temporal?: ReflectionTemporalOptions;
  /** Near-contact screen-space detail; world queries remain the source of every sample. */
  readonly screenTrace?: ReflectionScreenTraceOptions;
}

export interface ResolvedReflectionsOptions {
  readonly pixelSpacing: number;
  readonly maxAnchors: number;
  readonly samplesPerAnchor: number;
  readonly maxDistance: number;
  readonly maxGatherDistance: number;
  readonly planeTolerance: number;
  readonly normalBias: number;
  readonly fogSteps: number;
  readonly maxBytes: number;
  readonly skyRadiance: readonly [number, number, number];
  readonly textureLod: number;
  /** Undefined when history is not configured. */
  readonly temporal: ResolvedReflectionTemporalOptions | undefined;
  /** Undefined when the screen-space supplement is not configured. */
  readonly screenTrace: ResolvedReflectionScreenTraceOptions | undefined;
}

export interface ReflectionGridPlan {
  readonly spacing: number;
  readonly columns: number;
  readonly rows: number;
  readonly anchorCount: number;
  readonly rayCount: number;
}

export interface ReflectionGgxSample {
  readonly halfVector: readonly [number, number, number];
  readonly direction: readonly [number, number, number];
  /** Solid-angle PDF: D(NoH) * NoH / (4 * VoH). Zero for a rejected ray or ideal mirror. */
  readonly pdf: number;
  /** Geometry-only BRDF * NoL / pdf; uses the surface BRDF's NoV and denominator floors. Fresnel and radiance are separate. */
  readonly weight: number;
}

type Vec3 = readonly [number, number, number];

function boundedInteger(value: unknown, minimum: number, maximum: number, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`REFLECTIONS_${name}`);
  }
  return value;
}

function boundedNumber(value: unknown, minimum: number, maximum: number, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`REFLECTIONS_${name}`);
  }
  return value;
}

export function resolveReflectionsOptions(options: ReflectionsOptions = {}): ResolvedReflectionsOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new RangeError('REFLECTIONS_OPTIONS');
  const sky = options.skyRadiance === undefined ? [0, 0, 0] : options.skyRadiance;
  if (!Array.isArray(sky) || sky.length !== 3) throw new RangeError('REFLECTIONS_SKY_RADIANCE');
  const skyRadiance = Object.freeze([
    boundedNumber(sky[0], 0, 1024, 'SKY_RADIANCE'),
    boundedNumber(sky[1], 0, 1024, 'SKY_RADIANCE'),
    boundedNumber(sky[2], 0, 1024, 'SKY_RADIANCE'),
  ] as [number, number, number]);
  return Object.freeze({
    pixelSpacing: boundedInteger(options.pixelSpacing === undefined ? 4 : options.pixelSpacing, 1, 128, 'PIXEL_SPACING'),
    maxAnchors: boundedInteger(options.maxAnchors === undefined ? 16_384 : options.maxAnchors, 1, 65_536, 'MAX_ANCHORS'),
    samplesPerAnchor: boundedInteger(options.samplesPerAnchor === undefined ? 4 : options.samplesPerAnchor, 1, 16, 'SAMPLES_PER_ANCHOR'),
    maxDistance: boundedNumber(options.maxDistance === undefined ? 50 : options.maxDistance, 0.01, 1000, 'MAX_DISTANCE'),
    maxGatherDistance: boundedNumber(options.maxGatherDistance === undefined ? 0.5 : options.maxGatherDistance,
      0.001, 100, 'MAX_GATHER_DISTANCE'),
    planeTolerance: boundedNumber(options.planeTolerance === undefined ? 0.001 : options.planeTolerance,
      0.000001, 0.1, 'PLANE_TOLERANCE'),
    normalBias: boundedNumber(options.normalBias === undefined ? 0.0001 : options.normalBias, 0, 0.01, 'NORMAL_BIAS'),
    fogSteps: boundedInteger(options.fogSteps === undefined ? 32 : options.fogSteps, 1, 128, 'FOG_STEPS'),
    maxBytes: boundedInteger(options.maxBytes === undefined ? 32 * 1024 * 1024 : options.maxBytes,
      1, 128 * 1024 * 1024, 'MAX_BYTES'),
    skyRadiance,
    textureLod: boundedNumber(options.textureLod === undefined ? 0 : options.textureLod, 0, 16, 'TEXTURE_LOD'),
    temporal: resolveTemporal(options.temporal),
    screenTrace: resolveScreenTrace(options.screenTrace),
  });
}

function resolveScreenTrace(screen: ReflectionScreenTraceOptions | undefined): ResolvedReflectionScreenTraceOptions | undefined {
  if (screen === undefined) return undefined;
  if (!screen || typeof screen !== 'object' || Array.isArray(screen)) throw new RangeError('REFLECTIONS_SCREEN_TRACE');
  return Object.freeze({
    maxDistance: boundedNumber(screen.maxDistance ?? 1.5, 0.05, 20, 'SCREEN_MAX_DISTANCE'),
    steps: boundedInteger(screen.steps ?? 12, 4, 32, 'SCREEN_STEPS'),
    thickness: boundedNumber(screen.thickness ?? 0.1, 0.001, 2, 'SCREEN_THICKNESS'),
    distanceTolerance: boundedNumber(screen.distanceTolerance ?? 0.05, 0.001, 1, 'SCREEN_DISTANCE_TOLERANCE'),
    edgeFade: boundedNumber(screen.edgeFade ?? 0.1, 0.01, 0.5, 'SCREEN_EDGE_FADE'),
  });
}

function resolveTemporal(temporal: ReflectionTemporalOptions | undefined): ResolvedReflectionTemporalOptions | undefined {
  if (temporal === undefined) return undefined;
  if (!temporal || typeof temporal !== 'object' || Array.isArray(temporal)) throw new RangeError('REFLECTIONS_TEMPORAL');
  const maxHistoryFrames = boundedInteger(temporal.maxHistoryFrames ?? 32, 2, 128, 'MAX_HISTORY_FRAMES');
  return Object.freeze({
    maxHistoryFrames,
    minHistoryFrames: boundedInteger(temporal.minHistoryFrames ?? Math.min(8, maxHistoryFrames), 1, maxHistoryFrames,
      'MIN_HISTORY_FRAMES'),
    clampSigma: boundedNumber(temporal.clampSigma ?? 1.5, 0.25, 8, 'CLAMP_SIGMA'),
    parallaxRoughness: boundedNumber(temporal.parallaxRoughness ?? 0.35, 0, 1, 'PARALLAX_ROUGHNESS'),
  });
}

/** Roughness-aware history cap; WGSL `reflectionHistoryLimit` uses the same ramp. */
export function reflectionHistoryLimit(roughness: number, temporal: ResolvedReflectionTemporalOptions): number {
  boundedNumber(roughness, 0, 1, 'ROUGHNESS');
  const t = Math.min(1, Math.max(0, (roughness - 0.1) / 0.4));
  const ramp = t * t * (3 - 2 * t);
  return Math.max(1, Math.round(temporal.minHistoryFrames + (temporal.maxHistoryFrames - temporal.minHistoryFrames) * ramp));
}

/** 32-bit integer avalanche shared with WGSL `reflectionHash`. */
export function reflectionHash(value: number): number {
  let x = value >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/** R2 sequence steps (1/plastic number and its square) in 0.32 fixed point. */
const R2_STEP = [3242174889, 2447445414] as const;

/** Per-anchor Cranley-Patterson rotation for `frame` as two 0.32 fixed-point
 * words: a hashed anchor offset plus the R2 low-discrepancy step, so consecutive
 * frames stratify the GGX lobe. Integer wrapping matches WGSL `reflectionRotation` exactly. */
export function reflectionSampleRotation(frame: number, anchor: number): [number, number] {
  boundedInteger(frame, 0, 0xffff_ffff, 'FRAME');
  boundedInteger(anchor, 0, 0xffff_ffff, 'ANCHOR');
  const seed = reflectionHash((Math.imul(anchor, 0x9e3779b9) ^ 0x5bd1e995) >>> 0);
  const step = frame % 4096;
  return [(seed + Math.imul(step, R2_STEP[0])) >>> 0, (reflectionHash(seed) + Math.imul(step, R2_STEP[1])) >>> 0];
}

/** GGX (u1 azimuth, u2 polar CDF) for sample `index` of `count`. Without a
 * rotation this is the legacy fixed Hammersley set; with one it is the
 * temporal set. u1 keeps 24 exact bits; u2 follows the float32 WGSL sum. */
export function reflectionSampleCoordinates(index: number, count: number,
  rotation?: readonly [number, number]): [number, number] {
  boundedInteger(count, 1, 16, 'SAMPLES_PER_ANCHOR');
  boundedInteger(index, 0, count - 1, 'SAMPLE_INDEX');
  let reversed = 0;
  for (let bit = 0; bit < 32; bit++) if (index & (1 << bit)) reversed |= 1 << (31 - bit);
  reversed >>>= 0;
  if (!rotation) return [reversed / 4294967296, (index + 0.5) / count];
  const u1 = (((reversed + rotation[0]) >>> 0) >>> 8) / 16777216;
  const sum = Math.fround(Math.fround((index + 0.5) / count) + (rotation[1] >>> 8) / 16777216);
  return [u1, Math.min(sum - Math.floor(sum), 0.99999994)];
}

/** Previous-frame image of a reflected point across the receiving plane.
 * `hitPrevious` is the reflected hit's previous position; the receiver plane
 * passes through `receiverPrevious` with normal `normal`. A static hit gives a
 * camera-independent virtual point, which is the reflected-hit parallax target. */
export function reflectionVirtualPoint(hitPrevious: Vec3, receiverPrevious: Vec3, normal: Vec3): [number, number, number] {
  const n = normalized(normal, 'NORMAL');
  const offset = dot([hitPrevious[0] - receiverPrevious[0], hitPrevious[1] - receiverPrevious[1],
    hitPrevious[2] - receiverPrevious[2]], n);
  return [hitPrevious[0] - 2 * offset * n[0], hitPrevious[1] - 2 * offset * n[1], hitPrevious[2] - 2 * offset * n[2]];
}

/** Find the finest integer spacing that fits anchors, ray capacity and shared work units. */
export function planReflectionGrid(
  width: number, height: number, options: ReflectionsOptions | ResolvedReflectionsOptions,
  maxRays: number, maxTotalWorkUnits: number,
): ReflectionGridPlan {
  boundedInteger(width, 1, 16_384, 'WIDTH');
  boundedInteger(height, 1, 16_384, 'HEIGHT');
  boundedInteger(maxRays, 1, 0xffff_ffff, 'MAX_RAYS');
  boundedInteger(maxTotalWorkUnits, 1, 0xffff_ffff, 'MAX_TOTAL_WORK_UNITS');
  const resolved = resolveReflectionsOptions(options);
  const anchorBudget = Math.min(resolved.maxAnchors,
    Math.floor(maxRays / resolved.samplesPerAnchor),
    Math.floor(maxTotalWorkUnits / resolved.samplesPerAnchor));
  if (anchorBudget < 1) throw new RangeError('REFLECTIONS_QUERY_RAY_BUDGET');

  const fits = (spacing: number): boolean =>
    Math.ceil(width / spacing) * Math.ceil(height / spacing) <= anchorBudget;
  let low = resolved.pixelSpacing;
  let high = Math.max(low, width, height);
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (fits(middle)) high = middle;
    else low = middle + 1;
  }
  const spacing = low;
  const columns = Math.ceil(width / spacing);
  const rows = Math.ceil(height / spacing);
  const anchorCount = columns * rows;
  return Object.freeze({ spacing, columns, rows, anchorCount, rayCount: anchorCount * resolved.samplesPerAnchor });
}

function dot(a: Vec3, b: Vec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalized(input: Vec3, name: string): Vec3 {
  if (!Array.isArray(input) || input.length !== 3 || input.some(value => !Number.isFinite(value))) {
    throw new RangeError(`REFLECTIONS_${name}`);
  }
  const length = Math.hypot(...input);
  if (!(length > 0) || !Number.isFinite(length)) throw new RangeError(`REFLECTIONS_${name}`);
  return [input[0] / length, input[1] / length, input[2] / length];
}

/** Isotropic GGX NDF sampling, not VNDF. `view` points toward the eye.
 * `u1` is azimuth and `u2` is polar-CDF position: the GPU currently feeds
 * reverseBits(sampleIndex) to u1 and (sampleIndex + 0.5)/N to u2.
 * The GPU material path clamps roughness to at least 0.045, so this CPU
 * reference's zero-roughness delta (pdf/weight 0) is outside that path. */
export function sampleReflectionGgx(normal: Vec3, view: Vec3, roughness: number,
  u1: number, u2: number): ReflectionGgxSample {
  const n = normalized(normal, 'NORMAL'), v = normalized(view, 'VIEW');
  const noV = dot(n, v);
  if (!(noV > 0)) throw new RangeError('REFLECTIONS_VIEW_HEMISPHERE');
  boundedNumber(roughness, 0, 1, 'ROUGHNESS');
  if (!(u1 >= 0 && u1 < 1) || !Number.isFinite(u1)
    || !(u2 >= 0 && u2 < 1) || !Number.isFinite(u2)) throw new RangeError('REFLECTIONS_SAMPLE_U');

  const alpha = roughness * roughness;
  const alpha2 = alpha * alpha;
  // Float64 cannot resolve an NDF this narrow. A perfect mirror is a Dirac
  // delta, so no finite solid-angle PDF exists; signal it with pdf=0.
  const mirror = alpha2 < 1e-20;
  const cosTheta = mirror ? 1 : Math.sqrt((1 - u2) / (1 + (alpha2 - 1) * u2));
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const phi = 2 * Math.PI * u1;
  const helper: Vec3 = Math.abs(n[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  const tangent = normalized(cross(helper, n), 'TANGENT');
  const bitangent = cross(n, tangent);
  const halfVector = normalized([
    tangent[0] * sinTheta * Math.cos(phi) + bitangent[0] * sinTheta * Math.sin(phi) + n[0] * cosTheta,
    tangent[1] * sinTheta * Math.cos(phi) + bitangent[1] * sinTheta * Math.sin(phi) + n[1] * cosTheta,
    tangent[2] * sinTheta * Math.cos(phi) + bitangent[2] * sinTheta * Math.sin(phi) + n[2] * cosTheta,
  ], 'HALF_VECTOR');
  const voH = dot(v, halfVector);
  const direction = normalized([
    2 * voH * halfVector[0] - v[0],
    2 * voH * halfVector[1] - v[1],
    2 * voH * halfVector[2] - v[2],
  ], 'DIRECTION');
  const noL = dot(n, direction), noH = Math.max(0, dot(n, halfVector));
  if (mirror || voH <= 0 || noL <= 0) {
    return { halfVector, direction, pdf: 0, weight: 0 };
  }
  const denominator = (1 - noH * noH) + noH * noH * alpha2;
  const distribution = alpha2 / (Math.PI * denominator * denominator);
  const pdf = distribution * noH / (4 * voH);
  const k = (roughness + 1) ** 2 / 8;
  const shadingNoV = Math.max(noV, 0.001);
  const gV = shadingNoV / (shadingNoV * (1 - k) + k);
  const gL = noL / (noL * (1 - k) + k);
  const weight = gV * gL * (4 * noL * voH)
    / Math.max(noH * Math.max(4 * shadingNoV * noL, 0.001), 1e-20);
  return { halfVector, direction, pdf, weight };
}
