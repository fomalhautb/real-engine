/** CPU contracts and bounded screen grid planning for world-space reflection rays. */
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
  });
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
