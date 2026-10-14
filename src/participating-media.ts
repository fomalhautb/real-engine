/** Homogeneous participating-medium reference. Distances are metres; radiance is linear HDR. */
export interface MediumCoefficients {
  /** Extinction coefficient in inverse metres. */
  readonly extinction: number;
  /** Per-channel scattering fraction in [0, 1]. */
  readonly albedo: readonly [number, number, number];
  /** Per-channel emitted radiance per metre. */
  readonly emission: readonly [number, number, number];
  /** Henyey–Greenstein asymmetry parameter; omitted means isotropic. */
  readonly anisotropy?: number;
}

export interface MediumSegment {
  readonly transmittance: number;
  readonly scattering: readonly [number, number, number];
}

export type Rgb = readonly [number, number, number];
export type MediumSnapshot = Readonly<Required<MediumCoefficients>>;

function finiteFloat32(value: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`);
  }
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded) || (value !== 0 && rounded === 0)) {
    throw new RangeError(`${label} cannot be represented by float32`);
  }
  return value;
}

function nonnegative(value: number, label: string): number {
  finiteFloat32(value, label);
  if (value < 0) throw new RangeError(`${label} must be nonnegative`);
  return value;
}

function rgb(value: Rgb, label: string, maximum = Infinity): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new RangeError(`${label} must contain three channels`);
  }
  return value.map((channel, index) => {
    nonnegative(channel, `${label}[${index}]`);
    if (channel > maximum) throw new RangeError(`${label}[${index}] must be at most ${maximum}`);
    return channel;
  }) as [number, number, number];
}

function finiteResult(value: number, label: string): number {
  if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new RangeError(`${label} exceeds the finite float32 result range`);
  }
  return value;
}

/** Validate input and take an immutable snapshot, including both RGB tuples. */
export function snapshotMediumCoefficients(input: MediumCoefficients): MediumSnapshot {
  if (!input || typeof input !== 'object') throw new TypeError('medium coefficients must be an object');
  const extinction = nonnegative(input.extinction, 'extinction');
  const albedo = Object.freeze(rgb(input.albedo, 'albedo', 1));
  const emission = Object.freeze(rgb(input.emission, 'emission'));
  const anisotropy = finiteFloat32(input.anisotropy ?? 0, 'anisotropy');
  if (anisotropy < -0.95 || anisotropy > 0.95) {
    throw new RangeError('anisotropy must be in [-0.95, 0.95]');
  }
  return Object.freeze({ extinction, albedo, emission, anisotropy });
}

/**
 * HG phase density in inverse steradians. cosTheta is the dot product between
 * incoming light propagation direction and outgoing propagation toward the viewer.
 */
export function henyeyGreenstein(cosTheta: number, g: number): number {
  finiteFloat32(cosTheta, 'cosTheta');
  finiteFloat32(g, 'g');
  if (cosTheta < -1 || cosTheta > 1) throw new RangeError('cosTheta must be in [-1, 1]');
  if (g < -0.95 || g > 0.95) throw new RangeError('g must be in [-0.95, 0.95]');
  const denominator = 1 + g * g - 2 * g * cosTheta;
  return (1 - g * g) / (4 * Math.PI * denominator * Math.sqrt(denominator));
}

/**
 * Integrate one uniform segment: source = sigma_t * albedo * phase * incident + emission.
 * The result's scattering field includes both in-scattering and medium emission.
 */
export function integrateUniformMedium(
  coefficients: MediumCoefficients,
  distance: number,
  incidentRadiance: Rgb,
  cosTheta = 0,
): MediumSegment {
  const medium = snapshotMediumCoefficients(coefficients);
  nonnegative(distance, 'distance');
  const incident = rgb(incidentRadiance, 'incidentRadiance');
  const phase = henyeyGreenstein(cosTheta, medium.anisotropy);
  const opticalDepth = medium.extinction * distance;
  const transmittance = finiteResult(Math.exp(-opticalDepth), 'transmittance');
  const integral = medium.extinction === 0
    ? distance
    : -Math.expm1(-opticalDepth) / medium.extinction;
  finiteResult(integral, 'medium integral');
  const scattering = [0, 1, 2].map((channel) => {
    const source = medium.extinction * medium.albedo[channel] * phase * incident[channel]
      + medium.emission[channel];
    finiteResult(source, `source[${channel}]`);
    return finiteResult(source * integral, `scattering[${channel}]`);
  }) as [number, number, number];
  return { transmittance, scattering };
}

function validatedSegment(segment: MediumSegment, label: string): MediumSegment {
  if (!segment || typeof segment !== 'object') throw new TypeError(`${label} must be a medium segment`);
  const transmittance = finiteFloat32(segment.transmittance, `${label}.transmittance`);
  if (transmittance < 0 || transmittance > 1) {
    throw new RangeError(`${label}.transmittance must be in [0, 1]`);
  }
  return { transmittance, scattering: rgb(segment.scattering, `${label}.scattering`) };
}

/** Front-to-back recurrence: (T_f T_b, L_f + T_f L_b). Order matters. */
export function composeMediumSegments(front: MediumSegment, back: MediumSegment): MediumSegment {
  const f = validatedSegment(front, 'front');
  const b = validatedSegment(back, 'back');
  return {
    transmittance: finiteResult(f.transmittance * b.transmittance, 'composed transmittance'),
    scattering: [0, 1, 2].map((channel) =>
      finiteResult(f.scattering[channel] + f.transmittance * b.scattering[channel], `composed scattering[${channel}]`),
    ) as [number, number, number],
  };
}

/** Linear-HDR surface composition: L_out = L_medium + T_medium * L_surface. */
export function composeWithSurface(segment: MediumSegment, surfaceRGB: Rgb): [number, number, number] {
  const medium = validatedSegment(segment, 'segment');
  const surface = rgb(surfaceRGB, 'surfaceRGB');
  return [0, 1, 2].map((channel) =>
    finiteResult(medium.scattering[channel] + medium.transmittance * surface[channel], `composed RGB[${channel}]`),
  ) as [number, number, number];
}
