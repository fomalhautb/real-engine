/** Experimental, fixed-receiver CPU reference utility. Not integrated into the renderer.
 *
 * All quantities are pre-albedo irradiance: integral L(w) max(Ns.w, 0) dw
 * over the unchanged geometric hemisphere Ng.w > 0. Samples MUST be uniform
 * on that entire hemisphere. Merely supplying the correct numeric PDF does
 * not prove the caller sampled that distribution or used independent samples.
 *
 * The angular model is biased, even when its moments converge. Its deterministic
 * remainder bound applies to true integrals with a trusted radiance upper bound,
 * not to a noisy finite-sample estimate. Reported IID standard errors describe
 * sampling uncertainty only; for correlated/stratified/deterministic sequences
 * they are descriptive statistics, not confidence-interval coverage guarantees.
 */
export type GiAngularVector = readonly [number, number, number];
export const GI_ANGULAR_UNIFORM_PDF = 1 / (2 * Math.PI);
export const GI_ANGULAR_MAX_SAMPLES = 65_536;
export const GI_ANGULAR_MAX_ANGLE_DEGREES = 10;
export const GI_ANGULAR_MAX_RADIANCE = 1e12;

export interface GiAngularReceiver {
  readonly position: GiAngularVector;
  readonly shadingNormal: GiAngularVector;
  readonly geometricNormal: GiAngularVector;
}

export interface GiAngularMomentOptions {
  readonly receiver: GiAngularReceiver;
  /** A prior bound on ALL incident radiance, never the observed sample maximum.
   * This utility checks observed samples but cannot certify the caller's bound. */
  readonly radianceUpperBound: GiAngularVector;
  readonly maxSamples: number;
}

export interface GiAngularSample {
  readonly direction: GiAngularVector;
  readonly pdf: number;
  readonly radiance: GiAngularVector;
}

export interface GiAngularRgbStatistics {
  readonly mean: GiAngularVector;
  /** Row-major RGB covariance of the sample mean, assuming IID samples.
   * Null with fewer than two samples; zero is not evidence of convergence. */
  readonly iidCovarianceOfMean: readonly number[] | null;
  readonly iidStandardError: GiAngularVector | null;
  readonly luminanceMean: number;
  readonly luminanceIidStandardError: number | null;
}

export interface GiAngularMomentSnapshot {
  readonly sampleCount: number;
  readonly reference: GiAngularVector;
  readonly tangent: GiAngularVector;
  readonly bitangent: GiAngularVector;
  /** Full 9x9 sample covariance, order E.rgb, GT.rgb, GB.rgb. Not divided by n. */
  readonly sampleCovariance: readonly number[] | null;
}

export interface GiAngularReconstruction {
  readonly sampleCount: number;
  readonly angleDegrees: number;
  readonly linear: GiAngularRgbStatistics;
  /** Nonlinear clamping has different bias/variance; linear SE does NOT apply here. */
  readonly nonnegativeEstimate: GiAngularVector;
  /** 0 <= exact E(target) - exact linear <= pi Lmax (1-cos(angle)).
   * This bounds angular model error, not sampling error or spatial reuse. */
  readonly integralRemainderUpperBound: GiAngularVector;
}

const LUMINANCE: GiAngularVector = [0.2126, 0.7152, 0.0722];
const MAX_ANGLE_COS = Math.cos(GI_ANGULAR_MAX_ANGLE_DEGREES * Math.PI / 180);
const dot = (a: GiAngularVector, b: GiAngularVector) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const same = (a: GiAngularVector, b: GiAngularVector) => a.every((value, index) => value === b[index]);
const triple = (values: ArrayLike<number>, start = 0): GiAngularVector =>
  Object.freeze([values[start], values[start + 1], values[start + 2]]);

function vector(value: GiAngularVector, name: string): GiAngularVector {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) {
    throw new RangeError(`GI_ANGULAR_${name}`);
  }
  return triple(value);
}

function unit(value: GiAngularVector, name: string): GiAngularVector {
  const v = vector(value, name), length = Math.hypot(...v);
  if (Math.abs(length - 1) > 1e-6) throw new RangeError(`GI_ANGULAR_${name}_UNIT`);
  return triple(v.map(component => component / length));
}

function resolveReceiver(value: GiAngularReceiver): GiAngularReceiver {
  if (value === null || typeof value !== 'object') throw new RangeError('GI_ANGULAR_RECEIVER');
  const result = {
    position: vector(value.position, 'POSITION'),
    shadingNormal: unit(value.shadingNormal, 'SHADING_NORMAL'),
    geometricNormal: unit(value.geometricNormal, 'GEOMETRIC_NORMAL'),
  };
  if (dot(result.shadingNormal, result.geometricNormal) <= 0) throw new RangeError('GI_ANGULAR_NORMAL_FACING');
  return Object.freeze(result);
}

function basis(normal: GiAngularVector): readonly [GiAngularVector, GiAngularVector] {
  const axis: GiAngularVector = Math.abs(normal[2]) < 0.999 ? [0, 0, 1] : [0, 1, 0];
  const cross = (a: GiAngularVector, b: GiAngularVector): GiAngularVector =>
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const tangent = cross(axis, normal), length = Math.hypot(...tangent);
  const t = triple(tangent.map(component => component / length));
  return Object.freeze([t, triple(cross(normal, t))]);
}

/** u in (0,1), v in [0,1). Excludes measure-zero horizon/pole endpoints.
 * The caller supplies random numbers; no RNG or temporal sampling is implicit. */
export function sampleGiAngularHemisphere(geometricNormal: GiAngularVector, u: number, v: number) {
  if (!Number.isFinite(u) || u <= 0 || u >= 1 || !Number.isFinite(v) || v < 0 || v >= 1) {
    throw new RangeError('GI_ANGULAR_SAMPLE_COORDINATES');
  }
  const n = unit(geometricNormal, 'GEOMETRIC_NORMAL'), [t, b] = basis(n);
  const r = Math.sqrt(1 - u * u), x = r * Math.cos(2 * Math.PI * v), y = r * Math.sin(2 * Math.PI * v);
  return Object.freeze({ direction: triple(n.map((component, i) => component * u + t[i] * x + b[i] * y)),
    pdf: GI_ANGULAR_UNIFORM_PDF });
}

interface Context {
  readonly receiver: GiAngularReceiver;
  readonly upperBound: GiAngularVector;
  readonly maxSamples: number;
  readonly t: GiAngularVector;
  readonly b: GiAngularVector;
}

function context(options: GiAngularMomentOptions): Context {
  if (options === null || typeof options !== 'object') throw new RangeError('GI_ANGULAR_OPTIONS');
  if (!Number.isSafeInteger(options.maxSamples) || options.maxSamples < 1 || options.maxSamples > GI_ANGULAR_MAX_SAMPLES) {
    throw new RangeError('GI_ANGULAR_MAX_SAMPLES');
  }
  const upperBound = vector(options.radianceUpperBound, 'RADIANCE_BOUND');
  if (upperBound.some(value => value < 0 || value > GI_ANGULAR_MAX_RADIANCE)) throw new RangeError('GI_ANGULAR_RADIANCE_BOUND');
  const receiver = resolveReceiver(options.receiver), [t, b] = basis(receiver.shadingNormal);
  return Object.freeze({ receiver, upperBound, maxSamples: options.maxSamples, t, b });
}

function targetCoefficients(c: Context, target: GiAngularReceiver) {
  const receiver = resolveReceiver(target);
  if (!same(receiver.position, c.receiver.position)) throw new RangeError('GI_ANGULAR_POSITION_CHANGED');
  if (!same(receiver.geometricNormal, c.receiver.geometricNormal)) throw new RangeError('GI_ANGULAR_GEOMETRIC_NORMAL_CHANGED');
  const n = receiver.shadingNormal, reference = c.receiver.shadingNormal;
  const cosine = Math.min(1, Math.max(-1, dot(n, reference)));
  // Tolerance only accommodates roundoff at the exact 10-degree boundary.
  if (cosine < MAX_ANGLE_COS - 1e-14) throw new RangeError('GI_ANGULAR_ANGLE');
  const coefficients: GiAngularVector = same(n, reference) ? [1, 0, 0] : [dot(n, reference), dot(n, c.t), dot(n, c.b)];
  return { receiver, coefficients, angleDegrees: Math.acos(cosine) * 180 / Math.PI,
    remainder: triple(c.upperBound.map(value => Math.PI * value * (1 - cosine))) };
}

function observation(c: Context, sample: GiAngularSample) {
  if (sample === null || typeof sample !== 'object' || sample.pdf !== GI_ANGULAR_UNIFORM_PDF) {
    throw new RangeError('GI_ANGULAR_UNIFORM_PDF_REQUIRED');
  }
  const w = unit(sample.direction, 'DIRECTION'), radiance = vector(sample.radiance, 'RADIANCE');
  if (dot(w, c.receiver.geometricNormal) <= 0) throw new RangeError('GI_ANGULAR_SAMPLE_DOMAIN');
  if (radiance.some((value, i) => value < 0 || value > c.upperBound[i])) throw new RangeError('GI_ANGULAR_RADIANCE_BOUND');
  const cosine = dot(w, c.receiver.shadingNormal), active = cosine > 0;
  const factors = [Math.max(0, cosine), active ? dot(c.t, w) : 0, active ? dot(c.b, w) : 0];
  const response = factors.flatMap(factor => radiance.map(value => value * factor / sample.pdf));
  return { w, radiance, cosine, response };
}

/** Constant storage; symmetric Welford covariance, including every cross term. */
class RunningMoments {
  count = 0;
  readonly mean: Float64Array;
  readonly m2: Float64Array;
  constructor(readonly dimensions: number) {
    this.mean = new Float64Array(dimensions);
    this.m2 = new Float64Array(dimensions * dimensions);
  }
  add(value: readonly number[]) {
    const delta = value.map((entry, i) => entry - this.mean[i]);
    const n = ++this.count, factor = (n - 1) / n;
    for (let i = 0; i < this.dimensions; i++) {
      this.mean[i] += delta[i] / n;
      for (let j = i; j < this.dimensions; j++) {
        const index = i * this.dimensions + j;
        this.m2[index] += factor * delta[i] * delta[j];
        this.m2[j * this.dimensions + i] = this.m2[index];
      }
    }
  }
  covariance(): readonly number[] | null {
    return this.count < 2 ? null : Object.freeze(Array.from(this.m2, value => value / (this.count - 1)));
  }
  rgb(weights: readonly (readonly number[])[]): GiAngularRgbStatistics {
    const mean = triple(weights.map(row => row.reduce((sum, weight, i) => sum + weight * this.mean[i], 0)));
    let covariance: number[] | null = null;
    if (this.count >= 2) {
      covariance = [];
      for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) {
        let total = 0;
        for (let i = 0; i < this.dimensions; i++) for (let j = 0; j < this.dimensions; j++) {
          total += weights[c][i] * weights[d][j] * this.m2[i * this.dimensions + j];
        }
        covariance.push(total / (this.count * (this.count - 1)));
      }
    }
    const luminanceVariance = covariance?.reduce((sum, value, i) => sum + value * LUMINANCE[Math.floor(i / 3)] * LUMINANCE[i % 3], 0);
    return Object.freeze({ mean, iidCovarianceOfMean: covariance && Object.freeze(covariance),
      iidStandardError: covariance && triple([0, 4, 8].map(i => Math.sqrt(Math.max(0, covariance![i])))),
      luminanceMean: dot(mean, LUMINANCE), luminanceIidStandardError: luminanceVariance === undefined ? null : Math.sqrt(Math.max(0, luminanceVariance)) });
  }
}

function weights(coefficients: GiAngularVector): readonly (readonly number[])[] {
  return [0, 1, 2].map(channel => Array.from({ length: 9 }, (_, i) => i % 3 === channel ? coefficients[Math.floor(i / 3)] : 0));
}
const RGB_IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/** Fixed P/Ns0/Ng; no history, spatial filtering, or basis mixing. */
export class GiAngularMomentAccumulator {
  private readonly c: Context;
  private readonly stats = new RunningMoments(9);
  constructor(options: GiAngularMomentOptions) { this.c = context(options); }
  get sampleCount() { return this.stats.count; }
  add(sample: GiAngularSample): void {
    if (this.sampleCount >= this.c.maxSamples) throw new RangeError('GI_ANGULAR_SAMPLE_BUDGET');
    const value = observation(this.c, sample); // Reject invalid input before changing any state.
    this.stats.add(value.response);
  }
  snapshot(): GiAngularMomentSnapshot {
    return Object.freeze({ sampleCount: this.sampleCount, reference: triple(this.stats.mean), tangent: triple(this.stats.mean, 3),
      bitangent: triple(this.stats.mean, 6), sampleCovariance: this.stats.covariance() });
  }
  reconstruct(target: GiAngularReceiver): GiAngularReconstruction {
    if (!this.sampleCount) throw new RangeError('GI_ANGULAR_NO_SAMPLES');
    const { coefficients, angleDegrees, remainder } = targetCoefficients(this.c, target);
    const linear = this.stats.rgb(weights(coefficients));
    return Object.freeze({ sampleCount: this.sampleCount, angleDegrees, linear,
      nonnegativeEstimate: triple(linear.mean.map(value => Math.max(0, value))), integralRemainderUpperBound: remainder });
  }
}

export interface GiAngularPairedResult {
  readonly sampleCount: number;
  readonly directTarget: GiAngularRgbStatistics;
  readonly scalarReuse: GiAngularRgbStatistics;
  readonly linear: GiAngularRgbStatistics;
  readonly directMinusLinear: GiAngularRgbStatistics;
  readonly directMinusScalar: GiAngularRgbStatistics;
  readonly newHemisphereSamples: number;
  readonly lostHemisphereSamples: number;
  readonly positiveTargetSamples: number;
  readonly positiveReferenceSamples: number;
  readonly maximumAbsoluteLinearSample: GiAngularVector;
  readonly maximumDirectTargetSample: GiAngularVector;
  readonly integralRemainderUpperBound: GiAngularVector;
}

/** Independent bounded audit of ONE immutable target using the SAME input rays.
 * Differences are accumulated per sample, preserving paired covariance. The
 * new/lost hemisphere counts count directions; positive counts require radiance. */
export class GiAngularPairedAudit {
  private readonly c: Context;
  private readonly target: ReturnType<typeof targetCoefficients>;
  private readonly stats = Array.from({ length: 5 }, () => new RunningMoments(3));
  private newlyActive = 0;
  private lostActive = 0;
  private positiveTarget = 0;
  private positiveReference = 0;
  private readonly maximumLinear = [0, 0, 0];
  private readonly maximumDirect = [0, 0, 0];
  constructor(options: GiAngularMomentOptions, target: GiAngularReceiver) {
    this.c = context(options);
    this.target = targetCoefficients(this.c, target);
  }
  get sampleCount() { return this.stats[0].count; }
  add(sample: GiAngularSample): void {
    if (this.sampleCount >= this.c.maxSamples) throw new RangeError('GI_ANGULAR_SAMPLE_BUDGET');
    const { w, radiance, cosine, response } = observation(this.c, sample);
    const targetCosine = dot(w, this.target.receiver.shadingNormal);
    const direct = radiance.map(value => value * Math.max(0, targetCosine) / sample.pdf);
    const scalar = response.slice(0, 3), a = this.target.coefficients;
    const linear = scalar.map((value, i) => a[0] * value + a[1] * response[3 + i] + a[2] * response[6 + i]);
    const values = [direct, scalar, linear, direct.map((value, i) => value - linear[i]), direct.map((value, i) => value - scalar[i])];
    values.forEach((value, i) => this.stats[i].add(value));
    if (cosine <= 0 && targetCosine > 0) this.newlyActive++;
    if (cosine > 0 && targetCosine <= 0) this.lostActive++;
    if (direct.some(value => value > 0)) this.positiveTarget++;
    if (scalar.some(value => value > 0)) this.positiveReference++;
    for (let c = 0; c < 3; c++) {
      this.maximumLinear[c] = Math.max(this.maximumLinear[c], Math.abs(linear[c]));
      this.maximumDirect[c] = Math.max(this.maximumDirect[c], direct[c]);
    }
  }
  snapshot(): GiAngularPairedResult {
    if (!this.sampleCount) throw new RangeError('GI_ANGULAR_NO_SAMPLES');
    const [directTarget, scalarReuse, linear, directMinusLinear, directMinusScalar] = this.stats.map(value => value.rgb(RGB_IDENTITY));
    return Object.freeze({ sampleCount: this.sampleCount, directTarget, scalarReuse, linear, directMinusLinear, directMinusScalar,
      newHemisphereSamples: this.newlyActive, lostHemisphereSamples: this.lostActive,
      positiveTargetSamples: this.positiveTarget, positiveReferenceSamples: this.positiveReference,
      maximumAbsoluteLinearSample: triple(this.maximumLinear), maximumDirectTargetSample: triple(this.maximumDirect),
      integralRemainderUpperBound: this.target.remainder });
  }
}
