import { describe, expect, it } from 'vitest';
import {
  GI_ANGULAR_MAX_SAMPLES, GI_ANGULAR_UNIFORM_PDF,
  GiAngularMomentAccumulator, GiAngularPairedAudit, sampleGiAngularHemisphere,
  type GiAngularMomentOptions, type GiAngularReceiver, type GiAngularRgbStatistics,
  type GiAngularSample, type GiAngularVector,
} from '../src/gi-angular-moment';

const PI2 = 2 * Math.PI;
const dot = (a: GiAngularVector, b: GiAngularVector) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const normalize = (v: GiAngularVector): GiAngularVector => v.map(value => value / Math.hypot(...v)) as unknown as GiAngularVector;
function receiver(degrees = 0): GiAngularReceiver {
  const angle = degrees * Math.PI / 180;
  return { position: [1, 2, 3], shadingNormal: [Math.sin(angle), 0, Math.cos(angle)], geometricNormal: [0, 0, 1] };
}
function options(r = receiver(), maxSamples = 16_384, upper: GiAngularVector = [1, 1, 1]): GiAngularMomentOptions {
  return { receiver: r, maxSamples, radianceUpperBound: upper };
}
function quadrature(consume: (direction: GiAngularVector, pdf: number) => void) {
  // Equal-solid-angle midpoint quadrature, NOT independent random draws. Its SE
  // is only descriptive; analytic error gates below do not rely on that SE.
  for (let z = 0; z < 128; z++) for (let phi = 0; phi < 128; phi++) {
    const ray = sampleGiAngularHemisphere([0, 0, 1], (z + 0.5) / 128, (phi + 0.5) / 128);
    consume(ray.direction, ray.pdf);
  }
}
function rgbClose(actual: readonly number[], expected: readonly number[], tolerance = 1e-10) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, i) => expect(Math.abs(value - expected[i])).toBeLessThanOrEqual(tolerance));
}

function batchStatistics(samples: readonly (readonly number[])[]) {
  const n = samples.length;
  const mean = [0, 1, 2].map(c => samples.reduce((sum, value) => sum + value[c], 0) / n);
  const covariance = [0, 1, 2].flatMap(c => [0, 1, 2].map(d =>
    samples.reduce((sum, value) => sum + (value[c] - mean[c]) * (value[d] - mean[d]), 0) / (n * (n - 1))));
  const luminance = samples.map(value => value[0] * 0.2126 + value[1] * 0.7152 + value[2] * 0.0722);
  const luminanceMean = luminance.reduce((sum, value) => sum + value, 0) / n;
  const luminanceSE = Math.sqrt(luminance.reduce((sum, value) => sum + (value - luminanceMean) ** 2, 0) / (n * (n - 1)));
  return { mean, covariance, se: [0, 4, 8].map(i => Math.sqrt(covariance[i])), luminanceMean, luminanceSE };
}
function checkStatistics(actual: GiAngularRgbStatistics, samples: readonly (readonly number[])[]) {
  const expected = batchStatistics(samples);
  rgbClose(actual.mean, expected.mean);
  rgbClose(actual.iidCovarianceOfMean!, expected.covariance);
  rgbClose(actual.iidStandardError!, expected.se);
  expect(actual.luminanceMean).toBeCloseTo(expected.luminanceMean, 11);
  expect(actual.luminanceIidStandardError).toBeCloseTo(expected.luminanceSE, 11);
}

describe('experimental fixed-receiver angular irradiance moments', () => {
  it('samples the full geometric hemisphere with the solid-angle PDF, independently of Ns', () => {
    const ng = normalize([1, -2, 3]);
    for (const u of [0.0001, 0.25, 0.5, 0.9999]) for (const v of [0, 0.13, 0.5, 0.9999]) {
      const ray = sampleGiAngularHemisphere(ng, u, v);
      expect(ray.pdf).toBe(1 / PI2);
      expect(Math.hypot(...ray.direction)).toBeCloseTo(1, 12);
      expect(dot(ray.direction, ng)).toBeCloseTo(u, 12);
    }
    for (const [u, v] of [[0, 0], [1, 0], [0.5, -1], [0.5, 1], [NaN, 0]]) {
      expect(() => sampleGiAngularHemisphere(ng, u, v)).toThrow('SAMPLE_COORDINATES');
    }
  });

  it('owns immutable receiver and radiance bounds and enforces the explicit sample budget', () => {
    const position: [number, number, number] = [1, 2, 3];
    const normal: [number, number, number] = [0, 0, 1];
    const bound: [number, number, number] = [1, 2, 3];
    const m = new GiAngularMomentAccumulator({ receiver: { position, shadingNormal: normal, geometricNormal: normal },
      radianceUpperBound: bound, maxSamples: 2 });
    position[0] = 100; normal[0] = 100; bound[0] = 100;
    const sample = { direction: [0, 0, 1] as const, radiance: [1, 2, 3] as const, pdf: GI_ANGULAR_UNIFORM_PDF };
    m.add(sample); m.add(sample);
    expect(m.sampleCount).toBe(2);
    rgbClose(m.reconstruct(receiver()).linear.mean, [PI2, 2 * PI2, 3 * PI2]);
    expect(() => m.add(sample)).toThrow('SAMPLE_BUDGET');
    expect(m.sampleCount).toBe(2);
    const snapshot = m.snapshot();
    expect([snapshot, snapshot.reference, snapshot.tangent, snapshot.bitangent, snapshot.sampleCovariance].every(Object.isFrozen)).toBe(true);
    for (const maxSamples of [0, 0.5, GI_ANGULAR_MAX_SAMPLES + 1, Infinity]) {
      expect(() => new GiAngularMomentAccumulator(options(receiver(), maxSamples))).toThrow('MAX_SAMPLES');
    }
    for (const upper of [[-1, 1, 1], [Infinity, 1, 1], [1e13, 1, 1]]) {
      expect(() => new GiAngularMomentAccumulator(options(receiver(), 1, upper as unknown as GiAngularVector))).toThrow('RADIANCE_BOUND');
    }
  });

  it('rejects wrong PDF, energy, geometric domain and changed receivers without mutating statistics', () => {
    const m = new GiAngularMomentAccumulator(options(receiver(), 10));
    const valid: GiAngularSample = { direction: [0, 0, 1], radiance: [0.2, 0.3, 0.4], pdf: GI_ANGULAR_UNIFORM_PDF };
    m.add(valid);
    const before = m.snapshot();
    const invalid: GiAngularSample[] = [
      { ...valid, pdf: 1 / Math.PI }, // Cosine-hemisphere PDF at this direction.
      { ...valid, direction: [0, 0, -1] }, { ...valid, direction: [1, 0, 0] },
      { ...valid, direction: [0, 0, 2] }, { ...valid, radiance: [-0.1, 0, 0] },
      { ...valid, radiance: [1.001, 0, 0] }, { ...valid, radiance: [NaN, 0, 0] },
    ];
    for (const sample of invalid) {
      expect(() => m.add(sample)).toThrow();
      expect(m.snapshot()).toEqual(before);
    }
    expect(() => m.reconstruct({ ...receiver(), position: [1.000000001, 2, 3] })).toThrow('POSITION_CHANGED');
    expect(() => m.reconstruct({ ...receiver(), geometricNormal: receiver(1).shadingNormal })).toThrow('GEOMETRIC_NORMAL_CHANGED');
    expect(() => m.reconstruct(receiver(11))).toThrow('ANGLE');
    expect(() => m.reconstruct(receiver(180))).toThrow('NORMAL_FACING');
  });

  it('matches uniform-field integrals at 0, 5, 10 degrees without extra pi or energy normalization', () => {
    const radiance: GiAngularVector = [0.2, 0.7, 1];
    const m = new GiAngularMomentAccumulator(options(receiver(), 16_384, radiance));
    const targets = [0, 5, 10].map(receiver);
    const audits = targets.map(target => new GiAngularPairedAudit(options(receiver(), 16_384, radiance), target));
    quadrature((direction, pdf) => {
      const sample = { direction, pdf, radiance };
      m.add(sample); audits.forEach(audit => audit.add(sample));
    });
    rgbClose(m.snapshot().reference, radiance.map(value => Math.PI * value));
    targets.forEach((target, i) => {
      const cosine = dot(target.shadingNormal, receiver().shadingNormal);
      const reconstructed = m.reconstruct(target), audit = audits[i].snapshot();
      const truth = radiance.map(value => Math.PI * value * (1 + cosine) / 2);
      const linear = radiance.map(value => Math.PI * value * cosine);
      rgbClose(reconstructed.linear.mean, linear);
      rgbClose(audit.directTarget.mean, truth, 2e-4);
      rgbClose(reconstructed.linear.mean, audit.linear.mean);
      rgbClose(reconstructed.integralRemainderUpperBound, radiance.map(value => Math.PI * value * (1 - cosine)));
      rgbClose(audit.directMinusLinear.mean, truth.map((value, c) => value - linear[c]), 2e-4);
      const expectedSampleVariance = Math.PI ** 2 * (4 / 3 - cosine * cosine);
      expect(reconstructed.linear.iidCovarianceOfMean![8] * m.sampleCount).toBeCloseTo(expectedSampleVariance, 3);
      // Uniform sky is a counterexample to guaranteed improvement: the exact
      // scalar and linear model biases have equal magnitude and opposite signs.
      expect(Math.abs(linear[2] - truth[2])).toBeCloseTo(Math.abs(Math.PI - truth[2]), 12);
    });
    expect(m.reconstruct(targets[0]).linear.mean).toEqual(m.snapshot().reference);
    const varianceRatio = m.reconstruct(targets[2]).linear.iidCovarianceOfMean![8] / m.reconstruct(targets[0]).linear.iidCovarianceOfMean![8];
    expect(varianceRatio).toBeCloseTo(1 + 3 * Math.sin(10 * Math.PI / 180) ** 2, 3);
    expect(varianceRatio).toBeGreaterThan(1.09);
  });

  it('exactly reconstructs a bounded bright patch entirely in both positive Ns hemispheres', () => {
    const reference = receiver(20), target = receiver(30), upper: GiAngularVector = [2, 0.7, 1];
    const audit = new GiAngularPairedAudit(options(reference, 16_384, upper), target);
    quadrature((direction, pdf) => {
      const radiance: GiAngularVector = direction[2] > 0.75 && direction[0] > 0 ? upper : [0, 0, 0];
      audit.add({ direction, pdf, radiance });
    });
    const result = audit.snapshot();
    // Independent spherical-patch vector integral: z>.75, -pi/2<phi<pi/2.
    const xIntegral = Math.PI / 2 - (0.75 * Math.sqrt(1 - 0.75 ** 2) + Math.asin(0.75));
    const zIntegral = Math.PI * (1 - 0.75 ** 2) / 2;
    const expected = upper.map(value => value * (target.shadingNormal[0] * xIntegral + target.shadingNormal[2] * zIntegral));
    rgbClose(result.directTarget.mean, expected, 5e-4);
    rgbClose(result.linear.mean, result.directTarget.mean);
    rgbClose(result.directMinusLinear.mean, [0, 0, 0]);
    expect(result.directMinusLinear.iidStandardError![0]).toBeLessThan(1e-14);
    expect(Math.abs(result.directMinusScalar.mean[0])).toBeGreaterThan(0.01);
  });

  it('exposes unrepresented new-hemisphere light and negative linear estimates on a lost bright band', () => {
    const newReference = receiver(80), newTarget = receiver(70);
    const gained = new GiAngularPairedAudit(options(newReference), newTarget);
    const lost = new GiAngularPairedAudit(options(newTarget), newReference);
    const lostMoments = new GiAngularMomentAccumulator(options(newTarget));
    quadrature((direction, pdf) => {
      const lit = dot(newReference.shadingNormal, direction) <= 0 && dot(newTarget.shadingNormal, direction) > 0;
      const radiance: GiAngularVector = lit ? [1, 1, 1] : [0, 0, 0];
      const sample = { direction, pdf, radiance };
      gained.add(sample); lost.add(sample); lostMoments.add(sample);
    });
    const g = gained.snapshot(), l = lost.snapshot(), reconstructed = lostMoments.reconstruct(newReference);
    expect(g.newHemisphereSamples).toBeGreaterThan(0);
    expect(g.positiveReferenceSamples).toBe(0);
    expect(g.positiveTargetSamples).toBeGreaterThan(0);
    rgbClose(g.scalarReuse.mean, [0, 0, 0]); rgbClose(g.linear.mean, [0, 0, 0]);
    expect(g.directTarget.mean[0]).toBeGreaterThan(0.001);
    expect(g.directTarget.mean[0]).toBeLessThan(g.integralRemainderUpperBound[0] + 2e-4);
    rgbClose(g.directMinusLinear.mean, g.directTarget.mean);
    expect(l.lostHemisphereSamples).toBeGreaterThan(0);
    rgbClose(l.directTarget.mean, [0, 0, 0]);
    expect(l.scalarReuse.mean[0]).toBeGreaterThan(0);
    expect(l.linear.mean[0]).toBeLessThan(0);
    rgbClose(reconstructed.linear.mean, l.linear.mean);
    rgbClose(reconstructed.nonnegativeEstimate, [0, 0, 0]);
    expect(reconstructed.linear.iidStandardError![0]).toBeGreaterThan(0);
    for (const result of [g, l]) {
      expect(result.maximumAbsoluteLinearSample[0]).toBeLessThanOrEqual(PI2 + 1e-12);
      expect(result.maximumDirectTargetSample[0]).toBeLessThanOrEqual(PI2 + 1e-12);
    }
  });

  it('preserves full component/RGB covariance and paired-difference SE against direct batch calculations', () => {
    const target = receiver(8), cfg = options(receiver(), 5, [2, 2, 3]);
    const m = new GiAngularMomentAccumulator(cfg), audit = new GiAngularPairedAudit(cfg, target);
    const directions = [[1, 0, 1], [-1, 0.3, 0.03], [0.2, -1, 2], [-1, -1, 1], [1, 2, 0.2]].map(v => normalize(v as unknown as GiAngularVector));
    const radiances: GiAngularVector[] = [[1, 0, 3], [2, 1, 0], [0.25, 2, 1], [0.5, 0.5, 2], [2, 0.2, 0.3]];
    const raw: number[][] = [], direct: number[][] = [], scalar: number[][] = [], linear: number[][] = [];
    directions.forEach((direction, i) => {
      const radiance = radiances[i], sample = { direction, radiance, pdf: GI_ANGULAR_UNIFORM_PDF };
      m.add(sample); audit.add(sample);
      // Independent expression: on the reference positive hemisphere the linear
      // per-ray contribution is simply L*(Ntarget.w)/pdf, including its sign.
      raw.push([direction[2], direction[0], direction[1]].flatMap(cosine => radiance.map(value => PI2 * value * cosine)));
      direct.push(radiance.map(value => PI2 * value * Math.max(0, dot(target.shadingNormal, direction))));
      scalar.push(radiance.map(value => PI2 * value * direction[2]));
      linear.push(radiance.map(value => PI2 * value * dot(target.shadingNormal, direction)));
    });
    const snapshot = m.snapshot(), mean = Array.from({ length: 9 }, (_, c) => raw.reduce((sum, value) => sum + value[c], 0) / 5);
    const sampleCovariance = Array.from({ length: 81 }, (_, index) => {
      const c = Math.floor(index / 9), d = index % 9;
      return raw.reduce((sum, value) => sum + (value[c] - mean[c]) * (value[d] - mean[d]), 0) / 4;
    });
    rgbClose([...snapshot.reference, ...snapshot.tangent, ...snapshot.bitangent], mean);
    rgbClose(snapshot.sampleCovariance!, sampleCovariance);
    checkStatistics(m.reconstruct(target).linear, linear);
    const result = audit.snapshot();
    checkStatistics(result.directTarget, direct); checkStatistics(result.scalarReuse, scalar); checkStatistics(result.linear, linear);
    checkStatistics(result.directMinusLinear, direct.map((value, i) => value.map((channel, c) => channel - linear[i][c])));
    checkStatistics(result.directMinusScalar, direct.map((value, i) => value.map((channel, c) => channel - scalar[i][c])));
    const a = target.shadingNormal;
    const falselyIndependentVariance = (a[2] ** 2 * sampleCovariance[0] + a[0] ** 2 * sampleCovariance[30]) / 5;
    expect(Math.abs(falselyIndependentVariance - result.linear.iidCovarianceOfMean![0])).toBeGreaterThan(0.01);
  });

  it('does not present missing variance or the angular integral bound as finite-sample certainty', () => {
    const m = new GiAngularMomentAccumulator(options(receiver(), 1));
    const audit = new GiAngularPairedAudit(options(receiver(), 1), receiver());
    expect(() => m.reconstruct(receiver())).toThrow('NO_SAMPLES');
    expect(() => audit.snapshot()).toThrow('NO_SAMPLES');
    const ray = sampleGiAngularHemisphere([0, 0, 1], 0.99, 0.25);
    const sample = { ...ray, radiance: [1, 1, 1] as const };
    m.add(sample); audit.add(sample);
    const reconstructed = m.reconstruct(receiver());
    expect(reconstructed.linear.iidStandardError).toBeNull();
    expect(reconstructed.linear.luminanceIidStandardError).toBeNull();
    expect(m.snapshot().sampleCovariance).toBeNull();
    rgbClose(reconstructed.integralRemainderUpperBound, [0, 0, 0]);
    // Actual uniform-sky irradiance is pi, while this one uniform-proposal draw
    // is noisy despite zero angular approximation error at the reference Ns.
    expect(Math.abs(reconstructed.linear.mean[0] - Math.PI)).toBeGreaterThan(3);
    expect(audit.snapshot().directMinusLinear.iidStandardError).toBeNull();
  });
});
