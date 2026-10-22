import type { Renderer } from '../../../src/renderer';
import type { WorldHit, WorldQueryResult, WorldRay } from '../../../src/world-query';

type Vec3 = readonly [number, number, number];
type MutableVec3 = [number, number, number];
export interface IntelSponzaLocalReceiver {
  readonly pixel: { readonly x: number; readonly y: number };
  readonly position: Vec3;
  readonly normal: Vec3;
  readonly geometricNormal: Vec3;
  readonly batch: string;
  readonly entityId: number;
}

const SAMPLES = 4096;
const MAX_POINTS = 16;
const SEED = 0x6d2b79f5;
const NORMAL_BIAS = 0.0001;
const T_MIN = 0.00001;
const NEAR_DISTANCE = 0.001;
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): MutableVec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];
const offset = (p: Vec3, n: Vec3): MutableVec3 => [
  p[0] + n[0] * NORMAL_BIAS, p[1] + n[1] * NORMAL_BIAS, p[2] + n[2] * NORMAL_BIAS,
];
function vector(v: Vec3, label: string): MutableVec3 {
  if (!Array.isArray(v) || v.length !== 3 || !v.every(Number.isFinite)) throw Error('SPONZA_LOCAL_REFERENCE_' + label);
  return [v[0], v[1], v[2]];
}
function unit(v: Vec3, label: string): MutableVec3 {
  const value = vector(v, label), length = Math.hypot(...value);
  if (!(length > 0) || !Number.isFinite(length)) throw Error('SPONZA_LOCAL_REFERENCE_' + label);
  return [value[0] / length, value[1] / length, value[2] / length];
}
function hash(value: number): number {
  let v = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  v = Math.imul(v ^ (v >>> 15), 0x846ca68b);
  return (v ^ (v >>> 16)) >>> 0;
}
function moments() { return { count: 0, mean: [0, 0, 0] as MutableVec3, m2: [0, 0, 0] as MutableVec3,
  luminanceMean: 0, luminanceM2: 0 }; }
function include(stats: ReturnType<typeof moments>, value: Vec3): void {
  stats.count++;
  for (let channel = 0; channel < 3; channel++) {
    const delta = value[channel] - stats.mean[channel];
    stats.mean[channel] += delta / stats.count;
    stats.m2[channel] += delta * (value[channel] - stats.mean[channel]);
  }
  const luminance = dot(value, [0.2126, 0.7152, 0.0722]), delta = luminance - stats.luminanceMean;
  stats.luminanceMean += delta / stats.count;
  stats.luminanceM2 += delta * (luminance - stats.luminanceMean);
}
function summarize(stats: ReturnType<typeof moments>) {
  return { mean: stats.mean, standardError: stats.m2.map(value => Math.sqrt(Math.max(0, value) / (SAMPLES - 1) / SAMPLES)),
    luminance: { mean: stats.luminanceMean, standardError: Math.sqrt(Math.max(0, stats.luminanceM2) / (SAMPLES - 1) / SAMPLES) } };
}
const sameInstance = (hit: WorldHit, other: { batch: string; entityId: number }) =>
  hit.batch === other.batch && hit.entityId === other.entityId;

/** Diagnostic reference only: explicit actual-raster receivers, independent CPU
 * cosine integration, and the renderer's shared query geometry/materials. No
 * spatial filtering, probe cache, receiver albedo, exposure, or extra render.
 * Points are processed serially: at most 4096 primary + 4096 shadow rays and
 * O(4096) temporary CPU records per point; at most 16 points / 32 trace calls.
 * Source mip 0 means the finest currently resident query texture, and does not
 * reproduce a raster derivative footprint. This is not a geometry truth oracle.
 */
export async function intelSponzaLocalReference(renderer: Renderer, points: readonly IntelSponzaLocalReceiver[],
  sky: Vec3, sunColor: Vec3, sunDirection: Vec3, maxDistance: number, options: { seed?: number } = {}) {
  if (!Array.isArray(points) || points.length < 1 || points.length > MAX_POINTS) throw Error('SPONZA_LOCAL_REFERENCE_POINT_BUDGET');
  const randomSeed = options.seed ?? SEED;
  if (!Number.isSafeInteger(randomSeed) || randomSeed < 0 || randomSeed > 0xffff_ffff) throw Error('SPONZA_LOCAL_REFERENCE_SEED');
  const skyRadiance = vector(sky, 'SKY'), sunRadiance = vector(sunColor, 'SUN'), sun = unit(sunDirection, 'SUN_DIRECTION');
  if ([...skyRadiance, ...sunRadiance].some(value => value < 0 || value > 65504)
    || !Number.isFinite(maxDistance) || maxDistance <= T_MIN || maxDistance > 1000) throw Error('SPONZA_LOCAL_REFERENCE_LIGHTING');
  // Validate and snapshot every caller input before any GPU query starts.
  const receivers = points.map(point => {
    if (!point || !point.pixel || !Number.isSafeInteger(point.pixel.x) || point.pixel.x < 0
      || !Number.isSafeInteger(point.pixel.y) || point.pixel.y < 0 || typeof point.batch !== 'string' || !point.batch.length
      || !Number.isSafeInteger(point.entityId) || point.entityId < 0 || point.entityId > 0xffff_ffff) {
      throw Error('SPONZA_LOCAL_REFERENCE_RECEIVER');
    }
    return { pixel: { ...point.pixel }, position: vector(point.position, 'POSITION'),
      normal: unit(point.normal, 'NORMAL'), geometricNormal: unit(point.geometricNormal, 'GEOMETRIC_NORMAL'),
      batch: point.batch, entityId: point.entityId };
  });
  let snapshot: { frame: number; materialRevision: number } | undefined;
  let queryCalls = 0, primaryRayCount = 0, shadowRayCount = 0;
  const trace = async (rays: WorldRay[]): Promise<(WorldHit | null)[]> => {
    if (!rays.length) return [];
    if (rays.length > SAMPLES) throw Error('SPONZA_LOCAL_REFERENCE_RAY_BUDGET');
    const result: WorldQueryResult = await renderer.traceRays(rays, { textureLod: 0 });
    queryCalls++;
    if (result.hits.length !== rays.length) throw Error('SPONZA_LOCAL_REFERENCE_HIT_COUNT');
    if (!snapshot) snapshot = { frame: result.frame, materialRevision: result.materialRevision };
    if (result.frame !== snapshot.frame || result.materialRevision !== snapshot.materialRevision) {
      throw Error('SPONZA_LOCAL_REFERENCE_SNAPSHOT_CHANGED');
    }
    return result.hits;
  };
  const rows = [];
  for (const receiver of receivers) {
    const n = receiver.normal, g = receiver.geometricNormal;
    const tangent = unit(cross(Math.abs(n[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0], n), 'TANGENT');
    const bitangent = cross(n, tangent), origin = offset(receiver.position, g);
    const primaryRays: WorldRay[] = [], sampleIds: number[] = [];
    // Common random numbers make neighboring receiver comparisons and reordered
    // input lists reproducible. Directions are transformed by each local frame.
    for (let sample = 0; sample < SAMPLES; sample++) {
      const seed = hash(sample ^ randomSeed);
      const u = (hash(seed ^ 0x68bc21eb) + 0.5) / 0x1_0000_0000;
      const angle = 2 * Math.PI * (hash(seed ^ 0x02e5be93) + 0.5) / 0x1_0000_0000;
      const x = Math.sqrt(u) * Math.cos(angle), y = Math.sqrt(u) * Math.sin(angle), z = Math.sqrt(1 - u);
      const direction = unit([tangent[0] * x + bitangent[0] * y + n[0] * z,
        tangent[1] * x + bitangent[1] * y + n[1] * z, tangent[2] * x + bitangent[2] * y + n[2] * z], 'SAMPLE');
      if (dot(direction, g) <= 0) continue;
      sampleIds.push(sample);
      primaryRays.push({ origin, direction, tMin: T_MIN, tMax: maxDistance });
    }
    const hits = await trace(primaryRays);
    primaryRayCount += primaryRays.length;
    const shadowRays: WorldRay[] = [], shadowIds: number[] = [];
    const hasSun = sunRadiance.some(value => value > 0);
    for (let index = 0; index < hits.length; index++) {
      const hit = hits[index];
      if (!hit || hit.shading !== 'pbr' || !hasSun || dot(hit.normal, sun) <= 0 || dot(hit.geometricNormal, sun) <= 0) continue;
      shadowIds.push(index);
      shadowRays.push({ origin: offset(hit.position, hit.geometricNormal), direction: sun, tMin: T_MIN, tMax: maxDistance });
    }
    const shadowHits = await trace(shadowRays);
    shadowRayCount += shadowRays.length;
    const visible = new Uint8Array(hits.length);
    let shadowNear = 0, shadowNearSameSurface = 0, shadowNearSameReceiver = 0, shadowOccluded = 0;
    let nearestShadow: number | null = null;
    for (let index = 0; index < shadowHits.length; index++) {
      const hit = shadowHits[index];
      if (!hit) { visible[shadowIds[index]] = 1; continue; }
      shadowOccluded++;
      nearestShadow = Math.min(nearestShadow ?? Infinity, hit.distance);
      if (hit.distance < NEAR_DISTANCE) {
        shadowNear++;
        if (sameInstance(hit, hits[shadowIds[index]]!)) shadowNearSameSurface++;
        if (sameInstance(hit, receiver)) shadowNearSameReceiver++;
      }
    }
    const contributions = { sky: moments(), sun: moments(), emissive: moments(), total: moments() };
    let primaryNear = 0, primaryNearSameReceiver = 0, missed = 0, primaryIndex = 0;
    let nearestPrimary: number | null = null, samplesAboveHdrLimit = 0;
    for (let sample = 0; sample < SAMPLES; sample++) {
      const skyValue: MutableVec3 = [0, 0, 0], sunValue: MutableVec3 = [0, 0, 0], emissiveValue: MutableVec3 = [0, 0, 0];
      if (primaryIndex < sampleIds.length && sampleIds[primaryIndex] === sample) {
        const hit = hits[primaryIndex];
        if (!hit) {
          missed++;
          for (let c = 0; c < 3; c++) skyValue[c] = Math.PI * skyRadiance[c];
        } else {
          nearestPrimary = Math.min(nearestPrimary ?? Infinity, hit.distance);
          if (hit.distance < NEAR_DISTANCE) { primaryNear++; if (sameInstance(hit, receiver)) primaryNearSameReceiver++; }
          for (let c = 0; c < 3; c++) {
            emissiveValue[c] = Math.PI * (hit.emissive[c] + (hit.shading === 'unlit' ? hit.baseColor[c] : 0));
            // The cosine PDF cancels the secondary Lambert 1/pi exactly.
            if (visible[primaryIndex]) sunValue[c] = 0.96 * hit.baseColor[c] * (1 - hit.metallic)
              * sunRadiance[c] * Math.max(0, dot(hit.normal, sun));
          }
        }
        primaryIndex++;
      }
      const total: MutableVec3 = [skyValue[0] + sunValue[0] + emissiveValue[0],
        skyValue[1] + sunValue[1] + emissiveValue[1], skyValue[2] + sunValue[2] + emissiveValue[2]];
      if (!total.every(value => Number.isFinite(value) && value >= 0)) throw Error('SPONZA_LOCAL_REFERENCE_NONFINITE');
      if (total.some(value => value > 65504)) samplesAboveHdrLimit++;
      include(contributions.sky, skyValue); include(contributions.sun, sunValue);
      include(contributions.emissive, emissiveValue); include(contributions.total, total);
    }
    rows.push({ receiver, samples: SAMPLES, geometricHemisphereClipped: SAMPLES - primaryRays.length,
      primary: { traced: primaryRays.length, misses: missed, hits: hits.length - missed, nearHits: primaryNear,
        nearSameReceiver: primaryNearSameReceiver, nearestDistance: nearestPrimary },
      shadows: { traced: shadowRays.length, occluded: shadowOccluded, visible: shadowRays.length - shadowOccluded,
        nearHits: shadowNear, nearSameSurface: shadowNearSameSurface, nearSameReceiver: shadowNearSameReceiver,
        nearestDistance: nearestShadow },
      contributions: { sky: summarize(contributions.sky), sun: summarize(contributions.sun),
        emissive: summarize(contributions.emissive), total: summarize(contributions.total) }, samplesAboveHdrLimit });
  }
  return { method: 'Independent CPU cosine Monte Carlo single-bounce irradiance at supplied raster receivers; shared renderer query geometry and hit materials, not independent geometry truth.',
    seed: randomSeed, samplesPerPoint: SAMPLES, normalBias: NORMAL_BIAS, tMin: T_MIN, nearDistance: NEAR_DISTANCE,
    maxDistance, skyRadiance, sunRadiance, sunDirection: sun, snapshot: snapshot ?? null,
    budget: { maxPoints: MAX_POINTS, maxSamplesPerPoint: SAMPLES, maxRaysPerPoint: 2 * SAMPLES,
      maxTotalRays: 2 * MAX_POINTS * SAMPLES, maxTraceCalls: 2 * MAX_POINTS },
    actual: { points: receivers.length, queryCalls, primaryRays: primaryRayCount, shadowRays: shadowRayCount },
    limitations: ['Pre-albedo diffuse irradiance; no receiver BRDF, spatial filtering, exposure, or cache tail.',
      'No ignored entity; nearSameSurface shadow counts compare against the secondary surface that emitted the shadow ray.',
      'Source mip 0 uses the finest resident query texture; raster footprints can differ.',
      'Geometrically clipped samples contribute zero and remain in all 4096-sample statistics.',
      'Standard errors are sample estimates for deterministic pseudorandom directions, not rigorous error bounds.',
      'Reference estimates are not HDR-clamped; samplesAboveHdrLimit identifies possible production clamp disagreement.'], rows };
}
