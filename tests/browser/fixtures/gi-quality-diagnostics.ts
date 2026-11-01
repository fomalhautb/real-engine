/** Diagnostic decomposition of the joint GI L01 and L04 errors. Every number is
 * derived from explicit, bounded readbacks (probe field, HDR pixels, world
 * queries) after the production frames; nothing here runs per frame. */
import type { DiffuseGiOptions } from '../../../src/diffuse-gi';
import type { DiffuseGiUpdateOptions } from '../../../src/gi-probe-schedule';
import { createJointLeakScene, createJointRoomsScene, JOINT_CAMERAS, JOINT_LEAK_BOXES, JOINT_LEAK_GRID, JOINT_PANEL_ON,
  JOINT_PANEL_START, JOINT_ROIS, JOINT_SKY, JOINT_SUN, type JointCameraName } from '../../../src/gi-joint-scene';
import { giReplayBasis, giReplayProbePosition, replayGiSurfaceIrradiance, type GiReplayGrid, type GiReplayProbe,
  type GiReplayOverrides } from '../../../src/gi-interpolation';
import { Renderer } from '../../../src/renderer';
import { giAxialMoments } from '../../../src/gi-visibility';
import type { SunShadowOptions } from '../../../src/sun-shadow';
import type { WorldHit, WorldRay } from '../../../src/world-query';
import { camera, canvasFor, pathTrace, rayBudget, roiWorlds, roomGrid, verifiedPixels } from './gi-joint-acceptance';

type Vec3 = [number, number, number];
type Rgb = [number, number, number];

const add = (a: readonly number[], b: readonly number[], s = 1): Vec3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a: readonly number[], s: number): Rgb => [a[0] * s, a[1] * s, a[2] * s];
const QUERY_CHUNK = 32_768;

/** Uniform Fibonacci sphere, the same construction as the GPU probe rays. */
function fibonacci(index: number, count: number): Vec3 {
  const y = 1 - 2 * (index + 0.5) / count;
  const radius = Math.sqrt(Math.max(0, 1 - y * y));
  const phi = index * 2.399963229728653;
  return [Math.cos(phi) * radius, y, Math.sin(phi) * radius];
}

async function traceAll(r: Renderer, rays: readonly WorldRay[]): Promise<(WorldHit | null)[]> {
  const hits: (WorldHit | null)[] = [];
  for (let start = 0; start < rays.length; start += QUERY_CHUNK) {
    hits.push(...(await r.traceRays(rays.slice(start, start + QUERY_CHUNK))).hits);
  }
  return hits;
}

export interface DiagnosticReceiver {
  position: Vec3; normal: Vec3; geometricNormal: Vec3; baseColor: Vec3; metallic: number;
}
function receiverOf(hit: WorldHit): DiagnosticReceiver {
  return { position: [...hit.position], normal: [...hit.normal], geometricNormal: [...hit.geometricNormal],
    baseColor: [...hit.baseColor], metallic: hit.metallic };
}
const albedo = (receiver: Pick<DiagnosticReceiver, 'baseColor' | 'metallic'>) =>
  scale(receiver.baseColor, (1 - receiver.metallic) * 0.96 / Math.PI);

interface Field { probes: GiReplayProbe[]; grid: GiReplayGrid }
/** CPU axial-depth moments of every probe from the production capture rays (world queries). */
export async function probeAxialMoments(r: Renderer): Promise<number[][]> {
  const options = r.diagnostics().diffuseGi!.options;
  const rays = options.raysPerProbe, directions = Array.from({ length: rays }, (_, i) => fibonacci(i, rays));
  const all: WorldRay[] = [];
  for (let probe = 0; probe < options.probeCount; probe++) for (const direction of directions) {
    all.push({ origin: giReplayProbePosition(options, probe), direction, tMin: 0.0001, tMax: options.maxDistance });
  }
  const hits = await traceAll(r, all);
  return Array.from({ length: options.probeCount }, (_, probe) => giAxialMoments(directions,
    hits.slice(probe * rays, (probe + 1) * rays).map(hit => hit ? hit.distance : null), options.maxDistance));
}
async function readField(r: Renderer, grid: GiReplayGrid): Promise<Field> {
  const field = await r.probeDiffuseGi();
  if (field.invalidProbes) throw new Error('GI_DIAG_INVALID_PROBES ' + field.invalidProbes);
  const axial = grid.visibilityRefinement === 'grid-axis-neighbor' ? await probeAxialMoments(r) : undefined;
  return { probes: field.probes.map((probe, index) => ({ coefficients: probe.coefficients, moments: probe.moments,
    maxRadiance: probe.maxRadiance, valid: probe.valid, ...(axial ? { axial: axial[index] } : {}) })), grid };
}
function replayGrid(r: Renderer): GiReplayGrid {
  const options = r.diagnostics().diffuseGi!.options;
  return { origin: options.origin, spacing: options.spacing, counts: options.counts, normalBias: options.normalBias,
    visibilityWeightThreshold: options.visibilityWeightThreshold, visibilityFilter: options.visibilityFilter,
    visibilityRefinement: options.visibilityRefinement };
}

interface Emitter { batch: string; center: Vec3; size: Vec3; radiance: Rgb }
interface RadianceModel {
  sky: Rgb; sun: { direction: Vec3; color: Rgb }; maxDistance: number; shadowBias: number;
  /** Batches whose emission is integrated by explicit area samples instead of direction rays. */
  emitters: Emitter[];
  /** Cross-frame diffuse feedback from a converged field (multi-bounce capture model). */
  feedback?: Field;
}

/** Capture radiance (diffuse-gi-update.wgsl integrate) arriving along each ray. */
async function rayRadiance(r: Renderer, rays: readonly WorldRay[], model: RadianceModel): Promise<Rgb[]> {
  const hits = await traceAll(r, rays);
  const area = new Set(model.emitters.map(emitter => emitter.batch));
  const shadowIndex: number[] = [], shadowRays: WorldRay[] = [];
  hits.forEach((hit, i) => {
    if (!hit || hit.shading === 'unlit' || !model.sun.color.some(value => value > 0)) return;
    if (dot(hit.normal, model.sun.direction) <= 0 || dot(hit.geometricNormal, model.sun.direction) <= 0) return;
    shadowIndex.push(i);
    shadowRays.push({ origin: add(hit.position, hit.geometricNormal, model.shadowBias), direction: model.sun.direction,
      tMin: 0.0001, tMax: model.maxDistance });
  });
  const shadowHits = await traceAll(r, shadowRays);
  const lit = new Set(shadowIndex.filter((_, i) => !shadowHits[i]));
  return hits.map((hit, i) => {
    if (!hit) return [...model.sky] as Rgb;
    const sampled = area.has(hit.batch);
    const radiance: Rgb = sampled ? [0, 0, 0] : [...hit.emissive] as Rgb;
    if (hit.shading === 'unlit') { if (!sampled) for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c]; return radiance; }
    const a = albedo(hit);
    if (lit.has(i)) {
      const nl = Math.max(0, dot(hit.normal, model.sun.direction));
      for (let c = 0; c < 3; c++) radiance[c] += a[c] * model.sun.color[c] * nl;
    }
    if (model.feedback) {
      const e = replayGiSurfaceIrradiance(model.feedback.probes, model.feedback.grid, hit.position, hit.normal,
        hit.geometricNormal).irradiance;
      for (let c = 0; c < 3; c++) radiance[c] += a[c] * e[c];
    }
    return radiance.map(value => Math.min(Math.max(value, 0), 65504)) as Rgb;
  });
}

/** Stratified samples on every face of an axis-aligned emitter box. */
function emitterSamples(emitter: Emitter, density: number) {
  const samples: { point: Vec3; normal: Vec3; area: number }[] = [];
  for (let axis = 0; axis < 3; axis++) for (const side of [-1, 1]) {
    const u = (axis + 1) % 3, v = (axis + 2) % 3;
    const nu = Math.max(1, Math.round(emitter.size[u] * density)), nv = Math.max(1, Math.round(emitter.size[v] * density));
    const cellArea = emitter.size[u] * emitter.size[v] / (nu * nv);
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const point: Vec3 = [...emitter.center];
      point[axis] += side * emitter.size[axis] / 2;
      point[u] += ((i + 0.5) / nu - 0.5) * emitter.size[u];
      point[v] += ((j + 0.5) / nv - 0.5) * emitter.size[v];
      const normal: Vec3 = [0, 0, 0]; normal[axis] = side;
      samples.push({ point, normal, area: cellArea });
    }
  }
  return samples;
}

export interface OracleResult {
  /** Clamped-cosine SH9 irradiance of a probe placed at the (biased) receiver. */
  sh9: Rgb;
  /** Exact cosine-weighted hemisphere irradiance at the same point and radiance. */
  exact: Rgb;
  /** Upper/lower hemisphere irradiance split (diagnostic for SH ringing). */
  lowerHemisphereFlux: number;
}

/** A probe placed at each receiver with `directions` Fibonacci rays plus area-sampled emitters. */
async function oracle(r: Renderer, receivers: readonly DiagnosticReceiver[], model: RadianceModel,
  options: { directions: number; bias: number; emitterDensity: number }): Promise<OracleResult[]> {
  const directions = Array.from({ length: options.directions }, (_, i) => fibonacci(i, options.directions));
  const origins = receivers.map(receiver => add(receiver.position, receiver.geometricNormal, options.bias));
  const coefficients = receivers.map(() => Array.from({ length: 9 }, () => [0, 0, 0]));
  const exact = receivers.map(() => [0, 0, 0] as Rgb), maximum = receivers.map(() => [0, 0, 0] as Rgb);
  const lower = receivers.map(() => 0);
  const weight = 4 * Math.PI / options.directions;
  // Bounded batches of receivers keep each query under the chunk size.
  const perBatch = Math.max(1, Math.floor(QUERY_CHUNK / options.directions));
  for (let start = 0; start < receivers.length; start += perBatch) {
    const indices = receivers.slice(start, start + perBatch).map((_, local) => start + local);
    const rays = indices.flatMap(index => directions.map(direction => ({ origin: origins[index], direction,
      tMin: 0.0001, tMax: model.maxDistance })));
    const radiance = await rayRadiance(r, rays, model);
    indices.forEach((index, local) => directions.forEach((direction, d) => {
      const value = radiance[local * directions.length + d], basis = giReplayBasis(direction);
      const cosine = dot(direction, receivers[index].normal);
      for (let c = 0; c < 3; c++) {
        for (let k = 0; k < 9; k++) coefficients[index][k][c] += value[c] * basis[k] * weight;
        if (cosine > 0) exact[index][c] += value[c] * cosine * weight;
        maximum[index][c] = Math.max(maximum[index][c], value[c]);
      }
      if (cosine < 0) lower[index] += (0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2]) * -cosine * weight;
    }));
  }
  for (const emitter of model.emitters) {
    const samples = emitterSamples(emitter, options.emitterDensity);
    const emitted = receivers.map(() => false);
    const perEmitterBatch = Math.max(1, Math.floor(QUERY_CHUNK / samples.length));
    for (let start = 0; start < receivers.length; start += perEmitterBatch) {
      const jobs: { index: number; sample: (typeof samples)[number]; direction: Vec3; solidAngle: number }[] = [];
      for (let index = start; index < Math.min(receivers.length, start + perEmitterBatch); index++) {
        for (const sample of samples) {
          const delta = sub(sample.point, origins[index]), distance = Math.hypot(...delta);
          const direction = scale(delta, 1 / distance) as Vec3;
          const cosine = -dot(direction, sample.normal);
          if (cosine <= 0) continue;
          jobs.push({ index, sample, direction, solidAngle: sample.area * cosine / (distance * distance) });
        }
      }
      const hits = await traceAll(r, jobs.map(job => {
        const distance = Math.hypot(...sub(job.sample.point, origins[job.index]));
        return { origin: origins[job.index], direction: job.direction, tMin: 0.0001, tMax: Math.max(0.0002, distance - 0.0005) };
      }));
      jobs.forEach((job, i) => {
        if (hits[i]) return;
        emitted[job.index] = true;
        const basis = giReplayBasis(job.direction), cosine = dot(job.direction, receivers[job.index].normal);
        for (let c = 0; c < 3; c++) {
          for (let k = 0; k < 9; k++) coefficients[job.index][k][c] += emitter.radiance[c] * basis[k] * job.solidAngle;
          if (cosine > 0) exact[job.index][c] += emitter.radiance[c] * cosine * job.solidAngle;
        }
        if (cosine < 0) lower[job.index] += (0.2126 * emitter.radiance[0] + 0.7152 * emitter.radiance[1]
          + 0.0722 * emitter.radiance[2]) * -cosine * job.solidAngle;
      });
    }
    receivers.forEach((_, index) => { if (emitted[index]) for (let c = 0; c < 3; c++) maximum[index][c] += emitter.radiance[c]; });
  }
  const convolution = [Math.PI, 2.094395102, 2.094395102, 2.094395102, 0.785398163, 0.785398163, 0.785398163, 0.785398163, 0.785398163];
  return receivers.map((receiver, index) => {
    const basis = giReplayBasis(receiver.normal);
    const sh9: Rgb = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      for (let k = 0; k < 9; k++) sh9[c] += coefficients[index][k][c] * basis[k] * convolution[k];
      sh9[c] = Math.min(Math.max(sh9[c], 0), maximum[index][c] * Math.PI);
    }
    return { sh9, exact: exact[index], lowerHemisphereFlux: lower[index] };
  });
}

/** Binary segment visibility from each biased receiver to its interpolation corners. */
async function cornerVisibility(r: Renderer, field: Field, receivers: readonly DiagnosticReceiver[], bias: number) {
  const corners = receivers.map(receiver => replayGiSurfaceIrradiance(field.probes, field.grid, receiver.position,
    receiver.normal, receiver.geometricNormal, { normalBias: bias }).corners);
  const rays: WorldRay[] = [], owners: { receiver: number; probe: number }[] = [];
  receivers.forEach((receiver, index) => {
    const origin = add(receiver.position, receiver.geometricNormal, bias);
    for (const corner of corners[index]) {
      const delta = sub(corner.position, origin), distance = Math.hypot(...delta);
      if (distance < 0.0005) continue;
      rays.push({ origin, direction: scale(delta, 1 / distance) as Vec3, tMin: 0.0001, tMax: distance - 0.0002 });
      owners.push({ receiver: index, probe: corner.index });
    }
  });
  const hits = await traceAll(r, rays);
  const visible = receivers.map(() => new Map<number, boolean>());
  owners.forEach((owner, i) => visible[owner.receiver].set(owner.probe, !hits[i]));
  return visible;
}

type ReplayVariant = { name: string; overrides: (visible: Map<number, boolean>) => GiReplayOverrides; filter?: GiReplayGrid['visibilityFilter'] };
const REPLAY_VARIANTS: ReplayVariant[] = [
  { name: 'production', overrides: () => ({}) },
  { name: 'perTap', overrides: () => ({}), filter: 'per-tap-chebyshev-bilinear' },
  { name: 'exactVisibility', overrides: visible => ({ visibility: corner => visible.get(corner.index) === false ? 0 : 1 }) },
  { name: 'exactVisibilityNoFacing', overrides: visible => ({ visibility: corner => visible.get(corner.index) === false ? 0 : 1,
    facing: () => 1 }) },
  { name: 'trilinear', overrides: () => ({ visibility: () => 1, facing: () => 1 }) },
  // Remove only false visibility (leaks) or only false occlusion (self-shadowing).
  { name: 'leakRemoved', overrides: visible => ({ visibility: corner => visible.get(corner.index) === false ? 0 : corner.chebyshev }) },
  { name: 'selfShadowRemoved', overrides: visible => ({ visibility: corner => visible.get(corner.index) === false ? corner.chebyshev : 1 }) },
];

function replayVariants(field: Field, receivers: readonly DiagnosticReceiver[], visible: Map<number, boolean>[]) {
  return Object.fromEntries(REPLAY_VARIANTS.map(variant => {
    const grid = variant.filter ? { ...field.grid, visibilityFilter: variant.filter } : field.grid;
    return [variant.name, receivers.map((receiver, index) => {
      const result = replayGiSurfaceIrradiance(field.probes, grid, receiver.position, receiver.normal,
        receiver.geometricNormal, variant.overrides(visible[index]));
      return result.irradiance;
    })];
  })) as Record<string, Rgb[]>;
}

function cornerSummary(field: Field, receivers: readonly DiagnosticReceiver[], visible: Map<number, boolean>[]) {
  return receivers.map((receiver, index) => {
    const result = replayGiSurfaceIrradiance(field.probes, field.grid, receiver.position, receiver.normal, receiver.geometricNormal);
    const weightTotal = result.corners.reduce((sum, corner) => sum + corner.weight, 0);
    let occludedWeight = 0, visibleChebyshev = 0, visibleCount = 0, occludedChebyshev = 0, occludedCount = 0;
    for (const corner of result.corners) {
      if (visible[index].get(corner.index) === false) { occludedWeight += corner.weight; occludedChebyshev += corner.chebyshev; occludedCount++; }
      else { visibleChebyshev += corner.chebyshev; visibleCount++; }
    }
    return { support: result.support, supportWeight: result.supportWeight, occludedWeightShare: weightTotal ? occludedWeight / weightTotal : 0,
      occludedCorners: occludedCount, visibleCorners: visibleCount,
      meanChebyshevVisible: visibleCount ? visibleChebyshev / visibleCount : null,
      meanChebyshevOccluded: occludedCount ? occludedChebyshev / occludedCount : null,
      nearestProbeDistance: Math.min(...result.corners.map(corner => corner.distance)) };
  });
}

/** Every production probe ray of the field with the inputs the capture integrates:
 * [distance, position3, normal3, geometricNormal3, albedo3, sunLit, emissive3, unlit, areaSampled]
 * or null for a miss. Lets candidate interpolation and capture rules be iterated offline. */
async function probeCapture(r: Renderer, grid: GiReplayGrid, model: RadianceModel, raysPerProbe: number) {
  const count = grid.counts[0] * grid.counts[1] * grid.counts[2];
  const rays: WorldRay[] = [];
  for (let probe = 0; probe < count; probe++) for (let i = 0; i < raysPerProbe; i++) {
    rays.push({ origin: giReplayProbePosition(grid, probe), direction: fibonacci(i, raysPerProbe), tMin: 0.0001, tMax: model.maxDistance });
  }
  const hits = await traceAll(r, rays);
  const shadowIndex: number[] = [], shadowRays: WorldRay[] = [];
  hits.forEach((hit, i) => {
    if (!hit || hit.shading === 'unlit') return;
    if (dot(hit.normal, model.sun.direction) <= 0 || dot(hit.geometricNormal, model.sun.direction) <= 0) return;
    shadowIndex.push(i);
    shadowRays.push({ origin: add(hit.position, hit.geometricNormal, model.shadowBias), direction: model.sun.direction,
      tMin: 0.0001, tMax: model.maxDistance });
  });
  const shadowHits = await traceAll(r, shadowRays);
  const lit = new Set(shadowIndex.filter((_, i) => !shadowHits[i]));
  const area = new Set(model.emitters.map(emitter => emitter.batch));
  const round = (value: number) => Math.round(value * 1e6) / 1e6;
  return { raysPerProbe, rays: hits.map((hit, i) => hit ? [hit.distance, ...hit.position, ...hit.normal, ...hit.geometricNormal,
    ...albedo(hit), lit.has(i) ? 1 : 0, ...hit.emissive, hit.shading === 'unlit' ? 1 : 0, area.has(hit.batch) ? 1 : 0].map(round) : null) };
}
const fieldDump = (field: Field) => field.probes.map(probe => ({ coefficients: probe.coefficients, moments: probe.moments,
  maxRadiance: probe.maxRadiance, valid: probe.valid, ...(probe.axial ? { axial: probe.axial } : {}) }));

const emitterOf = (radiance: readonly number[]): Emitter =>
  ({ batch: 'joint-panel', center: [...JOINT_PANEL_START], size: [1.2, 0.5, 0.04], radiance: [radiance[0], radiance[1], radiance[2]] });

export interface L01DiagnosticsInput {
  width: number; height: number; roiGrid: number; warmup: number; singleBounceWarmup: number;
  update?: DiffuseGiUpdateOptions;
  /** Optional grid/visibility overrides for controlled experiments (density, filters). */
  grid?: Partial<DiffuseGiOptions>;
  oracleDirections: number; emitterDensity: number;
  references: { paths: number; bounces: number; seed: number; normalBias?: number }[];
  /** Include both probe fields and every production probe ray (offline iteration). */
  dumpFields?: boolean;
}

/** L01: GPU HDR, CPU replay variants, receiver-placed oracle probes and references at the frozen ROI points. */
export async function runL01Diagnostics(input: L01DiagnosticsInput) {
  const element = canvasFor(input.width, input.height);
  const scene = createJointRoomsScene();
  const grid: DiffuseGiOptions = { ...roomGrid({ update: input.update }), ...input.grid };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, scene.batches, { worldQueries: { maxRays: Math.max(QUERY_CHUNK, rayBudget(grid)) },
      diffuseGi: grid, antiAliasing: 'none' });
    const r = renderer;
    r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
    r.render(JOINT_CAMERAS.roomA.eye as Vec3, JOINT_CAMERAS.roomA.target as Vec3);
    const rois: { name: string; camera: JointCameraName; pixels: { x: number; y: number }[]; hits: WorldHit[] }[] = [];
    for (const roi of JOINT_ROIS.filter(item => item.camera !== 'roomBAway')) {
      const verified = await verifiedPixels(r, roi.camera, roiWorlds(roi, input.roiGrid),
        hit => hit.batch === roi.batch && hit.entityId === roi.entityId);
      rois.push({ name: roi.name, camera: roi.camera, pixels: verified.pixels, hits: verified.hits });
    }
    const receivers = rois.flatMap(roi => roi.hits.map(receiverOf));
    const read = async () => {
      const values: Record<string, number[][]> = {};
      for (const name of ['roomA', 'roomB'] as const) {
        r.render(JOINT_CAMERAS[name].eye as Vec3, JOINT_CAMERAS[name].target as Vec3);
        for (const roi of rois) if (roi.camera === name) values[roi.name] = (await r.probeHdr(roi.pixels)).map(v => v.slice(0, 3));
      }
      return rois.flatMap(roi => values[roi.name]);
    };
    const converge = async (frames: number) => {
      for (let i = 0; i < frames; i++) {
        const view = JOINT_CAMERAS[i % 2 ? 'roomB' : 'roomA'];
        r.render(view.eye as Vec3, view.target as Vec3);
      }
    };
    const gridReplay = replayGrid(r);
    const model = (feedback?: Field): RadianceModel => ({ sky: [...JOINT_SKY], maxDistance: r.diagnostics().diffuseGi!.options.maxDistance,
      shadowBias: r.diagnostics().diffuseGi!.options.shadowNormalBias,
      sun: { direction: [...JOINT_SUN.direction] as Vec3, color: [...JOINT_SUN.color] as Rgb },
      emitters: [emitterOf(JOINT_PANEL_ON)], feedback });
    await converge(input.warmup);
    const hdrMulti = await read();
    const multiField = await readField(r, gridReplay);
    r.setDiffuseGiEnabled(false);
    const hdrOff = await read();
    r.setDiffuseGiEnabled(true);
    r.setDiffuseGiMultiBounce(false);
    await converge(input.singleBounceWarmup);
    const hdrSingle = await read();
    const singleField = await readField(r, gridReplay);
    const bias = gridReplay.normalBias;
    const visible = await cornerVisibility(r, multiField, receivers, bias);
    const replay = { multi: replayVariants(multiField, receivers, visible), single: replayVariants(singleField, receivers, visible) };
    // Normal-bias sensitivity of the production interpolation.
    const biasSweep = Object.fromEntries([0.005, 0.015, 0.06, 0.1].map(value => [String(value), receivers.map(receiver =>
      replayGiSurfaceIrradiance(multiField.probes, multiField.grid, receiver.position, receiver.normal, receiver.geometricNormal,
        { normalBias: value }).irradiance)]));
    const corners = cornerSummary(multiField, receivers, visible);
    const oracleOptions = { directions: input.oracleDirections, bias, emitterDensity: input.emitterDensity };
    const oracleSingle = await oracle(r, receivers, model(), oracleOptions);
    const oracleMulti = await oracle(r, receivers, model(multiField), oracleOptions);
    const oracleMultiSurface = await oracle(r, receivers, model(multiField), { ...oracleOptions, bias: 0.0005 });
    const references = [];
    for (const reference of input.references) {
      const traced = await pathTrace(r, rois.flatMap(roi => roi.hits), { paths: reference.paths, bounces: reference.bounces,
        seed: reference.seed, batchRays: QUERY_CHUNK, sky: [...JOINT_SKY] as Vec3,
        sun: { direction: [...JOINT_SUN.direction] as Vec3, color: [...JOINT_SUN.color] as Vec3 },
        normalBias: reference.normalBias ?? grid.normalBias ?? 0.03 });
      references.push({ ...reference, oneBounce: traced.oneBounce, total: traced.total, perBounce: traced.perBounce, rays: traced.rays });
    }
    const capture = input.dumpFields ? await probeCapture(r, gridReplay, model(), r.diagnostics().diffuseGi!.options.raysPerProbe) : null;
    const errors = [...r.errors];
    const counts = rois.map(roi => ({ name: roi.name, count: roi.hits.length }));
    const result = { adapter: r.info.adapterInfo, grid: { origin: gridReplay.origin, spacing: gridReplay.spacing, counts: gridReplay.counts,
      normalBias: bias, visibilityFilter: gridReplay.visibilityFilter, threshold: gridReplay.visibilityWeightThreshold },
      rois: counts, receivers, albedo: receivers.map(albedo), hdr: { multi: hdrMulti, off: hdrOff, single: hdrSingle },
      replay, biasSweep, corners, oracle: { single: oracleSingle, multi: oracleMulti, multiSurface: oracleMultiSurface },
      references, errors,
      ...(input.dumpFields ? { fields: { multi: fieldDump(multiField), single: fieldDump(singleField) }, capture } : {}) };
    r.dispose();
    return { ...result, afterDispose: r.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

/** Box interior containing a world point, if any (leak scene). */
function boxOf(point: readonly number[]): keyof typeof JOINT_LEAK_BOXES | null {
  for (const [name, box] of Object.entries(JOINT_LEAK_BOXES) as [keyof typeof JOINT_LEAK_BOXES, { center: number }][]) {
    if (Math.abs(point[0] - box.center) < 0.5 && point[1] > 0.1 && point[1] < 1.1 && Math.abs(point[2]) < 0.5) return name;
  }
  return null;
}

export interface L04DiagnosticsInput {
  width: number; height: number; warmup: number; samples: [number, number];
  update?: DiffuseGiUpdateOptions; grid?: Partial<DiffuseGiOptions>;
  probeRays?: number;
  /** Include both probe fields and every production probe ray (offline iteration). */
  dumpFields?: boolean;
}

/** L04: interior receivers of the three boxes; direct (shadow map) and GI leak terms separated. */
export async function runL04Diagnostics(input: L04DiagnosticsInput) {
  const element = canvasFor(input.width, input.height);
  const { batches } = createJointLeakScene();
  const grid: DiffuseGiOptions = { ...JOINT_LEAK_GRID, update: input.update, ...input.grid };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, batches, { worldQueries: { maxRays: Math.max(QUERY_CHUNK, rayBudget(grid)) },
      diffuseGi: grid, antiAliasing: 'none' });
    const r = renderer;
    r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
    const views = ['blackBox', 'thinBox', 'slitBox'] as const;
    const warm = async (frames: number) => {
      for (let i = 0; i < Math.max(1, frames); i++) {
        const view = JOINT_CAMERAS[views[i % 3]];
        r.render(view.eye as Vec3, view.target as Vec3);
      }
    };
    await warm(input.warmup);
    const pixels: { x: number; y: number }[] = [];
    for (let j = 0; j < input.samples[1]; j++) for (let i = 0; i < input.samples[0]; i++) {
      pixels.push({ x: Math.floor((i + 0.5) * r.stats.width / input.samples[0]), y: Math.floor((j + 0.5) * r.stats.height / input.samples[1]) });
    }
    // Interior receivers seen by each view (camera rays; misses and exterior hits are dropped).
    const receivers: Record<string, { pixels: { x: number; y: number }[]; hits: WorldHit[] }> = {};
    for (const name of views) {
      const view = camera(name, r.stats.width, r.stats.height);
      const hits = (await r.traceRays(pixels.map(pixel => view.ray(pixel)))).hits;
      const kept = pixels.map((pixel, i) => ({ pixel, hit: hits[i] }))
        .filter((item): item is { pixel: { x: number; y: number }; hit: WorldHit } => !!item.hit
          && boxOf(add(item.hit.position, item.hit.geometricNormal, 0.01)) === JOINT_LEAK_BOXES_KEY[name]);
      receivers[name] = { pixels: kept.map(item => item.pixel), hits: kept.map(item => item.hit) };
    }
    const read = async () => {
      const values: Record<string, number[][]> = {};
      for (const name of views) {
        r.render(JOINT_CAMERAS[name].eye as Vec3, JOINT_CAMERAS[name].target as Vec3);
        values[name] = (await r.probeHdr(receivers[name].pixels)).map(v => v.slice(0, 3));
      }
      const outside = { eye: [0.25, 1.6, 2.4] as Vec3, target: [0.25, 0.2, 1.1] as Vec3 };
      r.render(outside.eye, outside.target);
      values.outside = (await r.probeHdr([{ x: Math.floor(r.stats.width / 2), y: Math.floor(r.stats.height * 0.7) }])).map(v => v.slice(0, 3));
      return values;
    };
    const gridReplay = replayGrid(r);
    const hdrMulti = await read();
    const multiField = await readField(r, gridReplay);
    r.setDiffuseGiMultiBounce(false);
    await warm(input.warmup);
    const hdrSingle = await read();
    const singleField = await readField(r, gridReplay);
    r.setDiffuseGiMultiBounce(true);
    r.setDiffuseGiEnabled(false);
    const hdrOff = await read();
    r.setDiffuseGiEnabled(true);
    const all = views.flatMap(name => receivers[name].hits);
    const sunRays = all.map(hit => ({ origin: add(hit.position, hit.geometricNormal, 0.0001), direction: [...JOINT_SUN.direction] as Vec3,
      tMin: 0.0001, tMax: 60 }));
    const sunHits = await traceAll(r, sunRays);
    const sunVisible = sunHits.map(hit => !hit);
    const allReceivers = all.map(receiverOf);
    const visible = await cornerVisibility(r, multiField, allReceivers, gridReplay.normalBias);
    // Classify interpolation corners by box membership of the probe position.
    const insideOnly = (receiverBox: string | null) => ({ visibility: (corner: { position: readonly number[]; chebyshev: number }) =>
      boxOf(corner.position) === receiverBox ? corner.chebyshev : 0 });
    const classify = (field: Field) => allReceivers.map(receiver => {
      const box = boxOf(add(receiver.position, receiver.geometricNormal, 0.01));
      const result = replayGiSurfaceIrradiance(field.probes, field.grid, receiver.position, receiver.normal, receiver.geometricNormal);
      let inside = 0, outside = 0;
      const outsideIrradiance: Rgb = [0, 0, 0];
      for (const corner of result.corners) {
        if (boxOf(corner.position) === box) inside += corner.weight;
        else { outside += corner.weight; for (let c = 0; c < 3; c++) outsideIrradiance[c] += corner.irradiance[c] * corner.weight; }
      }
      const total = inside + outside;
      const contribution = scale(outsideIrradiance, total ? result.supportWeight / total : 0);
      return { box, insideWeight: inside, outsideWeight: outside, outsideShare: total ? outside / total : 0,
        outsideContribution: contribution, irradiance: result.irradiance,
        insideOnly: replayGiSurfaceIrradiance(field.probes, field.grid, receiver.position, receiver.normal, receiver.geometricNormal,
          insideOnly(box)).irradiance };
    });
    const probeStats = (field: Field) => field.probes.map((probe, index) => {
      const position = giReplayProbePosition(field.grid, index);
      const box = boxOf(position);
      if (!box) return null;
      // Mean SH irradiance over the six axis normals: interior probes should be dark.
      const axes: Vec3[] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      const irradiance = axes.map(axis => {
        const basis = giReplayBasis(axis);
        const conv = [Math.PI, 2.094395102, 2.094395102, 2.094395102, 0.785398163, 0.785398163, 0.785398163, 0.785398163, 0.785398163];
        return [0, 1, 2].map(c => Math.max(0, probe.coefficients.reduce((sum, coefficient, k) => sum + coefficient[c] * basis[k] * conv[k], 0)));
      });
      return { index, position, box, meanAxisIrradiance: [0, 1, 2].map(c => irradiance.reduce((s, v) => s + v[c], 0) / 6),
        maxRadiance: probe.maxRadiance };
    }).filter(item => item);
    // Probe rays from interior probes: count escapes (miss or exterior hit) with the production directions.
    const interiorProbes = multiField.probes.map((_, index) => index).filter(index => boxOf(giReplayProbePosition(multiField.grid, index)));
    const raysPerProbe = input.probeRays ?? (r.diagnostics().diffuseGi!.options.raysPerProbe);
    const probeRays = interiorProbes.flatMap(index => Array.from({ length: raysPerProbe }, (_, i) => ({
      origin: giReplayProbePosition(multiField.grid, index), direction: fibonacci(i, raysPerProbe), tMin: 0.0001,
      tMax: r.diagnostics().diffuseGi!.options.maxDistance })));
    const probeHits = await traceAll(r, probeRays);
    const escapes = interiorProbes.map((index, p) => {
      const box = boxOf(giReplayProbePosition(multiField.grid, index));
      let escaped = 0;
      for (let i = 0; i < raysPerProbe; i++) {
        const hit = probeHits[p * raysPerProbe + i];
        if (!hit || boxOf(add(hit.position, hit.geometricNormal, 0.01)) !== box) escaped++;
      }
      return { index, box, escaped };
    });
    const replay = { multi: replayVariants(multiField, allReceivers, visible), single: replayVariants(singleField, allReceivers, visible) };
    const options = r.diagnostics().diffuseGi!.options;
    const capture = input.dumpFields ? await probeCapture(r, gridReplay, { sky: [...JOINT_SKY], maxDistance: options.maxDistance,
      shadowBias: options.shadowNormalBias, sun: { direction: [...JOINT_SUN.direction] as Vec3, color: [...JOINT_SUN.color] as Rgb },
      emitters: [] }, options.raysPerProbe) : null;
    const errors = [...r.errors];
    const result = { adapter: r.info.adapterInfo, grid: gridReplay,
      ...(input.dumpFields ? { fields: { multi: fieldDump(multiField), single: fieldDump(singleField) }, capture,
        outsideHdr: { multi: hdrMulti.outside, off: hdrOff.outside } } : {}),
      views: Object.fromEntries(views.map(name => [name, receivers[name].hits.length])), receivers: allReceivers,
      receiverBoxes: allReceivers.map(receiver => boxOf(add(receiver.position, receiver.geometricNormal, 0.01))),
      albedo: allReceivers.map(albedo), sunVisible,
      hdr: { multi: hdrMulti, single: hdrSingle, off: hdrOff },
      replay, classification: { multi: classify(multiField), single: classify(singleField) },
      corners: cornerSummary(multiField, allReceivers, visible),
      probes: { multi: probeStats(multiField), single: probeStats(singleField) }, escapes, errors };
    r.dispose();
    return { ...result, afterDispose: r.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

const JOINT_LEAK_BOXES_KEY = { blackBox: 'black', thinBox: 'thin', slitBox: 'slit' } as const;

/** L04 direct term only (GI disabled, no ambient fill): the sun shadow map
 * against exact world-query sun visibility, for several shadow configurations. */
export async function runL04DirectLeak(input: { width: number; height: number; samples: [number, number];
  shadows: { name: string; options?: SunShadowOptions }[] }) {
  const results: Record<string, unknown> = {};
  for (const shadow of input.shadows) {
    const element = canvasFor(input.width, input.height);
    const { batches } = createJointLeakScene();
    let renderer: Renderer | undefined;
    try {
      renderer = await Renderer.create(element, batches, { worldQueries: { maxRays: QUERY_CHUNK },
        diffuseGi: { ...JOINT_LEAK_GRID, raysPerProbe: 16 }, antiAliasing: 'none', sunShadow: shadow.options });
      const r = renderer;
      r.setDiffuseGiEnabled(false);
      r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
      r.render(JOINT_CAMERAS.blackBox.eye as Vec3, JOINT_CAMERAS.blackBox.target as Vec3);
      const pixels: { x: number; y: number }[] = [];
      for (let j = 0; j < input.samples[1]; j++) for (let i = 0; i < input.samples[0]; i++) {
        pixels.push({ x: Math.floor((i + 0.5) * r.stats.width / input.samples[0]), y: Math.floor((j + 0.5) * r.stats.height / input.samples[1]) });
      }
      const views: Record<string, unknown> = {};
      for (const name of ['blackBox', 'thinBox', 'slitBox'] as const) {
        const view = camera(name, r.stats.width, r.stats.height);
        r.render(JOINT_CAMERAS[name].eye as Vec3, JOINT_CAMERAS[name].target as Vec3);
        const hits = (await r.traceRays(pixels.map(pixel => view.ray(pixel)))).hits;
        const kept = pixels.map((pixel, i) => ({ pixel, hit: hits[i] }))
          .filter((item): item is { pixel: { x: number; y: number }; hit: WorldHit } => !!item.hit
            && boxOf(add(item.hit.position, item.hit.geometricNormal, 0.01)) === JOINT_LEAK_BOXES_KEY[name]);
        const hdr = (await r.probeHdr(kept.map(item => item.pixel))).map(v => v.slice(0, 3));
        const sun = await traceAll(r, kept.map(item => ({ origin: add(item.hit.position, item.hit.geometricNormal, 0.0001),
          direction: [...JOINT_SUN.direction] as Vec3, tMin: 0.0001, tMax: 60 })));
        const depth = await r.probeDepth(kept.map(item => item.hit.position));
        views[name] = { pixels: kept.map(item => item.pixel), positions: kept.map(item => item.hit.position),
          normals: kept.map(item => item.hit.geometricNormal), hdr, sunVisible: sun.map(hit => !hit), shadowDepth: depth.map(d => d.shadow) };
      }
      r.render([0.25, 1.6, 2.4], [0.25, 0.2, 1.1]);
      const outside = (await r.probeHdr([{ x: Math.floor(r.stats.width / 2), y: Math.floor(r.stats.height * 0.7) }]))[0].slice(0, 3);
      results[shadow.name] = { sunShadow: r.diagnostics().sunShadow, outside, views, errors: [...r.errors] };
      r.dispose();
      (results[shadow.name] as Record<string, unknown>).afterDispose = r.stats.ownedGpuBytes;
    } finally { renderer?.dispose(); element.remove(); }
  }
  return results;
}
