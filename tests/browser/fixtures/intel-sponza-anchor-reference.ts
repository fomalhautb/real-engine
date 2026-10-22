import { Renderer } from '../../../src/renderer';
import { multiply, perspective, lookAt } from '../../../src/math';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { planGeometryStreamBudgets } from '../../../src/geometry-stream-budget';
import type { SceneBatch } from '../../../src/contracts';
import { packWorldRays, type WorldRay } from '../../../src/world-query';
import { loadIntelSponzaAssets, type IntelSponzaAssetOptions } from './intel-sponza-assets';
import { intelSponzaLocalReference, type IntelSponzaLocalReceiver } from './intel-sponza-local-reference';

type Vec3 = readonly [number, number, number];
type LocalReference = Awaited<ReturnType<typeof intelSponzaLocalReference>>;
type Component = LocalReference['rows'][number]['contributions']['total'];
export interface IntelSponzaReferenceCase {
  label: string;
  receiver: IntelSponzaLocalReceiver;
  /** Provenance only; it does not affect query rays or the estimator. */
  provenance?: { pixel: { x: number; y: number }; anchorIndex?: number; variant: string };
}
export interface IntelSponzaVisibilityLink {
  label: string;
  /** Already biased by the caller, if desired; no extra offset is applied. */
  origin: Vec3;
  destination: Vec3;
}
export interface IntelSponzaAnchorReferenceInput {
  assets: IntelSponzaAssetOptions & { sceneBudget?: boolean };
  cases: readonly IntelSponzaReferenceCase[];
  comparisons?: readonly { label: string; left: string; right: string }[];
  visibilityLinks?: readonly IntelSponzaVisibilityLink[];
  lighting: { sky: Vec3; sunColor: Vec3; sunDirection: Vec3; maxDistance: number };
  batches?: number;
  seed?: number;
  maxWarmupFrames?: number;
  /** Taken from the exact capture whose receiver/anchor inputs are supplied. */
  expectedTextures: readonly { id: string; currentMip: number }[];
  expectedImageBytes: number;
  expectedQueryGeometry: NonNullable<ReturnType<Renderer['diagnostics']>['worldQueries']>['geometryRepresentations'];
}

const MiB = 1024 ** 2, samples = 4096, maxBatches = 8, maxCases = 16;
const maxLinks = 256, linkTMin = 0.0001, linkEndpointMargin = 0.0001, maxLinkDistance = 1000;
const componentNames = ['sky', 'sun', 'emissive', 'total'] as const;
const textureSignature = (textures: readonly { id: string; currentMip: number }[]) =>
  JSON.stringify(textures.map(({ id, currentMip }) => [id, currentMip]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))));

function mergeScalar(values: readonly { mean: number; standardError: number }[]) {
  let count = 0, mean = 0, m2 = 0;
  for (const value of values) {
    const next = count + samples, delta = value.mean - mean;
    m2 += value.standardError ** 2 * samples * (samples - 1) + delta ** 2 * count * samples / next;
    mean += delta * samples / next;
    count = next;
  }
  return { mean, standardError: Math.sqrt(Math.max(0, m2) / (count - 1) / count) };
}
function mergeComponents(values: readonly Component[]) {
  const channels = [0, 1, 2].map(channel => mergeScalar(values.map(value => ({
    mean: value.mean[channel], standardError: value.standardError[channel],
  }))));
  return { mean: channels.map(channel => channel.mean), standardError: channels.map(channel => channel.standardError),
    luminance: mergeScalar(values.map(value => value.luminance)) };
}
function planVisibilityLinks(links: readonly IntelSponzaVisibilityLink[] | undefined) {
  if (links === undefined) return [];
  if (!Array.isArray(links) || links.length > maxLinks) throw Error('SPONZA_VISIBILITY_LINK_BUDGET');
  const labels = new Set<string>();
  const planned = links.map(link => {
    if (!link || typeof link.label !== 'string' || !link.label.length || link.label.length > 128 || labels.has(link.label)
      || !Array.isArray(link.origin) || link.origin.length !== 3 || !link.origin.every(Number.isFinite)
      || !Array.isArray(link.destination) || link.destination.length !== 3 || !link.destination.every(Number.isFinite)) {
      throw Error('SPONZA_VISIBILITY_LINK_INPUT');
    }
    labels.add(link.label);
    const origin: [number, number, number] = [link.origin[0], link.origin[1], link.origin[2]];
    const destination: [number, number, number] = [link.destination[0], link.destination[1], link.destination[2]];
    const delta = destination.map((value, axis) => value - origin[axis]);
    const distance = Math.hypot(...delta);
    if (!(distance > linkTMin + linkEndpointMargin) || distance > maxLinkDistance) throw Error('SPONZA_VISIBILITY_LINK_DISTANCE');
    const direction = delta.map(value => value / distance) as [number, number, number];
    const ray: WorldRay = { origin, direction, tMin: linkTMin, tMax: distance - linkEndpointMargin };
    return { label: link.label, origin, destination, distance, direction, tMin: linkTMin, tMax: ray.tMax!, ray };
  });
  // Reuse the actual wire validator before creating a device or issuing any
  // earlier reference rays: finite doubles alone do not guarantee valid f32.
  packWorldRays(planned.map(link => link.ray), maxLinks);
  return planned;
}

/** One bounded query at the same rendered snapshot as the reference. Material
 * sidedness/alpha semantics remain those of the shared query scene; a reverse
 * link is a separate caller-supplied control, not an implied symmetry claim. */
export async function intelSponzaVisibilityLinks(renderer: Renderer, links: readonly IntelSponzaVisibilityLink[],
  snapshot: { frame: number; materialRevision: number }) {
  const planned = planVisibilityLinks(links);
  if (![snapshot.frame, snapshot.materialRevision].every(value => Number.isSafeInteger(value) && value >= 0 && value <= 0xffff_ffff)) {
    throw Error('SPONZA_VISIBILITY_LINK_SNAPSHOT');
  }
  const result = planned.length ? await renderer.traceRays(planned.map(link => link.ray), { textureLod: 0 }) : null;
  if (result && (result.frame !== snapshot.frame || result.materialRevision !== snapshot.materialRevision)) {
    throw Error('SPONZA_VISIBILITY_LINK_SNAPSHOT_CHANGED');
  }
  if (result && result.hits.length !== planned.length) throw Error('SPONZA_VISIBILITY_LINK_HIT_COUNT');
  const rows = planned.map(({ ray: _ray, ...link }, index) => {
    const hit = result!.hits[index];
    if (hit && (!Number.isFinite(hit.distance) || hit.distance < Math.fround(link.tMin) || hit.distance > Math.fround(link.tMax))) {
      throw Error('SPONZA_VISIBILITY_LINK_HIT_INTERVAL');
    }
    return { ...link, blocked: hit !== null, hit };
  });
  return { snapshot, rows, budget: { links: planned.length, maxLinks, maxDistance: maxLinkDistance,
    queryCalls: Number(planned.length > 0), maxRays: planned.length },
    limitations: ['Shared query geometry and material sidedness; blocked is not an independent source-geometry truth.',
      'A reverse link may differ for single-sided materials. No entity is ignored and no bias is added by this helper.',
      'The first and last 0.1 mm of the segment are excluded; endpoint coincidence is not measured.'] };
}
function validate(input: IntelSponzaAnchorReferenceInput) {
  const batches = input.batches ?? maxBatches, seed = input.seed ?? 0x6d2b79f5;
  if (!Number.isSafeInteger(batches) || batches < 1 || batches > maxBatches
    || !Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) throw Error('SPONZA_ANCHOR_REFERENCE_BUDGET');
  if (!Array.isArray(input.cases) || input.cases.length < 1 || input.cases.length > maxCases
    || input.cases.some(row => !row || typeof row.label !== 'string' || !row.label.length || row.label.length > 128)
    || new Set(input.cases.map(row => row.label)).size !== input.cases.length) throw Error('SPONZA_ANCHOR_REFERENCE_CASES');
  const labels = new Set(input.cases.map(row => row.label));
  if (input.comparisons && (!Array.isArray(input.comparisons) || input.comparisons.length > 32
    || input.comparisons.some(pair => !labels.has(pair.left) || !labels.has(pair.right)))) throw Error('SPONZA_ANCHOR_REFERENCE_PAIRS');
  const links = planVisibilityLinks(input.visibilityLinks);
  return { batches, seed, links };
}

/** Runs on an already settled renderer without advancing its rendered snapshot.
 * Each chunk retains the existing reference's 4096-ray query budget. Seeds are
 * independent of production slot/frame seeds; all cases share each chunk seed. */
export async function intelSponzaAnchorReference(renderer: Renderer, input: IntelSponzaAnchorReferenceInput,
  progress: (phase: string, data?: unknown) => void = () => {}) {
  const config = validate(input), batches: LocalReference[] = [], cumulative = [];
  const labels = input.cases.map(row => row.label);
  const summarize = () => input.cases.map((item, index) => {
    const rows = batches.map(batch => batch.rows[index]);
    const contributions = Object.fromEntries(componentNames.map(name => [name, mergeComponents(rows.map(row => row.contributions[name]))]));
    return { label: item.label, receiver: item.receiver, provenance: item.provenance,
      samples: rows.length * samples, contributions,
      geometricHemisphereClipped: rows.reduce((sum, row) => sum + row.geometricHemisphereClipped, 0),
      primary: Object.fromEntries(['traced', 'misses', 'hits', 'nearHits', 'nearSameReceiver'].map(key =>
        [key, rows.reduce((sum, row) => sum + row.primary[key as keyof Omit<typeof row.primary, 'nearestDistance'>], 0)])),
      shadows: Object.fromEntries(['traced', 'occluded', 'visible', 'nearHits', 'nearSameSurface', 'nearSameReceiver'].map(key =>
        [key, rows.reduce((sum, row) => sum + row.shadows[key as keyof Omit<typeof row.shadows, 'nearestDistance'>], 0)])),
      samplesAboveHdrLimit: rows.reduce((sum, row) => sum + row.samplesAboveHdrLimit, 0) };
  });
  for (let batch = 0; batch < config.batches; batch++) {
    const seed = (config.seed + Math.imul(batch, 0x9e3779b9)) >>> 0;
    const result = await intelSponzaLocalReference(renderer, input.cases.map(row => row.receiver),
      input.lighting.sky, input.lighting.sunColor, input.lighting.sunDirection, input.lighting.maxDistance, { seed });
    if (batches.length && JSON.stringify(result.snapshot) !== JSON.stringify(batches[0].snapshot)) {
      throw Error('SPONZA_ANCHOR_REFERENCE_SNAPSHOT_CHANGED');
    }
    batches.push(result);
    const checkpoint = { batch: batch + 1, seed, samplesPerCase: (batch + 1) * samples, rows: summarize() };
    cumulative.push(checkpoint);
    progress('reference-batch', checkpoint);
  }
  const comparisons = (input.comparisons ?? []).map(pair => {
    const left = labels.indexOf(pair.left), right = labels.indexOf(pair.right);
    const differences = batches.map(batch => batch.rows[left].contributions.total.luminance.mean
      - batch.rows[right].contributions.total.luminance.mean);
    const mean = differences.reduce((sum, value) => sum + value, 0) / differences.length;
    const standardError = differences.length > 1
      ? Math.sqrt(differences.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (differences.length - 1) / differences.length) : null;
    return { ...pair, meanLuminanceDifference: mean, pairedBatchStandardError: standardError, batchDifferences: differences };
  });
  if (config.links.length && !batches[0].snapshot) throw Error('SPONZA_VISIBILITY_LINK_NO_REFERENCE_SNAPSHOT');
  const visibilityLinks = config.links.length ? await intelSponzaVisibilityLinks(renderer, config.links, batches[0].snapshot!) : null;
  if (visibilityLinks) progress('visibility-links', { links: visibilityLinks.rows.length,
    blocked: visibilityLinks.rows.filter(link => link.blocked).length, snapshot: visibilityLinks.snapshot });
  return { method: 'Independent cosine Monte Carlo at explicit receiver/anchor coordinates; shared query geometry/materials, no GI cache or spatial reconstruction.',
    cases: input.cases, snapshot: batches[0].snapshot, rows: summarize(), comparisons, cumulative, batches, visibilityLinks,
    budget: { batches: config.batches, cases: input.cases.length, samplesPerCase: config.batches * samples,
      maxRaysPerTrace: samples, maxRays: config.batches * input.cases.length * samples * 2 + config.links.length,
      maxTraceCalls: config.batches * input.cases.length * 2 + Number(config.links.length > 0),
      visibilityLinkRays: config.links.length, cpuTemporaryRecordsPerPoint: samples },
    limitations: ['Batch seeds are deterministic pseudorandom replicates, not formal independent sampling guarantees.',
      'Paired comparison errors use variation of common-seed batch differences; pooled channel errors do not measure that covariance.',
      'The shared query geometry/material evaluator cannot establish independent raster or source-geometry truth.',
      'Crossed position/normal cases are diagnostic integrals, not physically present surface receivers.'] };
}

/** Independent scene entry point: no GI warmup or production GI rendering.
 * Scene asset identity, resident query texture mips and query geometry must match
 * the supplied capture before any reference ray is issued. */
export async function captureIntelSponzaAnchorReference(input: IntelSponzaAnchorReferenceInput) {
  validate(input);
  const maxWarmup = input.maxWarmupFrames ?? 600;
  if (!Number.isSafeInteger(maxWarmup) || maxWarmup < 12 || maxWarmup > 1200
    || !input.assets.queryRoot || !Array.isArray(input.expectedTextures) || !input.expectedTextures.length
    || !Array.isArray(input.expectedQueryGeometry) || !input.expectedQueryGeometry.length
    || !Number.isSafeInteger(input.expectedImageBytes) || input.expectedImageBytes < 0) throw Error('SPONZA_ANCHOR_REFERENCE_SNAPSHOT_INPUT');
  const started = performance.now();
  const progress = (phase: string, data: unknown = {}) => console.log('SPONZA_REFERENCE_PROGRESS ' + JSON.stringify({ phase, elapsedMs: performance.now() - started, data }));
  const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
  canvas.style.cssText = 'width:1280px;height:720px;display:block;flex:none';
  let renderer: Renderer | undefined;
  try {
    // Renderer.resize derives its targets from the attached canvas's client
    // size and DPR, rather than the canvas width/height attributes alone.
    document.body.append(canvas);
    if (devicePixelRatio !== 1 || canvas.clientWidth !== 1280 || canvas.clientHeight !== 720) {
      throw Error('SPONZA_ANCHOR_REFERENCE_CSS_VIEWPORT: expected 1280x720 at DPR 1');
    }
    const assets = await loadIntelSponzaAssets(input.assets, progress);
    const { batches, accounting, sources } = assets;
    const eye: [number, number, number] = [9, 1.8, 0], target: [number, number, number] = [-6, 3.5, 0];
    const viewProjection = multiply(perspective(Math.PI / 3.2, 1280 / 720, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
    if (input.assets.sceneBudget && !input.assets.sourceRaster) {
      const plan = planGeometryStreamBudgets(batches.map(batch => ({ directory: batch.geometryStream.directory,
        demands: new GeometryStreamCut(batch.geometryStream.directory).demands(batch.instances.map(instance => ({
          eye, near: 0.1, projectionScale: 720 / (2 * Math.tan(Math.PI / 6.4)), maxErrorPixels: 2,
          model: instance.model, viewProjection }))),
      })), { maxResidentBytes: accounting.reduce((sum, row) => sum + row.gpuBytes, 0) });
      for (const entry of plan.streams) {
        const stream = batches[entry.index].geometryStream;
        batches[entry.index].geometryStream = { ...stream, gpuPool: { ...stream.gpuPool, maxResidentBytes: entry.maxResidentBytes } };
      }
    }
    const scene = input.assets.sourceRaster ? batches.map(({ geometryStream, ...batch }): SceneBatch =>
      ({ ...batch, mesh: geometryStream.queryProxy.mesh })) : batches;
    renderer = await Renderer.create(canvas, scene, { maxBatches: 512, textureBudgetBytes: 512 * MiB,
      worldQueries: { maxRays: samples, maxBatches: 512, maxTrianglesPerMesh: 262144, maxBytes: 512 * MiB, maxTotalWorkUnits: 67108864 },
      textureStreaming: { sources, maxCpuBytes: 256 * MiB, maxInFlightBytes: 128 * MiB, maxUploadBytesPerFrame: 2 * MiB } });
    renderer.setSun({ direction: [...input.lighting.sunDirection], color: [...input.lighting.sunColor] });
    if (!input.assets.sourceRaster) renderer.setGeometryLod({ maxErrorPixels: 2, frustumCulling: true, debug: 'material' });
    let stableFrames = 0, renderedFrames = 0, priorSignature = '';
    for (; renderedFrames < maxWarmup && stableFrames < 12; renderedFrames++) {
      renderer.render(eye, target);
      if (renderer.stats.width !== 1280 || renderer.stats.height !== 720 || canvas.width !== 1280 || canvas.height !== 720) {
        throw Error(`SPONZA_ANCHOR_REFERENCE_RENDER_VIEWPORT: ${renderer.stats.width}x${renderer.stats.height}`);
      }
      await renderer.probeHdr([{ x: 640, y: 360 }]);
      await new Promise(requestAnimationFrame);
      if (renderer.errors.length) throw Error('SPONZA_ANCHOR_REFERENCE_RENDER: ' + renderer.errors.join('; '));
      const d = renderer.diagnostics(), textures = d.textureStreaming?.sources ?? [];
      if (textures.some(source => source.failure) || d.geometryStreams.some(stream => stream.cache.failedPages.length)) {
        throw Error('SPONZA_ANCHOR_REFERENCE_RESIDENCY_FAILED');
      }
      const signature = JSON.stringify({ textures: textureSignature(textures), imageBytes: d.materials.imageBytes,
        geometry: d.geometryStreams.map(stream => stream.pool.residentPages) });
      const settled = !d.textureStreaming?.active && textures.every(source => source.currentMip === source.plannedMip)
        && d.geometryStreams.every(stream => !stream.pool.pendingPages.length && !stream.cache.activeRequests
          && !stream.cache.queuedRequests && stream.allocation.pages.every(page => stream.pool.residentPages.includes(page)));
      stableFrames = settled && signature === priorSignature ? stableFrames + 1 : 0;
      priorSignature = signature;
      if (renderedFrames % 30 === 0) progress('reference-residency', { renderedFrames, stableFrames });
    }
    if (stableFrames < 12) throw Error('SPONZA_ANCHOR_REFERENCE_RESIDENCY_NOT_SETTLED');
    const before = renderer.diagnostics();
    if (textureSignature(before.textureStreaming?.sources ?? []) !== textureSignature(input.expectedTextures)
      || before.materials.imageBytes !== input.expectedImageBytes) throw Error('SPONZA_ANCHOR_REFERENCE_TEXTURE_MISMATCH');
    if (JSON.stringify(before.worldQueries?.geometryRepresentations) !== JSON.stringify(input.expectedQueryGeometry)) {
      throw Error('SPONZA_ANCHOR_REFERENCE_GEOMETRY_MISMATCH');
    }
    const reference = await intelSponzaAnchorReference(renderer, input, progress);
    const after = renderer.diagnostics();
    if (after.materialRevision !== before.materialRevision
      || textureSignature(after.textureStreaming?.sources ?? []) !== textureSignature(input.expectedTextures)) {
      throw Error('SPONZA_ANCHOR_REFERENCE_SCENE_CHANGED');
    }
    const result = { info: renderer.info, reference, errors: [...renderer.errors], renderedFrames, stableFrames,
      viewport: { width: renderer.stats.width, height: renderer.stats.height, devicePixelRatio },
      queryGeometry: before.worldQueries?.geometryRepresentations, textureMips: input.expectedTextures,
      imageBytes: before.materials.imageBytes, omitted: assets.omitted, afterDispose: 0 };
    renderer.dispose(); result.afterDispose = renderer.stats.ownedGpuBytes;
    return result;
  } finally { renderer?.dispose(); canvas.remove(); }
}
