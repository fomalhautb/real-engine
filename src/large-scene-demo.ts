/**
 * Large-scene walkthrough page (large-scene.html?set=/artifacts/.../asset-set.json).
 * Loads an offline-built asset set and renders its opaque paged geometry,
 * full-detail MASK primitives and streamed textures under one explicit geometry
 * budget (initial pools from the first pose's demand, then the renderer's
 * capacity coordinator), with the camera path's field of view and a far plane
 * that covers the scene bounds. Lighting (scheduled diffuse GI, temporal
 * reflections, height fog with optional GI air lighting) uses explicit budgets
 * that are validated before creation and on every resize: settings the engine
 * would coarsen are rejected with an error, never silently degraded.
 *
 * Query parameters: frames=N (walk N rendered frames, then stop and report through
 * window.__LARGE_SCENE__), exposure=EV, sun=x,y,z, walk=1, lod=px (0..16),
 * geometryMiB=N (live paged-geometry budget, default 1024), lighting=on|off,
 * gi=0|1 and reflections=0|1 (configure each pass; fog is configured with
 * lighting), fog=0|1 (initially enabled), air=1 (GI air lighting), probes=N
 * (8..512), reflectionSpacing=auto|px, fogPixel=auto|px. Frame counts are
 * frames the renderer rendered (stream backpressure skips do not count).
 * Frame intervals and GPU pass timings are diagnostics from a shared GPU, not
 * benchmark results.
 */
import { cameraPathKeyframes, parseCameraPath, sampleCameraPath, type CameraPath, type CameraPose, type Vec3 } from './camera-path';
import { resolveCameraProjection, type CameraProjection } from './camera-projection';
import { loadAssetSetRoots, loadGltfPackageScene, type GltfPackageScene } from './gltf-package-scene';
import { planLargeSceneGeometry, planLargeSceneLighting, type LargeSceneGeometryReport, type LargeSceneLightingPlan,
  type LargeSceneLightingRequest } from './large-scene-plan';
import { Renderer, type FrameStats } from './renderer';

type DebugView = 'material' | 'clusters' | 'levels';
type Status = 'idle' | 'loading' | 'ready' | 'done' | 'error' | 'disposed';
type LightingState = { gi: boolean; reflections: boolean; fog: boolean; airLighting: boolean };
type GpuPass = keyof NonNullable<FrameStats['gpuMs']>;
const MiB = 1024 ** 2;
const MAX_TIMING_SAMPLES = 512;
/** Callbacks one drain may wait for in-flight frames to finish: about a minute at 60 Hz. */
const MAX_DRAIN_CALLBACKS = 3600;
const params = new URLSearchParams(location.search);
const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = element<HTMLCanvasElement>('viewport');
const controls = new AbortController();
let renderer: Renderer | undefined, scene: GltfPackageScene | undefined, path: CameraPath | undefined;
let projection: CameraProjection | undefined, geometryReport: LargeSceneGeometryReport | undefined;
let lightingPlan: LargeSceneLightingPlan | undefined, lightingRequest: LargeSceneLightingRequest | undefined;
let validatedSize = '', configured: LightingState = { gi: false, reflections: false, fog: false, airLighting: false };
let enabled: LightingState = { ...configured };
let status: Status = 'idle', failure: string | null = null, disposed = false, raf = 0, walking = false;
let cameraTime = 0, lastTime = 0, lastUi = 0, interval = 0, debug: DebugView = 'material', maxErrorPixels = 2;
// Frame counts are frames the renderer actually rendered (renderer.stats.frames): a callback that
// stream backpressure skips renders nothing and does not count. runEnd is the renderer frame that
// ends the current run (0: run until stopped); runCallbacks bounds the callbacks one run may take.
let runEnd = 0, runStartFrame = 0, runCallbacks = 0, callbacks = 0, exposureEV = 3, sunDirection: Vec3 = [0.35, 0.8, 0.4];
let settleUntil = 0, settleStable = 0, lastSettle: { settled: boolean; frames: number } | null = null;
// Every 15 rendered frames the page checks for staged pages that in-flight leases block (see geometryState) and
// then drains once; drains counts completed drains.
let drainPending = false, drainCallbacks = 0, drains = 0;
let timingSamples: { frame: number; ms: NonNullable<FrameStats['gpuMs']> }[] = [], lastSampleFrame: number | null = null;

function setStatus(next: Status, text: string): void {
  status = next;
  element('status').textContent = text;
}

function fail(error: unknown): void {
  cancelAnimationFrame(raf);
  failure = error instanceof Error ? error.message : String(error);
  setStatus('error', 'ERROR');
  const panel = element('error');
  panel.hidden = false; panel.textContent = failure;
}

function vec3Param(name: string, fallback: Vec3): Vec3 {
  const text = params.get(name);
  if (text === null) return fallback;
  const values = text.split(',').map(Number);
  if (values.length !== 3 || !values.every(Number.isFinite)) throw new Error('LARGE_SCENE_PARAM: ' + name);
  return values as Vec3;
}

function numberParam(name: string, fallback: number, minimum: number, maximum: number, integer: boolean): number {
  const text = params.get(name);
  const value = text === null ? fallback : Number(text);
  if (!Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isSafeInteger(value))) {
    throw new Error(`LARGE_SCENE_PARAM: ${name} must be ${integer ? 'an integer ' : ''}in [${minimum}, ${maximum}]`);
  }
  return value;
}

function autoParam(name: string, minimum: number, maximum: number): number | 'auto' {
  return params.get(name) === null || params.get(name) === 'auto' ? 'auto' : numberParam(name, minimum, minimum, maximum, true);
}

function flagParam(name: string, fallback: boolean): boolean {
  const text = params.get(name);
  if (text === null) return fallback;
  if (!['0', '1', 'on', 'off'].includes(text)) throw new Error('LARGE_SCENE_PARAM: ' + name);
  return text === '1' || text === 'on';
}

/** Same drawing-buffer size the renderer picks on its next frame. */
function drawingSize(): { width: number; height: number } {
  const ratio = Math.min(devicePixelRatio, 1.5);
  return { width: Math.max(1, Math.round(canvas.clientWidth * ratio)), height: Math.max(1, Math.round(canvas.clientHeight * ratio)) };
}

/** Eight-sample orbit around the bounds, used when the asset set has no camera path. */
function orbitPath(bounds: GltfPackageScene['bounds']): CameraPath {
  const center = [0, 1, 2].map(c => (bounds.min[c] + bounds.max[c]) / 2) as Vec3;
  const radius = 0.35 * Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[2] - bounds.min[2]);
  const samples = Array.from({ length: 9 }, (_, i) => {
    const angle = (i / 8) * 2 * Math.PI, position: Vec3 = [center[0] + radius * Math.cos(angle), center[1], center[2] + radius * Math.sin(angle)];
    return { time: i * 5, position, forward: [center[0] - position[0], 0, center[2] - position[2]], up: [0, 1, 0], key: i < 8 };
  });
  return parseCameraPath({ format: 'real-engine-camera-path', name: 'bounds-orbit', samples });
}

function pose(): CameraPose { return sampleCameraPath(path!, cameraTime); }

/** Debug views show their palette unexposed (EV 0); the material view uses the scene exposure. */
function applyLod(): void {
  renderer?.setGeometryLod({ maxErrorPixels, frustumCulling: true, debug });
  renderer?.setExposure(debug === 'material' ? exposureEV : 0);
}

function lightingRequestFor(plan: LargeSceneLightingPlan | undefined, base: LargeSceneLightingRequest): LargeSceneLightingRequest {
  // After creation the chosen spacing/froxel size is explicit: a resize may not coarsen it.
  return plan ? { ...base, reflectionSpacing: plan.report.reflections?.spacing ?? 'auto', fogPixelSize: plan.report.fog?.pixelSize ?? 'auto' } : base;
}

/** Re-validates the lighting budgets when the drawing buffer changes size (throws instead of degrading). */
function checkViewport(): void {
  const { width, height } = drawingSize(), key = `${width}x${height}`;
  if (key === validatedSize || !lightingRequest || !scene || !projection) return;
  planLargeSceneLighting(scene.bounds, width, height, projection.far, scene.batches.length, lightingRequestFor(lightingPlan, lightingRequest));
  validatedSize = key;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b), middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Per-pass GPU timestamp summary of the current run (diagnostic: shared GPU, sampled every 30 frames). */
function gpuTimings() {
  if (!timingSamples.length) return { samples: 0, passes: null };
  const passes = Object.keys(timingSamples[0].ms) as GpuPass[];
  const quantile = (values: number[], q: number) => { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]; };
  return { samples: timingSamples.length, frames: [timingSamples[0].frame, timingSamples.at(-1)!.frame],
    passes: Object.fromEntries(passes.map(pass => {
      const values = timingSamples.map(sample => sample.ms[pass]);
      // Samples bit-identical to the previous non-zero sample are almost surely stale query values (for example
      // the cached world BVH refit after its first frame), not a repeated measurement.
      const repeated = values.filter((value, index) => index > 0 && value > 0 && value === values[index - 1]).length;
      return [pass, { median: median(values), p90: quantile(values, 0.9), max: Math.max(...values), repeated }];
    })),
    totalMedian: median(timingSamples.map(sample => passes.reduce<number>((sum, pass) => sum + sample.ms[pass], 0))) };
}

function geometryState() {
  if (!renderer) return null;
  const d = renderer.diagnostics(), budget = d.geometryRuntimeBudget;
  let missingAllocatedPages = 0, budgetDeferredGroups = 0, dependencyDeferredGroups = 0, pendingPages = 0, failedPages = 0, requests = 0;
  let leaseBlockedStreams = 0;
  for (const stream of d.geometryStreams) {
    const pool = stream.pool, resident = new Set(pool.residentPages);
    for (const page of stream.allocation.pages) if (!resident.has(page)) missingAllocatedPages++;
    budgetDeferredGroups += stream.allocation.budgetDeferredGroups.length;
    dependencyDeferredGroups += stream.allocation.dependencyDeferredGroups.length;
    pendingPages += pool.pendingPages.length; failedPages += stream.cache.failedPages.length;
    requests += stream.cache.activeRequests + stream.cache.queuedRequests;
    // The pool evicts only resident pages that are neither pinned nor leased, and every submitted frame leases
    // all resident pages. A staged page that needs a slot in a full pool whose only unpinned pages are leased
    // can therefore upload only while no frame is in flight.
    if (pool.pendingPages.length && pool.activePage === null && pool.framesInFlight && pool.residentPages.length >= pool.slotCount) {
      const pinned = new Set(pool.pinnedPages), leased = new Set(pool.leasedPages);
      const unpinned = pool.residentPages.filter(page => !pinned.has(page));
      if (unpinned.length && unpinned.every(page => leased.has(page))) leaseBlockedStreams++;
    }
  }
  const textures = d.textureStreaming;
  const texturesSettled = !textures || (!textures.active && textures.sources.every(source => source.currentMip === source.plannedMip));
  const capacityIdle = !budget.enabled || (budget.phase === 'idle' && !budget.backlog);
  return { missingAllocatedPages, budgetDeferredGroups, dependencyDeferredGroups, pendingPages, failedPages, requests, leaseBlockedStreams, texturesSettled, capacityIdle,
    settled: !missingAllocatedPages && !pendingPages && !requests && texturesSettled && capacityIdle,
    capacity: budget.enabled ? { phase: budget.phase, error: budget.error, liveBytes: budget.liveBytes, peakBytes: budget.peakBytes,
      migrations: budget.migrations, deferred: budget.deferred, backlog: budget.backlog, limits: budget.limits } : null };
}

function refreshUi(): void {
  if (!renderer || !scene) return;
  const stats = renderer.stats;
  element('frame').textContent = String(stats.frames);
  element('interval').textContent = interval ? `${interval.toFixed(1)} ms` : '—';
  element('memory').textContent = `${(stats.ownedGpuBytes / MiB).toFixed(1)} MiB`;
  element('time').textContent = `${cameraTime.toFixed(2)} / ${path!.duration.toFixed(1)} s`;
  const budget = renderer.diagnostics().geometryRuntimeBudget;
  if (budget.enabled) element('geometry').textContent = `${(budget.liveBytes / MiB).toFixed(0)} / ${(budget.limits.maxResidentBytes / MiB).toFixed(0)} MiB · ${budget.phase}`;
  const ms = stats.gpuMs;
  element('gpu').textContent = ms ? `surface ${ms.surface.toFixed(2)} · shadow ${ms.shadow.toFixed(2)} · GI ${ms.diffuseGi.toFixed(2)} · refl ${ms.reflections.toFixed(2)} · fog ${ms.volumetricFog.toFixed(2)} ms`
    : 'timestamp-query unavailable';
}

const renderedFrames = () => renderer?.stats.frames ?? 0;

/** Most submitted frames still in flight over the streamed batches' page pools (CPU-side diagnostics, no readback). */
function geometryFramesInFlight(): number {
  let inFlight = 0;
  for (const stream of renderer!.diagnostics().geometryStreams) inFlight = Math.max(inFlight, stream.pool.framesInFlight);
  return inFlight;
}

/** Renders the current pose; false when stream backpressure skipped the frame. */
function renderOnce(): boolean {
  checkViewport();
  const camera = pose(), before = renderer!.stats.frames;
  renderer!.render(camera.eye, camera.target);
  if (renderer!.errors.length) throw new Error(renderer!.errors.join('\n'));
  const stats = renderer!.stats;
  if (stats.gpuMs && stats.gpuSampleFrame !== null && stats.gpuSampleFrame !== lastSampleFrame) {
    lastSampleFrame = stats.gpuSampleFrame;
    // Sample frames use the renderer's numbering, like runStartFrame. A configured pass that is disabled at
    // runtime still writes an empty timestamped pass, and Apple was observed to keep the previously written
    // values for it (identical to the nanosecond across samples), so report disabled lighting passes as 0.
    const ms = { ...stats.gpuMs };
    if (!enabled.gi) { ms.diffuseGi = 0; ms.finalGather = 0; ms.supplementalReceivers = 0; }
    if (!enabled.reflections) ms.reflections = 0;
    if (stats.gpuSampleFrame > runStartFrame) timingSamples.push({ frame: stats.gpuSampleFrame, ms });
    if (timingSamples.length > MAX_TIMING_SAMPLES) timingSamples.shift();
  }
  return stats.frames > before;
}

function frame(now: number): void {
  if (disposed || !renderer) return;
  try {
    const dt = Math.min(0.1, Math.max(0, (now - lastTime) / 1000));
    interval = interval ? interval * 0.9 + (now - lastTime) * 0.1 : now - lastTime;
    lastTime = now;
    // Interactive walking follows wall-clock time; a fixed-length run advances 1/60 s per rendered frame.
    if (walking && !runEnd) cameraTime = (cameraTime + dt) % path!.duration;
    // A drain holds rendering until the GPU has finished every submitted frame, so that the next frame's
    // uploads may evict pages that in-flight leases kept resident. Bounded, and it fails on renderer errors.
    let held = false;
    if (drainPending) {
      if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
      held = geometryFramesInFlight() > 0;
      if (held && ++drainCallbacks > MAX_DRAIN_CALLBACKS) throw new Error(`LARGE_SCENE_DRAIN_STALLED: frames still in flight after ${MAX_DRAIN_CALLBACKS} callbacks`);
      if (!held) { drainPending = false; drainCallbacks = 0; drains++; }
    }
    const rendered = !held && renderOnce(), frames = renderedFrames();
    callbacks++;
    if (rendered && walking && runEnd) cameraTime = (cameraTime + 1 / 60) % path!.duration;
    if (now - lastUi > 250) { refreshUi(); lastUi = now; }
    if (rendered && (frames - runStartFrame) % 15 === 0) {
      const state = geometryState()!;
      if (state.leaseBlockedStreams) drainPending = true;
      if (settleUntil) {
        settleStable = state.settled ? settleStable + 1 : 0;
        if (settleStable >= 2 || frames >= settleUntil) {
          lastSettle = { settled: settleStable >= 2, frames: frames - runStartFrame }; settleUntil = 0; runEnd = frames;
        }
      }
    }
    if (runEnd && frames >= runEnd) { refreshUi(); setStatus('done', `DONE · ${frames} FRAMES`); return; }
    if (runEnd && --runCallbacks <= 0) {
      throw new Error(`LARGE_SCENE_STALLED: ${frames - runStartFrame} of ${runEnd - runStartFrame} frames rendered; stream backpressure skipped the rest`);
    }
    raf = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

function start(frames: number): void {
  if (!renderer || disposed || status === 'error') throw new Error('LARGE_SCENE_NOT_READY');
  cancelAnimationFrame(raf);
  runStartFrame = renderedFrames(); timingSamples = [];
  runEnd = runStartFrame + frames;
  // Bounded: backpressure may skip many callbacks per rendered frame, but not without limit.
  runCallbacks = 64 * frames + 3600;
  setStatus('ready', `RUNNING ${frames} FRAMES`);
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}

/** Renders `frames` more frames (walking or not), then stops with status 'done' again. */
function run(frames: number, walk: boolean): void {
  if (!Number.isSafeInteger(frames) || frames < 1 || frames > 100_000) throw new Error('LARGE_SCENE_FRAMES');
  setWalk(walk); settleUntil = 0; start(frames);
}

/** Holds the camera and renders until geometry, textures and capacity migrations settle, at most maxFrames. */
function settle(maxFrames: number): void {
  if (!Number.isSafeInteger(maxFrames) || maxFrames < 30 || maxFrames > 20_000) throw new Error('LARGE_SCENE_FRAMES');
  setWalk(false); lastSettle = null; settleStable = 0; start(maxFrames); settleUntil = renderedFrames() + maxFrames;
}

function setWalk(value: boolean): void {
  walking = value;
  element('walk').setAttribute('aria-pressed', String(value));
}

function showKeyframe(index: number): void {
  if (!path || !Number.isSafeInteger(index) || index < 0 || index >= path.samples.length) throw new Error('LARGE_SCENE_KEYFRAME');
  setWalk(false);
  cameraTime = path.samples[index].time - path.samples[0].time;
}

function setDebug(value: DebugView): void {
  if (!['material', 'clusters', 'levels'].includes(value)) throw new Error('LARGE_SCENE_DEBUG_VIEW');
  debug = value; element<HTMLSelectElement>('debug').value = value; applyLod();
}

/** Runtime toggles of the configured lighting passes; unconfigured passes stay off. */
function setLighting(next: Partial<LightingState>): void {
  if (!renderer) throw new Error('LARGE_SCENE_NOT_READY');
  for (const key of Object.keys(next) as (keyof LightingState)[]) {
    if (!(key in enabled) || typeof next[key] !== 'boolean') throw new Error('LARGE_SCENE_LIGHTING_TOGGLE: ' + key);
    if (next[key] && !configured[key]) throw new Error('LARGE_SCENE_LIGHTING_NOT_CONFIGURED: ' + key);
  }
  enabled = { ...enabled, ...next };
  if (configured.gi) renderer.setDiffuseGiEnabled(enabled.gi);
  if (configured.reflections) renderer.setReflectionsEnabled(enabled.reflections);
  if (configured.fog) renderer.setVolumetricFogEnabled(enabled.fog);
  if (configured.airLighting) renderer.setVolumetricFogAirLighting({ gi: enabled.airLighting });
  for (const key of Object.keys(enabled) as (keyof LightingState)[]) element<HTMLInputElement>('light-' + key).checked = enabled[key];
}

function setSun(direction: Vec3): void {
  if (!direction.every(Number.isFinite) || Math.hypot(...direction) < 1e-6) throw new Error('LARGE_SCENE_SUN');
  sunDirection = [...direction];
  renderer?.setSun({ direction: sunDirection, color: [4, 3.8, 3.5] });
  const length = Math.hypot(...direction);
  element<HTMLInputElement>('sun-elevation').value = String(Math.round(Math.asin(direction[1] / length) * 180 / Math.PI));
  element<HTMLInputElement>('sun-azimuth').value = String((Math.round(Math.atan2(direction[2], direction[0]) * 180 / Math.PI) + 360) % 360);
}

/**
 * Coverage audit of the last rendered camera through the renderer's world queries (the
 * camera-independent shadow/query representation, exact for fine-proxy asset sets). Returns
 * pixel centres on a `step` grid that are interior to opaque streamed geometry (centre and the
 * four rays `offset` px away hit opaque geometry at depths within 10 %), and sky samples where all
 * five rays miss. A rendered background colour at an interior sample is a hole. Diagnostic,
 * explicit one-off readback; never part of the frame path.
 */
async function auditCoverage(step = 8, offset = 3) {
  if (!renderer || !scene || !projection) throw new Error('LARGE_SCENE_NOT_READY');
  if (!Number.isSafeInteger(step) || step < 4 || step > 64 || !Number.isSafeInteger(offset) || offset < 1 || offset >= step) throw new Error('LARGE_SCENE_AUDIT');
  const { width, height } = renderer.stats, camera = pose();
  const z = camera.eye.map((v, a) => v - camera.target[a]), zl = Math.hypot(...z), zn = z.map(v => v / zl);
  const xr = [zn[2], 0, -zn[0]], xl = Math.hypot(...xr), xn = xr.map(v => v / xl);
  const yn = [zn[1] * xn[2] - zn[2] * xn[1], zn[2] * xn[0] - zn[0] * xn[2], zn[0] * xn[1] - zn[1] * xn[0]];
  const t = Math.tan(projection.verticalFovRadians / 2), aspect = width / height;
  const ray = (px: number, py: number) => {
    const nx = ((px + 0.5) / width * 2 - 1) * t * aspect, ny = (1 - (py + 0.5) / height * 2) * t;
    const d = [0, 1, 2].map(a => xn[a] * nx + yn[a] * ny - zn[a]), l = Math.hypot(...d);
    return { origin: camera.eye, direction: d.map(v => v / l) as Vec3, tMin: projection!.near, tMax: projection!.far };
  };
  const opaque = new Set(scene.batches.filter(batch => batch.geometryStream).map(batch => batch.name));
  const centres: [number, number][] = [];
  for (let y = offset; y < height - offset; y += step) for (let x = offset; x < width - offset; x += step) centres.push([x, y]);
  const rays = centres.flatMap(([x, y]) => [ray(x, y), ray(x - offset, y), ray(x + offset, y), ray(x, y - offset), ray(x, y + offset)]);
  // The world-query work cap is shared by all rays of one query: shrink the chunk until no ray overflows.
  const hits: (number | null)[] = [];
  let chunk = 8192, overflowRetries = 0;
  for (let first = 0; first < rays.length;) {
    try {
      const result = await renderer.traceRays(rays.slice(first, first + chunk));
      for (const hit of result.hits) hits.push(hit && opaque.has(hit.batch) ? hit.distance : hit ? -1 : null);
      first += chunk;
    } catch (error) {
      if (!String(error).includes('WORLD_QUERY_WORK_BUDGET') || chunk <= 256) throw error;
      chunk >>= 1; overflowRetries++;
    }
  }
  const interior: [number, number][] = [], sky: [number, number][] = [];
  centres.forEach((centre, i) => {
    const group = hits.slice(i * 5, i * 5 + 5), d = group[0];
    if (group.every(value => value === null)) sky.push(centre);
    else if (d !== null && d > 0 && group.every(value => value !== null && value > 0 && Math.abs(value - d) <= 0.1 * d)) interior.push(centre);
  });
  return { width, height, step, offset, samples: centres.length, interior, sky, cameraTime, raysPerQuery: chunk, overflowRetries };
}

function dispose(): void {
  if (disposed) return;
  disposed = true;
  cancelAnimationFrame(raf);
  controls.abort();
  renderer?.dispose();
  setStatus('disposed', 'DISPOSED');
}

function snapshot() {
  return { status, failure, assetSet: params.get('set'), renderedFrames: renderedFrames(), callbacks, drains,
    backpressureFrames: renderer?.stats.streamBackpressureFrames ?? 0, cameraTime, walking, debug, maxErrorPixels,
    info: renderer?.info ?? null, stats: renderer ? { ...renderer.stats } : null, projection: projection ?? null,
    scene: scene ? { batches: scene.batches.length, streamedBatches: scene.batches.filter(batch => batch.geometryStream).length,
      textures: scene.sources.length, sourceInstances: scene.sourceInstances,
      pagedTriangles: scene.pagedTriangles, omittedTriangles: scene.omittedTriangles, omitted: scene.omitted.length,
      omittedByReason: scene.omitted.reduce<Record<string, number>>((counts, entry) => { counts[entry.reason] = (counts[entry.reason] ?? 0) + 1; return counts; }, {}),
      mask: scene.mask, queryProxies: scene.queryProxies,
      conversions: scene.conversions, unsupportedMaterialInputs: scene.unsupportedMaterialInputs.length, bounds: scene.bounds } : null,
    geometry: geometryReport ? { plan: geometryReport, live: geometryState() } : null,
    lighting: lightingPlan ? { configured, enabled, ...lightingPlan.report, passes: renderer?.diagnostics().passes ?? [] } : null,
    sun: sunDirection, gpuTimings: gpuTimings(), lastSettle,
    cameraPath: path ? { name: path.name, samples: path.samples.length, duration: path.duration,
      verticalFovRadians: path.verticalFovRadians, keyframes: cameraPathKeyframes(path) } : null };
}

function wireControls(): void {
  const signal = controls.signal;
  element('walk').addEventListener('click', () => setWalk(!walking), { signal });
  element<HTMLSelectElement>('debug').addEventListener('change', event => setDebug((event.target as HTMLSelectElement).value as DebugView), { signal });
  element<HTMLInputElement>('lod-error').addEventListener('input', event => {
    maxErrorPixels = Number((event.target as HTMLInputElement).value);
    element('lod-error-value').textContent = `${maxErrorPixels} px`;
    applyLod();
  }, { signal });
  for (const key of ['gi', 'reflections', 'fog', 'airLighting'] as const) {
    element<HTMLInputElement>('light-' + key).addEventListener('change', event => {
      try { setLighting({ [key]: (event.target as HTMLInputElement).checked }); } catch (error) { fail(error); }
    }, { signal });
  }
  const sunFromSliders = () => {
    const elevation = Number(element<HTMLInputElement>('sun-elevation').value) * Math.PI / 180;
    const azimuth = Number(element<HTMLInputElement>('sun-azimuth').value) * Math.PI / 180;
    setSun([Math.cos(elevation) * Math.cos(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)]);
  };
  element('sun-elevation').addEventListener('input', sunFromSliders, { signal });
  element('sun-azimuth').addEventListener('input', sunFromSliders, { signal });
  window.addEventListener('keydown', event => {
    if (event.key.toLowerCase() === 'h' && !event.altKey && !event.ctrlKey && !event.metaKey) document.body.classList.toggle('ui-hidden');
  }, { signal });
  window.addEventListener('pagehide', dispose, { once: true, signal });
}

async function startPage(): Promise<void> {
  wireControls();
  window.__LARGE_SCENE__ = { snapshot, showKeyframe, setWalk, setDebug, setLighting, setSun, run, settle, auditCoverage, dispose,
    setUiHidden: (hidden: boolean) => { document.body.classList.toggle('ui-hidden', hidden); }, get renderer() { return renderer; } };
  const set = params.get('set');
  if (!set) { setStatus('idle', 'NO ASSET SET · add ?set=/artifacts/<name>/asset-set.json'); return; }
  const autorun = params.get('frames') === null ? 0 : numberParam('frames', 1, 1, 100_000, true);
  exposureEV = numberParam('exposure', 3, -16, 16, false);
  maxErrorPixels = numberParam('lod', 2, 0, 16, false);
  const geometryBytes = numberParam('geometryMiB', 1024, 64, 8191, true) * MiB;
  const lightingOn = flagParam('lighting', true);
  lightingRequest = { gi: lightingOn && flagParam('gi', true), reflections: lightingOn && flagParam('reflections', true), fog: lightingOn,
    airLighting: lightingOn && flagParam('air', false),
    maxProbes: numberParam('probes', 512, 8, 512, true), reflectionSpacing: autoParam('reflectionSpacing', 1, 128), fogPixelSize: autoParam('fogPixel', 1, 128) };
  if (lightingRequest.airLighting && !lightingRequest.gi) throw new Error('LARGE_SCENE_PARAM: air needs gi (GI air lighting)');
  const initialFog = flagParam('fog', true);
  sunDirection = vec3Param('sun', sunDirection);
  setStatus('loading', 'LOADING ASSET SET');
  const roots = await loadAssetSetRoots(set, controls.signal);
  scene = await loadGltfPackageScene(roots, { name: set.split('/').slice(-2, -1)[0] || 'scene', anisotropy: 8, signal: controls.signal,
    progress: (phase, data) => setStatus('loading', `${phase.toUpperCase()} ${data ? JSON.stringify(data) : ''}`) });
  if (disposed) return;
  path = roots.cameraPathUrl
    ? parseCameraPath(await (await fetch(roots.cameraPathUrl, { signal: controls.signal })).json())
    : orbitPath(scene.bounds);
  // The path's own field of view; a far plane that reaches every corner of the bounds from any eye inside them.
  const diagonal = Math.hypot(...[0, 1, 2].map(axis => scene!.bounds.max[axis] - scene!.bounds.min[axis]));
  projection = resolveCameraProjection({ ...(path.verticalFovRadians === null ? {} : { verticalFovRadians: path.verticalFovRadians }),
    near: 0.1, far: Math.max(150, Math.ceil(diagonal)) });
  const size = drawingSize(), first = sampleCameraPath(path, 0);
  setStatus('loading', 'PLANNING GEOMETRY BUDGET');
  const streamed = scene.batches.filter(batch => batch.geometryStream);
  const geometry = planLargeSceneGeometry(streamed.map(batch => ({ directory: batch.geometryStream!.directory, models: batch.instances.map(i => i.model) })),
    { eye: first.eye, target: first.target, width: size.width, height: size.height, projection, maxErrorPixels },
    { maxResidentBytes: geometryBytes },
    // Renderer requests the standard WebGPU limits; it re-validates its actual device limits itself.
    { maxBufferSize: 256 * MiB, maxStorageBufferBindingSize: 128 * MiB, maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 });
  geometryReport = geometry.report;
  streamed.forEach((batch, index) => {
    const stream = batch.geometryStream!;
    (batch as { geometryStream: typeof stream }).geometryStream = { ...stream, gpuPool: { ...stream.gpuPool, maxResidentBytes: geometry.poolBytes[index] } };
  });
  lightingPlan = planLargeSceneLighting(scene.bounds, size.width, size.height, projection.far, scene.batches.length, lightingRequest);
  validatedSize = `${size.width}x${size.height}`;
  configured = { gi: !!lightingPlan.options.diffuseGi, reflections: !!lightingPlan.options.reflections,
    fog: !!lightingPlan.options.volumetricFog, airLighting: lightingRequest.airLighting };
  setStatus('loading', 'CREATING RENDERER');
  const created = await Renderer.create(canvas, scene.batches, { maxBatches: Math.max(256, scene.batches.length), antiAliasing: 'fxaa',
    textureBudgetBytes: 1024 * MiB, projection, geometryRuntimeBudget: geometry.capacity, ...lightingPlan.options,
    textureStreaming: { sources: scene.sources, maxCpuBytes: 256 * MiB, maxInFlightBytes: 128 * MiB, maxUploadBytesPerFrame: 2 * MiB },
    sunShadow: { resolution: 2048, focusBounds: scene.bounds } });
  if (disposed) { created.dispose(); return; }
  renderer = created;
  setSun(sunDirection);
  for (const key of Object.keys(configured) as (keyof LightingState)[]) element<HTMLInputElement>('light-' + key).disabled = !configured[key];
  setLighting({ ...configured, fog: configured.fog && initialFog });
  applyLod();
  element('adapter').textContent = renderer.info.adapter;
  element('batches').textContent = `${streamed.length} paged · ${scene.mask.primitives} mask · textures ${scene.sources.length}`;
  element('triangles').textContent = `${scene.pagedTriangles.toLocaleString()} + ${scene.mask.triangles.toLocaleString()} mask`;
  element('omitted').textContent = `${scene.omitted.length} primitives · ${scene.omittedTriangles.toLocaleString()} tris`;
  element('projection').textContent = `${(projection.verticalFovRadians * 180 / Math.PI).toFixed(1)}° · ${projection.near}–${projection.far} m`;
  element('queries').textContent = `${scene.queryProxies.fine ? `fine (error ${scene.queryProxies.maxError})` : 'built-in'} · ${scene.queryProxies.triangles.toLocaleString()} tris`;
  const report = lightingPlan.report;
  element('lighting').textContent = [report.probes && `GI ${report.probes.counts.join('×')}`, report.reflections && `refl ${report.reflections.spacing} px`,
    report.fog && `fog ${report.fog.pixelSize} px × ${report.fog.slices}`].filter(Boolean).join(' · ') || 'direct only';
  const keyframes = element('keyframes');
  for (const [n, index] of cameraPathKeyframes(path).entries()) {
    const button = document.createElement('button');
    button.textContent = `V${n + 1}`;
    button.title = `t = ${(path.samples[index].time - path.samples[0].time).toFixed(1)} s`;
    button.addEventListener('click', () => showKeyframe(index), { signal: controls.signal });
    keyframes.append(button);
  }
  if (autorun) { setWalk(true); start(autorun); return; }
  setWalk(params.get('walk') === '1');
  setStatus('ready', `READY · ${path.name}`);
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}

export interface LargeSceneHarness {
  snapshot: typeof snapshot;
  showKeyframe(index: number): void;
  setWalk(value: boolean): void;
  setDebug(value: DebugView): void;
  setLighting(next: Partial<LightingState>): void;
  setSun(direction: Vec3): void;
  run(frames: number, walk: boolean): void;
  settle(maxFrames: number): void;
  auditCoverage: typeof auditCoverage;
  setUiHidden(hidden: boolean): void;
  dispose(): void;
  readonly renderer: Renderer | undefined;
}
declare global { interface Window { __LARGE_SCENE__?: LargeSceneHarness } }

void startPage().catch(error => { fail(error); renderer?.dispose(); renderer = undefined; });
