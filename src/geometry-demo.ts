import './geometry-demo.css';
import { GeometryDemoScene } from './geometry-demo-scene';
import { StreamedGalleryScene, STREAMED_GALLERY_TRAJECTORIES, type StreamedGalleryTrajectory } from './geometry-demo-streamed-scene';
import { GeometryStreamCut } from './geometry-stream-cut';
import { GEOMETRY_DEBUG_VIEWS, type GeometryDebugView } from './gpu-geometry-lod';
import { multiply, lookAt, perspective } from './math';
import { Renderer } from './renderer';

type DebugView = GeometryDebugView;
type ResidentTrajectory = 'combined' | 'orbit' | 'dolly';
type Trajectory = ResidentTrajectory | StreamedGalleryTrajectory | 'manual';
type Vec3 = [number, number, number];
type Probe = Awaited<ReturnType<Renderer['probeGeometryLod']>>;
type Mode = 'resident' | 'streamed';

function element<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id);
  if (!value) throw new Error(`Missing geometry demo element: ${id}`);
  return value as T;
}

const params = new URLSearchParams(location.search);
const mode: Mode = params.get('scene') === 'streamed' ? 'streamed' : 'resident';
const numberParam = (name: string) => params.has(name) ? Number(params.get(name)) : undefined;
const canvas = element<HTMLCanvasElement>('viewport');
const app = element<HTMLDivElement>('geometry-app');
const controls = new AbortController();
const benchmarkMode = params.get('benchmark') === '1';
/** Exact full-detail resident meshes, no paging, LOD or occlusion: the hole/mis-culling reference. */
const referenceMode = mode === 'streamed' && params.get('reference') === '1';
let resident: GeometryDemoScene | undefined;
let streamed: StreamedGalleryScene | undefined;
let planners: GeometryStreamCut[] = [];
let renderer: Renderer | undefined;
let raf = 0;
let disposed = false;
let paused = benchmarkMode;
let lastTime = performance.now();
let lastUI = 0;
let lastProbeRequest = -Infinity;
let probePending = false;
let diagnosticsEnabled = !benchmarkMode && !referenceMode;
let probe: Probe | null = null;
let cpuPrediction: { visibleObjects: number; fullResidencyTriangles: number; residentCutTriangles: number; frame: number } | null = null;
let frameInterval = 0;
let phase = 0;
let yaw = 0.35;
let elevation = 0.24;
let radius = 14.5;
let trajectory: Trajectory = mode === 'streamed' ? 'walk' : 'combined';
let debugView: DebugView = 'material';
let debugSplit = 0;
let maxErrorPixels = mode === 'streamed' ? 1 : 2;
let frustumCulling = true;
let frozen = false;
let occlusion = mode === 'streamed' && !referenceMode;
const target: Vec3 = [0, 0.15, 0];
let streamEye: Vec3 = [0, 1.65, -1.5];
let streamTarget: Vec3 = [0, 1.2, 4.5];
let lastDiscontinuity = -1;
const exportUrls = new Set<string>();
const exportTimers = new Set<number>();
const VIEW_LABELS: Record<DebugView, string> = { material: 'NORMAL MATERIAL', clusters: 'STABLE CLUSTER COLORS',
  levels: 'LOD LEVEL COLORS', residency: 'PAGE RESIDENCY / FALLBACK' };

function eye(): Vec3 {
  if (mode === 'streamed') return [...streamEye];
  return [Math.sin(yaw) * Math.cos(elevation) * radius,
    target[1] + Math.sin(elevation) * radius,
    Math.cos(yaw) * Math.cos(elevation) * radius];
}
function lookTarget(): Vec3 { return mode === 'streamed' ? [...streamTarget] : [...target]; }

function applyLod(): void {
  if (renderer && !referenceMode) {
    renderer.setGeometryLod({ maxErrorPixels, frustumCulling, frozen, debug: debugView, debugSplit });
    if (mode === 'streamed') renderer.setOcclusionEnabled(occlusion);
  }
  element<HTMLOutputElement>('error-value').textContent = `${maxErrorPixels.toFixed(1)} px`;
  element('stage-mode').textContent = debugSplit > 0 && debugView !== 'material' ? `MATERIAL | ${VIEW_LABELS[debugView]}` : VIEW_LABELS[debugView];
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-debug]')) {
    const active = button.dataset.debug === debugView;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
}

function advanceCamera(dt: number): void {
  if (trajectory === 'manual') return;
  phase += dt;
  if (mode === 'streamed' && streamed) {
    const pose = streamed.pose(trajectory as StreamedGalleryTrajectory, phase);
    streamEye = [...pose.eye]; streamTarget = [...pose.target];
    if (pose.discontinuity) lastDiscontinuity = renderer?.stats.frames ?? 0;
    return;
  }
  yaw = trajectory === 'dolly' ? 0.35 : 0.35 + phase * 0.27;
  radius = trajectory === 'orbit' ? 11.3 : 11.3 + 3.2 * Math.cos(phase * 0.43);
  elevation = 0.23 + 0.055 * Math.sin(phase * 0.31);
}

const mib = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;

function streamSummary() {
  if (!renderer || !streamed) return null;
  const diagnostics = renderer.diagnostics();
  let resident = 0, pending = 0, requested = 0, budgetDeferred = 0, uploads = 0, evictions = 0, poolBytes = 0, visible = 0, indices = 0;
  for (const stream of diagnostics.geometryStreams) {
    resident += stream.pool.residentPages.length; pending += stream.pool.pendingPages.length;
    requested += stream.allocation.requestedPages.length; budgetDeferred += stream.allocation.budgetDeferredGroups.length;
    uploads += stream.pool.uploads; evictions += stream.pool.evictions; poolBytes += stream.pool.allocatedGpuBytes;
    visible += stream.feedback?.visibleClusters ?? 0; indices += stream.feedback?.indices ?? 0;
  }
  const budget = diagnostics.geometryRuntimeBudget;
  return { streams: diagnostics.geometryStreams.length, residentPages: resident, totalPages: streamed.accounting.pages, pendingPages: pending,
    requestedPages: requested, budgetDeferredGroups: budgetDeferred, uploads, evictions, poolBytes,
    feedbackVisibleClusters: visible, feedbackTriangles: indices / 3, runtimeBudget: budget ?? null };
}

function refreshUI(): void {
  if (!renderer) return;
  const stats = renderer.stats;
  if (resident) {
    element('source-triangles').textContent = resident.sourceTriangles.toLocaleString();
    element('source-total').textContent = (resident.sourceTriangles * resident.instanceCount).toLocaleString();
  } else if (streamed) {
    element('source-triangles-streamed').textContent = streamed.accounting.sourceTriangles.toLocaleString();
  }
  element('actual-triangles').textContent = probe ? probe.triangles.toLocaleString() : '—';
  element('actual-indices').textContent = probe ? probe.indices.toLocaleString() : '—';
  element('selected-clusters').textContent = probe ? probe.selectedClusters.length.toLocaleString() : '—';
  element('probe-frame').textContent = probe ? String(probe.frame) : '—';
  const gpu = stats.gpuMs;
  element('gpu-lod-ms').textContent = gpu ? `${gpu.geometryLod.toFixed(2)} ms` : '—';
  element('gpu-frame-ms').textContent = gpu ? `${Object.values(gpu).reduce((sum, value) => sum + value, 0).toFixed(2)} ms` : '—';
  element('gpu-sample-frame').textContent = stats.gpuSampleFrame === null ? '—' : String(stats.gpuSampleFrame);
  element('cpu-submit-ms').textContent = `${stats.cpuSubmitMs.toFixed(2)} ms`;
  element('raf-ms').textContent = frameInterval ? `${frameInterval.toFixed(1)} ms` : '—';
  element('viewport-size').textContent = `${stats.width} × ${stats.height}`;
  element('gpu-memory').textContent = mib(stats.ownedGpuBytes);
  element('footer-frame').textContent = `FRAME ${stats.frames.toLocaleString()}`;
  if (streamed) {
    const summary = streamSummary()!, a = streamed.accounting;
    const fallback = probe ? probe.selectedClusters.filter(cluster => cluster.fallback).length : null;
    element('stream-pages').textContent = `${summary.residentPages.toLocaleString()} / ${summary.totalPages.toLocaleString()}`;
    element('stream-requested').textContent = `${summary.requestedPages} · ${summary.pendingPages}`;
    element('stream-deferred').textContent = summary.budgetDeferredGroups.toLocaleString();
    element('stream-fallback').textContent = fallback === null ? '—' : fallback.toLocaleString();
    element('stream-pool').textContent = `${mib(summary.poolBytes)} / ${mib(a.poolBudgetBytes)}`;
    element('stream-ratio').textContent = `${a.uniqueToPool.toFixed(2)}×`;
    element('stream-feedback').textContent = `${summary.feedbackVisibleClusters.toLocaleString()} · ${Math.round(summary.feedbackTriangles).toLocaleString()}`;
    const budget = summary.runtimeBudget;
    element('stream-migrations').textContent = budget && 'migrations' in budget ? `${budget.migrations} · ${budget.phase}` : '—';
    element('cpu-predicted').textContent = cpuPrediction
      ? `${cpuPrediction.fullResidencyTriangles.toLocaleString()} / ${cpuPrediction.residentCutTriangles.toLocaleString()}` : '—';
    element('stream-unique').textContent = `${mib(a.uniqueResidentBytes)} 独特驻留格式 · ${a.objects} 件 · ${a.pages.toLocaleString()} 页`;
  }
  element('probe-status').textContent = probe
    ? probe.overflow ? `帧 ${probe.frame}：簇选择溢出；统计可能不完整。`
      : `帧 ${probe.frame}：GPU 实际选择的诊断回读，最多每 500 ms 一次。`
    : '等待第一帧 GPU 统计…';
  element('status').textContent = paused ? 'PAUSED' : 'WEBGPU LIVE';
  element<HTMLButtonElement>('pause').textContent = paused ? '继续轨迹' : '暂停轨迹';
}

function snapshot() {
  const diagnostics = renderer?.diagnostics();
  return {
    page: mode === 'streamed' ? 'streamed-gpu-lod' : 'resident-gpu-lod',
    scope: { streaming: mode === 'streamed', occlusionCulling: mode === 'streamed' && occlusion },
    source: resident ? { trianglesPerInstance: resident.sourceTriangles,
      instances: resident.instanceCount, clusters: resident.hierarchy.clusters.length,
      groups: resident.hierarchy.groups.length, roots: resident.hierarchy.roots.length }
      : streamed ? { trianglesPerInstance: Math.round(streamed.accounting.sourceTriangles / streamed.accounting.objects),
        instances: streamed.accounting.objects, clusters: streamed.accounting.clusters, groups: null, roots: null } : null,
    accounting: streamed ? { ...streamed.accounting } : null,
    sceneOptions: streamed ? { ...streamed.options } : null,
    camera: { eye: eye(), target: lookTarget(), yaw, elevation, radius, phase, trajectory, lastDiscontinuityFrame: lastDiscontinuity },
    controls: { paused, diagnosticsEnabled, maxErrorPixels, frustumCulling, frozen, debug: debugView, debugSplit, occlusion },
    gpuProbe: probe ? structuredClone(probe) : null,
    cpuPrediction: cpuPrediction ? { ...cpuPrediction, basis: 'cpu-reference-cut-of-frustum-visible-objects' } : null,
    streams: streamSummary(),
    stats: renderer ? structuredClone(renderer.stats) : null,
    rafIntervalMs: frameInterval,
    geometryLod: diagnostics?.geometryLod ?? null,
    diagnostics: mode === 'resident' ? diagnostics ?? null : null,
    adapter: renderer?.info ?? null,
    errors: renderer ? [...renderer.errors] : [],
  };
}

/** CPU reference cut for frustum-visible objects: the complete cut a fully
 * resident pool would draw, and the cut the current GPU residency permits. */
function predict(): void {
  if (!streamed || !renderer) return;
  const view = renderer.diagnostics().geometryStreams;
  const vp = multiply(perspective(Math.PI / 3.2, renderer.stats.width / Math.max(1, renderer.stats.height), 0.1, 150), lookAt(eye(), lookTarget(), [0, 1, 0]));
  const projectionScale = renderer.stats.height / (2 * Math.tan(Math.PI / 6.4));
  let visibleObjects = 0, full = 0, residentCut = 0;
  streamed.objects.forEach((object, index) => {
    const c = object.center, clip = [0, 1, 2, 3].map(row => vp[row] * c[0] + vp[4 + row] * c[1] + vp[8 + row] * c[2] + vp[12 + row]);
    const r = object.radius * 2;
    if (clip[3] < -r || Math.abs(clip[0]) > clip[3] + r || Math.abs(clip[1]) > clip[3] + r) return;
    visibleObjects++;
    const planner = planners[index], stream = view.find(entry => entry.name === object.name);
    const settings = { eye: eye(), near: 0.1, projectionScale, maxErrorPixels, model: object.model };
    const all = new Set(object.directory.pages.map(page => page.id));
    const triangles = (ids: number[]) => ids.reduce((sum, id) => sum + object.directory.clusters[id].triangleCount, 0);
    full += triangles(planner.plan(all, settings).selectedClusters);
    if (stream) residentCut += triangles(planner.plan(new Set(stream.pool.residentPages), settings).selectedClusters);
  });
  cpuPrediction = { visibleObjects, fullResidencyTriangles: full, residentCutTriangles: residentCut, frame: renderer.stats.frames };
}

async function probeAll(active: Renderer): Promise<Probe> {
  if (!streamed) return active.probeGeometryLod('sculpture');
  const results = await Promise.all(streamed.streamBatchNames.map(name => active.probeGeometryLod(name)));
  return { frame: Math.min(...results.map(result => result.frame)), indices: results.reduce((sum, result) => sum + result.indices, 0),
    triangles: results.reduce((sum, result) => sum + result.triangles, 0), overflow: results.some(result => result.overflow),
    occlusion: { deferredClusters: results.reduce((sum, result) => sum + result.occlusion.deferredClusters, 0),
      culledClusters: results.reduce((sum, result) => sum + result.occlusion.culledClusters, 0),
      recoveredClusters: results.reduce((sum, result) => sum + result.occlusion.recoveredClusters, 0) },
    selectedClusters: results.flatMap((result, object) => result.selectedClusters.map(cluster => ({ ...cluster, instanceIndex: object }))) };
}

async function maybeProbe(now: number): Promise<void> {
  if (!diagnosticsEnabled || !renderer || disposed || probePending || now - lastProbeRequest < 500 || renderer.stats.frames === 0) return;
  const active = renderer;
  probePending = true;
  lastProbeRequest = now;
  try {
    const result = await probeAll(active);
    if (!disposed && renderer === active && (!probe || result.frame >= probe.frame)) {
      probe = result;
      predict();
      refreshUI();
    }
  } catch (error) {
    if (!disposed) fail(error);
  } finally {
    probePending = false;
  }
}

function renderOnce(now = performance.now()): void {
  if (!renderer || disposed) return;
  renderer.render(eye(), lookTarget());
  if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
  if (now - lastUI >= 150) { refreshUI(); lastUI = now; }
  void maybeProbe(now);
}

function renderIfPaused(): void {
  if (!paused) return;
  try { renderOnce(); } catch (error) { fail(error); }
}

function frame(now: number): void {
  if (disposed) return;
  try {
    const dt = Math.min(0.1, Math.max(0, (now - lastTime) / 1000));
    frameInterval = frameInterval ? frameInterval * 0.9 + (now - lastTime) * 0.1 : now - lastTime;
    lastTime = now;
    if (!paused) { advanceCamera(dt); renderOnce(now); }
    else if (renderer && probe?.frame !== renderer.stats.frames) void maybeProbe(now);
    raf = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

function fail(error: unknown): void {
  cancelAnimationFrame(raf);
  element('status').textContent = 'RUNTIME ERROR';
  const panel = element('error');
  panel.hidden = false;
  panel.textContent = String(error);
}

function pause(value: boolean): void {
  if (disposed) return;
  paused = value;
  lastTime = performance.now();
  refreshUI();
}

function step(dt = 1 / 60): void {
  if (!Number.isFinite(dt) || dt <= 0 || dt > 1) throw new RangeError('step requires 0 < dt <= 1 second');
  if (!renderer || disposed) throw new Error('GEOMETRY_DEMO_NOT_READY');
  pause(true);
  advanceCamera(dt);
  renderOnce();
  refreshUI();
}

/** Jump a fixed trajectory to time t without rendering; used by recordings and tests. */
function seek(value: Trajectory, t: number): void {
  if (!renderer || disposed) throw new Error('GEOMETRY_DEMO_NOT_READY');
  const allowed: readonly string[] = mode === 'streamed' ? [...STREAMED_GALLERY_TRAJECTORIES, 'manual'] : ['combined', 'orbit', 'dolly', 'manual'];
  if (!allowed.includes(value) || !Number.isFinite(t) || t < 0) throw new RangeError('invalid geometry demo trajectory');
  trajectory = value; phase = 0; element<HTMLSelectElement>('trajectory').value = value;
  advanceCamera(t);
}

function setDebug(value: DebugView): void {
  if (!(GEOMETRY_DEBUG_VIEWS as readonly string[]).includes(value)) throw new RangeError('invalid geometry debug view');
  debugView = value;
  applyLod();
  renderIfPaused();
}

function setSplit(value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError('invalid geometry debug split');
  debugSplit = value;
  element<HTMLInputElement>('split').checked = value > 0;
  applyLod();
  renderIfPaused();
}

function setMaxError(value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 64) throw new RangeError('invalid geometry error');
  maxErrorPixels = value;
  element<HTMLInputElement>('error-range').value = String(value);
  applyLod();
  renderIfPaused();
}

function setOcclusion(value: boolean): void {
  if (mode !== 'streamed' || referenceMode || typeof value !== 'boolean') throw new Error('GEOMETRY_DEMO_OCCLUSION');
  occlusion = value;
  element<HTMLInputElement>('occlusion').checked = value;
  applyLod();
  renderIfPaused();
}

function dispose(): void {
  if (disposed) return;
  disposed = true;
  cancelAnimationFrame(raf);
  controls.abort();
  renderer?.dispose();
  renderer = undefined;
  resident = undefined;
  streamed = undefined;
  planners = [];
  for (const url of exportUrls) URL.revokeObjectURL(url);
  exportUrls.clear();
  for (const timer of exportTimers) clearTimeout(timer);
  exportTimers.clear();
  delete window.__GEOMETRY_DEMO__;
}

function wireControls(): void {
  const signal = controls.signal;
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-debug]')) {
    button.addEventListener('click', () => setDebug(button.dataset.debug as DebugView), { signal });
  }
  element<HTMLInputElement>('error-range').addEventListener('input', event => {
    maxErrorPixels = Number((event.target as HTMLInputElement).value);
    applyLod(); renderIfPaused();
  }, { signal });
  element<HTMLInputElement>('freeze').addEventListener('change', event => {
    frozen = (event.target as HTMLInputElement).checked;
    applyLod(); renderIfPaused();
  }, { signal });
  element<HTMLInputElement>('frustum').addEventListener('change', event => {
    frustumCulling = (event.target as HTMLInputElement).checked;
    applyLod(); renderIfPaused();
  }, { signal });
  element<HTMLInputElement>('split').addEventListener('change', event => {
    setSplit((event.target as HTMLInputElement).checked ? 0.5 : 0);
  }, { signal });
  element<HTMLInputElement>('occlusion').addEventListener('change', event => {
    try { setOcclusion((event.target as HTMLInputElement).checked); } catch (error) { fail(error); }
  }, { signal });
  element<HTMLSelectElement>('trajectory').addEventListener('change', event => {
    trajectory = (event.target as HTMLSelectElement).value as Trajectory;
    if (trajectory !== 'manual') { phase = 0; advanceCamera(0); }
    renderIfPaused();
  }, { signal });
  element('pause').addEventListener('click', () => pause(!paused), { signal });
  element('step').addEventListener('click', () => { try { step(); } catch (error) { fail(error); } }, { signal });
  const toggleUI = () => {
    const hidden = app.classList.toggle('ui-hidden');
    element<HTMLButtonElement>('toggle-ui').textContent = hidden ? 'H / 显示界面' : 'H / 隐藏界面';
    element<HTMLButtonElement>('toggle-ui').setAttribute('aria-label', hidden ? '显示界面' : '隐藏界面');
    renderIfPaused();
  };
  element('toggle-ui').addEventListener('click', toggleUI, { signal });
  window.addEventListener('keydown', event => {
    if (event.key.toLowerCase() === 'h' && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault(); toggleUI();
    }
  }, { signal });
  element('export').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot(), null, 2)], { type: 'application/json' }));
    exportUrls.add(url);
    const link = document.createElement('a');
    link.href = url; link.download = 'real-engine-geometry-diagnostics.json'; link.click();
    const timer = window.setTimeout(() => {
      URL.revokeObjectURL(url); exportUrls.delete(url); exportTimers.delete(timer);
    }, 1000);
    exportTimers.add(timer);
  }, { signal });

  let dragging = false, px = 0, py = 0;
  const manual = () => { trajectory = 'manual'; element<HTMLSelectElement>('trajectory').value = 'manual'; };
  canvas.addEventListener('pointerdown', event => {
    dragging = true; px = event.clientX; py = event.clientY; canvas.setPointerCapture(event.pointerId);
  }, { signal });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    const dx = event.clientX - px, dy = event.clientY - py;
    px = event.clientX; py = event.clientY;
    if (mode === 'streamed') {
      // Look around the eye: yaw and pitch the view direction.
      const d = [streamTarget[0] - streamEye[0], streamTarget[1] - streamEye[1], streamTarget[2] - streamEye[2]];
      const length = Math.hypot(d[0], d[1], d[2]) || 1, heading = Math.atan2(d[0], d[2]) - dx * 0.005;
      const pitch = Math.min(1.2, Math.max(-1.2, Math.asin(d[1] / length) - dy * 0.004));
      streamTarget = [streamEye[0] + Math.sin(heading) * Math.cos(pitch) * 6, streamEye[1] + Math.sin(pitch) * 6,
        streamEye[2] + Math.cos(heading) * Math.cos(pitch) * 6];
    } else {
      yaw -= dx * 0.006;
      elevation = Math.min(1.2, Math.max(-0.16, elevation + dy * 0.005));
    }
    manual(); renderIfPaused();
  }, { signal });
  canvas.addEventListener('pointerup', () => { dragging = false; }, { signal });
  canvas.addEventListener('pointercancel', () => { dragging = false; }, { signal });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    if (mode === 'streamed') {
      const d = [streamTarget[0] - streamEye[0], streamTarget[1] - streamEye[1], streamTarget[2] - streamEye[2]];
      const length = Math.hypot(d[0], d[1], d[2]) || 1, move = -event.deltaY * 0.004;
      streamEye = [streamEye[0] + d[0] / length * move, streamEye[1] + d[1] / length * move, streamEye[2] + d[2] / length * move];
      streamTarget = [streamTarget[0] + d[0] / length * move, streamTarget[1] + d[1] / length * move, streamTarget[2] + d[2] / length * move];
    } else radius = Math.min(28, Math.max(6.5, radius * Math.exp(event.deltaY * 0.001)));
    manual(); renderIfPaused();
  }, { signal, passive: false });
  window.addEventListener('resize', renderIfPaused, { signal });
  document.addEventListener('visibilitychange', () => { lastTime = performance.now(); }, { signal });
  window.addEventListener('pagehide', dispose, { once: true, signal });
}

function configurePage(): void {
  const select = element<HTMLSelectElement>('trajectory');
  for (const node of document.querySelectorAll<HTMLElement>('[data-mode]')) node.hidden = node.dataset.mode !== mode;
  if (mode !== 'streamed') return;
  const labels: Record<string, string> = { walk: '穿过门洞游走', zoom: '拉近 / 拉远单件', orbit: '近距离绕行', turn: '门洞 180° 快速转头',
    teleport: '首尾房间瞬移', manual: '手动视角' };
  select.replaceChildren(...[...STREAMED_GALLERY_TRAJECTORIES, 'manual'].map(value => new Option(labels[value], value)));
  select.value = trajectory;
  element('geometry-heading-eyebrow').textContent = 'STREAMED GEOMETRY / PAGED LOD';
  element('geometry-heading-text').textContent = '每件雕塑都是独特几何。GPU 只驻留当前需要的页，缺页时由祖先簇补位。';
  element('scope-note').textContent = '页面流送 + 两阶段 Hi-Z 遮挡剔除。GPU 选择为实际绘制；CPU 参考切分单独标明。帧时间受同机其它负载影响，仅作诊断。';
  element<HTMLInputElement>('error-range').value = String(maxErrorPixels);
  element<HTMLInputElement>('occlusion').checked = occlusion;
}

async function start(): Promise<void> {
  configurePage();
  let created: Renderer;
  if (mode === 'streamed') {
    element('status').textContent = 'BUILDING PAGES';
    const shading = params.get('shading') === 'identity' ? 'identity' : 'lit';
    streamed = await StreamedGalleryScene.create({ objects: numberParam('objects'), around: numberParam('around'), across: numberParam('across'),
      providerLatencyMs: numberParam('latency'), targetUniqueToPoolRatio: numberParam('ratio'), shading,
      admissionIntervalFrames: numberParam('interval'), maxConcurrentMigrations: numberParam('migrations'),
      maxRequestsPerFrame: numberParam('requests'), admission: params.get('admission') === 'interval' ? 'interval' : undefined }, (done, total) => { element('status').textContent = `BUILDING ${done}/${total}`; });
    if (disposed) return;
    planners = streamed.objects.map(object => new GeometryStreamCut(object.directory));
    created = await Renderer.create(canvas, referenceMode ? streamed.referenceBatches : streamed.batches,
      referenceMode ? {} : { geometryRuntimeBudget: streamed.runtimeBudget, occlusion: {} });
    advanceCamera(0);
  } else {
    resident = await GeometryDemoScene.create();
    if (disposed) return;
    created = await Renderer.create(canvas, resident.batches);
  }
  if (disposed) { created.dispose(); return; }
  renderer = created;
  renderer.setSun({ direction: [-12, 24, 15], color: [3.3, 3.0, 2.8] });
  element('adapter').textContent = `${renderer.info.adapter} · ${renderer.info.timing}`;
  applyLod();
  wireControls();
  window.__GEOMETRY_DEMO__ = { get renderer() { return renderer; }, get mode() { return mode; }, snapshot, pause, step, seek, setDebug, setSplit,
    setMaxError, setOcclusion, setDiagnosticsEnabled, dispose };
  renderOnce();
  refreshUI();
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}

export interface GeometryDemoHarness {
  readonly renderer: Renderer | undefined;
  readonly mode: Mode;
  snapshot: typeof snapshot;
  pause(value: boolean): void;
  step(dt?: number): void;
  seek(trajectory: Trajectory, t: number): void;
  setDebug(value: DebugView): void;
  setSplit(value: number): void;
  setMaxError(value: number): void;
  setOcclusion(value: boolean): void;
  setDiagnosticsEnabled(value: boolean): void;
  dispose(): void;
}
function setDiagnosticsEnabled(value: boolean): void {
  if (disposed) throw new Error('GEOMETRY_DEMO_DISPOSED');
  if (typeof value !== 'boolean' || value && referenceMode) throw new Error('GEOMETRY_DEMO_DIAGNOSTICS');
  diagnosticsEnabled = value;
}
declare global { interface Window { __GEOMETRY_DEMO__?: GeometryDemoHarness } }

void start().catch(error => { fail(error); dispose(); });
if (import.meta.hot) import.meta.hot.dispose(dispose);
