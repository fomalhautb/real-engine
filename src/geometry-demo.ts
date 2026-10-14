import './geometry-demo.css';
import { GeometryDemoScene } from './geometry-demo-scene';
import { Renderer } from './renderer';

type DebugView = 'material' | 'clusters' | 'levels';
type Trajectory = 'combined' | 'orbit' | 'dolly' | 'manual';
type Vec3 = [number, number, number];
type Probe = Awaited<ReturnType<Renderer['probeGeometryLod']>>;

function element<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id);
  if (!value) throw new Error(`Missing geometry demo element: ${id}`);
  return value as T;
}

const canvas = element<HTMLCanvasElement>('viewport');
const app = element<HTMLDivElement>('geometry-app');
const controls = new AbortController();
const benchmarkMode = new URLSearchParams(location.search).get('benchmark') === '1';
let scene: GeometryDemoScene | undefined;
let renderer: Renderer | undefined;
let raf = 0;
let disposed = false;
let paused = benchmarkMode;
let lastTime = performance.now();
let lastUI = 0;
let lastProbeRequest = -Infinity;
let probePending = false;
let diagnosticsEnabled = !benchmarkMode;
let probe: Probe | null = null;
let phase = 0;
let yaw = 0.35;
let elevation = 0.24;
let radius = 14.5;
let trajectory: Trajectory = 'combined';
let debugView: DebugView = 'material';
let maxErrorPixels = 2;
let frustumCulling = true;
let frozen = false;
const target: Vec3 = [0, 0.15, 0];
const exportUrls = new Set<string>();
const exportTimers = new Set<number>();

function eye(): Vec3 {
  return [Math.sin(yaw) * Math.cos(elevation) * radius,
    target[1] + Math.sin(elevation) * radius,
    Math.cos(yaw) * Math.cos(elevation) * radius];
}

function applyLod(): void {
  renderer?.setGeometryLod({ maxErrorPixels, frustumCulling, frozen, debug: debugView });
  element<HTMLOutputElement>('error-value').textContent = `${maxErrorPixels.toFixed(1)} px`;
  const labels = { material: 'NORMAL MATERIAL', clusters: 'STABLE CLUSTER COLORS', levels: 'LOD LEVEL COLORS' };
  element('stage-mode').textContent = labels[debugView];
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-debug]')) {
    const active = button.dataset.debug === debugView;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
}

function advanceCamera(dt: number): void {
  if (trajectory === 'manual') return;
  phase += dt;
  yaw = trajectory === 'dolly' ? 0.35 : 0.35 + phase * 0.27;
  radius = trajectory === 'orbit' ? 11.3 : 11.3 + 3.2 * Math.cos(phase * 0.43);
  elevation = 0.23 + 0.055 * Math.sin(phase * 0.31);
}

function refreshUI(): void {
  if (!scene || !renderer) return;
  const stats = renderer.stats;
  element('source-triangles').textContent = scene.sourceTriangles.toLocaleString();
  element('source-total').textContent = (scene.sourceTriangles * scene.instanceCount).toLocaleString();
  element('actual-triangles').textContent = probe ? probe.triangles.toLocaleString() : '—';
  element('actual-indices').textContent = probe ? probe.indices.toLocaleString() : '—';
  element('selected-clusters').textContent = probe ? probe.selectedClusters.length.toLocaleString() : '—';
  element('probe-frame').textContent = probe ? String(probe.frame) : '—';
  element('gpu-lod-ms').textContent = stats.gpuMs ? `${stats.gpuMs.geometryLod.toFixed(2)} ms` : '—';
  element('gpu-sample-frame').textContent = stats.gpuSampleFrame === null ? '—' : String(stats.gpuSampleFrame);
  element('cpu-submit-ms').textContent = `${stats.cpuSubmitMs.toFixed(2)} ms`;
  element('viewport-size').textContent = `${stats.width} × ${stats.height}`;
  element('gpu-memory').textContent = `${(stats.ownedGpuBytes / 1048576).toFixed(1)} MiB`;
  element('footer-frame').textContent = `FRAME ${stats.frames.toLocaleString()}`;
  element('probe-status').textContent = probe
    ? probe.overflow ? `帧 ${probe.frame}：簇选择溢出；统计可能不完整。`
      : `帧 ${probe.frame}：GPU 选择结果，最多每 500 ms 更新一次。`
    : '等待第一帧 GPU 统计…';
  element('status').textContent = paused ? 'PAUSED' : 'WEBGPU LIVE';
  element<HTMLButtonElement>('pause').textContent = paused ? '继续轨迹' : '暂停轨迹';
}

function snapshot() {
  const diagnostics = renderer?.diagnostics();
  return {
    page: 'resident-gpu-lod', scope: { streaming: false, occlusionCulling: false },
    source: scene ? { trianglesPerInstance: scene.sourceTriangles,
      instances: scene.instanceCount, clusters: scene.hierarchy.clusters.length,
      groups: scene.hierarchy.groups.length, roots: scene.hierarchy.roots.length } : null,
    camera: { eye: eye(), target: [...target], yaw, elevation, radius, phase, trajectory },
    controls: { paused, diagnosticsEnabled, maxErrorPixels, frustumCulling, frozen, debug: debugView },
    gpuProbe: probe ? structuredClone(probe) : null,
    stats: renderer ? structuredClone(renderer.stats) : null,
    geometryLod: diagnostics?.geometryLod ?? null,
    diagnostics: diagnostics ?? null,
    adapter: renderer?.info ?? null,
    errors: renderer ? [...renderer.errors] : [],
  };
}

async function maybeProbe(now: number): Promise<void> {
  if (!diagnosticsEnabled || !renderer || disposed || probePending || now - lastProbeRequest < 500 || renderer.stats.frames === 0) return;
  const active = renderer;
  probePending = true;
  lastProbeRequest = now;
  try {
    const result = await active.probeGeometryLod('sculpture');
    if (!disposed && renderer === active && (!probe || result.frame >= probe.frame)) {
      probe = result;
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
  renderer.render(eye(), target);
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

function setDebug(value: DebugView): void {
  if (value !== 'material' && value !== 'clusters' && value !== 'levels') throw new RangeError('invalid geometry debug view');
  debugView = value;
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
  scene = undefined;
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
  canvas.addEventListener('pointerdown', event => {
    dragging = true; px = event.clientX; py = event.clientY; canvas.setPointerCapture(event.pointerId);
  }, { signal });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    yaw -= (event.clientX - px) * 0.006;
    elevation = Math.min(1.2, Math.max(-0.16, elevation + (event.clientY - py) * 0.005));
    px = event.clientX; py = event.clientY;
    trajectory = 'manual'; element<HTMLSelectElement>('trajectory').value = 'manual';
    renderIfPaused();
  }, { signal });
  canvas.addEventListener('pointerup', () => { dragging = false; }, { signal });
  canvas.addEventListener('pointercancel', () => { dragging = false; }, { signal });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    radius = Math.min(28, Math.max(6.5, radius * Math.exp(event.deltaY * 0.001)));
    trajectory = 'manual'; element<HTMLSelectElement>('trajectory').value = 'manual';
    renderIfPaused();
  }, { signal, passive: false });
  window.addEventListener('resize', renderIfPaused, { signal });
  document.addEventListener('visibilitychange', () => { lastTime = performance.now(); }, { signal });
  window.addEventListener('pagehide', dispose, { once: true, signal });
}

async function start(): Promise<void> {
  scene = await GeometryDemoScene.create();
  if (disposed) return;
  const created = await Renderer.create(canvas, scene.batches);
  if (disposed) { created.dispose(); return; }
  renderer = created;
  renderer.setSun({ direction: [-12, 24, 15], color: [3.3, 3.0, 2.8] });
  element('adapter').textContent = `${renderer.info.adapter} · ${renderer.info.timing}`;
  applyLod();
  wireControls();
  window.__GEOMETRY_DEMO__ = { get renderer() { return renderer; }, snapshot, pause, step, setDebug, setDiagnosticsEnabled, dispose };
  renderOnce();
  refreshUI();
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}

export interface GeometryDemoHarness {
  readonly renderer: Renderer | undefined;
  snapshot: typeof snapshot;
  pause(value: boolean): void;
  step(dt?: number): void;
  setDebug(value: DebugView): void;
  setDiagnosticsEnabled(value: boolean): void;
  dispose(): void;
}
function setDiagnosticsEnabled(value: boolean): void {
  if (disposed) throw new Error('GEOMETRY_DEMO_DISPOSED');
  if (typeof value !== 'boolean') throw new Error('GEOMETRY_DEMO_DIAGNOSTICS');
  diagnosticsEnabled = value;
}
declare global { interface Window { __GEOMETRY_DEMO__?: GeometryDemoHarness } }

void start().catch(error => { fail(error); dispose(); });
if (import.meta.hot) import.meta.hot.dispose(dispose);
