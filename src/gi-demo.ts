import './gi-demo.css';
import { createGiDemoScene, GI_PROBE_GRID } from './gi-demo-scene';
import { Renderer, type RendererOptions } from './renderer';

type Vec3 = [number, number, number];
type WallColor = 'coral' | 'gold' | 'mint';
type GiProbe = Awaited<ReturnType<Renderer['probeDiffuseGi']>>;

const WALL_COLORS: Record<WallColor, [number, number, number, number]> = {
  coral: [0.82, 0.105, 0.075, 1],
  gold: [0.72, 0.39, 0.085, 1],
  mint: [0.075, 0.46, 0.33, 1],
};

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`GI demo element missing: ${id}`);
  return found as T;
}

const app = element<HTMLDivElement>('gi-app');
const canvas = element<HTMLCanvasElement>('viewport');
const events = new AbortController();
const benchmarkMode = new URLSearchParams(location.search).get('benchmark') === '1';
const finalGatherPreview = new URLSearchParams(location.search).get('finalGather') === '1';
const exportUrls = new Set<string>();
const exportTimers = new Set<number>();
let renderer: Renderer | undefined;
let scene: ReturnType<typeof createGiDemoScene> | undefined;
let latestProbe: GiProbe | null = null;
let lastProbeRequest = -Infinity;
let probePending = false;
let diagnosticsEnabled = !benchmarkMode;
let lastUi = -Infinity;
let lastTime = performance.now();
let phase = 0;
let yaw = 0.18;
let elevation = 0.21;
let distance = 8.7;
let manualCamera = finalGatherPreview && !benchmarkMode;
let paused = benchmarkMode;
let giEnabled = true;
let multiBounce = true;
let wallColor: WallColor = 'coral';
let sunAzimuth = 72;
let sunElevation = 52;
let sunColor = '#fff1dc';
let sunIntensity = 3.2;
let raf = 0;
let disposed = false;
const target: Vec3 = [0, 1.15, -0.35];

function eye(): Vec3 {
  return [target[0] + Math.sin(yaw) * Math.cos(elevation) * distance,
    target[1] + Math.sin(elevation) * distance,
    target[2] + Math.cos(yaw) * Math.cos(elevation) * distance];
}

function applySun(): void {
  const az = sunAzimuth * Math.PI / 180;
  const el = sunElevation * Math.PI / 180;
  const unit = (offset: number) => {
    const srgb = parseInt(sunColor.slice(offset, offset + 2), 16) / 255;
    return srgb <= 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055) ** 2.4;
  };
  renderer?.setSun({
    direction: [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)],
    color: [unit(1) * sunIntensity, unit(3) * sunIntensity, unit(5) * sunIntensity],
  });
  element<HTMLOutputElement>('azimuth-value').textContent = `${sunAzimuth}°`;
  element<HTMLOutputElement>('elevation-value').textContent = `${sunElevation}°`;
  element<HTMLOutputElement>('intensity-value').textContent = sunIntensity.toFixed(1);
}

function setWallColor(value: WallColor): void {
  if (!(value in WALL_COLORS)) throw new RangeError('Unknown GI demo wall color');
  wallColor = value;
  renderer?.setMaterialParameters('red-wall', new Float32Array(WALL_COLORS[value]));
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-wall]')) {
    const active = button.dataset.wall === value;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
  renderIfPaused();
}

function toggleGi(enabled = !giEnabled): void {
  if (disposed) throw new Error('GI_DEMO_DISPOSED');
  giEnabled = enabled;
  renderer?.setDiffuseGiEnabled(enabled);
  element<HTMLInputElement>('gi-enabled').checked = enabled;
  refreshStageMode();
  renderIfPaused();
  refreshUi();
}

function refreshStageMode(): void {
  element('stage-mode').textContent = !giEnabled ? 'GI OFF · DIRECT SUN ONLY'
    : multiBounce ? 'GI ON · MULTI-BOUNCE DIFFUSE' : 'GI ON · SINGLE-BOUNCE DIFFUSE';
}

function setMultiBounce(enabled: boolean): void {
  if (disposed) throw new Error('GI_DEMO_DISPOSED');
  if (typeof enabled !== 'boolean') throw new RangeError('GI_DEMO_MULTI_BOUNCE');
  renderer?.setDiffuseGiMultiBounce(enabled);
  multiBounce = enabled;
  element<HTMLInputElement>('gi-multi-bounce').checked = enabled;
  refreshStageMode();
  renderIfPaused();
  refreshUi();
}

function animateCamera(dt: number): void {
  if (manualCamera) return;
  phase += dt;
  yaw = 0.18 + 0.23 * Math.sin(phase * 0.22);
  elevation = 0.21 + 0.035 * Math.sin(phase * 0.17);
  distance = 7.35 + 1.35 * Math.cos(phase * 0.31);
}

function refreshUi(): void {
  if (!renderer) return;
  const stats = renderer.stats;
  element('valid-probes').textContent = latestProbe
    ? `${latestProbe.probes.length - latestProbe.invalidProbes} / ${latestProbe.probes.length}` : '—';
  element('probe-frame').textContent = latestProbe ? String(latestProbe.frame) : '—';
  element('gpu-gi-ms').textContent = stats.gpuMs ? `${stats.gpuMs.diffuseGi.toFixed(2)} ms` : '—';
  element('gpu-sample-frame').textContent = stats.gpuSampleFrame === null ? '—' : String(stats.gpuSampleFrame);
  element('cpu-submit-ms').textContent = `${stats.cpuSubmitMs.toFixed(2)} ms`;
  element('scene-triangles').textContent = scene?.triangles.toLocaleString() ?? '—';
  element('gpu-memory').textContent = `${(stats.ownedGpuBytes / 1048576).toFixed(1)} MiB`;
  element('viewport-size').textContent = `${stats.width} × ${stats.height}`;
  element('footer-frame').textContent = `FRAME ${stats.frames.toLocaleString()}`;
  element('probe-status').textContent = !giEnabled
    ? `GI 已关闭；探针不再更新。${latestProbe ? `上次采样为帧 ${latestProbe.frame}。` : ''}`
    : latestProbe ? `帧 ${latestProbe.frame} 的显式读回；最多每秒一次。` : '等待第一帧探针诊断…';
  element('status').textContent = paused ? 'PAUSED' : 'WEBGPU LIVE';
  element<HTMLButtonElement>('pause').textContent = paused ? '继续轨迹' : '暂停轨迹';
}

function snapshot() {
  const diagnostics = renderer?.diagnostics();
  return {
    page: 'diffuse-gi-demo', scope: {
      transport: 'diffuse-only', probePlacement: 'fixed-manual-grid',
      multiBounce,
      surfaceFinalGather: finalGatherPreview,
      feedback: multiBounce ? 'previous-completed-probe-SH' : 'none',
      temporalFiltering: false, ambientFallback: false,
    },
    scene: scene ? { triangles: scene.triangles, batches: scene.batches.map(batch => batch.name) } : null,
    probeGrid: GI_PROBE_GRID,
    camera: { eye: eye(), target: [...target], yaw, elevation, distance, phase, manualCamera },
    controls: { paused, diagnosticsEnabled, giEnabled, multiBounce, wallColor, sunAzimuth, sunElevation, sunColor, sunIntensity },
    probe: latestProbe ? structuredClone(latestProbe) : null,
    diffuseGi: diagnostics?.diffuseGi ?? null,
    stats: renderer ? structuredClone(renderer.stats) : null,
    diagnostics: diagnostics ?? null,
    adapter: renderer?.info ?? null,
    errors: renderer ? [...renderer.errors] : [],
  };
}

async function maybeProbe(now: number): Promise<void> {
  if (!diagnosticsEnabled || !renderer || disposed || !giEnabled || probePending || now - lastProbeRequest < 1000 || renderer.stats.frames === 0) return;
  const active = renderer;
  lastProbeRequest = now;
  probePending = true;
  try {
    const result = await active.probeDiffuseGi();
    if (!disposed && renderer === active && (!latestProbe || result.frame >= latestProbe.frame)) {
      latestProbe = result;
      refreshUi();
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
  if (now - lastUi >= 150) { refreshUi(); lastUi = now; }
  void maybeProbe(now);
}

function renderIfPaused(): void {
  if (!paused || !renderer) return;
  try { renderOnce(); refreshUi(); } catch (error) { fail(error); }
}

function frame(now: number): void {
  if (disposed) return;
  try {
    const dt = Math.max(0, Math.min(0.1, (now - lastTime) / 1000));
    lastTime = now;
    if (!paused) { animateCamera(dt); renderOnce(now); }
    else if (renderer && giEnabled && latestProbe?.frame !== renderer.stats.frames) void maybeProbe(now);
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
  if (disposed) throw new Error('GI_DEMO_DISPOSED');
  paused = value;
  lastTime = performance.now();
  refreshUi();
}

function step(dt = 1 / 60): void {
  if (!Number.isFinite(dt) || dt <= 0 || dt > 1) throw new RangeError('step requires 0 < dt <= 1 second');
  if (!renderer || disposed) throw new Error('GI_DEMO_NOT_READY');
  pause(true);
  animateCamera(dt);
  renderOnce();
  refreshUi();
}

function dispose(): void {
  if (disposed) return;
  disposed = true;
  cancelAnimationFrame(raf);
  events.abort();
  renderer?.dispose();
  renderer = undefined;
  scene = undefined;
  for (const url of exportUrls) URL.revokeObjectURL(url);
  exportUrls.clear();
  for (const timer of exportTimers) clearTimeout(timer);
  exportTimers.clear();
  delete window.__GI_DEMO__;
}

function wireControls(): void {
  const signal = events.signal;
  element<HTMLInputElement>('gi-enabled').addEventListener('change', event =>
    toggleGi((event.target as HTMLInputElement).checked), { signal });
  element<HTMLInputElement>('gi-multi-bounce').addEventListener('change', event =>
    setMultiBounce((event.target as HTMLInputElement).checked), { signal });
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-wall]')) {
    button.addEventListener('click', () => setWallColor(button.dataset.wall as WallColor), { signal });
  }
  element<HTMLInputElement>('sun-azimuth').addEventListener('input', event => {
    sunAzimuth = Number((event.target as HTMLInputElement).value); applySun(); renderIfPaused();
  }, { signal });
  element<HTMLInputElement>('sun-elevation').addEventListener('input', event => {
    sunElevation = Number((event.target as HTMLInputElement).value); applySun(); renderIfPaused();
  }, { signal });
  element<HTMLInputElement>('sun-color').addEventListener('input', event => {
    sunColor = (event.target as HTMLInputElement).value; applySun(); renderIfPaused();
  }, { signal });
  element<HTMLInputElement>('sun-intensity').addEventListener('input', event => {
    sunIntensity = Number((event.target as HTMLInputElement).value); applySun(); renderIfPaused();
  }, { signal });
  element('pause').addEventListener('click', () => pause(!paused), { signal });
  element('step').addEventListener('click', () => { try { step(); } catch (error) { fail(error); } }, { signal });
  const toggleUi = () => {
    const hidden = app.classList.toggle('ui-hidden');
    element('toggle-ui').textContent = hidden ? 'H / 显示界面' : 'H / 隐藏界面';
    renderIfPaused();
  };
  element('toggle-ui').addEventListener('click', toggleUi, { signal });
  window.addEventListener('keydown', event => {
    if (event.key.toLowerCase() === 'h' && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault(); toggleUi();
    }
  }, { signal });
  element('export').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot(), null, 2)], { type: 'application/json' }));
    exportUrls.add(url);
    const link = document.createElement('a');
    link.href = url; link.download = 'real-engine-gi-diagnostics.json'; link.click();
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
    elevation = Math.max(-0.04, Math.min(0.9, elevation + (event.clientY - py) * 0.005));
    px = event.clientX; py = event.clientY;
    manualCamera = true; renderIfPaused();
  }, { signal });
  canvas.addEventListener('pointerup', () => { dragging = false; }, { signal });
  canvas.addEventListener('pointercancel', () => { dragging = false; }, { signal });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    distance = Math.max(5.2, Math.min(16, distance * Math.exp(event.deltaY * 0.001)));
    manualCamera = true; renderIfPaused();
  }, { signal, passive: false });
  window.addEventListener('resize', renderIfPaused, { signal });
  document.addEventListener('visibilitychange', () => { lastTime = performance.now(); }, { signal });
  window.addEventListener('pagehide', dispose, { signal, once: true });
}

async function start(): Promise<void> {
  scene = createGiDemoScene();
  const rayBudget = GI_PROBE_GRID.counts.reduce((a, b) => a * b, 1) * GI_PROBE_GRID.raysPerProbe;
  element('probe-budget').textContent = `${GI_PROBE_GRID.counts.reduce((a, b) => a * b, 1)} 个固定探针，每颗 ${GI_PROBE_GRID.raysPerProbe} 条方向光线＋${GI_PROBE_GRID.emissiveSamplesPerProbe} 个发光面样本`;
  const options: RendererOptions = { diffuseGi: finalGatherPreview ? { ...GI_PROBE_GRID,
    maxBytes: 64 * 1024 * 1024,
    finalGather: { pixelSpacing: 4, maxAnchors: 65536, raysPerAnchor: 1, maxHistory: 256,
      maxBytes: 64 * 1024 * 1024 } } : GI_PROBE_GRID,
    worldQueries: { maxRays: finalGatherPreview ? 65536 : rayBudget }, antiAliasing: 'fxaa',
    sunShadow: { resolution: 2048, focusBounds: { min: [-3.5, -0.25, -4], max: [3.5, 4, 4] },
      maxBytes: 16 * 1024 * 1024 } };
  const created = await Renderer.create(canvas, scene.batches, options);
  if (disposed) { created.dispose(); return; }
  renderer = created;
  applySun();
  element('adapter').textContent = `${renderer.info.adapter} · ${renderer.info.timing}`;
  wireControls();
  window.__GI_DEMO__ = { get renderer() { return renderer; }, snapshot, pause, step, toggleGi, setMultiBounce, setWallColor, setDiagnosticsEnabled, dispose };
  refreshStageMode();
  renderOnce();
  refreshUi();
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}

export interface GiDemoHarness {
  readonly renderer: Renderer | undefined;
  snapshot: typeof snapshot;
  pause(value: boolean): void;
  step(dt?: number): void;
  toggleGi(enabled?: boolean): void;
  setMultiBounce(enabled: boolean): void;
  setWallColor(value: WallColor): void;
  setDiagnosticsEnabled(value: boolean): void;
  dispose(): void;
}
function setDiagnosticsEnabled(value: boolean): void {
  if (disposed) throw new Error('GI_DEMO_DISPOSED');
  if (typeof value !== 'boolean') throw new Error('GI_DEMO_DIAGNOSTICS');
  diagnosticsEnabled = value;
}
declare global { interface Window { __GI_DEMO__?: GiDemoHarness } }

void start().catch(error => { fail(error); dispose(); });
if (import.meta.hot) import.meta.hot.dispose(dispose);
