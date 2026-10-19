import './fog-demo.css';
import { Renderer, type RendererOptions } from './renderer';
import type { InstanceState, SceneBatch } from './contracts';
import { createCube } from './geometry';
import { m, type MaterialDefinition } from './material';
import { multiply, normalMatrix, scaling, translation, type Mat4 } from './math';
import type { FogVolume } from './volumetric-fog';

type Vec3 = [number, number, number];
type Color = [number, number, number, number];

interface FogDemoHarness {
  pause(value: boolean): void;
  render(): void;
  resetHistory(): void;
  step(dt?: number): void;
  snapshot(): ReturnType<typeof snapshot>;
  dispose(): void;
}

declare global { interface Window { __FOG_DEMO__?: FogDemoHarness } }

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (!found) throw new Error(`FOG_DEMO_ELEMENT_MISSING:${id}`);
  return found as T;
};

const canvas = $<HTMLCanvasElement>('viewport');
const benchmarkMode = new URLSearchParams(location.search).get('benchmark') === '1';
const events = new AbortController();
const cube = createCube();
const material: MaterialDefinition = {
  shading: 'pbr', alphaMode: 'opaque', doubleSided: true,
  baseColor: m.instanceColor, metallic: 0, roughness: 0.84,
  emissive: m.constant([0, 0, 0, 1]), alphaCutoff: 0,
};

let nextId = 1;
const pieces: InstanceState[] = [];
function add(position: Vec3, size: Vec3, color: Color, roughness = 0.84): void {
  const model: Mat4 = multiply(translation(...position), scaling(...size));
  pieces.push({ entityId: nextId++, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color, metalness: 0, roughness, historyVersion: 0 });
}

const stone: Color = [0.73, 0.67, 0.56, 1];
const limestone: Color = [0.88, 0.83, 0.72, 1];
const roof: Color = [0.45, 0.47, 0.43, 1];
const bronze: Color = [0.39, 0.31, 0.21, 1];

// Every silhouette is a real opaque cube. The open cells between these roof
// members and the central opening in the rear wall are actual sun paths.
add([0, -0.20, 0], [11.5, 0.4, 11.5], stone);
add([0, -0.37, 0], [12.2, 0.2, 12.2], bronze);
for (const x of [-3.6, -2.25, -0.9, 0.45, 1.8, 3.15, 3.85]) {
  add([x, 3.52, 0], [0.42, 0.35, 8.7], roof);
}
for (const z of [-4.1, 0.0, 4.1]) add([0, 3.52, z], [8.1, 0.35, 0.48], roof);
for (const x of [-3.75, 3.75]) {
  for (const z of [-3.65, -1.2, 1.25, 3.7]) {
    add([x, 1.66, z], [0.52, 3.32, 0.52], limestone);
    add([x, 0.2, z], [0.82, 0.4, 0.82], stone);
    add([x, 3.25, z], [0.78, 0.32, 0.78], stone);
  }
}
// Rear wall, with a 2.1 m wide door left as empty space between solid slabs.
add([-2.62, 1.55, -4.35], [3.05, 3.1, 0.35], limestone);
add([2.62, 1.55, -4.35], [3.05, 3.1, 0.35], limestone);
add([0, 2.76, -4.35], [2.22, 0.7, 0.35], limestone);
add([-4.47, 1.52, 0], [0.24, 3.05, 8.8], stone);
add([4.47, 1.52, 0], [0.24, 3.05, 8.8], stone);
// Solid objects interrupt several shafts on their way to the floor.
add([-1.25, 0.72, -1.0], [0.95, 1.44, 0.95], bronze);
add([1.55, 0.48, 0.95], [1.7, 0.96, 0.82], stone);
add([0.45, 1.02, -2.4], [0.66, 2.04, 0.66], limestone);

const batch: SceneBatch = {
  name: 'fog-colonnade', mesh: cube, capacity: pieces.length, instances: pieces,
  inputMode: 'committed', deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
  material: { definition: material },
};
const scene = { batches: [batch], instances: pieces.length, triangles: pieces.length * cube.indices.length / 3 };

let renderer: Renderer | undefined;
let disposed = false;
let failed = false;
let paused = benchmarkMode;
let trajectory = true;
let fogEnabled = true;
let temporalEnabled = true;
let density = 0.14;
let sunAzimuth = 150;
let sunElevation = 50;
let sunIntensity = 9;
let phase = 0;
let raf = 0;
let lastTime = performance.now();
let lastUi = 0;
const target: Vec3 = [0, 1.34, -0.7];

function cameraEye(): Vec3 {
  const azimuth = 0.32 + (trajectory ? 0.09 * Math.sin(phase * 0.28) : 0);
  const elevation = 0.16 + (trajectory ? 0.022 * Math.sin(phase * 0.19) : 0);
  const distance = 11.3;
  return [target[0] + Math.sin(azimuth) * Math.cos(elevation) * distance,
    target[1] + Math.sin(elevation) * distance,
    target[2] + Math.cos(azimuth) * Math.cos(elevation) * distance];
}

function localMedium(): FogVolume[] {
  return [{ id: 1, shape: { kind: 'box', min: [-4.2, 0, -4.1], max: [4.2, 3.4, 4.1] },
    coefficients: { extinction: density, albedo: [0.93, 0.88, 0.77],
      emission: [0, 0, 0], anisotropy: 0.3 } }];
}

const options: RendererOptions = {
  antiAliasing: 'fxaa',
  sunShadow: { resolution: 2048,
    focusBounds: { min: [-6.2, -0.7, -6.2], max: [6.2, 4.8, 6.2] },
    maxBytes: 16 * 1024 * 1024 },
  volumetricFog: { pixelSize: 8, slices: 64, maxViewDepth: 30, volumes: localMedium(),
    maxBytes: 128 * 1024 * 1024, temporal: { maxHistoryFrames: 16 } },
};

function applySun(): void {
  const az = sunAzimuth * Math.PI / 180;
  const el = sunElevation * Math.PI / 180;
  renderer?.setSun({ direction: [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)],
    color: [sunIntensity, sunIntensity * 0.88, sunIntensity * 0.7] });
  $<HTMLOutputElement>('azimuth-value').textContent = `${sunAzimuth}°`;
  $<HTMLOutputElement>('elevation-value').textContent = `${sunElevation}°`;
  $<HTMLOutputElement>('intensity-value').textContent = sunIntensity.toFixed(1);
}

function updateMedium(): void {
  renderer?.setVolumetricFogMedia({ volumes: localMedium() });
  $<HTMLOutputElement>('density-value').textContent = `${density.toFixed(2)} / m`;
}

function updateUi(): void {
  if (!renderer) return;
  const stats = renderer.stats;
  const fog = renderer.diagnostics().volumetricFog;
  $('grid').textContent = fog?.grid ? `${fog.grid.columns} × ${fog.grid.rows} × ${fog.grid.slices} · ${fog.grid.pixelSize}px` : '—';
  $('froxels').textContent = fog?.grid ? fog.grid.froxels.toLocaleString() : '—';
  $('gpu-time').textContent = stats.gpuMs ? `${stats.gpuMs.volumetricFog.toFixed(2)} ms` : '—';
  $('gpu-frame').textContent = stats.gpuSampleFrame === null ? '—' : String(stats.gpuSampleFrame);
  $('gpu-bytes').textContent = `${(stats.ownedGpuBytes / 1048576).toFixed(1)} MiB`;
  $('viewport-size').textContent = `${stats.width} × ${stats.height}`;
  $('triangles').textContent = scene.triangles.toLocaleString();
  $('frame').textContent = stats.frames.toLocaleString();
  $('status').textContent = paused ? '已暂停' : 'WEBGPU LIVE';
  $('stage-mode').textContent = fogEnabled ? '雾已开启 · 太阳单次散射' : '雾已关闭 · PBR 与阴影保留';
  $<HTMLButtonElement>('pause').textContent = paused ? '继续轨迹' : '暂停轨迹';
  const adapter = renderer.info.adapterInfo;
  $('adapter').textContent = `${adapter.vendor || '未知供应商'} · ${adapter.architecture || '未知架构'}${adapter.isFallbackAdapter ? ' · 软件适配器' : ''}`;
}

function snapshot() {
  return { page: 'volumetric-fog-demo', scene: { ...scene, batches: [batch.name] },
    camera: { eye: cameraEye(), target: [...target], phase },
    controls: { paused, trajectory, fogEnabled, temporalEnabled, density, sunAzimuth, sunElevation, sunIntensity },
    fog: renderer?.diagnostics().volumetricFog ?? null,
    stats: renderer ? structuredClone(renderer.stats) : null,
    adapter: renderer?.info ?? null,
    errors: renderer ? [...renderer.errors] : [],
    disposed };
}

function render(): void {
  if (disposed) throw new Error('FOG_DEMO_DISPOSED');
  if (!renderer) return;
  renderer.render(cameraEye(), target);
  if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
  updateUi();
}

function pause(value: boolean): void {
  if (disposed) throw new Error('FOG_DEMO_DISPOSED');
  paused = value;
  lastTime = performance.now();
  updateUi();
}

function step(dt = 1 / 60): void {
  if (!Number.isFinite(dt) || dt < 0 || dt > 1) throw new RangeError('FOG_DEMO_STEP_DT');
  pause(true);
  if (trajectory) phase += dt;
  render();
}

function fail(error: unknown): void {
  failed = true;
  cancelAnimationFrame(raf);
  $('status').textContent = '渲染错误';
  const panel = $('error');
  panel.hidden = false;
  panel.textContent = String(error);
}

function frame(now: number): void {
  if (disposed || failed) return;
  try {
    const dt = Math.max(0, Math.min(0.1, (now - lastTime) / 1000));
    lastTime = now;
    if (!paused) {
      if (trajectory) phase += dt;
      renderer?.render(cameraEye(), target);
      if (renderer?.errors.length) throw new Error(renderer.errors.join('\n'));
      if (now - lastUi > 160) { updateUi(); lastUi = now; }
    }
    raf = requestAnimationFrame(frame);
  } catch (error) { if (!disposed) fail(error); }
}

function dispose(): void {
  if (disposed) return;
  disposed = true;
  cancelAnimationFrame(raf);
  events.abort();
  renderer?.dispose();
  renderer = undefined;
  $('status').textContent = '已释放';
}

window.__FOG_DEMO__ = { pause, render, step, snapshot, dispose, resetHistory: () => renderer?.invalidateHistory() };

async function main(): Promise<void> {
  try {
    renderer = await Renderer.create(canvas, scene.batches, options);
    if (disposed) { renderer.dispose(); renderer = undefined; return; }
    renderer.setExposure(-1);
    applySun();
    updateMedium();
    renderer.setVolumetricFogEnabled(fogEnabled);
    const signal = events.signal;
    $<HTMLInputElement>('temporal-enabled').addEventListener('change', event => {
      try { temporalEnabled = (event.target as HTMLInputElement).checked;
        renderer?.setVolumetricFogTemporalEnabled(temporalEnabled); render(); } catch (error) { fail(error); }
    }, { signal });
    $<HTMLInputElement>('fog-enabled').addEventListener('change', event => {
      try { fogEnabled = (event.target as HTMLInputElement).checked;
        renderer?.setVolumetricFogEnabled(fogEnabled); render(); } catch (error) { fail(error); }
    }, { signal });
    $<HTMLInputElement>('density').addEventListener('input', event => {
      try { density = Number((event.target as HTMLInputElement).value);
        updateMedium(); if (paused) render(); } catch (error) { fail(error); }
    }, { signal });
    for (const [id, setter] of [
      ['azimuth', (value: number) => { sunAzimuth = value; }],
      ['elevation', (value: number) => { sunElevation = value; }],
      ['intensity', (value: number) => { sunIntensity = value; }],
    ] as const) {
      $<HTMLInputElement>(id).addEventListener('input', event => {
        try { setter(Number((event.target as HTMLInputElement).value)); applySun();
          if (paused) render(); } catch (error) { fail(error); }
      }, { signal });
    }
    $<HTMLInputElement>('trajectory').addEventListener('change', event => {
      trajectory = (event.target as HTMLInputElement).checked;
      if (paused) render();
    }, { signal });
    $<HTMLButtonElement>('pause').addEventListener('click', () => pause(!paused), { signal });
    $<HTMLButtonElement>('step').addEventListener('click', () => {
      try { step(); } catch (error) { fail(error); }
    }, { signal });
    window.addEventListener('resize', () => {
      if (paused && !disposed) { try { render(); } catch (error) { fail(error); } }
    }, { signal });
    render();
    if (benchmarkMode) { render(); render(); }
    else raf = requestAnimationFrame(frame);
  } catch (error) { if (!disposed) fail(error); }
}

window.addEventListener('pagehide', dispose, { once: true });
void main();
