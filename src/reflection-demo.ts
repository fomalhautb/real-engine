import './reflection-demo.css';
import { Renderer, type RendererOptions } from './renderer';
import type { InstanceState, SceneBatch } from './contracts';
import { createCube } from './geometry';
import { DEFAULT_MATERIAL_DEFINITION, m } from './material';
import { multiply, normalMatrix, rotationY, scaling, translation, type Mat4 } from './math';

type Vec3 = [number, number, number];
type Color = [number, number, number, number];

interface ReflectionDemoHarness {
  pause(value: boolean): void;
  render(): void;
  step(dt?: number): void;
  snapshot(): ReturnType<typeof snapshot>;
  dispose(): void;
}
declare global { interface Window { __REFLECTION_DEMO__?: ReflectionDemoHarness } }

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (!found) throw new Error(`REFLECTION_DEMO_ELEMENT_MISSING:${id}`);
  return found as T;
};

const canvas = $<HTMLCanvasElement>('viewport');
const benchmarkMode = new URLSearchParams(location.search).get('benchmark') === '1';
const events = new AbortController();
const cube = createCube();
const eye: Vec3 = [0, 2, 6];
const target: Vec3 = [0, 1.6, 0];
let nextId = 1;

function instance(model: Mat4, color: Color, metalness: number, roughness: number, entityId = nextId++): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color, metalness, roughness, historyVersion: 0 };
}
function block(position: Vec3, size: Vec3, color: Color, metalness = 0, roughness = 0.78,
  entityId?: number): InstanceState {
  return instance(multiply(translation(...position), scaling(...size)), color, metalness, roughness, entityId);
}
function batch(name: string, instances: InstanceState[], material = DEFAULT_MATERIAL_DEFINITION): SceneBatch {
  return { name, mesh: cube, capacity: instances.length, instances, inputMode: 'committed',
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: { definition: material } };
}

const mirrorColor: Color = [0.84, 0.9, 0.94, 1];
let roughness = 0.045;
const mirror = batch('gallery-mirror', [block([0, 1.77, 0], [3.76, 3.18, 0.08], mirrorColor, 1, roughness)]);
const stone: Color = [0.47, 0.54, 0.58, 1];
const dark: Color = [0.13, 0.18, 0.22, 1];
const bronze: Color = [0.56, 0.37, 0.23, 1];
const pieces: InstanceState[] = [
  block([0, -0.16, 3.5], [10, 0.3, 17], stone),
  block([0, 1.75, -0.62], [8.5, 3.7, 0.26], dark),
  block([-1.98, 1.77, 0.12], [0.18, 3.55, 0.23], bronze, 0.62, 0.33),
  block([1.98, 1.77, 0.12], [0.18, 3.55, 0.23], bronze, 0.62, 0.33),
  block([0, 3.52, 0.12], [4.14, 0.18, 0.23], bronze, 0.62, 0.33),
  block([0, 0.02, 0.12], [4.14, 0.18, 0.23], bronze, 0.62, 0.33),
  block([-3.18, 1.45, 1.35], [0.52, 2.9, 0.52], stone),
  block([3.18, 1.45, 1.35], [0.52, 2.9, 0.52], stone),
  block([-3.18, 2.96, 1.35], [0.82, 0.17, 0.82], bronze),
  block([3.18, 2.96, 1.35], [0.82, 0.17, 0.82], bronze),
  block([-3.18, 0.17, 1.35], [0.82, 0.34, 0.82], dark),
  block([3.18, 0.17, 1.35], [0.82, 0.34, 0.82], dark),
  block([0, 0.04, 4.1], [4.8, 0.08, 2.5], dark),
];
const architecture = batch('gallery-architecture', pieces);

// These solid objects are behind the camera. Their color appears in the mirror
// through the same geometry queries used by the renderer, not a screen decal.
const luminous = { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const,
  doubleSided: true, baseColor: m.instanceColor };
const emitters = batch('gallery-rear-objects', [
  block([-1.15, 1.5, 8.5], [0.72, 2.5, 0.72], [0.16, 0.59, 1, 1]),
  block([1.15, 1.5, 8.5], [0.72, 2.5, 0.72], [1, 0.37, 0.15, 1]),
  block([0, 2.78, 9.9], [2.9, 0.17, 0.42], [0.89, 0.81, 0.55, 1]),
], luminous);
const movingId = nextId++;
function movingObject(time: number): InstanceState {
  const model = multiply(translation(0.76 * Math.sin(time * 0.82), 1.28 + 0.19 * Math.sin(time * 1.3), 9.25),
    multiply(rotationY(time * 0.62), scaling(0.74, 0.74, 0.74)));
  return instance(model, [1, 0.72, 0.28, 1], 0, 0.5, movingId);
}
let phase = 0;
const moving = batch('gallery-moving-object', [movingObject(phase)], luminous);
const scene = [mirror, architecture, emitters, moving];

const options: RendererOptions = {
  antiAliasing: 'fxaa',
  sunShadow: { resolution: 1024, focusBounds: { min: [-6, -1, -2], max: [6, 5, 12] }, maxBytes: 8 * 1024 * 1024 },
  worldQueries: { maxRays: 65_536, maxTotalWorkUnits: 16_777_216 },
  reflections: { pixelSpacing: 4, maxAnchors: 16_384, samplesPerAnchor: 4,
    maxDistance: 20, maxBytes: 32 * 1024 * 1024, skyRadiance: [0.025, 0.04, 0.07], fogSteps: 24 },
  volumetricFog: { pixelSize: 8, slices: 32, maxViewDepth: 25, maxFroxels: 300_000,
    maxBytes: 32 * 1024 * 1024, global: { coefficients: {
      extinction: 0.015, albedo: [0.67, 0.77, 0.86], emission: [0, 0, 0], anisotropy: 0.12 } } },
};

let renderer: Renderer | undefined;
let disposed = false;
let failed = false;
let paused = benchmarkMode;
let reflectionsEnabled = true;
let fogEnabled = true;
let raf = 0;
let lastTime = performance.now();
let lastUi = 0;

function updateUi(): void {
  if (!renderer) return;
  const stats = renderer.stats;
  const reflections = renderer.diagnostics().reflections;
  $('reflection-grid').textContent = reflections?.grid
    ? `${reflections.grid.columns} × ${reflections.grid.rows} · ${reflections.grid.spacing}px` : '—';
  $('reflection-rays').textContent = reflections?.grid
    ? `${reflections.grid.anchorCount.toLocaleString()} / ${reflections.grid.rayCount.toLocaleString()}` : '—';
  $('reflection-bytes').textContent = reflections
    ? `${(reflections.ownedGpuBytes / 1048576).toFixed(1)} MiB` : '—';
  $('gpu-time').textContent = stats.gpuMs ? `${stats.gpuMs.reflections.toFixed(2)} ms` : '—';
  $('gpu-frame').textContent = stats.gpuSampleFrame === null ? '—' : String(stats.gpuSampleFrame);
  $('gpu-bytes').textContent = `${(stats.ownedGpuBytes / 1048576).toFixed(1)} MiB`;
  $('viewport-size').textContent = `${stats.width} × ${stats.height}`;
  $('frame').textContent = stats.frames.toLocaleString();
  $('status').textContent = paused ? '已暂停' : 'WEBGPU LIVE';
  $('stage-mode').textContent = reflectionsEnabled ? '单段世界反射 · 开启' : '单段世界反射 · 关闭';
  $<HTMLButtonElement>('pause').textContent = paused ? '继续物体' : '暂停物体';
  const adapter = renderer.info.adapterInfo;
  $('adapter').textContent = `${adapter.vendor || '未知供应商'} · ${adapter.architecture || '未知架构'}${adapter.isFallbackAdapter ? ' · 软件适配器' : ''}`;
}

function snapshot() {
  return { page: 'reflection-demo', camera: { eye: [...eye], target: [...target] },
    controls: { paused, reflectionsEnabled, roughness, fogEnabled, phase },
    scene: { batches: scene.map(item => item.name), instances: scene.reduce((sum, item) => sum + item.instances.length, 0) },
    reflections: renderer?.diagnostics().reflections ?? null,
    stats: renderer ? structuredClone(renderer.stats) : null,
    adapter: renderer?.info ?? null,
    errors: renderer ? [...renderer.errors] : [], disposed };
}

function render(): void {
  if (disposed) throw new Error('REFLECTION_DEMO_DISPOSED');
  if (!renderer) return;
  renderer.render(eye, target);
  if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
  updateUi();
}
function pause(value: boolean): void {
  if (disposed) throw new Error('REFLECTION_DEMO_DISPOSED');
  paused = value;
  lastTime = performance.now();
  updateUi();
}
function advance(dt: number): void {
  phase += dt;
  moving.instances[0] = movingObject(phase);
}
function step(dt = 1 / 60): void {
  if (!Number.isFinite(dt) || dt < 0 || dt > 1) throw new RangeError('REFLECTION_DEMO_STEP_DT');
  pause(true);
  advance(dt);
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
      advance(dt);
      renderer?.render(eye, target);
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

window.__REFLECTION_DEMO__ = { pause, render, step, snapshot, dispose };

async function main(): Promise<void> {
  try {
    renderer = await Renderer.create(canvas, scene, options);
    if (disposed) { renderer.dispose(); renderer = undefined; return; }
    renderer.setExposure(-0.7);
    renderer.setSun({ direction: [0.42, 0.81, 0.39], color: [5.4, 5.1, 4.7] });
    const signal = events.signal;
    $<HTMLInputElement>('reflections-enabled').addEventListener('change', event => {
      try { reflectionsEnabled = (event.target as HTMLInputElement).checked;
        renderer?.setReflectionsEnabled(reflectionsEnabled); render(); } catch (error) { fail(error); }
    }, { signal });
    $<HTMLInputElement>('roughness').addEventListener('input', event => {
      try { roughness = Number((event.target as HTMLInputElement).value);
        mirror.instances[0] = block([0, 1.77, 0], [3.76, 3.18, 0.08], mirrorColor, 1, roughness,
          mirror.instances[0].entityId);
        $<HTMLOutputElement>('roughness-value').textContent = roughness.toFixed(3);
        if (paused) render(); } catch (error) { fail(error); }
    }, { signal });
    $<HTMLInputElement>('fog-enabled').addEventListener('change', event => {
      try { fogEnabled = (event.target as HTMLInputElement).checked;
        renderer?.setVolumetricFogEnabled(fogEnabled); render(); } catch (error) { fail(error); }
    }, { signal });
    $<HTMLButtonElement>('pause').addEventListener('click', () => pause(!paused), { signal });
    $<HTMLButtonElement>('step').addEventListener('click', () => { try { step(); } catch (error) { fail(error); } }, { signal });
    window.addEventListener('resize', () => { if (paused && !disposed) { try { render(); } catch (error) { fail(error); } } }, { signal });
    render();
    if (benchmarkMode) { render(); render(); }
    else raf = requestAnimationFrame(frame);
  } catch (error) { if (!disposed) fail(error); }
}

window.addEventListener('pagehide', dispose, { once: true });
void main();
