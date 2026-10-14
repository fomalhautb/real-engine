import './style.css';
import { DemoScene } from './scene';
import { Renderer } from './renderer';
import type { RayHit } from './physics';
import { loadGltf } from './gltf-load';

const element = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error('Missing UI element: ' + id);
  return node as T;
};
const canvas = element<HTMLCanvasElement>('viewport');
let scene: DemoScene | undefined, renderer: Renderer | undefined, raf = 0, disposed = false;
let yaw = 0.34, elevation = 0.40, radius = 14, lastTime = performance.now(), lastUI = 0;
const samples: number[] = [];
const controls: AbortController = new AbortController();
function camera(): [number, number, number] {
  return [Math.sin(yaw) * Math.cos(elevation) * radius, 1.2 + Math.sin(elevation) * radius, Math.cos(yaw) * Math.cos(elevation) * radius];
}
function fail(error: unknown): void {
  element('error').hidden = false;
  element('error').textContent = String(error);
  element('status').textContent = 'RUNTIME ERROR';
  cancelAnimationFrame(raf);
}
function snapshot() {
  return {
    stage: '1.2', time: scene?.time ?? 0, paused: scene?.paused ?? false,
    stats: renderer ? structuredClone(renderer.stats) : null, environment: renderer?.info ?? null,
    errors: [...renderer?.errors ?? []],
    renderGraph: renderer?.diagnostics() ?? null,
    sceneChanges: scene ? { revision: scene.committedRevision, lastFrame: scene.changeCounts } : null,
    physics: { bodies: scene?.physics.bodyCount ?? 0, active: scene?.physics.activeCount ?? 0, droppedSeconds: scene?.droppedSeconds ?? 0 },
    cpuSubmit: samples.length ? { samples: samples.length, p50: percentile(samples, 0.5), p95: percentile(samples, 0.95) } : null,
    scope: { virtualGeometry: false, worldSpaceGI: false, reflections: false, fullDeformationCompiler: false },
  };
}
function percentile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}
function render(): void { scene!.commitFrame(); renderer!.render(camera(), [0, 1.2, -1.5]); }
function frame(now: number): void {
  if (disposed) return;
  try {
    scene!.advance(Math.max(0, (now - lastTime) / 1000)); lastTime = now;
    render();
    if (renderer!.errors.length) throw new Error(renderer!.errors.join('\n'));
    samples.push(renderer!.stats.cpuSubmitMs); if (samples.length > 1800) samples.shift();
    if (now - lastUI > 250) { updateUI(); lastUI = now; }
    raf = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}
function updateUI(): void {
  const stats = renderer!.stats, gpu = stats.gpuMs;
  element('instances').textContent = String(scene!.batches.reduce((sum, batch) => sum + batch.instances.length, 0));
  element('vertices').textContent = stats.deformedVertices.toLocaleString();
  element('cpu').textContent = stats.cpuSubmitMs.toFixed(2) + ' ms';
  element('gpu').textContent = gpu ? (gpu.deformation + gpu.shadow + gpu.surface).toFixed(2) + ' ms' : '不可用 / 等待采样';
  element('memory').textContent = (stats.ownedGpuBytes / 1048576).toFixed(1) + ' MiB';
  element('physics').textContent = scene!.physics.activeCount + '/' + scene!.physics.bodyCount + ' active · 60 Hz';
  element('status').innerHTML = '<i></i>' + (scene!.paused ? 'PAUSED' : 'WEBGPU LIVE');
}
async function verify() {
  const button = element<HTMLButtonElement>('verify');
  button.disabled = true;
  try {
    const result = await renderer!.verifyDeformation();
    const passed = result.vertices > 0 && result.maxError < 0.0002 && result.maxNormalError < 0.0002 && result.boundsViolations === 0;
    element('verification').textContent = (passed ? 'PASS' : 'FAIL') + ' · ' + result.vertices + ' vertices · max error ' + result.maxError.toExponential(2);
    return { ...result, passed };
  } finally { button.disabled = false; }
}
function dispose(): void {
  if (disposed) return;
  disposed = true; cancelAnimationFrame(raf); controls.abort(); renderer?.dispose(); scene?.dispose();
}
const signal = controls.signal;
async function start(): Promise<void> {
  const queryCount = Number(new URLSearchParams(location.search).get('trees') ?? 32);
  const count = Number.isInteger(queryCount) ? Math.min(256, Math.max(1, queryCount)) : 32;
  const assets = await Promise.all([
    loadGltf('/assets/gltf/SimpleSkin/glTF/SimpleSkin.gltf'),
    loadGltf('/assets/gltf/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf'),
  ]);
  scene = await DemoScene.create(count, assets);
  renderer = await Renderer.create(canvas, scene.renderBatches);
  element('adapter').textContent = renderer.info.adapter + ' · ' + renderer.info.timing;
  element('pause').addEventListener('click', () => { scene!.paused = !scene!.paused; }, { signal });
  element('reset').addEventListener('click', () => { scene!.reset(); samples.length = 0; }, { signal });
  element('spawn').addEventListener('click', () => scene!.spawnBox(), { signal });
  element('remove').addEventListener('click', () => scene!.removeBox(), { signal });
  element('verify').addEventListener('click', () => void verify().catch(fail), { signal });
  element('export').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(snapshot(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = 'real-engine-diagnostics.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, { signal });
  let dragging = false, px = 0, py = 0;
  canvas.addEventListener('pointerdown', event => { dragging = true; px = event.clientX; py = event.clientY; canvas.setPointerCapture(event.pointerId); }, { signal });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    yaw -= (event.clientX - px) * 0.006; elevation = Math.min(1.25, Math.max(0.1, elevation + (event.clientY - py) * 0.005));
    px = event.clientX; py = event.clientY;
  }, { signal });
  canvas.addEventListener('pointerup', () => dragging = false, { signal });
  canvas.addEventListener('pointercancel', () => dragging = false, { signal });
  canvas.addEventListener('wheel', event => { event.preventDefault(); radius = Math.max(5, Math.min(45, radius * Math.exp(event.deltaY * 0.001))); }, { signal, passive: false });
  document.addEventListener('visibilitychange', () => { lastTime = performance.now(); }, { signal });
  window.addEventListener('pagehide', dispose, { once: true, signal });
  window.__REAL_ENGINE__ = {
    snapshot,
    pause(value: boolean) { scene!.paused = value; },
    step(dt: number) { const paused = scene!.paused; scene!.paused = false; scene!.advance(dt); scene!.paused = paused; render(); },
    reset() { scene!.reset(); render(); },
    spawn() { return scene!.spawnBox(); },
    remove() { return scene!.removeBox(); },
    raycast(origin, direction, maxDistance, excludeEntityId) { return scene!.physics.raycast(origin, direction, maxDistance, excludeEntityId); },
    verify,
    dispose,
  };
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}
export interface EngineHarness {
  snapshot: typeof snapshot;
  pause(value: boolean): void;
  step(dt: number): void;
  reset(): void;
  spawn(): boolean;
  remove(): boolean;
  raycast(origin: [number, number, number], direction: [number, number, number], maxDistance: number, excludeEntityId?: number): RayHit | null;
  verify: typeof verify;
  dispose(): void;
}
declare global { interface Window { __REAL_ENGINE__?: EngineHarness } }
void start().catch(error => { fail(error); dispose(); });
if (import.meta.hot) import.meta.hot.dispose(dispose);
