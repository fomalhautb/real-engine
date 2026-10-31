/**
 * Large-scene walkthrough page (large-scene.html?set=/artifacts/.../asset-set.json).
 * Loads an offline-built asset set, renders its opaque paged geometry with
 * streamed textures, and offers fixed camera keyframes, a continuous walk
 * along the set's camera path (or a bounds orbit when it has none) and the
 * renderer's material / clusters / levels views. Optional query parameters:
 * frames=N (walk N frames, then stop and report through window.__LARGE_SCENE__;
 * run(frames, walk) renders more frames after a view or keyframe change),
 * exposure=EV, sun=x,y,z. Frame intervals shown are browser rAF diagnostics.
 */
import { cameraPathKeyframes, parseCameraPath, sampleCameraPath, type CameraPath, type CameraPose, type Vec3 } from './camera-path';
import { loadAssetSetRoots, loadGltfPackageScene, type GltfPackageScene } from './gltf-package-scene';
import { Renderer } from './renderer';

type DebugView = 'material' | 'clusters' | 'levels';
type Status = 'idle' | 'loading' | 'ready' | 'done' | 'error' | 'disposed';
const MiB = 1024 ** 2;
const params = new URLSearchParams(location.search);
const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = element<HTMLCanvasElement>('viewport');
const controls = new AbortController();
let renderer: Renderer | undefined, scene: GltfPackageScene | undefined, path: CameraPath | undefined;
let status: Status = 'idle', failure: string | null = null, disposed = false, raf = 0, walking = false;
let cameraTime = 0, lastTime = 0, lastUi = 0, interval = 0, debug: DebugView = 'material', maxErrorPixels = 2;
let autorunFrames = 0, renderedFrames = 0, exposureEV = 3;

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

function refreshUi(): void {
  if (!renderer || !scene) return;
  const stats = renderer.stats;
  element('frame').textContent = String(stats.frames);
  element('interval').textContent = interval ? `${interval.toFixed(1)} ms` : '—';
  element('memory').textContent = `${(stats.ownedGpuBytes / MiB).toFixed(1)} MiB`;
  element('time').textContent = `${cameraTime.toFixed(2)} / ${path!.duration.toFixed(1)} s`;
}

function renderOnce(): void {
  if (!renderer) return;
  const camera = pose();
  renderer.render(camera.eye, camera.target);
  if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
  renderedFrames++;
}

function frame(now: number): void {
  if (disposed) return;
  try {
    const dt = Math.min(0.1, Math.max(0, (now - lastTime) / 1000));
    interval = interval ? interval * 0.9 + (now - lastTime) * 0.1 : now - lastTime;
    lastTime = now;
    if (walking) cameraTime = (cameraTime + (autorunFrames ? 1 / 60 : dt)) % path!.duration;
    renderOnce();
    if (now - lastUi > 250) { refreshUi(); lastUi = now; }
    if (autorunFrames && renderedFrames >= autorunFrames) { refreshUi(); setStatus('done', `DONE · ${renderedFrames} FRAMES`); return; }
    raf = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

/** Renders `frames` more frames (walking or not), then stops with status 'done' again. */
function run(frames: number, walk: boolean): void {
  if (!renderer || disposed || status === 'error') throw new Error('LARGE_SCENE_NOT_READY');
  if (!Number.isSafeInteger(frames) || frames < 1 || frames > 100_000) throw new Error('LARGE_SCENE_FRAMES');
  cancelAnimationFrame(raf);
  autorunFrames = renderedFrames + frames;
  setWalk(walk);
  setStatus('ready', `RUNNING ${frames} FRAMES`);
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
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

function dispose(): void {
  if (disposed) return;
  disposed = true;
  cancelAnimationFrame(raf);
  controls.abort();
  renderer?.dispose();
  setStatus('disposed', 'DISPOSED');
}

function snapshot() {
  return { status, failure, assetSet: params.get('set'), renderedFrames, cameraTime, walking, debug, maxErrorPixels,
    info: renderer?.info ?? null, stats: renderer ? { ...renderer.stats } : null,
    scene: scene ? { batches: scene.batches.length, textures: scene.sources.length, sourceInstances: scene.sourceInstances,
      pagedTriangles: scene.pagedTriangles, omittedTriangles: scene.omittedTriangles, omitted: scene.omitted.length,
      conversions: scene.conversions, unsupportedMaterialInputs: scene.unsupportedMaterialInputs.length, bounds: scene.bounds } : null,
    cameraPath: path ? { name: path.name, samples: path.samples.length, duration: path.duration } : null };
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
  window.addEventListener('keydown', event => {
    if (event.key.toLowerCase() === 'h' && !event.altKey && !event.ctrlKey && !event.metaKey) document.body.classList.toggle('ui-hidden');
  }, { signal });
  window.addEventListener('pagehide', dispose, { once: true, signal });
}

async function start(): Promise<void> {
  wireControls();
  window.__LARGE_SCENE__ = { snapshot, showKeyframe, setWalk, setDebug, run, dispose, get renderer() { return renderer; } };
  const set = params.get('set');
  if (!set) { setStatus('idle', 'NO ASSET SET · add ?set=/artifacts/<name>/asset-set.json'); return; }
  const frames = params.get('frames');
  if (frames !== null) {
    autorunFrames = Number(frames);
    if (!Number.isSafeInteger(autorunFrames) || autorunFrames < 1 || autorunFrames > 100_000) throw new Error('LARGE_SCENE_FRAMES');
  }
  setStatus('loading', 'LOADING ASSET SET');
  const roots = await loadAssetSetRoots(set, controls.signal);
  scene = await loadGltfPackageScene(roots, { name: set.split('/').slice(-2, -1)[0] || 'scene', anisotropy: 8, signal: controls.signal,
    progress: (phase, data) => setStatus('loading', `${phase.toUpperCase()} ${data ? JSON.stringify(data) : ''}`) });
  if (disposed) return;
  path = roots.cameraPathUrl
    ? parseCameraPath(await (await fetch(roots.cameraPathUrl, { signal: controls.signal })).json())
    : orbitPath(scene.bounds);
  setStatus('loading', 'CREATING RENDERER');
  const created = await Renderer.create(canvas, scene.batches, { maxBatches: Math.max(256, scene.batches.length), antiAliasing: 'fxaa',
    textureBudgetBytes: 1024 * MiB,
    textureStreaming: { sources: scene.sources, maxCpuBytes: 256 * MiB, maxInFlightBytes: 128 * MiB, maxUploadBytesPerFrame: 2 * MiB },
    sunShadow: { resolution: 2048, focusBounds: scene.bounds } });
  if (disposed) { created.dispose(); return; }
  renderer = created;
  renderer.setSun({ direction: vec3Param('sun', [0.35, 0.8, 0.4]), color: [4, 3.8, 3.5] });
  exposureEV = Number(params.get('exposure') ?? 3);
  if (!Number.isFinite(exposureEV)) throw new Error('LARGE_SCENE_PARAM: exposure');
  applyLod();
  element('adapter').textContent = renderer.info.adapter;
  element('batches').textContent = `${scene.batches.length} · textures ${scene.sources.length}`;
  element('triangles').textContent = scene.pagedTriangles.toLocaleString();
  element('omitted').textContent = `${scene.omitted.length} primitives · ${scene.omittedTriangles.toLocaleString()} tris`;
  const keyframes = element('keyframes');
  for (const [n, index] of cameraPathKeyframes(path).entries()) {
    const button = document.createElement('button');
    button.textContent = `V${n + 1}`;
    button.title = `t = ${(path.samples[index].time - path.samples[0].time).toFixed(1)} s`;
    button.addEventListener('click', () => showKeyframe(index), { signal: controls.signal });
    keyframes.append(button);
  }
  setWalk(autorunFrames > 0 || params.get('walk') === '1');
  setStatus('ready', `READY · ${path.name}`);
  lastTime = performance.now();
  raf = requestAnimationFrame(frame);
}

export interface LargeSceneHarness {
  snapshot: typeof snapshot;
  showKeyframe(index: number): void;
  setWalk(value: boolean): void;
  setDebug(value: DebugView): void;
  run(frames: number, walk: boolean): void;
  dispose(): void;
  readonly renderer: Renderer | undefined;
}
declare global { interface Window { __LARGE_SCENE__?: LargeSceneHarness } }

void start().catch(error => { fail(error); renderer?.dispose(); renderer = undefined; });
