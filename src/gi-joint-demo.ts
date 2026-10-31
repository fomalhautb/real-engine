import './gi-demo.css';
import { createJointLeakScene, createJointRoomsScene, JOINT_CAMERAS, JOINT_DOOR_CLOSED, JOINT_DOOR_OPEN, JOINT_FOREGROUND,
  JOINT_LEAK_GRID, JOINT_MOVER_PARKED_Z, JOINT_PANEL_END, JOINT_PANEL_OFF, JOINT_PANEL_ON, JOINT_PANEL_START,
  JOINT_ROOM_GRID, JOINT_SUN, jointMover, jointPanel, type JointCameraName, type JointRoomsScene } from './gi-joint-scene';
import { Renderer } from './renderer';

type Vec3 = [number, number, number];
type SceneName = 'rooms' | 'leak';
const CAMERAS: Record<SceneName, JointCameraName[]> = { rooms: ['roomA', 'roomB', 'roomBAway'], leak: ['blackBox', 'thinBox', 'slitBox'] };

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Joint demo element missing: ${id}`);
  return found as T;
}
const canvas = element<HTMLCanvasElement>('viewport');
let renderer: Renderer | undefined;
let rooms: JointRoomsScene | undefined;
let sceneName: SceneName = 'rooms';
let cameraName: JointCameraName = 'roomA';
let sunOn = true, panelOn = true, doorOpen = true, foreground = true;
let panelT = 0, panelTarget = 0, moverZ: number | null = null;
let raf = 0, lastUi = 0, generation = 0;

async function load(name: SceneName): Promise<void> {
  const token = ++generation;
  cancelAnimationFrame(raf);
  renderer?.dispose(); renderer = undefined;
  sceneName = name; cameraName = CAMERAS[name][0];
  sunOn = panelOn = doorOpen = foreground = true; panelT = panelTarget = 0; moverZ = null;
  element('status').textContent = 'LOADING';
  try {
    const scheduled = element<HTMLInputElement>('scheduled').checked ? { mode: 'scheduled' as const } : undefined;
    rooms = name === 'rooms' ? createJointRoomsScene() : undefined;
    const batches = rooms?.batches ?? createJointLeakScene().batches;
    const grid = name === 'rooms' ? JOINT_ROOM_GRID : JOINT_LEAK_GRID;
    const rays = grid.counts.reduce((a, b) => a * b, 1) * (grid.raysPerProbe ?? 64);
    const created = await Renderer.create(canvas, batches, { worldQueries: { maxRays: Math.max(8192, rays) },
      diffuseGi: { ...grid, update: scheduled }, antiAliasing: 'fxaa' });
    if (token !== generation) { created.dispose(); return; }
    renderer = created;
    renderer.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
    renderer.setDiffuseGiEnabled(element<HTMLInputElement>('gi-enabled').checked);
    const info = renderer.info.adapterInfo;
    element('adapter').textContent = `${info.vendor} / ${info.architecture}${info.isFallbackAdapter ? ' (fallback)' : ''}`;
    element('status').textContent = 'LIVE';
    buildCameraButtons();
    raf = requestAnimationFrame(frame);
  } catch (error) {
    const panel = element('error');
    panel.hidden = false;
    panel.textContent = error instanceof Error ? error.message : String(error);
    element('status').textContent = 'ERROR';
  }
}

function buildCameraButtons(): void {
  const group = element('cameras');
  group.replaceChildren(...CAMERAS[sceneName].map(name => {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = name;
    button.setAttribute('aria-pressed', String(name === cameraName));
    button.addEventListener('click', () => {
      cameraName = name;
      for (const other of group.querySelectorAll('button')) other.setAttribute('aria-pressed', String(other === button));
    });
    return button;
  }));
  for (const id of ['sun', 'panel', 'door', 'panel-move', 'foreground', 'mover']) {
    element<HTMLButtonElement>(id).disabled = sceneName !== 'rooms' && id !== 'sun';
  }
}

function frame(now: number): void {
  const r = renderer;
  if (!r) return;
  try {
    if (rooms && panelT !== panelTarget) {
      panelT = Math.max(0, Math.min(1, panelT + Math.sign(panelTarget - panelT) / 12));
      rooms.panel.instances = [jointPanel(JOINT_PANEL_START.map((value, axis) =>
        value + (JOINT_PANEL_END[axis] - value) * panelT) as Vec3)];
    }
    if (rooms && moverZ !== null) {
      moverZ += 4.1 / 30;
      if (moverZ > 2) { moverZ = null; rooms.mover.instances = [jointMover(JOINT_MOVER_PARKED_Z)]; }
      else rooms.mover.instances = [jointMover(moverZ)];
    }
    const view = JOINT_CAMERAS[cameraName];
    r.render(view.eye as Vec3, view.target as Vec3);
    if (now - lastUi > 500) { lastUi = now; refreshUi(r); }
    raf = requestAnimationFrame(frame);
  } catch (error) {
    element('status').textContent = 'ERROR';
    const panel = element('error');
    panel.hidden = false;
    panel.textContent = error instanceof Error ? error.message : String(error);
  }
}

function refreshUi(r: Renderer): void {
  const gi = r.diagnostics().diffuseGi;
  if (!gi) return;
  const schedule = gi.schedule;
  element('stage-mode').textContent = `${gi.enabled ? 'GI ON' : 'GI OFF'} · ${gi.update}`;
  element('captured').textContent = `${schedule.capturedProbesLastUpdate} / ${schedule.probeCount}`;
  element('max-age').textContent = `${schedule.maxAgeUpdates} ≤ ${schedule.maxStalenessUpdates - 1}`;
  element('pending').textContent = `${schedule.pendingDirtyProbes} / ${schedule.unrefreshedDirtyProbes}`;
  element('queries').textContent = String(gi.queriesPerUpdate);
  element('gpu-gi-ms').textContent = r.stats.gpuMs ? `${r.stats.gpuMs.diffuseGi.toFixed(3)} ms` : 'timestamp-query 不可用';
  element('frame').textContent = String(r.stats.frames);
}

element('sun').addEventListener('click', () => {
  sunOn = !sunOn;
  renderer?.setSun({ direction: [...JOINT_SUN.direction], color: sunOn ? [...JOINT_SUN.color] : [0, 0, 0] });
});
element('panel').addEventListener('click', () => {
  panelOn = !panelOn;
  renderer?.setMaterialParameters('joint-panel', new Float32Array(panelOn ? JOINT_PANEL_ON : JOINT_PANEL_OFF));
});
element('door').addEventListener('click', () => {
  if (!rooms) return;
  doorOpen = !doorOpen;
  rooms.door.instances = [doorOpen ? JOINT_DOOR_OPEN : JOINT_DOOR_CLOSED];
});
element('panel-move').addEventListener('click', () => { panelTarget = panelTarget ? 0 : 1; });
element('foreground').addEventListener('click', () => {
  if (!rooms) return;
  foreground = !foreground;
  rooms.foreground.instances = foreground ? [JOINT_FOREGROUND] : [];
});
element('mover').addEventListener('click', () => { if (rooms && moverZ === null) moverZ = JOINT_MOVER_PARKED_Z; });
element<HTMLInputElement>('scheduled').addEventListener('change', event => {
  renderer?.setDiffuseGiUpdate((event.target as HTMLInputElement).checked ? { mode: 'scheduled' } : undefined);
});
element<HTMLInputElement>('gi-enabled').addEventListener('change', event => {
  renderer?.setDiffuseGiEnabled((event.target as HTMLInputElement).checked);
});
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-scene]')) {
  button.addEventListener('click', () => { void load(button.dataset.scene as SceneName); });
}
window.addEventListener('pagehide', () => { cancelAnimationFrame(raf); renderer?.dispose(); renderer = undefined; });
void load('rooms');
