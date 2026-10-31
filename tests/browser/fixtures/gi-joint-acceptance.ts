import type { DiffuseGiOptions } from '../../../src/diffuse-gi';
import type { DiffuseGiUpdateOptions } from '../../../src/gi-probe-schedule';
import { createJointLeakScene, createJointRoomsScene, JOINT_CAMERAS, JOINT_DOOR_CLOSED, JOINT_DOOR_OPEN,
  JOINT_LEAK_GRID, JOINT_PANEL_END, JOINT_PANEL_OFF, JOINT_PANEL_ON, JOINT_PANEL_START, JOINT_ROIS,
  JOINT_ROOM_GRID, JOINT_SKY, JOINT_SUN, jointMover, jointPanel, JOINT_MOVER_PARKED_Z,
  type JointCameraName, type JointRoi } from '../../../src/gi-joint-scene';
import { luminance } from '../../../src/gi-acceptance';
import { lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldHit, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
type Pixel = { x: number; y: number };

export interface JointFrames {
  warmup: number; stability: number; baseline: number; response: number; moverFrames: number; panelMoveFrames: number;
}
export interface JointRunInput {
  update?: DiffuseGiUpdateOptions;
  width: number; height: number;
  raysPerProbe?: number;
  emissiveSamplesPerProbe?: number;
  frames: JointFrames;
  roiGrid: number;
  screenshots: boolean;
  /** false records the direct-only response (GI disabled for the whole run). */
  gi?: boolean;
}

const MOVER_END_Z = 2.0;

function sub(a: readonly number[], b: readonly number[]): Vec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add(a: readonly number[], b: readonly number[], scale = 1): Vec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}
function cross(a: readonly number[], b: readonly number[]): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a: readonly number[], b: readonly number[]): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function normalize(v: readonly number[]): Vec3 { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; }

/** Same projection and camera rays as Renderer.render. */
function camera(name: JointCameraName, width: number, height: number) {
  const { eye, target } = JOINT_CAMERAS[name];
  const viewProjection = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
  const forward = normalize(sub(target, eye)), right = normalize(cross(forward, [0, 1, 0])), up = cross(right, forward);
  const focal = Math.tan(Math.PI / 6.4);
  return {
    eye: eye as Vec3, target: target as Vec3,
    project(world: readonly number[]): Pixel | null {
      const ndc = transformPoint(viewProjection, world as Vec3);
      const pixel = { x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5), y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5) };
      return ndc[2] > 0 && ndc[2] < 1 && pixel.x >= 0 && pixel.y >= 0 && pixel.x < width && pixel.y < height ? pixel : null;
    },
    ray(pixel: Pixel): WorldRay {
      const x = (2 * (pixel.x + 0.5) / width - 1) * width / height * focal, y = (1 - 2 * (pixel.y + 0.5) / height) * focal;
      return { origin: eye as Vec3, direction: normalize(add(add(forward, right, x), up, y)), tMax: 60 };
    },
  };
}

async function verifiedPixels(r: Renderer, name: JointCameraName, worlds: readonly Vec3[], accept: (hit: WorldHit) => boolean) {
  const view = camera(name, r.stats.width, r.stats.height);
  const candidates = worlds.map(world => view.project(world)).filter((pixel): pixel is Pixel => !!pixel);
  const unique = [...new Map(candidates.map(pixel => [pixel.x + ',' + pixel.y, pixel])).values()];
  const hits = unique.length ? (await r.traceRays(unique.map(pixel => view.ray(pixel)))).hits : [];
  const pixels = unique.filter((_, index) => hits[index] && accept(hits[index]!));
  return { pixels, hits: hits.filter((hit, index) => hit && accept(hit) && unique[index]) as WorldHit[], total: worlds.length };
}

function roiWorlds(roi: JointRoi, grid: number): Vec3[] {
  const worlds: Vec3[] = [];
  for (let j = 0; j < grid; j++) for (let i = 0; i < grid; i++) {
    const s = (2 * (i + 0.5) / grid - 1) * 0.9, t = (2 * (j + 0.5) / grid - 1) * 0.9;
    worlds.push(add(add(add(roi.center, roi.normal, 0.002), roi.u, s), roi.v, t));
  }
  return worlds;
}

type PixelSet = { name: string; kind: string; start: number; count: number };
type Recorded = { frame: number; luminance: number[]; captured: number; queries: number;
  gpu: { frame: number; diffuseGiMs: number } | null };

async function capture(r: Renderer, canvas: HTMLCanvasElement, cameraName: JointCameraName, pixels: Pixel[]) {
  const view = JOINT_CAMERAS[cameraName];
  r.render(view.eye as Vec3, view.target as Vec3);
  const hdr = pixels.length ? await r.probeHdr(pixels) : [];
  const gi = r.diagnostics().diffuseGi;
  // Sparse asynchronous GPU samples; the analysis keeps each sample frame once.
  const gpu = r.stats.gpuMs && r.stats.gpuSampleFrame !== null ? { frame: r.stats.gpuSampleFrame, diffuseGiMs: r.stats.gpuMs.diffuseGi } : null;
  return { hdr, recorded: { frame: r.stats.frames, luminance: hdr.map(luminance), captured: gi?.schedule.capturedProbesLastUpdate ?? 0,
    queries: gi?.queriesPerUpdate ?? 0, gpu } satisfies Recorded, png: () => canvas.toDataURL('image/png') };
}

function rgbMeans(sets: PixelSet[], hdrFrames: number[][][]) {
  return Object.fromEntries(sets.map(set => {
    const sum = [0, 0, 0];
    let count = 0;
    for (const hdr of hdrFrames) for (let i = set.start; i < set.start + set.count; i++) {
      for (let c = 0; c < 3; c++) sum[c] += hdr[i][c];
      count++;
    }
    return [set.name, sum.map(value => value / Math.max(1, count))];
  }));
}

function canvasFor(width: number, height: number): HTMLCanvasElement {
  const element = document.createElement('canvas');
  element.style.cssText = `position:fixed;left:0;top:0;width:${width}px;height:${height}px;`;
  document.body.appendChild(element);
  return element;
}

function roomGrid(input: Pick<JointRunInput, 'raysPerProbe' | 'emissiveSamplesPerProbe' | 'update'>): DiffuseGiOptions {
  return { ...JOINT_ROOM_GRID, raysPerProbe: input.raysPerProbe ?? JOINT_ROOM_GRID.raysPerProbe,
    emissiveSamplesPerProbe: input.emissiveSamplesPerProbe ?? JOINT_ROOM_GRID.emissiveSamplesPerProbe, update: input.update };
}
function rayBudget(grid: DiffuseGiOptions): number {
  return Math.max(8192, grid.counts.reduce((a, b) => a * b, 1) * (grid.raysPerProbe ?? 64));
}

/** Scripted L02/L03/T01 sequence and the fixed-camera stability window. */
export async function runJointDynamics(input: JointRunInput) {
  const element = canvasFor(input.width, input.height);
  const scene = createJointRoomsScene();
  const grid = roomGrid(input);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, scene.batches, { worldQueries: { maxRays: rayBudget(grid) }, diffuseGi: grid,
      antiAliasing: 'none' });
    const r = renderer;
    r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
    if (input.gi === false) r.setDiffuseGiEnabled(false);
    const f = input.frames;
    const ownedBytes = new Set<number>();
    // ROI pixels are frozen per camera and verified against the intended surface.
    r.render(JOINT_CAMERAS.roomA.eye as Vec3, JOINT_CAMERAS.roomA.target as Vec3);
    const rois: Record<string, { camera: JointCameraName; kind: string; pixels: Pixel[]; total: number }> = {};
    for (const roi of JOINT_ROIS) {
      const { pixels, total } = await verifiedPixels(r, roi.camera, roiWorlds(roi, input.roiGrid),
        hit => hit.batch === roi.batch && hit.entityId === roi.entityId);
      rois[roi.name] = { camera: roi.camera, kind: roi.kind, pixels, total };
    }
    const setsFor = (cameraName: JointCameraName, extra: { name: string; kind: string; pixels: Pixel[] }[] = []) => {
      const sets: PixelSet[] = [], pixels: Pixel[] = [];
      for (const [name, roi] of Object.entries(rois)) if (roi.camera === cameraName && roi.pixels.length) {
        sets.push({ name, kind: roi.kind, start: pixels.length, count: roi.pixels.length }); pixels.push(...roi.pixels);
      }
      for (const set of extra) if (set.pixels.length) {
        sets.push({ name: set.name, kind: set.kind, start: pixels.length, count: set.pixels.length }); pixels.push(...set.pixels);
      }
      return { sets, pixels };
    };
    const phases: { name: string; camera: JointCameraName; eventIndex: number | null; stopIndex?: number; sets: PixelSet[];
      frames: Recorded[]; finalRgb: Record<string, number[]>; screenshots: { index: number; png: string }[] }[] = [];
    const phase = async (name: string, cameraName: JointCameraName, options: { baseline?: number; response?: number;
      event?: () => void | Promise<void>; during?: (index: number) => void; duringFrames?: number;
      extra?: { name: string; kind: string; pixels: Pixel[] }[] }) => {
      const { sets, pixels } = setsFor(cameraName, options.extra);
      const frames: Recorded[] = [], hdrs: number[][][] = [], screenshots: { index: number; png: string }[] = [];
      const baseline = options.baseline ?? f.baseline, response = options.response ?? f.response;
      const shots = new Set([baseline, baseline + 14, baseline + (options.duringFrames ?? 0) + response - 1]);
      for (let index = 0; index < baseline + (options.duringFrames ?? 0) + response; index++) {
        if (index === baseline && options.event) await options.event();
        if (options.during && index >= baseline && index < baseline + (options.duringFrames ?? 0)) options.during(index - baseline);
        const { hdr, recorded, png } = await capture(r, element, cameraName, pixels);
        frames.push(recorded); hdrs.push(hdr); ownedBytes.add(r.stats.ownedGpuBytes);
        if (input.screenshots && shots.has(index)) screenshots.push({ index, png: png() });
      }
      phases.push({ name, camera: cameraName, eventIndex: options.event || options.during ? baseline : null,
        stopIndex: options.duringFrames ? baseline + options.duringFrames : undefined, sets, frames,
        finalRgb: rgbMeans(sets, hdrs.slice(-Math.min(10, hdrs.length))), screenshots });
    };
    for (let i = 0; i < f.warmup; i++) r.render(JOINT_CAMERAS.roomA.eye as Vec3, JOINT_CAMERAS.roomA.target as Vec3);
    await phase('stability', 'roomA', { baseline: 0, response: f.stability });
    await phase('sunOff', 'roomA', { event: () => r.setSun({ direction: [...JOINT_SUN.direction], color: [0, 0, 0] }) });
    await phase('sunOn', 'roomA', { event: () => r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] }) });
    const moverPixels = rois['A.sunPatch'].pixels;
    await phase('mover', 'roomA', { duringFrames: f.moverFrames, extra: [{ name: 'mover.patch', kind: 'ghost', pixels: moverPixels }],
      during: index => {
        const z = JOINT_MOVER_PARKED_Z + (MOVER_END_Z - JOINT_MOVER_PARKED_Z) * (index + 1) / f.moverFrames;
        scene.mover.instances = [jointMover(z)];
      } });
    // Pixels the foreground object covers right now are revealed by its removal.
    const view = camera('roomA', r.stats.width, r.stats.height);
    const scan: Pixel[] = [];
    for (let y = 2; y < r.stats.height; y += Math.max(2, Math.round(r.stats.height / 48))) {
      for (let x = 2; x < r.stats.width; x += Math.max(2, Math.round(r.stats.width / 64))) scan.push({ x, y });
    }
    const scanHits = (await r.traceRays(scan.map(pixel => view.ray(pixel)))).hits;
    const revealed = scan.filter((_, index) => scanHits[index]?.batch === 'joint-foreground').slice(0, 192);
    await phase('removeForeground', 'roomA', { extra: [{ name: 'foreground.revealed', kind: 'ghost', pixels: revealed }],
      event: () => { scene.foreground.instances = []; } });
    // Camera teleport into room B: only the post-teleport frames are comparable.
    await phase('teleport', 'roomB', { baseline: 0, event: () => undefined });
    await phase('panelOff', 'roomB', { event: () => r.setMaterialParameters('joint-panel', new Float32Array(JOINT_PANEL_OFF)) });
    // With the panel off, room B is lit only through the door.
    await phase('doorClose', 'roomB', { event: () => { scene.door.instances = [JOINT_DOOR_CLOSED]; } });
    await phase('doorOpen', 'roomB', { event: () => { scene.door.instances = [JOINT_DOOR_OPEN]; } });
    await phase('panelOn', 'roomB', { event: () => r.setMaterialParameters('joint-panel', new Float32Array(JOINT_PANEL_ON)) });
    // L02: the same floor patch with the green wall and panel off-screen.
    await phase('away', 'roomBAway', { baseline: 0, event: () => undefined });
    const ghostWorlds: Vec3[] = [];
    for (let j = 0; j < 5; j++) for (let i = 0; i < 8; i++) ghostWorlds.push([1.3 + i * 0.2, 0.35 + j * 0.25, -2.498]);
    const panelGhost = (await verifiedPixels(r, 'roomB', ghostWorlds, hit => hit.batch === 'joint-shell' && hit.entityId === 2)).pixels;
    await phase('panelMove', 'roomB', { duringFrames: f.panelMoveFrames,
      extra: [{ name: 'panel.wall', kind: 'ghost', pixels: panelGhost }],
      during: index => {
        const t = (index + 1) / f.panelMoveFrames;
        scene.panel.instances = [jointPanel(JOINT_PANEL_START.map((value, axis) => value + (JOINT_PANEL_END[axis] - value) * t) as Vec3)];
      } });
    const diagnostics = r.diagnostics().diffuseGi;
    const errors = [...r.errors];
    const result = { adapter: r.info.adapterInfo, viewport: { width: r.stats.width, height: r.stats.height },
      grid: { counts: grid.counts, raysPerProbe: grid.raysPerProbe, probes: grid.counts.reduce((a, b) => a * b, 1) },
      update: diagnostics?.update, schedule: diagnostics?.schedule,
      rois: Object.fromEntries(Object.entries(rois).map(([name, roi]) => [name, { camera: roi.camera, kind: roi.kind,
        verified: roi.pixels.length, total: roi.total }])),
      revealedPixels: revealed.length, panelGhostPixels: panelGhost.length, phases, errors,
      ownedBytes: [...ownedBytes] };
    r.dispose();
    return { ...result, afterDispose: r.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

function hash32(value: number): number {
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}
function cosineDirection(normal: Vec3, sample: number, point: number, bounce: number): Vec3 {
  const seed = hash32(sample ^ Math.imul(point + 1, 0x9e3779b9) ^ Math.imul(bounce + 1, 0x85ebca6b));
  const u = (hash32(seed ^ 0x68bc21eb) + 0.5) / 0x100000000;
  const phi = 2 * Math.PI * (hash32(seed ^ 0x02e5be93) + 0.5) / 0x100000000;
  const radius = Math.sqrt(u);
  const tangent = normalize(cross(Math.abs(normal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], normal));
  const bitangent = cross(normal, tangent);
  return normalize(add(add(add([0, 0, 0], tangent, radius * Math.cos(phi)), bitangent, radius * Math.sin(phi)),
    normal, Math.sqrt(1 - u)));
}

/** Independent diffuse path-traced outgoing indirect radiance at receiver hits,
 * with the engine's GI transport model (diffuse-only, F0 0.04, binary sun
 * visibility through world queries, sky beyond the scene, unlit emitters). */
async function pathTrace(r: Renderer, receivers: readonly WorldHit[], options: { paths: number; bounces: number;
  sky: Vec3; sun: { direction: Vec3; color: Vec3 }; normalBias: number }) {
  const oneBounce = receivers.map(() => [0, 0, 0]), total = receivers.map(() => [0, 0, 0]);
  const batch = Math.max(1, Math.floor(8192 / options.paths));
  let rays = 0;
  for (let start = 0; start < receivers.length; start += batch) {
    const paths = receivers.slice(start, start + batch).flatMap((hit, local) => Array.from({ length: options.paths }, (_, sample) => {
      const normal = normalize(hit.normal);
      return { point: start + local, sample, origin: add(hit.position, normal, options.normalBias),
        direction: cosineDirection(normal, sample, start + local, 0),
        throughput: hit.baseColor.map(value => value * 0.96 * (1 - hit.metallic)), active: true };
    }));
    for (let bounce = 0; bounce < options.bounces; bounce++) {
      const active = paths.filter(path => path.active);
      if (!active.length) break;
      const hits = (await r.traceRays(active.map(path => ({ origin: path.origin, direction: path.direction, tMax: 60 })))).hits;
      rays += active.length;
      const shadowIndex: number[] = [];
      const shadowRays: WorldRay[] = [];
      hits.forEach((hit, i) => {
        if (!hit || hit.shading === 'unlit' || dot(hit.normal, options.sun.direction) <= 0
          || dot(hit.geometricNormal, options.sun.direction) <= 0 || !options.sun.color.some(value => value > 0)) return;
        shadowIndex.push(i);
        shadowRays.push({ origin: add(hit.position, hit.geometricNormal, options.normalBias), direction: options.sun.direction, tMax: 60 });
      });
      const shadowHits = shadowRays.length ? (await r.traceRays(shadowRays)).hits : [];
      rays += shadowRays.length;
      const lit = new Set(shadowIndex.filter((_, i) => !shadowHits[i]));
      active.forEach((path, i) => {
        const hit = hits[i];
        const radiance = hit ? [...hit.emissive] : [...options.sky];
        if (hit?.shading === 'unlit') for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c];
        else if (hit && lit.has(i)) {
          const nl = Math.max(0, dot(hit.normal, options.sun.direction));
          for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c] * (1 - hit.metallic) * 0.96 / Math.PI * options.sun.color[c] * nl;
        }
        for (let c = 0; c < 3; c++) {
          const contribution = path.throughput[c] * radiance[c] / options.paths;
          total[path.point][c] += contribution;
          if (bounce === 0) oneBounce[path.point][c] += contribution;
        }
        if (!hit || hit.shading === 'unlit' || bounce === options.bounces - 1) { path.active = false; return; }
        for (let c = 0; c < 3; c++) path.throughput[c] *= hit.baseColor[c] * 0.96 * (1 - hit.metallic);
        const normal = normalize(hit.normal);
        path.origin = add(hit.position, normal, options.normalBias);
        path.direction = cosineDirection(normal, path.sample, path.point, bounce + 1);
      });
    }
  }
  return { oneBounce, total, rays };
}

/** L01 steady state: GI on/off and single/multi bounce at frozen ROI pixels, plus
 * an independent path-traced reference of the same pixels (optional, costly). */
export async function runJointSteadyState(input: Omit<JointRunInput, 'frames' | 'screenshots' | 'gi'> & {
  warmup: number; singleBounceWarmup: number; reference?: { paths: number; bounces: number } }) {
  const element = canvasFor(input.width, input.height);
  const scene = createJointRoomsScene();
  const grid = roomGrid(input);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, scene.batches, { worldQueries: { maxRays: rayBudget(grid) }, diffuseGi: grid,
      antiAliasing: 'none' });
    const r = renderer;
    r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
    r.render(JOINT_CAMERAS.roomA.eye as Vec3, JOINT_CAMERAS.roomA.target as Vec3);
    // World queries allow one pending readback, so ROI verification is sequential.
    const rois: ({ roi: (typeof JOINT_ROIS)[number] } & Awaited<ReturnType<typeof verifiedPixels>>)[] = [];
    for (const roi of JOINT_ROIS.filter(item => item.camera !== 'roomBAway')) rois.push({ roi,
      ...await verifiedPixels(r, roi.camera, roiWorlds(roi, input.roiGrid), hit => hit.batch === roi.batch && hit.entityId === roi.entityId) });
    const read = async () => {
      const values: Record<string, number[][]> = {};
      for (const cameraName of ['roomA', 'roomB'] as const) {
        const view = JOINT_CAMERAS[cameraName];
        r.render(view.eye as Vec3, view.target as Vec3);
        for (const { roi, pixels } of rois) if (roi.camera === cameraName) values[roi.name] = await r.probeHdr(pixels);
      }
      return values;
    };
    const converge = async (frames: number) => {
      for (let i = 0; i < frames; i++) {
        const view = JOINT_CAMERAS[i % 2 ? 'roomB' : 'roomA'];
        r.render(view.eye as Vec3, view.target as Vec3);
      }
    };
    await converge(input.warmup);
    const multi = await read();
    r.setDiffuseGiEnabled(false);
    const off = await read();
    r.setDiffuseGiEnabled(true);
    r.setDiffuseGiMultiBounce(false);
    await converge(input.singleBounceWarmup);
    const single = await read();
    let reference: { oneBounce: Record<string, number[][]>; total: Record<string, number[][]>; rays: number;
      paths: number; bounces: number } | null = null;
    if (input.reference) {
      const receivers = rois.flatMap(({ hits }) => hits);
      const traced = await pathTrace(r, receivers, { ...input.reference, sky: JOINT_SKY,
        sun: { direction: [...JOINT_SUN.direction] as Vec3, color: [...JOINT_SUN.color] as Vec3 }, normalBias: grid.normalBias ?? 0.03 });
      let offset = 0;
      const oneBounce: Record<string, number[][]> = {}, total: Record<string, number[][]> = {};
      for (const { roi, hits } of rois) {
        oneBounce[roi.name] = traced.oneBounce.slice(offset, offset + hits.length);
        total[roi.name] = traced.total.slice(offset, offset + hits.length);
        offset += hits.length;
      }
      reference = { oneBounce, total, rays: traced.rays, ...input.reference };
    }
    const errors = [...r.errors];
    const result = { adapter: r.info.adapterInfo, viewport: { width: r.stats.width, height: r.stats.height },
      update: r.diagnostics().diffuseGi?.update,
      rois: Object.fromEntries(rois.map(({ roi, pixels, total }) => [roi.name, { kind: roi.kind, verified: pixels.length, total }])),
      multi, single, off, reference, errors };
    r.dispose();
    return { ...result, afterDispose: r.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

/** L04: interiors of closed black/thin-wall boxes and a slit box, GI on and off. */
export async function runJointLeak(input: { update?: DiffuseGiUpdateOptions; width: number; height: number; warmup: number;
  raysPerProbe?: number; samples: [number, number]; reference?: { paths: number; bounces: number }; screenshots: boolean }) {
  const element = canvasFor(input.width, input.height);
  const { batches } = createJointLeakScene();
  const grid: DiffuseGiOptions = { ...JOINT_LEAK_GRID, raysPerProbe: input.raysPerProbe ?? JOINT_LEAK_GRID.raysPerProbe,
    update: input.update };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, batches, { worldQueries: { maxRays: rayBudget(grid) }, diffuseGi: grid,
      antiAliasing: 'none' });
    const r = renderer;
    r.setSun({ direction: [...JOINT_SUN.direction], color: [...JOINT_SUN.color] });
    const views = ['blackBox', 'thinBox', 'slitBox'] as const;
    for (let i = 0; i < Math.max(1, input.warmup); i++) {
      const view = JOINT_CAMERAS[views[i % 3]];
      r.render(view.eye as Vec3, view.target as Vec3);
    }
    // The viewport size is known only after the first rendered frame.
    const pixels: Pixel[] = [];
    for (let j = 0; j < input.samples[1]; j++) for (let i = 0; i < input.samples[0]; i++) {
      pixels.push({ x: Math.floor((i + 0.5) * r.stats.width / input.samples[0]), y: Math.floor((j + 0.5) * r.stats.height / input.samples[1]) });
    }
    if (new Set(pixels.map(pixel => pixel.x + ',' + pixel.y)).size !== pixels.length) throw new Error('JOINT_LEAK_PIXELS_NOT_DISTINCT');
    const outside = { eye: [0.25, 1.6, 2.4] as Vec3, target: [0.25, 0.2, 1.1] as Vec3 };
    const read = async () => {
      const values: Record<string, number[][]> = {}, pngs: Record<string, string> = {};
      for (const name of views) {
        const view = JOINT_CAMERAS[name];
        r.render(view.eye as Vec3, view.target as Vec3);
        values[name] = await r.probeHdr(pixels);
        if (input.screenshots) pngs[name] = element.toDataURL('image/png');
      }
      r.render(outside.eye, outside.target);
      values.outside = await r.probeHdr([{ x: Math.floor(r.stats.width / 2), y: Math.floor(r.stats.height * 0.7) }]);
      return { values, pngs };
    };
    const on = await read();
    r.setDiffuseGiEnabled(false);
    const off = await read();
    let slitReference: number[][] | null = null;
    if (input.reference) {
      const view = JOINT_CAMERAS.slitBox;
      const forward = normalize(sub(view.target, view.eye)), right = normalize(cross(forward, [0, 1, 0])), up = cross(right, forward);
      const focal = Math.tan(Math.PI / 6.4), w = r.stats.width, h = r.stats.height;
      const hits = (await r.traceRays(pixels.map(pixel => ({ origin: view.eye as Vec3, tMax: 60,
        direction: normalize(add(add(forward, right, (2 * (pixel.x + 0.5) / w - 1) * w / h * focal), up, (1 - 2 * (pixel.y + 0.5) / h) * focal)) })))).hits;
      const receivers = hits.filter((hit): hit is WorldHit => !!hit);
      const traced = await pathTrace(r, receivers, { ...input.reference, sky: JOINT_SKY,
        sun: { direction: [...JOINT_SUN.direction] as Vec3, color: [...JOINT_SUN.color] as Vec3 }, normalBias: grid.normalBias ?? 0.03 });
      let next = 0;
      slitReference = hits.map(hit => hit ? traced.total[next++] : [0, 0, 0]);
    }
    const errors = [...r.errors];
    const result = { adapter: r.info.adapterInfo, viewport: { width: r.stats.width, height: r.stats.height },
      update: r.diagnostics().diffuseGi?.update, pixels: pixels.length, on: on.values, off: off.values, pngs: on.pngs, offPngs: off.pngs,
      slitReference, errors };
    r.dispose();
    return { ...result, afterDispose: r.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}
