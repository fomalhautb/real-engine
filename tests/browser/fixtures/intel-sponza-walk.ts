import { Renderer } from '../../../src/renderer';
import type { SceneBatch, StreamedSceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import { geometryCutCoverage } from '../../../src/geometry-stream-coverage';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import { oversubscribedGeometryBudget } from '../../../src/geometry-stream-oversubscription';
import { decodeGeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { decodeVerifiedGeometryPage } from '../../../src/geometry-page-decode';
import { parseCameraPath, sampleCameraPath } from '../../../src/camera-path';
import type { GpuGeometryStream } from '../../../src/gpu-geometry-stream';
import { loadIntelSponzaAssets, type IntelSponzaAssetOptions } from './intel-sponza-assets';
import { assertCapacityLedger, interpolateCapacityCamera, type CapacityCamera } from './intel-sponza-capacity-checks';

const MiB = 1024 ** 2;
const LIMITS = { maxBufferSize: 256 * MiB, maxStorageBufferBindingSize: 128 * MiB,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };
const TAN_HALF = Math.tan(Math.PI / 6.4);
/** Hole pixels classified per check; any beyond this count as unexplained. */
const HOLE_CLASSIFICATION_LIMIT = 32_768;
/** CPU ray-cast work per classified batch (world triangles, all instances). */
const CLASSIFY_TRIANGLES = 2_000_000;
/** Triangle tests per check across all classified pixels; the rest stay unclassified. */
const CLASSIFY_TESTS = 400_000_000;
/** Screen-tile bins: tile size and margin in pixels, total tile entries per bin set,
 * and the camera depth at or below which a triangle is tested by every ray. */
const BIN_TILE = 16, BIN_MARGIN = 128, BIN_ENTRIES = 24_000_000, BIN_NEAR_DEPTH = 1e-4;
/** Half-pixel offsets for raster ties at a drawn silhouette. */
const HALF_PIXEL: readonly (readonly [number, number])[] = [[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]];
const LEVELS = [0.1, 0.22, 0.34, 0.46, 0.58, 0.7, 0.82, 0.94];

/** 512 identity colors, at least 0.12 apart in every differing channel. */
export function walkIdentityColor(index: number): [number, number, number, number] {
  if (!Number.isSafeInteger(index) || index < 0 || index >= 512) throw new RangeError('SPONZA_WALK_IDENTITY');
  return [LEVELS[index & 7], LEVELS[(index >> 3) & 7], LEVELS[(index >> 6) & 7], 1];
}

type Vec3 = [number, number, number];
const A: CapacityCamera = { eye: [9, 1.8, 0], target: [-6, 3.5, 0] };
const B: CapacityCamera = { eye: [-9, 1.8, 0], target: [6, 3.5, 0] };
const A_TURNED: CapacityCamera = { eye: [9, 1.8, 0], target: [24, 3.5, 0] };
const AISLE: CapacityCamera = { eye: [-10, 1.7, 0], target: [-10, 2.7, 10] };
type Phase = { label: string; kind: 'cold' | 'teleport' | 'turn'; to: CapacityCamera; from?: CapacityCamera };
const SEQUENCE: readonly Phase[] = [
  { label: 'cold-a', kind: 'cold', to: A },
  { label: 'teleport-b', kind: 'teleport', to: B },
  { label: 'teleport-a', kind: 'teleport', to: A },
  { label: 'turn-180-at-a', kind: 'turn', from: A, to: A_TURNED },
  { label: 'turn-180-back', kind: 'turn', from: A_TURNED, to: A },
  { label: 'teleport-aisle', kind: 'teleport', to: AISLE },
  { label: 'teleport-a-again', kind: 'teleport', to: A },
];

export type IntelSponzaWalkMode = 'verify' | 'recovery' | 'present';
export interface IntelSponzaWalkOptions extends Omit<IntelSponzaAssetOptions, 'sourceRaster' | 'queryRoot'> {
  /** Unique resident-format bytes per byte of steady-state geometry pool (G03: 4–8). */
  ratio: number;
  mode: IntelSponzaWalkMode;
  /** Server path of a real-engine-camera-path file for the continuous walk. */
  cameraPathUrl: string;
  /** Fine LOD0 query-proxy package: the full-detail reference renderer (verify only). */
  fineQueryRoot?: string;
  width?: number;
  height?: number;
  maxErrorPixels?: number;
  /** Frames a settled signature must stay unchanged; must exceed the admission interval. */
  stableFrames?: number;
  maxSettleFrames?: number;
  /** verify: quality check every this many frames while streaming is busy (0: first frames and settled only). */
  checkEvery?: number;
  /** verify: quality check every this many walk frames. */
  walkCheckEvery?: number;
  /** Seconds of path time per walk frame in verify and present modes. */
  walkStep?: number;
  /** Wall-clock duration of each 180-degree turn. */
  turnSeconds?: number;
  intervalFrames?: number;
  maxConcurrentMigrations?: number;
  exposureEV?: number;
  /** present: fraction of the width shown in material shading; the rest shows residency. */
  debugSplit?: number;
  /** present: frames rendered after the walk at the final station. */
  presentTailFrames?: number;
}

export interface IntelSponzaWalkSink {
  /** present: one canvas frame (JPEG data URL), in order. */
  frame?(frame: { index: number; label: string; jpeg: string }): Promise<void>;
  /** verify: evidence images of failing checks. */
  capture?(capture: { label: string; png: string }): Promise<void>;
}

type PrivateRenderer = { device: GPUDevice; hdr: GPUTexture; geometryCapacity?: { dispose(): Promise<void> };
  batches: { scene: { name: string }; stream?: { runtime: GpuGeometryStream } }[] };
type Diagnostics = ReturnType<Renderer['diagnostics']>;
type Probe = Awaited<ReturnType<GpuGeometryStream['probe']>>;

async function readHdr(renderer: Renderer): Promise<{ width: number; height: number; rgb: Float32Array }> {
  const r = renderer as unknown as PrivateRenderer, width = renderer.stats.width, height = renderer.stats.height;
  const bytesPerRow = Math.ceil(width * 8 / 256) * 256;
  const buffer = r.device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = r.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: r.hdr }, { buffer, bytesPerRow }, { width, height });
    r.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint16Array(buffer.getMappedRange()), rgb = new Float32Array(width * height * 3);
    const half = (bits: number) => { const e = (bits >>> 10) & 31, f = bits & 1023; return e ? (1 + f / 1024) * 2 ** (e - 15) : f * 2 ** -24; };
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) rgb[(y * width + x) * 3 + c] = half(words[y * bytesPerRow / 2 + x * 4 + c]);
    }
    buffer.unmap();
    return { width, height, rgb };
  } finally { buffer.destroy(); }
}

/** Nearest identity label within an L1 distance of 0.03, else -1 (sky, background). */
function labels(image: { rgb: Float32Array }, palette: readonly number[][]): Int16Array {
  const result = new Int16Array(image.rgb.length / 3).fill(-1);
  for (let p = 0; p < result.length; p++) {
    const r = image.rgb[p * 3], g = image.rgb[p * 3 + 1], b = image.rgb[p * 3 + 2];
    // Palette index from the quantised channels, then an exact distance check.
    const level = (value: number) => Math.round((value - 0.1) / 0.12);
    const i = level(r), j = level(g), k = level(b);
    if (i < 0 || i > 7 || j < 0 || j > 7 || k < 0 || k > 7) continue;
    const index = i | j << 3 | k << 6;
    if (index >= palette.length) continue;
    const color = palette[index];
    if (Math.abs(r - color[0]) + Math.abs(g - color[1]) + Math.abs(b - color[2]) < 0.03) result[p] = index;
  }
  return result;
}

/** Pixels deep inside a reference object (every pixel within its radius has the
 * same label) that the streamed frame shows as anything else. LOD silhouette
 * error is excluded by the per-object radius; holes, cracks and mis-culling are not. */
function holes(reference: Int16Array, streamed: Int16Array, width: number, height: number, radii: readonly number[]) {
  const objects = radii.length, row = new Int32Array(width * height), interior = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    let last = -Infinity;
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      if (x === 0 || reference[p] !== reference[p - 1]) last = x;
      row[p] = x - last;
    }
    last = Infinity;
    for (let x = width - 1; x >= 0; x--) {
      const p = y * width + x;
      if (x === width - 1 || reference[p] !== reference[p + 1]) last = x;
      row[p] = Math.min(row[p], last - x);
    }
  }
  for (let x = 0; x < width; x++) for (let y = 1; y < height - 1; y++) {
    const p = y * width + x;
    if (reference[p] < 0 || reference[p] >= objects) continue;
    const radius = radii[reference[p]];
    if (!Number.isFinite(radius) || row[p] < radius || y < radius || y >= height - radius) continue;
    let ok = true;
    for (let dy = -radius; dy <= radius && ok; dy++) {
      const q = p + dy * width;
      if (reference[q] !== reference[p] || row[q] < radius) ok = false;
    }
    if (ok) interior[p] = 1;
  }
  let holePixels = 0, testedPixels = 0;
  const byObject = new Map<number, number>(), interiorObjects = new Set<number>(), holeList: number[] = [];
  for (let p = 0; p < interior.length; p++) {
    if (!interior[p]) continue;
    testedPixels++; interiorObjects.add(reference[p]);
    if (streamed[p] !== reference[p]) {
      holePixels++; byObject.set(reference[p], (byObject.get(reference[p]) ?? 0) + 1);
      if (holeList.length < HOLE_CLASSIFICATION_LIMIT) holeList.push(p);
    }
  }
  return { testedPixels, holePixels, objectsTested: interiorObjects.size, untestedObjects: radii.filter(r => !Number.isFinite(r)).length,
    holesByObject: [...byObject].sort((a, b) => b[1] - a[1]).slice(0, 8), interior, holeList };
}

const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a: readonly number[]): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

/** World-space triangles, 9 floats each, of local triangles under each model. */
function worldTriangles(positions: (vertex: number) => Vec3, corners: ArrayLike<number>, models: readonly Float32Array[], into: number[]) {
  for (const model of models) for (let t = 0; t < corners.length; t += 3) for (let k = 0; k < 3; k++) {
    const [x, y, z] = positions(corners[t + k]);
    into.push(model[0] * x + model[4] * y + model[8] * z + model[12], model[1] * x + model[5] * y + model[9] * z + model[13],
      model[2] * x + model[6] * y + model[10] * z + model[14]);
  }
}
/** Two-sided hit distance beyond the near plane of one triangle (9 floats at `t`), or null. */
function rayTriangle(triangles: Float32Array, t: number, eye: Vec3, direction: Vec3): number | null {
  const p0 = [triangles[t], triangles[t + 1], triangles[t + 2]];
  const e1 = [triangles[t + 3] - p0[0], triangles[t + 4] - p0[1], triangles[t + 5] - p0[2]];
  const e2 = [triangles[t + 6] - p0[0], triangles[t + 7] - p0[1], triangles[t + 8] - p0[2]];
  const p = cross(direction, e2), det = dot(e1, p);
  if (Math.abs(det) < 1e-14) return null;
  const s = sub(eye, p0), u = dot(s, p) / det, q = cross(s, e1), v = dot(direction, q) / det, distance = dot(e2, q) / det;
  return u < 0 || v < 0 || u + v > 1 || distance <= 0.1 ? null : distance;
}
/** Squared Euclidean distance from a point to one triangle (closest-point regions). */
function triangleDistanceSquared(triangles: Float32Array, t: number, point: Vec3): number {
  const a: Vec3 = [triangles[t], triangles[t + 1], triangles[t + 2]], b: Vec3 = [triangles[t + 3], triangles[t + 4], triangles[t + 5]];
  const c: Vec3 = [triangles[t + 6], triangles[t + 7], triangles[t + 8]];
  const ab = sub(b, a), ac = sub(c, a), ap = sub(point, a), d1 = dot(ab, ap), d2 = dot(ac, ap);
  const bp = sub(point, b), d3 = dot(ab, bp), d4 = dot(ac, bp), cp = sub(point, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
  let closest: Vec3;
  const along = (o: Vec3, d: Vec3, s: number): Vec3 => [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
  if (d1 <= 0 && d2 <= 0) closest = a;
  else if (d3 >= 0 && d4 <= d3) closest = b;
  else if (vc <= 0 && d1 >= 0 && d3 <= 0) closest = along(a, ab, d1 / (d1 - d3));
  else if (d6 >= 0 && d5 <= d6) closest = c;
  else if (vb <= 0 && d2 >= 0 && d6 <= 0) closest = along(a, ac, d2 / (d2 - d6));
  else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) closest = along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
  else { const denominator = 1 / (va + vb + vc); closest = along(along(a, ab, vb * denominator), ac, vc * denominator); }
  const d = sub(point, closest);
  return dot(d, d);
}

/** Camera frame of one check; pixel (x, y) has its centre at integer coordinates. */
type ScreenFrame = { eye: Vec3; forward: Vec3; right: Vec3; up: Vec3; width: number; height: number; focal: number };
/** Conservative screen-tile bins of world triangles for one camera. Each binned
 * triangle has all corners in front of the eye and is listed in every tile its
 * projected bounds, grown by one pixel, touch; the bins cover the screen plus a
 * margin. Triangles reaching the near plane are tested by every query. */
type TriangleBins = { triangles: Float32Array; tilesX: number; offsets: Uint32Array; items: Uint32Array; near: Uint32Array;
  stamp: Uint32Array; mark: number };
type ClassifyWork = { tests: number };

function rayAt(frame: ScreenFrame, x: number, y: number): Vec3 {
  const nx = ((x + 0.5) / frame.width) * 2 - 1, ny = 1 - ((y + 0.5) / frame.height) * 2;
  return unit([0, 1, 2].map(axis => frame.forward[axis] + frame.right[axis] * nx * TAN_HALF * frame.width / frame.height + frame.up[axis] * ny * TAN_HALF));
}

/** Returns null when the bins would exceed BIN_ENTRIES (the caller leaves those pixels unclassified). */
function binTriangles(triangles: Float32Array, frame: ScreenFrame): TriangleBins | null {
  const count = triangles.length / 9, tilesX = Math.ceil((frame.width + 2 * BIN_MARGIN) / BIN_TILE);
  const tilesY = Math.ceil((frame.height + 2 * BIN_MARGIN) / BIN_TILE);
  const boxes = new Int32Array(count * 4).fill(-1), counts = new Uint32Array(tilesX * tilesY + 1), near: number[] = [];
  let entries = 0;
  for (let i = 0; i < count; i++) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let k = 0; k < 3; k++) {
      const o = i * 9 + k * 3, dx = triangles[o] - frame.eye[0], dy = triangles[o + 1] - frame.eye[1], dz = triangles[o + 2] - frame.eye[2];
      const z = dx * frame.forward[0] + dy * frame.forward[1] + dz * frame.forward[2];
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
      if (z <= BIN_NEAR_DEPTH) continue;
      const x = frame.width / 2 + frame.focal * (dx * frame.right[0] + dy * frame.right[1] + dz * frame.right[2]) / z - 0.5;
      const y = frame.height / 2 - frame.focal * (dx * frame.up[0] + dy * frame.up[1] + dz * frame.up[2]) / z - 0.5;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    // Wholly behind the eye: no forward ray reaches it.
    if (maxZ <= 0) continue;
    if (minZ <= BIN_NEAR_DEPTH) { near.push(i); continue; }
    const x0 = Math.max(0, Math.floor((minX - 1 + BIN_MARGIN) / BIN_TILE)), x1 = Math.min(tilesX - 1, Math.floor((maxX + 1 + BIN_MARGIN) / BIN_TILE));
    const y0 = Math.max(0, Math.floor((minY - 1 + BIN_MARGIN) / BIN_TILE)), y1 = Math.min(tilesY - 1, Math.floor((maxY + 1 + BIN_MARGIN) / BIN_TILE));
    if (x0 > x1 || y0 > y1) continue;
    entries += (x1 - x0 + 1) * (y1 - y0 + 1);
    if (entries > BIN_ENTRIES) return null;
    boxes[i * 4] = x0; boxes[i * 4 + 1] = y0; boxes[i * 4 + 2] = x1; boxes[i * 4 + 3] = y1;
    for (let ty = y0; ty <= y1; ty++) for (let tx = x0; tx <= x1; tx++) counts[ty * tilesX + tx + 1]++;
  }
  for (let tile = 1; tile < counts.length; tile++) counts[tile] += counts[tile - 1];
  const items = new Uint32Array(entries), cursor = counts.slice(0, -1);
  for (let i = 0; i < count; i++) {
    if (boxes[i * 4] < 0) continue;
    for (let ty = boxes[i * 4 + 1]; ty <= boxes[i * 4 + 3]; ty++) for (let tx = boxes[i * 4]; tx <= boxes[i * 4 + 2]; tx++) items[cursor[ty * tilesX + tx]++] = i;
  }
  return { triangles, tilesX, offsets: counts, items, near: Uint32Array.from(near), stamp: new Uint32Array(count), mark: 0 };
}

/** Nearest hit of the ray through pixel coordinates (x, y), which must lie within
 * the bins. The hit point projects to (x, y), inside the hit triangle's bounds. */
function castBinned(bins: TriangleBins, frame: ScreenFrame, x: number, y: number, work: ClassifyWork): number | null {
  const direction = rayAt(frame, x, y), tile = Math.floor((y + BIN_MARGIN) / BIN_TILE) * bins.tilesX + Math.floor((x + BIN_MARGIN) / BIN_TILE);
  let best: number | null = null;
  const test = (triangle: number) => {
    const distance = rayTriangle(bins.triangles, triangle * 9, frame.eye, direction);
    if (distance !== null && (best === null || distance < best)) best = distance;
  };
  for (let k = bins.offsets[tile]; k < bins.offsets[tile + 1]; k++) test(bins.items[k]);
  for (const triangle of bins.near) test(triangle);
  work.tests += bins.offsets[tile + 1] - bins.offsets[tile] + bins.near.length;
  return best;
}

/** Is any binned triangle within `limit` (world units) of `point`, the hit at camera
 * depth `depth` of the ray through pixel (x, y)? A point Q within `limit` has depth at
 * least depth - limit, so its projection lies within focal * limit / (depth - limit)
 * * (1 + |offset of (x, y) from the principal point| / focal) pixels of (x, y): only
 * tiles touching that window and near-plane triangles are visited. Null when the
 * window leaves the bins. */
function withinBinned(bins: TriangleBins, frame: ScreenFrame, point: Vec3, depth: number, limit: number, x: number, y: number,
  work: ClassifyWork): boolean | null {
  if (!(limit < depth)) return null;
  const offAxis = Math.hypot(x + 0.5 - frame.width / 2, y + 0.5 - frame.height / 2) / frame.focal;
  const reach = Math.ceil(frame.focal * limit / (depth - limit) * (1 + offAxis)) + 2;
  if (reach > BIN_MARGIN - BIN_TILE) return null;
  if (++bins.mark === 0xffffffff) { bins.stamp.fill(0); bins.mark = 1; }
  const limit2 = limit * limit, tile = (value: number) => Math.floor((value + BIN_MARGIN) / BIN_TILE);
  const near = (triangle: number) => {
    if (bins.stamp[triangle] === bins.mark) return false;
    bins.stamp[triangle] = bins.mark; work.tests++;
    return triangleDistanceSquared(bins.triangles, triangle * 9, point) <= limit2;
  };
  for (let ty = tile(y - reach); ty <= tile(y + reach); ty++) for (let tx = tile(x - reach); tx <= tile(x + reach); tx++) {
    const index = ty * bins.tilesX + tx;
    for (let k = bins.offsets[index]; k < bins.offsets[index + 1]; k++) if (near(bins.items[k])) return true;
  }
  for (const triangle of bins.near) if (near(triangle)) return true;
  return false;
}

/** Oversubscribed Intel Sponza walk: teleports, 180-degree turns and a camera-path
 * walk with the runtime capacity budget and two-phase occlusion. verify checks
 * GPU cuts and identity images against a full-detail LOD0 reference (test-only
 * readbacks); recovery measures streaming recovery without any probe; present
 * renders the textured scene at a fixed step for a clip. Timings are diagnostic. */
export async function runIntelSponzaWalk(options: IntelSponzaWalkOptions, sink: IntelSponzaWalkSink = {}) {
  const mode = options.mode;
  if (mode !== 'verify' && mode !== 'recovery' && mode !== 'present') throw new Error('SPONZA_WALK_MODE');
  const width = options.width ?? (mode === 'present' ? 1280 : 960), height = options.height ?? (mode === 'present' ? 720 : 540);
  const maxErrorPixels = options.maxErrorPixels ?? 1, stableFrames = options.stableFrames ?? 45;
  const maxSettleFrames = options.maxSettleFrames ?? 1800, checkEvery = options.checkEvery ?? 15;
  const walkCheckEvery = options.walkCheckEvery ?? 30, walkStep = options.walkStep ?? 1 / 30, turnSeconds = options.turnSeconds ?? 0.25;
  const intervalFrames = options.intervalFrames ?? 30, maxConcurrentMigrations = options.maxConcurrentMigrations ?? 4;
  for (const [value, low, high, name] of [[width, 64, 3840, 'WIDTH'], [height, 64, 2160, 'HEIGHT'], [stableFrames, 2, 600, 'STABLE'],
    [maxSettleFrames, 10, 20000, 'SETTLE'], [checkEvery, 0, 600, 'CHECK'], [walkCheckEvery, 1, 3600, 'WALK_CHECK'],
    [intervalFrames, 1, 600, 'INTERVAL'], [maxConcurrentMigrations, 1, 16, 'MIGRATIONS']] as const) {
    if (!Number.isSafeInteger(value) || value < low || value > high) throw new RangeError('SPONZA_WALK_' + name);
  }
  if (stableFrames <= intervalFrames) throw new RangeError('SPONZA_WALK_STABLE_WITHIN_INTERVAL');
  if (!Number.isFinite(maxErrorPixels) || maxErrorPixels <= 0 || maxErrorPixels > 16) throw new RangeError('SPONZA_WALK_LOD_ERROR');
  if (!Number.isFinite(walkStep) || walkStep <= 0 || walkStep > 1 || !Number.isFinite(turnSeconds) || turnSeconds <= 0 || turnSeconds > 2) {
    throw new RangeError('SPONZA_WALK_TIME_STEP');
  }
  if (mode === 'verify' && !options.fineQueryRoot) throw new Error('SPONZA_WALK_VERIFY_REQUIRES_FINE_QUERY');
  const started = performance.now(), projectionScale = height / (2 * TAN_HALF);
  const progress = (phase: string, data: unknown = {}) => console.log('SPONZA_WALK_PROGRESS ' + JSON.stringify({ phase, elapsedMs: Math.round(performance.now() - started), data }));
  const pathResponse = await fetch(options.cameraPathUrl);
  if (!pathResponse.ok) throw new Error('SPONZA_WALK_CAMERA_PATH_HTTP');
  const path = parseCameraPath(await pathResponse.json());

  const panel = document.createElement('section');
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#0b1418;overflow:auto';
  const canvases = (mode === 'verify' ? [0, 1] : [0]).map(() => {
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    canvas.style.cssText = `width:${width}px;height:${height}px;display:block`; panel.appendChild(canvas); return canvas;
  });
  document.body.appendChild(panel);
  let streamed: Renderer | undefined, reference: Renderer | undefined;
  try {
    const assets = await loadIntelSponzaAssets(options, progress), { batches, bounds, sources } = assets;
    const inputs = batches.map(batch => ({ directory: batch.geometryStream.directory, capacity: batch.capacity }));
    const budget = oversubscribedGeometryBudget(inputs, options.ratio, LIMITS, maxConcurrentMigrations);
    const identity = mode === 'verify';
    const palette = batches.map((_, index) => walkIdentityColor(index));
    const identityMaterial = { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const, baseColor: m.instanceColor } };
    // Cold start: every pool holds its roots only; the capacity budget grows them.
    const streamedBatches: StreamedSceneBatch[] = batches.map((batch, index) => ({
      ...batch, ...(identity ? { material: identityMaterial, instances: batch.instances.map(instance => ({ ...instance, color: palette[index] })) } : {}),
      geometryStream: { ...batch.geometryStream, gpuPool: { ...batch.geometryStream.gpuPool,
        maxResidentBytes: geometryGpuPoolLayout(batch.geometryStream.directory).rootBytes } } }));
    progress('budget', budget);
    let fine: MeshData[] | null = null;
    if (identity) {
      const manifest = await (await fetch(options.fineQueryRoot + 'manifest.json')).json();
      if (manifest.sourcePackage?.assetIdentity !== assets.manifest.assetIdentity) throw new Error('SPONZA_WALK_FINE_PROXY_SOURCE');
      const entries = new Map<string, any>(manifest.primitives.map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p.queryProxy]));
      fine = [];
      for (const batch of batches) {
        const key = batch.name.slice('sponza:'.length), entry = entries.get(key);
        if (!entry) throw new Error('SPONZA_WALK_FINE_PROXY_MISSING: ' + key);
        const response = await fetch(options.fineQueryRoot + entry.file);
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (bytes.byteLength !== entry.bytes) throw new Error('SPONZA_WALK_FINE_PROXY_BYTES: ' + key);
        const proxy = await decodeGeometryQueryProxy(batch.geometryStream.directory, bytes, entry.sha256,
          { maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB });
        if (proxy.requestedError !== 0 || proxy.mesh.indices.length / 3 !== batch.geometryStream.directory.sourceTriangleCount) {
          throw new Error('SPONZA_WALK_FINE_PROXY_NOT_LOD0: ' + key);
        }
        fine.push(proxy.mesh);
      }
      progress('fine-proxies', { batches: fine.length, triangles: fine.reduce((n, mesh) => n + mesh.indices.length / 3, 0) });
    }
    const runtimeBudget = { maxResidentBytes: budget.maxResidentBytes, maxPeakBytes: budget.maxPeakBytes, intervalFrames,
      maxConcurrentMigrations, maxCopyPages: 4096, maxCopyBytes: 64 * MiB };
    streamed = await Renderer.create(canvases[0], streamedBatches, identity
      ? { maxBatches: 512, geometryRuntimeBudget: runtimeBudget, occlusion: {} }
      : { maxBatches: 512, antiAliasing: 'fxaa', textureBudgetBytes: 512 * MiB, occlusion: {}, geometryRuntimeBudget: runtimeBudget,
        textureStreaming: { sources, maxCpuBytes: 256 * MiB, maxInFlightBytes: 128 * MiB, maxUploadBytesPerFrame: 2 * MiB },
        sunShadow: { resolution: 2048, focusBounds: bounds as { min: Vec3; max: Vec3 } } });
    const s = streamed, internals = s as unknown as PrivateRenderer;
    if (identity) {
      const referenceBatches: SceneBatch[] = batches.map(({ geometryStream: _, ...batch }, index) => ({ ...batch, mesh: fine![index],
        material: identityMaterial, instances: batch.instances.map(instance => ({ ...instance, color: palette[index] })) }));
      reference = await Renderer.create(canvases[1], referenceBatches, { maxBatches: 512 });
    } else {
      s.setSun({ direction: [0.35, 0.8, 0.4], color: [4, 3.8, 3.5] });
      s.setExposure(options.exposureEV ?? 3);
    }
    s.setGeometryLod({ maxErrorPixels, frustumCulling: true,
      ...(mode === 'present' ? { debug: 'residency' as const, debugSplit: options.debugSplit ?? 0.5 } : {}) });
    const runtimes = batches.map(batch => {
      const entry = internals.batches.find(item => item.scene.name === batch.name);
      if (!entry?.stream) throw new Error('SPONZA_WALK_STREAM_MISSING: ' + batch.name);
      return entry.stream;
    });
    const planners = inputs.map(input => new GeometryStreamCut(input.directory));
    let camera: CapacityCamera = { eye: [...A.eye], target: [...A.target] };
    const view = (model: Float32Array): GeometryStreamView => ({ eye: camera.eye, near: 0.1, projectionScale, maxErrorPixels, model });
    const pace = () => new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    let frameIndex = 0, peakAccountedBytes = 0, maxLiveBytes = 0, withheld = 0;
    // Where wall time goes (diagnostic): Renderer.render calls, frame pacing
    // including frames withheld by stream backpressure, and diagnostics reads.
    const time = { renderCallMs: 0, paceMs: 0, diagnosticsMs: 0, checkMs: 0 };
    const render = async (pose: CapacityCamera, label: string): Promise<Diagnostics> => {
      camera = { eye: [...pose.eye] as Vec3, target: [...pose.target] as Vec3 };
      // A frame withheld by stream backpressure keeps the previous image: retry.
      let jpeg: string | null = null;
      for (let attempt = 0; ; attempt++) {
        const before = s.stats.frames, callStarted = performance.now();
        s.render(camera.eye, camera.target);
        time.renderCallMs += performance.now() - callStarted;
        if (s.errors.length) throw new Error('SPONZA_WALK_RENDER: ' + s.errors.join('; '));
        const rendered = s.stats.frames === before + 1;
        // Read the canvas in the task that rendered it, before it is presented.
        if (rendered && mode === 'present' && sink.frame) jpeg = canvases[0].toDataURL('image/jpeg', 0.92);
        const paceStarted = performance.now();
        await pace();
        time.paceMs += performance.now() - paceStarted;
        if (rendered) break;
        withheld++;
        if (attempt >= 120) throw new Error('SPONZA_WALK_FRAME_WITHHELD');
      }
      frameIndex++;
      const diagnosticsStarted = performance.now();
      const diagnostics = s.diagnostics(), budgetState = diagnostics.geometryRuntimeBudget;
      time.diagnosticsMs += performance.now() - diagnosticsStarted;
      if (!budgetState.enabled) throw new Error('SPONZA_WALK_BUDGET_DISABLED');
      assertCapacityLedger(budgetState);
      peakAccountedBytes = Math.max(peakAccountedBytes, budgetState.accountedBytes); maxLiveBytes = Math.max(maxLiveBytes, budgetState.liveBytes);
      if (jpeg) await sink.frame!({ index: frameIndex, label, jpeg });
      return diagnostics;
    };

    // Busy: admission not idle, pages pending or allocated but missing, requests
    // in flight, owners preparing or retiring. Key: per-stream epoch, slots and
    // upload/eviction counters (any residency change advances one of them).
    const signature = (d: Diagnostics) => {
      const budgetState = d.geometryRuntimeBudget;
      let busy = !budgetState.enabled || budgetState.phase !== 'idle';
      const key = d.geometryStreams.map(stream => `${stream.capacity.epoch}:${stream.pool.slotCount}:${stream.pool.uploads}:${stream.pool.evictions}`).join('|');
      let resident = 0, requested = 0, deferred = 0, missing = 0, uploads = 0, evictions = 0;
      for (const stream of d.geometryStreams) {
        resident += stream.pool.residentPages.length; requested += stream.allocation.requestedPages.length;
        deferred += stream.allocation.budgetDeferredGroups.length; uploads += stream.pool.uploads; evictions += stream.pool.evictions;
        const set = new Set(stream.pool.residentPages);
        let absent = 0;
        for (const page of stream.allocation.pages) if (!set.has(page)) absent++;
        missing += absent;
        if (absent || stream.pool.pendingPages.length || stream.cache.activeRequests || stream.cache.queuedRequests
          || stream.capacity.preparing || stream.capacity.retiringOwners) busy = true;
      }
      const error = d.geometryStreams.find(stream => stream.error || stream.pool.error || stream.selection.error)?.error ?? (budgetState.enabled ? budgetState.error : null);
      return { busy, key, resident, requested, deferred, missing, uploads, evictions, error,
        migrations: budgetState.enabled ? budgetState.migrations : 0, liveBytes: budgetState.enabled ? budgetState.liveBytes : 0 };
    };
    // Pool counters restart when a migration replaces a pool; track evictions by resident-set change instead.
    const residentSets = () => new Map(s.diagnostics().geometryStreams.map((stream, index) => [index, new Set(stream.pool.residentPages)]));
    const droppedEver = new Map<number, Set<number>>();
    let streamingTotals = { loaded: 0, dropped: 0, reloaded: 0 };
    const exchange = (before: Map<number, Set<number>>) => {
      let loaded = 0, dropped = 0, reloaded = 0;
      for (const [index, pages] of residentSets()) {
        const previous = before.get(index) ?? new Set<number>(), history = droppedEver.get(index) ?? new Set<number>();
        for (const page of pages) if (!previous.has(page)) { loaded++; if (history.has(page)) reloaded++; }
        for (const page of previous) if (!pages.has(page)) { dropped++; history.add(page); }
        droppedEver.set(index, history);
      }
      streamingTotals = { loaded: streamingTotals.loaded + loaded, dropped: streamingTotals.dropped + dropped, reloaded: streamingTotals.reloaded + reloaded };
      return { loaded, dropped, reloaded };
    };

    let evidenceBudget = 6;
    const pageCache = new Map<string, { rest: Uint8Array; localTriangles: Uint8Array; sourceVertexIds: Uint32Array }>();
    const decodedCluster = async (index: number, clusterId: number) => {
      const directory = inputs[index].directory, page = directory.pages[directory.clusterPages[clusterId]];
      const key = `${index}:${clusterId}`;
      if (!pageCache.has(key)) {
        const bytes = await batches[index].geometryStream.provider(page, new AbortController().signal);
        const decoded = await decodeVerifiedGeometryPage(page, new Uint8Array(bytes));
        for (const cluster of decoded.clusters) pageCache.set(`${index}:${cluster.id}`, cluster);
      }
      const cluster = pageCache.get(key);
      if (!cluster) throw new Error('SPONZA_WALK_CLUSTER_PAYLOAD: ' + key);
      return cluster;
    };
    /** Test-only attribution of hole pixels: reference object X, streamed image shows
     * something else. The ray through the pixel centre hits full-detail X (fine LOD0
     * proxy) at P, else the pixel is a raster tie. Then the drawn cuts (decoded probed
     * clusters) decide, nearest surface first:
     *  - the shown object's drawn cut is nearest (or within a depth tie): the image
     *    agrees with the cuts, and a neighbour's coarser cut covers X (neighbour);
     *    a ray half a pixel away reaching it is a raster tie at its drawn silhouette;
     *  - X's drawn cut covers the ray but the image shows something behind it or
     *    nothing: a missed GPU draw (mis-culling, a dropped cluster), unless a ray
     *    half a pixel away leaves X's drawn cut (a raster tie at its silhouette);
     *  - X's drawn cut misses the ray: explained when P lies within X's band of the
     *    drawn cut (LOD error), else unexplained.
     * A neighbour's own error is not bounded here; G01 measures LOD error. */
    const classify = async (holeList: readonly number[], ref: Int16Array, str: Int16Array, cuts: readonly Probe[], radii: readonly number[]) => {
      const counts = { explained: 0, neighbour: 0, ties: 0, missedDraws: 0, unclassified: 0 }, work: ClassifyWork = { tests: 0 };
      const forward = unit(sub(camera.target, camera.eye)), right = unit(cross(forward, [0, 1, 0])), up = cross(right, forward);
      const frame: ScreenFrame = { eye: camera.eye, forward, right, up, width, height, focal: projectionScale };
      const models = (object: number) => batches[object].instances.map(instance => instance.model as Float32Array);
      const fullBins = new Map<number, TriangleBins | null>(), cutBins = new Map<number, TriangleBins | null>();
      const full = (object: number) => {
        if (!fullBins.has(object)) {
          const mesh = fine![object], positions = new DataView(mesh.vertices), instanceModels = models(object);
          let bins: TriangleBins | null = null;
          if (mesh.indices.length / 3 * instanceModels.length <= CLASSIFY_TRIANGLES) {
            const list: number[] = [];
            worldTriangles(vertex => [positions.getFloat32(vertex * 80, true), positions.getFloat32(vertex * 80 + 4, true),
              positions.getFloat32(vertex * 80 + 8, true)], mesh.indices, instanceModels, list);
            bins = binTriangles(Float32Array.from(list), frame);
          }
          fullBins.set(object, bins);
        }
        return fullBins.get(object)!;
      };
      const drawn = async (object: number) => {
        if (!cutBins.has(object)) {
          const selected = cuts[object].selectedClusters, directory = inputs[object].directory, instanceModels = models(object);
          const visible = (selected ?? []).filter(cluster => cluster.visible);
          let bins: TriangleBins | null = null;
          if (selected && visible.reduce((n, cluster) => n + directory.clusters[cluster.clusterId].triangleCount, 0) <= CLASSIFY_TRIANGLES) {
            const list: number[] = [];
            for (const cluster of visible) {
              const payload = await decodedCluster(object, cluster.clusterId), rest = new DataView(payload.rest.buffer, payload.rest.byteOffset, payload.rest.byteLength);
              worldTriangles(vertex => [rest.getFloat32(vertex * 80, true), rest.getFloat32(vertex * 80 + 4, true), rest.getFloat32(vertex * 80 + 8, true)],
                payload.localTriangles, [instanceModels[cluster.instanceIndex]], list);
            }
            bins = binTriangles(Float32Array.from(list), frame);
          }
          cutBins.set(object, bins);
        }
        return cutBins.get(object)!;
      };
      for (const p of holeList) {
        if (work.tests > CLASSIFY_TESTS) { counts.unclassified++; continue; }
        const object = ref[p], shown = str[p], x = p % width, y = Math.floor(p / width);
        const fullX = full(object), cutX = await drawn(object);
        const cutShown = shown >= 0 && shown !== object ? await drawn(shown) : undefined;
        if (!fullX || !cutX || cutShown === null) { counts.unclassified++; continue; }
        const hit = castBinned(fullX, frame, x, y, work);
        if (hit === null) { counts.ties++; continue; }
        const drawnHit = castBinned(cutX, frame, x, y, work);
        if (cutShown) {
          const front = (distance: number | null) => distance !== null && (drawnHit === null || distance <= drawnHit * (1 + 2e-4) + 1e-3);
          if (front(castBinned(cutShown, frame, x, y, work))) { counts.neighbour++; continue; }
          if (HALF_PIXEL.some(([dx, dy]) => front(castBinned(cutShown, frame, x + dx, y + dy, work)))) { counts.ties++; continue; }
        }
        if (drawnHit !== null) {
          if (HALF_PIXEL.some(([dx, dy]) => castBinned(cutX, frame, x + dx, y + dy, work) === null)) counts.ties++;
          else counts.missedDraws++;
          continue;
        }
        const direction = rayAt(frame, x, y), point: Vec3 = [camera.eye[0] + direction[0] * hit, camera.eye[1] + direction[1] * hit, camera.eye[2] + direction[2] * hit];
        const depth = dot(sub(point, camera.eye), forward);
        const within = withinBinned(cutX, frame, point, depth, radii[object] * depth / projectionScale, x, y, work);
        if (within === null) counts.unclassified++;
        else if (within) counts.explained++;
      }
      pageCache.clear();
      return { ...counts, tests: work.tests };
    };

    const check = async (label: string, settled: boolean) => {
      if (!reference || !fine) throw new Error('SPONZA_WALK_CHECK_REQUIRES_VERIFY');
      const checkStarted = performance.now();
      reference.render(camera.eye, camera.target);
      if (reference.errors.length) throw new Error('SPONZA_WALK_REFERENCE: ' + reference.errors.join('; '));
      const cuts: Probe[] = [];
      for (let first = 0; first < runtimes.length; first += 32) {
        cuts.push(...await Promise.all(runtimes.slice(first, first + 32).map(stream => stream.runtime.probe())));
      }
      let uncovered = 0, overlapping = 0, partialGroups = 0, visible = 0, fallback = 0, visibleFallback = 0, roots = 0, triangles = 0, cpuMismatches = 0, overflow = 0;
      const radii: number[] = [], mismatchExamples: string[] = [];
      let maxBand = 0;
      for (let index = 0; index < runtimes.length; index++) {
        const cut = cuts[index], directory = inputs[index].directory, instances = batches[index].instances;
        if (cut.overflow || !cut.selectedClusters) { overflow++; radii.push(Infinity); continue; }
        const perInstance = instances.map(() => [] as number[]);
        let band = maxErrorPixels;
        for (const cluster of cut.selectedClusters) {
          perInstance[cluster.instanceIndex].push(cluster.clusterId);
          if (cluster.visible) { visible++; triangles += directory.clusters[cluster.clusterId].triangleCount; }
          if (cluster.fallback) { fallback++; if (cluster.visible) visibleFallback++; }
          if (cluster.root) roots++;
          const refined = directory.clusters[cluster.clusterId].refinedGroup;
          if (refined !== null) band = Math.max(band, planners[index].groupErrorPixels(refined, view(instances[cluster.instanceIndex].model as Float32Array)));
        }
        maxBand = Math.max(maxBand, band);
        const radius = Math.ceil(1.5 * band) + 2;
        radii.push(radius <= 48 ? radius : Infinity);
        const resident = runtimes[index].runtime.pool.residentPageIds();
        perInstance.forEach((ids, instance) => {
          const coverage = geometryCutCoverage(directory, ids);
          uncovered += coverage.uncovered; overlapping += coverage.overlapping; partialGroups += coverage.partialGroups;
          if (settled) {
            const expected = planners[index].plan(resident, view(instances[instance].model as Float32Array)).selectedClusters;
            const actual = [...ids].sort((a, b) => a - b);
            if (actual.length !== expected.length || actual.some((id, k) => id !== expected[k])) {
              cpuMismatches++;
              if (mismatchExamples.length < 4) mismatchExamples.push(`${batches[index].name}#${instance}:${actual.length}/${expected.length}`);
            }
          }
        });
      }
      const [a, b] = await Promise.all([readHdr(reference), readHdr(s)]);
      const ref = labels(a, palette), str = labels(b, palette);
      const result = holes(ref, str, width, height, radii);
      const classified = result.holePixels ? await classify(result.holeList, ref, str, cuts, radii)
        : { explained: 0, neighbour: 0, ties: 0, missedDraws: 0, unclassified: 0, tests: 0 };
      const listed = result.holeList.length;
      const unexplained = result.holePixels - classified.explained - classified.neighbour - classified.ties;
      let evidence: string | null = null;
      if (unexplained > 0 && evidenceBudget > 0 && sink.capture) {
        evidenceBudget--;
        const encode = async (image: { rgb: Float32Array }, mark: boolean, suffix: string) => {
          const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
          const g = canvas.getContext('2d')!, data = g.createImageData(width, height);
          for (let p = 0; p < width * height; p++) {
            const hole = mark && result.interior[p] && str[p] !== ref[p];
            for (let c = 0; c < 3; c++) data.data[p * 4 + c] = hole ? (c === 0 ? 255 : 0) : Math.min(255, Math.round(Math.sqrt(image.rgb[p * 3 + c]) * 255));
            data.data[p * 4 + 3] = 255;
          }
          g.putImageData(data, 0, 0);
          const name = `${label.replace(/[^a-z0-9-]/g, '-')}-${suffix}`;
          await sink.capture!({ label: name, png: canvas.toDataURL('image/png') });
          return name;
        };
        evidence = await encode(b, true, 'streamed');
        await encode(a, false, 'reference');
      }
      const { interior: _interior, holeList: _list, ...counts } = result;
      const checkMs = performance.now() - checkStarted;
      time.checkMs += checkMs;
      return { label, frame: frameIndex, settled, eye: camera.eye.map(v => +v.toFixed(3)), target: camera.target.map(v => +v.toFixed(3)),
        coverage: { uncovered, overlapping, partialGroups }, overflowStreams: overflow, visibleClusters: visible, fallbackClusters: fallback,
        visibleFallbackClusters: visibleFallback,
        rootClusters: roots, drawnTriangles: triangles, cpuCutMismatches: settled ? cpuMismatches : null, mismatchExamples,
        maxBandPixels: +maxBand.toFixed(3), pixels: { ...counts, classifiedHolePixels: listed, explainedHolePixels: classified.explained,
          neighbourCutPixels: classified.neighbour, rasterTiePixels: classified.ties,
          unclassifiedHolePixels: classified.unclassified + (result.holePixels - listed),
          missedDrawPixels: classified.missedDraws, unexplainedHolePixels: unexplained, classifyTests: classified.tests },
        evidence, checkMs: Math.round(checkMs) };
    };
    type Check = Awaited<ReturnType<typeof check>>;
    const checks: Check[] = [];

    // Settle at a pose: render until the signature is idle and unchanged for
    // stableFrames, recording the frame and time of the last change (recovery).
    const settle = async (label: string, pose: CapacityCamera, jumped: boolean) => {
      const startedAt = performance.now(), before = residentSets();
      let stable = 0, lastKey = '', frames = 0, lastChangeFrame = 0, lastChangeMs = 0, demandMetFrame: number | null = null, demandMetMs: number | null = null;
      let peakMissing = 0, peakRequested = 0;
      const series: { frame: number; ms: number; requested: number; missing: number; resident: number }[] = [];
      while (frames < maxSettleFrames) {
        const diagnostics = await render(pose, label); frames++;
        const state = signature(diagnostics), ms = performance.now() - startedAt;
        if (state.error) throw new Error(`SPONZA_WALK_STREAM_ERROR:${label}:${state.error}`);
        peakMissing = Math.max(peakMissing, state.missing); peakRequested = Math.max(peakRequested, state.requested);
        if (frames <= 120 || frames % 10 === 0) series.push({ frame: frames, ms: Math.round(ms), requested: state.requested, missing: state.missing, resident: state.resident });
        if (state.requested === 0 && state.missing === 0) { if (demandMetFrame === null) { demandMetFrame = frames; demandMetMs = Math.round(ms); } }
        else { demandMetFrame = null; demandMetMs = null; }
        if (state.busy || state.key !== lastKey) { stable = 0; lastKey = state.key; lastChangeFrame = frames; lastChangeMs = ms; }
        else if (++stable >= stableFrames) break;
        if (mode === 'verify' && (jumped && (frames === 1 || frames === 3) || checkEvery && stable === 0 && frames % checkEvery === 0)) {
          checks.push(await check(`${label}:${frames === 1 && jumped ? 'first-frame' : 'frame-' + frames}`, false));
        }
      }
      const end = signature(s.diagnostics());
      if (stable < stableFrames) throw new Error('SPONZA_WALK_DID_NOT_SETTLE: ' + JSON.stringify({ label, frames, end }));
      const settledCheck = mode === 'verify' ? await check(`${label}:settled`, true) : null;
      if (settledCheck) checks.push(settledCheck);
      const moved = exchange(before);
      return { label, framesRendered: frames, framesToLastChange: lastChangeFrame, msToLastChange: Math.round(lastChangeMs),
        framesToDemandMet: demandMetFrame, msToDemandMet: demandMetMs, peakRequestedPages: peakRequested, peakMissingAllocatedPages: peakMissing,
        residentPages: end.resident, requestedPages: end.requested, budgetDeferredGroups: end.deferred, migrations: end.migrations,
        liveBytes: end.liveBytes, ...moved, series: mode === 'recovery' ? series : undefined,
        settled: settledCheck };
    };

    const phases: Awaited<ReturnType<typeof settle>>[] = [];
    for (const phase of SEQUENCE) {
      if (phase.kind === 'turn') {
        const steps = Math.max(1, Math.round(turnSeconds * 60));
        for (let step = 1; step < steps; step++) await render(interpolateCapacityCamera(phase.from!, phase.to, step / steps), phase.label + '-turning');
      }
      progress('phase', { label: phase.label });
      phases.push(await settle(phase.label, phase.to, phase.kind !== 'cold'));
      progress('settled', { label: phase.label, framesToLastChange: phases.at(-1)!.framesToLastChange, msToLastChange: phases.at(-1)!.msToLastChange,
        framesToDemandMet: phases.at(-1)!.framesToDemandMet, migrations: phases.at(-1)!.migrations });
    }

    // Continuous walk along the camera path. recovery follows wall-clock time.
    const walkChecks: Check[] = [], walkFrames: { t: number; requested: number; missing: number; resident: number; dtMs: number }[] = [];
    const walkBefore = residentSets(), walkStarted = performance.now();
    let last = walkStarted, walkFrameCount = 0, peakWalkMissing = 0, framesWithMissing = 0;
    for (let t = 0; t <= path.duration; walkFrameCount++) {
      const pose = sampleCameraPath(path, t, false);
      const diagnostics = await render({ eye: pose.eye, target: pose.target }, 'walk');
      const now = performance.now(), state = signature(diagnostics);
      if (state.error) throw new Error('SPONZA_WALK_STREAM_ERROR:walk:' + state.error);
      peakWalkMissing = Math.max(peakWalkMissing, state.missing); if (state.missing) framesWithMissing++;
      if (walkFrameCount % 5 === 0) walkFrames.push({ t: +t.toFixed(3), requested: state.requested, missing: state.missing, resident: state.resident, dtMs: +(now - last).toFixed(2) });
      last = now;
      if (mode === 'verify' && walkFrameCount % walkCheckEvery === 0) walkChecks.push(await check(`walk:${t.toFixed(2)}`, false));
      t = mode === 'recovery' ? (now - walkStarted) / 1000 : t + walkStep;
    }
    const walkExchange = exchange(walkBefore);
    for (let frame = 0; mode === 'present' && frame < (options.presentTailFrames ?? 30); frame++) await render(A, 'tail');
    const diagnostics = s.diagnostics(), finalBudget = diagnostics.geometryRuntimeBudget;
    const errors = [...s.errors, ...(reference?.errors ?? [])];
    const allChecks = [...checks, ...walkChecks];
    const failures: string[] = [];
    if (allChecks.some(item => item.coverage.uncovered || item.coverage.overlapping || item.coverage.partialGroups)) failures.push('CUT_COVERAGE');
    if (allChecks.some(item => item.overflowStreams)) failures.push('PROBE_OVERFLOW');
    if (allChecks.some(item => item.cpuCutMismatches)) failures.push('CPU_CUT_MISMATCH');
    if (allChecks.some(item => item.pixels.unexplainedHolePixels > 0)) failures.push('UNEXPLAINED_HOLES');
    if (allChecks.some(item => item.pixels.missedDrawPixels > 0)) failures.push('MISSED_DRAWS');
    if (errors.length) failures.push('RENDERER_ERRORS');
    if (!streamingTotals.dropped) failures.push('NO_EVICTION_OBSERVED');
    const summary = mode === 'verify' ? {
      checks: allChecks.length, settledChecks: allChecks.filter(item => item.settled).length,
      firstFrameChecks: allChecks.filter(item => item.label.endsWith(':first-frame')).length,
      testedPixels: allChecks.reduce((n, item) => n + item.pixels.testedPixels, 0),
      holePixels: allChecks.reduce((n, item) => n + item.pixels.holePixels, 0),
      explainedHolePixels: allChecks.reduce((n, item) => n + item.pixels.explainedHolePixels, 0),
      neighbourCutPixels: allChecks.reduce((n, item) => n + item.pixels.neighbourCutPixels, 0),
      rasterTiePixels: allChecks.reduce((n, item) => n + item.pixels.rasterTiePixels, 0),
      missedDrawPixels: allChecks.reduce((n, item) => n + item.pixels.missedDrawPixels, 0),
      unclassifiedHolePixels: allChecks.reduce((n, item) => n + item.pixels.unclassifiedHolePixels, 0),
      unexplainedHolePixels: allChecks.reduce((n, item) => n + item.pixels.unexplainedHolePixels, 0),
      settledUnexplainedHolePixels: allChecks.filter(item => item.settled).reduce((n, item) => n + item.pixels.unexplainedHolePixels, 0),
      maxUntestedObjects: Math.max(...allChecks.map(item => item.pixels.untestedObjects)),
      maxFallbackClusters: Math.max(...allChecks.map(item => item.fallbackClusters)),
      maxVisibleFallbackClusters: Math.max(...allChecks.map(item => item.visibleFallbackClusters)),
      maxBandPixels: Math.max(...allChecks.map(item => item.maxBandPixels)),
    } : null;
    return { schemaVersion: 1, fixture: 'intel-sponza-oversubscribed-walk', mode, passed: failures.length === 0, failures,
      adapter: s.info.adapterInfo, viewport: { width, height }, elapsedMs: Math.round(performance.now() - started),
      content: { assetIdentity: assets.manifest.assetIdentity, geometryRoot: assets.root, batches: batches.length, sourceInstances: assets.sourceInstances,
        sourceTriangles: assets.sourceTriangles, omitted: assets.omitted.length, cameraPath: { name: path.name, duration: path.duration, samples: path.samples.length } },
      budget, limits: { maxErrorPixels, stableFrames, maxSettleFrames, checkEvery, walkCheckEvery, walkStep, turnSeconds, intervalFrames,
        maxConcurrentMigrations, admission: 'default (work-conserving)' },
      scope: { occlusion: 'two-phase Hi-Z enabled', shading: identity ? 'identity unlit, no FXAA, HDR rgba16f readback' : 'textured PBR, sun shadow, FXAA',
        timing: 'diagnostic only: other agents share the GPU' + (mode === 'verify' ? '; probes and readbacks stall every checked frame' : '') },
      phases, walk: { frames: walkFrameCount, wallMs: Math.round(performance.now() - walkStarted), peakMissingAllocatedPages: peakWalkMissing,
        framesWithMissingAllocatedPages: framesWithMissing, ...walkExchange, samples: walkFrames },
      streaming: streamingTotals, summary, checks, walkChecks, peakAccountedBytes, maxLiveBytes, withheldFrames: withheld,
      wallTime: { renderCallMs: Math.round(time.renderCallMs), paceMs: Math.round(time.paceMs), diagnosticsMs: Math.round(time.diagnosticsMs),
        checkMs: Math.round(time.checkMs), frames: frameIndex },
      finalBudget, backpressureFrames: s.stats.streamBackpressureFrames, ownedGpuBytes: s.stats.ownedGpuBytes, errors };
  } finally {
    streamed?.dispose(); reference?.dispose();
    if (streamed) await (streamed as unknown as PrivateRenderer).geometryCapacity?.dispose().catch(() => undefined);
    await new Promise(resolve => setTimeout(resolve, 0));
    panel.remove();
  }
}
