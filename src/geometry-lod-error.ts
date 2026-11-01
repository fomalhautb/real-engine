import type { GeometryHierarchy } from './geometry-hierarchy-types';
import type { GeometryCutCoverage, GeometryCutWatertightness } from './geometry-stream-coverage';

/** Pure CPU measurement of LOD error against a full-detail reference: binary
 * identity coverage, exact Euclidean distance transforms, silhouette and
 * popping counts, an incremental edge ledger and a coverage checker that
 * reproduce geometryCutWatertightness / geometryCutCoverage at per-frame
 * cost, and a cluster-culled CPU ray caster for surface gaps. No renderer or
 * GPU types; nothing here runs in the runtime frame loop. Every workload is
 * bounded by the image size or by the clusters passed in. */
export type Vec3 = readonly [number, number, number];

export const LOD_LABEL_BACKGROUND = 0;
export const LOD_LABEL_OBJECT = 1;
export const LOD_LABEL_UNKNOWN = 255;
const MAX_PIXELS = 1 << 24;
const EDGE_KEY = 0x4000000;
const MAX_LEDGER_TRIANGLES = 4_000_000;

function checkImage(width: number, height: number, length: number) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1
    || width * height > MAX_PIXELS || length !== width * height) throw new RangeError('LOD_ERROR_IMAGE');
}

/** Binary identity coverage from linear RGB triples: object or background when
 * the L1 distance to that colour is below tolerance (nearest wins), else unknown. */
export function lodCoverageLabels(rgb: Float32Array, object: readonly number[], background: readonly number[], tolerance = 0.03): Uint8Array {
  if (rgb.length % 3 || rgb.length / 3 > MAX_PIXELS || object.length < 3 || background.length < 3
    || !(tolerance > 0)) throw new RangeError('LOD_ERROR_LABELS');
  const labels = new Uint8Array(rgb.length / 3);
  for (let p = 0, i = 0; p < labels.length; p++, i += 3) {
    const r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
    const o = Math.abs(r - object[0]) + Math.abs(g - object[1]) + Math.abs(b - object[2]);
    const k = Math.abs(r - background[0]) + Math.abs(g - background[1]) + Math.abs(b - background[2]);
    labels[p] = o < tolerance && o <= k ? LOD_LABEL_OBJECT : k < tolerance ? LOD_LABEL_BACKGROUND : LOD_LABEL_UNKNOWN;
  }
  return labels;
}

const FAR = 1e20;
/** Felzenszwalb-Huttenlocher lower envelope: exact 1D squared distance. */
function envelope(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array) {
  let k = 0;
  v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const delta = q - v[k];
    d[q] = delta * delta + f[v[k]];
  }
}

/** Exact squared Euclidean distance, in pixels, from every pixel centre to the
 * nearest pixel centre labelled `target`. With outsideIsTarget the ring of
 * pixel centres just outside the image also counts as target: beyond the
 * viewport the label is unknown, so it may be the nearest opposite label. */
export function lodSquaredDistanceToLabel(labels: Uint8Array, width: number, height: number, target: number, outsideIsTarget: boolean): Float64Array {
  checkImage(width, height, labels.length);
  const w = width + 2, h = height + 2, grid = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const inside = x > 0 && y > 0 && x <= width && y <= height;
    const seed = inside ? labels[(y - 1) * width + (x - 1)] === target : outsideIsTarget;
    grid[y * w + x] = seed ? 0 : FAR;
  }
  const n = Math.max(w, h), f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = grid[y * w + x];
    envelope(f, h, d, v, z);
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = grid[y * w + x];
    envelope(f, w, d, v, z);
    for (let x = 0; x < w; x++) grid[y * w + x] = d[x];
  }
  const result = new Float64Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) result[y * width + x] = grid[(y + 1) * w + x + 1];
  return result;
}

/** Reference coverage with its two distance fields, shared by every compared frame. */
export interface LodReferenceSilhouette {
  readonly width: number;
  readonly height: number;
  readonly labels: Uint8Array;
  /** Squared distance to the nearest object (or out-of-image) pixel centre. */
  readonly toObject: Float64Array;
  /** Squared distance to the nearest background (or out-of-image) pixel centre. */
  readonly toBackground: Float64Array;
}
export function lodReferenceSilhouette(labels: Uint8Array, width: number, height: number): LodReferenceSilhouette {
  return { width, height, labels, toObject: lodSquaredDistanceToLabel(labels, width, height, LOD_LABEL_OBJECT, true),
    toBackground: lodSquaredDistanceToLabel(labels, width, height, LOD_LABEL_BACKGROUND, true) };
}

export interface LodSilhouetteComparison {
  /** Pixels whose binary coverage differs between the LOD and reference frames. */
  readonly mismatched: number;
  /** Covered only by the LOD frame / only by the reference frame. */
  readonly lodOnly: number;
  readonly referenceOnly: number;
  /** Pixels unknown in either frame; excluded from every other count. */
  readonly unknown: number;
  /** Mismatched pixel indices and, per pixel, the distance in pixels from its
   * centre to the nearest reference pixel centre of the opposite label (or the
   * out-of-image ring): how far the LOD silhouette is displaced there. */
  readonly pixels: Uint32Array;
  readonly distances: Float32Array;
}
export function lodCompareSilhouette(reference: LodReferenceSilhouette, lod: Uint8Array): LodSilhouetteComparison {
  checkImage(reference.width, reference.height, lod.length);
  const ref = reference.labels;
  let mismatched = 0, lodOnly = 0, referenceOnly = 0, unknown = 0;
  for (let p = 0; p < lod.length; p++) {
    if (ref[p] === LOD_LABEL_UNKNOWN || lod[p] === LOD_LABEL_UNKNOWN) { unknown++; continue; }
    if (ref[p] !== lod[p]) { mismatched++; if (lod[p] === LOD_LABEL_OBJECT) lodOnly++; else referenceOnly++; }
  }
  const pixels = new Uint32Array(mismatched), distances = new Float32Array(mismatched);
  for (let p = 0, i = 0; p < lod.length; p++) {
    if (ref[p] === LOD_LABEL_UNKNOWN || lod[p] === LOD_LABEL_UNKNOWN || ref[p] === lod[p]) continue;
    pixels[i] = p;
    distances[i++] = Math.sqrt(ref[p] === LOD_LABEL_OBJECT ? reference.toBackground[p] : reference.toObject[p]);
  }
  return { mismatched, lodOnly, referenceOnly, unknown, pixels, distances };
}

/** Secant of the angle between a pixel-centre ray and the view axis for a
 * symmetric perspective with vertical field of view fovY. */
export function lodPixelSecant(x: number, y: number, width: number, height: number, fovY: number): number {
  const t = Math.tan(fovY / 2), nx = (((x + 0.5) / width) * 2 - 1) * t * width / height, ny = (1 - ((y + 0.5) / height) * 2) * t;
  return Math.sqrt(1 + nx * nx + ny * ny);
}

/** Unit ray through a pixel centre for lookAt(eye, target, +Y) and a symmetric
 * perspective; forward is the view axis, so view depth = dot(p - eye, forward). */
export function lodPixelRay(eye: Vec3, target: Vec3, x: number, y: number, width: number, height: number, fovY: number) {
  const f = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const r = normalize([-f[2], 0, f[0]]); // cross(forward, +Y)
  const u: Vec3 = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const t = Math.tan(fovY / 2), nx = (((x + 0.5) / width) * 2 - 1) * t * width / height, ny = (1 - ((y + 0.5) / height) * 2) * t;
  const direction = normalize([f[0] + r[0] * nx + u[0] * ny, f[1] + r[1] * nx + u[1] * ny, f[2] + r[2] * nx + u[2] * ny]);
  return { direction, forward: f };
}
function normalize(a: readonly number[]): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (!(l > 0)) throw new RangeError('LOD_ERROR_DIRECTION');
  return [a[0] / l, a[1] / l, a[2] / l];
}

export interface LodLabelPopping {
  readonly lodChanged: number;
  readonly referenceChanged: number;
  /** Label changed between consecutive LOD frames but not between the
   * consecutive reference frames: change caused by LOD, not by motion. */
  readonly lodOnly: number;
  /** Reference changed but the LOD frame did not. */
  readonly referenceOnly: number;
}
/** Consecutive-frame label changes; a pixel unknown in any of the four frames is skipped. */
export function lodLabelPopping(previousLod: Uint8Array, lod: Uint8Array, previousReference: Uint8Array, reference: Uint8Array): LodLabelPopping {
  const n = lod.length;
  if (previousLod.length !== n || previousReference.length !== n || reference.length !== n) throw new RangeError('LOD_ERROR_IMAGE');
  let lodChanged = 0, referenceChanged = 0, lodOnly = 0, referenceOnly = 0;
  for (let p = 0; p < n; p++) {
    const a = previousLod[p], b = lod[p], c = previousReference[p], d = reference[p];
    if (a === LOD_LABEL_UNKNOWN || b === LOD_LABEL_UNKNOWN || c === LOD_LABEL_UNKNOWN || d === LOD_LABEL_UNKNOWN) continue;
    const l = a !== b, r = c !== d;
    if (l) lodChanged++;
    if (r) referenceChanged++;
    if (l && !r) lodOnly++;
    if (r && !l) referenceOnly++;
  }
  return { lodChanged, referenceChanged, lodOnly, referenceOnly };
}

/** Pixels whose known labels differ (e.g. the same camera drawn with two cuts). */
export function lodLabelDifference(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length) throw new RangeError('LOD_ERROR_IMAGE');
  let count = 0;
  for (let p = 0; p < a.length; p++) if (a[p] !== b[p] && a[p] !== LOD_LABEL_UNKNOWN && b[p] !== LOD_LABEL_UNKNOWN) count++;
  return count;
}

export interface LodDistribution {
  readonly count: number;
  readonly mean: number;
  readonly p50: number;
  readonly p90: number;
  readonly p99: number;
  readonly max: number;
}
/** Nearest-rank percentiles; an empty sample reports count 0 and zeros. */
export function lodDistribution(values: ArrayLike<number>): LodDistribution {
  const sorted = Float64Array.from(values).sort();
  const n = sorted.length;
  if (!n) return { count: 0, mean: 0, p50: 0, p90: 0, p99: 0, max: 0 };
  let sum = 0;
  for (let i = 0; i < n; i++) sum += sorted[i];
  const rank = (q: number) => sorted[Math.min(n - 1, Math.max(0, Math.ceil(q * n) - 1))];
  return { count: n, mean: sum / n, p50: rank(0.5), p90: rank(0.9), p99: rank(0.99), max: sorted[n - 1] };
}

/** Minimal DAG structure shared by GeometryHierarchy and a stream directory. */
export interface LodCutStructure {
  readonly clusters: readonly { readonly level: number; readonly group: number }[];
  readonly groups: readonly { readonly parents: readonly number[]; readonly terminal: boolean }[];
}

/** geometryCutCoverage semantics with the level order, parent lists and
 * scratch arrays prepared once, so a per-frame check costs O(clusters +
 * parent references) without revalidating the directory. Construction
 * validates the same structural invariants that function checks. */
export class LodCutCoverageChecker {
  private readonly order: Uint32Array;
  private readonly level: Int32Array;
  private readonly group: Int32Array;
  private readonly terminal: Uint8Array;
  private readonly parentStart: Uint32Array;
  private readonly parentList: Uint32Array;
  private readonly chosen: Uint8Array;
  private readonly represented: Uint32Array;
  private readonly groupState: Float64Array;

  constructor(structure: LodCutStructure) {
    const count = structure.clusters.length, groups = structure.groups.length;
    if (count < 1 || groups < 1 || count > 1 << 22 || groups > 1 << 22) throw new RangeError('LOD_ERROR_COVERAGE_STRUCTURE');
    this.level = new Int32Array(count); this.group = new Int32Array(count);
    for (let id = 0; id < count; id++) {
      const cluster = structure.clusters[id];
      if (!Number.isSafeInteger(cluster.level) || cluster.level < 0 || !Number.isSafeInteger(cluster.group)
        || cluster.group < 0 || cluster.group >= groups) throw new RangeError('LOD_ERROR_COVERAGE_STRUCTURE');
      this.level[id] = cluster.level; this.group[id] = cluster.group;
    }
    this.terminal = new Uint8Array(groups); this.parentStart = new Uint32Array(groups + 1);
    let references = 0;
    for (let g = 0; g < groups; g++) {
      this.terminal[g] = structure.groups[g].terminal ? 1 : 0;
      references += structure.groups[g].terminal ? 0 : structure.groups[g].parents.length;
      this.parentStart[g + 1] = references;
    }
    this.parentList = new Uint32Array(references);
    for (let g = 0, cursor = 0; g < groups; g++) if (!this.terminal[g]) for (const parent of structure.groups[g].parents) {
      if (!Number.isSafeInteger(parent) || parent < 0 || parent >= count) throw new RangeError('LOD_ERROR_COVERAGE_STRUCTURE');
      this.parentList[cursor++] = parent;
    }
    for (let id = 0; id < count; id++) {
      const g = this.group[id];
      for (let k = this.parentStart[g]; k < this.parentStart[g + 1]; k++) {
        if (this.level[this.parentList[k]] <= this.level[id]) throw new RangeError('GEOMETRY_CUT_COVERAGE_LEVEL_ORDER');
      }
    }
    this.order = Uint32Array.from({ length: count }, (_, id) => id)
      .sort((a, b) => this.level[b] - this.level[a] || a - b);
    this.chosen = new Uint8Array(count); this.represented = new Uint32Array(count); this.groupState = new Float64Array(groups);
  }

  check(selected: Iterable<number>): GeometryCutCoverage {
    const count = this.level.length, chosen = this.chosen;
    chosen.fill(0); this.groupState.fill(-1);
    let selectedClusters = 0;
    for (const id of selected) {
      if (!Number.isSafeInteger(id) || id < 0 || id >= count) throw new RangeError('GEOMETRY_CUT_COVERAGE_CLUSTER');
      if (chosen[id]) throw new RangeError('GEOMETRY_CUT_COVERAGE_DUPLICATE');
      chosen[id] = 1; selectedClusters++;
    }
    let partialGroups = 0, lod0Clusters = 0, uncovered = 0, overlapping = 0;
    const represented = this.represented, state = this.groupState;
    for (let i = 0; i < count; i++) {
      const id = this.order[i], g = this.group[id];
      let above = 0;
      if (!this.terminal[g]) {
        if (state[g] < 0) {
          let low = Infinity, high = 0;
          for (let k = this.parentStart[g]; k < this.parentStart[g + 1]; k++) {
            const value = represented[this.parentList[k]];
            if (value < low) low = value;
            if (value > high) high = value;
          }
          if (low !== high) partialGroups++;
          state[g] = Number.isFinite(low) ? low : 0;
        }
        above = state[g];
      }
      represented[id] = above + chosen[id];
      if (this.level[id] === 0) {
        lod0Clusters++;
        if (represented[id] === 0) uncovered++;
        else if (represented[id] > 1) overlapping++;
      }
    }
    return { lod0Clusters, uncovered, overlapping, partialGroups, selectedClusters };
  }
}

/** geometryCutWatertightness counts maintained incrementally across cuts:
 * positions are welded once by bit pattern, each cluster's directed edges are
 * cached on first use, and update() applies only the clusters that entered or
 * left the cut. Cut IDs must be unique (the reference function would count a
 * duplicate's edges twice; here it is an error). */
export class LodCutEdgeLedger {
  private readonly weld: Uint32Array;
  private readonly edges = new Map<number, Float64Array>();
  private readonly counts = new Map<number, number>();
  private current = new Set<number>();
  private boundary = 0;
  private repeated = 0;
  private triangleCount = 0;

  constructor(private readonly hierarchy: Pick<GeometryHierarchy, 'source' | 'clusters'>) {
    const source = hierarchy.source, view = new DataView(source.vertices), canonical = new Map<string, number>();
    if (!Number.isSafeInteger(source.vertexCount) || source.vertexCount < 1 || source.vertexCount >= EDGE_KEY
      || source.vertices.byteLength < source.vertexCount * 80) throw new RangeError('LOD_ERROR_LEDGER_SOURCE');
    this.weld = new Uint32Array(source.vertexCount);
    for (let vertex = 0; vertex < source.vertexCount; vertex++) {
      const offset = vertex * 80, key = `${view.getUint32(offset, true)},${view.getUint32(offset + 4, true)},${view.getUint32(offset + 8, true)}`;
      let id = canonical.get(key);
      if (id === undefined) { id = canonical.size; canonical.set(key, id); }
      this.weld[vertex] = id;
    }
  }

  get totals(): GeometryCutWatertightness {
    return { triangles: this.triangleCount, boundaryEdges: this.boundary, repeatedEdges: this.repeated };
  }

  update(selected: Iterable<number>): GeometryCutWatertightness {
    const next = new Set<number>();
    let triangles = 0;
    for (const id of selected) {
      const cluster = this.hierarchy.clusters[id];
      if (!Number.isSafeInteger(id) || !cluster) throw new RangeError('GEOMETRY_CUT_WATERTIGHT_CLUSTER');
      if (next.has(id)) throw new RangeError('LOD_ERROR_LEDGER_DUPLICATE');
      next.add(id); triangles += cluster.triangles.length / 3;
    }
    if (triangles > MAX_LEDGER_TRIANGLES) throw new RangeError('GEOMETRY_CUT_WATERTIGHT_BUDGET');
    for (const id of this.current) if (!next.has(id)) this.apply(id, -1);
    for (const id of next) if (!this.current.has(id)) this.apply(id, 1);
    this.current = next;
    return this.totals;
  }

  /** Use counts, in the current cut, of one cluster triangle's three welded
   * directed edges and of their reverses (1 and 1 on a closed, consistently
   * oriented surface; a repeated forward edge marks a fold or overlap). */
  edgeUse(id: number, triangle: number): { forward: number[]; reverse: number[] } {
    const cluster = this.hierarchy.clusters[id];
    if (!cluster || !Number.isSafeInteger(triangle) || triangle < 0 || triangle * 3 >= cluster.triangles.length) throw new RangeError('LOD_ERROR_LEDGER_TRIANGLE');
    const keys = this.keys(id), forward: number[] = [], reverse: number[] = [];
    for (let k = 0; k < 3; k++) {
      const key = keys[triangle * 3 + k], a = Math.floor(key / EDGE_KEY), b = key - a * EDGE_KEY;
      forward.push(this.counts.get(key) ?? 0); reverse.push(this.counts.get(b * EDGE_KEY + a) ?? 0);
    }
    return { forward, reverse };
  }

  private keys(id: number): Float64Array {
    let keys = this.edges.get(id);
    if (keys) return keys;
    const cluster = this.hierarchy.clusters[id], triangles = cluster.triangles, vertices = cluster.vertices;
    keys = new Float64Array(triangles.length);
    for (let t = 0; t < triangles.length; t += 3) {
      const a = this.weld[vertices[triangles[t]]], b = this.weld[vertices[triangles[t + 1]]], c = this.weld[vertices[triangles[t + 2]]];
      keys[t] = a * EDGE_KEY + b; keys[t + 1] = b * EDGE_KEY + c; keys[t + 2] = c * EDGE_KEY + a;
    }
    this.edges.set(id, keys);
    return keys;
  }

  private apply(id: number, delta: 1 | -1) {
    const keys = this.keys(id);
    for (let i = 0; i < keys.length; i++) this.bump(keys[i], delta);
    this.triangleCount += delta * keys.length / 3;
  }

  private bump(key: number, delta: number) {
    const a = Math.floor(key / EDGE_KEY), b = key - a * EDGE_KEY, reverse = b * EDGE_KEY + a;
    const before = this.counts.get(key) ?? 0, after = before + delta;
    if (after < 0) throw new Error('LOD_ERROR_LEDGER_UNDERFLOW');
    this.repeated += (after > 1 ? 1 : 0) - (before > 1 ? 1 : 0);
    if (reverse !== key) {
      // Boundary: a present directed edge whose reverse is absent, counted for this key and its reverse.
      const other = this.counts.get(reverse) ?? 0;
      this.boundary += (after > 0 && other === 0 ? 1 : 0) - (before > 0 && other === 0 ? 1 : 0)
        + (other > 0 && after === 0 ? 1 : 0) - (other > 0 && before === 0 ? 1 : 0);
    }
    if (after) this.counts.set(key, after); else this.counts.delete(key);
  }
}

export interface LodRayHit {
  readonly t: number;
  readonly cluster: number;
  /** Triangle index within the cluster. */
  readonly triangle: number;
  readonly front: boolean;
}

/** CPU ray casting and closest-point queries over source triangles of chosen
 * clusters, culled by per-cluster position bounds (computed from the cluster's
 * own vertices, cached). Test and record diagnostics only. */
export class LodClusterSurface {
  private readonly positions: Float32Array;
  private readonly boxes = new Map<number, Float64Array>();

  constructor(private readonly hierarchy: Pick<GeometryHierarchy, 'source' | 'clusters'>) {
    const source = hierarchy.source, view = new DataView(source.vertices);
    if (!Number.isSafeInteger(source.vertexCount) || source.vertexCount < 1
      || source.vertices.byteLength < source.vertexCount * 80) throw new RangeError('LOD_ERROR_SURFACE_SOURCE');
    this.positions = new Float32Array(source.vertexCount * 3);
    for (let v = 0; v < source.vertexCount; v++) for (let axis = 0; axis < 3; axis++) {
      this.positions[v * 3 + axis] = view.getFloat32(v * 80 + axis * 4, true);
    }
  }

  private box(id: number): Float64Array {
    let box = this.boxes.get(id);
    if (box) return box;
    const cluster = this.hierarchy.clusters[id];
    if (!cluster) throw new RangeError('LOD_ERROR_SURFACE_CLUSTER');
    box = new Float64Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (const vertex of cluster.vertices) for (let axis = 0; axis < 3; axis++) {
      const value = this.positions[vertex * 3 + axis];
      if (value < box[axis]) box[axis] = value;
      if (value > box[axis + 3]) box[axis + 3] = value;
    }
    this.boxes.set(id, box);
    return box;
  }

  /** Nearest hit with t > near (direction must be unit length). front reports
   * whether the triangle faces the ray origin (counter-clockwise seen from it,
   * the winding the renderer draws with back-face culling); frontOnly ignores
   * back-facing triangles, i.e. what a culled rasterizer can show. */
  raycast(clusters: ArrayLike<number>, origin: Vec3, direction: Vec3, near = 0, frontOnly = false): LodRayHit | null {
    const candidates: [number, number][] = [];
    for (let i = 0; i < clusters.length; i++) {
      const box = this.box(clusters[i]);
      let enter = near, exit = Infinity;
      for (let axis = 0; axis < 3; axis++) {
        if (direction[axis] === 0) {
          // Parallel to this slab: inside it for every t, or never.
          if (origin[axis] < box[axis] || origin[axis] > box[axis + 3]) { exit = -Infinity; break; }
          continue;
        }
        const inverse = 1 / direction[axis];
        let t0 = (box[axis] - origin[axis]) * inverse, t1 = (box[axis + 3] - origin[axis]) * inverse;
        if (t0 > t1) { const swap = t0; t0 = t1; t1 = swap; }
        if (t0 > enter) enter = t0;
        if (t1 < exit) exit = t1;
      }
      // A small slack keeps hits on box faces (flat clusters) from being culled by rounding.
      if (enter <= exit + 1e-6) candidates.push([enter, clusters[i]]);
    }
    candidates.sort((a, b) => a[0] - b[0]);
    const p = this.positions, [ox, oy, oz] = origin, [dx, dy, dz] = direction;
    let best: LodRayHit | null = null;
    for (const [enter, id] of candidates) {
      if (best && enter - 1e-6 > best.t) break;
      const cluster = this.hierarchy.clusters[id], triangles = cluster.triangles, vertices = cluster.vertices;
      for (let k = 0; k < triangles.length; k += 3) {
        const a = vertices[triangles[k]] * 3, b = vertices[triangles[k + 1]] * 3, c = vertices[triangles[k + 2]] * 3;
        const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
        const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
        const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        // det = -dot(direction, cross(e1, e2)): positive when the triangle faces the origin.
        if (Math.abs(det) < 1e-14 || (frontOnly && det < 0)) continue;
        const inv = 1 / det, sx = ox - p[a], sy = oy - p[a + 1], sz = oz - p[a + 2];
        const u = (sx * px + sy * py + sz * pz) * inv;
        if (u < 0 || u > 1) continue;
        const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (t > near && (!best || t < best.t)) best = { t, cluster: id, triangle: k / 3, front: det > 0 };
      }
    }
    return best;
  }

  /** Euclidean distance from a point to the nearest triangle of the clusters. */
  distance(clusters: ArrayLike<number>, point: Vec3): { distance: number; cluster: number } | null {
    const candidates: [number, number][] = [];
    for (let i = 0; i < clusters.length; i++) {
      const box = this.box(clusters[i]);
      let squared = 0;
      for (let axis = 0; axis < 3; axis++) {
        const delta = Math.max(box[axis] - point[axis], 0, point[axis] - box[axis + 3]);
        squared += delta * delta;
      }
      candidates.push([squared, clusters[i]]);
    }
    candidates.sort((a, b) => a[0] - b[0]);
    const p = this.positions;
    let best = Infinity, bestCluster = -1;
    for (const [bound, id] of candidates) {
      if (bound > best) break;
      const cluster = this.hierarchy.clusters[id], triangles = cluster.triangles, vertices = cluster.vertices;
      for (let k = 0; k < triangles.length; k += 3) {
        const squared = pointTriangleSquared(point, p, vertices[triangles[k]] * 3, vertices[triangles[k + 1]] * 3, vertices[triangles[k + 2]] * 3);
        if (squared < best) { best = squared; bestCluster = id; }
      }
    }
    return bestCluster < 0 ? null : { distance: Math.sqrt(best), cluster: bestCluster };
  }
}

/** Squared closest-point distance (Ericson, Real-Time Collision Detection 5.1.5). */
function pointTriangleSquared(point: Vec3, p: Float32Array, a: number, b: number, c: number): number {
  const ax = p[a], ay = p[a + 1], az = p[a + 2];
  const abx = p[b] - ax, aby = p[b + 1] - ay, abz = p[b + 2] - az, acx = p[c] - ax, acy = p[c + 1] - ay, acz = p[c + 2] - az;
  const apx = point[0] - ax, apy = point[1] - ay, apz = point[2] - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let u = 0, v = 0; // closest = a + ab*u + ac*v
  if (d1 <= 0 && d2 <= 0) { u = 0; v = 0; }
  else {
    const bpx = point[0] - p[b], bpy = point[1] - p[b + 1], bpz = point[2] - p[b + 2];
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = point[0] - p[c], cpy = point[1] - p[c + 1], cpz = point[2] - p[c + 2];
    const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) { u = 1; v = 0; }
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) { u = d1 / (d1 - d3); v = 0; }
    else if (d6 >= 0 && d5 <= d6) { u = 0; v = 1; }
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) { u = 0; v = d2 / (d2 - d6); }
    else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); u = 1 - w; v = w; }
    else { const denominator = 1 / (va + vb + vc); u = vb * denominator; v = vc * denominator; }
  }
  const x = apx - abx * u - acx * v, y = apy - aby * u - acy * v, z = apz - abz * u - acz * v;
  return x * x + y * y + z * z;
}

/** Sorted unique cluster IDs, the canonical form for comparing cuts. */
export function lodSortedCut(ids: Iterable<number>): Uint32Array {
  const sorted = Uint32Array.from(ids).sort();
  for (let i = 1; i < sorted.length; i++) if (sorted[i] === sorted[i - 1]) throw new RangeError('LOD_ERROR_CUT_DUPLICATE');
  return sorted;
}
/** Clusters entering and leaving between two sorted cuts. */
export function lodCutDifference(previous: Uint32Array, next: Uint32Array): { added: number; removed: number } {
  let i = 0, j = 0, added = 0, removed = 0;
  while (i < previous.length || j < next.length) {
    if (j >= next.length || (i < previous.length && previous[i] < next[j])) { removed++; i++; }
    else if (i >= previous.length || next[j] < previous[i]) { added++; j++; }
    else { i++; j++; }
  }
  return { added, removed };
}

export interface LodCutView {
  readonly eye: Vec3;
  readonly projectionScale: number;
  readonly near: number;
  readonly maxErrorPixels: number;
}
/** selectGeometryCut's projected group error: group.error * projectionScale /
 * max(near, distance from the eye to the group bounds). Identity model. */
export function lodGroupErrorPixels(group: { readonly error: number; readonly bounds: { readonly min: readonly number[]; readonly max: readonly number[] } },
  view: LodCutView): number {
  let squared = 0;
  for (let axis = 0; axis < 3; axis++) {
    const delta = Math.max(group.bounds.min[axis] - view.eye[axis], 0, view.eye[axis] - group.bounds.max[axis]);
    squared += delta * delta;
  }
  return group.error * view.projectionScale / Math.max(view.near, Math.sqrt(squared));
}

export interface LodCutComparison {
  /** In the observed cut but not the expected one, and the reverse. */
  readonly extra: number;
  readonly missing: number;
  /** Differing clusters whose group or refined group sits within relTolerance
   * of the threshold: a float32-versus-float64 decision, not a selection bug. */
  readonly borderline: number;
  readonly unexplained: number;
  readonly examples: readonly number[];
}
/** Compares an observed cut with the CPU reference cut for the same view. */
export function lodCompareCuts(hierarchy: Pick<GeometryHierarchy, 'clusters' | 'groups'>, observed: Uint32Array, expected: Uint32Array,
  view: LodCutView, relTolerance = 1e-4): LodCutComparison {
  if (!(view.maxErrorPixels > 0) || !(relTolerance >= 0)) throw new RangeError('LOD_ERROR_CUT_VIEW');
  const near = (groupId: number | null) => {
    if (groupId === null) return false;
    const group = hierarchy.groups[groupId];
    if (group.terminal) return false;
    return Math.abs(lodGroupErrorPixels(group, view) / view.maxErrorPixels - 1) <= relTolerance;
  };
  let i = 0, j = 0, extra = 0, missing = 0, borderline = 0, unexplained = 0;
  const examples: number[] = [];
  const classify = (id: number) => {
    const cluster = hierarchy.clusters[id];
    if (near(cluster.group) || near(cluster.refinedGroup)) borderline++;
    else { unexplained++; if (examples.length < 8) examples.push(id); }
  };
  while (i < observed.length || j < expected.length) {
    if (j >= expected.length || (i < observed.length && observed[i] < expected[j])) { extra++; classify(observed[i++]); }
    else if (i >= observed.length || expected[j] < observed[i]) { missing++; classify(expected[j++]); }
    else { i++; j++; }
  }
  return { extra, missing, borderline, unexplained, examples };
}
