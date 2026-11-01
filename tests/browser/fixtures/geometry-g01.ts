import type { InstanceState, MaterialBinding, SceneBatch, StreamedSceneBatch } from '../../../src/contracts';
import { G01Trajectory, G01_DETAIL_SPECTRUM, G01_NEAR, G01_SEED, G01_STEP_SECONDS, G01_VERTICAL_FOV, g01Mesh, type G01Pose } from '../../../src/geometry-g01-scene';
import { GEOMETRY_GPU_VERTEX_BYTES } from '../../../src/geometry-gpu-page';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import { buildGeometryHierarchy, selectGeometryCut } from '../../../src/geometry-hierarchy';
import type { GeometryHierarchy } from '../../../src/geometry-hierarchy-types';
import {
  LOD_LABEL_BACKGROUND, LOD_LABEL_OBJECT, LOD_LABEL_UNKNOWN, LodClusterSurface, LodCutCoverageChecker, LodCutEdgeLedger,
  lodCompareCuts, lodCompareSilhouette, lodCoverageLabels, lodCutDifference, lodDistribution, lodLabelDifference, lodLabelPopping,
  lodPixelRay, lodPixelSecant, lodReferenceSilhouette, lodSortedCut, type LodDistribution, type LodReferenceSilhouette,
} from '../../../src/geometry-lod-error';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { geometryCutCoverage, geometryCutWatertightness } from '../../../src/geometry-stream-coverage';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import type { GeometryPageProvider, GeometryStreamDirectory } from '../../../src/geometry-stream-types';
import type { GpuGeometryStream } from '../../../src/gpu-geometry-stream';
import { identity } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

/** G01: one closed high-poly procedural mesh on a fixed zoom-and-orbit
 * trajectory, drawn by the resident GPU LOD path and the paged stream path and
 * compared, frame by frame, with a full-detail reference renderer. Every
 * readback here is test-only diagnostics; the runtime never reads geometry or
 * images back to choose what it draws. */
export interface G01FixtureOptions {
  readonly around: number;
  readonly across: number;
  readonly width: number;
  readonly height: number;
  readonly maxErrorPixels?: number;
  /** Development only: stop after this many trajectory steps; the record uses all of them. */
  readonly steps?: number;
  /** Surface-gap samples every depthEvery steps, depthSamples pixels each. */
  readonly depthEvery?: number;
  readonly depthSamples?: number;
  /** Fast coverage/edge counts are re-derived with the reference functions every crossCheckEvery steps. */
  readonly crossCheckEvery?: number;
  readonly evidenceSteps?: readonly number[];
  /** Send a JPEG of each step to window.__g01VideoFrame (exposed by the spec). */
  readonly video?: boolean;
}

const NAME = 'g01-mesh';
const OBJECT = [0.9, 0.42, 0.18];
const BACKGROUND = [0.0035, 0.0055, 0.0065];
const BUILD = { maxClusterTriangles: 96, groupSize: 8, simplifyRatio: 0.5, maxLevels: 16 } as const;
const MAX_PAGE_BYTES = 32768;
/** Discretisation allowance of a silhouette displacement: one pixel diagonal
 * (pixel-centre sampling of both the mismatched pixel and the reference edge). */
const SILHOUETTE_ALLOWANCE = Math.SQRT2;
/** float32 positions versus the float64 CPU distance: far below a pixel. */
const DEPTH_ALLOWANCE = 1e-3;
const WORST_EXAMPLES = 8;

type PrivateRenderer = { device: GPUDevice; hdr: GPUTexture; batches: { scene: { name: string }; stream?: { runtime: GpuGeometryStream } }[] };
type Vec3 = [number, number, number];

const HALF = (() => {
  const table = new Float32Array(65536);
  for (let bits = 0; bits < 65536; bits++) {
    const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, fraction = bits & 1023;
    table[bits] = sign * (exponent === 31 ? (fraction ? NaN : Infinity) : exponent ? (1 + fraction / 1024) * 2 ** (exponent - 15) : fraction * 2 ** -24);
  }
  return table;
})();

/** Test-only HDR readback with one reusable staging buffer per renderer. */
class HdrReader {
  private readonly buffer: GPUBuffer;
  private readonly bytesPerRow: number;
  constructor(private readonly renderer: Renderer, readonly width: number, readonly height: number) {
    const r = renderer as unknown as PrivateRenderer;
    this.bytesPerRow = Math.ceil(width * 8 / 256) * 256;
    this.buffer = r.device.createBuffer({ label: 'G01 test-only HDR readback', size: this.bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }
  async read(): Promise<Float32Array> {
    const r = this.renderer as unknown as PrivateRenderer, { width, height } = this;
    const encoder = r.device.createCommandEncoder({ label: 'G01 test-only HDR copy' });
    encoder.copyTextureToBuffer({ texture: r.hdr }, { buffer: this.buffer, bytesPerRow: this.bytesPerRow }, { width, height });
    r.device.queue.submit([encoder.finish()]);
    await this.buffer.mapAsync(GPUMapMode.READ);
    try {
      const words = new Uint16Array(this.buffer.getMappedRange()), rgb = new Float32Array(width * height * 3), stride = this.bytesPerRow / 2;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const s = y * stride + x * 4, d = (y * width + x) * 3;
        rgb[d] = HALF[words[s]]; rgb[d + 1] = HALF[words[s + 1]]; rgb[d + 2] = HALF[words[s + 2]];
      }
      return rgb;
    } finally { this.buffer.unmap(); }
  }
  dispose() { this.buffer.destroy(); }
}

/** Renders one frame, retrying while stream backpressure skips it. */
async function renderFrame(renderer: Renderer, pose: G01Pose): Promise<number> {
  const before = renderer.stats.frames;
  for (let attempt = 0; attempt < 400; attempt++) {
    renderer.render([...pose.eye] as Vec3, [...pose.target] as Vec3);
    if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
    if (renderer.stats.frames > before) return attempt;
    await new Promise(resolve => setTimeout(resolve, 2));
  }
  throw new Error('G01_RENDER_BACKPRESSURE');
}

function instance(): InstanceState {
  return { entityId: 1001, entityGeneration: 1, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
    color: [OBJECT[0], OBJECT[1], OBJECT[2], 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
}

function identityMaterial(): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.instanceColor } };
}

/** Per-path, per-step record (columnar to keep the JSON small). */
function pathColumns() {
  return { step: [] as number[], clusters: [] as number[], cutTriangles: [] as number[], gpuTriangles: [] as number[],
    epsilonPixels: [] as number[], mismatched: [] as number[], lodOnly: [] as number[], referenceOnly: [] as number[],
    maxDistance: [] as number[], maxRatio: [] as number[], violations: [] as number[],
    cutAdded: [] as number[], cutRemoved: [] as number[], popLodOnly: [] as number[], popReferenceOnly: [] as number[], purePop: [] as number[],
    boundaryEdges: [] as number[], repeatedEdges: [] as number[], uncovered: [] as number[], overlapping: [] as number[], partialGroups: [] as number[],
    cpuExtra: [] as number[], cpuMissing: [] as number[], cpuBorderline: [] as number[], cpuUnexplained: [] as number[],
    fallbackClusters: [] as number[], idle: [] as number[] };
}
type Columns = ReturnType<typeof pathColumns>;
type Worst = { step: number; x: number; y: number; value: number; bound: number; epsilonPixels: number; ratio: number };

function pushWorst(list: Worst[], entry: Worst) {
  list.push(entry); list.sort((a, b) => b.ratio - a.ratio);
  if (list.length > WORST_EXAMPLES) list.length = WORST_EXAMPLES;
}

class PathState {
  readonly columns = pathColumns();
  readonly distances: number[] = [];
  readonly ratios: number[] = [];
  readonly gaps: number[] = [];
  readonly gapRatios: number[] = [];
  readonly gapSamples = { step: [] as number[], x: [] as number[], y: [] as number[], gapPixels: [] as number[], boundPixels: [] as number[] };
  readonly worstSilhouette: Worst[] = [];
  readonly worstGap: Worst[] = [];
  readonly crossChecks: { step: number; coverageEqual: boolean; edgesEqual: boolean; repeatedOrigins: Record<string, number> | null }[] = [];
  readonly frozenChecks: { step: number; equalToPreviousCut: boolean }[] = [];
  readonly violationDiagnoses: ReturnType<typeof diagnoseViolation>[] = [];
  readonly ledger: LodCutEdgeLedger;
  violationEvidence = 16;
  /** Violations whose 3D surface gap (full detail to cut for reference-only
   * pixels, cut to full detail for LOD-only pixels) is within eps * sec(theta):
   * see-through slivers at internal silhouettes that binary coverage cannot place. */
  violationsWithinSurfaceBound = 0;
  violationsCpuMiss = 0;
  previousLabels: Uint8Array | null = null;
  previousCut: Uint32Array | null = null;
  violations = 0;
  gapViolations = 0;
  rayMisses = 0;
  constructor(readonly name: 'resident' | 'stream', hierarchy: GeometryHierarchy) { this.ledger = new LodCutEdgeLedger(hierarchy); }
}

/** Repeated directed edges of a cut grouped by where the duplicate triangles come from. */
function repeatedEdgeOrigins(hierarchy: GeometryHierarchy, cut: Uint32Array): Record<string, number> {
  const view = new DataView(hierarchy.source.vertices), canonical = new Map<string, number>();
  const weld = (vertex: number) => {
    const o = vertex * 80, key = `${view.getUint32(o, true)},${view.getUint32(o + 4, true)},${view.getUint32(o + 8, true)}`;
    let id = canonical.get(key);
    if (id === undefined) { id = canonical.size; canonical.set(key, id); }
    return id;
  };
  const users = new Map<number, number[]>();
  for (const id of cut) {
    const cluster = hierarchy.clusters[id];
    for (let t = 0; t < cluster.triangles.length; t += 3) {
      const v = [0, 1, 2].map(k => weld(cluster.vertices[cluster.triangles[t + k]]));
      for (let k = 0; k < 3; k++) {
        const key = v[k] * 0x4000000 + v[(k + 1) % 3], list = users.get(key);
        if (list) list.push(id); else users.set(key, [id]);
      }
    }
  }
  const origins: Record<string, number> = { withinOneCluster: 0, siblingsOfOneSimplification: 0, acrossGroups: 0 };
  const levels: Record<string, number> = {};
  for (const list of users.values()) {
    if (list.length < 2) continue;
    const first = hierarchy.clusters[list[0]];
    if (list.every(id => id === list[0])) origins.withinOneCluster++;
    else if (first.refinedGroup !== null && list.every(id => hierarchy.clusters[id].refinedGroup === first.refinedGroup)) origins.siblingsOfOneSimplification++;
    else origins.acrossGroups++;
    const level = `level${Math.max(...list.map(id => hierarchy.clusters[id].level))}`;
    levels[level] = (levels[level] ?? 0) + 1;
  }
  return { ...origins, ...levels };
}

/** Test-only explanation of one silhouette-bound violation: the pixel ray
 * against full detail (front faces, as rasterised) and against the drawn cut
 * (nearest triangle of either facing, and nearest front face), with the edge
 * use of the nearest cut triangle in the welded cut. A back-facing nearest cut
 * triangle whose forward edge is repeated is a fold the rasteriser culls. */
function diagnoseViolation(context: { hierarchy: GeometryHierarchy; surface: LodClusterSurface; lod0: Uint32Array; width: number; height: number; projectionScale: number },
  ledger: LodCutEdgeLedger, cut: Uint32Array, pose: G01Pose, step: number, x: number, y: number, kind: 'lodOnly' | 'referenceOnly', distance: number, bound: number, epsilon: number) {
  const { hierarchy, surface, lod0, width, height, projectionScale } = context;
  const { direction, forward } = lodPixelRay(pose.eye, pose.target, x, y, width, height, G01_VERTICAL_FOV);
  const full = surface.raycast(lod0, pose.eye, direction, G01_NEAR, true);
  const nearest = surface.raycast(cut, pose.eye, direction, G01_NEAR), front = surface.raycast(cut, pose.eye, direction, G01_NEAR, true);
  // The surface point the frame shows at this pixel, and its distance to the other surface.
  const shown = kind === 'referenceOnly' ? full : front, other = kind === 'referenceOnly' ? cut : lod0;
  let gapPixels: number | null = null;
  if (shown) {
    const point: Vec3 = [pose.eye[0] + direction[0] * shown.t, pose.eye[1] + direction[1] * shown.t, pose.eye[2] + direction[2] * shown.t];
    const depth = (point[0] - pose.eye[0]) * forward[0] + (point[1] - pose.eye[1]) * forward[1] + (point[2] - pose.eye[2]) * forward[2];
    gapPixels = +(surface.distance(other, point)!.distance * projectionScale / depth).toFixed(4);
  }
  const surfaceBoundPixels = epsilon * lodPixelSecant(x, y, width, height, G01_VERTICAL_FOV) + DEPTH_ALLOWANCE;
  const hit = (entry: ReturnType<LodClusterSurface['raycast']>) => entry && { t: +entry.t.toFixed(5), cluster: entry.cluster,
    level: hierarchy.clusters[entry.cluster].level, triangle: entry.triangle, front: entry.front };
  return { step, x, y, kind, distance: +distance.toFixed(3), bound: +bound.toFixed(3), epsilonPixels: +epsilon.toFixed(4),
    fullDetailFront: hit(full), cutNearest: hit(nearest), cutNearestFront: hit(front),
    surfaceGapPixels: gapPixels, surfaceBoundPixels: +surfaceBoundPixels.toFixed(4), withinSurfaceBound: gapPixels !== null && gapPixels <= surfaceBoundPixels,
    cutNearestEdgeUse: nearest ? ledger.edgeUse(nearest.cluster, nearest.triangle) : null };
}

function encodePng(rgb: Float32Array, width: number, height: number, marks?: { reference: Uint8Array; lod: Uint8Array }) {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const g = canvas.getContext('2d')!, data = g.createImageData(width, height);
  for (let p = 0; p < width * height; p++) {
    const lodOnly = marks && marks.lod[p] === LOD_LABEL_OBJECT && marks.reference[p] === LOD_LABEL_BACKGROUND;
    const referenceOnly = marks && marks.lod[p] === LOD_LABEL_BACKGROUND && marks.reference[p] === LOD_LABEL_OBJECT;
    for (let c = 0; c < 3; c++) {
      data.data[p * 4 + c] = lodOnly ? [255, 0, 0][c] : referenceOnly ? [0, 255, 255][c] : Math.min(255, Math.round(Math.sqrt(Math.max(0, rgb[p * 3 + c])) * 255));
    }
    data.data[p * 4 + 3] = 255;
  }
  g.putImageData(data, 0, 0);
  return canvas.toDataURL('image/png');
}

function summary(values: ArrayLike<number>): LodDistribution {
  const d = lodDistribution(values), round = (value: number) => +value.toFixed(4);
  return { count: d.count, mean: round(d.mean), p50: round(d.p50), p90: round(d.p90), p99: round(d.p99), max: round(d.max) };
}

export async function runG01Fixture(options: G01FixtureOptions) {
  const maxErrorPixels = options.maxErrorPixels ?? 1, depthEvery = options.depthEvery ?? 8, depthSamples = options.depthSamples ?? 32;
  const crossCheckEvery = options.crossCheckEvery ?? 240, evidenceSteps = new Set(options.evidenceSteps ?? []);
  const timing: Record<string, number> = {};
  let started = performance.now();
  const mesh = g01Mesh(G01_SEED, options.around, options.across);
  timing.meshMs = performance.now() - started; started = performance.now();
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: `g01/${options.around}x${options.across}`, mesh, coverage: 'opaque' }, BUILD);
  timing.hierarchyMs = performance.now() - started; started = performance.now();
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: MAX_PAGE_BYTES });
  const directory: GeometryStreamDirectory = await buildGeometryStreamDirectory(hierarchy, encoded);
  timing.pagesMs = performance.now() - started; started = performance.now();
  const pages = encoded.pages.map(page => page.bytes);
  const provider: GeometryPageProvider = async (page, signal) => {
    signal.throwIfAborted();
    const bytes = pages[page.id];
    if (!bytes) throw new RangeError('G01_PAGE');
    return bytes;
  };
  const queryProxy = await buildGeometryQueryProxy(directory, provider);
  timing.queryProxyMs = performance.now() - started;
  const layout = geometryGpuPoolLayout(directory);
  const levels: Record<string, { clusters: number; triangles: number }> = {};
  for (const cluster of hierarchy.clusters) {
    const entry = levels[cluster.level] ??= { clusters: 0, triangles: 0 };
    entry.clusters++; entry.triangles += cluster.triangles.length / 3;
  }
  const lod0 = Uint32Array.from(hierarchy.clusters.flatMap((cluster, id) => cluster.level === 0 ? [id] : []));
  const trajectory = new G01Trajectory(), steps = Math.min(trajectory.steps, options.steps ?? trajectory.steps);

  const panel = document.createElement('section');
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#0b1418;color:#e8f1ef;padding:12px;font:13px system-ui;overflow:auto';
  panel.innerHTML = '<h1 style="font-size:16px;margin:0 0 6px">G01：完整细节参考 / 常驻 GPU LOD / 分页流送</h1>';
  const canvases = [0, 1, 2].map(() => {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `width:${options.width}px;height:${options.height}px;display:inline-block;margin:2px`;
    panel.appendChild(canvas); return canvas;
  });
  document.body.appendChild(panel);
  let reference: Renderer | undefined, resident: Renderer | undefined, stream: Renderer | undefined;
  const readers: HdrReader[] = [];
  try {
    const common = { name: NAME, capacity: 1, inputMode: 'committed' as const, deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
      material: identityMaterial(), queryIdentity: { assetId: `g01-${options.around}x${options.across}`, primitiveId: 0, topologyVersion: 1 }, instances: [instance()] };
    const referenceBatch: SceneBatch = { ...common, mesh: hierarchy.source };
    const residentBatch: SceneBatch = { ...common, mesh: hierarchy.source, geometryLod: hierarchy };
    // The pool holds every page the device limits allow: the record measures
    // selection and streaming latency, not eviction under a small budget.
    const bindingLimit = 128 * 1024 * 1024;
    const slots = Math.min(directory.pages.length, Math.floor(bindingLimit / (layout.verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES)),
      Math.floor(bindingLimit / (layout.indicesPerSlot * 4)), Math.floor((256 * 1024 * 1024 - layout.metadataBytes) / layout.bytesPerSlot));
    const gpuPool = { maxResidentBytes: slots * layout.bytesPerSlot + layout.metadataBytes };
    const streamBatch: StreamedSceneBatch = { ...common, geometryStream: { directory, provider, queryProxy, gpuPool, maxRequestsPerFrame: 16 } };
    reference = await Renderer.create(canvases[0], [referenceBatch]);
    resident = await Renderer.create(canvases[1], [residentBatch]);
    stream = await Renderer.create(canvases[2], [streamBatch]);
    for (const renderer of [resident, stream]) renderer.setGeometryLod({ maxErrorPixels, frustumCulling: false, frozen: false, debug: 'material' });
    // The canvas size is resolved on the first frame.
    for (const renderer of [reference, resident, stream]) await renderFrame(renderer, trajectory.pose(0));
    const width = reference.stats.width, height = reference.stats.height;
    if (width !== options.width || height !== options.height) throw new Error(`G01_VIEWPORT:${width}x${height}:dpr${window.devicePixelRatio}`);
    for (const renderer of [resident, stream]) {
      if (renderer.stats.width !== width || renderer.stats.height !== height) throw new Error('G01_VIEWPORT_MISMATCH');
    }
    readers.push(new HdrReader(reference, width, height), new HdrReader(resident, width, height), new HdrReader(stream, width, height));
    const [referenceReader, residentReader, streamReader] = readers;
    const runtime = (stream as unknown as PrivateRenderer).batches.find(batch => batch.scene.name === NAME)?.stream?.runtime;
    if (!runtime) throw new Error('G01_STREAM_RUNTIME_MISSING');
    const projectionScale = height / (2 * Math.tan(G01_VERTICAL_FOV / 2));
    const planner = new GeometryStreamCut(directory), checker = new LodCutCoverageChecker(hierarchy), surface = new LodClusterSurface(hierarchy);
    const streamView = (pose: G01Pose): GeometryStreamView => ({ eye: pose.eye, projectionScale, near: G01_NEAR, maxErrorPixels });
    const streamState = () => {
      const d = runtime.diagnostics(), resident = new Set(d.pool.residentPages);
      const missing = d.allocation.pages.reduce((count, page) => count + (resident.has(page) ? 0 : 1), 0);
      // allocation.requestedPages lists every admitted page with its priority,
      // resident or not; outstanding work is pending uploads, admitted pages
      // not yet resident, and active or queued provider requests.
      return { busy: d.pool.pendingPages.length > 0 || missing > 0 || d.cache.activeRequests > 0 || d.cache.queuedRequests > 0,
        admitted: d.allocation.pages.length, missing, pending: d.pool.pendingPages.length,
        resident: d.pool.residentPages.length, uploads: d.pool.uploads, evictions: d.pool.evictions, error: d.error, slots: d.pool.slotCount };
    };

    // Settle the stream at the first pose: every admitted page resident and nothing in flight for 30 frames.
    started = performance.now();
    const first = trajectory.pose(0);
    let stable = 0, settleFrames = 0;
    while (stable < 30) {
      if (++settleFrames > 2000) throw new Error('G01_STREAM_DID_NOT_SETTLE:' + JSON.stringify(streamState()));
      await renderFrame(stream, first);
      await new Promise(resolve => requestAnimationFrame(() => resolve(null)));
      const state = streamState();
      if (state.error) throw new Error('G01_STREAM_ERROR:' + state.error);
      stable = state.busy ? 0 : stable + 1;
    }
    timing.settleMs = performance.now() - started;
    const settled = { frames: settleFrames, ...streamState() };

    const paths = { resident: new PathState('resident', hierarchy), stream: new PathState('stream', hierarchy) };
    const evidence: { step: number; name: string; png: string }[] = [];
    const videoSink = options.video ? (window as unknown as { __g01VideoFrame?: (step: number, jpeg: string) => Promise<void> }).__g01VideoFrame : undefined;
    if (options.video && !videoSink) throw new Error('G01_VIDEO_SINK_MISSING');
    let previousReference: Uint8Array | null = null, unknownPixels = 0, backpressureRetries = 0;
    const groupPixels = new Map<number, number>();
    const epsilonOf = (cut: Uint32Array, view: GeometryStreamView) => {
      groupPixels.clear();
      let epsilon = 0;
      for (const id of cut) {
        const refined = hierarchy.clusters[id].refinedGroup;
        if (refined === null) continue;
        let pixels = groupPixels.get(refined);
        if (pixels === undefined) { pixels = planner.groupErrorPixels(refined, view); groupPixels.set(refined, pixels); }
        if (pixels > epsilon) epsilon = pixels;
      }
      return epsilon;
    };
    let seed = 0x9e3779b9;
    const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
    const loopStarted = performance.now();
    const diagnosisContext = { hierarchy, surface, lod0, width, height, projectionScale };
    const violationFrames = new Set<number>();

    const measure = (path: PathState, step: number, pose: G01Pose, labels: Uint8Array, cut: Uint32Array, gpuTriangles: number,
      ref: LodReferenceSilhouette, extras: { purePop: number; fallback: number; idle: boolean; expected: Uint32Array | null }) => {
      const c = path.columns, view = streamView(pose), epsilon = epsilonOf(cut, view);
      c.step.push(step); c.clusters.push(cut.length);
      c.cutTriangles.push(cut.reduce((sum, id) => sum + hierarchy.clusters[id].triangles.length / 3, 0));
      c.gpuTriangles.push(gpuTriangles); c.epsilonPixels.push(+epsilon.toFixed(4));
      // Cracks: exact structural coverage and welded edge topology of the probed cut.
      const coverage = checker.check(cut), edges = path.ledger.update(cut);
      // Silhouette: each mismatched pixel's displacement against eps*sec^2(theta) + sqrt(2).
      const silhouette = lodCompareSilhouette(ref, labels);
      let maxDistance = 0, maxRatio = 0, violations = 0;
      for (let i = 0; i < silhouette.pixels.length; i++) {
        const p = silhouette.pixels[i], x = p % width, y = Math.floor(p / width), d = silhouette.distances[i];
        const secant = lodPixelSecant(x, y, width, height, G01_VERTICAL_FOV), bound = epsilon * secant * secant + SILHOUETTE_ALLOWANCE;
        const ratio = d / bound;
        path.distances.push(d); path.ratios.push(ratio);
        if (d > maxDistance) maxDistance = d;
        if (ratio > maxRatio) maxRatio = ratio;
        if (d > bound) {
          violations++;
          // Bounded: every violation is classified up to 256 per path; 24 are kept in full.
          if (path.violations + violations <= 256) {
            const diagnosis = diagnoseViolation(diagnosisContext, path.ledger, cut, pose, step, x, y,
              labels[p] === LOD_LABEL_OBJECT ? 'lodOnly' : 'referenceOnly', d, bound, epsilon);
            if (diagnosis.withinSurfaceBound) path.violationsWithinSurfaceBound++;
            if (diagnosis.surfaceGapPixels === null) path.violationsCpuMiss++;
            if (path.violationDiagnoses.length < 24) path.violationDiagnoses.push(diagnosis);
          }
        }
        if (path.worstSilhouette.length < WORST_EXAMPLES || ratio > path.worstSilhouette[path.worstSilhouette.length - 1].ratio) {
          pushWorst(path.worstSilhouette, { step, x, y, value: +d.toFixed(3), bound: +bound.toFixed(3), epsilonPixels: +epsilon.toFixed(4), ratio: +ratio.toFixed(4) });
        }
      }
      path.violations += violations;
      c.mismatched.push(silhouette.mismatched); c.lodOnly.push(silhouette.lodOnly); c.referenceOnly.push(silhouette.referenceOnly);
      c.maxDistance.push(+maxDistance.toFixed(3)); c.maxRatio.push(+maxRatio.toFixed(4)); c.violations.push(violations);
      // Popping between consecutive steps.
      if (path.previousLabels && path.previousCut && previousReference) {
        const pop = lodLabelPopping(path.previousLabels, labels, previousReference, ref.labels), diff = lodCutDifference(path.previousCut, cut);
        c.cutAdded.push(diff.added); c.cutRemoved.push(diff.removed); c.popLodOnly.push(pop.lodOnly); c.popReferenceOnly.push(pop.referenceOnly);
      } else { c.cutAdded.push(0); c.cutRemoved.push(0); c.popLodOnly.push(0); c.popReferenceOnly.push(0); }
      c.purePop.push(extras.purePop);
      c.uncovered.push(coverage.uncovered); c.overlapping.push(coverage.overlapping); c.partialGroups.push(coverage.partialGroups);
      c.boundaryEdges.push(edges.boundaryEdges); c.repeatedEdges.push(edges.repeatedEdges);
      if (step % crossCheckEvery === 0 || step === steps - 1) {
        const slowCoverage = geometryCutCoverage(directory, cut), slowEdges = geometryCutWatertightness(hierarchy, cut);
        path.crossChecks.push({ step, coverageEqual: JSON.stringify(slowCoverage) === JSON.stringify(coverage),
          edgesEqual: JSON.stringify(slowEdges) === JSON.stringify(edges), repeatedOrigins: edges.repeatedEdges ? repeatedEdgeOrigins(hierarchy, cut) : null });
      }
      // CPU reference cut for the same camera.
      if (extras.expected) {
        const comparison = lodCompareCuts(hierarchy, cut, extras.expected, view);
        c.cpuExtra.push(comparison.extra); c.cpuMissing.push(comparison.missing); c.cpuBorderline.push(comparison.borderline); c.cpuUnexplained.push(comparison.unexplained);
      } else { c.cpuExtra.push(-1); c.cpuMissing.push(-1); c.cpuBorderline.push(-1); c.cpuUnexplained.push(-1); }
      c.fallbackClusters.push(extras.fallback); c.idle.push(extras.idle ? 1 : 0);
      if (violations && path.violationEvidence > 0) { path.violationEvidence--; violationFrames.add(step); }
      path.previousLabels = labels; path.previousCut = cut;
      return epsilon;
    };

    for (let step = 0; step < steps; step++) {
      const pose = trajectory.pose(step);
      backpressureRetries += await renderFrame(reference, pose);
      const referenceRgb = await referenceReader.read(), referenceLabels = lodCoverageLabels(referenceRgb, OBJECT, BACKGROUND);
      const ref = lodReferenceSilhouette(referenceLabels, width, height);
      // Resident: the previous cut at this camera (frozen selection camera), then the live cut.
      let frozenLabels: Uint8Array | null = null;
      if (step > 0) {
        resident.setGeometryLod({ frozen: true });
        await renderFrame(resident, pose);
        frozenLabels = lodCoverageLabels(await residentReader.read(), OBJECT, BACKGROUND);
        const verifyFrozen = paths.resident.frozenChecks.length < 8 || step % crossCheckEvery === 0;
        if (verifyFrozen) {
          const frozenCut = lodSortedCut((await resident.probeGeometryLod(NAME)).selectedClusters.map(cluster => cluster.clusterId));
          const previous = paths.resident.previousCut!;
          paths.resident.frozenChecks.push({ step, equalToPreviousCut: frozenCut.length === previous.length && frozenCut.every((id, i) => id === previous[i]) });
        }
        resident.setGeometryLod({ frozen: false });
      }
      backpressureRetries += await renderFrame(resident, pose);
      const residentRgb = await residentReader.read(), residentLabels = lodCoverageLabels(residentRgb, OBJECT, BACKGROUND);
      const residentProbe = await resident.probeGeometryLod(NAME);
      if (residentProbe.overflow) throw new Error('G01_RESIDENT_OVERFLOW:' + step);
      const residentCut = lodSortedCut(residentProbe.selectedClusters.map(cluster => cluster.clusterId));
      const expected = lodSortedCut(selectGeometryCut(hierarchy, { eye: pose.eye, projectionScale, near: G01_NEAR, maxErrorPixels }));
      const residentEpsilon = measure(paths.resident, step, pose, residentLabels, residentCut, residentProbe.triangles, ref,
        { purePop: frozenLabels ? lodLabelDifference(frozenLabels, residentLabels) : 0, fallback: 0, idle: true, expected });
      // Stream: one live frame per step (60 frames per trajectory second).
      backpressureRetries += await renderFrame(stream, pose);
      const streamRgb = await streamReader.read(), streamLabels = lodCoverageLabels(streamRgb, OBJECT, BACKGROUND);
      const probe = await runtime.probe();
      if (probe.overflow) throw new Error('G01_STREAM_OVERFLOW:' + step);
      const selected = probe.selectedClusters ?? [];
      const streamCut = lodSortedCut(selected.map(cluster => cluster.clusterId));
      const state = streamState();
      if (state.error) throw new Error('G01_STREAM_ERROR:' + state.error);
      const plan = lodSortedCut(planner.plan(runtime.pool.residentPageIds(), streamView(pose)).selectedClusters);
      const streamEpsilon = measure(paths.stream, step, pose, streamLabels, streamCut, probe.indices / 3, ref,
        { purePop: 0, fallback: selected.filter(cluster => cluster.fallback).length, idle: !state.busy, expected: plan });
      for (const labels of [referenceLabels, residentLabels, streamLabels]) for (let p = 0; p < labels.length; p++) if (labels[p] === LOD_LABEL_UNKNOWN) unknownPixels++;

      // Surface gaps: full-detail surface point through a pixel, distance to each drawn cut.
      if (step % depthEvery === 0) {
        const covered: number[] = [];
        for (let p = 0; p < referenceLabels.length; p++) if (referenceLabels[p] === LOD_LABEL_OBJECT) covered.push(p);
        seed = (G01_SEED ^ Math.imul(step + 1, 0x85ebca6b)) >>> 0;
        for (let k = 0; k < depthSamples && covered.length; k++) {
          const p = covered[Math.floor(random() * covered.length)], x = p % width, y = Math.floor(p / width);
          const { direction, forward } = lodPixelRay(pose.eye, pose.target, x, y, width, height, G01_VERTICAL_FOV);
          const hit = surface.raycast(lod0, pose.eye, direction, G01_NEAR);
          if (!hit) { paths.resident.rayMisses++; paths.stream.rayMisses++; continue; }
          const point: Vec3 = [pose.eye[0] + direction[0] * hit.t, pose.eye[1] + direction[1] * hit.t, pose.eye[2] + direction[2] * hit.t];
          const depth = (point[0] - pose.eye[0]) * forward[0] + (point[1] - pose.eye[1]) * forward[1] + (point[2] - pose.eye[2]) * forward[2];
          const secant = lodPixelSecant(x, y, width, height, G01_VERTICAL_FOV);
          for (const [path, cut, epsilon] of [[paths.resident, residentCut, residentEpsilon], [paths.stream, streamCut, streamEpsilon]] as const) {
            const nearest = surface.distance(cut, point)!;
            const gap = nearest.distance * projectionScale / depth, bound = epsilon * secant + DEPTH_ALLOWANCE, ratio = gap / bound;
            path.gaps.push(gap); path.gapRatios.push(ratio);
            path.gapSamples.step.push(step); path.gapSamples.x.push(x); path.gapSamples.y.push(y);
            path.gapSamples.gapPixels.push(+gap.toFixed(4)); path.gapSamples.boundPixels.push(+bound.toFixed(4));
            if (gap > bound) path.gapViolations++;
            if (path.worstGap.length < WORST_EXAMPLES || ratio > path.worstGap[path.worstGap.length - 1].ratio) {
              pushWorst(path.worstGap, { step, x, y, value: +gap.toFixed(4), bound: +bound.toFixed(4), epsilonPixels: +epsilon.toFixed(4), ratio: +ratio.toFixed(4) });
            }
          }
        }
      }
      if (evidenceSteps.has(step) || violationFrames.has(step)) {
        evidence.push({ step, name: 'reference', png: encodePng(referenceRgb, width, height) });
        evidence.push({ step, name: 'resident', png: encodePng(residentRgb, width, height, { reference: referenceLabels, lod: residentLabels }) });
        evidence.push({ step, name: 'stream', png: encodePng(streamRgb, width, height, { reference: referenceLabels, lod: streamLabels }) });
      }
      if (videoSink) {
        const canvas = document.createElement('canvas'); canvas.width = width * 3; canvas.height = height;
        const g = canvas.getContext('2d')!;
        const images = [encodePng(referenceRgb, width, height), encodePng(residentRgb, width, height, { reference: referenceLabels, lod: residentLabels }),
          encodePng(streamRgb, width, height, { reference: referenceLabels, lod: streamLabels })];
        for (const [index, url] of images.entries()) {
          const image = new Image(); image.src = url; await image.decode(); g.drawImage(image, index * width, 0);
        }
        await videoSink(step, canvas.toDataURL('image/jpeg', 0.9));
      }
      previousReference = referenceLabels;
    }
    timing.loopMs = performance.now() - loopStarted;

    const report = (path: PathState) => {
      const c = path.columns, changed = c.step.map((_, i) => i).filter(i => i > 0 && c.cutAdded[i] + c.cutRemoved[i] > 0);
      const unchanged = c.step.map((_, i) => i).filter(i => i > 0 && c.cutAdded[i] + c.cutRemoved[i] === 0);
      const at = (column: number[], indices: number[]) => indices.map(i => column[i]);
      const maxOf = (column: number[]) => column.reduce((a, b) => Math.max(a, b), 0);
      return {
        frames: c.step.length,
        cut: { clusters: summary(c.clusters), cutTriangles: summary(c.cutTriangles), gpuTriangles: summary(c.gpuTriangles),
          gpuTrianglesEqualCutTriangles: c.gpuTriangles.every((value, i) => value === c.cutTriangles[i]), epsilonPixels: summary(c.epsilonPixels) },
        silhouette: { boundDefinition: 'epsilon * sec^2(theta) + sqrt(2) px', mismatchedPixelsPerFrame: summary(c.mismatched),
          lodOnlyPerFrame: summary(c.lodOnly), referenceOnlyPerFrame: summary(c.referenceOnly), distancePixels: summary(path.distances),
          distanceOverBound: summary(path.ratios), violations: path.violations, worst: path.worstSilhouette,
          violationFrames: c.step.filter((_, i) => c.violations[i] > 0),
          violationsClassified: Math.min(path.violations, 256), violationsWithinSurfaceBound: path.violationsWithinSurfaceBound,
          violationsCpuMiss: path.violationsCpuMiss, violationDiagnoses: path.violationDiagnoses },
        depth: { boundDefinition: 'epsilon * sec(theta) + 0.001 px', samples: path.gaps.length, rayMisses: path.rayMisses,
          gapPixels: summary(path.gaps), gapOverBound: summary(path.gapRatios), violations: path.gapViolations, worst: path.worstGap },
        popping: { definition: 'pixels whose label changes between consecutive LOD frames while the consecutive reference frames keep it',
          framesWithCutChange: changed.length, popLodOnlyOnCutChange: summary(at(c.popLodOnly, changed)), maxPopLodOnly: maxOf(c.popLodOnly),
          popLodOnlyWithoutCutChange: summary(at(c.popLodOnly, unchanged)),
          referenceOnlyChange: summary(c.popReferenceOnly.slice(1)),
          ...(path.name === 'resident' ? { purePopDefinition: 'pixels that differ between the previous cut and the new cut drawn at the same camera',
            purePopOnCutChange: summary(at(c.purePop, changed)), maxPurePop: maxOf(c.purePop),
            purePopWithoutCutChange: summary(at(c.purePop, unchanged)), frozenChecks: path.frozenChecks } : {}) },
        cracks: { maxBoundaryEdges: maxOf(c.boundaryEdges), maxRepeatedEdges: maxOf(c.repeatedEdges),
          framesWithRepeatedEdges: c.repeatedEdges.filter(value => value > 0).length,
          maxUncovered: maxOf(c.uncovered), maxOverlapping: maxOf(c.overlapping), maxPartialGroups: maxOf(c.partialGroups), crossChecks: path.crossChecks },
        cpuCut: { comparedFrames: c.cpuExtra.filter(value => value >= 0).length,
          framesWithDifference: c.cpuExtra.filter((value, i) => value + c.cpuMissing[i] > 0).length,
          idleFramesWithDifference: c.cpuExtra.filter((value, i) => c.idle[i] && value + c.cpuMissing[i] > 0).length,
          borderlineClusters: c.cpuBorderline.reduce((a, b) => a + Math.max(b, 0), 0), unexplainedClusters: c.cpuUnexplained.reduce((a, b) => a + Math.max(b, 0), 0),
          idleUnexplainedClusters: c.cpuUnexplained.reduce((a, b, i) => a + (c.idle[i] ? Math.max(b, 0) : 0), 0) },
        ...(path.name === 'stream' ? { streaming: { idleFrames: c.idle.reduce((a, b) => a + b, 0), fallbackClusters: summary(c.fallbackClusters) } } : {}),
        columns: c, depthSamples: path.gapSamples,
      };
    };
    const end = streamState();
    return {
      adapter: stream.info.adapterInfo, rendererTiming: stream.info.timing,
      viewport: { width, height, devicePixelRatio: window.devicePixelRatio },
      parameters: { seed: G01_SEED, around: options.around, across: options.across, detailSpectrum: [...G01_DETAIL_SPECTRUM], build: BUILD, maxPageBytes: MAX_PAGE_BYTES,
        maxErrorPixels, verticalFov: G01_VERTICAL_FOV, near: G01_NEAR, projectionScale, stepSeconds: G01_STEP_SECONDS,
        trajectory: { steps: trajectory.steps, measuredSteps: steps, segments: trajectory.segments, options: trajectory.options },
        sampling: { depthEvery, depthSamples, crossCheckEvery, evidenceSteps: [...evidenceSteps] },
        allowances: { silhouettePixels: SILHOUETTE_ALLOWANCE, depthPixels: DEPTH_ALLOWANCE },
        stream: { gpuPool, slots, maxRequestsPerFrame: 16, uploadDefaults: 'GpuGeometryPagePool defaults', culling: false, occlusion: false },
        resident: { frustumCulling: false, occlusion: false } },
      source: { vertices: mesh.vertexCount, triangles: hierarchy.sourceTriangleCount, bounds: mesh.bounds },
      hierarchy: { clusters: hierarchy.clusters.length, groups: hierarchy.groups.length, roots: hierarchy.roots.length,
        rootTriangles: hierarchy.roots.reduce((sum, id) => sum + hierarchy.clusters[id].triangles.length / 3, 0), levels,
        pages: directory.pages.length, rootPages: directory.rootPages.length, layout, queryProxyTriangles: queryProxy.mesh.indices.length / 3 },
      streamSettle: settled, streamEnd: end, unknownPixels, backpressureRetries, backpressureFrames: stream.stats.streamBackpressureFrames,
      diagnosticTimingMs: Object.fromEntries(Object.entries(timing).map(([key, value]) => [key, Math.round(value)])),
      paths: { resident: report(paths.resident), stream: report(paths.stream) },
      evidence,
      errors: [...reference.errors, ...resident.errors, ...stream.errors],
    };
  } finally {
    for (const reader of readers) reader.dispose();
    stream?.dispose(); resident?.dispose(); reference?.dispose();
    await new Promise(resolve => setTimeout(resolve, 0));
    panel.remove();
  }
}
