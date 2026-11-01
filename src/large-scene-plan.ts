/**
 * Pre-creation geometry budget for the large-scene page. One explicit live
 * budget covers every paged stream (pools, selection, rendered current/previous
 * output and parameters, as GeometryStreamRuntimeBudget accounts them). The
 * initial pool of each stream comes from the CPU group demand of the first
 * camera pose, and the returned capacity-coordinator options keep later poses
 * inside the same budget. Pure CPU work over directories: no page decode, no
 * GPU allocation, no readback.
 */
import { projectionScale, type CameraProjection } from './camera-projection';
import type { GeometryStreamCapacityOptions } from './geometry-stream-capacity-coordinator';
import { GeometryStreamCut } from './geometry-stream-cut';
import { geometryGpuPoolLayout } from './geometry-gpu-pool-layout';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';
import { GeometryStreamRuntimeBudget, type GeometryStreamRuntimeLimits } from './geometry-stream-runtime-budget';
import type { GeometryStreamDirectory } from './geometry-stream-types';
import { lookAt, multiply, perspective, type Mat4 } from './math';
import { resolveDiffuseGiOptions } from './diffuse-gi';
import { planReflectionGrid } from './reflections';
import { resolveVolumetricFogOptions, selectFogGrid, type VolumetricFogOptions } from './volumetric-fog';
import type { RendererOptions } from './renderer';

const MAX_BYTES = 8 * 1024 ** 3;
/** GPU stream selection clamps the view distance to 0.1 when projecting LOD error; the CPU demand matches it. */
const LOD_DISTANCE_CLAMP = 0.1;

export interface LargeSceneStreamInput {
  readonly directory: GeometryStreamDirectory;
  /** One affine model matrix per drawn instance (the batch capacity). */
  readonly models: readonly Mat4[];
}

export interface LargeSceneView {
  readonly eye: readonly [number, number, number];
  readonly target: readonly [number, number, number];
  readonly width: number;
  readonly height: number;
  readonly projection: CameraProjection;
  readonly maxErrorPixels: number;
}

export interface LargeSceneGeometryBudgetOptions {
  /** Live paged-geometry bytes across all streams. */
  readonly maxResidentBytes: number;
  /** Migration overlap ceiling. Default: maxResidentBytes + the largest complete single-stream target + one slot. */
  readonly maxPeakBytes?: number;
  readonly maxConcurrentMigrations?: number;
  readonly maxCopyBytes?: number;
  readonly maxCopyPages?: number;
  readonly intervalFrames?: number;
}

export interface LargeSceneGeometryReport {
  streams: number;
  /** Root pages of every stream: the smallest admissible budget. */
  rootBytes: number;
  /** Every page of every stream resident (capped at each stream's largest valid pool). */
  allPagesBytes: number;
  /** Complete demand of the initial view with no budget limit. */
  initialDemandBytes: number;
  /** Initial allocation under the budget. */
  initialBytes: number;
  maxResidentBytes: number;
  maxPeakBytes: number;
  demandedGroups: number;
  /** Groups the initial view wants refined that the budget did not admit (drawn coarser). */
  budgetDeferredGroups: number;
  dependencyDeferredGroups: number;
  largestRootCopyBytes: number;
}

export interface LargeSceneGeometryPlan {
  /** Initial gpuPool.maxResidentBytes per input, in input order. */
  readonly poolBytes: number[];
  readonly capacity: GeometryStreamCapacityOptions;
  readonly report: LargeSceneGeometryReport;
}

function byteBudget(value: number | undefined, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || value < 1 || value > MAX_BYTES) throw new Error('LARGE_SCENE_GEOMETRY_OPTION: ' + name);
  return value;
}

function count(value: number | undefined, fallback: number, maximum: number, name: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new Error('LARGE_SCENE_GEOMETRY_OPTION: ' + name);
  return result;
}

function checkView(view: LargeSceneView): void {
  const finite = (values: readonly number[]) => values.length === 3 && values.every(Number.isFinite);
  if (!view || !finite(view.eye) || !finite(view.target) || view.eye.every((value, axis) => value === view.target[axis])
    || !Number.isSafeInteger(view.width) || view.width < 1 || view.width > 16384
    || !Number.isSafeInteger(view.height) || view.height < 1 || view.height > 16384
    || !Number.isFinite(view.maxErrorPixels) || view.maxErrorPixels < 0 || view.maxErrorPixels > 16) throw new Error('LARGE_SCENE_GEOMETRY_VIEW');
}

/** CPU group demand of one view for every stream, with frustum culling like the GPU feedback. */
export function largeSceneDemands(cuts: readonly GeometryStreamCut[], inputs: readonly LargeSceneStreamInput[], view: LargeSceneView) {
  checkView(view);
  const { verticalFovRadians, near, far } = view.projection;
  const viewProjection = multiply(perspective(verticalFovRadians, view.width / view.height, near, far),
    lookAt([...view.eye], [...view.target], [0, 1, 0]));
  const scale = projectionScale(view.projection, view.height);
  return cuts.map((cut, index) => cut.demands(inputs[index].models.map(model => ({ eye: view.eye, near: LOD_DISTANCE_CLAMP,
    projectionScale: scale, maxErrorPixels: view.maxErrorPixels, model, viewProjection }))));
}

export function planLargeSceneGeometry(inputs: readonly LargeSceneStreamInput[], view: LargeSceneView,
  options: LargeSceneGeometryBudgetOptions, limits: GeometryStreamRuntimeLimits): LargeSceneGeometryPlan {
  if (!Array.isArray(inputs) || inputs.length < 1 || inputs.length > 4096) throw new Error('LARGE_SCENE_GEOMETRY_INPUTS');
  for (const input of inputs) if (!Array.isArray(input?.models) || input.models.length < 1) throw new Error('LARGE_SCENE_GEOMETRY_INPUTS');
  const maxResidentBytes = byteBudget(options?.maxResidentBytes, 'maxResidentBytes');
  if (maxResidentBytes === undefined) throw new Error('LARGE_SCENE_GEOMETRY_OPTION: maxResidentBytes');
  const explicitPeak = byteBudget(options.maxPeakBytes, 'maxPeakBytes');
  if (explicitPeak !== undefined && explicitPeak < maxResidentBytes) throw new Error('LARGE_SCENE_GEOMETRY_OPTION: maxPeakBytes < maxResidentBytes');
  const maxConcurrentMigrations = count(options.maxConcurrentMigrations, 8, 16, 'maxConcurrentMigrations');
  const maxCopyBytes = count(options.maxCopyBytes, 64 * 1024 ** 2, 256 * 1024 ** 2, 'maxCopyBytes');
  const maxCopyPages = count(options.maxCopyPages, 4096, 4096, 'maxCopyPages');
  const intervalFrames = count(options.intervalFrames, 2, 600, 'intervalFrames');
  checkView(view);

  const runtimeInputs = inputs.map(input => ({ directory: input.directory, capacity: input.models.length }));
  const sizing = new GeometryStreamRuntimeBudget(runtimeInputs, { maxResidentBytes: MAX_BYTES, maxPeakBytes: MAX_BYTES }, limits);
  let allPagesBytes = 0, largestStream = 0, largestSlot = 0, largestRootCopyBytes = 0;
  // A stream whose complete page set exceeds a per-buffer limit tops out at its largest valid slot count.
  const fullState = (index: number) => {
    const { pages, rootPages } = inputs[index].directory;
    let low = rootPages.length, high = pages.length;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      try { sizing.state(index, middle); low = middle; } catch { high = middle - 1; }
    }
    return sizing.state(index, low);
  };
  inputs.forEach(({ directory }, index) => {
    const full = fullState(index), root = sizing.state(index, directory.rootPages.length);
    allPagesBytes += full.totalBytes; largestStream = Math.max(largestStream, full.totalBytes);
    largestSlot = Math.max(largestSlot, geometryGpuPoolLayout(directory).bytesPerSlot + root.outputBytes / directory.rootPages.length);
    let rootCopy = 0;
    for (const page of directory.rootPages) for (const cluster of directory.pages[page].clusterIds) {
      rootCopy += directory.clusters[cluster].vertexCount * GEOMETRY_GPU_VERTEX_BYTES + directory.clusters[cluster].triangleCount * 12;
    }
    largestRootCopyBytes = Math.max(largestRootCopyBytes, rootCopy);
  });
  const rootBytes = sizing.rootBytes;
  if (maxResidentBytes < rootBytes) {
    throw new Error(`LARGE_SCENE_GEOMETRY_BUDGET: ${maxResidentBytes} B is below the ${rootBytes} B every stream's root pages need`);
  }
  if (largestRootCopyBytes > maxCopyBytes) throw new Error(`LARGE_SCENE_GEOMETRY_COPY_BUDGET: a stream's roots copy ${largestRootCopyBytes} B > ${maxCopyBytes} B`);
  const maxPeakBytes = explicitPeak ?? Math.min(MAX_BYTES, maxResidentBytes + largestStream + Math.ceil(largestSlot));

  const cuts = inputs.map(input => new GeometryStreamCut(input.directory));
  const demands = largeSceneDemands(cuts, inputs, view);
  const unbounded = sizing.plan(demands);
  const planner = new GeometryStreamRuntimeBudget(runtimeInputs, { maxResidentBytes, maxPeakBytes }, limits);
  const initial = planner.plan(demands);
  const report: LargeSceneGeometryReport = { streams: inputs.length, rootBytes, allPagesBytes,
    initialDemandBytes: unbounded.allocatedBytes, initialBytes: initial.allocatedBytes, maxResidentBytes, maxPeakBytes,
    demandedGroups: demands.reduce((sum, list) => sum + list.length, 0),
    budgetDeferredGroups: initial.streams.reduce((sum, stream) => sum + stream.budgetDeferredGroups.length, 0),
    dependencyDeferredGroups: initial.streams.reduce((sum, stream) => sum + stream.dependencyDeferredGroups.length, 0),
    largestRootCopyBytes };
  return { poolBytes: initial.streams.map(stream => stream.poolBytes), report,
    capacity: { maxResidentBytes, maxPeakBytes, intervalFrames, maxConcurrentMigrations, maxCopyBytes, maxCopyPages } };
}

export type LargeSceneBounds = { readonly min: readonly [number, number, number]; readonly max: readonly [number, number, number] };

export interface LargeSceneLightingRequest {
  readonly gi: boolean;
  readonly reflections: boolean;
  readonly fog: boolean;
  /** GI-lit air in the fog (requires gi and fog). */
  readonly airLighting: boolean;
  /** Probe ceiling for the bounds-sized grid, 8..512. Default 512. */
  readonly maxProbes?: number;
  /** 16..256. Default 64. */
  readonly raysPerProbe?: number;
  /** Reflection anchor spacing in pixels, or 'auto' for the finest the ray budget allows at this size. */
  readonly reflectionSpacing?: number | 'auto';
  /** Fog froxel size in pixels, or 'auto' for the finest the froxel/byte budget allows at this size. */
  readonly fogPixelSize?: number | 'auto';
}

export interface LargeSceneLightingReport {
  probes: { counts: [number, number, number]; spacing: [number, number, number]; rays: number } | null;
  reflections: { spacing: number; anchors: number; rays: number } | null;
  fog: { pixelSize: number; slices: number; froxels: number; bytes: number } | null;
  maxRays: number;
}

export interface LargeSceneLightingPlan {
  readonly options: Pick<RendererOptions, 'worldQueries' | 'diffuseGi' | 'reflections' | 'volumetricFog'>;
  readonly report: LargeSceneLightingReport;
}

const QUERY_RAYS = 65_536;
const QUERY_WORK = 67_108_864;
// Uniform sky radiance (luminance about 0.113): an unoccluded upward surface then receives about
// the renderer's legacy ambient fill (0.11 x albedo), so enabling GI adds occlusion and bounce
// light instead of brightening the whole frame at the page's default exposure.
const SKY: [number, number, number] = [0.095, 0.115, 0.15];

/** Uniform-spacing probe grid over the bounds with at most maxProbes probes, 2..32 per axis, cell-centred. */
export function largeSceneProbeGrid(bounds: LargeSceneBounds, maxProbes = 512) {
  if (!Number.isSafeInteger(maxProbes) || maxProbes < 8 || maxProbes > 512) throw new Error('LARGE_SCENE_GI_PROBES');
  const extent = [0, 1, 2].map(axis => bounds.max[axis] - bounds.min[axis]);
  if (!extent.every(value => Number.isFinite(value) && value > 0)) throw new Error('LARGE_SCENE_GI_BOUNDS');
  const counts = (spacing: number) => extent.map(value => Math.min(32, Math.max(2, Math.ceil(value / spacing)))) as [number, number, number];
  const fits = (spacing: number) => counts(spacing).reduce((a, b) => a * b, 1) <= maxProbes;
  // Bisection on the spacing: the finest uniform spacing whose grid fits the probe ceiling.
  let low = Math.max(...extent) / 64, high = Math.max(...extent);
  for (let i = 0; i < 64; i++) { const middle = (low + high) / 2; if (fits(middle)) high = middle; else low = middle; }
  const chosen = counts(high);
  const spacing = extent.map((value, axis) => value / chosen[axis]) as [number, number, number];
  const origin = bounds.min.map((value, axis) => value + spacing[axis] / 2) as [number, number, number];
  return { origin, spacing, counts: chosen };
}

/** Explicit lighting budgets for one canvas size. A requested setting the engine would coarsen
 * at this size is rejected here, before Renderer.create, rather than silently degraded; 'auto'
 * picks the finest setting that fits once, and the caller re-validates it on every resize. */
export function planLargeSceneLighting(bounds: LargeSceneBounds, width: number, height: number, far: number,
  batches: number, request: LargeSceneLightingRequest): LargeSceneLightingPlan {
  if (!Number.isSafeInteger(width) || width < 1 || width > 16384 || !Number.isSafeInteger(height) || height < 1 || height > 16384) {
    throw new Error('LARGE_SCENE_LIGHTING_VIEWPORT');
  }
  if (!Number.isSafeInteger(batches) || batches < 1 || batches > 4096) throw new Error('LARGE_SCENE_LIGHTING_BATCHES');
  if (!Number.isFinite(far) || far <= 0) throw new Error('LARGE_SCENE_LIGHTING_FAR');
  if (request.airLighting && !(request.gi && request.fog)) throw new Error('LARGE_SCENE_LIGHTING: air lighting requires gi and fog');
  const report: LargeSceneLightingReport = { probes: null, reflections: null, fog: null, maxRays: QUERY_RAYS };
  const options: { -readonly [K in keyof LargeSceneLightingPlan['options']]: LargeSceneLightingPlan['options'][K] } = {};
  const diagonal = Math.hypot(...[0, 1, 2].map(axis => bounds.max[axis] - bounds.min[axis]));
  if (request.gi || request.reflections) {
    options.worldQueries = { maxRays: QUERY_RAYS, maxBatches: batches, maxTrianglesPerMesh: 262_144, maxBytes: 512 * 1024 ** 2,
      maxTotalWorkUnits: QUERY_WORK };
  }
  if (request.gi) {
    const raysPerProbe = request.raysPerProbe ?? 64;
    const grid = largeSceneProbeGrid(bounds, request.maxProbes ?? 512);
    const gi = { ...grid, raysPerProbe, maxDistance: Math.min(200, diagonal), normalBias: 0.05, skyRadiance: SKY, multiBounce: false,
      maxBytes: 128 * 1024 ** 2, update: { mode: 'scheduled' as const } };
    const resolved = resolveDiffuseGiOptions(gi);
    if (resolved.rayCount > QUERY_RAYS) throw new Error(`LARGE_SCENE_GI_BUDGET: ${resolved.rayCount} rays > ${QUERY_RAYS}`);
    options.diffuseGi = gi;
    report.probes = { counts: grid.counts, spacing: grid.spacing, rays: resolved.rayCount };
  }
  if (request.reflections) {
    const requested = request.reflectionSpacing ?? 'auto';
    const base = { maxAnchors: 32_768, samplesPerAnchor: 1, maxDistance: Math.min(100, diagonal), skyRadiance: SKY,
      maxBytes: 96 * 1024 ** 2, temporal: {} };
    const plan = planReflectionGrid(width, height, { ...base, pixelSpacing: requested === 'auto' ? 4 : requested }, QUERY_RAYS, QUERY_WORK);
    if (requested !== 'auto' && plan.spacing !== requested) {
      throw new Error(`LARGE_SCENE_REFLECTION_BUDGET: anchor spacing ${requested} px at ${width}x${height} exceeds the anchor/ray budget; it fits from ${plan.spacing} px`);
    }
    options.reflections = { ...base, pixelSpacing: plan.spacing };
    report.reflections = { spacing: plan.spacing, anchors: plan.anchorCount, rays: plan.rayCount };
  }
  if (request.fog) {
    const requested = request.fogPixelSize ?? 'auto';
    const fog: VolumetricFogOptions = { pixelSize: requested === 'auto' ? 8 : requested, slices: 48, maxViewDepth: Math.min(far, 160),
      maxFroxels: 1_048_576, maxBytes: 96 * 1024 ** 2, temporal: {},
      airLighting: request.airLighting ? { gi: true, outsideSkyVisibility: 0 } : false,
      global: { coefficients: { extinction: 0.012, albedo: [0.82, 0.86, 0.9], emission: [0, 0, 0], anisotropy: 0.35 },
        baseHeight: bounds.min[1], heightFalloff: 0.08 } };
    const grid = selectFogGrid(width, height, resolveVolumetricFogOptions(fog));
    if (requested !== 'auto' && grid.pixelSize !== requested) {
      throw new Error(`LARGE_SCENE_FOG_BUDGET: froxel size ${requested} px at ${width}x${height} exceeds the froxel/byte budget; it fits from ${grid.pixelSize} px`);
    }
    options.volumetricFog = { ...fog, pixelSize: grid.pixelSize };
    report.fog = { pixelSize: grid.pixelSize, slices: grid.slices, froxels: grid.froxels, bytes: grid.bytes };
  }
  return { options, report };
}
