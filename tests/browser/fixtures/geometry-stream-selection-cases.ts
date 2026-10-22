import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { createCube } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { decodeGeometryPage, encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryPageCache } from '../../../src/geometry-stream-cache';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import { GpuGeometryPagePool } from '../../../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../../../src/gpu-geometry-stream-selection';
import type { GpuOcclusion } from '../../../src/gpu-occlusion';
import type { GeometryStreamDirectory } from '../../../src/geometry-stream-types';
import { GEOMETRY_GPU_VERTEX_BYTES } from '../../../src/geometry-gpu-page';
import { identity, lookAt, multiply, normalMatrix, perspective, scaling, translation, type Mat4 } from '../../../src/math';

const eye: [number, number, number] = [0, 2, 12];
const projectionScale = 512 / (2 * Math.tan(Math.PI / 6.4));
const vp = multiply(perspective(Math.PI / 3.2, 1, 0.1, 200), lookAt(eye, [0, 0, 0], [0, 1, 0]));

function transforms(): Mat4[] {
  const shear = identity();
  shear[4] = 0.28; shear[8] = -0.12;
  return [
    multiply(translation(-1.6, 0, 0), scaling(0.82, 1.25, 0.68)),
    multiply(translation(1.6, 0.1, 0), multiply(shear, scaling(1.1, 0.72, 1.35))),
    multiply(translation(100, 0, 0), scaling(0.9, 1.4, 0.7)),
  ];
}

function instances(models: readonly Mat4[]): Float32Array {
  const data = new Float32Array(models.length * 56);
  for (let index = 0; index < models.length; index++) {
    const at = index * 56, model = models[index];
    data.set(model, at); data.set(model, at + 16); data.set(normalMatrix(model), at + 32);
    data.set([1, 1, 1, 1], at + 48);
  }
  return data;
}

function same<T>(actual: T, expected: T, label: string): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label}: ${JSON.stringify({ actual, expected })}`);
  }
}

type Camera = { eye: readonly number[]; vp: Mat4 };
type Demand = { groupId: number; projectedError: number };
const DEMAND_RELATIVE_TOLERANCE = 0.001;
const DEMAND_ABSOLUTE_TOLERANCE = 0.002;

function corners(bounds: { min: readonly number[]; max: readonly number[] }): [number, number, number][] {
  const points: [number, number, number][] = [];
  for (const x of [bounds.min[0], bounds.max[0]])
    for (const y of [bounds.min[1], bounds.max[1]])
      for (const z of [bounds.min[2], bounds.max[2]]) points.push([x, y, z]);
  return points;
}

function affinePoint(model: Mat4, point: readonly number[]): [number, number, number] {
  return [0, 1, 2].map(row => model[row] * point[0] + model[4 + row] * point[1]
    + model[8 + row] * point[2] + model[12 + row]) as [number, number, number];
}

function clipPoint(vp: Mat4, point: readonly number[]): [number, number, number, number] {
  return [0, 1, 2, 3].map(row => vp[row] * point[0] + vp[4 + row] * point[1]
    + vp[8 + row] * point[2] + vp[12 + row]) as [number, number, number, number];
}

/** Plane rejection on transformed corners is independent of the GPU's center/radius test. */
function intersectsFrustum(points: readonly (readonly number[])[], vp: Mat4): boolean {
  const clip = points.map(point => clipPoint(vp, point));
  return ![
    (p: readonly number[]) => p[0] < -p[3], (p: readonly number[]) => p[0] > p[3],
    (p: readonly number[]) => p[1] < -p[3], (p: readonly number[]) => p[1] > p[3],
    (p: readonly number[]) => p[2] < 0, (p: readonly number[]) => p[2] > p[3],
  ].some(outside => clip.every(outside));
}

/** The 1/∞ matrix-norm product bounds affine scale, including shear. */
function affineScaleBound(model: Mat4): number {
  const columns = [0, 1, 2].map(column => [0, 1, 2].reduce((sum, row) => sum + Math.abs(model[column * 4 + row]), 0));
  const rows = [0, 1, 2].map(row => [0, 1, 2].reduce((sum, column) => sum + Math.abs(model[column * 4 + row]), 0));
  return Math.sqrt(Math.max(...columns) * Math.max(...rows));
}

function demandReference(directory: GeometryStreamDirectory, models: readonly Mat4[], camera: Camera,
  maxErrorPixels: number, frustumCulling: boolean): Demand[] {
  const demand: Demand[] = [];
  for (const [groupId, group] of directory.groups.entries()) {
    if (group.terminal) continue;
    let largest = 0;
    for (const model of models) {
      const points = corners(group.bounds).map(point => affinePoint(model, point));
      if (frustumCulling && !intersectsFrustum(points, camera.vp)) continue;
      const minimum = [0, 1, 2].map(axis => Math.min(...points.map(point => point[axis])));
      const maximum = [0, 1, 2].map(axis => Math.max(...points.map(point => point[axis])));
      const distance = Math.hypot(...[0, 1, 2].map(axis =>
        Math.max(minimum[axis] - camera.eye[axis], 0, camera.eye[axis] - maximum[axis])));
      const pixels = group.error * affineScaleBound(model) * projectionScale / Math.max(0.1, distance);
      if (pixels > maxErrorPixels) largest = Math.max(largest, Math.min(pixels, 1e30));
    }
    if (largest > 0) demand.push({ groupId, projectedError: largest });
  }
  return demand;
}

function compareDemand(actual: readonly Demand[], expected: readonly Demand[], label: string): void {
  const byId = (entries: readonly Demand[]) => [...entries].sort((a, b) => a.groupId - b.groupId);
  const actualSorted = byId(actual), expectedSorted = byId(expected);
  same(actualSorted.map(entry => entry.groupId), expectedSorted.map(entry => entry.groupId), label + ': DEMAND_GROUP_IDS');
  for (let i = 0; i < expectedSorted.length; i++) {
    const observed = actualSorted[i].projectedError, wanted = expectedSorted[i].projectedError;
    const tolerance = Math.max(DEMAND_ABSOLUTE_TOLERANCE, wanted * DEMAND_RELATIVE_TOLERANCE);
    if (!Number.isFinite(observed) || Math.abs(observed - wanted) > tolerance) {
      throw new Error(`${label}: DEMAND_ERROR:${expectedSorted[i].groupId}: ${JSON.stringify({ observed, wanted, tolerance })}`);
    }
  }
}

/** A real 12-triangle cube is one cluster, unlike count=1 in the multi-cluster
 * demo. Capacity=2 also checks switching between single-pair and scan paths. */
async function singlePairCases(device: GPUDevice, occlusion: GpuOcclusion,
  planes: (mode: 'first' | 'recover' | 'reject') => void,
  readWords: (source: GPUBuffer, byteCount: number) => Promise<Uint32Array>) {
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'selection-single-pair-cube', mesh: createCube(), coverage: 'opaque' });
  const encoded = await encodeGeometryPages(hierarchy);
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  same([directory.clusters.length, directory.pages.length, directory.sourceTriangleCount], [1, 1, 12], 'SINGLE_PAIR_ASSET');
  const decoded = await decodeGeometryPage(encoded.pages[0].bytes);
  let pool: GpuGeometryPagePool | undefined, plain: GpuGeometryStreamSelection | undefined;
  let hiz: GpuGeometryStreamSelection | undefined, input: GPUBuffer | undefined;
  let poolOwned = 0, selectionOwned = 0;
  try {
    pool = await GpuGeometryPagePool.create(device, directory, {}, bytes => { poolOwned += bytes; });
    pool.enqueue(0, decoded); pool.uploadFrame(1);
    same(pool.diagnostics().rootsReady, true, 'SINGLE_PAIR_ROOT_READY');
    input = device.createBuffer({ size: 2 * 224, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    plain = await GpuGeometryStreamSelection.create(device, pool, input, 2, bytes => { selectionOwned += bytes; });
    hiz = await GpuGeometryStreamSelection.create(device, pool, input, 2, bytes => { selectionOwned += bytes; }, {}, occlusion);
    const physicalCount = plain.physicalVertexCount, location = pool.clusterLocation(0)!;
    const expectedInstance = [...decoded.clusters[0].localTriangles].map(value => value + location.baseVertex + location.firstVertex);
    const savedLocation = await readWords(pool.clusterLocations, 32);
    const savedIndex = await readWords(pool.indices, 4);
    const rows = [], faults = [];
    let frame = 0;
    const submit = async (selection: GpuGeometryStreamSelection, count: number, offscreen = false) => {
      device.queue.writeBuffer(input!, 0, instances([offscreen ? translation(1000, 0, 0) : identity(), identity()]));
      selection.setSettings({ maxErrorPixels: 0, frustumCulling: true });
      selection.update(++frame, count, vp, eye, projectionScale);
      const lease = pool!.acquireFrame([0]), encoder = device.createCommandEncoder();
      let submitted = false;
      try {
        selection.encode(encoder); selection.encodeRecovery(encoder);
        const completion = pool!.submitFrame(lease, [encoder.finish()]); submitted = true;
        const done = await completion; if (done.error) throw new Error(done.error);
      } finally { if (!submitted) pool!.cancelFrame(lease); }
      const feedback = await selection.feedback(true);
      const args = await readWords(selection.arguments, 64);
      const indices = await readWords(selection.indices, feedback.indices * 4);
      return { feedback, args, indices };
    };
    // All visible cases emit the full nonempty cube. Zero and two-active cases
    // ensure the optimization uses current count, not allocated owner capacity.
    for (const mode of ['plain', 'first', 'recover', 'reject', 'frustum'] as const) {
      if (mode !== 'plain' && mode !== 'frustum') planes(mode);
      const selection = mode === 'plain' || mode === 'frustum' ? plain : hiz;
      for (const count of mode === 'frustum' ? [1] : [0, 1, 0, 1, 2, 1]) {
        const { feedback, args, indices } = await submit(selection, count, mode === 'frustum');
        const visible = mode !== 'reject' && mode !== 'frustum';
        const expected = visible ? Array.from({ length: count }, (_, instance) => expectedInstance.map(value => value + instance * physicalCount)).flat() : [];
        same([...indices], expected, 'SINGLE_PAIR_EXACT_INDICES_' + mode);
        same(feedback.overflow, false, 'SINGLE_PAIR_VALID_OVERFLOW');
        const recovered = mode === 'recover', deferred = mode === 'recover' || mode === 'reject';
        const firstCount = recovered ? 0 : expected.length, recoveryCount = recovered ? expected.length : 0;
        same([args[0], args[8], args[10]], [firstCount, recoveryCount, firstCount], 'SINGLE_PAIR_INDIRECT');
        same(feedback.occlusion, { deferredClusters: deferred ? count : 0,
          culledClusters: mode === 'reject' ? count : 0, recoveredClusters: recovered ? count : 0 }, 'SINGLE_PAIR_HIZ');
        same(feedback.selectedClusters?.map(entry => entry.visible), Array(count).fill(visible), 'SINGLE_PAIR_FLAGS');
        rows.push({ mode, count, indices: feedback.indices, firstIndices: firstCount, recoveredIndices: recoveryCount });
      }
    }
    for (const mode of ['first', 'recover'] as const) for (const defect of ['count', 'output', 'input-range', 'vertex-range', 'index'] as const) {
      planes(mode);
      const broken = savedLocation.slice();
      if (defect === 'count') broken[4] = 1539;
      if (defect === 'output') broken[4] = directory.sourceTriangleCount * 3 * 2 + 3;
      if (defect === 'input-range') broken[3] = 0xffffffff;
      if (defect === 'vertex-range') broken[0] = physicalCount + 1;
      device.queue.writeBuffer(pool.clusterLocations, 0, broken);
      if (defect === 'index') device.queue.writeBuffer(pool.indices, 0, new Uint32Array([0xffffffff]));
      try {
        const { feedback, args, indices } = await submit(hiz, 1);
        same({ indices: feedback.indices, overflow: feedback.overflow }, { indices: 0, overflow: true }, 'SINGLE_PAIR_FAIL_CLOSED_' + mode + '_' + defect);
        same([args[0], args[8], args[6], args[14]], mode === 'first' ? [0, 0, 1, 0] : [0, 0, 0, 1], 'SINGLE_PAIR_FAILURE_PHASE');
        same(indices.length, 0, 'SINGLE_PAIR_FAILURE_DRAW');
        faults.push({ mode, defect, indices: feedback.indices, overflow: feedback.overflow });
      } finally {
        device.queue.writeBuffer(pool.clusterLocations, 0, savedLocation);
        device.queue.writeBuffer(pool.indices, 0, savedIndex);
      }
    }
    // A failed phase must not poison the next frame after valid input returns.
    planes('recover');
    const restored = await submit(hiz, 1);
    same([...restored.indices], expectedInstance, 'SINGLE_PAIR_RESTORED_INDICES');
    same(restored.feedback.overflow, false, 'SINGLE_PAIR_RESTORED_OVERFLOW');
    same([plain.diagnostics().compaction.scratchBytes, hiz.diagnostics().compaction.scratchBytes], [0, 0], 'SINGLE_PAIR_NO_SCRATCH');
    plain.dispose(); hiz.dispose(); pool.dispose(); await device.queue.onSubmittedWorkDone();
    return { clusters: directory.clusters.length, capacity: 2, trianglesPerInstance: directory.sourceTriangleCount,
      rows, faults, restoredIndices: restored.feedback.indices, poolOwned, selectionOwned };
  } finally { plain?.dispose(); hiz?.dispose(); pool?.dispose(); input?.destroy(); }
}

/** Native GPU numeric cut comparison; its only readbacks are test diagnostics. */
export async function runGeometryStreamSelectionCases() {
  const scene = await GeometryDemoScene.create();
  const encoded = await encodeGeometryPages(scene.hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(scene.hierarchy, encoded);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('GEOMETRY_SELECTION_CASES_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let cache: GeometryPageCache | undefined, pool: GpuGeometryPagePool | undefined;
  let selection: GpuGeometryStreamSelection | undefined, instanceBuffer: GPUBuffer | undefined;
  let poolOwned = 0, selectionOwned = 0;
  let recoverySelection: GpuGeometryStreamSelection | undefined;
  const testBuffers: GPUBuffer[] = [], testTextures: GPUTexture[] = [];
  try {
    const allPageIds = directory.pages.map(page => page.id);
    cache = await GeometryPageCache.create(directory, async page => encoded.pages[page.id].bytes,
      { maxResidentBytes: Math.max(1, directory.pages.reduce((sum, page) => sum + page.decodedBytes, 0)) });
    for (const id of allPageIds) cache.request(id);
    await cache.idle();
    same([...cache.residentPageIds()].sort((a, b) => a - b), allPageIds, 'CPU_CACHE_ALL_PAGES');
    const verticesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].vertexCount, 0)));
    const indicesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].triangleCount * 3, 0)));
    const poolBudget = directory.pages.length * (verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES + indicesPerSlot * 4)
      + directory.clusters.length * 32 + directory.pages.length * 4;
    pool = await GpuGeometryPagePool.create(device, directory, {
      maxResidentBytes: poolBudget, maxStagingBytes: 64 * 1024 * 1024,
      maxUploadBytesPerFrame: 16 * 1024 * 1024, maxUploadWritesPerFrame: 4096, maxUploadPagesPerFrame: 256,
    }, bytes => { poolOwned += bytes; });
    const currentPool = pool;
    let uploadFrame = 0;
    const flush = (ids: readonly number[]) => {
      for (let attempt = 0; attempt < 4096; attempt++) {
        if (ids.every(id => currentPool.residentPageIds().has(id))) return;
        const result = currentPool.uploadFrame(++uploadFrame);
        if (result.blocked && result.bytes === 0) throw new Error('GEOMETRY_SELECTION_CASES_POOL_BLOCKED');
      }
      throw new Error('GEOMETRY_SELECTION_CASES_UPLOAD_LIMIT');
    };
    for (const id of directory.rootPages) currentPool.enqueue(id, cache.get(id)!);
    flush(directory.rootPages);
    const models = transforms(), planner = new GeometryStreamCut(directory);
    const capacity = Math.max(3, Math.floor(512 / directory.clusters.length) + 1);
    const manyModels = Array.from({ length: capacity }, (_, index) => models[index % 2]);
    const input = instances(manyModels);
    instanceBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true });
    new Float32Array(instanceBuffer.getMappedRange()).set(input); instanceBuffer.unmap();
    selection = await GpuGeometryStreamSelection.create(device, currentPool, instanceBuffer, capacity,
      bytes => { selectionOwned += bytes; });
    const gpu = selection;
    let frame = 0;
    let demandCamera: Camera = { eye, vp };
    const reference = (resident: ReadonlySet<number>, count: number, maxErrorPixels: number,
      caseModels: readonly Mat4[] = models) => {
      const plans = caseModels.slice(0, count).map(model => planner.plan(resident,
        { eye, near: 0.1, projectionScale, maxErrorPixels, model } satisfies GeometryStreamView));
      const selected = plans.flatMap((plan, instanceIndex) => plan.selectedClusters.map(clusterId => ({ clusterId, instanceIndex,
        level: directory.clusters[clusterId].level, visible: true })));
      const requests = new Map<number, number>();
      for (const plan of plans) for (const request of plan.requestedPages) {
        requests.set(request.pageId, Math.max(requests.get(request.pageId) ?? -1, request.priority));
      }
      const requestedPages = [...requests].map(([pageId, priority]) => ({ pageId, priority }))
        .sort((a, b) => b.priority - a.priority || a.pageId - b.pageId);
      const indices = selected.reduce((sum, entry) => sum + directory.clusters[entry.clusterId].triangleCount * 3, 0);
      return { selected, requestedPages, indices };
    };
    const readWords = async (source: GPUBuffer, byteCount: number) => {
      if (!byteCount) return new Uint32Array();
      const read = device.createBuffer({ size: byteCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, read, 0, byteCount);
        device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
        return new Uint32Array(read.getMappedRange()).slice();
      } finally { read.destroy(); }
    };
    const physicalIndices = (selected: readonly { clusterId: number; instanceIndex: number; visible: boolean }[],
      pageIndices: Uint32Array, physicalVertexCount: number) => selected.flatMap(entry => {
      if (!entry.visible) return [];
      const location = currentPool.clusterLocation(entry.clusterId)!;
      return Array.from({ length: location.indexCount }, (_, i) => pageIndices[location.firstIndex + i]
        + location.baseVertex + entry.instanceIndex * physicalVertexCount);
    });
    const indexHash = (indices: Uint32Array) => {
      let hash = 2166136261;
      for (const value of indices) hash = Math.imul(hash ^ value, 16777619) >>> 0;
      return hash;
    };
    const capture = async (label: string, count: number, maxErrorPixels: number, frustumCulling: boolean,
      compareIndices = false, expectOffscreen = false, camera: Camera = { eye, vp }, caseModels: readonly Mat4[] = models) => {
      device.queue.writeBuffer(instanceBuffer!, 0, instances(caseModels));
      gpu.setSettings({ maxErrorPixels, frustumCulling });
      gpu.update(++frame, count, camera.vp, camera.eye, projectionScale);
      if (!gpu.settings.frozen) demandCamera = camera;
      const resident = currentPool.residentPageIds();
      const lease = currentPool.acquireFrame([...resident]);
      const encoder = device.createCommandEncoder(); gpu.encode(encoder);
      const complete = await currentPool.submitFrame(lease, [encoder.finish()]);
      if (complete.error) throw new Error(complete.error);
      const actual = await gpu.feedback(true), expected = reference(resident, count, maxErrorPixels, caseModels);
      if (actual.overflow) throw new Error(label + ': GPU_OVERFLOW');
      compareDemand(actual.desiredGroups, demandReference(directory, caseModels.slice(0, count), demandCamera,
        maxErrorPixels, frustumCulling), label);
      same(actual.selectedClusters?.map(({ visible, ...entry }) => entry),
        expected.selected.map(({ visible, ...entry }) => entry), label + ': CUT');
      same(actual.requestedPages, expected.requestedPages, label + ': REQUESTS');
      if (!frustumCulling) {
        same(actual.selectedClusters, expected.selected, label + ': VISIBLE');
        same(actual.indices, expected.indices, label + ': INDEX_COUNT');
      } else {
        const culled = actual.selectedClusters!.filter(entry => entry.instanceIndex === 2);
        if (expectOffscreen && (!culled.length || culled.some(entry => entry.visible))) {
          throw new Error(label + ': OFFSCREEN_VISIBILITY');
        }
        const visibleCount = actual.selectedClusters!.reduce((sum, entry) => sum
          + (entry.visible ? directory.clusters[entry.clusterId].triangleCount * 3 : 0), 0);
        same(actual.indices, visibleCount, label + ': CULLED_INDEX_COUNT');
      }
      let orderedHash: number | null = null;
      if (compareIndices && actual.indices) {
        const pageIndices = await readWords(currentPool.indices, currentPool.indices.size);
        const compacted = await readWords(gpu.indices, actual.indices * 4);
        const expectedPhysical = physicalIndices(actual.selectedClusters!, pageIndices, gpu.physicalVertexCount);
        same([...compacted], expectedPhysical, label + ': EXACT_ORDERED_PHYSICAL_INDICES');
        orderedHash = indexHash(compacted);
      }
      return { label, count, pairs: count * directory.clusters.length, orderedHash, maxErrorPixels, frustumCulling, clusters: expected.selected.length,
        indices: actual.indices, requests: actual.requestedPages.length, desiredGroups: actual.desiredGroups,
        visibleClusters: actual.selectedClusters!.filter(entry => entry.visible).length,
        culledInstanceClusters: actual.selectedClusters!.filter(entry => entry.instanceIndex === 2 && !entry.visible).length };
    };
    const rootOnly = await capture('roots-only-fine', 2, 0, false);
    for (const id of allPageIds) if (!currentPool.residentPageIds().has(id)) currentPool.enqueue(id, cache.get(id)!);
    flush(allPageIds);
    const resident = currentPool.residentPageIds();
    const fineCount = reference(resident, 2, 0).selected.length;
    const coarseCount = reference(resident, 2, 10000).selected.length;
    const intermediate = [0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512].find(threshold => {
      const count = reference(resident, 2, threshold).selected.length;
      return count > coarseCount && count < fineCount;
    });
    if (intermediate === undefined) throw new Error(`GEOMETRY_SELECTION_CASES_NO_INTERMEDIATE:${coarseCount}:${fineCount}`);
    const fine = await capture('all-resident-fine', 2, 0, false, true);
    same(rootOnly.desiredGroups, fine.desiredGroups, 'DEMAND_INDEPENDENT_OF_RESIDENCY');
    const middle = await capture('all-resident-middle', 2, intermediate, false);
    const farEye: [number, number, number] = [0, 2, 1000];
    const farCount = models.slice(0, 2).reduce((sum, model) => sum + planner.plan(resident,
      { eye: farEye, near: 0.1, projectionScale, maxErrorPixels: intermediate, model }).selectedClusters.length, 0);
    if (farCount === middle.clusters) throw new Error('GEOMETRY_SELECTION_CASES_FREEZE_NOT_DISCRIMINATING');
    gpu.setSettings({ frozen: true });
    const frozen = await capture('frozen-camera', 2, intermediate, false, false, false,
      { eye: farEye, vp: multiply(perspective(Math.PI / 3.2, 1, 0.1, 2000), lookAt(farEye, [0, 0, 0], [0, 1, 0])) });
    gpu.setSettings({ frozen: false });
    const coarse = await capture('all-resident-coarse', 2, 10000, false);
    const culled = await capture('frustum-offscreen', 3, intermediate, true, true, true);
    const zero = await capture('zero-active-instances', 0, intermediate, false);
    const allOffscreen = await capture('all-offscreen', 1, intermediate, true, false, false,
      { eye, vp }, [models[2]]);
    const swapped = await capture('swapped-affine-instances', 2, 0, false, true, false,
      { eye, vp }, [models[1], models[0]]);
    same(swapped.desiredGroups, fine.desiredGroups, 'DEMAND_INSTANCE_MAX_ORDER');
    // Same actual index sequence on static frames, both sides of a 512-pair
    // boundary; changing active count exercises scratch reuse and zero tails.
    const stableSmall = [], stableLarge = [];
    for (let repeat = 0; repeat < 12; repeat++) {
      stableSmall.push(await capture('stable-small-' + repeat, 1, 0, false, true));
      stableLarge.push(await capture('stable-large-' + repeat, capacity, 0, false, true, false, { eye, vp }, manyModels));
    }
    same(stableSmall.map(entry => entry.orderedHash), Array(12).fill(stableSmall[0].orderedHash), 'STATIC_SMALL_ORDER');
    same(stableLarge.map(entry => entry.orderedHash), Array(12).fill(stableLarge[0].orderedHash), 'STATIC_LARGE_ORDER');
    const boundaryBelow = await capture('below-block-boundary', Math.floor(512 / directory.clusters.length),
      0, false, true, false, { eye, vp }, manyModels);
    const boundaryZero = await capture('large-to-zero', 0, 0, false);
    const boundaryRegrow = await capture('zero-to-large', capacity, 0, false, true, false, { eye, vp }, manyModels);
    same(boundaryRegrow.orderedHash, stableLarge[0].orderedHash, 'REGROWN_ORDER');

    // Controlled Hi-Z planes exercise first-phase draws, all-deferred recovery,
    // current-frame rejection and mixed first/recovery output without readback
    // driving production selection. The projection maps every box to depth .5.
    const testLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
    ] });
    const occlusionUniform = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    testBuffers.push(occlusionUniform);
    const occlusionWords = new ArrayBuffer(96), floats = new Float32Array(occlusionWords), uints = new Uint32Array(occlusionWords);
    floats[14] = 0.5; floats[15] = 1; uints[16] = 1; uints[17] = 1;
    floats[20] = 1; floats[22] = 1;
    device.queue.writeBuffer(occlusionUniform, 0, occlusionWords);
    const plane = () => {
      const texture = device.createTexture({ size: [1, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      testTextures.push(texture);
      return { texture, group: device.createBindGroup({ layout: testLayout, entries: [
        { binding: 0, resource: { buffer: occlusionUniform } }, { binding: 1, resource: texture.createView() },
      ] }) };
    };
    const previous = plane(), current = plane();
    const occlusion = { testLayout, previousGroup: previous.group, currentGroup: current.group } as unknown as GpuOcclusion;
    recoverySelection = await GpuGeometryStreamSelection.create(device, currentPool, instanceBuffer, capacity,
      bytes => { selectionOwned += bytes; }, {}, occlusion);
    const recoveryGpu = recoverySelection, recoveryCases = [];
    device.queue.writeBuffer(instanceBuffer, 0, instances(manyModels));
    const pageWords = await readWords(currentPool.indices, currentPool.indices.size);
    for (const count of [1, capacity]) for (const mode of ['first', 'recover', 'reject', 'mixed'] as const) {
      const modeModels = mode === 'mixed' ? manyModels.map((model, index) => index % 2 ? model : multiply(translation(1000, 0, 0), model)) : manyModels;
      device.queue.writeBuffer(instanceBuffer, 0, instances(modeModels));
      floats[0] = mode === 'mixed' ? 0.01 : 0;
      device.queue.writeBuffer(occlusionUniform, 0, occlusionWords);
      device.queue.writeTexture({ texture: previous.texture }, new Float32Array([mode === 'first' ? 1 : 0]), { bytesPerRow: 4 }, [1, 1]);
      device.queue.writeTexture({ texture: current.texture }, new Float32Array([mode === 'reject' ? 0 : 1]), { bytesPerRow: 4 }, [1, 1]);
      const hashes = [];
      for (let repeat = 0; repeat < 4; repeat++) {
        recoveryGpu.setSettings({ maxErrorPixels: 0, frustumCulling: false });
        recoveryGpu.update(++frame, count, vp, eye, projectionScale);
        const lease = currentPool.acquireFrame([...currentPool.residentPageIds()]);
        const encoder = device.createCommandEncoder(); recoveryGpu.encode(encoder); recoveryGpu.encodeRecovery(encoder);
        const done = await currentPool.submitFrame(lease, [encoder.finish()]);
        if (done.error) throw new Error(done.error);
        const feedback = await recoveryGpu.feedback(true);
        if (feedback.overflow) throw new Error('RECOVERY_OVERFLOW');
        const wanted = reference(currentPool.residentPageIds(), count, 0, modeModels);
        const firstSelected = mode === 'first' ? wanted.selected : mode === 'mixed'
          ? wanted.selected.filter(entry => entry.instanceIndex % 2 === 0) : [];
        const recoverySelected = mode === 'recover' ? wanted.selected : mode === 'mixed'
          ? wanted.selected.filter(entry => entry.instanceIndex % 2 === 1) : [];
        const firstIndices = physicalIndices(firstSelected, pageWords, recoveryGpu.physicalVertexCount);
        const recoveredIndices = physicalIndices(recoverySelected, pageWords, recoveryGpu.physicalVertexCount);
        const expectedCount = mode === 'reject' ? 0 : wanted.indices;
        same(feedback.indices, expectedCount, 'RECOVERY_INDEX_COUNT');
        same(feedback.occlusion, { deferredClusters: wanted.selected.length - firstSelected.length,
          culledClusters: mode === 'reject' ? wanted.selected.length : 0,
          recoveredClusters: recoverySelected.length }, 'RECOVERY_COUNTERS');
        const indices = await readWords(recoveryGpu.indices, feedback.indices * 4);
        same([...indices], [...firstIndices, ...recoveredIndices], 'RECOVERY_EXACT_ORDERED_INDICES');
        const args = await readWords(recoveryGpu.arguments, 64);
        same([args[0], args[8], args[10]], [firstIndices.length, recoveredIndices.length, firstIndices.length], 'RECOVERY_OFFSETS');
        hashes.push(indexHash(indices));
      }
      same(hashes, Array(4).fill(hashes[0]), 'STATIC_RECOVERY_ORDER');
      recoveryCases.push({ count, pairs: count * directory.clusters.length, mode, hashes });
    }
    // Corrupt a visible cluster's count, then its physical input range. Both
    // fail closed without an out-of-bounds GPU write, including recovery.
    device.queue.writeBuffer(instanceBuffer, 0, instances(manyModels));
    floats[0] = 0; device.queue.writeBuffer(occlusionUniform, 0, occlusionWords);
    const firstCluster = reference(currentPool.residentPageIds(), 1, 0, manyModels).selected[0].clusterId;
    const savedLocation = (await readWords(currentPool.clusterLocations, currentPool.clusterLocations.size)).slice(firstCluster * 8, firstCluster * 8 + 8);
    const overflowCases = [];
    for (const mode of ['first', 'recover'] as const) for (const defect of ['count', 'range'] as const) {
      const broken = savedLocation.slice();
      if (defect === 'count') broken[4] = 1539; else broken[3] = 0xffffffff;
      device.queue.writeBuffer(currentPool.clusterLocations, firstCluster * 32, broken);
      device.queue.writeTexture({ texture: previous.texture }, new Float32Array([mode === 'first' ? 1 : 0]), { bytesPerRow: 4 }, [1, 1]);
      device.queue.writeTexture({ texture: current.texture }, new Float32Array([1]), { bytesPerRow: 4 }, [1, 1]);
      recoveryGpu.update(++frame, capacity, vp, eye, projectionScale);
      const lease = currentPool.acquireFrame([...currentPool.residentPageIds()]);
      const encoder = device.createCommandEncoder(); recoveryGpu.encode(encoder); recoveryGpu.encodeRecovery(encoder);
      const done = await currentPool.submitFrame(lease, [encoder.finish()]); if (done.error) throw new Error(done.error);
      const feedback = await recoveryGpu.feedback(true);
      same({ indices: feedback.indices, overflow: feedback.overflow }, { indices: 0, overflow: true }, 'FAIL_CLOSED_' + mode + '_' + defect);
      overflowCases.push({ mode, defect, indices: feedback.indices, overflow: feedback.overflow });
      device.queue.writeBuffer(currentPool.clusterLocations, firstCluster * 32, savedLocation);
    }
    recoverySelection.dispose();
    const singlePair = await singlePairCases(device, occlusion, mode => {
      device.queue.writeTexture({ texture: previous.texture }, new Float32Array([mode === 'first' ? 1 : 0]), { bytesPerRow: 4 }, [1, 1]);
      device.queue.writeTexture({ texture: current.texture }, new Float32Array([mode === 'reject' ? 0 : 1]), { bytesPerRow: 4 }, [1, 1]);
    }, readWords);
    selection.dispose(); currentPool.dispose(); await cache.dispose(); await device.queue.onSubmittedWorkDone();
    return { adapter: { vendor: adapter.info.vendor, isFallbackAdapter: adapter.info.isFallbackAdapter },
      pages: directory.pages.length, sourceTriangles: directory.sourceTriangleCount,
      rootOnly, fine, middle, frozen, farCount, coarse, culled, zero, allOffscreen, swapped, poolOwned, selectionOwned,
      stableSmall, stableLarge, boundaryBelow, boundaryZero, boundaryRegrow, recoveryCases, overflowCases, singlePair,
      pool: currentPool.diagnostics(), selection: selection.diagnostics(), errors };
  } finally {
    recoverySelection?.dispose(); selection?.dispose(); pool?.dispose();
    for (const buffer of testBuffers) buffer.destroy(); for (const texture of testTextures) texture.destroy(); await cache?.dispose(); instanceBuffer?.destroy(); device.destroy();
  }
}
