import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryPageCache } from '../../../src/geometry-stream-cache';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import { GpuGeometryPagePool } from '../../../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../../../src/gpu-geometry-stream-selection';
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
    const input = instances(models);
    instanceBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true });
    new Float32Array(instanceBuffer.getMappedRange()).set(input); instanceBuffer.unmap();
    selection = await GpuGeometryStreamSelection.create(device, currentPool, instanceBuffer, models.length,
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
    const triangleKeys = (words: Uint32Array) => {
      const keys: string[] = [];
      for (let i = 0; i < words.length; i += 3) keys.push(`${words[i]},${words[i + 1]},${words[i + 2]}`);
      return keys.sort();
    };
    const capture = async (label: string, count: number, maxErrorPixels: number, frustumCulling: boolean,
      compareIndices = false, expectOffscreen = false, camera: Camera = { eye, vp }, caseModels: readonly Mat4[] = models) => {
      if (caseModels !== models) device.queue.writeBuffer(instanceBuffer!, 0, instances(caseModels));
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
      if (compareIndices && actual.indices) {
        const pageIndices = await readWords(currentPool.indices, currentPool.indices.size);
        const compacted = await readWords(gpu.indices, actual.indices * 4);
        const expectedPhysical: number[] = [];
        for (const entry of actual.selectedClusters!) {
          if (!entry.visible) continue;
          const location = currentPool.clusterLocation(entry.clusterId)!;
          for (let i = 0; i < location.indexCount; i++) {
            expectedPhysical.push(pageIndices[location.firstIndex + i] + location.baseVertex
              + entry.instanceIndex * gpu.physicalVertexCount);
          }
        }
        same(triangleKeys(compacted), triangleKeys(Uint32Array.from(expectedPhysical)), label + ': PHYSICAL_INDICES');
      }
      return { label, count, maxErrorPixels, frustumCulling, clusters: expected.selected.length,
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
    const swapped = await capture('swapped-affine-instances', 2, 0, false, false, false,
      { eye, vp }, [models[1], models[0]]);
    same(swapped.desiredGroups, fine.desiredGroups, 'DEMAND_INSTANCE_MAX_ORDER');
    selection.dispose(); currentPool.dispose(); await cache.dispose(); await device.queue.onSubmittedWorkDone();
    return { adapter: { vendor: adapter.info.vendor, isFallbackAdapter: adapter.info.isFallbackAdapter },
      pages: directory.pages.length, sourceTriangles: directory.sourceTriangleCount,
      rootOnly, fine, middle, frozen, farCount, coarse, culled, zero, allOffscreen, swapped, poolOwned, selectionOwned,
      pool: currentPool.diagnostics(), selection: selection.diagnostics(), errors };
  } finally {
    selection?.dispose(); pool?.dispose(); await cache?.dispose(); instanceBuffer?.destroy(); device.destroy();
  }
}
