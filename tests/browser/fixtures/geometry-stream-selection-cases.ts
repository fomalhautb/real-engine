import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryPageCache } from '../../../src/geometry-stream-cache';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import { GpuGeometryPagePool } from '../../../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../../../src/gpu-geometry-stream-selection';
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
    const poolBudget = directory.pages.length * (verticesPerSlot * 128 + indicesPerSlot * 4)
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
    const reference = (resident: ReadonlySet<number>, count: number, maxErrorPixels: number) => {
      const plans = models.slice(0, count).map(model => planner.plan(resident,
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
      compareIndices = false, expectOffscreen = false, camera: { eye: readonly number[]; vp: Mat4 } = { eye, vp }) => {
      gpu.setSettings({ maxErrorPixels, frustumCulling });
      gpu.update(++frame, count, camera.vp, camera.eye, projectionScale);
      const resident = currentPool.residentPageIds();
      const lease = currentPool.acquireFrame([...resident]);
      const encoder = device.createCommandEncoder(); gpu.encode(encoder);
      const complete = await currentPool.submitFrame(lease, [encoder.finish()]);
      if (complete.error) throw new Error(complete.error);
      const actual = await gpu.feedback(true), expected = reference(resident, count, maxErrorPixels);
      if (actual.overflow) throw new Error(label + ': GPU_OVERFLOW');
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
        indices: actual.indices, requests: actual.requestedPages.length,
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
    selection.dispose(); currentPool.dispose(); await cache.dispose(); await device.queue.onSubmittedWorkDone();
    return { adapter: { vendor: adapter.info.vendor, isFallbackAdapter: adapter.info.isFallbackAdapter },
      pages: directory.pages.length, sourceTriangles: directory.sourceTriangleCount,
      rootOnly, fine, middle, frozen, farCount, coarse, culled, zero, poolOwned, selectionOwned,
      pool: currentPool.diagnostics(), selection: selection.diagnostics(), errors };
  } finally {
    selection?.dispose(); pool?.dispose(); await cache?.dispose(); instanceBuffer?.destroy(); device.destroy();
  }
}
