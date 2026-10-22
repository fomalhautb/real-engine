import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { RenderBatch, SceneBatch } from '../src/contracts';
import { GpuGeometryStream } from '../src/gpu-geometry-stream';
import type { GeometryStreamSelectionFeedback } from '../src/gpu-geometry-stream-selection';

const mocks = vi.hoisted(() => ({
  cacheCreate: vi.fn(), poolCreate: vi.fn(), selectionCreate: vi.fn(), validateProxy: vi.fn(),
}));
const zeroOcclusion = { deferredClusters: 0, culledClusters: 0, recoveredClusters: 0 };

vi.mock('../src/geometry-stream-cache', () => ({ GeometryPageCache: { create: mocks.cacheCreate } }));
vi.mock('../src/gpu-geometry-page-pool', () => ({ GpuGeometryPagePool: { create: mocks.poolCreate } }));
vi.mock('../src/gpu-geometry-stream-selection', () => ({ GpuGeometryStreamSelection: { create: mocks.selectionCreate } }));
vi.mock('../src/geometry-query-proxy', () => ({ validateGeometryQueryProxy: mocks.validateProxy }));
vi.mock('../src/geometry-stream-working-set', () => ({ GeometryStreamWorkingSet: class {
  plan(demands: { groupId: number; projectedError: number }[]) {
    return { pages: [0, ...demands.map(d => d.groupId)], admittedGroups: [], budgetDeferredGroups: [], dependencyDeferredGroups: [],
      requestedPages: demands.map(d => ({ pageId: d.groupId, priority: d.projectedError })) };
  }
} }));

function setup(maxRequestsPerFrame = 1) {
  const page = { clusters: [] };
  const directory = {
    sourceVertexCount: 1, sourceTriangleCount: 1, vertexStride: 80,
    rootPages: [0], pages: [0, 1, 2, 3].map(id => ({ id, decodedBytes: 100, clusterIds: [] })), clusters: [],
  };
  const scene = {
    geometryStream: { directory, provider: vi.fn(), maxRequestsPerFrame },
    geometryLod: undefined, deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    mesh: { vertexCount: 1, vertices: new ArrayBuffer(80), indices: new Uint32Array([0, 0, 0]) },
    capacity: 1,
  } as unknown as SceneBatch;
  const cache = {
    get: vi.fn((id: number) => id === 0 ? page : undefined),
    request: vi.fn(), retry: vi.fn(), setRequestWindow: vi.fn(),
    diagnostics: vi.fn(() => ({ failedPages: [], limits: { maxResidentBytes: 400 } })),
    dispose: vi.fn(async () => {}),
  };
  const pool = {
    directory, slotCount: 4, enqueue: vi.fn(), residentPageIds: vi.fn(() => new Set([0])),
    retainPendingPages: vi.fn(), setPinnedPages: vi.fn(),
    uploadFrame: vi.fn(() => ({ bytes: 0 })),
    diagnostics: vi.fn(() => ({ pendingPages: [] as number[], stagingBytes: 0, limits: { maxQueuedPages: 4, maxStagingBytes: 1024 } })),
    dispose: vi.fn(), migrateResident: vi.fn(),
  };
  const selection = {
    capacity: 1, update: vi.fn(() => 0),
    feedback: vi.fn(async (): Promise<GeometryStreamSelectionFeedback> =>
      ({ frame: 1, requestedPages: [], desiredGroups: [], indices: 0, overflow: false, occlusion: zeroOcclusion })),
    diagnostics: vi.fn(() => ({})), dispose: vi.fn(), copyViewStateFrom: vi.fn(),
  };
  const device = { queue: { onSubmittedWorkDone: vi.fn(async () => {}) } } as unknown as GPUDevice;
  mocks.cacheCreate.mockResolvedValue(cache);
  mocks.poolCreate.mockResolvedValue(pool);
  mocks.selectionCreate.mockResolvedValue(selection);
  return { scene, device, cache, pool, selection };
}

async function feedbackSettled(stream: GpuGeometryStream): Promise<void> {
  stream.afterSubmit();
  await (stream as unknown as { feedbackTask: Promise<void> }).feedbackTask;
}

beforeEach(() => { vi.resetAllMocks(); mocks.validateProxy.mockResolvedValue(undefined); });

describe('GPU geometry stream bridge lifecycle', () => {
  it('prepares a replacement with the same cache, commits synchronously, and retires old owners after feedback and fence', async () => {
    const { scene, device, cache, pool, selection } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    let finishFeedback!: (value: GeometryStreamSelectionFeedback) => void, finishFence!: () => void;
    selection.feedback.mockImplementationOnce(() => new Promise(resolve => { finishFeedback = resolve; }));
    stream.afterSubmit();
    const nextPool = { ...pool, slotCount: 3, dispose: vi.fn() };
    const nextSelection = { ...selection, feedback: vi.fn(async (): Promise<GeometryStreamSelectionFeedback> =>
      ({ frame: 2, requestedPages: [], desiredGroups: [{ groupId: 2, projectedError: 5 }], indices: 0, overflow: false, occlusion: zeroOcclusion })),
      dispose: vi.fn(), copyViewStateFrom: vi.fn() };
    pool.migrateResident.mockResolvedValue({ pool: nextPool, copiedPages: [0], copiedBytes: 100, peakBytes: 500 });
    mocks.selectionCreate.mockResolvedValueOnce(nextSelection);
    const candidate = await stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 });
    expect(stream.pool).toBe(pool); expect(stream.selection).toBe(selection); expect(cache.dispose).not.toHaveBeenCalled();
    expect(() => stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 })).toThrow('CAPACITY_BUSY');
    vi.mocked(device.queue.onSubmittedWorkDone).mockImplementationOnce(() => new Promise(resolve => { finishFence = () => resolve(undefined); }));
    const commit = stream.commitCapacity(candidate);
    expect(stream.pool).toBe(nextPool); expect(stream.selection).toBe(nextSelection); expect(stream.capacityEpoch).toBe(1);
    expect(nextSelection.copyViewStateFrom).toHaveBeenLastCalledWith(selection);
    expect(pool.dispose).not.toHaveBeenCalled(); expect(selection.dispose).not.toHaveBeenCalled();
    expect(() => stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 })).toThrow('CAPACITY_BUSY');
    await feedbackSettled(stream);
    finishFeedback({ frame: 1, requestedPages: [], desiredGroups: [{ groupId: 3, projectedError: 100 }], indices: 0,
      overflow: true, occlusion: zeroOcclusion });
    await Promise.resolve(); expect(stream.diagnostics().error).toBeNull();
    finishFence(); await commit.retired;
    expect(stream.diagnostics().allocation.pages).toEqual([0, 2]);
    expect(pool.dispose).toHaveBeenCalledOnce(); expect(selection.dispose).toHaveBeenCalledOnce();
    expect(() => stream.commitCapacity(candidate)).toThrow('CAPACITY_STALE');
    await stream.dispose(); expect(cache.dispose).toHaveBeenCalledOnce();
  });

  it('aborts an unused replacement and preserves the original stream when selection creation fails', async () => {
    const { scene, device, pool, selection } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    const nextPool = { ...pool, slotCount: 2, dispose: vi.fn() }, nextSelection = { ...selection, dispose: vi.fn() };
    pool.migrateResident.mockResolvedValue({ pool: nextPool, copiedPages: [0], copiedBytes: 100, peakBytes: 500 });
    mocks.selectionCreate.mockRejectedValueOnce(new Error('FAKE_SELECTION_FAILURE'));
    await expect(stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 })).rejects.toThrow('FAKE_SELECTION_FAILURE');
    expect(nextPool.dispose).toHaveBeenCalledOnce(); expect(pool.dispose).not.toHaveBeenCalled(); expect(stream.pool).toBe(pool);
    mocks.selectionCreate.mockResolvedValueOnce(nextSelection);
    const candidate = await stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 });
    stream.abortCapacity(candidate);
    expect(nextSelection.dispose).toHaveBeenCalledOnce(); expect(stream.pool).toBe(pool);
    expect(() => stream.commitCapacity(candidate)).toThrow('CAPACITY_STALE'); await stream.dispose();
  });

  it('awaits and cleans an in-progress preparation when the stream is disposed', async () => {
    const { scene, device, pool } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    let finishMigration!: (value: unknown) => void;
    pool.migrateResident.mockImplementationOnce(() => new Promise(resolve => { finishMigration = resolve; }));
    const pending = stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 });
    const rejection = expect(pending).rejects.toThrow('DISPOSED');
    let cleaned = false; const cleanup = stream.dispose().then(() => { cleaned = true; });
    await Promise.resolve(); expect(cleaned).toBe(false);
    const candidatePool = { ...pool, dispose: vi.fn() };
    finishMigration({ pool: candidatePool, copiedPages: [0], copiedBytes: 100, peakBytes: 500 });
    await rejection; await cleanup; expect(cleaned).toBe(true); expect(candidatePool.dispose).toHaveBeenCalledOnce();
  });

  it('still waits for migration cleanup when cache disposal throws synchronously', async () => {
    const { scene, device, cache, pool } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    let finishMigration!: (value: unknown) => void;
    pool.migrateResident.mockImplementationOnce(() => new Promise(resolve => { finishMigration = resolve; }));
    const pending = stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 });
    const rejection = expect(pending).rejects.toThrow('DISPOSED');
    cache.dispose.mockImplementationOnce(() => { throw new Error('FAKE_SYNC_CACHE_FAILURE'); });
    let cleaned = false;
    const cleanup = stream.dispose(); void cleanup.then(() => { cleaned = true; }, () => { cleaned = true; });
    const cleanupRejected = expect(cleanup).rejects.toThrow('GEOMETRY_STREAM_CLEANUP_FAILED');
    await Promise.resolve(); expect(cleaned).toBe(false);
    const candidatePool = { ...pool, dispose: vi.fn() };
    finishMigration({ pool: candidatePool, copiedPages: [0], copiedBytes: 100, peakBytes: 500 });
    await rejection; await cleanupRejected; expect(cleaned).toBe(true);
    expect(candidatePool.dispose).toHaveBeenCalledOnce();
  });

  it('reports a retirement fence failure and releases the old GPU owners', async () => {
    const { scene, device, pool, selection } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    const nextPool = { ...pool, slotCount: 3, dispose: vi.fn() }, nextSelection = { ...selection, dispose: vi.fn() };
    pool.migrateResident.mockResolvedValue({ pool: nextPool, copiedPages: [0], copiedBytes: 100, peakBytes: 500 });
    mocks.selectionCreate.mockResolvedValueOnce(nextSelection);
    const candidate = await stream.prepareCapacity({ maxResidentBytes: 300, maxPeakBytes: 500 });
    vi.mocked(device.queue.onSubmittedWorkDone).mockRejectedValueOnce(new Error('FAKE_RETIRE_QUEUE_FAILURE'));
    await expect(stream.commitCapacity(candidate).retired).rejects.toThrow('CAPACITY_RETIRE_FAILED');
    expect(stream.diagnostics().error).toContain('FAKE_RETIRE_QUEUE_FAILURE');
    expect(pool.dispose).toHaveBeenCalledOnce(); expect(selection.dispose).toHaveBeenCalledOnce();
    await stream.dispose(); expect(nextPool.dispose).toHaveBeenCalledOnce(); expect(nextSelection.dispose).toHaveBeenCalledOnce();
  });
  it('rejects proxy budgets without a proxy before GPU allocation', async () => {
    const { scene, device } = setup();
    const invalid = { ...scene, geometryStream: { ...scene.geometryStream!, queryProxyLimits: { maxTriangles: 1 } } };
    await expect(GpuGeometryStream.create(device, invalid, {} as GPUBuffer, () => {}))
      .rejects.toThrow('GEOMETRY_STREAM_QUERY_PROXY_LIMITS_WITHOUT_PROXY');
    expect(mocks.poolCreate).not.toHaveBeenCalled();
  });

  it('forwards explicit proxy budgets and propagates validation failure before GPU allocation', async () => {
    const { scene, device } = setup();
    const proxy = { mesh: scene.mesh } as NonNullable<typeof scene.geometryStream>['queryProxy'];
    const limits = { maxVertices: 4, maxTriangles: 2, maxBytes: 1024 };
    const paged = { ...scene, mesh: undefined, geometryStream: { ...scene.geometryStream!, queryProxy: proxy,
      queryProxyLimits: limits } } as unknown as RenderBatch;
    mocks.validateProxy.mockRejectedValueOnce(new RangeError('GEOMETRY_QUERY_PROXY_TRIANGLE_BUDGET'));
    await expect(GpuGeometryStream.create(device, paged, {} as GPUBuffer, () => {}))
      .rejects.toThrow('GEOMETRY_QUERY_PROXY_TRIANGLE_BUDGET');
    expect(mocks.validateProxy).toHaveBeenCalledWith(paged.geometryStream!.directory, proxy, limits);
    expect(mocks.poolCreate).not.toHaveBeenCalled();

    const stream = await GpuGeometryStream.create(device, paged, {} as GPUBuffer, () => {});
    expect(mocks.validateProxy).toHaveBeenLastCalledWith(paged.geometryStream!.directory, proxy, limits);
    await stream.dispose();
  });

  it('preserves initialization failure and cleans every owner when selection disposal throws', async () => {
    const { scene, device, cache, pool, selection } = setup();
    const initial = new Error('FAKE_ROOT_ENQUEUE_FAILURE');
    pool.enqueue.mockImplementation(() => { throw initial; });
    selection.dispose.mockImplementation(() => { throw new Error('FAKE_TRACKER_FAILURE'); });

    await expect(GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {})).rejects.toBe(initial);
    expect(selection.dispose).toHaveBeenCalledOnce();
    expect(pool.dispose).toHaveBeenCalledOnce();
    expect(cache.dispose).toHaveBeenCalledOnce();
  });

  it('starts both GPU cleanups synchronously, awaits cache cleanup and shares its disposal promise', async () => {
    const { scene, device, cache, pool, selection } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    let completeCache!: () => void;
    let cacheFinished = false;
    cache.dispose.mockImplementation(() => new Promise<void>(resolve => {
      completeCache = () => { cacheFinished = true; resolve(); };
    }));
    selection.dispose.mockImplementation(() => { throw new Error('FAKE_TRACKER_FAILURE'); });
    pool.dispose.mockImplementation(() => { throw new Error('FAKE_POOL_TRACKER_FAILURE'); });

    let cleanup!: Promise<void>;
    expect(() => { cleanup = stream.dispose(); }).not.toThrow();
    expect(cleanup).toBeInstanceOf(Promise);
    expect(selection.dispose).toHaveBeenCalledOnce();
    expect(pool.dispose).toHaveBeenCalledOnce();
    expect(cache.dispose).toHaveBeenCalledOnce();
    expect(stream.dispose()).toBe(cleanup);
    expect(cacheFinished).toBe(false);
    completeCache();
    await cleanup.catch(() => {});
    expect(cacheFinished).toBe(true);
    expect(selection.dispose).toHaveBeenCalledOnce();
    expect(pool.dispose).toHaveBeenCalledOnce();
    expect(cache.dispose).toHaveBeenCalledOnce();
  });

  it('defers a request when the CPU cache queue is saturated', async () => {
    const { scene, device, cache, pool, selection } = setup();
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    selection.feedback.mockResolvedValue({ frame: 1, requestedPages: [{ pageId: 1, priority: 3 }],
      desiredGroups: [{ groupId: 1, projectedError: 3 }], indices: 0, overflow: false, occlusion: zeroOcclusion });
    cache.request.mockImplementation(() => { throw new Error('GEOMETRY_STREAM_REQUEST_BUDGET'); });

    await feedbackSettled(stream);
    expect(() => stream.update(1, 1, new Float32Array(16), [0, 0, 0], 1)).not.toThrow();
    expect(cache.request).toHaveBeenCalledWith(1, 3);
    expect(pool.uploadFrame).toHaveBeenCalledOnce();
    expect(selection.update).toHaveBeenCalledOnce();
    cache.request.mockClear();
    cache.request.mockImplementation(() => {});
    stream.update(2, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(cache.request).toHaveBeenCalledWith(1, 3);
  });

  it('advances the pending cursor across repeated feedback when one request is allowed per frame', async () => {
    const { scene, device, cache, pool, selection } = setup(1);
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    selection.feedback.mockResolvedValue({ frame: 1, requestedPages: [
      { pageId: 1, priority: 3 }, { pageId: 2, priority: 2 }, { pageId: 3, priority: 1 },
    ], desiredGroups: [1, 2, 3].map(id => ({ groupId: id, projectedError: 4 - id })),
      indices: 0, overflow: false, occlusion: zeroOcclusion });

    for (let frame = 1; frame <= 3; frame++) {
      await feedbackSettled(stream);
      stream.update(frame, 1, new Float32Array(16), [0, 0, 0], 1);
      pool.residentPageIds.mockReturnValue(new Set(Array.from({ length: frame + 1 }, (_, id) => id)));
    }
    expect(cache.request.mock.calls.map(([pageId]) => pageId)).toEqual([1, 2, 3]);
  });

  it('keeps a slow CPU transfer pinned until upload staging owns it, then advances within the byte budget', async () => {
    const { scene, device, cache, pool, selection } = setup(2);
    cache.diagnostics.mockReturnValue({ failedPages: [], limits: { maxResidentBytes: 200 } });
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    selection.feedback.mockResolvedValue({ frame: 1, requestedPages: [],
      desiredGroups: [1, 2].map(groupId => ({ groupId, projectedError: 1 })),
      indices: 0, overflow: false, occlusion: zeroOcclusion });
    for (let frame = 1; frame <= 4; frame++) {
      await feedbackSettled(stream);
      stream.update(frame, 1, new Float32Array(16), [0, 0, 0], 1);
      expect(cache.setRequestWindow).toHaveBeenLastCalledWith([1]);
    }
    expect(cache.request.mock.calls.map(([id]) => id)).toEqual([1, 1, 1, 1]);
    cache.get.mockImplementation(id => id <= 1 ? { clusters: [] } : undefined);
    stream.update(5, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(pool.enqueue).toHaveBeenLastCalledWith(1, { clusters: [] }, 1);
    pool.diagnostics.mockReturnValue({ pendingPages: [1], stagingBytes: 4,
      limits: { maxQueuedPages: 4, maxStagingBytes: 1024 } });
    stream.update(6, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(cache.setRequestWindow).toHaveBeenLastCalledWith([2]);
    expect(cache.request).toHaveBeenLastCalledWith(2, 1);
    await stream.dispose();
  });

  it('drops obsolete transfers when the demanded group changes and retains only desired resident pages', async () => {
    const { scene, device, cache, pool, selection } = setup(1);
    const stream = await GpuGeometryStream.create(device, scene, {} as GPUBuffer, () => {});
    selection.feedback.mockResolvedValue({ frame: 1, requestedPages: [],
      desiredGroups: [{ groupId: 1, projectedError: 2 }], indices: 0, overflow: false, occlusion: zeroOcclusion });
    await feedbackSettled(stream);
    stream.update(1, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(cache.setRequestWindow).toHaveBeenLastCalledWith([1]);
    pool.residentPageIds.mockReturnValue(new Set([0, 3]));
    selection.feedback.mockResolvedValue({ frame: 2, requestedPages: [],
      desiredGroups: [{ groupId: 2, projectedError: 2 }], indices: 0, overflow: false, occlusion: zeroOcclusion });
    await feedbackSettled(stream);
    stream.update(2, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(cache.setRequestWindow).toHaveBeenLastCalledWith([2]);
    expect(pool.retainPendingPages).toHaveBeenLastCalledWith([0, 2]);
    expect(pool.setPinnedPages).toHaveBeenLastCalledWith([0]);
    await stream.dispose();
  });
});
