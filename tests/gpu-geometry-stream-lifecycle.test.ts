import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SceneBatch } from '../src/contracts';
import { GpuGeometryStream } from '../src/gpu-geometry-stream';
import type { GeometryStreamSelectionFeedback } from '../src/gpu-geometry-stream-selection';

const mocks = vi.hoisted(() => ({
  cacheCreate: vi.fn(), poolCreate: vi.fn(), selectionCreate: vi.fn(),
}));

vi.mock('../src/geometry-stream-cache', () => ({ GeometryPageCache: { create: mocks.cacheCreate } }));
vi.mock('../src/gpu-geometry-page-pool', () => ({ GpuGeometryPagePool: { create: mocks.poolCreate } }));
vi.mock('../src/gpu-geometry-stream-selection', () => ({ GpuGeometryStreamSelection: { create: mocks.selectionCreate } }));
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
    dispose: vi.fn(),
  };
  const selection = {
    capacity: 1, update: vi.fn(() => 0),
    feedback: vi.fn(async (): Promise<GeometryStreamSelectionFeedback> =>
      ({ frame: 1, requestedPages: [], desiredGroups: [], indices: 0, overflow: false })),
    diagnostics: vi.fn(() => ({})), dispose: vi.fn(),
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

beforeEach(() => { vi.resetAllMocks(); });

describe('GPU geometry stream bridge lifecycle', () => {
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
      desiredGroups: [{ groupId: 1, projectedError: 3 }], indices: 0, overflow: false });
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
    ], desiredGroups: [1, 2, 3].map(id => ({ groupId: id, projectedError: 4 - id })), indices: 0, overflow: false });

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
      desiredGroups: [1, 2].map(groupId => ({ groupId, projectedError: 1 })), indices: 0, overflow: false });
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
      desiredGroups: [{ groupId: 1, projectedError: 2 }], indices: 0, overflow: false });
    await feedbackSettled(stream);
    stream.update(1, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(cache.setRequestWindow).toHaveBeenLastCalledWith([1]);
    pool.residentPageIds.mockReturnValue(new Set([0, 3]));
    selection.feedback.mockResolvedValue({ frame: 2, requestedPages: [],
      desiredGroups: [{ groupId: 2, projectedError: 2 }], indices: 0, overflow: false });
    await feedbackSettled(stream);
    stream.update(2, 1, new Float32Array(16), [0, 0, 0], 1);
    expect(cache.setRequestWindow).toHaveBeenLastCalledWith([2]);
    expect(pool.retainPendingPages).toHaveBeenLastCalledWith([0, 2]);
    expect(pool.setPinnedPages).toHaveBeenLastCalledWith([0]);
    await stream.dispose();
  });
});
