import { describe, expect, it, vi } from 'vitest';
import { GeometryStreamCapacityCoordinator, type GeometryStreamCapacityTarget } from '../src/geometry-stream-capacity-coordinator';
import { GeometryStreamRuntimeBudget } from '../src/geometry-stream-runtime-budget';
import type { GpuGeometryStream, PreparedGeometryStreamCapacity } from '../src/gpu-geometry-stream';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import type { GeometryGroupDemand } from '../src/geometry-stream-working-set';
import { geometryGpuPoolLayout } from '../src/geometry-gpu-pool-layout';

const limits = { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };
function directory(): GeometryStreamDirectory {
  const bounds = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number] };
  return { version: 1, primitiveKey: 'coordinator', sourceVertexCount: 6, sourceTriangleCount: 2, vertexStride: 80,
    clusters: [0, 1, 2, 3, 4, 5].map(id => ({ level: Math.floor(id / 2), group: id < 4 ? id : 4,
      refinedGroup: id < 2 ? null : id - 2, vertexCount: 3, triangleCount: 1, bounds, lodBounds: bounds, error: id < 2 ? 0 : id < 4 ? 1 : 4 })),
    groups: [0, 1, 2, 3].map(id => ({ level: Math.floor(id / 2), children: [id], parents: [id + 2], bounds, error: id < 2 ? 1 : 4, terminal: false }))
      .concat([{ level: 2, children: [4, 5], parents: [], bounds, error: 4, terminal: true }]),
    roots: [4, 5], pages: [0, 1, 2, 3, 4, 5].map(id => ({ id, clusterIds: [id], encodedBytes: 41, decodedBytes: 272, sha256: 'a'.repeat(64) })),
    clusterPages: [0, 1, 2, 3, 4, 5], rootPages: [4, 5] };
}
function deferred<T>() { let resolve!: (value: T) => void, reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
async function settle() { for (let step = 0; step < 8; step++) await Promise.resolve(); }
const near = [{ groupId: 2, projectedError: 100 }, { groupId: 0, projectedError: 50 }];
type Bindings = { pool: object; physicalVertexCount: number; instanceHistory: object; dispose: () => void };

function setup(config: { slots?: number[]; concurrent?: number; peakCopies?: number; copyBytes?: number; copyPages?: number } = {}) {
  const initial = config.slots ?? [4, 2];
  const asset = directory(), layout = geometryGpuPoolLayout(asset), inputs = initial.map(() => ({ directory: asset, capacity: 1 }));
  const reference = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 1e6, maxPeakBytes: 2e6 }, limits);
  const resident = initial.reduce((sum, slots, index) => sum + reference.state(index, slots).totalBytes, 0);
  const options = { maxResidentBytes: resident, maxPeakBytes: resident + (config.peakCopies ?? 1) * reference.state(0, 2).totalBytes, intervalFrames: 1,
    maxConcurrentMigrations: config.concurrent ?? 1, maxCopyBytes: config.copyBytes, maxCopyPages: config.copyPages };
  const budget = new GeometryStreamRuntimeBudget(inputs, options, limits);
  let owned = 0;
  const owner = (bytes: number) => { owned += bytes; let disposed = false; return vi.fn(() => { if (!disposed) { disposed = true; owned -= bytes; } }); };
  const fixtures = initial.map((slots, index) => {
    const makePool = (count: number, pages: number[]) => ({ directory: asset, slotCount: count,
      residentPageIds: () => new Set(pages), diagnostics: () => ({ framesInFlight: 0, limits: { maxFramesInFlight: 3 } }) });
    let pool = makePool(slots, slots === 4 ? [0, 2, 4, 5] : [4, 5]);
    let epoch = 0;
    let release: () => void = owner(budget.state(index, slots).poolBytes + budget.state(index, slots).selectionBytes);
    let candidate: PreparedGeometryStreamCapacity | undefined, releaseCandidate: (() => void) | undefined;
    const fences: ReturnType<typeof deferred<void>>[] = [], history = {};
    const runtime = {
      get pool() { return pool; }, get capacityEpoch() { return epoch; },
      capacityDemand: [] as readonly GeometryGroupDemand[] | null,
      prepareCapacity: vi.fn(async (args: { maxResidentBytes: number; retainedPages: number[] }) => {
        const count = (args.maxResidentBytes - layout.metadataBytes) / layout.bytesPerSlot;
        const next = makePool(count, args.retainedPages);
        releaseCandidate = owner(budget.state(index, count).poolBytes + budget.state(index, count).selectionBytes);
        candidate = { pool: next, selection: { physicalVertexCount: count * layout.verticesPerSlot }, sourceEpoch: epoch } as unknown as PreparedGeometryStreamCapacity;
        return candidate;
      }),
      commitCapacity: vi.fn((next: PreparedGeometryStreamCapacity) => {
        if (next !== candidate || next.sourceEpoch !== epoch) throw new Error('STALE');
        const oldRelease = release; release = releaseCandidate!; releaseCandidate = undefined;
        pool = next.pool as unknown as typeof pool; candidate = undefined; epoch++;
        const fence = deferred<void>(); fences.push(fence);
        return { epoch, retired: fence.promise.finally(oldRelease) };
      }),
      abortCapacity: vi.fn(() => { releaseCandidate?.(); releaseCandidate = undefined; candidate = undefined; }),
      dispose: () => { release(); releaseCandidate?.(); },
      changeEpoch: () => { epoch++; },
    };
    const bindings = (nextPool: object, count: number): Bindings => ({ pool: nextPool, physicalVertexCount: count * layout.verticesPerSlot,
      instanceHistory: history, dispose: owner(budget.state(index, count).outputBytes + 32) });
    const target: GeometryStreamCapacityTarget<Bindings> = { runtime: runtime as unknown as GpuGeometryStream,
      bindings: bindings(pool, slots), prepareBindings: vi.fn(async next => bindings(next.pool, next.pool.slotCount)) };
    return { runtime, target, fences, history };
  });
  const coordinator = new GeometryStreamCapacityCoordinator(budget, fixtures.map(f => f.target), options);
  const close = async () => {
    for (const fixture of fixtures) { for (const fence of fixture.fences) fence.resolve(); }
    await coordinator.dispose(); for (const fixture of fixtures) { fixture.runtime.dispose(); fixture.target.bindings.dispose(); }
  };
  return { budget, options, coordinator, fixtures, close, owned: () => owned };
}

describe('Renderer capacity transaction coordinator', () => {
  it('exchanges two streams from GPU camera demand, switches every dependent owner together, and reserves through retirement', async () => {
    const s = setup(), [a, b] = s.fixtures, oldA = a.target.bindings;
    b.runtime.capacityDemand = near;
    s.coordinator.tick(1);
    expect(s.coordinator.diagnostics()).toMatchObject({ phase: 'preparing', reservedBytes: s.budget.state(0, 2).totalBytes });
    for (let frame = 2; frame < 20; frame++) s.coordinator.tick(frame);
    expect(a.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(b.runtime.prepareCapacity).not.toHaveBeenCalled();
    await settle(); expect(s.coordinator.diagnostics().phase).toBe('ready');
    expect(a.target.bindings).toBe(oldA); expect(a.runtime.pool.slotCount).toBe(4);
    s.coordinator.tick(20);
    expect(a.runtime.pool.slotCount).toBe(2); expect(a.target.bindings.pool).toBe(a.runtime.pool);
    expect(a.target.bindings.physicalVertexCount).toBe(6); expect(a.target.bindings.instanceHistory).toBe(a.history);
    expect([...a.runtime.pool.residentPageIds()]).toEqual([4, 5]);
    expect(oldA.dispose).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ phase: 'retiring', reservedBytes: s.budget.state(0, 4).totalBytes });
    s.coordinator.tick(21); expect(b.runtime.prepareCapacity).not.toHaveBeenCalled();
    a.fences[0].resolve(); await settle(); expect(oldA.dispose).toHaveBeenCalledOnce();
    s.coordinator.tick(22); await settle(); s.coordinator.tick(23);
    expect(b.runtime.pool.slotCount).toBe(4); expect(b.target.bindings.pool).toBe(b.runtime.pool);
    expect(b.target.bindings.physicalVertexCount).toBe(12); expect(b.target.bindings.instanceHistory).toBe(b.history);
    expect([...b.runtime.pool.residentPageIds()]).toEqual([4, 5]); // Unloaded detail falls back to mandatory roots.
    b.fences[0].resolve(); await settle();
    expect(s.coordinator.diagnostics()).toMatchObject({ migrations: 2, phase: 'idle', error: null, reservedBytes: 0 });
    expect(s.coordinator.diagnostics().peakBytes).toBeLessThanOrEqual(s.options.maxPeakBytes);
    expect(s.owned()).toBeLessThanOrEqual(s.options.maxResidentBytes);
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('keeps original owners usable after candidate binding failure and releases its full reservation', async () => {
    const s = setup(), [a, b] = s.fixtures, old = a.target.bindings, before = s.owned();
    b.runtime.capacityDemand = near;
    vi.mocked(a.target.prepareBindings).mockRejectedValueOnce(new Error('ALLOCATION_FAILED'));
    s.coordinator.tick(1); await settle();
    expect(a.target.bindings).toBe(old); expect(a.runtime.pool.slotCount).toBe(4);
    expect(a.runtime.abortCapacity).toHaveBeenCalledOnce(); expect(a.runtime.commitCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ error: 'Error: ALLOCATION_FAILED', reservedBytes: 0 });
    expect(s.owned()).toBe(before); await s.close(); expect(s.owned()).toBe(0);
  });

  it('rejects a candidate whose source epoch changed before the frame boundary', async () => {
    const s = setup(), [a, b] = s.fixtures;
    b.runtime.capacityDemand = near; s.coordinator.tick(1); await settle(); a.runtime.changeEpoch(); s.coordinator.tick(2);
    expect(a.runtime.commitCapacity).not.toHaveBeenCalled(); expect(a.runtime.abortCapacity).toHaveBeenCalledOnce();
    expect(s.coordinator.diagnostics()).toMatchObject({ error: 'Error: GEOMETRY_CAPACITY_STALE', reservedBytes: 0 });
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('waits for asynchronous binding preparation during dispose and cleans late candidate owners', async () => {
    const s = setup(), [a, b] = s.fixtures, pending = deferred<Bindings>();
    const original = a.target.prepareBindings;
    let prepared: Bindings | undefined;
    a.target.prepareBindings = async candidate => { prepared = await original(candidate); return pending.promise; };
    b.runtime.capacityDemand = near; s.coordinator.tick(1); await settle();
    let finished = false; const cleanup = s.coordinator.dispose().then(() => { finished = true; });
    await settle(); expect(finished).toBe(false);
    pending.resolve(prepared!); await cleanup;
    expect(prepared!.dispose).toHaveBeenCalledOnce(); expect(a.runtime.abortCapacity).toHaveBeenCalledOnce();
    expect(s.coordinator.diagnostics()).toMatchObject({ phase: 'disposed', liveBytes: 0, reservedBytes: 0 });
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('waits for retirement rejection, releases old bindings and never starts a competing transaction', async () => {
    const s = setup(), [a, b] = s.fixtures, old = a.target.bindings;
    b.runtime.capacityDemand = near; s.coordinator.tick(1); await settle(); s.coordinator.tick(2);
    a.fences[0].reject(new Error('QUEUE_LOST')); await settle();
    expect(old.dispose).toHaveBeenCalledOnce(); expect(s.coordinator.diagnostics()).toMatchObject({ error: 'Error: QUEUE_LOST', reservedBytes: 0 });
    s.coordinator.tick(3); expect(b.runtime.prepareCapacity).not.toHaveBeenCalled();
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('does not plan before all streams have feedback and rejects root copies outside the bounded workload', async () => {
    const s = setup(), [a, b] = s.fixtures;
    a.runtime.capacityDemand = null; b.runtime.capacityDemand = near; s.coordinator.tick(1); await settle();
    expect(a.runtime.prepareCapacity).not.toHaveBeenCalled();
    expect(() => new GeometryStreamCapacityCoordinator(s.budget, s.fixtures.map(f => f.target), { ...s.options, maxCopyPages: 1 }))
      .toThrow('ROOT_COPY_BUDGET');
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('overlaps independent prepares and retirement while charging every old/new owner', async () => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 2 }), [a, b, c, d] = s.fixtures;
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    s.coordinator.tick(1); await settle();
    expect(a.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(b.runtime.prepareCapacity).toHaveBeenCalledOnce();
    expect(c.runtime.prepareCapacity).not.toHaveBeenCalled(); expect(d.runtime.prepareCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ inFlightMigrations: 2, phase: 'ready', accountedBytes: s.options.maxPeakBytes });
    s.coordinator.tick(2);
    expect(a.runtime.pool.slotCount).toBe(2); expect(b.runtime.pool.slotCount).toBe(2);
    expect(s.coordinator.diagnostics()).toMatchObject({ migrations: 2, phase: 'retiring', accountedBytes: s.options.maxPeakBytes });
    // A's retired bytes fund C while B remains protected by its outstanding fence.
    a.fences[0].resolve(); await settle(); s.coordinator.tick(3); await settle();
    expect(c.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(d.runtime.prepareCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ inFlightMigrations: 2, phase: 'ready', accountedBytes: s.options.maxPeakBytes });
    expect(s.owned()).toBe(s.coordinator.diagnostics().accountedBytes);
    s.coordinator.tick(4); expect(c.target.bindings.pool).toBe(c.runtime.pool);
    expect(c.target.bindings.instanceHistory).toBe(c.history);
    b.fences[0].resolve(); c.fences[0].resolve(); await settle(); s.coordinator.tick(5); await settle(); s.coordinator.tick(6);
    expect(s.fixtures.map(f => f.runtime.pool.slotCount)).toEqual([2, 2, 4, 4]);
    expect(s.coordinator.diagnostics().peakBytes).toBeLessThanOrEqual(s.options.maxPeakBytes);
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('reserves uncommitted growth without borrowing a slow uncommitted shrink', async () => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 8 }), [a, b, c, d] = s.fixtures;
    const pending = deferred<Bindings>(), original = b.target.prepareBindings; let delayed: Bindings | undefined;
    b.target.prepareBindings = async candidate => { delayed = await original(candidate); return pending.promise; };
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    s.coordinator.tick(1); await settle(); s.coordinator.tick(2); await settle();
    expect(a.runtime.pool.slotCount).toBe(2); expect(b.runtime.pool.slotCount).toBe(4);
    expect(c.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(d.runtime.prepareCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics().projectedLiveBytes).toBe(s.options.maxResidentBytes);
    expect(s.coordinator.diagnostics().pendingGrowthBytes).toBe(s.budget.state(2, 4).totalBytes - s.budget.state(2, 2).totalBytes);
    s.coordinator.tick(3); expect(c.runtime.pool.slotCount).toBe(4);
    expect(s.coordinator.diagnostics().liveBytes).toBe(s.options.maxResidentBytes);
    expect(d.runtime.prepareCapacity).not.toHaveBeenCalled();
    pending.resolve(delayed!); await settle(); s.coordinator.tick(4); await settle(); s.coordinator.tick(5);
    expect(s.fixtures.map(f => f.runtime.pool.slotCount)).toEqual([2, 2, 4, 4]);
    expect(s.coordinator.diagnostics().projectedLiveBytes).toBeLessThanOrEqual(s.options.maxResidentBytes);
    await s.close(); expect(s.owned()).toBe(0);
  });

  it.each([{ copyBytes: 888 }, { copyPages: 2 }])('caps aggregate in-flight copy work through retirement: %j', async copy => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 8, ...copy }), [a, b, c, d] = s.fixtures;
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    s.coordinator.tick(1); await settle();
    expect(a.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(b.runtime.prepareCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics().copyReservations).toEqual({ bytes: 888, pages: 2 });
    s.coordinator.tick(2); s.coordinator.tick(3);
    expect(b.runtime.prepareCapacity).not.toHaveBeenCalled();
    a.fences[0].resolve(); await settle(); s.coordinator.tick(4); await settle();
    expect(b.runtime.prepareCapacity).toHaveBeenCalledOnce();
    expect(s.coordinator.diagnostics().copyReservations).toEqual({ bytes: 888, pages: 2 });
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('discards ready peers and waits for late preparation when another stream fails', async () => {
    const s = setup({ slots: [4, 4, 4, 2, 2, 2], concurrent: 4, peakCopies: 8 }), [a, b, c] = s.fixtures;
    for (const fixture of s.fixtures.slice(3)) fixture.runtime.capacityDemand = near;
    const failure = deferred<Bindings>(), late = deferred<Bindings>(), original = c.target.prepareBindings;
    let delayed: Bindings | undefined;
    b.target.prepareBindings = () => failure.promise;
    c.target.prepareBindings = async candidate => { delayed = await original(candidate); return late.promise; };
    const before = s.owned(); s.coordinator.tick(1); await settle();
    expect(s.coordinator.diagnostics().inFlightMigrations).toBe(3);
    failure.reject(new Error('PEER_ALLOCATION_FAILED')); await settle();
    expect(a.runtime.abortCapacity).toHaveBeenCalledOnce(); expect(a.runtime.commitCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ error: 'Error: PEER_ALLOCATION_FAILED', inFlightMigrations: 1 });
    let disposed = false; const completion = s.coordinator.dispose().then(() => { disposed = true; });
    await settle(); expect(disposed).toBe(false);
    late.resolve(delayed!); await completion;
    expect(delayed!.dispose).toHaveBeenCalledOnce(); expect(c.runtime.abortCapacity).toHaveBeenCalledOnce();
    expect(s.owned()).toBe(before);
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('stops the admission loop immediately after a synchronous preparation failure', async () => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 8 }), [a, b, c, d] = s.fixtures;
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    a.runtime.prepareCapacity.mockImplementationOnce(() => { throw new Error('SYNC_PREPARE_FAILED'); });
    s.coordinator.tick(1); await settle();
    expect(b.runtime.prepareCapacity).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ error: 'Error: SYNC_PREPARE_FAILED', inFlightMigrations: 0, reservedBytes: 0 });
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('cleans uncommitted peers on retirement failure but preserves already committed owners', async () => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 8 }), [a, b, c, d] = s.fixtures;
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    s.coordinator.tick(1); await settle(); s.coordinator.tick(2); await settle();
    expect(a.runtime.pool.slotCount).toBe(2); expect(b.runtime.pool.slotCount).toBe(2);
    expect(c.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(d.runtime.prepareCapacity).toHaveBeenCalledOnce();
    a.fences[0].reject(new Error('RETIRE_QUEUE_LOST')); await settle();
    expect(c.runtime.abortCapacity).toHaveBeenCalledOnce(); expect(d.runtime.abortCapacity).toHaveBeenCalledOnce();
    expect(a.target.bindings.dispose).not.toHaveBeenCalled(); expect(b.target.bindings.dispose).not.toHaveBeenCalled();
    expect(s.coordinator.diagnostics()).toMatchObject({ error: 'Error: RETIRE_QUEUE_LOST', inFlightMigrations: 1 });
    b.fences[0].resolve(); await settle();
    expect(s.coordinator.diagnostics().reservedBytes).toBe(0);
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('waits for current-epoch feedback after a commit without blocking a ready peer publication', async () => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 8 }), [a, b, c, d] = s.fixtures;
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    s.coordinator.tick(1); await settle();
    const commit = a.runtime.commitCapacity;
    a.runtime.commitCapacity = vi.fn(candidate => { const result = commit(candidate); a.runtime.capacityDemand = null; return result; });
    s.coordinator.tick(2); await settle();
    expect(a.runtime.pool.slotCount).toBe(2); expect(b.runtime.pool.slotCount).toBe(2);
    expect(c.runtime.prepareCapacity).not.toHaveBeenCalled(); expect(d.runtime.prepareCapacity).not.toHaveBeenCalled();
    a.runtime.capacityDemand = []; s.coordinator.tick(3); await settle();
    expect(c.runtime.prepareCapacity).toHaveBeenCalledOnce(); expect(d.runtime.prepareCapacity).toHaveBeenCalledOnce();
    await s.close(); expect(s.owned()).toBe(0);
  });

  it('disposes a mixed ready/preparing/retiring batch without dropping reservations or double-releasing owners', async () => {
    const s = setup({ slots: [4, 4, 2, 2], concurrent: 4, peakCopies: 8 }), [a, b, c, d] = s.fixtures;
    const pending = deferred<Bindings>(), original = b.target.prepareBindings; let delayed: Bindings | undefined;
    b.target.prepareBindings = async candidate => { delayed = await original(candidate); return pending.promise; };
    c.runtime.capacityDemand = near; d.runtime.capacityDemand = near;
    s.coordinator.tick(1); await settle(); s.coordinator.tick(2); await settle();
    expect(s.coordinator.diagnostics().targets.map(target => target.phase).sort()).toEqual(['preparing', 'ready', 'retiring']);
    let complete = false; const cleanup = s.coordinator.dispose().then(() => { complete = true; });
    expect(c.runtime.abortCapacity).toHaveBeenCalledOnce();
    // Match Renderer teardown: active owners may be destroyed before the
    // coordinator's pending candidate and old-owner fences have resolved.
    for (const fixture of s.fixtures) { fixture.runtime.dispose(); fixture.target.bindings.dispose(); }
    await settle(); expect(complete).toBe(false); expect(s.owned()).toBeGreaterThanOrEqual(0);
    a.fences[0].resolve(); await settle(); expect(complete).toBe(false);
    pending.resolve(delayed!); await cleanup;
    expect(s.coordinator.diagnostics()).toMatchObject({ phase: 'disposed', inFlightMigrations: 0, reservedBytes: 0,
      copyReservations: { bytes: 0, pages: 0 }, liveBytes: 0, pendingGrowthBytes: 0 });
    expect(s.owned()).toBe(0); await s.close(); expect(s.owned()).toBe(0);
  });

  it('defaults to four streams but rejects concurrency outside the hard limit', async () => {
    const s = setup();
    const defaults = new GeometryStreamCapacityCoordinator(s.budget, s.fixtures.map(f => f.target),
      { ...s.options, maxConcurrentMigrations: undefined });
    expect(defaults.diagnostics().scheduling.maxConcurrentMigrations).toBe(4);
    for (const value of [0, 17, 1.5, Number.NaN]) expect(() => new GeometryStreamCapacityCoordinator(s.budget,
      s.fixtures.map(f => f.target), { ...s.options, maxConcurrentMigrations: value })).toThrow('OPTION');
    await defaults.dispose(); await s.close(); expect(s.owned()).toBe(0);
  });

  it('bounds concurrent preparations and reduces a many-stream exchange in deterministic frame steps', async () => {
    const run = async (concurrent: number) => {
      const s = setup({ slots: [4, 4, 4, 4, 2, 2, 2, 2], concurrent, peakCopies: 8 });
      for (const fixture of s.fixtures.slice(4)) fixture.runtime.capacityDemand = near;
      let frames = 0, largest = 0;
      while (frames < 80) {
        s.coordinator.tick(++frames); await settle();
        const state = s.coordinator.diagnostics(); largest = Math.max(largest, state.inFlightMigrations);
        expect(state.inFlightMigrations).toBeLessThanOrEqual(concurrent);
        expect(state.projectedLiveBytes).toBeLessThanOrEqual(s.options.maxResidentBytes);
        expect(state.accountedBytes).toBeLessThanOrEqual(s.options.maxPeakBytes);
        expect(s.owned()).toBeLessThanOrEqual(state.accountedBytes);
        for (const fixture of s.fixtures) for (const fence of fixture.fences) fence.resolve();
        await settle();
        if (s.fixtures.every((fixture, index) => fixture.runtime.pool.slotCount === (index < 4 ? 2 : 4))
          && s.coordinator.diagnostics().phase === 'idle') break;
      }
      await s.close(); expect(s.owned()).toBe(0); return { frames, largest };
    };
    const serial = await run(1), bounded = await run(4);
    expect(bounded.largest).toBe(4); expect(bounded.frames).toBeLessThan(serial.frames / 2);
  });
});
