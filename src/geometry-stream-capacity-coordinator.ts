import { GeometryStreamRuntimeBudget, type GeometryStreamRuntimeBudgetOptions } from './geometry-stream-runtime-budget';
import type { GpuGeometryStream, PreparedGeometryStreamCapacity } from './gpu-geometry-stream';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';

export interface GeometryStreamCapacityOptions extends GeometryStreamRuntimeBudgetOptions {
  /** Default 30; admission visits at most once per this many rendered frames. */
  intervalFrames?: number;
  /** Future per-stream physical pool ceiling; initial gpuPool budgets remain unchanged. */
  maxPoolBytes?: number;
  /** Total vertex/index copy payload and pages reserved across ALL outstanding
   * migrations, retained until retirement. Roots must fit. Defaults 16 MiB / 256. */
  maxCopyBytes?: number;
  maxCopyPages?: number;
  /** Independent streams may prepare/retire concurrently. Default 4, hard limit 16. */
  maxConcurrentMigrations?: number;
}
export interface GeometryStreamRenderedOwner { dispose(): void }
export interface GeometryStreamCapacityTarget<T extends GeometryStreamRenderedOwner> {
  readonly runtime: GpuGeometryStream;
  bindings: T;
  /** Construct all dependent bindings before publication; clean up on rejection. */
  prepareBindings(candidate: PreparedGeometryStreamCapacity): Promise<T>;
}
type Action = NonNullable<ReturnType<GeometryStreamRuntimeBudget['next']>['action']>;
type Transaction<T> = {
  action: Action; sourceEpoch: number; candidate?: PreparedGeometryStreamCapacity; bindings?: T;
  copyBytes: number; copyPages: number;
  phase: 'preparing' | 'ready' | 'retiring'; done?: Promise<void>;
};
function bounded(value: number | undefined, fallback: number, maximum: number) {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new Error('GEOMETRY_CAPACITY_OPTION');
  return result;
}

/** Each reservation spans preparation, publication and retirement. tick()
 * is called only at a synchronous frame boundary, before any frame is encoded.
 * The next shared deformation pass evaluates current AND previous transforms
 * into the replacement output; physical slot addresses are never history IDs. */
export class GeometryStreamCapacityCoordinator<T extends GeometryStreamRenderedOwner> {
  private readonly interval: number;
  private readonly copyBytes: number;
  private readonly copyPages: number;
  private readonly concurrent: number;
  private readonly rootCopies: { bytes: number; pages: number }[];
  private readonly transactions = new Map<number, Transaction<T>>();
  private disposed = false;
  private cleanup?: Promise<void>;
  private nextFrame = 0;
  private error: string | null = null;
  private migrations = 0;
  private deferred = 0;
  private peakBytes = 0;

  constructor(private readonly budget: GeometryStreamRuntimeBudget,
    private readonly targets: readonly GeometryStreamCapacityTarget<T>[], options: GeometryStreamCapacityOptions) {
    this.interval = bounded(options.intervalFrames, 30, 600);
    this.copyBytes = bounded(options.maxCopyBytes, 16 * 1024 ** 2, 256 * 1024 ** 2);
    this.copyPages = bounded(options.maxCopyPages, 256, 4096);
    this.concurrent = bounded(options.maxConcurrentMigrations, 4, 16);
    const current = targets.reduce((sum, target, index) => sum + budget.state(index, target.runtime.pool.slotCount).totalBytes, 0);
    if (!targets.length || current > budget.options.maxResidentBytes) throw new Error('GEOMETRY_CAPACITY_INITIAL_BUDGET');
    this.peakBytes = current;
    this.rootCopies = targets.map(({ runtime }) => {
      const roots = runtime.pool.directory.rootPages;
      const bytes = roots.reduce((sum, id) => sum + this.pageBytes(runtime, id), 0);
      if (roots.length > this.copyPages || bytes > this.copyBytes) {
        throw new Error('GEOMETRY_CAPACITY_ROOT_COPY_BUDGET');
      }
      return { bytes, pages: roots.length };
    });
  }
  private pageBytes(runtime: GpuGeometryStream, id: number) {
    const directory = runtime.pool.directory;
    return directory.pages[id].clusterIds.reduce((sum, cluster) => sum
      + directory.clusters[cluster].vertexCount * GEOMETRY_GPU_VERTEX_BYTES + directory.clusters[cluster].triangleCount * 12, 0);
  }
  private fail(error: unknown) {
    if (this.disposed) return;
    this.error ??= String(error);
    // Never publish another candidate after a peer fails. In-progress preparation
    // cleans itself on resolution; already-published owners finish retirement.
    this.discardReady();
  }
  private discard(transaction: Transaction<T>) {
    const errors: unknown[] = [];
    try { transaction.bindings?.dispose(); } catch (error) { errors.push(error); }
    transaction.bindings = undefined;
    if (transaction.candidate) {
      try { this.targets[transaction.action.index].runtime.abortCapacity(transaction.candidate); }
      catch (error) { if (!this.disposed) errors.push(error); }
      transaction.candidate = undefined;
    }
    if (errors.length) throw new AggregateError(errors, 'GEOMETRY_CAPACITY_ABORT_FAILED');
  }
  private discardReady() {
    for (const [index, transaction] of this.transactions) if (transaction.phase === 'ready') {
      try { this.discard(transaction); } catch (error) { if (!this.disposed) this.error ??= String(error); }
      this.transactions.delete(index);
    }
  }
  private reservations() {
    let bytes = 0, growthBytes = 0, copyBytes = 0, copyPages = 0;
    for (const transaction of this.transactions.values()) {
      const old = this.budget.state(transaction.action.index, transaction.action.expectedSlots).totalBytes;
      bytes += transaction.phase === 'retiring' ? old : transaction.action.totalBytes;
      if (transaction.phase !== 'retiring') growthBytes += Math.max(0, transaction.action.totalBytes - old);
      copyBytes += transaction.copyBytes; copyPages += transaction.copyPages;
    }
    return { bytes, growthBytes, copyBytes, copyPages };
  }
  private prepare(action: Action, pages: number[], bytes: number) {
    const target = this.targets[action.index];
    // Charge the entire candidate AND its positive live delta before any await.
    // Same-stream owners remain excluded until their old resources retire.
    const next: Transaction<T> = { action, sourceEpoch: target.runtime.capacityEpoch, phase: 'preparing', copyBytes: bytes, copyPages: pages.length };
    this.transactions.set(action.index, next); this.peakBytes = Math.max(this.peakBytes, action.peakBytes);
    next.done = (async () => {
      try {
        next.candidate = await target.runtime.prepareCapacity({ maxResidentBytes: action.poolBytes,
          maxPeakBytes: action.poolPeakBytes, retainedPages: pages, maxCopyBytes: bytes, maxCopyPages: pages.length });
        if (this.disposed || this.error) { this.discard(next); return; }
        next.bindings = await target.prepareBindings(next.candidate);
        if (this.disposed || this.error) { this.discard(next); return; }
        next.phase = 'ready';
      } catch (error) {
        try { this.discard(next); } catch (cleanup) { this.fail(cleanup); }
        this.fail(error);
      } finally {
        if (next.phase !== 'ready' && this.transactions.get(action.index) === next) this.transactions.delete(action.index);
      }
    })();
  }
  tick(frame: number): void {
    if (this.disposed || this.error) return;
    if (!Number.isSafeInteger(frame) || frame < 0) throw new Error('GEOMETRY_CAPACITY_FRAME');
    // Every publication happens before this frame is encoded. Shrinks first is
    // deterministic; live growth reservations also protect any completion order.
    const ready = [...this.transactions.values()].filter(transaction => transaction.phase === 'ready').sort((a, b) => {
      const delta = (transaction: Transaction<T>) => transaction.action.totalBytes
        - this.budget.state(transaction.action.index, transaction.action.expectedSlots).totalBytes;
      return delta(a) - delta(b) || a.action.index - b.action.index;
    });
    for (const transaction of ready) {
      if (this.error) break;
      const target = this.targets[transaction.action.index];
      try {
        if (target.runtime.capacityEpoch !== transaction.sourceEpoch
          || target.runtime.pool.slotCount !== transaction.action.expectedSlots) throw new Error('GEOMETRY_CAPACITY_STALE');
        const oldBindings = target.bindings;
        const committed = target.runtime.commitCapacity(transaction.candidate!);
        // Plain assignment immediately after stream commit: no callback, await,
        // allocation or command encoding can observe mismatched physical owners.
        target.bindings = transaction.bindings!;
        transaction.candidate = undefined; transaction.bindings = undefined;
        transaction.phase = 'retiring'; this.migrations++;
        transaction.done = committed.retired.catch(error => { this.fail(error); }).then(() => {
          try { oldBindings.dispose(); } catch (error) { this.fail(error); }
          if (this.transactions.get(transaction.action.index) === transaction) this.transactions.delete(transaction.action.index);
        });
      } catch (error) {
        try { this.discard(transaction); } catch (cleanup) { this.fail(cleanup); }
        this.transactions.delete(transaction.action.index); this.fail(error);
      }
    }
    if (this.error || this.transactions.size >= this.concurrent) return;
    if (frame < this.nextFrame) return;
    const demands = this.targets.map(target => target.runtime.capacityDemand);
    if (demands.some(demand => demand === null)) return;
    this.nextFrame = frame + this.interval;
    try {
      const plan = this.budget.plan(demands.map(demand => demand!));
      const excluded = new Set(this.transactions.keys());
      // Busy frame leases may exclude additional streams, but cannot cause an
      // unbounded rescan. Each successful launch also consumes a concurrency slot.
      for (let attempt = 0; attempt < 64 && !this.error && !this.disposed && this.transactions.size < this.concurrent; attempt++) {
        const reserved = this.reservations(), availableBytes = this.copyBytes - reserved.copyBytes, availablePages = this.copyPages - reserved.copyPages;
        if (availableBytes <= 0 || availablePages <= 0) { this.deferred++; break; }
        this.rootCopies.forEach((root, index) => { if (root.bytes > availableBytes || root.pages > availablePages) excluded.add(index); });
        const decision = this.budget.next(plan, this.targets.map(target => target.runtime.pool.slotCount), reserved.bytes,
          { excluded: [...excluded], growthBytes: reserved.growthBytes });
        if (!decision.action) { if (decision.deferred) this.deferred++; break; }
        const action = decision.action, target = this.targets[action.index], state = target.runtime.pool.diagnostics();
        excluded.add(action.index);
        if (state.framesInFlight >= state.limits.maxFramesInFlight) { this.deferred++; continue; }
        const resident = target.runtime.pool.residentPageIds(), pages = [...target.runtime.pool.directory.rootPages], roots = new Set(pages);
        let bytes = this.rootCopies[action.index].bytes;
        for (const id of action.pages) if (!roots.has(id) && resident.has(id) && pages.length < availablePages) {
          const size = this.pageBytes(target.runtime, id);
          if (bytes + size <= availableBytes) { pages.push(id); bytes += size; }
        }
        this.prepare(action, pages, bytes);
      }
    } catch (error) { this.fail(error); }
  }
  diagnostics() {
    const liveBytes = this.disposed ? 0 : this.targets.reduce((sum, target, index) => sum
      + this.budget.state(index, target.runtime.pool.slotCount).totalBytes, 0);
    const transactions = [...this.transactions.values()], reserved = this.reservations();
    const phase = transactions.some(value => value.phase === 'preparing') ? 'preparing'
      : transactions.some(value => value.phase === 'ready') ? 'ready' : transactions.length ? 'retiring' : 'idle';
    return { enabled: true, phase: this.disposed ? 'disposed' : phase, error: this.error,
      liveBytes, reservedBytes: reserved.bytes, accountedBytes: liveBytes + reserved.bytes, peakBytes: this.peakBytes,
      pendingGrowthBytes: reserved.growthBytes, projectedLiveBytes: liveBytes + reserved.growthBytes,
      copyReservations: { bytes: reserved.copyBytes, pages: reserved.copyPages }, inFlightMigrations: transactions.length,
      scheduling: { maxConcurrentMigrations: this.concurrent, maxCopyBytes: this.copyBytes, maxCopyPages: this.copyPages, intervalFrames: this.interval },
      migrations: this.migrations, deferred: this.deferred, limits: this.budget.options,
      targets: transactions.map(value => ({ index: value.action.index, slots: value.action.slots, temporary: value.action.temporary, phase: value.phase })) };
  }
  /** Active bindings remain Renderer-owned; this waits only for pending/retiring
   * owners. Renderer's synchronous teardown may destroy them idempotently first. */
  dispose(): Promise<void> {
    if (this.cleanup) return this.cleanup;
    this.disposed = true;
    this.discardReady();
    this.cleanup = Promise.allSettled([...this.transactions.values()].map(transaction => transaction.done)).then(() => { this.transactions.clear(); });
    return this.cleanup;
  }
}
