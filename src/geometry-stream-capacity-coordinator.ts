import { GeometryStreamRuntimeBudget, type GeometryStreamRuntimeBudgetOptions } from './geometry-stream-runtime-budget';
import type { GpuGeometryStream, PreparedGeometryStreamCapacity } from './gpu-geometry-stream';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';

export interface GeometryStreamCapacityOptions extends GeometryStreamRuntimeBudgetOptions {
  /** Default 30; admission visits at most once per this many rendered frames. */
  intervalFrames?: number;
  /** Future per-stream physical pool ceiling; initial gpuPool budgets remain unchanged. */
  maxPoolBytes?: number;
  /** Bounded GPU copy work per migration. Roots must fit. Defaults 16 MiB / 256 pages. */
  maxCopyBytes?: number;
  maxCopyPages?: number;
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
  phase: 'preparing' | 'ready' | 'retiring'; done?: Promise<void>;
};
function bounded(value: number | undefined, fallback: number, maximum: number) {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new Error('GEOMETRY_CAPACITY_OPTION');
  return result;
}

/** One global reservation spans preparation, publication and retirement. tick()
 * is called only at a synchronous frame boundary, before any frame is encoded.
 * The next shared deformation pass evaluates current AND previous transforms
 * into the replacement output; physical slot addresses are never history IDs. */
export class GeometryStreamCapacityCoordinator<T extends GeometryStreamRenderedOwner> {
  private readonly interval: number;
  private readonly copyBytes: number;
  private readonly copyPages: number;
  private transaction?: Transaction<T>;
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
    const current = targets.reduce((sum, target, index) => sum + budget.state(index, target.runtime.pool.slotCount).totalBytes, 0);
    if (!targets.length || current > budget.options.maxResidentBytes) throw new Error('GEOMETRY_CAPACITY_INITIAL_BUDGET');
    this.peakBytes = current;
    for (const { runtime } of targets) {
      const roots = runtime.pool.directory.rootPages;
      if (roots.length > this.copyPages || roots.reduce((sum, id) => sum + this.pageBytes(runtime, id), 0) > this.copyBytes) {
        throw new Error('GEOMETRY_CAPACITY_ROOT_COPY_BUDGET');
      }
    }
  }
  private pageBytes(runtime: GpuGeometryStream, id: number) {
    const directory = runtime.pool.directory;
    return directory.pages[id].clusterIds.reduce((sum, cluster) => sum
      + directory.clusters[cluster].vertexCount * GEOMETRY_GPU_VERTEX_BYTES + directory.clusters[cluster].triangleCount * 12, 0);
  }
  private fail(error: unknown) { if (!this.disposed) this.error = String(error); }
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
  tick(frame: number): void {
    if (this.disposed || this.error) return;
    if (!Number.isSafeInteger(frame) || frame < 0) throw new Error('GEOMETRY_CAPACITY_FRAME');
    const transaction = this.transaction;
    if (transaction) {
      if (transaction.phase !== 'ready') return;
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
          if (this.transaction === transaction) this.transaction = undefined;
        });
      } catch (error) {
        try { this.discard(transaction); } catch (cleanup) { this.fail(cleanup); }
        this.transaction = undefined; this.fail(error);
      }
      return;
    }
    if (frame < this.nextFrame) return;
    const demands = this.targets.map(target => target.runtime.capacityDemand);
    if (demands.some(demand => demand === null)) return;
    this.nextFrame = frame + this.interval;
    try {
      const plan = this.budget.plan(demands.map(demand => demand!));
      const decision = this.budget.next(plan, this.targets.map(target => target.runtime.pool.slotCount));
      if (!decision.action) { if (decision.deferred) this.deferred++; return; }
      const action = decision.action, target = this.targets[action.index], state = target.runtime.pool.diagnostics();
      if (state.framesInFlight >= state.limits.maxFramesInFlight) { this.deferred++; return; }
      const resident = target.runtime.pool.residentPageIds();
      const pages = [...target.runtime.pool.directory.rootPages];
      const roots = new Set(pages);
      let bytes = pages.reduce((sum, id) => sum + this.pageBytes(target.runtime, id), 0);
      for (const id of action.pages) if (!roots.has(id) && resident.has(id) && pages.length < this.copyPages) {
        const size = this.pageBytes(target.runtime, id);
        if (bytes + size <= this.copyBytes) { pages.push(id); bytes += size; }
      }
      // Reserve the entire candidate before the first allocation or await. Only
      // one transaction can exist, including after commit while old owners live.
      const next: Transaction<T> = { action, sourceEpoch: target.runtime.capacityEpoch, phase: 'preparing' };
      this.transaction = next; this.peakBytes = Math.max(this.peakBytes, action.peakBytes);
      next.done = (async () => {
        try {
          next.candidate = await target.runtime.prepareCapacity({ maxResidentBytes: action.poolBytes,
            maxPeakBytes: action.poolPeakBytes, retainedPages: pages, maxCopyBytes: this.copyBytes, maxCopyPages: this.copyPages });
          if (this.disposed) { this.discard(next); return; }
          next.bindings = await target.prepareBindings(next.candidate);
          if (this.disposed) { this.discard(next); return; }
          next.phase = 'ready';
        } catch (error) {
          try { this.discard(next); } catch (cleanup) { this.fail(cleanup); }
          this.fail(error);
        } finally {
          if (next.phase !== 'ready' && this.transaction === next) this.transaction = undefined;
        }
      })();
    } catch (error) { this.fail(error); }
  }
  diagnostics() {
    const liveBytes = this.disposed ? 0 : this.targets.reduce((sum, target, index) => sum
      + this.budget.state(index, target.runtime.pool.slotCount).totalBytes, 0);
    const transaction = this.transaction;
    const reservedBytes = transaction ? transaction.phase === 'retiring'
      ? this.budget.state(transaction.action.index, transaction.action.expectedSlots).totalBytes : transaction.action.totalBytes : 0;
    return { enabled: true, phase: this.disposed ? 'disposed' : transaction?.phase ?? 'idle', error: this.error,
      liveBytes, reservedBytes, accountedBytes: liveBytes + reservedBytes, peakBytes: this.peakBytes,
      migrations: this.migrations, deferred: this.deferred, limits: this.budget.options,
      target: transaction ? { index: transaction.action.index, slots: transaction.action.slots, temporary: transaction.action.temporary } : null };
  }
  /** Active bindings remain Renderer-owned; this waits only for pending/retiring
   * owners. Renderer's synchronous teardown may destroy them idempotently first. */
  dispose(): Promise<void> {
    if (this.cleanup) return this.cleanup;
    this.disposed = true;
    const transaction = this.transaction;
    if (transaction?.phase === 'ready') {
      try { this.discard(transaction); } catch { /* Every owner was attempted. */ }
      this.transaction = undefined;
    }
    this.cleanup = Promise.resolve(transaction?.done).then(() => { this.transaction = undefined; });
    return this.cleanup;
  }
}
