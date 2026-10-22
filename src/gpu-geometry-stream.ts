import { GeometryPageCache, type GeometryPageCacheOptions } from './geometry-stream-cache';
import { GpuGeometryPagePool, type GeometryGpuPoolOptions, type GeometryGpuPoolMigrationOptions } from './gpu-geometry-page-pool';
import { GpuGeometryStreamSelection, type GeometryStreamSelectionFeedback, type GeometryStreamSelectionOptions } from './gpu-geometry-stream-selection';
import type { GeometryPageProvider, GeometryStreamDirectory } from './geometry-stream-types';
import { residentMesh, type RenderBatch } from './contracts';
import { validateGeometryQueryProxy, type GeometryQueryProxy, type GeometryQueryProxyOptions } from './geometry-query-proxy';
import type { Mat4 } from './math';
import { GeometryStreamWorkingSet } from './geometry-stream-working-set';
import type { GeometryPageDecodeService } from './geometry-decode-worker-pool';
import type { MeshData } from './geometry';
import type { GpuOcclusion } from './gpu-occlusion';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';

export interface GeometryStreamSource {
  /** Explicit fixed representation for a source-free StreamedSceneBatch. */
  readonly queryProxy?: GeometryQueryProxy;
  /** Optional retained-proxy budgets; defaults and hard ceilings remain those of validateGeometryQueryProxy. */
  readonly queryProxyLimits?: Pick<GeometryQueryProxyOptions, 'maxVertices' | 'maxTriangles' | 'maxBytes'>;
  readonly directory: GeometryStreamDirectory;
  readonly provider: GeometryPageProvider;
  readonly cpuCache?: GeometryPageCacheOptions;
  readonly gpuPool?: GeometryGpuPoolOptions;
  readonly selection?: GeometryStreamSelectionOptions;
  /** Default 16. Caps CPU request/enqueue visits per rendered frame, independently of byte budgets. */
  readonly maxRequestsPerFrame?: number;
}

export interface PreparedGeometryStreamCapacity {
  readonly pool: GpuGeometryPagePool;
  readonly selection: GpuGeometryStreamSelection;
  readonly sourceEpoch: number;
  readonly copiedPages: readonly number[];
  readonly copiedBytes: number;
  readonly poolPeakBytes: number;
}

/** Renderer bridge. Paged surfaces share instance/material state with either
 * an exact resident mesh or an explicit fixed shadow/query proxy. */
export class GpuGeometryStream {
  private source: GeometryStreamSource;
  private readonly sourceMesh: MeshData;
  private readonly fullSource?: MeshData;
  private readonly proxy?: GeometryQueryProxy;
  private readonly sourceVertexCount: number;
  private readonly sourceVertices: ArrayBuffer;
  private readonly sourceIndices: Uint32Array;
  private readonly sourceUvs?: Float32Array;
  private readonly sourceTangents?: Float32Array;
  private readonly sourceColors?: Float32Array;
  private readonly requestsPerFrame: number;
  private pending: GeometryStreamSelectionFeedback['requestedPages'] = [];
  private pendingCursor = 0;
  private workingSet: GeometryStreamWorkingSet;
  private allocation: ReturnType<GeometryStreamWorkingSet['plan']>;
  private cpuWindow: number[] = [];
  private cpuWindowBudgetDeferred: number[] = [];
  private uploadSerial = 0;
  private reading = false;
  private feedbackTask?: Promise<void>;
  private disposed = false;
  private error: string | null = null;
  private lastFeedback: GeometryStreamSelectionFeedback | null = null;
  private readonly checkedPages = new Set<number>();
  private cleanup?: Promise<void>;
  private epoch = 0;
  private preparing?: Promise<PreparedGeometryStreamCapacity>;
  private prepared?: PreparedGeometryStreamCapacity;
  private readonly retiring = new Set<{ pool: GpuGeometryPagePool; selection: GpuGeometryStreamSelection; done: Promise<void> }>();

  private constructor(readonly scene: RenderBatch, readonly cache: GeometryPageCache,
    private activePool: GpuGeometryPagePool, private activeSelection: GpuGeometryStreamSelection,
    private readonly device: GPUDevice, private readonly instances: GPUBuffer,
    private readonly track: (bytes: number) => void, private readonly occlusion?: GpuOcclusion) {
    this.source = scene.geometryStream!;
    this.fullSource = scene.mesh; this.proxy = this.source.queryProxy;
    const mesh = this.sourceMesh = residentMesh(scene);
    this.sourceVertexCount = mesh.vertexCount;
    this.sourceVertices = mesh.vertices; this.sourceIndices = mesh.indices;
    this.sourceUvs = mesh.uvs; this.sourceTangents = mesh.tangents; this.sourceColors = mesh.colors;
    this.requestsPerFrame = this.source.maxRequestsPerFrame ?? 16;
    this.workingSet = new GeometryStreamWorkingSet(this.pool.directory, this.pool.slotCount);
    this.allocation = this.workingSet.plan([]);
  }
  get pool() { return this.activePool; }
  get selection() { return this.activeSelection; }
  get capacityEpoch() { return this.epoch; }
  /** Bounded GPU demand, independent of residency. Null until this epoch has feedback. */
  get capacityDemand(): readonly GeometryStreamSelectionFeedback['desiredGroups'][number][] | null {
    return this.lastFeedback?.desiredGroups ?? null;
  }
  static async create(device: GPUDevice, scene: RenderBatch, instances: GPUBuffer, track: (bytes: number) => void,
    decoder?: GeometryPageDecodeService, occlusion?: GpuOcclusion) {
    const source = scene.geometryStream;
    if (!source || scene.geometryLod || scene.deformation.jointCount || scene.deformation.nodes.length || scene.morphTargets.length) {
      throw new Error('GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE');
    }
    const visits = source.maxRequestsPerFrame ?? 16;
    if (!Number.isSafeInteger(visits) || visits < 1 || visits > 256) throw new Error('GEOMETRY_STREAM_REQUEST_VISIT_BUDGET');
    if (!source.queryProxy && source.queryProxyLimits !== undefined) throw new Error('GEOMETRY_STREAM_QUERY_PROXY_LIMITS_WITHOUT_PROXY');
    if (source.queryProxy && scene.mesh) throw new Error('GEOMETRY_STREAM_AMBIGUOUS_REPRESENTATION');
    const mesh = residentMesh(scene);
    if (source.queryProxy) await validateGeometryQueryProxy(source.directory, source.queryProxy, source.queryProxyLimits);
    else if (source.directory.sourceVertexCount !== mesh.vertexCount || source.directory.sourceTriangleCount * 3 !== mesh.indices.length
      || !!source.directory.hasColors !== !!mesh.colors
      || source.directory.vertexStride !== 80 + (mesh.uvs ? 8 : 0) + (mesh.tangents ? 16 : 0) + (mesh.colors ? 16 : 0)) {
      throw new Error('GEOMETRY_STREAM_SOURCE_MISMATCH');
    }
    let cache: GeometryPageCache | undefined, pool: GpuGeometryPagePool | undefined, selection: GpuGeometryStreamSelection | undefined;
    try {
      pool = await GpuGeometryPagePool.create(device, source.directory, source.gpuPool, track);
      cache = await GeometryPageCache.create(source.directory, source.provider, source.cpuCache, decoder);
      selection = await GpuGeometryStreamSelection.create(device, pool, instances, scene.capacity, track, source.selection, occlusion);
      const result = new GpuGeometryStream(scene, cache, pool, selection, device, instances, track, occlusion);
      // Initialization is cooperative and completes every root before the first
      // frame. The regular render path performs exactly one bounded upload tick.
      for (const page of pool.directory.rootPages) {
        result.checkPage(page); pool.enqueue(page, cache.get(page)!);
        while (!pool.residentPageIds().has(page)) {
          pool.uploadFrame(++result.uploadSerial); await device.queue.onSubmittedWorkDone();
        }
      }
      return result;
    } catch (error) {
      // A diagnostic callback may throw; every owner must still be released and
      // the original initialization failure must remain the reported cause.
      try { selection?.dispose(); } catch { /* preserve initialization failure */ }
      try { pool?.dispose(); } catch { /* preserve initialization failure */ }
      try { await cache?.dispose(); } catch { /* preserve initialization failure */ }
      throw error;
    }
  }
  /** Prepare without changing the active stream or its CPU cache. The caller
   * reserves the complete scene overlap (selection and rendered state included)
   * before calling; maxPeakBytes here retains the pool-only migration meaning. */
  prepareCapacity(options: GeometryGpuPoolMigrationOptions): Promise<PreparedGeometryStreamCapacity> {
    this.validateSources();
    if (this.preparing || this.prepared || this.retiring.size) throw new Error('GEOMETRY_STREAM_CAPACITY_BUSY');
    const sourceEpoch = this.epoch, sourcePool = this.pool;
    const task = (async () => {
      let pool: GpuGeometryPagePool | undefined, selection: GpuGeometryStreamSelection | undefined;
      try {
        const migration = await sourcePool.migrateResident(options); pool = migration.pool;
        this.validateSources();
        if (this.epoch !== sourceEpoch || this.pool !== sourcePool) throw new Error('GEOMETRY_STREAM_CAPACITY_STALE');
        selection = await GpuGeometryStreamSelection.create(this.device, pool, this.instances, this.scene.capacity,
          this.track, this.source.selection, this.occlusion, this.selection);
        this.validateSources();
        if (this.epoch !== sourceEpoch || this.pool !== sourcePool) throw new Error('GEOMETRY_STREAM_CAPACITY_STALE');
        selection.copyViewStateFrom(this.selection);
        const candidate = Object.freeze({ pool, selection, sourceEpoch, copiedPages: Object.freeze([...migration.copiedPages]),
          copiedBytes: migration.copiedBytes, poolPeakBytes: migration.peakBytes });
        this.prepared = candidate; return candidate;
      } catch (error) {
        try { selection?.dispose(); } catch { /* Preserve the preparation failure. */ }
        try { pool?.dispose(); } catch { /* Preserve the preparation failure. */ }
        throw error;
      }
    })();
    this.preparing = task;
    void task.then(() => { if (this.preparing === task) this.preparing = undefined; },
      () => { if (this.preparing === task) this.preparing = undefined; });
    return task;
  }
  abortCapacity(candidate: PreparedGeometryStreamCapacity): void {
    if (this.prepared !== candidate) throw new Error('GEOMETRY_STREAM_CAPACITY_CANDIDATE');
    this.prepared = undefined; const errors: unknown[] = [];
    try { candidate.selection.dispose(); } catch (error) { errors.push(error); }
    try { candidate.pool.dispose(); } catch (error) { errors.push(error); }
    if (errors.length) throw new AggregateError(errors, 'GEOMETRY_STREAM_CAPACITY_CLEANUP');
  }
  /** Call at a synchronous frame boundary, then install the matching render
   * bindings without awaiting. Until retirement resolves, the caller continues
   * charging ALL old bindings and rendered buffers against the peak budget. */
  commitCapacity(candidate: PreparedGeometryStreamCapacity): { retired: Promise<void>; epoch: number } {
    this.validateSources();
    if (candidate !== this.prepared || candidate.sourceEpoch !== this.epoch) throw new Error('GEOMETRY_STREAM_CAPACITY_STALE');
    if (candidate.pool.diagnostics().disposed || candidate.selection.diagnostics().disposed) throw new Error('GEOMETRY_STREAM_CAPACITY_DISPOSED');
    candidate.selection.copyViewStateFrom(this.selection);
    const workingSet = new GeometryStreamWorkingSet(candidate.pool.directory, candidate.pool.slotCount);
    const allocation = workingSet.plan(this.lastFeedback?.desiredGroups ?? []);
    const pool = this.pool, selection = this.selection, feedback = this.feedbackTask;
    // Capture the fence before publishing. No frame encoding can interleave with
    // this synchronous commit; later submissions may only use the new owners.
    const fence = this.device.queue.onSubmittedWorkDone();
    this.activePool = candidate.pool; this.activeSelection = candidate.selection;
    this.workingSet = workingSet; this.allocation = allocation; this.pending = allocation.requestedPages;
    this.pendingCursor = 0; this.reading = false; this.feedbackTask = undefined; this.lastFeedback = null;
    this.prepared = undefined; this.epoch++;
    const retired = { pool, selection, done: Promise.resolve() };
    retired.done = Promise.allSettled([fence, feedback]).then(results => {
      const errors: unknown[] = results.filter(result => result.status === 'rejected').map(result => result.reason);
      try { selection.dispose(); } catch (error) { errors.push(error); }
      try { pool.dispose(); } catch (error) { errors.push(error); }
      this.retiring.delete(retired);
      if (errors.length) {
        if (!this.disposed) this.error = 'GEOMETRY_STREAM_CAPACITY_RETIRE_FAILED: ' + String(errors[0]);
        throw new AggregateError(errors, 'GEOMETRY_STREAM_CAPACITY_RETIRE_FAILED');
      }
    });
    this.retiring.add(retired);
    // The returned promise still rejects for the caller; this observer prevents
    // an unhandled rejection when the Renderer is disposed before observing it.
    void retired.done.catch(() => {});
    return { retired: retired.done, epoch: this.epoch };
  }
  validateSources() {
    if (this.disposed) throw new Error('GEOMETRY_STREAM_DISPOSED');
    if (this.error) throw new Error(this.error);
    const mesh = residentMesh(this.scene);
    if (this.scene.geometryStream !== this.source || this.source.queryProxy !== this.proxy || this.scene.mesh !== this.fullSource
      || mesh !== this.sourceMesh || mesh.vertexCount !== this.sourceVertexCount
      || mesh.vertices !== this.sourceVertices || mesh.indices !== this.sourceIndices
      || mesh.uvs !== this.sourceUvs || mesh.tangents !== this.sourceTangents
      || mesh.colors !== this.sourceColors
      || this.scene.capacity !== this.selection.capacity || this.scene.deformation.jointCount || this.scene.deformation.nodes.length || this.scene.morphTargets.length) {
      throw new Error('GEOMETRY_STREAM_ASSET_CHANGED');
    }
  }
  private checkPage(id: number) {
    if (this.checkedPages.has(id)) return;
    const page = this.cache.get(id); if (!page) throw new Error('GEOMETRY_STREAM_CPU_PAGE_MISSING');
    // Proxy assets and pages are bound to the same validated directory hashes.
    // Source-free inputs have no original attribute arrays to compare against.
    if (this.proxy) { this.checkedPages.add(id); return; }
    const original = new Uint8Array(this.sourceVertices);
    // Exact rest attributes/UV/tangents are shared with the full-detail query
    // representation. Source topology is immutable by the scene asset contract.
    for (const cluster of page.clusters) for (let vertex = 0; vertex < cluster.sourceVertexIds.length; vertex++) {
      const source = cluster.sourceVertexIds[vertex];
      for (let byte = 0; byte < 80; byte++) if (cluster.rest[vertex * 80 + byte] !== original[source * 80 + byte]) throw new Error('GEOMETRY_STREAM_SOURCE_ATTRIBUTES');
      if (cluster.uvs) for (let axis = 0; axis < 2; axis++) if (cluster.uvs[vertex * 2 + axis] !== this.sourceUvs![source * 2 + axis]) throw new Error('GEOMETRY_STREAM_SOURCE_UVS');
      if (cluster.tangents) for (let axis = 0; axis < 4; axis++) if (cluster.tangents[vertex * 4 + axis] !== this.sourceTangents![source * 4 + axis]) throw new Error('GEOMETRY_STREAM_SOURCE_TANGENTS');
      if (cluster.colors) for (let axis = 0; axis < 4; axis++) if (cluster.colors[vertex * 4 + axis] !== this.sourceColors![source * 4 + axis]) throw new Error('GEOMETRY_STREAM_SOURCE_COLORS');
    }
    this.checkedPages.add(id);
  }
  update(frame: number, count: number, vp: Mat4, eye: readonly number[], projectionScale: number) {
    this.validateSources();
    const resident = this.pool.residentPageIds();
    const desired = new Set(this.allocation.pages);
    this.pool.retainPendingPages(this.allocation.pages);
    this.pool.setPinnedPages(this.allocation.pages.filter(id => resident.has(id)));
    const poolState = this.pool.diagnostics(), staged = new Set(poolState.pendingPages);
    const cacheState = this.cache.diagnostics(), failed = new Set(cacheState.failedPages.map(page => page.pageId));
    const needsCpu = (id: number) => desired.has(id) && !resident.has(id) && !staged.has(id) && !failed.has(id);
    // Keep in-progress downloads/decoded pages until the pool owns a snapshot.
    // A bounded future pin window prevents CPU LRU from evicting them first.
    this.cpuWindow = this.cpuWindow.filter(needsCpu);
    const window = new Set(this.cpuWindow);
    let windowBytes = this.pool.directory.rootPages.reduce((sum, id) => sum + this.pool.directory.pages[id].decodedBytes, 0)
      + this.cpuWindow.reduce((sum, id) => sum + this.pool.directory.pages[id].decodedBytes, 0);
    this.cpuWindowBudgetDeferred = [];
    for (let visit = 0; visit < Math.min(this.requestsPerFrame, this.pending.length); visit++) {
      const request = this.pending[this.pendingCursor];
      this.pendingCursor = (this.pendingCursor + 1) % this.pending.length;
      if (!needsCpu(request.pageId) || window.has(request.pageId)) continue;
      const bytes = this.pool.directory.pages[request.pageId].decodedBytes;
      if (windowBytes + bytes > cacheState.limits.maxResidentBytes) { this.cpuWindowBudgetDeferred.push(request.pageId); continue; }
      if (window.size >= this.requestsPerFrame) break;
      window.add(request.pageId); windowBytes += bytes;
    }
    this.cpuWindow = [...window];
    this.cache.setRequestWindow(this.cpuWindow);
    const priorities = new Map(this.pending.map(request => [request.pageId, request.priority]));
    for (const pageId of this.cpuWindow) {
      const priority = priorities.get(pageId) ?? 0;
      try { this.cache.request(pageId, priority); }
      catch (error) {
        // Saturation is backpressure. Preserve the visible ancestor and revisit
        // this page on a later frame; other provider/asset failures still surface.
        if (error instanceof Error && error.message === 'GEOMETRY_STREAM_REQUEST_BUDGET') continue;
        throw error;
      }
      const page = this.cache.get(pageId); if (!page) continue;
      const state = this.pool.diagnostics();
      if (state.pendingPages.includes(pageId)) continue;
      const bytes = this.pool.directory.pages[pageId].clusterIds.reduce((sum, id) => sum
        + this.pool.directory.clusters[id].vertexCount * GEOMETRY_GPU_VERTEX_BYTES + this.pool.directory.clusters[id].triangleCount * 12 + 32, 4);
      if (state.pendingPages.length >= state.limits.maxQueuedPages || state.stagingBytes + bytes > state.limits.maxStagingBytes) break;
      this.checkPage(pageId); this.pool.enqueue(pageId, page, priority);
    }
    const upload = this.pool.uploadFrame(++this.uploadSerial);
    return upload.bytes + this.selection.update(frame, count, vp, eye, projectionScale);
  }
  /** Queue readback contains only bounded requests/counters, never geometry. */
  afterSubmit() {
    if (this.reading || this.disposed) return; this.reading = true;
    const selection = this.selection, epoch = this.epoch;
    const current = () => !this.disposed && this.selection === selection && this.epoch === epoch;
    this.feedbackTask = selection.feedback().then(result => {
      if (!current()) return;
      if (result.overflow) { this.error = 'GEOMETRY_STREAM_DRAW_OVERFLOW'; return; }
      this.lastFeedback = result; this.allocation = this.workingSet.plan(result.desiredGroups);
      this.pending = this.allocation.requestedPages;
      this.pendingCursor = this.pending.length ? this.pendingCursor % this.pending.length : 0;
    }).catch(error => { if (current()) this.error = 'GEOMETRY_STREAM_FEEDBACK_FAILED: ' + String(error); })
      .finally(() => { if (current()) this.reading = false; });
  }
  async probe() {
    await this.feedbackTask; this.validateSources();
    return this.selection.feedback(true);
  }
  retry(page: number) { this.validateSources(); this.cache.retry(page); }
  diagnostics() { return { pool: this.pool.diagnostics(), cache: this.cache.diagnostics(), selection: this.selection.diagnostics(),
    error: this.error, feedback: this.lastFeedback,
    capacity: { epoch: this.epoch, preparing: !!this.preparing, preparedSlots: this.prepared?.pool.slotCount ?? null,
      retiringOwners: this.retiring.size },
    sourceRepresentation: this.proxy ? 'fixed-shadow-query-proxy' as const : 'full-detail-shadow-query-and-masked' as const,
    queryProxy: this.proxy ? { primitiveKey: this.proxy.primitiveKey, directorySha256: this.proxy.directorySha256,
      topologyVersion: this.proxy.topologyVersion, vertices: this.sourceVertexCount, triangles: this.sourceIndices.length / 3,
      residentPayloadBytes: this.sourceVertices.byteLength + this.sourceIndices.byteLength + (this.sourceUvs?.byteLength ?? 0) + (this.sourceTangents?.byteLength ?? 0),
      approximateObjectSpaceError: this.proxy.approximateObjectSpaceError, errorMetric: 'accumulated-simplification',
      cameraDependent: false, exactSourceRetained: false } : null,
    scheduling: 'group-budget-with-ancestor-retention' as const, allocation: { ...this.allocation,
      pages: [...this.allocation.pages], admittedGroups: [...this.allocation.admittedGroups],
      budgetDeferredGroups: [...this.allocation.budgetDeferredGroups], dependencyDeferredGroups: [...this.allocation.dependencyDeferredGroups],
      requestedPages: this.allocation.requestedPages.map(page => ({ ...page })) },
    cpuWindow: [...this.cpuWindow], cpuWindowBudgetDeferred: [...this.cpuWindowBudgetDeferred], maxRequestsPerFrame: this.requestsPerFrame }; }
  dispose(): Promise<void> {
    if (this.cleanup) return this.cleanup; this.disposed = true;
    let resolve!: () => void, reject!: (error: unknown) => void;
    this.cleanup = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
    const errors: unknown[] = [];
    // GPU release stays synchronous so the Renderer can close its resource
    // ledger immediately. Publish cleanup first to make callbacks reentrant.
    try { this.selection.dispose(); } catch (error) { errors.push(error); }
    try { this.pool.dispose(); } catch (error) { errors.push(error); }
    if (this.prepared) try { this.abortCapacity(this.prepared); } catch (error) { errors.push(error); }
    for (const retired of this.retiring) {
      try { retired.selection.dispose(); } catch (error) { errors.push(error); }
      try { retired.pool.dispose(); } catch (error) { errors.push(error); }
    }
    this.pending = []; this.cpuWindow = []; this.cpuWindowBudgetDeferred = []; this.checkedPages.clear(); this.lastFeedback = null;
    const finish = () => errors.length ? reject(new AggregateError(errors, 'GEOMETRY_STREAM_CLEANUP_FAILED')) : resolve();
    let cacheCleanup: Promise<void>;
    try { cacheCleanup = Promise.resolve(this.cache.dispose()); }
    catch (error) { cacheCleanup = Promise.reject(error); }
    void Promise.allSettled([cacheCleanup, this.preparing, ...[...this.retiring].map(owner => owner.done)]).then(results => {
      if (results[0].status === 'rejected') errors.push(results[0].reason);
      finish();
    });
    return this.cleanup;
  }
}
