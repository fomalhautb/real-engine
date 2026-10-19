import { GeometryPageCache, type GeometryPageCacheOptions } from './geometry-stream-cache';
import { GpuGeometryPagePool, type GeometryGpuPoolOptions } from './gpu-geometry-page-pool';
import { GpuGeometryStreamSelection, type GeometryStreamSelectionFeedback, type GeometryStreamSelectionOptions } from './gpu-geometry-stream-selection';
import type { GeometryPageProvider, GeometryStreamDirectory } from './geometry-stream-types';
import type { SceneBatch } from './contracts';
import type { Mat4 } from './math';
import { GeometryStreamWorkingSet } from './geometry-stream-working-set';

export interface GeometryStreamSource {
  readonly directory: GeometryStreamDirectory;
  readonly provider: GeometryPageProvider;
  readonly cpuCache?: GeometryPageCacheOptions;
  readonly gpuPool?: GeometryGpuPoolOptions;
  readonly selection?: GeometryStreamSelectionOptions;
  /** Default 16. Caps CPU request/enqueue visits per rendered frame, independently of byte budgets. */
  readonly maxRequestsPerFrame?: number;
}

/** Renderer bridge. Surface geometry is physically paged; the existing mesh
 * remains an exact full-detail shadow/query/Masked compatibility representation.
 * This is not yet a bounded-memory query proxy or dynamic simplification path. */
export class GpuGeometryStream {
  private source: GeometryStreamSource;
  private readonly sourceMesh: SceneBatch['mesh'];
  private readonly sourceVertexCount: number;
  private readonly sourceVertices: ArrayBuffer;
  private readonly sourceIndices: Uint32Array;
  private readonly sourceUvs?: Float32Array;
  private readonly sourceTangents?: Float32Array;
  private readonly requestsPerFrame: number;
  private pending: GeometryStreamSelectionFeedback['requestedPages'] = [];
  private pendingCursor = 0;
  private readonly workingSet: GeometryStreamWorkingSet;
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

  private constructor(readonly scene: SceneBatch, readonly cache: GeometryPageCache,
    readonly pool: GpuGeometryPagePool, readonly selection: GpuGeometryStreamSelection) {
    this.source = scene.geometryStream!;
    this.sourceMesh = scene.mesh; this.sourceVertexCount = scene.mesh.vertexCount;
    this.sourceVertices = scene.mesh.vertices; this.sourceIndices = scene.mesh.indices;
    this.sourceUvs = scene.mesh.uvs; this.sourceTangents = scene.mesh.tangents;
    this.requestsPerFrame = this.source.maxRequestsPerFrame ?? 16;
    this.workingSet = new GeometryStreamWorkingSet(pool.directory, pool.slotCount);
    this.allocation = this.workingSet.plan([]);
  }
  static async create(device: GPUDevice, scene: SceneBatch, instances: GPUBuffer, track: (bytes: number) => void) {
    const source = scene.geometryStream;
    if (!source || scene.geometryLod || scene.deformation.jointCount || scene.deformation.nodes.length || scene.morphTargets.length) {
      throw new Error('GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE');
    }
    const visits = source.maxRequestsPerFrame ?? 16;
    if (!Number.isSafeInteger(visits) || visits < 1 || visits > 256) throw new Error('GEOMETRY_STREAM_REQUEST_VISIT_BUDGET');
    if (source.directory.sourceVertexCount !== scene.mesh.vertexCount || source.directory.sourceTriangleCount * 3 !== scene.mesh.indices.length
      || source.directory.vertexStride !== 80 + (scene.mesh.uvs ? 8 : 0) + (scene.mesh.tangents ? 16 : 0)) {
      throw new Error('GEOMETRY_STREAM_SOURCE_MISMATCH');
    }
    let cache: GeometryPageCache | undefined, pool: GpuGeometryPagePool | undefined, selection: GpuGeometryStreamSelection | undefined;
    try {
      pool = await GpuGeometryPagePool.create(device, source.directory, source.gpuPool, track);
      cache = await GeometryPageCache.create(source.directory, source.provider, source.cpuCache);
      selection = await GpuGeometryStreamSelection.create(device, pool, instances, scene.capacity, track, source.selection);
      const result = new GpuGeometryStream(scene, cache, pool, selection);
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
  validateSources() {
    if (this.disposed) throw new Error('GEOMETRY_STREAM_DISPOSED');
    if (this.error) throw new Error(this.error);
    if (this.scene.geometryStream !== this.source || this.scene.mesh !== this.sourceMesh || this.scene.mesh.vertexCount !== this.sourceVertexCount
      || this.scene.mesh.vertices !== this.sourceVertices || this.scene.mesh.indices !== this.sourceIndices
      || this.scene.mesh.uvs !== this.sourceUvs || this.scene.mesh.tangents !== this.sourceTangents
      || this.scene.capacity !== this.selection.capacity || this.scene.deformation.jointCount || this.scene.deformation.nodes.length || this.scene.morphTargets.length) {
      throw new Error('GEOMETRY_STREAM_ASSET_CHANGED');
    }
  }
  private checkPage(id: number) {
    if (this.checkedPages.has(id)) return;
    const page = this.cache.get(id); if (!page) throw new Error('GEOMETRY_STREAM_CPU_PAGE_MISSING');
    const original = new Uint8Array(this.sourceVertices);
    // Exact rest attributes/UV/tangents are shared with the full-detail query
    // representation. Source topology is immutable by the scene asset contract.
    for (const cluster of page.clusters) for (let vertex = 0; vertex < cluster.sourceVertexIds.length; vertex++) {
      const source = cluster.sourceVertexIds[vertex];
      for (let byte = 0; byte < 80; byte++) if (cluster.rest[vertex * 80 + byte] !== original[source * 80 + byte]) throw new Error('GEOMETRY_STREAM_SOURCE_ATTRIBUTES');
      if (cluster.uvs) for (let axis = 0; axis < 2; axis++) if (cluster.uvs[vertex * 2 + axis] !== this.sourceUvs![source * 2 + axis]) throw new Error('GEOMETRY_STREAM_SOURCE_UVS');
      if (cluster.tangents) for (let axis = 0; axis < 4; axis++) if (cluster.tangents[vertex * 4 + axis] !== this.sourceTangents![source * 4 + axis]) throw new Error('GEOMETRY_STREAM_SOURCE_TANGENTS');
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
        + this.pool.directory.clusters[id].vertexCount * 128 + this.pool.directory.clusters[id].triangleCount * 12 + 32, 4);
      if (state.pendingPages.length >= state.limits.maxQueuedPages || state.stagingBytes + bytes > state.limits.maxStagingBytes) break;
      this.checkPage(pageId); this.pool.enqueue(pageId, page, priority);
    }
    const upload = this.pool.uploadFrame(++this.uploadSerial);
    return upload.bytes + this.selection.update(frame, count, vp, eye, projectionScale);
  }
  /** Queue readback contains only bounded requests/counters, never geometry. */
  afterSubmit() {
    if (this.reading || this.disposed) return; this.reading = true;
    this.feedbackTask = this.selection.feedback().then(result => {
      if (this.disposed) return;
      if (result.overflow) { this.error = 'GEOMETRY_STREAM_DRAW_OVERFLOW'; return; }
      this.lastFeedback = result; this.allocation = this.workingSet.plan(result.desiredGroups);
      this.pending = this.allocation.requestedPages;
      this.pendingCursor = this.pending.length ? this.pendingCursor % this.pending.length : 0;
    }).catch(error => { if (!this.disposed) this.error = 'GEOMETRY_STREAM_FEEDBACK_FAILED: ' + String(error); }).finally(() => { this.reading = false; });
  }
  async probe() {
    await this.feedbackTask; this.validateSources();
    return this.selection.feedback(true);
  }
  retry(page: number) { this.validateSources(); this.cache.retry(page); }
  diagnostics() { return { pool: this.pool.diagnostics(), cache: this.cache.diagnostics(), selection: this.selection.diagnostics(),
    error: this.error, feedback: this.lastFeedback, sourceRepresentation: 'full-detail-shadow-query-and-masked' as const,
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
    this.pending = []; this.cpuWindow = []; this.cpuWindowBudgetDeferred = []; this.checkedPages.clear(); this.lastFeedback = null;
    const finish = () => errors.length ? reject(new AggregateError(errors, 'GEOMETRY_STREAM_CLEANUP_FAILED')) : resolve();
    try { void this.cache.dispose().then(finish, error => { errors.push(error); finish(); }); }
    catch (error) { errors.push(error); finish(); }
    return this.cleanup;
  }
}
