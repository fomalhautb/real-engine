import { decodeGeometryPage, type DecodedGeometryPage } from './geometry-pages';
import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryPageProvider, GeometryStreamDirectory, GeometryStreamPage } from './geometry-stream-types';

export interface GeometryPageCacheOptions {
  /** Decoded page payload accounting, including reservations, not JS/process RSS. */
  maxResidentBytes?: number;
  /** Per-request reservation: 2 * encodedBytes + 2 * decodedBytes. */
  maxInFlightBytes?: number;
  maxConcurrentRequests?: number;
  maxQueuedRequests?: number;
}
type Resident = { page: DecodedGeometryPage; bytes: number; touched: number };
type Request = { priority: number; sequence: number };
type Active = { controller: AbortController; promise: Promise<void>; discarded: boolean };

function limit(value: number | undefined, fallback: number, maximum: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new Error('GEOMETRY_STREAM_OPTION: ' + label);
  return result;
}

/** Bounded CPU fetch/decode cache. Decoded availability is NOT GPU residency.
 * Consumers must separately upload, fence and retire physical GPU page slots before
 * passing GPU-resident IDs to a rendering cut planner. Returned page data is immutable
 * asset input by contract; it must not be mutated or retained as an unaccounted cache. */
export class GeometryPageCache {
  readonly directory: GeometryStreamDirectory;
  private readonly options: Required<GeometryPageCacheOptions>;
  private readonly resident = new Map<number, Resident>();
  private readonly queued = new Map<number, Request>();
  private readonly active = new Map<number, Active>();
  private readonly failures = new Map<number, string>();
  private pinned: Set<number>;
  private idlePromise?: Promise<void>;
  private idleResolve?: () => void;
  private residentBytes = 0;
  private reservedBytes = 0;
  private inFlightBytes = 0;
  private sequence = 0;
  private scheduled = false;
  private disposed = false;
  private disposal?: Promise<void>;
  private requestsStarted = 0;
  private cacheHits = 0;
  private mergedRequests = 0;
  private evictions = 0;

  static async create(directory: GeometryStreamDirectory, provider: GeometryPageProvider,
    options: GeometryPageCacheOptions = {}): Promise<GeometryPageCache> {
    const result = new GeometryPageCache(directory, provider, options);
    try {
      for (const page of result.directory.rootPages) result.request(page, Number.MAX_SAFE_INTEGER);
      await result.idle();
      for (const page of result.directory.rootPages) {
        if (!result.resident.has(page)) throw new Error('GEOMETRY_STREAM_ROOT_LOAD_FAILED: ' + page + ': ' + result.failures.get(page));
      }
      return result;
    } catch (error) { await result.dispose(); throw error; }
  }

  private constructor(directory: GeometryStreamDirectory, private readonly provider: GeometryPageProvider,
    options: GeometryPageCacheOptions) {
    this.directory = validateGeometryStreamDirectory(directory);
    if (typeof provider !== 'function') throw new Error('GEOMETRY_STREAM_PROVIDER');
    this.options = {
      maxResidentBytes: limit(options.maxResidentBytes, 16 * 1024 * 1024, 256 * 1024 * 1024, 'maxResidentBytes'),
      maxInFlightBytes: limit(options.maxInFlightBytes, 16 * 1024 * 1024, 64 * 1024 * 1024, 'maxInFlightBytes'),
      maxConcurrentRequests: limit(options.maxConcurrentRequests, 2, 16, 'maxConcurrentRequests'),
      maxQueuedRequests: limit(options.maxQueuedRequests, 4096, 65_536, 'maxQueuedRequests'),
    };
    this.pinned = new Set(this.directory.rootPages);
    const rootBytes = this.directory.rootPages.reduce((sum, id) => sum + this.directory.pages[id].decodedBytes, 0);
    // Whole root pages may also contain unrelated clusters. Charge the entire page.
    if (rootBytes > this.options.maxResidentBytes || this.pinned.size > this.options.maxQueuedRequests) {
      throw new Error('GEOMETRY_STREAM_ROOT_BUDGET');
    }
    for (const page of this.directory.pages) {
      if (page.decodedBytes > this.options.maxResidentBytes
        || this.workBytes(page) > this.options.maxInFlightBytes) throw new Error('GEOMETRY_STREAM_PAGE_BUDGET: ' + page.id);
    }
  }

  private live(): void { if (this.disposed) throw new Error('GEOMETRY_STREAM_DISPOSED'); }
  private page(id: number): GeometryStreamPage {
    if (!Number.isSafeInteger(id) || id < 0 || id >= this.directory.pages.length) throw new Error('GEOMETRY_STREAM_PAGE_ID');
    return this.directory.pages[id];
  }
  private workBytes(page: GeometryStreamPage): number { return 2 * page.encodedBytes + 2 * page.decodedBytes; }

  /** Duplicate pending requests merge; failures require explicit retry, never a frame-by-frame retry storm. */
  request(pageId: number, priority = 0): void {
    this.live(); this.page(pageId);
    if (!Number.isFinite(priority)) throw new Error('GEOMETRY_STREAM_PRIORITY');
    const resident = this.resident.get(pageId);
    if (resident) { resident.touched = ++this.sequence; this.cacheHits++; return; }
    const queued = this.queued.get(pageId);
    if (queued || this.active.has(pageId)) {
      if (queued) queued.priority = Math.max(queued.priority, priority);
      this.mergedRequests++; return;
    }
    if (this.failures.has(pageId)) return;
    if (this.queued.size >= this.options.maxQueuedRequests) throw new Error('GEOMETRY_STREAM_REQUEST_BUDGET');
    this.queued.set(pageId, { priority, sequence: ++this.sequence });
    this.schedule();
  }

  retry(pageId: number, priority = 0): void {
    this.live(); this.page(pageId);
    if (!Number.isFinite(priority)) throw new Error('GEOMETRY_STREAM_PRIORITY');
    if (!this.resident.has(pageId) && !this.queued.has(pageId) && !this.active.has(pageId)
      && this.queued.size >= this.options.maxQueuedRequests) throw new Error('GEOMETRY_STREAM_REQUEST_BUDGET');
    this.failures.delete(pageId); this.request(pageId, priority);
  }

  get(pageId: number): DecodedGeometryPage | undefined {
    this.live(); this.page(pageId);
    const entry = this.resident.get(pageId);
    if (entry) entry.touched = ++this.sequence;
    return entry?.page;
  }
  residentPageIds(): ReadonlySet<number> { return new Set(this.resident.keys()); }

  /** CPU retention only. Roots remain pinned. GPU in-flight lifetimes are a separate consumer contract. */
  setPinnedPages(ids: readonly number[]): void {
    this.live();
    if (!Array.isArray(ids) || ids.length > this.directory.pages.length) throw new Error('GEOMETRY_STREAM_PIN_BUDGET');
    const next = new Set(this.directory.rootPages);
    for (const id of ids) {
      this.page(id);
      if (!this.resident.has(id)) throw new Error('GEOMETRY_STREAM_PIN_NOT_RESIDENT');
      next.add(id);
    }
    this.pinned = next; this.schedule();
  }

  /** Retain the requested CPU working set, including pages still awaiting decode.
   * Roots are always included. This does not enqueue missing pages. */
  setRequestWindow(ids: readonly number[]): void {
    this.live();
    if (!Array.isArray(ids) || ids.length > this.directory.pages.length) throw new Error('GEOMETRY_STREAM_WINDOW_BUDGET');
    const next = new Set(this.directory.rootPages);
    let bytes = [...next].reduce((sum, id) => sum + this.directory.pages[id].decodedBytes, 0);
    for (const id of ids) {
      const page = this.page(id);
      if (next.has(id)) continue;
      next.add(id);
      bytes += page.decodedBytes;
      if (bytes > this.options.maxResidentBytes) throw new Error('GEOMETRY_STREAM_WINDOW_BUDGET');
    }

    this.pinned = next;
    for (const id of this.queued.keys()) if (!next.has(id)) this.queued.delete(id);
    // Abort listeners may reenter with a newer window; check its current pins each time.
    for (const [id, entry] of [...this.active]) {
      if (this.active.get(id) !== entry || this.pinned.has(id)) continue;
      entry.discarded = true;
      entry.controller.abort();
    }
    this.schedule();
  }

  /** Wait for runnable queued work and active requests. Capacity-blocked requests may remain queued. */
  idle(): Promise<void> {
    if (this.disposed && !this.active.size) return Promise.resolve();
    if (this.idlePromise) return this.idlePromise;
    const promise = this.idlePromise = new Promise<void>(resolve => { this.idleResolve = resolve; });
    if (this.disposed) this.resolveIdle(); else this.schedule();
    return promise;
  }
  private schedule(): void {
    if (this.scheduled || this.disposed) return;
    this.scheduled = true;
    queueMicrotask(() => { this.scheduled = false; this.pump(); });
  }
  private resolveIdle(): void {
    if (this.active.size || this.scheduled) return;
    const resolve = this.idleResolve;
    this.idlePromise = undefined; this.idleResolve = undefined;
    resolve?.();
  }
  private evictionCandidates(): [number, Resident][] {
    return [...this.resident].filter(([id]) => !this.pinned.has(id))
      .sort((a, b) => a[1].touched - b[1].touched || a[0] - b[0]);
  }
  private pump(): void {
    if (this.disposed) { this.resolveIdle(); return; }
    const requests = [...this.queued].sort((a, b) => b[1].priority - a[1].priority || a[1].sequence - b[1].sequence);
    for (const [id] of requests) {
      if (this.active.size >= this.options.maxConcurrentRequests) break;
      const page = this.page(id), work = this.workBytes(page);
      if (work + this.inFlightBytes > this.options.maxInFlightBytes) continue;
      const evictable = this.evictionCandidates();
      const required = this.residentBytes + this.reservedBytes + page.decodedBytes - this.options.maxResidentBytes;
      if (required > evictable.reduce((sum, [, entry]) => sum + entry.bytes, 0)) continue;
      let remaining = required;
      for (const [victim, entry] of evictable) {
        if (remaining <= 0) break;
        this.resident.delete(victim); this.residentBytes -= entry.bytes; remaining -= entry.bytes; this.evictions++;
      }
      this.queued.delete(id);
      this.reservedBytes += page.decodedBytes; this.inFlightBytes += work; this.requestsStarted++;
      const active: Active = { controller: new AbortController(), promise: Promise.resolve(), discarded: false };
      let reservationHeld = true;
      this.active.set(id, active);
      // Publish the actual lifetime promise before invoking external provider code.
      // A provider may synchronously trigger disposal or enqueue another request.
      active.promise = Promise.resolve().then(() => this.load(page, active.controller.signal)).then(decoded => {
        this.reservedBytes -= page.decodedBytes; reservationHeld = false;
        if (this.disposed || active.discarded) return;
        this.resident.set(id, { page: decoded, bytes: page.decodedBytes, touched: ++this.sequence });
        this.residentBytes += page.decodedBytes;
      }).catch(error => {
        if (!this.disposed && !active.discarded) this.failures.set(id, error instanceof Error ? error.message : String(error));
      }).finally(() => {
        if (reservationHeld) this.reservedBytes -= page.decodedBytes;
        this.inFlightBytes -= work; this.active.delete(id);
        this.pump();
      });
    }
    this.resolveIdle();
  }

  private async load(page: GeometryStreamPage, signal: AbortSignal): Promise<DecodedGeometryPage> {
    signal.throwIfAborted();
    const response = await this.provider(page, signal);
    signal.throwIfAborted();
    if (!(response instanceof Uint8Array) || response.byteLength !== page.encodedBytes) throw new Error('GEOMETRY_STREAM_ENCODED_LENGTH: ' + page.id);
    // Snapshot external buffers before hashing/decoding, never trust a mutable provider view.
    const bytes = new Uint8Array(response);
    const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    signal.throwIfAborted();
    const actual = Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('');
    if (actual !== page.sha256) throw new Error('GEOMETRY_STREAM_HASH: ' + page.id);
    const decoded = await decodeGeometryPage(bytes, { maxPageBytes: page.decodedBytes, maxEncodedBytes: page.encodedBytes });
    signal.throwIfAborted();
    if (decoded.vertexStride !== this.directory.vertexStride || decoded.sourceVertexCount !== this.directory.sourceVertexCount
      || decoded.hierarchyClusterCount !== this.directory.clusters.length || decoded.clusters.length !== page.clusterIds.length) {
      throw new Error('GEOMETRY_STREAM_PAGE_IDENTITY: ' + page.id);
    }
    for (let ordinal = 0; ordinal < decoded.clusters.length; ordinal++) {
      const cluster = decoded.clusters[ordinal], expectedId = page.clusterIds[ordinal];
      const metadata = this.directory.clusters[expectedId];
      if (cluster.id !== expectedId || cluster.sourceVertexIds.length !== metadata.vertexCount
        || cluster.localTriangles.length !== metadata.triangleCount * 3) throw new Error('GEOMETRY_STREAM_CLUSTER_IDENTITY: ' + page.id);
    }
    return decoded;
  }

  diagnostics() {
    return { residentBytes: this.residentBytes, reservedBytes: this.reservedBytes, inFlightBytes: this.inFlightBytes,
      activeRequests: this.active.size, queuedRequests: this.queued.size,
      residentPages: [...this.resident.keys()].sort((a, b) => a - b), pinnedPages: [...this.pinned].sort((a, b) => a - b),
      failedPages: [...this.failures].map(([pageId, message]) => ({ pageId, message })),
      requestsStarted: this.requestsStarted, cacheHits: this.cacheHits, mergedRequests: this.mergedRequests,
      evictions: this.evictions, disposed: this.disposed, limits: { ...this.options }, gpuResidency: false as const };
  }

  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.disposed = true; this.queued.clear();
    for (const entry of this.active.values()) entry.controller.abort();
    this.disposal = Promise.all([...this.active.values()].map(entry => entry.promise)).then(() => {
      this.resident.clear(); this.failures.clear(); this.pinned.clear(); this.residentBytes = 0;
      this.resolveIdle();
    });
    return this.disposal;
  }
}
