import type { DecodedGeometryPage } from './geometry-pages';
import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';
import { GEOMETRY_GPU_VERTEX_BYTES, packGeometryGpuPage, type PackedGeometryGpuPage } from './geometry-gpu-page';
import { geometryGpuPoolLayout, GEOMETRY_GPU_LOCATION_BYTES as LOCATION_BYTES } from './geometry-gpu-pool-layout';

export interface GeometryGpuPoolOptions {
  /** Actual allocated buffers, including fixed-slot padding and metadata. */
  maxResidentBytes?: number;
  /** Owned upload payloads, not JS overhead, caller data or driver staging. */
  maxStagingBytes?: number;
  maxQueuedPages?: number;
  maxUploadBytesPerFrame?: number;
  maxUploadWritesPerFrame?: number;
  maxUploadPagesPerFrame?: number;
  maxFramesInFlight?: number;
}
export interface GeometryGpuClusterLocation {
  pageId: number; slot: number; generation: number;
  /** Add baseVertex to the page-local indices; firstVertex is local to the page. */
  baseVertex: number; firstVertex: number; vertexCount: number;
  firstIndex: number; indexCount: number;
}
export interface GeometryGpuFrameLease { readonly serial: number }
type Slot = { page: number | null; generation: number };
type Resident = { slot: number; generation: number; touched: number };
type Pending = { packed: PackedGeometryGpuPage; bytes: number; priority: number; sequence: number;
  slot?: number; locations?: Uint32Array<ArrayBuffer>; phase: number; offset: number };
type Frame = { pages: Set<number>; submitted: boolean };
const MiB = 1024 * 1024;

function option(value: number | undefined, fallback: number, minimum: number, maximum: number, name: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < minimum || result > maximum) throw new Error('GEOMETRY_GPU_POOL_OPTION: ' + name);
  return result;
}

/** Physical page storage, independent of selection, material, query and simulation.
 * Residency is published only after all uploads on the same queue. Callers must
 * acquire a lease BEFORE encoding draws and submit those buffers through this
 * pool; a lease cannot be cancelled after submission. No geometry readback is
 * needed to decide allocation or retirement. */
export class GpuGeometryPagePool {
  readonly directory: GeometryStreamDirectory;
  readonly options: Readonly<Required<GeometryGpuPoolOptions>>;
  readonly vertices: GPUBuffer;
  readonly indices: GPUBuffer;
  /** Eight u32: baseVertex, firstVertex, vertexCount, firstIndex, indexCount, pageId, generation, slot. */
  readonly clusterLocations: GPUBuffer;
  /** One u32 per page; locations and geometry may only be read when this is 1. */
  readonly pageResidency: GPUBuffer;
  readonly verticesPerSlot: number;
  readonly indicesPerSlot: number;
  readonly slotCount: number;
  readonly bytesPerSlot: number;
  private readonly slots: Slot[];
  private readonly rootSlots = new Map<number, number>();
  private readonly resident = new Map<number, Resident>();
  private readonly pending = new Map<number, Pending>();
  private readonly frames = new Map<GeometryGpuFrameLease, Frame>();
  private readonly leased = new Map<number, number>();
  private readonly buffers: GPUBuffer[] = [];
  private readonly locations: Uint32Array<ArrayBuffer>;
  private pinned = new Set<number>();
  private allocatedBytes = 0;
  private stagingBytes = 0;
  private sequence = 0;
  private frameSerial = 0;
  private uploadFrameNumber = 0;
  private activePage?: number;
  private disposed = false;
  private error: string | null = null;
  private uploads = 0;
  private evictions = 0;

  static async create(device: GPUDevice, directory: GeometryStreamDirectory, options: GeometryGpuPoolOptions = {},
    track: (bytes: number) => void = () => {}): Promise<GpuGeometryPagePool> {
    let pool: GpuGeometryPagePool | undefined;
    device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    try {
      pool = new GpuGeometryPagePool(device, directory, options, track);
    } catch (error) {
      await Promise.allSettled([device.popErrorScope(), device.popErrorScope()]); throw error;
    }
    try {
      const [validation, allocation] = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
      if (validation || allocation) throw new Error('GEOMETRY_GPU_POOL_INITIALIZATION: ' + (validation ?? allocation)!.message);
      pool.live(); return pool;
    } catch (error) { try { pool.dispose(); } catch { /* Preserve the initialization failure. */ } throw error; }
  }

  private constructor(private readonly device: GPUDevice, directory: GeometryStreamDirectory, options: GeometryGpuPoolOptions,
    private readonly track: (bytes: number) => void) {
    this.directory = validateGeometryStreamDirectory(directory);
    this.options = Object.freeze({
      maxResidentBytes: option(options.maxResidentBytes, 16 * MiB, 1, 256 * MiB, 'maxResidentBytes'),
      maxStagingBytes: option(options.maxStagingBytes, 16 * MiB, 1, 256 * MiB, 'maxStagingBytes'),
      maxQueuedPages: option(options.maxQueuedPages, 64, 1, 4096, 'maxQueuedPages'),
      maxUploadBytesPerFrame: option(options.maxUploadBytesPerFrame, 512 * 1024, 32, 16 * MiB, 'maxUploadBytesPerFrame'),
      maxUploadWritesPerFrame: option(options.maxUploadWritesPerFrame, 64, 1, 4096, 'maxUploadWritesPerFrame'),
      maxUploadPagesPerFrame: option(options.maxUploadPagesPerFrame, 4, 1, 256, 'maxUploadPagesPerFrame'),
      maxFramesInFlight: option(options.maxFramesInFlight, 3, 1, 8, 'maxFramesInFlight'),
    });
    if (this.options.maxUploadBytesPerFrame % 4) throw new Error('GEOMETRY_GPU_POOL_UPLOAD_ALIGNMENT');
    const layout = geometryGpuPoolLayout(this.directory);
    this.verticesPerSlot = layout.verticesPerSlot;
    this.indicesPerSlot = layout.indicesPerSlot;
    if (this.directory.pages.some(page => this.pagePayloadBytes(page.id) > this.options.maxStagingBytes)) {
      throw new Error('GEOMETRY_GPU_POOL_PAGE_STAGING_BUDGET');
    }
    this.bytesPerSlot = layout.bytesPerSlot;
    const metadataBytes = layout.metadataBytes;
    this.slotCount = Math.min(this.directory.pages.length, Math.floor((this.options.maxResidentBytes - metadataBytes) / this.bytesPerSlot));
    if (this.slotCount < this.directory.rootPages.length) throw new Error('GEOMETRY_GPU_POOL_ROOT_BUDGET');
    const sizes = [this.slotCount * this.verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES,
      this.slotCount * this.indicesPerSlot * 4, this.directory.clusters.length * LOCATION_BYTES, this.directory.pages.length * 4];
    if (sizes.some(size => size > device.limits.maxBufferSize || size > device.limits.maxStorageBufferBindingSize)) {
      throw new Error('GEOMETRY_GPU_POOL_DEVICE_LIMIT');
    }
    this.locations = new Uint32Array(this.directory.clusters.length * 8);
    this.slots = Array.from({ length: this.slotCount }, () => ({ page: null, generation: 0 }));
    this.directory.rootPages.forEach((page, slot) => this.rootSlots.set(page, slot));
    try {
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      this.vertices = this.buffer('paged geometry: 144-byte vertex records', sizes[0], usage);
      this.indices = this.buffer('paged geometry: page-local indices', sizes[1], usage | GPUBufferUsage.INDEX);
      this.clusterLocations = this.buffer('paged geometry: cluster locations', sizes[2], usage);
      this.pageResidency = this.buffer('paged geometry: publication flags', sizes[3], usage);
    } catch (error) { try { this.dispose(); } catch { /* Preserve the allocation failure. */ } throw error; }
    void device.lost.then(info => { if (!this.disposed) {
      this.error = 'GEOMETRY_GPU_POOL_DEVICE_LOST: ' + info.message;
      try { this.dispose(); } catch (error) { this.error += '; tracker: ' + String(error); }
    } });
  }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size, usage });
    this.buffers.push(buffer); this.allocatedBytes += size; this.track(size); return buffer;
  }
  private live(): void { if (this.disposed) throw new Error(this.error ?? 'GEOMETRY_GPU_POOL_DISPOSED'); }
  private page(id: number) {
    if (!Number.isSafeInteger(id) || id < 0 || id >= this.directory.pages.length) throw new Error('GEOMETRY_GPU_POOL_PAGE_ID');
    return this.directory.pages[id];
  }
  private pagePayloadBytes(id: number): number {
    return this.page(id).clusterIds.reduce((sum, cluster) => sum + GEOMETRY_GPU_VERTEX_BYTES * this.directory.clusters[cluster].vertexCount
      + this.directory.clusters[cluster].triangleCount * 12 + LOCATION_BYTES, 4);
  }

  /** Snapshot a decoded page; duplicates do not allocate another upload payload. */
  enqueue(pageId: number, decoded: DecodedGeometryPage, priority = 0): 'resident' | 'queued' | 'merged' {
    this.live(); this.page(pageId);
    if (!Number.isFinite(priority)) throw new Error('GEOMETRY_GPU_POOL_PRIORITY');
    const resident = this.resident.get(pageId);
    if (resident) { resident.touched = ++this.sequence; return 'resident'; }
    const pending = this.pending.get(pageId);
    if (pending) { pending.priority = Math.max(pending.priority, priority); return 'merged'; }
    const bytes = this.pagePayloadBytes(pageId);
    if (this.pending.size >= this.options.maxQueuedPages || this.stagingBytes + bytes > this.options.maxStagingBytes) {
      throw new Error('GEOMETRY_GPU_POOL_STAGING_BUDGET');
    }
    const packed = packGeometryGpuPage(this.directory, pageId, decoded);
    this.pending.set(pageId, { packed, bytes, priority, sequence: ++this.sequence, phase: 0, offset: 0 });
    this.stagingBytes += bytes; return 'queued';
  }
  /** Discard upload payloads outside the desired set. Root uploads are retained;
   * published pages and frame leases are independent of this staging decision. */
  retainPendingPages(ids: readonly number[]): void {
    this.live();
    if (!Array.isArray(ids) || ids.length > this.directory.pages.length) throw new Error('GEOMETRY_GPU_POOL_PAGE_SET');
    const desired = new Set<number>(this.directory.rootPages);
    for (const id of ids) { this.page(id); desired.add(id); }
    for (const [id, pending] of this.pending) {
      if (desired.has(id)) continue;
      if (pending.slot !== undefined) this.slots[pending.slot].page = null;
      if (this.activePage === id) this.activePage = undefined;
      this.pending.delete(id);
      this.stagingBytes -= pending.bytes;
    }
  }
  residentPageIds(): ReadonlySet<number> { return new Set(this.resident.keys()); }
  private pageSet(ids: readonly number[]): Set<number> {
    if (!Array.isArray(ids) || ids.length > this.directory.pages.length) throw new Error('GEOMETRY_GPU_POOL_PAGE_SET');
    const result = new Set<number>();
    for (const id of ids) { this.page(id); if (!this.resident.has(id)) throw new Error('GEOMETRY_GPU_POOL_NOT_RESIDENT: ' + id); result.add(id); }
    return result;
  }
  setPinnedPages(ids: readonly number[]): void { this.live(); const pages = this.pageSet(ids); this.pinned = pages; }
  clusterLocation(clusterId: number): GeometryGpuClusterLocation | undefined {
    this.live();
    if (!Number.isSafeInteger(clusterId) || clusterId < 0 || clusterId >= this.directory.clusters.length) throw new Error('GEOMETRY_GPU_POOL_CLUSTER_ID');
    const pageId = this.directory.clusterPages[clusterId], resident = this.resident.get(pageId);
    if (!resident) return undefined;
    resident.touched = ++this.sequence;
    const at = clusterId * 8;
    return { baseVertex: this.locations[at], firstVertex: this.locations[at + 1], vertexCount: this.locations[at + 2],
      firstIndex: this.locations[at + 3], indexCount: this.locations[at + 4], pageId, generation: resident.generation, slot: resident.slot };
  }

  /** A positive, monotonic frame number prevents repeated calls from resetting the budget. */
  uploadFrame(frame: number) {
    this.live();
    if (!Number.isSafeInteger(frame) || frame <= this.uploadFrameNumber) throw new Error('GEOMETRY_GPU_POOL_FRAME');
    this.uploadFrameNumber = frame;
    const result = { frame, bytes: 0, writes: 0, pagesUploaded: [] as number[], evictedPages: [] as number[], blocked: false };
    const write = (buffer: GPUBuffer, offset: number, data: ArrayBuffer, start: number, bytes: number) => {
      this.device.queue.writeBuffer(buffer, offset, data, start, bytes); result.bytes += bytes; result.writes++;
    };
    try {
      const ordered = [...this.pending].sort((a, b) => Number(this.rootSlots.has(b[0])) - Number(this.rootSlots.has(a[0]))
        || b[1].priority - a[1].priority || a[1].sequence - b[1].sequence).map(([id]) => id);
      if (this.activePage !== undefined) { ordered.splice(ordered.indexOf(this.activePage), 1); ordered.unshift(this.activePage); }
      for (const pageId of ordered) {
        if (result.writes >= this.options.maxUploadWritesPerFrame || result.bytes >= this.options.maxUploadBytesPerFrame
          || result.pagesUploaded.length >= this.options.maxUploadPagesPerFrame) break;
        const pending = this.pending.get(pageId)!;
        if (pending.slot === undefined) {
          let slot = this.rootSlots.get(pageId) ?? this.slots.findIndex((entry, id) => id >= this.rootSlots.size && entry.page === null);
          if (slot < 0) {
            const victims = [...this.resident].filter(([id]) => !this.rootSlots.has(id) && !this.pinned.has(id) && !this.leased.has(id))
              .sort((a, b) => a[1].touched - b[1].touched || a[0] - b[0]);
            if (!victims.length) { result.blocked = true; continue; }
            const [victim, state] = victims[0]; slot = state.slot;
            write(this.pageResidency, victim * 4, new Uint32Array([0]).buffer, 0, 4);
            this.resident.delete(victim); this.slots[slot].page = null; this.evictions++; result.evictedPages.push(victim);
          }
          const physical = this.slots[slot];
          if (physical.generation === 0xffffffff) throw new Error('GEOMETRY_GPU_POOL_GENERATION_EXHAUSTED');
          physical.page = pageId; physical.generation++; pending.slot = slot;
          pending.locations = new Uint32Array(pending.packed.clusters.length * 8);
          for (const [ordinal, cluster] of pending.packed.clusters.entries()) {
            pending.locations.set([slot * this.verticesPerSlot, cluster.vertexOffset, cluster.vertexCount,
              slot * this.indicesPerSlot + cluster.indexOffset, cluster.indexCount, pageId, physical.generation, slot], ordinal * 8);
          }
          this.activePage = pageId;
        }
        while (result.writes < this.options.maxUploadWritesPerFrame) {
          const remaining = this.options.maxUploadBytesPerFrame - result.bytes;
          if (remaining < 4) break;
          const slot = pending.slot!;
          if (pending.phase < 2) {
            const data = pending.phase === 0 ? pending.packed.vertices : pending.packed.indices.buffer;
            const target = pending.phase === 0 ? this.vertices : this.indices;
            const base = slot * (pending.phase === 0 ? this.verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES : this.indicesPerSlot * 4);
            const size = Math.min(remaining, data.byteLength - pending.offset);
            write(target, base + pending.offset, data, pending.offset, size);
            pending.offset += size;
            if (pending.offset === data.byteLength) { pending.phase++; pending.offset = 0; }
          } else if (pending.phase === 2) {
            if (remaining < LOCATION_BYTES) break;
            const cluster = pending.packed.clusters[pending.offset], data = pending.locations!;
            write(this.clusterLocations, cluster.id * LOCATION_BYTES, data.buffer, pending.offset * LOCATION_BYTES, LOCATION_BYTES);
            this.locations.set(data.subarray(pending.offset * 8, pending.offset * 8 + 8), cluster.id * 8);
            if (++pending.offset === pending.packed.clusters.length) { pending.phase++; pending.offset = 0; }
          } else {
            write(this.pageResidency, pageId * 4, new Uint32Array([1]).buffer, 0, 4);
            this.resident.set(pageId, { slot, generation: this.slots[slot].generation, touched: ++this.sequence });
            this.pending.delete(pageId); this.stagingBytes -= pending.bytes; this.activePage = undefined;
            this.uploads++; result.pagesUploaded.push(pageId); break;
          }
        }
        // Finish a partial page before reserving more slots or payload work.
        if (this.pending.has(pageId)) break;
      }
      return result;
    } catch (error) {
      this.error = 'GEOMETRY_GPU_POOL_UPLOAD_FAILED: ' + String(error);
      try { this.dispose(); } catch (trackerError) { this.error += '; tracker: ' + String(trackerError); }
      throw error;
    }
  }

  /** Acquire before constructing any commands that reference the selected pages. */
  acquireFrame(ids: readonly number[]): GeometryGpuFrameLease {
    this.live();
    if (!this.directory.rootPages.every(id => this.resident.has(id))) throw new Error('GEOMETRY_GPU_POOL_ROOTS_NOT_READY');
    if (this.frames.size >= this.options.maxFramesInFlight) throw new Error('GEOMETRY_GPU_POOL_FRAMES_IN_FLIGHT');
    const pages = this.pageSet(ids), token = Object.freeze({ serial: ++this.frameSerial });
    for (const id of pages) { this.leased.set(id, (this.leased.get(id) ?? 0) + 1); this.resident.get(id)!.touched = ++this.sequence; }
    this.frames.set(token, { pages, submitted: false }); return token;
  }
  private release(token: GeometryGpuFrameLease): void {
    const frame = this.frames.get(token); if (!frame) return;
    for (const id of frame.pages) {
      const count = this.leased.get(id)! - 1;
      if (count) this.leased.set(id, count); else this.leased.delete(id);
    }
    this.frames.delete(token);
  }
  cancelFrame(token: GeometryGpuFrameLease): void {
    this.live(); const frame = this.frames.get(token);
    if (!frame || frame.submitted) throw new Error('GEOMETRY_GPU_POOL_FRAME_LEASE'); this.release(token);
  }
  /** Submit one command batch that references leases from pools on one device.
   * Every lease stays pinned until the same queue completion fence settles. */
  static submitFrames(entries: readonly { pool: GpuGeometryPagePool; lease: GeometryGpuFrameLease }[],
    commands: readonly GPUCommandBuffer[]): Promise<{ error: string | null }> {
    // A renderer can contain up to 4096 independently paged batches. Every pool
    // must participate in the same queue submission and completion fence.
    if (!Array.isArray(entries) || entries.length < 1 || entries.length > 4096
      || !Array.isArray(commands) || commands.length < 1 || commands.length > 32) {
      throw new Error('GEOMETRY_GPU_POOL_FRAME_LEASE');
    }
    const seen = new Set<GpuGeometryPagePool>();
    let device: GPUDevice | undefined;
    const selected: { pool: GpuGeometryPagePool; lease: GeometryGpuFrameLease; frame: Frame }[] = [];
    for (const entry of entries) {
      const pool = entry?.pool;
      if (!(pool instanceof GpuGeometryPagePool) || seen.has(pool) || (device && pool.device !== device)) {
        throw new Error('GEOMETRY_GPU_POOL_FRAME_LEASE');
      }
      pool.live();
      const frame = pool.frames.get(entry.lease);
      if (!frame || frame.submitted) throw new Error('GEOMETRY_GPU_POOL_FRAME_LEASE');
      seen.add(pool); device = pool.device;
      selected.push({ pool, lease: entry.lease, frame });
    }
    // All validation is complete before the first lease or queue is changed.
    try { device!.queue.submit(commands); }
    catch (error) { for (const { pool, lease } of selected) pool.release(lease); throw error; }
    for (const { frame } of selected) frame.submitted = true;
    const failed = (cause: unknown): { error: string } => {
      const errors: string[] = [];
      for (const { pool, lease } of selected) {
        pool.release(lease);
        pool.error = 'GEOMETRY_GPU_POOL_QUEUE_FAILED: ' + String(cause);
        try { pool.dispose(); } catch (error) { pool.error += '; tracker: ' + String(error); }
        errors.push(pool.error);
      }
      return { error: [...new Set(errors)].join('; ') };
    };
    try {
      return device!.queue.onSubmittedWorkDone().then(() => {
        for (const { pool, lease } of selected) pool.release(lease);
        return { error: selected.find(({ pool }) => pool.error)?.pool.error ?? null };
      }, failed);
    } catch (error) { return Promise.resolve(failed(error)); }
  }
  /** Submit through the owning pool so queued GPU frames keep their slots pinned. */
  submitFrame(token: GeometryGpuFrameLease, commands: readonly GPUCommandBuffer[]): Promise<{ error: string | null }> {
    return GpuGeometryPagePool.submitFrames([{ pool: this, lease: token }], commands);
  }
  diagnostics() {
    return { disposed: this.disposed, error: this.error, allocatedGpuBytes: this.allocatedBytes,
      slotCount: this.slotCount, bytesPerSlot: this.bytesPerSlot, stagingBytes: this.stagingBytes,
      residentPages: [...this.resident.keys()].sort((a, b) => a - b), pendingPages: [...this.pending.keys()],
      rootPages: [...this.directory.rootPages], rootsReady: this.directory.rootPages.every(id => this.resident.has(id)),
      pinnedPages: this.disposed ? [] : [...new Set([...this.directory.rootPages, ...this.pinned])].sort((a, b) => a - b),
      leasedPages: [...this.leased.keys()].sort((a, b) => a - b), framesInFlight: this.frames.size,
      activePage: this.activePage ?? null, uploadFrame: this.uploadFrameNumber, uploads: this.uploads, evictions: this.evictions,
      residency: 'published-on-queue' as const, limits: { ...this.options } };
  }
  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy(); this.buffers.length = 0;
    const released = this.allocatedBytes; this.allocatedBytes = 0; this.stagingBytes = 0;
    this.pending.clear(); this.resident.clear(); this.frames.clear(); this.leased.clear(); this.pinned.clear(); this.activePage = undefined;
    this.track(-released);
  }
}
