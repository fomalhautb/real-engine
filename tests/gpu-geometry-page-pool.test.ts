import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { GeometryDemoScene } from '../src/geometry-demo-scene';
import { decodeGeometryPage, encodeGeometryPages, type DecodedGeometryPage } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { GpuGeometryPagePool, type GeometryGpuPoolOptions } from '../src/gpu-geometry-page-pool';
import { GEOMETRY_GPU_VERTEX_BYTES } from '../src/geometry-gpu-page';

type FakeBuffer = { label: string; size: number; data: Uint8Array; destroyed: boolean; destroy: () => void };

class FakeDevice {
  readonly buffers: FakeBuffer[] = [];
  readonly writes: { label: string; offset: number; bytes: number }[] = [];
  readonly submissions: (readonly GPUCommandBuffer[])[] = [];
  readonly fences: { resolve: () => void; reject: (error: Error) => void }[] = [];
  readonly limits = { maxBufferSize: 256 * 1024 * 1024, maxStorageBufferBindingSize: 256 * 1024 * 1024 };
  readonly lost: Promise<{ message: string }>;
  private lose!: (info: { message: string }) => void;
  failBufferAt = -1;

  constructor() { this.lost = new Promise(resolve => { this.lose = resolve; }); }
  createBuffer = ({ label, size }: { label: string; size: number }): GPUBuffer => {
    if (this.buffers.length === this.failBufferAt) throw new Error('FAKE_ALLOCATION_FAILED');
    const buffer: FakeBuffer = { label, size, data: new Uint8Array(size), destroyed: false,
      destroy() { this.destroyed = true; } };
    this.buffers.push(buffer);
    return buffer as unknown as GPUBuffer;
  };
  pushErrorScope = (_filter: string): void => {};
  popErrorScope = async (): Promise<null> => null;
  queue = {
    writeBuffer: (target: GPUBuffer, offset: number, source: ArrayBuffer, sourceOffset: number, size: number): void => {
      const buffer = target as unknown as FakeBuffer;
      if (buffer.destroyed || offset < 0 || offset + size > buffer.size) throw new Error('FAKE_WRITE_BOUNDS');
      buffer.data.set(new Uint8Array(source, sourceOffset, size), offset);
      this.writes.push({ label: buffer.label, offset, bytes: size });
    },
    submit: (commands: readonly GPUCommandBuffer[]): void => { this.submissions.push(commands); },
    onSubmittedWorkDone: (): Promise<void> => new Promise((resolve, reject) => {
      this.fences.push({ resolve, reject });
    }),
  };
  buffer(label: string): FakeBuffer { return this.buffers.find(buffer => buffer.label.includes(label))!; }
  completeNext(): void { this.fences.shift()!.resolve(); }
  loseDevice(): void { this.lose({ message: 'fake loss' }); }
}

let directory: GeometryStreamDirectory;
let pages: DecodedGeometryPage[];
let roots: number[];
let details: number[];
let bytesPerSlot: number;
let metadataBytes: number;
const pools: GpuGeometryPagePool[] = [];
const command = {} as GPUCommandBuffer;

async function poolWithSlots(device: FakeDevice, slots: number, options: GeometryGpuPoolOptions = {},
  track: (bytes: number) => void = () => {}): Promise<GpuGeometryPagePool> {
  const pool = await GpuGeometryPagePool.create(device as unknown as GPUDevice, directory,
    { maxResidentBytes: metadataBytes + bytesPerSlot * slots, ...options }, track);
  pools.push(pool);
  expect(pool.slotCount).toBe(slots);
  return pool;
}

function enqueueRoots(pool: GpuGeometryPagePool): void {
  for (const id of roots) expect(pool.enqueue(id, pages[id])).toBe('queued');
}

function uploadUntil(pool: GpuGeometryPagePool, wanted: readonly number[], firstFrame = 1): number {
  let frame = firstFrame;
  while (!wanted.every(id => pool.residentPageIds().has(id))) {
    expect(frame).toBeLessThan(firstFrame + 500);
    pool.uploadFrame(frame++);
  }
  return frame;
}

beforeAll(async () => {
  const hierarchy = (await GeometryDemoScene.create()).hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  pages = await Promise.all(encoded.pages.map(page => decodeGeometryPage(page.bytes, { maxPageBytes: 49152 })));
  roots = [...directory.rootPages];
  details = directory.pages.map(page => page.id).filter(id => !roots.includes(id));
  expect(roots).toHaveLength(4);
  expect(details.length).toBeGreaterThan(3);
  const verticesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce(
    (sum, id) => sum + directory.clusters[id].vertexCount, 0)));
  const indicesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce(
    (sum, id) => sum + directory.clusters[id].triangleCount * 3, 0)));
  bytesPerSlot = verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES + indicesPerSlot * 4;
  metadataBytes = directory.clusters.length * 32 + directory.pages.length * 4;
});

beforeEach(() => {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, INDEX: 8 });
});
afterEach(() => { for (const pool of pools.splice(0)) pool.dispose(); vi.unstubAllGlobals(); });
afterAll(() => { pages = []; });

describe('GPU geometry page pool CPU lifecycle with deferred fake queue', () => {
  it('rejects a root budget before allocating any buffers', async () => {
    const device = new FakeDevice();
    await expect(GpuGeometryPagePool.create(device as unknown as GPUDevice, directory,
      { maxResidentBytes: metadataBytes + bytesPerSlot * (roots.length - 1) }))
      .rejects.toThrow(/ROOT_BUDGET/);
    expect(device.buffers).toHaveLength(0);
  });

  it('bounds each frame, rejects a repeated frame number, and publishes only complete pages', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1, {
      maxUploadBytesPerFrame: 1024, maxUploadWritesPerFrame: 2, maxUploadPagesPerFrame: 1,
    });
    expect(() => { (pool.options as { maxUploadBytesPerFrame: number }).maxUploadBytesPerFrame = 4096; }).toThrow();
    enqueueRoots(pool);
    expect(() => pool.acquireFrame([])).toThrow(/ROOTS_NOT_READY/);
    let frame = 1, priorWrites = 0;
    while (!pool.diagnostics().rootsReady) {
      const result = pool.uploadFrame(frame);
      expect(result.bytes).toBeLessThanOrEqual(1024);
      expect(result.writes).toBeLessThanOrEqual(2);
      expect(result.pagesUploaded.length).toBeLessThanOrEqual(1);
      expect(device.writes.length - priorWrites).toBe(result.writes);
      priorWrites = device.writes.length;
      expect(() => pool.uploadFrame(frame)).toThrow(/GEOMETRY_GPU_POOL_FRAME/);
      for (const id of roots) {
        const published = new DataView(device.buffer('publication flags').data.buffer).getUint32(id * 4, true);
        expect(published).toBe(pool.residentPageIds().has(id) ? 1 : 0);
      }
      expect(frame++).toBeLessThan(500);
    }
    expect(pool.diagnostics().stagingBytes).toBe(0);
    expect(pool.diagnostics().pendingPages).toEqual([]);
    expect(device.writes.at(-1)?.label).toContain('publication flags');
  });

  it('merges duplicate requests, owns staging payloads, and blocks allocation over the queue budget', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1, { maxQueuedPages: 1 });
    const id = roots[0];
    expect(pool.enqueue(id, pages[id], 1)).toBe('queued');
    const staging = pool.diagnostics().stagingBytes;
    expect(staging).toBeGreaterThan(0);
    expect(pool.enqueue(id, pages[id], 5)).toBe('merged');
    expect(pool.diagnostics().stagingBytes).toBe(staging);
    expect(() => pool.enqueue(roots[1], pages[roots[1]])).toThrow(/STAGING_BUDGET/);
    uploadUntil(pool, [id]);
    expect(pool.enqueue(id, pages[id])).toBe('resident');
    expect(pool.diagnostics().stagingBytes).toBe(0);
  });

  it('retains root uploads while cancelling only undesired queued payloads', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1);
    enqueueRoots(pool);
    const rootStaging = pool.diagnostics().stagingBytes;
    const a = details[0], b = details[1];
    pool.enqueue(a, pages[a]); pool.enqueue(b, pages[b]);
    const allStaging = pool.diagnostics().stagingBytes;
    pool.retainPendingPages([a]);
    expect(pool.diagnostics()).toMatchObject({ pendingPages: [...roots, a], activePage: null });
    expect(pool.diagnostics().stagingBytes).toBeLessThan(allStaging);
    expect(pool.diagnostics().stagingBytes).toBeGreaterThan(rootStaging);
    pool.retainPendingPages([]);
    expect(pool.diagnostics().pendingPages).toEqual(roots);
    expect(pool.diagnostics().stagingBytes).toBe(rootStaging);
    expect(pool.enqueue(a, pages[a])).toBe('queued');
  });

  it('cancels a partial unpublished page and reuses its slot with a newer generation', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1, {
      maxUploadBytesPerFrame: 1024, maxUploadWritesPerFrame: 2,
    });
    enqueueRoots(pool);
    let frame = uploadUntil(pool, roots);
    const a = details[0], b = details[1];
    pool.enqueue(a, pages[a]);
    const staging = pool.diagnostics().stagingBytes;
    const partial = pool.uploadFrame(frame++);
    expect(partial.writes).toBeGreaterThan(0);
    expect(pool.diagnostics()).toMatchObject({ pendingPages: [a], activePage: a, stagingBytes: staging });
    expect(pool.clusterLocation(directory.pages[a].clusterIds[0])).toBeUndefined();
    const writesBeforeCancellation = device.writes.length;
    pool.retainPendingPages([]);
    expect(device.writes).toHaveLength(writesBeforeCancellation);
    expect(pool.diagnostics()).toMatchObject({ pendingPages: [], activePage: null, stagingBytes: 0,
      residentPages: [...roots].sort((x, y) => x - y), evictions: 0 });
    expect(new DataView(device.buffer('publication flags').data.buffer).getUint32(a * 4, true)).toBe(0);
    pool.enqueue(b, pages[b]);
    frame = uploadUntil(pool, [b], frame);
    const replacement = pool.clusterLocation(directory.pages[b].clusterIds[0])!;
    expect(replacement.slot).toBe(roots.length);
    expect(replacement.generation).toBe(2);
    expect(pool.enqueue(a, pages[a])).toBe('queued');
    uploadUntil(pool, [a], frame);
    expect(pool.clusterLocation(directory.pages[a].clusterIds[0])!.generation).toBeGreaterThan(replacement.generation);
  });

  it('does not alter published pages or held and submitted leases while cancelling pending uploads', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 2, { maxUploadBytesPerFrame: 1024 });
    enqueueRoots(pool);
    let frame = uploadUntil(pool, roots);
    const a = details[0], b = details[1];
    pool.enqueue(a, pages[a]); frame = uploadUntil(pool, [a], frame);
    const location = pool.clusterLocation(directory.pages[a].clusterIds[0])!;
    const held = pool.acquireFrame([a]), submitted = pool.acquireFrame([a]);
    const completion = pool.submitFrame(submitted, [command]);
    pool.enqueue(b, pages[b]); pool.uploadFrame(frame++);
    expect(pool.diagnostics().activePage).toBe(b);
    pool.retainPendingPages([]);
    expect(pool.diagnostics()).toMatchObject({ pendingPages: [], activePage: null, framesInFlight: 2, leasedPages: [a] });
    expect(pool.clusterLocation(directory.pages[a].clusterIds[0])).toMatchObject(location);
    expect(new DataView(device.buffer('publication flags').data.buffer).getUint32(a * 4, true)).toBe(1);
    expect(() => pool.cancelFrame(submitted)).toThrow(/FRAME_LEASE/);
    pool.cancelFrame(held);
    device.completeNext();
    await expect(completion).resolves.toEqual({ error: null });
    expect(pool.diagnostics().framesInFlight).toBe(0);
    expect(pool.residentPageIds().has(a)).toBe(true);
  });

  it('validates the complete desired set before changing pending uploads', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1, { maxUploadBytesPerFrame: 1024 });
    enqueueRoots(pool);
    let frame = uploadUntil(pool, roots);
    const a = details[0], b = details[1];
    pool.enqueue(a, pages[a]); pool.enqueue(b, pages[b]);
    pool.uploadFrame(frame++);
    const before = pool.diagnostics();
    const writes = device.writes.length;
    expect(() => pool.retainPendingPages([a, directory.pages.length])).toThrow(/PAGE_ID/);
    expect(() => pool.retainPendingPages([a, -1])).toThrow(/PAGE_ID/);
    expect(() => pool.retainPendingPages([a, 1.5])).toThrow(/PAGE_ID/);
    expect(() => pool.retainPendingPages(Array(directory.pages.length + 1).fill(a))).toThrow(/PAGE_SET/);
    expect(() => pool.retainPendingPages(new Set([a]) as unknown as number[])).toThrow(/PAGE_SET/);
    expect(pool.diagnostics()).toMatchObject({ pendingPages: before.pendingPages, activePage: before.activePage,
      stagingBytes: before.stagingBytes, residentPages: before.residentPages, evictions: before.evictions });
    expect(device.writes).toHaveLength(writes);
    pool.retainPendingPages([a]);
    expect(pool.diagnostics().pendingPages).toEqual([a]);
  });

  it('keeps pinned and held or submitted frame pages out of LRU reuse until release', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1);
    enqueueRoots(pool);
    let frame = uploadUntil(pool, roots);
    const a = details[0], b = details[1];
    expect(pool.enqueue(a, pages[a])).toBe('queued');
    frame = uploadUntil(pool, [a], frame);
    const location = pool.clusterLocation(directory.pages[a].clusterIds[0])!;
    pool.setPinnedPages([a]);
    expect(pool.enqueue(b, pages[b])).toBe('queued');
    expect(pool.uploadFrame(frame++).blocked).toBe(true);
    expect(pool.residentPageIds().has(a)).toBe(true);
    pool.setPinnedPages([]);
    const held = pool.acquireFrame([a]);
    expect(pool.uploadFrame(frame++).blocked).toBe(true);
    pool.cancelFrame(held);
    expect(() => pool.cancelFrame(held)).toThrow(/FRAME_LEASE/);
    const submitted = pool.acquireFrame([a]);
    const completion = pool.submitFrame(submitted, [command]);
    expect(() => pool.cancelFrame(submitted)).toThrow(/FRAME_LEASE/);
    expect(() => pool.submitFrame(submitted, [command])).toThrow(/FRAME_LEASE/);
    expect(pool.uploadFrame(frame++).blocked).toBe(true);
    expect(device.fences).toHaveLength(1);
    device.completeNext();
    await expect(completion).resolves.toEqual({ error: null });
    frame = uploadUntil(pool, [b], frame);
    expect(pool.residentPageIds().has(a)).toBe(false);
    expect(pool.clusterLocation(directory.pages[a].clusterIds[0])).toBeUndefined();
    expect(pool.clusterLocation(directory.pages[b].clusterIds[0])!.slot).toBe(location.slot);
    expect(pool.clusterLocation(directory.pages[b].clusterIds[0])!.generation).toBeGreaterThan(location.generation);
    expect(pool.diagnostics().evictions).toBe(1);
    expect(pool.diagnostics().framesInFlight).toBe(0);
  });

  it('submits two pools on one device with one fence and keeps both leases pinned', async () => {
    const device = new FakeDevice();
    const first = await poolWithSlots(device, roots.length + 1);
    const second = await poolWithSlots(device, roots.length + 1);
    enqueueRoots(first); enqueueRoots(second);
    let firstFrame = uploadUntil(first, roots);
    let secondFrame = uploadUntil(second, roots);
    const a = details[0], b = details[1];
    first.enqueue(a, pages[a]); second.enqueue(a, pages[a]);
    firstFrame = uploadUntil(first, [a], firstFrame);
    secondFrame = uploadUntil(second, [a], secondFrame);
    first.enqueue(b, pages[b]); second.enqueue(b, pages[b]);
    const firstLease = first.acquireFrame([a]);
    const secondLease = second.acquireFrame([a]);
    const completion = GpuGeometryPagePool.submitFrames([
      { pool: first, lease: firstLease }, { pool: second, lease: secondLease },
    ], [command]);
    expect(device.submissions).toEqual([[command]]);
    expect(device.fences).toHaveLength(1);
    expect(first.uploadFrame(firstFrame++).blocked).toBe(true);
    expect(second.uploadFrame(secondFrame++).blocked).toBe(true);
    expect(() => first.cancelFrame(firstLease)).toThrow(/FRAME_LEASE/);
    expect(() => second.cancelFrame(secondLease)).toThrow(/FRAME_LEASE/);
    device.completeNext();
    await expect(completion).resolves.toEqual({ error: null });
    expect(first.diagnostics().framesInFlight).toBe(0);
    expect(second.diagnostics().framesInFlight).toBe(0);
    uploadUntil(first, [b], firstFrame);
    uploadUntil(second, [b], secondFrame);
    expect(first.residentPageIds().has(a)).toBe(false);
    expect(second.residentPageIds().has(a)).toBe(false);
  });

  it('rejects different devices, duplicate pools and forged leases without changing valid leases', async () => {
    const device = new FakeDevice(), otherDevice = new FakeDevice();
    const first = await poolWithSlots(device, roots.length + 1);
    const second = await poolWithSlots(device, roots.length + 1);
    const other = await poolWithSlots(otherDevice, roots.length + 1);
    for (const pool of [first, second, other]) { enqueueRoots(pool); uploadUntil(pool, roots); }
    const firstLease = first.acquireFrame(roots), secondLease = second.acquireFrame(roots);
    const otherLease = other.acquireFrame(roots);
    expect(() => GpuGeometryPagePool.submitFrames([], [command])).toThrow(/FRAME_LEASE/);
    expect(() => GpuGeometryPagePool.submitFrames(
      Array(257).fill({ pool: first, lease: firstLease }), [command])).toThrow(/FRAME_LEASE/);
    expect(() => GpuGeometryPagePool.submitFrames([{ pool: first, lease: firstLease }], [])).toThrow(/FRAME_LEASE/);
    expect(() => GpuGeometryPagePool.submitFrames(
      [{ pool: first, lease: firstLease }], Array(33).fill(command))).toThrow(/FRAME_LEASE/);
    expect(() => GpuGeometryPagePool.submitFrames([
      { pool: first, lease: firstLease }, { pool: other, lease: otherLease },
    ], [command])).toThrow(/FRAME_LEASE/);
    expect(() => GpuGeometryPagePool.submitFrames([
      { pool: first, lease: firstLease }, { pool: first, lease: firstLease },
    ], [command])).toThrow(/FRAME_LEASE/);
    expect(() => GpuGeometryPagePool.submitFrames([
      { pool: first, lease: firstLease }, { pool: second, lease: { serial: secondLease.serial } },
    ], [command])).toThrow(/FRAME_LEASE/);
    expect(device.submissions).toHaveLength(0);
    expect(otherDevice.submissions).toHaveLength(0);
    for (const pool of [first, second, other]) expect(pool.diagnostics().framesInFlight).toBe(1);
    first.cancelFrame(firstLease); second.cancelFrame(secondLease); other.cancelFrame(otherLease);
  });

  it('releases every selected lease if queue submission throws synchronously', async () => {
    const device = new FakeDevice();
    const first = await poolWithSlots(device, roots.length + 1);
    const second = await poolWithSlots(device, roots.length + 1);
    for (const pool of [first, second]) { enqueueRoots(pool); uploadUntil(pool, roots); }
    const firstLease = first.acquireFrame(roots), secondLease = second.acquireFrame(roots);
    device.queue.submit = () => { throw new Error('FAKE_SUBMIT_FAILURE'); };
    expect(() => GpuGeometryPagePool.submitFrames([
      { pool: first, lease: firstLease }, { pool: second, lease: secondLease },
    ], [command])).toThrow('FAKE_SUBMIT_FAILURE');
    expect(device.fences).toHaveLength(0);
    expect(first.diagnostics().framesInFlight).toBe(0);
    expect(second.diagnostics().framesInFlight).toBe(0);
    expect(first.diagnostics().disposed).toBe(false);
    expect(second.diagnostics().disposed).toBe(false);
  });

  it('cleans all pools after a shared fence failure even when one tracker throws', async () => {
    const device = new FakeDevice();
    const first = await poolWithSlots(device, roots.length + 1, {}, bytes => {
      if (bytes < 0) throw new Error('FAKE_TRACKER_FAILURE');
    });
    const second = await poolWithSlots(device, roots.length + 1);
    for (const pool of [first, second]) { enqueueRoots(pool); uploadUntil(pool, roots); }
    const completion = GpuGeometryPagePool.submitFrames([
      { pool: first, lease: first.acquireFrame(roots) },
      { pool: second, lease: second.acquireFrame(roots) },
    ], [command]);
    expect(device.fences).toHaveLength(1);
    device.fences.shift()!.reject(new Error('FAKE_QUEUE_FAILURE'));
    expect((await completion).error).toMatch(/QUEUE_FAILED.*FAKE_QUEUE_FAILURE.*FAKE_TRACKER_FAILURE/);
    for (const pool of [first, second]) {
      expect(pool.diagnostics()).toMatchObject({ disposed: true, allocatedGpuBytes: 0,
        framesInFlight: 0, leasedPages: [] });
    }
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
  });

  it('rejects forged leases and cleans up tracked buffers and pending payload on device loss', async () => {
    const device = new FakeDevice();
    let tracked = 0;
    const pool = await poolWithSlots(device, roots.length + 1, {}, bytes => { tracked += bytes; });
    expect(tracked).toBe(pool.diagnostics().allocatedGpuBytes);
    enqueueRoots(pool);
    uploadUntil(pool, roots);
    expect(() => pool.cancelFrame({ serial: 1 })).toThrow(/FRAME_LEASE/);
    expect(() => pool.submitFrame({ serial: 1 }, [command])).toThrow(/FRAME_LEASE/);
    pool.enqueue(details[0], pages[details[0]]);
    expect(pool.diagnostics().stagingBytes).toBeGreaterThan(0);
    device.loseDevice();
    await Promise.resolve();
    expect(pool.diagnostics()).toMatchObject({ disposed: true, allocatedGpuBytes: 0, stagingBytes: 0,
      pendingPages: [], residentPages: [] });
    expect(pool.diagnostics().error).toMatch(/DEVICE_LOST/);
    expect(tracked).toBe(0);
    expect(device.buffers).toHaveLength(4);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
  });

  it('destroys earlier allocations if later buffer creation fails', async () => {
    const device = new FakeDevice();
    device.failBufferAt = 2;
    let tracked = 0;
    await expect(GpuGeometryPagePool.create(device as unknown as GPUDevice, directory,
      { maxResidentBytes: metadataBytes + bytesPerSlot * (roots.length + 1) },
      bytes => { tracked += bytes; })).rejects.toThrow(/FAKE_ALLOCATION_FAILED/);
    expect(tracked).toBe(0);
    expect(device.buffers).toHaveLength(2);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
  });

  it('reports a failed submitted fence and releases pages, staging and buffers', async () => {
    const device = new FakeDevice();
    let tracked = 0;
    const pool = await poolWithSlots(device, roots.length + 1, {}, bytes => { tracked += bytes; });
    enqueueRoots(pool); uploadUntil(pool, roots);
    pool.enqueue(details[0], pages[details[0]]);
    const token = pool.acquireFrame(roots);
    const completion = pool.submitFrame(token, [command]);
    device.fences.shift()!.reject(new Error('FAKE_QUEUE_FAILURE'));
    expect((await completion).error).toMatch(/QUEUE_FAILED.*FAKE_QUEUE_FAILURE/);
    expect(pool.diagnostics()).toMatchObject({ disposed: true, allocatedGpuBytes: 0, stagingBytes: 0,
      framesInFlight: 0, residentPages: [], pendingPages: [], leasedPages: [] });
    expect(tracked).toBe(0);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(() => pool.uploadFrame(100)).toThrow(/QUEUE_FAILED/);
  });

  it('preserves upload failure and finishes cleanup even when the accounting callback throws', async () => {
    const device = new FakeDevice();
    const pool = await poolWithSlots(device, roots.length + 1, {}, bytes => {
      if (bytes < 0) throw new Error('FAKE_TRACKER_FAILURE');
    });
    enqueueRoots(pool);
    device.queue.writeBuffer = () => { throw new Error('FAKE_UPLOAD_FAILURE'); };
    expect(() => pool.uploadFrame(1)).toThrow('FAKE_UPLOAD_FAILURE');
    expect(pool.diagnostics()).toMatchObject({ disposed: true, allocatedGpuBytes: 0, stagingBytes: 0,
      framesInFlight: 0, residentPages: [], pendingPages: [] });
    expect(pool.diagnostics().error).toMatch(/UPLOAD_FAILED.*FAKE_UPLOAD_FAILURE.*FAKE_TRACKER_FAILURE/);
    expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
    expect(() => pool.dispose()).not.toThrow();
  });
});
