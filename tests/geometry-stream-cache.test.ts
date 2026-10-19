import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { MeshData } from '../src/geometry';
import { buildGeometryHierarchy } from '../src/geometry-hierarchy';
import { encodeGeometryPages, type EncodedGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import { GeometryPageCache } from '../src/geometry-stream-cache';
import type { GeometryPageProvider, GeometryStreamDirectory } from '../src/geometry-stream-types';

function grid(size = 12): MeshData {
  const side = size + 1, vertexCount = side * side;
  const vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
  const uvs = new Float32Array(vertexCount * 2), indices: number[] = [];
  for (let y = 0; y <= size; y++) for (let x = 0; x <= size; x++) {
    const vertex = y * side + x, offset = vertex * 80;
    view.setFloat32(offset, x, true);
    view.setFloat32(offset + 4, y, true);
    view.setFloat32(offset + 8, 0.1 * Math.sin(x * 0.5) * Math.cos(y * 0.4), true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setFloat32(offset + 48, 1, true);
    uvs[vertex * 2] = x / size;
    uvs[vertex * 2 + 1] = y / size;
  }
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const a = y * side + x, b = a + 1, c = a + side, d = c + 1;
    indices.push(a, b, d, a, d, c);
  }
  return { vertices, vertexCount, indices: Uint32Array.from(indices), uvs,
    bounds: { min: [0, 0, -0.1], max: [size, size, 0.1] } };
}

let directory: GeometryStreamDirectory;
let encoded: EncodedGeometryPages;
let otherIds: number[];

function bytes(id: number): Uint8Array { return new Uint8Array(encoded.pages[id].bytes); }
function provider(): GeometryPageProvider { return page => Promise.resolve(bytes(page.id)); }
function rootBytes(): number { return directory.rootPages.reduce((sum, id) => sum + directory.pages[id].decodedBytes, 0); }

async function eventually(test: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (test()) return;
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  throw new Error('timed out waiting for cache state');
}

function heldProvider(held: ReadonlySet<number>) {
  const started: number[] = [];
  const waiting = new Map<number, { signal: AbortSignal; release: () => void }>();
  const fetch: GeometryPageProvider = (page, signal) => {
    started.push(page.id);
    if (!held.has(page.id)) return Promise.resolve(bytes(page.id));
    return new Promise<Uint8Array>((resolve, reject) => {
      const abort = () => { waiting.delete(page.id); reject(new DOMException('aborted', 'AbortError')); };
      signal.addEventListener('abort', abort, { once: true });
      waiting.set(page.id, { signal, release: () => {
        signal.removeEventListener('abort', abort);
        waiting.delete(page.id);
        resolve(bytes(page.id));
      } });
      if (signal.aborted) abort();
    });
  };
  return { fetch, started, waiting };
}

describe('bounded CPU geometry page cache', () => {
  beforeAll(async () => {
    const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'cache-grid', mesh: grid(), coverage: 'opaque' }, {
      maxClusterVertices: 12, maxClusterTriangles: 6, groupSize: 2, maxLevels: 6,
      simplifyRatio: 0.5, maxSimplificationError: 2,
    });
    encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 2048 });
    directory = await buildGeometryStreamDirectory(hierarchy, encoded);
    otherIds = directory.pages.map(page => page.id).filter(id => !directory.rootPages.includes(id));
    expect(directory.rootPages.length).toBeGreaterThan(0);
    expect(otherIds.length).toBeGreaterThanOrEqual(3);
  });

  it('loads and pins every root before create resolves, then serves decoded metadata', async () => {
    const calls: number[] = [];
    const fetch: GeometryPageProvider = (page, signal) => {
      expect(signal.aborted).toBe(false);
      calls.push(page.id);
      return Promise.resolve(bytes(page.id));
    };
    const cache = await GeometryPageCache.create(directory, fetch);
    try {
      expect(calls.sort((a, b) => a - b)).toEqual([...directory.rootPages].sort((a, b) => a - b));
      const state = cache.diagnostics();
      expect(state.residentBytes).toBe(rootBytes());
      expect(state.pinnedPages.sort((a, b) => a - b)).toEqual([...directory.rootPages].sort((a, b) => a - b));
      expect(cache.residentPageIds().size).toBe(directory.rootPages.length);
      for (const id of directory.rootPages) {
        const decoded = cache.get(id)!;
        expect(decoded.clusters.map(cluster => cluster.id)).toEqual(directory.pages[id].clusterIds);
        expect(decoded.vertexStride).toBe(directory.vertexStride);
        expect(decoded.sourceVertexCount).toBe(directory.sourceVertexCount);
        for (const cluster of decoded.clusters) {
          expect(cluster.sourceVertexIds).toHaveLength(directory.clusters[cluster.id].vertexCount);
          expect(cluster.localTriangles).toHaveLength(directory.clusters[cluster.id].triangleCount * 3);
        }
        cache.request(id); // Resident requests count as cache hits.
      }
      expect(cache.diagnostics().cacheHits).toBeGreaterThanOrEqual(directory.rootPages.length);
    } finally { await cache.dispose(); }
  });

  it('rejects a root set that cannot fit before contacting the provider', async () => {
    const fetch = vi.fn(provider());
    await expect(GeometryPageCache.create(directory, fetch, { maxResidentBytes: rootBytes() - 1 }))
      .rejects.toThrow();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('merges duplicate requests and runs the highest-priority queued page next', async () => {
    const [active, low, high] = otherIds;
    const held = heldProvider(new Set([active, low, high]));
    const cache = await GeometryPageCache.create(directory, held.fetch, {
      maxConcurrentRequests: 1, maxResidentBytes: directory.pages.reduce((sum, page) => sum + page.decodedBytes, 0),
      maxInFlightBytes: 4 * 1024 * 1024,
    });
    try {
      cache.request(active);
      await eventually(() => held.waiting.has(active));
      cache.request(active, 3);
      cache.request(low, 1);
      cache.request(high, 9);
      cache.request(high, 10);
      expect(cache.diagnostics().mergedRequests).toBeGreaterThanOrEqual(2);
      expect(held.started.filter(id => id === active)).toHaveLength(1);
      expect(cache.diagnostics().activeRequests).toBe(1);
      expect(cache.diagnostics().reservedBytes).toBe(directory.pages[active].decodedBytes);
      expect(cache.diagnostics().inFlightBytes).toBeLessThanOrEqual(4 * 1024 * 1024);
      held.waiting.get(active)!.release();
      await eventually(() => held.waiting.has(high));
      held.waiting.get(high)!.release();
      await eventually(() => held.waiting.has(low));
      held.waiting.get(low)!.release();
      await cache.idle();
      expect(held.started.filter(id => [active, low, high].includes(id))).toEqual([active, high, low]);
      for (const id of [active, low, high]) expect(cache.get(id)).toBeDefined();
      expect(cache.diagnostics().reservedBytes).toBe(0);
      expect(cache.diagnostics().activeRequests).toBe(0);
    } finally { await cache.dispose(); }
  });

  it('keeps roots pinned, blocks on an extra pin, then evicts the old leaf after unpinning', async () => {
    const [first, second] = otherIds;
    const capacity = rootBytes() + Math.max(directory.pages[first].decodedBytes, directory.pages[second].decodedBytes);
    const calls: number[] = [];
    const fetch: GeometryPageProvider = page => { calls.push(page.id); return Promise.resolve(bytes(page.id)); };
    const cache = await GeometryPageCache.create(directory, fetch, { maxResidentBytes: capacity,
      maxInFlightBytes: 4 * 1024 * 1024 });
    try {
      cache.request(first);
      await cache.idle();
      expect(cache.get(first)).toBeDefined();
      cache.setPinnedPages([first]);
      expect(() => cache.setPinnedPages([first, second])).toThrow(/GEOMETRY_STREAM_PIN_NOT_RESIDENT/);
      expect(cache.diagnostics().pinnedPages).toContain(first); // Invalid update was atomic.
      cache.request(second);
      await cache.idle(); // Capacity-blocked work does not keep idle pending forever.
      expect(calls).not.toContain(second);
      expect(cache.diagnostics().queuedRequests).toBe(1);
      expect(cache.get(first)).toBeDefined();
      cache.setPinnedPages([]);
      await cache.idle();
      expect(calls).toContain(second);
      expect(cache.get(second)).toBeDefined();
      expect(cache.get(first)).toBeUndefined();
      expect(cache.diagnostics().evictions).toBeGreaterThanOrEqual(1);
      for (const root of directory.rootPages) expect(cache.get(root)).toBeDefined();
      expect(cache.diagnostics().residentBytes).toBeLessThanOrEqual(capacity);
    } finally { await cache.dispose(); }
  });

  it('validates a request window atomically and charges unique pages plus roots', async () => {
    const [first, second] = otherIds;
    const capacity = rootBytes() + directory.pages[first].decodedBytes;
    const cache = await GeometryPageCache.create(directory, provider(), { maxResidentBytes: capacity });
    try {
      cache.setRequestWindow([first, first, directory.rootPages[0]]);
      const pinned = cache.diagnostics().pinnedPages;
      expect(pinned).toContain(first);
      expect(pinned).toEqual(expect.arrayContaining([...directory.rootPages]));
      expect(() => cache.setRequestWindow([first, second])).toThrow(/GEOMETRY_STREAM_WINDOW_BUDGET/);
      expect(() => cache.setRequestWindow([first, -1])).toThrow(/GEOMETRY_STREAM_PAGE_ID/);
      expect(() => cache.setRequestWindow(Array(directory.pages.length + 1).fill(first)))
        .toThrow(/GEOMETRY_STREAM_WINDOW_BUDGET/);
      expect(cache.diagnostics().pinnedPages).toEqual(pinned);
      expect(cache.diagnostics().queuedRequests).toBe(0);
    } finally { await cache.dispose(); }
  });

  it('retains a future decoded page until the window changes, then unblocks queued work', async () => {
    const [first, second] = otherIds;
    const capacity = rootBytes() + Math.max(directory.pages[first].decodedBytes, directory.pages[second].decodedBytes);
    const cache = await GeometryPageCache.create(directory, provider(), { maxResidentBytes: capacity });
    try {
      cache.setRequestWindow([first]);
      expect(cache.get(first)).toBeUndefined();
      cache.request(first);
      await cache.idle();
      expect(cache.get(first)).toBeDefined();
      cache.request(second);
      await cache.idle();
      expect(cache.diagnostics().queuedRequests).toBe(1);
      expect(cache.get(first)).toBeDefined();
      cache.setRequestWindow([second]);
      await cache.idle();
      expect(cache.get(first)).toBeUndefined();
      expect(cache.get(second)).toBeDefined();
      for (const root of directory.rootPages) expect(cache.get(root)).toBeDefined();
    } finally { await cache.dispose(); }
  });

  it('removes excluded queued requests without starting them', async () => {
    const [active, queued] = otherIds;
    const held = heldProvider(new Set([active]));
    const cache = await GeometryPageCache.create(directory, held.fetch, { maxConcurrentRequests: 1 });
    try {
      cache.request(active);
      await eventually(() => held.waiting.has(active));
      cache.request(queued);
      cache.setRequestWindow([active]);
      expect(cache.diagnostics().queuedRequests).toBe(0);
      held.waiting.get(active)!.release();
      await cache.idle();
      expect(held.started).not.toContain(queued);
    } finally { await cache.dispose(); }
  });

  it('aborts excluded active requests and discards late results and errors', async () => {
    const [lateResult, lateError] = otherIds;
    const waiting = new Map<number, { signal: AbortSignal; resolve: (value: Uint8Array) => void;
      reject: (error: Error) => void }>();
    const fetch: GeometryPageProvider = (page, signal) => {
      if (directory.rootPages.includes(page.id)) return Promise.resolve(bytes(page.id));
      return new Promise((resolve, reject) => waiting.set(page.id, { signal, resolve, reject }));
    };
    const cache = await GeometryPageCache.create(directory, fetch, { maxConcurrentRequests: 2 });
    try {
      cache.request(lateResult);
      cache.request(lateError);
      await eventually(() => waiting.size === 2);
      const reservation = cache.diagnostics().reservedBytes;
      cache.setRequestWindow([]);
      expect(waiting.get(lateResult)!.signal.aborted).toBe(true);
      expect(waiting.get(lateError)!.signal.aborted).toBe(true);
      expect(cache.diagnostics().reservedBytes).toBe(reservation);
      expect(cache.diagnostics().activeRequests).toBe(2);
      waiting.get(lateResult)!.resolve(bytes(lateResult));
      waiting.get(lateError)!.reject(new Error('late provider failure'));
      await cache.idle();
      expect(cache.get(lateResult)).toBeUndefined();
      expect(cache.get(lateError)).toBeUndefined();
      expect(cache.diagnostics().failedPages).toEqual([]);
      expect(cache.diagnostics().reservedBytes).toBe(0);
      expect(cache.diagnostics().inFlightBytes).toBe(0);
      expect(cache.diagnostics().pinnedPages).toEqual([...directory.rootPages].sort((a, b) => a - b));
    } finally { await cache.dispose(); }
  });

  it('lets a reentrant newer window retain an active page', async () => {
    const [first, second] = otherIds;
    const waiting = new Map<number, { signal: AbortSignal; release: () => void }>();
    let cache!: GeometryPageCache;
    const fetch: GeometryPageProvider = (page, signal) => {
      if (directory.rootPages.includes(page.id)) return Promise.resolve(bytes(page.id));
      return new Promise(resolve => {
        waiting.set(page.id, { signal, release: () => resolve(bytes(page.id)) });
        if (page.id === first) signal.addEventListener('abort', () => cache.setRequestWindow([second]), { once: true });
      });
    };
    cache = await GeometryPageCache.create(directory, fetch, { maxConcurrentRequests: 2,
      maxResidentBytes: rootBytes() + directory.pages[first].decodedBytes + directory.pages[second].decodedBytes });
    try {
      cache.request(first);
      cache.request(second);
      await eventually(() => waiting.size === 2);
      cache.setRequestWindow([]);
      expect(waiting.get(first)!.signal.aborted).toBe(true);
      expect(waiting.get(second)!.signal.aborted).toBe(false);
      waiting.get(first)!.release();
      waiting.get(second)!.release();
      await cache.idle();
      expect(cache.get(first)).toBeUndefined();
      expect(cache.get(second)).toBeDefined();
      expect(cache.diagnostics().failedPages).toEqual([]);
    } finally { await cache.dispose(); }
  });

  it('evicts the least recently used unpinned leaf after a read refreshes its age', async () => {
    const bySize = [...otherIds].sort((a, b) => directory.pages[a].decodedBytes - directory.pages[b].decodedBytes);
    const incoming = bySize[0], oldest = bySize[bySize.length - 1], recent = bySize[1];
    expect(new Set([incoming, oldest, recent]).size).toBe(3);
    const capacity = rootBytes() + directory.pages[oldest].decodedBytes + directory.pages[recent].decodedBytes;
    const cache = await GeometryPageCache.create(directory, provider(), { maxResidentBytes: capacity,
      maxInFlightBytes: 4 * 1024 * 1024 });
    try {
      cache.request(recent);
      await cache.idle();
      cache.request(oldest);
      await cache.idle();
      expect(cache.get(recent)).toBeDefined(); // Most recent use wins over insertion order.
      cache.request(incoming);
      await cache.idle();
      expect(cache.get(recent)).toBeDefined();
      expect(cache.get(incoming)).toBeDefined();
      expect(cache.get(oldest)).toBeUndefined();
      expect(cache.diagnostics().evictions).toBeGreaterThanOrEqual(1);
      for (const id of directory.rootPages) expect(cache.get(id)).toBeDefined();
    } finally { await cache.dispose(); }
  });

  it('rejects wrong length and hash, keeps ancestors, and fetches again only on explicit retry', async () => {
    const [lengthId, hashId] = otherIds;
    const attempts = new Map<number, number>();
    const fetch: GeometryPageProvider = page => {
      const count = (attempts.get(page.id) ?? 0) + 1;
      attempts.set(page.id, count);
      const valid = bytes(page.id);
      if (page.id === lengthId && count === 1) return Promise.resolve(valid.subarray(0, valid.length - 1));
      if (page.id === hashId && count === 1) { valid[valid.length - 1] ^= 1; return Promise.resolve(valid); }
      return Promise.resolve(valid);
    };
    const cache = await GeometryPageCache.create(directory, fetch);
    try {
      cache.request(lengthId);
      cache.request(hashId);
      await cache.idle();
      expect(cache.get(lengthId)).toBeUndefined();
      expect(cache.get(hashId)).toBeUndefined();
      expect(cache.diagnostics().failedPages.map(failure => failure.pageId).sort((a, b) => a - b))
        .toEqual([lengthId, hashId].sort((a, b) => a - b));
      expect(cache.diagnostics().failedPages.find(failure => failure.pageId === lengthId)?.message)
        .toMatch(/GEOMETRY_STREAM_ENCODED_LENGTH/);
      expect(cache.diagnostics().failedPages.find(failure => failure.pageId === hashId)?.message)
        .toMatch(/GEOMETRY_STREAM_HASH/);
      for (const root of directory.rootPages) expect(cache.get(root)).toBeDefined();
      cache.request(lengthId);
      cache.request(hashId);
      await cache.idle();
      expect(attempts.get(lengthId)).toBe(1);
      expect(attempts.get(hashId)).toBe(1);
      cache.retry(lengthId);
      cache.retry(hashId);
      await cache.idle();
      expect(attempts.get(lengthId)).toBe(2);
      expect(attempts.get(hashId)).toBe(2);
      expect(cache.get(lengthId)).toBeDefined();
      expect(cache.get(hashId)).toBeDefined();
      expect(cache.diagnostics().failedPages).toEqual([]);
    } finally { await cache.dispose(); }
  });

  it('rejects correctly hashed pages whose decoded source or cluster identity disagrees with metadata', async () => {
    const mismatchedSource = { ...directory, sourceVertexCount: directory.sourceVertexCount + 1 };
    const rootFetch = vi.fn(provider());
    await expect(GeometryPageCache.create(mismatchedSource, rootFetch))
      .rejects.toThrow(/GEOMETRY_STREAM_ROOT_LOAD_FAILED.*GEOMETRY_STREAM_PAGE_IDENTITY/);
    expect(rootFetch).toHaveBeenCalled();
    expect(rootFetch.mock.calls.length).toBeLessThanOrEqual(directory.rootPages.length);

    const pageId = otherIds[0], clusterId = directory.pages[pageId].clusterIds[0];
    const clusters = directory.clusters.map((entry, id) => id === clusterId
      ? { ...entry, vertexCount: entry.vertexCount + 1 } : entry);
    const pages = directory.pages.map(page => page.id === pageId
      ? { ...page, decodedBytes: page.decodedBytes + 4 + directory.vertexStride } : page);
    const cache = await GeometryPageCache.create({ ...directory, clusters, pages }, provider());
    try {
      cache.request(pageId);
      await cache.idle();
      expect(cache.get(pageId)).toBeUndefined();
      expect(cache.diagnostics().failedPages.find(failure => failure.pageId === pageId)?.message)
        .toMatch(/GEOMETRY_STREAM_CLUSTER_IDENTITY/);
      for (const root of directory.rootPages) expect(cache.get(root)).toBeDefined();
    } finally { await cache.dispose(); }
  });

  it('awaits the true active lifetime when a provider synchronously disposes the cache', async () => {
    const pageId = otherIds[0];
    let cache!: GeometryPageCache;
    let disposal: Promise<void> | undefined;
    let stateAtDispose: ReturnType<GeometryPageCache['diagnostics']> | undefined;
    const fetch: GeometryPageProvider = page => {
      if (page.id === pageId) {
        stateAtDispose = cache.diagnostics();
        disposal = cache.dispose();
      }
      return Promise.resolve(bytes(page.id));
    };
    cache = await GeometryPageCache.create(directory, fetch);
    cache.request(pageId);
    const idle = cache.idle();
    await eventually(() => disposal !== undefined);
    await Promise.all([disposal!, idle]);
    expect(stateAtDispose?.activeRequests).toBe(1);
    expect(stateAtDispose?.reservedBytes).toBe(directory.pages[pageId].decodedBytes);
    const state = cache.diagnostics();
    expect(state.disposed).toBe(true);
    expect(state.activeRequests).toBe(0);
    expect(state.reservedBytes).toBe(0);
    expect(state.inFlightBytes).toBe(0);
    expect(state.residentBytes).toBe(0);
    expect(state.residentPages).toEqual([]);
    expect(() => cache.get(pageId)).toThrow(/GEOMETRY_STREAM_DISPOSED/);
  });

  it('aborts active work on dispose and prevents a late page from becoming resident', async () => {
    const id = otherIds[0], held = heldProvider(new Set([id]));
    const cache = await GeometryPageCache.create(directory, held.fetch);
    cache.request(id);
    await eventually(() => held.waiting.has(id));
    const signal = held.waiting.get(id)!.signal;
    await cache.dispose();
    expect(signal.aborted).toBe(true);
    expect(cache.diagnostics().disposed).toBe(true);
    expect(cache.diagnostics().activeRequests).toBe(0);
    expect(cache.diagnostics().residentPages).toEqual([]);
    expect(cache.diagnostics().residentBytes).toBe(0);
    expect(() => cache.get(id)).toThrow(/GEOMETRY_STREAM_DISPOSED/);
  });
});
