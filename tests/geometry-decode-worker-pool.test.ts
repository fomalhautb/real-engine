import { afterEach, describe, expect, it, vi } from 'vitest';
import type { DecodedGeometryPage } from '../src/geometry-pages';
import { GeometryDecodeWorkerPool, type GeometryDecodePage } from '../src/geometry-decode-worker-pool';

const decoded: DecodedGeometryPage = {
  version: 1, vertexStride: 80, hasUvs: false, hasTangents: false,
  sourceVertexCount: 3, hierarchyClusterCount: 1, clusters: [],
};
const descriptor = (id: number, encodedBytes = 4, decodedBytes = 4): GeometryDecodePage =>
  ({ id, encodedBytes, decodedBytes, sha256: 'a'.repeat(64) });
const input = (length = 4) => new Uint8Array(new ArrayBuffer(length));

class FakeWorkerPort {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  readonly posts: { message: { id: number; page: GeometryDecodePage; bytes: Uint8Array }; transfer: Transferable[] }[] = [];
  readonly terminate = vi.fn();
  postMessage = vi.fn((message: { id: number; page: GeometryDecodePage; bytes: Uint8Array }, transfer: Transferable[]) => {
    this.posts.push({ message, transfer });
  });
  decoded(): void {
    const id = this.posts.at(-1)!.message.id;
    this.onmessage?.({ data: { type: 'decoded', id, page: decoded } } as MessageEvent);
  }
  fail(message: string): void {
    const id = this.posts.at(-1)!.message.id;
    this.onmessage?.({ data: { type: 'failed', id, error: message } } as MessageEvent);
  }
  error(message: string): void {
    this.onerror?.({ message, preventDefault: vi.fn() } as unknown as ErrorEvent);
  }
  messageError(): void { this.onmessageerror?.({} as MessageEvent); }
}

function harness(options: ConstructorParameters<typeof GeometryDecodeWorkerPool>[0] = {}) {
  const ports: FakeWorkerPort[] = [];
  const factory = vi.fn(() => {
    const port = new FakeWorkerPort(); ports.push(port);
    return port as unknown as Worker;
  });
  const pool = new GeometryDecodeWorkerPool(options, factory);
  return { pool, ports, factory };
}

afterEach(() => { vi.useRealTimers(); });

describe('shared geometry decode worker pool', () => {
  it('uses idle slots before applying the waiting queue limit within a synchronous burst', async () => {
    const { pool } = harness({ workers: 2, maxQueuedRequests: 1 });
    const first = pool.acquire(descriptor(0), new AbortController().signal);
    const second = pool.acquire(descriptor(1), new AbortController().signal);
    const waiting = pool.acquire(descriptor(2), new AbortController().signal);
    const leases = await Promise.all([first, second]);
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 2, waitingRequests: 1 });
    await expect(pool.acquire(descriptor(3), new AbortController().signal)).rejects.toThrow('QUEUE_BUDGET');
    leases[0].release();
    const third = await waiting;
    leases[1].release(); third.release(); pool.dispose();
  });

  it('reserves a slot and bytes at acquire, while queued requests retain metadata only', async () => {
    const { pool, ports } = harness({ workers: 1, maxInFlightBytes: 16, maxQueuedRequests: 1 });
    const first = await pool.acquire(descriptor(0), new AbortController().signal);
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 1, waitingRequests: 0, inFlightBytes: 16,
      peakInFlightBytes: 16, workers: 0 });
    const waiting = pool.acquire(descriptor(1), new AbortController().signal);
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 1, waitingRequests: 1, inFlightBytes: 16 });
    await expect(pool.acquire(descriptor(2), new AbortController().signal)).rejects.toThrow('QUEUE_BUDGET');
    expect(ports).toHaveLength(0);
    first.release();
    const second = await waiting;
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 1, waitingRequests: 0, inFlightBytes: 16 });
    second.release(); pool.dispose();
    expect(pool.diagnostics().inFlightBytes).toBe(0);
  });

  it('shares worker and byte budgets across leases, transfers one exact input, and reuses a worker', async () => {
    const { pool, ports, factory } = harness({ workers: 2, maxInFlightBytes: 32 });
    const first = await pool.acquire(descriptor(0), new AbortController().signal);
    const second = await pool.acquire(descriptor(1), new AbortController().signal);
    const thirdPending = pool.acquire(descriptor(2), new AbortController().signal);
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 2, waitingRequests: 1, inFlightBytes: 32, workers: 0 });
    const firstBytes = input(), secondBytes = input();
    const firstJob = first.decode(firstBytes), secondJob = second.decode(secondBytes);
    expect(factory).toHaveBeenCalledTimes(2);
    expect(ports[0].posts).toHaveLength(1);
    expect(ports[0].posts[0].message).toMatchObject({ id: 1, page: descriptor(0), bytes: firstBytes });
    expect(ports[0].posts[0].transfer).toEqual([firstBytes.buffer]);
    expect(ports[1].posts[0].transfer).toEqual([secondBytes.buffer]);
    await expect(first.decode(input())).rejects.toThrow('LEASE_USED');
    ports[0].decoded(); ports[1].decoded();
    await expect(firstJob).resolves.toBe(decoded);
    await expect(secondJob).resolves.toBe(decoded);
    first.release();
    const third = await thirdPending;
    const thirdJob = third.decode(input());
    expect(factory).toHaveBeenCalledTimes(2);
    expect(ports[0].posts).toHaveLength(2);
    ports[0].decoded(); await expect(thirdJob).resolves.toBe(decoded);
    second.release(); third.release(); pool.dispose();
    expect(pool.diagnostics()).toMatchObject({ inFlightBytes: 0, completed: 3, workersCreated: 2,
      workersTerminated: 2 });
  });

  it('aborts a queued descriptor without touching the active lease', async () => {
    const { pool } = harness({ workers: 1 });
    const active = await pool.acquire(descriptor(0), new AbortController().signal);
    const controller = new AbortController();
    const waiting = pool.acquire(descriptor(1), controller.signal);
    controller.abort();
    await expect(waiting).rejects.toMatchObject({ name: 'AbortError' });
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 1, waitingRequests: 0, cancelled: 1,
      inFlightBytes: 16 });
    active.release(); pool.dispose();
  });

  it('terminates an active decode on abort and starts a replacement worker', async () => {
    const { pool, ports } = harness({ workers: 1 });
    const controller = new AbortController();
    const lease = await pool.acquire(descriptor(0), controller.signal);
    const job = lease.decode(input());
    controller.abort();
    await expect(job).rejects.toMatchObject({ name: 'AbortError' });
    expect(ports[0].terminate).toHaveBeenCalledOnce();
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 0, inFlightBytes: 0, cancelled: 1 });
    const next = await pool.acquire(descriptor(1), new AbortController().signal);
    const nextJob = next.decode(input());
    expect(ports).toHaveLength(2);
    ports[1].decoded(); await expect(nextJob).resolves.toBe(decoded);
    next.release(); pool.dispose();
  });

  it.each(['error', 'messageerror', 'failed'] as const)('settles a %s response and recovers after release', async kind => {
    const { pool, ports } = harness({ workers: 1 });
    const lease = await pool.acquire(descriptor(0), new AbortController().signal);
    const job = lease.decode(input());
    if (kind === 'error') ports[0].error('FAKE_WORKER_ERROR');
    else if (kind === 'messageerror') ports[0].messageError();
    else ports[0].fail('FAKE_DECODE_FAILURE');
    await expect(job).rejects.toThrow(kind === 'error' ? 'FAKE_WORKER_ERROR'
      : kind === 'messageerror' ? 'MESSAGE_ERROR' : 'FAKE_DECODE_FAILURE');
    lease.release();
    const next = await pool.acquire(descriptor(1), new AbortController().signal);
    const nextJob = next.decode(input());
    ports.at(-1)!.decoded(); await expect(nextJob).resolves.toBe(decoded);
    next.release(); pool.dispose();
    expect(pool.diagnostics()).toMatchObject({ activeRequests: 0, inFlightBytes: 0, failed: 1, completed: 1 });
  });

  it('times out a decode, terminates the worker and recovers', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { pool, ports } = harness({ workers: 1, timeoutMs: 5 });
    const lease = await pool.acquire(descriptor(0), new AbortController().signal);
    const job = lease.decode(input());
    const timedOut = expect(job).rejects.toThrow('GEOMETRY_DECODE_TIMEOUT');
    await vi.advanceTimersByTimeAsync(5);
    await timedOut;
    expect(ports[0].terminate).toHaveBeenCalledOnce();
    lease.release();
    const next = await pool.acquire(descriptor(1), new AbortController().signal);
    const nextJob = next.decode(input()); ports[1].decoded();
    await expect(nextJob).resolves.toBe(decoded);
    next.release(); pool.dispose();
  });

  it('handles a throwing worker factory and lets a later decode succeed', async () => {
    const port = new FakeWorkerPort();
    let attempts = 0;
    const pool = new GeometryDecodeWorkerPool({ workers: 1 }, () => {
      if (++attempts === 1) throw new Error('FAKE_FACTORY_FAILURE');
      return port as unknown as Worker;
    });
    const first = await pool.acquire(descriptor(0), new AbortController().signal);
    await expect(first.decode(input())).rejects.toThrow('FAKE_FACTORY_FAILURE');
    first.release();
    const second = await pool.acquire(descriptor(1), new AbortController().signal);
    const job = second.decode(input()); port.decoded();
    await expect(job).resolves.toBe(decoded);
    second.release(); pool.dispose();
    expect(pool.diagnostics()).toMatchObject({ workersCreated: 1, failed: 1, completed: 1,
      activeRequests: 0, inFlightBytes: 0 });
  });

  it('disposes queued and running work, rejects both, and clears all counters', async () => {
    const { pool, ports } = harness({ workers: 1 });
    const active = await pool.acquire(descriptor(0), new AbortController().signal);
    const job = active.decode(input());
    const waiting = pool.acquire(descriptor(1), new AbortController().signal);
    pool.dispose();
    await expect(job).rejects.toThrow('GEOMETRY_DECODE_DISPOSED');
    await expect(waiting).rejects.toThrow('GEOMETRY_DECODE_DISPOSED');
    await expect(pool.acquire(descriptor(2), new AbortController().signal)).rejects.toThrow('GEOMETRY_DECODE_DISPOSED');
    await expect(active.decode(input())).rejects.toThrow('GEOMETRY_DECODE_DISPOSED');
    expect(ports[0].terminate).toHaveBeenCalledOnce();
    expect(pool.diagnostics()).toMatchObject({ disposed: true, activeRequests: 0, waitingRequests: 0,
      workers: 0, inFlightBytes: 0 });
    expect(() => pool.dispose()).not.toThrow();
  });

  it('rejects malformed options, descriptors and input views without consuming a lease', async () => {
    for (const options of [{ workers: 0 }, { workers: 5 }, { maxInFlightBytes: 0 },
      { maxQueuedRequests: 4097 }, { timeoutMs: 0 }]) {
      expect(() => new GeometryDecodeWorkerPool(options)).toThrow('GEOMETRY_DECODE_OPTION');
    }
    const { pool, ports } = harness({ workers: 1, maxInFlightBytes: 16 });
    for (const page of [{ ...descriptor(0), id: -1 }, { ...descriptor(0), encodedBytes: 0 },
      { ...descriptor(0), decodedBytes: 0 }, { ...descriptor(0), sha256: 'bad' }]) {
      await expect(pool.acquire(page, new AbortController().signal)).rejects.toThrow('GEOMETRY_DECODE_PAGE');
    }
    await expect(pool.acquire(descriptor(0, 5), new AbortController().signal)).rejects.toThrow('PAGE_BUDGET');
    const mutable = { ...descriptor(7) } as { -readonly [K in keyof GeometryDecodePage]: GeometryDecodePage[K] };
    const lease = await pool.acquire(mutable, new AbortController().signal);
    mutable.id = 99; mutable.sha256 = 'b'.repeat(64);
    await expect(lease.decode(new Uint8Array(new ArrayBuffer(8), 2, 4) as Uint8Array<ArrayBuffer>))
      .rejects.toThrow('INPUT_SNAPSHOT');
    await expect(lease.decode(input(3))).rejects.toThrow('INPUT_SNAPSHOT');
    const job = lease.decode(input());
    expect(ports[0].posts[0].message.page).toEqual(descriptor(7));
    ports[0].decoded(); await expect(job).resolves.toBe(decoded);
    lease.release(); pool.dispose();
  });

  it('cleans other workers and settles requests even when a worker termination throws', async () => {
    const { pool, ports } = harness({ workers: 2 });
    const first = await pool.acquire(descriptor(0), new AbortController().signal);
    const second = await pool.acquire(descriptor(1), new AbortController().signal);
    const a = first.decode(input()).catch(error => error.message);
    const b = second.decode(input()).catch(error => error.message);
    ports[0].terminate.mockImplementation(() => { throw new Error('FAKE_TERMINATE_FAILURE'); });
    expect(() => pool.dispose()).not.toThrow();
    expect(await Promise.all([a, b])).toEqual(['GEOMETRY_DECODE_DISPOSED', 'GEOMETRY_DECODE_DISPOSED']);
    expect(ports[1].terminate).toHaveBeenCalledOnce();
    expect(pool.diagnostics()).toMatchObject({ workers: 0, activeRequests: 0, inFlightBytes: 0, cleanupFailures: 1 });
  });
});
