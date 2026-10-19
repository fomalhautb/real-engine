import type { DecodedGeometryPage } from './geometry-pages';
import type { GeometryStreamPage } from './geometry-stream-types';

export type GeometryDecodePage = Pick<GeometryStreamPage, 'id' | 'encodedBytes' | 'decodedBytes' | 'sha256'>;
export interface GeometryDecodeLease {
  /** Also cancels a pending provider fetch when the shared service is disposed. */
  readonly signal: AbortSignal;
  /** Takes ownership of this exact-sized snapshot; it may be detached on return. */
  decode(bytes: Uint8Array<ArrayBuffer>): Promise<DecodedGeometryPage>;
  release(): void;
}
export interface GeometryPageDecodeService {
  readonly mode: 'worker';
  /** Reserve before fetching, so queued requests retain metadata, not payloads. */
  acquire(page: GeometryDecodePage, signal: AbortSignal): Promise<GeometryDecodeLease>;
}
export interface GeometryDecodeWorkerOptions {
  /** Shared by all streamed batches in one Renderer. Default 2; range 1–4. */
  workers?: number;
  /** Includes provider snapshot, encoded input and decoded working/result payloads. Default 64 MiB. */
  maxInFlightBytes?: number;
  /** Waiting descriptors contain no page payloads. Default/hard maximum 4096.
   * Must accommodate root initialization concurrency; overflow rejects explicitly. */
  maxQueuedRequests?: number;
  /** Covers worker module startup, hashing and decode; excludes provider fetch. Default 30 s. */
  timeoutMs?: number;
}
type WorkerPort = Pick<Worker, 'postMessage' | 'terminate' | 'onmessage' | 'onerror' | 'onmessageerror'>;
type Ticket = {
  id: number; page: GeometryDecodePage; signal: AbortSignal; bytes: number;
  controller: AbortController;
  resolve: (lease: GeometryDecodeLease) => void; reject: (error: Error) => void; abort: () => void;
  slot?: Slot; released: boolean; started: boolean;
  job?: { resolve: (page: DecodedGeometryPage) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> };
};
type Slot = { worker?: WorkerPort; ticket?: Ticket };

function integer(value: number | undefined, fallback: number, maximum: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new Error('GEOMETRY_DECODE_OPTION: ' + label);
  return result;
}
const aborted = () => new DOMException('Geometry page decode cancelled', 'AbortError');

/** One bounded service per Renderer, with transferable data and explicit ownership.
 * A slot spans fetch through decode; slow providers therefore exert backpressure.
 * Cancelling active WASM work terminates that worker rather than pretending that
 * a message can preempt synchronous decode. The replacement starts on demand. */
export class GeometryDecodeWorkerPool implements GeometryPageDecodeService {
  readonly mode = 'worker' as const;
  readonly limits: Readonly<Required<GeometryDecodeWorkerOptions>>;
  private readonly slots: Slot[];
  private readonly waiting: Ticket[] = [];
  private sequence = 0;
  private bytes = 0;
  private peakBytes = 0;
  private disposed = false;
  private completed = 0;
  private failed = 0;
  private cancelled = 0;
  private created = 0;
  private terminated = 0;
  private cleanupFailures = 0;

  constructor(options: GeometryDecodeWorkerOptions = {},
    private readonly factory: () => WorkerPort = () => new Worker(new URL('./geometry-page.worker.ts', import.meta.url), { type: 'module' })) {
    this.limits = Object.freeze({ workers: integer(options.workers, 2, 4, 'workers'),
      maxInFlightBytes: integer(options.maxInFlightBytes, 64 * 1024 * 1024, 256 * 1024 * 1024, 'maxInFlightBytes'),
      maxQueuedRequests: integer(options.maxQueuedRequests, 4096, 4096, 'maxQueuedRequests'),
      timeoutMs: integer(options.timeoutMs, 30000, 120000, 'timeoutMs') });
    this.slots = Array.from({ length: this.limits.workers }, () => ({}));
  }

  acquire(page: GeometryDecodePage, signal: AbortSignal): Promise<GeometryDecodeLease> {
    if (this.disposed) return Promise.reject(new Error('GEOMETRY_DECODE_DISPOSED'));
    if (signal.aborted) return Promise.reject(aborted());
    if (!page || !Number.isSafeInteger(page.id) || page.id < 0
      || !Number.isSafeInteger(page.encodedBytes) || page.encodedBytes < 1 || page.encodedBytes > 16 * 1024 * 1024
      || !Number.isSafeInteger(page.decodedBytes) || page.decodedBytes < 1 || page.decodedBytes > 4 * 1024 * 1024
      || !/^[0-9a-f]{64}$/.test(page.sha256)) return Promise.reject(new Error('GEOMETRY_DECODE_PAGE'));
    const bytes = 2 * page.encodedBytes + 2 * page.decodedBytes;
    if (bytes > this.limits.maxInFlightBytes) return Promise.reject(new Error('GEOMETRY_DECODE_PAGE_BUDGET'));
    if (this.waiting.length >= this.limits.maxQueuedRequests) return Promise.reject(new Error('GEOMETRY_DECODE_QUEUE_BUDGET'));
    if (this.sequence === Number.MAX_SAFE_INTEGER) return Promise.reject(new Error('GEOMETRY_DECODE_REQUEST_IDS_EXHAUSTED'));
    return new Promise((resolve, reject) => {
      const ticket: Ticket = { id: ++this.sequence,
        page: { id: page.id, encodedBytes: page.encodedBytes, decodedBytes: page.decodedBytes, sha256: page.sha256 },
        signal, bytes, controller: new AbortController(), resolve, reject, released: false, started: false,
        abort: () => { this.cancelled++; this.release(ticket, aborted()); } };
      signal.addEventListener('abort', ticket.abort, { once: true });
      this.waiting.push(ticket); this.schedule();
    });
  }

  private schedule() {
    if (this.disposed) return;
    // Book available capacity synchronously, so the queue limit counts only
    // descriptors actually waiting for a slot or bytes, even within one turn.
    for (const slot of this.slots) {
      if (slot.ticket) continue;
      const index = this.waiting.findIndex(ticket => ticket.bytes + this.bytes <= this.limits.maxInFlightBytes);
      if (index < 0) break;
      const ticket = this.waiting.splice(index, 1)[0];
      slot.ticket = ticket; ticket.slot = slot; this.bytes += ticket.bytes;
      this.peakBytes = Math.max(this.peakBytes, this.bytes);
      ticket.resolve({ signal: ticket.controller.signal,
        decode: bytes => this.decode(ticket, bytes), release: () => this.release(ticket) });
    }
  }
  private stopWorker(slot: Slot) {
    const worker = slot.worker; if (!worker) return;
    slot.worker = undefined;
    worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null;
    try { worker.terminate(); this.terminated++; }
    catch { this.cleanupFailures++; }
  }
  private release(ticket: Ticket, reason = new Error('GEOMETRY_DECODE_LEASE_RELEASED')) {
    if (ticket.released) return;
    ticket.released = true; ticket.signal.removeEventListener('abort', ticket.abort);
    if (ticket.slot) {
      const slot = ticket.slot;
      if (ticket.job) {
        clearTimeout(ticket.job.timer); ticket.job.reject(reason); ticket.job = undefined;
        this.stopWorker(slot);
      }
      slot.ticket = undefined; ticket.slot = undefined; this.bytes -= ticket.bytes;
    } else {
      const index = this.waiting.indexOf(ticket);
      if (index >= 0) this.waiting.splice(index, 1);
      ticket.reject(reason);
    }
    ticket.controller.abort(reason);
    this.schedule();
  }
  private fail(slot: Slot, message: string) {
    const ticket = slot.ticket;
    if (ticket?.job) {
      clearTimeout(ticket.job.timer); ticket.job.reject(new Error(message)); ticket.job = undefined; this.failed++;
    }
    this.stopWorker(slot);
  }
  private decode(ticket: Ticket, bytes: Uint8Array<ArrayBuffer>): Promise<DecodedGeometryPage> {
    if (this.disposed) return Promise.reject(new Error('GEOMETRY_DECODE_DISPOSED'));
    if (ticket.released || ticket.signal.aborted) return Promise.reject(aborted());
    if (ticket.started) return Promise.reject(new Error('GEOMETRY_DECODE_LEASE_USED'));
    if (!(bytes instanceof Uint8Array) || !(bytes.buffer instanceof ArrayBuffer)
      || bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength || bytes.byteLength !== ticket.page.encodedBytes) {
      return Promise.reject(new Error('GEOMETRY_DECODE_INPUT_SNAPSHOT'));
    }
    ticket.started = true;
    const slot = ticket.slot!;
    return new Promise((resolve, reject) => {
      ticket.job = { resolve, reject, timer: setTimeout(() => this.fail(slot, 'GEOMETRY_DECODE_TIMEOUT'), this.limits.timeoutMs) };
      try {
        if (!slot.worker) {
          const worker = this.factory(); slot.worker = worker; this.created++;
          worker.onmessage = event => {
            if (slot.worker !== worker) return;
            const current = slot.ticket, response = event.data;
            if (!current?.job || !response || response.id !== current.id
              || !['decoded', 'failed'].includes(response.type)) { this.fail(slot, 'GEOMETRY_DECODE_PROTOCOL'); return; }
            const job = current.job; current.job = undefined; clearTimeout(job.timer);
            if (response.type === 'failed') { this.failed++; job.reject(new Error(String(response.error))); }
            else if (!response.page || response.page.version !== 1 || !Array.isArray(response.page.clusters)) {
              this.failed++; job.reject(new Error('GEOMETRY_DECODE_PROTOCOL')); this.stopWorker(slot);
            } else { this.completed++; job.resolve(response.page); }
          };
          worker.onerror = event => { if (slot.worker === worker) { event.preventDefault(); this.fail(slot, 'GEOMETRY_DECODE_WORKER: ' + event.message); } };
          worker.onmessageerror = () => { if (slot.worker === worker) this.fail(slot, 'GEOMETRY_DECODE_MESSAGE_ERROR'); };
        }
        slot.worker.postMessage({ type: 'decode', id: ticket.id, page: ticket.page, bytes }, [bytes.buffer]);
      } catch (error) { this.fail(slot, 'GEOMETRY_DECODE_WORKER: ' + String(error)); }
    });
  }
  diagnostics() {
    return { mode: this.mode, disposed: this.disposed, waitingRequests: this.waiting.length,
      activeRequests: this.slots.filter(slot => slot.ticket).length,
      workers: this.slots.filter(slot => slot.worker).length, inFlightBytes: this.bytes, peakInFlightBytes: this.peakBytes,
      completed: this.completed, failed: this.failed, cancelled: this.cancelled,
      workersCreated: this.created, workersTerminated: this.terminated, cleanupFailures: this.cleanupFailures, limits: { ...this.limits } };
  }
  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    const error = new Error('GEOMETRY_DECODE_DISPOSED');
    for (const ticket of [...this.waiting]) this.release(ticket, error);
    for (const slot of this.slots) {
      if (slot.ticket) this.release(slot.ticket, error);
      this.stopWorker(slot);
    }
  }
}
