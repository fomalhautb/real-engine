// Bounded worker_threads pool for offline BC7 encoding. Pixels are shared with
// workers through a SharedArrayBuffer copy per level (read-only by contract);
// block rows are split into fixed-size chunks and reassembled in order, so the
// output is identical for any worker count or completion order.
import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';

const MAX_WORKERS = 64;

export function defaultWorkerCount() {
  return Math.max(1, Math.min(8, availableParallelism() - 2));
}

export class TextureCodecPool {
  #workers = [];
  #idle = [];
  #queue = [];
  #pending = new Map();
  #nextId = 1;
  #closed = false;

  constructor(workers = defaultWorkerCount()) {
    if (!Number.isSafeInteger(workers) || workers < 1 || workers > MAX_WORKERS) throw Error('TEXTURE_CODEC_WORKERS');
    for (let i = 0; i < workers; i++) {
      const worker = new Worker(new URL('./texture-codec-worker.mjs', import.meta.url));
      worker.on('message', message => this.#settle(worker, message));
      worker.on('error', error => this.#fail(worker, error));
      worker.on('exit', code => { if (!this.#closed && code !== 0) this.#fail(worker, Error('TEXTURE_CODEC_WORKER_EXIT:' + code)); });
      this.#workers.push(worker);
      this.#idle.push(worker);
    }
  }

  get size() { return this.#workers.length; }

  #dispatch() {
    while (this.#idle.length && this.#queue.length) {
      const worker = this.#idle.pop(), task = this.#queue.shift();
      task.worker = worker;
      this.#pending.set(task.message.id, task);
      worker.postMessage(task.message);
    }
  }

  #settle(worker, message) {
    const task = this.#pending.get(message.id);
    if (!task) return;
    this.#pending.delete(message.id);
    this.#idle.push(worker);
    if (message.error) task.reject(Error('TEXTURE_CODEC_TASK: ' + message.error));
    else task.resolve(message.blocks);
    this.#dispatch();
  }

  #fail(worker, error) {
    for (const [id, task] of this.#pending) if (task.worker === worker) { this.#pending.delete(id); task.reject(error); }
  }

  #run(message) {
    if (this.#closed) return Promise.reject(Error('TEXTURE_CODEC_POOL_CLOSED'));
    return new Promise((resolve, reject) => {
      this.#queue.push({ message: { ...message, id: this.#nextId++ }, resolve, reject });
      this.#dispatch();
    });
  }

  /** Encodes one RGBA8 level to BC7, returning tightly packed blocks. */
  async encodeBc7(pixels, width, height, options = {}, rowsPerTask = 16) {
    if (!(pixels instanceof Uint8Array) || pixels.byteLength !== width * height * 4) throw Error('TEXTURE_CODEC_PIXELS');
    const shared = new SharedArrayBuffer(pixels.byteLength);
    new Uint8Array(shared).set(pixels);
    const rows = Math.ceil(height / 4), bw = Math.ceil(width / 4);
    const jobs = [];
    for (let rowStart = 0; rowStart < rows; rowStart += rowsPerTask) {
      const rowEnd = Math.min(rows, rowStart + rowsPerTask);
      jobs.push(this.#run({ pixels: shared, width, height, rowStart, rowEnd, options }).then(blocks => ({ rowStart, blocks })));
    }
    const out = new Uint8Array(rows * bw * 16);
    for (const { rowStart, blocks } of await Promise.all(jobs)) out.set(blocks, rowStart * bw * 16);
    return out;
  }

  async close() {
    this.#closed = true;
    for (const task of this.#queue) task.reject(Error('TEXTURE_CODEC_POOL_CLOSED'));
    this.#queue.length = 0;
    await Promise.all(this.#workers.map(worker => worker.terminate()));
  }
}
