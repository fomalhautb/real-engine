import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages, decodeGeometryPage } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryDecodeWorkerPool } from '../../../src/geometry-decode-worker-pool';
import { geometryPageTransferBuffers } from '../../../src/geometry-page-decode';

/** Actual Worker transfers, cancellation and reuse; no GPU readback involved. */
export async function runGeometryWorkerFixture(workerUrl?: string) {
  const hierarchy = (await GeometryDemoScene.create()).hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const page = directory.pages[directory.rootPages[0]], original = encoded.pages[page.id].bytes;
  const budget = 2 * page.encodedBytes + 2 * page.decodedBytes;
  const pool = new GeometryDecodeWorkerPool({ workers: 1, maxInFlightBytes: budget },
    workerUrl ? () => new Worker(workerUrl, { type: 'module' }) : undefined);
  const errorName = (error: unknown) => error instanceof Error ? error.name + ': ' + error.message : String(error);
  try {
    const abort = new AbortController(), queuedAbort = new AbortController();
    const first = await pool.acquire(page, abort.signal);
    const queued = pool.acquire(page, queuedAbort.signal).then(() => 'unexpected success', errorName);
    queuedAbort.abort();
    const queuedError = await queued;
    const snapshot = new Uint8Array(original);
    const pending = first.decode(snapshot).then(() => 'unexpected success', errorName);
    const detachedInput = snapshot.byteLength === 0;
    abort.abort();
    const cancelledError = await pending; first.release();
    const afterCancel = pool.diagnostics();

    const second = await pool.acquire(page, new AbortController().signal);
    const result = await second.decode(new Uint8Array(original)); second.release();
    const reference = await decodeGeometryPage(original, { maxPageBytes: page.decodedBytes, maxEncodedBytes: page.encodedBytes });
    const expected = geometryPageTransferBuffers(reference), actual = geometryPageTransferBuffers(result);
    let mismatches = 0, comparedBytes = 0;
    if (actual.length !== expected.length) throw new Error('WORKER_TRANSFER_STREAMS');
    actual.forEach((buffer, index) => {
      if (buffer.byteLength !== expected[index].byteLength) throw new Error('WORKER_TRANSFER_LENGTH');
      const a = new Uint8Array(buffer), b = new Uint8Array(expected[index]);
      for (let byte = 0; byte < a.length; byte++) { comparedBytes++; if (a[byte] !== b[byte]) mismatches++; }
    });
    const afterRecovery = pool.diagnostics();
    const third = await pool.acquire(page, new AbortController().signal);
    const disposalPending = third.decode(new Uint8Array(original)).then(() => 'unexpected success', errorName);
    const disposalQueued = pool.acquire(page, new AbortController().signal).then(() => 'unexpected success', errorName);
    pool.dispose();
    const disposalErrors = await Promise.all([disposalPending, disposalQueued]); third.release();
    return { detachedInput, providerBufferPreserved: original.byteLength === page.encodedBytes,
      queuedError, cancelledError, afterCancel, afterRecovery, afterDispose: pool.diagnostics(), disposalErrors,
      mismatches, comparedBytes, budget };
  } finally { pool.dispose(); }
}
