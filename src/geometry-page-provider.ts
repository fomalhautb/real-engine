import type { GeometryPageProvider } from './geometry-stream-types';

const MAX_PAGES = 65_536;
const MAX_ENCODED_BYTES = 8 * 1024 * 1024 + 40;
const MAX_READS = 65_536;

export interface HttpGeometryPageProviderOptions {
  fetch?: typeof fetch;
  /** Per-page wire byte limit; the hard ceiling is 8 MiB plus the 40-byte page header. */
  maxEncodedBytes?: number;
}

function fail(reason: string): never { throw new Error(`GEOMETRY_PAGE_PROVIDER_${reason}`); }

function httpUrl(value: string | URL, base?: URL): URL {
  let url: URL;
  try { url = new URL(value, base); }
  catch { return fail('URL'); }
  if ((url.protocol !== 'http:' && url.protocol !== 'https:') || url.username || url.password) fail('URL');
  return url;
}

/** Wait for a stream read while retaining an abort path for virtual or stalled streams. */
function readWithAbort(reader: ReadableStreamDefaultReader<Uint8Array>, signal: AbortSignal): Promise<ReadableStreamReadResult<Uint8Array>> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const aborted = () => reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
    signal.addEventListener('abort', aborted, { once: true });
    reader.read().then(resolve, reject).finally(() => signal.removeEventListener('abort', aborted));
  });
}

/** Fetches encoded pages by directory ID. Hash, decoding, and topology remain cache responsibilities. */
export function createHttpGeometryPageProvider(
  baseUrl: string | URL, pagePaths: readonly string[], options: HttpGeometryPageProviderOptions = {},
): GeometryPageProvider {
  const base = httpUrl(baseUrl);
  if (!Array.isArray(pagePaths) || pagePaths.length < 1 || pagePaths.length > MAX_PAGES) fail('PATHS');
  const paths = Array.from(pagePaths, path => {
    if (typeof path !== 'string' || !path.trim()) fail('PATH');
    return httpUrl(path, base).href;
  });
  const maxEncodedBytes = options.maxEncodedBytes ?? MAX_ENCODED_BYTES;
  if (!Number.isSafeInteger(maxEncodedBytes) || maxEncodedBytes < 1 || maxEncodedBytes > MAX_ENCODED_BYTES) fail('BUDGET');
  if (options.fetch !== undefined && typeof options.fetch !== 'function') fail('FETCH');
  const fetcher = options.fetch ?? fetch;

  return async (page, signal) => {
    if (!page || !Number.isSafeInteger(page.id) || page.id < 0 || page.id >= paths.length) fail('PAGE_ID');
    const expected = page.encodedBytes;
    if (!Number.isSafeInteger(expected) || expected < 1 || expected > maxEncodedBytes) fail('PAGE_BYTES');
    signal.throwIfAborted();
    const response = await fetcher(new URL(paths[page.id]), { credentials: 'omit', signal });
    try {
      signal.throwIfAborted();
      if (!response.ok) fail(`HTTP_${response.status}`);
      const length = response.headers.get('content-length');
      const encoding = response.headers.get('content-encoding')?.trim().toLowerCase();
      // Fetch exposes the decoded body, but Content-Length describes the HTTP
      // representation. CORS may also hide Content-Encoding. Only compare a
      // known identity response; every body still has the exact asset-byte cap.
      const identity = encoding === 'identity' || !encoding && response.type !== 'cors';
      if (length !== null && (!/^\d+$/.test(length) || !Number.isSafeInteger(Number(length))
        || identity && Number(length) !== expected)) fail('CONTENT_LENGTH');
      if (!response.body) fail('BODY');
    } catch (error) {
      // Cancellation is requested, but a custom stream's cancellation promise
      // must not be able to prevent abort/error propagation and cache disposal.
      void response.body?.cancel().catch(() => {});
      throw error;
    }
    const reader = response.body!.getReader();
    let total = 0, reads = 0;
    try {
      const result = new Uint8Array(expected);
      for (;;) {
        if (++reads > MAX_READS) fail('READ_BUDGET');
        signal.throwIfAborted();
        const { done, value } = await readWithAbort(reader, signal);
        signal.throwIfAborted();
        if (done) break;
        if (!(value instanceof Uint8Array) || value.byteLength === 0 || value.byteLength > expected - total) fail('BODY_LENGTH');
        result.set(value, total);
        total += value.byteLength;
      }
      if (total !== expected) fail('BODY_LENGTH');
      signal.throwIfAborted();
      return result;
    } catch (error) {
      void reader.cancel().catch(() => {});
      throw error;
    } finally { reader.releaseLock(); }
  };
}
