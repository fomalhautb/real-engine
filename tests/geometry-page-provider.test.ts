import { describe, expect, it, vi } from 'vitest';
import { createHttpGeometryPageProvider } from '../src/geometry-page-provider';
import type { GeometryStreamPage } from '../src/geometry-stream-types';

const page = (id: number, encodedBytes: number): GeometryStreamPage => ({
  id, encodedBytes, clusterIds: [], decodedBytes: 1, sha256: '',
});
const base = 'https://assets.example.test/scenes/one/';

function streamed(chunks: Uint8Array[], headers: HeadersInit = {}, cancel = vi.fn()): Response {
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) { for (const chunk of chunks) controller.enqueue(chunk); controller.close(); },
    cancel,
  }), { headers });
}

describe('HTTP geometry page provider', () => {
  it('resolves snapshotted relative paths and returns an exact streamed body with omitted credentials', async () => {
    const paths = ['pages/zero.gpg'];
    const fetcher = vi.fn(async (_input: URL, _init: RequestInit) => streamed(
      [new Uint8Array([1, 2]), new Uint8Array([3, 4])], { 'Content-Length': '4' },
    ));
    const provider = createHttpGeometryPageProvider(base, paths, { fetch: fetcher as typeof fetch });
    paths[0] = 'other.gpg';
    const signal = new AbortController().signal;
    expect(await provider(page(0, 4), signal)).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(fetcher).toHaveBeenCalledOnce();
    const [url, request] = fetcher.mock.calls[0];
    expect(url.href).toBe('https://assets.example.test/scenes/one/pages/zero.gpg');
    expect(request).toEqual({ credentials: 'omit', signal });
  });

  it('accepts an exact streamed body without Content-Length', async () => {
    const provider = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => streamed([new Uint8Array([7]), new Uint8Array([8, 9])]),
    });
    expect(await provider(page(0, 3), new AbortController().signal)).toEqual(new Uint8Array([7, 8, 9]));
  });

  it('bounds the decoded asset body when HTTP compression or CORS makes Content-Length incomparable', async () => {
    for (const encoding of ['gzip', 'br', undefined]) {
      const headers: Record<string, string> = { 'Content-Length': '27' };
      if (encoding) headers['Content-Encoding'] = encoding;
      const response = streamed([new Uint8Array([7, 8, 9])], headers);
      if (!encoding) Object.defineProperty(response, 'type', { value: 'cors' });
      const provider = createHttpGeometryPageProvider(base, ['page.gpg'], { fetch: async () => response });
      expect(await provider(page(0, 3), new AbortController().signal)).toEqual(new Uint8Array([7, 8, 9]));
    }
    for (const body of [new Uint8Array(2), new Uint8Array(4)]) {
      const provider = createHttpGeometryPageProvider(base, ['page.gpg'], {
        fetch: async () => streamed([body], { 'Content-Length': '27', 'Content-Encoding': 'gzip' }),
      });
      await expect(provider(page(0, 3), new AbortController().signal)).rejects.toThrow(/BODY_LENGTH/);
    }
  });

  it('cancels before reading a mismatched or malformed Content-Length', async () => {
    for (const length of ['5', 'garbage', '-1', '9007199254740992']) {
      const cancel = vi.fn(), pull = vi.fn();
      const fetcher = vi.fn(async () => new Response(new ReadableStream<Uint8Array>({ pull, cancel }, { highWaterMark: 0 }),
        { headers: { 'Content-Length': length } }));
      const provider = createHttpGeometryPageProvider(base, ['page.gpg'], { fetch: fetcher as typeof fetch });
      await expect(provider(page(0, 4), new AbortController().signal)).rejects.toThrow(/CONTENT_LENGTH/);
      expect(cancel).toHaveBeenCalledOnce();
      expect(pull).not.toHaveBeenCalled();
      expect(fetcher).toHaveBeenCalledOnce();
    }
  });

  it('cancels streaming overflow and rejects a short body', async () => {
    const overflowCancel = vi.fn();
    const overflowChunks = [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5])];
    const overflow = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => new Response(new ReadableStream<Uint8Array>({
        pull(controller) { controller.enqueue(overflowChunks.shift()!); }, cancel: overflowCancel,
      }, { highWaterMark: 0 })),
    });
    await expect(overflow(page(0, 4), new AbortController().signal)).rejects.toThrow(/BODY_LENGTH/);
    expect(overflowCancel).toHaveBeenCalledOnce();

    const short = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => streamed([new Uint8Array([1, 2])]),
    });
    await expect(short(page(0, 4), new AbortController().signal)).rejects.toThrow(/BODY_LENGTH/);
  });

  it('rejects empty chunks so a stream cannot consume unbounded reads without byte progress', async () => {
    const cancel = vi.fn(), pull = vi.fn((controller: ReadableStreamDefaultController<Uint8Array>) => {
      controller.enqueue(new Uint8Array(0));
    });
    const provider = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => new Response(new ReadableStream<Uint8Array>({ pull, cancel }, { highWaterMark: 0 })),
    });
    await expect(provider(page(0, 1), new AbortController().signal)).rejects.toThrow(/BODY_LENGTH/);
    expect(pull).toHaveBeenCalledOnce();
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('rejects HTTP errors and cancels their bodies without retrying', async () => {
    const cancel = vi.fn(), fetcher = vi.fn(async () => new Response(
      new ReadableStream<Uint8Array>({ cancel }, { highWaterMark: 0 }), { status: 404 },
    ));
    const provider = createHttpGeometryPageProvider(base, ['page.gpg'], { fetch: fetcher as typeof fetch });
    await expect(provider(page(0, 4), new AbortController().signal)).rejects.toThrow(/HTTP_404/);
    expect(cancel).toHaveBeenCalledOnce();
    expect(fetcher).toHaveBeenCalledOnce();
  });

  it('checks abort before fetch and while a stream read is pending', async () => {
    const fetcher = vi.fn(async () => streamed([new Uint8Array([1])]));
    const provider = createHttpGeometryPageProvider(base, ['page.gpg'], { fetch: fetcher as typeof fetch });
    const initial = new AbortController();
    initial.abort();
    await expect(provider(page(0, 1), initial.signal)).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetcher).not.toHaveBeenCalled();

    const controller = new AbortController(), cancel = vi.fn();
    let reading!: () => void;
    const readStarted = new Promise<void>(resolve => { reading = resolve; });
    const stalled = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => new Response(new ReadableStream<Uint8Array>({ pull: reading, cancel }, { highWaterMark: 0 })),
    });
    const pending = stalled(page(0, 1), controller.signal);
    await readStarted;
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('rejects malformed URLs, path metadata, budgets and page descriptors before fetch', async () => {
    expect(() => createHttpGeometryPageProvider('file:///asset/', ['p'])).toThrow(/URL/);
    expect(() => createHttpGeometryPageProvider('https://user:password@example.test/', ['p'])).toThrow(/URL/);
    expect(() => createHttpGeometryPageProvider(base, ['javascript:alert(1)'])).toThrow(/URL/);
    expect(() => createHttpGeometryPageProvider(base, ['https://user@example.test/p'])).toThrow(/URL/);
    expect(() => createHttpGeometryPageProvider(base, [])).toThrow(/PATHS/);
    expect(() => createHttpGeometryPageProvider(base, new Array(65_537).fill('p'))).toThrow(/PATHS/);
    expect(() => createHttpGeometryPageProvider(base, new Array<string>(1))).toThrow(/PATH/);
    expect(() => createHttpGeometryPageProvider(base, [' '])).toThrow(/PATH/);
    for (const limit of [0, 1.5, Infinity, 8 * 1024 * 1024 + 41]) {
      expect(() => createHttpGeometryPageProvider(base, ['p'], { maxEncodedBytes: limit })).toThrow(/BUDGET/);
    }
    const fetcher = vi.fn(async () => streamed([new Uint8Array([1])]));
    const provider = createHttpGeometryPageProvider(base, ['p'], { fetch: fetcher as typeof fetch, maxEncodedBytes: 3 });
    for (const descriptor of [page(-1, 1), page(1, 1), page(0.5, 1), page(0, 0), page(0, 1.5), page(0, 4)]) {
      await expect(provider(descriptor, new AbortController().signal)).rejects.toThrow(/PAGE_ID|PAGE_BYTES/);
    }
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('propagates abort and header errors even when custom stream cancellation never settles', async () => {
    const controller = new AbortController();
    const cancel = vi.fn(() => new Promise<void>(() => {}));
    let reading!: () => void;
    const readStarted = new Promise<void>(resolve => { reading = resolve; });
    const stalled = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => new Response(new ReadableStream<Uint8Array>({ pull: reading, cancel }, { highWaterMark: 0 })),
    });
    const pending = stalled(page(0, 1), controller.signal);
    await readStarted;
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancel).toHaveBeenCalledOnce();

    const invalidHeader = createHttpGeometryPageProvider(base, ['page.gpg'], {
      fetch: async () => new Response(new ReadableStream<Uint8Array>({ cancel }, { highWaterMark: 0 }),
        { headers: { 'Content-Length': '2' } }),
    });
    await expect(invalidHeader(page(0, 1), new AbortController().signal)).rejects.toThrow(/CONTENT_LENGTH/);
    expect(cancel).toHaveBeenCalledTimes(2);
  });
});
