import { createHash } from 'node:crypto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createHttpTextureMipProvider, validateTextureStreamSource,
  verifyTextureStreamFallback, verifyTextureStreamMip,
} from '../src/texture-stream-source';
import type { TextureStreamMip, TextureStreamSource } from '../src/texture-stream-types';
import type { PreparedTextureAsset, PreparedTextureFormat } from '../src/texture-storage';

const digest = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');
const signal = () => new AbortController().signal;

function fixture(width = 7, height = 3, format: PreparedTextureFormat = 'rgba8', baseMipLevel = 1): TextureStreamSource {
  const mips: TextureStreamMip[] = [];
  const levels: PreparedTextureAsset['levels'][number][] = [];
  for (let level = 0; ; level++) {
    const w = Math.max(1, Math.floor(width / 2 ** level));
    const h = Math.max(1, Math.floor(height / 2 ** level));
    const bytes = format === 'rgba8' ? w * h * 4 : Math.ceil(w / 4) * Math.ceil(h / 4) * 16;
    const data = new Uint8Array(bytes).fill(level + 1);
    mips.push({ level, width: w, height: h, byteLength: bytes, sha256: digest(data) });
    if (level >= baseMipLevel) levels.push({ width: w, height: h, data });
    if (w === 1 && h === 1) break;
  }
  const fallback: PreparedTextureAsset = {
    kind: 'prepared', width, height, format, colorSpace: 'linear', baseMipLevel, levels,
  };
  return { id: 'terrain/albedo', width, height, format, colorSpace: 'linear', mips,
    fallback, provider: async mip => new Uint8Array(mip.byteLength).fill(mip.level + 1) };
}

afterEach(() => vi.unstubAllGlobals());

describe('texture stream source', () => {
  it('accepts a NPOT native chain and preserves fallback identity while freezing metadata', async () => {
    const input = fixture();
    const source = validateTextureStreamSource(input);
    expect(source).not.toBe(input);
    expect(source.fallback).toBe(input.fallback);
    expect(source.provider).toBe(input.provider);
    expect(source.mips.map(mip => [mip.width, mip.height, mip.byteLength])).toEqual([
      [7, 3, 84], [3, 1, 12], [1, 1, 4],
    ]);
    expect(Object.isFrozen(source)).toBe(true);
    expect(Object.isFrozen(source.mips)).toBe(true);
    expect(Object.isFrozen(source.mips[0])).toBe(true);
    await expect(verifyTextureStreamFallback(source, signal())).resolves.toBeUndefined();
    const fetched = await source.provider(source.mips[0], signal());
    await expect(verifyTextureStreamMip(source.mips[0], fetched, signal())).resolves.toBe(fetched);
  });

  it('checks compressed block padding and rejects malformed manifests', () => {
    const valid = fixture(8, 8, 'bc7', 1);
    expect(validateTextureStreamSource(valid).mips.map(mip => mip.byteLength)).toEqual([64, 16, 16, 16]);
    expect(() => validateTextureStreamSource({ ...valid, mips: valid.mips.slice(1) })).toThrow(/complete chain/);
    expect(() => validateTextureStreamSource({ ...valid, mips: valid.mips.map((mip, i) => i === 2
      ? { ...mip, byteLength: 15 } : mip) })).toThrow(/native mip 2/);
    expect(() => validateTextureStreamSource({ ...valid, width: 7 })).toThrow(/divisible by four/);
    expect(() => validateTextureStreamSource({ ...valid, id: 'x'.repeat(129) })).toThrow(/id/);
    expect(() => validateTextureStreamSource({ ...valid, fallback: { ...valid.fallback, format: 'rgba8' } }))
      .toThrow();
  });

  it('rejects corrupt hashes, shared backing allocations and aborts', async () => {
    const source = validateTextureStreamSource(fixture());
    const mip = source.mips[0];
    await expect(verifyTextureStreamMip(mip, new Uint8Array(mip.byteLength), signal())).rejects.toThrow(/SHA-256/);
    const backing = new Uint8Array(mip.byteLength + 1);
    await expect(verifyTextureStreamMip(mip, backing.subarray(0, mip.byteLength), signal()))
      .rejects.toThrow(/owned/);
    source.fallback.levels[0].data[0] ^= 1;
    await expect(verifyTextureStreamFallback(source, signal())).rejects.toThrow(/SHA-256/);
    const controller = new AbortController();
    controller.abort();
    await expect(verifyTextureStreamMip(mip, new Uint8Array(mip.byteLength), controller.signal))
      .rejects.toMatchObject({ name: 'AbortError' });
  });

  it('bounds HTTP reads and cancels oversized or short bodies', async () => {
    const cancels: number[] = [];
    const response = (chunks: Uint8Array[], close = true) => ({
      ok: true, status: 200,
      body: new ReadableStream<Uint8Array>({
        start(controller) { for (const chunk of chunks) controller.enqueue(chunk); if (close) controller.close(); },
        cancel() { cancels.push(1); },
      }),
    }) as Response;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response([new Uint8Array(5)], false))
      .mockResolvedValueOnce(response([new Uint8Array(2)]))
      .mockResolvedValueOnce(response([new Uint8Array(4)]));
    vi.stubGlobal('fetch', fetchMock);
    const urls = ['a'];
    const provider = createHttpTextureMipProvider(urls);
    urls[0] = 'changed';
    const mip = { level: 0, width: 1, height: 1, byteLength: 4, sha256: digest(new Uint8Array(4)) };
    await expect(provider(mip, signal())).rejects.toThrow(/oversized/);
    await expect(provider(mip, signal())).rejects.toThrow(/short/);
    await expect(provider(mip, signal())).resolves.toEqual(new Uint8Array(4));
    expect(fetchMock.mock.calls.map(call => call[0])).toEqual(['a', 'a', 'a']);
    expect(cancels).toHaveLength(1); // Closing a short body leaves no open stream to cancel.
  });

  it('rejects failed HTTP responses and forwards abort signals', async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404,
      body: { cancel } }));
    const provider = createHttpTextureMipProvider(['a']);
    const controller = new AbortController();
    const mip = fixture(1, 1, 'rgba8', 0).mips[0];
    await expect(provider(mip, controller.signal)).rejects.toThrow(/HTTP 404/);
    expect(cancel).toHaveBeenCalledOnce();
    expect(vi.mocked(fetch).mock.calls[0][1]).toEqual({ signal: controller.signal });
  });

  it('cancels a pending body read when aborted', async () => {
    let resolveRead!: (value: ReadableStreamReadResult<Uint8Array>) => void;
    let started!: () => void;
    const reading = new Promise<void>(resolve => { started = resolve; });
    const reader = {
      read: vi.fn(() => { started(); return new Promise<ReadableStreamReadResult<Uint8Array>>(
        resolve => { resolveRead = resolve; }); }),
      cancel: vi.fn(async () => { resolveRead({ done: true, value: undefined }); }),
      releaseLock: vi.fn(),
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200,
      body: { getReader: () => reader } }));
    const controller = new AbortController();
    const provider = createHttpTextureMipProvider(['a']);
    const pending = provider(fixture(1, 1, 'rgba8', 0).mips[0], controller.signal);
    await reading;
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(reader.cancel).toHaveBeenCalled();
    expect(reader.releaseLock).toHaveBeenCalledOnce();
  });
});
