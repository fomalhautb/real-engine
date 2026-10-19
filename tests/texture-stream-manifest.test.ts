import { createHash } from 'node:crypto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadTextureStreamSource, validateTextureStreamManifest } from '../src/texture-stream-manifest';
import type { PreparedTextureFormat } from '../src/texture-storage';
import { verifyTextureStreamMip } from '../src/texture-stream-source';

const hash = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');
const origin = 'https://assets.example/game/pack/';

function fixture(width = 8, height = 8, format: PreparedTextureFormat = 'rgba8', fallbackMip = 2) {
  const data: Uint8Array[] = [];
  const mips = [];
  for (let level = 0; ; level++) {
    const w = Math.max(1, Math.floor(width / 2 ** level));
    const h = Math.max(1, Math.floor(height / 2 ** level));
    const bytes = format === 'rgba8' ? w * h * 4 : Math.ceil(w / 4) * Math.ceil(h / 4) * 16;
    const payload = new Uint8Array(bytes).fill(level + 1);
    data.push(payload);
    mips.push({ level, width: w, height: h, byteLength: bytes, sha256: hash(payload), file: `tiles/mip-${level}.bin` });
    if (w === 1 && h === 1) break;
  }
  return { manifest: { version: 1, id: 'terrain/albedo', width, height, colorSpace: 'linear',
    format, fallbackMip, mips }, data };
}

function streamed(bytes: Uint8Array, url: string, chunks = [bytes], declared?: string, close = true) {
  const cancel = vi.fn();
  const body = new ReadableStream<Uint8Array>({
    start(controller) { for (const chunk of chunks) controller.enqueue(chunk); if (close) controller.close(); },
    cancel,
  });
  return { response: { ok: true, status: 200, body, url,
    headers: new Headers(declared === undefined ? {} : { 'content-length': declared }) } as Response, cancel };
}

function mockPack(manifest: object, data: Uint8Array[]) {
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url === `${origin}manifest.json`) return streamed(manifestBytes, url).response;
    const match = /mip-(\d+)\.bin$/.exec(url);
    if (!match) throw new Error(`unexpected URL ${url}`);
    return streamed(data[Number(match[1])], url).response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe('texture stream manifest', () => {
  it('owns canonical metadata and checks RGBA and compressed block chains', () => {
    for (const format of ['rgba8', 'bc7', 'etc2-rgba8', 'astc-4x4'] as const) {
      const { manifest } = fixture(8, 8, format, format === 'rgba8' ? 2 : 1);
      const checked = validateTextureStreamManifest(manifest);
      expect(checked).not.toBe(manifest);
      expect(Object.isFrozen(checked)).toBe(true);
      expect(Object.isFrozen(checked.mips)).toBe(true);
      expect(Object.isFrozen(checked.mips[0])).toBe(true);
      expect(checked.mips.at(-1)).toMatchObject({ width: 1, height: 1, byteLength: format === 'rgba8' ? 4 : 16 });
      manifest.mips[0].file = 'changed.bin';
      expect(checked.mips[0].file).toBe('tiles/mip-0.bin');
    }
  });

  it('rejects malformed metadata, unsafe files, and invalid compressed fallback', () => {
    const { manifest } = fixture();
    for (const patch of [{ version: 2 }, { id: '' }, { width: 8193 }, { colorSpace: 'wide' },
      { format: 'png' }, { fallbackMip: 9 }, { mips: manifest.mips.slice(1) }]) {
      expect(() => validateTextureStreamManifest({ ...manifest, ...patch })).toThrow();
    }
    for (const file of ['', '../x', 'a/../x', 'a//x', '/x', 'https://evil/x', 'x?y', 'x#y',
      'a%2fb', 'a\\b', './x', 'a/.']) {
      expect(() => validateTextureStreamManifest({ ...manifest, mips: [{ ...manifest.mips[0], file }, ...manifest.mips.slice(1)] }))
        .toThrow(/mip 0/);
    }
    expect(() => validateTextureStreamManifest({ ...manifest,
      mips: [{ ...manifest.mips[0], sha256: 'bad' }, ...manifest.mips.slice(1)] })).toThrow(/mip 0/);
    expect(() => validateTextureStreamManifest({ ...manifest,
      mips: [{ ...manifest.mips[0], byteLength: 7 }, ...manifest.mips.slice(1)] })).toThrow(/mip 0/);
    expect(() => validateTextureStreamManifest(fixture(8, 8, 'bc7', 2).manifest)).toThrow(/fallback/);
  });

  it('loads a complete fallback and leaves the provider with an independent signal', async () => {
    const { manifest, data } = fixture();
    const fetchMock = mockPack(manifest, data);
    const loadSignal = new AbortController();
    const source = await loadTextureStreamSource(`${origin}manifest.json`, { signal: loadSignal.signal });
    expect(source.id).toBe(manifest.id);
    expect(source.fallback.baseMipLevel).toBe(2);
    expect(source.fallback.levels.map(level => level.data)).toEqual(data.slice(2));
    expect(fetchMock.mock.calls.map(call => call[0])).toEqual([
      `${origin}manifest.json`, `${origin}tiles/mip-2.bin`, `${origin}tiles/mip-3.bin`,
    ]);
    loadSignal.abort();
    const later = new AbortController();
    const detail = await source.provider(source.mips[0], later.signal);
    await expect(verifyTextureStreamMip(source.mips[0], detail, later.signal)).resolves.toBe(detail);
    expect(fetchMock.mock.calls.at(-1)?.[1]).toEqual({ signal: later.signal });
  });

  it('loads a compressed resident tail with padded small mips', async () => {
    const { manifest, data } = fixture(8, 8, 'bc7', 1);
    mockPack(manifest, data);
    const source = await loadTextureStreamSource(`${origin}manifest.json`);
    expect(source.fallback.levels.map(level => level.data.byteLength)).toEqual([16, 16, 16]);
  });

  it('checks the total fallback budget before fetching payloads', async () => {
    const { manifest, data } = fixture();
    const fetchMock = mockPack(manifest, data);
    await expect(loadTextureStreamSource(`${origin}manifest.json`, { maxFallbackBytes: 19 }))
      .rejects.toThrow(/budget/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await expect(loadTextureStreamSource(`${origin}manifest.json`, { maxManifestBytes: 1048577 }))
      .rejects.toThrow(/maxManifestBytes/);
  });

  it('rejects short, oversized, corrupt, and aborted payloads', async () => {
    const { manifest, data } = fixture();
    for (const bad of [data[2].subarray(0, -1), new Uint8Array(data[2].byteLength + 1),
      new Uint8Array(data[2].byteLength)]) {
      const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
      vi.stubGlobal('fetch', vi.fn(async (url: string) => url.endsWith('manifest.json')
        ? streamed(manifestBytes, url).response : streamed(bad, url).response));
      await expect(loadTextureStreamSource(`${origin}manifest.json`)).rejects.toThrow();
    }
    const controller = new AbortController();
    controller.abort();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    await expect(loadTextureStreamSource(`${origin}manifest.json`, { signal: controller.signal }))
      .rejects.toMatchObject({ name: 'AbortError' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('bounds streamed manifest bytes and rejects cross-origin redirects', async () => {
    const { manifest } = fixture();
    const bytes = new TextEncoder().encode(JSON.stringify(manifest));
    const oversized = streamed(bytes, `${origin}manifest.json`, [bytes], String(bytes.byteLength), false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(oversized.response));
    await expect(loadTextureStreamSource(`${origin}manifest.json`, { maxManifestBytes: 20 }))
      .rejects.toThrow(/oversized manifest/);
    expect(oversized.cancel).toHaveBeenCalledOnce();
    const streamedOversize = streamed(bytes, `${origin}manifest.json`, [bytes], undefined, false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamedOversize.response));
    await expect(loadTextureStreamSource(`${origin}manifest.json`, { maxManifestBytes: 20 }))
      .rejects.toThrow(/oversized manifest/);
    expect(streamedOversize.cancel).toHaveBeenCalledOnce();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamed(bytes, 'https://evil.example/manifest.json').response));
    await expect(loadTextureStreamSource(`${origin}manifest.json`)).rejects.toThrow(/redirect/);
  });

  it('cancels a pending manifest read on abort', async () => {
    let started!: () => void;
    const reading = new Promise<void>(resolve => { started = resolve; });
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      pull() { started(); },
      cancel,
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200,
      url: `${origin}manifest.json`, body, headers: new Headers() }));
    const controller = new AbortController();
    const pending = loadTextureStreamSource(`${origin}manifest.json`, { signal: controller.signal });
    await reading;
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancel).toHaveBeenCalled();
  });
});
