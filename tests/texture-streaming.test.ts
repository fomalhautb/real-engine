import { createHash } from 'node:crypto';
import { describe, expect, it, vi } from 'vitest';
import type { MaterialBinding } from '../src/contracts';
import type { MaterialTexturePin, PreparedTextureUpload, TextureReplacementResult } from '../src/gpu-material';
import { textureStorageLayout, type PreparedTextureAsset } from '../src/texture-storage';
import type { TextureStreamDemand, TextureStreamSource, TextureStreamingOptions } from '../src/texture-stream-types';
import { TextureStreaming, type TextureStreamingBackend } from '../src/texture-streaming';

const hash = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');

function fixture(id: string, baseMipLevel = 2): TextureStreamSource {
  const data = [
    new Uint8Array(256).fill(11), new Uint8Array(64).fill(22),
    new Uint8Array(16).fill(33), new Uint8Array(4).fill(44),
  ];
  const mips = data.map((payload, level) => ({ level, width: 8 >> level, height: 8 >> level,
    byteLength: payload.byteLength, sha256: hash(payload) }));
  const fallback: PreparedTextureAsset = { kind: 'prepared', width: 8, height: 8,
    format: 'rgba8', colorSpace: 'linear', baseMipLevel,
    levels: data.slice(baseMipLevel).map((payload, index) => ({
      width: 8 >> (index + baseMipLevel), height: 8 >> (index + baseMipLevel), data: payload,
    })) };
  return { id, width: 8, height: 8, format: 'rgba8', colorSpace: 'linear', mips,
    fallback, provider: async mip => new Uint8Array(data[mip.level]) };
}

class Backend implements TextureStreamingBackend {
  readonly refs = new Map<PreparedTextureAsset, { count: number; bytes: number }>();
  readonly materials: { textures: PreparedTextureAsset[] }[] = [];
  readonly writes: number[] = [];
  pins = 0;
  stages = 0;
  replacements = 0;
  unmanagedBytes = 0;
  constructor(readonly budgetBytes: number) {}
  imageBytes(): number {
    return this.unmanagedBytes + [...this.refs.values()].reduce((sum, image) => sum + image.bytes, 0);
  }
  private retain(asset: PreparedTextureAsset): void {
    const existing = this.refs.get(asset);
    if (existing) existing.count++;
    else this.refs.set(asset, { count: 1, bytes: textureStorageLayout(asset).byteLength });
    if (this.imageBytes() > this.budgetBytes) throw new Error('fake backend over budget');
  }
  private release(asset: PreparedTextureAsset): void {
    const image = this.refs.get(asset);
    if (!image) throw new Error('fake backend released unknown image');
    if (--image.count === 0) this.refs.delete(asset);
  }
  bind(...textures: PreparedTextureAsset[]): { textures: PreparedTextureAsset[] } {
    textures.forEach(asset => this.retain(asset));
    const material = { textures: [...textures] };
    this.materials.push(material);
    return material;
  }
  pin(asset: PreparedTextureAsset): MaterialTexturePin {
    this.retain(asset); this.pins++;
    let disposed = false;
    return { dispose: () => {
      if (disposed) return;
      disposed = true; this.pins--; this.release(asset);
    } };
  }
  stage(asset: PreparedTextureAsset): PreparedTextureUpload {
    this.retain(asset); this.stages++;
    const layout = textureStorageLayout(asset);
    let remaining = layout.byteLength;
    let level = 0, row = 0, disposed = false;
    const nextRowBytes = () => remaining ? layout.levels[level].bytesPerRow : 0;
    return {
      asset, get complete() { return remaining === 0; }, get disposed() { return disposed; },
      diagnostics: () => ({ complete: remaining === 0, remainingBytes: remaining,
        nextRowBytes: nextRowBytes(), allocationBytes: disposed ? 0 : layout.byteLength }),
      upload: maxBytes => {
        if (disposed) throw new Error('upload disposed');
        let written = 0;
        while (remaining && written + nextRowBytes() <= maxBytes) {
          const bytes = nextRowBytes();
          written += bytes; remaining -= bytes; row++;
          if (row === layout.levels[level].rowsPerImage) { level++; row = 0; }
        }
        this.writes.push(written);
        return written;
      },
      dispose: () => { if (!disposed) { disposed = true; this.stages--; this.release(asset); } },
    };
  }
  replace(source: PreparedTextureAsset, next: PreparedTextureAsset): TextureReplacementResult {
    let materials = 0, slots = 0;
    for (const material of this.materials) {
      let touched = false;
      material.textures.forEach((asset, index) => {
        if (asset !== source) return;
        this.retain(next); this.release(source); material.textures[index] = next;
        slots++; touched = true;
      });
      if (touched) materials++;
    }
    this.replacements++;
    return { materials, slots };
  }
}

const demand = (source: TextureStreamSource, mip = 0, priority = 1): TextureStreamDemand =>
  ({ source: source.fallback, mip, priority });
const current = (stream: TextureStreaming, id: string) =>
  stream.diagnostics().sources.find(source => source.id === id)!.currentMip;

async function frames(stream: TextureStreaming, demands: readonly TextureStreamDemand[], count = 30): Promise<number[]> {
  const uploaded: number[] = [];
  for (let i = 0; i < count; i++) {
    uploaded.push(stream.tick(demands));
    await stream.idle();
  }
  return uploaded;
}

describe('TextureStreaming', () => {
  it('shares constrained detail across equally visible images instead of starving later materials', async () => {
    for (const reverse of [false, true]) {
      const sources = ['a', 'b', 'c', 'd'].map(id => fixture(id));
      const backend = new Backend(440);
      const stream = await TextureStreaming.create({ sources: reverse ? [...sources].reverse() : sources,
        maxCpuBytes: 400, maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 }, backend);
      const demands = sources.map(source => demand(source));
      await frames(stream, demands, 60);
      // Four mip-1 tails fit; greedy whole-tail allocation used to give the
      // first image mip 0 and leave the other three at their 2x2 fallback.
      expect(sources.map(source => current(stream, source.id))).toEqual([1, 1, 1, 1]);
      expect(stream.diagnostics().cpuBytes).toBe(336);
      expect(backend.imageBytes()).toBe(336);
      const publications = stream.diagnostics().publications;
      await frames(stream, demands, 10);
      expect(stream.diagnostics().publications).toBe(publications);
      await stream.dispose();
      expect(backend.imageBytes()).toBe(0);
    }
  });

  it('uploads bounded rows, retains old bindings until completion, and publishes all shared slots', async () => {
    const source = fixture('wall');
    const backend = new Backend(400);
    const stream = await TextureStreaming.create({ sources: [source], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 }, backend);
    const old = source.fallback;
    const a = backend.bind(old, old), b = backend.bind(old);
    expect(stream.source(old)?.id).toBe('wall');
    expect(stream.owns(old)).toBe(true);
    expect(stream.resolve({ textures: [old] } as unknown as MaterialBinding)?.textures?.[0]).toBe(old);

    stream.tick([demand(source)]);
    await stream.idle();
    expect(stream.diagnostics().requests).toBe(2);
    expect(current(stream, 'wall')).toBe(2);
    expect(stream.tick([demand(source)])).toBe(32);
    expect(a.textures).toEqual([old, old]);
    expect(b.textures).toEqual([old]);
    expect(backend.stages).toBe(1);
    expect(stream.diagnostics().frameUploadedBytes).toBe(32);

    const uploads = await frames(stream, [demand(source)], 20);
    expect(uploads.every(bytes => bytes <= 32)).toBe(true);
    expect(current(stream, 'wall')).toBe(0);
    const next = a.textures[0];
    expect(next).not.toBe(old);
    expect(a.textures).toEqual([next, next]);
    expect(b.textures).toEqual([next]);
    expect(stream.resolve({ textures: [old] } as unknown as MaterialBinding)?.textures?.[0]).toBe(next);
    expect(stream.owns(next)).toBe(true);
    expect(backend.pins).toBe(1);
    expect(backend.stages).toBe(0);
    expect(backend.replacements).toBe(1);
    expect(backend.imageBytes()).toBe(340);
    const requests = stream.diagnostics().requests;
    await frames(stream, [demand(source)], 5);
    expect(stream.diagnostics().requests).toBe(requests);
    await stream.dispose();
    expect(backend.pins).toBe(0);
  });

  it('moves its single fine residency with changing priorities without crossing GPU or CPU budgets', async () => {
    const a = fixture('a'), b = fixture('b');
    const backend = new Backend(400);
    const stream = await TextureStreaming.create({ sources: [a, b], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32, downgradeDelayFrames: 1 }, backend);
    const aMaterial = backend.bind(a.fallback), bMaterial = backend.bind(b.fallback);
    for (const winner of [a, b, a]) {
      const loser = winner === a ? b : a;
      const demands = [demand(winner, 0, 10), demand(loser, 0, 1)];
      const uploads = await frames(stream, demands, 30);
      expect(uploads.every(bytes => bytes <= 32)).toBe(true);
      expect(current(stream, winner.id)).toBe(0);
      expect(current(stream, loser.id)).toBe(2);
      expect(stream.diagnostics().cpuBytes).toBe(360);
      expect(backend.imageBytes()).toBe(360);
      expect(backend.imageBytes()).toBeLessThanOrEqual(backend.budgetBytes);
      expect(aMaterial.textures[0].baseMipLevel).toBe(current(stream, 'a'));
      expect(bMaterial.textures[0].baseMipLevel).toBe(current(stream, 'b'));
    }
    expect(stream.diagnostics().evictions).toBe(2);
    await stream.dispose();
  });

  it('uses the root as an intermediate when old and finer images cannot coexist', async () => {
    const a = fixture('a'), b = fixture('b');
    const backend = new Backend(400);
    const stream = await TextureStreaming.create({ sources: [a, b], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 }, backend);
    const material = backend.bind(a.fallback);
    await frames(stream, [demand(a, 1)], 10);
    expect(current(stream, 'a')).toBe(1);
    const middle = material.textures[0];
    expect(backend.imageBytes()).toBe(104); // mip 1 tail (84) + other root (20)

    expect(stream.tick([demand(a, 0)])).toBe(20); // Complete root bridge in this frame.
    expect(current(stream, 'a')).toBe(2);
    expect(material.textures[0]).toBe(a.fallback);
    expect(material.textures[0]).not.toBe(middle);
    expect(backend.imageBytes()).toBe(40);
    const uploaded = await frames(stream, [demand(a, 0)], 20);
    expect(uploaded.every(bytes => bytes <= 32)).toBe(true);
    expect(current(stream, 'a')).toBe(0);
    expect(backend.imageBytes()).toBe(360);
    expect(stream.diagnostics().evictions).toBe(1);
    await stream.dispose();
  });

  it('delays lower demand and respects unmanaged GPU occupancy', async () => {
    const source = fixture('a');
    const backend = new Backend(400);
    const stream = await TextureStreaming.create({ sources: [source], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32, downgradeDelayFrames: 3 }, backend);
    backend.unmanagedBytes = 350;
    await frames(stream, [demand(source)], 5);
    expect(stream.diagnostics().sources[0].plannedMip).toBe(2);
    expect(current(stream, 'a')).toBe(2);
    expect(stream.diagnostics().budgetBlocked).toBe(false);
    expect(backend.imageBytes()).toBe(370);

    backend.unmanagedBytes = 0;
    await frames(stream, [demand(source)], 20);
    expect(current(stream, 'a')).toBe(0);
    stream.tick([]);
    stream.tick([]);
    expect(current(stream, 'a')).toBe(0);
    expect(stream.diagnostics().sources[0].wantedMip).toBe(0);
    stream.tick([]);
    expect(current(stream, 'a')).toBe(2);
    expect(stream.diagnostics().sources[0].wantedMip).toBe(2);
    await stream.dispose();
  });

  it('keeps a cancelled provider reservation until its ignored abort settles', async () => {
    const a = fixture('a'), b = fixture('b');
    let release!: () => void;
    const pending = new Promise<void>(resolve => { release = resolve; });
    let began!: () => void;
    const started = new Promise<void>(resolve => { began = resolve; });
    let aborted = false;
    const provider = vi.fn(async (mip: TextureStreamSource['mips'][number], signal: AbortSignal) => {
      began(); signal.addEventListener('abort', () => { aborted = true; });
      await pending; return new Uint8Array(mip.byteLength).fill(mip.level === 0 ? 11 : 22);
    });
    const slow = { ...a, provider };
    const stream = await TextureStreaming.create({ sources: [slow, b], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 }, new Backend(400));
    stream.tick([demand(slow)]);
    await started;
    expect(stream.diagnostics().loading).toBe(true);
    expect(stream.diagnostics().inFlightBytes).toBe(512);
    stream.tick([demand(b)]);
    expect(aborted).toBe(true);
    expect(stream.diagnostics().active).toBe(1);
    expect(stream.diagnostics().reservedCpuBytes).toBe(320);
    expect(stream.tick([demand(b)])).toBe(0);
    expect(stream.diagnostics().requests).toBe(1);
    release(); await stream.idle();
    expect(stream.diagnostics().active).toBe(0);
    expect(stream.diagnostics().inFlightBytes).toBe(0);
    await frames(stream, [demand(b)], 20);
    expect(current(stream, 'b')).toBe(0);
    expect(provider).toHaveBeenCalledOnce();
    await stream.dispose();
  });

  it('records a hash failure once and resumes only after explicit retry', async () => {
    const source = fixture('broken');
    const provider = vi.fn().mockResolvedValueOnce(new Uint8Array(256)).mockImplementation(
      async (mip: TextureStreamSource['mips'][number]) => new Uint8Array(mip.byteLength).fill(mip.level === 0 ? 11 : 22));
    const bad = { ...source, provider };
    const stream = await TextureStreaming.create({ sources: [bad], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 }, new Backend(400));
    stream.tick([demand(bad)]); await stream.idle();
    expect(stream.diagnostics().sources[0].failure?.message).toMatch(/SHA-256/);
    expect(current(stream, bad.id)).toBe(2);
    await frames(stream, [demand(bad)], 8);
    expect(provider).toHaveBeenCalledOnce();
    stream.retry(bad.id);
    await frames(stream, [demand(bad)], 20);
    expect(current(stream, bad.id)).toBe(0);
    expect(provider).toHaveBeenCalledTimes(3);
    await stream.dispose();
  });

  it('releases pins and waits for an in-flight provider during disposal', async () => {
    const source = fixture('pending');
    let release!: () => void;
    const pending = new Promise<void>(resolve => { release = resolve; });
    let began!: () => void;
    const started = new Promise<void>(resolve => { began = resolve; });
    const provider = vi.fn(async (mip: TextureStreamSource['mips'][number]) => {
      began(); await pending; return new Uint8Array(mip.byteLength);
    });
    const backend = new Backend(400);
    const stream = await TextureStreaming.create({ sources: [{ ...source, provider }], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 }, backend);
    stream.tick([demand(source)]); await started;
    let settled = false;
    const disposal = stream.dispose().then(() => { settled = true; });
    expect(backend.pins).toBe(0);
    expect(settled).toBe(false);
    release(); await disposal;
    expect(settled).toBe(true);
    expect(backend.imageBytes()).toBe(0);
    expect(stream.diagnostics().cpuBytes).toBe(0);
    expect(() => stream.tick([])).toThrow('TEXTURE_STREAM_DISPOSED');
  });

  it('rejects impossible roots, row/work budgets, duplicate sources and invalid demand', async () => {
    const a = fixture('a'), b = fixture('b');
    const options: TextureStreamingOptions = { sources: [a], maxCpuBytes: 400,
      maxInFlightBytes: 512, maxUploadBytesPerFrame: 32 };
    await expect(TextureStreaming.create({ ...options, maxCpuBytes: 19 }, new Backend(400)))
      .rejects.toThrow('TEXTURE_STREAM_ROOT_BUDGET');
    await expect(TextureStreaming.create(options, new Backend(39)))
      .rejects.toThrow('TEXTURE_STREAM_ROOT_BUDGET');
    await expect(TextureStreaming.create({ ...options, sources: [fixture('root', 0)],
      maxUploadBytesPerFrame: 31 }, new Backend(800)))
      .rejects.toThrow('TEXTURE_STREAM_MIP_BUDGET');
    await expect(TextureStreaming.create({ ...options, maxInFlightBytes: 511 }, new Backend(400)))
      .rejects.toThrow('TEXTURE_STREAM_MIP_BUDGET');
    await expect(TextureStreaming.create({ ...options, sources: [a, a] }, new Backend(400)))
      .rejects.toThrow('TEXTURE_STREAM_DUPLICATE_SOURCE');
    const stream = await TextureStreaming.create({ ...options, sources: [a, b] }, new Backend(400));
    expect(() => stream.tick([{ source: { ...a.fallback }, mip: 0, priority: 1 }]))
      .toThrow('TEXTURE_STREAM_DEMAND');
    expect(() => stream.tick([demand(a, 4)])).toThrow('TEXTURE_STREAM_DEMAND');
    expect(() => stream.tick([demand(a, 0, Number.NaN)])).toThrow('TEXTURE_STREAM_DEMAND');
    expect(stream.source({ ...a.fallback })).toBeUndefined();
    expect(stream.owns({ ...a.fallback })).toBe(false);
    await stream.dispose();
  });
});
