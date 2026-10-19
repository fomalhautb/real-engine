import { describe, expect, it } from 'vitest';
import { GpuTextureUpload } from '../src/gpu-texture-upload';
import { textureStorageLayout, type PreparedTextureAsset } from '../src/texture-storage';

type Write = { destination: GPUImageCopyTexture; data: Uint8Array; layout: GPUImageDataLayout;
  size: GPUExtent3DStrict };

function fixture(width: number, height: number, format: PreparedTextureAsset['format'],
  baseMipLevel = 0): { asset: PreparedTextureAsset; payloads: Uint8Array[] } {
  const blockSize = format === 'rgba8' ? 1 : 4;
  const blockBytes = format === 'rgba8' ? 4 : 16;
  const mipCount = 1 + Math.floor(Math.log2(Math.max(width, height)));
  const levels = Array.from({ length: mipCount - baseMipLevel }, (_, index) => {
    const mip = index + baseMipLevel;
    const w = Math.max(1, width >> mip), h = Math.max(1, height >> mip);
    const bytes = Math.ceil(w / blockSize) * Math.ceil(h / blockSize) * blockBytes;
    return { width: w, height: h, data: Uint8Array.from({ length: bytes }, (_, i) => (i + index * 37) & 255) };
  });
  return { asset: { kind: 'prepared', width, height, format, colorSpace: 'linear', baseMipLevel, levels },
    payloads: levels.map(level => level.data) };
}

function fakeQueue(failAt = -1) {
  const writes: Write[] = [];
  let calls = 0;
  const texture = {} as GPUTexture;
  const device = { queue: { writeTexture(destination: GPUImageCopyTexture, data: Uint8Array,
    layout: GPUImageDataLayout, size: GPUExtent3DStrict) {
    calls++;
    if (calls === failAt) throw new Error('injected write failure');
    writes.push({ destination, data, layout, size });
  } } } as unknown as GPUDevice;
  return { device, texture, writes };
}

describe('GpuTextureUpload', () => {
  it('walks RGBA NPOT rows and mips without exceeding a frame budget', () => {
    const { asset, payloads } = fixture(3, 5, 'rgba8');
    const layout = textureStorageLayout(asset);
    const { device, texture, writes } = fakeQueue();
    const upload = new GpuTextureUpload(device, texture, layout, payloads);
    expect(upload.remainingBytes).toBe(72);
    expect(upload.nextRowBytes).toBe(12);
    expect(upload.upload(11)).toBe(0);
    expect(writes).toHaveLength(0);
    expect(upload.upload(24)).toBe(24);
    expect(writes[0]).toMatchObject({ destination: { texture, mipLevel: 0,
      origin: { x: 0, y: 0, z: 0 } }, layout: { bytesPerRow: 12, rowsPerImage: 2 }, size: [3, 2, 1] });
    expect(upload.upload(36)).toBe(36);
    expect(writes[1]).toMatchObject({ destination: { mipLevel: 0, origin: { y: 2 } }, size: [3, 3, 1] });
    expect(upload.nextRowBytes).toBe(4);
    expect(upload.upload(8)).toBe(8);
    expect(writes[2]).toMatchObject({ destination: { mipLevel: 1, origin: { y: 0 } }, size: [1, 2, 1] });
    expect(upload.upload(4)).toBe(4);
    expect(upload.done).toBe(true);
    expect(upload.remainingBytes).toBe(0);
    expect(upload.nextRowBytes).toBe(0);
    expect(upload.upload(100)).toBe(0);
    expect(writes.map(write => write.data.byteLength)).toEqual([24, 36, 8, 4]);
  });

  it('uses compressed block rows and padded copy extents for tail mips', () => {
    const { asset, payloads } = fixture(8, 8, 'bc7');
    const layout = textureStorageLayout(asset);
    const { device, texture, writes } = fakeQueue();
    const upload = new GpuTextureUpload(device, texture, layout, payloads);
    expect(upload.upload(16)).toBe(0);
    expect(upload.upload(48)).toBe(32);
    expect(upload.remainingBytes).toBe(layout.byteLength - 32);
    expect(upload.upload(64)).toBe(64);
    expect(writes[1]).toMatchObject({ destination: { mipLevel: 0,
      origin: { x: 0, y: 4, z: 0 } }, layout: { bytesPerRow: 32, rowsPerImage: 1 }, size: [8, 4, 1] });
    expect(writes[2]).toMatchObject({ destination: { mipLevel: 1 }, size: [4, 4, 1] });
    expect(writes[3]).toMatchObject({ destination: { mipLevel: 2 }, size: [4, 4, 1] });
    expect(upload.upload(16)).toBe(16);
    expect(writes[4]).toMatchObject({ destination: { mipLevel: 3 }, size: [4, 4, 1] });
    expect(upload.done).toBe(true);
  });

  it('retains the original payload buffer and byte offset, without copying', () => {
    const { asset, payloads } = fixture(2, 2, 'rgba8');
    const storage = new Uint8Array(64);
    storage.set(payloads[0], 7);
    asset.levels[0].data = storage.subarray(7, 23);
    const { device, texture, writes } = fakeQueue();
    const upload = new GpuTextureUpload(device, texture, textureStorageLayout(asset),
      asset.levels.map(level => level.data));
    expect(upload.upload(8)).toBe(8);
    expect(writes[0].data.buffer).toBe(storage.buffer);
    expect(writes[0].data.byteOffset).toBe(7);
    expect(upload.upload(8)).toBe(8);
    expect(writes[1].data.buffer).toBe(storage.buffer);
    expect(writes[1].data.byteOffset).toBe(15);
  });

  it('does not advance a failed write and retries from that mip', () => {
    const { asset, payloads } = fixture(2, 2, 'rgba8');
    const { device, texture, writes } = fakeQueue(2);
    const upload = new GpuTextureUpload(device, texture, textureStorageLayout(asset), payloads);
    expect(() => upload.upload(100)).toThrow('injected write failure');
    expect(writes).toHaveLength(1);
    expect(upload.remainingBytes).toBe(4);
    expect(upload.nextRowBytes).toBe(4);
    expect(upload.upload(4)).toBe(4);
    expect(writes[1].destination.mipLevel).toBe(1);
    expect(upload.done).toBe(true);
  });

  it('validates payloads and positive integral budgets before writing', () => {
    const { asset, payloads } = fixture(2, 2, 'rgba8');
    const layout = textureStorageLayout(asset);
    const { device, texture, writes } = fakeQueue();
    expect(() => new GpuTextureUpload(device, texture, layout, payloads.slice(1))).toThrow('TEXTURE_UPLOAD_LEVEL_COUNT');
    expect(() => new GpuTextureUpload(device, texture, layout, [payloads[0].subarray(1), payloads[1]]))
      .toThrow('TEXTURE_UPLOAD_LEVEL_LAYOUT');
    const upload = new GpuTextureUpload(device, texture, layout, payloads);
    for (const budget of [0, -1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => upload.upload(budget)).toThrow('TEXTURE_UPLOAD_BUDGET');
    }
    expect(writes).toHaveLength(0);
    expect(upload.remainingBytes).toBe(layout.byteLength);
  });
});
