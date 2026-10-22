import { expect, it, vi } from 'vitest';
import { rasterReceiverBytes } from '../src/surface-receivers';
import { GpuSurfaceReceivers } from '../src/gpu-surface-receivers';

it('accounts full resolution attachments plus identity rows before resize', () => {
  const bytes = 1280 * 720 * 32 + 401 * 32;
  expect(rasterReceiverBytes(1280, 720, 401, bytes)).toBe(bytes);
  expect(() => rasterReceiverBytes(1280, 720, 401, bytes - 1)).toThrow('RASTER_RECEIVER_BYTE_BUDGET');
  expect(() => rasterReceiverBytes(1920, 1080, 401, bytes)).toThrow('RASTER_RECEIVER_BYTE_BUDGET');
});

it('rejects dimension and storage limits before GPU allocation', () => {
  for (const rows of [0, 1.5, NaN, 0xffff_ffff]) {
    expect(() => rasterReceiverBytes(64, 64, rows, 128 * 1024 * 1024)).toThrow();
  }
  expect(() => rasterReceiverBytes(65, 64, 1, 128 * 1024 * 1024, 64)).toThrow('INVALID_RASTER_RECEIVER_LAYOUT');
  expect(() => rasterReceiverBytes(64, 64, 3, 128 * 1024 * 1024, 64, 64)).toThrow('INVALID_RASTER_RECEIVER_LAYOUT');
});

it('retains old receivers and accounting if a replacement allocation fails', () => {
  vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4 });
  let allocations = 0, owned = 0, peak = 0;
  const textures: { destroy: ReturnType<typeof vi.fn> }[] = [];
  const device = { limits: { maxTextureDimension2D: 8192, maxBufferSize: 128 * 1024 * 1024,
    maxStorageBufferBindingSize: 128 * 1024 * 1024, maxColorAttachmentBytesPerSample: 32, maxColorAttachments: 8 },
  createBuffer: () => ({ destroy: vi.fn() }), createTexture: () => {
    if (++allocations === 4) throw new Error('SIMULATED_ALLOCATION_FAILURE');
    const texture = { destroy: vi.fn() }; textures.push(texture); return texture;
  } } as unknown as GPUDevice;
  const receiver = new GpuSurfaceReceivers(device, 1, 1024 * 1024, bytes => { owned += bytes; peak = Math.max(peak, owned); });
  try {
    const original = receiver.resize(64, 64), before = receiver.diagnostics();
    expect(() => receiver.resize(128, 128)).toThrow('SIMULATED_ALLOCATION_FAILURE');
    expect(receiver.diagnostics()).toEqual(before);
    expect(owned).toBe(before.ownedGpuBytes);
    expect(receiver.resize(64, 64).motionDepth).toBe(original.motionDepth);
    expect(textures[0].destroy).not.toHaveBeenCalled(); expect(textures[1].destroy).not.toHaveBeenCalled();
    expect(textures[2].destroy).toHaveBeenCalledOnce();
    expect(() => receiver.preflight(180, 180)).toThrow('RASTER_RECEIVER_RESIZE_BYTE_BUDGET');
    expect(peak).toBeLessThanOrEqual(1024 * 1024);
  } finally { receiver.dispose(); vi.unstubAllGlobals(); }
  expect(owned).toBe(0);
});
