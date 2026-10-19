import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MaterialResources } from '../src/gpu-material';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../src/material';
import type { MaterialBinding } from '../src/contracts';
import type { TextureAsset } from '../src/texture-assets';
import type { PreparedTextureAsset, PreparedTextureFormat } from '../src/texture-storage';

type FakeTexture = { descriptor: { label?: string; format?: string; size?: readonly number[]; mipLevelCount?: number };
  destroyed: boolean; destroy(): void; createView(): { texture: FakeTexture } };
type FakeBuffer = { bytes: Uint8Array; destroyed: boolean; destroy(): void };

class FakeDevice {
  adapterInfo = { architecture: 'native', description: 'native GPU' };
  features = new Set<string>();
  limits = { maxTextureDimension2D: 8192 };
  textures: FakeTexture[] = [];
  buffers: FakeBuffer[] = [];
  samplers: Record<string, unknown>[] = [];
  writes: { texture: FakeTexture; mipLevel: number; data: Uint8Array;
    layout: { bytesPerRow: number; rowsPerImage?: number }; extent: readonly number[] }[] = [];
  bufferWrites: { target: FakeBuffer; offset: number; byteLength: number }[] = [];
  events: string[] = [];
  submissions = 0;
  failSubmit = false;
  failTextureAt = 0;
  failWriteTextureAt = 0;
  failBufferAt = 0;
  failBindGroupAt = 0;
  bindGroupCalls = 0;
  createBindGroupLayout = () => ({});
  createShaderModule = () => ({});
  createPipelineLayout = () => ({});
  createRenderPipeline = () => ({});
  createCommandEncoder = () => ({ beginRenderPass: () => ({ setPipeline: () => {},
    setBindGroup: () => {}, draw: () => this.events.push('draw'), end: () => {} }),
  finish: () => ({}) });
  createSampler = (descriptor: Record<string, unknown>) => {
    this.samplers.push(descriptor);
    return { descriptor };
  };
  createTexture = (descriptor: FakeTexture['descriptor']): FakeTexture => {
    if (this.textures.length + 1 === this.failTextureAt) {
      this.failTextureAt = 0;
      throw new Error('FAKE_TEXTURE_FAILURE');
    }
    const texture: FakeTexture = { descriptor, destroyed: false,
      destroy() { this.destroyed = true; }, createView() { return { texture: this }; } };
    this.textures.push(texture);
    return texture;
  };
  createBuffer = ({ size }: { size: number }): FakeBuffer => {
    if (this.buffers.length + 1 === this.failBufferAt) {
      this.failBufferAt = 0;
      throw new Error('FAKE_BUFFER_FAILURE');
    }
    const buffer: FakeBuffer = { bytes: new Uint8Array(size), destroyed: false,
      destroy() { this.destroyed = true; } };
    this.buffers.push(buffer);
    return buffer;
  };
  createBindGroup = () => {
    this.bindGroupCalls++;
    if (this.bindGroupCalls === this.failBindGroupAt) {
      this.failBindGroupAt = 0;
      throw new Error('FAKE_BIND_GROUP_FAILURE');
    }
    return {};
  };
  queue = {
    writeTexture: (destination: { texture: FakeTexture; mipLevel?: number }, data: Uint8Array,
      layout: { bytesPerRow: number; rowsPerImage?: number }, extent: readonly number[]) => {
      if (this.writes.length + 1 === this.failWriteTextureAt) {
        this.failWriteTextureAt = 0;
        throw new Error('FAKE_UPLOAD_FAILURE');
      }
      this.writes.push({ texture: destination.texture, mipLevel: destination.mipLevel ?? 0,
        data, layout, extent });
      this.events.push(`upload:${destination.mipLevel ?? 0}`);
    },
    writeBuffer: (target: FakeBuffer, offset: number, data: Float32Array) => {
      const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      target.bytes.set(source, offset);
      this.bufferWrites.push({ target, offset, byteLength: source.byteLength });
    },
    submit: (_commands: unknown[]) => {
      this.events.push('submit'); this.submissions++;
      if (this.failSubmit) { this.failSubmit = false; throw new Error('FAKE_PREPARATION_SUBMIT_FAILURE'); }
    },
  };
}

function ledger() {
  let bytes = 0;
  const changes: number[] = [];
  return { account: (delta: number) => { bytes += delta; changes.push(bytes); },
    get bytes() { return bytes; }, get changes() { return changes; } };
}

function prepared(width: number, height: number, format: PreparedTextureFormat = 'bc7', baseMipLevel = 0): PreparedTextureAsset {
  const block = format === 'rgba8' ? 1 : 4;
  const blockBytes = format === 'rgba8' ? 4 : 16;
  const levels: { width: number; height: number; data: Uint8Array }[] = [];
  for (let level = baseMipLevel; ; level++) {
    const w = Math.max(1, Math.floor(width / 2 ** level));
    const h = Math.max(1, Math.floor(height / 2 ** level));
    const data = new Uint8Array(Math.ceil(w / block) * Math.ceil(h / block) * blockBytes);
    data.fill(level + 17);
    levels.push({ width: w, height: h, data });
    if (w === 1 && h === 1) break;
  }
  return { kind: 'prepared', width, height, format, colorSpace: 'srgb', baseMipLevel, levels };
}

function legacy(): TextureAsset {
  return { width: 4, height: 4, colorSpace: 'linear', pixels: new Uint8Array(64).fill(150) };
}

const textured = { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.texture(0) };
const parametrized = { ...DEFAULT_MATERIAL_DEFINITION,
  baseColor: m.multiply(m.texture(0), m.parameter(0)) };
function binding(asset: PreparedTextureAsset | TextureAsset, extras: Partial<MaterialBinding> = {}): MaterialBinding {
  return { definition: textured, textures: [asset], ...extras };
}

beforeEach(() => {
  vi.stubGlobal('GPUShaderStage', { FRAGMENT: 1, COMPUTE: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 });
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
});
afterEach(() => vi.unstubAllGlobals());

describe('material texture residency and ownership', () => {
  it.each([
    ['bc7', 'texture-compression-bc'],
    ['etc2-rgba8', 'texture-compression-etc2'],
    ['astc-4x4', 'texture-compression-astc'],
  ] as const)('rejects unavailable %s before allocating its texture', (format, feature) => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    expect(() => resources.create(binding(prepared(4, 4, format)))).toThrow(feature);
    expect(device.textures).toHaveLength(1); // Constructor fallback only.
    expect(device.buffers).toHaveLength(0);
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    expect(tracked.bytes).toBe(4);
    resources.dispose();
    expect(device.textures[0].destroyed).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('uses native compressed bytes and requires the old image to fit alongside its replacement', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account, 150);
    const compressed = resources.create(binding(prepared(8, 8)));
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 112 });
    expect(tracked.bytes).toBe(4 + 112 + 320);
    expect(() => resources.create(binding(legacy()))).toThrow(/MATERIAL_TEXTURE_BUDGET/);
    expect(device.textures).toHaveLength(2);
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 112 });
    resources.release(compressed);
    const replacement = resources.create(binding(legacy()));
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 84 });
    resources.release(replacement);
    resources.dispose();
    expect(tracked.bytes).toBe(0);
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.changes.every(bytes => bytes >= 0)).toBe(true);
  });

  it('shares one texture by asset identity and releases it after the last material', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(4, 4), first = resources.create(binding(asset)), second = resources.create(binding(asset));
    const image = device.textures[1];
    expect(device.textures).toHaveLength(2);
    expect(resources.diagnostics()).toEqual({ materials: 2, images: 1, imageBytes: 48 });
    resources.release(first);
    expect(image.destroyed).toBe(false);
    expect(resources.diagnostics()).toEqual({ materials: 1, images: 1, imageBytes: 48 });
    resources.release(first); // Idempotent.
    resources.release(second);
    expect(image.destroyed).toBe(true);
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    resources.dispose(); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it.each(['texture', 'upload', 'buffer', 'bindGroup'] as const)(
    'cleans all acquired resources when %s creation fails', failure => {
      const device = new FakeDevice(), tracked = ledger();
      device.features.add('texture-compression-bc');
      const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
      if (failure === 'texture') device.failTextureAt = 2;
      if (failure === 'upload') device.failWriteTextureAt = 3; // fallback, mip 0, then mip 1.
      if (failure === 'buffer') device.failBufferAt = 1;
      if (failure === 'bindGroup') device.failBindGroupAt = 1;
      expect(() => resources.create(binding(prepared(8, 8)))).toThrow(/FAKE_.*_FAILURE/);
      expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
      expect(tracked.bytes).toBe(4);
      expect(device.textures.slice(1).every(texture => texture.destroyed)).toBe(true);
      expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
      const recovered = resources.create(binding(prepared(4, 4)));
      resources.release(recovered); resources.dispose();
      expect(tracked.bytes).toBe(0);
    },
  );

  it('uploads prepared payloads by identity with exact block-padded extents', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(8, 8), material = resources.create(binding(asset));
    const writes = device.writes.slice(1);
    expect(device.textures[1].descriptor).toMatchObject({ size: [8, 8], mipLevelCount: 4,
      format: 'bc7-rgba-unorm-srgb' });
    expect(writes.map(write => [write.mipLevel, write.layout.bytesPerRow, write.layout.rowsPerImage,
      write.extent])).toEqual([
      [0, 32, 2, [8, 8]], [1, 16, 1, [4, 4]],
      [2, 16, 1, [4, 4]], [3, 16, 1, [4, 4]],
    ]);
    writes.forEach((write, index) => expect(write.data).toBe(asset.levels[index].data));
    resources.release(material); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('translates source-space sampler clamps and preserves residency metadata on parameter updates', () => {
    const device = new FakeDevice(), tracked = ledger();
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(11, 9, 'rgba8', 1);
    const material = resources.create(binding(asset, { definition: parametrized,
      parameters: new Float32Array([0.1, 0.2, 0.3, 0.4]),
      samplers: [{ lodMinClamp: 0.5, lodMaxClamp: 6, minFilter: 'linear' }] }));
    const sampler = device.samplers.at(-1)!;
    expect(sampler.lodMinClamp).toBe(0);
    expect(sampler.lodMaxClamp).toBe(5);
    const stored = new Float32Array(device.buffers[0].bytes.buffer);
    expect(Array.from(stored.slice(64, 68))).toEqual([1, Math.fround(1.1), Math.fround(1.125), 0]);
    const before = Array.from(stored.slice(64));
    resources.updateParameters(material, new Float32Array([0.8, 0.7, 0.6, 0.5]));
    expect(device.bufferWrites.at(-1)).toMatchObject({ target: device.buffers[0], offset: 0, byteLength: 256 });
    expect(Array.from(stored.slice(0, 4))).toEqual(Array.from(new Float32Array([0.8, 0.7, 0.6, 0.5])));
    expect(Array.from(stored.slice(64))).toEqual(before);
    resources.release(material); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('prepares one shared compressed image only after all mip uploads on SwiftShader', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.adapterInfo = { architecture: 'SwiftShader', description: 'software adapter' };
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const asset = prepared(8, 8);
    const first = resources.create(binding(asset)), second = resources.create(binding(asset));
    expect(resources.preparationDiagnostics()).toEqual({ mode: 'swiftshader-render',
      imagesPrepared: 1, ownedGpuBytes: 4 });
    expect(device.submissions).toBe(1);
    expect(device.textures.filter(texture => texture.descriptor.label === 'compressed image preparation target')).toHaveLength(1);
    expect(device.events.slice(1, 7)).toEqual(['upload:0', 'upload:1', 'upload:2', 'upload:3', 'draw', 'submit']);
    expect(tracked.bytes).toBe(4 + 112 + 4 + 2 * 320);
    resources.release(first); resources.release(second); resources.dispose();
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });

  it('does not prepare RGBA8 images even when SwiftShader mode is active', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.adapterInfo = { architecture: 'swiftshader', description: '' };
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    const one = resources.create(binding(prepared(4, 4, 'rgba8')));
    const two = resources.create(binding(legacy()));
    expect(resources.preparationDiagnostics()).toEqual({ mode: 'swiftshader-render',
      imagesPrepared: 0, ownedGpuBytes: 0 });
    expect(device.submissions).toBe(0);
    expect(device.textures.filter(texture => texture.descriptor.label === 'compressed image preparation target')).toHaveLength(0);
    resources.release(one); resources.release(two); resources.dispose();
    expect(tracked.bytes).toBe(0);
  });

  it('does not publish a compressed image when preparation submission fails', () => {
    const device = new FakeDevice(), tracked = ledger();
    device.adapterInfo = { architecture: 'SwiftShader', description: '' };
    device.features.add('texture-compression-bc');
    const resources = new MaterialResources(device as unknown as GPUDevice, tracked.account);
    device.failSubmit = true;
    expect(() => resources.create(binding(prepared(4, 4)))).toThrow('FAKE_PREPARATION_SUBMIT_FAILURE');
    expect(resources.diagnostics()).toEqual({ materials: 0, images: 0, imageBytes: 0 });
    expect(resources.preparationDiagnostics().imagesPrepared).toBe(0);
    expect(device.textures[1].destroyed).toBe(true);
    expect(device.buffers).toHaveLength(0);
    resources.dispose();
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
  });
});
