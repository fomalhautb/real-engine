import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuVolumetricFog } from '../src/gpu-volumetric-fog';

type FakeBuffer = { label: string; destroyed: boolean; destroy(): void };
type FakeTexture = { label: string; destroyed: boolean; createView(): object; destroy(): void };

class FakeDevice {
  readonly limits = { maxStorageBuffersPerShaderStage: 3, maxTextureDimension2D: 4096,
    maxBufferSize: 128 * 1024 * 1024, maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535 };
  readonly buffers: FakeBuffer[] = [];
  readonly textures: FakeTexture[] = [];
  bufferCalls = 0;
  bindGroupCalls = 0;
  computePipelineCalls = 0;
  failBufferCall = -1;
  failBufferLabel = '';
  failTexture = false;
  failBindGroupCall = -1;
  failComputePipelineCall = -1;

  createBuffer = ({ label }: { label: string }): GPUBuffer => {
    this.bufferCalls++;
    if (this.bufferCalls === this.failBufferCall || label === this.failBufferLabel) {
      this.failBufferCall = -1; this.failBufferLabel = '';
      throw new Error('FAKE_BUFFER_FAILURE');
    }
    const buffer: FakeBuffer = { label, destroyed: false, destroy() { this.destroyed = true; } };
    this.buffers.push(buffer);
    return buffer as unknown as GPUBuffer;
  };
  createTexture = ({ label }: { label: string }): GPUTexture => {
    if (this.failTexture) { this.failTexture = false; throw new Error('FAKE_TEXTURE_FAILURE'); }
    const texture: FakeTexture = { label, destroyed: false, createView: () => ({}), destroy() { this.destroyed = true; } };
    this.textures.push(texture);
    return texture as unknown as GPUTexture;
  };
  createShaderModule = () => ({ getCompilationInfo: async () => ({ messages: [] }) });
  createBindGroupLayout = () => ({});
  createPipelineLayout = () => ({});
  createBindGroup = () => {
    this.bindGroupCalls++;
    if (this.bindGroupCalls === this.failBindGroupCall) throw new Error('FAKE_BIND_GROUP_FAILURE');
    return {};
  };
  createSampler = () => ({});
  createComputePipelineAsync = async () => {
    this.computePipelineCalls++;
    if (this.computePipelineCalls === this.failComputePipelineCall) throw new Error('FAKE_PIPELINE_FAILURE');
    return {};
  };
  createRenderPipelineAsync = async () => ({});
  queue = { writeBuffer() {} };
}

function attachment(): GPUTexture {
  return { createView: () => ({}) } as unknown as GPUTexture;
}
function ledger() {
  let bytes = 0;
  const balances: number[] = [];
  return { account: (delta: number) => { bytes += delta; balances.push(bytes); },
    get bytes() { return bytes; }, get balances() { return balances; } };
}
function allDestroyed(device: FakeDevice): boolean {
  return [...device.buffers, ...device.textures].every(resource => resource.destroyed);
}
function resize(fog: GpuVolumetricFog, width: number, height: number): void {
  fog.resize(width, height, attachment(), attachment(), attachment());
}

beforeEach(() => {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 });
  vi.stubGlobal('GPUTextureUsage', { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1, FRAGMENT: 2 });
});
afterEach(() => vi.unstubAllGlobals());

describe('GPU volumetric fog ownership', () => {
  it('bounds sun radiance before optical products can overflow f32', () => {
    expect(() => GpuVolumetricFog.validateSunColor([1e20, 0, 1])).not.toThrow();
    for (const color of [[1e21, 0, 0], [Infinity, 0, 0], [-1, 0, 0], [1, 2]]) {
      expect(() => GpuVolumetricFog.validateSunColor(color)).toThrow('FOG_SUN_RADIANCE_LIMIT');
    }
  });
  it('rejects an impossible budget without allocating persistent buffers', async () => {
    const device = new FakeDevice(), tracked = ledger();
    await expect(GpuVolumetricFog.create(device as unknown as GPUDevice, { maxBytes: 1 }, tracked.account)).rejects.toThrow('fog grid cannot fit');
    expect(device.buffers).toHaveLength(0); expect(tracked.bytes).toBe(0);
  });
  it('releases the first allocation when construction fails and both persistent buffers when pipeline creation fails', async () => {
    const secondBuffer = new FakeDevice(), firstLedger = ledger();
    secondBuffer.failBufferCall = 2;
    await expect(GpuVolumetricFog.create(secondBuffer as unknown as GPUDevice, {}, firstLedger.account))
      .rejects.toThrow('FAKE_BUFFER_FAILURE');
    expect(secondBuffer.buffers).toHaveLength(1);
    expect(allDestroyed(secondBuffer)).toBe(true);
    expect(firstLedger.bytes).toBe(0);

    const pipeline = new FakeDevice(), secondLedger = ledger();
    pipeline.failComputePipelineCall = 2;
    await expect(GpuVolumetricFog.create(pipeline as unknown as GPUDevice, {}, secondLedger.account))
      .rejects.toThrow('FAKE_PIPELINE_FAILURE');
    expect(pipeline.buffers).toHaveLength(2);
    expect(allDestroyed(pipeline)).toBe(true);
    expect(secondLedger.bytes).toBe(0);
    expect(secondLedger.balances.every(bytes => bytes >= 0)).toBe(true);
  });

  it.each(['buffer', 'texture', 'bindGroup'] as const)(
    'cleans a partial %s resize, keeps the ledger nonnegative, and permits retry', async failure => {
      const device = new FakeDevice(), tracked = ledger();
      const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice, {}, tracked.account);
      try {
        resize(fog, 64, 64);
        const oldOutput = fog.colorTexture as unknown as FakeTexture;
        if (failure === 'buffer') device.failBufferLabel = 'fog cumulative radiance and transmission';
        if (failure === 'texture') device.failTexture = true;
        if (failure === 'bindGroup') device.failBindGroupCall = device.bindGroupCalls + 2;
        expect(() => resize(fog, 96, 64)).toThrow(`FAKE_${failure === 'bindGroup' ? 'BIND_GROUP' : failure.toUpperCase()}_FAILURE`);
        expect(oldOutput.destroyed).toBe(true); // Old view is retired before the new allocation.
        expect(fog.colorTexture).toBeUndefined();
        expect(fog.diagnostics().grid).toBeNull();
        expect(tracked.bytes).toBe(5376);
        expect(tracked.balances.every(bytes => bytes >= 0)).toBe(true);
        expect(device.buffers.filter(buffer => buffer.label.startsWith('fog ') && buffer.label !== 'fog frame'
          && buffer.label !== 'fog local media').every(buffer => buffer.destroyed)).toBe(true);
        expect(device.textures.every(texture => texture.destroyed)).toBe(true);

        resize(fog, 96, 64);
        expect(fog.colorTexture).toBeDefined();
        expect(tracked.bytes).toBe(fog.diagnostics().grid!.bytes);
      } finally { fog.dispose(); }
      expect(tracked.bytes).toBe(0);
      expect(tracked.balances.every(bytes => bytes >= 0)).toBe(true);
      expect(allDestroyed(device)).toBe(true);
    },
  );

  it('rejects byte and device preflight before retiring a valid view', async () => {
    for (const [options, rejectedWidth, rejectedHeight, error] of [
      [{ maxBytes: 20_000, slices: 8 }, 64, 64, 'fog grid cannot fit'],
      [{}, 5000, 64, 'FOG_DEVICE_BUDGET'],
    ] as const) {
      const device = new FakeDevice(), tracked = ledger();
      const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice, options, tracked.account);
      try {
        resize(fog, 32, 32);
        const oldOutput = fog.colorTexture as unknown as FakeTexture;
        const oldBytes = tracked.bytes;
        expect(() => resize(fog, rejectedWidth, rejectedHeight)).toThrow(error);
        expect(fog.colorTexture).toBe(oldOutput as unknown as GPUTexture);
        expect(oldOutput.destroyed).toBe(false);
        expect(tracked.bytes).toBe(oldBytes);
      } finally { fog.dispose(); }
      expect(tracked.bytes).toBe(0);
      expect(allDestroyed(device)).toBe(true);
    }
  });
});
