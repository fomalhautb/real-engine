import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuOcclusion } from '../src/gpu-occlusion';
import { identity } from '../src/math';
import { planOcclusionPyramid } from '../src/occlusion';
import { createTimestampQuerySet } from '../src/gpu-timestamp-span';

type FakeBuffer = { id: number; destroyed: boolean; destroy(): void };
type FakeTexture = { id: number; destroyed: boolean; createView(): object; destroy(): void };
type Write = { buffer: FakeBuffer; data: Uint8Array };

class FakeDevice {
  limits = { maxTextureDimension2D: 4096, maxComputeWorkgroupsPerDimension: 65_535 };
  buffers: FakeBuffer[] = [];
  textures: FakeTexture[] = [];
  writes: Write[] = [];
  bufferCalls = 0;
  textureCalls = 0;
  pipelineCalls = 0;
  bindGroupCalls = 0;
  failBufferCall = -1;
  failTextureCall = -1;
  failPipelineCall = -1;
  failBindGroupCall = -1;
  shaderErrors = false;

  createBuffer = (): GPUBuffer => {
    if (++this.bufferCalls === this.failBufferCall) throw new Error('FAKE_BUFFER_FAILURE');
    const resource: FakeBuffer = { id: this.bufferCalls, destroyed: false, destroy() { this.destroyed = true; } };
    this.buffers.push(resource);
    return resource as unknown as GPUBuffer;
  };
  createTexture = (): GPUTexture => {
    if (++this.textureCalls === this.failTextureCall) throw new Error('FAKE_TEXTURE_FAILURE');
    const resource: FakeTexture = { id: this.textureCalls, destroyed: false,
      createView() { return { texture: this.id }; }, destroy() { this.destroyed = true; } };
    this.textures.push(resource);
    return resource as unknown as GPUTexture;
  };
  createShaderModule = () => ({ getCompilationInfo: async () => ({ messages: this.shaderErrors
    ? [{ type: 'error', message: 'invalid fake shader' }] : [] }) });
  createComputePipelineAsync = async () => {
    if (++this.pipelineCalls === this.failPipelineCall) throw new Error('FAKE_PIPELINE_FAILURE');
    return { getBindGroupLayout: () => ({}) };
  };
  createBindGroupLayout = () => ({});
  createPipelineLayout = () => ({});
  createBindGroup = ({ entries }: { entries: object[] }) => {
    if (++this.bindGroupCalls === this.failBindGroupCall) throw new Error('FAKE_BIND_GROUP_FAILURE');
    return { entries };
  };
  queue = { writeBuffer: (buffer: FakeBuffer, _offset: number, data: ArrayBufferView) => {
    this.writes.push({ buffer, data: Uint8Array.from(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)) });
  } };
}

function ledger() {
  let bytes = 0;
  const balances: number[] = [];
  return { account: (delta: number) => { bytes += delta; balances.push(bytes); },
    get bytes() { return bytes; }, get balances() { return balances; } };
}

function depth(): GPUTexture { return { createView: () => ({ depth: true }) } as unknown as GPUTexture; }
function encoder() {
  let passes = 0, dispatches = 0;
  const descriptors: GPUComputePassDescriptor[] = [];
  return { gpu: { beginComputePass: (descriptor: GPUComputePassDescriptor) => {
    descriptors.push(descriptor);
    passes++;
    return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { dispatches++; }, end() {} };
  } } as unknown as GPUCommandEncoder, descriptors,
  get passes() { return passes; }, get dispatches() { return dispatches; } };
}
const allDestroyed = (device: FakeDevice) => [...device.buffers, ...device.textures].every(item => item.destroyed);

beforeEach(() => {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
  vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 1, STORAGE_BINDING: 2 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
});
afterEach(() => vi.unstubAllGlobals());

describe('GPU occlusion ownership and frame state', () => {
  it('rejects an impossible parameter budget before allocating', async () => {
    const device = new FakeDevice(), tracked = ledger();
    await expect(GpuOcclusion.create(device as unknown as GPUDevice,
      { maxBytes: 207 }, tracked.account)).rejects.toThrow('OCCLUSION_BYTE_BUDGET');
    expect(device.buffers).toHaveLength(0);
    expect(tracked.bytes).toBe(0);
  });

  it.each([2, 3])('cleans partial buffer allocation at call %i', async failedCall => {
    const device = new FakeDevice(), tracked = ledger();
    device.failBufferCall = failedCall;
    await expect(GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account))
      .rejects.toThrow('FAKE_BUFFER_FAILURE');
    expect(device.buffers).toHaveLength(failedCall - 1);
    expect(allDestroyed(device)).toBe(true);
    expect(tracked.bytes).toBe(0);
    expect(tracked.balances.every(value => value >= 0)).toBe(true);
  });

  it.each(['shader', 'first pipeline', 'second pipeline'] as const)(
    'cleans owned buffers after %s initialization failure', async failure => {
      const device = new FakeDevice(), tracked = ledger();
      if (failure === 'shader') device.shaderErrors = true;
      else device.failPipelineCall = failure === 'first pipeline' ? 1 : 2;
      await expect(GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account))
        .rejects.toThrow(failure === 'shader' ? 'invalid fake shader' : 'FAKE_PIPELINE_FAILURE');
      expect(device.buffers).toHaveLength(3);
      expect(allDestroyed(device)).toBe(true);
      expect(tracked.bytes).toBe(0);
    });

  it('reallocates only on view changes, keeps exact ledger, and disposes idempotently', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    const borrowedDepth = depth();
    expect(tracked.bytes).toBe(208);
    try {
      expect(occlusion.update(1, 32, 20, identity(), borrowedDepth)).toBe(208);
      const first = planOcclusionPyramid(32, 20, {});
      expect(tracked.bytes).toBe(first.bytes);
      expect(occlusion.diagnostics().ownedGpuBytes).toBe(first.bytes);
      const initialTextures = [...device.textures];
      occlusion.update(2, 32, 20, identity(), borrowedDepth);
      expect(device.textures).toHaveLength(2);
      expect(tracked.bytes).toBe(first.bytes);
      occlusion.update(3, 64, 20, identity(), borrowedDepth);
      const resized = planOcclusionPyramid(64, 20, {});
      expect(initialTextures.every(texture => texture.destroyed)).toBe(true);
      expect(device.textures).toHaveLength(4);
      expect(tracked.bytes).toBe(resized.bytes);
      expect(occlusion.diagnostics().grid).toEqual(resized);
    } finally { occlusion.dispose(); }
    const balances = [...tracked.balances];
    occlusion.dispose();
    expect(tracked.balances).toEqual(balances);
    expect(tracked.bytes).toBe(0);
    expect(tracked.balances.every(value => value >= 0)).toBe(true);
    expect(allDestroyed(device)).toBe(true);
    expect(() => occlusion.setEnabled(true)).toThrow('OCCLUSION_DISPOSED');
  });

  it('rejects invalid view and device limits before retiring a valid pyramid', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    try {
      const borrowedDepth = depth();
      occlusion.update(1, 32, 20, identity(), borrowedDepth);
      const oldTextures = [...device.textures], oldBytes = tracked.bytes, oldWrites = device.writes.length;
      const badMatrix = identity(); badMatrix[0] = Infinity;
      for (const attempt of [
        () => occlusion.update(2, 0, 20, identity(), borrowedDepth),
        () => occlusion.update(2, 32, 16_385, identity(), borrowedDepth),
        () => occlusion.update(2, 32, 20, badMatrix, borrowedDepth),
        () => occlusion.update(2, 32, 20, new Float32Array(15), borrowedDepth),
      ]) expect(attempt).toThrow();
      device.limits.maxTextureDimension2D = 16;
      expect(() => occlusion.update(2, 64, 20, identity(), borrowedDepth)).toThrow('OCCLUSION_DEVICE_LIMIT');
      device.limits.maxTextureDimension2D = 4096;
      device.limits.maxComputeWorkgroupsPerDimension = 1;
      expect(() => occlusion.update(2, 64, 20, identity(), borrowedDepth)).toThrow('OCCLUSION_DEVICE_LIMIT');
      expect(device.textures).toEqual(oldTextures);
      expect(oldTextures.every(texture => !texture.destroyed)).toBe(true);
      expect(tracked.bytes).toBe(oldBytes);
      expect(device.writes).toHaveLength(oldWrites);
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('preflights an over-budget resize without changing a live view or its history', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice,
      { maxBytes: 3000 }, tracked.account);
    try {
      const borrowedDepth = depth();
      occlusion.update(1, 32, 20, identity(), borrowedDepth);
      occlusion.encodePyramid(encoder().gpu);
      const before = occlusion.diagnostics();
      const textures = [...device.textures], writes = device.writes.length;
      const balances = [...tracked.balances];
      expect(before).toMatchObject({ historyValid: true, frame: 1 });
      expect(() => occlusion.planResize(64, 20)).toThrow('OCCLUSION_BYTE_BUDGET');
      expect(occlusion.diagnostics()).toEqual(before);
      expect(device.textures).toEqual(textures);
      expect(textures.every(texture => !texture.destroyed)).toBe(true);
      expect(device.writes).toHaveLength(writes);
      expect(tracked.balances).toEqual(balances);
      occlusion.update(2, 32, 20, identity(), borrowedDepth);
      expect(new Float32Array(device.writes.at(-3)!.data.buffer)[20]).toBe(1);
      occlusion.encodePyramid(encoder().gpu);
      expect(occlusion.diagnostics()).toMatchObject({ historyValid: true, frame: 2 });
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
    expect(() => occlusion.planResize(32, 20)).toThrow('OCCLUSION_DISPOSED');
  });

  it('cleans a partial texture resize and allows a retry', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    try {
      occlusion.update(1, 32, 20, identity(), depth());
      device.failTextureCall = 4; // Second allocation of the replacement pyramid.
      expect(() => occlusion.update(2, 64, 20, identity(), depth())).toThrow('FAKE_TEXTURE_FAILURE');
      expect(device.textures.every(texture => texture.destroyed)).toBe(true);
      expect(occlusion.diagnostics().grid).toBeNull();
      expect(tracked.bytes).toBe(208);
      occlusion.update(2, 64, 20, identity(), depth());
      expect(tracked.bytes).toBe(planOcclusionPyramid(64, 20, {}).bytes);
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('cleans failed bind-group construction and retries the same dimensions', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    try {
      const borrowedDepth = depth();
      device.failBindGroupCall = 2;
      expect(() => occlusion.update(1, 32, 20, identity(), borrowedDepth)).toThrow('FAKE_BIND_GROUP_FAILURE');
      expect(device.textures.every(texture => texture.destroyed)).toBe(true);
      expect(occlusion.diagnostics().grid).toBeNull();
      expect(tracked.bytes).toBe(208);
      occlusion.update(1, 32, 20, identity(), borrowedDepth);
      expect(tracked.bytes).toBe(planOcclusionPyramid(32, 20, {}).bytes);
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('keeps prediction and current parameter slots separate across camera changes and history resets', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    try {
      const borrowedDepth = depth(), first = identity();
      occlusion.update(1, 32, 20, first, borrowedDepth);
      expect(device.writes.slice(-3).map(write => write.buffer.id)).toEqual([1, 2, 3]);
      expect(new Float32Array(device.writes.at(-3)!.data.buffer)[20]).toBe(0);
      expect(occlusion.previousGroup).not.toBe(occlusion.currentGroup);
      const parameterId = (group: GPUBindGroup) => (group as unknown as {
        entries: { binding: number; resource: { buffer: FakeBuffer } }[];
      }).entries.find(entry => entry.binding === 0)!.resource.buffer.id;
      expect([parameterId(occlusion.previousGroup), parameterId(occlusion.currentGroup)])
        .toEqual([1, 2]);
      const firstEncode = encoder(); occlusion.encodePyramid(firstEncode.gpu);
      expect(firstEncode.dispatches).toBe(planOcclusionPyramid(32, 20, {}).mipLevelCount);
      const moved = identity(); moved[12] = 1;
      occlusion.update(2, 32, 20, moved, borrowedDepth);
      expect(new Float32Array(device.writes.at(-3)!.data.buffer)[20]).toBe(1);
      expect(new Float32Array(device.writes.at(-2)!.data.buffer)[12]).toBe(1);
      occlusion.encodePyramid(encoder().gpu);
      occlusion.setEnabled(false);
      occlusion.update(3, 32, 20, moved, borrowedDepth);
      expect(new Float32Array(device.writes.at(-3)!.data.buffer)[20]).toBe(0);
      const disabled = encoder(); occlusion.encodePyramid(disabled.gpu);
      expect(disabled.passes).toBe(1);
      expect(disabled.dispatches).toBe(0);
      occlusion.setEnabled(true);
      occlusion.update(4, 32, 20, moved, borrowedDepth);
      expect(new Float32Array(device.writes.at(-3)!.data.buffer)[20]).toBe(0);
      occlusion.encodePyramid(encoder().gpu);
      occlusion.update(5, 64, 20, moved, borrowedDepth);
      expect(new Float32Array(device.writes.at(-3)!.data.buffer)[20]).toBe(0);
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('writes full timestamp pairs only on the endpoints of a multi-mip sequence and both ends of a disabled pass', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    try {
      const borrowedDepth = depth();
      // Slots 0..3 are measured; slot 4 is the reserved scratch slot.
      const querySet = createTimestampQuerySet({ createQuerySet: () => ({}) } as unknown as GPUDevice, 4);
      const timestamps = { querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
      occlusion.update(1, 32, 20, identity(), borrowedDepth);
      const active = encoder();
      occlusion.encodePyramid(active.gpu, timestamps);
      expect(active.passes).toBeGreaterThan(2);
      // No pass writes only one end (that loses the device on Apple/Metal); the end a pass does not own goes to scratch.
      expect(active.descriptors[0].timestampWrites).toEqual({ querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 4 });
      for (const middle of active.descriptors.slice(1, -1)) expect(middle.timestampWrites).toBeUndefined();
      expect(active.descriptors.at(-1)!.timestampWrites).toEqual({ querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 3 });

      occlusion.setEnabled(false);
      occlusion.update(2, 32, 20, identity(), borrowedDepth);
      const disabled = encoder();
      occlusion.encodePyramid(disabled.gpu, timestamps);
      expect(disabled.passes).toBe(1);
      expect(disabled.dispatches).toBe(0);
      expect(disabled.descriptors[0].timestampWrites).toEqual(timestamps);
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('temporarily suppresses both prediction and recovery without changing user enablement', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const occlusion = await GpuOcclusion.create(device as unknown as GPUDevice, {}, tracked.account);
    try {
      const borrowedDepth = depth();
      occlusion.update(1, 32, 20, identity(), borrowedDepth);
      occlusion.encodePyramid(encoder().gpu);
      expect(occlusion.diagnostics()).toMatchObject({ enabled: true, activeForView: true, historyValid: true });

      occlusion.update(2, 32, 20, identity(), borrowedDepth, false);
      const suppressedWrites = device.writes.slice(-3);
      expect(new Float32Array(suppressedWrites[0].data.buffer)[20]).toBe(0); // Previous prediction.
      expect(new Float32Array(suppressedWrites[1].data.buffer)[20]).toBe(0); // Current recovery.
      expect(occlusion.diagnostics()).toMatchObject({ enabled: true, activeForView: false });
      const suppressed = encoder();
      occlusion.encodePyramid(suppressed.gpu);
      expect(suppressed.passes).toBe(1);
      expect(suppressed.dispatches).toBe(0);
      expect(occlusion.diagnostics()).toMatchObject({ enabled: true, historyValid: false });

      occlusion.update(3, 32, 20, identity(), borrowedDepth, true);
      const resumedWrites = device.writes.slice(-3);
      expect(new Float32Array(resumedWrites[0].data.buffer)[20]).toBe(0); // Stale pyramid cannot predict.
      expect(new Float32Array(resumedWrites[1].data.buffer)[20]).toBe(1); // Current depth may recover.
      const resumed = encoder();
      occlusion.encodePyramid(resumed.gpu);
      expect(resumed.dispatches).toBe(planOcclusionPyramid(32, 20, {}).mipLevelCount);
      expect(occlusion.diagnostics()).toMatchObject({ enabled: true, activeForView: true, historyValid: true });
    } finally { occlusion.dispose(); }
    expect(tracked.bytes).toBe(0);
  });
});
