import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuVolumetricFog } from '../src/gpu-volumetric-fog';
import { createTimestampQuerySet } from '../src/gpu-timestamp-span';
import { identity } from '../src/math';

type FakeBuffer = { label: string; destroyed: boolean; destroy(): void };
type FakeTexture = { label: string; destroyed: boolean; createView(): object; destroy(): void };

class FakeDevice {
  readonly limits = { maxStorageBuffersPerShaderStage: 5, maxTextureDimension2D: 4096,
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
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8 });
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
  it('preflights the extra temporal storage bindings before allocating resources', async () => {
    const device = new FakeDevice(), tracked = ledger();
    device.limits.maxStorageBuffersPerShaderStage = 4;
    await expect(GpuVolumetricFog.create(device as unknown as GPUDevice,
      { temporal: {} }, tracked.account)).rejects.toThrow('FOG_STORAGE_BINDING_LIMIT');
    expect(device.buffers).toHaveLength(0);
    expect(tracked.bytes).toBe(0);
    // The non-temporal path continues to fit the three-binding baseline.
    device.limits.maxStorageBuffersPerShaderStage = 3;
    const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice, {}, tracked.account);
    fog.dispose();
    expect(tracked.bytes).toBe(0);
    expect(allDestroyed(device)).toBe(true);
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
        expect(tracked.bytes).toBe(5504);
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

  it.each([5, 6, 7, 8])('releases every partially allocated temporal history buffer when allocation %i fails',
    async failedCall => {
      const device = new FakeDevice(), tracked = ledger();
      const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice,
        { slices: 8, temporal: { maxHistoryFrames: 6 } }, tracked.account);
      try {
        device.failBufferCall = failedCall;
        expect(() => resize(fog, 32, 32)).toThrow('FAKE_BUFFER_FAILURE');
        expect(fog.diagnostics().grid).toBeNull();
        expect(fog.colorTexture).toBeUndefined();
        expect(tracked.bytes).toBe(5504);
        expect(device.buffers.filter(buffer => buffer.label.startsWith('fog history'))
          .every(buffer => buffer.destroyed)).toBe(true);
        expect(device.buffers.filter(buffer => buffer.label === 'fog source and extinction'
          || buffer.label === 'fog cumulative radiance and transmission')
          .every(buffer => buffer.destroyed)).toBe(true);
        expect(device.textures.every(texture => texture.destroyed)).toBe(true);
        expect(tracked.balances.every(bytes => bytes >= 0)).toBe(true);

        resize(fog, 32, 32);
        expect(fog.diagnostics().grid?.bytes).toBe(tracked.bytes);
        expect(device.buffers.filter(buffer => buffer.label.startsWith('fog history') && !buffer.destroyed)).toHaveLength(4);
      } finally { fog.dispose(); }
      expect(tracked.bytes).toBe(0);
      expect(allDestroyed(device)).toBe(true);
    });

  it('cleans temporal history after bind-group initialization fails, then retries', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice, { temporal: {} }, tracked.account);
    try {
      device.failBindGroupCall = device.bindGroupCalls + 3;
      expect(() => resize(fog, 32, 32)).toThrow('FAKE_BIND_GROUP_FAILURE');
      expect(fog.diagnostics().grid).toBeNull();
      expect(tracked.bytes).toBe(5504);
      expect(device.buffers.filter(buffer => buffer.label.startsWith('fog history')).every(buffer => buffer.destroyed)).toBe(true);
      expect(device.textures.every(texture => texture.destroyed)).toBe(true);
      resize(fog, 32, 32);
      expect(fog.diagnostics().grid?.bytes).toBe(tracked.bytes);
    } finally { fog.dispose(); }
    expect(tracked.bytes).toBe(0);
    expect(allDestroyed(device)).toBe(true);
  });

  it('accounts for temporal history bytes and records history only after submission', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice,
      { pixelSize: 8, slices: 8, temporal: { maxHistoryFrames: 4 }, maxBytes: 7680 }, tracked.account);
    try {
      expect(fog.planResize(16, 8)).toEqual({ columns: 2, rows: 1, slices: 8,
        pixelSize: 8, froxels: 16, bytes: 7680 });
      resize(fog, 16, 8);
      expect(tracked.bytes).toBe(7680);
      const camera = identity();
      const update = () => fog.update([0, 1, 5], [0, 1, 0], Math.PI / 3, 0.1, 100,
        [0, 1, -1], [2, 2, 2], camera, camera,
        { cameraCut: false, sceneChanged: false, materialRevision: 1 });
      expect(update()).toBe(384 + 64 * 80);
      expect(fog.diagnostics().history).toMatchObject({ configured: true, valid: false, framesSinceReset: 0, maxFrames: 4 });
      fog.afterSubmit();
      expect(fog.diagnostics().history).toMatchObject({ configured: true, valid: true, framesSinceReset: 1 });
      expect(update()).toBe(384);
      fog.afterSubmit();
      expect(fog.diagnostics().history.framesSinceReset).toBe(2);
      fog.invalidateHistory('test-cut');
      expect(fog.diagnostics().history).toMatchObject({ valid: false, framesSinceReset: 0, lastResetReason: 'test-cut' });
    } finally { fog.dispose(); }
    expect(tracked.bytes).toBe(0);
    expect(allDestroyed(device)).toBe(true);

    const constrained = new FakeDevice(), constrainedLedger = ledger();
    const tight = await GpuVolumetricFog.create(constrained as unknown as GPUDevice,
      { pixelSize: 8, slices: 8, temporal: {}, maxBytes: 7679 }, constrainedLedger.account);
    try {
      expect(tight.planResize(16, 8)).toEqual({ columns: 1, rows: 1, slices: 8,
        pixelSize: 16, froxels: 8, bytes: 7104 });
    } finally { tight.dispose(); }
    expect(constrainedLedger.bytes).toBe(0);
    expect(allDestroyed(constrained)).toBe(true);
  });
});

type Write = { label: string; offset: number; data: Float32Array };
function recordWrites(device: FakeDevice): Write[] {
  const writes: Write[] = [];
  device.queue.writeBuffer = ((buffer: FakeBuffer, offset: number, data: ArrayBufferView) => {
    writes.push({ label: buffer.label, offset,
      data: new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)) });
  }) as never;
  return writes;
}
function fakeGi(device: FakeDevice) {
  const probes = [device.createBuffer({ label: 'gi probes 0' }), device.createBuffer({ label: 'gi probes 1' })];
  const parameters = device.createBuffer({ label: 'gi parameters' });
  return { lightingProbeBuffers: probes, lightingParameters: parameters, lightingIndex: 0, historyEpoch: 0 };
}
const view = identity();
// Orthographic sun: NDC x = x / 10, y = z / 10, depth = 0.5 - y / 20.
const sun = [0.1, 0, 0, 0, 0, 0, -0.05, 0, 0, 0.1, 0, 0, 0, 0, 0.5, 1];

describe('GPU volumetric fog air lighting and local history rejection', () => {
  it('requires a borrowed GI field and preflights its extra storage binding', async () => {
    const device = new FakeDevice(), tracked = ledger();
    await expect(GpuVolumetricFog.create(device as unknown as GPUDevice, { airLighting: {} }, tracked.account))
      .rejects.toThrow('FOG_AIR_LIGHTING_REQUIRES_DIFFUSE_GI');
    expect(device.buffers).toHaveLength(0);
    const gi = fakeGi(device);
    device.limits.maxStorageBuffersPerShaderStage = 3;
    await expect(GpuVolumetricFog.create(device as unknown as GPUDevice, { airLighting: {} }, tracked.account,
      gi as never)).rejects.toThrow('FOG_STORAGE_BINDING_LIMIT');
    expect(device.buffers).toHaveLength(3);
    expect(tracked.bytes).toBe(0);
    device.limits.maxStorageBuffersPerShaderStage = 4;
    const groups = device.bindGroupCalls;
    const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice, { airLighting: {} }, tracked.account, gi as never);
    try {
      expect(fog.airLightingConfigured).toBe(true);
      // One borrowed GI bind group per GI ping-pong probe buffer.
      expect(device.bindGroupCalls - groups).toBe(2);
      expect(fog.diagnostics().airLighting).toMatchObject({ gi: true, outsideSkyVisibility: 0, configured: true,
        representation: 'gi-incident-radiance-sh9-hg-zonal', evaluation: 'cell-centre-each-frame-after-history' });
      expect(() => fog.reflectionShaderSource(false)).toThrow('FOG_AIR_LIGHTING_REFLECTION_GI_REQUIRED');
      expect(fog.reflectionShaderSource(true)).toContain('fn fogAirField(');
    } finally { fog.dispose(); }
    expect(tracked.bytes).toBe(0);
    // GI owns its buffers; fog destroys only its own.
    expect(device.buffers.filter(buffer => buffer.label.startsWith('gi ')).every(buffer => !buffer.destroyed)).toBe(true);
    expect(device.buffers.filter(buffer => buffer.label.startsWith('fog ')).every(buffer => buffer.destroyed)).toBe(true);

    const plain = await GpuVolumetricFog.create(device as unknown as GPUDevice, {}, tracked.account, gi as never);
    try {
      // Without air lighting the GI field is not borrowed and reflections use the stub.
      expect(plain.airLightingConfigured).toBe(false);
      expect(plain.reflectionShaderSource(false))
        .toContain('fn fogAirSource(world: vec3f, direction: vec3f) -> vec3f { return vec3f(0.0); }');
      expect(() => plain.setAirLighting({ gi: false })).toThrow('FOG_AIR_LIGHTING_NOT_CONFIGURED');
      expect(plain.diagnostics().airLighting).toBeNull();
    } finally { plain.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('packs air lanes and the split debug view, keeps air light out of history, and splits the integrate timestamp span', async () => {
    const device = new FakeDevice(), tracked = ledger(), writes = recordWrites(device), gi = fakeGi(device);
    const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice,
      { pixelSize: 8, slices: 8, temporal: {}, airLighting: { outsideSkyVisibility: 0.5 } }, tracked.account, gi as never);
    try {
      resize(fog, 16, 16);
      const frame = () => {
        fog.update([0, 1, 5], [0, 1, 0], Math.PI / 3, 0.1, 100, [0, 1, 0], [1, 1, 1], view, view,
          { cameraCut: false, sceneChanged: false, materialRevision: 1 });
        const uniform = writes.filter(write => write.label === 'fog frame').at(-1)!.data;
        fog.afterSubmit();
        return Array.from(uniform.subarray(92, 96));
      };
      expect(frame()).toEqual([0, 1, 0.5, 0]);
      expect(frame()).toEqual([0, 1, 0.5, 0]);
      expect(fog.diagnostics().history.framesSinceReset).toBe(2);
      fog.setDebugView('scattering-transmittance');
      expect(frame()[0]).toBe(1);
      expect(fog.diagnostics().debugView).toBe('scattering-transmittance');
      expect(() => fog.setDebugView('depth' as never)).toThrow('INVALID_FOG_DEBUG_VIEW');
      fog.setDebugView('composite');
      // Air light is evaluated after the temporal resolve: toggling it keeps history.
      fog.setAirLighting({ gi: false });
      expect(frame()).toEqual([0, 0, 0.5, 0]);
      expect(fog.diagnostics().history).toMatchObject({ framesSinceReset: 4, lastResetReason: 'resize' });
      expect(() => fog.setAirLighting({ outsideSkyVisibility: 2 })).toThrow(/outsideSkyVisibility/);
      expect(fog.diagnostics().airLighting).toMatchObject({ gi: false, outsideSkyVisibility: 0.5,
        evaluation: 'cell-centre-each-frame-after-history' });

      // One timestamp span: begins on the air pass and ends on the integration pass.
      const passes: { label: string; timestampWrites?: Record<string, unknown> }[] = [];
      const groups: number[] = [];
      const encoder = { beginComputePass: (descriptor: { label: string; timestampWrites?: Record<string, unknown> }) => {
        passes.push(descriptor);
        return { setPipeline() {}, setBindGroup: (index: number) => { groups.push(index); }, dispatchWorkgroups() {}, end() {} };
      } };
      // Each pass writes both ends; the end it does not own goes to the reserved scratch slot (8).
      const querySet = createTimestampQuerySet({ createQuerySet: (descriptor: GPUQuerySetDescriptor) => ({ count: descriptor.count }) } as never, 8);
      fog.encodeIntegrate(encoder as never, { querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 });
      expect(passes.map(pass => pass.label)).toEqual(['fog air lighting', 'fog integration']);
      expect(passes[0].timestampWrites).toEqual({ querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 8 });
      expect(passes[1].timestampWrites).toEqual({ querySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 5 });
      expect(groups).toEqual([0, 1, 0]);
      passes.length = 0;
      fog.encodeInject(encoder as never);
      expect(passes.map(pass => pass.label)).toEqual(['fog density and lighting']);
    } finally { fog.dispose(); }
    expect(tracked.bytes).toBe(0);

    const plain = await GpuVolumetricFog.create(device as unknown as GPUDevice, { pixelSize: 8, slices: 8 }, tracked.account);
    try {
      resize(plain, 16, 16);
      const passes: { timestampWrites?: unknown }[] = [];
      const encoder = { beginComputePass: (descriptor: { timestampWrites?: unknown }) => {
        passes.push(descriptor);
        return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
      } };
      const timing = { querySet: {}, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 };
      plain.encodeIntegrate(encoder as never, timing as never);
      expect(passes).toHaveLength(1);
      expect(passes[0].timestampWrites).toBe(timing);
    } finally { plain.dispose(); }
    expect(tracked.bytes).toBe(0);
  });

  it('localizes changed casters, falls back to a global reset when it cannot, and releases the region uniform', async () => {
    const device = new FakeDevice(), tracked = ledger(), writes = recordWrites(device);
    const fog = await GpuVolumetricFog.create(device as unknown as GPUDevice,
      { pixelSize: 8, slices: 8, temporal: { occluderRejection: 'local' } }, tracked.account);
    try {
      expect(fog.localRejection).toBe(true);
      expect(tracked.bytes).toBe(5504 + 1040);
      expect(device.buffers.filter(buffer => buffer.label === 'fog changed occluder regions')).toHaveLength(1);
      resize(fog, 16, 16);
      expect(tracked.bytes).toBe(fog.diagnostics().grid!.bytes);
      let eye: [number, number, number] = [0, 1, 5];
      type State = { sceneChanged: boolean; changedBounds?: { min: number[]; max: number[] }[] | null };
      const frame = (state: State, light: number[] = sun) => {
        writes.length = 0;
        fog.update(eye, [0, 1, 0], Math.PI / 3, 0.1, 100, [0, 1, 0], [1, 1, 1], light as never, view,
          { cameraCut: false, materialRevision: 1, shadowResolution: 512, ...state });
        fog.afterSubmit();
        const regions = writes.filter(write => write.label === 'fog changed occluder regions').at(-1)?.data;
        return { regions, staticView: writes.find(write => write.label === 'fog frame')!.data[91] };
      };
      // The constructor schedules one empty header; history then grows.
      expect(Array.from(frame({ sceneChanged: false }).regions!)).toEqual([0, 0, 0, 0]);
      expect(frame({ sceneChanged: false }).regions).toBeUndefined();
      expect(fog.diagnostics().history.framesSinceReset).toBe(2);
      const moved = frame({ sceneChanged: true,
        changedBounds: [{ min: [0, 0, 0], max: [1, 1, 1] }, { min: [1, 0, 0], max: [2, 1, 1] }] });
      expect(moved.regions!.length).toBe(1040 / 4);
      expect(Array.from(moved.regions!.subarray(0, 2))).toEqual([2, 0]);
      expect(moved.staticView).toBe(1);
      expect(fog.diagnostics().history).toMatchObject({ framesSinceReset: 3, occluderRejection: 'local',
        occluderRegions: { framesWithRegions: 1, lastRegions: 2, lastMerged: 0, totalMerged: 0 } });
      // The next frame clears the regions with one empty header.
      expect(Array.from(frame({ sceneChanged: false }).regions!)).toEqual([0, 0, 0, 0]);
      expect(fog.diagnostics().history.occluderRegions).toMatchObject({ lastRegions: 0 });
      eye = [0.5, 1, 5];
      expect(frame({ sceneChanged: false }).staticView).toBe(0);
      expect(frame({ sceneChanged: false }).staticView).toBe(1);
      const projective = [...sun.slice(0, 11), 0.01, ...sun.slice(12)];
      for (const [state, light] of [
        [{ sceneChanged: true, changedBounds: null }, sun],
        [{ sceneChanged: true, changedBounds: [] }, sun],
        [{ sceneChanged: true, changedBounds: [{ min: [0, 0, 0], max: [1, 1, 1] }] }, projective],
      ] as [State, number[]][]) {
        frame({ sceneChanged: false }, light);
        frame({ sceneChanged: false }, light);
        expect(fog.diagnostics().history.framesSinceReset).toBeGreaterThan(1);
        frame(state, light);
        expect(fog.diagnostics().history).toMatchObject({ framesSinceReset: 1, lastResetReason: 'scene-change' });
      }
    } finally { fog.dispose(); }
    expect(tracked.bytes).toBe(0);
    expect(allDestroyed(device)).toBe(true);

    const global = await GpuVolumetricFog.create(device as unknown as GPUDevice, { temporal: {} }, tracked.account);
    try {
      expect(global.localRejection).toBe(false);
      expect(device.buffers.filter(buffer => buffer.label === 'fog changed occluder regions')).toHaveLength(1);
      expect(tracked.bytes).toBe(5504);
      expect(global.diagnostics().history).toMatchObject({ occluderRejection: 'global', occluderRegions: null });
    } finally { global.dispose(); }
    expect(tracked.bytes).toBe(0);
  });
});
