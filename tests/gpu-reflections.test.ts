import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuReflections, type ReflectionUpdate } from '../src/gpu-reflections';
import type { GpuWorldQuery } from '../src/gpu-world-query';
import type { GpuDiffuseGi } from '../src/gpu-diffuse-gi';
import type { GpuVolumetricFog } from '../src/gpu-volumetric-fog';

type FakeBuffer = { label: string; destroyed: boolean; destroy(): void };

function buffer(label: string): FakeBuffer {
  return { label, destroyed: false, destroy() { this.destroyed = true; } };
}

type FakeTexture = { label: string; size: number[]; destroyed: boolean; destroy(): void; createView(): object };

class FakeDevice {
  limits = { maxStorageBuffersPerShaderStage: 7, maxBufferSize: 128 * 1024 * 1024,
    maxStorageBufferBindingSize: 128 * 1024 * 1024, maxComputeWorkgroupsPerDimension: 65_535, maxTextureDimension2D: 8192 };
  readonly buffers: FakeBuffer[] = [];
  readonly textures: FakeTexture[] = [];
  createTexture = ({ label, size }: { label: string; size: number[] }): GPUTexture => {
    const created: FakeTexture = { label, size, destroyed: false, destroy() { this.destroyed = true; }, createView: () => ({}) };
    this.textures.push(created);
    return created as unknown as GPUTexture;
  };
  readonly writes: { label: string; bytes: number }[] = [];
  bufferCalls = 0;
  bindGroupCalls = 0;
  failBufferCall = -1;
  failBindGroupCall = -1;
  failPipeline = false;
  shaderErrors = false;
  createBuffer = ({ label }: { label: string }): GPUBuffer => {
    this.bufferCalls++;
    if (this.bufferCalls === this.failBufferCall) {
      this.failBufferCall = -1;
      throw new Error('FAKE_BUFFER_FAILURE');
    }
    const created = buffer(label);
    this.buffers.push(created);
    return created as unknown as GPUBuffer;
  };
  createShaderModule = () => ({ getCompilationInfo: async () => ({ messages: this.shaderErrors
    ? [{ type: 'error', lineNum: 17, message: 'invalid fake shader' }] : [] }) });
  createComputePipelineAsync = async () => {
    if (this.failPipeline) throw new Error('FAKE_PIPELINE_FAILURE');
    return { getBindGroupLayout: () => ({}) };
  };
  createBindGroup = () => {
    this.bindGroupCalls++;
    if (this.bindGroupCalls === this.failBindGroupCall) throw new Error('FAKE_BIND_GROUP_FAILURE');
    return {};
  };
  createSampler = () => ({});
  queue = { writeBuffer: (target: FakeBuffer, _offset: number, data: ArrayBufferView) => {
    this.writes.push({ label: target.label, bytes: data.byteLength });
  } };
}

function ledger() {
  let bytes = 0;
  const balances: number[] = [];
  return { account: (delta: number) => { bytes += delta; balances.push(bytes); },
    get bytes() { return bytes; }, get balances() { return balances; } };
}

function borrowed(maxRays = 16, maxTotalWorkUnits = 16, maxTraceParameterSetsPerEncoder = 8) {
  const ray = buffer('borrowed query rays'), hit = buffer('borrowed query hits');
  const query = { options: { maxRays, maxTotalWorkUnits, maxTraceParameterSetsPerEncoder },
    rayBuffer: ray, hitBuffer: hit, encodeTrace: vi.fn(() => 16) } as unknown as GpuWorldQuery;
  return { query, ray, hit };
}

function lightingAndFog() {
  const parameters = buffer('borrowed GI parameters');
  const probes = [buffer('borrowed GI probes 0'), buffer('borrowed GI probes 1')];
  const worldParameters = buffer('borrowed fog parameters');
  const worldVolumes = buffer('borrowed fog volumes');
  const shadow = { createView: () => ({}) } as unknown as GPUTexture;
  const gi = { options: { finalGather: false }, lightingParameters: parameters,
    lightingProbeBuffers: probes, lightingIndex: 0 } as unknown as GpuDiffuseGi;
  const fog = { worldParameters, worldVolumes } as unknown as GpuVolumetricFog;
  return { gi, fog, shadow, borrowed: [parameters, ...probes, worldParameters, worldVolumes] };
}

const frame: ReflectionUpdate = {
  frame: 1, materialRevision: 1, width: 32, height: 24,
  eye: [0, 1, 5], target: [0, 1, 0], fovY: Math.PI / 3, near: 0.1, far: 100,
  sunDirection: [0, 1, -1], sunColor: [2, 2, 2],
};

beforeEach(() => vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, STORAGE: 4,
  COPY_SRC: 8, MAP_READ: 16 }));
afterEach(() => vi.unstubAllGlobals());

describe('GPU reflections ownership and preflight', () => {
  it('rejects impossible query, parameter, storage, fog and memory budgets before allocating', async () => {
    for (const configure of [
      () => ({ query: borrowed(3, 16).query, options: {}, error: 'REFLECTIONS_QUERY_RAY_BUDGET' }),
      () => ({ query: borrowed(16, 16, 1).query, options: {}, error: 'REFLECTIONS_QUERY_PARAMETER_BUDGET' }),
      () => ({ query: borrowed().query, options: {}, error: 'REFLECTIONS_STORAGE_BINDING_LIMIT', storage: 4 }),
      () => ({ query: borrowed().query, options: { maxBytes: 1 }, error: 'REFLECTIONS_MEMORY_BUDGET' }),
    ]) {
      const device = new FakeDevice(), tracked = ledger(), test = configure();
      const storage = (test as { storage?: number }).storage;
      if (storage !== undefined) device.limits.maxStorageBuffersPerShaderStage = storage;
      await expect(GpuReflections.create(device as unknown as GPUDevice, test.query,
        test.options, tracked.account)).rejects.toThrow(test.error);
      expect(device.buffers).toHaveLength(0);
      expect(tracked.bytes).toBe(0);
    }
    const device = new FakeDevice(), tracked = ledger(), query = borrowed().query;
    const { fog } = lightingAndFog();
    await expect(GpuReflections.create(device as unknown as GPUDevice, query, {},
      tracked.account, undefined, fog)).rejects.toThrow('REFLECTIONS_FOG_SHADOW_REQUIRED');
    expect(device.buffers).toHaveLength(0);
    expect(tracked.bytes).toBe(0);
    for (const [limit, value, error] of [
      ['maxBufferSize', 100, 'REFLECTIONS_BUFFER_LIMIT'],
      ['maxStorageBufferBindingSize', 100, 'REFLECTIONS_BUFFER_LIMIT'],
      ['maxComputeWorkgroupsPerDimension', 0, 'REFLECTIONS_DISPATCH_LIMIT'],
    ] as const) {
      const limited = new FakeDevice(), balance = ledger();
      limited.limits[limit] = value;
      await expect(GpuReflections.create(limited as unknown as GPUDevice,
        borrowed().query, {}, balance.account)).rejects.toThrow(error);
      expect(limited.buffers).toHaveLength(0);
      expect(balance.bytes).toBe(0);
    }
  });

  it.each([1, 2, 3, 4, 5, 6])('cleans partial buffer allocation at call %i', async failedCall => {
    const device = new FakeDevice(), tracked = ledger(), { query, ray, hit } = borrowed();
    device.failBufferCall = failedCall;
    await expect(GpuReflections.create(device as unknown as GPUDevice, query, {}, tracked.account))
      .rejects.toThrow('FAKE_BUFFER_FAILURE');
    expect(device.buffers).toHaveLength(failedCall - 1);
    expect(device.buffers.every(item => item.destroyed)).toBe(true);
    expect(tracked.bytes).toBe(0);
    expect(tracked.balances.every(bytes => bytes >= 0)).toBe(true);
    expect(ray.destroyed).toBe(false);
    expect(hit.destroyed).toBe(false);
    const recovered = await GpuReflections.create(device as unknown as GPUDevice, query, {}, tracked.account);
    recovered.dispose();
    expect(tracked.bytes).toBe(0);
    expect(device.buffers.every(item => item.destroyed)).toBe(true);
  });

  it.each(['shader', 'pipeline', 'bindGroup'] as const)(
    'cleans owned buffers after %s initialization failure without releasing borrowed GI/Fog', async failure => {
      const device = new FakeDevice(), tracked = ledger(), { query, ray, hit } = borrowed();
      const lights = lightingAndFog();
      if (failure === 'shader') device.shaderErrors = true;
      if (failure === 'pipeline') device.failPipeline = true;
      if (failure === 'bindGroup') device.failBindGroupCall = 6; // After two GI integrate groups, at fog group.
      await expect(GpuReflections.create(device as unknown as GPUDevice, query, {}, tracked.account,
        lights.gi, lights.fog, lights.shadow)).rejects.toThrow(failure === 'shader'
        ? 'invalid fake shader' : failure === 'pipeline' ? 'FAKE_PIPELINE_FAILURE' : 'FAKE_BIND_GROUP_FAILURE');
      expect(device.buffers).toHaveLength(6);
      expect(device.buffers.every(item => item.destroyed)).toBe(true);
      expect(tracked.bytes).toBe(0);
      expect(tracked.balances.every(bytes => bytes >= 0)).toBe(true);
      expect([ray, hit, ...lights.borrowed].every(item => !item.destroyed)).toBe(true);
    });

  it('owns exactly six buffers, accounts once, and disposes idempotently without touching borrowed resources', async () => {
    const device = new FakeDevice(), tracked = ledger(), { query, ray, hit } = borrowed();
    const lights = lightingAndFog();
    const reflections = await GpuReflections.create(device as unknown as GPUDevice, query, {}, tracked.account,
      lights.gi, lights.fog, lights.shadow);
    expect(device.buffers).toHaveLength(6);
    expect(reflections.diagnostics()).toMatchObject({ diffuseGi: true, worldMedia: true,
      ownedGpuBytes: tracked.bytes, capacity: 4 });
    expect(tracked.bytes).toBeGreaterThan(0);
    reflections.dispose();
    const balancesAfterDispose = [...tracked.balances];
    reflections.dispose();
    expect(tracked.balances).toEqual(balancesAfterDispose);
    expect(tracked.bytes).toBe(0);
    expect(device.buffers.every(item => item.destroyed)).toBe(true);
    expect([ray, hit, ...lights.borrowed].every(item => !item.destroyed)).toBe(true);
    expect(() => reflections.setEnabled(true)).toThrow('REFLECTIONS_DISPOSED');
  });

  it('rejects malformed frame updates and encodes a disabled frame without dispatch or query work', async () => {
    const device = new FakeDevice(), tracked = ledger(), { query } = borrowed();
    const reflections = await GpuReflections.create(device as unknown as GPUDevice, query, {}, tracked.account);
    try {
      expect(() => reflections.encode({} as GPUCommandEncoder)).toThrow('REFLECTIONS_NO_FRAME');
      for (const patch of [{ frame: 0 }, { materialRevision: -1 }, { width: 0 },
        { eye: [0, NaN, 1] }, { sunColor: [1e21, 0, 0] }, { sunDirection: [0, 0, 0] },
        { sunDirection: [1.5e308, 1.5e308, 1.5e308] },
        { fovY: 0 }, { near: 0 }, { far: 0.05 }]) {
        expect(() => reflections.update({ ...frame, ...patch })).toThrow();
      }
      expect(device.writes).toHaveLength(0);
      expect(() => reflections.setEnabled(1 as unknown as boolean)).toThrow('REFLECTIONS_ENABLED');
      reflections.setEnabled(false);
      expect(reflections.update(frame)).toBe(304);
      expect(device.writes).toEqual([{ label: 'reflection parameters', bytes: 304 }]);
      let passes = 0, dispatches = 0;
      const encoder = { beginComputePass: ({ label }: { label: string }) => {
        expect(label).toBe('reflections disabled'); passes++;
        return { dispatchWorkgroups: () => { dispatches++; }, end() {} };
      } } as unknown as GPUCommandEncoder;
      expect(reflections.encode(encoder)).toBe(0);
      expect(passes).toBe(1);
      expect(dispatches).toBe(0);
      expect(query.encodeTrace).not.toHaveBeenCalled();
    } finally { reflections.dispose(); }
    expect(tracked.bytes).toBe(0);
  });
});

function fakeEncoder() {
  const labels: string[] = [];
  const encoder = { beginComputePass: ({ label }: { label: string }) => {
    labels.push(label);
    return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  }, copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
  return { encoder, labels };
}

const identityMatrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const moved = identityMatrix.map((value, i) => i === 12 ? 0.5 : value);

describe('GPU reflection history', () => {
  it('allocates two budgeted history buffers only when configured and rejects a budget without room for them', async () => {
    const device = new FakeDevice(), tracked = ledger();
    const plain = await GpuReflections.create(device as unknown as GPUDevice, borrowed().query, {}, tracked.account);
    const plainBytes = tracked.bytes;
    plain.dispose();
    const temporal = await GpuReflections.create(device as unknown as GPUDevice, borrowed().query,
      { temporal: {} }, tracked.account);
    try {
      expect(device.buffers.filter(item => item.label.startsWith('reflection reprojected history'))).toHaveLength(2);
      expect(tracked.bytes - plainBytes).toBe(2 * 4 * 96);
      expect(temporal.diagnostics().temporal).toMatchObject({ configured: true, enabled: true, historyBytes: 768,
        minHistoryFrames: 8, maxHistoryFrames: 32 });
    } finally { temporal.dispose(); }
    expect(tracked.bytes).toBe(0);
    const tight = new FakeDevice(), balance = ledger();
    await expect(GpuReflections.create(tight as unknown as GPUDevice, borrowed().query,
      { maxBytes: plainBytes, temporal: {} }, balance.account)).rejects.toThrow('REFLECTIONS_MEMORY_BUDGET');
    expect(tight.buffers).toHaveLength(0);
    expect(balance.bytes).toBe(0);
  });

  it('names every discrete history reset and keeps continuous frames accumulating', async () => {
    const device = new FakeDevice(), tracked = ledger(), { query } = borrowed();
    const reflections = await GpuReflections.create(device as unknown as GPUDevice, query, { temporal: {} }, tracked.account);
    try {
      let index = 0;
      const step = (patch: Partial<ReflectionUpdate> = {}, encode = true) => {
        index++;
        reflections.update({ ...frame, frame: index, viewProjection: identityMatrix,
          previousViewProjection: index > 1 ? identityMatrix : undefined, sceneMotion: false, historyRevision: 0, ...patch });
        const { encoder, labels } = fakeEncoder();
        if (encode) reflections.encode(encoder);
        return { history: reflections.diagnostics().temporal, labels };
      };
      expect(() => reflections.update({ ...frame, viewProjection: undefined })).toThrow('REFLECTIONS_TEMPORAL_VIEW_PROJECTION');
      const first = step();
      expect(first.history).toMatchObject({ active: true, valid: true, framesSinceReset: 1, lastResetReason: 'initial' });
      expect(first.labels.at(-1)).toBe('reflections temporal');
      expect(step().history).toMatchObject({ framesSinceReset: 2, lastResetReason: 'initial' });
      expect(step({ previousViewProjection: undefined }).history).toMatchObject({ framesSinceReset: 1, lastResetReason: 'camera-cut' });
      expect(step({ previousViewProjection: moved }).history.lastResetReason).toBe('camera-cut');
      expect(step({ materialRevision: 2 }).history.lastResetReason).toBe('material-changed');
      expect(step({ materialRevision: 2, sunColor: [2, 2, 1.5] }).history.lastResetReason).toBe('lighting-changed');
      const stable = step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 0 });
      expect(stable.history).toMatchObject({ framesSinceReset: 2, lastResetReason: 'lighting-changed' });
      expect(step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1 }).history.lastResetReason).toBe('scene-identity-changed');
      expect(step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1, width: 40 }).history.lastResetReason).toBe('grid-changed');
      step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1, width: 40 }, false);
      expect(step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1, width: 40 }).history.lastResetReason).toBe('unrendered-history');
      reflections.invalidateHistory('gi-changed');
      expect(step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1, width: 40 }).history.lastResetReason).toBe('gi-changed');
      reflections.setTemporalEnabled(false);
      const off = step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1, width: 40 });
      expect(off.history).toMatchObject({ active: false, valid: false });
      expect(off.labels).not.toContain('reflections temporal');
      reflections.setTemporalEnabled(true);
      expect(step({ materialRevision: 2, sunColor: [2, 2, 1.5], historyRevision: 1, width: 40 }).history)
        .toMatchObject({ active: true, framesSinceReset: 1, lastResetReason: 'temporal-toggle' });
      expect(device.writes.every(write => write.bytes === 304)).toBe(true);
    } finally { reflections.dispose(); }
    expect(tracked.bytes).toBe(0);
    const untimed = await GpuReflections.create(device as unknown as GPUDevice, borrowed().query, {}, tracked.account);
    expect(() => untimed.setTemporalEnabled(true)).toThrow('REFLECTIONS_TEMPORAL_NOT_CONFIGURED');
    untimed.dispose();
  });
});

describe('GPU reflection screen-space supplement', () => {
  beforeEach(() => vi.stubGlobal('GPUTextureUsage', { TEXTURE_BINDING: 4, COPY_DST: 2 }));

  it('owns budgeted previous-frame copies, names every inactive state and stays world-only over budget', async () => {
    const device = new FakeDevice(), tracked = ledger(), { query } = borrowed();
    const reflections = await GpuReflections.create(device as unknown as GPUDevice, query, { screenTrace: {} }, tracked.account);
    const hdr = { label: 'renderer HDR' } as unknown as GPUTexture, depth = { label: 'renderer depth' } as unknown as GPUTexture;
    const copies: unknown[][] = [];
    const encoder = { copyTextureToTexture: (...args: unknown[]) => { copies.push(args); } } as unknown as GPUCommandEncoder;
    const step = (patch: Partial<ReflectionUpdate>) => {
      reflections.update({ ...frame, viewProjection: identityMatrix, ...patch });
      return reflections.diagnostics().screenTrace;
    };
    try {
      expect(device.textures.map(texture => texture.label)).toEqual(['reflection previous HDR', 'reflection previous depth']);
      expect(reflections.screenTraceConfigured).toBe(true);
      expect(reflections.diagnostics().screenTrace).toMatchObject({ configured: true, enabled: true, active: false,
        state: 'awaiting-resize', captureBytes: 12, maxDistance: 1.5, steps: 12 });
      const before = tracked.bytes;
      expect(step({ frame: 1 }).state).toBe('awaiting-resize');
      reflections.resize(32, 24, hdr, depth);
      expect(tracked.bytes - before).toBe(32 * 24 * 12 - 12);
      expect(device.textures.slice(0, 2).every(texture => texture.destroyed)).toBe(true);
      expect(step({ frame: 2 }).state).toBe('awaiting-capture');
      reflections.encodeCapture(encoder);
      expect(copies.map(args => [(args[0] as { texture: unknown }).texture, args[2]])).toEqual([[hdr, [32, 24]], [depth, [32, 24]]]);
      expect(step({ frame: 3, previousViewProjection: identityMatrix })).toMatchObject({ state: 'active', active: true });
      reflections.encodeCapture(encoder);
      expect(step({ frame: 4, viewProjection: moved, previousViewProjection: moved }).state).toBe('camera-cut');
      reflections.encodeCapture(encoder);
      expect(step({ frame: 5, viewProjection: moved }).state).toBe('camera-cut');
      reflections.setScreenTraceEnabled(false);
      expect(step({ frame: 6 })).toMatchObject({ state: 'disabled', active: false });
      reflections.setScreenTraceEnabled(true);
      expect(step({ frame: 7, width: 40 }).state).toBe('awaiting-resize');
      // 2048^2 copies (48 MiB) exceed the default 32 MiB reflection budget.
      reflections.resize(2048, 2048, hdr, depth);
      expect(reflections.diagnostics().screenTrace).toMatchObject({ captureBytes: 12, width: 1, height: 1 });
      expect(step({ frame: 8 }).state).toBe('over-budget');
      const copied = copies.length;
      reflections.encodeCapture(encoder);
      expect(copies).toHaveLength(copied);
      expect(() => reflections.resize(0, 24, hdr, depth)).toThrow('REFLECTIONS_SCREEN_SIZE');
      expect(() => reflections.setScreenTraceEnabled(1 as never)).toThrow('REFLECTIONS_SCREEN_TRACE_ENABLED');
      expect(device.writes.every(write => write.bytes === 304)).toBe(true);
    } finally { reflections.dispose(); }
    expect(tracked.bytes).toBe(0);
    expect(device.textures.every(texture => texture.destroyed)).toBe(true);
    const plain = await GpuReflections.create(device as unknown as GPUDevice, borrowed().query, {}, tracked.account);
    expect(() => plain.setScreenTraceEnabled(true)).toThrow('REFLECTIONS_SCREEN_TRACE_NOT_CONFIGURED');
    expect(plain.diagnostics().screenTrace).toMatchObject({ configured: false, state: 'not-configured' });
    plain.resize(32, 24, hdr, depth);
    expect(device.textures).toHaveLength(6);
    plain.dispose();
    expect(tracked.bytes).toBe(0);
  });
});
