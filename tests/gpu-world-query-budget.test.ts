import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SceneBatch } from '../src/contracts';
import { createPlane } from '../src/geometry';
import { GpuWorldQuery } from '../src/gpu-world-query';
import { identity } from '../src/math';

interface FakeBuffer {
  label: string;
  size: number;
  usage: number;
  bytes: ArrayBuffer;
  destroy(): void;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

async function setup(maxNodeVisits = 1000, sceneBvh = false) {
  const buffers: FakeBuffer[] = [];
  const writeBuffer = vi.fn((buffer: FakeBuffer, offset: number, data: ArrayBuffer | ArrayBufferView,
    dataOffset = 0, size?: number) => {
    const view = ArrayBuffer.isView(data);
    const elementBytes = view && 'BYTES_PER_ELEMENT' in data ? Number(data.BYTES_PER_ELEMENT) : 1;
    const source = new Uint8Array(view ? data.buffer : data,
      (view ? data.byteOffset : 0) + dataOffset * elementBytes,
      size === undefined ? data.byteLength - dataOffset * elementBytes : size * elementBytes);
    new Uint8Array(buffer.bytes, offset, source.length).set(source);
  });
  const pipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 16 * 1024 * 1024,
      maxBufferSize: 16 * 1024 * 1024 },
    queue: { writeBuffer },
    createBuffer: ({ label = '', size, usage }: GPUBufferDescriptor) => {
      const bytes = new ArrayBuffer(size);
      const buffer = { label, size, usage, bytes, destroy: vi.fn(), getMappedRange: () => bytes, unmap() {} };
      buffers.push(buffer); return buffer;
    },
    createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}), createBindGroup: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: async () => pipeline,
  } as unknown as GPUDevice;
  const query = await GpuWorldQuery.create(device, {} as GPUBindGroupLayout, () => {}, {
    maxRays: 8, maxBatches: 8, maxTotalWorkUnits: 100, maxNodeVisits, sceneAcceleration: sceneBvh ? 'bvh' : 'linear',
  });
  const scene: SceneBatch = { name: 'budget-plane', mesh: createPlane(), capacity: 1,
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    instances: [{ entityId: 1, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
      color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 }] };
  for (let index = 0; index < (sceneBvh ? 8 : 1); index++) query.addBatch({
    scene: { ...scene, name: `budget-plane-${index}`, instances: index ? [] : scene.instances },
    output: {} as GPUBuffer, instances: {} as GPUBuffer,
    getPipeline: () => pipeline as unknown as GPUComputePipeline,
    getMaterial: () => ({ bindGroup: {} }) as ReturnType<Parameters<typeof query.addBatch>[0]['getMaterial']> });
  query.updateSnapshot(1, 2);
  const pass = { setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(), end: vi.fn() };
  const encoder = { beginComputePass: vi.fn(() => pass) };
  writeBuffer.mockClear();
  const quotas = () => writeBuffer.mock.calls.filter(([buffer, , data]) => data.byteLength === 96)
    .map(([buffer, offset]) => new Uint32Array(buffer.bytes, offset, 24)[9]);
  return { query, buffers, writeBuffer, encoder, quotas };
}

beforeEach(() => {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8, MAP_READ: 16 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
});
afterEach(() => vi.unstubAllGlobals());

describe('world query per-trace work budget', () => {
  it('records matching scene/triangle partitions once and isolates different block layouts in immutable slots', async () => {
    const { query, writeBuffer, encoder } = await setup(1000, true);
    try {
      const gpuEncoder = encoder as unknown as GPUCommandEncoder;
      const partition = { raysPerBlock: 4, primaryRaysPerBlock: 2, supplementalMaxWorkUnits: 40 };
      query.encodeTrace(gpuEncoder, 8, 0, 2, 100, partition);
      const writes = () => writeBuffer.mock.calls.filter(([buffer]) =>
        ['world batch parameter slots', 'world query scene candidate parameter slots'].includes(buffer.label))
        .map(([buffer, offset, data]) => ({ label: buffer.label, words: Array.from(new Uint32Array(buffer.bytes, offset, data.byteLength / 4)) }));
      const first = writes();
      expect(first).toHaveLength(2);
      expect(first[0].words.slice(0, 2)).toEqual([8, 25]);
      expect(first[0].words.slice(8, 12)).toEqual([4, 2, 10, 0]);
      expect(first[1].words[9]).toBe(25);
      expect(first[1].words.slice(17, 20)).toEqual([4, 2, 10]);
      // One reset, one scene candidate pass, one active mesh pass: no second trace.
      expect(encoder.beginComputePass).toHaveBeenCalledTimes(3);
      expect(query.encodeTrace(gpuEncoder, 8, 0, 2, 100, { ...partition, supplementalMaxWorkUnits: 43 })).toBe(0);
      query.encodeTrace(gpuEncoder, 8, 0, 2, 100, { ...partition, raysPerBlock: 8, primaryRaysPerBlock: 4 });
      expect(writes()).toHaveLength(4);
      expect(writes()[0]).toEqual(first[0]); expect(writes()[1]).toEqual(first[1]);
      expect(writes()[2].words.slice(8, 12)).toEqual([8, 4, 10, 0]);
      query.encodeTrace(gpuEncoder, 8, 0, 2);
      expect(writes().at(-1)!.words.slice(17, 20)).toEqual([0, 0, 0]);
    } finally { query.dispose(); }
  });
  it('keeps default and supplemental quotas in separate immutable slots and reuses equivalent quotas', async () => {
    const { query, quotas, encoder } = await setup();
    try {
      const gpuEncoder = encoder as unknown as GPUCommandEncoder;
      query.encodeTrace(gpuEncoder, 4, 0, 2);
      query.encodeTrace(gpuEncoder, 4, 0, 2, 40);
      expect(quotas()).toEqual([25, 10]);
      expect(query.encodeTrace(gpuEncoder, 4, 0, 2)).toBe(0);
      expect(query.encodeTrace(gpuEncoder, 4, 0, 2, 43)).toBe(0);
      expect(quotas()).toEqual([25, 10]);
      expect(query.options.maxTotalWorkUnits).toBe(100);
      expect(query.rayBuffer.usage & GPUBufferUsage.COPY_SRC).toBeTruthy();
    } finally { query.dispose(); }
  });

  it('caps each quota at maxNodeVisits and accepts a one-unit trace', async () => {
    const { query, quotas, encoder } = await setup(12);
    try {
      query.encodeTrace(encoder as unknown as GPUCommandEncoder, 4, 0, 2, 100);
      query.encodeTrace(encoder as unknown as GPUCommandEncoder, 1, 0, 2, 1);
      expect(quotas()).toEqual([12, 1]);
    } finally { query.dispose(); }
  });

  it('rejects invalid or insufficient budgets before recording any dispatch, including empty queries', async () => {
    const { query, writeBuffer, encoder } = await setup();
    try {
      for (const invalid of [0, -1, 101, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, null, '10']) {
        for (const count of [0, 4]) expect(() => query.encodeTrace(encoder as unknown as GPUCommandEncoder,
          count, 0, 2, invalid as number)).toThrow('WORLD_QUERY_TRACE_WORK_BUDGET');
      }
      expect(() => query.encodeTrace(encoder as unknown as GPUCommandEncoder, 4, 0, 2, 3))
        .toThrow('WORLD_QUERY_WORK_BUDGET');
      expect(query.encodeTrace(encoder as unknown as GPUCommandEncoder, 0, 0, 2, 1)).toBe(0);
      expect(encoder.beginComputePass).not.toHaveBeenCalled();
      expect(writeBuffer).not.toHaveBeenCalled();
    } finally { query.dispose(); }
  });
});
