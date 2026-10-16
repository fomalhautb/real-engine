import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { GeometryDemoScene } from '../src/geometry-demo-scene';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import type { GpuGeometryPagePool } from '../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../src/gpu-geometry-stream-selection';
import { identity } from '../src/math';

type FakeBuffer = { label: string; size: number; bytes: ArrayBuffer; destroyed: boolean;
  getMappedRange: () => ArrayBuffer; unmap: () => void; mapAsync: () => Promise<void>; destroy: () => void };

function buffer(label: string, size: number): FakeBuffer {
  return { label, size, bytes: new ArrayBuffer(size), destroyed: false,
    getMappedRange() { return this.bytes; }, unmap() {}, async mapAsync() {},
    destroy() { this.destroyed = true; } };
}

class FakeDevice {
  readonly buffers: FakeBuffer[] = [];
  readonly limits = { maxBufferSize: 256 * 1024 * 1024, maxStorageBufferBindingSize: 256 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535, minUniformBufferOffsetAlignment: 256 };
  readonly lost: Promise<{ message: string }>;
  private lose!: (info: { message: string }) => void;
  pushes = 0;
  pops = 0;
  failBufferAt = -1;
  failPipelineAt = -1;
  pipelines = 0;

  constructor() { this.lost = new Promise(resolve => { this.lose = resolve; }); }
  loseDevice() { this.lose({ message: 'fake loss' }); }
  pushErrorScope() { this.pushes++; }
  async popErrorScope() { this.pops++; return null; }
  createBuffer = ({ label, size }: { label: string; size: number }): GPUBuffer => {
    if (this.buffers.length === this.failBufferAt) throw new Error('FAKE_ALLOCATION_FAILURE');
    const result = buffer(label, size); this.buffers.push(result); return result as unknown as GPUBuffer;
  };
  createShaderModule = () => ({ getCompilationInfo: async () => ({ messages: [] }) });
  createBindGroupLayout = () => ({});
  createPipelineLayout = () => ({});
  createBindGroup = () => ({});
  createComputePipelineAsync = async () => {
    if (this.pipelines++ === this.failPipelineAt) throw new Error('FAKE_PIPELINE_FAILURE');
    return {};
  };
  createCommandEncoder = () => ({
    clearBuffer() {}, copyBufferToBuffer() {}, finish: () => ({}),
    beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
  });
  queue = { writeBuffer() {}, submit() {} };
}

let directory: GeometryStreamDirectory;
const selections: GpuGeometryStreamSelection[] = [];

function fakePool(): GpuGeometryPagePool {
  return { directory, slotCount: directory.pages.length, verticesPerSlot: 1024,
    pageResidency: buffer('pool residency', directory.pages.length * 4),
    clusterLocations: buffer('pool locations', directory.clusters.length * 32),
    indices: buffer('pool indices', 4096), diagnostics: () => ({ rootsReady: true }),
  } as unknown as GpuGeometryPagePool;
}

async function selection(device: FakeDevice, track: (bytes: number) => void = () => {}, maxGpuBytes?: number) {
  const result = await GpuGeometryStreamSelection.create(device as unknown as GPUDevice, fakePool(),
    buffer('instances', 224) as unknown as GPUBuffer, 1, track, { maxGpuBytes });
  selections.push(result);
  return result;
}

beforeAll(async () => {
  const hierarchy = (await GeometryDemoScene.create()).hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  directory = await buildGeometryStreamDirectory(hierarchy, encoded);
});
beforeEach(() => {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8,
    INDEX: 16, INDIRECT: 32, MAP_READ: 64 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
  vi.stubGlobal('GPUMapMode', { READ: 1 });
});
afterEach(() => {
  for (const entry of selections.splice(0)) entry.dispose();
  vi.unstubAllGlobals();
});

describe('GPU streamed selection ownership and failure lifecycle', () => {
  it('destroys partial allocations and balances both scopes while retaining the allocation error', async () => {
    const device = new FakeDevice(); device.failBufferAt = 3;
    await expect(selection(device, bytes => {
      if (bytes < 0) throw new Error('FAKE_CLEANUP_TRACKER_FAILURE');
    })).rejects.toThrow('FAKE_ALLOCATION_FAILURE');
    expect(device.buffers).toHaveLength(3);
    expect(device.buffers.every(entry => entry.destroyed)).toBe(true);
    expect([device.pushes, device.pops]).toEqual([2, 2]);
  });

  it('destroys all persistent buffers after asynchronous pipeline failure even if cleanup accounting throws', async () => {
    const device = new FakeDevice(); device.failPipelineAt = 1;
    await expect(selection(device, bytes => {
      if (bytes < 0) throw new Error('FAKE_CLEANUP_TRACKER_FAILURE');
    })).rejects.toThrow('FAKE_PIPELINE_FAILURE');
    expect(device.buffers).toHaveLength(7);
    expect(device.buffers.every(entry => entry.destroyed)).toBe(true);
    expect([device.pushes, device.pops]).toEqual([2, 2]);
  });

  it('releases a failed feedback scratch buffer and permits a later feedback request', async () => {
    const device = new FakeDevice();
    let tracked = 0, failFeedback = false;
    const current = await selection(device, bytes => {
      tracked += bytes;
      if (failFeedback && bytes > 0) throw new Error('FAKE_FEEDBACK_TRACKER_FAILURE');
    });
    current.update(1, 1, identity(), [0, 0, 4], 400);
    current.encode(device.createCommandEncoder() as unknown as GPUCommandEncoder);
    const persistent = current.diagnostics().gpuBytes;
    failFeedback = true;
    await expect(current.feedback(true)).rejects.toThrow('FAKE_FEEDBACK_TRACKER_FAILURE');
    const failedReadback = device.buffers.at(-1)!;
    expect(failedReadback.label).toContain('bounded stream requests');
    expect(failedReadback.destroyed).toBe(true);
    expect(current.diagnostics()).toMatchObject({ feedbackPending: false, gpuBytes: persistent });
    expect(tracked).toBe(persistent);
    failFeedback = false;
    await expect(current.feedback(true)).resolves.toMatchObject({ frame: 1, indices: 0, overflow: false });
    expect(current.diagnostics()).toMatchObject({ feedbackPending: false, gpuBytes: persistent });
    current.dispose();
    expect(tracked).toBe(0);
  });

  it('contains a throwing cleanup tracker on device loss and records it in diagnostics', async () => {
    const device = new FakeDevice();
    const current = await selection(device, bytes => {
      if (bytes < 0) throw new Error('FAKE_LOSS_TRACKER_FAILURE');
    });
    device.loseDevice(); await Promise.resolve();
    expect(current.diagnostics()).toMatchObject({ disposed: true, gpuBytes: 0, feedbackPending: false });
    expect(current.diagnostics().error).toMatch(/GEOMETRY_STREAM_DEVICE_LOST.*FAKE_LOSS_TRACKER_FAILURE/);
    expect(device.buffers.every(entry => entry.destroyed)).toBe(true);
    expect(() => current.dispose()).not.toThrow();
  });

  it('requires room for maximum feedback scratch in the declared GPU byte budget', async () => {
    const device = new FakeDevice();
    const current = await selection(device);
    const { gpuBytes, maxFeedbackBytes } = current.diagnostics();
    expect(maxFeedbackBytes).toBe(32 + directory.pages.length * 4 + directory.clusters.length * 4);
    const allocations = device.buffers.length;
    await expect(selection(device, () => {}, gpuBytes + maxFeedbackBytes - 1))
      .rejects.toThrow('GEOMETRY_STREAM_RESOURCE_BUDGET');
    expect(device.buffers).toHaveLength(allocations);
    const exact = await selection(device, () => {}, gpuBytes + maxFeedbackBytes);
    expect(exact.diagnostics().maxGpuBytes).toBe(gpuBytes + maxFeedbackBytes);
  });
});
