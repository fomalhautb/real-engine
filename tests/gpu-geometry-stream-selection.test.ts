import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { GeometryDemoScene } from '../src/geometry-demo-scene';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { GpuGeometryPagePool } from '../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../src/gpu-geometry-stream-selection';
import { identity } from '../src/math';
import type { GpuOcclusion } from '../src/gpu-occlusion';
import { packGeometryStreamSelection } from '../src/geometry-stream-selection-data';

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
  readonly boundGroups: { index: number; group: unknown }[] = [];

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
    clearBuffer() {}, copyBufferToBuffer(source: FakeBuffer, sourceOffset: number, target: FakeBuffer, targetOffset: number, size: number) {
      new Uint8Array(target.bytes, targetOffset, size).set(new Uint8Array(source.bytes, sourceOffset, size));
    }, finish: () => ({}),
    beginComputePass: () => ({ setPipeline() {}, setBindGroup: (index: number, group: unknown) => {
      this.boundGroups.push({ index, group });
    }, dispatchWorkgroups() {}, end() {} }),
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

async function selection(device: FakeDevice, track: (bytes: number) => void = () => {}, maxGpuBytes?: number,
  occlusion?: GpuOcclusion) {
  const result = await GpuGeometryStreamSelection.create(device as unknown as GPUDevice, fakePool(),
    buffer('instances', 224) as unknown as GPUBuffer, 1, track, { maxGpuBytes }, occlusion);
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
  it('shares one loss observer with page pools and detaches disposed owners before device loss', async () => {
    const device = new FakeDevice(), observed = vi.spyOn(device.lost, 'then');
    const pool = await GpuGeometryPagePool.create(device as unknown as GPUDevice, directory);
    const retired = await selection(device), active = await selection(device);
    pool.dispose(); retired.dispose();
    expect(observed).toHaveBeenCalledOnce();
    device.loseDevice(); await Promise.resolve();
    expect(pool.diagnostics().error).toBeNull(); expect(retired.diagnostics().error).toBeNull();
    expect(active.diagnostics()).toMatchObject({ disposed: true, gpuBytes: 0, error: 'GEOMETRY_STREAM_DEVICE_LOST' });
  });

  it('reuses compiled pipelines during capacity changes and closes allocation scopes before awaiting', async () => {
    const device = new FakeDevice(), current = await selection(device);
    const compiled = device.pipelines, allocated = device.buffers.length;
    const replacement = GpuGeometryStreamSelection.create(device as unknown as GPUDevice,
      { ...fakePool(), slotCount: directory.rootPages.length } as GpuGeometryPagePool,
      buffer('instances', 224) as unknown as GPUBuffer, 1, () => {}, {}, undefined, current);
    expect(device.pushes).toBe(device.pops); expect(device.pipelines).toBe(compiled);
    const next = await replacement; selections.push(next);
    expect(next.physicalVertexCount).toBe(directory.rootPages.length * 1024);
    expect(device.buffers.length - allocated).toBe(7);
    current.dispose();
    next.update(1, 1, identity(), [0, 0, 4], 400);
    expect(() => next.encode(device.createCommandEncoder() as unknown as GPUCommandEncoder)).not.toThrow();
    expect(device.buffers.slice(allocated).every(entry => !entry.destroyed)).toBe(true);
  });

  it('cleans partial reused selection allocations while keeping the old owner usable', async () => {
    const device = new FakeDevice(), current = await selection(device), allocated = device.buffers.length;
    device.failBufferAt = allocated + 2;
    await expect(GpuGeometryStreamSelection.create(device as unknown as GPUDevice, fakePool(),
      buffer('instances', 224) as unknown as GPUBuffer, 1, () => {}, {}, undefined, current)).rejects.toThrow('FAKE_ALLOCATION_FAILURE');
    expect(device.buffers.slice(allocated).every(entry => entry.destroyed)).toBe(true);
    expect(current.diagnostics().disposed).toBe(false); expect(device.pushes).toBe(device.pops);
  });

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
    expect(maxFeedbackBytes).toBe(64 + (directory.pages.length + directory.groups.length + directory.clusters.length) * 4);
    const allocations = device.buffers.length;
    await expect(selection(device, () => {}, gpuBytes + maxFeedbackBytes - 1))
      .rejects.toThrow('GEOMETRY_STREAM_RESOURCE_BUDGET');
    expect(device.buffers).toHaveLength(allocations);
    const exact = await selection(device, () => {}, gpuBytes + maxFeedbackBytes);
    expect(exact.diagnostics().maxGpuBytes).toBe(gpuBytes + maxFeedbackBytes);
  });

  it('keeps the legacy one-pass path and exposes zero occlusion counters', async () => {
    const device = new FakeDevice();
    const current = await selection(device);
    current.update(1, 1, identity(), [0, 0, 4], 400);
    current.encode(device.createCommandEncoder() as unknown as GPUCommandEncoder);
    current.encodeRecovery(device.createCommandEncoder() as unknown as GPUCommandEncoder);
    expect(device.pipelines).toBe(5);
    expect(device.boundGroups.every(binding => binding.index === 0)).toBe(true);
    expect(await current.feedback()).toMatchObject({ indices: 0, overflow: false,
      occlusion: { deferredClusters: 0, culledClusters: 0, recoveredClusters: 0 } });
  });

  it('requires recovery before feedback and combines both draw records with bounded offsets', async () => {
    const device = new FakeDevice();
    const previousGroup = { name: 'previous' }, currentGroup = { name: 'current' };
    const occlusion = { testLayout: {}, previousGroup, currentGroup } as unknown as GpuOcclusion;
    const current = await selection(device, () => {}, undefined, occlusion);
    current.update(1, 1, identity(), [0, 0, 4], 400);
    current.encode(device.createCommandEncoder() as unknown as GPUCommandEncoder);
    expect(device.pipelines).toBe(7);
    expect(device.boundGroups.some(binding => binding.index === 1 && binding.group === previousGroup)).toBe(true);
    await expect(current.feedback()).rejects.toThrow('GEOMETRY_STREAM_RECOVERY_PENDING');
    current.encodeRecovery(device.createCommandEncoder() as unknown as GPUCommandEncoder);
    expect(device.boundGroups.some(binding => binding.index === 1 && binding.group === currentGroup)).toBe(true);
    expect(() => current.encodeRecovery(device.createCommandEncoder() as unknown as GPUCommandEncoder))
      .toThrow('GEOMETRY_STREAM_RECOVERY_FRAME');

    const args = new Uint32Array(device.buffers.find(entry => entry.label === 'stream indirect draw arguments')!.bytes);
    args[0] = 6; args[7] = 2; args[8] = 3; args[13] = 1; args[15] = 1;
    const requests = new Uint32Array(device.buffers.find(entry => entry.label === 'stream page requests and group demand')!.bytes);
    requests[0] = 6;
    new Float32Array(requests.buffer)[directory.pages.length] = 4.5;
    const selected = new Uint32Array(device.buffers.find(entry => entry.label === 'stream cut and visible flags')!.bytes);
    selected[0] = 2; selected[1] = 1;
    const feedback = await current.feedback(true);
    expect(feedback).toMatchObject({ frame: 1, indices: 9, overflow: false,
      requestedPages: [{ pageId: 0, priority: 5 }],
      desiredGroups: [{ groupId: packGeometryStreamSelection(directory).groupIds[0], projectedError: 4.5 }],
      occlusion: { deferredClusters: 2, culledClusters: 1, recoveredClusters: 1 } });
    expect(feedback.selectedClusters?.slice(0, 2)).toMatchObject([
      { clusterId: 0, visible: true }, { clusterId: 1, visible: false },
    ]);
    args[14] = 1;
    expect(await current.feedback()).toMatchObject({ overflow: true });
  });
});
