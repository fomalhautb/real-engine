import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DIFFUSE_GI_PROBE_BYTES } from '../src/diffuse-gi';
import { resolveFinalGatherOptions } from '../src/final-gather';
import { GpuFinalGather } from '../src/gpu-final-gather';
import type { GpuWorldQuery } from '../src/gpu-world-query';
import { WORLD_HIT_BYTES, WORLD_RAY_BYTES } from '../src/world-query';

interface FakeBuffer {
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  destroyed: boolean;
  destroy(): void;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

function fakeBuffer(label: string, size: number, usage: number): FakeBuffer {
  const bytes = new ArrayBuffer(size);
  return {
    label, size, usage, destroyed: false,
    destroy() { this.destroyed = true; },
    getMappedRange() { return bytes; },
    unmap() {},
  };
}

class FakeDevice {
  readonly limits = {
    maxStorageBuffersPerShaderStage: 6,
    maxBufferSize: 128 * 1024 * 1024,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxComputeWorkgroupsPerDimension: 65_535,
  };
  readonly buffers: FakeBuffer[] = [];
  bufferCalls = 0;
  failBufferCall = -1;

  createBuffer = ({ label, size, usage }: GPUBufferDescriptor): GPUBuffer => {
    this.bufferCalls++;
    if (this.bufferCalls === this.failBufferCall) throw new Error('INJECTED_FINAL_GATHER_ALLOCATION_FAILURE');
    const buffer = fakeBuffer(label ?? '', size, usage);
    this.buffers.push(buffer);
    return buffer as unknown as GPUBuffer;
  };
}

function borrowedBuffers() {
  const maxRays = 16;
  const ray = fakeBuffer('borrowed query rays', maxRays * WORLD_RAY_BYTES, GPUBufferUsage.STORAGE);
  const hit = fakeBuffer('borrowed query hits', maxRays * WORLD_HIT_BYTES, GPUBufferUsage.STORAGE);
  const giParameters = fakeBuffer('borrowed GI parameters', 112, GPUBufferUsage.UNIFORM);
  const giProbes = [0, 1].map(index => fakeBuffer(`borrowed GI probes ${index}`,
    DIFFUSE_GI_PROBE_BYTES, GPUBufferUsage.STORAGE)) as [FakeBuffer, FakeBuffer];
  const queries = {
    options: { maxRays, maxTotalWorkUnits: maxRays, maxTraceParameterSetsPerEncoder: 4 },
    rayBuffer: ray, hitBuffer: hit,
  } as unknown as GpuWorldQuery;
  return { queries, giParameters, giProbes, all: [ray, hit, giParameters, ...giProbes] };
}

function ledger() {
  let bytes = 0;
  const balances: number[] = [];
  return {
    account(delta: number) { bytes += delta; balances.push(bytes); },
    get bytes() { return bytes; },
    get balances() { return balances; },
  };
}

beforeEach(() => vi.stubGlobal('GPUBufferUsage', {
  UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8, MAP_READ: 16,
}));
afterEach(() => vi.unstubAllGlobals());

describe('final gather path resource ownership', () => {
  const options = resolveFinalGatherOptions({ pathSegments: 4 });
  const failedAllocations = [
    [4, 'surface final gather path throughput'],
    [5, 'surface final gather immutable path segment 0'],
    [6, 'surface final gather immutable path segment 1'],
    [7, 'surface final gather immutable path segment 2'],
    [8, 'surface final gather immutable path segment 3'],
    [14, 'final gather surface candidates'],
    [15, 'final gather coverage prefix sums'],
    [16, 'final gather coverage map 0'],
    [17, 'final gather coverage map 1'],
  ] as const;

  it.each(failedAllocations)(
    'releases earlier owned buffers when allocation %i (%s) fails', async (failedCall, label) => {
      const device = new FakeDevice();
      const tracked = ledger();
      const borrowed = borrowedBuffers();
      device.failBufferCall = failedCall;

      await expect(GpuFinalGather.create(device as unknown as GPUDevice, borrowed.queries,
        borrowed.giParameters as unknown as GPUBuffer,
        borrowed.giProbes as unknown as [GPUBuffer, GPUBuffer], options,
        tracked.account, options.maxBytes)).rejects.toThrow('INJECTED_FINAL_GATHER_ALLOCATION_FAILURE');

      expect(device.bufferCalls).toBe(failedCall);
      expect(device.buffers).toHaveLength(failedCall - 1);
      expect(device.buffers.map(buffer => buffer.label)).not.toContain(label);
      expect(device.buffers.every(buffer => buffer.destroyed)).toBe(true);
      expect(tracked.bytes).toBe(0);
      expect(tracked.balances.every(bytes => bytes >= 0)).toBe(true);
      expect(tracked.balances.at(-1)).toBe(0);
      expect(borrowed.all.every(buffer => !buffer.destroyed)).toBe(true);
    },
  );

  it.each([
    ['local option', resolveFinalGatherOptions({ pathSegments: 4, maxBytes: 1 }), 16 * 1024 * 1024],
    ['remaining shared budget', options, 1],
  ] as const)('rejects insufficient %s memory before creating any buffer', async (_budget, config, remaining) => {
    const device = new FakeDevice();
    const tracked = ledger();
    const borrowed = borrowedBuffers();

    await expect(GpuFinalGather.create(device as unknown as GPUDevice, borrowed.queries,
      borrowed.giParameters as unknown as GPUBuffer,
      borrowed.giProbes as unknown as [GPUBuffer, GPUBuffer], config,
      tracked.account, remaining)).rejects.toThrow('FINAL_GATHER_MEMORY_BUDGET');

    expect(device.bufferCalls).toBe(0);
    expect(device.buffers).toHaveLength(0);
    expect(tracked.balances.every(bytes => bytes === 0)).toBe(true);
    expect(tracked.bytes).toBe(0);
    expect(borrowed.all.every(buffer => !buffer.destroyed)).toBe(true);
  });
});
