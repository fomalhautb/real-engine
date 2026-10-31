import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SceneBatch } from '../src/contracts';
import type { MorphTarget } from '../src/deformation';
import { createPlane } from '../src/geometry';
import { createTimestampQuerySet, timestampSpanEnd, timestampSpanStart } from '../src/gpu-timestamp-span';
import { GpuWorldQuery } from '../src/gpu-world-query';
import type { MaterialProgram } from '../src/material';
import { identity, translation } from '../src/math';
import type { WorldQueryOptions } from '../src/world-query';

interface FakeBuffer { label: string; size: number; usage: number; bytes: ArrayBuffer; destroy(): void }
type Dispatch = { entryPoint: string; groups: number };

function program(alphaMode: 'opaque' | 'mask', doubleSided = false): MaterialProgram {
  return { wgsl: '', shading: 'pbr', alphaMode, doubleSided, normalMapped: false } as unknown as MaterialProgram;
}

async function setup(options: WorldQueryOptions, count = 10, overrides: (index: number) => Partial<SceneBatch> = () => ({})) {
  const buffers: FakeBuffer[] = [];
  const writeBuffer = vi.fn((buffer: FakeBuffer, offset: number, data: ArrayBuffer | ArrayBufferView) => {
    const source = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
    new Uint8Array(buffer.bytes, offset, source.byteLength).set(source);
  });
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256, maxUniformBufferBindingSize: 65536,
      maxStorageBufferBindingSize: 128 << 20, maxBufferSize: 256 << 20, maxComputeWorkgroupsPerDimension: 65535 },
    queue: { writeBuffer },
    createBuffer: ({ label = '', size, usage, mappedAtCreation }: GPUBufferDescriptor) => {
      const bytes = new ArrayBuffer(size);
      const buffer = { label, size, usage, bytes, destroy: vi.fn(), getMappedRange: () => bytes, unmap() {}, mappedAtCreation };
      buffers.push(buffer); return buffer;
    },
    createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => ({ descriptor }),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) =>
      ({ entryPoint: descriptor.compute.entryPoint, getBindGroupLayout: () => ({}) }),
  } as unknown as GPUDevice;
  let accounted = 0;
  const query = await GpuWorldQuery.create(device, {} as GPUBindGroupLayout, bytes => { accounted += bytes; },
    { maxRays: 64, maxBatches: 16, ...options });
  const trace = await query.createPipeline(program('opaque'));
  const materials = Array.from({ length: count }, () => ({ bindGroup: {}, program: program('opaque') }));
  const dirty = Array.from({ length: count }, () => true);
  const scenes: SceneBatch[] = Array.from({ length: count }, (_, index) => ({
    name: `shared-${index}`, mesh: createPlane(), capacity: 2, inputMode: 'committed',
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    instances: [{ entityId: index + 1, model: translation(index * 3, 0, 0), normalMatrix: identity(), bones: [],
      deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 }],
    ...overrides(index) }));
  scenes.forEach((scene, index) => query.addBatch({ scene, output: {} as GPUBuffer, instances: {} as GPUBuffer,
    getPipeline: () => trace, getMaterial: () => materials[index] as never, getDirty: () => dirty[index] }));
  const record = () => {
    const passes: { label: string; dispatches: Dispatch[]; timestampWrites?: GPUComputePassTimestampWrites }[] = [];
    const encoder = {
      beginComputePass: vi.fn((descriptor?: GPUComputePassDescriptor) => {
        const pass = { label: descriptor?.label ?? '', dispatches: [] as Dispatch[], timestampWrites: descriptor?.timestampWrites };
        passes.push(pass);
        let pipeline: { entryPoint: string } | undefined;
        return { setPipeline: (value: { entryPoint: string }) => { pipeline = value; }, setBindGroup: vi.fn(),
          dispatchWorkgroups: (groups: number) => pass.dispatches.push({ entryPoint: pipeline!.entryPoint, groups }), end: vi.fn() };
      }),
      copyBufferToBuffer: vi.fn(),
    };
    return { encoder: encoder as unknown as GPUCommandEncoder, passes,
      entryPoints: () => passes.flatMap(pass => pass.dispatches.map(dispatch => dispatch.entryPoint)) };
  };
  /** Parameter slot 0 of a batch: the first trace recorded into the latest encoder. */
  const traceWords = (batch: number) =>
    new Uint32Array(buffers.filter(buffer => buffer.label === 'world batch parameter slots')[batch].bytes, 0, 28);
  return { query, scenes, materials, dirty, buffers, writeBuffer, record, traceWords, accounted: () => accounted };
}

beforeEach(() => {
  vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8, MAP_READ: 16 });
  vi.stubGlobal('GPUShaderStage', { COMPUTE: 1 });
});
afterEach(() => vi.unstubAllGlobals());

describe('shared static world query encoding', () => {
  it('replaces per-batch traversal by one pooled dispatch and per-batch material resolves', async () => {
    const { query, record } = await setup({ staticTraversal: 'shared' });
    try {
      query.updateSnapshot(1, 0);
      const shared = query.diagnostics().sharedStatic!;
      expect(shared).toMatchObject({ pooledBatches: 10, excludedBatches: 0, members: 10, leaves: 20, tlasNodes: 39 });
      const refit = record();
      query.encodeRefit(refit.encoder);
      const update = refit.passes.find(pass => pass.label === 'world shared static pool update')!;
      expect(update.dispatches.filter(dispatch => dispatch.entryPoint === 'updatePool')).toHaveLength(10);
      expect(update.dispatches.slice(10).every(dispatch => dispatch.entryPoint === 'refitTlas')).toBe(true);
      expect(query.diagnostics().sharedStatic!.lastUpdate).toMatchObject({ frame: 1, batches: 10, dispatches: update.dispatches.length });
      const traced = record();
      query.encodeTrace(traced.encoder, 64, 0, 0);
      // No candidate pass: every active batch is a pooled member.
      expect(traced.entryPoints()).toEqual(['main', 'traverse', ...Array(10).fill('resolveShared')]);
      expect(traced.passes).toHaveLength(12);
      expect(query.diagnostics()).toMatchObject({ staticTraversal: 'shared', dispatchEncoding: 'pass-per-batch' });
    } finally { query.dispose(); }
  });
  it('spans an optional trace timestamp pair over the first and last trace pass only', async () => {
    const querySet = createTimestampQuerySet({ createQuerySet: (descriptor: GPUQuerySetDescriptor) => ({ count: descriptor.count }) } as unknown as GPUDevice, 2);
    const timing = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
    for (const dispatchEncoding of ['pass-per-batch', 'single-pass'] as const) {
      const { query, record } = await setup({ staticTraversal: 'shared', dispatchEncoding });
      try {
        query.updateSnapshot(1, 0);
        const traced = record();
        query.encodeTrace(traced.encoder, 64, 0, 0, undefined, undefined, 'material', timing);
        const writes = traced.passes.map(pass => pass.timestampWrites);
        if (dispatchEncoding === 'single-pass') expect(writes).toEqual([timing]);
        else expect(writes).toEqual([timestampSpanStart(timing), ...Array(10).fill(undefined), timestampSpanEnd(timing)]);
        const untimed = record();
        query.encodeTrace(untimed.encoder, 64, 0, 0);
        expect(untimed.passes.every(pass => pass.timestampWrites === undefined)).toBe(true);
      } finally { query.dispose(); }
    }
  });
  it('records the same ordered dispatches in one pass for single-pass encoding', async () => {
    for (const staticTraversal of ['per-batch', 'shared'] as const) {
      const { query, record } = await setup({ staticTraversal, dispatchEncoding: 'single-pass' });
      try {
        query.updateSnapshot(1, 0);
        const traced = record();
        query.encodeTrace(traced.encoder, 64, 0, 0);
        expect(traced.passes).toHaveLength(1);
        expect(traced.entryPoints()).toEqual(staticTraversal === 'shared'
          ? ['main', 'traverse', ...Array(10).fill('resolveShared')]
          : ['main', 'selectBatches', ...Array(10).fill('main')]);
      } finally { query.dispose(); }
    }
  });
  it('keeps deformed, mutable and Masked batches on the per-batch path, including a switch after the snapshot', async () => {
    const { query, materials, record, traceWords } = await setup({ staticTraversal: 'shared' }, 10, index => index === 1
      ? { inputMode: 'mutable' } : index === 2 ? { morphTargets: [{} as MorphTarget] } : {});
    try {
      materials[0].program = program('mask');
      query.updateSnapshot(1, 0);
      expect(query.diagnostics().sharedStatic).toMatchObject({ pooledBatches: 8, members: 7,
        excluded: { 'mutable-input': 1, deformation: 1, budget: 0 } });
      const first = record();
      query.encodeTrace(first.encoder, 64, 0, 0);
      expect(first.entryPoints()).toEqual(['main', 'selectBatches', 'traverse', 'main', 'main', 'main',
        ...Array(7).fill('resolveShared')]);
      expect([0, 1, 2].map(batch => traceWords(batch)[14])).toEqual([1, 1, 1]);
      // Masked after the snapshot: still exact, by scanning its instances until the next snapshot.
      materials[3].program = program('mask');
      expect(() => query.encodeTrace(first.encoder, 64, 0, 0)).toThrow('WORLD_QUERY_ENCODER_SNAPSHOT_CHANGED');
      const second = record();
      query.encodeTrace(second.encoder, 64, 0, 0);
      expect(second.entryPoints()).toEqual(['main', 'selectBatches', 'traverse', 'main', 'main', 'main', 'main',
        ...Array(6).fill('resolveShared')]);
      expect(traceWords(3)[14]).toBe(0);
      expect(query.diagnostics().sharedStatic!.members).toBe(6);
      query.updateSnapshot(2, 0);
      const third = record();
      query.encodeTrace(third.encoder, 64, 0, 0);
      expect(traceWords(3)[14]).toBe(1);
      // Opaque again: back to the pool, no longer read from the candidate tree.
      materials[0].program = program('opaque');
      materials[3].program = program('opaque', true);
      const fourth = record();
      query.encodeTrace(fourth.encoder, 64, 0, 0);
      expect(fourth.entryPoints()).toEqual(['main', 'selectBatches', 'traverse', 'resolveShared', 'main', 'main',
        ...Array(7).fill('resolveShared')]);
    } finally { query.dispose(); }
  });
  it('updates only changed pooled batches and pays nothing for an unchanged static scene', async () => {
    const { query, dirty, record } = await setup({ staticTraversal: 'shared' });
    try {
      query.updateSnapshot(1, 0);
      query.encodeRefit(record().encoder);
      dirty.fill(false);
      query.updateSnapshot(2, 0);
      const still = record();
      query.encodeRefit(still.encoder);
      expect(still.passes).toHaveLength(0);
      dirty[4] = true;
      query.updateSnapshot(3, 0);
      const moved = record();
      query.encodeRefit(moved.encoder);
      const update = moved.passes.find(pass => pass.label === 'world shared static pool update')!;
      expect(update.dispatches.filter(dispatch => dispatch.entryPoint === 'updatePool')).toHaveLength(1);
    } finally { query.dispose(); }
  });
  it('admits only the budgeted subset and accounts every pooled byte in the query ledger', async () => {
    const reference = await setup({ staticTraversal: 'per-batch' });
    reference.query.updateSnapshot(1, 0);
    const referenceBytes = reference.query.diagnostics().ownedGpuBytes;
    reference.query.dispose();
    const full = await setup({ staticTraversal: 'shared' });
    full.query.updateSnapshot(1, 0);
    const fullPlan = full.query.diagnostics().sharedStatic!;
    const pooledBytes = full.query.diagnostics().ownedGpuBytes - referenceBytes;
    expect(pooledBytes).toBeGreaterThan(fullPlan.bytes.total - fullPlan.bytes.fixed);
    expect(pooledBytes).toBeLessThanOrEqual(fullPlan.bytes.total);
    expect(full.accounted()).toBe(full.query.diagnostics().ownedGpuBytes);
    full.query.dispose();
    expect(full.accounted()).toBe(0);
    const limited = await setup({ staticTraversal: 'shared', maxSharedStaticBytes: Math.floor(fullPlan.bytes.total * 0.6) });
    try {
      limited.query.updateSnapshot(1, 0);
      const plan = limited.query.diagnostics().sharedStatic!;
      expect(plan.pooledBatches).toBeGreaterThan(0);
      expect(plan.pooledBatches).toBeLessThan(10);
      expect(plan.excluded.budget).toBe(10 - plan.pooledBatches);
      expect(plan.bytes.total).toBeLessThanOrEqual(Math.floor(fullPlan.bytes.total * 0.6));
      const traced = limited.record();
      limited.query.encodeTrace(traced.encoder, 64, 0, 0);
      expect(traced.entryPoints().filter(entry => entry === 'resolveShared')).toHaveLength(plan.pooledBatches);
      expect(traced.entryPoints().filter(entry => entry === 'main')).toHaveLength(1 + 10 - plan.pooledBatches);
    } finally { limited.query.dispose(); }
  });
});
