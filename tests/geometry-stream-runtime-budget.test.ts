import { describe, expect, it } from 'vitest';
import { GeometryStreamRuntimeBudget } from '../src/geometry-stream-runtime-budget';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';

const limits = { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };
function branches(key = 'runtime/branches'): GeometryStreamDirectory {
  const left = { min: [-2, 0, 0] as [number, number, number], max: [-1, 1, 1] as [number, number, number] };
  const right = { min: [1, 0, 0] as [number, number, number], max: [2, 1, 1] as [number, number, number] };
  return { version: 1, primitiveKey: key, sourceVertexCount: 6, sourceTriangleCount: 2, vertexStride: 80,
    clusters: [0, 1, 2, 3, 4, 5].map(id => ({ level: Math.floor(id / 2), group: id < 4 ? id : 4,
      refinedGroup: id < 2 ? null : id - 2, vertexCount: 3, triangleCount: 1,
      bounds: id % 2 ? right : left, lodBounds: id % 2 ? right : left, error: id < 2 ? 0 : id < 4 ? 1 : 4 })),
    groups: [0, 1, 2, 3].map(id => ({ level: Math.floor(id / 2), children: [id], parents: [id + 2],
      bounds: id % 2 ? right : left, error: id < 2 ? 1 : 4, terminal: false })).concat([
      { level: 2, children: [4, 5], parents: [], bounds: { min: [-2, 0, 0], max: [2, 1, 1] }, error: 4, terminal: true },
    ]), roots: [4, 5], pages: [0, 1, 2, 3, 4, 5].map(id => ({ id, clusterIds: [id], encodedBytes: 41, decodedBytes: 272, sha256: 'a'.repeat(64) })),
    clusterPages: [0, 1, 2, 3, 4, 5], rootPages: [4, 5] };
}
const demand = (groupId: number, projectedError: number) => ({ groupId, projectedError });
const inputs = [{ directory: branches('a'), capacity: 1 }, { directory: branches('b'), capacity: 1 }];
const generous = () => new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 1024 ** 2, maxPeakBytes: 2 * 1024 ** 2 }, limits);

describe('runtime scene geometry budgets including rendered dependents', () => {
  it('charges physical slots, instantiated rendered output, both metadata sets, parameters and maximum selection feedback', () => {
    const scheduler = generous(), state = scheduler.state(0, 4);
    expect(state.poolBytes).toBe(6 * 32 + 6 * 4 + 4 * (3 * 144 + 3 * 4));
    expect(state.outputBytes).toBe(4 * 3 * 64); expect(state.parameterBytes).toBe(32);
    expect(state.selectionBytes).toBeGreaterThan(0);
    expect(state.totalBytes).toBe(state.poolBytes + state.outputBytes + state.parameterBytes + state.selectionBytes);
    const instanced = new GeometryStreamRuntimeBudget([{ directory: branches(), capacity: 2 }],
      { maxResidentBytes: 1024 ** 2, maxPeakBytes: 2 * 1024 ** 2 }, limits).state(0, 4);
    expect(instanced.poolBytes).toBe(state.poolBytes); expect(instanced.outputBytes).toBe(2 * state.outputBytes);
    expect(instanced.selectionBytes).toBeGreaterThan(state.selectionBytes);
  });

  it('moves capacity between views with complete dependency chains and preserves every stream root', () => {
    const reference = generous(), liveBudget = reference.state(0, 4).totalBytes + reference.state(1, 2).totalBytes;
    const scheduler = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: liveBudget, maxPeakBytes: liveBudget * 2 }, limits);
    const nearA = scheduler.plan([[demand(2, 100), demand(0, 50)], []]);
    expect(nearA.streams.map(stream => stream.slots)).toEqual([4, 2]);
    expect(nearA.streams[0].pages).toEqual([0, 2, 4, 5]); expect(nearA.allocatedBytes).toBe(liveBudget);
    const nearB = scheduler.plan([[], [demand(3, 100), demand(1, 50)]]);
    expect(nearB.streams.map(stream => stream.slots)).toEqual([2, 4]);
    expect(nearB.streams[1].pages).toEqual([1, 3, 4, 5]);
    const first = scheduler.next(nearB, [4, 2]);
    expect(first.action).toMatchObject({ index: 0, slots: 2, expectedSlots: 4, temporary: false });
    expect(first.action!.peakBytes).toBe(liveBudget + reference.state(0, 2).totalBytes);
    const second = scheduler.next(nearB, [2, 2]);
    expect(second.action).toMatchObject({ index: 1, slots: 4, expectedSlots: 2 });
    expect(scheduler.next(nearB, [2, 4])).toMatchObject({ action: null, deferred: false });
  });

  it('counts retiring owners in overlap and reports a deferred migration instead of overcommitting', () => {
    const reference = generous(), liveBudget = reference.state(0, 4).totalBytes + reference.state(1, 2).totalBytes;
    const peakBudget = liveBudget + reference.state(0, 2).totalBytes;
    const scheduler = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: liveBudget, maxPeakBytes: peakBudget }, limits);
    const plan = scheduler.plan([[], [demand(3, 100), demand(1, 50)]]);
    const first = scheduler.next(plan, [4, 2]); expect(first.action?.index).toBe(0);
    const waiting = scheduler.next(plan, [2, 2], reference.state(0, 4).totalBytes);
    expect(waiting).toMatchObject({ action: null, deferred: true });
    const released = scheduler.next(plan, [2, 2]); expect(released.action?.index).toBe(1);
    expect(released.action!.peakBytes).toBeLessThanOrEqual(peakBudget);
  });

  it('can use an explicit temporary root-only shrink when a smaller desired capacity cannot overlap', () => {
    const reference = generous(), liveBudget = reference.state(0, 4).totalBytes + reference.state(1, 4).totalBytes;
    const scheduler = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: liveBudget,
      maxPeakBytes: liveBudget + reference.state(0, 2).totalBytes }, limits);
    const plan = scheduler.plan([[demand(2, 100), demand(0, 50)], [demand(3, 90), demand(1, 40)]]);
    expect(plan.streams.map(stream => stream.slots)).toEqual([4, 4]);
    const first = scheduler.next(plan, [2, 6]);
    expect(first.action).toMatchObject({ index: 1, slots: 2, expectedSlots: 6, temporary: true, pages: [4, 5] });
    expect(first.action!.peakBytes).toBeLessThanOrEqual(scheduler.options.maxPeakBytes);
  });

  it('preflights impossible roots, malformed inputs, invalid plans, and existing over-budget owners', () => {
    expect(() => new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 1, maxPeakBytes: 1 }, limits)).toThrow('ROOT_BUDGET');
    expect(() => new GeometryStreamRuntimeBudget(new Array(1), { maxResidentBytes: 1024 ** 2, maxPeakBytes: 2 * 1024 ** 2 }, limits)).toThrow();
    expect(() => new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 1024 ** 2, maxPeakBytes: 2 * 1024 ** 2 }, {} as typeof limits)).toThrow('DEVICE_LIMITS');
    const scheduler = generous(), plan = scheduler.plan([[], []]);
    expect(() => scheduler.plan([[]])).toThrow('DEMAND');
    expect(() => scheduler.next(plan, [1, 2])).toThrow('CAPACITY');
    expect(() => scheduler.next(plan, [2, 2], scheduler.options.maxPeakBytes)).toThrow('EXISTING_PEAK');
    expect(() => scheduler.next({ ...plan, streams: plan.streams.map(stream => ({ ...stream, poolBytes: 1 })) }, [2, 2])).toThrow('PLAN');
    expect(() => scheduler.next({ ...plan, streams: plan.streams.map(stream => ({ ...stream, pages: [4, 4] })) }, [2, 2])).toThrow('PLAN');
  });

  it('does not spend a pending shrink or overbook positive growth reserved by another candidate', () => {
    const sizing = generous(), resident = sizing.state(0, 4).totalBytes + sizing.state(1, 2).totalBytes;
    const scheduler = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: resident, maxPeakBytes: resident * 3 }, limits);
    const plan = scheduler.plan([[], [demand(2, 100), demand(0, 50)]]);
    expect(scheduler.next(plan, [4, 2], scheduler.state(0, 2).totalBytes, { excluded: [0] })).toMatchObject({ action: null, deferred: true });
    const delta = scheduler.state(1, 4).totalBytes - scheduler.state(1, 2).totalBytes;
    expect(scheduler.next(plan, [2, 2], 0, { growthBytes: delta })).toMatchObject({ action: null, deferred: true });
    expect(scheduler.next(plan, [2, 2], 0).action).toMatchObject({ index: 1, slots: 4 });
    expect(() => scheduler.next(plan, [4, 2], 0, { growthBytes: 1 })).toThrow('EXISTING_LIVE');
    expect(() => scheduler.next(plan, [2, 2], 0, { excluded: [2] })).toThrow('STATE');
  });
});
