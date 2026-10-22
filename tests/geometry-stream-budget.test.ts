import { describe, expect, it } from 'vitest';
import { planGeometryStreamBudgets } from '../src/geometry-stream-budget';
import { geometryGpuPoolLayout } from '../src/geometry-gpu-pool-layout';
import { GeometryStreamWorkingSet } from '../src/geometry-stream-working-set';
import { GeometryStreamCut, type GeometryStreamView } from '../src/geometry-stream-cut';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { identity } from '../src/math';

// Two independent chains allow a high-error deep refinement to compete with a
// low-error shallow one. Every group has its own physical page in this fixture.
function branches(key = 'scene/branches'): GeometryStreamDirectory {
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

describe('scene geometry pool capacity planning', () => {
  it('moves a fixed scene budget toward visible error while preserving every root and metadata byte', () => {
    const a = branches('visible'), b = branches('not-visible'), layout = geometryGpuPoolLayout(a);
    const budget = 2 * layout.rootBytes + 2 * layout.bytesPerSlot;
    const result = planGeometryStreamBudgets([{ directory: a, demands: [demand(2, 100), demand(0, 50)] },
      { directory: b, demands: [] }], { maxResidentBytes: budget });
    expect(result.allocatedBytes).toBe(budget); expect(result.unallocatedBytes).toBe(0);
    expect(result.streams.map(stream => stream.slots)).toEqual([4, 2]);
    expect(result.streams[0].pages).toEqual([0, 2, 4, 5]); expect(result.streams[1].pages).toEqual([4, 5]);
    expect(result.streams.reduce((bytes, stream) => bytes + stream.metadataBytes + stream.slots * stream.bytesPerSlot, 0)).toBe(budget);
    // A static 50/50 capacity split cannot admit both refinements in stream A.
    expect(new GeometryStreamWorkingSet(a, 3).plan([demand(2, 100), demand(0, 50)]).budgetDeferredGroups).toEqual([0]);
    expect(new GeometryStreamWorkingSet(a, result.streams[0].slots).plan([demand(2, 100), demand(0, 50)]).pages).toEqual(result.streams[0].pages);
  });

  it('prioritizes reachable high-error descendants, changes promptly with the view, and bounds hysteresis', () => {
    const directory = branches(), working = new GeometryStreamWorkingSet(directory, 4);
    const left = [demand(2, 100), demand(0, 50), demand(3, 10), demand(1, 5)];
    expect(working.plan(left)).toMatchObject({ pages: [0, 2, 4, 5], admittedGroups: [2, 0], budgetDeferredGroups: [3], dependencyDeferredGroups: [1] });
    const right = [demand(3, 100), demand(1, 50), demand(2, 10), demand(0, 5)];
    expect(working.plan(right).pages).toEqual([1, 3, 4, 5]);
    expect(working.plan(left).pages).toEqual([0, 2, 4, 5]);
    const one = new GeometryStreamWorkingSet(directory, 3);
    expect(one.plan([demand(2, 10), demand(3, 9)]).admittedGroups).toEqual([2]);
    expect(one.plan([demand(2, 10), demand(3, 10.5)]).admittedGroups).toEqual([2]);
    expect(one.plan([demand(2, 10), demand(3, 11.1)]).admittedGroups).toEqual([3]);
  });

  it('does not split a multi-page group or let an unaffordable group block a cheaper stream', () => {
    const large = branches('large'), small = branches('small');
    // Bundle both branch refinements into one atomic group with two child pages.
    const bundled: GeometryStreamDirectory = { ...large,
      clusters: large.clusters.map((c, id) => ({ ...c, group: id === 3 ? 2 : id === 4 || id === 5 ? 3 : c.group,
        lodBounds: id >= 4 ? large.groups[4].bounds : c.lodBounds,
        refinedGroup: id === 5 ? 2 : c.refinedGroup })),
      groups: [large.groups[0], large.groups[1], { ...large.groups[2], children: [2, 3], parents: [4, 5],
        bounds: large.groups[4].bounds }, { ...large.groups[4] }] };
    const bigLayout = geometryGpuPoolLayout(bundled), smallLayout = geometryGpuPoolLayout(small);
    const result = planGeometryStreamBudgets([{ directory: bundled, demands: [demand(2, 100)] },
      { directory: small, demands: [demand(2, 10)] }],
    { maxResidentBytes: bigLayout.rootBytes + smallLayout.rootBytes + smallLayout.bytesPerSlot });
    expect(result.streams[0].budgetDeferredGroups).toEqual([2]); expect(result.streams[0].pages).toEqual([4, 5]);
    expect(result.streams[1].admittedGroups).toEqual([2]); expect(result.unallocatedBytes).toBe(0);
  });

  it('reports missing dependencies and unknown demand instead of reserving invisible fine detail', () => {
    const directory = branches(), layout = geometryGpuPoolLayout(directory);
    const result = planGeometryStreamBudgets([{ directory }, { directory, demands: [demand(0, 50)] }],
      { maxResidentBytes: 2 * layout.fullBytes });
    expect(result.streams[0].demandStatus).toBe('unknown'); expect(result.streams[1].demandStatus).toBe('known');
    expect(result.streams[1].dependencyDeferredGroups).toEqual([0]);
    expect(result.allocatedBytes).toBe(2 * layout.rootBytes);
    expect(result.unallocatedBytes).toBe(2 * (layout.fullBytes - layout.rootBytes));
  });

  it('honors per-pool ceilings, deterministic ties, and rejects impossible or malformed budgets', () => {
    const directory = branches(), layout = geometryGpuPoolLayout(directory), demands = [demand(2, 10), demand(0, 5)];
    const inputs = [{ directory, demands, maxPoolBytes: layout.rootBytes }, { directory, demands }];
    const result = planGeometryStreamBudgets(inputs, { maxResidentBytes: 2 * layout.rootBytes + layout.bytesPerSlot });
    expect(result.streams[0].budgetDeferredGroups).toEqual([2]); expect(result.streams[1].admittedGroups).toEqual([2]);
    expect(planGeometryStreamBudgets([{ directory, demands }, { directory, demands }],
      { maxResidentBytes: 2 * layout.rootBytes + layout.bytesPerSlot }).streams.map(s => s.admittedGroups)).toEqual([[2], []]);
    for (const budget of [0, 1.5, Infinity, 8 * 1024 ** 3 + 1]) expect(() => planGeometryStreamBudgets([{ directory }], { maxResidentBytes: budget })).toThrow('SCENE_BUDGET');
    expect(() => planGeometryStreamBudgets([{ directory }], { maxResidentBytes: layout.rootBytes - 1 })).toThrow('SCENE_ROOT_BUDGET');
    expect(() => planGeometryStreamBudgets([{ directory, maxPoolBytes: layout.rootBytes - 1 }], { maxResidentBytes: layout.fullBytes })).toThrow('ROOT_POOL_BUDGET');
    expect(() => planGeometryStreamBudgets([{ directory, demands: [demand(2, 10), demand(2, 1)] }], { maxResidentBytes: layout.fullBytes })).toThrow('WORKING_SET_DEMAND');
    expect(() => planGeometryStreamBudgets([{ directory, demands: null as never }], { maxResidentBytes: layout.fullBytes })).toThrow('WORKING_SET_DEMAND');
    expect(() => planGeometryStreamBudgets(new Array(1), { maxResidentBytes: layout.fullBytes })).toThrow('SCENE_INPUTS');
  });
});

describe('CPU pre-creation group demand', () => {
  const view: GeometryStreamView = { eye: [-1.5, 0.5, 3], projectionScale: 100, near: 0.1, maxErrorPixels: 2 };
  it('matches full residency refinement and combines instances/views by maximum error', () => {
    const directory = branches(), cut = new GeometryStreamCut(directory);
    const demands = cut.demands([view]);
    expect(demands.map(d => d.groupId).sort()).toEqual(cut.plan(new Set(directory.pages.map(p => p.id)), view).refinedGroups.sort());
    const other = { ...view, eye: [1.5, 0.5, 3] as const };
    const merged = cut.demands([view, other]), right = cut.demands([other]);
    for (const d of merged) expect(d.projectedError).toBe(Math.max(demands.find(x => x.groupId === d.groupId)!.projectedError, right.find(x => x.groupId === d.groupId)!.projectedError));
    expect(cut.demands([])).toEqual([]);
  });
  it('includes off-screen demand only without a frustum and accounts for object transforms', () => {
    const cut = new GeometryStreamCut(branches()), vp = identity();
    vp[0] = 2; vp[12] = 3; // Clip X range [-2,-1], preserving only the left branch.
    expect(cut.demands([{ ...view, viewProjection: vp }]).map(d => d.groupId)).toEqual([0, 2]);
    const model = identity(); model[12] = -3;
    expect(cut.demands([{ ...view, model, viewProjection: vp }]).map(d => d.groupId)).toEqual([1, 3]);
    expect(() => cut.demands([{ ...view, near: 0 }])).toThrow('STREAM_VIEW');
    expect(() => cut.demands(new Array(257).fill(view))).toThrow('DEMAND_WORK_BUDGET');
  });
});
