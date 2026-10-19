import { beforeAll, describe, expect, it } from 'vitest';
import { GeometryDemoScene } from '../src/geometry-demo-scene';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { GeometryStreamWorkingSet, type GeometryGroupDemand } from '../src/geometry-stream-working-set';

function twoBranches(): GeometryStreamDirectory {
  const left = { min: [-2, 0, 0] as [number, number, number], max: [-1, 1, 1] as [number, number, number] };
  const right = { min: [1, 0, 0] as [number, number, number], max: [2, 1, 1] as [number, number, number] };
  const both = { min: [-2, 0, 0] as [number, number, number], max: [2, 1, 1] as [number, number, number] };
  return {
    version: 1, primitiveKey: 'working-set/two-branches', sourceVertexCount: 6,
    sourceTriangleCount: 2, vertexStride: 80,
    clusters: [
      { level: 0, group: 0, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds: left, lodBounds: left, error: 0 },
      { level: 0, group: 1, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds: right, lodBounds: right, error: 0 },
      { level: 1, group: 2, refinedGroup: 0, vertexCount: 3, triangleCount: 1,
        bounds: left, lodBounds: left, error: 1 },
      { level: 1, group: 2, refinedGroup: 1, vertexCount: 3, triangleCount: 1,
        bounds: right, lodBounds: right, error: 1 },
    ],
    groups: [
      { level: 0, children: [0], parents: [2], bounds: left, error: 1, terminal: false },
      { level: 0, children: [1], parents: [3], bounds: right, error: 1, terminal: false },
      { level: 1, children: [2, 3], parents: [], bounds: both, error: 1, terminal: true },
    ],
    roots: [2, 3],
    pages: [
      { id: 0, clusterIds: [0], encodedBytes: 41, decodedBytes: 272, sha256: 'a'.repeat(64) },
      { id: 1, clusterIds: [1], encodedBytes: 41, decodedBytes: 272, sha256: 'b'.repeat(64) },
      { id: 2, clusterIds: [2, 3], encodedBytes: 41, decodedBytes: 544, sha256: 'c'.repeat(64) },
    ],
    clusterPages: [0, 1, 2, 2], rootPages: [2],
  };
}

function multiPageGroup(): GeometryStreamDirectory {
  const bounds = { min: [0, 0, 0] as [number, number, number],
    max: [2, 1, 1] as [number, number, number] };
  return {
    version: 1, primitiveKey: 'working-set/atomic-group', sourceVertexCount: 6,
    sourceTriangleCount: 2, vertexStride: 80,
    clusters: [
      { level: 0, group: 0, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds, lodBounds: bounds, error: 0 },
      { level: 0, group: 0, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds, lodBounds: bounds, error: 0 },
      { level: 1, group: 1, refinedGroup: 0, vertexCount: 3, triangleCount: 1,
        bounds, lodBounds: bounds, error: 1 },
    ],
    groups: [
      { level: 0, children: [0, 1], parents: [2], bounds, error: 1, terminal: false },
      { level: 1, children: [2], parents: [], bounds, error: 1, terminal: true },
    ],
    roots: [2],
    pages: [0, 1, 2].map(id => ({ id, clusterIds: [id], encodedBytes: 41,
      decodedBytes: 272, sha256: 'a'.repeat(64) })),
    clusterPages: [0, 1, 2], rootPages: [2],
  };
}

const demand = (groupId: number, projectedError: number): GeometryGroupDemand => ({ groupId, projectedError });
let real: GeometryStreamDirectory;
beforeAll(async () => {
  const hierarchy = (await GeometryDemoScene.create()).hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  real = await buildGeometryStreamDirectory(hierarchy, encoded);
});

describe('geometry stream working-set admission', () => {
  it('keeps roots under an exact budget and never requests part of a group', () => {
    expect(() => new GeometryStreamWorkingSet(twoBranches(), 0)).toThrow('WORKING_SET_BUDGET');
    expect(() => new GeometryStreamWorkingSet(twoBranches(), 4)).toThrow('WORKING_SET_BUDGET');
    expect(() => new GeometryStreamWorkingSet(twoBranches(), 1.5)).toThrow('WORKING_SET_BUDGET');
    const rootsOnly = new GeometryStreamWorkingSet(twoBranches(), 1);
    expect(rootsOnly.plan([demand(0, 10), demand(1, 9)])).toEqual({ pages: [2], admittedGroups: [],
      budgetDeferredGroups: [0, 1], dependencyDeferredGroups: [], requestedPages: [] });
    const oneBranch = new GeometryStreamWorkingSet(twoBranches(), 2);
    expect(oneBranch.plan([demand(0, 10), demand(1, 9)])).toEqual({ pages: [0, 2], admittedGroups: [0],
      budgetDeferredGroups: [1], dependencyDeferredGroups: [], requestedPages: [{ pageId: 0, priority: 0 }] });
    const short = new GeometryStreamWorkingSet(multiPageGroup(), 2);
    expect(short.plan([demand(0, 4)])).toEqual({ pages: [2], admittedGroups: [],
      budgetDeferredGroups: [0], dependencyDeferredGroups: [], requestedPages: [] });
    const exact = new GeometryStreamWorkingSet(multiPageGroup(), 3);
    expect(exact.plan([demand(0, 4)])).toEqual({ pages: [0, 1, 2], admittedGroups: [0],
      budgetDeferredGroups: [], dependencyDeferredGroups: [],
      requestedPages: [{ pageId: 0, priority: 0 }, { pageId: 1, priority: 0 }] });
  });

  it('retains an admitted group through small error jitter, then releases it when another view wins', () => {
    const working = new GeometryStreamWorkingSet(twoBranches(), 2);
    working.plan([demand(0, 10), demand(1, 9)]);
    expect(working.plan([demand(1, 10), demand(0, 9.5)]).admittedGroups).toEqual([0]);
    expect(working.plan([demand(0, 8), demand(1, 10)])).toMatchObject({ pages: [1, 2],
      admittedGroups: [1], budgetDeferredGroups: [0], requestedPages: [{ pageId: 1, priority: 0 }] });
    expect(working.plan([])).toEqual({ pages: [2], admittedGroups: [], budgetDeferredGroups: [],
      dependencyDeferredGroups: [], requestedPages: [] });
    expect(working.plan([demand(1, 10), demand(0, 10)]).admittedGroups).toEqual([0]);
  });

  it('validates all demands before changing retained admission state', () => {
    const working = new GeometryStreamWorkingSet(twoBranches(), 2);
    working.plan([demand(0, 10), demand(1, 9)]);
    const malformed: unknown[] = [
      [demand(0, 1), demand(0, 2)], [demand(2, 1)], [demand(3, 1)],
      [demand(1, 0)], [demand(1, Infinity)], [demand(1, 3.5e38)],
      [demand(1, 1), { groupId: -1, projectedError: 1 }],
      new Array<GeometryGroupDemand>(1),
    ];
    for (const input of malformed) expect(() => working.plan(input as GeometryGroupDemand[])).toThrow('WORKING_SET_DEMAND');
    expect(working.plan([demand(1, 10), demand(0, 9.5)]).admittedGroups).toEqual([0]);
    expect(Object.isFrozen(working.directory)).toBe(true);
    expect(Object.isFrozen(working.directory.groups[0].children)).toBe(true);
  });

  it('requires ancestors on the actual encoded demo hierarchy and carries their child pages', () => {
    const roots = new Set(real.roots);
    const target = real.groups.findIndex(group => !group.terminal && group.parents.some(id => !roots.has(id)));
    expect(target).toBeGreaterThanOrEqual(0);
    const working = new GeometryStreamWorkingSet(real, real.pages.length);
    const unavailable = working.plan([demand(target, 1)]);
    expect(unavailable.admittedGroups).toEqual([]);
    expect(unavailable.dependencyDeferredGroups).toEqual([target]);
    expect(unavailable.pages).toEqual([...real.rootPages]);
    expect(unavailable.requestedPages).toEqual([]);

    const dependencies = new Set<number>([target]);
    const visit = (groupId: number): void => {
      for (const parentId of real.groups[groupId].parents) {
        if (roots.has(parentId)) continue;
        const ancestor = real.clusters[parentId].group;
        if (!dependencies.has(ancestor)) { dependencies.add(ancestor); visit(ancestor); }
      }
    };
    visit(target);
    const demands = [...dependencies].map(id => demand(id, 1));
    const admitted = working.plan(demands);
    expect(admitted.admittedGroups).toHaveLength(demands.length);
    expect(admitted.budgetDeferredGroups).toEqual([]);
    expect(admitted.dependencyDeferredGroups).toEqual([]);
    const expectedPages = [...new Set([...real.rootPages, ...[...dependencies].flatMap(id =>
      real.groups[id].children.map(child => real.clusterPages[child]))])].sort((a, b) => a - b);
    expect(admitted.pages).toEqual(expectedPages);
    const priorities = new Map<number, number>();
    for (const id of dependencies) for (const child of real.groups[id].children) {
      const pageId = real.clusterPages[child];
      if (!real.rootPages.includes(pageId)) priorities.set(pageId,
        Math.max(priorities.get(pageId) ?? -1, real.groups[id].level));
    }
    expect(admitted.requestedPages).toEqual([...priorities].map(([pageId, priority]) => ({ pageId, priority }))
      .sort((a, b) => b.priority - a.priority || a.pageId - b.pageId));
  });
});
