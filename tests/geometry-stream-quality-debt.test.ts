import { describe, expect, it } from 'vitest';
import { diagnoseGeometryStreamQualityDebt } from '../src/geometry-stream-quality-debt';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';
import { identity, translation } from '../src/math';

// One replacement needs both child pages. The root alone remains drawable.
function twoPageReplacement(): GeometryStreamDirectory {
  const bounds = { min: [0, 0, 0] as [number, number, number],
    max: [1, 1, 1] as [number, number, number] };
  return {
    version: 1, primitiveKey: 'quality-debt/two-page', sourceVertexCount: 6,
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

function threeLevelRefinement(): GeometryStreamDirectory {
  const base = twoPageReplacement(), bounds = base.groups[0].bounds;
  return {
    ...base, primitiveKey: 'quality-debt/three-level',
    clusters: [...base.clusters,
      { level: 2, group: 2, refinedGroup: 1, vertexCount: 3, triangleCount: 1,
        bounds, lodBounds: bounds, error: 2 }],
    groups: [base.groups[0], { ...base.groups[1], parents: [3], error: 2, terminal: false },
      { level: 2, children: [3], parents: [], bounds, error: 2, terminal: true }],
    roots: [3],
    pages: [0, 1, 2, 3].map(id => ({ id, clusterIds: [id], encodedBytes: 41,
      decodedBytes: 272, sha256: 'a'.repeat(64) })),
    clusterPages: [0, 1, 2, 3], rootPages: [3],
  };
}

const nearView = { eye: [0.5, 0.5, 3] as [number, number, number],
  projectionScale: 100, near: 0.1, maxErrorPixels: 2, model: identity() };
const pages = (...ids: number[]) => new Set(ids);
const diagnose = (residentPages: ReadonlySet<number>, admittedPages: ReadonlySet<number>, failedPages = pages()) =>
  diagnoseGeometryStreamQualityDebt({ directory: twoPageReplacement(), views: [nearView],
    residentPages, admittedPages, failedPages });

describe('CPU predicted geometry-stream quality debt', () => {
  it('reports capacity-deferred child pages at the selected fallback frontier', () => {
    const debt = diagnose(pages(2), pages(2));
    expect(debt).toMatchObject({ basis: 'cpu-predicted-visible-cut', instanceCount: 1,
      fallbackGroupInstances: 1, groupsByCause: { 'capacity-deferred': 1,
        'admitted-not-resident': 0, failed: 0, mixed: 0 } });
    expect(debt.maxProjectedApproximateErrorPixels).toBeCloseTo(50, 3);
    expect(debt.details).toMatchObject([{ instanceIndex: 0, groupId: 0,
      requestedErrorPixels: 2, cause: 'capacity-deferred',
      missingChildPages: [0, 1], capacityDeferredPages: [0, 1], admittedNotResidentPages: [], failedPages: [] }]);
    expect(debt.details[0].projectedApproximateErrorPixels).toBeCloseTo(50, 3);
  });

  it('separates admitted-but-unresident and failed pages, and reports mixed causes', () => {
    expect(diagnose(pages(2), pages(0, 1, 2)).details[0]).toMatchObject({
      cause: 'admitted-not-resident', capacityDeferredPages: [], admittedNotResidentPages: [0, 1], failedPages: [],
    });
    expect(diagnose(pages(1, 2), pages(0, 1, 2), pages(0)).details[0]).toMatchObject({
      cause: 'failed', capacityDeferredPages: [], admittedNotResidentPages: [], failedPages: [0],
    });
    expect(diagnose(pages(2), pages(0, 2)).details[0]).toMatchObject({
      cause: 'mixed', capacityDeferredPages: [1], admittedNotResidentPages: [0], failedPages: [],
    });
  });

  it('reports no debt once the complete replacement is resident', () => {
    expect(diagnose(pages(0, 1, 2), pages(0, 1, 2))).toMatchObject({
      fallbackGroupInstances: 0, maxProjectedApproximateErrorPixels: 0, details: [],
      groupsByCause: { 'capacity-deferred': 0, 'admitted-not-resident': 0, failed: 0, mixed: 0 },
    });
  });

  it('excludes an off-screen fallback from visible quality debt', () => {
    const debt = diagnoseGeometryStreamQualityDebt({ directory: twoPageReplacement(),
      views: [{ ...nearView, viewProjection: translation(10, 0, 0) }],
      residentPages: pages(2), admittedPages: pages(2) });
    expect(debt.fallbackGroupInstances).toBe(0);
    expect(debt.details).toEqual([]);
  });

  it('counts instance-specific frontier error instead of applying a global maximum to every instance', () => {
    const debt = diagnoseGeometryStreamQualityDebt({ directory: twoPageReplacement(),
      views: [nearView, { ...nearView, model: translation(0, 0, 100) }],
      residentPages: pages(2), admittedPages: pages(2) });
    expect(debt.instanceCount).toBe(2);
    expect(debt.fallbackGroupInstances).toBe(1);
    expect(debt.details.map(detail => detail.instanceIndex)).toEqual([0]);
  });

  it('moves debt to a deeper frontier only after its parent replacement is resident', () => {
    const directory = threeLevelRefinement();
    const replay = (residentPages: ReadonlySet<number>, admittedPages: ReadonlySet<number>) =>
      diagnoseGeometryStreamQualityDebt({ directory, views: [nearView], residentPages, admittedPages });
    const rootOnly = replay(pages(3), pages(3));
    expect(rootOnly.fallbackGroupInstances).toBe(1);
    expect(rootOnly.details).toMatchObject([{ groupId: 1, missingChildPages: [2],
      capacityDeferredPages: [2] }]);
    expect(rootOnly.maxProjectedApproximateErrorPixels).toBeCloseTo(100, 3);

    const middleResident = replay(pages(2, 3), pages(2, 3));
    expect(middleResident.fallbackGroupInstances).toBe(1);
    expect(middleResident.details).toMatchObject([{ groupId: 0, missingChildPages: [0, 1],
      capacityDeferredPages: [0, 1] }]);
    expect(middleResident.maxProjectedApproximateErrorPixels).toBeCloseTo(50, 3);
    expect(replay(pages(0, 1, 2, 3), pages(0, 1, 2, 3)).fallbackGroupInstances).toBe(0);
  });

  it('rejects invalid page snapshots and excessive replay work', () => {
    expect(() => diagnose(pages(), pages(2))).toThrow('ROOT_PAGE_MISSING');
    expect(() => diagnose(pages(2), pages())).toThrow('QUALITY_DEBT_ADMISSION');
    expect(() => diagnose(pages(2, 99), pages(2))).toThrow('QUALITY_DEBT_RESIDENCY');
    const negativeSize = pages(2);
    Object.defineProperty(negativeSize, 'size', { value: -1 });
    expect(() => diagnose(negativeSize, pages(2))).toThrow('QUALITY_DEBT_RESIDENCY');
    expect(() => diagnoseGeometryStreamQualityDebt({ directory: twoPageReplacement(),
      views: Array(257).fill(nearView), residentPages: pages(2), admittedPages: pages(2) }))
      .toThrow('QUALITY_DEBT_WORK_BUDGET');
  });
});
