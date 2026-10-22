import { describe, expect, it } from 'vitest';
import { planWorldQueryTraceBudget, type WorldQueryWorkPartition } from '../src/gpu-world-query';

const limits = { maxRays: 65_536, maxNodeVisits: 262_144, maxTotalWorkUnits: 67_108_864 };

describe('world query periodic budget partition', () => {
  it('preserves the Sponza primary quota while adding an explicitly bounded supplemental quota', () => {
    const baseline = planWorldQueryTraceBudget(57_600, limits);
    const combined = planWorldQueryTraceBudget(65_322, limits, undefined,
      { raysPerBlock: 65_322, primaryRaysPerBlock: 57_600, supplementalMaxWorkUnits: 8_388_608 });
    expect(baseline.primaryWorkPerRay).toBe(1165);
    expect(combined).toEqual({ raysPerBlock: 65_322, primaryRaysPerBlock: 57_600,
      primaryWorkPerRay: 1165, supplementalWorkPerRay: 1086, primaryRayCount: 57_600,
      supplementalRayCount: 7722, allocatedWorkUnits: 75_490_092, maxTotalWorkUnits: 75_497_472 });
    expect(combined.primaryWorkPerRay).toBe(baseline.primaryWorkPerRay);
    expect(Object.isFrozen(combined)).toBe(true);
  });

  it('assigns primary prefixes within every source-major area block, rather than one global prefix', () => {
    const plan = planWorldQueryTraceBudget(24, limits, 100,
      { raysPerBlock: 8, primaryRaysPerBlock: 5, supplementalMaxWorkUnits: 18 });
    const quotas = Array.from({ length: 24 }, (_, ray) => ray % plan.raysPerBlock < plan.primaryRaysPerBlock
      ? plan.primaryWorkPerRay : plan.supplementalWorkPerRay);
    expect(quotas).toEqual([6, 6, 6, 6, 6, 2, 2, 2, 6, 6, 6, 6, 6, 2, 2, 2, 6, 6, 6, 6, 6, 2, 2, 2]);
    expect(plan.primaryRayCount).toBe(15); expect(plan.supplementalRayCount).toBe(9);
    expect(quotas.reduce((sum, quota) => sum + quota, 0)).toBe(108);
    expect(plan.maxTotalWorkUnits).toBe(118);
  });

  it('bounds both subsets and matches separate-query quotas across block shapes and node caps', () => {
    for (const maxNodeVisits of [1, 7, 262_144]) for (let block = 2; block <= 16; block++) {
      for (let prefix = 1; prefix < block; prefix++) for (let blocks = 1; blocks <= 8; blocks++) {
        const options = { ...limits, maxNodeVisits }, count = block * blocks;
        const plan = planWorldQueryTraceBudget(count, options, 1000,
          { raysPerBlock: block, primaryRaysPerBlock: prefix, supplementalMaxWorkUnits: 500 });
        expect(plan.primaryWorkPerRay).toBe(planWorldQueryTraceBudget(blocks * prefix, options, 1000).primaryWorkPerRay);
        expect(plan.supplementalWorkPerRay).toBe(planWorldQueryTraceBudget(blocks * (block - prefix), options, 500).primaryWorkPerRay);
        expect(plan.primaryRayCount * plan.primaryWorkPerRay).toBeLessThanOrEqual(1000);
        expect(plan.supplementalRayCount * plan.supplementalWorkPerRay).toBeLessThanOrEqual(500);
        expect(plan.allocatedWorkUnits).toBeLessThanOrEqual(plan.maxTotalWorkUnits);
      }
    }
  });

  it('rejects ambiguous shapes and budgets that cannot fund both nonempty subsets', () => {
    const valid = { raysPerBlock: 8, primaryRaysPerBlock: 4, supplementalMaxWorkUnits: 100 };
    for (const value of [0, -1, 1.5, NaN, Infinity, '8', null]) {
      for (const key of ['raysPerBlock', 'primaryRaysPerBlock'] as const) {
        expect(() => planWorldQueryTraceBudget(16, limits, 100,
          { ...valid, [key]: value } as unknown as WorldQueryWorkPartition)).toThrow('WORLD_QUERY_WORK_PARTITION');
      }
    }
    for (const shape of [{ raysPerBlock: 1 }, { raysPerBlock: 17 }, { raysPerBlock: 3 },
      { primaryRaysPerBlock: 8 }, { primaryRaysPerBlock: 9 }]) {
      expect(() => planWorldQueryTraceBudget(16, limits, 100, { ...valid, ...shape })).toThrow('WORLD_QUERY_WORK_PARTITION');
    }
    expect(() => planWorldQueryTraceBudget(0, limits, 100, valid)).toThrow('WORLD_QUERY_WORK_PARTITION');
    expect(() => planWorldQueryTraceBudget(16, limits, 100, null as unknown as WorldQueryWorkPartition))
      .toThrow('WORLD_QUERY_WORK_PARTITION');
    for (const budget of [0, -1, 1.5, NaN, Infinity, 67_108_865]) {
      expect(() => planWorldQueryTraceBudget(16, limits, 100, { ...valid, supplementalMaxWorkUnits: budget }))
        .toThrow('WORLD_QUERY_TRACE_WORK_BUDGET');
    }
    expect(() => planWorldQueryTraceBudget(16, limits, 7, valid)).toThrow('WORLD_QUERY_WORK_BUDGET');
    expect(() => planWorldQueryTraceBudget(16, limits, 100, { ...valid, supplementalMaxWorkUnits: 7 }))
      .toThrow('WORLD_QUERY_WORK_BUDGET');
    expect(planWorldQueryTraceBudget(0, limits).allocatedWorkUnits).toBe(0);
  });
});
