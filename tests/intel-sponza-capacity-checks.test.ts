import { describe, expect, it } from 'vitest';
import { assertCapacityCut, assertCapacityLedger, capacityExchange, interpolateCapacityCamera } from './browser/fixtures/intel-sponza-capacity-checks';

const ledger = { liveBytes: 80, reservedBytes: 40, accountedBytes: 120, peakBytes: 120, error: null,
  limits: { maxResidentBytes: 100, maxPeakBytes: 140 } };
const bounds = { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] };
// The oracle consumes cluster counts and page IDs, not directory construction.
const directory = { clusters: [2, 3, 1].map(triangleCount => ({ triangleCount, vertexCount: 3, bounds, lodBounds: bounds,
  level: 0, group: 0, refinedGroup: null, error: 0 })), clusterPages: [0, 1, 2] };
const selected = [{ clusterId: 0, instanceIndex: 0, visible: true }, { clusterId: 1, instanceIndex: 0, visible: false },
  { clusterId: 2, instanceIndex: 1, visible: true }];
const feedback = { indices: 9, overflow: false, selectedClusters: selected };

describe('real Sponza capacity acceptance oracles', () => {
  it('charges retiring/candidate reservation and permits only declared geometry budgets', () => {
    expect(() => assertCapacityLedger(ledger)).not.toThrow();
    for (const mutation of [{ liveBytes: 101, accountedBytes: 141, peakBytes: 141 }, { reservedBytes: 61, accountedBytes: 141, peakBytes: 141 },
      { accountedBytes: 80 }, { peakBytes: 119 }, { reservedBytes: -1 }, { error: 'queue failed' }]) {
      expect(() => assertCapacityLedger({ ...ledger, ...mutation })).toThrow('SPONZA_CAPACITY_LEDGER');
    }
  });
  it('checks pending growth, aggregate copy budgets and concurrency, while allowing historical ledgers', () => {
    const parallel = { ...ledger, pendingGrowthBytes: 20, projectedLiveBytes: 100, inFlightMigrations: 2,
      copyReservations: { bytes: 30, pages: 4 }, scheduling: { maxCopyBytes: 32, maxCopyPages: 4, maxConcurrentMigrations: 4, intervalFrames: 1 } };
    expect(() => assertCapacityLedger(ledger)).not.toThrow();
    expect(() => assertCapacityLedger(parallel)).not.toThrow();
    for (const mutation of [{ projectedLiveBytes: 80 }, { pendingGrowthBytes: 21, projectedLiveBytes: 101 },
      { inFlightMigrations: 5 }, { copyReservations: { bytes: 33, pages: 4 } }, { copyReservations: { bytes: 30, pages: 5 } },
      { pendingGrowthBytes: -1 }, { pendingGrowthBytes: undefined }, { scheduling: undefined }, { inFlightMigrations: 0 }]) {
      expect(() => assertCapacityLedger({ ...parallel, ...mutation })).toThrow('PARALLEL_LEDGER');
    }
  });
  it('accepts a complete resident cut including culled members and counts only actually drawn triangles', () => {
    expect(assertCapacityCut(directory, new Set([0, 1, 2]), [[0, 1], [2]], feedback))
      .toEqual({ completeClusters: 3, visibleClusters: 2, triangles: 3, indices: 9 });
  });
  it('rejects an incomplete cut even when drawn triangle counters remain plausible', () => {
    expect(() => assertCapacityCut(directory, new Set([0, 1, 2]), [[0, 1], [2]],
      { ...feedback, selectedClusters: [selected[0], selected[2]] })).toThrow('INCOMPLETE_CUT');
  });
  it('rejects duplicate clusters, wrong instance IDs, and nonresident selection', () => {
    for (const extra of [selected[0], { ...selected[0], instanceIndex: 2 }, { ...selected[0], clusterId: 9 }]) {
      expect(() => assertCapacityCut(directory, new Set([0, 1, 2]), [[0, 1], [2]],
        { ...feedback, selectedClusters: [...selected, extra] })).toThrow('CUT_MEMBER');
    }
    expect(() => assertCapacityCut(directory, new Set([0, 2]), [[0, 1], [2]], feedback)).toThrow('CUT_MEMBER');
  });
  it('rejects overflow and draws that disagree with visible cluster triangle counts', () => {
    expect(() => assertCapacityCut(directory, new Set([0, 1, 2]), [[0, 1], [2]], { ...feedback, overflow: true })).toThrow('CUT_UNAVAILABLE');
    expect(() => assertCapacityCut(directory, new Set([0, 1, 2]), [[0, 1], [2]], { ...feedback, indices: 18 })).toThrow('DRAW_INDICES');
  });
  it('requires capacity to move in both directions for an exchange, not just grow unconstrained', () => {
    expect(capacityExchange([4, 2, 1], [2, 4, 1])).toEqual({ grown: [1], shrunk: [0], exchanged: true });
    expect(capacityExchange([4, 2, 1], [4, 4, 1]).exchanged).toBe(false);
    expect(capacityExchange([4, 2, 1], [4, 2, 1]).exchanged).toBe(false);
  });
  it('keeps eye/target distinct throughout an opposite-heading walk and a stationary 180 degree turn', () => {
    const a = { eye: [9, 1.8, 0] as [number, number, number], target: [-6, 3.5, 0] as [number, number, number] };
    for (const b of [{ eye: [-9, 1.8, 0] as [number, number, number], target: [6, 3.5, 0] as [number, number, number] },
      { eye: a.eye, target: [24, 3.5, 0] as [number, number, number] }]) {
      for (let step = 0; step <= 48; step++) {
        const camera = interpolateCapacityCamera(a, b, step / 48);
        expect(Math.hypot(...camera.target.map((v, i) => v - camera.eye[i]))).toBeGreaterThan(15);
        expect(camera.eye[1]).toBeCloseTo(1.8);
      }
      interpolateCapacityCamera(a, b, 0).target.forEach((value, axis) => expect(value).toBeCloseTo(a.target[axis]));
      interpolateCapacityCamera(a, b, 1).target.forEach((value, axis) => expect(value).toBeCloseTo(b.target[axis]));
    }
  });
});
