import { describe, expect, it } from 'vitest';
import { GI_MAX_CHANGE_REGIONS_PER_UPDATE, GiProbeScheduler, resolveDiffuseGiUpdateOptions,
  type GiProbeGrid } from '../src/gi-probe-schedule';
import { resolveDiffuseGiOptions } from '../src/diffuse-gi';
import { lookAt, multiply, perspective } from '../src/math';

const grid: GiProbeGrid = { origin: [0, 0, 0], spacing: [1, 1, 1], counts: [8, 4, 8], maxDistance: 12 };
const probeCount = 8 * 4 * 8;
function scheduler(options: Parameters<typeof resolveDiffuseGiUpdateOptions>[0] = { mode: 'scheduled' }) {
  return new GiProbeScheduler(grid, resolveDiffuseGiUpdateOptions(options, probeCount));
}
function probeIndex(x: number, y: number, z: number): number { return x + 8 * (y + 4 * z); }
function random(seed: number) {
  return () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
}

describe('GI probe update policy', () => {
  it('defaults to the legacy full update and rejects scheduled-only options there', () => {
    const full = resolveDiffuseGiUpdateOptions(undefined, 216);
    expect(full).toMatchObject({ mode: 'full', maxProbesPerUpdate: 216, maxStalenessUpdates: 1, staleReserve: 216 });
    expect(Object.isFrozen(full)).toBe(true);
    expect(() => resolveDiffuseGiUpdateOptions({ mode: 'full', maxProbesPerUpdate: 8 }, 216))
      .toThrow('DIFFUSE_GI_UPDATE_FULL_MODE_OPTION');
    expect(resolveDiffuseGiOptions({ origin: [0, 0, 0], spacing: [1, 1, 1], counts: [2, 2, 2] }).update.mode).toBe('full');
  });

  it('resolves an already resolved policy to the same values', () => {
    for (const input of [undefined, { mode: 'scheduled' as const, maxStalenessUpdates: 5, maxProbesPerUpdate: 60 }]) {
      const once = resolveDiffuseGiUpdateOptions(input, 216);
      expect(resolveDiffuseGiUpdateOptions(once, 216)).toEqual(once);
    }
    const options = resolveDiffuseGiOptions({ origin: [0, 0, 0], spacing: [1, 1, 1], counts: [4, 4, 4],
      update: { mode: 'scheduled' } });
    expect(resolveDiffuseGiOptions(options).update).toEqual(options.update);
    expect(resolveDiffuseGiOptions(resolveDiffuseGiOptions({ origin: [0, 0, 0], spacing: [1, 1, 1], counts: [4, 4, 4] })).update.mode)
      .toBe('full');
  });

  it('resolves bounded scheduled defaults and refuses budgets that cannot meet the staleness bound', () => {
    const value = resolveDiffuseGiUpdateOptions({ mode: 'scheduled' }, 216);
    expect(value).toMatchObject({ mode: 'scheduled', maxStalenessUpdates: 8, staleReserve: 27, maxProbesPerUpdate: 108,
      convergenceUpdates: 8, dirtyThreshold: 0.002, offscreenPriority: 0.25 });
    expect(resolveDiffuseGiUpdateOptions({ mode: 'scheduled', maxStalenessUpdates: 4, maxProbesPerUpdate: 54 }, 216).staleReserve)
      .toBe(54);
    expect(() => resolveDiffuseGiUpdateOptions({ mode: 'scheduled', maxStalenessUpdates: 4, maxProbesPerUpdate: 53 }, 216))
      .toThrow('DIFFUSE_GI_UPDATE_STALENESS_BUDGET');
    for (const invalid of [{ maxProbesPerUpdate: 0 }, { maxProbesPerUpdate: 217 }, { maxProbesPerUpdate: 1.5 },
      { maxStalenessUpdates: 0 }, { maxStalenessUpdates: 1025 }, { convergenceUpdates: 0 }, { convergenceUpdates: 17 },
      { dirtyThreshold: 0 }, { dirtyThreshold: 1.1 }, { dirtyThreshold: NaN }, { offscreenPriority: 0 }]) {
      expect(() => resolveDiffuseGiUpdateOptions({ mode: 'scheduled', ...invalid }, 216)).toThrow(/DIFFUSE_GI_UPDATE_/);
    }
    expect(() => resolveDiffuseGiUpdateOptions({ mode: 'sometimes' as 'full' }, 216)).toThrow('DIFFUSE_GI_UPDATE_MODE');
    expect(() => resolveDiffuseGiOptions({ origin: [0, 0, 0], spacing: [1, 1, 1], counts: [2, 2, 2],
      update: { mode: 'scheduled', maxProbesPerUpdate: 9 } })).toThrow('DIFFUSE_GI_UPDATE_BUDGET');
  });

  it('starts with one full refresh, then never exceeds the budget or the staleness bound', () => {
    const options = { mode: 'scheduled' as const, maxStalenessUpdates: 6, maxProbesPerUpdate: 64 };
    const value = scheduler(options);
    const next = random(7);
    let maxAge = 0;
    for (let frame = 1; frame <= 600; frame++) {
      // Random local changes, occasional global light changes.
      for (let mark = 0; mark < 3; mark++) if (next() < 0.5) {
        const x = next() * 7, y = next() * 3, z = next() * 7, size = next() * 2;
        value.markRegion({ min: [x, y, z], max: [x + size, y + size, z + size] }, 'transform');
      }
      if (next() < 0.02) value.markAll('light', frame);
      const plan = value.plan(frame);
      if (frame === 1) {
        expect(plan).toMatchObject({ fullRefresh: true, fullRefreshReason: 'initial', count: probeCount });
      } else {
        expect(plan.fullRefresh).toBe(false);
        expect(plan.count).toBeLessThanOrEqual(64);
        expect(plan.staleProbes).toBe(Math.ceil(probeCount / 6));
        expect(new Set(plan.probes).size).toBe(plan.count);
      }
      value.commit(plan);
      for (let probe = 0; probe < probeCount; probe++) maxAge = Math.max(maxAge, value.ageOf(probe));
      const diagnostics = value.diagnostics();
      expect(diagnostics.maxAgeUpdates).toBeLessThan(6);
      expect(diagnostics.neverCapturedProbes).toBe(0);
    }
    // Age is measured right after a commit; between captures a probe is at most
    // maxStalenessUpdates updates old when it is captured again.
    expect(maxAge).toBe(5);
    expect(value.diagnostics().totals).toMatchObject({ updates: 600, fullRefreshes: 1 });
  });

  it('converges the whole grid after a hard reset, then idles at the staleness reserve', () => {
    const value = scheduler({ mode: 'scheduled', maxStalenessUpdates: 8, maxProbesPerUpdate: 64, convergenceUpdates: 3 });
    let plan = value.plan(1); value.commit(plan);
    expect(value.diagnostics()).toMatchObject({ pendingDirtyProbes: probeCount, pendingCaptures: probeCount * 2 });
    const counts: number[] = [];
    for (let frame = 2; frame <= 40; frame++) { plan = value.plan(frame); counts.push(plan.count); value.commit(plan); }
    // 256 probes x 2 remaining captures at 64 per update take 8 updates.
    expect(counts.slice(0, 8).every(count => count === 64)).toBe(true);
    expect(counts.slice(8).every(count => count === 32)).toBe(true);
    expect(value.diagnostics().pendingDirtyProbes).toBe(0);
    value.requestFullRefresh('explicit');
    plan = value.plan(41);
    expect(plan).toMatchObject({ fullRefresh: true, fullRefreshReason: 'explicit', count: probeCount });
    value.commit(plan);
    expect(value.diagnostics().totals.fullRefreshes).toBe(2);
  });

  it('prioritizes probes near a local change and leaves distant probes to background refresh', () => {
    const value = scheduler({ mode: 'scheduled', maxStalenessUpdates: 32, maxProbesPerUpdate: 40, convergenceUpdates: 1 });
    let frame = 1;
    for (; frame <= 12; frame++) value.commit(value.plan(frame));
    expect(value.diagnostics().pendingDirtyProbes).toBe(0);
    value.markRegion({ min: [1.9, 0.9, 1.9], max: [2.1, 1.1, 2.1] }, 'transform');
    const plan = value.plan(frame);
    const dirty = [...plan.probes.subarray(plan.staleProbes)];
    expect(plan.staleProbes).toBe(8);
    // Strongest influence first: the probe at the change, then by solid angle,
    // which decreases with distance.
    expect(dirty[0]).toBe(probeIndex(2, 1, 2));
    expect(dirty).not.toContain(probeIndex(7, 3, 7));
    const distance = (probe: number) => Math.hypot(probe % 8 - 2, Math.floor(probe / 8) % 4 - 1, Math.floor(probe / 32) - 2);
    for (let i = 1; i < dirty.length; i++) expect(distance(dirty[i])).toBeGreaterThanOrEqual(distance(dirty[i - 1]) - 1e-9);
    // Solid-angle fraction >= 0.002 of a 0.17 m sphere reaches ~1.94 m (27
    // probes); probes within one cell are marked regardless.
    expect(dirty).toHaveLength(27);
    expect(value.diagnostics()).toMatchObject({ pendingDirtyProbes: 27, unrefreshedDirtyProbes: 27 });
    value.commit(plan);
    const after = value.diagnostics();
    expect(after).toMatchObject({ pendingDirtyProbes: 0, unrefreshedDirtyProbes: 0 });
    expect(after.totals).toMatchObject({ regionMarks: 1, markedProbes: 27 });
  });

  it('marks probes within one cell of a tiny change and gives first captures priority over convergence', () => {
    const value = scheduler({ mode: 'scheduled', maxStalenessUpdates: 64, maxProbesPerUpdate: 12, convergenceUpdates: 4 });
    value.commit(value.plan(1));
    for (let frame = 2; frame <= 400 && value.diagnostics().pendingDirtyProbes; frame++) value.commit(value.plan(frame));
    expect(value.diagnostics().pendingDirtyProbes).toBe(0);
    let frame = value.diagnostics().lastCommittedFrame + 1;
    // A 1 cm change subtends far less than the threshold, yet its cell corners recapture.
    value.markRegion({ min: [3.495, 1.495, 3.495], max: [3.505, 1.505, 3.505] }, 'external');
    let plan = value.plan(frame);
    const corners = [probeIndex(3, 1, 3), probeIndex(4, 1, 3), probeIndex(3, 2, 3), probeIndex(4, 2, 3),
      probeIndex(3, 1, 4), probeIndex(4, 1, 4), probeIndex(3, 2, 4), probeIndex(4, 2, 4)];
    expect(value.diagnostics()).toMatchObject({ pendingDirtyProbes: 8, unrefreshedDirtyProbes: 8 });
    expect([...plan.probes]).toEqual(expect.arrayContaining(corners));
    value.commit(plan); frame++;
    expect(value.diagnostics()).toMatchObject({ pendingDirtyProbes: 8, unrefreshedDirtyProbes: 0 });
    // A second change elsewhere is captured before the first one converges.
    value.markRegion({ min: [0, 0, 0], max: [0.1, 0.1, 0.1] }, 'transform');
    plan = value.plan(frame);
    const fresh = [probeIndex(0, 0, 0), probeIndex(1, 0, 0), probeIndex(0, 1, 0), probeIndex(0, 0, 1)];
    expect(value.diagnostics().unrefreshedDirtyProbes).toBe(4);
    expect([...plan.probes]).toEqual(expect.arrayContaining(fresh));
    const dirty = [...plan.probes.subarray(plan.staleProbes)];
    const firstConverging = dirty.findIndex(probe => corners.includes(probe));
    expect(firstConverging).toBeGreaterThan(-1);
    for (const probe of fresh) if (dirty.includes(probe)) expect(dirty.indexOf(probe)).toBeLessThan(firstConverging);
    value.commit(plan);
    expect(value.diagnostics().unrefreshedDirtyProbes).toBe(0);
  });

  it('orders equally dirty probes by the expanded view frustum', () => {
    const value = scheduler({ mode: 'scheduled', maxStalenessUpdates: 64, maxProbesPerUpdate: 12, convergenceUpdates: 1 });
    value.commit(value.plan(1));
    value.markAll('light', 2);
    // Camera at x=-6 looking along +x down the y=1.5, z=0..1 corner of the grid.
    const viewProjection = multiply(perspective(0.4, 1, 0.1, 100), lookAt([-6, 1.5, 0.5], [0, 1.5, 0.5], [0, 1, 0]));
    value.setView({ eye: [-6, 1.5, 0.5], viewProjection });
    const plan = value.plan(2);
    const dirty = [...plan.probes.subarray(plan.staleProbes)];
    expect(dirty).toHaveLength(8);
    for (const probe of dirty) expect(Math.floor(probe / 32)).toBeLessThanOrEqual(2);
  });

  it('bounds per-update region work by merging marks beyond the cap', () => {
    const value = scheduler({ mode: 'scheduled' });
    value.commit(value.plan(1));
    for (let i = 0; i < GI_MAX_CHANGE_REGIONS_PER_UPDATE + 10; i++) {
      value.markRegion({ min: [i % 8, 0, 0], max: [i % 8 + 0.1, 0.1, 0.1] }, 'transform');
    }
    value.commit(value.plan(2));
    expect(value.diagnostics().totals).toMatchObject({ regionMarks: GI_MAX_CHANGE_REGIONS_PER_UPDATE, mergedRegionMarks: 11 });
    expect(() => value.markRegion({ min: [1, 0, 0], max: [0, 1, 1] }, 'transform')).toThrow('DIFFUSE_GI_CHANGE_REGION');
    expect(() => value.markRegion({ min: [0, 0, NaN], max: [1, 1, 1] }, 'transform')).toThrow('DIFFUSE_GI_CHANGE_REGION');
  });

  it('rejects stale plans and non-increasing frames', () => {
    const value = scheduler({ mode: 'scheduled' });
    const first = value.plan(1);
    value.commit(first);
    expect(() => value.commit(first)).toThrow('DIFFUSE_GI_SCHEDULE_PLAN_MISMATCH');
    expect(() => value.plan(1)).toThrow('DIFFUSE_GI_SCHEDULE_FRAME');
    const replaced = value.plan(2), current = value.plan(3);
    expect(() => value.commit(replaced)).toThrow('DIFFUSE_GI_SCHEDULE_PLAN_MISMATCH');
    value.commit(current);
    expect(value.diagnostics().lastCommittedFrame).toBe(3);
  });

  it('full mode plans the identity schedule every update', () => {
    const value = new GiProbeScheduler(grid, resolveDiffuseGiUpdateOptions(undefined, probeCount));
    for (let frame = 1; frame <= 3; frame++) {
      const plan = value.plan(frame);
      expect(plan.count).toBe(probeCount);
      expect([...plan.probes]).toEqual(Array.from({ length: probeCount }, (_, index) => index));
      value.commit(plan);
      expect(value.diagnostics().maxAgeUpdates).toBe(0);
    }
  });
});
