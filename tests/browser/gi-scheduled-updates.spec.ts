import { expect, test, type Page } from '@playwright/test';
import { evaluateGiIrradiance } from '../../src/diffuse-gi';
import type { runScheduledGiConvergence, runScheduledGiFixture, runScheduledGiSwitchFixture } from './fixtures/gi-scheduled-updates';

type Run = Awaited<ReturnType<typeof runScheduledGiFixture>>;
type Snapshot = Run['settled'];
const axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const;
const policy = { mode: 'scheduled', maxProbesPerUpdate: 8, maxStalenessUpdates: 8, convergenceUpdates: 2 } as const;

async function fixturePage(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return errors;
}
async function run(page: Page, input: Parameters<typeof runScheduledGiFixture>[0]): Promise<Run> {
  return await page.evaluate(async input => {
    const module = await import('/tests/browser/fixtures/' + 'gi-scheduled-updates.ts') as {
      runScheduledGiFixture: typeof runScheduledGiFixture };
    return module.runScheduledGiFixture(input);
  }, input) as Run;
}
function irradiance(snapshot: Snapshot): number[][] {
  return snapshot.probe.probes.map(probe => axes.flatMap(axis => evaluateGiIrradiance(probe.coefficients.flat(), axis)));
}
/** Largest per-probe difference, relative to that probe's largest irradiance. */
function fieldDifference(a: Snapshot, b: Snapshot): number {
  const left = irradiance(a), right = irradiance(b);
  let worst = 0;
  left.forEach((values, probe) => {
    const scale = Math.max(1e-4, ...values.map(Math.abs), ...right[probe].map(Math.abs));
    for (let i = 0; i < values.length; i++) worst = Math.max(worst, Math.abs(values[i] - right[probe][i]) / scale);
  });
  return worst;
}
function captured(snapshot: Snapshot): number[] {
  return snapshot.probe.probes.flatMap((probe, index) => probe.captureFrame === snapshot.probe.frame ? [index] : []);
}
function healthy(result: Run, pageErrors: string[]) {
  for (const snapshot of [result.initial, result.settled, result.moved, result.movedSettled, result.sunChanged,
    result.sunSettled, result.emissionChanged, result.emissionSettled, result.externalMarked, result.hardReset]) {
    expect(snapshot.probe.invalidProbes).toBe(0);
    expect(snapshot.probe.frame).toBe(snapshot.frame);
    for (const probe of snapshot.probe.probes) {
      expect(probe.primaryOverflows + probe.shadowOverflows + probe.emissiveOverflows).toBe(0);
      expect(probe.captureFrame).toBeGreaterThan(0);
      expect(probe.captureFrame).toBeLessThanOrEqual(snapshot.frame);
    }
  }
  expect(new Set(result.ownedBytes).size).toBe(1);
  expect(result.afterDispose).toBe(0);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
}

test('scheduled GI captures bounded dirty subsets, respects staleness and matches full single-bounce fields', async ({ page }) => {
  test.setTimeout(300_000);
  const pageErrors = await fixturePage(page);
  const full = await run(page, { multiBounce: false });
  const scheduled = await run(page, { multiBounce: false, update: policy, traceAges: true });
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(scheduled.adapter.isFallbackAdapter).toBe(false);
  healthy(full, pageErrors); healthy(scheduled, pageErrors);
  const difference = {
    initial: fieldDifference(full.initial, scheduled.initial), settled: fieldDifference(full.settled, scheduled.settled),
    moved: fieldDifference(full.moved, scheduled.moved), movedSettled: fieldDifference(full.movedSettled, scheduled.movedSettled),
    sunChanged: fieldDifference(full.sunChanged, scheduled.sunChanged), sunSettled: fieldDifference(full.sunSettled, scheduled.sunSettled),
    emissionSettled: fieldDifference(full.emissionSettled, scheduled.emissionSettled),
  };
  console.log('GI_SCHEDULED_SINGLE ' + JSON.stringify({ difference, adapter: scheduled.adapter,
    queries: { full: full.settled.queriesPerUpdate, scheduled: scheduled.settled.queriesPerUpdate },
    moved: captured(scheduled.moved), schedule: scheduled.emissionSettled.schedule,
    maxAge: Math.max(...scheduled.ages.map(age => age.maxAge)) }));
  expect(full.settled.update).toBe('full-grid-every-frame');
  expect(scheduled.settled.update).toBe('change-driven-bounded');
  // A full refresh is the same capture; after every probe has been recaptured
  // since the last change, single-bounce fields agree with full updates.
  for (const key of ['initial', 'settled', 'movedSettled', 'sunSettled', 'emissionSettled'] as const) {
    expect(difference[key], key).toBeLessThan(1e-4);
  }
  // Between changes the scheduled field is visibly stale, bounded by the policy.
  expect(difference.sunChanged).toBeGreaterThan(0.05);
  expect(captured(scheduled.initial)).toHaveLength(32);
  for (const age of scheduled.ages) {
    expect(age.maxAge).toBeLessThan(8);
    if (age.frame !== 1 && age.frame !== scheduled.hardReset.frame) expect(age.captured.length).toBeLessThanOrEqual(8);
  }
  // The moved block (old and new positions) owns the dirty slots right away.
  const movedCaptured = captured(scheduled.moved);
  expect(movedCaptured).toHaveLength(8);
  const index = (x: number, y: number, z: number) => x + 4 * (y + 2 * z);
  expect(movedCaptured).toEqual(expect.arrayContaining([index(2, 0, 2), index(1, 0, 1)]));
  const moved = scheduled.moved.schedule as { lastPlan: { dirtyProbes: number; staleProbes: number };
    changes: { transform: number }; pendingDirtyProbes: number };
  expect(moved.lastPlan).toMatchObject({ staleProbes: 4, dirtyProbes: 4 });
  // Disjoint old and new placements are two regions.
  expect(moved.changes.transform).toBe(2);
  const sun = scheduled.sunChanged.schedule as { feedbackCaptureFloorFrame: number; changes: { globalLight: number };
    pendingDirtyProbes: number; unrefreshedDirtyProbes: number };
  expect(sun.feedbackCaptureFloorFrame).toBe(scheduled.sunChanged.frame);
  expect(sun.changes.globalLight).toBe(1);
  expect(sun).toMatchObject({ pendingDirtyProbes: 32, unrefreshedDirtyProbes: 24 });
  expect(scheduled.sunSettled.schedule).toMatchObject({ pendingDirtyProbes: 0, unrefreshedDirtyProbes: 0 });
  const emission = scheduled.emissionChanged.schedule as { changes: { material: number; globalFallback: number } };
  expect(emission.changes).toMatchObject({ material: 1, globalFallback: 0 });
  expect((scheduled.externalMarked.schedule as { changes: { external: number } }).changes.external).toBe(1);
  expect(captured(scheduled.hardReset)).toHaveLength(32);
  expect(scheduled.hardReset.schedule).toMatchObject({ lastPlan: { fullRefresh: true, fullRefreshReason: 'explicit' } });
  // Query work follows the captured probe count: an idle update captures only
  // the 4-probe staleness reserve, a change uses the 8-probe budget (of 32).
  expect(scheduled.settled.queriesPerUpdate * 8).toBe(full.settled.queriesPerUpdate);
  expect(scheduled.moved.queriesPerUpdate * 4).toBe(full.moved.queriesPerUpdate);
});

test('scheduled multi-bounce feedback converges to the full-update fixed point', async ({ page }) => {
  test.setTimeout(300_000);
  const pageErrors = await fixturePage(page);
  const full = await run(page, { multiBounce: true });
  const scheduled = await run(page, { multiBounce: true, update: policy });
  healthy(full, pageErrors); healthy(scheduled, pageErrors);
  const trace = async (update?: typeof policy) => await page.evaluate(async update => {
    const module = await import('/tests/browser/fixtures/' + 'gi-scheduled-updates.ts') as {
      runScheduledGiConvergence: typeof runScheduledGiConvergence };
    return module.runScheduledGiConvergence({ update, multiBounce: true, frames: 160, every: 8 });
  }, update);
  const reference = await trace(), bounded = await trace(policy);
  expect(reference.errors).toEqual([]); expect(bounded.errors).toEqual([]);
  const final = reference.trace.at(-1)!;
  const asSnapshot = (coefficients: number[][]) => ({ probe: { probes: coefficients.map(flat =>
    ({ coefficients: Array.from({ length: 9 }, (_, j) => flat.slice(j * 3, j * 3 + 3)) })) } }) as unknown as Snapshot;
  const curve = bounded.trace.map(sample => ({ frame: sample.frame,
    difference: fieldDifference(asSnapshot(sample.coefficients), asSnapshot(final.coefficients)) }));
  const fullCurve = reference.trace.map(sample => ({ frame: sample.frame,
    difference: fieldDifference(asSnapshot(sample.coefficients), asSnapshot(final.coefficients)) }));
  console.log('GI_SCHEDULED_MULTI ' + JSON.stringify({ adapter: scheduled.adapter, full: fullCurve, scheduled: curve,
    settled: fieldDifference(full.settled, scheduled.settled) }));
  // The full-update reference has converged, and the scheduled asynchronous
  // iteration approaches the same fixed point without regressing.
  expect(fullCurve.at(-2)!.difference).toBeLessThan(1e-4);
  for (let i = 1; i < curve.length; i++) expect(curve[i].difference).toBeLessThanOrEqual(curve[i - 1].difference + 1e-4);
  expect(curve.at(-1)!.difference).toBeLessThan(0.005);
});

test('runtime GI update policy switches reuse allocations and recapture once', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-scheduled-updates.ts') as {
      runScheduledGiSwitchFixture: typeof runScheduledGiSwitchFixture };
    return module.runScheduledGiSwitchFixture();
  });
  expect(result.frames.map(frame => frame.capturedProbes)).toEqual([32, 32, 32, 8, 8, 8, 32, 32]);
  expect(result.frames.map(frame => frame.update)).toEqual([
    ...Array(2).fill('full-grid-every-frame'), ...Array(4).fill('change-driven-bounded'), ...Array(2).fill('full-grid-every-frame')]);
  expect(result.frames[2]).toMatchObject({ fullRefresh: true, reason: 'update-policy-changed', historyResetReason: 'update-policy-changed' });
  expect(result.frames[6]).toMatchObject({ historyResetReason: 'update-policy-changed' });
  expect(result.invalid).toEqual([expect.stringContaining('DIFFUSE_GI_UPDATE_STALENESS_BUDGET'),
    expect.stringContaining('DIFFUSE_GI_UPDATE_BUDGET'), expect.stringContaining('DIFFUSE_GI_UPDATE_FULL_MODE_OPTION')]);
  expect(new Set(result.frames.map(frame => frame.bytes)).size).toBe(1);
  expect(result.afterDispose).toBe(0);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
