import { expect, test, type Page } from '@playwright/test';
import { giRayDirection } from '../../src/diffuse-gi';
import type { runGiSupportFixture, runGiSupportHistoryFixture } from './fixtures/gi-support';

async function setup(page: Page) {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__);
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return errors;
}
type Capture = Awaited<ReturnType<typeof runGiSupportFixture>>['ordinaryBefore'];
function moments(capture: Capture) {
  expect(capture.actual.invalidProbes).toBe(0);
  capture.actual.probes.forEach((probe, p) => {
    expect([probe.primaryOverflows, probe.shadowOverflows, probe.emissiveOverflows, probe.visibilityOverflows]).toEqual([0,0,0,0]);
    probe.moments.forEach((value, cell) => value.forEach((channel, c) => {
      expect(Math.abs(channel - capture.expected[p][cell][c]), `${capture.direction} probe ${p}, cell ${cell}, channel ${c}`).toBeLessThan(c === 1 ? 0.005 : 0.001);
    }));
  });
}
for (const filter of ['moment-bilinear', 'per-tap-chebyshev-bilinear'] as const)
test(`receiver-directed GI moments match analytic sided planes with ${filter} without changing sun or emissive SH`, async ({ page }) => {
  const errors = await setup(page);
  const r = await page.evaluate(async filter => {
    const f = await import('/tests/browser/fixtures/' + 'gi-support.ts') as { runGiSupportFixture: typeof runGiSupportFixture };
    return f.runGiSupportFixture(false, filter);
  }, filter);
  r.rayDirections.forEach((direction, i) => {
    const expected = giRayDirection(i, 64);
    direction.forEach((v, axis) => expect(Math.abs(v - expected[axis]), `ray ${i}, axis ${axis}`).toBeLessThan(0.001));
    expect(Math.abs(Math.hypot(...direction) - 1), `ray ${i} length`).toBeLessThan(0.001);
  });
  for (const c of [r.ordinaryBefore, r.reversed, r.ordinaryAfter, ...r.sequence]) moments(c);
  for (let p = 0; p < 8; p++) {
    expect(r.reversed.actual.probes[p].coefficients).toEqual(r.ordinaryBefore.actual.probes[p].coefficients);
    expect(r.ordinaryAfter.actual.probes[p].coefficients).toEqual(r.ordinaryBefore.actual.probes[p].coefficients);
  }
  expect(r.reversed.actual.probes.map(p => p.moments)).not.toEqual(r.ordinaryBefore.actual.probes.map(p => p.moments));
  expect(r.queryAfter.hits).toEqual(r.queryBefore.hits);
  expect(r.queryBefore.hits[0]?.distance).toBeCloseTo(2, 5); expect(r.queryBefore.hits[1]).toBeNull();
  expect(r.reversed.bytes).toBe(r.ordinaryBefore.bytes);
  expect(r.reversed.diagnostics!.queriesPerUpdate - r.ordinaryBefore.diagnostics!.queriesPerUpdate).toBe(512);
  expect(r.reversed.diagnostics!.maxWorkUnitsPerUpdate - r.ordinaryBefore.diagnostics!.maxWorkUnitsPerUpdate).toBe(65536);
  expect(r.reversed.diagnostics!.visibilityQueriesPerUpdate).toBe(512);
  for (const capture of [r.ordinaryBefore, r.reversed, r.ordinaryAfter, ...r.sequence]) {
    expect(capture.diagnostics!.visibilityFilter).toBe(filter);
  }
  expect(errors).toEqual([]); expect(r.errors).toEqual([]); expect(r.afterDispose).toBe(0);
});

test('support-only traversal overflow invalidates probes and switching back clears its metadata', async ({ page }) => {
  const errors = await setup(page);
  const r = await page.evaluate(async () => {
    const f = await import('/tests/browser/fixtures/' + 'gi-support.ts') as { runGiSupportFixture: typeof runGiSupportFixture };
    return f.runGiSupportFixture(true);
  });
  moments(r.ordinaryBefore); moments(r.ordinaryAfter);
  expect(r.reversed.actual.invalidProbes).toBeGreaterThan(0);
  for (const p of r.reversed.actual.probes) {
    expect([p.primaryOverflows, p.shadowOverflows, p.emissiveOverflows]).toEqual([0,0,0]);
    expect(p.valid).toBe(p.visibilityOverflows === 0);
  }
  expect(errors).toEqual([]); expect(r.errors).toEqual([]); expect(r.afterDispose).toBe(0);
});

test('support direction switches invalidate world and final-gather history with reflections enabled', async ({ page }) => {
  const errors = await setup(page);
  const r = await page.evaluate(async () => {
    const f = await import('/tests/browser/fixtures/' + 'gi-support.ts') as { runGiSupportHistoryFixture: typeof runGiSupportHistoryFixture };
    return f.runGiSupportHistoryFixture();
  });
  expect(r.stable.gi!.historyUpdates).toBeGreaterThan(1);
  expect(r.stable.gi!.finalGather!.historyUpdates).toBeGreaterThan(1);
  for (const c of [r.switched, r.restored]) {
    expect(c.gi!.historyResetReason).toBe('visibility-direction-changed');
    expect(c.gi!.historyUpdates).toBe(1); expect(c.gi!.lastHistoryFrame).toBe(0);
    expect(c.gi!.finalGather!.historyUpdates).toBe(1);
    expect(c.gi!.finalGather!.historyResetReason).toBe('scene-lighting-discontinuity');
  }
  expect(r.recovered.gi!.historyUpdates).toBeGreaterThan(1);
  for (const c of [r.stable, r.switched, r.recovered, r.restored]) {
    expect(c.probe.invalidProbes).toBe(0); expect(c.hdr.every(Number.isFinite)).toBe(true);
    expect(c.reflections!.enabled).toBe(true);
  }
  expect(errors).toEqual([]); expect(r.errors).toEqual([]); expect(r.afterDispose).toBe(0);
});
