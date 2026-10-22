import { expect, test } from '@playwright/test';
import type { runWorldQueryFacePolicyFixture } from './fixtures/world-query-face-policy';

for (const sceneBvh of [false, true]) test(`world query face policy preserves material coverage and isolates encoder slots (${sceneBvh ? 'scene BVH' : 'linear'})`, async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async accelerated => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'world-query-face-policy.ts') as {
      runWorldQueryFacePolicyFixture: typeof runWorldQueryFacePolicyFixture;
    };
    return fixture.runWorldQueryFacePolicyFixture(accelerated);
  }, sceneBvh);
  const hit = (trace: number, label: string) => result.results[trace].result.hits[result.labels.indexOf(label)];
  for (const [label, ordinary, reverse] of [
    ['single-front', 100, null], ['single-back', null, 100],
    ['reversed-winding-front', null, 101], ['reversed-winding-back', 101, null],
    ['double-front', 102, 102], ['double-back', 102, 102],
  ] as const) {
    expect(hit(0, label)?.entityId ?? null, label).toBe(ordinary);
    expect(hit(1, label)?.entityId ?? null, label).toBe(reverse);
    for (const trace of [0, 1]) if (hit(trace, label)) expect(hit(trace, label)!.distance).toBeCloseTo(2, 5);
  }
  for (const [cell, covered] of [['hole', false], ['below', false], ['equal', true], ['solid', true]] as const) {
    expect(Boolean(hit(0, `mask-${cell}-front`)), cell).toBe(covered);
    expect(hit(0, `mask-${cell}-back`)).toBeNull();
    expect(hit(1, `mask-${cell}-front`)).toBeNull();
    expect(Boolean(hit(1, `mask-${cell}-back`)), cell).toBe(covered);
    for (const trace of [0, 1]) for (const side of ['front', 'back']) {
      expect(Boolean(hit(trace, `mask-double-${cell}-${side}`)), `${cell} double ${side}`).toBe(covered);
    }
  }
  expect(hit(0, 'mask-equal-front')!.opacity).toBeCloseTo(128 / 255, 6);
  expect(hit(1, 'mask-equal-back')!.opacity).toBeCloseTo(128 / 255, 6);
  expect(hit(0, 'layers-probe')).toBeNull();
  expect(hit(1, 'layers-receiver')).toBeNull();
  const fromReceiver = hit(0, 'layers-receiver')!, fromProbe = hit(1, 'layers-probe')!;
  expect(fromReceiver.entityId).toBe(105); expect(fromReceiver.distance).toBeCloseTo(1.75, 5);
  expect(fromProbe.entityId).toBe(104); expect(fromProbe.distance).toBeCloseTo(1.25, 5);
  // Same segment blockedness does not imply identical nearest shading hits.
  expect(fromReceiver.entityId).not.toBe(fromProbe.entityId);
  for (const label of ['short-interval', 'excluded-endpoint', 'parallel']) {
    expect(hit(0, label)).toBeNull(); expect(hit(1, label)).toBeNull();
  }
  expect(result.results[2].result).toEqual(result.results[0].result);
  expect(result.results[3].result).toEqual(result.results[1].result);
  expect(result.publicDefault).toEqual(result.results[0].result);
  expect(result.uploads.slice(0, 2).every(bytes => bytes > 0)).toBe(true);
  expect(result.uploads.slice(2)).toEqual([0, 0]);
  expect(result.invalidPolicy).toContain('WORLD_QUERY_FACE_POLICY');
  expect(result.slotBudget).toContain('WORLD_QUERY_ENCODER_PARAMETER_BUDGET');
  expect(result.reuseAfterRejection).toBe(0);
  expect(result.diagnostics.traceParameterBytes).toBe(112);
  expect(result.diagnostics.sceneAcceleration).toBe(sceneBvh ? 'batch-bvh' : 'linear-batches');
  expect(result.cleanup.ownedBytesAfterDispose).toBe(0);
  expect(result.errors).toEqual([]); expect(pageErrors).toEqual([]);
});
