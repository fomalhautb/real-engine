import { expect, test } from '@playwright/test';
import type { runFinalGatherHistoryFixture, runFinalGatherDisocclusionFixture } from './fixtures/gi-final-gather-history';

async function ready(page: import('@playwright/test').Page) {
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
}

test('raw final gather anchors carry camera motion, then reset at explicit, light and identity cuts', async ({ page }) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await ready(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-final-gather-history.ts') as {
      runFinalGatherHistoryFixture: typeof runFinalGatherHistoryFixture;
    };
    return module.runFinalGatherHistoryFixture();
  });
  console.log('FINAL_GATHER_HISTORY ' + JSON.stringify({
    counts: [result.staticAnchor.historyUpdates, result.movedAnchor.historyUpdates,
      result.cutAnchor.historyUpdates, result.resumedAnchor.historyUpdates,
      result.lightChangedAnchor.historyUpdates, result.identityChangedAnchor.historyUpdates],
    errors: result.errors, afterDispose: result.afterDispose }));
  expect(pageErrors).toEqual([]); expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  for (const anchor of [result.staticAnchor, result.movedAnchor, result.cutAnchor,
    result.resumedAnchor, result.lightChangedAnchor, result.identityChangedAnchor]) {
    expect(anchor.valid).toBe(true);
    expect(Math.abs(anchor.position[2])).toBeLessThan(0.01);
  }
  expect(result.staticAnchor.historyUpdates).toBe(32);
  expect(result.movedAnchor.historyUpdates).toBeGreaterThan(1);
  expect(result.movedAnchor.historyUpdates).toBeLessThanOrEqual(8);
  expect(result.cutAnchor.historyUpdates).toBe(1);
  expect(result.resumedAnchor.historyUpdates).toBe(2);
  expect(result.explicitGiResetAnchor.historyUpdates).toBe(1);
  expect(result.lightChangedAnchor.historyUpdates).toBe(1);
  expect(result.identityChangedAnchor.historyUpdates).toBe(1);
  expect(result.probeErrors.beforeFrame).toContain('FINAL_GATHER_NO_FRAME');
  expect(result.probeErrors.duplicates).toContain('FINAL_GATHER_PROBE_INDICES');
  expect(result.probeErrors.outsideGrid).toContain('FINAL_GATHER_PROBE_INDICES');
  expect(result.probeErrors.overBudget).toContain('FINAL_GATHER_PROBE_INDEX_BUDGET');
  expect(result.probeErrors.concurrent).toContain('FINAL_GATHER_READBACK_PENDING');
  expect(result.probeErrors.disabled).toContain('FINAL_GATHER_DISABLED');
  expect(result.snapshot.capturedFrame).toBe(result.snapshot.requestedFrame);
  expect(result.snapshot.anchorFrame).toBe(result.snapshot.requestedFrame);
  for (let channel = 0; channel < 3; channel++) {
    const expected = 0.96 * result.base[channel] * result.sky[channel];
    for (const hdr of [result.staticHdr, result.movedHdr]) {
      expect(Math.abs(hdr[channel] - expected)).toBeLessThan(0.001);
    }
  }
});

test('a newly exposed receiver anchor rejects foreground history while another carries', async ({ page }) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await ready(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-final-gather-history.ts') as {
      runFinalGatherDisocclusionFixture: typeof runFinalGatherDisocclusionFixture;
    };
    return module.runFinalGatherDisocclusionFixture();
  });
  console.log('FINAL_GATHER_DISOCCLUSION ' + JSON.stringify(result));
  expect(pageErrors).toEqual([]); expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  expect(result.newlyExposed.length).toBeGreaterThan(0);
  for (const anchor of result.newlyExposed) {
    expect(anchor.previousHistory).toBeGreaterThan(1);
    expect(anchor.currentHistory).toBe(1);
  }
  expect(result.carriedReceiver?.historyUpdates).toBeGreaterThan(1);
});
