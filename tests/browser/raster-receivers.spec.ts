import { expect, test } from '@playwright/test';
import type { runRasterReceiverFixture } from './fixtures/raster-receivers';

test.use({ deviceScaleFactor: 1 });
test('GI receivers follow actual LOD depth, material coverage and stable identity', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'raster-receivers.ts') as { runRasterReceiverFixture: typeof runRasterReceiverFixture };
    return fixture.runRasterReceiverFixture();
  });
  await testInfo.attach('raster-receivers.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  expect(pageErrors).toEqual([]); expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0);
  expect(result.coarse.cut.triangles).toBeLessThan(result.fine.cut.triangles);
  expect(result.diagnostics.surfaceReceivers?.ownedGpuBytes).toBe(128 * 128 * 32 + 32);
  expect(result.diagnostics.passes.indexOf('surface-receivers')).toBeLessThan(result.diagnostics.passes.indexOf('final-gather'));
  for (const capture of [result.fine, result.coarse, result.stable, result.moved]) {
    expect(capture.anchors.every(anchor => anchor.valid)).toBe(true);
    expect(Math.max(...capture.depthError)).toBeLessThan(0.000002);
    const selected = new Set(capture.cut.selectedClusters.map(cluster => cluster.clusterId + 1));
    for (const anchor of capture.anchors) {
      expect(anchor.identity[0]).toBe(1); expect(anchor.identity[1]).toBe(0xffff_ff01);
      expect(selected.has(anchor.identity[3])).toBe(true);
      expect(anchor.versions[0]).toBe(capture.frame); expect(anchor.versions[2]).toBe(5);
      expect(anchor.normal[0]).toBeGreaterThan(0.15);
    }
  }
  expect(result.masked.anchors.every(anchor => !anchor.valid)).toBe(true);
  const changed = result.coarse.anchors.filter((anchor, i) => anchor.identity[3] !== result.fine.anchors[i].identity[3]);
  expect(changed.length).toBeGreaterThan(0);
  expect(changed.every(anchor => anchor.historyUpdates === 1)).toBe(true);
  for (const anchor of result.moved.anchors) {
    expect(anchor.identity[2]).toBe(8); expect(anchor.versions[1]).toBe(4); expect(anchor.historyUpdates).toBe(1);
  }
});
