import { expect, test } from '@playwright/test';
import type { runOcclusionFixture } from './fixtures/occlusion';

type Result = Awaited<ReturnType<typeof runOcclusionFixture>>;

test.use({ deviceScaleFactor: 1 });

for (const mode of ['resident', 'stream'] as const) {
  test(`two-phase current-depth recovery preserves ${mode} geometry pixels and rejects stale occlusion in the first frame`,
    async ({ page }, testInfo) => {
      test.setTimeout(180_000);
      const pageErrors: string[] = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.goto('/?trees=1');
      await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
      await expect(page.locator('#error')).toBeHidden();
      const result = await page.evaluate(async selectedMode => {
        window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
        const fixture = await import('/tests/browser/fixtures/' + 'occlusion.ts') as {
          runOcclusionFixture: typeof runOcclusionFixture;
        };
        return fixture.runOcclusionFixture(selectedMode);
      }, mode) as Result;
      const png = testInfo.outputPath(`occlusion-${mode}.png`);
      await page.locator(`#occlusion-${mode}-visual`).screenshot({ path: png });
      await testInfo.attach(`occlusion-${mode}.png`, { path: png, contentType: 'image/png' });
      await testInfo.attach(`occlusion-${mode}.json`, {
        body: JSON.stringify(result, null, 2), contentType: 'application/json',
      });
      if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
      if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
        expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
      }
      expect(result.diagnostics.algorithm).toBe('two-phase-current-depth-recovery');
      expect(result.diagnostics.ownedGpuBytes).toBeGreaterThan(0);
      expect(result.settledFrames, 'history and full stream should settle').toBeLessThan(80);
      expect(result.stable.cut.occlusion.deferredClusters).toBeGreaterThan(0);
      expect(result.stable.cut.occlusion.culledClusters).toBeGreaterThan(0);
      expect(result.stable.cut.triangles).toBeLessThan(result.stable.baseline.triangles * 0.4);
      expect(result.stable.baseline.triangles).toBe(result.sourceTriangles);
      if (mode === 'stream') expect(result.pages).toBeGreaterThan(1);
      const captures = [result.stable, result.movedFirst, result.movedNext, result.turnedFirst, result.nearPlane,
        result.disabled, result.reenabled, result.resized, result.recoveredAfterBudget, result.empty,
        ...(result.maskFirst ? [result.maskFirst] : [])];
      for (const [index, capture] of captures.entries()) {
        expect(capture.error.max, `${mode} capture ${index} HDR maximum`).toBeLessThanOrEqual(0.005);
        expect(capture.error.mean, `${mode} capture ${index} HDR mean`).toBeLessThanOrEqual(0.0005);
        expect(capture.error.visible, `${mode} capture ${index} foreground count relative to reference`)
          .toBe(capture.error.referenceVisible);
        for (let point = 0; point < capture.referenceDepth.length; point++) {
          expect(Math.abs(capture.referenceDepth[point] - capture.actualDepth[point]),
            `${mode} capture ${index} depth point ${point}`).toBeLessThanOrEqual(0.0001);
        }
      }
      expect(result.stable.error.visible).toBeGreaterThan(100);
      // The rear board projects to 8×8 of the 16×16 probes after the wall
      // moves; compare to the unculled image and require a meaningful footprint.
      expect(result.movedFirst.error.referenceVisible).toBeGreaterThanOrEqual(32);
      expect(result.movedFirst.cut.occlusion.deferredClusters).toBeGreaterThan(0);
      expect(result.movedFirst.cut.occlusion.recoveredClusters).toBeGreaterThan(0);
      expect(result.movedFirst.cut.triangles).toBe(result.movedFirst.baseline.triangles);
      if (result.maskFirst) {
        expect(result.maskFirst.cut.occlusion.deferredClusters).toBeGreaterThan(0);
        expect(result.maskFirst.cut.occlusion.recoveredClusters).toBeGreaterThan(0);
        expect(result.maskFirst.cut.triangles).toBeGreaterThan(result.stable.cut.triangles);
        const center = result.maskFirst.actualHdr[8 * 16 + 8];
        expect(center[0], 'red backboard is visible through the mask hole').toBeGreaterThan(center[2]);
      }
      expect(result.turnedFirst.cut.occlusion.deferredClusters).toBeGreaterThan(0);
      expect(result.turnedFirst.cut.occlusion.recoveredClusters).toBeGreaterThan(0);
      expect(result.disabled.occlusion.enabled).toBe(false);
      expect(result.reenabled.occlusion.enabled).toBe(true);
      expect(result.resized.occlusion.grid).toBeTruthy();
      expect(result.resized.actualHdr).toHaveLength(259); // 16×16 grid plus three final partial-tile edges
      expect(result.failedResize.error).toContain('OCCLUSION_BYTE_BUDGET');
      expect(result.failedResize.before.stats.width).toBe(319);
      expect(result.failedResize.before.stats.height).toBe(223);
      expect(result.failedResize.after).toEqual(result.failedResize.before);
      expect(result.recoveredAfterBudget.actualHdr).toHaveLength(259);
      expect(result.empty.cut.triangles).toBe(0);
      expect(result.queryHidden.hits[0]?.entityId).toBe(110);
      expect(result.queryAfterResize.hits[0]?.entityId).toBe(110);
      expect(result.errors).toEqual([]);
      expect(result.ownedAfterDispose).toEqual([0, 0]);
      expect(pageErrors).toEqual([]);
    });
}
