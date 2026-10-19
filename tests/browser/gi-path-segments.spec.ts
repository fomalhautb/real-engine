import { expect, test } from '@playwright/test';
import type { runGiPathSegmentsFixture, runGiLaterEmissionFixture,
  runGiRoomCacheFixture } from './fixtures/gi-path-segments';

type Result = Awaited<ReturnType<typeof runGiPathSegmentsFixture>>;
type LaterResult = Awaited<ReturnType<typeof runGiLaterEmissionFixture>>;
type RoomResult = Awaited<ReturnType<typeof runGiRoomCacheFixture>>;

for (const segments of [2, 3, 4] as const) {
  test(`final gather ${segments}-segment paths terminate on miss and Unlit while preserving area emission`,
    async ({ page }, testInfo) => {
      test.setTimeout(180_000);
      const pageErrors: string[] = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.goto('/?trees=1');
      await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
      await expect(page.locator('#error')).toBeHidden();
      const result = await page.evaluate(async count => {
        window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
        const fixture = await import('/tests/browser/fixtures/' + 'gi-path-segments.ts') as {
          runGiPathSegmentsFixture: typeof runGiPathSegmentsFixture;
        };
        return fixture.runGiPathSegmentsFixture(count);
      }, segments) as Result;
      await testInfo.attach(`gi-path-segments-${segments}.json`, {
        body: JSON.stringify(result, null, 2), contentType: 'application/json',
      });
      if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
      if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
        expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
      }
      const snapshots = [result.missSingle, result.missMulti, result.unlitSingle,
        result.unlitMulti, result.doubled, result.morphedAway];
      for (const snapshot of snapshots) {
        expect(snapshot.anchor.valid).toBe(true);
        expect(snapshot.anchor.historyUpdates).toBeGreaterThan(0);
        expect(snapshot.diagnostics.queryRaysPerUpdate).toBeLessThanOrEqual(1024 * snapshot.diagnostics.queryPassesPerUpdate);
      }
      for (const snapshot of [result.missSingle, result.unlitSingle]) {
        expect(snapshot.diagnostics.pathSegmentsPerUpdate).toBe(1);
        expect(snapshot.diagnostics.queryPassesPerUpdate).toBe(snapshot.diagnostics.areaRaysPerUpdate ? 4 : 3);
      }
      for (const snapshot of [result.missMulti, result.unlitMulti, result.doubled, result.morphedAway]) {
        expect(snapshot.diagnostics.pathSegmentsPerUpdate).toBe(segments);
        expect(snapshot.diagnostics.queryPassesPerUpdate).toBe(1 + 2 * segments
          + Number(snapshot.diagnostics.areaRaysPerUpdate > 0));
      }
      for (let c = 0; c < 3; c++) {
        const skyIrradiance = Math.PI * result.sky[c];
        const skyHdr = result.base[c] * 0.96 * result.sky[c];
        for (const snapshot of [result.missSingle, result.missMulti, result.morphedAway]) {
          expect(Math.abs(snapshot.anchor.filteredIrradiance[c] - skyIrradiance),
            `segment ${segments} terminal miss ${c}`).toBeLessThanOrEqual(0.003);
          expect(Math.abs(snapshot.hdr[c] - skyHdr),
            `segment ${segments} terminal HDR ${c}`).toBeLessThanOrEqual(0.003);
        }
        for (const snapshot of [result.unlitSingle, result.unlitMulti]) {
          expect(Math.abs(snapshot.anchor.filteredIrradiance[c] - result.reference[c]),
            `segment ${segments} area light ${c}`).toBeLessThanOrEqual(0.035);
          expect(Math.abs(snapshot.hdr[c] - result.base[c] * 0.96 / Math.PI * result.reference[c]),
            `segment ${segments} area HDR ${c}`).toBeLessThanOrEqual(0.012);
        }
        expect(Math.abs(result.unlitMulti.anchor.filteredIrradiance[c]
          - result.unlitSingle.anchor.filteredIrradiance[c]),
        `segment ${segments} Unlit must not continue ${c}`).toBeLessThanOrEqual(0.035);
        expect(Math.abs(result.doubled.anchor.filteredIrradiance[c] - result.doubledReference[c]),
          `segment ${segments} material-updated emitter ${c}`).toBeLessThanOrEqual(0.055);
      }
      expect(result.unlitMulti.diagnostics.activeEmitters).toBe(1);
      expect(result.missMulti.diagnostics.activeEmitters).toBe(0);
      expect(result.morphedAway.diagnostics.activeEmitters).toBe(1);
      expect(result.errors).toEqual([]);
      expect(result.afterDispose).toBe(0);
      expect(pageErrors).toEqual([]);
    });
}

for (const segments of [2, 3, 4] as const) {
  test(`final gather ${segments}-segment terminal cache converges in a uniform closed room`,
    async ({ page }, testInfo) => {
      test.setTimeout(180_000);
      const pageErrors: string[] = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.goto('/?trees=1');
      await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
      await expect(page.locator('#error')).toBeHidden();
      const result = await page.evaluate(async count => {
        window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
        const fixture = await import('/tests/browser/fixtures/' + 'gi-path-segments.ts') as {
          runGiRoomCacheFixture: typeof runGiRoomCacheFixture;
        };
        return fixture.runGiRoomCacheFixture(count);
      }, segments) as RoomResult;
      await testInfo.attach(`gi-path-room-cache-${segments}.json`, {
        body: JSON.stringify(result, null, 2), contentType: 'application/json',
      });
      if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
      if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
        expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
      }
      expect(result.anchor.valid).toBe(true);
      expect(result.anchor.historyUpdates).toBeGreaterThanOrEqual(64);
      expect(result.diagnostics.pathSegmentsPerUpdate).toBe(segments);
      expect(result.diagnostics.queryPassesPerUpdate).toBe(1 + 2 * segments);
      expect(result.diagnostics.areaRaysPerUpdate).toBe(0);
      expect(result.diagnostics.activeEmitters).toBe(0);
      expect(result.diagnostics.queryRaysPerUpdate).toBeLessThanOrEqual(
        1024 * result.diagnostics.queryPassesPerUpdate);
      // Every bounce sees the same Le and rho. The expected infinite series
      // is Le/(1-rho), hence E=pi*Le/(1-rho). Check unfiltered data so spatial
      // filtering cannot conceal a missing terminal cache at segment four.
      for (let c = 0; c < 3; c++) {
        expect(Number.isFinite(result.anchor.irradiance[c])).toBe(true);
        expect(Math.abs(result.anchor.irradiance[c] - result.expectedIrradiance[c]),
          `segment ${segments} raw room irradiance channel ${c}`)
          .toBeLessThanOrEqual(result.expectedIrradiance[c] * 0.03);
      }
      expect(result.errors).toEqual([]);
      expect(result.afterDispose).toBe(0);
      expect(pageErrors).toEqual([]);
    });
}

for (const segments of [2, 3, 4] as const) {
  test(`final gather ${segments}-segment paths count a second-hit Unlit emitter once`, async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    const pageErrors: string[] = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.goto('/?trees=1');
    await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    const result = await page.evaluate(async count => {
      window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
      const fixture = await import('/tests/browser/fixtures/' + 'gi-path-segments.ts') as {
        runGiLaterEmissionFixture: typeof runGiLaterEmissionFixture;
      };
      return fixture.runGiLaterEmissionFixture(count);
    }, segments) as LaterResult;
    await testInfo.attach(`gi-path-later-emission-${segments}.json`, {
      body: JSON.stringify(result, null, 2), contentType: 'application/json',
    });
    if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
    if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
      expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
    }
    expect(result.anchor.valid).toBe(true);
    expect(result.fraction).toBeGreaterThan(0.1);
    expect(result.diagnostics.pathSegmentsPerUpdate).toBe(segments);
    expect(result.diagnostics.queryPassesPerUpdate).toBe(2 + 2 * segments);
    expect(result.diagnostics.areaRaysPerUpdate).toBe(result.diagnostics.anchorsPerUpdate);
    expect(result.diagnostics.activeEmitters).toBe(1);
    expect(result.finiteSecondSegmentLossBound).toBeLessThan(0.01);
    for (let c = 0; c < 3; c++) {
      expect(result.anchor.filteredIrradiance[c], `segment ${segments} later Unlit channel ${c}`).toBeGreaterThan(0);
      expect(Math.abs(result.anchor.filteredIrradiance[c] - result.expected[c]),
        `segment ${segments} later Unlit channel ${c}, expected ${result.expected[c]}`)
        .toBeLessThanOrEqual(Math.max(0.012, result.expected[c] * 0.15));
    }
    expect(result.errors).toEqual([]);
    expect(result.afterDispose).toBe(0);
    expect(pageErrors).toEqual([]);
  });
}
