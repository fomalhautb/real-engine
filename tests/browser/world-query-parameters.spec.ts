import { expect, test } from '@playwright/test';
import type { runWorldQueryParametersFixture } from './fixtures/world-query-parameters';

test('linear and scene-BVH world queries retain encoder parameters, masks and slot budgets', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const variants = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'world-query-parameters.ts') as {
      runWorldQueryParametersFixture: typeof runWorldQueryParametersFixture;
    };
    const linear = await module.runWorldQueryParametersFixture(false);
    const accelerated = await module.runWorldQueryParametersFixture(true);
    return { linear, accelerated };
  });

  for (const [mode, result] of Object.entries(variants)) {
    expect(result.sceneDiagnostics).toEqual(mode === 'accelerated'
      ? { mode: 'batch-bvh', nodes: 511, batches: 8 }
      : { mode: 'linear-batches', nodes: 0, batches: 1 });
    expect(result.results.map(value => value.hits.length)).toEqual([3, 1, 2, 3]);
    for (const [index, query] of result.results.entries()) {
      expect(query.frame).toBe(17);
      expect(query.materialRevision).toBe(23);
      const red = [1, 0.25098, 0.06275, 1][index];
      for (const hit of query.hits) {
        expect(hit).not.toBeNull();
        expect(hit!.batch).toBe('parameter-triangle');
        expect(hit!.entityId).toBe(712);
        expect(hit!.entityGeneration).toBe(2);
        expect(hit!.historyVersion).toBe(5);
        expect(hit!.topologyVersion).toBe(9);
        expect(hit!.distance).toBeCloseTo(5, 4);
        expect(hit!.baseColor[0]).toBeCloseTo(red, 2);
        expect(hit!.baseColor[1]).toBeCloseTo(0, 4);
        expect(hit!.baseColor[2]).toBeCloseTo(0, 4);
      }
    }
    expect(result.maskResults.map(value => value.hits[0]?.entityId)).toEqual([712, undefined, undefined, 712]);
    expect(result.maskUploads).toEqual([0, 0, 0, 0]);
    expect(result.maskWork[2]).toBe(0);
    if (mode === 'accelerated') expect(result.maskWork[1]).toBe(1);
    expect(result.uploads[0]).toBeGreaterThan(0);
    expect(result.uploads[1]).toBeGreaterThan(0);
    expect(result.uploads[2]).toBeGreaterThan(0);
    expect(result.uploads[3]).toBe(0);
    expect(result.zeroUpload).toBe(0);
    expect(result.invalidCount).toContain('WORLD_QUERY_RAY_BUDGET');
    expect(result.invalidLod).toContain('WORLD_QUERY_TEXTURE_LOD');
    expect(result.snapshotChange).toContain('WORLD_QUERY_ENCODER_SNAPSHOT_CHANGED');
    expect(result.frameChange).toContain('WORLD_QUERY_ENCODER_SNAPSHOT_CHANGED');
    expect(result.sourceCountChange).toContain('WORLD_QUERY_ENCODER_SNAPSHOT_CHANGED');
    expect(result.reuseAfterSnapshotReject).toBe(0);
    expect(result.budgetUploads.every(value => value > 0)).toBe(true);
    expect(result.budgetError).toContain('WORLD_QUERY_ENCODER_PARAMETER_BUDGET');
    expect(result.reuseAfterReject).toBe(0);
    expect(result.preserved.map(value => value.hits[0]?.entityId)).toEqual([712, 712]);
    expect(result.preserved.map(value => value.hits[0]?.baseColor[0])).toEqual([1, 1]);
    expect(result.freshUpload).toBeGreaterThan(0);
    expect(result.strides.trace % result.strides.uniformAlignment).toBe(0);
    expect(result.strides.reset % result.strides.uniformAlignment).toBe(0);
    expect(result.cleanup).toEqual({ firstOwnedBytesAfterDispose: 0, secondOwnedBytesAfterDispose: 0 });
    expect(result.validation).toEqual([]);
  }
  // Candidate selection adds one 32-byte dispatch/layout snapshot per distinct slot.
  for (const index of [0, 1, 2]) {
    expect(variants.accelerated.uploads[index] - variants.linear.uploads[index]).toBe(32);
  }
  for (const index of [0, 1]) {
    expect(variants.accelerated.budgetUploads[index] - variants.linear.budgetUploads[index]).toBe(32);
  }
  expect(variants.accelerated.freshUpload - variants.linear.freshUpload).toBe(32);
  expect(pageErrors).toEqual([]);
});
