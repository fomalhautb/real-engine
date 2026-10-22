import { expect, test } from '@playwright/test';
import type { runGeometryPoolMigrationFixture } from './fixtures/geometry-pool-migration';

test('GPU geometry capacity migration preserves draws, supports later refinement, and shrinks to complete roots', async ({ page }, testInfo) => {
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-pool-migration.ts') as { runGeometryPoolMigrationFixture: typeof runGeometryPoolMigrationFixture };
    return fixture.runGeometryPoolMigrationFixture();
  });
  expect(result.unchangedBytes).toBe(0); expect(result.stages[0]).toEqual(result.stages[1]);
  expect(result.sourceStillResident).toEqual(result.initiallyRetained);
  expect([...result.copiedPages].sort((a, b) => a - b)).toEqual(result.initiallyRetained);
  expect(result.copiedBytes).toBeGreaterThan(0);
  expect(result.stages[2].triangles).toBeGreaterThan(result.stages[0].triangles);
  expect(result.stages[3].triangles).toBeLessThan(result.stages[2].triangles);
  expect(result.stages.every(stage => stage.foreground > 5000)).toBe(true);
  expect(result.shrinkRetained).toEqual(result.roots);
  expect(result.poolPeak).toBeLessThanOrEqual(result.initialBudget + result.fullBudget);
  expect(result.poolOwnedAfterDispose).toBe(0); expect(result.selectionOwnedAfterDispose).toBe(0);
  expect(result.errors).toEqual([]); expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  const path = testInfo.outputPath('geometry-pool-migration.png');
  await page.locator('#geometry-pool-migration-visual').screenshot({ path });
  await testInfo.attach('GPU geometry pool growth and shrink', { path, contentType: 'image/png' });
  console.log('GEOMETRY_POOL_MIGRATION_DIAGNOSTIC', JSON.stringify(result));
});
