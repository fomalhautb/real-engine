import { expect, test } from '@playwright/test';
import type { runGeometryRuntimeCapacityFixture } from './fixtures/geometry-runtime-capacity';

test('Renderer trades physical stream capacity between camera views without losing roots, bindings or history', async ({ page }, testInfo) => {
  test.setTimeout(120000);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const r = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-runtime-capacity.ts') as { runGeometryRuntimeCapacityFixture: typeof runGeometryRuntimeCapacityFixture };
    return fixture.runGeometryRuntimeCapacityFixture();
  });
  expect(r.endpoints.map(point => point.slots[1 - point.view])).toEqual([r.roots, r.roots, r.roots]);
  for (const endpoint of r.endpoints) {
    expect(endpoint.slots[endpoint.view]).toBeGreaterThan(r.roots);
    expect(endpoint.maxError).toBeLessThan(0.002); expect(endpoint.velocityError).toBeLessThan(0.002);
    expect(endpoint.foreground).toBeGreaterThan(10);
  }
  expect(r.transitionChecks.length).toBeGreaterThanOrEqual(5);
  for (const check of r.transitionChecks) {
    expect(check.samples).toBeGreaterThan(0); expect(check.current).toBeLessThan(0.0001); expect(check.previous).toBeLessThan(0.0001);
  }
  expect(r.peakAccounted).toBeLessThanOrEqual(r.maxPeakBytes); expect(r.minimumOwned).toBeGreaterThan(0);
  expect(r.states.some(state => state.reservedBytes > 0)).toBe(true);
  expect(r.diagnostics).toMatchObject({ enabled: true, error: null });
  expect(r.afterDispose).toMatchObject({ phase: 'disposed', liveBytes: 0, reservedBytes: 0 });
  expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]); expect(r.ownedAfterDispose).toEqual([0, 0]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(r.adapter.isFallbackAdapter).toBe(false);
  const screenshot = testInfo.outputPath('geometry-runtime-capacity.png');
  await page.locator('#geometry-runtime-capacity-visual').screenshot({ path: screenshot });
  await testInfo.attach('Camera-driven shared geometry capacity', { path: screenshot, contentType: 'image/png' });
  console.log('GEOMETRY_RUNTIME_CAPACITY', JSON.stringify(r));
});
