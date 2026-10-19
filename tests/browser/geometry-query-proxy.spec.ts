import { expect, test } from '@playwright/test';
import type { runGeometryQueryProxyFixture } from './fixtures/geometry-query-proxy';

test('source-free paged surface shares a fixed proxy for shadows, queries, GI and area sampling', async ({ page }, testInfo) => {
  test.setTimeout(120000);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const r = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-query-proxy.ts') as { runGeometryQueryProxyFixture: typeof runGeometryQueryProxyFixture };
    return fixture.runGeometryQueryProxyFixture();
  });
  console.log('QUERY_PROXY_CHECKS', JSON.stringify({ sourceVertices: r.sourceVertices, proxyVertices: r.proxyVertices,
    sourceTriangles: r.sourceTriangles, proxyTriangles: r.proxyTriangles, surfaceMaxError: r.surfaceMaxError,
    sameQuery: r.sameQuery, shadowEqual: r.shadowEqual, movingQueryEqual: r.movingQueryEqual,
    giEqual: r.giEqual, gatherEqual: r.gatherEqual, gatherValid: r.gatherValid, ledger: r.ledger }));
  expect(r.noSourceMesh).toBe(true);
  expect(r.proxyTriangles).toBeLessThan(r.sourceTriangles / 2); expect(r.proxyVertices).toBeLessThan(r.sourceVertices);
  expect(r.coarseTriangles).toBeLessThan(r.sourceTriangles * 3);
  expect(r.surfaceMaxError).toBeLessThan(0.002); expect(r.foreground).toBeGreaterThan(100);
  expect(r.queryCoarse.hits).toEqual(r.queryDetailed.hits); expect(r.sameQuery).toBe(true);
  const hits = r.queryDetailed.hits.filter(hit => hit?.batch === 'sculpture');
  expect(hits).toHaveLength(3);
  for (const hit of hits) {
    expect(hit!.geometry).toMatchObject({ kind: 'fixed-query-proxy', topologyVersion: 7, errorMetric: 'accumulated-simplification' });
    expect(hit!.topologyVersion).toBe(3); expect(hit!.triangleIndex).toBeLessThan(r.proxyTriangles);
  }
  expect(r.shadowEqual).toBe(true); expect(r.ledger.proxy).toBeLessThan(r.ledger.full);
  expect(r.maskError).toContain('GEOMETRY_PROXY_MASK_REQUIRES_FULL_SOURCE'); expect(r.afterRevision).toBe(r.beforeRevision);
  expect(r.replacementError).toContain('GEOMETRY_STREAM_ASSET_CHANGED');
  expect(r.velocityEqual).toBe(true); expect(r.velocityNonzero).toBe(true); expect(r.movingQueryEqual).toBe(true);
  expect(r.giEqual).toBe(true); expect(r.giInvalid).toEqual([0, 0]);
  expect(r.gatherEqual).toBe(true); expect(r.gatherValid).toBeGreaterThan(20);
  expect(r.diagnostics.geometryStreams[0].sourceRepresentation).toBe('fixed-shadow-query-proxy');
  expect(r.diagnostics.geometryStreams[0].queryProxy).toMatchObject({ exactSourceRetained: false, cameraDependent: false });
  expect(r.visiblePixels.every(count => count > 10000)).toBe(true);
  expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]); expect(r.afterDispose).toEqual([0, 0, 0]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(r.adapter.isFallbackAdapter).toBe(false);
  const screenshot = testInfo.outputPath('geometry-query-proxy.png'); await page.locator('#query-proxy-visual').screenshot({ path: screenshot });
  await testInfo.attach('Source-free surface and fixed query proxy', { path: screenshot, contentType: 'image/png' });
  console.log('QUERY_PROXY_DIAGNOSTIC', JSON.stringify(r));
});
