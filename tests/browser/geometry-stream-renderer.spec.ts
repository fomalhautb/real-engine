import { expect, test } from '@playwright/test';
import type { runGeometryStreamRendererFixture } from './fixtures/geometry-stream-renderer';

test('main renderer consumes streamed pages with shared material, motion, shadow, query and GI semantics', async ({ page }, testInfo) => {
  test.setTimeout(120000);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  const workers: string[] = []; page.on('worker', worker => workers.push(worker.url()));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const r = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-stream-renderer.ts') as { runGeometryStreamRendererFixture: typeof runGeometryStreamRendererFixture };
    return fixture.runGeometryStreamRendererFixture();
  });
  expect(r.coarse.reduce((sum, cut) => sum + cut.triangles, 0)).toBeLessThan(r.sourceTriangles * 3);
  expect(r.detailed.reduce((sum, cut) => sum + cut.triangles, 0)).toBe(r.sourceTriangles * 3);
  expect(r.rootsOnly).toEqual([r.rootPages, r.rootPages]);
  expect(r.requestedPages.length).toBeGreaterThan(r.rootPages.length);
  for (const value of [r.opaque, r.normalUpdate, r.moving, r.mask, r.gi]) {
    expect(value.meanError).toBeLessThan(0.0001); expect(value.maxError).toBeLessThan(0.002); expect(value.foreground).toBeGreaterThan(50);
  }
  expect(r.variantsAfterParameters).toBe(r.variants);
  expect(r.queryDetailed.hits).toEqual(r.queryReference.hits); expect(r.queryCoarse.hits).toEqual(r.queryDetailed.hits);
  expect(r.queryDetailed.hits.some(hit => hit?.entityId === 2)).toBe(true);
  expect(r.depth.streamed).toEqual(r.depth.reference);
  expect(r.velocity.streamed).toEqual(r.velocity.reference);
  expect(r.velocity.streamed.some(pixel => Math.abs(pixel[0]) + Math.abs(pixel[1]) > 0.0001)).toBe(true);
  expect(r.maskCuts.every(cut => cut.indices === 0)).toBe(true);
  expect(r.maskQuery.streamed.hits).toEqual(r.maskQuery.reference.hits);
  expect(r.diagnostics.geometryStreams).toHaveLength(2);
  expect(r.diagnostics.geometryStreams.every(stream => stream.cache.decoding === 'worker')).toBe(true);
  expect(workers.filter(url => url.includes('geometry-page.worker'))).toHaveLength(2);
  expect(r.diagnostics.geometryDecoding).toMatchObject({ workers: 2, failed: 0, waitingRequests: 0, activeRequests: 0 });
  expect(r.diagnostics.geometryDecoding!.completed).toBe(r.pages * 2);
  expect(r.diagnostics.geometryDecoding!.peakInFlightBytes).toBeLessThanOrEqual(r.diagnostics.geometryDecoding!.limits.maxInFlightBytes);
  expect(r.decoderAfterDispose).toMatchObject({ disposed: true, workers: 0, activeRequests: 0, waitingRequests: 0, inFlightBytes: 0 });
  expect(r.visiblePixels.every(count => count > 10000)).toBe(true);
  expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]); expect(r.ownedAfterDispose).toEqual([0, 0]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(r.adapter.isFallbackAdapter).toBe(false);
  const screenshot = testInfo.outputPath('streamed-main-renderer.png'); await page.locator('#stream-renderer-visual').screenshot({ path: screenshot });
  await testInfo.attach('Streamed main renderer and resident reference', { path: screenshot, contentType: 'image/png' });
  console.log('STREAM_RENDERER_DIAGNOSTIC', JSON.stringify({ ...r, velocity: { samples: r.velocity.streamed.length },
    decoder: r.diagnostics.geometryDecoding, diagnostics: r.diagnostics.geometryStreams }));
});
