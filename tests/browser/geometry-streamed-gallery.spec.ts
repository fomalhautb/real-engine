import { writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import type { runStreamedGalleryFixture, StreamedGalleryFixtureOptions } from './fixtures/geometry-streamed-gallery';

const hardware = process.env.PLAYWRIGHT_GPU === 'hardware';

async function run(page: import('@playwright/test').Page, options: StreamedGalleryFixtureOptions) {
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  return page.evaluate(async fixtureOptions => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-streamed-gallery.ts') as { runStreamedGalleryFixture: typeof runStreamedGalleryFixture };
    return fixture.runStreamedGalleryFixture(fixtureOptions);
  }, options);
}

test('oversubscribed unique gallery streams without holes, mis-culling or incomplete cuts across teleports and turns', async ({ page }, testInfo) => {
  test.setTimeout(hardware ? 300_000 : 600_000);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  const r = await run(page, hardware ? { scene: { objects: 24, around: 192, across: 48 }, width: 640, height: 400 } : {});
  const all = [...r.phases.map(phase => phase.settled), ...r.firstFrames, ...r.transitionChecks, ...r.walkChecks];
  for (const check of all) if (check.evidence) {
    for (const [name, url] of Object.entries(check.evidence)) {
      const path = testInfo.outputPath(`${check.label.replace(/[^a-z0-9.-]+/gi, '_')}-${name}.png`);
      writeFileSync(path, Buffer.from(url.split(',')[1], 'base64'));
      await testInfo.attach(`${check.label}-${name}.png`, { path, contentType: 'image/png' });
    }
    check.evidence = null;
  }
  writeFileSync(testInfo.outputPath('streamed-gallery.json'), JSON.stringify(r, null, 2));
  await testInfo.attach('streamed-gallery.json', { path: testInfo.outputPath('streamed-gallery.json'), contentType: 'application/json' });
  console.log('GEOMETRY_STREAMED_GALLERY', JSON.stringify({ adapter: r.adapter, accounting: r.accounting,
    phases: r.phases.map(({ settled, ...phase }) => ({ ...phase, settled: { ...settled, eye: undefined } })),
    firstFrames: r.firstFrames, walk: r.walkChecks.map(check => [check.label, check.fallbackClusters, check.pixels.testedPixels, check.pixels.holePixels, check.pixels.minRadius, check.pixels.maxRadius]),
    holes: all.filter(check => check.pixels.holePixels).map(check => ({ label: check.label, eye: check.eye, pixels: check.pixels,
      radii: check.radii, diagnosis: check.holeDiagnosis })) }));
  // G03 sizing: unique resident-format data oversubscribes the geometry pool at least 4x.
  expect(r.accounting.uniqueToPool).toBeGreaterThanOrEqual(4);
  expect(r.accounting.fullResidencyToResidentBudget).toBeGreaterThan(1);
  expect(all.length).toBeGreaterThan(20);
  for (const check of all) {
    // Exact structural completeness of every probed GPU cut, settled or not.
    expect(check.coverage, check.label).toEqual({ uncovered: 0, overlapping: 0, partialGroups: 0 });
    // Crack-free drawn geometry: welded edges all have their reverse (non-manifold pinches are reported, not holes).
    expect(check.edges.boundaryEdges, check.label).toBe(0);
    expect(check.unclassifiedPixels, check.label).toBe(0);
    // No reference-interior sculpture pixel shows anything else beyond the error
    // band, unless the full-detail surface seen there lies within the drawn cut's
    // declared error (a gap at an internal self-occlusion silhouette).
    expect(check.pixels.missingObjects, check.label).toEqual([]);
    expect(check.pixels.unexplainedHolePixels, check.label).toBe(0);
  }
  for (const phase of r.phases) {
    expect(phase.settled.cpuCutMismatches, phase.label).toBe(0);
  }
  // Budget-limited settled views keep fallback on near sculptures, widening their
  // bands; across all stops the pixel test must still cover real interiors.
  expect(r.phases.reduce((sum, phase) => sum + phase.settled.pixels.testedPixels, 0)).toBeGreaterThan(2000);
  expect(r.phases.some(phase => phase.settled.nominalRadiusObjects > 0)).toBe(true);
  // Oversubscription must actually exercise eviction and visible ancestor fallback.
  expect(r.phases.reduce((sum, phase) => sum + phase.droppedPages, 0)).toBeGreaterThan(0);
  expect(r.phases.reduce((sum, phase) => sum + phase.reloadedPages, 0)).toBeGreaterThan(0);
  expect(Math.max(...[...r.firstFrames, ...r.transitionChecks, ...r.walkChecks].map(check => check.fallbackClusters))).toBeGreaterThan(0);
  expect(r.firstFrames.length).toBe(4);
  expect(r.budget).toMatchObject({ enabled: true, error: null });
  expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]);
  if (hardware) expect(r.adapter.isFallbackAdapter).toBe(false);
});
