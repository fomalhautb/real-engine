import { writeFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import type { runGeometrySilentDropFixture, runGeometrySilentDropRejections } from './fixtures/geometry-silent-drop';

// Silent-drop audit. Masked, skinned and morph batches share one Renderer with a
// source-free paged stream (fixed query/shadow proxy), a full-source paged
// stream, resident GPU LOD, Masked LOD and Masked stream batches, an
// oversubscribed cross-stream capacity budget and two-phase occlusion. Every
// batch must stay in the main view (identity-unlit colour over dense 0.1 m
// grids, cutout holes open, deformed not rest positions), in the sun shadow map
// (caster depth, and the shadowed ground against a ground-only render), and in
// world queries (identity, deformed distance, cutout pass-through), at rest and
// under load: camera paths that occlude every panel and then disocclude them,
// Masked/Opaque material switches, occlusion toggles and animated poses, with a
// few probes per frame. Combinations the renderer cannot draw must be refused
// with an explicit error. Readbacks are test-only diagnostics.

type Fixtures = { runGeometrySilentDropFixture: typeof runGeometrySilentDropFixture; runGeometrySilentDropRejections: typeof runGeometrySilentDropRejections };
type Result = Awaited<ReturnType<typeof runGeometrySilentDropFixture>>;

const hardware = process.env.PLAYWRIGHT_GPU === 'hardware';
const vendor = process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR;
const PANELS = ['skinned', 'stream-proxy', 'stream-source', 'lod', 'masked', 'masked-lod', 'masked-stream', 'morph'] as const;
/** Covered and empty samples per batch in one static audit at camera A. */
const STATIC_SAMPLES: Record<(typeof PANELS)[number], [number, number]> = {
  skinned: [7, 1], 'stream-proxy': [9, 0], 'stream-source': [9, 0], lod: [9, 0],
  masked: [8, 3], 'masked-lod': [8, 3], 'masked-stream': [8, 3], morph: [4, 1],
};
/** Dense 0.1 m coverage grid sizes per batch: surface covered/empty, ground shadowed/lit. */
const COVERAGE = {
  opaque: [285, 0, 221, 0], masked: [186, 63, 78, 35], skinned: [435, 40, 351, 40], morph: [285, 57, 221, 57],
} as const;
function expectedCoverage(masked: readonly string[], withShadow: boolean) {
  return Object.fromEntries(PANELS.map(name => {
    const [covered, empty, shadowed, lit] = COVERAGE[name === 'skinned' || name === 'morph' ? name : masked.includes(name) ? 'masked' : 'opaque'];
    return [name, { 'surface-covered': covered, ...(empty ? { 'surface-empty': empty } : {}),
      ...(withShadow ? { 'ground-shadowed': shadowed, ...(lit ? { 'ground-lit': lit } : {}) } : {}) }];
  }));
}
const MASKED = ['masked', 'masked-lod', 'masked-stream'];
const OPAQUE_PHASE = (label: string) => label.startsWith('opaque') || label === 'masked LOD/stream switched to Opaque';

async function openEngine(page: Page, pageErrors: string[]) {
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
}

function expectAdapter(adapter: { vendor: string; isFallbackAdapter?: boolean | null }) {
  if (hardware) expect(adapter.isFallbackAdapter).toBe(false);
  if (vendor) expect(adapter.vendor.toLowerCase()).toBe(vendor.toLowerCase());
}

for (const staticTraversal of ['per-batch', 'shared'] as const) {
  test(`Masked, skinned and morph batches stay drawn, shadowed and queryable beside streams, GPU LOD, budget and occlusion (${staticTraversal} queries)`, async ({ page }, testInfo) => {
    test.setTimeout(hardware ? 180_000 : 420_000);
    const pageErrors: string[] = [];
    await openEngine(page, pageErrors);
    const r: Result = await page.evaluate(async traversal => {
      window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
      const fixture = await import('/tests/browser/fixtures/' + 'geometry-silent-drop.ts') as Fixtures;
      return fixture.runGeometrySilentDropFixture({ staticTraversal: traversal });
    }, staticTraversal);
    const record = testInfo.outputPath(`silent-drop-${staticTraversal}.json`);
    writeFileSync(record, JSON.stringify(r, null, 1));
    await testInfo.attach(`Silent-drop record (${staticTraversal})`, { path: record, contentType: 'application/json' });
    const screenshot = testInfo.outputPath(`silent-drop-${staticTraversal}.png`);
    await page.locator('#silent-drop-visual').screenshot({ path: screenshot });
    await testInfo.attach(`Silent-drop frames (${staticTraversal})`, { path: screenshot, contentType: 'image/png' });
    console.log('SILENT_DROP', JSON.stringify({ adapter: r.adapter, staticTraversal: r.staticTraversal, oversubscription: r.oversubscription,
      settledFrames: r.settledFrames, skippedFrames: r.skippedFrames, failureCount: r.failureCount, failures: r.failures.slice(0, 12),
      lodActive: r.lodActive, representations: r.representations, deformation: r.deformation, budget: r.budget, coverage: r.coverage,
      occlusion: r.occlusion, worldQueries: r.worldQueries, staticVirtual: r.staticVirtual,
      frames: r.frames.map(frame => ({ label: frame.label, checked: frame.checked, occluded: frame.occludedChecks, skipped: frame.skipped,
        queries: frame.queries, shadow: frame.shadowChecks, drawCalls: frame.drawCalls, live: frame.budget.liveBytes, slots: frame.budget.slots,
        members: frame.sharedMembers, coverage: frame.coverage ? Object.values(frame.coverage).reduce((sum, counts) => sum + Object.values(counts).reduce((a, b) => a + b, 0), 0) : undefined,
        virtual: frame.virtual?.map(cut => [cut.name, cut.indices, cut.fallbackClusters, cut.occlusion.culledClusters, cut.occlusion.recoveredClusters].join(':')) })) }));

    // Nothing failed, and nothing leaked.
    expect(r.failures).toEqual([]); expect(r.failureCount).toBe(0);
    expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]); expect(r.ownedAfterDispose).toEqual([0, 0]);
    expect(r.decoderAfterDispose).toMatchObject({ disposed: true, workers: 0, activeRequests: 0, waitingRequests: 0, inFlightBytes: 0 });
    expectAdapter(r.adapter);

    // Every batch was checked in every channel, before and after the load.
    for (const audit of [r.static, r.final]) {
      for (const name of PANELS) {
        const [covered, empty] = STATIC_SAMPLES[name];
        expect(audit.counts[name], `${name} static checks`).toEqual({ 'main-covered': covered, 'shadow-caster': covered, 'ground-shadowed': covered,
          'query-hit': covered, ...(empty ? { 'main-empty': empty, 'shadow-gap': empty, 'ground-lit': empty, 'query-pass': empty } : {}) });
        expect(audit.identityError[name]).toBeLessThanOrEqual(0.02);
      }
    }
    // Dense coverage: whole surfaces and ground shadows at rest, whole surfaces on key load frames.
    expect(r.coverage.static).toEqual(expectedCoverage(MASKED, true));
    expect(r.coverage.final).toEqual(expectedCoverage(MASKED, true));
    const coverageFrames = r.frames.filter(entry => entry.coverage);
    expect(coverageFrames.map(entry => entry.label)).toEqual(['teleport to A', 'masked LOD/stream switched to Opaque', 'opaque teleport to A',
      'switched back to Masked', 'occlusion disabled', 'occlusion re-enabled']);
    for (const entry of coverageFrames) {
      expect(entry.coverage, entry.label).toEqual(expectedCoverage(OPAQUE_PHASE(entry.label) ? ['masked'] : MASKED, false));
    }

    // The shared budget really is oversubscribed and stays within its limits.
    const o = r.oversubscription;
    expect(o.pages).toBeGreaterThan(o.roots * 2);
    expect(o.opaqueDemandBytes).toBeGreaterThan(o.maxResidentBytes);
    const budget = r.budget as { enabled: boolean; error: string | null; liveBytes: number; peakBytes: number; migrations: number };
    expect(budget).toMatchObject({ enabled: true, error: null });
    expect(budget.migrations).toBeGreaterThan(0);
    expect(budget.peakBytes).toBeLessThanOrEqual(o.maxPeakBytes);
    for (const frame of r.frames) expect(frame.budget.liveBytes, frame.label).toBeLessThanOrEqual(o.maxResidentBytes);
    expect(r.requestedPages).toBeGreaterThan(o.roots);

    // Representation rules: Masked batches keep the full source, the source-free
    // stream shadows and queries its fixed proxy, deformation stays full detail.
    expect(r.lodActive).toEqual({ lod: true, 'masked-lod': false });
    expect(r.representations).toEqual({ 'stream-proxy': 'fixed-shadow-query-proxy', 'stream-source': 'full-detail-shadow-query-and-masked',
      'masked-stream': 'full-detail-shadow-query-and-masked' });
    expect(r.worldQueries.staticTraversal).toBe(staticTraversal);
    expect(Object.fromEntries(r.worldQueries.geometryRepresentations.map(entry => [entry.batch, entry.geometry.kind]))).toEqual({
      ground: 'source', occluder: 'source', ...Object.fromEntries(PANELS.map(name => [name, name === 'stream-proxy' ? 'fixed-query-proxy' : 'source'])) });
    if (staticTraversal === 'shared') {
      // Rigid committed batches are pooled; deformed batches stay per batch, and
      // Masked batches leave the pooled membership while they are Masked.
      expect(r.worldQueries.sharedStatic).toMatchObject({ pooledBatches: 8, excludedBatches: 2, members: 5 });
      expect(r.worldQueries.sharedStatic!.excluded.deformation).toBe(2);
      for (const entry of r.frames) expect(entry.sharedMembers, entry.label).toBe(OPAQUE_PHASE(entry.label) ? 7 : 5);
    } else {
      expect(r.worldQueries.sharedStatic).toBeNull();
      for (const entry of r.frames) expect(entry.sharedMembers, entry.label).toBeNull();
    }
    const cut = <T extends { name: string }>(cuts: readonly T[] | undefined, name: string): T => cuts!.find(entry => entry.name === name)!;
    for (const name of ['stream-proxy', 'stream-source', 'lod']) expect(cut(r.staticVirtual, name).indices).toBeGreaterThan(0);
    // Masked LOD and Masked stream batches select nothing: they draw their full source.
    for (const name of ['masked-lod', 'masked-stream']) for (const cuts of [r.staticVirtual, r.finalVirtual]) {
      expect(cut(cuts, name)).toMatchObject({ indices: 0, occlusion: { deferredClusters: 0, culledClusters: 0, recoveredClusters: 0 } });
    }

    // Deformed output matches the independent CPU oracle, away from the rest pose.
    expect(r.deformation.maxError).toBeLessThan(1e-4); expect(r.deformation.boundsViolations).toBe(0);
    for (const sample of r.deformation.samples) { expect(sample.error).toBeLessThan(1e-4); expect(sample.restDistance).toBeGreaterThan(0.5); }

    // Under load: every frame at camera A probed every batch, and draw-call accounting matched.
    const frame = (label: string) => r.frames.find(entry => entry.label === label)!;
    for (const entry of r.frames) {
      expect(entry.drawCalls, entry.label).toBe(entry.expectedDrawCalls);
      expect(entry.queries, entry.label).toBeGreaterThanOrEqual(18); expect(entry.shadowChecks, entry.label).toBe(entry.queries);
      if (entry.eye[0] === 0) for (const name of PANELS) expect(entry.checked[name] ?? 0, `${entry.label} ${name}`).toBeGreaterThan(0);
    }
    // At camera B the occluder hides every panel: two-phase occlusion culls every
    // cluster drawn through LOD or a stream, while the same frames' shadow-map and
    // world-query checks above still find each caster. The first frame back at A
    // recovers them in phase two and draws every batch.
    const opaqueVirtual = (label: string) => ['stream-proxy', 'stream-source', 'lod', ...(label.startsWith('opaque') ? ['masked-lod', 'masked-stream'] : [])];
    for (const label of ['hold B 2', 'opaque hold B 1']) for (const name of opaqueVirtual(label)) {
      expect(cut(frame(label).virtual, name).occlusion.culledClusters, `${label} ${name}`).toBeGreaterThan(0);
    }
    for (const label of ['teleport to A', 'opaque teleport to A']) for (const name of opaqueVirtual(label)) {
      const probe = cut(frame(label).virtual, name);
      expect(probe.occlusion.recoveredClusters, `${label} ${name}`).toBeGreaterThan(0); expect(probe.indices, `${label} ${name}`).toBeGreaterThan(0);
    }
    // Switching the Masked LOD and stream batches to Opaque moves them onto the
    // virtual path; switching back returns them to the full source.
    for (const name of ['masked-lod', 'masked-stream']) {
      expect(cut(frame('masked LOD/stream switched to Opaque').virtual, name).indices, name).toBeGreaterThan(0);
      for (const label of ['hold B 2', 'teleport to A', 'switched back to Masked']) expect(cut(frame(label).virtual, name).indices, `${label} ${name}`).toBe(0);
    }
    // The oversubscribed streams kept coarser resident ancestors instead of dropping clusters.
    expect(r.frames.some(entry => entry.virtual?.some(probe => probe.fallbackClusters > 0))).toBe(true);
  });
}

test('combinations the renderer cannot draw are refused explicitly, never accepted and left undrawn', async ({ page }) => {
  test.setTimeout(hardware ? 60_000 : 180_000);
  const pageErrors: string[] = [];
  await openEngine(page, pageErrors);
  const r = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-silent-drop.ts') as Fixtures;
    return fixture.runGeometrySilentDropRejections();
  });
  console.log('SILENT_DROP_REJECTIONS', JSON.stringify(r));
  expect(r.results.map(entry => entry.name)).toEqual(['skinned + geometryLod', 'morph + geometryLod', 'wave + geometryLod',
    'skinned + full-source stream', 'morph + full-source stream', 'skinned + source-free proxy stream', 'morph + source-free proxy stream',
    'Masked + source-free proxy stream', 'geometryLod + geometryStream', 'occlusion without virtual geometry']);
  for (const entry of r.results) {
    expect(entry.createdRenderer, entry.name).toBe(false);
    expect(entry.error, entry.name).toContain(entry.expected);
  }
  for (const entry of r.offline) expect(entry.error, entry.name).toContain(entry.expected);
  // A refused Masked switch on a source-free proxy leaves the batch drawn and queryable as before.
  const { before, after, maskError } = r.proxySwitch;
  expect(maskError).toContain('GEOMETRY_PROXY_MASK_REQUIRES_FULL_SOURCE');
  for (const look of [before, after]) {
    expect(Math.max(...look.hdr.map((value, channel) => Math.abs(value - [0.9, 0.1, 0.1][channel])))).toBeLessThanOrEqual(0.02);
    expect(look.hit).toEqual({ batch: 'proxy', entityId: 11 });
    expect(look.indices).toBeGreaterThan(0);
  }
  expect(after.materialRevision).toBe(before.materialRevision);
  expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]); expect(r.ownedAfterDispose).toBe(0);
  expectAdapter(r.adapter);
});
