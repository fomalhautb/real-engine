import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import type * as G05 from './fixtures/geometry-g05';

// G05 geometry record: micro-triangles, overlapping thin sheets and multi-material
// batches from fixed procedural content and seeds. Correctness checks are hard
// assertions; every timing is DIAGNOSTIC (the GPU is shared with other agents) and
// is recorded in the JSON report, never gated.
//
// Environment:
//   G05_SCALE=large      Larger content for the Apple hardware target (1.35M-triangle
//                        sphere, 1920x1080). Skipped on software adapters. The sphere's
//                        packed LOD hierarchy (about 2.06x its source triangles) must fit
//                        the 3M packed-triangle budget of src/geometry-lod-data.ts.
//   G05_REPORT_DIR=<dir> Also copy each JSON report into <dir>, named by adapter.
const hardware = process.env.PLAYWRIGHT_GPU === 'hardware';
const scale = process.env.G05_SCALE ?? 'default';
if (scale !== 'default' && scale !== 'large') throw new Error('G05_SCALE must be unset or "large"');
const large = scale === 'large';
const profile = large ? 'large' as const : 'default' as const;

// Unquantized timestamps (Chrome otherwise rounds them to 65.536 us) for this diagnostic only;
// the override applies to hardware runs so a software run keeps its SwiftShader arguments.
if (hardware) {
  test.use({ launchOptions: { args: [...(process.env.PLAYWRIGHT_HIGH_PERFORMANCE_GPU === '1' ? ['--force-high-performance-gpu'] : []),
    '--enable-webgpu-developer-features'] } });
}
test.skip(large && !hardware, 'G05_SCALE=large is a hardware-only variant');

const frames: G05.G05FramePlan = large ? { warmup: 30, frames: 240, serialized: 60 }
  : hardware ? { warmup: 30, frames: 120, serialized: 30 } : { warmup: 3, frames: 8, serialized: 4 };
const micro: G05.G05MicroOptions = large
  ? { profile, frames, subdivisions: 336, width: 1920, height: 1080, diametersPx: [100, 300], stream: true, maxSettleFrames: 900, stableFrames: 30 }
  : { profile, frames, subdivisions: 296, width: 448, height: 336, diametersPx: [100, 300], stream: true,
    maxSettleFrames: hardware ? 900 : 400, stableFrames: hardware ? 30 : 12 };
const sheets: G05.G05SheetOptions = { profile, frames, width: large ? 1920 : 320, height: large ? 1080 : 200, layers: [1, 8, 32, 64], seed: 0x6505 };
const materials: G05.G05MaterialOptions = { profile, frames, width: large ? 1920 : 480, height: large ? 1080 : 300,
  quads: large ? 512 : 64, materials: [1, 8, 64, 256] };
const timeout = large ? 900_000 : hardware ? 300_000 : 600_000;

type Fixture = typeof G05;
type Runner = 'runG05MicroTriangles' | 'runG05ThinSheets' | 'runG05MultiMaterial';
type Result<K extends Runner> = Awaited<ReturnType<Fixture[K]>>;

async function run<K extends Runner>(page: Page, name: K, options: Parameters<Fixture[K]>[0]): Promise<Result<K>> {
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  return page.evaluate(async ({ name, options }) => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-g05.ts') as Fixture;
    return (fixture[name] as (value: typeof options) => Promise<unknown>)(options);
  }, { name, options }) as Promise<Result<K>>;
}

/** Writes evidence PNGs and the JSON report (evidence replaced by file names); optionally copies the report. */
async function record(testInfo: TestInfo, name: string, result: { evidence: Record<string, string>; adapter: { vendor: string; architecture: string } }) {
  const files: string[] = [];
  for (const [label, url] of Object.entries(result.evidence)) {
    const file = `${name}-${label.replace(/[^a-z0-9.-]+/gi, '_')}.png`, path = testInfo.outputPath(file);
    writeFileSync(path, Buffer.from(url.split(',')[1], 'base64'));
    await testInfo.attach(file, { path, contentType: 'image/png' });
    files.push(file);
  }
  const path = testInfo.outputPath(`${name}.json`);
  writeFileSync(path, JSON.stringify({ ...result, evidence: files }, null, 2));
  await testInfo.attach(`${name}.json`, { path, contentType: 'application/json' });
  const directory = process.env.G05_REPORT_DIR;
  if (directory) {
    mkdirSync(directory, { recursive: true });
    const adapter = `${result.adapter.vendor || 'unknown'}-${result.adapter.architecture || 'unknown'}`.replace(/[^a-z0-9.-]+/gi, '_');
    copyFileSync(path, join(directory, `${name}${large ? '-large' : ''}.${adapter}.json`));
  }
}

function expectAdapter(result: { adapter: { vendor: string; isFallbackAdapter: boolean | null } }) {
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  if (hardware) expect(result.adapter.isFallbackAdapter).not.toBe(true);
}
function expectTiming(timing: G05.G05Timing, label: string) {
  // Harness integrity only; the numbers themselves are diagnostic and never gated.
  expect(timing.measuredFrames, label).toBe(frames.frames);
  expect(timing.serializedFrameMs?.n, label).toBe(frames.serialized);
  if (timing.crossCheck) expect(timing.crossCheck.maxAbsDiffMs, `${label} timestamp cross-check`).toBe(0);
  // Render passes always execute, so their slots must be rewritten every timed frame; the
  // frame span is then defined in every frame. Compute passes without a dispatch may keep
  // stale slots on some adapters; those are counted in the report and excluded, not gated.
  if (timing.timestampIntegrity) {
    for (const pass of ['shadow', 'surface', 'resolve'] as const) {
      expect(timing.timestampIntegrity.byPass[pass]?.fresh, `${label} fresh ${pass} timestamps`).toBe(frames.frames);
    }
    expect(timing.gpuFrameSpanMs?.n, `${label} frame spans`).toBe(frames.frames);
  }
}
function expectCleanup(cleanup: { label: string; errors: string[]; ownedGpuBytesAfterDispose: number }[]) {
  for (const entry of cleanup) {
    expect(entry.errors, entry.label).toEqual([]);
    expect(entry.ownedGpuBytesAfterDispose, entry.label).toBe(0);
  }
}

test(`G05 micro-triangles: dense sphere through full, GPU-LOD and paged paths (${profile}; timings diagnostic)`, async ({ page }, testInfo) => {
  test.setTimeout(timeout);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  const r = await run(page, 'runG05MicroTriangles', micro);
  await record(testInfo, 'g05-micro-triangles', r);
  console.log('GEOMETRY_G05_MICRO', JSON.stringify({ adapter: r.adapter, labels: r.labels, mesh: r.mesh, stream: r.stream,
    runs: r.runs.map(run => ({ label: run.label, drawn: run.drawnTriangles, front: run.frontFacingTriangles, covered: run.silhouette.coveredPixels,
      frontPerPixel: run.frontFacingTrianglesPerCoveredPixel, beyondBand: run.silhouette.beyondBand, iou: run.silhouette.iouAnalytic,
      surfaceMs: run.timing.gpuPassMs.surface?.median ?? null, shadowMs: run.timing.gpuPassMs.shadow?.median ?? null,
      spanMs: run.timing.gpuFrameSpanMs?.median ?? null, cpuMs: run.timing.cpuSubmitMs?.median ?? null })) }));
  expectAdapter(r);
  // Per diameter: full detail, GPU LOD at 0 px and 1 px, and the paged stream at 1 px.
  const runsPerDiameter = 3 + (micro.stream ? 1 : 0);
  expect(r.runs.length).toBe(micro.diametersPx.length * runsPerDiameter);
  for (const run of r.runs) {
    // Coverage silhouette within the stated band of the exact sphere, every pixel classified.
    expect(run.silhouette.unclassifiedPixels, run.label).toBe(0);
    expect(run.silhouette.beyondBand, run.label).toBe(0);
    expect(run.silhouette.coveredPixels, run.label).toBeGreaterThan(0);
    if (run.variant === 'full' || run.maxErrorPixels === 0) expect(run.drawnTriangles, run.label).toBe(r.mesh.sourceTriangles);
    if (run.maxErrorPixels === 1) expect(run.drawnTriangles, run.label).toBeLessThan(r.mesh.sourceTriangles);
    if (run.probe) expect(run.probe.overflow, run.label).toBe(false);
    // The paged workload must not change while it is timed.
    if (run.stream) expect(run.stream.uploadsDuringMeasurement, run.label).toBe(0);
    expectTiming(run.timing, run.label);
  }
  // The content really is in the micro-triangle regime at full detail.
  for (const run of r.runs.filter(entry => entry.variant === 'full')) expect(run.frontFacingTrianglesPerCoveredPixel, run.label).toBeGreaterThan(1);
  expectCleanup(r.cleanup);
  expect(pageErrors).toEqual([]);
});

test(`G05 overlapping thin sheets: opaque and masked stacks, both submission orders (${profile}; timings diagnostic)`, async ({ page }, testInfo) => {
  test.setTimeout(timeout);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  const r = await run(page, 'runG05ThinSheets', sheets);
  await record(testInfo, 'g05-thin-sheets', r);
  console.log('GEOMETRY_G05_SHEETS', JSON.stringify({ adapter: r.adapter, labels: r.labels, comparisons: r.comparisonsAtMaxLayers,
    scaling: r.scaling.map(entry => ({ variant: entry.variant, order: entry.order, surface: entry.surfaceMedianMsByLayers, fit: entry.surfaceFitMsPerLayer })),
    runs: r.runs.map(run => ({ label: run.label, robust: run.correctness.robustPixels, mismatches: run.correctness.robustMismatches,
      nonFront: run.correctness.nonFrontSheetPixels, orderDiff: run.correctness.differsFromOtherOrder,
      shaded: run.estimate.idealEarlyZShadedFragments, surfaceMs: run.timing.gpuPassMs.surface?.median ?? null })) }));
  expectAdapter(r);
  expect(r.runs.length).toBe(sheets.layers.length * 4);
  for (const run of r.runs) {
    expect(run.correctness.unclassifiedPixels, run.label).toBe(0);
    expect(run.correctness.robustPixels, run.label).toBeGreaterThan(sheets.width * sheets.height / 4);
    expect(run.correctness.robustMismatches, run.label).toBe(0);
    // Depth testing, not submission order, decides visibility.
    expect(run.correctness.differsFromOtherOrder, run.label).toBe(0);
    // Opaque: only the front sheet is visible. Masked: deeper sheets show through the cutouts.
    if (run.variant === 'opaque') expect(run.correctness.nonFrontSheetPixels, run.label).toBe(0);
    if (run.variant === 'masked' && run.layers >= 2) {
      expect(run.correctness.expectedNonFrontPixels, run.label).toBeGreaterThan(0);
      expect(run.correctness.nonFrontSheetPixels, run.label).toBeGreaterThan(0);
    }
    if (run.layers === 1) expect(run.correctness.nonFrontSheetPixels, run.label).toBe(0);
    expectTiming(run.timing, run.label);
  }
  expectCleanup(r.cleanup);
  expect(pageErrors).toEqual([]);
});

test(`G05 multi-material: one wall split into M materials and batches with shadows (${profile}; timings diagnostic)`, async ({ page }, testInfo) => {
  test.setTimeout(timeout);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  const r = await run(page, 'runG05MultiMaterial', materials);
  await record(testInfo, 'g05-multi-material', r);
  console.log('GEOMETRY_G05_MATERIALS', JSON.stringify({ adapter: r.adapter, labels: r.labels, scaling: r.scaling,
    runs: r.runs.map(run => ({ label: run.label, batches: run.batches, robust: run.correctness.robustWallPixels,
      minPerTile: run.correctness.minRobustPixelsPerTile, mismatches: run.correctness.robustMismatches, shadow: run.correctness.shadow })) }));
  expectAdapter(r);
  expect(r.runs.map(run => run.materials)).toEqual(materials.materials);
  for (const run of r.runs) {
    // Every material's identity colour fills its own tile, and only there.
    expect(run.correctness.tilesWithoutRobustPixels, run.label).toEqual([]);
    expect(run.correctness.robustMismatches, run.label).toBe(0);
    // Shadows are on: the wall darkens the floor band its sun rays hit.
    expect(run.correctness.shadow.litPixels, run.label).toBeGreaterThan(50);
    expect(run.correctness.shadow.shadowPixels, run.label).toBeGreaterThan(50);
    expect(run.correctness.shadow.shadowOverLit, run.label).toBeLessThan(0.5);
    // M material instances share one compiled tile program plus the floor's.
    expect(run.materialVariants, run.label).toBe(2);
    expect(run.batches, run.label).toBe(run.materials + 1);
    expectTiming(run.timing, run.label);
  }
  expectCleanup(r.cleanup);
  expect(pageErrors).toEqual([]);
});
