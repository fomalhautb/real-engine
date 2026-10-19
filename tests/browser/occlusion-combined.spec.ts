import { expect, test } from '@playwright/test';
import type { runOcclusionCombinedFixture } from './fixtures/occlusion-combined';

type Result = Awaited<ReturnType<typeof runOcclusionCombinedFixture>>;

test.use({ deviceScaleFactor: 1 });

const energy = (values: readonly number[]) => values[0] + values[1] + values[2];
const distance = (left: readonly number[], right: readonly number[]) =>
  Math.max(...[0, 1, 2].map(channel => Math.abs(left[channel] - right[channel])));

test('recovered virtual PBR geometry preserves combined GI, final gather, reflections and fog', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'occlusion-combined.ts') as {
      runOcclusionCombinedFixture: typeof runOcclusionCombinedFixture;
    };
    return fixture.runOcclusionCombinedFixture();
  }) as Result;
  await testInfo.attach('occlusion-combined.json', {
    body: JSON.stringify(result, null, 2), contentType: 'application/json',
  });
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  expect(result.diagnostics.occlusion?.algorithm).toBe('two-phase-current-depth-recovery');
  expect(result.diagnostics.diffuseGi?.finalGather).not.toBeNull();
  expect(result.diagnostics.diffuseGi?.finalGather?.pathSegmentsPerUpdate).toBe(2);
  expect(result.diagnostics.reflections?.diffuseGi).toBe(true);
  expect(result.diagnostics.reflections?.worldMedia).toBe(true);
  expect(result.diagnostics.volumetricFog).not.toBeNull();
  expect(result.settledFrames).toBeLessThan(24);
  expect(result.closed.occlusion?.deferredClusters).toBeGreaterThan(0);
  expect(result.closed.occlusion?.culledClusters).toBeGreaterThan(0);
  expect(result.closed.triangles, 'the opaque wall must remove actual virtual triangles')
    .toBeLessThan(result.closed.baselineTriangles);
  expect(result.closed.baselineTriangles).toBe(result.sourceTriangles);
  expect(result.recovered.occlusion?.deferredClusters).toBeGreaterThan(0);
  expect(result.recovered.occlusion?.recoveredClusters).toBeGreaterThan(0);
  expect(result.recovered.triangles).toBe(result.recovered.baselineTriangles);
  expect(result.open.triangles).toBe(result.open.baselineTriangles);
  for (const [label, frame] of [['closed', result.closed], ['recovered', result.recovered], ['open', result.open]] as const) {
    expect(frame.hdr.max, `${label} HDR`).toBeLessThanOrEqual(0.006);
    expect(frame.hdr.mean, `${label} HDR mean`).toBeLessThanOrEqual(0.001);
    expect(frame.depthMax, `${label} depth`).toBeLessThanOrEqual(0.0001);
  }
  expect(result.recovered.hdr.visible).toBeGreaterThan(8);
  expect(result.hiddenQuery.hits[0]?.entityId).toBe(101);
  expect(result.contributions.reflection.valid).toBe(true);
  expect(result.contributions.reflection.identity[1]).toBe(101);
  expect(result.contributions.reflection.failedQueries).toBe(0);
  expect(energy(result.contributions.reflection.radiance)).toBeGreaterThan(0.1);
  expect(distance(result.contributions.surfaceOn, result.contributions.reflectionOff),
    'reflection lights the recovered PBR surface').toBeGreaterThan(0.03);
  expect(result.contributions.giEnergy).toBeGreaterThan(0.1);
  expect(result.contributions.gather.valid).toBe(true);
  expect(energy(result.contributions.gather.irradiance)).toBeGreaterThan(0.005);
  expect(distance(result.contributions.surfaceOn, result.contributions.giOff),
    'GI lights the recovered PBR surface').toBeGreaterThan(0.005);
  expect(distance(result.contributions.surfaceOn, result.contributions.hdrOn),
    'fog changes the recovered surface radiance in the same frame').toBeGreaterThan(0.02);
  expect(distance(result.contributions.fogOffSurface, result.contributions.fogOff),
    'disabled fog passes through the same frame surface radiance').toBeLessThanOrEqual(0.006);
  expect(result.errors).toEqual([]);
  expect(result.ownedAfterDispose).toEqual([0, 0]);
  expect(pageErrors).toEqual([]);
});
