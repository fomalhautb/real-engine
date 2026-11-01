import { expect, test, type BrowserServer, type Page, type TestInfo } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

test('large-scene page loads without an asset set and explains how to provide one', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/large-scene.html');
  await page.waitForFunction(() => window.__LARGE_SCENE__?.snapshot().status === 'idle');
  await expect(page.locator('#status')).toContainText('NO ASSET SET');
  await expect(page.locator('#error')).toBeHidden();
  expect(errors).toEqual([]);
});

test('large-scene page rejects a non-local asset set path explicitly', async ({ page }) => {
  await page.goto('/large-scene.html?set=https://example.com/asset-set.json');
  await page.waitForFunction(() => window.__LARGE_SCENE__?.snapshot().status === 'error');
  await expect(page.locator('#error')).toContainText('GLTF_PACKAGE_ROOT: assetSet');
});

test('large-scene page rejects out-of-range budgets before loading anything', async ({ page }) => {
  for (const [query, message] of [['geometryMiB=8', 'LARGE_SCENE_PARAM: geometryMiB'], ['probes=4096', 'LARGE_SCENE_PARAM: probes'],
    ['fogPixel=0', 'LARGE_SCENE_PARAM: fogPixel'], ['lighting=maybe', 'LARGE_SCENE_PARAM: lighting'], ['reflections=2', 'LARGE_SCENE_PARAM: reflections'],
    ['gi=0&air=1', 'LARGE_SCENE_PARAM: air']] as const) {
    await page.goto(`/large-scene.html?set=/artifacts/none/asset-set.json&${query}`);
    await page.waitForFunction(() => window.__LARGE_SCENE__?.snapshot().status === 'error');
    await expect(page.locator('#error')).toContainText(message);
  }
});

interface Png { width: number; height: number; data: Uint8Array }
const wait = (page: Page, timeout: number) => page.waitForFunction(() => ['done', 'error'].includes(window.__LARGE_SCENE__?.snapshot().status ?? ''),
  undefined, { timeout, polling: 500 });
/** Decodes a screenshot in the page (as engine.spec does) without colour conversion; returns its RGBA bytes. */
async function decode(page: Page, png: Buffer): Promise<Png> {
  const { width, height, rgba } = await page.evaluate(async base64 => {
    const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), context = canvas.getContext('2d', { willReadFrequently: true })!;
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let binary = '';
    for (let i = 0; i < data.length; i += 0x8000) binary += String.fromCharCode(...data.subarray(i, i + 0x8000));
    return { width: canvas.width, height: canvas.height, rgba: btoa(binary) };
  }, png.toString('base64'));
  return { width, height, data: Buffer.from(rgba, 'base64') };
}
const meanDifference = (a: Png, b: Png) => {
  let sum = 0;
  for (let i = 0; i < a.data.length; i += 4) sum += Math.abs(a.data[i] - b.data[i]) + Math.abs(a.data[i + 1] - b.data[i + 1]) + Math.abs(a.data[i + 2] - b.data[i + 2]);
  return sum / (3 * a.width * a.height);
};
const delay = (ms: number) => new Promise<false>(resolve => setTimeout(() => resolve(false), ms));

/**
 * Closes the asset-set test's own browser server with a bounded wait. On the target Mac (Chrome 156),
 * after world queries ran over Bistro Exterior's 1,591 batches, Chrome's crash handler outlives the
 * browser and keeps the browser's stdout/stderr open; Playwright waits for those pipes, so a normal
 * close never resolves (docs/pending/large-scene.md). Once the browser process itself has exited,
 * the pipes are released explicitly; the result is reported, and a browser that does not exit fails.
 */
async function closeServer(server: BrowserServer) {
  const child = server.process(), started = Date.now();
  const exited = new Promise<true>(resolve => {
    if (child.exitCode !== null || child.signalCode !== null) resolve(true); else child.once('exit', () => resolve(true));
  });
  const closing = server.close();
  const browserExited = await Promise.race([exited, delay(60_000)]);
  const pipesHeld = child.stdio.filter(stream => stream && !stream.destroyed).length;
  for (const stream of child.stdio) stream?.destroy();
  const closed = await Promise.race([closing.then(() => true as const), delay(10_000)]);
  return { browserExited, closed, pipesHeldAfterExit: pipesHeld, ms: Date.now() - started };
}

// Local large assets only (CI has none): LARGE_SCENE_ASSET_SET is the server
// path of an asset-set.json written by scripts/write-asset-set.mjs, e.g.
// /artifacts/bistro-exterior-asset-set-fine-mac-20261031/asset-set.json.
const assetSet = process.env.LARGE_SCENE_ASSET_SET;
test('large-scene walk renders a local asset set with lighting, has no coverage holes and disposes everything', async ({ playwright, launchOptions,
  channel, headless, baseURL }, testInfo) => {
  test.skip(!assetSet, 'LARGE_SCENE_ASSET_SET is not set (local large assets only)');
  test.setTimeout(1_800_000);
  const server = await playwright.chromium.launchServer({ ...launchOptions, channel, headless });
  const browser = await playwright.chromium.connect(server.wsEndpoint());
  let teardown: Awaited<ReturnType<typeof closeServer>>;
  try {
    await walkAuditAndCompare(await browser.newPage({ baseURL, viewport: { width: 1440, height: 1000 } }), testInfo);
  } finally {
    await browser.close().catch(() => undefined);
    teardown = await closeServer(server);
    console.log('LARGE_SCENE_TEARDOWN ' + JSON.stringify(teardown));
  }
  expect(teardown.browserExited).toBe(true);
  expect(teardown.closed).toBe(true);
});

async function walkAuditAndCompare(page: Page, testInfo: TestInfo): Promise<void> {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const frames = Number(process.env.LARGE_SCENE_FRAMES ?? 240);
  await page.goto(`/large-scene.html?set=${encodeURIComponent(assetSet!)}&frames=${frames}`);
  await wait(page, 900_000);
  const result = await page.evaluate(() => window.__LARGE_SCENE__!.snapshot());
  await page.screenshot({ path: testInfo.outputPath('large-scene-walk-final.png') });
  expect(result.failure).toBeNull();
  expect(result.status).toBe('done');
  expect(result.renderedFrames).toBe(frames);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.info!.adapterInfo.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.info!.adapterInfo.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  // The camera path's FOV is applied and the far plane reaches across the whole scene.
  const bounds = result.scene!.bounds, diagonal = Math.hypot(...[0, 1, 2].map(axis => bounds.max[axis] - bounds.min[axis]));
  expect(result.projection!.verticalFovRadians).toBe(result.cameraPath!.verticalFovRadians ?? Math.PI / 3.2);
  expect(result.projection!.far).toBeGreaterThanOrEqual(diagonal);
  // MASK primitives are drawn full-detail; anything omitted is BLEND or mirrored, each with its reason.
  expect(Object.keys(result.scene!.omittedByReason).every(reason => reason.startsWith('BLEND') || reason.startsWith('mirrored'))).toBe(true);
  expect(result.scene!.mask.budget).not.toBeNull();
  // The default geometry budget admits the whole first-view demand; the capacity coordinator stays error-free.
  expect(result.geometry!.plan.budgetDeferredGroups).toBe(0);
  expect(result.geometry!.plan.initialBytes).toBeLessThanOrEqual(result.geometry!.plan.maxResidentBytes);
  expect(result.geometry!.live!.capacity!.error).toBeNull();
  expect(result.geometry!.live!.failedPages).toBe(0);
  expect(result.lighting!.configured).toEqual({ gi: true, reflections: true, fog: true, airLighting: false });
  for (const pass of ['world-query-refit', 'diffuse-gi', 'reflections', 'fog-inject', 'fog-integrate', 'fog-compose']) expect(result.lighting!.passes).toContain(pass);

  // Fixed first keyframe: hold the camera until pages, textures and capacity migrations settle (bounded).
  const keyframe = result.cameraPath!.keyframes[0];
  await page.evaluate(index => { const demo = window.__LARGE_SCENE__!; demo.setUiHidden(true); demo.setDebug('material'); demo.showKeyframe(index); demo.settle(3000); }, keyframe);
  await wait(page, 1_200_000);
  const settled = await page.evaluate(() => window.__LARGE_SCENE__!.snapshot());
  expect(settled.failure).toBeNull();
  expect(settled.lastSettle?.settled, `settle ${JSON.stringify(settled.lastSettle)}: ${JSON.stringify(settled.geometry?.live)}`).toBe(true);
  expect(settled.geometry!.live!.budgetDeferredGroups).toBe(0);

  // Hole audit: cluster view, no anti-aliasing, no fog, so an uncovered pixel shows the exact clear colour.
  await page.evaluate(() => { const demo = window.__LARGE_SCENE__!; demo.setLighting({ fog: false }); demo.setDebug('clusters'); demo.renderer!.setAntiAliasing('none'); demo.run(4, false); });
  await wait(page, 300_000);
  const clusters = await decode(page, await page.screenshot({ path: testInfo.outputPath('large-scene-keyframe0-clusters.png') }));
  const audit = await page.evaluate(() => window.__LARGE_SCENE__!.auditCoverage(8, 3));
  expect([audit.width, audit.height]).toEqual([clusters.width, clusters.height]);
  const at = (x: number, y: number) => [0, 1, 2].map(c => clusters.data[(y * clusters.width + x) * 4 + c]);
  const skyColours = new Map<string, number>();
  for (const [x, y] of audit.sky) { const key = at(x, y).join(','); skyColours.set(key, (skyColours.get(key) ?? 0) + 1); }
  // An uncovered interior pixel shows the clear colour, measured from the sky samples when sky is in view. Without
  // sky (Sponza keyframe 0) the check relies on cluster colours never being that dark: every channel is at least
  // 0.08 linear before exposure (85/255 or more at the default exposure in both asset sets) while the clear colour
  // is (3, 6, 7), so a pixel no brighter than DARK in every channel is a hole as well.
  const DARK = 20;
  const [background, backgroundCount] = [...skyColours].sort((a, b) => b[1] - a[1])[0] ?? [null, 0];
  const clear = background?.split(',').map(Number) ?? null;
  const isHole = (pixel: number[]) => Math.max(...pixel) <= DARK
    || (clear !== null && pixel.reduce((sum, value, c) => sum + Math.abs(value - clear[c]), 0) <= 3);
  const holes = audit.interior.filter(([x, y]) => isHole(at(x, y)));
  expect(audit.interior.length).toBeGreaterThan(1000);
  expect(backgroundCount).toBeGreaterThanOrEqual(0.95 * audit.sky.length);
  if (clear) expect(Math.max(...clear)).toBeLessThanOrEqual(DARK);
  expect(holes).toEqual([]);

  // Lighting and fog comparisons at the same keyframe (material view, FXAA).
  const capture = async (name: string, lighting: Record<string, boolean>, settleFrames: number) => {
    await page.evaluate(([state, n]) => { const demo = window.__LARGE_SCENE__!; demo.setDebug('material'); demo.renderer!.setAntiAliasing('fxaa');
      demo.setLighting(state as Record<string, boolean>); demo.run(n as number, false); }, [lighting, settleFrames] as const);
    await wait(page, 600_000);
    const snapshot = await page.evaluate(() => window.__LARGE_SCENE__!.snapshot());
    expect(snapshot.failure).toBeNull();
    return { png: await decode(page, await page.screenshot({ path: testInfo.outputPath(`large-scene-keyframe0-${name}.png`) })), snapshot };
  };
  const litFog = await capture('lit-fog', { gi: true, reflections: true, fog: true }, 90);
  const litNoFog = await capture('lit-nofog', { fog: false }, 30);
  const direct = await capture('direct', { gi: false, reflections: false }, 30);
  // The sun control: an overhead sun reaches the street floor that the default sun leaves in shadow.
  const defaultSun = result.sun;
  await page.evaluate(() => window.__LARGE_SCENE__!.setSun([0.1, 0.99, 0.1]));
  const overheadSun = await capture('direct-overhead-sun', {}, 30);
  await page.evaluate(sun => window.__LARGE_SCENE__!.setSun(sun), defaultSun);
  const differences = { fog: meanDifference(litFog.png, litNoFog.png), lighting: meanDifference(litNoFog.png, direct.png),
    sun: meanDifference(direct.png, overheadSun.png) };
  expect(differences.fog).toBeGreaterThan(1);
  expect(differences.lighting).toBeGreaterThan(1);
  expect(differences.sun).toBeGreaterThan(1);
  // With timestamp queries, every enabled lighting pass recorded GPU work (diagnostic spans, sampled every 30 frames).
  if (result.info!.timing.startsWith('GPU timestamp-query')) {
    expect(litFog.snapshot.gpuTimings.samples).toBeGreaterThan(0);
    for (const pass of ['diffuseGi', 'reflections', 'volumetricFog'] as const) expect(litFog.snapshot.gpuTimings.passes![pass].max).toBeGreaterThan(0);
  }

  const final = await page.evaluate(() => window.__LARGE_SCENE__!.snapshot());
  const afterDispose = await page.evaluate(() => { const r = window.__LARGE_SCENE__!.renderer!; window.__LARGE_SCENE__!.dispose(); return r.stats.ownedGpuBytes; });
  const report = { walk: result, settled: { lastSettle: settled.lastSettle, drains: settled.drains, geometry: settled.geometry }, audit: { samples: audit.samples, interior: audit.interior.length,
    sky: audit.sky.length, background: clear, backgroundCount, holes, raysPerQuery: audit.raysPerQuery, overflowRetries: audit.overflowRetries },
    differences, timings: { walk: result.gpuTimings, litFog: litFog.snapshot.gpuTimings, litNoFog: litNoFog.snapshot.gpuTimings, direct: direct.snapshot.gpuTimings,
      overheadSun: overheadSun.snapshot.gpuTimings },
    final: { drains: final.drains, geometry: final.geometry, lighting: final.lighting, stats: final.stats }, afterDispose, errors };
  await writeFile(testInfo.outputPath('large-scene-report.json'), JSON.stringify(report, null, 2));
  console.log('LARGE_SCENE ' + JSON.stringify({ frames: result.renderedFrames, adapter: result.info?.adapterInfo, projection: result.projection,
    scene: { ...result.scene, bounds: undefined }, plan: result.geometry?.plan, settle: settled.lastSettle,
    streaming: { walkDrains: result.drains, settledDrains: settled.drains, finalDrains: final.drains,
      walkCallbacks: result.callbacks, walkBackpressure: result.backpressureFrames, capacity: final.geometry?.live?.capacity
        && { migrations: final.geometry.live.capacity.migrations, deferred: final.geometry.live.capacity.deferred } },
    audit: report.audit, differences, ownedGpuBytes: final.stats?.ownedGpuBytes, afterDispose }));
  expect(errors).toEqual([]);
  expect(afterDispose).toBe(0);
}
