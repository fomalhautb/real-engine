import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runPreparedTextureSurfaceFixture, runPreparedTextureLightingFixture,
  runAutomaticTextureLightingFixture } from './fixtures/prepared-textures';

test.use({ deviceScaleFactor: 1 });
function adapterCheck(adapter: { isFallbackAdapter: boolean | null; vendor: string }) {
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
}
function near(actual: readonly number[], expected: readonly number[], tolerance: number, label: string) {
  for (let c = 0; c < expected.length; c++) expect(Math.abs(actual[c] - expected[c]), `${label}/${c}`).toBeLessThan(tolerance);
}

test('prepared mip tails and compressed textures agree in raster, shadow and world queries', async ({ page }, info) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'prepared-textures.ts') as {
      runPreparedTextureSurfaceFixture: typeof runPreparedTextureSurfaceFixture };
    return module.runPreparedTextureSurfaceFixture();
  });
  const report = JSON.stringify({ ...result,
    samples: result.samples.map(({ png: _, ...sample }) => sample),
    compressed: result.compressed.map(({ png: _, ...sample }) => sample) }, null, 2);
  await writeFile(info.outputPath('prepared-surface.json'), report);
  await info.attach('prepared-surface.json', { body: report, contentType: 'application/json' });
  adapterCheck(result.adapter);
  for (const sample of [...result.samples, ...result.compressed]) {
    await writeFile(info.outputPath(`${sample.name}.png`), Buffer.from(sample.png.split(',')[1], 'base64'));
  }
  const samples = Object.fromEntries(result.samples.map(sample => [sample.name, sample]));
  const red = [192, 32, 16].map(v => v / 255), green = [32, 192, 16].map(v => v / 255), blue = [16, 32, 192].map(v => v / 255);
  for (const [name, hdr, query] of [
    ['full', red, red], ['tail', green, green], ['tail-after-rejections', green, green], ['tail-explicit-lod2', green, blue],
    ['tail-parameter-update', [green[0] * 0.5, ...green.slice(1)], [green[0] * 0.5, ...green.slice(1)]],
    ['tail-sampler-lod2', blue, blue],
  ] as const) {
    near(samples[name].hdr, hdr, 0.004, `${name} surface`);
    expect(samples[name].hit).not.toBeNull();
    near(samples[name].hit!.baseColor, query, 0.004, `${name} query`);
    expect(samples[name].depth.surface).toBeLessThan(1);
    expect(samples[name].depth.shadow).toBeLessThan(1);
  }
  expect(samples.full.materials.imageBytes).toBe(340);
  expect(samples.tail.materials.imageBytes).toBe(84);
  expect(result.failures.length).toBeGreaterThanOrEqual(1);
  for (const failure of result.failures) {
    expect(failure.unchanged, failure.name).toBe(true);
    expect(failure.error).toContain(failure.name === 'budget' ? 'MATERIAL_TEXTURE_BUDGET' : 'MATERIAL_TEXTURE_FEATURE');
  }
  expect(samples['tail-mask-hole'].hit).toBeNull();
  expect(samples['tail-mask-hole'].depth).toMatchObject({ surface: 1, shadow: 1 });
  const formatCount = result.features.filter(feature => ['texture-compression-bc', 'texture-compression-etc2', 'texture-compression-astc'].includes(feature)).length;
  expect(result.compressed).toHaveLength(formatCount * 2);
  for (const sample of result.compressed) {
    expect(sample.materials.imageBytes).toBe(48);
    if (sample.expected[3] < 0.5) {
      expect(sample.hit, sample.name).toBeNull();
      expect(sample.depth).toMatchObject({ surface: 1, shadow: 1 });
    } else {
      expect(sample.hit, sample.name).not.toBeNull();
      near(sample.hdr, sample.expected.slice(0, 3), sample.tolerance, sample.name + ' HDR');
      near(sample.hit!.baseColor, sample.expected.slice(0, 3), sample.tolerance, sample.name + ' query');
      expect(sample.depth.surface).toBeLessThan(1); expect(sample.depth.shadow).toBeLessThan(1);
    }
  }
  const mixed = green.map((value, c) => (value + blue[c]) / 2);
  for (const [name, sample] of [['full', result.npotFull], ['tail', result.npotTail]] as const) {
    near(sample.hdr, mixed, 0.015, 'NPOT implicit LOD ' + name);
    near(sample.hit!.baseColor, mixed, 0.004, 'NPOT explicit LOD ' + name);
  }
  expect(result.npotFull.materials.imageBytes).toBe(224);
  expect(result.npotTail.materials.imageBytes).toBe(44);
  expect(result.errors).toEqual([]); expect(result.afterDispose).toEqual([0, 0]); expect(errors).toEqual([]);
});

test('offscreen prepared and compressed emitters update GI and reflection through Mask and deformation', async ({ page }, info) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'prepared-textures.ts') as {
      runPreparedTextureLightingFixture: typeof runPreparedTextureLightingFixture };
    return module.runPreparedTextureLightingFixture();
  });
  await writeFile(info.outputPath('prepared-lighting.json'), JSON.stringify(result, null, 2));
  await info.attach('prepared-lighting.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  adapterCheck(result.adapter);
  const formatCount = result.features.filter(feature => ['texture-compression-bc', 'texture-compression-etc2', 'texture-compression-astc'].includes(feature)).length;
  expect(result.samples).toHaveLength(4 + formatCount * 2);
  for (const sample of result.samples) {
    expect(sample.anchor.valid, sample.name).toBe(true);
    expect(sample.anchor.failedQueries, sample.name).toBe(0);
    near(sample.anchor.radiance, sample.expected, 0.015, sample.name + ' reflection');
    near(sample.giMax, sample.expected, 0.015, sample.name + ' GI');
  }
  expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0); expect(errors).toEqual([]);
});

test('bounded staged texture uploads publish once while GI and reflections retain their values', async ({ page }, info) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'prepared-textures.ts') as {
      runPreparedTextureLightingFixture: typeof runPreparedTextureLightingFixture };
    return module.runPreparedTextureLightingFixture(true);
  });
  const report = JSON.stringify(result, null, 2);
  await writeFile(info.outputPath('prepared-lighting-staged.json'), report);
  await info.attach('prepared-lighting-staged.json', { body: report, contentType: 'application/json' });
  adapterCheck(result.adapter);
  const formatCount = result.features.filter(feature => ['texture-compression-bc', 'texture-compression-etc2', 'texture-compression-astc'].includes(feature)).length;
  expect(result.samples).toHaveLength(4 + formatCount * 2);
  expect(result.stagedUploads).toHaveLength(formatCount ? 1 + formatCount * 2 : 0);
  for (const upload of result.stagedUploads) {
    expect(upload.bytes.length).toBeGreaterThan(0);
    for (const bytes of upload.bytes) expect(bytes).toBeGreaterThan(0);
    for (const bytes of upload.bytes) expect(bytes).toBeLessThanOrEqual(16);
    expect(upload.total).toBe(upload.layoutBytes);
    expect(upload.total).toBe(upload.bytes.reduce((sum, bytes) => sum + bytes, 0));
    expect(upload.afterRevision).toBe(upload.beforeRevision + 1);
    expect(upload.afterVariants).toBe(upload.beforeVariants);
    expect(upload.materials).toBe(1);
    expect(upload.slots).toBe(1);
  }
  for (const sample of result.samples) {
    expect(sample.anchor.valid, sample.name).toBe(true);
    expect(sample.anchor.failedQueries, sample.name).toBe(0);
    near(sample.anchor.radiance, sample.expected, 0.015, sample.name + ' reflection');
    near(sample.giMax, sample.expected, 0.015, sample.name + ' GI');
  }
  expect(result.errors).toEqual([]); expect(result.afterDispose).toBe(0); expect(errors).toEqual([]);
});

test('automatic texture residency updates offscreen GI and reflections without changing the emitter path', async ({ page }, info) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'prepared-textures.ts') as {
      runAutomaticTextureLightingFixture: typeof runAutomaticTextureLightingFixture };
    return module.runAutomaticTextureLightingFixture();
  });
  const report = JSON.stringify(result, null, 2);
  await writeFile(info.outputPath('automatic-texture-lighting.json'), report);
  await info.attach('automatic-texture-lighting.json', { body: report, contentType: 'application/json' });
  adapterCheck(result.adapter);
  const coarse = [32, 192, 16].map(value => value / 255);
  const fine = [192, 32, 16].map(value => value / 255);
  for (const [sample, mip, color] of [
    [result.coarse, 1, coarse], [result.fine, 0, fine], [result.coarseAgain, 1, coarse],
    [result.morphAway, 1, [0, 0, 0]], [result.morphRestored, 1, coarse],
  ] as const) {
    expect(sample.mip, sample.name).toBe(mip);
    expect(sample.anchor.valid, sample.name).toBe(true);
    expect(sample.anchor.failedQueries, sample.name).toBe(0);
    near(sample.anchor.radiance, color, 0.015, sample.name + ' reflection');
    near(sample.giMax, color, 0.015, sample.name + ' GI');
    expect(sample.variants).toBe(result.coarse.variants);
  }
  expect(result.fine.revision).toBeGreaterThan(result.coarse.revision);
  expect(result.coarseAgain.revision).toBeGreaterThan(result.fine.revision);
  expect(result.morphAway.revision).toBe(result.coarseAgain.revision);
  expect(result.morphRestored.revision).toBe(result.coarseAgain.revision);
  expect(result.downgradeFrames).toBeGreaterThan(15);
  for (const frame of result.frames) {
    const streaming = frame.streaming;
    expect(streaming.frameUploadedBytes).toBeLessThanOrEqual(32);
    expect(streaming.cpuBytes + streaming.reservedCpuBytes).toBeLessThanOrEqual(512);
    expect(streaming.inFlightBytes).toBeLessThanOrEqual(512);
    expect(frame.imageBytes).toBeLessThanOrEqual(500);
    expect(streaming.sources.every(source => source.failure === null)).toBe(true);
  }
  expect(result.errors).toEqual([]);
  expect(errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
});
