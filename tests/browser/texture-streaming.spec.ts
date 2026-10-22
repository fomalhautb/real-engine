import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import type { runTextureStreamingFixture } from './fixtures/texture-streaming';

test.use({ deviceScaleFactor: 1 });

for (const [useManifest, useGltf] of [[false, false], [true, false], [true, true]]) {
test(useGltf ? 'glTF materials preserve manifest texture identity, sRGB sampling, and automatic mip publication'
  : useManifest
  ? 'manifest texture streaming preserves coarse images, budgets, and shared material publication'
  : 'automatic texture streaming preserves coarse images, budgets, and shared material publication', async ({ page }, info) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  const colors = {
    a: [[192, 32, 16], [32, 192, 16], [16, 32, 192], [128, 64, 32]],
    b: [[16, 192, 96], [192, 16, 96], [96, 16, 192], [32, 128, 64]],
  };
  const httpRequests: string[] = [];
  const mipBytes = (source: 'a' | 'b', level: number) => {
    const bytes = Buffer.alloc((8 >> level) ** 2 * 4), rgb = colors[source][level];
    for (let offset = 0; offset < bytes.length; offset += 4) bytes.set([...rgb, 255], offset);
    return bytes;
  };
  await page.route('**/__texture-stream/**', async route => {
    const match = new URL(route.request().url()).pathname.match(/^\/__texture-stream\/(a|b)\/(manifest\.json|[0-3]\.bin)$/);
    if (!match) throw new Error('Unexpected texture mip URL');
    const source = match[1] as 'a' | 'b';
    if (match[2] === 'manifest.json') {
      httpRequests.push(`${source}/manifest`);
      const mips = Array.from({ length: 4 }, (_, level) => {
        const bytes = mipBytes(source, level);
        return { level, width: 8 >> level, height: 8 >> level, byteLength: bytes.byteLength,
          sha256: createHash('sha256').update(bytes).digest('hex'), file: `${level}.bin` };
      });
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
        version: 1, id: source, width: 8, height: 8, colorSpace: useGltf ? 'srgb' : 'linear', format: 'rgba8',
        fallbackMip: 2, mips,
      }) });
      return;
    }
    const level = Number(match[2][0]);
    httpRequests.push(`${source}/${level}`);
    await route.fulfill({ status: 200, contentType: 'application/octet-stream', body: mipBytes(source, level) });
  });
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async ({ manifest, gltf }) => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'texture-streaming.ts') as {
      runTextureStreamingFixture: typeof runTextureStreamingFixture };
    return module.runTextureStreamingFixture(manifest, gltf);
  }, { manifest: useManifest, gltf: useGltf });
  const samples = [result.coarse, result.partial, result.aFine, result.aLateMaterial,
    result.bFine, result.bLateMaterial, result.aReturn];
  await writeFile(info.outputPath('texture-streaming.json'), JSON.stringify({ ...result, httpRequests,
    coarse: { ...result.coarse, png: undefined }, partial: { ...result.partial, png: undefined },
    aFine: { ...result.aFine, png: undefined }, aLateMaterial: { ...result.aLateMaterial, png: undefined },
    bFine: { ...result.bFine, png: undefined }, bLateMaterial: { ...result.bLateMaterial, png: undefined },
    aReturn: { ...result.aReturn, png: undefined } }, null, 2));
  for (const sample of [result.coarse, result.partial, result.aFine, result.bFine, result.aReturn]) {
    await writeFile(info.outputPath(`texture-${sample.name}.png`), Buffer.from(sample.png.split(',')[1], 'base64'));
  }
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);

  const near = (actual: readonly number[], expected: readonly number[], label: string) => {
    expected.forEach((value, channel) => expect(Math.abs(actual[channel] - value), `${label}/${channel}`).toBeLessThan(0.004));
  };
  const expected = (source: 'a' | 'b', mip: number, tint: readonly number[] = [1, 1, 1]) =>
    colors[source][mip].map((value, channel) => {
      const c = value / 255;
      return (useGltf ? (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4) : c) * tint[channel];
    });
  for (const sample of samples) {
    const current = Object.fromEntries(sample.streaming.sources.map(source => [source.id, source.currentMip]));
    const rightTint = sample === result.coarse || sample === result.partial || sample === result.aFine
      ? [1, 0.5, 1] : [0.5, 1, 1];
    const queryColors = [expected('a', current.a), expected('a', current.a, rightTint), expected('b', current.b)];
    sample.hits.forEach((hit, index) => {
      expect(hit, `${sample.name} query ${index}`).not.toBeNull();
      near(hit!.baseColor, queryColors[index], `${sample.name} query ${index}`);
    });
    const visible = sample.view === 'a' ? queryColors.slice(0, 2) : queryColors.slice(2);
    sample.hdr.forEach((pixel, index) => near(pixel, visible[index], `${sample.name} HDR ${index}`));
    for (const point of sample.depth) {
      expect(point.surface, sample.name).toBeLessThan(1);
      expect(point.shadow, sample.name).toBeLessThan(1);
    }
    expect(sample.variants).toBe(result.coarse.variants);
  }

  expect(result.coarse.streaming.sources.map(source => source.currentMip)).toEqual([2, 2]);
  expect(result.partial.streaming.sources.map(source => source.currentMip)).toEqual([2, 2]);
  expect(result.partial.streaming.frameUploadedBytes).toBeGreaterThan(0);
  expect(result.aFine.streaming.sources.map(source => source.currentMip)).toEqual([0, 2]);
  expect(result.bFine.streaming.sources.map(source => source.currentMip)).toEqual([2, 0]);
  expect(result.aReturn.streaming.sources.map(source => source.currentMip)).toEqual([0, 2]);
  expect(result.managedError).toContain('TEXTURE_STREAM_MANAGED_IDENTITY');
  expect(result.bFine.streaming.evictions).toBeGreaterThan(0);
  expect(result.aReturn.streaming.evictions).toBeGreaterThan(result.bFine.streaming.evictions);
  for (const frame of result.frames) {
    const state = frame.streaming!;
    expect(state.frameUploadedBytes).toBeLessThanOrEqual(32);
    expect(state.cpuBytes + state.reservedCpuBytes).toBeLessThanOrEqual(400);
    expect(state.inFlightBytes).toBeLessThanOrEqual(512);
    expect(frame.imageBytes).toBeLessThanOrEqual(400);
    expect(state.sources.every(source => source.failure === null)).toBe(true);
  }
  for (const key of ['requests', 'uploadedBytes', 'publications', 'evictions'] as const) {
    expect(result.settledEnd[key], key).toBe(result.settledStart[key]);
  }
  expect(result.settledEnd.active).toBe(0);
  const detailRequests = httpRequests.filter(request => /^[ab]\/[01]$/.test(request));
  expect(detailRequests).toEqual(['a/0', 'a/1', 'b/0', 'b/1', 'a/0', 'a/1']);
  if (useManifest) {
    for (const source of ['a', 'b']) {
      expect(httpRequests.filter(request => request.startsWith(`${source}/`)).slice(0, 3))
        .toEqual([`${source}/manifest`, `${source}/2`, `${source}/3`]);
    }
  } else {
    expect(httpRequests).toEqual(detailRequests);
  }
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(result.afterDispose).toBe(0);
});
}
