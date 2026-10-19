import { expect, test, type TestInfo } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runVertexColorSurfaceFixture, runVertexColorHierarchyFixture } from './fixtures/vertex-color';

type Surface = Awaited<ReturnType<typeof runVertexColorSurfaceFixture>>;
type Hierarchy = Awaited<ReturnType<typeof runVertexColorHierarchyFixture>>;
test.use({ deviceScaleFactor: 1 });

async function savePng(testInfo: TestInfo, name: string, url: string) {
  const data = Buffer.from(url.split(',')[1], 'base64');
  await writeFile(testInfo.outputPath(name), data);
  await testInfo.attach(name, { body: data, contentType: 'image/png' });
}

function expectAdapter(adapter: { isFallbackAdapter: boolean | null; vendor: string }) {
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
}

test('vertex RGBA interpolates independently through raster, mask, shadow, query and shared deformation',
  async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    const pageErrors: string[] = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.goto('/?trees=1');
    await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    const result = await page.evaluate(async () => {
      window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
      const fixture = await import('/tests/browser/fixtures/' + 'vertex-color.ts') as {
        runVertexColorSurfaceFixture: typeof runVertexColorSurfaceFixture;
      };
      return fixture.runVertexColorSurfaceFixture();
    }) as Surface;
    await testInfo.attach('vertex-color-surface.json', {
      body: JSON.stringify({ ...result, opaque: { ...result.opaque, png: undefined },
        masked: { ...result.masked, png: undefined }, moved: { ...result.moved, png: undefined },
        white: { ...result.white, png: undefined } }, null, 2),
      contentType: 'application/json',
    });
    for (const state of ['opaque', 'masked', 'moved'] as const) {
      await savePng(testInfo, `vertex-color-${state}.png`, result[state].png);
    }
    await savePng(testInfo, 'vertex-color-missing-white.png', result.white.png);
    expectAdapter(result.adapter);
    for (let i = 0; i < result.points.length; i++) {
      const x = result.points[i][0];
      const rgba = [0.5 + 0.3 * x, 0.4, 0.25, 0.5 + 0.3 * x];
      const expectedHdr = [rgba[0] * 1.2, rgba[1] * 1.2, rgba[2] * 1.2];
      const visible = x > 0;
      for (const state of ['opaque', 'masked', 'moved'] as const) {
        const capture = result[state], hit = capture.hits[i];
        if (state !== 'opaque' && !visible) {
          expect(hit, `${state} x=${x} query hole`).toBeNull();
          expect(capture.depth[i].surface, `${state} x=${x} surface hole`).toBe(1);
          expect(capture.depth[i].shadow, `${state} x=${x} shadow hole`).toBe(1);
          continue;
        }
        expect(hit, `${state} x=${x} query hit`).not.toBeNull();
        for (let c = 0; c < 3; c++) {
          expect(Math.abs(hit!.baseColor[c] - rgba[c]), `${state} x=${x} query colour ${c}`).toBeLessThan(0.008);
          expect(Math.abs(hit!.emissive[c] - rgba[c] * 0.2),
            `${state} x=${x} emission ${c}`).toBeLessThan(0.008);
          expect(Math.abs(capture.hdr[i][c] - expectedHdr[c]),
            `${state} x=${x} HDR ${c}`).toBeLessThan(0.015);
        }
        expect(Math.abs(hit!.opacity - rgba[3]), `${state} x=${x} opacity`).toBeLessThan(0.008);
        expect(capture.depth[i].surface).toBeGreaterThan(0);
        expect(capture.depth[i].surface).toBeLessThan(1);
        expect(capture.depth[i].shadow).toBeGreaterThan(0);
        expect(capture.depth[i].shadow).toBeLessThan(1);
      }
    }
    const radiance = (probe: Surface['giEmissive']) => [0, 1, 2].map(channel =>
      Math.max(...probe.probes.map(item => item.maxRadiance[channel])));
    const lit = radiance(result.giEmissive), dark = radiance(result.giDark);
    // The emission multiplier is neutral 0.2. Unequal channel output must
    // therefore come from the authored vertex colours at real GI ray hits.
    expect(lit[0], 'GI red from vertex-coloured hits').toBeGreaterThan(0.08);
    expect(lit[0] / lit[2], 'GI red/blue authored ratio').toBeGreaterThan(1.5);
    expect(lit[1] / lit[2], 'GI green/blue authored ratio').toBeGreaterThan(1.1);
    for (const channel of dark) expect(channel, 'GI emission removed').toBeLessThan(0.005);
    expect(result.white.hit).not.toBeNull();
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(result.white.hit!.baseColor[c] - 1), `missing colour query ${c}`).toBeLessThan(0.005);
      expect(Math.abs(result.white.hdr[c] - 1.2), `missing colour HDR ${c}`).toBeLessThan(0.015);
    }
    expect(result.white.hit!.opacity).toBe(1);
    expect(result.errors).toEqual([]);
    expect(result.afterDispose).toBe(0);
    expect(result.whiteAfterDispose).toBe(0);
    expect(pageErrors).toEqual([]);
  });

test('resident and source-free paged geometry retain authored vertex colours', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'vertex-color.ts') as {
      runVertexColorHierarchyFixture: typeof runVertexColorHierarchyFixture;
    };
    return fixture.runVertexColorHierarchyFixture();
  }) as Hierarchy;
  await testInfo.attach('vertex-color-hierarchy.json', {
    body: JSON.stringify({ ...result, samples: result.samples.map(({ png: _png, ...sample }) => sample),
      coarse: result.coarse.map(({ png: _png, ...sample }) => sample),
      sourceColors: `${result.sourceColors.length} values`, proxyColors: `${result.proxyColors.length} values` }, null, 2),
    contentType: 'application/json',
  });
  for (let i = 0; i < result.samples.length; i++) {
    await savePng(testInfo, `vertex-color-coarse-${i}.png`, result.coarse[i].png);
    await savePng(testInfo, `vertex-color-hierarchy-${i}.png`, result.samples[i].png);
  }
  expectAdapter(result.adapter);
  expect(result.pages).toBeGreaterThan(0);
  expect(result.sourceColors.length).toBeGreaterThan(0);
  expect(result.proxyColors.length).toBeGreaterThan(0);
  for (let i = 0; i < 3; i++) for (let p = 0; p < 3; p++) {
    const x = result.points[p][0], expected = [0.5 + 0.3 * x, 0.4, 0.25];
    const sample = result.samples[i], hit = sample.hits[p];
    expect(hit, `representation ${i} point ${p}`).not.toBeNull();
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(hit!.baseColor[c] - expected[c]), `representation ${i} query ${p}/${c}`).toBeLessThan(0.025);
      expect(Math.abs(sample.hdr[p][c] - expected[c] * 1.2),
        `representation ${i} HDR ${p}/${c}`).toBeLessThan(0.035);
      expect(Math.abs(result.coarse[i].hdr[p][c] - expected[c] * 1.2),
        `coarse representation ${i} HDR ${p}/${c}`).toBeLessThan(0.06);
    }
  }
  expect(result.samples[1].lod!.triangles).toBeGreaterThan(0);
  expect(result.samples[2].lod!.triangles).toBeGreaterThan(0);
  expect(result.samples[1].lod!.triangles).toBe(result.sourceTriangles);
  expect(result.samples[2].lod!.triangles).toBe(result.sourceTriangles);
  expect(result.coarse[1].lod!.triangles).toBeLessThan(result.sourceTriangles);
  expect(result.coarse[2].lod!.triangles).toBeLessThan(result.sourceTriangles);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toEqual([0, 0, 0]);
  expect(pageErrors).toEqual([]);
});

test('prepared RGBA8 mip tail multiplies vertex colour in full, resident and source-free paths',
  async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    const pageErrors: string[] = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.goto('/?trees=1');
    await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    const result = await page.evaluate(async () => {
      window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
      const fixture = await import('/tests/browser/fixtures/' + 'vertex-color.ts') as {
        runVertexColorHierarchyFixture: typeof runVertexColorHierarchyFixture;
      };
      return fixture.runVertexColorHierarchyFixture(true);
    }) as Hierarchy;
    await testInfo.attach('vertex-color-prepared-tail.json', {
      body: JSON.stringify({ ...result, samples: result.samples.map(({ png: _png, ...sample }) => sample),
        coarse: result.coarse.map(({ png: _png, ...sample }) => sample),
        sourceColors: `${result.sourceColors.length} values`, proxyColors: `${result.proxyColors.length} values` }, null, 2),
      contentType: 'application/json',
    });
    await savePng(testInfo, 'vertex-color-tail-paged-coarse.png', result.coarse[2].png);
    await savePng(testInfo, 'vertex-color-tail-paged-fine.png', result.samples[2].png);
    expectAdapter(result.adapter);
    expect(result.preparedBaseMipLevel).toBe(1);
    expect(result.pages).toBeGreaterThan(0);
    expect(result.sourceColors.length).toBeGreaterThan(0);
    expect(result.proxyColors.length).toBeGreaterThan(0);
    // The full-resolution mip is absent. Resident mip 0 corresponds to authored
    // mip 1; the two later mips have deliberately different sentinel colours.
    const tailColor = [192 / 255, 128 / 255, 64 / 255];
    for (const [state, samples] of [['coarse', result.coarse], ['fine', result.samples]] as const) {
      for (let representation = 0; representation < 3; representation++) {
        for (let point = 0; point < result.points.length; point++) {
          const x = result.points[point][0];
          const vertex = [0.5 + 0.3 * x, 0.4, 0.25];
          const hit = samples[representation].hits[point];
          expect(hit, `${state} representation ${representation} point ${point}`).not.toBeNull();
          for (let channel = 0; channel < 3; channel++) {
            const base = vertex[channel] * tailColor[channel];
            expect(Math.abs(hit!.baseColor[channel] - base),
              `${state} representation ${representation} query base ${point}/${channel}`).toBeLessThan(0.025);
            expect(Math.abs(hit!.emissive[channel] - base * 0.2),
              `${state} representation ${representation} query emission ${point}/${channel}`).toBeLessThan(0.025);
            expect(Math.abs(samples[representation].hdr[point][channel] - base * 1.2),
              `${state} representation ${representation} HDR ${point}/${channel}`).toBeLessThan(state === 'coarse' ? 0.06 : 0.035);
          }
          expect(Math.abs(hit!.opacity - (0.5 + 0.3 * x)),
            `${state} representation ${representation} query opacity ${point}`).toBeLessThan(0.025);
        }
      }
    }
    for (const representation of [1, 2]) {
      expect(result.coarse[representation].lod!.triangles).toBeGreaterThan(0);
      expect(result.coarse[representation].lod!.triangles).toBeLessThan(result.sourceTriangles);
      expect(result.samples[representation].lod!.triangles).toBe(result.sourceTriangles);
    }
    expect(result.errors).toEqual([]);
    expect(result.afterDispose).toEqual([0, 0, 0]);
    expect(pageErrors).toEqual([]);
  });
