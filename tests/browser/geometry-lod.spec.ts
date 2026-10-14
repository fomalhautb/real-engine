import { expect, test } from '@playwright/test';
import type {
  startGeometryLodFixture, captureGeometryLodFixtureDebug, exerciseGeometryLodCoverageFallback,
  disposeGeometryLodFixture,
} from './fixtures/geometry-lod';

type Setup = Awaited<ReturnType<typeof startGeometryLodFixture>>;
type Capture = Awaited<ReturnType<typeof captureGeometryLodFixtureDebug>>;
type Fallback = Awaited<ReturnType<typeof exerciseGeometryLodCoverageFallback>>;

function selected(capture: { selectedClusters: { clusterId: number; instanceIndex: number; level: number }[] }): number[] {
  return capture.selectedClusters.map(cluster => cluster.clusterId).sort((a, b) => a - b);
}

function difference(a: number[][], b: number[][]): number {
  return Math.max(...a.flatMap((pixel, index) => pixel.slice(0, 3).map((value, channel) =>
    Math.abs(value - b[index][channel]))));
}

async function attachScreenshot(page: import('@playwright/test').Page,
  testInfo: import('@playwright/test').TestInfo, name: string, selector?: string): Promise<void> {
  const path = testInfo.outputPath(`${name}.png`);
  if (selector) await page.locator(selector).screenshot({ path });
  else await page.screenshot({ path });
  await testInfo.attach(name, { path, contentType: 'image/png' });
}

test('GPU geometry cut matches CPU reference and debug modes preserve selection', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 960, height: 640 });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const setup = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-lod.ts') as {
      startGeometryLodFixture: typeof startGeometryLodFixture;
    };
    return fixture.startGeometryLodFixture();
  });
  let material: Capture | undefined, clusters: Capture | undefined, levels: Capture | undefined;
  let fallback: Fallback | undefined;
  let cleanup: ReturnType<typeof disposeGeometryLodFixture> | undefined;
  try {
    for (const debug of ['material', 'clusters', 'levels'] as const) {
      const capture = await page.evaluate(async value => {
        const fixture = await import('/tests/browser/fixtures/' + 'geometry-lod.ts') as {
          captureGeometryLodFixtureDebug: typeof captureGeometryLodFixtureDebug;
        };
        return fixture.captureGeometryLodFixtureDebug(value);
      }, debug);
      if (debug === 'material') material = capture;
      if (debug === 'clusters') clusters = capture;
      if (debug === 'levels') levels = capture;
      await attachScreenshot(page, testInfo, `geometry-fixture-${debug}`, 'canvas[data-geometry-lod-fixture]');
    }
    fallback = await page.evaluate(async () => {
      const fixture = await import('/tests/browser/fixtures/' + 'geometry-lod.ts') as {
        exerciseGeometryLodCoverageFallback: typeof exerciseGeometryLodCoverageFallback;
      };
      return fixture.exerciseGeometryLodCoverageFallback();
    });
  } finally {
    cleanup = await page.evaluate(async () => {
      const fixture = await import('/tests/browser/fixtures/' + 'geometry-lod.ts') as {
        disposeGeometryLodFixture: typeof disposeGeometryLodFixture;
      };
      return fixture.disposeGeometryLodFixture();
    });
  }

  expect(setup.sourceTriangles).toBe(512);
  expect(setup.fine.probe.overflow).toBe(false);
  expect(setup.coarse.probe.overflow).toBe(false);
  expect(setup.fine.expected.indices).toBe(setup.sourceTriangles * 3);
  expect(setup.fine.probe.indices).toBe(setup.fine.expected.indices);
  expect(selected(setup.fine.probe)).toEqual([...setup.fine.expected.ids].sort((a, b) => a - b));
  expect(setup.fine.probe.selectedClusters.every(cluster => cluster.instanceIndex === 0 && cluster.level === 0)).toBe(true);
  expect(setup.intermediate.map(cut => cut.maxErrorPixels)).toEqual([0.25, 1, 4, 16]);
  let previousIndices = setup.fine.probe.indices;
  for (const cut of setup.intermediate) {
    expect(cut.probe.overflow).toBe(false);
    expect(cut.probe.indices).toBe(cut.expected.indices);
    expect(selected(cut.probe)).toEqual([...cut.expected.ids].sort((a, b) => a - b));
    expect(cut.probe.indices).toBeLessThanOrEqual(previousIndices);
    previousIndices = cut.probe.indices;
  }
  expect(setup.far.maxErrorPixels).toBe(1);
  expect(setup.far.viewEye).toEqual([1.4, 0.8, 12]);
  expect(setup.far.probe.overflow).toBe(false);
  expect(setup.far.probe.indices).toBe(setup.far.expected.indices);
  expect(selected(setup.far.probe)).toEqual([...setup.far.expected.ids].sort((a, b) => a - b));
  expect(setup.far.probe.indices).toBeLessThanOrEqual(setup.intermediate[1].probe.indices);
  expect(setup.coarse.probe.indices).toBe(setup.coarse.expected.indices);
  expect(setup.coarse.probe.indices).toBeLessThan(setup.fine.probe.indices);
  expect(selected(setup.coarse.probe)).toEqual([...setup.coarse.expected.ids].sort((a, b) => a - b));
  expect(selected(setup.coarse.probe)).toEqual([...setup.roots].sort((a, b) => a - b));
  expect(setup.coarse.probe.selectedClusters.every(cluster =>
    cluster.level === setup.levelsByCluster[cluster.clusterId])).toBe(true);
  expect(selected(setup.frozen)).toEqual(selected(setup.coarse.probe));
  expect(setup.culled.indices).toBe(0);
  expect(setup.culled.selectedClusters).toEqual([]);
  expect(setup.restored.indices).toBe(setup.fine.probe.indices);
  expect(selected(setup.restored)).toEqual(selected(setup.fine.probe));

  expect(material).toBeDefined(); expect(clusters).toBeDefined(); expect(levels).toBeDefined();
  const modes = [material!, clusters!, levels!];
  for (const capture of modes) {
    expect(capture.probe.overflow).toBe(false);
    expect(selected(capture.probe)).toEqual(selected(setup.fine.probe));
    expect(capture.hit.hits[0]).not.toBeNull();
    expect(capture.hit.hits[0]?.shading).toBe('unlit');
    for (const [index, expected] of [0.18, 0.34, 0.63].entries()) {
      expect(capture.hit.hits[0]?.baseColor[index]).toBeCloseTo(expected, 4);
    }
    expect(capture.errors).toEqual([]);
    if (capture.timing) expect(capture.timing.geometryLod).toBeGreaterThanOrEqual(0);
  }
  expect(difference(material!.hdr, clusters!.hdr)).toBeGreaterThan(0.02);
  expect(difference(material!.hdr, levels!.hdr)).toBeGreaterThan(0.02);
  expect(difference(clusters!.hdr, levels!.hdr)).toBeGreaterThan(0.02);
  expect(fallback).toBeDefined();
  expect(fallback!.masked.probe.indices).toBe(0);
  expect(fallback!.masked.probe.selectedClusters).toEqual([]);
  expect(fallback!.restored.probe.indices).toBe(setup.fine.probe.indices);
  expect(selected(fallback!.restored.probe)).toEqual(selected(setup.fine.probe));
  expect(difference(fallback!.masked.hdr, material!.hdr)).toBeLessThan(0.01);
  expect(difference(fallback!.restored.hdr, material!.hdr)).toBeLessThan(0.01);
  expect(fallback!.masked.hit.hits[0]?.shading).toBe('unlit');
  expect(fallback!.restored.hit.hits[0]?.shading).toBe('unlit');
  expect(fallback!.errors).toEqual([]);
  expect(cleanup).toEqual({ errors: [], bytesAfterDispose: 0 });
  expect(pageErrors).toEqual([]);
});

test('geometry demo captures normal and stable cluster-color views', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto('/geometry.html');
  await page.waitForFunction(() => window.__GEOMETRY_DEMO__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  try {
    for (const debug of ['material', 'clusters'] as const) {
      await page.evaluate(mode => {
        const demo = window.__GEOMETRY_DEMO__!;
        demo.pause(true);
        demo.setDebug(mode);
      }, debug);
      // The demo limits readbacks to one per 500 ms. Reapply the view if an
      // older probe was still pending, without moving the paused camera.
      let probed = false;
      for (let attempt = 0; attempt < 3 && !probed; attempt++) {
        await page.waitForTimeout(550);
        const frame = await page.evaluate(mode => {
          const demo = window.__GEOMETRY_DEMO__!;
          demo.setDebug(mode);
          return demo.snapshot().stats!.frames;
        }, debug);
        probed = await page.waitForFunction(requiredFrame =>
          (window.__GEOMETRY_DEMO__?.snapshot().gpuProbe?.frame ?? -1) >= requiredFrame,
        frame, { timeout: 2000 }).then(() => true, () => false);
      }
      expect(probed).toBe(true);
      const snapshot = await page.evaluate(() => window.__GEOMETRY_DEMO__!.snapshot());
      expect(snapshot.controls.paused).toBe(true);
      expect(snapshot.controls.debug).toBe(debug);
      expect(snapshot.source?.trianglesPerInstance).toBeGreaterThan(0);
      expect(snapshot.gpuProbe?.overflow).toBe(false);
      expect(snapshot.gpuProbe?.selectedClusters.length).toBeGreaterThan(0);
      expect(snapshot.errors).toEqual([]);
      await attachScreenshot(page, testInfo, `geometry-demo-${debug}`);
    }
  } finally {
    await page.evaluate(() => window.__GEOMETRY_DEMO__?.dispose());
  }
  expect(pageErrors).toEqual([]);
});
