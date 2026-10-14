import { expect, test, type Page } from '@playwright/test';
import { evaluateGiIrradiance } from '../../src/diffuse-gi';
import type { runDiffuseGiSkyFixture, runDiffuseGiSceneFixture,
  runDiffuseGiOverflowFixture } from './fixtures/diffuse-gi';

type SkyResult = Awaited<ReturnType<typeof runDiffuseGiSkyFixture>>;
type SceneResult = Awaited<ReturnType<typeof runDiffuseGiSceneFixture>>;
type OverflowResult = Awaited<ReturnType<typeof runDiffuseGiOverflowFixture>>;
type ProbeSet = SkyResult['first'];

async function fixturePage(page: Page): Promise<string[]> {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return pageErrors;
}

function near(actual: readonly number[], expected: readonly number[], tolerance: number): void {
  expect(actual).toHaveLength(expected.length);
  for (let channel = 0; channel < expected.length; channel++) {
    expect(Math.abs(actual[channel] - expected[channel]), `channel ${channel}`).toBeLessThan(tolerance);
  }
}

function averageIrradiance(result: ProbeSet): [number, number, number] {
  const total: [number, number, number] = [0, 0, 0];
  for (const probe of result.probes) {
    expect(probe.valid).toBe(true);
    const values = evaluateGiIrradiance(probe.coefficients.flat(), [0, 0, 1]);
    for (let channel = 0; channel < 3; channel++) total[channel] += values[channel];
  }
  return total.map(value => value / result.probes.length) as [number, number, number];
}

function maxCoefficientDifference(a: ProbeSet, b: ProbeSet): number {
  expect(a.probes.length).toBe(b.probes.length);
  let largest = 0;
  for (let i = 0; i < a.probes.length; i++) {
    const left = a.probes[i].coefficients.flat(), right = b.probes[i].coefficients.flat();
    for (let j = 0; j < left.length; j++) largest = Math.max(largest, Math.abs(left[j] - right[j]));
  }
  return largest;
}

test('empty-world GI probes recover constant sky irradiance and honor bounded lifecycle', async ({ page }) => {
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'diffuse-gi.ts') as {
      runDiffuseGiSkyFixture: typeof runDiffuseGiSkyFixture;
    };
    return module.runDiffuseGiSkyFixture();
  }) as SkyResult;
  console.log('GI_SKY_GPU_DIAGNOSTIC ' + JSON.stringify({ frame: result.first.frame,
    valid: result.first.probes.length - result.first.invalidProbes, errors: result.errors }));

  expect(result.beforeFrame).toContain('DIFFUSE_GI_NO_FRAME');
  expect(result.first.probes).toHaveLength(8);
  expect(result.first.invalidProbes).toBe(0);
  for (const probe of result.first.probes) {
    expect(probe.valid).toBe(true);
    expect(probe.primaryOverflows).toBe(0);
    expect(probe.shadowOverflows).toBe(0);
    // Independent constant-environment integral: E = pi * incident radiance.
    for (const normal of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const) {
      near(evaluateGiIrradiance(probe.coefficients.flat(), normal),
        [0.1 * Math.PI, 0.2 * Math.PI, 0.3 * Math.PI], 0.025);
    }
    for (const face of probe.moments) {
      expect(face[0]).toBeCloseTo(12, 4);
      expect(face[1]).toBeCloseTo(144, 3);
      expect(face[2]).toBeGreaterThan(0);
    }
  }
  expect(result.initial?.enabled).toBe(true);
  expect(result.disabled?.enabled).toBe(false);
  expect(result.restored.frame).toBeGreaterThan(result.first.frame);
  expect(maxCoefficientDifference(result.first, result.restored)).toBeLessThan(0.001);
  expect(result.overRayBudget).toContain('DIFFUSE_GI_QUERY_RAY_BUDGET');
  expect(result.overWorkBudget).toContain('DIFFUSE_GI_QUERY_WORK_BUDGET');
  expect(result.overMemoryBudget).toContain('DIFFUSE_GI_MEMORY_BUDGET');
  expect(result.frames).toBe(3);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('offscreen PBR wall, sun visibility, mask and updates alter real GPU probe transport', async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'diffuse-gi.ts') as {
      runDiffuseGiSceneFixture: typeof runDiffuseGiSceneFixture;
    };
    return module.runDiffuseGiSceneFixture();
  }) as SceneResult;

  const initial = averageIrradiance(result.initial);
  const cameraMoved = averageIrradiance(result.cameraMoved);
  const shadowed = averageIrradiance(result.shadowed);
  const green = averageIrradiance(result.green);
  const noSun = averageIrradiance(result.noSun);
  const masked = averageIrradiance(result.masked);
  const removed = averageIrradiance(result.removed);
  console.log('GI_SCENE_GPU_DIAGNOSTIC ' + JSON.stringify({ initial, cameraMoved,
    shadowed, green, noSun, masked, removed, hdrOn: result.hdrOn, hdrOff: result.hdrOff,
    errors: result.errors }));

  for (const snapshot of [result.initial, result.cameraMoved, result.shadowed, result.green,
    result.noSun, result.masked, result.removed]) {
    expect(snapshot.invalidProbes).toBe(0);
    expect(snapshot.probes).toHaveLength(8);
    expect(snapshot.probes.every(probe => probe.primaryOverflows === 0 && probe.shadowOverflows === 0)).toBe(true);
  }
  expect(initial[0]).toBeGreaterThan(0.1);
  expect(initial[0]).toBeGreaterThan(initial[1] * 3);
  // Camera direction does not enter fixed world-space probe capture.
  near(cameraMoved, initial, 0.001);
  expect(maxCoefficientDifference(result.initial, result.cameraMoved)).toBeLessThan(0.001);
  // The one-sided blocker is invisible to primary rays; only the second,
  // wall-to-sun world query removes this bounce.
  expect(shadowed[0]).toBeLessThan(initial[0] * 0.2);
  expect(green[1]).toBeGreaterThan(green[0] * 3);
  expect(green[1]).toBeGreaterThan(0.1);
  expect(noSun[1]).toBeLessThan(green[1] * 0.2);
  expect(masked[0]).toBeLessThan(initial[0] * 0.95);
  expect(masked[0]).toBeGreaterThan(0);
  near(removed, [0, 0, 0], 0.01);
  expect(result.hdrOn[0]).toBeGreaterThan(result.hdrOff[0] + 0.02);
  expect(result.hdrOff[0]).toBeLessThan(0.02);
  expect(result.frames).toBe(8);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('query work overflow invalidates GI probes instead of publishing partial cache', async ({ page }) => {
  const pageErrors = await fixturePage(page);
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'diffuse-gi.ts') as {
      runDiffuseGiOverflowFixture: typeof runDiffuseGiOverflowFixture;
    };
    return module.runDiffuseGiOverflowFixture();
  }) as OverflowResult;
  expect(result.result.probes).toHaveLength(8);
  expect(result.result.invalidProbes).toBe(8);
  expect(result.result.probes.every(probe => !probe.valid && probe.primaryOverflows > 0)).toBe(true);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('GI demonstration captures comparable enabled and disabled frames', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.setViewportSize({ width: 1024, height: 768 });
  try {
    await page.goto('/gi.html');
    await page.waitForFunction(() => window.__GI_DEMO__?.renderer && window.__GI_DEMO__.renderer.stats.frames > 0
      || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    const before = await page.evaluate(() => {
      window.__GI_DEMO__!.pause(true);
      return window.__GI_DEMO__!.snapshot();
    });
    // Save explicit output files: attachments supplied only as buffers are not
    // retained by the default line reporter's successful-test artifact path.
    const capture = async (name: string) => {
      // Complete the GPU work before asking the compositor for its image.
      await page.evaluate(async () => {
        const renderer = window.__GI_DEMO__!.renderer!;
        await renderer.probeHdr([{ x: Math.floor(renderer.stats.width / 2), y: Math.floor(renderer.stats.height / 2) }]);
      });
      const path = testInfo.outputPath(name + '.png');
      await page.screenshot({ path });
      await testInfo.attach(name, { path, contentType: 'image/png' });
    };
    await capture('diffuse-gi-enabled-ui');
    await page.keyboard.press('h');
    await capture('diffuse-gi-enabled');
    const disabled = await page.evaluate(() => {
      window.__GI_DEMO__!.toggleGi(false);
      return window.__GI_DEMO__!.snapshot();
    });
    await capture('diffuse-gi-disabled');
    expect(before.controls.giEnabled).toBe(true);
    expect(disabled.controls.giEnabled).toBe(false);
    expect(disabled.camera).toEqual(before.camera);
    expect(disabled.stats!.frames).toBeGreaterThan(before.stats!.frames);
    expect(disabled.errors).toEqual([]);
    expect(pageErrors).toEqual([]);
  } finally {
    await page.evaluate(() => window.__GI_DEMO__?.dispose()).catch(() => {});
  }
});
