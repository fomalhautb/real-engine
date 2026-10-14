import { writeFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type { runShadowPrecisionFixture } from './fixtures/shadow-precision';

type Result = Awaited<ReturnType<typeof runShadowPrecisionFixture>>;
type Mode = Result['modes'][number];

test.use({ deviceScaleFactor: 1 });

function visibility(mode: Mode, index: number): number {
  const ambient = mode.ambient[index];
  const lit = mode.lit[index];
  const blocked = mode.blocked[index];
  const direct = [0, 1, 2].map(channel => lit[channel] - ambient[channel]);
  const meanDirect = direct.reduce((a, b) => a + b, 0) / 3;
  expect(meanDirect, `unoccluded direct light at sample ${index}`).toBeGreaterThan(0.12);
  return [0, 1, 2].reduce((sum, channel) =>
    sum + (blocked[channel] - ambient[channel]) / direct[channel], 0) / 3;
}

test('focused sun map follows an analytic diagonal edge more closely while retaining contact', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'shadow-precision.ts') as {
      runShadowPrecisionFixture: typeof runShadowPrecisionFixture;
    };
    return fixture.runShadowPrecisionFixture();
  }) as Result;

  expect(result.modes.map(mode => mode.mode)).toEqual(['default', 'focused']);
  expect(result.points.length).toBeGreaterThan(680);
  const reports = result.modes.map(mode => {
    const values = result.points.map((_, index) => visibility(mode, index));
    const edge = values.flatMap((value, index) => {
      const geometry = result.geometry[index];
      if (result.tags[index] !== 'edge' || Math.abs(geometry.u) > 1.25 ||
          Math.abs(geometry.edgeDistance) < 0.01 || Math.abs(geometry.edgeDistance) > 0.13) return [];
      return [{ value, ideal: geometry.occluded ? 0 : 1,
        error: Math.abs(value - (geometry.occluded ? 0 : 1)),
        edgeDistance: geometry.edgeDistance }];
    });
    const meanError = edge.reduce((sum, sample) => sum + sample.error, 0) / edge.length;
    const controls = Object.fromEntries(['deep-shadow', 'clear-light', 'contact', 'contact-clear']
      .map(tag => [tag, values.filter((_, index) => result.tags[index] === tag)]));
    return { mode: mode.mode, meanError, edgeCount: edge.length, controls,
      diagnostic: mode.diagnostics, focusChange: mode.focusChange,
      bytesBeforeFocusChange: mode.bytesBeforeFocusChange,
      ownedGpuBytesAfterDispose: mode.ownedGpuBytesAfterDispose,
      errors: mode.errors, edge };
  });

  for (const mode of result.modes) {
    const path = testInfo.outputPath(`shadow-${mode.mode}.png`);
    await writeFile(path, Buffer.from(mode.png.slice(mode.png.indexOf(',') + 1), 'base64'));
    await testInfo.attach(`shadow-${mode.mode}`, { path, contentType: 'image/png' });
  }
  const jsonPath = testInfo.outputPath('shadow-precision.json');
  await writeFile(jsonPath, JSON.stringify({ size: result.size, points: result.points,
    tags: result.tags, geometry: result.geometry, contactChecks: result.contactChecks, reports }, null, 2));
  await testInfo.attach('shadow-precision-measurements', { path: jsonPath, contentType: 'application/json' });
  console.log('SHADOW_PRECISION ' + JSON.stringify(reports.map(report => ({
    mode: report.mode, meanError: report.meanError, edgeCount: report.edgeCount,
    controls: Object.fromEntries(Object.entries(report.controls).map(([key, value]) =>
      [key, { min: Math.min(...value), max: Math.max(...value) }])),
    diagnostic: report.diagnostic,
  }))));

  const [coarse, refined] = reports;
  expect(result.contactChecks.map(check => [check.cameraHitsCube, check.sunHitsCube, check.sunHitsSlab]))
    .toEqual([[false, true, false], [false, false, false]]);
  expect(coarse.edgeCount).toBeGreaterThan(500);
  expect(refined.edgeCount).toBe(coarse.edgeCount);
  expect(coarse.meanError).toBeGreaterThan(0.025);
  expect(refined.meanError, `refined ${refined.meanError}, coarse ${coarse.meanError}`)
    .toBeLessThan(coarse.meanError * 0.2);
  expect(refined.meanError).toBeLessThan(0.02);
  expect(coarse.diagnostic.resolution).toBe(1024);
  expect(coarse.diagnostic.projection).toBe('legacy-fixed');
  expect(refined.diagnostic.resolution).toBe(2048);
  expect(refined.diagnostic.projection).toBe('world-bounds-snapped');
  expect(refined.diagnostic.texelWorldSize).toBeLessThan(coarse.diagnostic.texelWorldSize * 0.2);
  expect(refined.focusChange).not.toBeNull();
  expect(refined.focusChange!.changed.width).not.toBe(refined.focusChange!.original.width);
  expect(refined.focusChange!.bytesAfter).toBe(refined.bytesBeforeFocusChange);
  expect(refined.focusChange!.invalidMessage).toContain('INVALID_SUN_SHADOW');
  expect(refined.focusChange!.afterInvalid).toEqual(refined.focusChange!.changed);

  for (const report of reports) {
    expect(report.controls['deep-shadow'].length).toBeGreaterThan(10);
    expect(Math.max(...report.controls['deep-shadow'])).toBeLessThan(0.22);
    expect(Math.min(...report.controls['clear-light'])).toBeGreaterThan(0.85);
    expect(report.controls['contact']).toHaveLength(1);
    expect(report.controls['contact'][0]).toBeLessThan(0.4);
    expect(report.controls['contact-clear'][0]).toBeGreaterThan(0.85);
    expect(report.errors).toEqual([]);
    expect(report.ownedGpuBytesAfterDispose).toBe(0);
  }
  expect(pageErrors).toEqual([]);
});
