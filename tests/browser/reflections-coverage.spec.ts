import { writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import type { runReflectionsCoverageFixture } from './fixtures/reflections-coverage';

test.use({ deviceScaleFactor: 1 });

const luminance = (rgb: readonly number[]) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

test('curved and thin mirror receivers keep reconstructed reflections at coarser anchor spacing', async ({ page }, testInfo) => {
  test.setTimeout(300_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-coverage.ts') as {
      runReflectionsCoverageFixture: typeof runReflectionsCoverageFixture;
    };
    return fixture.runReflectionsCoverageFixture();
  });
  const reference = result.runs[0];
  expect(reference.spacing).toBe(1);
  const objects = [{ name: 'sphere', id: result.ids.sphere }, ...result.ids.rods.map((id, index) => ({
    name: `rod ${(result.rodHalfWidths[index] * 2).toFixed(3)} m`, id }))];
  // Under a uniform sky of radiance 1 a white metal mirror reflects about 1;
  // a pixel without any reconstructed anchor keeps only the dim ambient fill.
  const report = result.runs.map(run => ({ spacing: run.spacing, objects: objects.map(object => {
    const pixels = result.mask.map((entity, i) => ({ entity, i })).filter(item => item.entity === object.id
      && luminance(reference.hdr[item.i]) > 0.5);
    const ratios = pixels.map(item => luminance(run.hdr[item.i]) / luminance(reference.hdr[item.i]));
    return { name: object.name, pixels: pixels.length,
      omitted: ratios.filter(ratio => ratio < 0.5).length / Math.max(pixels.length, 1),
      meanAbsoluteRatioError: ratios.reduce((sum, ratio) => sum + Math.abs(ratio - 1), 0) / Math.max(ratios.length, 1) };
  }) }));
  const receivers = result.mask.filter(entity => entity !== 0).length;
  const lit = result.mask.filter((entity, i) => entity !== 0 && luminance(reference.hdr[i]) > 0.5).length;
  await testInfo.attach('reflection-coverage.json', { body: writeReport(testInfo.outputPath('reflection-coverage.json'), { receivers, lit, report,
    grids: result.runs.map(run => run.grid) }), contentType: 'application/json' });
  for (const run of result.runs) {
    expect(run.errors).toEqual([]);
    expect(run.ownedBytesAfterDispose).toBe(0);
  }
  // Every pixel anchor reconstructs itself: nearly all receivers see the bright sky.
  expect(lit / receivers).toBeGreaterThan(0.95);
  for (const object of report[0].objects) expect(object.pixels, object.name).toBeGreaterThan(0);
  for (const run of report.slice(1)) {
    const sphere = run.objects[0];
    expect(sphere.omitted, `sphere omitted at spacing ${run.spacing}`).toBeLessThanOrEqual(0.01);
    expect(sphere.meanAbsoluteRatioError, `sphere error at spacing ${run.spacing}`).toBeLessThan(0.05);
  }
  expect(pageErrors).toEqual([]);
});

function writeReport(path: string, value: unknown): string {
  const body = JSON.stringify(value, null, 2);
  writeFileSync(path, body);
  return body;
}
