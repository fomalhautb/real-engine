import { expect, test } from '@playwright/test';
import type { runGiReconstructionDiagnosticFixture } from './fixtures/gi-reconstruction-diagnostic';

test('GPU reconstruction diagnostic reports actual branches and leaves the rendered frame intact', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const fixture = await import('/tests/browser/fixtures/' + 'gi-reconstruction-diagnostic.ts') as {
      runGiReconstructionDiagnosticFixture: typeof runGiReconstructionDiagnosticFixture;
    };
    return fixture.runGiReconstructionDiagnosticFixture();
  });
  await testInfo.attach('gi-reconstruction-diagnostic.json', {
    body: JSON.stringify(result, null, 2), contentType: 'application/json',
  });
  expect(pageErrors).toEqual([]);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  const luminance = (rgb: readonly number[]) => rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
  for (const capture of [result.active, result.disabled]) {
    expect(capture.diagnostic.width).toBe(128);
    expect(capture.diagnostic.height).toBe(128);
    expect(capture.diagnostic.frame).toBe(capture.before.frames);
    expect(capture.diagnostic.points.map(point => point.pixel)).toEqual(result.pixels);
    expect(capture.diagnostic.temporaryBytes).toBeGreaterThan(0);
    expect(capture.diagnostic.temporaryBytes).toBeLessThan(64 * 1024 * 1024);
    expect(capture.after).toEqual(capture.before);
    expect(capture.hdrAfter).toEqual(capture.hdrBefore);
    for (const point of capture.diagnostic.points) {
      expect(point.candidateRejections).toHaveLength(16);
      expect(Number.isFinite(point.totalWeight)).toBe(true);
      expect(Number.isFinite(point.selectedLuminance)).toBe(true);
      expect(Number.isFinite(point.worldLuminance)).toBe(true);
    }
  }
  const active = result.active.diagnostic.points;
  expect(active[0].branch).toBe('primary');
  expect(active[1].branch).toBe('primary');
  expect(active[0].primaryAccepted).toBeGreaterThan(0);
  expect(active[1].primaryAccepted).toBeGreaterThan(0);
  expect(active[1].selectedLuminance, 'the black Mask cutout reveals the lit white wall').toBeGreaterThan(0.1);
  expect(luminance(result.active.hdrBefore[1])).toBeGreaterThan(0.01);
  expect(active[2].branch).toBe('unlit');
  expect(active[3].branch).toBe('background');
  expect(active[3].surface).toBeNull();
  for (const index of [0, 1]) {
    expect(active[index].surface?.batch).toBe('white-wall');
    expect(active[index].surface?.entityId).toBe(1);
    expect(Math.abs(active[index].surface!.position[2])).toBeLessThan(1e-6);
    expect(active[index].surface!.normal[2]).toBeGreaterThan(0.9999);
    expect(active[index].surface!.geometricNormal[2]).toBeGreaterThan(0.9999);
  }
  expect(active[2].surface?.batch).toBe('unlit-cover');
  expect(active[2].surface?.entityId).toBe(3);
  expect(active[2].surface!.position[2]).toBeCloseTo(0.04, 6);
  for (const index of [0, 1]) {
    // White nonmetal PBR, zero sun and emission: the actual HDR color is the
    // chosen irradiance times 0.96/pi, independent of the sky reference.
    const expected = luminance(result.active.hdrBefore[index]) * (Math.PI / 0.96);
    expect(Math.abs(active[index].selectedLuminance - expected)).toBeLessThan(0.002);
  }
  const disabled = result.disabled.diagnostic.points;
  expect(disabled[0].branch).toBe('disabled');
  expect(disabled[1].branch).toBe('disabled');
  expect(disabled[2].branch).toBe('unlit');
  expect(disabled[3].branch).toBe('background');
});
