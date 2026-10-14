import { expect, test } from '@playwright/test';
import type { runAntiAliasingFixture } from './fixtures/anti-aliasing';

test.use({ deviceScaleFactor: 1 });

for (const orientation of ['diagonal', 'near-vertical', 'near-horizontal'] as const) {
test(`bounded FXAA smooths a ${orientation} edge while preserving flat HDR output and legal borders`, async ({ page }) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async selectedOrientation => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'anti-aliasing.ts') as {
      runAntiAliasingFixture: typeof runAntiAliasingFixture;
    };
    return fixture.runAntiAliasingFixture(selectedOrientation);
  }, orientation);
  console.log('ANTI_ALIASING_GPU_DIAGNOSTIC ' + JSON.stringify(result));

  expect(result.orientation).toBe(orientation);
  expect(result.lowSize).toEqual({ width: 256, height: 256 });
  expect(result.highSize).toEqual({ width: 1024, height: 1024 });
  expect(result.finalSize).toEqual(result.lowSize);
  expect(result.edgeCount).toBeGreaterThan(100);
  expect(result.fxaaError).toBeLessThan(result.noneError * 0.75);
  expect(result.flatPreserved).toBe(true);
  expect(result.borderValid).toBe(true);
  expect(result.noneRoundtripSame).toBe(true);
  expect(result.fxaaAfterInvalidSame).toBe(true);
  expect(result.fxaaAfterResizeSame).toBe(true);
  expect(result.hdrFxaa).toEqual(result.hdrNone);
  expect(result.fxaaMode).toBe('fxaa');
  expect(result.modeAfterInvalid).toBe('fxaa');
  expect(result.invalidError).toContain('INVALID_ANTI_ALIASING');
  expect(result.highBytes - result.lowBytes).toBe((1024 * 1024 - 256 * 256) * 16);
  expect(result.finalBytes).toBe(result.lowBytes);
  expect(result.bytesAfterDispose).toBe(0);
  expect(result.frames).toBe(6);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
}
