import { expect, test } from '@playwright/test';
import type { runHdrFixture } from './fixtures/hdr';

function aces(value: number): number {
  return Math.max(0, Math.min(1, value * (2.51 * value + 0.03) / (value * (2.43 * value + 0.59) + 0.14)));
}
function linearToSrgb(value: number): number {
  return value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
}
function near(actual: readonly number[], expected: readonly number[], tolerance: number): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i] - expected[i]), `channel ${i}`).toBeLessThan(tolerance);
  }
}

test('linear HDR radiance survives exposure changes and resolves to sRGB', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('HDR_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'hdr.ts') as { runHdrFixture: typeof runHdrFixture };
    return module.runHdrFixture();
  });
  console.log('HDR_GPU_DIAGNOSTIC ' + JSON.stringify({
    hdr: [result.firstHdr, result.secondHdr, result.resizedHdr],
    output: [result.firstOutput, result.secondOutput, result.resizedOutput],
    sizes: [result.firstSize, result.resizedSize], bytes: [result.firstBytes, result.resizedBytes],
    errors: result.errors,
  }));

  const radiance = [5, 0.75, 0.125, 1];
  for (const values of [result.firstHdr, result.secondHdr, result.resizedHdr]) {
    expect(values).toHaveLength(1);
    near(values[0], radiance, 0.004);
  }
  const resolved = (ev: number) => [...radiance.slice(0, 3).map(value => linearToSrgb(aces(value * 2 ** ev))), 1];
  near(result.firstOutput, resolved(0), 0.015);
  near(result.secondOutput, resolved(-2), 0.015);
  near(result.resizedOutput, resolved(-2), 0.015);
  expect(result.firstOutput[1] - result.secondOutput[1]).toBeGreaterThan(0.25);
  expect(result.velocityAfterExposure).toHaveLength(1);
  expect(result.velocityAfterExposure[0][0]).toBeGreaterThan(0.01);
  expect(Math.abs(result.velocityAfterExposure[0][1])).toBeLessThan(0.001);
  expect(result.secondMaterialVariants).toBe(result.materialVariants);

  expect(result.noFrame).not.toBe('');
  expect(result.invalidPixel).not.toBe('');
  expect(result.tooManyPixels).not.toBe('');
  for (const rejection of result.invalidExposures) expect(rejection).not.toBe('');
  expect(result.resizedSize.width).toBeGreaterThan(result.firstSize.width);
  expect(result.resizedSize.height).toBe(result.firstSize.height);
  expect(result.resizedBytes - result.firstBytes).toBe(
    (result.resizedSize.width * result.resizedSize.height - result.firstSize.width * result.firstSize.height) * 16);
  expect(result.frames).toBe(3);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
