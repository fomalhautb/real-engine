import { expect, test } from '@playwright/test';
import type { runFinalGatherEmissiveFixture } from './fixtures/gi-final-gather-emissive';

/** Independently integrate the rectangle's solid angle and Lambertian response
 * at the actual center-pixel floor hit, using the renderer's public camera FOV. */
function referenceRadiance(result: Awaited<ReturnType<typeof runFinalGatherEmissiveFixture>>): number[] {
  const forward = [0, -3 / Math.sqrt(13), -2 / Math.sqrt(13)];
  const up = [0, 2 / Math.sqrt(13), -3 / Math.sqrt(13)];
  const slope = Math.tan(Math.PI / 6.4) / 256;
  const direction = [forward[0] + slope, forward[1] - slope * up[1],
    forward[2] - slope * up[2]];
  const travel = -result.eye[1] / direction[1];
  const floorX = result.eye[0] + travel * direction[0];
  const floorZ = result.eye[2] + travel * direction[2];
  const side = 128;
  const cellArea = result.size[0] * result.size[1] / (side * side);
  let irradianceFactor = 0;
  for (let row = 0; row < side; row++) for (let column = 0; column < side; column++) {
    const x = ((column + 0.5) / side - 0.5) * result.size[0];
    const z = ((row + 0.5) / side - 0.5) * result.size[1];
    const distanceSquared = (x - floorX) ** 2 + 1 + (z - floorZ) ** 2;
    irradianceFactor += cellArea / (distanceSquared * distanceSquared);
  }
  return result.emission.map((le, channel) =>
    le * irradianceFactor * result.floorBase[channel] * 0.96 / Math.PI);
}

test('final gather integrates small emissive area once and respects shared coverage and occlusion', async ({ page }) => {
  test.setTimeout(360_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'gi-final-gather-emissive.ts') as {
      runFinalGatherEmissiveFixture: typeof runFinalGatherEmissiveFixture;
    };
    return module.runFinalGatherEmissiveFixture();
  });
  const reference = referenceRadiance(result);
  console.log('FINAL_GATHER_EMISSIVE ' + JSON.stringify({ reference, initial: result.initial,
    reflective: result.reflective, masked: result.masked, blocked: result.blocked,
    restored: result.restored, errors: result.errors, afterDispose: result.afterDispose }));
  expect(pageErrors).toEqual([]);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(result.diagnostics?.enabled).toBe(true);
  for (let channel = 0; channel < 3; channel++) {
    expect(reference[channel]).toBeGreaterThan(0);
    for (const sample of [result.initial, result.reflective, result.restored]) {
      expect(Math.abs(sample[channel] - reference[channel]) / reference[channel]).toBeLessThan(0.03);
    }
    expect(result.masked[channel]).toBeLessThan(1e-7);
    expect(result.blocked[channel]).toBeLessThan(1e-7);
  }
});
