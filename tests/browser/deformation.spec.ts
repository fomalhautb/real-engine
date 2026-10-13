import { expect, test } from '@playwright/test';
import type { runDeformationFixture } from './deformation-fixture';

test('compiled GPU graphs support six bones, ordered multi-Morph/wave and entity history', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('DEFORMATION_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    // Keep the path dynamic so Playwright's Node transform does not import WGSL.
    const module = await import('/tests/browser/' + 'deformation-fixture.ts') as { runDeformationFixture: typeof runDeformationFixture };
    return module.runDeformationFixture();
  });
  console.log('DEFORMATION_GPU_DIAGNOSTIC ' + JSON.stringify(result));
  for (const verification of [result.first, result.second, result.reordered]) {
    expect(verification.vertices).toBe(result.expectedVerticesPerVerification);
    expect(verification.maxError).toBeLessThan(0.0002);
    expect(verification.maxNormalError).toBeLessThan(0.0002);
    expect(verification.boundsViolations).toBe(0);
  }
  for (const distance of [...result.historyMovement, ...result.reorderedHistoryMovement]) expect(distance).toBeGreaterThan(0.1);
  expect(result.historyMaxError, 'GPU previous position must match the retained prior state of the same entity').toBeLessThan(0.0002);
  expect(result.currentMaxError, 'GPU current position must match the independently retained submitted state').toBeLessThan(0.0002);
  expect(result.orderDifference).toBeGreaterThan(0.05);
  expect(result.diagnostics.deformationVariants).toBe(2);
  expect(result.diagnostics.bounds).toHaveLength(4);
  expect(result.frames).toBe(3);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
