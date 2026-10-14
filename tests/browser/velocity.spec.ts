import { expect, test } from '@playwright/test';
import type { runVelocityFixture } from './fixtures/velocity';

type Result = Awaited<ReturnType<typeof runVelocityFixture>>;
type Capture = Result['first'];

function near(actual: readonly number[], expected: readonly number[], tolerance = 0.0001): void {
  expect(actual).toHaveLength(2);
  expect(expected).toHaveLength(2);
  for (let axis = 0; axis < 2; axis++) expect(Math.abs(actual[axis] - expected[axis])).toBeLessThan(tolerance);
}

function verify(capture: Capture): void {
  expect(capture.values).toHaveLength(6);
  for (let i = 0; i < 4; i++) near(capture.values[i], capture.expected[i]);
  near(capture.values[4], [0, 0]); // Masked alpha hole does not write velocity.
  near(capture.values[5], [0, 0]); // Uncovered background is attachment clear color.
}

test('velocity attachment tracks camera, object, skin, Morph, cuts, and resize', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('VELOCITY_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    // Keep the fixture import in the browser; Playwright's Node transform cannot import WGSL.
    const module = await import('/tests/browser/fixtures/' + 'velocity.ts') as { runVelocityFixture: typeof runVelocityFixture };
    return module.runVelocityFixture();
  });
  console.log('VELOCITY_GPU_DIAGNOSTIC ' + JSON.stringify({
    camera: result.camera.values, moved: result.moved.values, cut: result.cut.values,
    resized: { width: result.resized.width, height: result.resized.height, values: result.resized.values },
    frames: result.frames, errors: result.errors,
  }));
  expect(result.noFrameRejection).toContain('NO_RENDERED_VELOCITY');
  expect(result.invalidProbeRejection).toContain('INVALID_VELOCITY_PROBES');
  for (const capture of [result.first, result.camera, result.moved, result.settled,
    result.entityReset, result.cut, result.resized]) verify(capture);

  expect(result.camera.values[0][0]).toBeLessThan(-0.005); // Camera moves right; static world shifts left.
  expect(result.moved.values[0][0]).toBeCloseTo(0, 4); // Camera has stopped.
  expect(result.moved.values[1][0]).toBeGreaterThan(0.005);
  expect(result.moved.values[1][1]).toBeGreaterThan(0.005); // Object moved down in screen UV.
  expect(result.moved.values[2][0]).toBeGreaterThan(0.005);
  expect(result.moved.values[2][1]).toBeLessThan(-0.003); // Skin moved up in screen UV.
  expect(result.moved.values[3][0]).toBeGreaterThan(0.005);
  expect(result.moved.values[3][1]).toBeGreaterThan(0.003); // Morph moved down.
  expect(result.resized.width).not.toBe(result.cut.width);
  expect(result.resized.height).toBe(result.cut.height);
  expect(result.frames).toBe(7);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
