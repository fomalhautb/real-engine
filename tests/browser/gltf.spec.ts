import { expect, test } from '@playwright/test';
import type { runGltfFixture } from './fixtures/gltf';

type Result = Awaited<ReturnType<typeof runGltfFixture>>;
type Verification = Result['first'];

function sample(verification: Verification, batch: string, entityId: number, vertexIndex: number) {
  const matches = verification.samples.filter(value => value.batch === batch && value.entityId === entityId && value.vertexIndex === vertexIndex);
  expect(matches, `one sample for ${batch} entity ${entityId} vertex ${vertexIndex}`).toHaveLength(1);
  return matches[0];
}

function close(actual: readonly number[], expected: readonly number[], tolerance = 0.0002): void {
  expect(actual).toHaveLength(expected.length);
  for (let axis = 0; axis < expected.length; axis++) expect(Math.abs(actual[axis] - expected[axis])).toBeLessThan(tolerance);
}

test('glTF skin and Morph clips retain independent GPU history with affine normals', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('GLTF_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    // Keep this import dynamic so Playwright's Node transform never imports WGSL.
    const module = await import('/tests/browser/fixtures/' + 'gltf.ts') as { runGltfFixture: typeof runGltfFixture };
    return module.runGltfFixture();
  });
  console.log('GLTF_GPU_DIAGNOSTIC ' + JSON.stringify({
    first: { vertices: result.first.vertices, maxError: result.first.maxError, maxNormalError: result.first.maxNormalError,
      boundsViolations: result.first.boundsViolations },
    second: { vertices: result.second.vertices, maxError: result.second.maxError, maxNormalError: result.second.maxNormalError,
      boundsViolations: result.second.boundsViolations },
    frames: result.frames, errors: result.errors,
  }));
  for (const verification of [result.first, result.second]) {
    expect(verification.vertices).toBe(result.expectedVertices);
    expect(verification.maxError).toBeLessThan(0.0002);
    expect(verification.maxNormalError).toBeLessThan(0.0002);
    expect(verification.boundsViolations).toBe(0);
  }

  // The selected source vertex is [0.5, 2, 0] with full weight on the animated joint.
  // At source t=1 it rotates about the y=1 pivot to approximately [-1, 1.5, 0].
  const leftFirst = sample(result.first, result.names.firstSkin, 701, result.topVertex);
  const rightFirst = sample(result.first, result.names.secondSkin, 702, result.topVertex);
  const leftSecond = sample(result.second, result.names.firstSkin, 701, result.topVertex);
  const rightSecond = sample(result.second, result.names.secondSkin, 702, result.topVertex);
  close(leftFirst.current, [-3.5, 2, 0], 0.002);
  close(leftFirst.previous, [-3.5, 2, 0], 0.002);
  close(rightFirst.current, [3, 1.5, 0], 0.002);
  close(rightFirst.previous, [3, 1.5, 0], 0.002);
  close(leftSecond.current, [-5, 1.5, 0], 0.002);
  close(leftSecond.previous, leftFirst.current);
  close(rightSecond.current, [4.5, 2, 0], 0.002);
  close(rightSecond.previous, rightFirst.current);

  // Morph oracle directly combines imported source position and both target deltas.
  const morphFirst = sample(result.first, result.names.morph, 703, result.morphVertex);
  const morphSecond = sample(result.second, result.names.morph, 703, result.morphVertex);
  expect(result.greatestMotion).toBeGreaterThan(0.1);
  close(morphFirst.current, result.morphFirst);
  close(morphFirst.previous, result.morphFirst);
  close(morphSecond.current, result.morphSecond);
  close(morphSecond.previous, result.morphFirst);

  // The equal blend of two sheared/nonuniform bone matrices maps a source Z
  // normal to the analytic direction [-1, 0, 6] (up to source winding).
  close(result.affineCpuNormal, result.affineExpectedNormal);
  const affineFirst = sample(result.first, result.names.affine, 704, result.blendVertex);
  const affineSecond = sample(result.second, result.names.affine, 704, result.blendVertex);
  close(affineFirst.current, [0.75, 2.5, -4.875]);
  close(affineSecond.current, affineFirst.current);
  close(affineSecond.previous, affineFirst.current);

  expect(result.frames).toBe(2);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
