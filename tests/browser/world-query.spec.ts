import { expect, test } from '@playwright/test';
import type { runWorldQueryFixture } from './fixtures/world-query';

type Result = Awaited<ReturnType<typeof runWorldQueryFixture>>;
type Hit = NonNullable<Result['initial']['hits'][number]>;

function hit(value: Hit | null): Hit {
  expect(value).not.toBeNull();
  if (!value) throw new Error('Expected a world-space ray hit');
  return value;
}
function near(actual: readonly number[], expected: readonly number[], tolerance = 0.0002): void {
  expect(actual).toHaveLength(expected.length);
  for (let axis = 0; axis < expected.length; axis++) expect(Math.abs(actual[axis] - expected[axis])).toBeLessThan(tolerance);
}

for (const instanceBvh of [false, true]) test(`GPU world queries follow shared deformation and stable scene identities offscreen (${instanceBvh ? 'instance BVH' : 'small batch'})`, async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('WORLD_QUERY_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async (instanceBvh) => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    // Playwright's Node transform must not try to import WGSL from the renderer.
    const module = await import('/tests/browser/fixtures/' + 'world-query.ts') as { runWorldQueryFixture: typeof runWorldQueryFixture };
    return module.runWorldQueryFixture(instanceBvh);
  }, instanceBvh);
  console.log('WORLD_QUERY_GPU_DIAGNOSTIC ' + JSON.stringify({
    frames: result.frames, initialHits: result.initial.hits.map((value: Hit | null) => value?.batch ?? null),
    movedHits: result.moved.hits.map((value: Hit | null) => value?.batch ?? null),
    errors: result.errors, topologyRejected: result.topologyRejected,
  }));

  expect(result.initial.hits).toHaveLength(13);
  const offscreen = hit(result.initial.hits[0]);
  expect(offscreen.batch).toBe('query-offscreen-static');
  expect(offscreen.entityId).toBe(100);
  expect(offscreen.entityGeneration).toBe(1);
  expect(offscreen.topologyVersion).toBe(1);
  expect(offscreen.historyVersion).toBe(0);
  expect(offscreen.triangleIndex).toBeGreaterThanOrEqual(0);
  expect(offscreen.triangleIndex).toBeLessThan(2);
  expect(offscreen.distance).toBeCloseTo(5, 4);
  near(offscreen.position, [-5, 0, 0]);
  near(offscreen.previousPosition, [-5, 0, 0]);
  near(offscreen.normal, [0, 0, 1]);
  near(offscreen.uv, [0.5, 0.5]);
  near(offscreen.baseColor, [0.2, 0.4, 0.6]);
  expect(offscreen.opacity).toBeCloseTo(1, 5);
  expect(offscreen.metallic).toBeCloseTo(0.25, 5);
  expect(offscreen.roughness).toBeCloseTo(0.7, 5);
  near(offscreen.emissive, [0, 0, 0]);
  expect(offscreen.shading).toBe('pbr');
  expect(hit(result.initial.hits[1]).entityId).toBe(200);
  expect(hit(result.initial.hits[2]).entityId).toBe(300);
  expect(hit(result.initial.hits[3]).entityId).toBe(301);
  near(hit(result.initial.hits[3]).position, [2.5, 1, 0]);

  // The front plane has a transparent center texel and opaque border. Rejected
  // Masked candidates must let the farther plane win the same world ray.
  const throughHole = hit(result.initial.hits[4]);
  expect(throughHole.batch).toBe('query-behind-mask');
  expect(throughHole.entityId).toBe(400);
  expect(throughHole.distance).toBeCloseTo(5, 4);
  const opaqueBorder = hit(result.initial.hits[5]);
  expect(opaqueBorder.batch).toBe('query-mask-front');
  expect(opaqueBorder.entityId).toBe(401);
  expect(opaqueBorder.distance).toBeCloseTo(4, 4);
  near(opaqueBorder.uv, [0.32 / 0.9 + 0.5, 0.32 / 0.9 + 0.5], 0.002);

  expect(result.initial.hits[6]).toBeNull(); // Back face of a single-sided material.
  expect(hit(result.initial.hits[7]).batch).toBe('query-double-sided');
  expect(result.initial.hits[8]).toBeNull(); // tMin beyond the plane.
  expect(result.initial.hits[9]).toBeNull(); // tMax before the plane.
  expect(result.initial.hits[10]).toBeNull(); // Exact entity generation ignored.
  expect(hit(result.initial.hits[11]).entityId).toBe(100); // Wrong generation is not ignored.
  expect(result.initial.hits[12]).toBeNull(); // Empty world space.
  expect(result.empty.hits).toEqual([]);
  expect(hit(result.normalizedDirection.hits[0]).distance).toBeCloseTo(5, 4);
  expect(result.invalidDirection).not.toBe('');
  expect(result.invalidInterval).not.toBe('');
  expect(result.overBudget).not.toBe('');

  expect(result.beforeRender.frame).toBe(result.initial.frame);
  expect(hit(result.beforeRender.hits[0]).entityId).toBe(200);
  expect(result.beforeRender.hits[1]).toBeNull();
  expect(result.moved.frame).toBeGreaterThan(result.initial.frame);
  expect(hit(result.moved.hits[0]).entityId).toBe(100); // Camera points elsewhere; ray still sees the object.
  const movedBody = hit(result.moved.hits[1]);
  expect(movedBody.entityId).toBe(200);
  near(movedBody.position, [-1, 0, 0]);
  near(movedBody.previousPosition, [-2, 0, 0]);
  expect(result.moved.hits[2]).toBeNull();
  const deformed = hit(result.moved.hits[3]);
  expect(deformed.entityId).toBe(300);
  near(deformed.position, [0.25, 0.2, 0]);
  near(deformed.previousPosition, [0, 0, 0]);
  expect(hit(result.moved.hits[4]).entityId).toBe(301);

  expect(hit(result.reordered.hits[0]).entityId).toBe(301);
  expect(hit(result.reordered.hits[1]).entityId).toBe(300);
  expect(result.removed.hits[0]).toBeNull();
  expect(hit(result.removed.hits[1]).entityId).toBe(301);
  const reused = hit(result.reused.hits[0]);
  expect(reused.entityId).toBe(300);
  expect(reused.entityGeneration).toBe(2);
  expect(reused.historyVersion).toBe(1);
  near(reused.position, [0, 0, 0]);
  near(reused.previousPosition, [0, 0, 0]);
  expect(hit(result.reused.hits[1]).entityGeneration).toBe(2); // Stale generation cannot hide a new entity.
  expect(result.reused.hits[2]).toBeNull();

  expect(result.topologyRejected).not.toBe('');
  expect(result.disabledQuery).not.toBe('');
  expect(result.frames).toBe(5);
  expect(result.diagnostics?.acceleratedBatches).toBe(instanceBvh ? 6 : 0);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(result.noQueryBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
