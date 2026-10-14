import { expect, test } from '@playwright/test';
import type { runWorldQueryBoundsFixture } from './fixtures/world-query-bounds';

type Result = Awaited<ReturnType<typeof runWorldQueryBoundsFixture>>;
type Hit = NonNullable<Result['initial']['hits'][number]>;

function hit(value: Hit | null): Hit {
  expect(value).not.toBeNull();
  if (!value) throw new Error('Expected a world-space ray hit');
  return value;
}

function near(actual: readonly number[], expected: readonly number[], tolerance = 0.0002): void {
  expect(actual).toHaveLength(expected.length);
  for (let axis = 0; axis < expected.length; axis++) {
    expect(Math.abs(actual[axis] - expected[axis])).toBeLessThan(tolerance);
  }
}

test('multilevel world-query BVH refits, current material, and explicit budgets', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') console.log('WORLD_QUERY_BOUNDS_BROWSER_ERROR ' + message.text());
  });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'world-query-bounds.ts') as {
      runWorldQueryBoundsFixture: typeof runWorldQueryBoundsFixture;
    };
    return module.runWorldQueryBoundsFixture();
  });
  console.log('WORLD_QUERY_BOUNDS_DIAGNOSTIC ' + JSON.stringify({
    frames: [result.initial.frame, result.moved.frame, result.skinned.frame],
    revisions: [result.skinned.materialRevision, result.parameters.materialRevision, result.masked.materialRevision],
    traversalBudgetError: result.traversalBudgetError, memoryBudgetError: result.memoryBudgetError,
    errors: result.errors, budgetErrors: result.budgetErrors,
  }));

  // The first triangle lies in the leftmost leaf and the last in the rightmost leaf.
  // The first ray's tMax equals the exact plane intersection distance.
  const initialFirst = hit(result.initial.hits[0]);
  const initialLast = hit(result.initial.hits[1]);
  expect(initialFirst.batch).toBe('bounds-front');
  expect(initialLast.batch).toBe('bounds-front');
  expect(initialFirst.triangleIndex).toBe(0);
  expect(initialLast.triangleIndex).toBe(15);
  expect(initialFirst.distance).toBeCloseTo(5, 4);
  near(initialFirst.position, [-16, 0, 0]);
  near(initialLast.position, [14, 0, 0]);
  near(initialFirst.baseColor, [0.2, 0.3, 0.4]);
  expect(initialFirst.shading).toBe('unlit');

  // Moving every triangle forces the root and both inner branches to refit.
  expect(result.moved.frame).toBe(result.initial.frame + 1);
  const movedFirst = hit(result.moved.hits[0]);
  const movedLast = hit(result.moved.hits[1]);
  expect(movedFirst.triangleIndex).toBe(0);
  expect(movedLast.triangleIndex).toBe(15);
  near(movedFirst.position, [84, 0, 0]);
  near(movedLast.position, [114, 0, 0]);
  near(movedFirst.previousPosition, [-16, 0, 0]);
  near(movedLast.previousPosition, [14, 0, 0]);
  expect(result.moved.hits[2]).toBeNull();
  expect(result.moved.hits[3]).toBeNull();

  // A joint palette edit changes both extremal leaves and their common ancestors.
  expect(result.skinned.frame).toBe(result.moved.frame + 1);
  const skinnedFirst = hit(result.skinned.hits[0]);
  const skinnedLast = hit(result.skinned.hits[1]);
  expect(skinnedFirst.triangleIndex).toBe(0);
  expect(skinnedLast.triangleIndex).toBe(15);
  near(skinnedFirst.position, [85, 0, 0]);
  near(skinnedLast.position, [115, 0, 0]);
  near(skinnedFirst.previousPosition, [84, 0, 0]);
  near(skinnedLast.previousPosition, [114, 0, 0]);
  expect(result.skinned.hits[2]).toBeNull();
  expect(result.skinned.hits[3]).toBeNull();

  // Material mutations are visible immediately against the last geometry frame.
  expect(result.parameters.frame).toBe(result.skinned.frame);
  expect(result.parameters.materialRevision).toBe(result.skinned.materialRevision + 1);
  const hdr = hit(result.parameters.hits[0]);
  expect(hdr.batch).toBe('bounds-front');
  expect(hdr.shading).toBe('unlit');
  near(hdr.baseColor, [2, 0.5, 0.25]);
  near(hdr.position, [85, 0, 0]);
  expect(result.masked.frame).toBe(result.skinned.frame);
  expect(result.masked.materialRevision).toBe(result.parameters.materialRevision + 1);
  const throughMask = hit(result.masked.hits[0]);
  expect(throughMask.batch).toBe('bounds-back');
  expect(throughMask.entityId).toBe(102);
  expect(throughMask.distance).toBeCloseTo(6, 4);
  near(throughMask.position, [85, 0, -1]);

  expect(result.traversalBudgetError).toContain('WORLD_QUERY_WORK_BUDGET');
  expect(result.memoryBudgetError).toContain('WORLD_QUERY_MEMORY_BUDGET');
  expect(result.errors).toEqual([]);
  expect(result.budgetErrors).toEqual([]);
  expect(result.bytesAfterDispose).toBe(0);
  expect(result.budgetBytesAfterDispose).toBe(0);
  expect(result.unexpectedMemoryBytesAfterDispose).toBeNull();
  expect(pageErrors).toEqual([]);
});
