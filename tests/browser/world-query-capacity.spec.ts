import { expect, test } from '@playwright/test';
import type { runSceneBvhMaskCapacityFixture, runWorldQueryCapacityFixture } from './fixtures/world-query-capacity';

type Result = Awaited<ReturnType<typeof runWorldQueryCapacityFixture>>;
type Hit = NonNullable<Result['results'][number]['initial']['hits'][number]>;

function hit(value: Hit | null): Hit {
  expect(value).not.toBeNull();
  if (!value) throw new Error('Expected an authored capacity hit');
  return value;
}

function near(actual: readonly number[], expected: readonly number[], tolerance = 0.0005): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((component, axis) =>
    expect(Math.abs(actual[axis] - component)).toBeLessThan(tolerance));
}

function center(index: number, moved = false): [number, number, number] {
  return [((index % 27) - 13) * 1.5 + (moved ? 0.9 : 0),
    (Math.floor(index / 27) - 7) * 1.5, 0];
}

test('405 scene batches retain identities and materials across candidate mask words', async ({ page }) => {
  test.setTimeout(180000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('framenavigated', frame => { if (frame === page.mainFrame()) console.log('CAPACITY_NAVIGATION ' + frame.url()); });
  page.on('console', message => {
    if (message.type() === 'error' || message.text().includes('[vite]')) console.log('CAPACITY_BROWSER ' + message.text());
  });
  await page.goto('/tests/browser/fixtures/world-query-scene.html');
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'world-query-capacity.ts') as {
      runWorldQueryCapacityFixture: typeof runWorldQueryCapacityFixture;
    };
    return module.runWorldQueryCapacityFixture();
  });

  expect(result.count).toBe(405);
  result.invalidBudgets.forEach(error => expect(error).toContain('INVALID_BATCH_BUDGET'));
  expect(result.defaultOverflow).toContain('SCENE_BATCH_BUDGET');
  expect(result.rendererOverflow).toContain('SCENE_BATCH_BUDGET');
  expect(result.queryOverflow).toContain('WORLD_QUERY_BATCH_BUDGET');
  expect(result.results.map(value => value.mode)).toEqual(['bvh', 'linear']);
  const [accelerated, linear] = result.results;
  expect(accelerated.initial.hits).toEqual(linear.initial.hits);
  expect(accelerated.moved.hits).toEqual(linear.moved.hits);
  expect(accelerated.empty.hits).toEqual(linear.empty.hits);
  expect(accelerated.restored.hits).toEqual(linear.restored.hits);
  expect(accelerated.materialChanged.hits).toEqual(linear.materialChanged.hits);

  expect(accelerated.initialDiagnostics.sceneAcceleration).toBe('batch-bvh');
  expect(accelerated.initialDiagnostics.sceneBvhNodes).toBe(809); // Exact 2 * 405 - 1 topology.
  expect(accelerated.initialDiagnostics.sceneCandidateWordsPerRay).toBe(13);
  expect(accelerated.initialDiagnostics.sceneCandidateBytes).toBe(512 * 13 * 4);
  expect(linear.initialDiagnostics.sceneAcceleration).toBe('linear-batches');
  expect(linear.initialDiagnostics.sceneBvhNodes).toBe(0);
  expect(linear.initialDiagnostics.sceneCandidateWordsPerRay).toBe(0);
  expect(accelerated.initialDiagnostics.lastReadbackQuery?.overflows).toBe(0);
  expect(linear.initialDiagnostics.lastReadbackQuery?.overflows).toBe(0);

  // Every ray has an independently authored target. Checking all 405 also detects lost interior words.
  expect(accelerated.initial.hits).toHaveLength(405);
  accelerated.initial.hits.forEach((value, index) => {
    const found = hit(value);
    expect(found.batch).toBe(`capacity-${index}`);
    expect(found.entityId).toBe(10000 + index);
    expect(found.entityGeneration).toBe(1);
    expect(found.topologyVersion).toBe(1);
    expect([0, 1]).toContain(found.triangleIndex);
    near(found.position, center(index));
    expect(found.distance).toBeCloseTo(5, 4);
    near(found.baseColor, [(index + 1) / 512, 0.25, 0.6]);
  });
  for (const index of [31, 32, 255, 256, 383, 384, 404]) {
    expect(hit(accelerated.initial.hits[index]).batch).toBe(`capacity-${index}`);
  }

  expect(accelerated.moved.hits[0]).toBeNull();
  near(hit(accelerated.moved.hits[1]).position, center(404, true));
  expect(hit(accelerated.moved.hits[1]).entityGeneration).toBe(1);
  expect(accelerated.empty.hits.slice(0, 2)).toEqual([null, null]);
  expect(accelerated.restored.hits[0]).toBeNull();
  near(hit(accelerated.restored.hits[1]).position, center(404, true));
  expect(hit(accelerated.restored.hits[1]).entityGeneration).toBe(2);
  expect(hit(accelerated.restored.hits[1]).historyVersion).toBe(1);
  for (const stage of [accelerated.moved, accelerated.empty, accelerated.restored]) {
    expect(hit(stage.hits[2]).entityId).toBe(10383);
    expect(hit(stage.hits[3]).entityId).toBe(10384);
  }

  expect(accelerated.materialChanged.frame).toBe(accelerated.restored.frame);
  expect(accelerated.materialChanged.materialRevision).toBe(accelerated.restored.materialRevision + 1);
  near(hit(accelerated.materialChanged.hits[0]).baseColor, [256 / 512, 0.25, 0.6]);
  near(hit(accelerated.materialChanged.hits[1]).baseColor, [0.9, 0.1, 0.3]);
  near(hit(accelerated.materialChanged.hits[2]).baseColor, [258 / 512, 0.25, 0.6]);

  for (const record of result.results) {
    expect(record.errors).toEqual([]);
    expect(record.ownedAfterDispose).toBe(0);
    if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(record.adapter.isFallbackAdapter).toBe(false);
    if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
      expect(record.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
    }
  }
  expect(pageErrors).toEqual([]);
  console.log('WORLD_QUERY_CAPACITY_DIAGNOSTIC ' + JSON.stringify({
    adapter: accelerated.adapter, count: result.count,
    sceneBvhNodes: accelerated.initialDiagnostics.sceneBvhNodes,
    maskWords: accelerated.initialDiagnostics.sceneCandidateWordsPerRay,
    acceleratedWork: accelerated.initialDiagnostics.lastReadbackQuery,
    linearWork: linear.initialDiagnostics.lastReadbackQuery,
  }));
});

test('scene BVH masks use exact capacity through the 4096th slot', async ({ page }) => {
  test.setTimeout(120000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/tests/browser/fixtures/world-query-scene.html');
  const result = await page.evaluate(async () => {
    const module = await import('/tests/browser/fixtures/' + 'world-query-capacity.ts') as {
      runSceneBvhMaskCapacityFixture: typeof runSceneBvhMaskCapacityFixture;
    };
    return module.runSceneBvhMaskCapacityFixture();
  });

  function expected(capacity: number, stage: 'initial' | 'moved' | 'deleted' | 'restored'): number[] {
    const wordsPerRay = Math.ceil(capacity / 32);
    const words = Array<number>(4 * wordsPerRay).fill(0);
    const last = capacity - 1;
    const set = (ray: number, index: number) => {
      const word = ray * wordsPerRay + (index >>> 5);
      words[word] = (words[word] | (1 << (index & 31))) >>> 0;
    };
    if (last !== 0) set(0, 0);
    if (stage === 'initial' || stage === 'restored') set(last === 0 ? 0 : 1, last);
    if (stage === 'moved') set(2, last);
    return words;
  }

  expect(result.records.map(record => record.capacity)).toEqual([1, 33, 4096]);
  for (const record of result.records) {
    expect(record.nodeCount).toBe(2 * record.capacity - 1);
    expect(record.maskWords).toBe(Math.ceil(record.capacity / 32));
    for (const stage of ['initial', 'moved', 'deleted', 'restored'] as const) {
      expect(record[stage]).toEqual(expected(record.capacity, stage));
    }
  }
  expect(result.records[2].initial[127]).toBe(0); // Ray 0 has only batch 0.
  expect(result.records[2].initial[128 + 127]).toBe(0x80000000); // Ray 1 selects batch 4095.
  expect(result.records[2].moved[128 + 127]).toBe(0); // Old ray loses that bit after refit.
  expect(result.records[2].moved[256 + 127]).toBe(0x80000000); // New ray gains it.
  expect(result.records[2].deleted[256 + 127]).toBe(0);
  expect(result.errors).toEqual([]);
  expect(result.ownedAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  }
  console.log('SCENE_BVH_MASK_CAPACITY_DIAGNOSTIC ' + JSON.stringify({
    adapter: result.adapter, capacities: result.records.map(record => ({
      capacity: record.capacity, nodes: record.nodeCount, words: record.maskWords,
    })), errors: result.errors,
  }));
});
