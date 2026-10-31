import { expect, test } from '@playwright/test';
import type { runWorldQuerySharedParityFixture } from './fixtures/world-query-shared';

type Result = Awaited<ReturnType<typeof runWorldQuerySharedParityFixture>>;
type Stage = { label: string; frame: number; labelled: Record<string, { batch: string; entityId: number; entityGeneration: number;
  distance: number } | null>; arms: { arm: string; decodedEqual: boolean; diagnostics: { sharedStatic: null | {
    pooledBatches: number; members: number; excluded: Record<string, number>; lastUpdate: { frame: number; batches: number } | null } };
  configs: { config: string; rays: number; equal: number; mismatches: number; referenceHits: number; overflowReference: number;
    overflowActual: number; overflowBoth: number; examples: unknown[]; words: Record<string, { count: number; maxUlps: number }>;
    work: { units: number; tests: number; max: number; overflows: number; quotaViolations: number } }[] }[] };

test('shared static traversal returns the per-batch hit records word for word across edits, policies and budgets', async ({ page }) => {
  test.setTimeout(300_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result: Result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'world-query-shared.ts') as {
      runWorldQuerySharedParityFixture: typeof runWorldQuerySharedParityFixture };
    return fixture.runWorldQuerySharedParityFixture();
  });
  const stages = result.stages as unknown as Stage[];
  expect(stages.map(stage => stage.label)).toEqual(['initial', 'steady', 'moved', 'deleted', 'restored',
    'switch-masked-unsnapshotted', 'switch-masked', 'switch-opaque-double']);
  const summary = stages.flatMap(stage => stage.arms.map(arm => ({ stage: stage.label, arm: arm.arm, decodedEqual: arm.decodedEqual,
    ...Object.fromEntries(arm.configs.map(config => [config.config, { hits: config.referenceHits, equal: config.equal,
      mismatches: config.mismatches, examples: config.examples, words: config.words,
      overflow: [config.overflowReference, config.overflowActual, config.overflowBoth],
      units: config.work.units, tests: config.work.tests, maxUnits: config.work.max }])) })));
  // Logged before any assertion so a failing gate keeps its data.
  console.log('WORLD_QUERY_SHARED_PARITY', JSON.stringify({ rays: stages[0].arms[0].configs[0].rays, summary }));
  for (const stage of stages) {
    for (const arm of stage.arms) {
      expect(arm.decodedEqual, `${stage.label} ${arm.arm} decoded`).toBe(true);
      for (const config of arm.configs) {
        const where = `${stage.label} ${arm.arm} ${config.config}`;
        expect(config.mismatches, `${where} ${JSON.stringify(config.examples)}`).toBe(0);
        expect(config.work.quotaViolations, where).toBe(0);
        if (config.config === 'material' || config.config === 'reverse-support') {
          expect(config.overflowReference + config.overflowActual + config.overflowBoth, where).toBe(0);
          expect(config.equal, where).toBe(config.rays);
        }
      }
      const shared = arm.diagnostics.sharedStatic;
      if (arm.arm.startsWith('shared')) {
        expect(shared).toMatchObject({ pooledBatches: 13, excluded: { 'mutable-input': 1, deformation: 1, budget: 0 } });
        const members = stage.label.startsWith('switch-masked') ? 10 : 11;
        expect(shared!.members, `${stage.label} ${arm.arm} members`).toBe(members);
      } else expect(shared).toBeNull();
    }
    // The tight budgets must bite in the reference while leaving comparable rays.
    for (const label of ['tight', 'partition']) {
      const reference = stage.arms[0].configs.find(config => config.config === label)!;
      expect(reference.work.overflows, `${stage.label} ${label}`).toBeGreaterThan(0);
      for (const arm of stage.arms) expect(arm.configs.find(config => config.config === label)!.equal,
        `${stage.label} ${arm.arm} ${label}`).toBeGreaterThan(100);
    }
    // Pool updates follow changed batches only; unchanged frames keep the last update.
    const lastUpdate = stage.arms.find(arm => arm.arm === 'shared')!.diagnostics.sharedStatic!.lastUpdate;
    const expectedUpdate: Record<string, [number, number]> = { initial: [1, 13], steady: [1, 13], moved: [3, 1],
      deleted: [4, 3], restored: [5, 2], 'switch-masked-unsnapshotted': [5, 2], 'switch-masked': [5, 2], 'switch-opaque-double': [5, 2] };
    expect(lastUpdate, stage.label).toMatchObject({ frame: expectedUpdate[stage.label][0], batches: expectedUpdate[stage.label][1] });
  }
  // Authored semantics of the reference itself.
  const initial = stages[0].labelled;
  expect(initial['tie-front-16.25']).toMatchObject({ entityId: 400, distance: 3 });
  expect(initial['tie-front-16.5']).toMatchObject({ entityId: 400, distance: 3 });
  expect(initial['tie-ignore-first-16.25']).toMatchObject({ entityId: 401 });
  expect(initial['tie-ignore-stale-16.25']).toMatchObject({ entityId: 400 });
  expect(initial['tie-back-16.25']).toMatchObject({ entityId: 402, distance: 2.5 });
  expect(initial['single-front']).toMatchObject({ entityId: 300 });
  expect(initial['single-back']).toBeNull();
  expect(initial['reversed-front']).toBeNull();
  expect(initial['reversed-back']).toMatchObject({ entityId: 301 });
  expect(initial['double-back']).toMatchObject({ entityId: 302 });
  expect(initial['mask-hole-front']).toMatchObject({ entityId: 700, distance: 4 });
  expect(initial['mask-solid-front']).toMatchObject({ entityId: 303 });
  expect(initial['switch-hole-front']).toMatchObject({ entityId: 502 });
  expect(initial['mutable-front']).toMatchObject({ entityId: 500 });
  expect(initial['deformed-front']).toMatchObject({ entityId: 501 });
  for (const label of ['switch-masked-unsnapshotted', 'switch-masked']) {
    const stage = stages.find(item => item.label === label)!;
    expect(stage.labelled['switch-hole-front'], label).toBeNull();
    expect(stage.labelled['switch-front'], label).toMatchObject({ entityId: 502 });
  }
  expect(stages[7].labelled['switch-back']).toMatchObject({ entityId: 502 });
  expect(result.errors).toEqual([]);
  expect(result.cleanup.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
