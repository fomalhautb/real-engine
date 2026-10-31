import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';
import type { FogBudgetConfig, runFogBudgetFixture } from './fixtures/volumetric-fog-budget';

test.use({ deviceScaleFactor: 1, viewport: { width: 1920, height: 1080 } });

const BUDGET_BYTES = 64 * 1024 * 1024;
const base: FogBudgetConfig = { label: 'baseline', pixelSize: 8, slices: 64, maxViewDepth: 64, volumes: 4, air: false, rejection: 'global' };
const vary = (label: string, change: Partial<FogBudgetConfig>): FogBudgetConfig => ({ ...base, ...change, label });
// A/B pairs interleave ABBA around the baseline; the sweep changes one factor at a time.
const CONFIGS: FogBudgetConfig[] = [
  base, vary('air-gi', { air: true }), vary('air-gi', { air: true }), base,
  vary('local-rejection', { rejection: 'local' }), vary('local-rejection', { rejection: 'local' }), base,
  vary('default-budget', { budget: 'default' }),
  vary('pixel-4', { pixelSize: 4 }), vary('pixel-12', { pixelSize: 12 }),
  vary('slices-48', { slices: 48 }), vary('slices-96', { slices: 96 }),
  vary('depth-32', { maxViewDepth: 32 }), vary('depth-128', { maxViewDepth: 128 }),
  vary('volumes-0', { volumes: 0 }), vary('volumes-16', { volumes: 16 }), vary('volumes-64', { volumes: 64 }),
];

function percentile(values: readonly number[], p: number) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)] : NaN;
}

test('F05: fog grid, view distance and overlap sweep at 1080p against the 64 MiB / 2 ms prototype budget (hardware, diagnostic timing)', async ({ page }) => {
  test.skip(process.env.PLAYWRIGHT_GPU !== 'hardware', 'F05 timing needs a hardware adapter; software adapters prove function only');
  test.setTimeout(600_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const results = await page.evaluate(async configs => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'volumetric-fog-budget.ts') as { runFogBudgetFixture: typeof runFogBudgetFixture };
    return fixture.runFogBudgetFixture(configs);
  }, CONFIGS);
  const pooled = new Map<string, { samples: number[]; runs: (typeof results)[number][] }>();
  for (const result of results) {
    expect(result.errors, result.label).toEqual([]);
    expect([result.width, result.height]).toEqual([1920, 1080]);
    expect(result.fog.length, `${result.label} samples`).toBeGreaterThanOrEqual(10);
    const entry = pooled.get(result.label) ?? { samples: [], runs: [] };
    entry.samples.push(...result.fog); entry.runs.push(result);
    pooled.set(result.label, entry);
  }
  const table = [...pooled].map(([label, { samples, runs }]) => {
    const run = runs[0];
    return { label, pixelSize: run.pixelSize, slices: run.slices, maxViewDepth: run.maxViewDepth, volumes: run.volumes, air: run.air,
      rejection: run.rejection, budget: run.budget ?? 'unbounded', grid: run.grid, ownedMiB: run.ownedBytes / 1048576,
      withinMemoryBudget: run.ownedBytes <= BUDGET_BYTES, samples: samples.length, p50: percentile(samples, 0.5), p95: percentile(samples, 0.95),
      runs: runs.length };
  });
  // The renderer's default budget must keep fog within 64 MiB at 1080p by choosing the grid.
  const defaults = table.find(row => row.label === 'default-budget')!;
  expect(defaults.withinMemoryBudget).toBe(true);
  for (const row of table) expect(Number.isFinite(row.p95), row.label).toBe(true);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) {
    expect(results[0].adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  const report = { contended: 'diagnostic only: other agents share the GPU; timing is not an acceptance result',
    adapter: results[0].adapter, budget: { ownedMiB: 64, p95Ms: 2 }, order: CONFIGS.map(config => config.label), table };
  if (process.env.FOG_EVIDENCE_DIR) writeFileSync(join(process.env.FOG_EVIDENCE_DIR, 'f05-sweep.json'), JSON.stringify(report, null, 2));
  console.log('FOG_F05 ' + JSON.stringify(report));
});
