import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Gate } from '../../src/gi-acceptance';
import type { DiffuseGiUpdateOptions } from '../../src/gi-probe-schedule';
import type { JointFrames, runJointDynamics, runJointLeak, runJointSteadyState } from './fixtures/gi-joint-acceptance';
import { analyzeDynamics, analyzeLeak, analyzeSteady, dynamicGates, JOINT_ANALYSIS, roiSeries } from './gi-joint-analysis';

type Dynamics = Awaited<ReturnType<typeof runJointDynamics>>;

async function fixturePage(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  await page.evaluate(() => { window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose(); });
  return errors;
}
type Fixture = typeof import('./fixtures/gi-joint-acceptance');
async function call<K extends 'runJointDynamics' | 'runJointSteadyState' | 'runJointLeak'>(page: Page, name: K,
  input: Parameters<Fixture[K]>[0]): Promise<Awaited<ReturnType<Fixture[K]>>> {
  return await page.evaluate(async ({ name, input }) => {
    const module = await import('/tests/browser/fixtures/' + 'gi-joint-acceptance.ts') as Fixture;
    return (module[name] as (value: unknown) => Promise<unknown>)(input);
  }, { name, input }) as Awaited<ReturnType<Fixture[K]>>;
}

function healthy(run: { errors: string[]; afterDispose: number; ownedBytes?: number[] }) {
  expect(run.errors).toEqual([]);
  expect(run.afterDispose).toBe(0);
  if (run.ownedBytes) expect(run.ownedBytes).toHaveLength(1);
}
function frozenRois(run: Dynamics) {
  for (const [name, roi] of Object.entries(run.rois)) expect(roi.verified, name).toBeGreaterThanOrEqual(Math.ceil(roi.total / 2));
  expect(run.revealedPixels).toBeGreaterThan(0);
  expect(run.panelGhostPixels).toBeGreaterThan(0);
  for (const phase of run.phases) for (const frame of phase.frames) expect(frame.luminance.every(Number.isFinite)).toBe(true);
}
function csv(run: Dynamics): string {
  const rows = ['phase,index,frame,set,luminance,capturedProbes,queries'];
  for (const phase of run.phases) phase.frames.forEach((frame, index) => {
    for (const set of phase.sets) rows.push([phase.name, index, frame.frame, set.name,
      roiSeries(phase, set.name)[index].toFixed(6), frame.captured, frame.queries].join(','));
  });
  return rows.join('\n');
}
function savePngs(directory: string, prefix: string, shots: { index: number; png: string }[]) {
  for (const shot of shots) writeFileSync(join(directory, `${prefix}-${String(shot.index).padStart(3, '0')}.png`),
    Buffer.from(shot.png.slice(shot.png.indexOf(',') + 1), 'base64'));
}

const SCHEDULED: DiffuseGiUpdateOptions = { mode: 'scheduled' };

test('joint GI acceptance harness runs every dynamic, steady and leak case on frozen ROIs', async ({ page }) => {
  test.setTimeout(900_000);
  const pageErrors = await fixturePage(page);
  const frames: JointFrames = { warmup: 4, stability: 6, baseline: 4, response: 12, moverFrames: 4, panelMoveFrames: 3 };
  const small = { width: 320, height: 180, raysPerProbe: 32, emissiveSamplesPerProbe: 4, roiGrid: 3 };
  for (const update of [undefined, SCHEDULED]) {
    const run = await call(page, 'runJointDynamics', { ...small, update, frames, screenshots: false });
    healthy(run); frozenRois(run);
    expect(run.update).toBe(update ? 'change-driven-bounded' : 'full-grid-every-frame');
    const metrics = analyzeDynamics(run);
    console.log('GI_JOINT_FUNCTIONAL ' + JSON.stringify({ update: run.update, rois: run.rois,
      response: Object.fromEntries(Object.entries(metrics.response).map(([event, rois]) => [event,
        Object.fromEntries(Object.entries(rois).map(([roi, value]) => [roi, [value.initial, value.final]]))])) }));
    // Physical direction of every scripted event (not a quality verdict).
    const sun = metrics.response.sunOff['A.sunPatch'];
    expect(sun.final).toBeLessThan(0.5 * sun.initial);
    expect(metrics.response.doorClose['B.floorNearDoor'].change).toBeLessThan(0);
    expect(metrics.response.doorOpen['B.floorNearDoor'].change).toBeGreaterThan(0);
    for (const [event, rois] of Object.entries(metrics.response)) for (const [roi, value] of Object.entries(rois)) {
      expect(Math.max(value.initial, value.final), `${event} ${roi}`).toBeGreaterThan(0);
    }
    expect(metrics.response.panelOff['B.backWallBelowPanel'].change).toBeLessThan(0);
    expect(metrics.response.panelOn['B.backWallBelowPanel'].change).toBeGreaterThan(0);
    // Recorded frames follow the warm-up full refresh: global light changes fill the
    // scheduled budget exactly and never exceed it; full mode captures all 240 probes.
    const budget = run.schedule!.maxProbesPerUpdate;
    if (!update) expect(budget).toBe(240);
    expect(metrics.cost.maxCapturedProbes).toBe(budget);
    if (update) expect(metrics.cost.meanCapturedProbes).toBeLessThan(budget);
  }
  const steady = await call(page, 'runJointSteadyState', { ...small, update: SCHEDULED, warmup: 12, singleBounceWarmup: 4 });
  healthy(steady);
  for (const name of Object.keys(steady.multi)) expect(steady.multi[name].flat().every(Number.isFinite)).toBe(true);
  const leak = await call(page, 'runJointLeak', { width: 320, height: 180, warmup: 6, raysPerProbe: 32, samples: [8, 5],
    screenshots: false, update: SCHEDULED });
  healthy(leak);
  expect(leak.on.outside[0][1]).toBeGreaterThan(0.1);
  expect(pageErrors).toEqual([]);
});

test('joint GI quality capture records L01-L04, T01 and stability verdicts on hardware', async ({ page }, testInfo) => {
  test.skip(process.env.PLAYWRIGHT_GPU !== 'hardware', 'Quality and response captures require the target hardware adapter.');
  test.setTimeout(3_600_000);
  const pageErrors = await fixturePage(page);
  const out = process.env.GI_JOINT_OUT ?? testInfo.outputPath('gi-joint');
  mkdirSync(out, { recursive: true });
  const frames: JointFrames = { warmup: 120, stability: 120, baseline: 6, response: 90, moverFrames: 30, panelMoveFrames: 12 };
  // GI_JOINT_POLICIES: JSON {name: options | null}; null selects full-grid updates.
  const parsed: Record<string, DiffuseGiUpdateOptions | null> | null = JSON.parse(process.env.GI_JOINT_POLICIES ?? 'null');
  const policies: Record<string, DiffuseGiUpdateOptions | undefined> = parsed ? Object.fromEntries(Object.entries(parsed).map(([key, value]) => [key, value ?? undefined])) : {
    full: undefined, scheduled: SCHEDULED };
  const base = { width: 640, height: 360, roiGrid: 4 };
  const summary: Record<string, unknown> = { analysis: JOINT_ANALYSIS, frames, policies };
  const gates: Gate[] = [];
  const direct = await call(page, 'runJointDynamics', { ...base, frames, screenshots: false, gi: false });
  healthy(direct); frozenRois(direct);
  const directMetrics = analyzeDynamics(direct);
  summary.direct = { response: { sunOff: directMetrics.response.sunOff, sunOn: directMetrics.response.sunOn } };
  writeFileSync(join(out, 'dynamics-direct.csv'), csv(direct));
  for (const [name, update] of Object.entries(policies)) {
    const run = await call(page, 'runJointDynamics', { ...base, update, frames, screenshots: true });
    healthy(run); frozenRois(run);
    expect(run.adapter.isFallbackAdapter).toBe(false);
    const metrics = analyzeDynamics(run);
    gates.push(...dynamicGates(name, metrics, name === Object.keys(policies)[0] ? directMetrics : undefined));
    summary[name] = { adapter: run.adapter, viewport: run.viewport, grid: run.grid, update: run.update, rois: run.rois,
      revealedPixels: run.revealedPixels, panelGhostPixels: run.panelGhostPixels, schedule: run.schedule, metrics };
    writeFileSync(join(out, `dynamics-${name}.csv`), csv(run));
    for (const phase of run.phases) savePngs(out, `${name}-${phase.name}`, phase.screenshots);
  }
  const reference = { paths: 512, bounces: 8 };
  for (const [name, update] of Object.entries(policies)) {
    const steady = await call(page, 'runJointSteadyState', { ...base, update, warmup: 240, singleBounceWarmup: 60,
      reference: name === Object.keys(policies)[0] ? reference : undefined });
    healthy(steady);
    summary[`steady-${name}`] = steady;
    if (steady.reference) summary.reference = steady.reference;
  }
  for (const name of Object.keys(policies)) {
    const steady = summary[`steady-${name}`] as Awaited<ReturnType<typeof runJointSteadyState>>;
    const analyzed = analyzeSteady({ ...steady, reference: (summary.reference ?? null) as typeof steady.reference }, name);
    summary[`L01-${name}`] = analyzed.rois;
    gates.push(...analyzed.gates);
  }
  for (const [name, update] of Object.entries(policies)) {
    const leak = await call(page, 'runJointLeak', { width: 640, height: 360, warmup: 120, samples: [16, 9], update,
      screenshots: true, reference: name === Object.keys(policies)[0] ? { paths: 512, bounces: 6 } : undefined });
    healthy(leak);
    const analyzed = analyzeLeak(leak, name);
    summary[`L04-${name}`] = analyzed;
    gates.push(...analyzed.gates);
    for (const [view, png] of Object.entries(leak.pngs)) savePngs(out, `${name}-leak-${view}`, [{ index: 0, png }]);
    for (const [view, png] of Object.entries(leak.offPngs)) savePngs(out, `${name}-leak-${view}-giOff`, [{ index: 0, png }]);
  }
  summary.gates = gates;
  writeFileSync(join(out, 'joint-acceptance.json'), JSON.stringify(summary, null, 1));
  console.log('GI_JOINT_GATES ' + JSON.stringify(gates.map(item => [item.metric, item.value, item.verdict])));
  expect(gates.length).toBeGreaterThan(20);
  // Regression floor (Apple M3 Max, 2026-10-31): the full-grid and default scheduled
  // policies met every response, stability, off-screen and recovery gate. L01 and
  // L04 quality gates still fail and are recorded, not asserted.
  for (const item of gates) if (['L03', 'stability', 'L02', 'T01'].includes(item.id)
    && /^(full|scheduled|direct)\./.test(item.metric)) expect.soft(item.verdict, item.metric).toBe('pass');
  expect(pageErrors).toEqual([]);
});
