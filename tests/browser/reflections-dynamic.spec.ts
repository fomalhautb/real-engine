import { writeFileSync } from 'node:fs';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import type { ReflectionsOptions } from '../../src/reflections';
import type { runReflectionsDynamicFixture } from './fixtures/reflections-dynamic';
import { REFLECTION_QUALITY_OPTIONS } from './reflection-quality-options';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runReflectionsDynamicFixture>>;
type Stage = Result['stages'][number];
const luminance = (rgb: readonly number[]) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
const mean = (values: readonly number[]) => values.reduce((a, b) => a + b, 0) / values.length;

/** Frames until the displayed center luminance stays within 10% of the step. */
function response(stage: Stage, before: number) {
  const series = stage.frames.map(frame => mean(frame.radiance.map(luminance)));
  // History output before the display-only spatial filter (equal to series without it).
  const unfiltered = stage.frames.map(frame => mean(frame.unfiltered.map(luminance)));
  const raw = stage.frames.map(frame => mean(frame.raw.map(luminance)));
  const settled = mean(raw.slice(-16));
  const step = settled - before;
  const within = (value: number) => Math.abs(value - settled) <= 0.1 * Math.abs(step);
  let t90 = series.length + 1;
  for (let k = series.length - 1; k >= 0 && within(series[k]); k--) t90 = k + 1;
  return { label: stage.label, roughness: stage.roughness, reason: stage.reason, before, settled, step,
    t90, residual: series.slice(0, 4).map(value => Math.abs(step) > 1e-6 ? (value - settled) / step : 0),
    series, unfiltered, raw, firstResponse: mean(stage.frames[0].response), firstReset: stage.frames[0].reason, firstModes: stage.frames[0].mode,
    firstCounts: stage.frames[0].count, firstRejected: stage.frames[0].rejected, firstValid: stage.frames[0].valid,
    invalidPositions: stage.frames[0].position.filter((_, i) => !stage.frames[0].valid[i]) };
}

async function dynamicCase(page: Page, testInfo: TestInfo, roughPoseTarget: boolean, reflections?: ReflectionsOptions) {
  test.setTimeout(600_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async options => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-dynamic.ts') as {
      runReflectionsDynamicFixture: typeof runReflectionsDynamicFixture;
    };
    return fixture.runReflectionsDynamicFixture({ motionFrames: 48, probeEvery: 4, reflections: options });
  }, reflections);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  const dynamicIds = [result.ids.character, result.ids.side, ...result.ids.boxes];
  const columns = result.size / 2;

  // 1. Every dynamic object is outside the camera frustum, yet the dominant
  // reflected hit of each anchor equals an exact mirror ray of the same snapshot.
  const seen = new Map<number, number>(dynamicIds.map(id => [id, 0]));
  const motion = result.motion.map(frame => {
    expect(frame.inFrustum.character, `character in frustum at ${frame.frame}`).toBe(false);
    expect(frame.inFrustum.side).toBe(false);
    expect(frame.inFrustum.boxes).toEqual([false, false, false]);
    const valid = frame.anchors.filter(anchor => anchor.valid);
    const agree = valid.filter(anchor => anchor.entity === anchor.expected);
    for (const id of dynamicIds) {
      if (valid.some(anchor => anchor.expected === id && anchor.entity === id)) seen.set(id, seen.get(id)! + 1);
    }
    // Interior anchors: the exact mirror ray and its four grid neighbours hit the
    // same object, so this frame's estimate is unambiguous. History must not lag.
    const interior = frame.anchors.filter((anchor, i) => {
      if (!anchor.valid || anchor.entity !== anchor.expected) return false;
      const x = i % columns, y = Math.floor(i / columns);
      if (x === 0 || y === 0 || x === columns - 1 || y === Math.floor(frame.anchors.length / columns) - 1) return false;
      return [i - 1, i + 1, i - columns, i + columns].every(j => frame.anchors[j].expected === anchor.expected);
    });
    const lag = interior.map(anchor => Math.abs(luminance(anchor.radiance) - luminance(anchor.raw))
      / Math.max(luminance(anchor.raw), 0.05));
    const dynamicInterior = interior.filter(anchor => dynamicIds.includes(anchor.expected as never)).length;
    return { frame: frame.frame, valid: valid.length, agreement: agree.length / valid.length,
      interior: interior.length, dynamicInterior, maxLag: Math.max(0, ...lag),
      lagOver5: lag.filter(value => value > 0.05).length,
      modes: Object.fromEntries(['none', 'stationary', 'surface', 'parallax', 'rejected'].map(mode =>
        [mode, frame.anchors.filter(anchor => anchor.mode === mode).length])),
      clamped: frame.anchors.filter(anchor => anchor.clamped).length, history: frame.history };
  });

  // 2. Response curves after discrete changes, for a smooth and a rough mirror.
  const curves = result.stages.map((stage, index) => {
    const previous = result.stages[index - 1];
    const before = previous ? mean(previous.frames.at(-1)!.radiance.map(luminance)) : luminance(result.sky);
    return response(stage, before);
  }).filter(curve => curve.label !== 'settle');
  const report = JSON.stringify({ adapter: result.adapter, reflections: reflections ?? null, motion, seen: [...seen], curves,
    diagnostics: result.diagnostics }, null, 2);
  writeFileSync(testInfo.outputPath('reflection-dynamic-d02.json'), report);
  await testInfo.attach('reflection-dynamic-d02.json', { body: report, contentType: 'application/json' });

  expect(motion).toHaveLength(12);
  for (const frame of motion) {
    expect(frame.valid, `valid anchors at ${frame.frame}`).toBe(result.size * result.size / 4);
    expect(frame.agreement, `mirror-ray identity agreement at ${frame.frame}`).toBeGreaterThanOrEqual(0.97);
    expect(frame.maxLag, `interior history lag at ${frame.frame}`).toBeLessThanOrEqual(0.1);
  }
  // Lag beyond 5% must stay exceptional over all probed interior anchors.
  const interiorTotal = motion.reduce((sum, frame) => sum + frame.interior, 0);
  expect(motion.reduce((sum, frame) => sum + frame.lagOver5, 0), 'interior anchors lagging over 5%')
    .toBeLessThanOrEqual(Math.floor(interiorTotal * 0.001));
  for (const id of dynamicIds) expect(seen.get(id), `dynamic entity ${id} reflected`).toBeGreaterThanOrEqual(3);
  expect(motion.reduce((sum, frame) => sum + frame.dynamicInterior, 0)).toBeGreaterThan(100);

  const curve = (label: string, roughness: number) => curves.find(item => item.label === label && item.roughness === roughness)!;
  for (const roughness of [0.045, 0.3]) {
    for (const label of ['teleport-in', 'material-parameter', 'history-version', 'teleport-away']) {
      expect(Math.abs(curve(label, roughness).step), `${label}@${roughness} step`).toBeGreaterThan(0.1);
    }
    // Pose events (no reset contract) reach the new state within two frames. An
    // object appearing inside a rough lobe is the measured gap (separate test).
    expect(curve('teleport-away', roughness).t90, `teleport-away@${roughness} T90`).toBeLessThanOrEqual(2);
    if (roughness < 0.1 || roughPoseTarget) {
      expect(curve('teleport-in', roughness).t90, `teleport-in@${roughness} T90`).toBeLessThanOrEqual(2);
    }
    // Material and history-version events reset every anchor: the first frame
    // uses no history at all (checked before the display-only spatial filter). At
    // roughness 0.3 the four-sample estimate after a reset is noisy, so T90 is
    // only gated on the smooth mirror.
    for (const label of ['material-parameter', 'history-version']) {
      const stage = curve(label, roughness);
      expect(stage.firstCounts.every(count => count === 1), `${label}@${roughness} reset`).toBe(true);
      expect(Math.abs(stage.unfiltered[0] - stage.raw[0]), `${label}@${roughness} first frame equals raw`)
        .toBeLessThanOrEqual(1e-6 * Math.max(1, stage.raw[0]));
      if (roughness < 0.1) expect(stage.t90, `${label}@${roughness} T90`).toBeLessThanOrEqual(2);
    }
    expect(curve('material-parameter', roughness).firstReset).toBe('material-changed');
    expect(curve('history-version', roughness).firstReset).toBe('scene-identity-changed');
  }
  // A smooth mirror's instance-tint change is caught by the motion clamp.
  expect(curve('instance-color', 0.045).t90).toBeLessThanOrEqual(2);
  expect(curve('receiver-roughness', 0.045).t90).toBeLessThanOrEqual(1);
  const roughnessChange = curve('receiver-roughness', 0.045);
  expect(roughnessChange.firstModes.every((mode, i) => mode === 'rejected' || !roughnessChange.firstValid[i])).toBe(true);
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
}

test('off-screen skinned and rigid-body reflections follow world queries and history responds to discrete changes', async ({ page }, testInfo) => {
  await dynamicCase(page, testInfo, false);
});

test('D02 target: an object appearing inside a roughness-0.3 lobe reaches 90% within two frames', async ({ page }, testInfo) => {
  // Original threshold, measured with the opt-in quality configuration: the
  // lobe-motion response rebuilds the history from this frame's compatible
  // neighbours where reflected content moved, and the spatial filter is on. The
  // default configuration still needs 31 frames (docs/pending/reflections-quality.md).
  await dynamicCase(page, testInfo, true, REFLECTION_QUALITY_OPTIONS);
});
