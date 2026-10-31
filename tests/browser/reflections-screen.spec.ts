import { writeFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import type { runReflectionsScreenFixture } from './fixtures/reflections-screen';
import { reflectionSampleCoordinates, sampleReflectionGgx } from '../../src/reflections';

test.use({ deviceScaleFactor: 1 });

const luminance = (rgb: readonly number[]) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
const total = (screen: { accepted: number; offscreen: number; unresolved: number; disagree: number }) =>
  screen.accepted + screen.offscreen + screen.unresolved + screen.disagree;

test('screen-space detail supplements on-screen contact reflections and falls back to world queries everywhere else', async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-screen.ts') as {
      runReflectionsScreenFixture: typeof runReflectionsScreenFixture;
    };
    return fixture.runReflectionsScreenFixture();
  });
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  const { worldOnly, supplemented, cut, ids } = result;
  expect(worldOnly.state).toMatchObject({ configured: true, enabled: false, active: false, state: 'disabled' });
  expect(supplemented.state).toMatchObject({ enabled: true, active: true, state: 'active',
    captureBytes: result.size * result.size * 12 });
  expect(cut.state).toMatchObject({ active: false, state: 'camera-cut' });
  expect(result.panelOffscreen).toBe(true);

  const groups = { box: [] as number[], panel: [] as number[], sky: [] as number[] };
  supplemented.anchors.forEach((anchor, i) => {
    if (!anchor.valid || anchor.identity !== ids.floor) return;
    if (anchor.reflected === ids.box) groups.box.push(i);
    else if (anchor.reflected === ids.panel) groups.panel.push(i);
    else if (anchor.reflected === 0) groups.sky.push(i);
  });
  const outcome = (indices: number[]) => indices.reduce((sum, i) => {
    const screen = supplemented.anchors[i].screen;
    return { accepted: sum.accepted + screen.accepted, offscreen: sum.offscreen + screen.offscreen,
      unresolved: sum.unresolved + screen.unresolved, disagree: sum.disagree + screen.disagree };
  }, { accepted: 0, offscreen: 0, unresolved: 0, disagree: 0 });
  // World-only and camera-cut frames never use the screen.
  for (const frame of [worldOnly, cut]) expect(frame.anchors.every(anchor => total(anchor.screen) === 0)).toBe(true);
  // Off-screen and sky content: explicit fallback, bit-identical to world queries.
  const exact = (indices: number[]) => indices.filter(i => supplemented.anchors[i].raw.every((value, c) =>
    value === worldOnly.anchors[i].raw[c] && value === cut.anchors[i].raw[c])).length;
  expect(groups.panel.length, 'anchors reflecting the off-screen panel').toBeGreaterThan(20);
  expect(outcome(groups.panel).accepted).toBe(0);
  expect(exact(groups.panel)).toBe(groups.panel.length);
  expect(outcome(groups.sky).accepted).toBe(0);
  expect(exact(groups.sky)).toBe(groups.sky.length);
  // On-screen contact content: most samples are resolved by the screen.
  const box = outcome(groups.box);
  const samples = groups.box.length * 4;
  expect(groups.box.length, 'anchors reflecting the on-screen box').toBeGreaterThan(40);
  expect(box.accepted / samples, 'screen acceptance on contact reflections').toBeGreaterThan(0.5);
  // Accepted screen detail is the rendered surface at the world hit.
  const fully = result.mirrorHits.filter(item => item.pixel && item.hdr && item.pixel.entity === ids.box
    && supplemented.anchors[item.anchor].screen.accepted === 4);
  // raw = sum of BRDF/PDF sample weight x incoming radiance / samples. The mirror
  // lobe is far narrower than a pixel at these distances, so every sample sees
  // the rendered hit; without temporal history the GPU uses the fixed set.
  const sampleWeight = (index: number) => {
    const anchor = supplemented.anchors[index];
    const toEye = result.eye.map((value, axis) => value - anchor.position[axis]), length = Math.hypot(...toEye);
    const view = toEye.map(value => value / length) as [number, number, number];
    return [0, 1, 2, 3].reduce((sum, sample) => {
      const [u1, u2] = reflectionSampleCoordinates(sample, 4);
      return sum + sampleReflectionGgx(anchor.normal as [number, number, number], view, 0.045, u1, u2).weight / 4;
    }, 0);
  };
  const surfaceError = fully.map(item => {
    const expected = luminance(item.hdr!) * sampleWeight(item.anchor);
    return Math.abs(luminance(supplemented.anchors[item.anchor].raw) - expected) / Math.max(expected, 0.02);
  }).sort((a, b) => a - b);
  const worldDifference = groups.box.map(i => Math.abs(luminance(supplemented.anchors[i].raw) - luminance(worldOnly.anchors[i].raw))
    / Math.max(luminance(worldOnly.anchors[i].raw), 0.02)).sort((a, b) => a - b);
  const quantile = (values: number[], q: number) => values.length ? values[Math.min(values.length - 1, Math.floor(q * values.length))] : NaN;
  await testInfo.attach('reflection-screen-trace.json', { body: writeReport(testInfo.outputPath('reflection-screen-trace.json'), { adapter: result.adapter,
    anchors: { box: groups.box.length, panel: groups.panel.length, sky: groups.sky.length },
    outcomes: { box, panel: outcome(groups.panel), sky: outcome(groups.sky) },
    surfaceError: { count: surfaceError.length, median: quantile(surfaceError, 0.5), p90: quantile(surfaceError, 0.9) },
    screenVersusWorld: { median: quantile(worldDifference, 0.5), p90: quantile(worldDifference, 0.9) },
    fullySamples: fully.slice(0, 24).map(item => ({ anchor: item.anchor, x: item.anchor % result.size, y: Math.floor(item.anchor / result.size),
      hit: item.pixel, raw: supplemented.anchors[item.anchor].raw, world: worldOnly.anchors[item.anchor].raw, hdr: item.hdr,
      weight: sampleWeight(item.anchor),
      distance: supplemented.anchors[item.anchor].distance })),
    diagnostics: result.diagnostics }), contentType: 'application/json' });
  expect(fully.length).toBeGreaterThan(10);
  expect(quantile(surfaceError, 0.5), 'median screen-detail error versus the rendered hit').toBeLessThan(0.1);
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

function writeReport(path: string, value: unknown): string {
  const body = JSON.stringify(value, null, 2);
  writeFileSync(path, body);
  return body;
}
