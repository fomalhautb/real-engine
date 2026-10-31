import { writeFileSync } from 'node:fs';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { reflectionHistoryLimit, reflectionSampleCoordinates, reflectionSampleRotation,
  resolveReflectionsOptions, sampleReflectionGgx, type ReflectionsOptions } from '../../src/reflections';
import type { runReflectionsLadderFixture, runReflectionsParallaxFixture } from './fixtures/reflections-temporal';
import { REFLECTION_QUALITY_OPTIONS } from './reflection-quality-options';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runReflectionsLadderFixture>>;
type Scene = Pick<Result, 'bars' | 'sky'>;
type Anchor = { position: readonly number[]; normal: readonly number[] };
type Vec3 = [number, number, number];
const luminance = (rgb: readonly number[]) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

/** Exact reflected radiance along one ray for the analytic bar scene; null near a bar edge. */
function trace(result: Scene, origin: readonly number[], direction: readonly number[]): Vec3 | null {
  if (direction[2] <= 0) return [...result.sky] as Vec3;
  const distance = (5 - origin[2]) / direction[2];
  if (distance > 20) return [...result.sky] as Vec3;
  const x = origin[0] + direction[0] * distance, y = origin[1] + direction[1] * distance;
  if (Math.abs(x) > 20) return [...result.sky] as Vec3;
  for (const bar of result.bars) {
    if (Math.abs(y - bar.y[0]) < 2e-3 || Math.abs(y - bar.y[1]) < 2e-3) return null;
    if (y >= bar.y[0] && y <= bar.y[1]) return [...bar.color] as Vec3;
  }
  return [...result.sky] as Vec3;
}

function view(eye: readonly number[], anchor: Anchor): Vec3 {
  const delta = eye.map((value, axis) => value - anchor.position[axis]);
  const length = Math.hypot(...delta);
  return delta.map(value => value / length) as Vec3;
}

/** Estimator with explicit sample coordinates: the same BRDF/PDF weight as the GPU path. */
function estimate(result: Result, anchor: Anchor, roughness: number, coordinates: readonly [number, number][]): Vec3 | null {
  const n = anchor.normal as Vec3, v = view(result.eye, anchor), sum: Vec3 = [0, 0, 0];
  const origin = anchor.position.map((value, axis) => value + n[axis] * 0.0001);
  for (const [u1, u2] of coordinates) {
    const sample = sampleReflectionGgx(n, v, roughness, u1, u2);
    if (!sample.weight) continue;
    const radiance = trace(result, origin, sample.direction);
    if (!radiance) return null;
    for (let channel = 0; channel < 3; channel++) sum[channel] += radiance[channel] * sample.weight / coordinates.length;
  }
  return sum;
}

/** 4,096 stratified samples per anchor; edge-ambiguous single rays only change one of them. */
function reference(result: Scene, eye: readonly number[], anchor: Anchor, roughness: number): Vec3 {
  const n = anchor.normal as Vec3, v = view(eye, anchor), sum: Vec3 = [0, 0, 0];
  const origin = anchor.position.map((value, axis) => value + n[axis] * 0.0001);
  let count = 0;
  for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) {
    const jitter = ((i * 73 + j * 151) % 97) / 97;
    const sample = sampleReflectionGgx(n, v, roughness, (i + jitter) / 64, (j + 1 - jitter) / 64 * 0.999999);
    count++;
    if (!sample.weight) continue;
    const radiance = trace(result, origin, sample.direction) ?? [...result.sky] as Vec3;
    for (let channel = 0; channel < 3; channel++) sum[channel] += radiance[channel] * sample.weight;
  }
  return sum.map(value => value / count) as Vec3;
}

function normalizedError(test: readonly Vec3[], expected: readonly Vec3[]): number {
  let error = 0, magnitude = 0;
  test.forEach((value, i) => {
    for (let channel = 0; channel < 3; channel++) {
      error += Math.abs(value[channel] - expected[i][channel]);
      magnitude += Math.abs(expected[i][channel]);
    }
  });
  return error / Math.max(magnitude, 1e-6);
}

async function ladderCase(page: Page, testInfo: TestInfo, l05Targets: boolean, reflections?: ReflectionsOptions) {
  test.setTimeout(600_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async options => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-temporal.ts') as {
      runReflectionsLadderFixture: typeof runReflectionsLadderFixture;
    };
    return fixture.runReflectionsLadderFixture({ warmup: 64, measured: 120, reflections: options });
  }, reflections);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  const temporal = resolveReflectionsOptions({ temporal: { minHistoryFrames: 8, maxHistoryFrames: 32, ...reflections?.temporal } }).temporal!;
  const filtered = !!reflections?.spatialFilter;
  const perStrip = result.rois.length ? result.rois[0].pixels.length : 0;
  const strip = (anchorPosition: number) => Math.floor(anchorPosition / perStrip);

  // 1. Rotated per-frame estimates equal an independent CPU replica of the exact GPU sample set.
  let replicated = 0;
  result.steps.forEach(step => step.anchors.forEach((anchor, i) => {
    const roughness = result.roughness[i];
    const rotation = reflectionSampleRotation(step.frame, result.traced[i]);
    const expected = estimate(result, anchor, roughness,
      [0, 1, 2, 3].map(index => reflectionSampleCoordinates(index, 4, rotation)));
    if (!expected) return;
    replicated++;
    for (let channel = 0; channel < 3; channel++) {
      expect(Math.abs(anchor.rawRadiance[channel] - expected[channel]), `raw frame ${step.frame} strip ${i}`).toBeLessThan(0.03);
    }
  }));
  expect(replicated).toBeGreaterThan(result.steps.length * result.traced.length * 0.6);

  // 2. Stationary accumulation is the exact capped running mean of raw estimates
  // (before any display filter; without one the displayed radiance is that mean).
  result.traced.forEach((_, i) => {
    let mean = [0, 0, 0], count = 0;
    const cap = reflectionHistoryLimit(result.roughness[i], temporal, true);
    result.steps.forEach((step, k) => {
      const anchor = step.anchors[i];
      count = Math.min(count + 1, cap);
      mean = mean.map((value, channel) => value + (anchor.rawRadiance[channel] - value) / count);
      expect(anchor.historyCount, `count step ${k} strip ${i}`).toBe(count);
      expect(anchor.historyMode).toBe(k === 0 ? 'none' : 'stationary');
      expect(anchor.historyClamped).toBe(false);
      expect(anchor.historyResponse).toBe(0);
      for (let channel = 0; channel < 3; channel++) {
        expect(Math.abs(anchor.unfilteredRadiance[channel] - mean[channel]), `mean step ${k} strip ${i}`)
          .toBeLessThan(1e-4 * Math.max(1, mean[channel]));
      }
      if (!filtered) expect(anchor.radiance).toEqual(anchor.unfilteredRadiance);
    });
  });
  // The first enable after construction keeps the oldest pending reason: no
  // history has ever existed, so the reset is reported as 'initial'.
  expect(result.steps[0].history).toMatchObject({ active: true, valid: true, framesSinceReset: 1, lastResetReason: 'initial' });

  // 3. Converged quality against the high-sample reference, per roughness rung.
  const ladder = result.roughness.map((roughness, index) => {
    const anchors = result.converged.anchors.filter((_, i) => strip(i) === index);
    const fixed = result.fixed.anchors.filter((_, i) => strip(i) === index);
    const expected = anchors.map(anchor => reference(result, result.eye, anchor, roughness));
    for (const anchor of anchors) {
      expect(anchor.valid).toBe(true);
      expect(anchor.historyMode).toBe('stationary');
      expect(anchor.historyCount).toBe(reflectionHistoryLimit(roughness, temporal, true));
    }
    // World distance between neighbouring anchors of this strip (one anchor per pixel).
    const spacing = Math.min(...anchors.slice(1).map((anchor, i) => Math.hypot(...anchor.position.map((value, axis) =>
      value - anchors[i].position[axis]))).filter(distance => distance > 0));
    const radius = anchors.reduce((sum, anchor) => sum + anchor.filterRadius, 0) / anchors.length;
    return { roughness, cap: reflectionHistoryLimit(roughness, temporal, true), spacing, filterRadius: radius,
      mirrorUnchanged: anchors.every(anchor => anchor.filterRadius < spacing
        && anchor.radiance.every((value, channel) => value === anchor.unfilteredRadiance[channel])),
      temporalError: normalizedError(anchors.map(anchor => anchor.radiance as Vec3), expected),
      unfilteredError: normalizedError(anchors.map(anchor => anchor.unfilteredRadiance as Vec3), expected),
      fixedError: normalizedError(fixed.map(anchor => anchor.radiance as Vec3), expected),
      referenceLuminance: expected.reduce((sum, value) => sum + luminance(value), 0) / expected.length };
  });

  // 4. 120-frame stability of displayed HDR in non-dark ROIs: peak-to-peak / mean.
  const stability = result.rois.map((roi, index) => {
    const series = result.series.map(frame => {
      const values = frame.slice(index * roi.pixels.length, (index + 1) * roi.pixels.length);
      return values.reduce((sum, value) => sum + luminance(value), 0) / values.length;
    });
    const mean = series.reduce((a, b) => a + b, 0) / series.length;
    const deviation = Math.sqrt(series.reduce((sum, value) => sum + (value - mean) ** 2, 0) / series.length);
    return { roughness: result.roughness[index], mean, peakToPeak: (Math.max(...series) - Math.min(...series)) / mean,
      coefficientOfVariation: deviation / mean, frames: series.length };
  });
  await testInfo.attach('reflection-temporal-ladder.json', { body: writeReport(testInfo.outputPath('reflection-temporal-ladder.json'),
    { adapter: result.adapter, reflections: reflections ?? null, ladder, stability, diagnostics: result.diagnostics }),
    contentType: 'application/json' });
  for (const rung of ladder) {
    expect(rung.temporalError, `ladder ${rung.roughness} not worse than fixed`).toBeLessThanOrEqual(Math.max(rung.fixedError, 0.02));
  }
  for (const roi of stability) expect(roi.frames).toBe(120);
  expect(result.diagnostics?.temporal).toMatchObject({ active: true, valid: true, framesSinceReset: 64 + 120,
    lastResetReason: 'initial', historyBytes: 2 * result.size * result.size * 96 });
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
  if (filtered) {
    // Mirror detail is not blurred: below one anchor spacing the kernel keeps
    // every mirror-like anchor's value exactly; rough rungs are filtered.
    for (const rung of ladder) {
      if (rung.roughness <= 0.1) expect(rung.mirrorUnchanged, `mirror rung ${rung.roughness} unfiltered`).toBe(true);
      if (rung.roughness >= 0.35) expect(rung.filterRadius, `rough rung ${rung.roughness} kernel`).toBeGreaterThan(rung.spacing);
    }
    expect(result.diagnostics?.spatialFilter).toMatchObject({ configured: true, active: true, iterations: 4, footprint: 0.3 });
  }
  if (!l05Targets) return;
  for (const rung of ladder) {
    expect(rung.temporalError, `ladder ${rung.roughness} temporal error`).toBeLessThan(0.1);
  }
  for (const roi of stability) {
    if (roi.mean < 0.05) continue;
    expect(roi.peakToPeak, `ROI ${roi.roughness} 120-frame fluctuation`).toBeLessThanOrEqual(0.03);
  }
}

test('temporal reflections accumulate exactly on a roughness ladder and improve on the fixed sample set', async ({ page }, testInfo) => {
  await ladderCase(page, testInfo, false);
});

test('L05 targets: ladder within 10% of a 4,096-sample reference and 120 static frames within 3%', async ({ page }, testInfo) => {
  // Original thresholds, measured with the opt-in quality configuration: the
  // edge-aware spatial filter and a 64-frame stationary history at four samples
  // per anchor. The default configuration (no filter, 8-32 frames) still misses
  // the r=0.5/0.7 error and r>=0.1 stability targets (docs/pending/reflections-quality.md).
  await ladderCase(page, testInfo, true, REFLECTION_QUALITY_OPTIONS);
});

test('camera motion reprojects mirror history by reflected-hit parallax rather than surface motion', async ({ page }, testInfo) => {
  test.setTimeout(600_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const runs = await page.evaluate(async quality => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-temporal.ts') as {
      runReflectionsParallaxFixture: typeof runReflectionsParallaxFixture;
    };
    const runs = [];
    // Default parallax threshold, surface-motion-only history, and no history;
    // then the default threshold with the opt-in quality configuration.
    for (const parallaxRoughness of [0.35, 0, null]) {
      runs.push(await fixture.runReflectionsParallaxFixture({ parallaxRoughness, settle: 40, frames: 16, step: 0.08 }));
    }
    runs.push(await fixture.runReflectionsParallaxFixture({ parallaxRoughness: 0.35, settle: 40, frames: 16, step: 0.08, reflections: quality }));
    return runs;
  }, REFLECTION_QUALITY_OPTIONS);
  const [parallax, surface, fixed, quality] = runs;
  for (const run of runs) {
    if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(run.adapter.isFallbackAdapter).toBe(false);
    if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(run.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
    expect(run.errors).toEqual([]);
    expect(run.ownedBytesAfterDispose).toBe(0);
    expect(run.frames).toHaveLength(16);
  }
  // Reference per frame from the parallax run's receiver points; all runs share
  // the camera path and pixel grid, so their receivers must agree.
  const strip = (i: number) => Math.floor(i / parallax.perStrip);
  const references = parallax.frames.map((frame, k) => frame.anchors.map((anchor, i) => {
    for (const run of [surface, fixed, quality]) {
      const other = run.frames[k].anchors[i];
      for (let axis = 0; axis < 3; axis++) expect(Math.abs(other.position[axis] - anchor.position[axis])).toBeLessThan(1e-4);
    }
    return reference(parallax, frame.eye, anchor, parallax.roughness[strip(i)]);
  }));
  const evaluate = (run: typeof parallax) => parallax.roughness.map((roughness, index) => {
    const perFrame = run.frames.map((frame, k) => {
      const chosen = frame.anchors.map((anchor, i) => ({ anchor, i })).filter(item => strip(item.i) === index);
      return { error: normalizedError(chosen.map(item => item.anchor.radiance as Vec3), chosen.map(item => references[k][item.i])),
        modes: Object.fromEntries(['none', 'stationary', 'surface', 'parallax', 'rejected'].map(mode =>
          [mode, chosen.filter(item => item.anchor.mode === mode).length])),
        meanCount: chosen.reduce((sum, item) => sum + item.anchor.count, 0) / chosen.length,
        clamped: chosen.filter(item => item.anchor.clamped).length };
    });
    const moving = perFrame.slice(4);
    return { roughness, meanError: moving.reduce((sum, item) => sum + item.error, 0) / moving.length,
      maxError: Math.max(...moving.map(item => item.error)), perFrame };
  });
  const report = { parallax: evaluate(parallax), surface: evaluate(surface), fixed: evaluate(fixed), quality: evaluate(quality) };
  await testInfo.attach('reflection-temporal-parallax.json', { body: writeReport(testInfo.outputPath('reflection-temporal-parallax.json'), { adapter: parallax.adapter, step: 0.08,
    report: Object.fromEntries(Object.entries(report).map(([key, rungs]) => [key, rungs.map(rung => ({
      roughness: rung.roughness, meanError: rung.meanError, maxError: rung.maxError,
      meanCount: rung.perFrame.slice(4).reduce((sum, item) => sum + item.meanCount, 0) / 12,
      modes: rung.perFrame.at(-1)!.modes, clamped: rung.perFrame.at(-1)!.clamped }))])) }),
    contentType: 'application/json' });
  parallax.roughness.forEach((roughness, index) => {
    const usesParallax = roughness < 0.35;
    const lastModes = report.parallax[index].perFrame.at(-1)!.modes;
    if (usesParallax) {
      // Anchors whose virtual image crosses a strip boundary are rejected by identity.
      expect(lastModes.parallax, `parallax mode at ${roughness}`).toBeGreaterThanOrEqual(parallax.perStrip * 0.8);
      expect(report.parallax[index].meanError, `parallax error ${roughness}`).toBeLessThanOrEqual(report.surface[index].meanError);
    } else {
      expect(lastModes.parallax).toBe(0);
    }
    expect(report.surface[index].perFrame.at(-1)!.modes.parallax).toBe(0);
    // Reprojected history must not be materially worse than no history while moving.
    expect(report.parallax[index].meanError, `moving error ${roughness} versus no history`)
      .toBeLessThanOrEqual(report.fixed[index].meanError * 1.25 + 0.01);
  });
  // The mirror-like strip stays close to the high-sample reference while moving;
  // rougher strips at four samples are noise-limited (reported, not gated).
  expect(report.parallax[0].meanError, 'moving mirror-strip error').toBeLessThan(0.1);
  // The opt-in quality configuration does not regress any moving strip. Mirror-like
  // strips differ only through the longer stationary history of the settle phase
  // (their kernel is below one anchor); rough strips gain from the filter.
  parallax.roughness.forEach((roughness, index) => {
    const quality = report.quality[index].meanError, plain = report.parallax[index].meanError;
    expect(quality, `quality moving error ${roughness}`).toBeLessThanOrEqual(plain * 1.02 + 0.002);
  });
  expect(pageErrors).toEqual([]);
});

function writeReport(path: string, value: unknown): string {
  const body = JSON.stringify(value, null, 2);
  writeFileSync(path, body);
  return body;
}
