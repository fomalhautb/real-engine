import { expect, test } from '@playwright/test';
import { sampleReflectionGgx } from '../../src/reflections';
import type { runReflectionsFixture, runReflectionsCombinedFixture } from './fixtures/reflections';

test.use({ deviceScaleFactor: 1 });

type Result = Awaited<ReturnType<typeof runReflectionsFixture>>;
type Anchor = NonNullable<Result['baseline']['reflection']>['anchors'][number];
type Vec3 = readonly [number, number, number];

function near(actual: readonly number[], expected: readonly number[], tolerance: number, label: string): void {
  for (let channel = 0; channel < expected.length; channel++) {
    expect(Math.abs(actual[channel] - expected[channel]), `${label} channel ${channel}`).toBeLessThan(tolerance);
  }
}

function expectedAnchor(anchor: Anchor, eye: Vec3, color: Vec3, roughness: number,
  extinction = 0): number[] {
  const n = anchor.normal as unknown as Vec3;
  const delta = eye.map((value, axis) => value - anchor.position[axis]);
  const length = Math.hypot(...delta);
  const view = delta.map(value => value / length) as [number, number, number];
  const sum = [0, 0, 0];
  for (let index = 0; index < 4; index++) {
    // reverseBits(index) / 2^32 for the first four indices.
    const azimuth = [0, 0.5, 0.25, 0.75][index];
    const sample = sampleReflectionGgx(n, view, roughness, azimuth, (index + 0.5) / 4);
    if (!sample.weight || sample.direction[2] <= 0) continue;
    const origin = anchor.position.map((value, axis) => value + n[axis] * 0.0001);
    const distance = (5 - origin[2]) / sample.direction[2];
    const x = origin[0] + sample.direction[0] * distance;
    const y = origin[1] + sample.direction[1] * distance;
    if (distance <= 0 || Math.abs(x) > 20 || Math.abs(y) > 20) continue;
    const transmission = Math.exp(-extinction * distance);
    for (let channel = 0; channel < 3; channel++) {
      // Metal=1 and base=1 make the Schlick Fresnel factor exactly one.
      sum[channel] += color[channel] * sample.weight * transmission / 4;
    }
  }
  return sum;
}

test('bounded world-space reflections show an offscreen emitter and update after scene, material, roughness and media edits', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections.ts') as {
      runReflectionsFixture: typeof runReflectionsFixture;
    };
    return fixture.runReflectionsFixture();
  });

  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  await testInfo.attach('reflection-numeric.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  expect(result.baseline.diagnostics?.grid).toEqual({ spacing: 1, columns: 32, rows: 32,
    anchorCount: 1024, rayCount: 4096 });
  expect(result.baseline.diagnostics?.ownedGpuBytes).toBeGreaterThan(0);
  expect(result.baseline.diagnostics?.ownedGpuBytes).toBeLessThanOrEqual(16 * 1024 * 1024);
  expect(result.baseline.reflection?.anchors.map(anchor => anchor.index)).toEqual(result.anchorIndices);
  for (let index = 0; index < result.pixels.length; index++) {
    const anchor = result.baseline.reflection!.anchors[index];
    expect(anchor.valid, `anchor ${index}`).toBe(true);
    expect(anchor.failedQueries).toBe(0);
    expect(anchor.identity[0]).toBeGreaterThan(0);
    expect(anchor.identity[1]).toBe(0xffc00001);
    near(anchor.radiance, expectedAnchor(anchor, result.eye, result.emitterColor, 0.045), 0.025,
      `offscreen anchor ${index}`);
    const contribution = result.baseline.surface[index].map((value, channel) => value - result.disabled.surface[index][channel]);
    near(contribution, anchor.radiance, 0.025, `surface reflection ${index}`);
    expect(contribution[0]).toBeGreaterThan(1.5);
    near(result.restored.surface[index], result.baseline.surface[index], 0.025, 're-enabled reflection');
    near(result.movedBack.surface[index], result.baseline.surface[index], 0.025, 'moved-back emitter');
    near(result.recreated.surface[index], result.baseline.surface[index], 0.025, 'recreated emitter');
    near(result.smoothAgain.surface[index], result.baseline.surface[index], 0.025, 'smooth mirror restored');
    near(result.movedAway.surface[index], result.disabled.surface[index], 0.025, 'offscreen emitter moved away');
    near(result.deleted.surface[index], result.disabled.surface[index], 0.025, 'offscreen emitter deleted');
    near(result.recolored.reflection!.anchors[index].radiance,
      expectedAnchor(result.recolored.reflection!.anchors[index], result.eye, [0.5, 1.5, 2], 0.045),
      0.025, `hot material ${index}`);
    near(result.rough.reflection!.anchors[index].radiance,
      expectedAnchor(result.rough.reflection!.anchors[index], result.eye, result.emitterColor, 0.45),
      0.04, `rough GGX ${index}`);
    const expectedAbsorbed = expectedAnchor(result.absorbed.reflection!.anchors[index], result.eye,
      result.emitterColor, 0.045, 0.2);
    near(result.absorbed.reflection!.anchors[index].radiance, expectedAbsorbed, 0.025,
      `reflected medium path ${index}`);
    const absorbedContribution = result.absorbed.surface[index].map((value, channel) =>
      value - result.absorbedDisabled.surface[index][channel]);
    near(absorbedContribution, expectedAbsorbed, 0.025, `pre-camera-fog HDR ${index}`);
    expect(absorbedContribution[0]).toBeLessThan(contribution[0] * 0.45);
  }
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('combined GI, final gather and fog preserve reflected offscreen vertex color, Mask, Morph, skin and wave edits', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections.ts') as {
      runReflectionsCombinedFixture: typeof runReflectionsCombinedFixture;
    };
    return fixture.runReflectionsCombinedFixture();
  });
  const samples = Object.fromEntries(result.samples.map(sample => [sample.label, sample]));
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  await testInfo.attach('reflection-combinations.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  expect(result.samples).toHaveLength(10);
  expect(samples.opaque.reflection?.grid).toEqual({ spacing: 1, columns: 32, rows: 32,
    anchorCount: 1024, rayCount: 4096 });
  expect(samples.opaque.reflection?.diffuseGi).toBe(true);
  expect(samples.opaque.reflection?.worldMedia).toBe(true);
  expect(samples.opaque.gi?.finalGather).not.toBeNull();
  for (const sample of result.samples) {
    expect(sample.anchor.valid, sample.label).toBe(true);
    expect(sample.anchor.failedQueries, sample.label).toBe(0);
    expect(sample.reflection?.ownedGpuBytes, sample.label).toBeLessThanOrEqual(16 * 1024 * 1024);
    const contribution = sample.on.map((value, channel) => value - sample.off[channel]);
    near(contribution, sample.anchor.radiance, 0.03, `${sample.label} surface contribution`);
  }
  const expected = expectedAnchor(samples.opaque.anchor, result.eye, result.emitterColor, 0.045);
  near(samples.opaque.anchor.radiance, expected, 0.025, 'combined opaque source');
  near(samples['opaque-again'].anchor.radiance, expected, 0.025, 'Mask hot update restored');
  for (const label of ['morph-restored', 'skin-restored', 'wave-restored']) {
    near(samples[label].anchor.radiance, expected, 0.025, `${label} reflected source`);
  }
  for (const label of ['transparent', 'morph-away', 'skin-away', 'wave-away', 'transparent-again']) {
    near(samples[label].anchor.radiance, [0, 0, 0], 0.02, `${label} reflected source removed`);
    const contribution = samples[label].on.map((value, channel) => value - samples[label].off[channel]);
    near(contribution, [0, 0, 0], 0.03, `${label} reflected HDR removed`);
  }
  expect(samples.opaque.on[0] - samples.opaque.off[0]).toBeGreaterThan(1.5);
  expect(result.errors).toEqual([]);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
