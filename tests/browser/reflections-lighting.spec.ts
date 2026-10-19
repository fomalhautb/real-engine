import { expect, test } from '@playwright/test';
import { sampleReflectionGgx } from '../../src/reflections';
import type { runReflectionsLightingFixture } from './fixtures/reflections-lighting';

test.use({ deviceScaleFactor: 1 });
type Result = Awaited<ReturnType<typeof runReflectionsLightingFixture>>;
type Anchor = NonNullable<Result['lit']['reflection']>['anchors'][number];
type Vec3 = readonly [number, number, number];

function dot(a: readonly number[], b: readonly number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function unit(value: readonly number[]): [number, number, number] {
  const length = Math.hypot(...value);
  return [value[0] / length, value[1] / length, value[2] / length];
}
function close(actual: readonly number[], expected: readonly number[], tolerance: number, label: string): void {
  for (let channel = 0; channel < expected.length; channel++) {
    expect(Math.abs(actual[channel] - expected[channel]), `${label} channel ${channel}`).toBeLessThan(tolerance);
  }
}

/** Independent scalar GGX/Schlick evaluation at the secondary PBR hit. */
function directBrdf(base: Vec3, metallic: number, roughness: number,
  normal: Vec3, view: Vec3, light: Vec3): number[] {
  const half = unit(light.map((value, axis) => value + view[axis]));
  const nv = Math.max(dot(normal, view), 0.001);
  const nl = dot(normal, light) > 0 ? Math.max(dot(normal, light), 0) : 0;
  const nh = Math.max(dot(normal, half), 0), vh = Math.max(dot(view, half), 0);
  const alpha = roughness * roughness, a2 = alpha * alpha;
  const denominator = (1 - nh * nh) + nh * nh * a2;
  const distribution = a2 / Math.max(Math.PI * denominator * denominator, 1e-20);
  const k = (roughness + 1) ** 2 / 8;
  const geometry = nv / (nv * (1 - k) + k) * nl / (nl * (1 - k) + k);
  return base.map(value => {
    const f0 = 0.04 * (1 - metallic) + value * metallic;
    const fresnel = f0 + (1 - f0) * (1 - vh) ** 5;
    const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
    const diffuse = (1 - fresnel) * (1 - metallic) * value / Math.PI;
    return (diffuse + specular) * nl;
  });
}

function expectedAnchor(anchor: Anchor, eye: Vec3, base: Vec3, sunColor: Vec3,
  includeDirect: boolean): number[] {
  const mirrorNormal = unit(anchor.normal);
  const cameraView = unit(eye.map((value, axis) => value - anchor.position[axis]));
  const receiverNormal: Vec3 = [0, 0, -1], light = unit([1, 0, -1]);
  const ambient = base.map(value => value * 0.045 * (1 - 0.2 * 0.6));
  const sum = [0, 0, 0];
  for (let sampleIndex = 0; sampleIndex < 4; sampleIndex++) {
    const azimuth = [0, 0.5, 0.25, 0.75][sampleIndex];
    const sample = sampleReflectionGgx(mirrorNormal, cameraView, 0.045,
      azimuth, (sampleIndex + 0.5) / 4);
    if (!sample.weight || sample.direction[2] <= 0) continue;
    const origin = anchor.position.map((value, axis) => value + mirrorNormal[axis] * 0.0001);
    const distance = (5 - origin[2]) / sample.direction[2];
    const hitX = origin[0] + sample.direction[0] * distance;
    const hitY = origin[1] + sample.direction[1] * distance;
    if (distance <= 0 || Math.abs(hitX) > 20 || Math.abs(hitY) > 20) continue;
    const view = unit(sample.direction.map(value => -value));
    const brdf = directBrdf(base, 0.2, 0.5, receiverNormal, view, light);
    for (let channel = 0; channel < 3; channel++) {
      sum[channel] += sample.weight * (ambient[channel]
        + (includeDirect ? brdf[channel] * sunColor[channel] : 0)) / 4;
    }
  }
  return sum;
}

test('reflected PBR hit follows independent GGX direct light, sun occlusion, and nonzero GI', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'reflections-lighting.ts') as {
      runReflectionsLightingFixture: typeof runReflectionsLightingFixture;
    };
    return fixture.runReflectionsLightingFixture();
  });

  for (const adapter of result.adapters) {
    if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(adapter.isFallbackAdapter).toBe(false);
    if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);
  }
  await testInfo.attach('reflection-numeric.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
  expect(result.lit.diagnostics?.grid).toEqual({ spacing: 1, columns: 32, rows: 32,
    anchorCount: 1024, rayCount: 4096 });
  for (let index = 0; index < result.anchorIndices.length; index++) {
    const lit = result.lit.reflection!.anchors[index];
    const unlit = result.noSun.reflection!.anchors[index];
    const shadow = result.shadowed.reflection!.anchors[index];
    const restored = result.restored.reflection!.anchors[index];
    for (const anchor of [lit, unlit, shadow, restored]) {
      expect(anchor.valid, `valid anchor ${index}`).toBe(true);
      expect(anchor.failedQueries, `query failures ${index}`).toBe(0);
    }
    const directReference = expectedAnchor(lit, result.eye, result.base, result.sun, true);
    const ambientReference = expectedAnchor(unlit, result.eye, result.base, result.sun, false);
    close(lit.radiance, directReference, 0.02, `PBR reflected direct ${index}`);
    close(unlit.radiance, ambientReference, 0.01, `PBR reflected ambient ${index}`);
    close(shadow.radiance, ambientReference, 0.015, `sun-blocked reflected hit ${index}`);
    close(restored.radiance, directReference, 0.02, `restored reflected direct ${index}`);
    expect(lit.radiance[0] - shadow.radiance[0]).toBeGreaterThan(0.1);
    const hdrReflection = result.lit.surface[index].map((value, channel) =>
      value - result.disabled.surface[index][channel]);
    close(hdrReflection, lit.radiance, 0.025, `native HDR reflected PBR ${index}`);
    close(result.shadowed.surface[index], result.noSun.surface[index], 0.02,
      `sun shadow only at secondary hit ${index}`);
  }

  expect(result.giDiagnostics).not.toBeNull();
  for (let index = 0; index < result.anchorIndices.length; index++) {
    const enabled = result.giEnabled.reflection!.anchors[index];
    const disabled = result.giDisabled.reflection!.anchors[index];
    expect(enabled.valid).toBe(true);
    expect(enabled.failedQueries).toBe(0);
    expect(disabled.valid).toBe(true);
    expect(disabled.failedQueries).toBe(0);
    close(disabled.radiance, [0, 0, 0], 0.01, `GI disabled reflected hit ${index}`);
    for (let channel = 0; channel < 3; channel++) {
      expect(enabled.radiance[channel], `GI reflected channel ${index}:${channel}`).toBeGreaterThan(0.005);
      expect(enabled.radiance[channel], `GI energy bound ${index}:${channel}`).toBeLessThan(3);
    }
    const hdrReflection = result.giEnabled.surface[index].map((value, channel) =>
      value - result.giNoReflection.surface[index][channel]);
    close(hdrReflection, enabled.radiance, 0.025, `GI reflected native HDR ${index}`);
    close(result.giDisabled.surface[index], result.giNoReflection.surface[index], 0.015,
      `disabled GI/reflective diffuse ${index}`);
  }
  expect(result.directErrors).toEqual([]);
  expect(result.giErrors).toEqual([]);
  expect(result.directOwnedAfterDispose).toBe(0);
  expect(result.ownedBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
