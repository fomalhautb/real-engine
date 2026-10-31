import { expect, test } from '@playwright/test';
import type { LocalLightDescriptor } from '../../src/local-lights';
import { sampleReflectionGgx } from '../../src/reflections';
import type { LocalLightReflectionPlan, runLocalLightReflectionFixture } from './fixtures/local-lights-reflections';
import { brdf, irradiance, normalize, sub, type Vec3 } from './local-light-oracle';

test.use({ deviceScaleFactor: 1 });
type Result = Awaited<ReturnType<typeof runLocalLightReflectionFixture>>;
type Anchor = NonNullable<Result['lit']['reflection']>['anchors'][number];

const light: LocalLightDescriptor = { kind: 'point', position: [1.2, 0.3, 3.5], color: [1, 0.8, 0.6], intensity: 8, range: 6 };
const plan: LocalLightReflectionPlan = {
  eye: [0, 0, 3],
  // Four anchors whose wall hits the strip shadows, and two whose shadow rays pass beside it. No anchor has
  // |x| = |y| about the centre: such camera or mirror rays meet the quads exactly on a triangle diagonal.
  pixels: [{ x: 16, y: 17 }, { x: 15, y: 17 }, { x: 14, y: 16 }, { x: 14, y: 15 }, { x: 10, y: 16 }, { x: 8, y: 20 }],
  base: [0.6, 0.3, 0.15], light,
  blocker: { center: [0.65, 0, 4.2], halfX: 0.3, halfY: 20 },
  shortRange: 1.5,
};
const BLOCKED = [true, true, true, true, false, false];
const WALL_Z = 5, WALL_NORMAL: Vec3 = [0, 0, -1], WALL_METALNESS = 0.2, WALL_ROUGHNESS = 0.5;

// --- Independent reference ------------------------------------------------------------------------
/** The renderer's deterministic 4-sample GGX pattern from the decoded camera hit, traced to the wall plane. */
function wallSamples(anchor: Anchor) {
  const n = normalize(anchor.normal), v = normalize(sub(plan.eye, anchor.position));
  const list: { weight: number; hit: Vec3; view: Vec3; origin: Vec3; direction: Vec3 }[] = [];
  for (let i = 0; i < 4; i++) {
    const sample = sampleReflectionGgx(n, v, 0.045, [0, 0.5, 0.25, 0.75][i], (i + 0.5) / 4);
    if (!sample.weight || sample.direction[2] <= 0) continue;
    const origin = anchor.position.map((value, axis) => value + n[axis] * 1e-4) as Vec3;
    const t = (WALL_Z - origin[2]) / sample.direction[2];
    const hit = origin.map((value, axis) => value + sample.direction[axis] * t) as Vec3;
    list.push({ weight: sample.weight, hit, view: sample.direction.map(value => -value) as Vec3, origin, direction: [...sample.direction] as Vec3 });
  }
  return list;
}
/** Signed distance inside the blocker rectangle of a point in its plane (positive inside). */
function insideBlocker(x: number, y: number): number {
  const { center, halfX, halfY } = plan.blocker;
  return Math.min(halfX - Math.abs(x - center[0]), halfY - Math.abs(y - center[1]));
}
function shadowCrossing(hit: Vec3): number {
  const s = (plan.blocker.center[2] - hit[2]) / (light.position[2] - hit[2]);
  return insideBlocker(hit[0] + (light.position[0] - hit[0]) * s, hit[1] + (light.position[1] - hit[1]) * s);
}
function expectedAnchor(anchor: Anchor, lit: LocalLightDescriptor | null, blocked: boolean): Vec3 {
  // Legacy reflected-hit fill without GI: base × (0.045 + 0.065 max(n.y, 0)) × (1 − 0.6 metalness), n.y = 0 on the wall.
  const ambient = plan.base.map(value => value * 0.045 * (1 - 0.6 * WALL_METALNESS));
  const sum: Vec3 = [0, 0, 0];
  for (const sample of wallSamples(anchor)) {
    const incoming = [...ambient];
    if (lit && !(blocked && shadowCrossing(sample.hit) > 0)) {
      const { l, e } = irradiance(lit, sample.hit), response = brdf(plan.base, WALL_METALNESS, WALL_ROUGHNESS, WALL_NORMAL, sample.view, l);
      for (let c = 0; c < 3; c++) incoming[c] += response[c] * e[c];
    }
    for (let c = 0; c < 3; c++) sum[c] += sample.weight * incoming[c] / 4;
  }
  return sum;
}
function close(actual: readonly number[], expected: readonly number[], relative: number, label: string): number {
  let worst = 0;
  for (let c = 0; c < 3; c++) {
    const error = Math.abs(actual[c] - expected[c]);
    expect(error, `${label} channel ${c}: actual ${actual[c]} expected ${expected[c]}`).toBeLessThanOrEqual(relative * Math.abs(expected[c]) + 1e-6);
    worst = Math.max(worst, error / Math.max(Math.abs(expected[c]), 1e-6));
  }
  return worst;
}

test('reflected PBR hits receive local light direct lighting and its atlas shadow', async ({ page }) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async input => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'local-lights-reflections.ts') as {
      runLocalLightReflectionFixture: typeof runLocalLightReflectionFixture };
    return fixture.runLocalLightReflectionFixture(input);
  }, plan);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(result.adapter.isFallbackAdapter).toBe(false);
  if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(result.adapter.vendor.toLowerCase()).toBe(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR);

  const captures = { dark: result.dark, lit: result.lit, shadowOpen: result.shadowOpen, shadowBlocked: result.shadowBlocked,
    shadowRestored: result.shadowRestored, disabled: result.disabled, outOfRange: result.outOfRange, relit: result.relit };
  for (const [name, capture] of Object.entries(captures)) {
    for (const anchor of capture.reflection!.anchors) {
      expect(anchor.valid, `${name} anchor ${anchor.index} ${JSON.stringify(anchor)}`).toBe(true);
      expect(anchor.failedQueries, `${name} anchor ${anchor.index} queries`).toBe(0);
    }
  }
  const errors = { dark: 0, lit: 0, blocked: 0, composite: 0 };
  result.lit.reflection!.anchors.forEach((anchor, index) => {
    // Plan guards: the strip never intersects a mirror ray, and every shadow ray clears its edges by 5 cm.
    for (const sample of wallSamples(anchor)) {
      const t = (plan.blocker.center[2] - sample.origin[2]) / sample.direction[2];
      expect(insideBlocker(sample.origin[0] + sample.direction[0] * t, sample.origin[1] + sample.direction[1] * t)).toBeLessThan(-0.05);
      const margin = shadowCrossing(sample.hit);
      if (BLOCKED[index]) expect(margin, `anchor ${index} blocked margin`).toBeGreaterThan(0.05);
      else expect(margin, `anchor ${index} clear margin`).toBeLessThan(-0.05);
      expect(Math.hypot(...sub(light.position, sample.hit))).toBeGreaterThan(plan.shortRange);
    }
    const ambient = expectedAnchor(anchor, null, false), direct = expectedAnchor(anchor, light, false);
    // The light is over 10% of the reflected radiance, 50× the 2e-3 tolerance: a missing term cannot pass.
    for (let c = 0; c < 3; c++) expect(direct[c] - ambient[c], `anchor ${index} light share ${c}`).toBeGreaterThan(0.1 * direct[c]);
    errors.dark = Math.max(errors.dark, close(result.dark.reflection!.anchors[index].radiance, ambient, 2e-3, `dark ${index}`));
    errors.lit = Math.max(errors.lit, close(anchor.radiance, direct, 2e-3, `local light ${index}`));
    errors.blocked = Math.max(errors.blocked, close(result.shadowBlocked.reflection!.anchors[index].radiance,
      expectedAnchor(anchor, light, true), 2e-3, `atlas shadow ${index}`));
    // Shadows without an occluder, toggles and range changes are exact.
    expect(result.shadowOpen.reflection!.anchors[index].radiance).toEqual(anchor.radiance);
    if (BLOCKED[index]) expect(result.shadowBlocked.reflection!.anchors[index].radiance).toEqual(result.dark.reflection!.anchors[index].radiance);
    else expect(result.shadowBlocked.reflection!.anchors[index].radiance).toEqual(anchor.radiance);
    // Composite: the reflected radiance enters the HDR surface; rgba16float rounds each value by ≤ 2^-11.
    const composite = result.relit.surface[index].map((value, c) => value - result.noReflection.surface[index][c]);
    for (let c = 0; c < 3; c++) {
      const error = Math.abs(composite[c] - result.relit.reflection!.anchors[index].radiance[c]);
      expect(error, `HDR composite ${index}:${c}`).toBeLessThanOrEqual(1e-3 * (Math.abs(result.relit.surface[index][c])
        + Math.abs(result.noReflection.surface[index][c])) + 1e-5);
      errors.composite = Math.max(errors.composite, error);
    }
  });
  const radiance = (capture: typeof result.dark) => capture.reflection!.anchors.map(anchor => anchor.radiance);
  expect(radiance(result.shadowRestored)).toEqual(radiance(result.shadowOpen));
  expect(radiance(result.disabled)).toEqual(radiance(result.dark));
  expect(radiance(result.outOfRange)).toEqual(radiance(result.dark));
  expect(radiance(result.relit)).toEqual(radiance(result.shadowOpen));
  expect(result.diagnostics.reflections).toMatchObject({ localLights: true });
  expect(result.diagnostics.localLights).toMatchObject({ cameraCulling: false, shadowedLights: 1, renderedShadowViews: 6 });
  expect(result.afterDispose).toBe(0);
  console.log('LOCAL_LIGHTS_REFLECTION_DIAGNOSTIC ' + JSON.stringify({ adapter: result.adapter, errors,
    lit: radiance(result.lit), dark: radiance(result.dark), blocked: radiance(result.shadowBlocked) }));
});
