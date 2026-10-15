import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runMaterialNormalFixture, runMaterialNormalBacklightFixture,
  runMaterialNormalTinyTriangleFixture, runMaterialNormalGatherHemisphereFixture,
  runMaterialNormalLodFixture } from './fixtures/material-normal';

type Vec3 = [number, number, number];
type Result = Awaited<ReturnType<typeof runMaterialNormalFixture>>;
test.use({ deviceScaleFactor: 1 });

const baseColor: Vec3 = [0.65, 0.45, 0.3];
const normalize = (a: readonly number[]): Vec3 => {
  const magnitude = Math.hypot(...a);
  return [a[0] / magnitude, a[1] / magnitude, a[2] / magnitude];
};
const dot = (a: readonly number[], b: readonly number[]) =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const encoded = (bytes: readonly number[]): Vec3 => normalize(bytes.map(value => value / 255 * 2 - 1));

function near(actual: readonly number[], expected: readonly number[], tolerance: number, label: string): void {
  expect(actual, label).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Number.isFinite(actual[i]), `${label}[${i}] finite`).toBe(true);
    expect(Math.abs(actual[i] - expected[i]),
      `${label}[${i}]: ${actual[i]} versus ${expected[i]}`).toBeLessThan(tolerance);
  }
}

/** Closed form matching the declared GGX/Smith/Schlick direct-light contract. */
function direct(normal: Vec3, world: Vec3, light: Vec3, sunColor: Vec3, eyeZ = 4): Vec3 {
  const view = normalize([-world[0], -world[1], eyeZ - world[2]]);
  const half = normalize(light.map((value, i) => value + view[i]));
  const nv = Math.max(dot(normal, view), 0.001);
  const nl = Math.max(dot(normal, light), 0);
  const nh = Math.max(dot(normal, half), 0);
  const vh = Math.max(dot(view, half), 0);
  const a2 = 1; // Roughness = 1, metallic = 0.
  const denominator = (1 - nh * nh) + nh * nh * a2;
  const distribution = a2 / (Math.PI * denominator * denominator);
  const k = 0.5;
  const geometry = nv / (nv * (1 - k) + k) * nl / (nl * (1 - k) + k);
  const fresnel = 0.04 + 0.96 * (1 - vh) ** 5;
  const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
  return baseColor.map((base, channel) =>
    ((1 - fresnel) * base / Math.PI + specular) * sunColor[channel] * nl) as Vec3;
}

test('authored normal maps share raster/query tangent frames and preserve geometric visibility', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('NORMAL_MAP_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result: Result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'material-normal.ts') as {
      runMaterialNormalFixture: typeof runMaterialNormalFixture;
    };
    return module.runMaterialNormalFixture();
  });
  for (const name of ['neutral', 'tilted', 'noSun'] as const) {
    const path = testInfo.outputPath(`normal-map-${name}.png`);
    await writeFile(path, Buffer.from(result[name].png.split(',')[1], 'base64'));
    await testInfo.attach(`normal-map-${name}.png`, {
      path, contentType: 'image/png',
    });
  }
  console.log('NORMAL_MAP_DIAGNOSTIC ' + JSON.stringify({
    neutral: { hdr: result.neutral.hdr, hits: result.neutral.hits },
    tilted: { hdr: result.tilted.hdr, hits: result.tilted.hits },
    pixels: result.pixels, variants: [result.neutral.variants, result.tilted.variants],
    errors: result.errors,
  }));
  expect(result.size).toEqual({ width: 512, height: 384 });
  expect(result.neutral.hits).toHaveLength(8);
  expect(result.tilted.hits).toHaveLength(8);
  expect(result.tilted.variants).toBe(result.neutral.variants);
  expect(result.noSun.variants).toBe(result.neutral.variants);

  const left = encoded([191, 128, 238]);
  const right = encoded([64, 128, 238]);
  const mirror = encoded([128, 191, 238]);
  const dynamicGeometric = normalize(cross([1.8, 0, 0.224], [0, 0.8, 0]));
  const dynamicTangent = normalize([1.8, 0, 0.224]);
  const dynamicMapped = normalize([
    dynamicTangent[0] * mirror[0] + dynamicGeometric[0] * mirror[2],
    mirror[1], dynamicTangent[2] * mirror[0] + dynamicGeometric[2] * mirror[2],
  ]);
  const geometric: Vec3[] = [[0, 0, 1], [0, 0, 1], [0, 0, -1],
    [0, 0, 1], dynamicGeometric, [0, 0, 1]];
  const shading: Vec3[] = [left, right, left.map(value => -value) as Vec3,
    [-mirror[0], -mirror[1], mirror[2]], dynamicMapped, mirror];
  for (let i = 0; i < 6; i++) {
    const before = result.neutral.hits[i], after = result.tilted.hits[i];
    expect(after.batch).toBe(before.batch);
    expect(after.entityId).toBe(before.entityId);
    expect(after.triangleIndex).toBe(before.triangleIndex);
    expect(Math.abs(after.distance - before.distance)).toBeLessThan(0.0002);
    near(after.position, before.position, 0.0002, `hit ${i} unchanged position`);
    near(after.geometricNormal, before.geometricNormal, 0.0002, `hit ${i} unchanged geometry`);
    near(after.geometricNormal, geometric[i], 0.004, `hit ${i} analytic triangle normal`);
    near(after.normal, shading[i], 0.018, `hit ${i} analytic mapped normal`);
    expect(dot(after.normal, after.geometricNormal), `hit ${i} visible hemisphere`).toBeGreaterThan(0.6);
    near(result.giHits[i].normal, after.normal, 0.003, `hit ${i} with GI enabled`);
    near(result.giHits[i].geometricNormal, after.geometricNormal, 0.003, `geometry ${i} with GI enabled`);
  }
  near(result.neutral.hits[0].normal, [0, 0, 1], 0.003, 'zero scale neutral normal');
  near(result.neutral.hits[1].normal, [0, 0, 1], 0.003, 'zero scale second texel');
  expect(result.tilted.hits[0].normal[0] - result.tilted.hits[1].normal[0]).toBeGreaterThan(0.8);
  // Dynamic triangle still intersects the center ray at the analytically known origin.
  near(result.tilted.hits[4].position, [4, 0, 0], 0.003, 'affine morph position');
  expect(Math.abs(result.tilted.hits[4].distance - 5)).toBeLessThan(0.003);
  const waveExpected = (tangent: Vec3): Vec3 => {
    const bitangent = cross([0, 0, 1], tangent);
    return normalize([tangent[0] * mirror[0] + bitangent[0] * mirror[1],
      tangent[1] * mirror[0] + bitangent[1] * mirror[1], mirror[2]]);
  };
  const slope = 0.25 * Math.PI / 2;
  const lower = normalize([slope, 1, 0]), upper = normalize([-slope, 1, 0]);
  const updatedTangent = normalize(lower.map((value, axis) =>
    value * 0.25 + upper[axis] * 0.75));
  near(result.waveBefore.normal, waveExpected([0, 1, 0]), 0.018, 'wave phase zero tangent frame');
  near(result.waveAfter.normal, waveExpected(updatedTangent), 0.018, 'wave phase half-pi tangent frame');
  near(result.waveBefore.geometricNormal, [0, 0, 1], 0.003, 'wave geometric normal before');
  near(result.waveAfter.geometricNormal, [0, 0, 1], 0.003, 'wave geometric normal after');
  near(result.waveAfter.position, result.waveBefore.position, 0.003, 'wave fixed query point');
  expect(result.waveAfter.uv[0] - result.waveBefore.uv[0]).toBeGreaterThan(0.05);
  expect(result.waveBefore.normal[1] - result.waveAfter.normal[1]).toBeGreaterThan(0.08);

  for (let i = 0; i < 2; i++) {
    near([result.tilted.depth[i].surface, result.tilted.depth[i].shadow],
      [result.neutral.depth[i].surface, result.neutral.depth[i].shadow],
      0.00002, `pixel ${i} unchanged depth and shadow`);
    const actualDirect = result.tilted.hdr[i].slice(0, 3).map((value, c) => value - result.noSun.hdr[i][c]);
    const pixelHit = result.tilted.hits[6 + i];
    expect(pixelHit.batch).toBe('normal-checker');
    near(pixelHit.normal, result.tilted.hits[i].normal, 0.003, `pixel ${i} normal agreement`);
    const expectedDirect = direct(pixelHit.normal, pixelHit.position,
      result.sunDirection, result.sunColor);
    near(actualDirect, expectedDirect, 0.012, `pixel ${i} raster/query PBR direct`);
  }
  expect(result.tilted.hdr[0][0] - result.tilted.hdr[1][0]).toBeGreaterThan(0.2);
  expect(Math.abs(result.neutral.hdr[0][0] - result.neutral.hdr[1][0])).toBeLessThan(0.03);
  expect(result.neutral.png).not.toBe(result.tilted.png);
  expect(result.tangentReplacementError).toContain('WORLD_QUERY_TOPOLOGY_CHANGED');
  near(result.restoredTangentHit.normal, result.tilted.hits[0].normal, 0.0001, 'original immutable tangent stream restored after rejection');
  expect(result.verification.vertices).toBeGreaterThan(0);
  expect(result.verification.maxError).toBeLessThan(3e-4);
  expect(result.verification.maxNormalError).toBeLessThan(3e-4);
  expect(result.verification.maxTangentError).toBeLessThan(3e-4);
  expect(result.verification.boundsViolations).toBe(0);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('sun below the real face cannot illuminate a tilted normal through direct, world GI, or final gathering', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'material-normal.ts') as {
      runMaterialNormalBacklightFixture: typeof runMaterialNormalBacklightFixture;
    };
    return module.runMaterialNormalBacklightFixture();
  });
  expect(result.hit.batch).toBe('normal-backlight-barrier');
  near(result.hit.geometricNormal, [0, 0, 1], 0.003, 'true upper face');
  expect(dot(result.hit.geometricNormal, result.backSun)).toBeLessThan(-0.09);
  expect(dot(result.hit.normal, result.backSun)).toBeGreaterThan(0.9);
  near(result.directOnly, result.dark, 0.007, 'backside direct sun rejected');
  near(result.worldGi, result.dark, 0.007, 'backside world GI rejected');
  near(result.finalGather, result.dark, 0.007, 'backside final gathering rejected');
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('tiny triangle world query returns finite unit geometric and neutral material normals', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'material-normal.ts') as {
      runMaterialNormalTinyTriangleFixture: typeof runMaterialNormalTinyTriangleFixture;
    };
    return module.runMaterialNormalTinyTriangleFixture();
  });
  expect(result.hit.batch).toBe('normal-tiny-triangle');
  expect(result.hit.triangleIndex).toBe(0);
  expect(result.extent).toBe(1e-14);
  expect(Math.abs(result.hit.distance - 1)).toBeLessThan(0.00001);
  near(result.hit.position, [0, 0, 0], 1e-12, 'tiny triangle position');
  near(result.hit.geometricNormal, [0, 0, 1], 0.0001, 'tiny triangle geometric normal');
  near(result.hit.normal, [0, 0, 1], 0.0001, 'tiny triangle neutral shading normal');
  expect(Math.abs(Math.hypot(...result.hit.geometricNormal) - 1)).toBeLessThan(0.0001);
  expect(Math.abs(Math.hypot(...result.hit.normal) - 1)).toBeLessThan(0.0001);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('final gather clips a 60-degree shading hemisphere at the real plane under uniform sky', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'material-normal.ts') as {
      runMaterialNormalGatherHemisphereFixture: typeof runMaterialNormalGatherHemisphereFixture;
    };
    return module.runMaterialNormalGatherHemisphereFixture();
  });
  console.log('NORMAL_GATHER_HEMISPHERE_DIAGNOSTIC ' + JSON.stringify({
    hdr: result.hdr, anchorHistory: result.anchors.map(anchor => anchor.historyUpdates),
    anchorIrradiance: result.anchors.map(anchor => anchor.irradiance),
    spacing: result.diagnostics.actualSpacing, rays: result.diagnostics.hemisphereRaysPerUpdate,
    errors: result.errors,
  }));
  expect(result.hit.batch).toBe('normal-hemisphere-receiver');
  near(result.hit.geometricNormal, [0, 0, 1], 0.003, 'uniform sky geometric normal');
  near(result.hit.normal, [Math.sqrt(3) / 2, 0, 0.5], 0.003, 'uniform sky 60-degree shading normal');
  expect(result.diagnostics.actualSpacing).toBe(4);
  expect(result.diagnostics.anchorsPerUpdate).toBe(1024);
  expect(result.diagnostics.hemisphereRaysPerUpdate).toBe(8192);
  expect(result.anchors).toHaveLength(4);
  // A cosine-weighted shading hemisphere retains probability (1 + cos 60°)/2
  // above the geometric plane. At 128 history frames × 8 rays, Bernoulli
  // standard error is sqrt(.75*.25/1024)/.75 ≈ 1.8%; 5% allows ~2.8σ per anchor.
  const fraction = 0.75;
  for (const anchor of result.anchors) {
    expect(anchor.valid, `anchor ${anchor.index}`).toBe(true);
    expect(anchor.historyUpdates, `anchor ${anchor.index} effective history`).toBeGreaterThanOrEqual(96);
    near(anchor.normal, [Math.sqrt(3) / 2, 0, 0.5], 0.003, `anchor ${anchor.index} normal`);
    const expectedIrradiance = result.sky.map(value => Math.PI * value * fraction);
    for (let channel = 0; channel < 3; channel++) {
      const relative = Math.abs(anchor.irradiance[channel] - expectedIrradiance[channel])
        / expectedIrradiance[channel];
      expect(relative, `anchor ${anchor.index} sky channel ${channel}`).toBeLessThan(0.05);
    }
  }
  expect(result.hdr).toHaveLength(5);
  const expectedHdr = result.base.map((value, channel) => 0.96 * value * result.sky[channel] * fraction);
  for (const [index, measured] of result.hdr.entries()) {
    for (let channel = 0; channel < 3; channel++) {
      const relative = Math.abs(measured[channel] - expectedHdr[channel]) / expectedHdr[channel];
      expect(relative, `pixel ${index} sky channel ${channel}`).toBeLessThan(0.05);
    }
    expect(measured[3]).toBe(1);
  }
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('actual reduced GPU LOD raster agrees with full source world query under a normal-map parameter update', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'material-normal.ts') as {
      runMaterialNormalLodFixture: typeof runMaterialNormalLodFixture;
    };
    return module.runMaterialNormalLodFixture();
  });
  console.log('NORMAL_LOD_DIAGNOSTIC ' + JSON.stringify({
    source: result.sourceTriangles, roots: result.rootTriangles,
    selected: result.tilted.lod.triangles, clusters: result.tilted.lod.selectedClusters,
    hdr: result.tilted.hdr, query: result.tilted.hits, errors: result.errors,
  }));
  expect(result.sourceTriangles).toBe(512);
  expect(result.rootTriangles).toBeGreaterThan(0);
  expect(result.rootTriangles).toBeLessThan(result.sourceTriangles);
  const selected = result.neutral.lod;
  expect(selected.overflow).toBe(false);
  expect(selected.selectedClusters.length).toBeGreaterThan(0);
  expect(selected.indices).toBe(selected.triangles * 3);
  expect(selected.triangles).toBeGreaterThan(0);
  expect(selected.triangles).toBeLessThan(result.sourceTriangles);
  const clusterIds = selected.selectedClusters.map(cluster => cluster.clusterId).sort((a, b) => a - b);
  for (const capture of [result.neutralDark, result.tilted, result.tiltedDark]) {
    expect(capture.lod.overflow).toBe(false);
    expect(capture.lod.indices).toBe(selected.indices);
    expect(capture.lod.selectedClusters.map(cluster => cluster.clusterId).sort((a, b) => a - b))
      .toEqual(clusterIds);
    expect(capture.variants).toBe(result.neutral.variants);
    expect(capture.hits).toHaveLength(4);
    expect(capture.hits.every(hit => hit.batch === 'normal-lod-grid' && hit.opacity === 1)).toBe(true);
  }
  const left = encoded([191, 128, 238]), right = encoded([64, 128, 238]);
  for (let i = 0; i < 4; i++) {
    const normal = i % 2 === 0 ? left : right;
    const before = result.neutral.hits[i], after = result.tilted.hits[i];
    near(after.geometricNormal, [0, 0, 1], 0.003, `LOD source hit ${i} geometry`);
    near(after.normal, normal, 0.008, `LOD source hit ${i} mapped normal`);
    near(before.normal, [0, 0, 1], 0.003, `LOD source hit ${i} neutral normal`);
    near(after.position, before.position, 0.0002, `LOD source hit ${i} unchanged position`);
    expect(Math.abs(after.distance - before.distance)).toBeLessThan(0.0002);
  }
  for (let i = 0; i < 2; i++) {
    near([result.tilted.depth[i].surface, result.tilted.depth[i].shadow],
      [result.neutral.depth[i].surface, result.neutral.depth[i].shadow],
      0.00002, `LOD point ${i} unchanged depth`);
    const pixelHit = result.tilted.hits[2 + i];
    const measured = result.tilted.hdr[i].slice(0, 3)
      .map((value, channel) => value - result.tiltedDark.hdr[i][channel]);
    const expected = direct(pixelHit.normal, pixelHit.position,
      result.sunDirection, result.sunColor, 12);
    near(measured, expected, 0.015, `LOD pixel ${i} raster/full-query direct PBR`);
    const neutralMeasured = result.neutral.hdr[i].slice(0, 3)
      .map((value, channel) => value - result.neutralDark.hdr[i][channel]);
    const neutralExpected = direct(result.neutral.hits[2 + i].normal,
      result.neutral.hits[2 + i].position, result.sunDirection, result.sunColor, 12);
    near(neutralMeasured, neutralExpected, 0.015, `LOD pixel ${i} neutral direct PBR`);
  }
  expect(result.tilted.hdr[0][0] - result.tilted.hdr[1][0]).toBeGreaterThan(0.2);
  expect(result.errors).toEqual([]);
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});
