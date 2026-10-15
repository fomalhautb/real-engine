import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runGltfTangentsFixture } from './fixtures/gltf-tangents';

type Vec3 = [number, number, number];
type Result = Awaited<ReturnType<typeof runGltfTangentsFixture>>;
test.use({ deviceScaleFactor: 1 });

const normalize = (v: readonly number[]): Vec3 => {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
};
const dot = (a: readonly number[], b: readonly number[]) =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function near(actual: readonly number[], expected: readonly number[], tolerance: number, label: string): void {
  expect(actual, label).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Number.isFinite(actual[i]), `${label}[${i}] finite`).toBe(true);
    expect(Math.abs(actual[i] - expected[i]), `${label}[${i}]: ${actual[i]} versus ${expected[i]}`)
      .toBeLessThan(tolerance);
  }
}

// Independent analytic surface: Morph makes z=.2x+.1y; one affine bone scales
// x,y,z by 1.1,.9,1.2. Each UV island maps U along a different world axis.
const geometricNormal = normalize([-0.2 / 1.1, -0.1 / 0.9, 1 / 1.2]);
const tangents: Vec3[] = [normalize([1.1, 0, 0.24]), normalize([0, 0.9, 0.12])];
const handedness = [1, -1];
const baseColor: Vec3 = [0.65, 0.45, 0.3];
function mappedNormal(triangle: number, pixel: readonly number[]): Vec3 {
  const encoded = normalize(pixel.slice(0, 3).map(byte => byte / 255 * 2 - 1));
  const tangent = tangents[triangle];
  const bitangent = cross(geometricNormal, tangent).map(value => value * handedness[triangle]);
  return normalize([0, 1, 2].map(axis => tangent[axis] * encoded[0]
    + bitangent[axis] * encoded[1] + geometricNormal[axis] * encoded[2]));
}
function direct(normal: Vec3, world: Vec3, light: Vec3, sunColor: Vec3): Vec3 {
  const view = normalize([-world[0], -world[1], 4 - world[2]]);
  const half = normalize(light.map((value, i) => value + view[i]));
  const nv = Math.max(dot(normal, view), 0.001), nl = Math.max(dot(normal, light), 0);
  const nh = Math.max(dot(normal, half), 0), vh = Math.max(dot(view, half), 0);
  const denominator = (1 - nh * nh) + nh * nh; // roughness = 1
  const distribution = 1 / (Math.PI * denominator * denominator);
  const geometry = nv / (nv * 0.5 + 0.5) * nl / (nl * 0.5 + 0.5);
  const fresnel = 0.04 + 0.96 * (1 - vh) ** 5;
  const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
  return baseColor.map((base, channel) =>
    ((1 - fresnel) * base / Math.PI + specular) * sunColor[channel] * nl) as Vec3;
}

test('glTF generates mirrored Mikk tangents through Morph, affine skin, raster, and world query',
  async ({ page }, testInfo) => {
    const pageErrors: string[] = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.goto('/?trees=1');
    await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
    await expect(page.locator('#error')).toBeHidden();
    const result: Result = await page.evaluate(async () => {
      window.__REAL_ENGINE__!.pause(true);
      window.__REAL_ENGINE__!.dispose();
      const module = await import('/tests/browser/fixtures/' + 'gltf-tangents.ts') as {
        runGltfTangentsFixture: typeof runGltfTangentsFixture;
      };
      return module.runGltfTangentsFixture();
    });
    for (const name of ['neutral', 'mapped'] as const) {
      const path = testInfo.outputPath(`gltf-tangents-${name}.png`);
      await writeFile(path, Buffer.from(result[name].png.split(',')[1], 'base64'));
      await testInfo.attach(`gltf-tangents-${name}.png`, { path, contentType: 'image/png' });
    }
    expect(result.size).toEqual({ width: 512, height: 384 });
    expect(result.vertices).toHaveLength(6);
    expect(result.indices).toHaveLength(6);
    expect(result.warnings).toContain('mesh 0 primitive 0: MikkTSpace-compatible tangents generated for normal texture with vertex splits.');
    const sourcePositions: Vec3[] = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
    const sourceUvs = [[0, 0], [1, 0], [0, 1], [1, 0]];
    const sourceCorners = [0, 1, 2, 0, 2, 3];
    const morphNormalDelta = normalize([-0.2, -0.1, 1]).map((value, i) => value - Number(i === 2));
    for (let corner = 0; corner < 6; corner++) {
      const source = sourceCorners[corner], triangle = Math.floor(corner / 3);
      const vertex = result.vertices[result.indices[corner]];
      near(vertex.position, sourcePositions[source], 1e-6, `corner ${corner} position`);
      near(vertex.normal, [0, 0, 1], 1e-6, `corner ${corner} normal`);
      near(vertex.uv, sourceUvs[source], 1e-6, `corner ${corner} UV`);
      expect(vertex.joints).toEqual([0, 0, 0, 0]);
      near(vertex.weights, [1, 0, 0, 0], 1e-6, `corner ${corner} skin weights`);
      near(vertex.tangent, [...(triangle ? [0, 1, 0] : [1, 0, 0]), handedness[triangle]],
        0.002, `corner ${corner} generated tangent`);
      near(vertex.morphPosition, [0, 0, 0.2 * sourcePositions[source][0] + 0.1 * sourcePositions[source][1]],
        1e-6, `corner ${corner} Morph position`);
      near(vertex.morphNormal, morphNormalDelta, 1e-6, `corner ${corner} Morph normal`);
      near(vertex.morphTangent, tangents[triangle].map((value, axis) => {
        // The source-space target tangent precedes affine skinning.
        const target = triangle ? normalize([0, 1, 0.1]) : normalize([1, 0, 0.2]);
        return target[axis] - Number(axis === triangle);
      }), 0.003, `corner ${corner} Morph tangent`);
    }
    for (const capture of [result.neutral, result.mapped, result.dark]) {
      expect(capture.hits).toHaveLength(4);
      expect(capture.variants).toBe(result.neutral.variants);
      expect(capture.hits.every(hit => hit.batch === 'gltf-generated-mirrored')).toBe(true);
    }
    for (let triangle = 0; triangle < 2; triangle++) {
      const neutral = result.neutral.hits[triangle], mapped = result.mapped.hits[triangle];
      expect(mapped.triangleIndex).toBe(triangle);
      near(neutral.normal, geometricNormal, 0.005, `triangle ${triangle} neutral normal`);
      near(mapped.geometricNormal, geometricNormal, 0.005, `triangle ${triangle} geometric normal`);
      near(mapped.normal, mappedNormal(triangle, result.mapPixel), 0.02,
        `triangle ${triangle} analytically mapped normal`);
      near(mapped.position, neutral.position, 0.0002, `triangle ${triangle} geometry unchanged`);
      expect(dot(mapped.normal, mapped.geometricNormal)).toBeGreaterThan(0.5);
      near(result.mapped.hits[triangle + 2].normal, mapped.normal, 0.005,
        `triangle ${triangle} pixel query normal`);
      const pixelHit = result.mapped.hits[triangle + 2];
      const measuredDirect = result.mapped.hdr[triangle].slice(0, 3)
        .map((value, channel) => value - result.dark.hdr[triangle][channel]);
      near(measuredDirect, direct(pixelHit.normal, pixelHit.position,
        result.sunDirection, result.sunColor), 0.02, `triangle ${triangle} raster/query direct light`);
    }
    expect(result.mapped.png).not.toBe(result.neutral.png);
    expect(result.verification.vertices).toBe(6);
    expect(result.verification.maxError).toBeLessThan(3e-4);
    expect(result.verification.maxNormalError).toBeLessThan(3e-4);
    expect(result.verification.maxTangentError).toBeLessThan(3e-4);
    expect(result.verification.boundsViolations).toBe(0);
    expect(result.errors).toEqual([]);
    expect(result.ownedGpuBytesAfterDispose).toBe(0);
    expect(pageErrors).toEqual([]);
  });
