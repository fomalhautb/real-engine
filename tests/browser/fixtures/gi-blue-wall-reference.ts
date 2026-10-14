import { createGiDemoScene, GI_PROBE_GRID } from '../../../src/gi-demo-scene';
import { lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldHit, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const eye: Vec3 = [1.523339179904896, 2.963601128661066, 8.021397057187407];
const target: Vec3 = [0, 1.15, -0.35];
const wallBase: Vec3 = [0.08, 0.24, 0.72];
const rayCount = 1024;
export type GiBlueWallQuality = 'baseline' | 'dense-rays' | 'dense-grid' | 'dense-grid-fast';
type Grid = Omit<typeof GI_PROBE_GRID, 'raysPerProbe' | 'emissiveSamplesPerProbe'>
  & { raysPerProbe: number; emissiveSamplesPerProbe: number };
// Freeze the original demo layout as an authored comparison baseline. If the
// production default changes, this fixture still measures the same 4^3 field.
const BASELINE_GRID: Grid = {
  origin: [-2.55, 0.35, -2.85], spacing: [1.7, 1.1, 1.9],
  counts: [4, 4, 4], raysPerProbe: 64,
  maxDistance: GI_PROBE_GRID.maxDistance,
  normalBias: GI_PROBE_GRID.normalBias,
  skyRadiance: GI_PROBE_GRID.skyRadiance,
  multiBounce: GI_PROBE_GRID.multiBounce,
  emissiveSamplesPerProbe: 0,
};
function gridForQuality(quality: GiBlueWallQuality): Grid {
  if (quality === 'baseline') return BASELINE_GRID;
  if (quality === 'dense-rays') return { ...BASELINE_GRID, raysPerProbe: 256 };
  if (quality === 'dense-grid' || quality === 'dense-grid-fast') {
    return { ...GI_PROBE_GRID,
      raysPerProbe: quality === 'dense-grid' ? GI_PROBE_GRID.raysPerProbe : 64 };
  }
  throw new RangeError(`Unknown blue-wall quality ${quality}`);
}
const nominalWallPoints = [-3.5, -0.5, 2.5].flatMap(z =>
  [0.7, 1.7, 2.7].map(y => [3.2, y, z] as Vec3));

function dot(a: readonly number[], b: readonly number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: readonly number[], b: readonly number[]): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v: readonly number[]): Vec3 {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}
function add(a: readonly number[], b: readonly number[], scale = 1): Vec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}
function srgb(byte: number): number {
  const value = byte / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}
const azimuth = 72 * Math.PI / 180, elevation = 52 * Math.PI / 180;
const sunDirection: Vec3 = [Math.sin(azimuth) * Math.cos(elevation), Math.sin(elevation),
  Math.cos(azimuth) * Math.cos(elevation)];
const sunColor: Vec3 = [srgb(255) * 3.2, srgb(241) * 3.2, srgb(220) * 3.2];

function screenSamples(width: number, height: number) {
  const viewProjection = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150),
    lookAt(eye, target, [0, 1, 0]));
  const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const right = normalize(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const focal = Math.tan(Math.PI / 6.4);
  return nominalWallPoints.map(world => {
    const ndc = transformPoint(viewProjection, world);
    const pixel = { x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5),
      y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5) };
    const x = (2 * (pixel.x + 0.5) / width - 1) * width / height * focal;
    const y = (1 - 2 * (pixel.y + 0.5) / height) * focal;
    const direction = normalize(add(add(forward, right, x), up, y));
    return { world, ndc, pixel, ray: { origin: eye, direction } satisfies WorldRay };
  });
}

/** CPU evaluation of the surface shader's sun BRDF, for diagnostics only.
 * The production GI capture intentionally uses its diffuse term with F=.04. */
function directBrdf(hit: WorldHit, view: Vec3, visible: boolean): Vec3 {
  if (!visible || hit.shading === 'unlit') return [0, 0, 0];
  const n = hit.normal, l = sunDirection, v = normalize(view), h = normalize(add(l, v));
  const nv = Math.max(dot(n, v), 0.001), nl = Math.max(dot(n, l), 0);
  const nh = Math.max(dot(n, h), 0), vh = Math.max(dot(v, h), 0);
  const a2 = hit.roughness ** 4;
  const denominator = (1 - nh * nh) + nh * nh * a2;
  const distribution = a2 / Math.max(Math.PI * denominator * denominator, 1e-20);
  const k = (hit.roughness + 1) ** 2 / 8;
  const geometry = nv / (nv * (1 - k) + k) * nl / (nl * (1 - k) + k);
  return hit.baseColor.map((base, channel) => {
    const f0 = 0.04 * (1 - hit.metallic) + base * hit.metallic;
    const fresnel = f0 + (1 - f0) * (1 - vh) ** 5;
    const specular = distribution * geometry * fresnel / Math.max(4 * nv * nl, 0.001);
    const diffuse = (1 - fresnel) * (1 - hit.metallic) * base / Math.PI;
    return (diffuse + specular) * sunColor[channel] * nl;
  }) as Vec3;
}

function cosineDirections(normal: Vec3): Vec3[] {
  // Golden-angle disk samples. The radial stratification gives a cosine-weighted
  // hemisphere without calling the engine's SH or GI interpolation helpers.
  const tangent = normalize(cross(Math.abs(normal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], normal));
  const bitangent = cross(normal, tangent);
  return Array.from({ length: rayCount }, (_, i) => {
    const r = Math.sqrt((i + 0.5) / rayCount);
    const phi = i * 2.399963229728653;
    return normalize(add(add(add([0, 0, 0], tangent, r * Math.cos(phi)),
      bitangent, r * Math.sin(phi)), normal, Math.sqrt(1 - r * r)));
  });
}

async function reference(renderer: Renderer, wallHits: WorldHit[], grid: Grid) {
  const directions = wallHits.flatMap(hit => cosineDirections(normalize(hit.normal)));
  const rays: WorldRay[] = wallHits.flatMap((hit, index) => {
    const origin = add(hit.position, hit.normal, grid.normalBias);
    return directions.slice(index * rayCount, (index + 1) * rayCount).map(direction =>
      ({ origin, direction, tMax: grid.maxDistance }));
  });
  if (rays.length > 8192) throw new Error('BLUE_WALL_REFERENCE_RAY_BUDGET');
  const primary = (await renderer.traceRays(rays)).hits;
  const shadowRays: WorldRay[] = [];
  const shadowIndices: number[] = [];
  for (let i = 0; i < primary.length; i++) {
    const hit = primary[i];
    if (!hit || hit.shading === 'unlit' || dot(hit.normal, sunDirection) <= 0) continue;
    shadowIndices.push(i);
    shadowRays.push({ origin: add(hit.position, hit.normal, grid.normalBias),
      direction: sunDirection, tMax: grid.maxDistance });
  }
  const shadowHits = (await renderer.traceRays(shadowRays)).hits;
  const visible = new Set<number>();
  for (let i = 0; i < shadowIndices.length; i++) if (!shadowHits[i]) visible.add(shadowIndices[i]);
  return wallHits.map((wall, pointIndex) => {
    const sum: Vec3 = [0, 0, 0], fullBrdfSum: Vec3 = [0, 0, 0];
    const categories: Record<string, number> = {};
    for (let sample = 0; sample < rayCount; sample++) {
      const index = pointIndex * rayCount + sample;
      const hit = primary[index];
      const radiance: Vec3 = hit ? [...hit.emissive] : [...grid.skyRadiance];
      const fullBrdf: Vec3 = [...radiance];
      const category = hit?.batch ?? 'sky';
      categories[category] = (categories[category] ?? 0) + 1;
      if (hit) {
        if (hit.shading === 'unlit') {
          for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c];
          for (let c = 0; c < 3; c++) fullBrdf[c] += hit.baseColor[c];
        } else if (visible.has(index)) {
          const nl = Math.max(0, dot(hit.normal, sunDirection));
          const brdf = directBrdf(hit, directions[index].map(value => -value) as Vec3, true);
          for (let c = 0; c < 3; c++) {
            radiance[c] += hit.baseColor[c] * (1 - hit.metallic) * (0.96 / Math.PI) * sunColor[c] * nl;
            fullBrdf[c] += brdf[c];
          }
        }
      }
      for (let c = 0; c < 3; c++) {
        sum[c] += radiance[c] / rayCount;
        fullBrdfSum[c] += fullBrdf[c] / rayCount;
      }
    }
    // Cosine PDF is cos(theta)/pi, so E = pi * mean(Li). The wall's
    // outgoing diffuse component is base * .96/pi * E.
    const irradiance = sum.map(value => Math.PI * value) as Vec3;
    const outgoingDiffuse = sum.map((value, c) => value * wallBase[c] * 0.96) as Vec3;
    return { world: wall.position, normal: wall.normal, samples: rayCount, categories,
      meanIncoming: sum, irradiance, outgoingDiffuse,
      fullBrdfOutgoingDiffuse: fullBrdfSum.map((value, c) => value * wallBase[c] * 0.96) as Vec3 };
  });
}

export async function runGiBlueWallReferenceFixture(quality: GiBlueWallQuality = 'baseline') {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:1280px;height:720px;';
  document.body.appendChild(canvas);
  const scene = createGiDemoScene();
  const grid = gridForQuality(quality);
  const gridRayCount = grid.counts.reduce((product, count) => product * count, 1) * grid.raysPerProbe;
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scene.batches, {
      diffuseGi: grid,
      worldQueries: { maxRays: Math.max(8192, gridRayCount) },
      sunShadow: { resolution: 2048,
        focusBounds: { min: [-3.5, -0.25, -4], max: [3.5, 4, 4] },
        maxBytes: 16 * 1024 * 1024 },
    });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    renderer.setDiffuseGiEnabled(false);
    renderer.render(eye, target); // Establish the actual device-pixel viewport before projection.
    const samples = screenSamples(renderer.stats.width, renderer.stats.height);
    const capture = async (screenshot: boolean) => {
      renderer!.render(eye, target);
      const hdr = await renderer!.probeHdr(samples.map(sample => sample.pixel));
      const png = screenshot ? canvas.toDataURL('image/png') : null;
      return { frame: renderer!.stats.frames, hdr, png,
        gi: renderer!.diagnostics().diffuseGi };
    };
    const off = await capture(true);
    const cameraHits = (await renderer.traceRays(samples.map(sample => sample.ray))).hits;
    renderer.setDiffuseGiEnabled(true);
    renderer.setDiffuseGiMultiBounce(false);
    const single = await capture(true);
    const probe = await renderer.probeDiffuseGi();
    const selected = cameraHits.map((hit, index) => ({ hit, index }))
      .filter((entry): entry is { hit: WorldHit; index: number } =>
        entry.hit?.batch === 'blue-wall' && entry.hit.normal[0] < -0.9).slice(0, 6);
    const oneBounce = await reference(renderer, selected.map(entry => entry.hit), grid);
    renderer.setDiffuseGiMultiBounce(true);
    let multi = await capture(false);
    for (let i = 1; i < 12; i++) multi = await capture(i === 11);
    renderer.setDiffuseGiMultiBounce(false);
    renderer.setSun({ direction: sunDirection, color: [0, 0, 0] });
    const sunOff = await capture(true);
    renderer.setSun({ direction: sunDirection, color: sunColor });
    const whiteForms = scene.batches.find(batch => batch.name === 'white-forms');
    if (!whiteForms) throw new Error('BLUE_WALL_REFERENCE_MISSING_WHITE_FORMS');
    whiteForms.instances = [];
    const noWhiteForms = await capture(false);
    const result = { quality, camera: { eye, target }, sunDirection, sunColor, grid,
      canvas: { width: renderer.stats.width, height: renderer.stats.height },
      samples: samples.map((sample, index) => ({ world: sample.world, ndc: sample.ndc,
        pixel: sample.pixel, cameraHit: cameraHits[index],
        oneBounce: oneBounce.find((_, selectedIndex) => selected[selectedIndex]?.index === index) ?? null })),
      reference: { selectedIndices: selected.map(entry => entry.index), rayCount, wallBase,
        normalBias: grid.normalBias, note: 'Full-BRDF diagnostic is distinct from diffuse-only GI capture.' },
      cases: { off, single, multi, sunOff, noWhiteForms },
      probe: { frame: probe.frame, invalidProbes: probe.invalidProbes },
      errors: [...renderer.errors] };
    renderer.dispose();
    return { ...result, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
