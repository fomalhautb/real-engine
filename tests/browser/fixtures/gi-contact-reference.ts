import { createGiDemoScene, GI_PROBE_GRID } from '../../../src/gi-demo-scene';
import { lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldHit, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const eye: Vec3 = [1.523339179904896, 2.963601128661066, 8.021397057187407];
const target: Vec3 = [0, 1.15, -0.35];
const block = { entityId: 12, center: [1.3, 0.79, 0.55] as Vec3,
  size: [0.72, 1.58, 0.72] as Vec3, frontZ: 0.91 };
const distances = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4];
const pathsPerPoint = 4096, maxBatch = 8192, bounces = 12;
const tMin = 1e-5, primaryBias = 1e-4;
const sunDirection: Vec3 = [Math.sin(72 * Math.PI / 180) * Math.cos(52 * Math.PI / 180),
  Math.sin(52 * Math.PI / 180), Math.cos(72 * Math.PI / 180) * Math.cos(52 * Math.PI / 180)];
const linear = (byte: number) => {
  const x = byte / 255;
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};
const sunColor: Vec3 = [linear(255) * 3.2, linear(241) * 3.2, linear(220) * 3.2];
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: readonly number[], b: readonly number[]): Vec3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const plus = (a: readonly number[], b: readonly number[], s = 1): Vec3 =>
  [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const unit = (a: readonly number[]): Vec3 => {
  const length = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / length, a[1] / length, a[2] / length];
};
function hash(n: number) {
  n = Math.imul(n ^ (n >>> 16), 0x7feb352d);
  n = Math.imul(n ^ (n >>> 15), 0x846ca68b);
  return (n ^ (n >>> 16)) >>> 0;
}
function direction(normal: Vec3, point: number, sample: number, bounce: number): Vec3 {
  const seed = hash(sample ^ Math.imul(point + 1, 0x9e3779b9) ^ Math.imul(bounce + 1, 0x85ebca6b));
  const r = Math.sqrt((hash(seed ^ 0x68bc21eb) + 0.5) / 0x100000000);
  const phi = 2 * Math.PI * (hash(seed ^ 0x02e5be93) + 0.5) / 0x100000000;
  const tangent = unit(cross(Math.abs(normal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], normal));
  const bitangent = cross(normal, tangent);
  return unit(plus(plus(plus([0, 0, 0], tangent, r * Math.cos(phi)),
    bitangent, r * Math.sin(phi)), normal, Math.sqrt(1 - r * r)));
}
type Path = { point: number; sample: number; origin: Vec3; direction: Vec3;
  throughput: Vec3; one: Vec3; full: Vec3; active: boolean };
function summary(values: Vec3[]) {
  const mean: Vec3 = [0, 0, 0], variance: Vec3 = [0, 0, 0];
  for (const value of values) for (let c = 0; c < 3; c++) mean[c] += value[c] / values.length;
  for (const value of values) for (let c = 0; c < 3; c++) variance[c] +=
    (value[c] - mean[c]) ** 2 / (values.length - 1);
  const standardError = variance.map(v => Math.sqrt(v / values.length)) as Vec3;
  return { mean, sampleVariance: variance, standardError,
    approximate95PercentInterval: mean.map((v, c) =>
      [Math.max(0, v - 1.96 * standardError[c]), v + 1.96 * standardError[c]]) };
}
async function traceReference(renderer: Renderer, floor: WorldHit[], pointOffset: number,
  sampleBias: number, bounceCount: number) {
  if (floor.length * pathsPerPoint > maxBatch || bounceCount < 1 || bounceCount > bounces) {
    throw new RangeError('GI_CONTACT_REFERENCE_BUDGET');
  }
  const paths: Path[] = floor.flatMap((hit, local) => Array.from({ length: pathsPerPoint }, (_, sample) => ({
    point: local, sample, origin: plus(hit.position, hit.normal, sampleBias),
    direction: direction(unit(hit.normal), pointOffset + local, sample, 0),
    throughput: hit.baseColor.map(v => v * 0.96 * (1 - hit.metallic)) as Vec3,
    one: [0, 0, 0] as Vec3, full: [0, 0, 0] as Vec3, active: true,
  })));
  const firstHits = floor.map(() => ({} as Record<string, number>));
  const sources = floor.map(() => ({} as Record<string, Vec3>));
  const firstHitSources = floor.map(() => ({} as Record<string, Vec3>));
  const activeCounts: number[] = [];
  for (let bounce = 0; bounce < bounceCount; bounce++) {
    const active = paths.filter(path => path.active);
    activeCounts.push(active.length);
    if (!active.length) break;
    const hits = (await renderer.traceRays(active.map(path => ({ origin: path.origin,
      direction: path.direction, tMin, tMax: GI_PROBE_GRID.maxDistance })))).hits;
    const shadowIndices: number[] = [], shadowRays: WorldRay[] = [];
    for (let i = 0; i < hits.length; i++) {
      const hit = hits[i];
      if (!hit || hit.shading === 'unlit' || dot(hit.normal, sunDirection) <= 0) continue;
      shadowIndices.push(i);
      shadowRays.push({ origin: plus(hit.position, hit.normal, sampleBias),
        direction: sunDirection, tMin, tMax: GI_PROBE_GRID.maxDistance });
    }
    const shadowHits = (await renderer.traceRays(shadowRays)).hits;
    const visible = new Set(shadowIndices.filter((_, i) => !shadowHits[i]));
    for (let i = 0; i < active.length; i++) {
      const path = active[i], hit = hits[i], category = hit?.batch ?? 'sky';
      if (bounce === 0) firstHits[path.point][category] = (firstHits[path.point][category] ?? 0) + 1;
      const radiance: Vec3 = hit ? [...hit.emissive] : [...GI_PROBE_GRID.skyRadiance];
      if (hit?.shading === 'unlit') {
        for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c];
      } else if (hit && visible.has(i)) {
        const nl = Math.max(0, dot(hit.normal, sunDirection));
        for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c] *
          (1 - hit.metallic) * 0.96 / Math.PI * sunColor[c] * nl;
      }
      for (let c = 0; c < 3; c++) {
        const contribution = path.throughput[c] * radiance[c];
        path.full[c] += contribution;
        if (bounce === 0) path.one[c] = contribution;
        (sources[path.point][category] ??= [0, 0, 0])[c] += contribution / pathsPerPoint;
        if (bounce === 0) (firstHitSources[path.point][category] ??= [0, 0, 0])[c] +=
          contribution / pathsPerPoint;
      }
      if (!hit || hit.shading === 'unlit' || bounce === bounceCount - 1) {
        path.active = false;
      } else {
        for (let c = 0; c < 3; c++) path.throughput[c] *=
          hit.baseColor[c] * 0.96 * (1 - hit.metallic);
        path.origin = plus(hit.position, hit.normal, sampleBias);
        path.direction = direction(unit(hit.normal), pointOffset + path.point, path.sample, bounce + 1);
      }
    }
  }
  return floor.map((hit, point) => {
    const group = paths.slice(point * pathsPerPoint, (point + 1) * pathsPerPoint);
    return { world: hit.position, firstHits: firstHits[point], sources: sources[point],
      firstHitSources: firstHitSources[point], oneBounce: summary(group.map(path => path.one)),
      twelveBounce: summary(group.map(path => path.full)), activeCounts };
  });
}
function cameraSample(world: Vec3, width: number, height: number) {
  const vp = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150),
    lookAt(eye, target, [0, 1, 0]));
  const ndc = transformPoint(vp, world);
  const pixel = { x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5),
    y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5) };
  const forward = unit(plus(target, eye, -1));
  const right = unit(cross(forward, [0, 1, 0]));
  const up = cross(right, forward), focal = Math.tan(Math.PI / 6.4);
  const x = (2 * (pixel.x + 0.5) / width - 1) * width / height * focal;
  const y = (1 - 2 * (pixel.y + 0.5) / height) * focal;
  return { world, ndc, pixel, ray: { origin: eye,
    direction: unit(plus(plus(forward, right, x), up, y)) } satisfies WorldRay };
}
export async function runGiContactReferenceFixture(options: { finalGather?: boolean } = {}) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:1280px;height:720px;';
  document.body.appendChild(canvas);
  const scene = createGiDemoScene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scene.batches, {
      diffuseGi: options.finalGather ? { ...GI_PROBE_GRID, maxBytes: 64 * 1024 * 1024,
        finalGather: { pixelSpacing: 4, raysPerAnchor: 1, maxAnchors: 65536,
          maxHistory: 256, maxBytes: 64 * 1024 * 1024 } } : GI_PROBE_GRID,
      worldQueries: { maxRays: options.finalGather ? 65536 : 32768 }, antiAliasing: 'fxaa',
      sunShadow: { resolution: 2048,
        focusBounds: { min: [-3.5, -0.25, -4], max: [3.5, 4, 4] },
        maxBytes: 16 * 1024 * 1024 },
    });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    if (options.finalGather) renderer.setDiffuseGiFinalGatherEnabled(false);
    renderer.setDiffuseGiEnabled(false);
    renderer.render(eye, target);
    const requested = distances.map(distance => [block.center[0], 0, block.frontZ + distance] as Vec3);
    const floor = (await renderer.traceRays(requested.map(world => ({
      origin: [world[0], 0.25, world[2]] as Vec3, direction: [0, -1, 0] as Vec3,
      tMin, tMax: 1,
    })))).hits;
    if (floor.some(hit => !hit || hit.batch !== 'room-neutral' || hit.entityId !== 1 ||
      hit.normal[1] < 0.9)) throw new Error('GI_CONTACT_EXPECTED_FLOOR');
    const floorHits = floor as WorldHit[];
    const samples = floorHits.map(hit => cameraSample(hit.position, renderer!.stats.width, renderer!.stats.height));
    const cameraHits = (await renderer.traceRays(samples.map(sample => sample.ray))).hits;
    const capture = async (screenshot = false) => {
      renderer!.render(eye, target);
      return { hdr: await renderer!.probeHdr(samples.map(sample => sample.pixel)),
        gi: renderer!.diagnostics().diffuseGi,
        png: screenshot ? canvas.toDataURL('image/png') : null };
    };
    const off = await capture();
    renderer.setDiffuseGiEnabled(true);
    renderer.setDiffuseGiMultiBounce(false);
    const single = await capture();
    renderer.setDiffuseGiMultiBounce(true);
    let multi = await capture();
    for (let i = 1; i < 12; i++) multi = await capture(i === 11);
    let gather: { single: Awaited<ReturnType<typeof capture>>; multi: Awaited<ReturnType<typeof capture>> } | null = null;
    if (options.finalGather) {
      renderer.setDiffuseGiFinalGatherEnabled(true);
      renderer.setDiffuseGiMultiBounce(false);
      for (let i = 0; i < 256; i++) renderer.render(eye, target);
      const gatheredSingle = await capture();
      renderer.setDiffuseGiMultiBounce(true);
      for (let i = 0; i < 320; i++) renderer.render(eye, target);
      const gatheredMulti = await capture(true);
      gather = { single: gatheredSingle, multi: gatheredMulti };
    }
    const reference: Awaited<ReturnType<typeof traceReference>> = [];
    for (let start = 0; start < floorHits.length; start += 2) reference.push(
      ...await traceReference(renderer, floorHits.slice(start, start + 2), start, primaryBias, bounces));
    // Screen pixels quantize the intended positions, and the nearest pixel can
    // even hit the block at 1 cm. Trace a separate reference at each actual
    // floor hit so an HDR comparison never uses a displaced surface position.
    const matched = cameraHits.map((hit, index) => ({ hit, index })).filter(
      (entry): entry is { hit: WorldHit; index: number } =>
        entry.hit?.batch === 'room-neutral' && entry.hit.entityId === 1 &&
        entry.hit.normal[1] > 0.9 && entry.hit.position[2] > block.frontZ);
    const matchedReferences: Awaited<ReturnType<typeof traceReference>> = [];
    for (let start = 0; start < matched.length; start += 2) matchedReferences.push(
      ...await traceReference(renderer, matched.slice(start, start + 2).map(item => item.hit),
        start, primaryBias, bounces));
    const pixelReferenceByIndex = new Map(matched.map((item, index) =>
      [item.index, matchedReferences[index]] as const));
    const biasChecks = [];
    for (const bias of [5e-5, 2e-4]) biasChecks.push({ bias,
      estimates: await traceReference(renderer, floorHits.slice(0, 2), 0, bias, 1) });
    const result = { camera: { eye, target }, block, distances, sunDirection, sunColor,
      grid: GI_PROBE_GRID, canvas: { width: renderer.stats.width, height: renderer.stats.height },
      referenceConfig: { pathsPerPoint, totalPaths: floorHits.length * pathsPerPoint,
        maxLiveRayBatch: maxBatch, bounces, tMin, primaryBias,
        limitation: 'Finite, diffuse-only 12-bounce path estimate. Approximate confidence intervals describe sample variance, not transport truncation. The GI comparison is diagnostic; no accuracy tolerance is imposed.' },
      samples: samples.map((sample, index) => ({ distance: distances[index],
        requestedWorld: requested[index], referenceWorld: floorHits[index].position,
        ndc: sample.ndc, pixel: sample.pixel, cameraHit: cameraHits[index],
        pixelDistanceFromFront: cameraHits[index]?.position[2] === undefined ? null :
          cameraHits[index].position[2] - block.frontZ,
        pixelWorldOffsetMeters: cameraHits[index] ?
          Math.hypot(...cameraHits[index]!.position.map((value, axis) =>
            value - floorHits[index].position[axis])) : null,
        pixelWorldOffsetMillimeters: cameraHits[index] ?
          1000 * Math.hypot(...cameraHits[index]!.position.map((value, axis) =>
            value - floorHits[index].position[axis])) : null,
        pixelReference: pixelReferenceByIndex.get(index) ?? null,
        reference: reference[index] })),
      biasChecks, cases: { off, single, multi }, gather, errors: [...renderer.errors] };
    renderer.dispose();
    return { ...result, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
