import { createGiDemoScene, GI_PROBE_GRID } from '../../../src/gi-demo-scene';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldHit, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const eye: Vec3 = [1.523339179904896, 2.963601128661066, 8.021397057187407];
const target: Vec3 = [0, 1.15, -0.35];
const pathsPerPoint = 4096;
const maxReferenceRayBatch = 8192;
const pathBounces = 12;
const wallArea = [0.65, 1.15, 1.65, 2.15, 2.65].flatMap(x =>
  [1.1, 1.6, 2.1, 2.6].map(z => [x, 0, z] as Vec3));

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
function hash32(value: number): number {
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
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
  return wallArea.map(world => {
    const ndc = transformPoint(viewProjection, world);
    const pixel = { x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5),
      y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5) };
    return { world, ndc, pixel, ray: cameraRay(pixel, width, height) };
  });
}

function cameraRay(pixel: { x: number; y: number }, width: number, height: number): WorldRay {
  const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const right = normalize(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const focal = Math.tan(Math.PI / 6.4);
  const x = (2 * (pixel.x + 0.5) / width - 1) * width / height * focal;
  const y = (1 - 2 * (pixel.y + 0.5) / height) * focal;
  return { origin: eye, direction: normalize(add(add(forward, right, x), up, y)) };
}

function cosineDirection(normal: Vec3, sample: number, point: number, bounce: number): Vec3 {
  // Separate integer hashes for radius and angle at every bounce. The
  // deterministic streams do not reuse a path's earlier direction sequence.
  const seed = hash32(sample ^ Math.imul(point + 1, 0x9e3779b9)
    ^ Math.imul(bounce + 1, 0x85ebca6b));
  const u = (hash32(seed ^ 0x68bc21eb) + 0.5) / 0x100000000;
  const phi = 2 * Math.PI * (hash32(seed ^ 0x02e5be93) + 0.5) / 0x100000000;
  const r = Math.sqrt(u);
  const tangent = normalize(cross(Math.abs(normal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], normal));
  const bitangent = cross(normal, tangent);
  return normalize(add(add(add([0, 0, 0], tangent, r * Math.cos(phi)),
    bitangent, r * Math.sin(phi)), normal, Math.sqrt(1 - r * r)));
}

type Path = {
  point: number; sample: number; origin: Vec3; direction: Vec3;
  throughput: Vec3; oneBounce: Vec3; fourBounce: Vec3;
  eightBounce: Vec3; twelveBounce: Vec3; active: boolean;
};

function estimate(values: readonly Vec3[]) {
  const mean: Vec3 = [0, 0, 0], variance: Vec3 = [0, 0, 0];
  for (const value of values) for (let c = 0; c < 3; c++) mean[c] += value[c] / values.length;
  for (const value of values) for (let c = 0; c < 3; c++) {
    variance[c] += (value[c] - mean[c]) ** 2 / (values.length - 1);
  }
  const standardError = variance.map(value => Math.sqrt(value / values.length)) as Vec3;
  return { mean, sampleVariance: variance, standardError,
    approximate95PercentInterval: mean.map((value, c) =>
      [Math.max(0, value - 1.96 * standardError[c]), value + 1.96 * standardError[c]]) as [number, number][] };
}

type ReferenceOptions = { samples?: number; bounces?: number; seedOffset?: number };

async function independentReferenceBatch(renderer: Renderer, floorHits: WorldHit[],
  options: ReferenceOptions = {}) {
  const sampleCount = options.samples ?? pathsPerPoint;
  const bounceCount = options.bounces ?? pathBounces;
  const seedOffset = options.seedOffset ?? 0;
  const paths: Path[] = floorHits.flatMap((floor, point) =>
    Array.from({ length: sampleCount }, (_, sample) => ({
      point, sample, origin: add(floor.position, floor.normal, GI_PROBE_GRID.normalBias),
      direction: cosineDirection(normalize(floor.normal), sample, point + seedOffset, 0),
      throughput: floor.baseColor.map(value => value * 0.96 * (1 - floor.metallic)) as Vec3,
      oneBounce: [0, 0, 0] as Vec3, fourBounce: [0, 0, 0] as Vec3,
      eightBounce: [0, 0, 0] as Vec3, twelveBounce: [0, 0, 0] as Vec3, active: true,
    })));
  if (paths.length > maxReferenceRayBatch) throw new Error('GI_FLOOR_REFERENCE_RAY_BUDGET');
  const firstHitCategories = floorHits.map(() => ({} as Record<string, number>));
  const firstHitContributions = floorHits.map(() => ({} as Record<string, Vec3>));
  const allHitCategories = floorHits.map(() => ({} as Record<string, number>));
  const allSourceContributions = floorHits.map(() => ({} as Record<string, Vec3>));
  const emitterHitCountsByBounce = floorHits.map(() => Array.from({ length: bounceCount }, () => 0));
  const emitterContributionsByBounce = floorHits.map(() =>
    Array.from({ length: bounceCount }, () => [0, 0, 0] as Vec3));
  const perBounceActiveRays: number[] = [];
  for (let bounce = 0; bounce < bounceCount; bounce++) {
    const active = paths.filter(path => path.active);
    perBounceActiveRays.push(active.length);
    if (!active.length) continue;
    const hits = (await renderer.traceRays(active.map(path => ({
      origin: path.origin, direction: path.direction, tMax: GI_PROBE_GRID.maxDistance,
    })))).hits;
    const shadowRays: WorldRay[] = [];
    const shadowIndices: number[] = [];
    for (let i = 0; i < hits.length; i++) {
      const hit = hits[i];
      if (!hit || hit.shading === 'unlit' || dot(hit.normal, sunDirection) <= 0) continue;
      shadowIndices.push(i);
      shadowRays.push({ origin: add(hit.position, hit.normal, GI_PROBE_GRID.normalBias),
        direction: sunDirection, tMax: GI_PROBE_GRID.maxDistance });
    }
    const shadowHits = (await renderer.traceRays(shadowRays)).hits;
    const sunVisible = new Set<number>();
    for (let i = 0; i < shadowIndices.length; i++) if (!shadowHits[i]) sunVisible.add(shadowIndices[i]);
    for (let i = 0; i < active.length; i++) {
      const path = active[i], hit = hits[i];
      const category = hit?.batch ?? 'sky';
      allHitCategories[path.point][category] = (allHitCategories[path.point][category] ?? 0) + 1;
      if (category === 'emissive-strip') emitterHitCountsByBounce[path.point][bounce]++;
      if (bounce === 0) {
        firstHitCategories[path.point][category] =
          (firstHitCategories[path.point][category] ?? 0) + 1;
      }
      const radiance: Vec3 = hit ? [...hit.emissive] : [...GI_PROBE_GRID.skyRadiance];
      if (hit?.shading === 'unlit') {
        for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c];
      } else if (hit && sunVisible.has(i)) {
        const nl = Math.max(0, dot(hit.normal, sunDirection));
        for (let c = 0; c < 3; c++) radiance[c] += hit.baseColor[c]
          * (1 - hit.metallic) * 0.96 / Math.PI * sunColor[c] * nl;
      }
      for (let c = 0; c < 3; c++) {
        const contribution = path.throughput[c] * radiance[c];
        path.twelveBounce[c] += contribution;
        if (bounce < 8) path.eightBounce[c] += contribution;
        if (bounce < 4) path.fourBounce[c] += contribution;
        if (bounce === 0) path.oneBounce[c] = contribution;
        const sourceSum = allSourceContributions[path.point][category] ??= [0, 0, 0];
        sourceSum[c] += contribution / sampleCount;
        if (category === 'emissive-strip') {
          emitterContributionsByBounce[path.point][bounce][c] += contribution / sampleCount;
        }
        if (bounce === 0) {
          const categorySum = firstHitContributions[path.point][category] ??= [0, 0, 0];
          categorySum[c] += contribution / sampleCount;
        }
      }
      if (!hit || hit.shading === 'unlit' || bounce === bounceCount - 1) {
        path.active = false;
        continue;
      }
      for (let c = 0; c < 3; c++) path.throughput[c] *=
        hit.baseColor[c] * 0.96 * (1 - hit.metallic);
      path.origin = add(hit.position, hit.normal, GI_PROBE_GRID.normalBias);
      path.direction = cosineDirection(normalize(hit.normal), path.sample,
        path.point + seedOffset, bounce + 1);
    }
  }
  return floorHits.map((floor, point) => {
    const start = point * sampleCount;
    const subset = paths.slice(start, start + sampleCount);
    return { world: floor.position, normal: floor.normal,
      floorBase: floor.baseColor, firstHitCategories: firstHitCategories[point],
      firstHitContributions: firstHitContributions[point],
      allHitCategories: allHitCategories[point],
      allSourceContributions: allSourceContributions[point],
      emitterHitCountsByBounce: emitterHitCountsByBounce[point],
      emitterContributionsByBounce: emitterContributionsByBounce[point],
      oneBounce: estimate(subset.map(path => path.oneBounce)),
      truncatedFourBounce: estimate(subset.map(path => path.fourBounce)),
      truncatedEightBounce: estimate(subset.map(path => path.eightBounce)),
      truncatedTwelveBounce: estimate(subset.map(path => path.twelveBounce)) };
  }).map((value, index) => ({ ...value,
    pathCount: sampleCount, maxBounces: bounceCount,
    perBounceActiveRays: index === 0 ? perBounceActiveRays : undefined }));
}

async function independentReference(renderer: Renderer, floorHits: WorldHit[],
  options: ReferenceOptions = {}) {
  const sampleCount = options.samples ?? pathsPerPoint;
  if (!Number.isInteger(sampleCount) || sampleCount < 2 || sampleCount > maxReferenceRayBatch) {
    throw new RangeError('GI_FLOOR_REFERENCE_SAMPLE_BUDGET');
  }
  const pointsPerBatch = Math.floor(maxReferenceRayBatch / sampleCount);
  const results: Awaited<ReturnType<typeof independentReferenceBatch>> = [];
  for (let start = 0; start < floorHits.length; start += pointsPerBatch) {
    const batch = floorHits.slice(start, start + pointsPerBatch);
    results.push(...await independentReferenceBatch(renderer, batch,
      { ...options, seedOffset: (options.seedOffset ?? 0) + start }));
  }
  return results;
}

function directAtFloor(hit: WorldHit, sunVisible: boolean): Vec3 {
  if (!sunVisible) return [0, 0, 0];
  const n = hit.normal, l = sunDirection;
  const v = normalize([eye[0] - hit.position[0], eye[1] - hit.position[1],
    eye[2] - hit.position[2]]);
  const h = normalize(add(l, v));
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

function displayByte(radiance: number): number {
  const x = Math.max(0, radiance); // Exposure EV 0, same filmic fit as resolve.wgsl.
  const mapped = Math.max(0, Math.min(1,
    x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)));
  const encoded = mapped > 0.0031308
    ? 1.055 * mapped ** (1 / 2.4) - 0.055 : 12.92 * mapped;
  return Math.round(Math.max(0, Math.min(1, encoded)) * 255);
}

async function renderReferenceCrop(renderer: Renderer, actualPng: string) {
  const crop = { x: 630, y: 440, width: 310, height: 150, scale: 2 };
  const width = crop.width / crop.scale, height = crop.height / crop.scale;
  if (renderer.stats.width !== 1280 || renderer.stats.height !== 720) {
    throw new Error('GI_FLOOR_REFERENCE_CROP_REQUIRES_1280x720');
  }
  const actualImage = new Image();
  actualImage.src = actualPng;
  await actualImage.decode();
  const actualCanvas = document.createElement('canvas');
  actualCanvas.width = width; actualCanvas.height = height;
  const actualContext = actualCanvas.getContext('2d');
  if (!actualContext) throw new Error('GI_FLOOR_ACTUAL_CROP_CONTEXT');
  actualContext.imageSmoothingEnabled = false;
  actualContext.drawImage(actualImage, crop.x, crop.y, crop.width, crop.height,
    0, 0, width, height);

  const referenceCanvas = document.createElement('canvas');
  referenceCanvas.width = width; referenceCanvas.height = height;
  const context = referenceCanvas.getContext('2d');
  if (!context) throw new Error('GI_FLOOR_REFERENCE_CROP_CONTEXT');
  const image = context.createImageData(width, height);
  const rays: WorldRay[] = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    rays.push(cameraRay({ x: crop.x + x * crop.scale + 1,
      y: crop.y + y * crop.scale + 1 }, 1280, 720));
  }
  const cameraHits = (await renderer.traceRays(rays)).hits;
  const floorPixels = cameraHits.map((hit, pixel) => ({ hit, pixel }))
    .filter((item): item is { hit: WorldHit; pixel: number } =>
      item.hit?.batch === 'room-neutral' && item.hit.entityId === 1
      && item.hit.normal[1] > 0.9);
  // Neutral gray is a mask for non-floor pixels, not a reconstructed material.
  for (let pixel = 0; pixel < width * height; pixel++) {
    image.data.set([80, 80, 80, 255], pixel * 4);
  }
  const tilePixels = 64, samplesPerPixel = 128, bounces = 8;
  let tracedPathSegments = 0;
  for (let start = 0; start < floorPixels.length; start += tilePixels) {
    const tile = floorPixels.slice(start, start + tilePixels);
    const hits = tile.map(entry => entry.hit);
    const estimates = await independentReference(renderer, hits,
      { samples: samplesPerPixel, bounces, seedOffset: start });
    tracedPathSegments += estimates[0]?.perBounceActiveRays?.reduce((sum, value) => sum + value, 0) ?? 0;
    const sunOcclusion = (await renderer.traceRays(hits.map(hit => ({
      origin: add(hit.position, hit.normal, GI_PROBE_GRID.normalBias),
      direction: sunDirection, tMax: GI_PROBE_GRID.maxDistance,
    })))).hits;
    for (let i = 0; i < tile.length; i++) {
      const direct = directAtFloor(hits[i], sunOcclusion[i] === null);
      const diffuse = estimates[i].truncatedEightBounce.mean;
      for (let c = 0; c < 3; c++) {
        image.data[tile[i].pixel * 4 + c] = displayByte(direct[c] + diffuse[c]);
      }
    }
  }
  context.putImageData(image, 0, 0);
  return { crop, width, height, floorPixels: floorPixels.length,
    nonFloorPixels: width * height - floorPixels.length,
    tilePixels, samplesPerPixel, bounces, tracedPathSegments,
    maxWorldQueryBatch: Math.max(width * height, tilePixels * samplesPerPixel),
    nonFloorColor: [80, 80, 80],
    approximation: 'Floor pixels only; independent 8-bounce diffuse paths plus binary world-query sun and CPU GGX direct, EV0 filmic/sRGB. 128 paths per output pixel remain a finite noisy estimate; non-floor pixels are neutral gray. Actual crop is sampled from the rendered GI PNG without smoothing.',
    actualGiCropPng: actualCanvas.toDataURL('image/png'),
    referencePng: referenceCanvas.toDataURL('image/png') };
}

export async function runGiFloorReferenceFixture(renderReferenceImage = false,
  raysPerProbeOverride?: 128 | 256, gridMode?: '8',
  emissiveSamplesOverride?: 0) {
  const denseSpacing: Vec3 = [6.8 / 8, 4.4 / 8, 7.6 / 8];
  const baseGrid = gridMode === '8' ? {
    ...GI_PROBE_GRID,
    origin: [-3.4 + denseSpacing[0] / 2, -0.2 + denseSpacing[1] / 2,
      -3.8 + denseSpacing[2] / 2] as Vec3,
    spacing: denseSpacing, counts: [8, 8, 8] as Vec3, raysPerProbe: 128,
  } : GI_PROBE_GRID;
  const rayGrid = raysPerProbeOverride === undefined ? baseGrid
    : { ...baseGrid, raysPerProbe: raysPerProbeOverride };
  const grid = emissiveSamplesOverride === undefined ? rayGrid
    : { ...rayGrid, emissiveSamplesPerProbe: emissiveSamplesOverride };
  const gridRayCount = grid.counts.reduce((product, count) => product * count, 1)
    * grid.raysPerProbe;
  if (gridRayCount > 65536) throw new RangeError('GI_FLOOR_GRID_RAY_BUDGET');
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:1280px;height:720px;';
  document.body.appendChild(canvas);
  const scene = createGiDemoScene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scene.batches, {
      diffuseGi: grid,
      worldQueries: { maxRays: Math.max(8192, gridRayCount) },
      antiAliasing: 'fxaa',
      sunShadow: { resolution: 2048,
        focusBounds: { min: [-3.5, -0.25, -4], max: [3.5, 4, 4] },
        maxBytes: 16 * 1024 * 1024 },
    });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    renderer.setDiffuseGiEnabled(false);
    renderer.render(eye, target);
    const samples = screenSamples(renderer.stats.width, renderer.stats.height);
    const capture = async () => {
      renderer!.render(eye, target);
      const hdr = await renderer!.probeHdr(samples.map(sample => sample.pixel));
      return { frame: renderer!.stats.frames, hdr, png: canvas.toDataURL('image/png'),
        gi: renderer!.diagnostics().diffuseGi };
    };
    const off = await capture();
    const cameraHits = (await renderer.traceRays(samples.map(sample => sample.ray))).hits;
    renderer.setDiffuseGiEnabled(true);
    renderer.setDiffuseGiMultiBounce(false);
    const single = await capture();
    const singleProbe = await renderer.probeDiffuseGi();
    const selectionOrder = [0, 3, 5, 10, 14, 19,
      ...Array.from({ length: wallArea.length }, (_, index) => index)];
    const selected = selectionOrder.filter((index, position) => selectionOrder.indexOf(index) === position)
      .map(index => ({ index, hit: cameraHits[index] }))
      .filter((item): item is { index: number; hit: WorldHit } =>
        item.hit?.batch === 'room-neutral' && item.hit.entityId === 1 && item.hit.normal[1] > 0.9)
      .slice(0, 6);
    const reference = await independentReference(renderer, selected.map(item => item.hit));
    renderer.setDiffuseGiMultiBounce(true);
    let multi = await capture();
    for (let i = 1; i < 12; i++) multi = await capture();
    const multiProbe = await renderer.probeDiffuseGi();
    const referenceImage = renderReferenceImage ? await renderReferenceCrop(renderer, multi.png) : null;
    renderer.setSun({ direction: sunDirection, color: [0, 0, 0] });
    let sunOffMulti = await capture();
    for (let i = 1; i < 12; i++) sunOffMulti = await capture();
    const sunOffProbe = await renderer.probeDiffuseGi();
    // Keep the strip geometry and its visibility exactly as authored. Only its
    // material stops emitting, so this isolates its transport contribution.
    renderer.setSun({ direction: sunDirection, color: sunColor });
    await renderer.setMaterial('emissive-strip', { definition: {
      ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.constant([0, 0, 0, 1]),
      emissive: m.constant([0, 0, 0, 1]),
      metallic: 0, roughness: 1,
    } });
    renderer.invalidateDiffuseGiHistory();
    let lampOffMulti = await capture();
    for (let i = 1; i < 12; i++) lampOffMulti = await capture();
    const lampOffProbe = await renderer.probeDiffuseGi();
    const lampOffReference = await independentReference(renderer, selected.map(item => item.hit));
    const lampOffStripHit = (await renderer.traceRays([{
      origin: [0, 2.75, -2], direction: [0, 0, -1], tMax: 3,
    }])).hits[0];
    const result = { camera: { eye, target }, sunDirection, sunColor, grid,
      canvas: { width: renderer.stats.width, height: renderer.stats.height },
      samples: samples.map((sample, index) => ({ world: sample.world, ndc: sample.ndc,
        pixel: sample.pixel, cameraHit: cameraHits[index],
        reference: reference.find((_, selection) => selected[selection]?.index === index) ?? null,
        lampOffReference: lampOffReference.find((_, selection) =>
          selected[selection]?.index === index) ?? null })),
      reference: { selectedIndices: selected.map(item => item.index), pathsPerPoint,
        totalReferencePaths: selected.length * pathsPerPoint,
        pointsPerRayBatch: Math.floor(maxReferenceRayBatch / pathsPerPoint),
        pointBatches: Math.ceil(selected.length * pathsPerPoint / maxReferenceRayBatch),
        maxReferenceRayBatch, maxBounces: pathBounces,
        normalBias: GI_PROBE_GRID.normalBias,
        limitation: 'Diffuse only; 12 ray legs maximum, finite 4096-path deterministic hashed samples per point, processed in ray batches of at most 8192. Reported 95% intervals are approximate Monte Carlo indicators, not a bound on the truncated tail.' },
      cases: { off, single, multi, sunOffMulti, lampOffMulti },
      lampAblation: { material: 'black unlit emissive-strip; geometry retained',
        stripHit: lampOffStripHit,
        measuredCurrentMinusLampOff: multi.hdr.map((pixel, index) =>
          pixel.slice(0, 3).map((value, channel) => value - lampOffMulti.hdr[index][channel])),
        referenceCurrentMinusLampOff: selected.map((item, index) => ({ index: item.index,
          oneBounce: reference[index].oneBounce.mean.map((value, channel) =>
            value - lampOffReference[index].oneBounce.mean[channel]),
          fourBounce: reference[index].truncatedFourBounce.mean.map((value, channel) =>
            value - lampOffReference[index].truncatedFourBounce.mean[channel]),
          eightBounce: reference[index].truncatedEightBounce.mean.map((value, channel) =>
            value - lampOffReference[index].truncatedEightBounce.mean[channel]),
          twelveBounce: reference[index].truncatedTwelveBounce.mean.map((value, channel) =>
            value - lampOffReference[index].truncatedTwelveBounce.mean[channel]) })) },
      referenceImage,
      probes: { single: { frame: singleProbe.frame, invalid: singleProbe.invalidProbes },
        multi: { frame: multiProbe.frame, invalid: multiProbe.invalidProbes,
          probes: multiProbe.probes },
        sunOffMulti: { frame: sunOffProbe.frame, invalid: sunOffProbe.invalidProbes },
        lampOffMulti: { frame: lampOffProbe.frame, invalid: lampOffProbe.invalidProbes,
          probes: lampOffProbe.probes } },
      errors: [...renderer.errors] };
    renderer.dispose();
    return { ...result, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
