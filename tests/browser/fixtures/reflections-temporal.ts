import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

export const LADDER_SIZE = 48;
export const LADDER_EYE: [number, number, number] = [0, 0, 3];
const TARGET: [number, number, number] = [0, 0, 0];
/** Mirror strips, left to right; metal=1 and white base make Fresnel exactly one. */
export const LADDER_ROUGHNESS = [0.045, 0.1, 0.2, 0.35, 0.5, 0.7] as const;
/** Unlit bars on the plane z = 5 behind the camera: [yMin, yMax, rgb]. */
export const LADDER_BARS: readonly { y: [number, number]; color: [number, number, number] }[] = [
  { y: [0.25, 0.8], color: [3, 0.3, 0.2] },
  { y: [-0.3, 0.1], color: [0.3, 2.5, 0.4] },
  { y: [-1.0, -0.55], color: [0.3, 0.5, 3] },
];
export const LADDER_SKY: [number, number, number] = [0.05, 0.05, 0.08];

export function quad(): MeshData {
  const points = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  points.forEach((point, index) => {
    const offset = index * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, point[axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setFloat32(offset + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

export function instance(entityId: number, model: Float32Array, roughness: number, metalness = 1,
  color: [number, number, number, number] = [1, 1, 1, 1]): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color, metalness, roughness, historyVersion: 0 };
}

/** One strip per roughness; the visible half-width at z=0 is about 1.6 m. */
export function ladderStrip(index: number): Float32Array {
  const width = 3.3 / LADDER_ROUGHNESS.length;
  return multiply(translation(-1.65 + width * (index + 0.5), 0, 0), scaling(width / 2, 2, 1));
}

/** Pixel ROIs in each strip, away from strip edges: 4 columns x 12 rows. */
export function ladderRois(): { strip: number; pixels: { x: number; y: number }[] }[] {
  const width = LADDER_SIZE / LADDER_ROUGHNESS.length;
  return LADDER_ROUGHNESS.map((_, strip) => {
    const pixels = [];
    for (let y = 18; y < 30; y++) for (let x = 0; x < 4; x++) pixels.push({ x: Math.round(strip * width + 2) + x, y });
    return { strip, pixels };
  });
}

/** Static roughness ladder: fixed samples, exact accumulation steps, then 120 measured frames. */
export async function runReflectionsLadderFixture(options: { warmup: number; measured: number }) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${LADDER_SIZE}px;height:${LADDER_SIZE}px`;
  document.body.appendChild(canvas);
  const mirror: SceneBatch = { name: 'ladder-mirror', mesh: quad(), capacity: LADDER_ROUGHNESS.length,
    inputMode: 'committed', instances: LADDER_ROUGHNESS.map((roughness, index) => instance(10 + index, ladderStrip(index), roughness)),
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } } };
  const bars: SceneBatch = { name: 'ladder-offscreen-bars', mesh: quad(), capacity: LADDER_BARS.length,
    inputMode: 'committed', instances: LADDER_BARS.map((bar, index) => instance(20 + index,
      multiply(translation(0, (bar.y[0] + bar.y[1]) / 2, 5), scaling(20, (bar.y[1] - bar.y[0]) / 2, 1)), 1, 0,
      [...bar.color, 1])),
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.instanceColor } } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [mirror, bars], { antiAliasing: 'none',
      worldQueries: { maxRays: LADDER_SIZE * LADDER_SIZE * 4 },
      reflections: { pixelSpacing: 1, maxAnchors: LADDER_SIZE * LADDER_SIZE, samplesPerAnchor: 4, maxDistance: 20,
        maxBytes: 16 * 1024 * 1024, skyRadiance: LADDER_SKY, temporal: { minHistoryFrames: 8, maxHistoryFrames: 32 } } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const rois = ladderRois();
    const anchors = rois.flatMap(roi => roi.pixels.map(pixel => pixel.y * LADDER_SIZE + pixel.x));
    active.setReflectionsTemporalEnabled(false);
    active.render(LADDER_EYE, TARGET);
    const fixed = await active.probeReflections(anchors);
    const fixedHdr = await active.probeSurfaceHdr(rois.flatMap(roi => roi.pixels));
    active.setReflectionsTemporalEnabled(true);
    // Exact accumulation: a handful of anchors per strip across the first frames.
    const traced = rois.map(roi => anchors[rois.indexOf(roi) * roi.pixels.length + 21]);
    const steps = [];
    for (let frame = 0; frame < 6; frame++) {
      active.render(LADDER_EYE, TARGET);
      steps.push({ ...(await active.probeReflections(traced)), history: active.diagnostics().reflections?.temporal });
    }
    for (let frame = 6; frame < options.warmup; frame++) active.render(LADDER_EYE, TARGET);
    const series: number[][][] = [];
    for (let frame = 0; frame < options.measured; frame++) {
      active.render(LADDER_EYE, TARGET);
      series.push(await active.probeSurfaceHdr(rois.flatMap(roi => roi.pixels)));
    }
    const converged = await active.probeReflections(anchors);
    const diagnostics = active.diagnostics().reflections;
    const errors = [...active.errors];
    active.dispose();
    return { adapter: active.info.adapterInfo, eye: LADDER_EYE, size: LADDER_SIZE, roughness: LADDER_ROUGHNESS,
      bars: LADDER_BARS, sky: LADDER_SKY, rois, anchors, traced, fixed, fixedHdr, steps, series, converged, diagnostics,
      errors, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Camera strafes vertically over the ladder: reflections slide across the
 * mirror at a different screen velocity than the surface itself. */
export async function runReflectionsParallaxFixture(options: { parallaxRoughness: number | null; settle: number;
  frames: number; step: number }) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${LADDER_SIZE}px;height:${LADDER_SIZE}px`;
  document.body.appendChild(canvas);
  const width = 3.3 / LADDER_ROUGHNESS.length;
  const mirror: SceneBatch = { name: 'parallax-mirror', mesh: quad(), capacity: LADDER_ROUGHNESS.length,
    inputMode: 'committed', instances: LADDER_ROUGHNESS.map((roughness, index) => instance(10 + index,
      multiply(translation(-1.65 + width * (index + 0.5), 0, 0), scaling(width / 2, 3, 1)), roughness)),
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } } };
  const bars: SceneBatch = { name: 'parallax-offscreen-bars', mesh: quad(), capacity: LADDER_BARS.length,
    inputMode: 'committed', instances: LADDER_BARS.map((bar, index) => instance(20 + index,
      multiply(translation(0, (bar.y[0] + bar.y[1]) / 2, 5), scaling(20, (bar.y[1] - bar.y[0]) / 2, 1)), 1, 0,
      [...bar.color, 1])),
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.instanceColor } } };
  let renderer: Renderer | undefined;
  try {
    const temporal = options.parallaxRoughness === null ? undefined
      : { minHistoryFrames: 8, maxHistoryFrames: 32, parallaxRoughness: options.parallaxRoughness };
    renderer = await Renderer.create(canvas, [mirror, bars], { antiAliasing: 'none',
      worldQueries: { maxRays: LADDER_SIZE * LADDER_SIZE * 4 },
      reflections: { pixelSpacing: 1, maxAnchors: LADDER_SIZE * LADDER_SIZE, samplesPerAnchor: 4, maxDistance: 20,
        maxBytes: 16 * 1024 * 1024, skyRadiance: LADDER_SKY, temporal } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const rois = ladderRois();
    const anchors = rois.flatMap(roi => roi.pixels.map(pixel => pixel.y * LADDER_SIZE + pixel.x));
    const start = -options.step * options.frames / 2;
    const eyeAt = (y: number): [number, number, number] => [0, y, 3];
    for (let frame = 0; frame < options.settle; frame++) active.render(eyeAt(start), [0, start, 0]);
    const frames = [];
    for (let frame = 0; frame < options.frames; frame++) {
      const y = start + options.step * (frame + 1);
      active.render(eyeAt(y), [0, y, 0]);
      const probe = await active.probeReflections(anchors);
      frames.push({ eye: eyeAt(y), anchors: probe.anchors.map(anchor => ({ position: anchor.position, normal: anchor.normal,
        valid: anchor.valid, radiance: anchor.radiance, raw: anchor.rawRadiance, mode: anchor.historyMode,
        count: anchor.historyCount, clamped: anchor.historyClamped, rejected: anchor.rejectedHistory })) });
    }
    const diagnostics = active.diagnostics().reflections;
    const errors = [...active.errors];
    active.dispose();
    return { adapter: active.info.adapterInfo, parallaxRoughness: options.parallaxRoughness, roughness: LADDER_ROUGHNESS,
      perStrip: rois[0].pixels.length, bars: LADDER_BARS, sky: LADDER_SKY, frames, diagnostics, errors,
      ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
