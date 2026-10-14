import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
type Pixel = { x: number; y: number };
export type AntiAliasingOrientation = 'diagonal' | 'near-vertical' | 'near-horizontal';

const SIZE = 256;
const SCALE = 4;
const eye: Vec3 = [0, 0, 6];
const target: Vec3 = [0, 0, 0];

function mesh(points: readonly Vec3[], indices: readonly number[]): MeshData {
  const vertices = new ArrayBuffer(points.length * 80);
  const view = new DataView(vertices);
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < points.length; index++) {
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(index * 80 + axis * 4, points[index][axis], true);
      min[axis] = Math.min(min[axis], points[index][axis]);
      max[axis] = Math.max(max[axis], points[index][axis]);
    }
    view.setFloat32(index * 80 + 12, 1, true);
    view.setFloat32(index * 80 + 24, 1, true);
    view.setUint32(index * 80 + 32, 0, true);
    view.setFloat32(index * 80 + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array(indices), vertexCount: points.length, bounds: { min, max } };
}

function batch(name: string, id: number, geometry: MeshData, color: [number, number, number, number]): SceneBatch {
  const state: InstanceState = { entityId: id, model: identity(), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
  return { name, mesh: geometry, capacity: 1, inputMode: 'committed', instances: [state],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.constant(color) } } };
}

function blueHalfPlane(orientation: AntiAliasingOrientation): MeshData {
  const [normalX, normalY, threshold] = orientation === 'diagonal' ? [0.8, 1, -1]
    : orientation === 'near-vertical' ? [1, 0.18, 0] : [0.18, 1, 0];
  const square: Vec3[] = [[-5, -5, 0.05], [5, -5, 0.05], [5, 5, 0.05], [-5, 5, 0.05]];
  const distance = (point: Vec3) => normalX * point[0] + normalY * point[1] - threshold;
  // Clip one square against the authored half-plane. This makes the visible
  // edge tilt while keeping the foreground valid all the way to the viewport.
  const polygon: Vec3[] = [];
  let previous = square[square.length - 1];
  let previousDistance = distance(previous);
  for (const current of square) {
    const currentDistance = distance(current);
    if ((previousDistance <= 0) !== (currentDistance <= 0)) {
      const fraction = previousDistance / (previousDistance - currentDistance);
      polygon.push([previous[0] + fraction * (current[0] - previous[0]),
        previous[1] + fraction * (current[1] - previous[1]), 0.05]);
    }
    if (currentDistance <= 0) polygon.push(current);
    previous = current;
    previousDistance = currentDistance;
  }
  const indices: number[] = [];
  for (let i = 1; i + 1 < polygon.length; i++) indices.push(0, i, i + 1);
  return mesh(polygon, indices);
}

function flatPatches(orientation: AntiAliasingOrientation): Pixel[] {
  return orientation === 'near-vertical' ? [{ x: 32, y: 128 }, { x: 224, y: 128 }]
    : [{ x: 128, y: 32 }, { x: 128, y: 224 }];
}

function scene(orientation: AntiAliasingOrientation): SceneBatch[] {
  const white = mesh([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], [0, 1, 2, 0, 2, 3]);
  return [batch('aa-white-background', 950, white, [2, 2, 2, 1]),
    batch('aa-blue-slope', 951, blueHalfPlane(orientation), [0.015, 0.1, 1.5, 1])];
}

/** Capture the presented pixels before the next frame acquires a canvas texture. */
async function capture(canvas: HTMLCanvasElement): Promise<Uint8ClampedArray> {
  const png = canvas.toDataURL('image/png');
  if (!png.startsWith('data:image/png;base64,')) throw new Error('AA_PNG_CAPTURE');
  const image = new Image();
  image.src = png;
  await image.decode();
  const copy = document.createElement('canvas');
  copy.width = image.naturalWidth; copy.height = image.naturalHeight;
  const context = copy.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('AA_2D_CONTEXT');
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, copy.width, copy.height).data;
}

function same(a: Uint8ClampedArray, b: Uint8ClampedArray): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function offset(width: number, x: number, y: number): number { return (y * width + x) * 4; }

function measure(none: Uint8ClampedArray, fxaa: Uint8ClampedArray, reference: Uint8ClampedArray) {
  let edgeCount = 0, noneError = 0, fxaaError = 0;
  for (let y = 12; y < SIZE - 12; y++) for (let x = 12; x < SIZE - 12; x++) {
    const average = [0, 0, 0];
    let blue = 0, white = 0;
    for (let sy = 0; sy < SCALE; sy++) for (let sx = 0; sx < SCALE; sx++) {
      const high = offset(SIZE * SCALE, x * SCALE + sx, y * SCALE + sy);
      if (reference[high] < 128) blue++; else white++;
      for (let channel = 0; channel < 3; channel++) average[channel] += reference[high + channel] / (SCALE * SCALE);
    }
    if (blue === 0 || white === 0) continue;
    edgeCount++;
    const low = offset(SIZE, x, y);
    for (let channel = 0; channel < 3; channel++) {
      noneError += Math.abs(none[low + channel] - average[channel]);
      fxaaError += Math.abs(fxaa[low + channel] - average[channel]);
    }
  }
  return { edgeCount, noneError: noneError / Math.max(edgeCount, 1),
    fxaaError: fxaaError / Math.max(edgeCount, 1) };
}

function flatRegionPreserved(none: Uint8ClampedArray, fxaa: Uint8ClampedArray,
  patches: readonly Pixel[]): boolean {
  // Both patches lie well away from the authored boundary and canvas edge.
  for (const patch of patches) {
    for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
      const i = offset(SIZE, patch.x + dx, patch.y + dy);
      for (let channel = 0; channel < 4; channel++) if (none[i + channel] !== fxaa[i + channel]) return false;
    }
  }
  return true;
}

function borderValid(fxaa: Uint8ClampedArray): boolean {
  const border: Pixel[] = [{ x: 0, y: 0 }, { x: SIZE - 1, y: 0 },
    { x: 0, y: SIZE - 1 }, { x: SIZE - 1, y: SIZE - 1 },
    { x: 0, y: 64 }, { x: 0, y: 65 }, { x: SIZE - 1, y: 128 }];
  return border.every(({ x, y }) => {
    const i = offset(SIZE, x, y);
    return fxaa[i + 3] === 255 && Math.max(fxaa[i], fxaa[i + 1], fxaa[i + 2]) > 100;
  });
}

export async function runAntiAliasingFixture(orientation: AntiAliasingOrientation = 'diagonal') {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px;`;
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scene(orientation), { antiAliasing: 'none' });
    const patches = flatPatches(orientation);
    const probe: Pixel[] = patches;
    renderer.render(eye, target);
    const lowSize = { width: renderer.stats.width, height: renderer.stats.height };
    const none = await capture(canvas);
    const hdrNone = await renderer.probeHdr(probe);
    const lowBytes = renderer.stats.ownedGpuBytes;

    renderer.setAntiAliasing('fxaa');
    renderer.render(eye, target);
    const fxaa = await capture(canvas);
    const hdrFxaa = await renderer.probeHdr(probe);
    const fxaaMode = renderer.diagnostics().output.antiAliasing;

    let invalidError = '';
    try { renderer.setAntiAliasing('bogus' as 'fxaa'); }
    catch (error) { invalidError = String(error); }
    const modeAfterInvalid = renderer.diagnostics().output.antiAliasing;
    renderer.render(eye, target);
    const afterInvalid = await capture(canvas);

    renderer.setAntiAliasing('none');
    renderer.render(eye, target);
    const noneAgain = await capture(canvas);

    canvas.style.width = `${SIZE * SCALE}px`;
    canvas.style.height = `${SIZE * SCALE}px`;
    renderer.render(eye, target);
    const highSize = { width: renderer.stats.width, height: renderer.stats.height };
    const reference = await capture(canvas);
    const highBytes = renderer.stats.ownedGpuBytes;

    canvas.style.width = `${SIZE}px`;
    canvas.style.height = `${SIZE}px`;
    renderer.setAntiAliasing('fxaa');
    renderer.render(eye, target);
    const resizedBack = await capture(canvas);
    const finalSize = { width: renderer.stats.width, height: renderer.stats.height };
    const finalBytes = renderer.stats.ownedGpuBytes;
    const result = { orientation, ...measure(none, fxaa, reference), lowSize, highSize, finalSize,
      flatPreserved: flatRegionPreserved(none, fxaa, patches), borderValid: borderValid(fxaa),
      noneRoundtripSame: same(none, noneAgain), fxaaAfterInvalidSame: same(fxaa, afterInvalid),
      fxaaAfterResizeSame: same(fxaa, resizedBack), hdrNone, hdrFxaa, lowBytes, highBytes, finalBytes,
      fxaaMode, modeAfterInvalid, invalidError, errors: [...renderer.errors], frames: renderer.stats.frames };
    renderer.dispose();
    return { ...result, bytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
