import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Pixel = { x: number; y: number };

/** A front-facing quad large enough to cover the center through a small move and resize. */
function quad(): MeshData {
  const positions = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
  const vertices = new ArrayBuffer(positions.length * 80);
  const view = new DataView(vertices);
  for (let i = 0; i < positions.length; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, positions[i][axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setUint32(offset + 32, 0, true);
    view.setFloat32(offset + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4,
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function state(x: number): InstanceState {
  return { entityId: 901, model: translation(x, 0, 0), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
}

async function rejection(action: () => unknown): Promise<string> {
  try { await action(); return ''; } catch (error) { return String(error); }
}

/** toDataURL runs synchronously, before another task can acquire a new swapchain image. */
async function capturePresentation(canvas: HTMLCanvasElement, pixel: Pixel): Promise<number[]> {
  const png = canvas.toDataURL('image/png');
  if (!png.startsWith('data:image/png;base64,')) throw new Error('HDR_FIXTURE_PNG_CAPTURE');
  const image = new Image();
  image.src = png;
  await image.decode();
  const copy = document.createElement('canvas');
  copy.width = image.naturalWidth; copy.height = image.naturalHeight;
  const context = copy.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('HDR_FIXTURE_2D_CONTEXT');
  context.drawImage(image, 0, 0);
  return Array.from(context.getImageData(pixel.x, pixel.y, 1, 1).data, component => component / 255);
}

/** Reads the linear attachment and the actual canvas presentation for two exposures. */
export async function runHdrFixture() {
  const batch: SceneBatch = { name: 'hdr-unlit-quad', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [state(0)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.constant([4, 0.5, 0.125, 1]), emissive: m.constant([1, 0.25, 0, 0]) } } };
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  const eye: [number, number, number] = [0, 0, 6];
  const target: [number, number, number] = [0, 0, 0];
  try {
    renderer = await Renderer.create(canvas, [batch]);
    const noFrame = await rejection(() => renderer!.probeHdr([{ x: 0, y: 0 }]));

    renderer.render(eye, target);
    const firstSize = { width: renderer.stats.width, height: renderer.stats.height };
    const firstPoint = { x: Math.floor(firstSize.width / 2), y: Math.floor(firstSize.height / 2) };
    const firstPresentation = capturePresentation(canvas, firstPoint);
    const firstHdr = await renderer.probeHdr([firstPoint]);
    const firstOutput = await firstPresentation;
    const firstBytes = renderer.stats.ownedGpuBytes;
    const materialVariants = renderer.diagnostics().materialVariants;
    const invalidPixel = await rejection(() => renderer!.probeHdr([{ x: -1, y: 0 }]));
    const tooManyPixels = await rejection(() => renderer!.probeHdr(Array.from({ length: 1025 }, () => firstPoint)));

    renderer.setExposure(-2);
    batch.instances = [state(0.2)];
    renderer.render(eye, target);
    const secondPresentation = capturePresentation(canvas, firstPoint);
    const secondHdr = await renderer.probeHdr([firstPoint]);
    const secondOutput = await secondPresentation;
    const velocityAfterExposure = await renderer.probeVelocity([firstPoint]);
    const secondMaterialVariants = renderer.diagnostics().materialVariants;
    const invalidExposures = await Promise.all([
      rejection(() => renderer!.setExposure(Number.NaN)),
      rejection(() => renderer!.setExposure(Number.POSITIVE_INFINITY)),
      rejection(() => renderer!.setExposure(-17)),
      rejection(() => renderer!.setExposure(17)),
    ]);

    canvas.style.width = '320px';
    renderer.render(eye, target);
    const resizedSize = { width: renderer.stats.width, height: renderer.stats.height };
    const resizedPoint = { x: Math.floor(resizedSize.width / 2), y: Math.floor(resizedSize.height / 2) };
    const resizedPresentation = capturePresentation(canvas, resizedPoint);
    const resizedHdr = await renderer.probeHdr([resizedPoint]);
    const resizedOutput = await resizedPresentation;
    const resizedBytes = renderer.stats.ownedGpuBytes;
    const frames = renderer.stats.frames, errors = [...renderer.errors];
    renderer.dispose();
    return { firstHdr, firstOutput, secondHdr, secondOutput, resizedHdr, resizedOutput,
      firstSize, resizedSize, firstBytes, resizedBytes, materialVariants, secondMaterialVariants,
      velocityAfterExposure, noFrame, invalidPixel, tooManyPixels, invalidExposures,
      frames, errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
