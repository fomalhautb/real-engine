import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { TextureAsset } from '../../../src/texture-assets';
import type { WorldRay, WorldQueryResult } from '../../../src/world-query';

type Vec3 = [number, number, number];
type QueryDiagnostics = {
  sceneAcceleration: 'batch-bvh' | 'linear-batches'; sceneBvhNodes: number; sceneCandidateBytes: number;
  instanceAcceleration: 'batch-bvh' | 'linear-root-scan';
  instanceBvhNodes: number; acceleratedBatches: number;
  lastReadbackQuery: { frame: number; rays: number; workUnits: number; triangleTests: number;
    maxWorkUnits: number; overflows: number } | null;
};
const COUNT = 64, DEFORM = 11, CROWD = 9, PARAM = 5, MASK_BACK = 17, MASK_FRONT = 42;
const TIE_FIRST = 21, TIE_SECOND = 54;
const EYE: Vec3 = [0, 0, 42], TARGET: Vec3 = [0, 0, 0];
const CORNERS = [[-0.8, -0.8], [0.8, -0.8], [0, 0.8]] as const;
const name = (index: number) => `scene-batch-${index}`;

function triangle(): MeshData {
  const vertices = new ArrayBuffer(3 * 80), data = new DataView(vertices);
  CORNERS.forEach(([x, y], i) => {
    const at = i * 80;
    data.setFloat32(at, x, true); data.setFloat32(at + 4, y, true);
    data.setFloat32(at + 12, 1, true); data.setFloat32(at + 24, 1, true);
    data.setUint32(at + 32, 0, true); data.setFloat32(at + 48, 1, true);
  });
  return { vertices, vertexCount: 3, indices: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0.5, 1]),
    bounds: { min: [-0.8, -0.8, 0], max: [0.8, 0.8, 0] } };
}
function grid(index: number): [number, number] {
  const cell = index * 37 % COUNT;
  return [(cell % 8 - 3.5) * 2.4, (Math.floor(cell / 8) - 3.5) * 2.4];
}
function location(index: number): [number, number, number] {
  if (index === MASK_BACK || index === MASK_FRONT) return [-18.5, 0, index === MASK_FRONT ? 1 : 0];
  if (index === TIE_FIRST) return [18.3, 0, 0];
  if (index === TIE_SECOND) return [18, 0, 0];
  return [...grid(index), 0];
}
function deformedCenter(moved: boolean): [number, number] {
  const boneY = moved ? 3.2 : 0.2, weight = moved ? 0.4 : 1, phase = moved ? 1.1 : 0.3;
  const bottomWave = 0.05 * Math.sin((boneY - 0.8) * 2 + phase);
  const topWave = 0.05 * Math.sin((boneY + 0.8) * 2 + phase);
  return [0.1 * weight + 0.5 * (bottomWave + topWave), boneY];
}
function pose(index: number, generation = 1, moved = false, crowdSlot = -1): InstanceState {
  const [x, y, z] = crowdSlot >= 0 ? [25 + crowdSlot % 4 * 2.4, 22 + Math.floor(crowdSlot / 4) * 2.4, 0]
    : moved && index === DEFORM ? [24, 4, 0] : location(index);
  const scale = index === TIE_FIRST ? 2 : 1;
  const model = multiply(translation(x, y, z), scaling(scale, scale, 1));
  const deformed = index === DEFORM;
  return { entityId: crowdSlot >= 0 ? 5000 + crowdSlot : 1000 + index,
    entityGeneration: generation, historyVersion: generation - 1,
    model, normalMatrix: normalMatrix(model),
    bones: deformed ? [translation(0, moved ? 3.2 : 0.2, 0)] : [],
    deformScalars: deformed ? (moved ? [0.4, 1.1] : [1, 0.3]) : [],
    color: [1, 1, 1, 1], metalness: 0.1, roughness: 0.6 };
}
function maskTexture(): TextureAsset {
  const pixels = new Uint8Array(3 * 3 * 4);
  for (let i = 0; i < 9; i++) pixels.set([255, 255, 255, i === 4 ? 0 : 255], i * 4);
  return { width: 3, height: 3, colorSpace: 'srgb', pixels };
}
function batches(mesh: MeshData): SceneBatch[] {
  const morphPositions = new Float32Array(9);
  for (let i = 0; i < 3; i++) morphPositions[i * 3] = 0.1;
  return Array.from({ length: COUNT }, (_, index): SceneBatch => {
    const color = new Float32Array([0.25 + index % 4 * 0.14, 0.3 + index % 5 * 0.1, 0.5, 1]);
    const material = index === MASK_FRONT ? {
      definition: { ...DEFAULT_MATERIAL_DEFINITION, alphaMode: 'mask' as const,
        baseColor: m.texture(0), alphaCutoff: 0.5 },
      textures: [maskTexture()],
      samplers: [{ addressModeU: 'clamp-to-edge' as const, addressModeV: 'clamp-to-edge' as const,
        minFilter: 'nearest' as const, magFilter: 'nearest' as const, mipmapFilter: 'nearest' as const }],
    } : { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.parameter(0),
      emissive: m.constant([0.2, 0.08, 0.04, 1]) }, parameters: color };
    return { name: name(index), mesh, capacity: index === CROWD ? 8 : 1, inputMode: 'committed',
      queryIdentity: { assetId: 'scene-query-triangle', primitiveId: index, topologyVersion: 1 },
      instances: index === CROWD ? Array.from({ length: 8 }, (_, slot) => pose(index, 1, false, slot)) : [pose(index)],
      deformation: index === DEFORM ? { jointCount: 1, nodes: [
        { kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' },
        { kind: 'wave', amplitude: 0.05, frequency: 2, phase: 1 },
      ] } : { jointCount: 0, nodes: [] },
      morphTargets: index === DEFORM ? [{ positions: morphPositions, normals: new Float32Array(9) }] : [],
      material };
  });
}
function ray(x: number, y: number, ignore?: WorldRay['ignore']): WorldRay {
  return { origin: [x, y, 5], direction: [0, 0, -1], ...(ignore ? { ignore } : {}) };
}
function batchRay(index: number): WorldRay {
  const [x, y] = location(index);
  return index === DEFORM ? ray(x + deformedCenter(false)[0], y + deformedCenter(false)[1]) : ray(x, y);
}
function movedRay(): WorldRay {
  const [x, y] = deformedCenter(true);
  return ray(24 + x, 4 + y);
}
const MISS = ray(100, 100);
function initialRays(): WorldRay[] {
  return [batchRay(0), batchRay(PARAM), batchRay(DEFORM), ray(25, 22),
    ray(-18.5, 0), ray(-18.5, 0.55), ray(18, 0), MISS,
    ray(location(0)[0], location(0)[1], { entityId: 1000, generation: 1 }),
    ray(location(0)[0], location(0)[1], { entityId: 1000, generation: 2 }),
    batchRay(63)];
}
function diagnostics(renderer: Renderer): QueryDiagnostics {
  const value = renderer.diagnostics().worldQueries as QueryDiagnostics | null;
  if (!value) throw new Error('SCENE_QUERY_DIAGNOSTICS_MISSING');
  return value;
}
type Stage = { accelerated: WorldQueryResult; linear: WorldQueryResult;
  acceleratedDiagnostics: QueryDiagnostics; linearDiagnostics: QueryDiagnostics };
async function capture(accelerated: Renderer, linear: Renderer, rays: WorldRay[], render = true): Promise<Stage> {
  if (render) { accelerated.render(EYE, TARGET); linear.render(EYE, TARGET); }
  const a = await accelerated.traceRays(rays, { textureLod: 0 });
  const b = await linear.traceRays(rays, { textureLod: 0 });
  return { accelerated: a, linear: b,
    acceleratedDiagnostics: diagnostics(accelerated), linearDiagnostics: diagnostics(linear) };
}
async function image(canvas: HTMLCanvasElement): Promise<{ node: HTMLImageElement; visiblePixels: number }> {
  const node = document.createElement('img'); node.src = canvas.toDataURL('image/png'); await node.decode();
  const check = document.createElement('canvas'); check.width = canvas.width; check.height = canvas.height;
  const context = check.getContext('2d')!; context.drawImage(node, 0, 0);
  const pixels = context.getImageData(0, 0, check.width, check.height).data;
  let visiblePixels = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] && pixels[i] + pixels[i + 1] + pixels[i + 2] > 80) visiblePixels++;
  }
  return { node, visiblePixels };
}
async function rejection(operation: () => Promise<unknown>): Promise<string> {
  try { await operation(); return ''; } catch (error) { return String(error); }
}

export async function runWorldQuerySceneFixture(instanceAcceleration: 'bvh' | 'linear' = 'linear') {
  const canvases = [document.querySelector<HTMLCanvasElement>('#accelerated')!,
    document.querySelector<HTMLCanvasElement>('#linear')!];
  const mesh = triangle(), scenes = [batches(mesh), batches(mesh)];
  const options = { maxRays: 512, maxTrianglesPerMesh: 1, maxInstancesPerBatch: 8,
    instanceAcceleration };
  const gi = { origin: [-8.4, -8.4, 0.05] as Vec3, spacing: [2.4, 2.4, 0.5] as Vec3,
    counts: [2, 2, 2] as Vec3, raysPerProbe: 64, maxDistance: 5,
    skyRadiance: [0, 0, 0] as Vec3 };
  const overflowCanvas = document.createElement('canvas'); overflowCanvas.style.display = 'none';
  document.body.appendChild(overflowCanvas);
  let accelerated: Renderer | undefined, linear: Renderer | undefined, overflowRenderer: Renderer | undefined;
  let acceleratedAfterDispose = -1, linearAfterDispose = -1, overflowAfterDispose = -1;
  try {
    accelerated = await Renderer.create(canvases[0], scenes[0], { worldQueries: { ...options,
      sceneAcceleration: 'bvh' }, diffuseGi: gi });
    linear = await Renderer.create(canvases[1], scenes[1], { worldQueries: { ...options,
      sceneAcceleration: 'linear' }, diffuseGi: gi });
    const initial = await capture(accelerated, linear, initialRays());
    const pictureEqual = canvases[0].toDataURL('image/png') === canvases[1].toDataURL('image/png');
    const steady = await capture(accelerated, linear, initialRays(), false);
    const staticFrame = await capture(accelerated, linear, initialRays());
    const acceleratedGi = await accelerated.probeDiffuseGi(), linearGi = await linear.probeDiffuseGi();
    accelerated.setDiffuseGiEnabled(false); linear.setDiffuseGiEnabled(false);

    for (const scene of scenes) scene[DEFORM].instances = [pose(DEFORM, 1, true)];
    const moved = await capture(accelerated, linear, [movedRay(), batchRay(DEFORM), batchRay(0),
      ray(-18.5, 0), ray(18, 0)]);
    for (const scene of scenes) scene[DEFORM].instances = [];
    const empty = await capture(accelerated, linear, [movedRay(), batchRay(DEFORM), batchRay(0)]);
    for (const scene of scenes) scene[DEFORM].instances = [pose(DEFORM, 2, true)];
    const restored = await capture(accelerated, linear, [movedRay(),
      { ...movedRay(), ignore: { entityId: 1000 + DEFORM, generation: 1 } },
      { ...movedRay(), ignore: { entityId: 1000 + DEFORM, generation: 2 } }, batchRay(0)]);

    const beforeMutationFrame = restored.accelerated.frame;
    for (const renderer of [accelerated, linear]) {
      renderer.setMaterialParameters(name(PARAM), new Float32Array([0.9, 0.1, 0.3, 1]));
      await renderer.setMaterial(name(MASK_FRONT), { definition: {
        ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([0.8, 0.15, 0.1, 1]),
      } });
    }
    const materialChanged = await capture(accelerated, linear, [batchRay(PARAM), ray(-18.5, 0)], false);

    accelerated.render(EYE, TARGET); linear.render(EYE, TARGET);
    const images = await Promise.all(canvases.map(image));
    const visiblePixels = images.map(item => item.visiblePixels);
    canvases[0].replaceWith(images[0].node); canvases[1].replaceWith(images[1].node);
    const adapter = accelerated.info.adapterInfo;
    const errors = { accelerated: [...accelerated.errors], linear: [...linear.errors] };
    accelerated.dispose(); acceleratedAfterDispose = accelerated.stats.ownedGpuBytes; accelerated = undefined;
    linear.dispose(); linearAfterDispose = linear.stats.ownedGpuBytes; linear = undefined;

    overflowRenderer = await Renderer.create(overflowCanvas, batches(mesh).slice(0, 8), { worldQueries: {
      ...options, maxRays: 1, maxNodeVisits: 1, maxTotalWorkUnits: 1, sceneAcceleration: 'bvh',
    } });
    overflowRenderer.render(EYE, TARGET);
    const overflow = await rejection(() => overflowRenderer!.traceRays([batchRay(0)]));
    const overflowDiagnostics = diagnostics(overflowRenderer);
    const overflowErrors = [...overflowRenderer.errors];
    overflowRenderer.dispose(); overflowAfterDispose = overflowRenderer.stats.ownedGpuBytes; overflowRenderer = undefined;

    return { adapter, instanceAcceleration, initial, steady, staticFrame, moved, empty, restored, materialChanged,
      acceleratedGi, linearGi, pictureEqual, visiblePixels, beforeMutationFrame,
      initialDeformedPosition: [location(DEFORM)[0] + deformedCenter(false)[0],
        location(DEFORM)[1] + deformedCenter(false)[1], 0] as Vec3,
      movedPosition: [24 + deformedCenter(true)[0], 4 + deformedCenter(true)[1], 0] as Vec3,
      overflow, overflowDiagnostics, overflowErrors, errors,
      acceleratedAfterDispose, linearAfterDispose, overflowAfterDispose };
  } finally {
    accelerated?.dispose(); linear?.dispose(); overflowRenderer?.dispose(); overflowCanvas.remove();
  }
}
