import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { compileDeformation, conservativeDeformationBounds, validateMorphTargets } from '../../../src/deformation';
import type { MeshData } from '../../../src/geometry';
import { orderInstanceBounds } from '../../../src/instance-bvh';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldRay, WorldQueryResult } from '../../../src/world-query';

type Vec3 = [number, number, number];
type Mode = 'bvh' | 'linear';
type QueryDiagnostics = {
  instanceAcceleration: 'batch-bvh' | 'linear-root-scan';
  instanceBvhNodes: number;
  acceleratedBatches: number;
  lastReadbackQuery: { frame: number; rays: number; workUnits: number; triangleTests: number;
    maxWorkUnits: number; overflows: number } | null;
};
const CAMERA: Vec3 = [0, 0, 46];
const TARGET: Vec3 = [0, 0, 0];
const CORNERS = [[-0.8, -0.8], [0.8, -0.8], [0, 0.8]] as const;
const MOVED_ID = 1200;
const OVERLAP_FIRST = 9001;
const OVERLAP_SECOND = 9002;

function triangle(): MeshData {
  const vertices = new ArrayBuffer(3 * 80), view = new DataView(vertices);
  CORNERS.forEach(([x, y], i) => {
    const offset = i * 80;
    view.setFloat32(offset, x, true); view.setFloat32(offset + 4, y, true);
    view.setFloat32(offset + 12, 1, true); view.setFloat32(offset + 24, 1, true);
    view.setUint32(offset + 32, 0, true); view.setFloat32(offset + 48, 1, true);
  });
  return { vertices, vertexCount: 3, indices: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0.5, 1]),
    bounds: { min: [-0.8, -0.8, 0], max: [0.8, 0.8, 0] } };
}

/** Analytic centroid of authored Morph -> skin -> wave triangle, independent of GPU traversal. */
function localHit(moved = false): [number, number] {
  let x = 0, y = 0;
  for (const [px, py] of CORNERS) {
    const skinnedY = py + (moved ? 3.2 : 0.2);
    x += px + 0.1 * (moved ? 0.4 : 1) + 0.05 * Math.sin(skinnedY * 2 + (moved ? 1.1 : 0.3));
    y += skinnedY;
  }
  return [x / 3, y / 3];
}

function cell(id: number): [number, number] {
  const index = id - 1000;
  return [(index % 16 - 7.5) * 2, (Math.floor(index / 16) - 7.5) * 2];
}
function pose(id: number, generation = 1, moved = false): InstanceState {
  const [x, y] = id === OVERLAP_FIRST ? [19.3, 0] : id === OVERLAP_SECOND ? [19, 0]
    : moved && id === MOVED_ID ? [23, 6] : cell(id);
  const scale = id === OVERLAP_FIRST ? 2 : 1;
  const model = multiply(translation(x, y, 0), scaling(scale, scale, 1));
  const deformedMove = moved && id === MOVED_ID;
  return { entityId: id, entityGeneration: generation, historyVersion: generation - 1,
    model, normalMatrix: normalMatrix(model), bones: [translation(0, deformedMove ? 3.2 : 0.2, 0)],
    deformScalars: deformedMove ? [0.4, 1.1] : [1, 0.3],
    color: [0.65, 0.75, 0.9, 1], metalness: 0.1, roughness: 0.6 };
}
function instances(moved = false, generation = 1): InstanceState[] {
  // Multiplication by 73 permutes 0..255: adjacent slots are spatially distant.
  const result = Array.from({ length: 256 }, (_, slot) => pose(1000 + (slot * 73) % 256,
    1000 + (slot * 73) % 256 === MOVED_ID ? generation : 1, moved));
  result.push(pose(OVERLAP_FIRST), pose(OVERLAP_SECOND));
  return result;
}
function scene(mesh: MeshData, name: string): SceneBatch {
  const morph = new Float32Array(9);
  for (let i = 0; i < 3; i++) morph[i * 3] = 0.1;
  return { name, mesh, capacity: 258, inputMode: 'committed', instances: instances(),
    queryIdentity: { assetId: 'world-query-instances', primitiveId: 0, topologyVersion: 1 },
    deformation: { jointCount: 1, nodes: [
      { kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' },
      { kind: 'wave', amplitude: 0.05, frequency: 2, phase: 1 },
    ] }, morphTargets: [{ positions: morph, normals: new Float32Array(9) }] };
}

function rayFor(id: number, moved = false, ignore?: WorldRay['ignore']): WorldRay {
  const [x, y] = id === OVERLAP_SECOND ? [19, 0] : moved && id === MOVED_ID ? [23, 6] : cell(id);
  const [localX, localY] = localHit(moved && id === MOVED_ID);
  // The overlap ray starts exactly on the common z=0 plane. A zero numerator
  // gives an exact distance tie on every adapter; coplanar triangles at a
  // nonzero distance can differ by one ULP after independent f32 divisions.
  return { origin: [x + localX, y + localY, id === OVERLAP_SECOND ? 0 : 5],
    direction: [0, 0, -1], ...(id === OVERLAP_SECOND ? { tMin: 0 } : {}), ...(ignore ? { ignore } : {}) };
}
const MISS: WorldRay = { origin: [100, 100, 5], direction: [0, 0, -1] };
function initialRays(): WorldRay[] {
  const ids = [1000, 1001, 1017, 1063, 1127, MOVED_ID, 1255];
  return [...ids.map(id => rayFor(id)), MISS,
    rayFor(1000, false, { entityId: 1000, generation: 1 }),
    rayFor(1000, false, { entityId: 1000, generation: 2 }),
    rayFor(OVERLAP_SECOND)];
}

type Stage = { accelerated: WorldQueryResult; linear: WorldQueryResult;
  acceleratedDiagnostics: QueryDiagnostics; linearDiagnostics: QueryDiagnostics };
function queryDiagnostics(renderer: Renderer): QueryDiagnostics {
  const result = renderer.diagnostics().worldQueries as QueryDiagnostics | null;
  if (!result) throw new Error('INSTANCE_BVH_QUERY_DIAGNOSTICS_MISSING');
  return result;
}
async function capture(accelerated: Renderer, linear: Renderer, rays: WorldRay[]): Promise<Stage> {
  accelerated.render(CAMERA, TARGET); linear.render(CAMERA, TARGET);
  const a = await accelerated.traceRays(rays), b = await linear.traceRays(rays);
  return { accelerated: a, linear: b,
    acceleratedDiagnostics: queryDiagnostics(accelerated), linearDiagnostics: queryDiagnostics(linear) };
}
async function rejection(operation: () => Promise<unknown>): Promise<string> {
  try { await operation(); return ''; } catch (error) { return String(error); }
}
async function stillImage(canvas: HTMLCanvasElement): Promise<{ image: HTMLImageElement; visiblePixels: number }> {
  const image = document.createElement('img');
  image.src = canvas.toDataURL('image/png');
  await image.decode();
  const check = document.createElement('canvas'); check.width = canvas.width; check.height = canvas.height;
  const context = check.getContext('2d')!; context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, check.width, check.height).data;
  let visiblePixels = 0;
  for (let at = 0; at < pixels.length; at += 4) {
    if (pixels[at + 3] && pixels[at] + pixels[at + 1] + pixels[at + 2] > 80) visiblePixels++;
  }
  return { image, visiblePixels };
}

function overlapTraversesLaterSlotFirst(batch: SceneBatch): boolean {
  const program = compileDeformation(batch.deformation);
  const morphBounds = validateMorphTargets(program, batch.mesh.vertexCount, batch.morphTargets);
  const bounds = batch.instances.map(state => conservativeDeformationBounds(program, batch.mesh.bounds,
    morphBounds, state.bones, state.deformScalars, state.model));
  const order = orderInstanceBounds(bounds, batch.capacity);
  return order.indexOf(257) < order.indexOf(256);
}

export async function runWorldQueryInstancesFixture() {
  const acceleratedCanvas = document.querySelector<HTMLCanvasElement>('#accelerated')!;
  const linearCanvas = document.querySelector<HTMLCanvasElement>('#linear')!;
  const mesh = triangle(), acceleratedScene = scene(mesh, 'many-deformed-instances');
  const linearScene = scene(mesh, 'many-deformed-instances');
  const overlapTraversalOrder = overlapTraversesLaterSlotFirst(acceleratedScene);
  const gi = { origin: [-3, -3, 0.05] as Vec3, spacing: [2, 2, 0.5] as Vec3,
    counts: [2, 2, 2] as Vec3, raysPerProbe: 64, maxDistance: 3,
    skyRadiance: [0, 0, 0] as Vec3 };
  const query = { maxRays: 512, maxTrianglesPerMesh: 1, maxInstancesPerBatch: 258 };
  const overflowCanvas = document.createElement('canvas');
  overflowCanvas.style.display = 'none'; document.body.appendChild(overflowCanvas);
  let accelerated: Renderer | undefined, linear: Renderer | undefined, overflowRenderer: Renderer | undefined;
  let acceleratedAfterDispose = -1, linearAfterDispose = -1, overflowAfterDispose = -1;
  try {
    accelerated = await Renderer.create(acceleratedCanvas, [acceleratedScene], {
      worldQueries: { ...query, instanceAcceleration: 'bvh' }, diffuseGi: gi });
    linear = await Renderer.create(linearCanvas, [linearScene], {
      worldQueries: { ...query, instanceAcceleration: 'linear' }, diffuseGi: gi });
    const initial = await capture(accelerated, linear, initialRays());
    const tieWitness = await accelerated.traceRays([
      rayFor(OVERLAP_SECOND, false, { entityId: OVERLAP_SECOND, generation: 1 }),
      rayFor(OVERLAP_SECOND, false, { entityId: OVERLAP_FIRST, generation: 1 }),
    ]);
    const pictureEqual = acceleratedCanvas.toDataURL('image/png') === linearCanvas.toDataURL('image/png');
    const initialVisiblePixels = (await Promise.all([stillImage(acceleratedCanvas), stillImage(linearCanvas)]))
      .map(snapshot => snapshot.visiblePixels);
    const acceleratedGi = await accelerated.probeDiffuseGi(), linearGi = await linear.probeDiffuseGi();
    accelerated.setDiffuseGiEnabled(false); linear.setDiffuseGiEnabled(false);

    const movedStates = instances(true);
    acceleratedScene.instances = movedStates; linearScene.instances = movedStates;
    const moved = await capture(accelerated, linear, [rayFor(MOVED_ID, true), rayFor(MOVED_ID),
      rayFor(1000), rayFor(OVERLAP_SECOND)]);

    const reorderedStates = [...movedStates].reverse();
    acceleratedScene.instances = reorderedStates; linearScene.instances = reorderedStates;
    const reordered = await capture(accelerated, linear, [rayFor(MOVED_ID, true), rayFor(OVERLAP_SECOND), rayFor(1000)]);

    acceleratedScene.instances = []; linearScene.instances = [];
    const empty = await capture(accelerated, linear, [rayFor(MOVED_ID, true), rayFor(OVERLAP_SECOND), MISS]);

    const restoredStates = instances(true, 2);
    acceleratedScene.instances = restoredStates; linearScene.instances = restoredStates;
    const restored = await capture(accelerated, linear, [rayFor(MOVED_ID, true),
      rayFor(MOVED_ID, true, { entityId: MOVED_ID, generation: 1 }),
      rayFor(MOVED_ID, true, { entityId: MOVED_ID, generation: 2 }),
      rayFor(OVERLAP_SECOND)]);

    accelerated.render(CAMERA, TARGET); linear.render(CAMERA, TARGET);
    const finalImages = await Promise.all([stillImage(acceleratedCanvas), stillImage(linearCanvas)]);
    const finalVisiblePixels = finalImages.map(snapshot => snapshot.visiblePixels);
    acceleratedCanvas.replaceWith(finalImages[0].image);
    linearCanvas.replaceWith(finalImages[1].image);
    const adapter = accelerated.info.adapterInfo;

    const errors = { accelerated: [...accelerated.errors], linear: [...linear.errors] };
    accelerated.dispose(); acceleratedAfterDispose = accelerated.stats.ownedGpuBytes; accelerated = undefined;
    linear.dispose(); linearAfterDispose = linear.stats.ownedGpuBytes; linear = undefined;

    const overflowScene = scene(mesh, 'many-deformed-instances');
    overflowRenderer = await Renderer.create(overflowCanvas, [overflowScene], { worldQueries: {
      ...query, maxRays: 1, maxNodeVisits: 1, maxTotalWorkUnits: 1, instanceAcceleration: 'bvh',
    } });
    overflowRenderer.render(CAMERA, TARGET);
    const overflow = await rejection(() => overflowRenderer!.traceRays([rayFor(1000)]));
    const overflowDiagnostics = queryDiagnostics(overflowRenderer);
    const overflowErrors = [...overflowRenderer.errors];
    overflowRenderer.dispose(); overflowAfterDispose = overflowRenderer.stats.ownedGpuBytes; overflowRenderer = undefined;

    return { adapter, initial, tieWitness, moved, reordered, empty, restored, acceleratedGi, linearGi, pictureEqual,
      initialVisiblePixels, finalVisiblePixels, overlapTraversalOrder,
      knownDeformedPosition: [cell(1000)[0] + localHit()[0], cell(1000)[1] + localHit()[1], 0] as Vec3,
      movedPosition: [23 + localHit(true)[0], 6 + localHit(true)[1], 0] as Vec3,
      overlapFirst: OVERLAP_FIRST, overlapSecond: OVERLAP_SECOND, movedId: MOVED_ID,
      overflow, overflowDiagnostics, errors, overflowErrors,
      acceleratedAfterDispose, linearAfterDispose, overflowAfterDispose };
  } finally {
    accelerated?.dispose(); linear?.dispose(); overflowRenderer?.dispose();
    overflowCanvas.remove();
  }
}
