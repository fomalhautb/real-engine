import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import { DIFFUSE_GI_PROBE_BYTES } from '../../../src/diffuse-gi';
import type { TextureAsset } from '../../../src/texture-assets';

type Vec3 = [number, number, number];
const eye: Vec3 = [0, 0, 5];
const target: Vec3 = [0, 0, 0];
const sunDirection: Vec3 = [0.8, 0, -0.6];
const sunColor: Vec3 = [4, 4, 4];
const wallName = 'gi-offscreen-wall';

const skyGrid = { origin: [-1, -1, -1] as Vec3, spacing: [2, 2, 2] as Vec3,
  counts: [2, 2, 2] as Vec3, raysPerProbe: 64, maxDistance: 12,
  skyRadiance: [0.1, 0.2, 0.3] as Vec3 };
const roomGrid = { origin: [-1, -1, 0.5] as Vec3, spacing: [2, 2, 2] as Vec3,
  counts: [2, 2, 2] as Vec3, raysPerProbe: 128, maxDistance: 20,
  skyRadiance: [0, 0, 0] as Vec3 };

function canvas(): HTMLCanvasElement {
  const result = document.createElement('canvas');
  result.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(result);
  return result;
}

function quad(z: number, halfSize: number, normalZ: -1 | 1): MeshData {
  const points = normalZ === 1
    ? [[-halfSize, -halfSize, z], [halfSize, -halfSize, z], [halfSize, halfSize, z], [-halfSize, halfSize, z]]
    : [[-halfSize, -halfSize, z], [-halfSize, halfSize, z], [halfSize, halfSize, z], [halfSize, -halfSize, z]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  for (let id = 0; id < 4; id++) {
    for (let axis = 0; axis < 3; axis++) view.setFloat32(id * 80 + axis * 4, points[id][axis], true);
    view.setFloat32(id * 80 + 12, 1, true);
    view.setFloat32(id * 80 + 24, normalZ, true);
    view.setUint32(id * 80 + 32, 0, true);
    view.setFloat32(id * 80 + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4,
    uvs: new Float32Array(normalZ === 1 ? [0, 0, 1, 0, 1, 1, 0, 1] : [0, 0, 0, 1, 1, 1, 1, 0]),
    bounds: { min: [-halfSize, -halfSize, z], max: [halfSize, halfSize, z] } };
}

function state(entityId: number): InstanceState {
  return { entityId, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

function batch(name: string, mesh: MeshData, entityId: number): SceneBatch {
  return { name, mesh, capacity: 1, inputMode: 'committed', instances: [state(entityId)],
    queryIdentity: { assetId: 'diffuse-gi-fixture', primitiveId: entityId, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

function scene(): { batches: SceneBatch[]; wall: SceneBatch; blocker: SceneBatch } {
  const receiver = batch('gi-visible-receiver', quad(0, 1.5, 1), 101);
  receiver.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION,
    baseColor: m.constant([1, 1, 1, 1]), metallic: 0, roughness: 1 } };
  // Behind the camera, but in the probe grid's +Z hemisphere.
  const wall = batch(wallName, quad(8, 5, -1), 102);
  wall.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
    baseColor: m.parameter(0), metallic: 0, roughness: 1 },
    parameters: new Float32Array([0.9, 0.05, 0.05, 1]) };
  // A one-sided +Z plate: +Z primary rays see its back face, whereas sun
  // visibility rays leaving the wall toward -Z hit its front face.
  const blocker = batch('gi-sun-blocker', quad(6, 12, 1), 103);
  blocker.instances = [];
  blocker.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
    doubleSided: false, baseColor: m.constant([0, 0, 0, 1]) } };
  return { batches: [receiver, wall, blocker], wall, blocker };
}

function center(renderer: Renderer): { x: number; y: number } {
  return { x: Math.floor(renderer.stats.width / 2), y: Math.floor(renderer.stats.height / 2) };
}

async function rejection(action: () => unknown): Promise<string> {
  try { await action(); return ''; } catch (error) { return String(error); }
}

function maskWithCenterHole(): TextureAsset {
  const pixels = new Uint8Array(3 * 3 * 4);
  for (let i = 0; i < 9; i++) pixels.set([255, 255, 255, i === 4 ? 0 : 255], i * 4);
  return { width: 3, height: 3, colorSpace: 'srgb', pixels };
}

export async function runDiffuseGiSkyFixture() {
  const element = canvas();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, [], { worldQueries: { maxRays: 512 }, diffuseGi: skyGrid });
    const beforeFrame = await rejection(() => renderer!.probeDiffuseGi());
    renderer.setSun({ direction: [0, 0, -1], color: [0, 0, 0] });
    renderer.render(eye, target);
    const first = await renderer.probeDiffuseGi();
    const initial = renderer.diagnostics().diffuseGi;
    renderer.setDiffuseGiEnabled(false);
    renderer.render(eye, target);
    const disabled = renderer.diagnostics().diffuseGi;
    renderer.setDiffuseGiEnabled(true);
    renderer.render(eye, target);
    const restored = await renderer.probeDiffuseGi();
    const errors = [...renderer.errors], frames = renderer.stats.frames;
    renderer.dispose();
    const afterDispose = renderer.stats.ownedGpuBytes;
    const overRayBudget = await rejection(() => Renderer.create(element, [], {
      worldQueries: { maxRays: 511 }, diffuseGi: skyGrid,
    }));
    const overMemoryBudget = await rejection(() => Renderer.create(element, [], {
      // The CPU contract accepts one field, but all GPU-owned fields, hit
      // backup, capture weights and readback must fit the complete budget.
      worldQueries: { maxRays: 512 }, diffuseGi: { ...skyGrid, maxBytes: 8 * DIFFUSE_GI_PROBE_BYTES },
    }));
    const overWorkBudget = await rejection(() => Renderer.create(element, [], {
      worldQueries: { maxRays: 512, maxTotalWorkUnits: 511 }, diffuseGi: skyGrid,
    }));
    return { beforeFrame, first, restored, initial, disabled, frames, errors,
      afterDispose, overRayBudget, overMemoryBudget, overWorkBudget };
  } finally { renderer?.dispose(); element.remove(); }
}

export async function runDiffuseGiSceneFixture() {
  const element = canvas(), { batches, wall, blocker } = scene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, batches, {
      worldQueries: { maxRays: 1024, maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1 }, diffuseGi: roomGrid,
    });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    renderer.render(eye, target);
    const initial = await renderer.probeDiffuseGi();
    const hdrOn = (await renderer.probeHdr([center(renderer)]))[0];
    renderer.setDiffuseGiEnabled(false);
    renderer.render(eye, target);
    const hdrOff = (await renderer.probeHdr([center(renderer)]))[0];
    renderer.setDiffuseGiEnabled(true);
    renderer.render([2, 0, 5], [2, 0, 0]);
    const cameraMoved = await renderer.probeDiffuseGi();

    blocker.instances = [state(103)];
    renderer.render(eye, target);
    const shadowed = await renderer.probeDiffuseGi();
    blocker.instances = [];
    renderer.setMaterialParameters(wallName, new Float32Array([0.05, 0.9, 0.05, 1]));
    renderer.render(eye, target);
    const green = await renderer.probeDiffuseGi();

    renderer.setSun({ direction: sunDirection, color: [0, 0, 0] });
    renderer.render(eye, target);
    const noSun = await renderer.probeDiffuseGi();
    renderer.setSun({ direction: sunDirection, color: sunColor });
    await renderer.setMaterial(wallName, { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      doubleSided: true, alphaMode: 'mask', baseColor: m.multiply(m.parameter(0), m.texture(0)),
      metallic: 0, roughness: 1, alphaCutoff: 0.5 },
      parameters: new Float32Array([0.9, 0.05, 0.05, 1]), textures: [maskWithCenterHole()],
      samplers: [{ addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
        minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest' }] });
    renderer.render(eye, target);
    const masked = await renderer.probeDiffuseGi();
    wall.instances = [];
    renderer.render(eye, target);
    const removed = await renderer.probeDiffuseGi();

    const errors = [...renderer.errors], frames = renderer.stats.frames;
    renderer.dispose();
    return { initial, cameraMoved, shadowed, green, noSun, masked, removed,
      hdrOn, hdrOff, errors, frames, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

export async function runDiffuseGiOverflowFixture() {
  const element = canvas(), { batches } = scene();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, batches, { worldQueries: {
      maxRays: 1024, maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1,
      maxNodeVisits: 1, maxTotalWorkUnits: 1024,
    }, diffuseGi: roomGrid });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    renderer.render(eye, target);
    const result = await renderer.probeDiffuseGi();
    const errors = [...renderer.errors];
    renderer.dispose();
    return { result, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}
