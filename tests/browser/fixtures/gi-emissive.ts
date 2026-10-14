import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { multiply, normalMatrix, rotationX, scaling, translation, type Mat4 } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const emitterName = 'gi-area-emitter';
const probeGrid = { origin: [-0.3, 0.2, 0.5] as Vec3,
  spacing: [0.6, 0.5, 0.6] as Vec3, counts: [2, 2, 2] as Vec3,
  raysPerProbe: 64, maxDistance: 12, skyRadiance: [0, 0, 0] as Vec3,
  multiBounce: false, emissiveSamplesPerProbe: 32 };
const radiance: Vec3 = [8, 4, 2];
const tinySize: [number, number] = [0.12, 0.08];
const largeSize: [number, number] = [3, 2];
const cameraEye: Vec3 = [0, 1.5, 3];
const cameraTarget: Vec3 = [0, 0.8, 0];

function model(centerY: number, size: readonly [number, number], centerZ = -2): Mat4 {
  // createPlane lies in XZ with a +Y normal. This maps its short axis to Y
  // and its normal to +Z, facing every probe in the fixture.
  return multiply(translation(0, centerY, centerZ),
    multiply(rotationX(Math.PI / 2), scaling(size[0], 1, size[1])));
}

function state(id: number, transform: Mat4, version = 0): InstanceState {
  return { entityId: id, model: transform, normalMatrix: normalMatrix(transform),
    bones: [], deformScalars: [], color: [1, 1, 1, 1],
    metalness: 0, roughness: 1, historyVersion: version };
}

function material(le: Vec3, opacity = 1, masked = false): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
    alphaMode: masked ? 'mask' : 'opaque', alphaCutoff: 0.5,
    baseColor: m.constant([0, 0, 0, opacity]), emissive: m.constant([...le, 1]) } };
}

function reflectiveEmitterMaterial(): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'pbr', doubleSided: true,
    baseColor: m.constant([0.9, 0.8, 0.7, 1]), metallic: 0, roughness: 1,
    emissive: m.constant([...radiance, 1]) } };
}

function emitterBatch(): SceneBatch {
  return { name: emitterName, mesh: createPlane(), capacity: 1,
    inputMode: 'committed', instances: [state(101, model(1, tinySize))],
    emissiveSampling: 'triangle-area',
    queryIdentity: { assetId: 'gi-emissive/plane', primitiveId: 1, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: material(radiance) };
}

function blockerBatch(): SceneBatch {
  return { name: 'gi-emissive-black-occluder', mesh: createPlane(), capacity: 1,
    inputMode: 'committed', instances: [],
    queryIdentity: { assetId: 'gi-emissive/blocker', primitiveId: 2, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: material([0, 0, 0]) };
}

export async function runGiEmissiveFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  const emitter = emitterBatch(), blocker = blockerBatch();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [emitter, blocker], {
      worldQueries: { maxRays: 512, maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1 },
      diffuseGi: probeGrid,
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const capture = async () => {
      renderer!.render(cameraEye, cameraTarget);
      return renderer!.probeDiffuseGi();
    };

    const initial = await capture();
    emitter.instances = [state(101, model(1, largeSize), 1)];
    const enlarged = await capture();
    await renderer.setMaterial(emitterName, reflectiveEmitterMaterial());
    renderer.setSun({ direction: [0, 0, 1], color: [3, 3, 3] });
    const reflective = await capture();
    await renderer.setMaterial(emitterName, material(radiance));
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    emitter.instances = [state(101, model(1, tinySize), 2)];

    blocker.instances = [state(102, model(1, [4, 4], -1))];
    const occluded = await capture();
    blocker.instances = [];

    await renderer.setMaterial(emitterName, material(radiance, 0, true));
    const masked = await capture();

    const doubledRadiance: Vec3 = radiance.map(value => value * 2) as Vec3;
    await renderer.setMaterial(emitterName, material(doubledRadiance));
    const doubled = await capture();

    emitter.instances = [state(101, model(2, tinySize), 3)];
    const moved = await capture();

    emitter.instances = [];
    const removed = await capture();
    const diagnostics = renderer.diagnostics().diffuseGi;
    const errors = [...renderer.errors], frames = renderer.stats.frames;
    renderer.dispose();
    return { grid: probeGrid, radiance, tinySize, largeSize,
      initial, enlarged, reflective, occluded, masked, doubled, moved, removed,
      diagnostics, errors, frames, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** A three-visit ray cap leaves sphere rays that miss the tiny plane complete,
 * but forces each endpoint-directed area ray to exhaust its triangle work. */
export async function runGiEmissiveOverflowFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [emitterBatch()], {
      worldQueries: { maxRays: 512, maxTrianglesPerMesh: 2,
        maxInstancesPerBatch: 1, maxNodeVisits: 3 },
      diffuseGi: probeGrid,
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    renderer.render(cameraEye, cameraTarget);
    const probe = await renderer.probeDiffuseGi();
    const errors = [...renderer.errors], diagnostics = renderer.diagnostics().diffuseGi;
    renderer.dispose();
    return { grid: probeGrid, tinySize, probe, diagnostics, errors,
      afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

export async function runGiEmissiveBlocksFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  const first = emitterBatch(), second = emitterBatch();
  second.name = 'gi-area-emitter-second';
  second.queryIdentity = { assetId: 'gi-emissive/second-plane', primitiveId: 2, topologyVersion: 1 };
  const secondModel = multiply(translation(0.8, 0, 0), model(1, tinySize));
  second.instances = [state(202, secondModel)];
  second.material = material([2, 6, 3]);
  const firstPose = first.instances, secondPose = second.instances;
  const grid = { ...probeGrid, emissiveSamplesPerProbe: 16 };
  let renderer: Renderer | undefined, unexpected: Renderer | undefined;
  try {
    let parameterBudgetError = '';
    try {
      unexpected = await Renderer.create(canvas, [first], {
        worldQueries: { maxRays: 512, maxTraceParameterSetsPerEncoder: 1 }, diffuseGi: grid,
      });
    } catch (error) { parameterBudgetError = String(error); }
    const unexpectedlyCreated = !!unexpected;
    unexpected?.dispose();
    renderer = await Renderer.create(canvas, [first, second], {
      worldQueries: { maxRays: 512, maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1 }, diffuseGi: grid,
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const capture = async () => {
      renderer!.render(cameraEye, cameraTarget);
      return { probe: await renderer!.probeDiffuseGi(), diagnostics: renderer!.diagnostics().diffuseGi };
    };
    const both = await capture();
    second.instances = [];
    const firstOnly = await capture();
    first.instances = []; second.instances = secondPose;
    const secondOnly = await capture();
    second.instances = [];
    const empty = await capture();
    first.instances = firstPose; second.instances = secondPose;
    const restored = await capture();
    const bytesBeforeDisable = renderer.stats.ownedGpuBytes;
    renderer.setDiffuseGiEnabled(false);
    renderer.render(cameraEye, cameraTarget);
    const disabled = renderer.diagnostics().diffuseGi;
    const errors = [...renderer.errors];
    renderer.dispose();
    return { both, firstOnly, secondOnly, empty, restored, disabled, parameterBudgetError,
      unexpectedlyCreated, bytesBeforeDisable, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); unexpected?.dispose(); canvas.remove(); }
}
