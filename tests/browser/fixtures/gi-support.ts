import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import { giRayDirection, type GiVisibilityDirection } from '../../../src/diffuse-gi';
import { createPlane, type MeshData } from '../../../src/geometry';
import { buildGiDistanceWeights, GI_DISTANCE_CELLS } from '../../../src/gi-visibility';
import { identity, normalMatrix, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const grid = { origin: [-1, -0.5, -0.5] as Vec3, spacing: [2, 1, 1] as Vec3,
  counts: [2, 2, 2] as Vec3, raysPerProbe: 64, maxDistance: 6, skyRadiance: [0.2, 0.3, 0.4] as Vec3 };
const eye: Vec3 = [3, 2, 3], target: Vec3 = [0, 0, 0];
function canvas() {
  const value = document.createElement('canvas');
  value.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(value); return value;
}
function state(x = 0, generation = 1): InstanceState {
  const model = translation(x, 0, 0);
  return { entityId: 1, entityGeneration: generation, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}
function material(doubleSided = false, opacity = 1): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided, alphaMode: 'mask', alphaCutoff: 0.5,
    baseColor: m.constant([0.4, 0.3, 0.2, opacity]), metallic: 0, roughness: 1,
    emissive: m.constant([0.5, 0.3, 0.1, 1]) } };
}
function door(): SceneBatch {
  // Independent plane equation x=0, front normal +X. Extent covers every
  // possible intersection within the six-metre ray range from this grid.
  const vertices = new Float32Array([[0,-8,-8], [0,8,-8], [0,8,8], [0,-8,8]]
    .flatMap(p => [...p, 1, 1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]));
  const mesh: MeshData = { vertices: vertices.buffer, vertexCount: 4, indices: new Uint32Array([0,1,2,0,2,3]),
    bounds: { min: [0,-8,-8], max: [0,8,8] },
    uvs: new Float32Array([0,0,1,0,1,1,0,1]) };
  return { name: 'support-door', mesh, capacity: 1, inputMode: 'committed', instances: [state()],
    queryIdentity: { assetId: 'gi-support/door', primitiveId: 0, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: material(), emissiveSampling: 'triangle-area' };
}
function expectedMoments(direction: GiVisibilityDirection, x = 0, doubleSided = false, present = true) {
  const rays = Array.from({ length: 64 }, (_, i) => giRayDirection(i, 64)), weights = buildGiDistanceWeights(rays);
  return Array.from({ length: 8 }, (_, p) => {
    const px = grid.origin[0] + (p & 1) * grid.spacing[0];
    const distances = rays.map(d => {
      const t = (x - px) / d[0];
      const eligible = doubleSided || (direction === 'receiver-to-probe' ? d[0] > 0 : d[0] < 0);
      return present && eligible && t >= 0.0001 && t <= grid.maxDistance ? t : grid.maxDistance;
    });
    return Array.from({ length: GI_DISTANCE_CELLS }, (_, cell) => {
      let total = 0, mean = 0, square = 0;
      for (let i = 0; i < rays.length; i++) {
        const weight = weights[cell * rays.length + i]; total += weight;
        mean += weight * distances[i]; square += weight * distances[i] ** 2;
      }
      return [mean / total, square / total, total];
    });
  });
}
export async function runGiSupportFixture(lowBudget = false) {
  const element = canvas(), batch = door(); let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, [batch], {
      worldQueries: { maxRays: 512, maxTotalWorkUnits: 65536, maxTraceParameterSetsPerEncoder: 3 },
      diffuseGi: { ...grid, emissiveSamplesPerProbe: 4, maxVisibilityWorkUnits: lowBudget ? 512 : 65536 },
    });
    renderer.setSun({ direction: [1, 0, 0], color: [3, 2, 1] });
    const capture = async (direction: GiVisibilityDirection, x = 0, doubleSided = false, present = true) => {
      renderer!.setDiffuseGiVisibilityDirection(direction); renderer!.render(eye, target);
      return { direction, actual: await renderer!.probeDiffuseGi(), expected: expectedMoments(direction, x, doubleSided, present),
        diagnostics: renderer!.diagnostics().diffuseGi, bytes: renderer!.stats.ownedGpuBytes };
    };
    const ordinaryBefore = await capture('probe-to-receiver');
    const queryBefore = await renderer.traceRays([{ origin: [2, 0, 0], direction: [-1, 0, 0], tMax: 4 },
      { origin: [-2, 0, 0], direction: [1, 0, 0], tMax: 4 }]);
    const reversed = await capture('receiver-to-probe');
    const queryAfter = await renderer.traceRays([{ origin: [2, 0, 0], direction: [-1, 0, 0], tMax: 4 },
      { origin: [-2, 0, 0], direction: [1, 0, 0], tMax: 4 }]);
    const ordinaryAfter = await capture('probe-to-receiver');
    const sequence = [];
    if (!lowBudget) {
      batch.instances = [state(0.25)]; sequence.push(await capture('receiver-to-probe', 0.25));
      await renderer.setMaterial(batch.name, material(true));
      sequence.push(await capture('receiver-to-probe', 0.25, true));
      await renderer.setMaterial(batch.name, material(false, 0));
      sequence.push(await capture('receiver-to-probe', 0.25, false, false));
      await renderer.setMaterial(batch.name, material(false, 0.5));
      sequence.push(await capture('receiver-to-probe', 0.25));
      batch.instances = []; sequence.push(await capture('receiver-to-probe', 0.25, false, false));
      batch.instances = [state(-0.25, 2)]; sequence.push(await capture('receiver-to-probe', -0.25));
      renderer.setDiffuseGiEnabled(false); renderer.render(eye, target);
      const disabled = renderer.diagnostics().diffuseGi;
      if (disabled?.visibilityQueriesPerUpdate !== 0) throw Error('SUPPORT_DISABLED_WORK');
      renderer.setDiffuseGiEnabled(true); sequence.push(await capture('receiver-to-probe', -0.25));
    }
    const errors = [...renderer.errors]; renderer.dispose();
    return { lowBudget, ordinaryBefore, reversed, ordinaryAfter, sequence, queryBefore, queryAfter,
      errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

/** Existing surface history, world feedback and reflections must share the new
 * cache field without mixing incompatible support across the mode boundary. */
export async function runGiSupportHistoryFixture() {
  const element = canvas(); let renderer: Renderer | undefined;
  const floor: SceneBatch = { name: 'support-floor', mesh: createPlane(), capacity: 1,
    instances: [{ ...state(), model: identity(), normalMatrix: identity() }], inputMode: 'committed',
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([0.6, 0.4, 0.2, 1]), metallic: 0, roughness: 1 } } };
  try {
    renderer = await Renderer.create(element, [floor], {
      worldQueries: { maxRays: 1024, maxTraceParameterSetsPerEncoder: 9 },
      diffuseGi: { origin: [-1, 0.3, -1], spacing: [2, 1, 2], counts: [2, 2, 2], multiBounce: true,
        skyRadiance: [0.1, 0.2, 0.3], maxVisibilityWorkUnits: 65536,
        finalGather: { pixelSpacing: 16, maxAnchors: 512, maxHistory: 16, receiver: 'raster' } },
      reflections: { pixelSpacing: 32, maxAnchors: 128, samplesPerAnchor: 1 },
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const camera: Vec3 = [0, 3, 2];
    const capture = async (frames: number) => {
      for (let i = 0; i < frames; i++) renderer!.render(camera, target);
      const probe = await renderer!.probeDiffuseGi();
      return { probe, gi: renderer!.diagnostics().diffuseGi, reflections: renderer!.diagnostics().reflections,
        hdr: (await renderer!.probeHdr([{ x: 128, y: 128 }]))[0] };
    };
    const stable = await capture(8);
    renderer.setDiffuseGiVisibilityDirection('receiver-to-probe'); const switched = await capture(1);
    const recovered = await capture(4);
    renderer.setDiffuseGiVisibilityDirection('probe-to-receiver'); const restored = await capture(1);
    const errors = [...renderer.errors]; renderer.dispose();
    return { stable, switched, recovered, restored, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}
