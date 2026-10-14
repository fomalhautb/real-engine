import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createCube } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const eye: Vec3 = [0, 0, 0];
const target: Vec3 = [0, 0, -1];
const emission = [0.2, 0.1, 0.05] as const;
const grid = {
  origin: [-1, -1, -1] as Vec3, spacing: [2, 2, 2] as Vec3,
  counts: [2, 2, 2] as Vec3, raysPerProbe: 128, maxDistance: 12,
  skyRadiance: [0, 0, 0] as Vec3, multiBounce: true,
};

function cavity(): SceneBatch {
  const model = scaling(4, 4, 4); // createCube spans +/-0.5, so interior walls are at +/-2 m.
  const occupant: InstanceState = {
    entityId: 42, entityGeneration: 0, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0,
    roughness: 1, historyVersion: 0,
  };
  return {
    name: 'closed-diffuse-cavity', mesh: createCube(), capacity: 1,
    instances: [occupant], inputMode: 'committed',
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'gi-multibounce/cube', primitiveId: 0, topologyVersion: 1 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      shading: 'pbr', doubleSided: true, baseColor: m.constant([0.5, 0.5, 0.5, 1]),
      metallic: 0, roughness: 1, emissive: m.parameter(0) },
    parameters: new Float32Array([...emission, 1]) },
  };
}

/** Closed constant-emission cavity: all probe rays hit PBR surfaces, and the
 * analytic uniform-radiance recurrence is L[n] = emission + 0.48 L[n-1]. */
export async function runGiMultiBounceFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:192px;height:192px;';
  document.body.appendChild(canvas);
  const walls = cavity();
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [walls], {
      worldQueries: { maxRays: 1024, maxTrianglesPerMesh: 12, maxInstancesPerBatch: 1 },
      diffuseGi: grid,
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const ownedBytes: number[] = [];
    const capture = async (cameraEye: Vec3 = eye, cameraTarget: Vec3 = target) => {
      renderer!.render(cameraEye, cameraTarget);
      ownedBytes.push(renderer!.stats.ownedGpuBytes);
      return { probe: await renderer!.probeDiffuseGi(), diagnostics: renderer!.diagnostics().diffuseGi };
    };

    const first = await capture();
    const second = await capture();
    const third = await capture();
    for (let frame = 4; frame < 16; frame++) await capture();
    const settled = await capture();
    const cameraMoved = await capture([0.25, 0, 0], [0.25, 0, -1]);

    // Purely continuous motion preserves the global history. The cavity still
    // encloses all probe centers and remains a constant-emission reference.
    const moved = multiply(translation(0.05, 0, 0), scaling(4, 4, 4));
    walls.instances = [{ ...walls.instances[0], model: moved, normalMatrix: normalMatrix(moved) }];
    const modelMoved = await capture();
    walls.instances = [{ ...walls.instances[0], historyVersion: 1 }];
    const historyChanged = await capture();
    await capture();
    walls.instances = [{ ...walls.instances[0], entityGeneration: 1 }];
    const generationChanged = await capture();
    await capture();

    renderer.setSun({ direction: [1, 0, 0], color: [0, 0, 0] });
    const sunChanged = await capture();
    await capture();
    renderer.invalidateDiffuseGiHistory();
    const explicitReset = await capture();
    await capture();

    renderer.setMaterialParameters(walls.name, new Float32Array([0, 0, 0, 1]));
    const emissionRemoved = await capture();
    renderer.setMaterialParameters(walls.name, new Float32Array([...emission, 1]));
    const emissionRestored = await capture();
    renderer.setDiffuseGiMultiBounce(false);
    const singleBounce = await capture();
    renderer.setDiffuseGiMultiBounce(true);
    const multiRestored = await capture();
    const multiWarmed = await capture();

    renderer.setDiffuseGiEnabled(false);
    renderer.render(eye, target);
    ownedBytes.push(renderer.stats.ownedGpuBytes);
    const disabled = renderer.diagnostics().diffuseGi;
    renderer.setDiffuseGiEnabled(true);
    const reenabled = await capture();

    const previousInstance = walls.instances[0];
    walls.instances = [];
    const removed = await capture();
    walls.instances = [{ ...previousInstance, entityId: 42 }];
    const readded = await capture();
    const errors = [...renderer.errors];
    renderer.dispose();
    return { first, second, third, settled, cameraMoved, modelMoved,
      historyChanged, generationChanged, sunChanged, explicitReset,
      emissionRemoved, emissionRestored, singleBounce, multiRestored,
      multiWarmed, disabled, reenabled, removed, readded,
      ownedBytes, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
