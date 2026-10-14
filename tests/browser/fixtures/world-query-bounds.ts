import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const camera = { eye: [0, 0, 6] as Vec3, target: [0, 0, 0] as Vec3 };
const frontName = 'bounds-front';
const firstCenter = -16;
const lastCenter = 14;

/** Sixteen separated triangles force a BVH with a root, inner nodes, and leaves. */
function triangles(count: number, z: number): MeshData {
  const vertices = new ArrayBuffer(count * 3 * 80);
  const data = new DataView(vertices);
  const indices = new Uint32Array(count * 3);
  const uvs = new Float32Array(count * 3 * 2);
  const corners: readonly [number, number][] = [[-0.4, -0.3], [0.4, -0.3], [0, 0.4]];
  for (let triangle = 0; triangle < count; triangle++) {
    const center = firstCenter + 2 * triangle;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = 3 * triangle + corner;
      const byte = vertex * 80;
      data.setFloat32(byte, center + corners[corner][0], true);
      data.setFloat32(byte + 4, corners[corner][1], true);
      data.setFloat32(byte + 8, z, true);
      data.setFloat32(byte + 12, 1, true);
      data.setFloat32(byte + 24, 1, true); // normal +Z
      data.setUint32(byte + 32, 0, true); // first joint
      data.setFloat32(byte + 48, 1, true); // first weight
      indices[vertex] = vertex;
      uvs[2 * vertex] = corner === 1 ? 1 : corner === 2 ? 0.5 : 0;
      uvs[2 * vertex + 1] = corner === 2 ? 1 : 0;
    }
  }
  return {
    vertices, indices, uvs, vertexCount: count * 3,
    bounds: { min: [firstCenter - 0.4, -0.3, z], max: [firstCenter + 2 * (count - 1) + 0.4, 0.4, z] },
  };
}

function pose(entityId: number, modelX: number, boneX: number): InstanceState {
  return {
    entityId, entityGeneration: 1, model: translation(modelX, 0, 0), normalMatrix: identity(),
    bones: [translation(boneX, 0, 0)], deformScalars: [], color: [1, 1, 1, 1],
    metalness: 0, roughness: 0.5, historyVersion: 0,
  };
}

function scene(): { front: SceneBatch; back: SceneBatch } {
  return {
    front: {
      name: frontName, mesh: triangles(16, 0), capacity: 1, inputMode: 'committed',
      queryIdentity: { assetId: 'bounds-front-16', primitiveId: 0, topologyVersion: 1 },
      instances: [pose(101, 0, 0)], deformation: { jointCount: 1, nodes: [{ kind: 'skin' }] },
      morphTargets: [],
      material: {
        definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.parameter(0) },
        parameters: new Float32Array([0.2, 0.3, 0.4, 1]),
      },
    },
    back: {
      name: 'bounds-back', mesh: triangles(1, -1), capacity: 1, inputMode: 'committed',
      queryIdentity: { assetId: 'bounds-back-1', primitiveId: 0, topologyVersion: 1 },
      instances: [pose(102, 0, 0)], deformation: { jointCount: 1, nodes: [{ kind: 'skin' }] },
      morphTargets: [],
    },
  };
}

function ray(x: number, tMax?: number): WorldRay {
  return { origin: [x, 0, 5], direction: [0, 0, -1], ...(tMax === undefined ? {} : { tMax }) };
}

async function rejection(operation: () => Promise<unknown>): Promise<string> {
  try { await operation(); return ''; } catch (error) { return String(error); }
}

/** All expected positions come from authored triangle centers, not a second intersection implementation. */
export async function runWorldQueryBoundsFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  let budgetRenderer: Renderer | undefined;
  try {
    const { front, back } = scene();
    renderer = await Renderer.create(canvas, [front, back], {
      worldQueries: { maxRays: 8, maxTrianglesPerMesh: 16, maxInstancesPerBatch: 1 },
    });
    renderer.render(camera.eye, camera.target);
    const initial = await renderer.traceRays([ray(firstCenter, 5), ray(lastCenter)]);

    front.instances = [pose(101, 100, 0)];
    back.instances = [pose(102, 100, 0)];
    renderer.render(camera.eye, camera.target);
    const moved = await renderer.traceRays([
      ray(firstCenter + 100), ray(lastCenter + 100), ray(firstCenter), ray(lastCenter),
    ]);

    front.instances = [pose(101, 100, 1)];
    back.instances = [pose(102, 100, 1)];
    renderer.render(camera.eye, camera.target);
    const skinned = await renderer.traceRays([
      ray(firstCenter + 101), ray(lastCenter + 101), ray(firstCenter + 100), ray(lastCenter + 100),
    ]);

    renderer.setMaterialParameters(frontName, new Float32Array([2, 0.5, 0.25, 1]));
    const parameters = await renderer.traceRays([ray(firstCenter + 101)]);
    await renderer.setMaterial(frontName, {
      definition: {
        ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', alphaMode: 'mask',
        baseColor: m.parameter(0), alphaCutoff: 0.5,
      },
      parameters: new Float32Array([0.7, 0.2, 0.3, 0]),
    });
    const masked = await renderer.traceRays([ray(firstCenter + 101)]);
    const errors = [...renderer.errors];
    renderer.dispose();
    const bytesAfterDispose = renderer.stats.ownedGpuBytes;
    renderer = undefined;

    const budgetScene = scene();
    budgetRenderer = await Renderer.create(canvas, [budgetScene.front], {
      worldQueries: { maxRays: 1, maxTrianglesPerMesh: 16, maxInstancesPerBatch: 1, maxNodeVisits: 1 },
    });
    budgetRenderer.render(camera.eye, camera.target);
    const traversalBudgetError = await rejection(() => budgetRenderer!.traceRays([ray(firstCenter)]));
    const budgetErrors = [...budgetRenderer.errors];
    budgetRenderer.dispose();
    const budgetBytesAfterDispose = budgetRenderer.stats.ownedGpuBytes;
    budgetRenderer = undefined;

    let unexpectedMemoryRenderer: Renderer | undefined;
    const memoryBudgetError = await rejection(async () => {
      try {
        unexpectedMemoryRenderer = await Renderer.create(canvas, [scene().front], {
          worldQueries: { maxRays: 1, maxTrianglesPerMesh: 16, maxInstancesPerBatch: 1, maxBytes: 1 },
        });
      } finally { unexpectedMemoryRenderer?.dispose(); }
    });
    return {
      initial, moved, skinned, parameters, masked, traversalBudgetError, memoryBudgetError,
      errors, budgetErrors, bytesAfterDispose, budgetBytesAfterDispose,
      unexpectedMemoryBytesAfterDispose: unexpectedMemoryRenderer?.stats.ownedGpuBytes ?? null,
    };
  } finally {
    budgetRenderer?.dispose();
    renderer?.dispose();
    canvas.remove();
  }
}
