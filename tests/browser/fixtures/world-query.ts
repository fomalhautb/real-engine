import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { TextureAsset } from '../../../src/texture-assets';

type Vec3 = [number, number, number];
type Ray = { origin: Vec3; direction: Vec3; tMin?: number; tMax?: number; ignore?: { entityId: number; generation?: number } };
const ray = (x: number, y = 0, z = 5, direction: Vec3 = [0, 0, -1]): Ray => ({ origin: [x, y, z], direction });
const camera = { eye: [0, 0, 6] as Vec3, target: [0, 0, 0] as Vec3 };

/** Two front-facing triangles, with an explicit UV stream for Masked hit tests. */
function plane(): MeshData {
  const coordinates = [[-0.45, -0.45, 0], [0.45, -0.45, 0], [0.45, 0.45, 0], [-0.45, 0.45, 0]];
  const vertices = new ArrayBuffer(4 * 80), data = new DataView(vertices);
  for (let i = 0; i < 4; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) data.setFloat32(offset + axis * 4, coordinates[i][axis], true);
    data.setFloat32(offset + 12, 1, true);
    data.setFloat32(offset + 24, 1, true);
    data.setUint32(offset + 32, 0, true);
    data.setFloat32(offset + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4,
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    bounds: { min: [-0.45, -0.45, 0], max: [0.45, 0.45, 0] } };
}

function pose(entityId: number, generation: number, x: number, y = 0, z = 0,
  boneY = 0, morph = 0, historyVersion = 0): InstanceState {
  return { entityId, entityGeneration: generation, model: translation(x, y, z), normalMatrix: identity(),
    bones: [translation(0, boneY, 0)], deformScalars: [morph],
    color: [0.2, 0.4, 0.6, 1], metalness: 0.25, roughness: 0.7, historyVersion };
}

function maskedTexture(): TextureAsset {
  const pixels = new Uint8Array(3 * 3 * 4);
  for (let i = 0; i < 9; i++) pixels.set([255, 255, 255, i === 4 ? 0 : 255], i * 4);
  return { width: 3, height: 3, colorSpace: 'srgb', pixels };
}

async function rejection(operation: () => Promise<unknown> | unknown): Promise<string> {
  try { await operation(); return ''; } catch (error) { return String(error); }
}

/** Exact rays against authored planes; no production CPU triangle traversal is used as an oracle. */
export async function runWorldQueryFixture(instanceBvh = false) {
  const mesh = plane();
  const morphPositions = new Float32Array(4 * 3);
  for (let i = 0; i < 4; i++) morphPositions[i * 3] = 1;
  const morphTarget = { positions: morphPositions, normals: new Float32Array(4 * 3) };
  const staticBatch: SceneBatch = { name: 'query-offscreen-static', mesh, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'query-plane', primitiveId: 0, topologyVersion: 1 },
    instances: [pose(100, 1, -5)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
  const movingBatch: SceneBatch = { name: 'query-moving', mesh, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'query-plane', primitiveId: 1, topologyVersion: 1 },
    instances: [pose(200, 1, -2)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
  const sharedBatch: SceneBatch = { name: 'query-shared-skin-morph', mesh, capacity: 2, inputMode: 'committed',
    queryIdentity: { assetId: 'query-plane', primitiveId: 2, topologyVersion: 1 },
    instances: [pose(300, 1, 0), pose(301, 4, 2, 0, 0, 1, 0.5)],
    deformation: { jointCount: 1, nodes: [{ kind: 'skin' }, { kind: 'morph', target: 0, weight: 0 }] },
    morphTargets: [morphTarget] };
  const backBatch: SceneBatch = { name: 'query-behind-mask', mesh, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'query-plane', primitiveId: 3, topologyVersion: 1 },
    instances: [pose(400, 1, 4)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
  const frontBatch: SceneBatch = { name: 'query-mask-front', mesh, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'query-plane', primitiveId: 4, topologyVersion: 1 },
    instances: [pose(401, 1, 4, 0, 1)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, alphaMode: 'mask', baseColor: m.texture(0), alphaCutoff: 0.5 },
      textures: [maskedTexture()], samplers: [{ addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
        minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest' }] } };
  const doubleSidedBatch: SceneBatch = { name: 'query-double-sided', mesh, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'query-plane', primitiveId: 5, topologyVersion: 1 },
    instances: [pose(500, 1, 6)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true } } };
  const batches = [staticBatch, movingBatch, sharedBatch, backBatch, frontBatch, doubleSidedBatch];
  if (instanceBvh) for (const batch of batches) batch.capacity = 8;
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches, { worldQueries: { maxRays: 16, maxTrianglesPerMesh: 8,
      maxInstancesPerBatch: instanceBvh ? 8 : 4 } });
    renderer.render(camera.eye, camera.target);
    const initialRays: Ray[] = [
      ray(-5), ray(-2), ray(0), ray(2.5, 1), ray(4), ray(4.32, 0.32),
      ray(-5, 0, -5, [0, 0, 1]), ray(6, 0, -5, [0, 0, 1]),
      { ...ray(-5), tMin: 5.1 }, { ...ray(-5), tMax: 4.9 },
      { ...ray(-5), ignore: { entityId: 100, generation: 1 } },
      { ...ray(-5), ignore: { entityId: 100, generation: 2 } },
      ray(100),
    ];
    const initial = await renderer.traceRays(initialRays, { textureLod: 0 });
    const empty = await renderer.traceRays([]);
    const normalizedDirection = await renderer.traceRays([ray(-5, 0, 5, [0, 0, -2])]);
    const invalidDirection = await rejection(() => renderer!.traceRays([ray(-5, 0, 5, [0, 0, 0])]));
    const invalidInterval = await rejection(() => renderer!.traceRays([{ ...ray(-5), tMin: 6, tMax: 5 }]));
    const overBudget = await rejection(() => renderer!.traceRays(Array.from({ length: 17 }, () => ray(-5))));

    // An unrendered mutation must not change the committed query snapshot.
    movingBatch.instances = [pose(200, 1, -1)];
    sharedBatch.instances = [pose(300, 1, 0, 0, 0, 0.2, 0.25), sharedBatch.instances[1]];
    const beforeRender = await renderer.traceRays([ray(-2), ray(-1)]);
    renderer.render([20, 0, 6], [20, 0, 0]); // Every query object is now offscreen.
    const moved = await renderer.traceRays([ray(-5), ray(-1), ray(-2), ray(0.25, 0.2), ray(2.5, 1)]);

    sharedBatch.instances = [sharedBatch.instances[1], sharedBatch.instances[0]];
    renderer.render([20, 0, 6], [20, 0, 0]);
    const reordered = await renderer.traceRays([ray(2.5, 1), ray(0.25, 0.2)]);

    sharedBatch.instances = [sharedBatch.instances[0]]; // Remove entity 300 while 301 remains.
    renderer.render([20, 0, 6], [20, 0, 0]);
    const removed = await renderer.traceRays([ray(0.25, 0.2), ray(2.5, 1)]);
    sharedBatch.instances = [sharedBatch.instances[0], pose(300, 2, 0, 0, 0, 0, 0, 1)];
    renderer.render([20, 0, 6], [20, 0, 0]);
    const reused = await renderer.traceRays([ray(0),
      { ...ray(0), ignore: { entityId: 300, generation: 1 } },
      { ...ray(0), ignore: { entityId: 300, generation: 2 } }]);

    const errors = [...renderer.errors], frames = renderer.stats.frames, diagnostics = renderer.diagnostics().worldQueries;
    staticBatch.queryIdentity = { assetId: 'query-plane', primitiveId: 0, topologyVersion: 2 };
    const topologyRejected = await rejection(async () => {
      try { await renderer!.traceRays([ray(-5)]); }
      catch (error) { throw error; }
      renderer!.render(camera.eye, camera.target);
    });
    renderer.dispose();
    const ownedGpuBytesAfterDispose = renderer.stats.ownedGpuBytes;

    const noQuery = await Renderer.create(canvas, batches);
    let disabledQuery = '';
    try { disabledQuery = await rejection(() => noQuery.traceRays([ray(-5)])); }
    finally { noQuery.dispose(); }
    return { initial, empty, normalizedDirection, invalidDirection, invalidInterval, overBudget,
      beforeRender, moved, reordered, removed, reused, topologyRejected, disabledQuery,
      frames, errors, diagnostics, ownedGpuBytesAfterDispose, noQueryBytesAfterDispose: noQuery.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
