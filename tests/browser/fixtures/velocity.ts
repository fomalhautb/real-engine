import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

const HALF_SIZE = 0.48;
const FOCAL = 1 / Math.tan(Math.PI / 6.4); // Renderer uses fovY = PI / 3.2.

function quad(): MeshData {
  const positions = [
    [-HALF_SIZE, -HALF_SIZE, 0], [HALF_SIZE, -HALF_SIZE, 0],
    [HALF_SIZE, HALF_SIZE, 0], [-HALF_SIZE, HALF_SIZE, 0],
  ];
  const vertices = new ArrayBuffer(positions.length * 80), view = new DataView(vertices);
  for (let vertex = 0; vertex < positions.length; vertex++) {
    const base = vertex * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(base + axis * 4, positions[vertex][axis], true);
    view.setFloat32(base + 12, 1, true);
    view.setFloat32(base + 24, 1, true);
    view.setUint32(base + 32, 0, true);
    view.setFloat32(base + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4,
    bounds: { min: [-HALF_SIZE, -HALF_SIZE, 0], max: [HALF_SIZE, HALF_SIZE, 0] } };
}

function state(entityId: number, x: number, y = 0, boneX = 0, boneY = 0, scalar = 0, historyVersion = 0): InstanceState {
  return { entityId, model: translation(x, y, 0), normalMatrix: identity(),
    bones: [translation(boneX, boneY, 0)], deformScalars: [scalar],
    color: [0.9, 0.7, 0.4, 1], metalness: 0, roughness: 0.7, historyVersion };
}

type Center = readonly [number, number];
const initialCenters: Center[] = [[-2.7, 0], [-0.9, 0], [0.9, 0], [2.7, 0]];

/** Independent front-facing projection: eye and target both translate in X, with Z fixed at 8. */
function uv(world: Center, eyeX: number, width: number, height: number): [number, number] {
  const aspect = width / height;
  return [0.5 + FOCAL * (world[0] - eyeX) / (16 * aspect), 0.5 - FOCAL * world[1] / 16];
}
function expectedVelocity(current: Center, previous: Center, currentEyeX: number, previousEyeX: number,
  width: number, height: number): [number, number] {
  const now = uv(current, currentEyeX, width, height), before = uv(previous, previousEyeX, width, height);
  return [now[0] - before[0], now[1] - before[1]];
}

/** Renders real velocity attachments and probes only exact interior pixels. */
export async function runVelocityFixture() {
  const mesh = quad();
  const morphPositions = new Float32Array(mesh.vertexCount * 3);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    morphPositions[vertex * 3] = 1;
    morphPositions[vertex * 3 + 1] = -0.5;
  }
  const morphTarget = { positions: morphPositions, normals: new Float32Array(mesh.vertexCount * 3) };
  const batches: SceneBatch[] = [
    { name: 'velocity-static', mesh, capacity: 1, inputMode: 'committed', instances: [state(801, -2.7)],
      deformation: { jointCount: 0, nodes: [] }, morphTargets: [] },
    { name: 'velocity-object', mesh, capacity: 1, inputMode: 'committed', instances: [state(802, -0.9)],
      deformation: { jointCount: 0, nodes: [] }, morphTargets: [] },
    { name: 'velocity-skin', mesh, capacity: 1, inputMode: 'committed', instances: [state(803, 0.9)],
      deformation: { jointCount: 1, nodes: [{ kind: 'skin' }] }, morphTargets: [] },
    { name: 'velocity-morph', mesh, capacity: 1, inputMode: 'committed', instances: [state(804, 2.7)],
      deformation: { jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] }, morphTargets: [morphTarget] },
    { name: 'velocity-mask-hole', mesh, capacity: 1, inputMode: 'committed', instances: [state(805, 0, 1.5)],
      deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
      material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, alphaMode: 'mask', baseColor: m.constant([1, 1, 1, 0]), alphaCutoff: 0.5 } } },
  ];
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches);
    let noFrameRejection = '';
    try { await renderer.probeVelocity([{ x: 0, y: 0 }]); } catch (error) { noFrameRejection = String(error); }
    const capture = async (eyeX: number, centers: readonly Center[], previousCenters: readonly Center[], previousEyeX: number,
      reset = false) => {
      renderer!.render([eyeX, 0, 8], [eyeX, 0, 0]);
      const width = renderer!.stats.width, height = renderer!.stats.height;
      const points = centers.map(center => {
        const projected = uv(center, eyeX, width, height);
        const x = Math.floor(projected[0] * width), y = Math.floor(projected[1] * height);
        if (x < 0 || y < 0 || x >= width || y >= height) throw new Error('VELOCITY_FIXTURE_POINT_OUTSIDE_VIEW');
        return { x, y };
      });
      const holeUv = uv([0, 1.5], eyeX, width, height);
      points.push({ x: Math.floor(holeUv[0] * width), y: Math.floor(holeUv[1] * height) });
      points.push({ x: width - 1, y: height - 1 });
      const values = await renderer!.probeVelocity(points);
      return { width, height, points, values,
        expected: centers.map((center, i) => reset ? [0, 0] : expectedVelocity(center, previousCenters[i], eyeX, previousEyeX, width, height)) };
    };

    const first = await capture(0, initialCenters, initialCenters, 0);
    let invalidProbeRejection = '';
    try { await renderer.probeVelocity([{ x: -1, y: 0 }]); } catch (error) { invalidProbeRejection = String(error); }
    const camera = await capture(0.3, initialCenters, initialCenters, 0);

    batches[1].instances = [state(802, -0.6, -0.2)];
    batches[2].instances = [state(803, 0.9, 0, 0.2, 0.1)];
    batches[3].instances = [state(804, 2.7, 0, 0, 0, 0.3)];
    const movedCenters: Center[] = [[-2.7, 0], [-0.6, -0.2], [1.1, 0.1], [3, -0.15]];
    const moved = await capture(0.3, movedCenters, initialCenters, 0.3);
    const settled = await capture(0.3, movedCenters, movedCenters, 0.3);

    // Per-entity discontinuity suppresses morph velocity without affecting others.
    batches[3].instances = [state(804, 2.7, 0, 0, 0, 0.45, 1)];
    const invalidatedCenters: Center[] = [[-2.7, 0], [-0.6, -0.2], [1.1, 0.1], [3.15, -0.225]];
    const entityReset = await capture(0.3, invalidatedCenters, movedCenters, 0.3, true);

    renderer.invalidateHistory();
    batches[1].instances = [state(802, -0.4, -0.3)];
    batches[2].instances = [state(803, 0.9, 0, 0.5, 0.2)];
    batches[3].instances = [state(804, 2.7, 0, 0, 0, 0.7, 1)];
    const cutCenters: Center[] = [[-2.7, 0], [-0.4, -0.3], [1.4, 0.2], [3.4, -0.35]];
    const cut = await capture(1.1, cutCenters, invalidatedCenters, 0.3, true);

    canvas.style.width = '320px';
    batches[1].instances = [state(802, -0.2, -0.3)];
    const resizedCenters: Center[] = [[-2.7, 0], [-0.2, -0.3], [1.4, 0.2], [3.4, -0.35]];
    const resized = await capture(1.3, resizedCenters, cutCenters, 1.1, true);
    const frames = renderer.stats.frames, errors = [...renderer.errors];
    renderer.dispose();
    return { first, camera, moved, settled, entityReset, cut, resized,
      noFrameRejection, invalidProbeRejection, frames, errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
