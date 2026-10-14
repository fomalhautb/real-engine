import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { DEFAULT_MATERIAL_DEFINITION } from '../../../src/material';
import { identity, lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
export type ShadowSample = { x: number; y: number; baseline: number[]; lit: number[] };

const eye: Vec3 = [0, 0, 7];
const target: Vec3 = [0, 0, 0];
const sunDirection: Vec3 = [0.8, 0, 0.6];
const sunColor: Vec3 = [2.5, 2.5, 2.5];
const baseColor: Vec3 = [0.8, 0.8, 0.8];
const roughness = 1;
const samplePoints = [80, 96, 112, 128, 144, 160, 176].flatMap(x =>
  [80, 104, 128, 152, 176].map(y => ({ x, y })));

function quad(points: readonly Vec3[], normal: Vec3): MeshData {
  const vertices = new ArrayBuffer(4 * 80);
  const data = new DataView(vertices);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let vertex = 0; vertex < 4; vertex++) {
    for (let axis = 0; axis < 3; axis++) {
      data.setFloat32(vertex * 80 + axis * 4, points[vertex][axis], true);
      data.setFloat32(vertex * 80 + 16 + axis * 4, normal[axis], true);
      min[axis] = Math.min(min[axis], points[vertex][axis]);
      max[axis] = Math.max(max[axis], points[vertex][axis]);
    }
    data.setFloat32(vertex * 80 + 12, 1, true);
    data.setUint32(vertex * 80 + 32, 0, true);
    data.setFloat32(vertex * 80 + 48, 1, true);
  }
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min, max } };
}

/** z = slope*x, with a constant geometric normal and no possible external blocker. */
function receiver(slope: number): MeshData {
  const half = 4;
  const length = Math.hypot(slope, 1);
  return quad([[-half, -half, -half * slope], [half, -half, half * slope],
    [half, half, half * slope], [-half, half, -half * slope]],
  [-slope / length, 0, 1 / length]);
}

/** Along the center receiver-to-sun ray, but outside the center camera ray. */
function smallCaster(): MeshData {
  const x0 = 0.24, dx = 0.11, dy = 0.28, z = 0.18;
  return quad([[x0 - dx, -dy, z], [x0 + dx, -dy, z],
    [x0 + dx, dy, z], [x0 - dx, dy, z]], [0, 0, 1]);
}

function sealedFloor(): MeshData {
  return quad([[-3, 0, 2], [0.2, 0, 2], [0.2, 0, -2], [-3, 0, -2]], [0, 1, 0]);
}

/** The wall overlaps the floor by 15 cm, removing a real geometric crack. */
function sealedWall(): MeshData {
  return quad([[0, -0.15, -2], [0, -0.15, 2], [0, 2, 2], [0, 2, -2]], [-1, 0, 0]);
}

function pixelsForWorldPoints(width: number, height: number, cameraEye: Vec3,
  cameraTarget: Vec3, world: readonly Vec3[]): { x: number; y: number }[] {
  const projection = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150),
    lookAt(cameraEye, cameraTarget, [0, 1, 0]));
  return world.map(point => {
    const ndc = transformPoint(projection, point);
    return { x: Math.round((ndc[0] * 0.5 + 0.5) * width - 0.5),
      y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * height - 0.5) };
  });
}

function pose(entityId: number): InstanceState {
  return { entityId, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
    color: [...baseColor, 1], metalness: 0, roughness, historyVersion: 0 };
}

function batch(name: string, mesh: MeshData, instances: InstanceState[]): SceneBatch {
  return { name, mesh, capacity: 1, inputMode: 'committed', instances,
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

export async function runShadowQualityFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:257px;height:257px;z-index:9999;pointer-events:none;';
  document.body.appendChild(canvas);
  const low = batch('shadow-low-slope', receiver(0), [pose(1)]);
  const high = batch('shadow-high-slope', receiver(0.5), []);
  const caster = batch('shadow-near-caster', smallCaster(), []);
  const floor = batch('shadow-contact-floor', sealedFloor(), []);
  const wall = batch('shadow-contact-wall', sealedWall(), []);
  wall.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [low, high, caster, floor, wall]);
    const capture = async (sunStrength: number, points = samplePoints,
      cameraEye = eye, cameraTarget = target, lightDirection = sunDirection): Promise<number[][]> => {
      renderer!.setSun({ direction: lightDirection,
        color: sunColor.map(value => value * sunStrength) as Vec3 });
      renderer!.render(cameraEye, cameraTarget);
      if (renderer!.stats.width !== 257 || renderer!.stats.height !== 257) {
        throw new Error(`SHADOW_QUALITY_PIXEL_SIZE: ${renderer!.stats.width}x${renderer!.stats.height}`);
      }
      if (renderer!.errors.length) throw new Error(renderer!.errors.join('\n'));
      return renderer!.probeHdr(points);
    };
    const lowBaseline = await capture(0);
    const lowLit = await capture(1);
    low.instances = [];
    high.instances = [pose(2)];
    const highBaseline = await capture(0);
    const highLit = await capture(1);
    high.instances = [];
    low.instances = [pose(1)];
    caster.instances = [pose(3)];
    const contact = await capture(1, [{ x: 128, y: 128 }, { x: 80, y: 128 }]);
    low.instances = [];
    caster.instances = [];
    floor.instances = [pose(4)];
    const wallEye: Vec3 = [-2, 2, 4], wallTarget: Vec3 = [-0.4, 0, 0];
    const wallSun: Vec3 = [0.6, 0.8, 0];
    const wallPoints = pixelsForWorldPoints(renderer.stats.width, renderer.stats.height,
      wallEye, wallTarget, [[-0.06, 0, 0], [-0.14, 0, 0], [-0.3, 0, 0]]);
    const wallOpenBaseline = await capture(0, wallPoints, wallEye, wallTarget, wallSun);
    const wallOpenLit = await capture(1, wallPoints, wallEye, wallTarget, wallSun);
    wall.instances = [pose(5)];
    const wallSealedBaseline = await capture(0, wallPoints, wallEye, wallTarget, wallSun);
    const wallSealedLit = await capture(1, wallPoints, wallEye, wallTarget, wallSun);
    const samples = (baseline: number[][], lit: number[][]): ShadowSample[] =>
      samplePoints.map((point, index) => ({ ...point, baseline: baseline[index], lit: lit[index] }));
    const result = { width: renderer.stats.width, height: renderer.stats.height,
      eye, sunDirection, sunColor, baseColor, roughness,
      low: { slope: 0, samples: samples(lowBaseline, lowLit) },
      high: { slope: 0.5, samples: samples(highBaseline, highLit) },
      contact, wallContact: { points: wallPoints, openBaseline: wallOpenBaseline,
        openLit: wallOpenLit, sealedBaseline: wallSealedBaseline, sealedLit: wallSealedLit },
      frames: renderer.stats.frames, errors: [...renderer.errors] };
    renderer.dispose();
    return { ...result, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
