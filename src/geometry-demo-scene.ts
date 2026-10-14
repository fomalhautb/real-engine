import { buildGeometryHierarchy, type GeometryHierarchy } from './geometry-hierarchy';
import { createCube, createPlane, type MeshData } from './geometry';
import { multiply, normalMatrix, rotationX, rotationY, scaling, translation } from './math';
import type { InstanceState, SceneBatch } from './contracts';

const AROUND = 128;
const ACROSS = 20;

function point(u: number, v: number): [number, number, number] {
  const ridge = 1 + 0.095 * Math.cos(9 * u + 4 * v) + 0.035 * Math.sin(17 * u - 3 * v);
  const tube = 0.47 * ridge;
  const radial = 1.5 + 0.16 * Math.cos(3 * u) + tube * Math.cos(v);
  return [radial * Math.cos(u), 0.24 * Math.sin(3 * u) + tube * Math.sin(v), radial * Math.sin(u)];
}

function sculpture(): MeshData {
  const vertexCount = AROUND * ACROSS;
  const vertices = new ArrayBuffer(vertexCount * 80);
  const view = new DataView(vertices);
  const indices = new Uint32Array(AROUND * ACROSS * 6);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const h = 0.0005;
  for (let uIndex = 0; uIndex < AROUND; uIndex++) {
    const u = uIndex * Math.PI * 2 / AROUND;
    for (let vIndex = 0; vIndex < ACROSS; vIndex++) {
      const v = vIndex * Math.PI * 2 / ACROSS;
      const p = point(u, v);
      const pu = point(u + h, v), mu = point(u - h, v);
      const pv = point(u, v + h), mv = point(u, v - h);
      const du = pu.map((value, axis) => value - mu[axis]);
      const dv = pv.map((value, axis) => value - mv[axis]);
      const n = [dv[1] * du[2] - dv[2] * du[1],
        dv[2] * du[0] - dv[0] * du[2],
        dv[0] * du[1] - dv[1] * du[0]];
      const length = Math.hypot(...n);
      if (!(length > 0)) throw new Error('GEOMETRY_DEMO_DEGENERATE_NORMAL');
      const vertex = uIndex * ACROSS + vIndex;
      const offset = vertex * 80;
      for (let axis = 0; axis < 3; axis++) {
        view.setFloat32(offset + axis * 4, p[axis], true);
        view.setFloat32(offset + 16 + axis * 4, n[axis] / length, true);
        min[axis] = Math.min(min[axis], p[axis]);
        max[axis] = Math.max(max[axis], p[axis]);
      }
      view.setFloat32(offset + 12, 1, true);
      for (let joint = 0; joint < 4; joint++) view.setUint32(offset + 32 + joint * 4, 0, true);
      view.setFloat32(offset + 48, 1, true);
    }
  }
  let cursor = 0;
  for (let u = 0; u < AROUND; u++) for (let v = 0; v < ACROSS; v++) {
    const a = u * ACROSS + v;
    const b = ((u + 1) % AROUND) * ACROSS + v;
    const c = ((u + 1) % AROUND) * ACROSS + (v + 1) % ACROSS;
    const d = u * ACROSS + (v + 1) % ACROSS;
    indices.set([a, d, b, b, d, c], cursor);
    cursor += 6;
  }
  return { vertices, indices, vertexCount, bounds: { min, max } };
}

function instance(entityId: number, model: Float32Array, color: [number, number, number, number],
  metallic: number, roughness: number): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color, metalness: metallic, roughness, historyVersion: 0 };
}

export class GeometryDemoScene {
  readonly batches: SceneBatch[];
  readonly hierarchy: GeometryHierarchy;
  readonly sourceTriangles: number;
  readonly instanceCount = 3;

  private constructor(hierarchy: GeometryHierarchy) {
    this.hierarchy = hierarchy;
    this.sourceTriangles = hierarchy.sourceTriangleCount;
    const sculptureTransforms = [
      multiply(translation(-4.3, 0.32, 0), multiply(rotationX(0.48), scaling(0.85, 0.85, 0.85))),
      multiply(translation(0, 0.4, 0), multiply(rotationY(0.25), scaling(1.04, 1.04, 1.04))),
      multiply(translation(4.3, 0.32, 0), multiply(rotationX(-0.38), scaling(0.85, 0.85, 0.85))),
    ];
    const ground = createPlane();
    const plinth = createCube();
    const emptyDeformation = { jointCount: 0, nodes: [] };
    const sculptureBatch: SceneBatch =
      { name: 'sculpture', mesh: hierarchy.source, geometryLod: hierarchy,
        capacity: 3, inputMode: 'committed', deformation: emptyDeformation, morphTargets: [],
        instances: [
          instance(1, sculptureTransforms[0], [0.33, 0.81, 0.78, 1], 0.48, 0.34),
          instance(2, sculptureTransforms[1], [0.96, 0.57, 0.30, 1], 0.52, 0.3),
          instance(3, sculptureTransforms[2], [0.68, 0.55, 0.96, 1], 0.42, 0.36),
        ] };
    this.batches = [
      sculptureBatch,
      { name: 'plinths', mesh: plinth, capacity: 3, inputMode: 'committed',
        deformation: emptyDeformation, morphTargets: [],
        instances: [-4.3, 0, 4.3].map((x, index) => instance(10 + index,
          multiply(translation(x, -0.94, 0), scaling(index === 1 ? 2.45 : 2.05, 0.3, 2.05)),
          [0.22, 0.30, 0.36, 1], 0.08, 0.64)) },
      { name: 'ground', mesh: ground, capacity: 1, inputMode: 'committed',
        deformation: emptyDeformation, morphTargets: [],
        instances: [instance(20, multiply(translation(0, -1.34, 0), scaling(24, 1, 18)),
          [0.11, 0.17, 0.21, 1], 0, 0.9)] },
    ];
  }

  static async create(): Promise<GeometryDemoScene> {
    const mesh = sculpture();
    const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'geometry-demo-sculpture-v1',
      mesh, coverage: 'opaque' }, { maxClusterTriangles: 96, simplifyRatio: 0.5, maxLevels: 8 });
    return new GeometryDemoScene(hierarchy);
  }
}
