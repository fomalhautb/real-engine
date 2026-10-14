import type { InstanceState, SceneBatch } from './contracts';
import { createCube, createPlane } from './geometry';
import { m, type MaterialDefinition } from './material';
import { multiply, normalMatrix, rotationX, scaling, translation, type Mat4 } from './math';

type Vec3 = [number, number, number];

const emptyDeformation = { jointCount: 0, nodes: [] };
const white: MaterialDefinition = {
  shading: 'pbr', alphaMode: 'opaque', doubleSided: true,
  baseColor: m.instanceColor, metallic: 0, roughness: 0.82,
  emissive: m.constant([0, 0, 0, 1]), alphaCutoff: 0,
};
const coloredWall: MaterialDefinition = {
  shading: 'pbr', alphaMode: 'opaque', doubleSided: true,
  baseColor: m.parameter(0), metallic: 0, roughness: 0.96,
  emissive: m.constant([0, 0, 0, 1]), alphaCutoff: 0,
};
const lamp: MaterialDefinition = {
  shading: 'unlit', alphaMode: 'opaque', doubleSided: true,
  baseColor: m.constant([0.06, 0.04, 0.025, 1]), metallic: 0, roughness: 1,
  emissive: m.constant([5.5, 2.6, 0.85, 1]), alphaCutoff: 0,
};

function instance(entityId: number, model: Mat4, color: [number, number, number, number]): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color, metalness: 0, roughness: 0.82, historyVersion: 0 };
}

function batch(name: string, mesh: SceneBatch['mesh'], instances: InstanceState[],
  definition: MaterialDefinition, parameters?: Float32Array): SceneBatch {
  return { name, mesh, capacity: instances.length, instances, inputMode: 'committed',
    deformation: emptyDeformation, morphTargets: [], material: { definition, parameters },
    queryIdentity: { assetId: `gi-demo/${name}/v2`, primitiveId: 0, topologyVersion: 1 } };
}

function transform(position: Vec3, size: Vec3): Mat4 {
  return multiply(translation(...position), scaling(...size));
}

/** An open roof and front keep direct sun visible while side walls supply color bleed. */
export function createGiDemoScene(): { batches: SceneBatch[]; triangles: number } {
  const plane = createPlane();
  const cube = createCube();
  const pale: [number, number, number, number] = [0.82, 0.84, 0.82, 1];
  // Overlap the floor and solid wall slabs: a real 2.5 cm opening here previously
  // let sun through even with GI disabled. Floor extensions lie inside the slabs.
  const floor = instance(1, transform([0, 0, 0], [6.6, 1, 7.4]), [0.76, 0.79, 0.76, 1]);
  const back = instance(2, transform([0, 1.775, -3.675], [6.7, 3.85, 0.15]), pale);
  const left = instance(3, transform([-3.275, 1.775, 0], [0.15, 3.85, 7.5]), [1, 1, 1, 1]);
  const right = instance(4, transform([3.275, 1.775, 0], [0.15, 3.85, 7.5]), [1, 1, 1, 1]);
  const objects = [
    instance(10, transform([0, 1.05, -0.25], [0.87, 2.1, 0.87]), [0.91, 0.89, 0.84, 1]),
    instance(11, transform([-1.3, 0.59, 0.75], [0.72, 1.18, 0.72]), [0.92, 0.94, 0.92, 1]),
    instance(12, transform([1.3, 0.79, 0.55], [0.72, 1.58, 0.72]), [0.88, 0.90, 0.94, 1]),
  ];
  // The luminous strip is a real world-query material hit, rather than a baked fill light.
  const luminousStrip = instance(20,
    multiply(translation(0, 2.75, -3.58), multiply(rotationX(Math.PI / 2), scaling(1.6, 1, 0.22))),
    [1, 1, 1, 1]);
  const batches = [
    batch('room-neutral', plane, [floor], white),
    batch('room-back-wall', cube, [back], white),
    batch('red-wall', cube, [left], coloredWall, new Float32Array([0.82, 0.105, 0.075, 1])),
    batch('blue-wall', cube, [right], coloredWall, new Float32Array([0.08, 0.24, 0.72, 1])),
    batch('white-forms', cube, objects, white),
    batch('emissive-strip', plane, [luminousStrip], lamp),
  ];
  return { batches, triangles: batches.reduce((count, item) => count + item.mesh.indices.length / 3 * item.instances.length, 0) };
}

export const GI_PROBE_GRID = Object.freeze({
  // Half-cell support covers the complete solid walls plus normal bias:
  // x +/-3.4, y -0.2..4.2, z +/-3.8. Centers stay inside the open room.
  origin: [-2.55, 0.35, -2.85] as [number, number, number],
  spacing: [1.7, 1.1, 1.9] as [number, number, number],
  counts: [4, 4, 4] as [number, number, number],
  raysPerProbe: 64,
  maxDistance: 12,
  normalBias: 0.03,
  skyRadiance: [0.045, 0.055, 0.075] as [number, number, number],
  multiBounce: true,
});
