import { describe, expect, it } from 'vitest';
import type { SceneBatch } from '../src/contracts';
import { createGiDemoScene, GI_PROBE_GRID } from '../src/gi-demo-scene';
import { transformPoint } from '../src/math';

type Vec3 = [number, number, number];
type Bounds = { min: Vec3; max: Vec3 };

function worldBounds(batch: SceneBatch, instanceIndex: number): Bounds {
  const { mesh } = batch;
  const view = new DataView(mesh.vertices);
  const model = batch.instances[instanceIndex].model;
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    const local: Vec3 = [0, 1, 2].map(axis =>
      view.getFloat32(vertex * 80 + axis * 4, true)) as Vec3;
    for (let axis = 0; axis < 3; axis++) {
      expect(local[axis]).toBeGreaterThanOrEqual(mesh.bounds.min[axis] - 1e-6);
      expect(local[axis]).toBeLessThanOrEqual(mesh.bounds.max[axis] + 1e-6);
    }
    const world = transformPoint(model, local);
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], world[axis]);
      max[axis] = Math.max(max[axis], world[axis]);
    }
  }
  return { min, max };
}

function probeCenters(): Vec3[] {
  const { origin, spacing, counts } = GI_PROBE_GRID;
  const centers: Vec3[] = [];
  for (let z = 0; z < counts[2]; z++) for (let y = 0; y < counts[1]; y++) {
    for (let x = 0; x < counts[0]; x++) {
      centers.push([origin[0] + spacing[0] * x,
        origin[1] + spacing[1] * y,
        origin[2] + spacing[2] * z]);
    }
  }
  return centers;
}

describe('GI demo asset and fixed world probe grid', () => {
  it('covers every transformed surface with its normal bias and leaves probes in free space', () => {
    const scene = createGiDemoScene();
    const { origin, spacing, counts, normalBias } = GI_PROBE_GRID;
    const supportMin = origin.map((value, axis) => value - spacing[axis] * 0.5);
    const supportMax = origin.map((value, axis) => value + (counts[axis] - 0.5) * spacing[axis]);
    const solidNames = new Set(['room-neutral', 'room-back-wall', 'red-wall', 'blue-wall', 'white-forms']);
    const solids: { name: string; bounds: Bounds }[] = [];

    for (const batch of scene.batches) for (let index = 0; index < batch.instances.length; index++) {
      const bounds = worldBounds(batch, index);
      for (let axis = 0; axis < 3; axis++) {
        expect(bounds.min[axis] - normalBias,
          `${batch.name}[${index}] min axis ${axis} lies outside GI half-cell support`)
          .toBeGreaterThanOrEqual(supportMin[axis] - 1e-5);
        expect(bounds.max[axis] + normalBias,
          `${batch.name}[${index}] max axis ${axis} lies outside GI half-cell support`)
          .toBeLessThanOrEqual(supportMax[axis] + 1e-5);
      }
      if (solidNames.has(batch.name)) solids.push({ name: `${batch.name}[${index}]`, bounds });
    }

    const centers = probeCenters();
    expect(centers).toHaveLength(counts[0] * counts[1] * counts[2]);
    expect(solids.length).toBeGreaterThanOrEqual(6);
    for (const [index, center] of centers.entries()) for (const solid of solids) {
      // A positive separating distance on at least one axis proves the probe
      // is outside the solid AABB; require room for the normal bias as well.
      const separation = Math.max(...[0, 1, 2].map(axis =>
        Math.max(solid.bounds.min[axis] - center[axis], center[axis] - solid.bounds.max[axis])));
      expect(separation, `probe ${index} intersects or touches ${solid.name}`)
        .toBeGreaterThan(normalBias);
    }
  });
});
