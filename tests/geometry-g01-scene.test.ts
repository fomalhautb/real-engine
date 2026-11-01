import { describe, expect, it } from 'vitest';
import { G01Trajectory, G01_DETAIL_SPECTRUM, G01_SEED, G01_STEP_SECONDS, g01Mesh, g01ProjectionScale, g01ShapeParameters } from '../src/geometry-g01-scene';

describe('G01 mesh', () => {
  it('is a closed, consistently oriented manifold for the record and reduced resolutions', () => {
    for (const [around, across] of [[64, 16], [256, 64]]) {
      const mesh = g01Mesh(G01_SEED, around, across);
      expect(mesh.indices.length / 3).toBe(2 * around * across);
      const edges = new Map<number, number>();
      for (let t = 0; t < mesh.indices.length; t += 3) for (let k = 0; k < 3; k++) {
        const key = mesh.indices[t + k] * mesh.vertexCount + mesh.indices[t + (k + 1) % 3];
        edges.set(key, (edges.get(key) ?? 0) + 1);
      }
      let boundary = 0, repeated = 0;
      for (const [key, count] of edges) {
        if (count > 1) repeated++;
        const a = Math.floor(key / mesh.vertexCount), b = key % mesh.vertexCount;
        if (!edges.has(b * mesh.vertexCount + a)) boundary++;
      }
      expect({ boundary, repeated }).toEqual({ boundary: 0, repeated: 0 });
      // Winding agrees with the stored normals (front faces are counter-clockwise outside).
      const view = new DataView(mesh.vertices), at = (v: number, o: number) => [0, 1, 2].map(axis => view.getFloat32(v * 80 + o + axis * 4, true));
      let agree = 0;
      for (let t = 0; t < mesh.indices.length; t += 3) {
        const [a, b, c] = [0, 1, 2].map(k => at(mesh.indices[t + k], 0)), n = at(mesh.indices[t], 16);
        const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const face = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        if (face[0] * n[0] + face[1] * n[1] + face[2] * n[2] > 0) agree++;
      }
      expect(agree / (mesh.indices.length / 3)).toBeGreaterThan(0.99);
    }
  });

  it('is deterministic per seed, with stored-float bounds and unit normals', () => {
    const a = g01Mesh(G01_SEED, 64, 16), b = g01Mesh(G01_SEED, 64, 16), c = g01Mesh(G01_SEED + 1, 64, 16);
    expect(new Uint8Array(a.vertices)).toEqual(new Uint8Array(b.vertices));
    expect(new Uint8Array(a.vertices)).not.toEqual(new Uint8Array(c.vertices));
    expect(g01ShapeParameters(G01_SEED)).toEqual(g01ShapeParameters(G01_SEED));
    expect(g01ShapeParameters(G01_SEED).detail).toHaveLength(G01_DETAIL_SPECTRUM.length * 2);
    const view = new DataView(a.vertices), min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < a.vertexCount; v++) {
      for (let axis = 0; axis < 3; axis++) {
        const value = view.getFloat32(v * 80 + axis * 4, true);
        min[axis] = Math.min(min[axis], value); max[axis] = Math.max(max[axis], value);
      }
      expect(Math.hypot(...[0, 1, 2].map(axis => view.getFloat32(v * 80 + 16 + axis * 4, true)))).toBeCloseTo(1, 5);
    }
    expect(a.bounds).toEqual({ min, max });
    expect(() => g01Mesh(G01_SEED, 8, 8)).toThrow('G01_MESH_RESOLUTION');
    expect(() => g01Mesh(G01_SEED, 64, 16, [0.2])).toThrow('G01_MESH_DETAIL');
    expect(g01Mesh(G01_SEED, 64, 16, []).vertexCount).toBe(64 * 16);
  });
});

describe('G01 trajectory', () => {
  it('is the fixed zoom and three orbits at 1/60 s steps', () => {
    const trajectory = new G01Trajectory();
    expect(trajectory.steps).toBe(1921);
    expect(trajectory.segments.map(segment => segment.name)).toEqual(['zoom-in', 'orbit-near', 'dolly-mid', 'orbit-mid', 'dolly-outer', 'orbit-outer']);
    expect(trajectory.pose(0)).toMatchObject({ segment: 'zoom-in', distance: 30, seconds: 0 });
    expect(trajectory.pose(600)).toMatchObject({ segment: 'orbit-near', distance: 2.6 });
    expect(trajectory.pose(1920).seconds).toBeCloseTo(1920 * G01_STEP_SECONDS, 12);
    let largestStep = 0, previous = trajectory.pose(0);
    for (let step = 1; step < trajectory.steps; step++) {
      const pose = trajectory.pose(step);
      largestStep = Math.max(largestStep, Math.hypot(pose.eye[0] - previous.eye[0], pose.eye[1] - previous.eye[1], pose.eye[2] - previous.eye[2]) / pose.distance);
      expect(Math.hypot(...pose.eye)).toBeCloseTo(pose.distance, 9);
      previous = pose;
    }
    // Continuous path: each fixed step moves the eye by under 2% of its distance (orbits turn 1 degree).
    expect(largestStep).toBeLessThan(0.02);
    // The zoom is exponential: equal relative speed per step.
    const ratio = trajectory.pose(1).distance / trajectory.pose(0).distance;
    expect(trajectory.pose(301).distance / trajectory.pose(300).distance).toBeCloseTo(ratio, 12);
    // Each orbit closes: start and end eyes coincide.
    for (const [start, end] of [[600, 960], [1080, 1440], [1560, 1920]]) {
      trajectory.pose(start).eye.forEach((value, axis) => expect(trajectory.pose(end).eye[axis]).toBeCloseTo(value, 9));
    }
    expect(new G01Trajectory().pose(777)).toEqual(trajectory.pose(777));
    expect(() => trajectory.pose(trajectory.steps)).toThrow('G01_TRAJECTORY_STEP');
    expect(() => new G01Trajectory({ nearDistance: 10 })).toThrow('G01_TRAJECTORY_DISTANCES');
    expect(g01ProjectionScale(320)).toBeCloseTo(320 / (2 * Math.tan(Math.PI / 6.4)), 12);
  });
});
