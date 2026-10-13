import { describe, expect, it } from 'vitest';
import { createCube, createPlane, type MeshData } from '../src/geometry';
import { authoredMorph, buildMorphTarget } from '../src/morph-assets';
import type { MorphTarget } from '../src/deformation';

function vertex(mesh: MeshData, index: number): { position: [number, number, number]; normal: [number, number, number] } {
  const data = new DataView(mesh.vertices);
  const read = (offset: number): [number, number, number] => [0, 1, 2].map(axis =>
    data.getFloat32(index * 80 + offset + axis * 4, true)) as [number, number, number];
  return { position: read(0), normal: read(16) };
}

function targetNormal(mesh: MeshData, target: MorphTarget, index: number): [number, number, number] {
  const base = vertex(mesh, index).normal;
  return [0, 1, 2].map(axis => base[axis] + target.normals[index * 3 + axis]) as [number, number, number];
}

describe('morph asset normal reconstruction', () => {
  it('matches the finite-difference normal of a tilted plane', () => {
    const mesh = createPlane();
    const height = (x: number, z: number): number => 0.3 * x - 0.2 * z;
    const target = buildMorphTarget(mesh, ([x, , z]) => [0, height(x, z), 0]);
    const epsilon = 0.001;
    for (let index = 0; index < mesh.vertexCount; index++) {
      const { position } = vertex(mesh, index);
      const dx = (height(position[0] + epsilon, position[2]) - height(position[0] - epsilon, position[2])) / (2 * epsilon);
      const dz = (height(position[0], position[2] + epsilon) - height(position[0], position[2] - epsilon)) / (2 * epsilon);
      const length = Math.hypot(-dx, 1, -dz);
      const expected = [-dx / length, 1 / length, -dz / length];
      const actual = targetNormal(mesh, target, index);
      for (let axis = 0; axis < 3; axis++) expect(actual[axis]).toBeCloseTo(expected[axis], 5);
    }
  });

  it('keeps coincident hard-edge vertices separate after a shear', () => {
    const mesh = createCube();
    const target = buildMorphTarget(mesh, ([, y]) => [0.3 * y, 0, 0]);
    const corner = Array.from({ length: mesh.vertexCount }, (_, index) => index).filter(index =>
      vertex(mesh, index).position.every(component => component === 0.5));
    expect(corner).toHaveLength(3);
    const xFace = corner.find(index => vertex(mesh, index).normal[0] === 1)!;
    const yFace = corner.find(index => vertex(mesh, index).normal[1] === 1)!;
    const xNormal = targetNormal(mesh, target, xFace);
    const yNormal = targetNormal(mesh, target, yFace);
    const length = Math.hypot(1, -0.3);
    expect(xNormal[0]).toBeCloseTo(1 / length, 5);
    expect(xNormal[1]).toBeCloseTo(-0.3 / length, 5);
    expect(yNormal).toEqual([0, 1, 0]);
  });

  it('returns zero deltas for an identity morph, including authored zero morphs', () => {
    const mesh = createCube();
    for (const target of [buildMorphTarget(mesh, () => [0, 0, 0]), authoredMorph(mesh)]) {
      for (const value of target.positions) expect(value).toBe(0);
      for (const value of target.normals) expect(value).toBeCloseTo(0, 5);
    }
  });

  it('rejects a degenerate target surface', () => {
    const mesh = createPlane();
    expect(() => buildMorphTarget(mesh, position =>
      [-position[0], -position[1], -position[2]])).toThrow(/degenerate vertex normal/);
  });
});
