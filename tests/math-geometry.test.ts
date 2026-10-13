import { describe, expect, it } from 'vitest';
import {
  identity, lookAt, multiply, perspective, quaternionMatrix, rotationX, rotationY,
  rotationZ, scaling, transformPoint, translation,
} from '../src/math';
import { createCharacter, createCube, createPlane, createTree, type MeshData } from '../src/geometry';

const close = (actual: readonly number[], expected: readonly number[]) => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 5);
};

describe('column-major transforms', () => {
  it('multiplies in application order and includes translation', () => {
    close(transformPoint(identity(), [2, 3, 4]), [2, 3, 4]);
    const model = multiply(translation(5, -1, 2), scaling(2, 3, 4));
    close(transformPoint(model, [1, 2, 3]), [7, 5, 14]);
  });

  it('maps the view eye to the origin and the target down -Z', () => {
    const eye = [2, 3, 7];
    const target = [-1, 1, 2];
    const view = lookAt(eye, target, [0, 1, 0]);
    close(transformPoint(view, eye), [0, 0, 0]);
    close(transformPoint(view, target), [0, 0, -Math.hypot(3, 2, 5)]);
  });

  it('maps camera near and far planes to WebGPU depth 0 and 1', () => {
    const projection = perspective(Math.PI / 2, 2, 0.1, 100);
    expect(transformPoint(projection, [0, 0, -0.1])[2]).toBeCloseTo(0, 5);
    expect(transformPoint(projection, [0, 0, -100])[2]).toBeCloseTo(1, 5);
    expect(transformPoint(projection, [2, 0, -2])[0]).toBeCloseTo(0.5, 5);
  });

  it('agrees with axis rotations converted from quaternions', () => {
    const angle = 0.7;
    const half = angle / 2;
    const pairs = [
      [rotationX(angle), quaternionMatrix(Math.sin(half), 0, 0, Math.cos(half))],
      [rotationY(angle), quaternionMatrix(0, Math.sin(half), 0, Math.cos(half))],
      [rotationZ(angle), quaternionMatrix(0, 0, Math.sin(half), Math.cos(half))],
    ];
    for (const [axis, quaternion] of pairs) close([...axis], [...quaternion]);
  });
});

describe('procedural meshes', () => {
  const meshes = {
    character: createCharacter(),
    tree: createTree(),
    cube: createCube(),
    plane: createPlane(),
  };

  for (const [name, mesh] of Object.entries(meshes)) {
    it(`${name} has valid interleaved vertices, indices, bounds, and CCW faces`, () => {
      validateMesh(mesh);
    });
  }

  it('has the requested size, skin groups, and morph targets', () => {
    close(meshes.character.bounds.min.slice(1, 2), [0]);
    close(meshes.character.bounds.max.slice(1, 2), [3]);
    close(meshes.tree.bounds.min.slice(1, 2), [0]);
    close(meshes.tree.bounds.max.slice(1, 2), [4]);
    expect(usedJoints(meshes.character)).toEqual(new Set([0, 1, 2, 3]));
    expect(usedJoints(meshes.tree)).toEqual(new Set([0, 1, 2, 3]));
    expect(hasMorph(meshes.character)).toBe(true);
    expect(hasMorph(meshes.cube)).toBe(false);
    expect(hasBlendedSkin(meshes.tree)).toBe(true);
  });
});

function validateMesh(mesh: MeshData): void {
  expect(mesh.vertexCount).toBeGreaterThan(0);
  expect(mesh.vertices.byteLength).toBe(mesh.vertexCount * 80);
  expect(mesh.indices.length % 3).toBe(0);
  const view = new DataView(mesh.vertices);
  const p = (index: number): number[] => [0, 4, 8].map((offset) => view.getFloat32(index * 80 + offset, true));
  const n = (index: number): number[] => [16, 20, 24].map((offset) => view.getFloat32(index * 80 + offset, true));
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.vertexCount; i++) {
    const position = p(i), normal = n(i);
    expect([...position, ...normal].every(Number.isFinite)).toBe(true);
    expect(view.getFloat32(i * 80 + 12, true)).toBe(1);
    expect(view.getFloat32(i * 80 + 28, true)).toBe(0);
    expect(view.getFloat32(i * 80 + 76, true)).toBe(0);
    const weights = [0, 1, 2, 3].map((j) => view.getFloat32(i * 80 + 48 + 4 * j, true));
    for (const weight of weights) expect(weight).toBeGreaterThanOrEqual(0);
    expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 5);
    for (let j = 0; j < 4; j++) {
      expect(view.getUint32(i * 80 + 32 + 4 * j, true)).toBe(j);
    }
    for (let j = 0; j < 3; j++) {
      min[j] = Math.min(min[j], position[j]);
      max[j] = Math.max(max[j], position[j]);
    }
    for (const offset of [64, 68, 72]) expect(Number.isFinite(view.getFloat32(i * 80 + offset, true))).toBe(true);
  }
  close(mesh.bounds.min, min.slice(0, 3));
  close(mesh.bounds.max, max.slice(0, 3));
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const [a, b, c] = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
    expect(a).toBeLessThan(mesh.vertexCount);
    expect(b).toBeLessThan(mesh.vertexCount);
    expect(c).toBeLessThan(mesh.vertexCount);
    const pa = p(a), pb = p(b), pc = p(c), normal = n(a);
    const ab = pb.map((value, j) => value - pa[j]);
    const ac = pc.map((value, j) => value - pa[j]);
    const cross = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    expect(cross.reduce((sum, value, j) => sum + value * normal[j], 0)).toBeGreaterThan(0);
  }
}

function usedJoints(mesh: MeshData): Set<number> {
  const view = new DataView(mesh.vertices);
  const joints = new Set<number>();
  for (let i = 0; i < mesh.vertexCount; i++) {
    for (let j = 0; j < 4; j++) {
      if (view.getFloat32(i * 80 + 48 + j * 4, true) > 0) joints.add(j);
    }
  }
  return joints;
}

function hasMorph(mesh: MeshData): boolean {
  const view = new DataView(mesh.vertices);
  for (let i = 0; i < mesh.vertexCount; i++) {
    for (const offset of [64, 68, 72]) {
      if (view.getFloat32(i * 80 + offset, true) !== 0) return true;
    }
  }
  return false;
}

function hasBlendedSkin(mesh: MeshData): boolean {
  const view = new DataView(mesh.vertices);
  for (let i = 0; i < mesh.vertexCount; i++) {
    const root = view.getFloat32(i * 80 + 48, true);
    const upper = view.getFloat32(i * 80 + 52, true);
    if (root > 0 && upper > 0) return true;
  }
  return false;
}
