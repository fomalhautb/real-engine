import { beforeAll, describe, expect, it } from 'vitest';
import type { MorphTarget } from '../src/deformation';
import type { MeshData } from '../src/geometry';
import { generateGltfTangents, prepareGltfTangents } from '../src/gltf-tangents';

const budget = { maxVertices: 64, maxTangentBytes: 1024 * 1024 };
type Vec2 = readonly [number, number];
type Vec3 = readonly [number, number, number];

function meshOf(positions: readonly Vec3[], uvs: readonly Vec2[] | undefined, indices: readonly number[]): MeshData {
  const vertices = new ArrayBuffer(positions.length * 80), view = new DataView(vertices);
  for (let i = 0; i < positions.length; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, positions[i][axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true); // +Z normal.
    // Distinct uint32 joints catch copying through a Float32Array, which loses integer bits.
    for (let joint = 0; joint < 4; joint++) view.setUint32(offset + 32 + joint * 4, 0x01000001 + i * 17 + joint, true);
    view.setFloat32(offset + 48, 0.1 * (i + 1), true);
    view.setFloat32(offset + 52, 0.2, true);
    view.setFloat32(offset + 56, 0.3, true);
    view.setFloat32(offset + 60, 0.4, true);
    view.setFloat32(offset + 64, i + 0.125, true); // Embedded rest morph data.
  }
  return {
    vertices, indices: Uint32Array.from(indices), vertexCount: positions.length,
    ...(uvs ? { uvs: Float32Array.from(uvs.flat()) } : {}),
    bounds: { min: [0, 0, 0], max: [1, 1, 0] },
  };
}

function plane(): MeshData {
  return meshOf([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 2, 0, 2, 3]);
}

function at(mesh: MeshData, vertex: number): Vec3 {
  const view = new DataView(mesh.vertices), offset = vertex * 80;
  return [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)];
}

function tangent(mesh: MeshData, vertex: number): number[] {
  return Array.from(mesh.tangents!.subarray(vertex * 4, vertex * 4 + 4));
}

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, digits = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i], digits);
}

describe('Mikk-compatible glTF tangent generation', () => {
  beforeAll(async () => { await prepareGltfTangents(); });

  it('matches the independent planar +X,+Z golden frame without deindexing a smooth plane', () => {
    const source = plane();
    const originalIndices = new Uint32Array(source.indices), originalVertices = source.vertices.slice(0);
    const { mesh, morphTargets } = generateGltfTangents(source, [], budget);
    expect(mesh.vertexCount).toBe(4);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2, 0, 2, 3]);
    expect(morphTargets).toHaveLength(0);
    for (let i = 0; i < 4; i++) close(tangent(mesh, i), [1, 0, 0, 1]);
    expect(Array.from(source.indices)).toEqual(Array.from(originalIndices));
    expect(new Uint8Array(source.vertices)).toEqual(new Uint8Array(originalVertices));
    expect(source.tangents).toBeUndefined();
  });

  it('uses angle-weighted Compatible smoothing at a shared planar corner', () => {
    // The two corner angles at 0 are both 45 degrees. Their face tangents are
    // +X and unit(1,-2,0), so the expected sum has no area weighting.
    const source = meshOf([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 3, 0]],
      [[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 2, 0, 2, 3]);
    const { mesh } = generateGltfTangents(source, [], budget);
    expect(mesh.vertexCount).toBe(4);
    const weight = 1 / Math.sqrt(5), length = Math.hypot(1 + weight, 2 * weight);
    close(tangent(mesh, mesh.indices[0]), [(1 + weight) / length, -2 * weight / length, 0, 1], 3);
    expect(mesh.indices[0]).toBe(mesh.indices[3]);
  });

  it('keeps the same frame at very small and very large uniform scales', () => {
    for (const scale of [1e-20, 1e20]) {
      const source = meshOf([[0, 0, 0], [scale, 0, 0], [0, scale, 0]],
        [[0, 0], [1, 0], [0, 1]], [0, 1, 2]);
      const { mesh } = generateGltfTangents(source, [], budget);
      for (let vertex = 0; vertex < 3; vertex++) close(tangent(mesh, vertex), [1, 0, 0, 1]);
    }
  });

  it('splits shared corners with opposite UV orientation and copies every source stream byte-for-byte', () => {
    // UV triangle 0 has positive determinant; triangle 1 has negative determinant.
    // Both triangles share source vertices 0 and 2, which cannot carry both w signs.
    const source = meshOf([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
      [[0, 0], [1, 0], [0, 1], [1, 0]], [0, 1, 2, 0, 2, 3]);
    source.colors = new Float32Array([1, 0, 0, 0.25, 0, 1, 0, 0.5, 0, 0, 1, 0.75, 1, 1, 1, 1]);
    const original = source.vertices.slice(0), originalUvs = new Float32Array(source.uvs!);
    const morph: MorphTarget = { positions: new Float32Array(12), normals: new Float32Array(12) };
    for (let i = 0; i < 4; i++) {
      morph.positions[i * 3] = 0.01 * (i + 1);
      morph.normals[i * 3 + 1] = 0.01 * (i + 1);
    }
    const { mesh, morphTargets } = generateGltfTangents(source, [morph], budget);
    expect(mesh.vertexCount).toBe(6);
    const cornerSource = [0, 1, 2, 0, 2, 3];
    for (let corner = 0; corner < 6; corner++) {
      const output = mesh.indices[corner], input = cornerSource[corner];
      expect(new Uint8Array(mesh.vertices, output * 80, 80)).toEqual(new Uint8Array(original, input * 80, 80));
      close(mesh.uvs!.subarray(output * 2, output * 2 + 2), originalUvs.subarray(input * 2, input * 2 + 2));
      close(mesh.colors!.subarray(output * 4, output * 4 + 4), source.colors.subarray(input * 4, input * 4 + 4));
      close(morphTargets[0].positions.subarray(output * 3, output * 3 + 3),
        morph.positions.subarray(input * 3, input * 3 + 3));
      close(morphTargets[0].normals.subarray(output * 3, output * 3 + 3),
        morph.normals.subarray(input * 3, input * 3 + 3));
      expect(tangent(mesh, output)[3]).toBe(corner < 3 ? 1 : -1);
      expect(at(mesh, output)).toEqual(at(source, input));
    }
    expect(mesh.indices[0]).not.toBe(mesh.indices[3]);
    expect(mesh.indices[2]).not.toBe(mesh.indices[4]);
    expect(new Uint8Array(source.vertices)).toEqual(new Uint8Array(original));
    expect(Array.from(source.uvs!)).toEqual(Array.from(originalUvs));
    expect(Array.from(source.indices)).toEqual(cornerSource);
    expect(morph.tangents).toBeUndefined();
  });

  it('generates morph tangent deltas and splits a target-only tangent seam', () => {
    // Two triangle fans meet only at vertex 0. Both base frames are +X, but a
    // morph rotates the tangent of the second fan, requiring two copies of 0.
    const source = meshOf([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
      [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], [0, 1, 2, 0, 3, 4]);
    const morph: MorphTarget = { positions: new Float32Array(15), normals: new Float32Array(15) };
    morph.positions[3 * 3 + 1] = 0.5; // Rotate only the second fan's tangent.
    const { mesh, morphTargets } = generateGltfTangents(source, [morph], budget);
    expect(mesh.vertexCount).toBe(6);
    expect(morphTargets[0].tangents).toHaveLength(mesh.vertexCount * 3);
    const first = mesh.indices[0], second = mesh.indices[3];
    expect(at(mesh, first)).toEqual([0, 0, 0]);
    expect(at(mesh, second)).toEqual([0, 0, 0]);
    expect(first).not.toBe(second);
    close(tangent(mesh, first), [1, 0, 0, 1]);
    close(tangent(mesh, second), [1, 0, 0, 1]);
    close(morphTargets[0].tangents!.subarray(first * 3, first * 3 + 3), [0, 0, 0]);
    const targetDelta = morphTargets[0].tangents!.subarray(second * 3, second * 3 + 3);
    expect(Math.hypot(...targetDelta)).toBeGreaterThan(0.1);
    expect(targetDelta[1]).toBeLessThan(-0.1);
    expect(morph.tangents).toBeUndefined();
  });

  it('rejects a morph target whose frame changes handedness', () => {
    const source = meshOf([[0, 0, 0], [1, 0, 0], [0, 1, 0]],
      [[0, 0], [1, 0], [0, 1]], [0, 1, 2]);
    const morph: MorphTarget = { positions: new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
      normals: new Float32Array(9) }; // Collapse the first edge; Mikk's degenerate frame changes w.
    expect(() => generateGltfTangents(source, [morph], budget)).toThrow(/GLTF_TANGENT_HANDEDNESS_CHANGE/);
  });

  it('enforces final vertex and temporary work budgets, including invalid options', () => {
    const source = plane();
    expect(() => generateGltfTangents(source, [], { ...budget, maxVertices: 3 }))
      .toThrow(/GLTF_VERTEX_BUDGET/);
    const mirrored = meshOf([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
      [[0, 0], [1, 0], [0, 1], [1, 0]], [0, 1, 2, 0, 2, 3]);
    expect(() => generateGltfTangents(mirrored, [], { ...budget, maxVertices: 5 }))
      .toThrow(/GLTF_VERTEX_BUDGET/); // Four inputs but six final seam vertices.
    expect(() => generateGltfTangents(source, [], { ...budget, maxTangentBytes: 1 }))
      .toThrow(/GLTF_TANGENT_WORK_BUDGET/);
    for (const maxVertices of [0, -1, 1.5, Number.NaN, 10_000_001]) {
      expect(() => generateGltfTangents(source, [], { ...budget, maxVertices }))
        .toThrow(/GLTF_VERTEX_BUDGET/);
    }
    for (const maxTangentBytes of [0, -1, 1.5, Number.NaN, 512 * 1024 * 1024 + 1]) {
      expect(() => generateGltfTangents(source, [], { ...budget, maxTangentBytes }))
        .toThrow(/GLTF_TANGENT_WORK_BUDGET/);
    }
  });

  it('requires UV0 for tangent generation', () => {
    const source = plane();
    delete source.uvs;
    expect(() => generateGltfTangents(source, [], budget)).toThrow(/GLTF_NORMAL_TEXTURE_REQUIRES_UV0/);
  });

  it('bounds repeated high-valence triangles before invoking the WASM generator', () => {
    const source = meshOf([[0, 0, 0], [1, 0, 0], [0, 1, 0]],
      [[0, 0], [1, 0], [0, 1]], Array.from({ length: 5_000 }, () => [0, 1, 2]).flat());
    expect(() => generateGltfTangents(source, [], { maxVertices: 64, maxTangentBytes: 16 * 1024 * 1024 }))
      .toThrow(/GLTF_TANGENT_ADJACENCY_WORK_BUDGET/);
  });
});
