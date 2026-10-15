import { describe, expect, it } from 'vitest';
import { importGltf } from '../src/gltf-import';
import type { GltfAccessor, GltfBufferView } from '../src/gltf-binary';
import type { MeshData } from '../src/geometry';

class BinaryFixture {
  bytes: number[] = [];
  bufferViews: GltfBufferView[] = [];
  accessors: GltfAccessor[] = [];

  add(data: Uint8Array | Uint16Array | Float32Array, componentType: number, type: string, count: number, normalized = false): number {
    while (this.bytes.length % 4) this.bytes.push(0);
    const offset = this.bytes.length;
    this.bytes.push(...new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    const bufferView = this.bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: data.byteLength }) - 1;
    return this.accessors.push({ bufferView, componentType, count, type, normalized }) - 1;
  }

  asset(primitive: Record<string, unknown>, options: { maxVertices?: number } = {}) {
    const binary = Uint8Array.from(this.bytes).buffer;
    const doc = { asset: { version: '2.0' }, buffers: [{ byteLength: binary.byteLength }],
      bufferViews: this.bufferViews, accessors: this.accessors, meshes: [{ primitives: [primitive] }] };
    return importGltf(doc, [binary], options);
  }
}

function components(mesh: MeshData, vertex: number, start: number, count: number): number[] {
  const data = new DataView(mesh.vertices);
  return Array.from({ length: count }, (_, axis) => data.getFloat32(vertex * 80 + start + axis * 4, true));
}

describe('glTF mesh conversion', () => {
  it('deindexes missing-normal faces and remaps UVs, weights, and morph deltas', () => {
    const fixture = new BinaryFixture();
    const position = fixture.add(new Float32Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    ]), 5126, 'VEC3', 4);
    const uv = fixture.add(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 5126, 'VEC2', 4);
    const joints = fixture.add(new Uint8Array([
      1, 2, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0,
    ]), 5121, 'VEC4', 4);
    const weights = fixture.add(new Float32Array([
      2, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    ]), 5126, 'VEC4', 4);
    const indices = fixture.add(new Uint16Array([0, 1, 2, 0, 3, 1]), 5123, 'SCALAR', 6);
    const targetPosition = fixture.add(new Float32Array([
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    ]), 5126, 'VEC3', 4);
    const asset = fixture.asset({ attributes: { POSITION: position, TEXCOORD_0: uv, JOINTS_0: joints, WEIGHTS_0: weights },
      indices, targets: [{ POSITION: targetPosition }] });
    const { mesh, morphTargets } = asset.meshes[0].primitives[0];
    expect(mesh.vertexCount).toBe(6);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(components(mesh, 0, 16, 3)).toEqual([0, 0, 1]);
    expect(components(mesh, 3, 16, 3)).toEqual([0, 1, 0]);
    expect(Array.from(mesh.uvs!)).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0]);
    expect(components(mesh, 0, 48, 4)).toEqual([0.5, 0.5, 0, 0]);
    expect(components(mesh, 3, 48, 4)).toEqual([0.5, 0.5, 0, 0]);
    expect(new DataView(mesh.vertices).getUint32(3 * 80 + 32, true)).toBe(1);
    expect(Array.from(morphTargets[0].positions)).toEqual([
      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ]);
    expect(morphTargets[0].normals[0]).toBeCloseTo(-Math.SQRT1_2);
    expect(morphTargets[0].normals[2]).toBeCloseTo(Math.SQRT1_2 - 1);
    expect(morphTargets[0].normals[9]).toBeCloseTo(0);
    expect(asset.warnings).toContain('mesh 0: missing normals generated as flat face normals.');
    expect(() => fixture.asset({ attributes: { POSITION: position }, indices }, { maxVertices: 5 })).toThrow(/VERTEX_BUDGET/);
  });

  it('preserves authored normals and base/target tangents while omitted target normals stay zero', () => {
    const fixture = new BinaryFixture();
    const position = fixture.add(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 5126, 'VEC3', 3);
    const normal = fixture.add(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), 5126, 'VEC3', 3);
    const tangent = fixture.add(new Float32Array([1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1]), 5126, 'VEC4', 3);
    const targetPosition = fixture.add(new Float32Array([0, 0, 0, 0, 0, 1, 0, 0, 0]), 5126, 'VEC3', 3);
    const targetTangent = fixture.add(new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]), 5126, 'VEC3', 3);
    const asset = fixture.asset({ attributes: { POSITION: position, NORMAL: normal, TANGENT: tangent },
      targets: [{ POSITION: targetPosition, TANGENT: targetTangent }, {}] });
    const { mesh, morphTargets } = asset.meshes[0].primitives[0];
    expect(mesh.vertexCount).toBe(3);
    expect(Array.from(mesh.tangents!)).toEqual([1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1]);
    expect(Array.from(morphTargets[0].tangents!)).toEqual([0, 1, 0, 0, 1, 0, 0, 1, 0]);
    expect(Array.from(morphTargets[0].normals)).toEqual(new Array(9).fill(0));
    expect(Array.from(morphTargets[1].normals)).toEqual(new Array(9).fill(0));
    expect(asset.warnings.some(warning => warning.includes('current isotropic material path'))).toBe(false);
  });

  it('rejects morph normal/tangent attributes without corresponding base attributes', () => {
    const fixture = new BinaryFixture();
    const position = fixture.add(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 5126, 'VEC3', 3);
    const normal = fixture.add(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), 5126, 'VEC3', 3);
    expect(() => fixture.asset({ attributes: { POSITION: position }, targets: [{ NORMAL: normal }] }))
      .toThrow(/MORPH_NORMAL_REQUIRES_BASE_NORMAL/);
    expect(() => fixture.asset({ attributes: { POSITION: position, NORMAL: normal }, targets: [{ TANGENT: normal }] }))
      .toThrow(/MORPH_TANGENT_REQUIRES_BASE_TANGENT/);
  });
});
