import { describe, expect, it } from 'vitest';
import { parseObj } from '../src/mesh-import';
import type { MeshData } from '../src/geometry';

function vertex(mesh: MeshData, index: number): { position: number[]; normal: number[]; uv: number[] } {
  const view = new DataView(mesh.vertices);
  const base = index * 80;
  return {
    position: [0, 4, 8].map((offset) => view.getFloat32(base + offset, true)),
    normal: [16, 20, 24].map((offset) => view.getFloat32(base + offset, true)),
    uv: [...mesh.uvs!.slice(index * 2, index * 2 + 2)],
  };
}

describe('Wavefront OBJ importer', () => {
  it('indexes a convex quad with negative indices, UVs, bounds, winding and flat normals', () => {
    const result = parseObj(`
      v -1 0 -1
      v 1 0 -1
      v 1 0 1
      v -1 0 1
      vt 0 0
      vt 1 0
      vt 1 1
      vt 0 1
      f -4/-4 -3/-3 -2/-2 -1/-1
    `);
    const mesh = result.primitives[0].mesh;
    expect(mesh.vertexCount).toBe(4);
    expect(mesh.indices).toEqual(new Uint32Array([0, 1, 2, 0, 2, 3]));
    expect(mesh.vertices.byteLength).toBe(4 * 80);
    expect(mesh.uvs).toEqual(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]));
    expect(mesh.bounds).toEqual({ min: [-1, 0, -1], max: [1, 0, 1] });
    expect(vertex(mesh, 0).normal).toEqual([0, -1, 0]);
    expect(result.warnings).toEqual([]);
  });

  it('splits UV seams and explicit hard edges but reuses matching corners', () => {
    const result = parseObj(`
      v 0 0 0
      v 1 0 0
      v 0 1 0
      v 0 0 1
      vt 0 0
      vt 1 0
      vt 0 1
      vt 0.5 0.5
      vn 0 0 1
      vn 0 -1 0
      f 1/1/1 2/2/1 3/3/1
      f 1/1/1 3/3/1 2/2/1
      f 1/4/2 2/2/2 4/3/2
    `);
    const mesh = result.primitives[0].mesh;
    expect(mesh.vertexCount).toBe(6);
    expect([...mesh.indices]).toEqual([0, 1, 2, 0, 2, 1, 3, 4, 5]);
    expect(vertex(mesh, 0).uv).toEqual([0, 0]);
    expect(vertex(mesh, 3).uv).toEqual([0.5, 0.5]);
    expect(vertex(mesh, 0).normal).toEqual([0, 0, 1]);
    expect(vertex(mesh, 3).normal).toEqual([0, -1, 0]);
    expect(mesh.bounds).toEqual({ min: [0, 0, 0], max: [1, 1, 1] });
  });

  it('generates separate flat normals on adjacent faces without vn', () => {
    const mesh = parseObj(`
      v 0 0 0
      v 1 0 0
      v 0 1 0
      v 0 0 1
      f 1 2 3
      f 1 4 2
    `).primitives[0].mesh;
    expect(mesh.vertexCount).toBe(6);
    expect(vertex(mesh, 0).normal).toEqual([0, 0, 1]);
    expect(vertex(mesh, 3).normal).toEqual([0, 1, 0]);
    expect(mesh.uvs).toEqual(new Float32Array(12));
  });

  it('accepts v//vn corners with negative normal references', () => {
    const mesh = parseObj(`
      v 0 0 0
      v 1 0 0
      v 0 1 0
      vn 0 0 2
      f 1//-1 2//-1 3//-1
    `).primitives[0].mesh;
    expect(mesh.vertexCount).toBe(3);
    expect(vertex(mesh, 0).normal).toEqual([0, 0, 1]);
  });

  it('preserves contiguous object, group and material sections', () => {
    const result = parseObj(`
      mtllib scene.mtl
      o chapel
      v 0 0 0
      v 1 0 0
      v 0 1 0
      usemtl brick
      f 1 2 3
      g trim
      usemtl copper
      f 1 2 3
      usemtl brick
      f 1 2 3
    `);
    expect(result.materialLibraries).toEqual(['scene.mtl']);
    expect(result.primitives.map(({ name, materialName }) => [name, materialName])).toEqual([
      ['chapel', 'brick'], ['chapel/trim', 'copper'], ['chapel/trim', 'brick'],
    ]);
    expect(result.primitives.every(({ mesh }) => mesh.vertexCount === 3)).toBe(true);
  });

  it('warns about smoothing directives and rejects concave or non-planar polygons', () => {
    const flat = parseObj('v 0 0 0\nv 1 0 0\nv 0 1 0\ns 1\nf 1 2 3');
    expect(flat.warnings).toContain('OBJ smoothing groups are ignored; faces without vn receive flat normals.');
    expect(() => parseObj('v 0 0 0\nv 2 0 0\nv 1 0.5 0\nv 2 2 0\nv 0 2 0\nf 1 2 3 4 5'))
      .toThrow(/concave/);
    expect(() => parseObj('v 0 0 0\nv 1 0 0\nv 1 1 1\nv 0 1 0\nf 1 2 3 4'))
      .toThrow(/non-planar/);
  });

  it.each([
    ['zero index', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3', /position index/],
    ['out-of-range index', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4', /out of range/],
    ['bad UV index', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1\/1 2\/1 3\/1', /texture index/],
    ['degenerate face', 'v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3', /degenerate/],
    ['nonfinite position', 'v 1e100 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3', /finite float32/],
    ['homogeneous position', 'v 0 0 0 1\nv 1 0 0\nv 0 1 0\nf 1 2 3', /homogeneous/],
    ['unsupported geometry', 'l 1 2', /unsupported directive/],
  ])('rejects %s', (_name, source, message) => {
    expect(() => parseObj(source)).toThrow(message);
  });

  it('enforces output and polygon budgets', () => {
    const source = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 4 2';
    expect(() => parseObj(source, { maxVertices: 3 })).toThrow(/position record budget/);
    expect(() => parseObj(source, { maxIndices: 3 })).toThrow(/index budget/);
    expect(() => parseObj(source, { maxVertices: 4 })).toThrow(/vertex budget/);
    const quad = 'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4';
    expect(() => parseObj(quad, { maxFaceVertices: 3 })).toThrow(/3 to 3 corners/);
  });
});
