import { describe, expect, it } from 'vitest';
import { cleanStaticGeometry } from '../src/geometry-clean';
import type { MeshData } from '../src/geometry';

const REST_BYTES = 80;

function fixture(): MeshData {
  const positions = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0],
    [9, 9, 9], [2 ** -100, 0, 0], [0, 2 ** -100, 0],
  ];
  const vertices = new ArrayBuffer(positions.length * REST_BYTES), view = new DataView(vertices);
  const uvs = new Float32Array(positions.length * 2);
  const tangents = new Float32Array(positions.length * 4);
  const colors = new Float32Array(positions.length * 4);
  for (let vertex = 0; vertex < positions.length; vertex++) {
    const offset = vertex * REST_BYTES;
    for (let word = 0; word < 20; word++) {
      if (word >= 8 && word < 12) view.setUint32(offset + word * 4, 0x7fc0_0000 + vertex + word, true);
      else view.setFloat32(offset + word * 4, word / 32 + vertex / 16, true);
    }
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, positions[vertex][axis], true);
    uvs.set([vertex / 8, vertex / 16], vertex * 2);
    tangents.set([vertex / 8, 0.25, 0.5, vertex % 2 ? -1 : 1], vertex * 4);
    colors.set([vertex / 8, 0.25, 0.75, 1], vertex * 4);
  }
  return { vertices, vertexCount: positions.length,
    // Triangles 0/1 share positions but use different seam vertices. Triangle 4
    // is a visible duplicate. Triangle 3 has tiny but nonzero exact f32 area.
    indices: new Uint32Array([0, 1, 2, 3, 1, 2, 0, 0, 1, 0, 5, 6, 0, 1, 2]),
    uvs, tangents, colors, bounds: { min: [0, 0, 0], max: [9, 9, 9] } };
}

describe('offline static geometry cleaning', () => {
  it('keeps triangle order, winding, duplicates, seams, exact bytes and provenance', () => {
    const source = fixture(), original = structuredClone(source);
    const result = cleanStaticGeometry(source, { removeZeroAreaTriangles: true });
    expect(result.sourceTriangles).toEqual(new Uint32Array([0, 1, 3, 4]));
    expect(result.removedTriangles).toEqual(new Uint32Array([2]));
    expect(result.sourceVertices).toEqual(new Uint32Array([0, 1, 2, 3, 5, 6]));
    expect(result.mesh.indices).toEqual(new Uint32Array([0, 1, 2, 3, 1, 2, 0, 4, 5, 0, 1, 2]));
    expect(result.mesh.bounds).toEqual({ min: [0, 0, 0], max: [1, 1, 0] });
    const originalRest = new Uint8Array(source.vertices), outputRest = new Uint8Array(result.mesh.vertices);
    for (let vertex = 0; vertex < result.mesh.vertexCount; vertex++) {
      const sourceId = result.sourceVertices[vertex];
      expect(outputRest.slice(vertex * REST_BYTES, (vertex + 1) * REST_BYTES))
        .toEqual(originalRest.slice(sourceId * REST_BYTES, (sourceId + 1) * REST_BYTES));
      expect(result.mesh.uvs!.slice(vertex * 2, vertex * 2 + 2))
        .toEqual(source.uvs!.slice(sourceId * 2, sourceId * 2 + 2));
      expect(result.mesh.tangents!.slice(vertex * 4, vertex * 4 + 4))
        .toEqual(source.tangents!.slice(sourceId * 4, sourceId * 4 + 4));
      expect(result.mesh.colors!.slice(vertex * 4, vertex * 4 + 4))
        .toEqual(source.colors!.slice(sourceId * 4, sourceId * 4 + 4));
    }
    expect(source).toEqual(original);
    result.mesh.colors![0] = 1;
    expect(source.colors![0]).toBe(0);
  });

  it('rejects zero area by default and never treats a tiny nonzero triangle as zero', () => {
    const source = fixture();
    expect(() => cleanStaticGeometry(source)).toThrow('GEOMETRY_CLEAN_ZERO_AREA_TRIANGLE');
    const near = { ...source, indices: new Uint32Array([0, 5, 6]) };
    const result = cleanStaticGeometry(near);
    expect(result.removedTriangles).toHaveLength(0);
    expect(result.sourceTriangles).toEqual(new Uint32Array([0]));
    expect(result.mesh.vertexCount).toBe(3);
    expect(() => cleanStaticGeometry({ ...source, indices: new Uint32Array([0, 0, 1]) },
      { removeZeroAreaTriangles: true })).toThrow('GEOMETRY_CLEAN_EMPTY');
    const collinearRest = source.vertices.slice(0);
    const view = new DataView(collinearRest);
    view.setFloat32(4 * REST_BYTES, 2, true);
    view.setFloat32(4 * REST_BYTES + 4, 0, true);
    view.setFloat32(4 * REST_BYTES + 8, 0, true);
    expect(() => cleanStaticGeometry({ ...source, vertices: collinearRest,
      indices: new Uint32Array([0, 1, 4]) })).toThrow('GEOMETRY_CLEAN_ZERO_AREA_TRIANGLE');
  });

  it('uses exact float32 geometry when double subtraction rounds a nonzero area to zero', () => {
    const source = fixture(), vertices = source.vertices.slice(0), view = new DataView(vertices);
    const large = 2 ** 100, ulp = 2 ** 77;
    const setPosition = (vertex: number, x: number, y: number) => {
      view.setFloat32(vertex * REST_BYTES, x, true);
      view.setFloat32(vertex * REST_BYTES + 4, y, true);
      view.setFloat32(vertex * REST_BYTES + 8, 0, true);
    };
    setPosition(0, large, large);
    setPosition(1, 0, 1);
    setPosition(2, large + ulp, large + ulp);
    // In JS doubles, (1 - large) rounds to -large, making both products equal.
    expect((0 - large) * ulp - (1 - large) * ulp).toBe(0);
    const mesh = { ...source, vertices, indices: new Uint32Array([0, 1, 2]),
      bounds: { min: [0, 0, 0] as [number, number, number],
        max: [large + ulp, large + ulp, 0] as [number, number, number] } };
    const result = cleanStaticGeometry(mesh, { removeZeroAreaTriangles: true });
    expect(result.sourceTriangles).toEqual(new Uint32Array([0]));
    expect(result.removedTriangles).toHaveLength(0);
    expect(result.mesh.indices).toEqual(new Uint32Array([0, 1, 2]));
  });

  it('checks layout, all attributes, indices, unknown extensions and budgets before cleaning', () => {
    const source = fixture();
    expect(() => cleanStaticGeometry({ ...source, vertexCount: 8 })).toThrow('GEOMETRY_CLEAN_VERTEX_LAYOUT');
    expect(() => cleanStaticGeometry({ ...source, indices: new Uint32Array([0, 1]) }))
      .toThrow('GEOMETRY_CLEAN_TRIANGLE_LAYOUT');
    expect(() => cleanStaticGeometry({ ...source, indices: new Uint32Array([0, 1, 7]) }))
      .toThrow('GEOMETRY_CLEAN_INDEX');
    expect(() => cleanStaticGeometry({ ...source, indices: new Uint32Array() })).toThrow('GEOMETRY_CLEAN_EMPTY');
    expect(() => cleanStaticGeometry({ ...source, bounds: { min: [0, 0, 0], max: [NaN, 1, 1] } }))
      .toThrow('GEOMETRY_CLEAN_BOUNDS');
    expect(() => cleanStaticGeometry({ ...source, uvs: new Float32Array(2) })).toThrow('GEOMETRY_CLEAN_UVS');
    const badTangents = source.tangents!.slice(); badTangents[3] = 0;
    expect(() => cleanStaticGeometry({ ...source, tangents: badTangents })).toThrow(/tangent.*w/);
    const badColors = source.colors!.slice(); badColors[0] = 1.1;
    expect(() => cleanStaticGeometry({ ...source, colors: badColors })).toThrow('INVALID_MESH_COLORS');
    const badRest = source.vertices.slice(0); new DataView(badRest).setFloat32(4 * 4, Infinity, true);
    expect(() => cleanStaticGeometry({ ...source, vertices: badRest })).toThrow('GEOMETRY_CLEAN_NONFINITE_REST');
    expect(() => cleanStaticGeometry({ ...source, customWeights: new Float32Array(7) } as MeshData))
      .toThrow('GEOMETRY_CLEAN_UNSUPPORTED_ATTRIBUTE');
    expect(() => cleanStaticGeometry(source, { maxVertices: 6 })).toThrow('GEOMETRY_CLEAN_VERTEX_LAYOUT');
    expect(() => cleanStaticGeometry(source, { maxTriangles: 4 })).toThrow('GEOMETRY_CLEAN_TRIANGLE_LAYOUT');
    expect(() => cleanStaticGeometry(source, { maxVertices: 4_000_001 })).toThrow('GEOMETRY_CLEAN_MAX_VERTICES');
    expect(() => cleanStaticGeometry(source, { maxTriangles: 0 })).toThrow('GEOMETRY_CLEAN_MAX_TRIANGLES');
    expect(() => cleanStaticGeometry(source, { removeZeroAreaTriangles: 1 as never }))
      .toThrow('GEOMETRY_CLEAN_OPTIONS');
  });
});
