import { describe, expect, it } from 'vitest';
import { extractGltfStaticPrimitive } from '../src/gltf-geometry-build';

type Doc = Record<string, any>;
function fixture() {
  const buffer = new ArrayBuffer(256), data = new DataView(buffer);
  const positions = [-1, -1, 0, 1, -1, 0, 0, 1, 0];
  const normals = [0, 0, 1, 0, 0, 1, 0, 0, 1];
  const tangents = [1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1];
  const uv0 = [0, 0, 1, 0, 0.5, 1], uv1 = [1, 1, 0, 1, 0.5, 0];
  const colors = [255, 0, 32, 64, 0, 255, 16, 128, 128, 64, 255, 255];
  let offset = 0;
  const views: { buffer: number; byteOffset: number; byteLength: number }[] = [];
  const accessors: Doc[] = [];
  const floats = (values: number[], type: string, count: number) => {
    const start = offset;
    for (const value of values) { data.setFloat32(offset, value, true); offset += 4; }
    views.push({ buffer: 0, byteOffset: start, byteLength: offset - start });
    accessors.push({ bufferView: views.length - 1, componentType: 5126, count, type });
    return accessors.length - 1;
  };
  const attrs: Doc = {
    POSITION: floats(positions, 'VEC3', 3), NORMAL: floats(normals, 'VEC3', 3),
    TANGENT: floats(tangents, 'VEC4', 3), TEXCOORD_0: floats(uv0, 'VEC2', 3),
    TEXCOORD_1: floats(uv1, 'VEC2', 3),
  };
  const colorStart = offset;
  for (const value of colors) data.setUint8(offset++, value);
  views.push({ buffer: 0, byteOffset: colorStart, byteLength: 12 });
  attrs.COLOR_0 = accessors.push({ bufferView: views.length - 1, componentType: 5121,
    normalized: true, count: 3, type: 'VEC4' }) - 1;
  const indexStart = offset;
  for (const value of [0, 1, 2]) { data.setUint16(offset, value, true); offset += 2; }
  views.push({ buffer: 0, byteOffset: indexStart, byteLength: 6 });
  const indexAccessor = accessors.push({ bufferView: views.length - 1, componentType: 5123,
    count: 3, type: 'SCALAR' }) - 1;
  const json: Doc = { asset: { version: '2.0' }, buffers: [{ byteLength: offset }],
    bufferViews: views, accessors, textures: [], materials: [{ alphaMode: 'OPAQUE' }],
    meshes: [{ primitives: [{ attributes: attrs, indices: indexAccessor, material: 0 }] }] };
  return { json, buffers: [buffer], attrs, indexAccessor };
}

describe('static glTF primitive extraction', () => {
  it('preserves colour, tangent, UV0 and BLEND classification without changing the source', () => {
    const { json, buffers } = fixture();
    json.materials[0].alphaMode = 'BLEND';
    const before = JSON.stringify(json), binary = buffers[0].slice(0);
    const result = extractGltfStaticPrimitive(json, buffers, 0, 0);
    expect(result.coverage).toBe('blend');
    expect(result.materialIndex).toBe(0);
    expect(result.ignoredAttributes).toEqual(['TEXCOORD_1']);
    expect(result.mesh.vertexCount).toBe(3);
    expect(Array.from(result.mesh.indices)).toEqual([0, 1, 2]);
    expect(Array.from(result.mesh.tangents!)).toEqual([1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1]);
    expect(Array.from(result.mesh.uvs!)).toEqual([0, 0, 1, 0, 0.5, 1]);
    expect(result.mesh.colors![0]).toBeCloseTo(1);
    expect(result.mesh.colors![3]).toBeCloseTo(64 / 255);
    expect(result.mesh.colors![5]).toBeCloseTo(1);
    expect(JSON.stringify(json)).toBe(before);
    expect(new Uint8Array(buffers[0])).toEqual(new Uint8Array(binary));
  });

  it('omits only proven-unused higher UV sets, including with geometry-unrelated extensions', () => {
    const { json, buffers } = fixture();
    json.extensionsRequired = ['KHR_lights_punctual', 'KHR_materials_unlit'];
    json.materials[0].extensions = { KHR_materials_unlit: {} };
    expect(extractGltfStaticPrimitive(json, buffers, 0, 0).ignoredAttributes).toEqual(['TEXCOORD_1']);
    json.textures = [{ source: 0 }];
    json.materials[0].pbrMetallicRoughness = { baseColorTexture: { index: 0, texCoord: 1 } };
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_UV_IN_USE');
    json.materials[0].pbrMetallicRoughness.baseColorTexture.texCoord = 0;
    json.materials[0].pbrMetallicRoughness.baseColorTexture.extensions = { KHR_texture_transform: {} };
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_UV_IN_USE');
    delete json.materials[0].pbrMetallicRoughness.baseColorTexture.extensions;
    json.textures[0].newUvSelector = 1;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_UV_UNKNOWN_TEXTURE_FIELD');
    delete json.textures[0].newUvSelector;
    json.materials[0].newTextureInfo = { index: 0, texCoord: 1 };
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_UV_UNKNOWN_MATERIAL_FIELD');
    delete json.materials[0].newTextureInfo;
    json.materials[0].extensions.UNKNOWN = {};
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_UV_MATERIAL_EXTENSION');
    delete json.materials[0].extensions.UNKNOWN;
    json.extensionsRequired.push('UNKNOWN_required');
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_REQUIRED_EXTENSION');
  });

  it('rejects dynamic geometry, unsupported primitive semantics and missing tangent-space inputs', () => {
    const { json, buffers, attrs } = fixture();
    const primitive = json.meshes[0].primitives[0];
    primitive.targets = [{ POSITION: attrs.POSITION }];
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_DYNAMIC_GEOMETRY');
    delete primitive.targets;
    primitive.attributes.JOINTS_0 = attrs.COLOR_0;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_DYNAMIC_GEOMETRY');
    delete primitive.attributes.JOINTS_0;
    primitive.extensions = { ANY: {} };
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_PRIMITIVE_EXTENSION');
    delete primitive.extensions;
    primitive.mode = 1;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_TRIANGLES_REQUIRED');
    primitive.mode = 4;
    json.textures = [{ source: 0 }];
    json.materials[0].normalTexture = { index: 0 };
    const uv0 = primitive.attributes.TEXCOORD_0;
    delete primitive.attributes.TEXCOORD_0;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_MISSING_UV0');
    primitive.attributes.TEXCOORD_0 = uv0;
    delete primitive.attributes.TANGENT;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_MISSING_TANGENT');
    delete primitive.attributes.NORMAL;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_MISSING_NORMAL');
  });

  it('checks source indices and allocation budgets before import', () => {
    const { json, buffers, indexAccessor } = fixture();
    expect(() => extractGltfStaticPrimitive(json, buffers, -1, 0)).toThrow('GLTF_BUILD_INDEX');
    const view = json.bufferViews[json.accessors[indexAccessor].bufferView];
    new DataView(buffers[0]).setUint16(view.byteOffset + 2, 3, true);
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_INDICES_OR_BUDGET');
    new DataView(buffers[0]).setUint16(view.byteOffset + 2, 1, true);
    json.accessors[json.meshes[0].primitives[0].attributes.POSITION].count = 4_000_001;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_VERTEX_BUDGET');
    json.accessors[json.meshes[0].primitives[0].attributes.POSITION].count = 3;
    json.accessors[indexAccessor].count = 12_000_001;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_INDEX_BUDGET');
    json.accessors[indexAccessor].count = 3;
    json.meshes[0].primitives[0].attributes.TEXCOORD_1 = 999;
    expect(() => extractGltfStaticPrimitive(json, buffers, 0, 0)).toThrow('GLTF_BUILD_INDEX');
  });
});
