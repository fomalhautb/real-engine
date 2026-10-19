import type { MeshData } from './geometry';
import { readAccessor, type GltfBinaryDocument } from './gltf-binary';
import { importGltf } from './gltf-import';

type Json = Record<string, unknown>;
const MAX_VERTICES = 4_000_000;
const MAX_INDICES = 12_000_000;
const NON_UV_EXTENSIONS = new Set(['KHR_lights_punctual', 'KHR_materials_unlit']);
const CORE_TEXTURE_INFOS = ['baseColorTexture', 'metallicRoughnessTexture'] as const;
const MATERIAL_FIELDS = new Set(['name', 'extras', 'extensions', 'pbrMetallicRoughness',
  'normalTexture', 'occlusionTexture', 'emissiveTexture', 'emissiveFactor',
  'alphaMode', 'alphaCutoff', 'doubleSided']);
const PBR_FIELDS = new Set(['baseColorFactor', 'baseColorTexture', 'metallicFactor',
  'roughnessFactor', 'metallicRoughnessTexture']);
const TEXTURE_INFO_FIELDS = new Set(['index', 'texCoord', 'scale', 'strength', 'extras', 'extensions']);
const TEXTURE_FIELDS = new Set(['source', 'sampler', 'name', 'extras', 'extensions']);

function object(value: unknown, label: string): Json {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('GLTF_BUILD_OBJECT: ' + label);
  return value as Json;
}
function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value) || value.length > 65_536) throw new Error('GLTF_BUILD_ARRAY: ' + label);
  return value;
}
function index(value: unknown, length: number, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0 || (value as number) >= length) {
    throw new Error('GLTF_BUILD_INDEX: ' + label);
  }
  return value as number;
}

/** Extracts only immutable geometry. The caller retains the original glTF
 * document for scene/material reconstruction; this function never converts a
 * material or treats an unsupported rendering mode as Opaque. */
export function extractGltfStaticPrimitive(json: unknown, buffers: readonly ArrayBuffer[],
  meshIndex: number, primitiveIndex: number): {
    mesh: MeshData; materialIndex: number | null; coverage: 'opaque' | 'mask' | 'blend';
    ignoredAttributes: string[]; warnings: string[];
  } {
  const doc = object(json, 'root');
  const asset = object(doc.asset, 'asset');
  if (asset.version !== '2.0' || (asset.minVersion !== undefined && asset.minVersion !== '2.0')) {
    throw new Error('GLTF_BUILD_VERSION');
  }
  const required = array(doc.extensionsRequired ?? [], 'extensionsRequired');
  for (const name of required) if (typeof name !== 'string' || !NON_UV_EXTENSIONS.has(name)) {
    throw new Error('GLTF_BUILD_REQUIRED_EXTENSION: ' + String(name));
  }
  const meshes = array(doc.meshes, 'meshes');
  const sourceMesh = object(meshes[index(meshIndex, meshes.length, 'meshIndex')], 'mesh');
  const primitives = array(sourceMesh.primitives, 'primitives');
  const primitive = object(primitives[index(primitiveIndex, primitives.length, 'primitiveIndex')], 'primitive');
  if ((primitive.mode ?? 4) !== 4) throw new Error('GLTF_BUILD_TRIANGLES_REQUIRED');
  if (primitive.extensions !== undefined) throw new Error('GLTF_BUILD_PRIMITIVE_EXTENSION');
  if (array(primitive.targets ?? [], 'targets').length ||
    (sourceMesh.weights !== undefined && array(sourceMesh.weights, 'weights').length)) {
    throw new Error('GLTF_BUILD_DYNAMIC_GEOMETRY: morph');
  }
  const nodes = array(doc.nodes ?? [], 'nodes');
  for (const value of nodes) {
    const node = object(value, 'node');
    if (node.mesh === meshIndex && (node.skin !== undefined || node.weights !== undefined)) {
      throw new Error('GLTF_BUILD_DYNAMIC_GEOMETRY: node skin/weights');
    }
  }
  const attributes = object(primitive.attributes, 'attributes');
  for (const name of Object.keys(attributes)) {
    if (/^(JOINTS|WEIGHTS)_\d+$/.test(name)) throw new Error('GLTF_BUILD_DYNAMIC_GEOMETRY: ' + name);
    if (!['POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'COLOR_0'].includes(name)
      && !/^TEXCOORD_[1-9]\d*$/.test(name)) throw new Error('GLTF_BUILD_ATTRIBUTE: ' + name);
  }
  if (attributes.NORMAL === undefined) throw new Error('GLTF_BUILD_MISSING_NORMAL');
  const accessors = array(doc.accessors, 'accessors');
  const position = object(accessors[index(attributes.POSITION, accessors.length, 'POSITION')], 'POSITION');
  if (!Number.isSafeInteger(position.count) || (position.count as number) < 1
    || (position.count as number) > MAX_VERTICES) throw new Error('GLTF_BUILD_VERTEX_BUDGET');
  const indexCount = primitive.indices === undefined ? position.count as number
    : object(accessors[index(primitive.indices, accessors.length, 'indices')], 'indices').count;
  if (!Number.isSafeInteger(indexCount) || (indexCount as number) < 3
    || (indexCount as number) > MAX_INDICES) throw new Error('GLTF_BUILD_INDEX_BUDGET');

  const materials = array(doc.materials ?? [], 'materials');
  const materialIndex = primitive.material === undefined ? null
    : index(primitive.material, materials.length, 'material');
  const material = materialIndex === null ? {} : object(materials[materialIndex], 'material');
  const alpha = material.alphaMode ?? 'OPAQUE';
  if (alpha !== 'OPAQUE' && alpha !== 'MASK' && alpha !== 'BLEND') {
    throw new Error('GLTF_BUILD_ALPHA_MODE: ' + String(alpha));
  }
  const coverage = alpha.toLowerCase() as 'opaque' | 'mask' | 'blend';
  const ignoredAttributes = Object.keys(attributes).filter(name => /^TEXCOORD_[1-9]\d*$/.test(name));
  if (ignoredAttributes.length > 8) throw new Error('GLTF_BUILD_UV_SET_BUDGET');
  for (const name of ignoredAttributes) {
    const accessorIndex = index(attributes[name], accessors.length, name);
    const accessor = object(accessors[accessorIndex], name);
    if (accessor.count !== position.count || accessor.type !== 'VEC2') {
      throw new Error('GLTF_BUILD_UV_LAYOUT: ' + name);
    }
    readAccessor(doc as GltfBinaryDocument, buffers, accessorIndex);
  }
  if (ignoredAttributes.length) {
    // UV sets beyond zero are omitted only after proving that no known core
    // textureInfo selects them and no extension can reinterpret UV binding.
    for (const name of array(doc.extensionsUsed ?? [], 'extensionsUsed')) {
      if (typeof name !== 'string' || !NON_UV_EXTENSIONS.has(name)) throw new Error('GLTF_BUILD_UV_EXTENSION: ' + String(name));
    }
    for (const name of Object.keys(object(doc.extensions ?? {}, 'extensions'))) {
      if (!NON_UV_EXTENSIONS.has(name)) throw new Error('GLTF_BUILD_UV_EXTENSION: ' + name);
    }
    const extensions = object(material.extensions ?? {}, 'material.extensions');
    if (Object.keys(material).some(name => !MATERIAL_FIELDS.has(name))) {
      throw new Error('GLTF_BUILD_UV_UNKNOWN_MATERIAL_FIELD');
    }
    for (const name of Object.keys(extensions)) {
      if (!NON_UV_EXTENSIONS.has(name)) throw new Error('GLTF_BUILD_UV_MATERIAL_EXTENSION: ' + name);
    }
    const pbr = object(material.pbrMetallicRoughness ?? {}, 'pbrMetallicRoughness');
    if (Object.keys(pbr).some(name => !PBR_FIELDS.has(name))) {
      throw new Error('GLTF_BUILD_UV_UNKNOWN_MATERIAL_FIELD');
    }
    const infos = [...CORE_TEXTURE_INFOS.map(name => pbr[name]), material.normalTexture,
      material.occlusionTexture, material.emissiveTexture];
    const textures = array(doc.textures ?? [], 'textures');
    for (const value of infos) {
      if (value === undefined) continue;
      const info = object(value, 'textureInfo');
      if (Object.keys(info).some(name => !TEXTURE_INFO_FIELDS.has(name))) {
        throw new Error('GLTF_BUILD_UV_UNKNOWN_TEXTURE_INFO');
      }
      if (info.extensions !== undefined || (info.texCoord ?? 0) !== 0) {
        throw new Error('GLTF_BUILD_UV_IN_USE');
      }
      const texture = object(textures[index(info.index, textures.length, 'textureInfo.index')], 'texture');
      if (Object.keys(texture).some(name => !TEXTURE_FIELDS.has(name))) {
        throw new Error('GLTF_BUILD_UV_UNKNOWN_TEXTURE_FIELD');
      }
      if (texture.extensions !== undefined) throw new Error('GLTF_BUILD_UV_TEXTURE_EXTENSION');
    }
  }
  if (material.normalTexture !== undefined && attributes.TANGENT === undefined) {
    throw new Error('GLTF_BUILD_MISSING_TANGENT');
  }
  if (material.normalTexture !== undefined && attributes.TEXCOORD_0 === undefined) {
    throw new Error('GLTF_BUILD_MISSING_UV0');
  }
  // Retain the original accessors/views/buffer descriptors, but expose only
  // the selected primitive and its geometry attributes to the existing binary
  // decoder. No material, texture, image, scene, or extension is synthesized.
  const filtered = Object.fromEntries(Object.entries(attributes).filter(([name]) =>
    !ignoredAttributes.includes(name)));
  const isolated = { asset: { version: '2.0' }, buffers: doc.buffers,
    bufferViews: doc.bufferViews, accessors: doc.accessors,
    meshes: [{ primitives: [{ attributes: filtered, ...(primitive.indices === undefined ? {} : { indices: primitive.indices }), mode: 4 }] }] };
  const imported = importGltf(isolated, buffers, { maxVertices: MAX_VERTICES, maxIndices: MAX_INDICES });
  return { mesh: imported.meshes[0].primitives[0].mesh, materialIndex, coverage,
    ignoredAttributes, warnings: imported.warnings };
}
