#!/usr/bin/env node
import { open, readFile } from 'node:fs/promises';
import { extname, resolve, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';

const MiB = 1024 * 1024;
const MAX_JSON_BYTES = 128 * MiB;
const supportedAttributes = new Set(['POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'COLOR_0', 'JOINTS_0', 'WEIGHTS_0']);
const supportedSlots = ['baseColorTexture', 'metallicRoughnessTexture', 'emissiveTexture', 'normalTexture'];
const issue = (code, path, detail) => ({ code, path, detail });
const items = value => Array.isArray(value) ? value : [];
const record = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const integer = value => Number.isSafeInteger(value) && value >= 0 ? value : null;

async function readJson(path) {
  const file = await open(path, 'r');
  try {
    const size = (await file.stat()).size;
    const head = Buffer.alloc(20);
    const readExact = async (buffer, position) => {
      for (let offset = 0; offset < buffer.length;) {
        const { bytesRead } = await file.read(buffer, offset, buffer.length - offset, position + offset);
        if (bytesRead === 0) throw new Error('GLTF_SHORT_READ');
        offset += bytesRead;
      }
    };
    await readExact(head.subarray(0, Math.min(20, size)), 0);
    if (head.readUInt32LE(0) === 0x46546c67) {
      if (size < 20 || head.readUInt32LE(4) !== 2 || head.readUInt32LE(8) !== size || head.readUInt32LE(16) !== 0x4e4f534a) throw new Error('INVALID_GLB_HEADER');
      const length = head.readUInt32LE(12);
      if (!length || length > MAX_JSON_BYTES || length % 4 || 20 + length > size) throw new Error('INVALID_GLB_JSON_LENGTH');
      const data = Buffer.alloc(length);
      await readExact(data, 20);
      return { json: JSON.parse(data.toString('utf8').replace(/\u0000+$/g, '').trimEnd()), container: 'glb', containerBytes: size, jsonBytes: length };
    }
    if (extname(path).toLowerCase() !== '.gltf') throw new Error('EXPECTED_GLTF_OR_GLB');
    if (size > MAX_JSON_BYTES) throw new Error('GLTF_JSON_AUDIT_BUDGET');
    const data = await readFile(path);
    if (data.length !== size) throw new Error('GLTF_SHORT_READ');
    return { json: JSON.parse(data.toString('utf8')), container: 'gltf', containerBytes: size, jsonBytes: size };
  } finally { await file.close(); }
}

function determinant3(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 16) return null;
  const m = matrix;
  return m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
}

function imageDimensions(bytes) {
  if (bytes.length >= 24 && bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) return { format: 'png', width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    for (let offset = 2; offset + 9 < bytes.length;) {
      if (bytes[offset++] !== 0xff) break;
      const marker = bytes[offset++];
      if (marker === 0xd9 || marker === 0xda) break;
      const length = bytes.readUInt16BE(offset);
      if (length < 2 || offset + length > bytes.length) break;
      if ([0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf].includes(marker)) return { format: 'jpeg', width: bytes.readUInt16BE(offset + 5), height: bytes.readUInt16BE(offset + 3) };
      offset += length;
    }
    return { format: 'jpeg', width: null, height: null };
  }
  return { format: 'unknown', width: null, height: null };
}

async function imageInfo(image, base) {
  const uri = typeof image.uri === 'string' ? image.uri : null;
  const result = { uri, mimeType: image.mimeType ?? null, uriFormat: uri?.startsWith('data:') ? null : uri ? extname(uri).toLowerCase().slice(1) || null : null, bufferView: image.bufferView ?? null, width: null, height: null, fileBytes: null, detectedFormat: null };
  if (uri && !uri.startsWith('data:')) {
    const target = resolve(dirname(base), decodeURIComponent(uri));
    if (!target.startsWith(dirname(base) + '\\') && !target.startsWith(dirname(base) + '/')) return { ...result, note: 'path escapes asset directory' };
    try {
      const file = await open(target, 'r');
      try {
        result.fileBytes = (await file.stat()).size;
        const data = Buffer.alloc(Math.min(result.fileBytes, 1024 * 1024));
        await file.read(data, 0, data.length, 0);
        const dimensions = imageDimensions(data);
        result.width = dimensions.width; result.height = dimensions.height; result.detectedFormat = dimensions.format;
      } finally { await file.close(); }
    } catch (error) { result.note = `unreadable: ${error.code ?? error.message}`; }
  } else if (uri?.startsWith('data:')) result.note = 'embedded data URI not decoded by audit';
  else result.note = 'bufferView image dimensions not decoded by audit';
  return result;
}

/** Metadata-only audit. Accessor declarations are checked; vertex values are never decoded or validated. */
export async function auditGltf(input) {
  const path = resolve(input), { json, container, containerBytes, jsonBytes } = await readJson(path);
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw new Error('GLTF_ROOT_OBJECT');
  const accessors = items(json.accessors), meshes = items(json.meshes), nodes = items(json.nodes), materials = items(json.materials);
  const scenes = items(json.scenes), textures = items(json.textures), images = items(json.images);
  const blockers = [], warnings = [];
  const add = (code, where, detail) => blockers.push(issue(code, where, detail));
  if (json.asset?.version !== '2.0') add('GLTF_VERSION', 'asset.version', json.asset?.version ?? null);
  for (const name of items(json.extensionsRequired)) if (name !== 'KHR_materials_unlit') add('GLTF_REQUIRED_EXTENSION', 'extensionsRequired', name);
  for (const name of items(json.extensionsUsed)) if (name !== 'KHR_materials_unlit' && !items(json.extensionsRequired).includes(name)) add('OPTIONAL_EXTENSION_NOT_INTERPRETED', 'extensionsUsed', name);
  const primitiveReports = [], positionAccessors = new Set(), indexAccessors = new Set();
  let primitiveVertices = 0, primitiveIndices = 0, primitiveTriangles = 0, missingNormalsExtraVertices = 0;
  let color0Primitives = 0, color0DeclaredVertices = 0;
  for (const [mi, mesh] of meshes.entries()) for (const [pi, primitive] of items(mesh.primitives).entries()) {
    const where = `meshes[${mi}].primitives[${pi}]`, attrs = record(primitive.attributes);
    const pos = accessors[attrs.POSITION], idx = accessors[primitive.indices];
    const vertices = integer(pos?.count), indices = primitive.indices === undefined ? vertices : integer(idx?.count);
    const triangles = indices === null || indices % 3 ? null : indices / 3;
    if (vertices === null || indices === null) add('INVALID_ACCESSOR_COUNT', where, null);
    if ((primitive.mode ?? 4) !== 4) add('GLTF_UNSUPPORTED_PRIMITIVE', where, primitive.mode);
    if (primitive.extensions !== undefined) add('GLTF_UNSUPPORTED_PRIMITIVE_EXTENSION', where, Object.keys(record(primitive.extensions)));
    if (pos?.componentType !== 5126) add('GLTF_POSITION_FLOAT_REQUIRED', where, pos?.componentType ?? null);
    for (const key of Object.keys(attrs)) if (!supportedAttributes.has(key)) add('GLTF_UNSUPPORTED_ATTRIBUTE', where, key);
    let color0 = null;
    if (attrs.COLOR_0 !== undefined) {
      color0Primitives++;
      const color = Number.isSafeInteger(attrs.COLOR_0) && attrs.COLOR_0 >= 0 ? accessors[attrs.COLOR_0] : undefined;
      const count = integer(color?.count), componentType = color?.componentType ?? null;
      color0DeclaredVertices += count ?? 0;
      color0 = { accessor: attrs.COLOR_0, count, type: color?.type ?? null, componentType,
        normalized: color?.normalized ?? false, valuesRead: false, valuesValidated: false };
      if (count !== vertices || !['VEC3', 'VEC4'].includes(color?.type)
        || (color.normalized !== undefined && typeof color.normalized !== 'boolean')
        || !(componentType === 5126 && color.normalized !== true
          || (componentType === 5121 || componentType === 5123) && color.normalized === true)) {
        add('GLTF_COLOR_LAYOUT', `${where}.attributes.COLOR_0`, color0);
      }
    }
    if (Number.isSafeInteger(attrs.POSITION)) positionAccessors.add(attrs.POSITION);
    if (Number.isSafeInteger(primitive.indices)) indexAccessors.add(primitive.indices);
    if (attrs.NORMAL === undefined && indices !== null) missingNormalsExtraVertices += Math.max(0, indices - (vertices ?? 0));
    if (vertices !== null) primitiveVertices += vertices;
    if (indices !== null) primitiveIndices += indices;
    if (triangles !== null) primitiveTriangles += triangles;
    primitiveReports.push({ mesh: mi, primitive: pi, material: primitive.material ?? null, vertices, indices, triangles, attributes: Object.keys(attrs), color0, mode: primitive.mode ?? 4 });
  }
  const uniqueAccessorVertices = [...positionAccessors].reduce((sum, i) => sum + (integer(accessors[i]?.count) ?? 0), 0);
  const uniqueAccessorIndices = [...indexAccessors].reduce((sum, i) => sum + (integer(accessors[i]?.count) ?? 0), 0);
  const textureUsages = [];
  for (const [mi, material] of materials.entries()) {
    const pbr = record(material.pbrMetallicRoughness);
    if (material.occlusionTexture !== undefined) add('GLTF_UNSUPPORTED_MATERIAL', `materials[${mi}]`, 'occlusionTexture');
    if (!['OPAQUE', 'MASK'].includes(material.alphaMode ?? 'OPAQUE')) add('GLTF_UNSUPPORTED_ALPHA_MODE', `materials[${mi}]`, material.alphaMode);
    for (const name of Object.keys(record(material.extensions))) if (name !== 'KHR_materials_unlit') add('GLTF_UNSUPPORTED_MATERIAL_EXTENSION', `materials[${mi}]`, name);
    for (const [slot, info] of [['baseColorTexture',pbr.baseColorTexture],['metallicRoughnessTexture',pbr.metallicRoughnessTexture],['emissiveTexture',material.emissiveTexture],['normalTexture',material.normalTexture],['occlusionTexture',material.occlusionTexture]]) {
      if (info === undefined) continue;
      textureUsages.push({ material: mi, slot, texture: info.index ?? null, texCoord: info.texCoord ?? 0, extensions: Object.keys(record(info.extensions)) });
      if (supportedSlots.includes(slot) && ((info.texCoord ?? 0) !== 0 || info.extensions !== undefined)) add('GLTF_UNSUPPORTED_TEXTURE_COORDINATES', `materials[${mi}].${slot}`, { texCoord: info.texCoord ?? 0, extensions: Object.keys(record(info.extensions)) });
    }
  }
  for (const [ti, texture] of textures.entries()) if (texture.extensions !== undefined) add('GLTF_UNSUPPORTED_TEXTURE_EXTENSION', `textures[${ti}]`, Object.keys(record(texture.extensions)));
  const imageReports = [];
  for (const image of images) imageReports.push(await imageInfo(record(image), path));
  for (const [i, image] of imageReports.entries()) if ((image.mimeType && !['image/png','image/jpeg'].includes(image.mimeType)) || (!image.mimeType && image.uriFormat && !['png','jpg','jpeg'].includes(image.uriFormat))) add('GLTF_IMAGE_FORMAT', `images[${i}]`, image.mimeType ?? image.uriFormat);
  const localSigns = nodes.map((node, i) => {
    const determinant = node.matrix !== undefined ? determinant3(node.matrix) : items(node.scale).length === 3 ? node.scale.reduce((a,b) => a*b, 1) : 1;
    if (determinant === null || !Number.isFinite(determinant)) { add('INVALID_NODE_TRANSFORM', `nodes[${i}]`, null); return 0; }
    return Math.sign(determinant);
  });
  const childSet = new Set(nodes.flatMap(node => items(node.children)));
  const defaultRoots = nodes.flatMap((_, i) => childSet.has(i) ? [] : [i]);
  if (nodes.length && !defaultRoots.length && !scenes.length) add('NODE_NO_ROOTS', 'nodes', 'implicit scene has no root; possible cycle');
  const selectedScenes = scenes.length ? scenes : [{ nodes: defaultRoots }];
  const sceneReports = selectedScenes.map((scene, si) => {
    let instances = 0, expandedVertices = 0, expandedIndices = 0, expandedTriangles = 0;
    const negativeNodes = [], repeatedNodes = [], visited = new Set();
    const stack = items(scene.nodes).map(id => ({ id, sign: 1 })).reverse();
    while (stack.length) {
      const { id, sign } = stack.pop();
      if (!Number.isSafeInteger(id) || id < 0 || id >= nodes.length) { add('INVALID_NODE_INDEX', `scenes[${si}]`, id); continue; }
      if (visited.has(id)) { repeatedNodes.push(id); continue; }
      visited.add(id);
      const node = nodes[id], worldSign = sign * localSigns[id];
      if (Number.isSafeInteger(node.mesh) && meshes[node.mesh]) {
        if (worldSign <= 0) negativeNodes.push(id);
        for (const primitive of items(meshes[node.mesh].primitives)) {
          const pos = accessors[record(primitive.attributes).POSITION], idx = accessors[primitive.indices];
          const v = integer(pos?.count) ?? 0, n = primitive.indices === undefined ? v : integer(idx?.count) ?? 0;
          instances++; expandedVertices += v; expandedIndices += n; if (n % 3 === 0) expandedTriangles += n / 3;
        }
      }
      for (const child of items(node.children)) stack.push({ id: child, sign: worldSign });
    }
    for (const id of negativeNodes) add('GLTF_UNSUPPORTED_REFLECTED_OR_SINGULAR_MESH_TRANSFORM', `nodes[${id}]`, `scene ${si}`);
    for (const id of new Set(repeatedNodes)) add('NODE_REPEATED_IN_SCENE', `nodes[${id}]`, `scene ${si}; multiple parents, duplicate root or cycle`);
    return { scene: si, name: scene.name ?? null, instances, expandedVertices, expandedIndices, expandedTriangles, negativeDeterminantMeshNodes: [...new Set(negativeNodes)], repeatedNodes: [...new Set(repeatedNodes)] };
  });
  if (nodes.length > 4096) add('GLTF_ARRAY_LIMIT', 'nodes', nodes.length);
  if (primitiveVertices + missingNormalsExtraVertices > 1_000_000) add('GLTF_VERTEX_BUDGET', 'meshes', primitiveVertices + missingNormalsExtraVertices);
  if (primitiveIndices > 3_000_000) add('GLTF_INDICES_OR_BUDGET', 'meshes', primitiveIndices);
  const declaredBufferBytes = items(json.buffers).reduce((sum, buffer) => sum + (integer(buffer.byteLength) ?? 0), 0);
  if (containerBytes + declaredBufferBytes + imageReports.reduce((sum, image) => sum + (image.fileBytes ?? 0), 0) > 128 * MiB) warnings.push(issue('DEFAULT_DOWNLOAD_BUDGET_LIKELY', 'resources', 'Uses file bytes as a conservative proxy; embedded and duplicate resources may be double counted.'));
  return { schemaVersion: 1, input: path, container, containerBytes, jsonBytes, counts: { meshes: meshes.length, primitives: primitiveReports.length, nodes: nodes.length, scenes: sceneReports.length, materials: materials.length, textures: textures.length, images: images.length, uniquePositionAccessors: positionAccessors.size, uniqueAccessorVertices, uniqueIndexAccessors: indexAccessors.size, uniqueAccessorIndices, primitiveVertices, primitiveIndices, primitiveTriangles, color0Primitives, color0DeclaredVertices, declaredBufferBytes }, scenes: sceneReports, primitives: primitiveReports, materials: materials.map((m, i) => ({ index: i, name: m.name ?? null, alphaMode: m.alphaMode ?? 'OPAQUE', alphaCutoff: m.alphaCutoff ?? 0.5, doubleSided: m.doubleSided === true, extensions: Object.keys(record(m.extensions)) })), textureUsages, images: imageReports, extensionsUsed: items(json.extensionsUsed), extensionsRequired: items(json.extensionsRequired), blockers, warnings, limitations: ['Accessor values, including COLOR_0 channel ranges and finiteness, were not read or validated.'], assessment: blockers.length ? 'blocked' : 'metadata-compatible-unverified' };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    if (process.argv.length !== 3) throw new Error('usage: node scripts/audit-gltf.mjs path.gltf|path.glb');
    process.stdout.write(JSON.stringify(await auditGltf(process.argv[2]), null, 2) + '\n');
  } catch (error) { process.stderr.write(`audit-gltf: ${error.message}\n`); process.exitCode = 1; }
}
