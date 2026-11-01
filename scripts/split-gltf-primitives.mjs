#!/usr/bin/env node
// Deterministic offline split of oversized glTF triangle primitives.
//   node scripts/split-gltf-primitives.mjs --input scene.gltf --output new-directory
//     --max-triangles N --max-vertices N [--primitives mesh:primitive[,...]] [--license file]
// Without --primitives every primitive above either limit is split; every other
// primitive keeps its accessor contents byte for byte. A split is a recursive
// median split of the triangle centroids (float64, from the float32 positions)
// along the longest axis of the current set's centroid bounds (ties: lower
// axis; equal keys: lower source triangle index), repeated until each part has
// at most N triangles and references at most N vertices. A part keeps source
// triangle and corner order and references only its own vertices, in ascending
// source vertex order. Vertices used on both sides of a cut are duplicated with
// identical bytes, so every new open border is exactly the shared seam (the
// geometry builder locks open borders, so no crack can open there). The parts
// replace the source primitive in place inside the same mesh: nodes, node
// transforms, materials, textures, samplers, images and extensions are kept.
// Output: <stem>.split.gltf, <stem>.split.bin, the images hard-linked (copied
// across devices) at their original relative URIs, an optional license copy,
// uint32 little-endian provenance maps per split part and manifest.json, which
// is written last as the completion marker. Every written byte is a pure
// function of the input bytes and options; no timestamps or paths are stored.
import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { copyFile, link, mkdir, mkdtemp, open, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const SCRIPT = fileURLToPath(import.meta.url);
const MiB = 1024 * 1024;
const LIMITS = Object.freeze({
  jsonBytes: 64 * MiB, bufferBytes: 1024 * MiB, inputBytes: 2048 * MiB, outputBinBytes: 1024 * MiB,
  imageBytes: 1024 * MiB, imagesBytes: 8192 * MiB, licenseBytes: 4 * MiB,
  buffers: 256, bufferViews: 65_536, accessors: 65_536, meshes: 65_536, images: 4096,
  sourcePrimitives: 16_384, outputPrimitives: 16_384, partsPerPrimitive: 4096,
  sourcePrimitiveTriangles: 67_108_864, sourcePrimitiveVertices: 67_108_864,
  maxTrianglesOption: 16_777_216, maxVerticesOption: 16_777_216,
});
const USAGE = 'usage: split-gltf-primitives --input scene.gltf --output new-directory --max-triangles N --max-vertices N [--primitives mesh:primitive[,...]] [--license file]';
const COMPONENT_BYTES = Object.freeze({ 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 });
const TYPE_COMPONENTS = Object.freeze({ SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 });
const INDEX_RESTART = Object.freeze({ 5121: 0xff, 5123: 0xffff, 5125: 0xffff_ffff });
// Extensions that never reference buffers, buffer views, accessors or files
// other than images. Any other extension could be invalidated by the rewritten
// binary layout, so it is rejected instead of being copied unchecked.
const PRESERVED_EXTENSIONS = /^(KHR_materials_[a-z_]+|KHR_texture_transform|KHR_lights_punctual|KHR_mesh_quantization|KHR_texture_basisu|EXT_texture_webp|EXT_texture_avif)$/;
const ALGORITHM = 'recursive median split of float64 triangle centroids along the longest axis of the current centroid bounds '
  + '(ties: lower axis; equal keys: lower source triangle index; left half = floor(n/2) triangles) until every part fits '
  + 'maxTriangles and maxVertices; parts keep source triangle and corner order and reference only their own vertices in '
  + 'ascending source vertex order; vertices shared across a cut are duplicated byte-exactly';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonBytes = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
const align4 = value => Math.ceil(value / 4) * 4;
function fail(code, detail) { throw new Error(detail === undefined ? code : `${code}: ${detail}`); }
function object(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('SPLIT_OBJECT', label);
  return value;
}
function list(value, label, max) {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > max) fail('SPLIT_ARRAY', label);
  return value;
}
function index(value, length, label) {
  if (!Number.isSafeInteger(value) || value < 0 || value >= length) fail('SPLIT_INDEX', label);
  return value;
}
function within(parent, target) {
  const path = relative(parent, target);
  return path !== '' && !isAbsolute(path) && path !== '..' && !path.startsWith('..' + sep);
}
function uint32Bytes(values) {
  const bytes = Buffer.alloc(values.length * 4);
  for (let i = 0; i < values.length; i++) bytes.writeUInt32LE(values[i], i * 4);
  return bytes;
}
const byPath = (a, b) => (a.file < b.file ? -1 : a.file > b.file ? 1 : 0);

function parseInteger(text, min, max, code) {
  if (!/^[1-9]\d*$/.test(text)) fail(code, text);
  const value = Number(text);
  if (!Number.isSafeInteger(value) || value < min || value > max) fail(code, `${text} (allowed ${min}..${max})`);
  return value;
}

function argumentsOf(argv) {
  const keys = ['--input', '--output', '--max-triangles', '--max-vertices', '--primitives', '--license'];
  const values = new Map();
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i], value = argv[i + 1];
    if (!keys.includes(key) || values.has(key) || value === undefined || value.startsWith('--')) fail('SPLIT_ARGUMENTS', `${key}; ${USAGE}`);
    values.set(key, value); i++;
  }
  for (const key of ['--input', '--output', '--max-triangles', '--max-vertices']) if (!values.has(key)) fail('SPLIT_ARGUMENTS', USAGE);
  const input = resolve(values.get('--input')), output = resolve(values.get('--output'));
  if (extname(input).toLowerCase() !== '.gltf') fail('SPLIT_INPUT_FORMAT', 'only a .gltf with external buffer files is supported');
  if (input === output) fail('SPLIT_ARGUMENTS', 'input and output are the same path');
  const maxTriangles = parseInteger(values.get('--max-triangles'), 1, LIMITS.maxTrianglesOption, 'SPLIT_MAX_TRIANGLES');
  // A single triangle references up to three vertices, so three always terminates.
  const maxVertices = parseInteger(values.get('--max-vertices'), 3, LIMITS.maxVerticesOption, 'SPLIT_MAX_VERTICES');
  let primitives = null;
  if (values.has('--primitives')) {
    const text = values.get('--primitives');
    if (!/^\d{1,9}:\d{1,9}(,\d{1,9}:\d{1,9})*$/.test(text)) fail('SPLIT_PRIMITIVE_SELECTION', text);
    primitives = text.split(',').map(pair => pair.split(':').map(Number));
    if (primitives.length > LIMITS.sourcePrimitives) fail('SPLIT_PRIMITIVE_SELECTION', 'too many entries');
    if (new Set(primitives.map(pair => pair.join(':'))).size !== primitives.length) fail('SPLIT_PRIMITIVE_SELECTION', 'duplicate entry in ' + text);
  }
  return { input, output, maxTriangles, maxVertices, primitives,
    license: values.has('--license') ? resolve(values.get('--license')) : null };
}

/** Reads a whole regular file once through its canonical path and rejects concurrent changes. */
async function readBounded(path, limit, code) {
  let canonical;
  try { canonical = await realpath(path); } catch (error) { fail(code + '_MISSING', `${basename(path)} (${error.code})`); }
  const handle = await open(canonical, 'r');
  try {
    const before = await handle.stat();
    if (!before.isFile() || before.size < 1 || before.size > limit) fail(code + '_BYTE_BUDGET', `${basename(path)} (${before.size} bytes, limit ${limit})`);
    const bytes = Buffer.alloc(before.size);
    for (let offset = 0; offset < bytes.length;) {
      const { bytesRead } = await handle.read(bytes, offset, Math.min(16 * MiB, bytes.length - offset), offset);
      if (!bytesRead) fail('SPLIT_SOURCE_CHANGED', basename(path));
      offset += bytesRead;
    }
    const extra = await handle.read(Buffer.alloc(1), 0, 1, bytes.length), after = await handle.stat();
    if (extra.bytesRead || after.size !== before.size || after.mtimeMs !== before.mtimeMs
      || await realpath(path) !== canonical) fail('SPLIT_SOURCE_CHANGED', basename(path));
    return { bytes, canonical, dev: before.dev, ino: before.ino };
  } finally { await handle.close(); }
}

/** A relative local file URI below `base` (mirrors the geometry builder's rules). */
function localReference(base, uri, code) {
  if (typeof uri !== 'string' || !uri.length || /^[a-z][a-z0-9+.-]*:/i.test(uri)) fail(code, String(uri));
  let decoded;
  try { decoded = decodeURIComponent(uri); } catch { fail(code, uri); }
  if (!decoded || /[\\\u0000?#:]/.test(decoded) || isAbsolute(decoded)) fail(code, uri);
  const path = resolve(base, decoded);
  if (!within(base, path)) fail(code, uri + ' leaves the source directory');
  return { path, file: relative(base, path).split(sep).join('/') };
}

/** Rejects every document feature whose meaning the rewritten layout could change. */
function validateDocument(doc) {
  const asset = object(doc.asset, 'asset');
  if (asset.version !== '2.0' || (asset.minVersion !== undefined && asset.minVersion !== '2.0')) fail('SPLIT_VERSION');
  for (const key of ['extensionsUsed', 'extensionsRequired']) for (const name of list(doc[key], key, 256)) {
    if (typeof name !== 'string' || !PRESERVED_EXTENSIONS.test(name)) fail('SPLIT_EXTENSION_UNSUPPORTED', String(name));
  }
  if (list(doc.animations, 'animations', LIMITS.accessors).length) fail('SPLIT_ANIMATIONS_UNSUPPORTED');
  if (list(doc.skins, 'skins', LIMITS.accessors).length) fail('SPLIT_SKINS_UNSUPPORTED');
  for (const [i, node] of list(doc.nodes, 'nodes', LIMITS.meshes).entries()) {
    if (object(node, `nodes[${i}]`).weights !== undefined) fail('SPLIT_MORPH_TARGETS_UNSUPPORTED', `nodes[${i}].weights`);
  }
  const meshes = list(doc.meshes, 'meshes', LIMITS.meshes);
  if (!meshes.length) fail('SPLIT_NO_MESHES');
  let primitives = 0;
  for (const [m, mesh] of meshes.entries()) {
    if (object(mesh, `meshes[${m}]`).weights !== undefined) fail('SPLIT_MORPH_TARGETS_UNSUPPORTED', `meshes[${m}].weights`);
    const entries = list(mesh.primitives, `meshes[${m}].primitives`, LIMITS.sourcePrimitives);
    if (!entries.length) fail('SPLIT_ARRAY', `meshes[${m}].primitives is empty`);
    for (const [p, primitive] of entries.entries()) {
      object(primitive, `meshes[${m}].primitives[${p}]`);
      object(primitive.attributes, `meshes[${m}].primitives[${p}].attributes`);
      if (primitive.targets !== undefined) fail('SPLIT_MORPH_TARGETS_UNSUPPORTED', `meshes[${m}].primitives[${p}].targets`);
    }
    primitives += entries.length;
  }
  if (primitives > LIMITS.sourcePrimitives) fail('SPLIT_PRIMITIVE_BUDGET', primitives);
  for (const [i, accessor] of list(doc.accessors, 'accessors', LIMITS.accessors).entries()) {
    if (object(accessor, `accessors[${i}]`).sparse !== undefined) fail('SPLIT_SPARSE_ACCESSOR_UNSUPPORTED', `accessors[${i}]`);
    if (accessor.extensions !== undefined) fail('SPLIT_LAYOUT_EXTENSION_UNSUPPORTED', `accessors[${i}]`);
  }
  for (const [i, view] of list(doc.bufferViews, 'bufferViews', LIMITS.bufferViews).entries()) {
    if (object(view, `bufferViews[${i}]`).extensions !== undefined) fail('SPLIT_LAYOUT_EXTENSION_UNSUPPORTED', `bufferViews[${i}]`);
  }
  for (const [i, image] of list(doc.images, 'images', LIMITS.images).entries()) {
    if (object(image, `images[${i}]`).bufferView !== undefined) fail('SPLIT_EMBEDDED_IMAGE_UNSUPPORTED', `images[${i}]`);
    if (typeof image.uri !== 'string') fail('SPLIT_IMAGE_URI', `images[${i}]`);
  }
  if (!list(doc.buffers, 'buffers', LIMITS.buffers).length) fail('SPLIT_BUFFER_URI', 'no buffers');
}

async function loadSource(input) {
  const json = await readBounded(input, LIMITS.jsonBytes, 'SPLIT_JSON');
  let doc;
  try { doc = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(json.bytes)); }
  catch (error) { fail('SPLIT_JSON_PARSE', error.message); }
  object(doc, 'root');
  validateDocument(doc);
  const base = dirname(json.canonical), buffers = [];
  let total = 0;
  for (const [i, value] of doc.buffers.entries()) {
    const definition = object(value, `buffers[${i}]`);
    if (definition.extensions !== undefined) fail('SPLIT_LAYOUT_EXTENSION_UNSUPPORTED', `buffers[${i}]`);
    if (!Number.isSafeInteger(definition.byteLength) || definition.byteLength < 1 || definition.byteLength > LIMITS.bufferBytes) fail('SPLIT_BUFFER_LENGTH', `buffers[${i}]`);
    if (typeof definition.uri !== 'string' || definition.uri.startsWith('data:')) fail('SPLIT_BUFFER_URI', `buffers[${i}] needs an external file URI`);
    const reference = localReference(base, definition.uri, 'SPLIT_BUFFER_URI');
    const file = await readBounded(reference.path, LIMITS.bufferBytes, 'SPLIT_BUFFER');
    if (!within(base, file.canonical)) fail('SPLIT_BUFFER_OUTSIDE_SOURCE', definition.uri);
    if (file.bytes.length < definition.byteLength) fail('SPLIT_BUFFER_LENGTH', `buffers[${i}] file is shorter than byteLength`);
    total += file.bytes.length;
    if (total > LIMITS.inputBytes) fail('SPLIT_INPUT_BYTE_BUDGET', total);
    buffers.push({ index: i, uri: definition.uri, file: reference.file, data: file.bytes, byteLength: definition.byteLength,
      bytes: file.bytes.length, sha256: sha256(file.bytes) });
  }
  return { doc, json, base, buffers };
}

/** Validated element layout of one accessor; supports any stride, offset and component type. */
function layoutOf(doc, buffers, accessorIndex, label) {
  const accessors = doc.accessors ?? [], views = doc.bufferViews ?? [];
  const accessor = object(accessors[index(accessorIndex, accessors.length, label)], label);
  if (accessor.sparse !== undefined) fail('SPLIT_SPARSE_ACCESSOR_UNSUPPORTED', label);
  if (accessor.bufferView === undefined) fail('SPLIT_ACCESSOR_WITHOUT_BUFFER_VIEW', label);
  const componentBytes = COMPONENT_BYTES[accessor.componentType], components = TYPE_COMPONENTS[accessor.type];
  if (!componentBytes || !components) fail('SPLIT_ACCESSOR_TYPE', `${label} ${accessor.type}/${accessor.componentType}`);
  if (!Number.isSafeInteger(accessor.count) || accessor.count < 1) fail('SPLIT_ACCESSOR_COUNT', label);
  if (accessor.normalized !== undefined && (typeof accessor.normalized !== 'boolean'
    || (accessor.normalized && (accessor.componentType === 5125 || accessor.componentType === 5126)))) fail('SPLIT_ACCESSOR_NORMALIZED', label);
  const view = object(views[index(accessor.bufferView, views.length, label + '.bufferView')], label + '.bufferView');
  const buffer = buffers[index(view.buffer, buffers.length, label + '.buffer')];
  const viewOffset = view.byteOffset ?? 0, accessorOffset = accessor.byteOffset ?? 0, elementBytes = componentBytes * components;
  if (!Number.isSafeInteger(viewOffset) || viewOffset < 0 || !Number.isSafeInteger(accessorOffset) || accessorOffset < 0
    || !Number.isSafeInteger(view.byteLength) || view.byteLength < 1 || viewOffset + view.byteLength > buffer.byteLength) fail('SPLIT_ACCESSOR_LAYOUT', label + ' view range');
  if (view.byteStride !== undefined && (!Number.isSafeInteger(view.byteStride) || view.byteStride < 4 || view.byteStride > 252
    || view.byteStride % 4)) fail('SPLIT_ACCESSOR_LAYOUT', label + ' byteStride');
  const stride = view.byteStride ?? elementBytes;
  if (stride < elementBytes || stride % componentBytes || (viewOffset + accessorOffset) % componentBytes) fail('SPLIT_ACCESSOR_LAYOUT', label + ' alignment');
  if (accessorOffset + (accessor.count - 1) * stride + elementBytes > view.byteLength) fail('SPLIT_ACCESSOR_LAYOUT', label + ' element range');
  return { accessor, view, data: buffer.data, base: viewOffset + accessorOffset, stride, elementBytes, componentBytes,
    components, count: accessor.count, componentType: accessor.componentType };
}

function component(layout, element, componentIndex) {
  const offset = layout.base + element * layout.stride + componentIndex * layout.componentBytes, data = layout.data;
  switch (layout.componentType) {
    case 5120: return data.readInt8(offset);
    case 5121: return data.readUInt8(offset);
    case 5122: return data.readInt16LE(offset);
    case 5123: return data.readUInt16LE(offset);
    case 5125: return data.readUInt32LE(offset);
    default: return data.readFloatLE(offset);
  }
}

/** Accessor min/max in stored (not normalized) component values over the given elements. */
function minMax(layout, elements, count, label) {
  const min = Array(layout.components).fill(Infinity), max = Array(layout.components).fill(-Infinity);
  for (let j = 0; j < count; j++) {
    const element = elements ? elements[j] : j;
    for (let c = 0; c < layout.components; c++) {
      const value = component(layout, element, c);
      if (!Number.isFinite(value)) fail('SPLIT_NONFINITE_ATTRIBUTE', label);
      if (value < min[c]) min[c] = value;
      if (value > max[c]) max[c] = value;
    }
  }
  return { min: min.map(value => value + 0), max: max.map(value => value + 0) };
}

function readIndices(layout, vertexCount, label) {
  if (layout.accessor.type !== 'SCALAR' || ![5121, 5123, 5125].includes(layout.componentType) || layout.accessor.normalized) fail('SPLIT_INDEX_TYPE', label);
  if (layout.view.byteStride !== undefined) fail('SPLIT_INDEX_BYTE_STRIDE', label);
  if (layout.count % 3) fail('SPLIT_INDEX_COUNT', label);
  const values = new Uint32Array(layout.count), restart = INDEX_RESTART[layout.componentType];
  for (let i = 0; i < layout.count; i++) {
    const value = component(layout, i, 0);
    if (value >= vertexCount || value === restart) fail('SPLIT_INDEX_RANGE', `${label}[${i}] = ${value}`);
    values[i] = value;
  }
  return values;
}

function readPositions(layout, label) {
  if (layout.accessor.type !== 'VEC3' || layout.componentType !== 5126) fail('SPLIT_POSITION_FLOAT_REQUIRED', label);
  const positions = new Float32Array(layout.count * 3);
  for (let vertex = 0; vertex < layout.count; vertex++) for (let axis = 0; axis < 3; axis++) {
    const value = layout.data.readFloatLE(layout.base + vertex * layout.stride + axis * 4);
    if (!Number.isFinite(value)) fail('SPLIT_NONFINITE_POSITION', `${label}[${vertex}]`);
    positions[vertex * 3 + axis] = value;
  }
  return positions;
}

/**
 * Recursive median split. `order` holds source triangle ids; every segment of
 * it stays in ascending id order, which is both the in-part triangle order and
 * the tie-break among equal centroid keys. Returns depth-first, left-first leaves.
 */
function partition(indices, positions, vertexCount, maxTriangles, maxVertices, label) {
  const triangles = indices.length / 3;
  const centroids = new Float64Array(triangles * 3);
  for (let t = 0; t < triangles; t++) {
    const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
    for (let axis = 0; axis < 3; axis++) {
      // +0 folds -0 into 0, so typed-array sorting and comparisons agree.
      centroids[t * 3 + axis] = (positions[a + axis] + positions[b + axis] + positions[c + axis]) / 3 + 0;
    }
  }
  const order = new Uint32Array(triangles), scratch = new Uint32Array(triangles), stamps = new Uint32Array(vertexCount);
  for (let t = 0; t < triangles; t++) order[t] = t;
  let stamp = 0, maxDepth = 0;
  const parts = [], stack = [{ start: 0, end: triangles, depth: 0 }];
  while (stack.length) {
    const { start, end, depth } = stack.pop(), count = end - start;
    stamp++;
    let vertices = 0;
    for (let i = start; i < end; i++) {
      const t = order[i];
      for (let corner = 0; corner < 3; corner++) {
        const vertex = indices[t * 3 + corner];
        if (stamps[vertex] !== stamp) { stamps[vertex] = stamp; vertices++; }
      }
    }
    if (count <= maxTriangles && vertices <= maxVertices) {
      if (parts.length >= LIMITS.partsPerPrimitive) fail('SPLIT_PART_BUDGET', `${label} needs more than ${LIMITS.partsPerPrimitive} parts`);
      parts.push({ start, end, depth });
      maxDepth = Math.max(maxDepth, depth);
      continue;
    }
    if (count < 2) fail('SPLIT_UNSATISFIABLE', label);
    const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) for (let axis = 0; axis < 3; axis++) {
      const value = centroids[order[i] * 3 + axis];
      if (value < low[axis]) low[axis] = value;
      if (value > high[axis]) high[axis] = value;
    }
    let axis = 0;
    for (let candidate = 1; candidate < 3; candidate++) if (high[candidate] - low[candidate] > high[axis] - low[axis]) axis = candidate;
    const keys = new Float64Array(count);
    for (let i = start; i < end; i++) keys[i - start] = centroids[order[i] * 3 + axis];
    const sorted = keys.slice().sort(), half = Math.floor(count / 2), pivot = sorted[half - 1];
    let less = 0;
    for (let lo = 0, hi = half - 1; lo <= hi;) { // first index whose key equals the pivot
      const middle = (lo + hi) >>> 1;
      if (sorted[middle] < pivot) { less = middle + 1; lo = middle + 1; } else hi = middle - 1;
    }
    let equal = half - less, left = start, right = start + half;
    for (let i = start; i < end; i++) {
      const key = keys[i - start];
      if (key < pivot || (key === pivot && equal > 0)) { if (key === pivot) equal--; scratch[left++] = order[i]; }
      else scratch[right++] = order[i];
    }
    if (left !== start + half || right !== end) fail('SPLIT_INTERNAL_PARTITION', label);
    order.set(scratch.subarray(start, end), start);
    stack.push({ start: start + half, end, depth: depth + 1 }, { start, end: start + half, depth: depth + 1 });
  }
  return { order, parts, maxDepth };
}

/** Builds the part's ascending vertex list and its local (renumbered) index buffer. */
function partGeometry(order, part, indices, positions, stamps, local, stampValue) {
  const triangles = order.slice(part.start, part.end), referenced = [];
  for (const t of triangles) for (let corner = 0; corner < 3; corner++) {
    const vertex = indices[t * 3 + corner];
    if (stamps[vertex] !== stampValue) { stamps[vertex] = stampValue; referenced.push(vertex); }
  }
  const vertices = Uint32Array.from(referenced).sort();
  for (let j = 0; j < vertices.length; j++) local[vertices[j]] = j;
  const localIndices = new Uint32Array(triangles.length * 3);
  for (let i = 0; i < triangles.length; i++) for (let corner = 0; corner < 3; corner++) {
    localIndices[i * 3 + corner] = local[indices[triangles[i] * 3 + corner]];
  }
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const vertex of vertices) for (let axis = 0; axis < 3; axis++) {
    const value = positions[vertex * 3 + axis];
    if (value < min[axis]) min[axis] = value;
    if (value > max[axis]) max[axis] = value;
  }
  return { triangles, vertices, localIndices, bounds: { min: min.map(v => v + 0), max: max.map(v => v + 0) } };
}

/** Copies an accessor definition in source key order with new layout fields. */
function accessorJson(source, bufferView, count, range) {
  const result = {};
  for (const [key, value] of Object.entries(source)) {
    if (key === 'bufferView') result.bufferView = bufferView;
    else if (key === 'byteOffset') continue;
    else if (key === 'count') result.count = count;
    else if (key === 'min' || key === 'max') { if (range) result[key] = range[key]; else result[key] = structuredClone(value); }
    else result[key] = structuredClone(value);
  }
  if (range && result.min === undefined) { result.min = range.min; result.max = range.max; }
  return result;
}
function primitiveJson(source, attributes, indices) {
  const result = {};
  for (const [key, value] of Object.entries(source)) {
    if (key === 'attributes') result.attributes = attributes;
    else if (key === 'indices') result.indices = indices;
    else result[key] = structuredClone(value);
  }
  return result;
}

function writeIndex(out, offset, componentType, value) {
  if (componentType === 5121) out.writeUInt8(value, offset);
  else if (componentType === 5123) out.writeUInt16LE(value, offset);
  else out.writeUInt32LE(value, offset);
}

/** Plans and fills the complete output document and binary buffer in memory. */
function buildOutput(source, config, names) {
  const { doc, buffers } = source;
  const layouts = new Map();
  const layout = (accessorIndex, label) => {
    if (!layouts.has(accessorIndex)) layouts.set(accessorIndex, layoutOf(doc, buffers, accessorIndex, label));
    return layouts.get(accessorIndex);
  };
  const selected = new Set((config.primitives ?? []).map(pair => pair.join(':')));
  if (config.primitives) for (const [m, p] of config.primitives) {
    if (!doc.meshes[m] || !doc.meshes[m].primitives[p]) fail('SPLIT_PRIMITIVE_SELECTION', `${m}:${p} does not exist`);
  }
  // Pass 1: classify every source primitive and partition the selected ones.
  const sourceRecords = [];
  for (const [m, mesh] of doc.meshes.entries()) for (const [p, primitive] of mesh.primitives.entries()) {
    const label = `meshes[${m}].primitives[${p}]`, mode = primitive.mode ?? 4;
    const attributeNames = Object.keys(primitive.attributes);
    if (!attributeNames.length) fail('SPLIT_ATTRIBUTE_COUNT', label + ' has no attributes');
    const first = layout(primitive.attributes.POSITION ?? primitive.attributes[attributeNames[0]], `${label}.attributes`);
    const vertices = first.count;
    const indexCount = primitive.indices === undefined ? vertices : layout(primitive.indices, label + '.indices').count;
    const triangles = mode === 4 && indexCount % 3 === 0 ? indexCount / 3 : null;
    const exceeds = vertices > config.maxVertices || (triangles ?? 0) > config.maxTriangles;
    const split = config.primitives ? selected.has(`${m}:${p}`) : exceeds;
    const record = { meshIndex: m, primitiveIndex: p, mode, materialIndex: primitive.material ?? null, triangles, vertices, exceedsLimits: exceeds, split };
    if (split) {
      if (mode !== 4 || primitive.indices === undefined) fail('SPLIT_TRIANGLES_WITH_INDICES_REQUIRED', label);
      if (primitive.attributes.POSITION === undefined) fail('SPLIT_POSITION_FLOAT_REQUIRED', label + ' has no POSITION');
      if (vertices > LIMITS.sourcePrimitiveVertices) fail('SPLIT_VERTEX_BUDGET', label);
      if (indexCount / 3 > LIMITS.sourcePrimitiveTriangles) fail('SPLIT_TRIANGLE_BUDGET', label);
      for (const name of attributeNames) {
        const attribute = layout(primitive.attributes[name], `${label}.attributes.${name}`);
        if (attribute.count !== vertices) fail('SPLIT_ATTRIBUTE_COUNT', `${label}.attributes.${name}`);
      }
      const positions = readPositions(layout(primitive.attributes.POSITION, label + '.POSITION'), label + '.POSITION');
      const indices = readIndices(layout(primitive.indices, label + '.indices'), vertices, label + '.indices');
      const { order, parts, maxDepth } = partition(indices, positions, vertices, config.maxTriangles, config.maxVertices, label);
      const stamps = new Uint32Array(vertices), local = new Uint32Array(vertices);
      record.parts = parts.map((part, k) => partGeometry(order, part, indices, positions, stamps, local, k + 1));
      record.maxDepth = maxDepth;
      // Exact coverage: every source triangle in exactly one part.
      const hits = new Uint8Array(triangles);
      for (const part of record.parts) for (const t of part.triangles) if (hits[t]++) fail('SPLIT_COVERAGE', `${label} triangle ${t} repeated`);
      if (hits.some(hit => hit !== 1)) fail('SPLIT_COVERAGE', `${label} triangle missing`);
      stamps.fill(0);
      for (const part of record.parts) for (const vertex of part.vertices) stamps[vertex] = 1;
      record.referencedVertices = stamps.reduce((sum, value) => sum + value, 0);
      record.partVertices = record.parts.reduce((sum, part) => sum + part.vertices.length, 0);
    }
    sourceRecords.push(record);
  }
  // Pass 2: output accessors, one tightly planned buffer view each.
  const plan = [], copies = new Map();
  const add = entry => {
    if (plan.length >= LIMITS.accessors) fail('SPLIT_ACCESSOR_BUDGET');
    plan.push(entry);
    return plan.length - 1;
  };
  const copyAccessor = (accessorIndex, role, label) => {
    const key = `${role}:${accessorIndex}`;
    if (copies.has(key)) return copies.get(key);
    const source = layout(accessorIndex, label);
    if (role === 'index' && source.view.byteStride !== undefined) fail('SPLIT_INDEX_BYTE_STRIDE', label);
    const stride = role === 'vertex' ? align4(source.elementBytes) : source.elementBytes;
    const id = add({ role, json: accessorJson(source.accessor, plan.length, source.count, null), count: source.count, stride,
      byteStride: role === 'vertex' && stride !== source.elementBytes ? stride : undefined, target: source.view.target,
      fill: (out, base) => {
        for (let j = 0; j < source.count; j++) {
          const from = source.base + j * source.stride;
          source.data.copy(out, base + j * stride, from, from + source.elementBytes);
        }
      } });
    copies.set(key, id);
    return id;
  };
  const meshes = [], outputRecords = [];
  for (const [m, mesh] of doc.meshes.entries()) {
    const primitives = [];
    for (const [p, primitive] of mesh.primitives.entries()) {
      const record = sourceRecords.find(item => item.meshIndex === m && item.primitiveIndex === p);
      const label = `meshes[${m}].primitives[${p}]`;
      record.outputPrimitives = { first: primitives.length, count: record.split ? record.parts.length : 1 };
      if (!record.split) {
        const attributes = {};
        for (const [name, accessor] of Object.entries(primitive.attributes)) attributes[name] = copyAccessor(accessor, 'vertex', `${label}.attributes.${name}`);
        const indices = primitive.indices === undefined ? undefined : copyAccessor(primitive.indices, 'index', label + '.indices');
        const position = primitive.attributes.POSITION === undefined ? null : layout(primitive.attributes.POSITION, label + '.POSITION');
        const bounds = position && position.componentType === 5126 && position.accessor.type === 'VEC3'
          ? minMax(position, null, position.count, label + '.POSITION') : null;
        outputRecords.push({ meshIndex: m, primitiveIndex: primitives.length, source: { meshIndex: m, primitiveIndex: p }, split: false,
          part: 0, parts: 1, materialIndex: record.materialIndex, triangleCount: record.triangles, vertexCount: record.vertices, bounds, provenance: null });
        primitives.push(primitiveJson(primitive, attributes, indices));
        continue;
      }
      const indexLayout = layout(primitive.indices, label + '.indices');
      record.parts.forEach((part, k) => {
        const attributes = {};
        for (const [name, accessor] of Object.entries(primitive.attributes)) {
          const source = layout(accessor, `${label}.attributes.${name}`), stride = align4(source.elementBytes);
          const range = name === 'POSITION' ? part.bounds
            : source.accessor.min !== undefined || source.accessor.max !== undefined ? minMax(source, part.vertices, part.vertices.length, `${label}.attributes.${name}`) : null;
          attributes[name] = add({ role: 'vertex', json: accessorJson(source.accessor, plan.length, part.vertices.length, range),
            count: part.vertices.length, stride, byteStride: stride !== source.elementBytes ? stride : undefined, target: source.view.target,
            fill: (out, base) => {
              for (let j = 0; j < part.vertices.length; j++) {
                const from = source.base + part.vertices[j] * source.stride;
                source.data.copy(out, base + j * stride, from, from + source.elementBytes);
              }
            } });
        }
        let indexRange = null;
        if (indexLayout.accessor.min !== undefined || indexLayout.accessor.max !== undefined) {
          let low = Infinity, high = -Infinity;
          for (const value of part.localIndices) { if (value < low) low = value; if (value > high) high = value; }
          indexRange = { min: [low], max: [high] };
        }
        const indexBytes = indexLayout.componentBytes;
        const indices = add({ role: 'index', json: accessorJson(indexLayout.accessor, plan.length, part.localIndices.length, indexRange),
          count: part.localIndices.length, stride: indexBytes, byteStride: undefined, target: indexLayout.view.target,
          fill: (out, base) => {
            for (let i = 0; i < part.localIndices.length; i++) writeIndex(out, base + i * indexBytes, indexLayout.componentType, part.localIndices[i]);
          } });
        const outputPrimitive = primitives.length;
        const file = `provenance/mesh-${m}-primitive-${outputPrimitive}`;
        outputRecords.push({ meshIndex: m, primitiveIndex: outputPrimitive, source: { meshIndex: m, primitiveIndex: p }, split: true,
          part: k, parts: record.parts.length, materialIndex: record.materialIndex, triangleCount: part.triangles.length,
          vertexCount: part.vertices.length, bounds: part.bounds,
          provenance: { triangles: { file: `${file}-source-triangles.bin`, data: part.triangles },
            vertices: { file: `${file}-source-vertices.bin`, data: part.vertices } } });
        primitives.push(primitiveJson(primitive, attributes, indices));
      });
    }
    if (primitives.length > LIMITS.outputPrimitives) fail('SPLIT_PRIMITIVE_BUDGET', 'output');
    const result = {};
    for (const [key, value] of Object.entries(mesh)) result[key] = key === 'primitives' ? primitives : structuredClone(value);
    meshes.push(result);
  }
  if (outputRecords.length > LIMITS.outputPrimitives) fail('SPLIT_PRIMITIVE_BUDGET', `${outputRecords.length} output primitives`);
  // Pass 3: one buffer view per accessor at a 4-byte aligned offset.
  let offset = 0;
  const views = plan.map(entry => {
    offset = align4(offset);
    const view = { buffer: 0, byteOffset: offset, byteLength: entry.count * entry.stride };
    if (entry.byteStride !== undefined) view.byteStride = entry.byteStride;
    if (entry.target !== undefined) view.target = entry.target;
    entry.offset = offset;
    offset += view.byteLength;
    if (offset > LIMITS.outputBinBytes) fail('SPLIT_OUTPUT_BYTE_BUDGET', `binary buffer exceeds ${LIMITS.outputBinBytes} bytes`);
    return view;
  });
  const bin = Buffer.alloc(align4(offset));
  for (const entry of plan) entry.fill(bin, entry.offset);
  const output = {};
  for (const [key, value] of Object.entries(doc)) {
    if (key === 'accessors') output.accessors = plan.map(entry => entry.json);
    else if (key === 'bufferViews') output.bufferViews = views;
    else if (key === 'buffers') output.buffers = [{ uri: encodeURIComponent(names.bin), byteLength: bin.length }];
    else if (key === 'meshes') output.meshes = meshes;
    else output[key] = structuredClone(value);
  }
  if (output.bufferViews === undefined) output.bufferViews = views;
  return { output, bin, sourceRecords, outputRecords };
}

const sameBytes = (a, aStart, b, bStart, length) => a.compare(b, bStart, bStart + length, aStart, aStart + length) === 0;

/** Independent read-back of the produced document against the source bytes. */
function verifyOutput(source, built) {
  const { doc } = source, out = built.output;
  const outBuffers = [{ data: built.bin, byteLength: built.bin.length }];
  const strip = value => JSON.stringify({ ...value, accessors: null, bufferViews: null, buffers: null,
    meshes: (value.meshes ?? []).map(mesh => ({ ...mesh, primitives: null })) });
  if (strip(doc) !== strip(out)) fail('SPLIT_VERIFY_DOCUMENT');
  const checked = { vertices: 0, attributeElements: 0, triangles: 0 };
  for (const record of built.outputRecords) {
    const label = `output meshes[${record.meshIndex}].primitives[${record.primitiveIndex}]`;
    const produced = out.meshes[record.meshIndex].primitives[record.primitiveIndex];
    const original = doc.meshes[record.source.meshIndex].primitives[record.source.primitiveIndex];
    const without = value => JSON.stringify(Object.fromEntries(Object.entries(value).filter(([key]) => key !== 'attributes' && key !== 'indices')));
    if (without(produced) !== without(original)) fail('SPLIT_VERIFY_PRIMITIVE_FIELDS', label);
    const names = Object.keys(original.attributes);
    if (JSON.stringify(Object.keys(produced.attributes)) !== JSON.stringify(names)) fail('SPLIT_VERIFY_ATTRIBUTES', label);
    const sourceVertices = record.provenance?.vertices.data, sourceTriangles = record.provenance?.triangles.data;
    for (const name of names) {
      const o = layoutOf(out, outBuffers, produced.attributes[name], `${label}.${name}`);
      const s = layoutOf(doc, source.buffers, original.attributes[name], `${label}.${name}`);
      if (o.componentType !== s.componentType || o.accessor.type !== s.accessor.type
        || (o.accessor.normalized ?? false) !== (s.accessor.normalized ?? false) || o.base % 4 || o.stride % 4) fail('SPLIT_VERIFY_ACCESSOR', `${label}.${name}`);
      const count = sourceVertices ? sourceVertices.length : s.count;
      if (o.count !== count) fail('SPLIT_VERIFY_COUNT', `${label}.${name}`);
      for (let j = 0; j < count; j++) {
        const from = s.base + (sourceVertices ? sourceVertices[j] : j) * s.stride;
        if (!sameBytes(o.data, o.base + j * o.stride, s.data, from, s.elementBytes)) fail('SPLIT_VERIFY_ATTRIBUTE_BYTES', `${label}.${name}[${j}]`);
      }
      if (name === 'POSITION' && sourceVertices) {
        const range = minMax(o, null, o.count, label);
        if (JSON.stringify(range) !== JSON.stringify({ min: o.accessor.min, max: o.accessor.max })) fail('SPLIT_VERIFY_BOUNDS', label);
      }
      checked.attributeElements += count;
    }
    checked.vertices += sourceVertices ? sourceVertices.length : 0;
    if (original.indices === undefined) continue;
    const o = layoutOf(out, outBuffers, produced.indices, label + '.indices');
    const s = layoutOf(doc, source.buffers, original.indices, label + '.indices');
    if (o.componentType !== s.componentType) fail('SPLIT_VERIFY_ACCESSOR', label + '.indices');
    if (!sourceTriangles) {
      if (o.count !== s.count) fail('SPLIT_VERIFY_COUNT', label + '.indices');
      for (let i = 0; i < s.count; i++) if (!sameBytes(o.data, o.base + i * o.stride, s.data, s.base + i * s.stride, s.elementBytes)) fail('SPLIT_VERIFY_INDEX_BYTES', label);
      continue;
    }
    if (o.count !== sourceTriangles.length * 3) fail('SPLIT_VERIFY_COUNT', label + '.indices');
    for (let i = 0; i < sourceTriangles.length; i++) for (let corner = 0; corner < 3; corner++) {
      const localIndex = component(o, i * 3 + corner, 0);
      if (localIndex >= sourceVertices.length
        || sourceVertices[localIndex] !== component(s, sourceTriangles[i] * 3 + corner, 0)) fail('SPLIT_VERIFY_TRIANGLE', `${label} triangle ${i}`);
    }
    checked.triangles += sourceTriangles.length;
  }
  for (const record of built.sourceRecords.filter(item => item.split)) {
    const hits = new Uint8Array(record.triangles);
    for (const item of built.outputRecords) {
      if (item.split && item.source.meshIndex === record.meshIndex && item.source.primitiveIndex === record.primitiveIndex) {
        for (const t of item.provenance.triangles.data) hits[t]++;
      }
    }
    if (hits.some(hit => hit !== 1)) fail('SPLIT_VERIFY_COVERAGE', `meshes[${record.meshIndex}].primitives[${record.primitiveIndex}]`);
  }
  return checked;
}

async function linkOrCopy(from, to) {
  try { await link(from, to); return 'link'; }
  catch (error) {
    if (error.code !== 'EXDEV' && error.code !== 'EPERM') throw error;
    await copyFile(from, to, constants.COPYFILE_EXCL);
    return 'copy';
  }
}

async function main() {
  const started = performance.now();
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) fail('SPLIT_OUTPUT_EXISTS', config.output);
  const source = await loadSource(config.input);
  const stem = basename(source.json.canonical).replace(/\.gltf$/i, '');
  const names = { gltf: `${stem}.split.gltf`, bin: `${stem}.split.bin` };
  // Images stay in the document unchanged; their files are linked at the same relative path.
  const images = [], imageFiles = new Map();
  let imageBytes = 0;
  for (const [i, image] of (source.doc.images ?? []).entries()) {
    if (image.uri.startsWith('data:')) { images.push({ image: i, uri: null, file: null, inline: true }); continue; }
    const reference = localReference(source.base, image.uri, 'SPLIT_IMAGE_URI');
    if (!imageFiles.has(reference.file)) {
      const read = await readBounded(reference.path, LIMITS.imageBytes, 'SPLIT_IMAGE');
      if (!within(source.base, read.canonical)) fail('SPLIT_IMAGE_OUTSIDE_SOURCE', image.uri);
      imageBytes += read.bytes.length;
      if (imageBytes > LIMITS.imagesBytes) fail('SPLIT_IMAGE_BYTE_BUDGET', imageBytes);
      imageFiles.set(reference.file, { canonical: read.canonical, dev: read.dev, ino: read.ino, bytes: read.bytes.length, sha256: sha256(read.bytes) });
    }
    images.push({ image: i, uri: image.uri, file: reference.file, inline: false });
  }
  let license = null;
  if (config.license) {
    const read = await readBounded(config.license, LIMITS.licenseBytes, 'SPLIT_LICENSE');
    license = { file: basename(read.canonical), data: read.bytes };
  }
  if (license && ([names.gltf, names.bin, 'manifest.json'].includes(license.file))) fail('SPLIT_OUTPUT_NAME_COLLISION', license.file);
  const reserved = new Set([names.gltf, names.bin, 'manifest.json', ...(license ? [license.file] : [])]);
  for (const file of imageFiles.keys()) {
    if (reserved.has(file) || file.startsWith('provenance/')) fail('SPLIT_OUTPUT_NAME_COLLISION', file);
  }
  const built = buildOutput(source, config, names);
  const checked = verifyOutput(source, built);
  const gltfBytes = jsonBytes(built.output);
  if (gltfBytes.length > LIMITS.jsonBytes) fail('SPLIT_JSON_BYTE_BUDGET', 'output document');

  const parent = dirname(config.output);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(join(parent, `.${basename(config.output)}.tmp-`));
  try {
    const files = [];
    const write = async (file, bytes) => {
      const path = resolve(temporary, file);
      if (!within(temporary, path)) fail('SPLIT_OUTPUT_PATH', file);
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, bytes, { flag: 'wx' });
      const entry = { file, bytes: bytes.byteLength, sha256: sha256(bytes) };
      files.push(entry);
      return entry;
    };
    const gltf = await write(names.gltf, gltfBytes), bin = await write(names.bin, built.bin);
    const methods = {};
    for (const [file, image] of imageFiles) {
      const path = resolve(temporary, file);
      if (!within(temporary, path)) fail('SPLIT_OUTPUT_PATH', file);
      await mkdir(dirname(path), { recursive: true });
      const method = await linkOrCopy(image.canonical, path);
      methods[method] = (methods[method] ?? 0) + 1;
      const linked = await stat(path);
      if (linked.size !== image.bytes || (method === 'link' && (linked.ino !== image.ino || linked.dev !== image.dev))
        || (method === 'copy' && sha256(await readFile(path)) !== image.sha256)) fail('SPLIT_SOURCE_CHANGED', file);
      files.push({ file, bytes: image.bytes, sha256: image.sha256 });
    }
    const licenseEntry = license ? await write(license.file, license.data) : null;
    for (const record of built.outputRecords) {
      if (!record.provenance) continue;
      const triangles = await write(record.provenance.triangles.file, uint32Bytes(record.provenance.triangles.data));
      const vertices = await write(record.provenance.vertices.file, uint32Bytes(record.provenance.vertices.data));
      record.provenance = { triangles, vertices };
    }
    const sourcePrimitives = built.sourceRecords.map(record => {
      const entry = { meshIndex: record.meshIndex, primitiveIndex: record.primitiveIndex, mode: record.mode,
        materialIndex: record.materialIndex, triangles: record.triangles, vertices: record.vertices,
        exceedsLimits: record.exceedsLimits, split: record.split, outputPrimitives: record.outputPrimitives };
      if (!record.split) return entry;
      const triangleCounts = record.parts.map(part => part.triangles.length), vertexCounts = record.parts.map(part => part.vertices.length);
      return { ...entry, parts: record.parts.length, maxDepth: record.maxDepth,
        partTriangles: { min: Math.min(...triangleCounts), max: Math.max(...triangleCounts) },
        partVertices: { min: Math.min(...vertexCounts), max: Math.max(...vertexCounts), total: record.partVertices },
        referencedVertices: record.referencedVertices, unreferencedVertices: record.vertices - record.referencedVertices,
        duplicatedVertices: record.partVertices - record.referencedVertices,
        coverage: { sourceTriangles: record.triangles, coveredTriangles: triangleCounts.reduce((a, b) => a + b, 0), exactlyOnce: true } };
    });
    const splitRecords = built.sourceRecords.filter(record => record.split);
    const manifest = { version: 1, format: 'real-engine-gltf-primitive-split',
      tool: { script: 'scripts/split-gltf-primitives.mjs', sha256: sha256(await readFile(SCRIPT)) },
      purpose: 'offline deterministic split of oversized triangle primitives; geometry, textures and scene loading remain separate tools',
      algorithm: ALGORITHM,
      source: { file: basename(source.json.canonical), bytes: source.json.bytes.length, sha256: sha256(source.json.bytes),
        buffers: source.buffers.map(buffer => ({ index: buffer.index, uri: buffer.uri, file: buffer.file, byteLength: buffer.byteLength,
          bytes: buffer.bytes, sha256: buffer.sha256 })) },
      options: { maxTriangles: config.maxTriangles, maxVertices: config.maxVertices,
        selection: config.primitives ? 'listed' : 'exceeding-limits',
        primitives: config.primitives ? config.primitives.map(([meshIndex, primitiveIndex]) => ({ meshIndex, primitiveIndex })) : null },
      limits: LIMITS,
      output: { gltf, bin },
      license: licenseEntry,
      images: images.map(image => image.inline ? { image: image.image, inline: true }
        : { image: image.image, uri: image.uri, file: image.file, ...(({ bytes, sha256: hash }) => ({ bytes, sha256: hash }))(imageFiles.get(image.file)),
          identicalToSource: true }),
      preserved: 'asset, scenes, nodes (transforms and mesh references), materials, textures, samplers, images, extensions and extras are unchanged; '
        + 'unsplit primitives keep their accessor element bytes; split parts replace their source primitive in place inside the same mesh',
      provenance: 'uint32 little-endian; *-source-triangles.bin: output triangle -> source primitive triangle index; '
        + '*-source-vertices.bin: output vertex -> source primitive vertex index (ascending)',
      verification: { ...checked, attributeBytesIdentical: true, trianglesIdentical: true, coverageExactlyOnce: true,
        documentOutsideMeshesIdentical: true },
      totals: {
        sourcePrimitives: built.sourceRecords.length, outputPrimitives: built.outputRecords.length,
        splitPrimitives: splitRecords.length, parts: splitRecords.reduce((sum, record) => sum + record.parts.length, 0),
        sourceTriangles: built.sourceRecords.reduce((sum, record) => sum + (record.triangles ?? 0), 0),
        outputTriangles: built.outputRecords.reduce((sum, record) => sum + (record.triangleCount ?? 0), 0),
        splitSourceTriangles: splitRecords.reduce((sum, record) => sum + record.triangles, 0),
        splitSourceVertices: splitRecords.reduce((sum, record) => sum + record.vertices, 0),
        partVertices: splitRecords.reduce((sum, record) => sum + record.partVertices, 0),
        duplicatedVertices: splitRecords.reduce((sum, record) => sum + record.partVertices - record.referencedVertices, 0),
        binBytes: built.bin.length },
      sourcePrimitives,
      primitives: built.outputRecords,
      files: [...files].sort(byPath) };
    await write('manifest.json', jsonBytes(manifest));
    await publishDirectory(temporary, config.output, { hardLinkFiles: true });
    process.stdout.write(JSON.stringify({ output: config.output, totals: manifest.totals, images: methods,
      elapsedMs: Math.round(performance.now() - started) }) + '\n');
  } finally { await removeOwnTemporary(temporary, parent, config.output); }
}

try { await main(); }
catch (error) { process.stderr.write(`split-gltf-primitives: ${error.message}\n`); process.exitCode = 1; }
