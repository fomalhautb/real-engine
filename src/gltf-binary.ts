/** The glTF 2.0 binary fields needed by the importer. */
export interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
  sparse?: {
    count: number;
    indices: { bufferView: number; byteOffset?: number; componentType: number };
    values: { bufferView: number; byteOffset?: number };
  };
}

export interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
}

export interface GltfBinaryDocument {
  buffers?: { byteLength: number; uri?: string }[];
  bufferViews?: GltfBufferView[];
  accessors?: GltfAccessor[];
}

export interface GltfAccessorData {
  values: Float32Array | Uint32Array;
  count: number;
  components: number;
  componentType: number;
  type: string;
  normalized: boolean;
}

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;
const DEFAULT_MAX_ELEMENTS = 16_000_000;

function integer(value: number, label: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${label} must be a safe integer >= ${minimum}`);
  }
  return value;
}

function checkedEnd(start: number, length: number, limit: number, label: string): number {
  const end = start + length;
  if (!Number.isSafeInteger(end) || end > limit) {
    throw new Error(`${label} is out of bounds`);
  }
  return end;
}

/** Parses a complete GLB 2 container. Unknown extension chunks after JSON/BIN are skipped. */
export function parseGlb(data: ArrayBuffer): { json: Record<string, unknown>; bin?: ArrayBuffer } {
  if (!(data instanceof ArrayBuffer) || data.byteLength < 20 || data.byteLength % 4 !== 0) {
    throw new Error('Invalid GLB length');
  }
  const view = new DataView(data);
  if (view.getUint32(0, true) !== GLB_MAGIC || view.getUint32(4, true) !== 2) {
    throw new Error('Invalid GLB magic or version');
  }
  if (view.getUint32(8, true) !== data.byteLength) {
    throw new Error('GLB declared length does not match input');
  }

  let offset = 12;
  let chunkIndex = 0;
  let json: Record<string, unknown> | undefined;
  let bin: ArrayBuffer | undefined;
  while (offset < data.byteLength) {
    if (offset + 8 > data.byteLength) throw new Error('Truncated GLB chunk header');
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    offset += 8;
    if (length % 4 !== 0) throw new Error('Unaligned GLB chunk length');
    const end = checkedEnd(offset, length, data.byteLength, 'GLB chunk');
    if (chunkIndex === 0) {
      if (type !== JSON_CHUNK || length === 0) throw new Error('GLB JSON chunk must be first');
      const content = new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(data, offset, length));
      const parsed: unknown = JSON.parse(content);
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('GLB JSON root must be an object');
      }
      json = parsed as Record<string, unknown>;
    } else if (type === BIN_CHUNK) {
      if (chunkIndex !== 1 || bin !== undefined) throw new Error('GLB BIN chunk must occur once, second');
      bin = data.slice(offset, end);
    } else if (type === JSON_CHUNK) {
      throw new Error('Duplicate GLB JSON chunk');
    }
    offset = end;
    chunkIndex++;
  }
  if (!json) throw new Error('Missing GLB JSON chunk');
  return bin === undefined ? { json } : { json, bin };
}

function componentBytes(type: number): number {
  switch (type) {
    case 5120: case 5121: return 1;
    case 5122: case 5123: return 2;
    case 5125: case 5126: return 4;
    default: throw new Error(`Unsupported glTF component type ${type}`);
  }
}

function componentCount(type: string): number {
  switch (type) {
    case 'SCALAR': return 1;
    case 'VEC2': return 2;
    case 'VEC3': return 3;
    case 'VEC4': case 'MAT2': return 4;
    case 'MAT3': return 9;
    case 'MAT4': return 16;
    default: throw new Error(`Unsupported glTF accessor type ${type}`);
  }
}

interface Layout {
  components: number;
  componentBytes: number;
  columnRows: number;
  columnStride: number;
  elementStride: number;
  lastByte: number;
}

function layout(type: string, componentType: number): Layout {
  const components = componentCount(type);
  const bytes = componentBytes(componentType);
  const columns = type.startsWith('MAT') ? Number(type[3]) : 1;
  const rows = components / columns;
  const columnBytes = rows * bytes;
  const columnStride = columns === 1 ? columnBytes : Math.ceil(columnBytes / 4) * 4;
  return {
    components,
    componentBytes: bytes,
    columnRows: rows,
    columnStride,
    elementStride: columns * columnStride,
    lastByte: (columns - 1) * columnStride + columnBytes,
  };
}

interface ResolvedView {
  data: DataView;
  byteLength: number;
  byteStride?: number;
  target?: number;
}

function resolveView(doc: GltfBinaryDocument, buffers: readonly ArrayBuffer[], index: number): ResolvedView {
  integer(index, 'bufferView index');
  const definition = doc.bufferViews?.[index];
  if (!definition) throw new Error(`Missing glTF bufferView ${index}`);
  const bufferIndex = integer(definition.buffer, 'buffer index');
  const buffer = buffers[bufferIndex];
  if (!(buffer instanceof ArrayBuffer)) throw new Error(`Missing glTF buffer ${bufferIndex}`);
  const declaredLength = doc.buffers?.[bufferIndex]?.byteLength;
  if (doc.buffers && !doc.buffers[bufferIndex]) throw new Error(`Missing glTF buffer definition ${bufferIndex}`);
  const limit = declaredLength === undefined ? buffer.byteLength : integer(declaredLength, 'buffer byteLength');
  if (buffer.byteLength < limit) throw new Error(`glTF buffer ${bufferIndex} is shorter than declared`);
  const offset = integer(definition.byteOffset ?? 0, 'bufferView byteOffset');
  const length = integer(definition.byteLength, 'bufferView byteLength', 1);
  checkedEnd(offset, length, limit, 'bufferView');
  const stride = definition.byteStride;
  if (stride !== undefined && (integer(stride, 'bufferView byteStride', 4) > 252 || stride % 4 !== 0)) {
    throw new Error('Invalid bufferView byteStride');
  }
  return { data: new DataView(buffer, offset, length), byteLength: length, byteStride: stride, target: definition.target };
}

function validateSlice(view: ResolvedView, offset: number, count: number, stride: number, lastByte: number, bytes: number, label: string): void {
  integer(offset, `${label} byteOffset`);
  if (offset % bytes !== 0 || (view.data.byteOffset + offset) % bytes !== 0) {
    throw new Error(`${label} byteOffset is misaligned`);
  }
  const length = (count - 1) * stride + lastByte;
  if (!Number.isSafeInteger(length)) throw new Error(`${label} byte length is unsafe`);
  checkedEnd(offset, length, view.byteLength, label);
}

function readComponent(view: DataView, offset: number, type: number): number {
  switch (type) {
    case 5120: return view.getInt8(offset);
    case 5121: return view.getUint8(offset);
    case 5122: return view.getInt16(offset, true);
    case 5123: return view.getUint16(offset, true);
    case 5125: return view.getUint32(offset, true);
    case 5126: {
      const value = view.getFloat32(offset, true);
      if (!Number.isFinite(value)) throw new Error('glTF accessor contains non-finite FLOAT value');
      return value;
    }
    default: throw new Error(`Unsupported glTF component type ${type}`);
  }
}

function normalize(value: number, type: number): number {
  switch (type) {
    case 5120: return Math.max(value / 127, -1);
    case 5121: return value / 255;
    case 5122: return Math.max(value / 32767, -1);
    case 5123: return value / 65535;
    default: return value;
  }
}

function copyElements(
  output: Float32Array | Uint32Array,
  destinationIndices: readonly number[] | undefined,
  source: ResolvedView,
  sourceOffset: number,
  count: number,
  stride: number,
  shape: Layout,
  componentType: number,
  normalized: boolean,
): void {
  for (let element = 0; element < count; element++) {
    const base = sourceOffset + element * stride;
    const destination = (destinationIndices?.[element] ?? element) * shape.components;
    for (let component = 0; component < shape.components; component++) {
      const column = Math.floor(component / shape.columnRows);
      const row = component % shape.columnRows;
      const raw = readComponent(source.data, base + column * shape.columnStride + row * shape.componentBytes, componentType);
      output[destination + component] = normalized ? normalize(raw, componentType) : raw;
    }
  }
}

/** Decodes a bounded accessor, including interleaving, matrix padding, and sparse overlays. */
export function readAccessor(
  doc: GltfBinaryDocument,
  buffers: readonly ArrayBuffer[],
  index: number,
  options: { maxElements?: number } = {},
): GltfAccessorData {
  integer(index, 'accessor index');
  const accessor = doc.accessors?.[index];
  if (!accessor) throw new Error(`Missing glTF accessor ${index}`);
  const count = integer(accessor.count, 'accessor count', 1);
  const shape = layout(accessor.type, accessor.componentType);
  const maxElements = integer(options.maxElements ?? DEFAULT_MAX_ELEMENTS, 'maxElements', 1);
  if (count > Math.floor(maxElements / shape.components)) throw new Error('glTF accessor exceeds element limit');
  if (accessor.normalized !== undefined && typeof accessor.normalized !== 'boolean') {
    throw new Error('accessor normalized must be boolean');
  }
  const normalized = accessor.normalized ?? false;
  if (normalized && (accessor.componentType === 5125 || accessor.componentType === 5126)) {
    throw new Error('Normalized FLOAT or UNSIGNED_INT accessor is invalid');
  }
  const unsigned = accessor.componentType === 5121 || accessor.componentType === 5123 || accessor.componentType === 5125;
  const values = unsigned && !normalized
    ? new Uint32Array(count * shape.components)
    : new Float32Array(count * shape.components);

  if (accessor.bufferView !== undefined) {
    const source = resolveView(doc, buffers, accessor.bufferView);
    const offset = accessor.byteOffset ?? 0;
    const stride = source.byteStride ?? shape.elementStride;
    if (stride < shape.elementStride || stride % shape.componentBytes !== 0) {
      throw new Error('Invalid accessor byteStride');
    }
    validateSlice(source, offset, count, stride, shape.lastByte, shape.componentBytes, 'accessor');
    copyElements(values, undefined, source, offset, count, stride, shape, accessor.componentType, normalized);
  } else if (accessor.byteOffset !== undefined || accessor.sparse === undefined) {
    throw new Error('Accessor without bufferView requires sparse data and no byteOffset');
  }

  const sparse = accessor.sparse;
  if (sparse !== undefined) {
    const sparseCount = integer(sparse.count, 'sparse count', 1);
    if (sparseCount > count) throw new Error('Sparse count exceeds accessor count');
    const indexType = sparse.indices.componentType;
    if (indexType !== 5121 && indexType !== 5123 && indexType !== 5125) {
      throw new Error('Invalid sparse index component type');
    }
    const indexBytes = componentBytes(indexType);
    const indexSource = resolveView(doc, buffers, sparse.indices.bufferView);
    const valueSource = resolveView(doc, buffers, sparse.values.bufferView);
    if (indexSource.byteStride !== undefined || indexSource.target !== undefined ||
        valueSource.byteStride !== undefined || valueSource.target !== undefined) {
      throw new Error('Sparse bufferViews cannot have byteStride or target');
    }
    const indexOffset = sparse.indices.byteOffset ?? 0;
    const valueOffset = sparse.values.byteOffset ?? 0;
    validateSlice(indexSource, indexOffset, sparseCount, indexBytes, indexBytes, indexBytes, 'sparse indices');
    validateSlice(valueSource, valueOffset, sparseCount, shape.elementStride, shape.lastByte, shape.componentBytes, 'sparse values');
    const destinationIndices: number[] = [];
    let previous = -1;
    for (let i = 0; i < sparseCount; i++) {
      const destination = readComponent(indexSource.data, indexOffset + i * indexBytes, indexType);
      if (destination <= previous || destination >= count) throw new Error('Sparse indices must strictly increase within accessor bounds');
      destinationIndices.push(destination);
      previous = destination;
    }
    copyElements(values, destinationIndices, valueSource, valueOffset, sparseCount, shape.elementStride, shape, accessor.componentType, normalized);
  }

  return { values, count, components: shape.components, componentType: accessor.componentType, type: accessor.type, normalized };
}
