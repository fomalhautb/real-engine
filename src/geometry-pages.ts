import { MeshoptDecoder } from 'meshoptimizer/decoder';
import type { GeometryHierarchy, GeometryCluster } from './geometry-hierarchy-types';

/** Self-contained, lossless page encoding for offline geometry residency experiments. */
export interface GeometryPageOptions {
  /** Maximum decoded payload bytes resident for one page. Default 64 KiB. */
  maxPageBytes?: number;
  /** Maximum encoded plus decoded bytes across the output. Default 256 MiB. */
  maxTotalBytes?: number;
}

export interface GeometryPage {
  id: number;
  clusterIds: number[];
  decodedBytes: number;
  bytes: Uint8Array;
}

export interface EncodedGeometryPages {
  version: 1;
  /** 80 rest bytes plus independent optional UV (8), tangent (16), and color (16) streams. */
  vertexStride: number;
  hasColors?: true;
  pages: GeometryPage[];
  clusterPages: number[];
  rootPages: number[];
}

export interface DecodedGeometryCluster {
  id: number;
  sourceVertexIds: Uint32Array;
  /** Copied, interleaved 80-byte rest records in local vertex order. */
  rest: Uint8Array;
  uvs?: Float32Array;
  tangents?: Float32Array;
  colors?: Float32Array;
  localTriangles: Uint8Array;
}

export interface DecodedGeometryPage {
  version: 1;
  vertexStride: number;
  hasUvs: boolean;
  hasTangents: boolean;
  hasColors?: boolean;
  sourceVertexCount: number;
  hierarchyClusterCount: number;
  clusters: DecodedGeometryCluster[];
}

export interface DecodeGeometryPageOptions {
  maxPageBytes?: number;
  maxEncodedBytes?: number;
}

const PAGE_MAGIC = 0x4750_4752; // ASCII RGPG in little-endian storage.
const HEADER_BYTES = 40;
const CLUSTER_HEADER_BYTES = 16;
const REST_BYTES = 80;
const FLAG_UV = 1;
const FLAG_TANGENT = 2;
const FLAG_COLOR = 4;
const DEFAULT_PAGE_BYTES = 65_536;
const HARD_PAGE_BYTES = 4 * 1024 * 1024;
const DEFAULT_TOTAL_BYTES = 256 * 1024 * 1024;
const HARD_TOTAL_BYTES = 512 * 1024 * 1024;
const MAX_SOURCE_VERTICES = 16_000_000;
const MAX_CLUSTERS = 1_000_000;
const MAX_LOCAL_VERTICES = 256; // Local triangle indices are u8.
const MAX_LOCAL_TRIANGLES = 4_096;

function positiveInteger(value: number, label: string, hardMax: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > hardMax) {
    throw new RangeError(`${label} must be a positive integer <= ${hardMax}`);
  }
  return value;
}

function validOptions(options: GeometryPageOptions): Required<GeometryPageOptions> {
  return {
    maxPageBytes: positiveInteger(options.maxPageBytes ?? DEFAULT_PAGE_BYTES, 'maxPageBytes', HARD_PAGE_BYTES),
    maxTotalBytes: positiveInteger(options.maxTotalBytes ?? DEFAULT_TOTAL_BYTES, 'maxTotalBytes', HARD_TOTAL_BYTES),
  };
}

function aligned4(value: number): number { return (value + 3) & ~3; }

function checksum(bytes: Uint8Array): number {
  let hash = 0x811c9dc5;
  for (const byte of bytes) hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
  return hash;
}

function validateRestFloats(view: DataView, byteOffset: number, label: string): void {
  for (let word = 0; word < 20; word++) {
    if (word >= 8 && word <= 11) continue; // Four joint IDs are u32, not floats.
    if (!Number.isFinite(view.getFloat32(byteOffset + word * 4, true))) {
      const field = word < 4 ? 'position' : word < 8 ? 'normal' : word < 16 ? 'weight' : 'morph';
      throw new RangeError(`${label} has nonfinite ${field} rest float`);
    }
  }
}

async function requireEncoder() {
  // Offline encoding must not initialize its WASM module in decode-only Workers.
  const { MeshoptEncoder } = await import('meshoptimizer/encoder');
  if (!MeshoptEncoder.supported) throw new Error('GEOMETRY_PAGE_MESHOPT_ENCODER_UNSUPPORTED');
  try { await MeshoptEncoder.ready; }
  catch (error) { throw new Error(`GEOMETRY_PAGE_MESHOPT_ENCODER_UNAVAILABLE: ${String(error)}`); }
  if (!MeshoptEncoder.supported) throw new Error('GEOMETRY_PAGE_MESHOPT_ENCODER_UNSUPPORTED');
  return MeshoptEncoder;
}

async function requireDecoder(): Promise<void> {
  if (!MeshoptDecoder.supported) throw new Error('GEOMETRY_PAGE_MESHOPT_DECODER_UNSUPPORTED');
  try { await MeshoptDecoder.ready; }
  catch (error) { throw new Error(`GEOMETRY_PAGE_MESHOPT_DECODER_UNAVAILABLE: ${String(error)}`); }
  if (!MeshoptDecoder.supported) throw new Error('GEOMETRY_PAGE_MESHOPT_DECODER_UNSUPPORTED');
}

function flagsAndStride(hierarchy: GeometryHierarchy): { flags: number; stride: number } {
  const source = hierarchy.source;
  const flags = (source.uvs ? FLAG_UV : 0) | (source.tangents ? FLAG_TANGENT : 0) | (source.colors ? FLAG_COLOR : 0);
  return { flags, stride: REST_BYTES + (source.uvs ? 8 : 0) + (source.tangents ? 16 : 0) + (source.colors ? 16 : 0) };
}

function validateSource(hierarchy: GeometryHierarchy): void {
  if (hierarchy.version !== 1 || !hierarchy.source || !Array.isArray(hierarchy.clusters)
    || !Array.isArray(hierarchy.roots)) throw new RangeError('Invalid geometry hierarchy');
  const source = hierarchy.source;
  positiveInteger(source.vertexCount, 'source vertexCount', MAX_SOURCE_VERTICES);
  positiveInteger(hierarchy.clusters.length, 'cluster count', MAX_CLUSTERS);
  if (!(source.vertices instanceof ArrayBuffer) || source.vertices.byteLength !== source.vertexCount * REST_BYTES) {
    throw new RangeError('Source rest buffer length does not match vertexCount');
  }
  if (source.uvs && (!(source.uvs instanceof Float32Array) || source.uvs.length !== source.vertexCount * 2)) {
    throw new RangeError('Source UV length does not match vertexCount');
  }
  if (source.tangents && (!(source.tangents instanceof Float32Array) || source.tangents.length !== source.vertexCount * 4)) {
    throw new RangeError('Source tangent length does not match vertexCount');
  }
  if (source.colors && (!(source.colors instanceof Float32Array) || source.colors.length !== source.vertexCount * 4
    || !source.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1))) {
    throw new RangeError('Source color length does not match vertexCount');
  }
  const roots = new Set<number>();
  for (const id of hierarchy.roots) {
    if (!Number.isInteger(id) || id < 0 || id >= hierarchy.clusters.length || roots.has(id)) {
      throw new RangeError('Root cluster ID is invalid or duplicated');
    }
    roots.add(id);
  }
  if (roots.size === 0) throw new RangeError('Hierarchy requires root clusters');
}

function clusterDecodedBytes(cluster: GeometryCluster, id: number, vertexCount: number, stride: number,
  restView: DataView, uvs?: Float32Array, tangents?: Float32Array, colors?: Float32Array): number {
  if (!(cluster.vertices instanceof Uint32Array) || !(cluster.triangles instanceof Uint8Array)) {
    throw new RangeError(`Cluster ${id} arrays have invalid types`);
  }
  const vertices = cluster.vertices.length, indices = cluster.triangles.length;
  if (vertices < 1 || vertices > MAX_LOCAL_VERTICES || indices < 3 || indices % 3 !== 0
    || indices / 3 > MAX_LOCAL_TRIANGLES) throw new RangeError(`Cluster ${id} has invalid local counts`);
  const seen = new Set<number>();
  for (const sourceId of cluster.vertices) {
    if (sourceId >= vertexCount || seen.has(sourceId)) throw new RangeError(`Cluster ${id} has invalid or duplicate source vertex ID`);
    seen.add(sourceId);
    validateRestFloats(restView, sourceId * REST_BYTES, `Cluster ${id} source`);
    if (uvs) for (let axis = 0; axis < 2; axis++) {
      if (!Number.isFinite(uvs[sourceId * 2 + axis])) throw new RangeError(`Cluster ${id} has nonfinite UV`);
    }
    if (tangents) for (let axis = 0; axis < 4; axis++) {
      if (!Number.isFinite(tangents[sourceId * 4 + axis])) throw new RangeError(`Cluster ${id} has nonfinite tangent`);
    }
    if (colors) for (let axis = 0; axis < 4; axis++) {
      if (!Number.isFinite(colors[sourceId * 4 + axis]) || colors[sourceId * 4 + axis] < 0
        || colors[sourceId * 4 + axis] > 1) throw new RangeError(`Cluster ${id} has invalid color`);
    }
  }
  for (const localIndex of cluster.triangles) {
    if (localIndex >= vertices) throw new RangeError(`Cluster ${id} has out-of-range local triangle index`);
  }
  return CLUSTER_HEADER_BYTES + vertices * (4 + stride) + aligned4(indices);
}

function appendCluster(target: Uint8Array, offset: number, cluster: GeometryCluster, id: number,
  flags: number, stride: number, source: GeometryHierarchy['source']): number {
  const vertexCount = cluster.vertices.length, indexCount = cluster.triangles.length;
  const view = new DataView(target.buffer, target.byteOffset, target.byteLength);
  view.setUint32(offset, id, true);
  view.setUint32(offset + 4, vertexCount, true);
  view.setUint32(offset + 8, indexCount / 3, true);
  view.setUint32(offset + 12, flags, true);
  offset += CLUSTER_HEADER_BYTES;
  for (const sourceId of cluster.vertices) { view.setUint32(offset, sourceId, true); offset += 4; }
  const rest = new Uint8Array(source.vertices);
  for (const sourceId of cluster.vertices) {
    target.set(rest.subarray(sourceId * REST_BYTES, (sourceId + 1) * REST_BYTES), offset);
    offset += REST_BYTES;
    if (flags & FLAG_UV) {
      for (let axis = 0; axis < 2; axis++) view.setFloat32(offset + axis * 4, source.uvs![sourceId * 2 + axis], true);
      offset += 8;
    }
    if (flags & FLAG_TANGENT) {
      for (let axis = 0; axis < 4; axis++) view.setFloat32(offset + axis * 4, source.tangents![sourceId * 4 + axis], true);
      offset += 16;
    }
    if (flags & FLAG_COLOR) {
      for (let axis = 0; axis < 4; axis++) view.setFloat32(offset + axis * 4, source.colors![sourceId * 4 + axis], true);
      offset += 16;
    }
  }
  if (offset % 4 !== 0) throw new Error(`Cluster ${id} record alignment failed (${stride})`);
  target.set(cluster.triangles, offset);
  return offset + aligned4(indexCount);
}

/** Greedy stable packing; root clusters precede and never share pages with details.
 * Each page is independently decodable, with no runtime streaming implied. */
export async function encodeGeometryPages(
  hierarchy: GeometryHierarchy, options: GeometryPageOptions = {},
): Promise<EncodedGeometryPages> {
  const limits = validOptions(options);
  validateSource(hierarchy);
  const source = hierarchy.source;
  const { flags, stride } = flagsAndStride(hierarchy);
  const restView = new DataView(source.vertices);
  const sizes = hierarchy.clusters.map((cluster, id) => {
    const bytes = clusterDecodedBytes(cluster, id, source.vertexCount, stride, restView, source.uvs, source.tangents, source.colors);
    if (bytes > limits.maxPageBytes) throw new RangeError(`Cluster ${id} exceeds maxPageBytes decoded residency`);
    return bytes;
  });
  const MeshoptEncoder = await requireEncoder();
  const pages: GeometryPage[] = [];
  const clusterPages = new Array<number>(hierarchy.clusters.length);
  let decodedTotal = 0, encodedTotal = 0;
  let ids: number[] = [], rawBytes = 0;
  const flush = (): void => {
    if (ids.length === 0) return;
    if (decodedTotal + encodedTotal + rawBytes > limits.maxTotalBytes) {
      throw new RangeError('Geometry pages exceed maxTotalBytes cumulative encoded plus decoded budget');
    }
    const raw = new Uint8Array(rawBytes);
    let offset = 0;
    for (const id of ids) offset = appendCluster(raw, offset, hierarchy.clusters[id], id, flags, stride, source);
    if (offset !== raw.length) throw new Error('Geometry page layout mismatch');
    const compressed = MeshoptEncoder.encodeVertexBuffer(raw, raw.length / 4, 4);
    const page = new Uint8Array(HEADER_BYTES + compressed.length);
    const header = new DataView(page.buffer);
    header.setUint32(0, PAGE_MAGIC, true);
    header.setUint16(4, 1, true);
    header.setUint16(6, flags, true);
    header.setUint16(8, stride, true);
    header.setUint32(12, ids.length, true);
    header.setUint32(16, raw.length, true);
    header.setUint32(20, compressed.length, true);
    header.setUint32(24, source.vertexCount, true);
    header.setUint32(28, hierarchy.clusters.length, true);
    header.setUint32(32, checksum(raw), true);
    page.set(compressed, HEADER_BYTES);
    decodedTotal += raw.length;
    encodedTotal += page.length;
    if (decodedTotal + encodedTotal > limits.maxTotalBytes) {
      throw new RangeError('Geometry pages exceed maxTotalBytes cumulative encoded plus decoded budget');
    }
    const pageId = pages.length;
    for (const id of ids) clusterPages[id] = pageId;
    pages.push({ id: pageId, clusterIds: [...ids], decodedBytes: raw.length, bytes: page });
    ids = []; rawBytes = 0;
  };
  const append = (id: number): void => {
    if (ids.length > 0 && rawBytes + sizes[id] > limits.maxPageBytes) flush();
    ids.push(id);
    rawBytes += sizes[id];
  };
  const rootIds = [...hierarchy.roots].sort((a, b) => a - b), roots = new Set(rootIds);
  for (const id of rootIds) append(id);
  flush(); // The last root page cannot acquire detail clusters.
  for (let id = 0; id < sizes.length; id++) if (!roots.has(id)) append(id);
  flush();
  const rootPages = [...new Set(hierarchy.roots.map((id) => clusterPages[id]))].sort((a, b) => a - b);
  return { version: 1, vertexStride: stride, ...(source.colors ? { hasColors: true as const } : {}),
    pages, clusterPages, rootPages };
}

/** Decode one untrusted page after checking the declared allocation size and exact wire length. */
export async function decodeGeometryPage(
  bytes: Uint8Array, options: DecodeGeometryPageOptions = {},
): Promise<DecodedGeometryPage> {
  const maxPageBytes = positiveInteger(options.maxPageBytes ?? DEFAULT_PAGE_BYTES, 'maxPageBytes', HARD_PAGE_BYTES);
  const maxEncodedBytes = positiveInteger(options.maxEncodedBytes ?? maxPageBytes * 2 + HEADER_BYTES,
    'maxEncodedBytes', HARD_PAGE_BYTES * 2 + HEADER_BYTES);
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < HEADER_BYTES || bytes.byteLength > maxEncodedBytes) {
    throw new RangeError('Geometry page encoded byte length is invalid or over budget');
  }
  const header = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = header.getUint32(0, true), version = header.getUint16(4, true);
  const flags = header.getUint16(6, true), stride = header.getUint16(8, true);
  const reserved16 = header.getUint16(10, true), clusterCount = header.getUint32(12, true);
  const decodedBytes = header.getUint32(16, true), encodedBytes = header.getUint32(20, true);
  const sourceVertexCount = header.getUint32(24, true), hierarchyClusterCount = header.getUint32(28, true);
  const expectedChecksum = header.getUint32(32, true), reserved32 = header.getUint32(36, true);
  if (magic !== PAGE_MAGIC || version !== 1 || reserved16 !== 0 || reserved32 !== 0
    || (flags & ~(FLAG_UV | FLAG_TANGENT | FLAG_COLOR)) !== 0
    || stride !== REST_BYTES + ((flags & FLAG_UV) ? 8 : 0) + ((flags & FLAG_TANGENT) ? 16 : 0)
      + ((flags & FLAG_COLOR) ? 16 : 0)) {
    throw new RangeError('Geometry page header has invalid magic, version, flags, or stride');
  }
  if (clusterCount < 1 || clusterCount > hierarchyClusterCount || clusterCount > maxPageBytes / CLUSTER_HEADER_BYTES
    || sourceVertexCount < 1 || sourceVertexCount > MAX_SOURCE_VERTICES
    || hierarchyClusterCount < 1 || hierarchyClusterCount > MAX_CLUSTERS) {
    throw new RangeError('Geometry page header has invalid counts');
  }
  if (decodedBytes < clusterCount * CLUSTER_HEADER_BYTES || decodedBytes > maxPageBytes || decodedBytes % 4 !== 0
    || encodedBytes < 1 || encodedBytes + HEADER_BYTES !== bytes.byteLength) {
    throw new RangeError('Geometry page header has invalid payload lengths or exceeds budget');
  }
  await requireDecoder();
  // This is the first allocation influenced by the untrusted page header.
  const raw = new Uint8Array(decodedBytes);
  try { MeshoptDecoder.decodeVertexBuffer(raw, raw.length / 4, 4, bytes.subarray(HEADER_BYTES)); }
  catch (error) { throw new RangeError(`Geometry page compressed payload is invalid: ${String(error)}`); }
  if (checksum(raw) !== expectedChecksum) throw new RangeError('Geometry page checksum mismatch');
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const clusters: DecodedGeometryCluster[] = [];
  const seen = new Set<number>();
  let offset = 0;
  for (let ordinal = 0; ordinal < clusterCount; ordinal++) {
    if (offset + CLUSTER_HEADER_BYTES > raw.length) throw new RangeError('Truncated geometry cluster header');
    const id = view.getUint32(offset, true), vertexCount = view.getUint32(offset + 4, true);
    const triangleCount = view.getUint32(offset + 8, true), clusterFlags = view.getUint32(offset + 12, true);
    offset += CLUSTER_HEADER_BYTES;
    if (id >= hierarchyClusterCount || seen.has(id) || clusterFlags !== flags
      || vertexCount < 1 || vertexCount > MAX_LOCAL_VERTICES
      || triangleCount < 1 || triangleCount > MAX_LOCAL_TRIANGLES) {
      throw new RangeError('Geometry page has duplicate/out-of-range cluster ID, flags, or counts');
    }
    seen.add(id);
    const indexBytes = triangleCount * 3;
    const recordBytes = vertexCount * (4 + stride) + aligned4(indexBytes);
    if (recordBytes > raw.length - offset) throw new RangeError('Truncated geometry cluster data');
    const sourceVertexIds = new Uint32Array(vertexCount), localIds = new Set<number>();
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const sourceId = view.getUint32(offset, true);
      if (sourceId >= sourceVertexCount || localIds.has(sourceId)) throw new RangeError('Out-of-range or duplicate source vertex ID');
      sourceVertexIds[vertex] = sourceId; localIds.add(sourceId); offset += 4;
    }
    const rest = new Uint8Array(vertexCount * REST_BYTES);
    const uvs = flags & FLAG_UV ? new Float32Array(vertexCount * 2) : undefined;
    const tangents = flags & FLAG_TANGENT ? new Float32Array(vertexCount * 4) : undefined;
    const colors = flags & FLAG_COLOR ? new Float32Array(vertexCount * 4) : undefined;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      rest.set(raw.subarray(offset, offset + REST_BYTES), vertex * REST_BYTES);
      validateRestFloats(view, offset, 'Geometry page');
      offset += REST_BYTES;
      if (uvs) for (let axis = 0; axis < 2; axis++) {
        const value = view.getFloat32(offset, true);
        if (!Number.isFinite(value)) throw new RangeError('Geometry page contains nonfinite UV');
        uvs[vertex * 2 + axis] = value; offset += 4;
      }
      if (tangents) for (let axis = 0; axis < 4; axis++) {
        const value = view.getFloat32(offset, true);
        if (!Number.isFinite(value)) throw new RangeError('Geometry page contains nonfinite tangent');
        tangents[vertex * 4 + axis] = value; offset += 4;
      }
      if (colors) for (let axis = 0; axis < 4; axis++) {
        const value = view.getFloat32(offset, true);
        if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError('Geometry page contains invalid color');
        colors[vertex * 4 + axis] = value; offset += 4;
      }
    }
    const localTriangles = raw.slice(offset, offset + indexBytes);
    for (const index of localTriangles) if (index >= vertexCount) throw new RangeError('Out-of-range local triangle index');
    offset += indexBytes;
    for (let padding = indexBytes; padding < aligned4(indexBytes); padding++) {
      if (raw[offset++] !== 0) throw new RangeError('Geometry page has nonzero cluster padding');
    }
    clusters.push({ id, sourceVertexIds, rest, uvs, tangents, colors, localTriangles });
  }
  if (offset !== raw.length) throw new RangeError('Geometry page has trailing decoded bytes');
  return { version: 1, vertexStride: stride, hasUvs: Boolean(flags & FLAG_UV), hasTangents: Boolean(flags & FLAG_TANGENT),
    ...(flags & FLAG_COLOR ? { hasColors: true } : {}),
    sourceVertexCount, hierarchyClusterCount, clusters };
}
