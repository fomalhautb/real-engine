import type { MeshData } from './geometry';
import { decodeVerifiedGeometryPage } from './geometry-page-decode';
import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryPageProvider, GeometryStreamDirectory } from './geometry-stream-types';

/** A fixed, source-free shadow/query approximation. Its metric is the largest
 * selected hierarchy error, not a certified Hausdorff or visibility bound. */
export interface GeometryQueryProxy {
  readonly version: 1;
  readonly primitiveKey: string;
  readonly directorySha256: string;
  readonly topologyVersion: number;
  readonly requestedError: number;
  readonly approximateObjectSpaceError: number;
  readonly selectedClusters: readonly number[];
  readonly mesh: MeshData;
}

export interface GeometryQueryProxyOptions {
  maxError?: number;
  topologyVersion?: number;
  maxVertices?: number;
  maxTriangles?: number;
  maxBytes?: number;
}

const MAGIC = 0x5051_4752; // RGQP
const HEADER_BYTES = 16;
const MAX_METADATA_BYTES = 1024 * 1024;
const DEFAULT_VERTICES = 262_144;
const DEFAULT_TRIANGLES = 65_536;
const DEFAULT_BYTES = 32 * 1024 * 1024;
const HARD_VERTICES = 1_048_576;
const HARD_TRIANGLES = 262_144;
const HARD_BYTES = 128 * 1024 * 1024;
const UINT32_MAX = 0xffff_ffff;

type Limits = Required<GeometryQueryProxyOptions>;
type Bounds = MeshData['bounds'];

function fail(reason: string): never { throw new RangeError(`GEOMETRY_QUERY_PROXY_${reason}`); }
function integer(value: unknown, min: number, max: number, reason: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) fail(reason);
  return value;
}
function finite(value: unknown, reason: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1e30) fail(reason);
  return value;
}
function limits(options: GeometryQueryProxyOptions = {}): Limits {
  if (!options || typeof options !== 'object') fail('OPTIONS');
  return {
    maxError: finite(options.maxError ?? 1e30, 'MAX_ERROR'),
    topologyVersion: integer(options.topologyVersion ?? 0, 0, UINT32_MAX, 'TOPOLOGY_VERSION'),
    maxVertices: integer(options.maxVertices ?? DEFAULT_VERTICES, 1, HARD_VERTICES, 'VERTEX_BUDGET'),
    maxTriangles: integer(options.maxTriangles ?? DEFAULT_TRIANGLES, 1, HARD_TRIANGLES, 'TRIANGLE_BUDGET'),
    maxBytes: integer(options.maxBytes ?? DEFAULT_BYTES, 1, HARD_BYTES, 'BYTE_BUDGET'),
  };
}
function rawBytes(vertexCount: number, triangleCount: number, hasUvs: boolean, hasTangents: boolean,
  hasColors: boolean): number {
  return vertexCount * (80 + (hasUvs ? 8 : 0) + (hasTangents ? 16 : 0) + (hasColors ? 16 : 0)) + triangleCount * 12;
}
function flags(directory: GeometryStreamDirectory) {
  const hasColors = directory.hasColors === true, base = directory.vertexStride - (hasColors ? 16 : 0);
  return { hasUvs: base === 88 || base === 104,
    hasTangents: base === 96 || base === 104, hasColors };
}
function selectedCut(directory: GeometryStreamDirectory, maxError: number): number[] {
  const selected = new Set(directory.roots);
  const groups = directory.groups.map((_, id) => id).sort((a, b) =>
    directory.groups[b].level - directory.groups[a].level || a - b);
  for (const id of groups) {
    const group = directory.groups[id];
    if (group.terminal || !group.parents.every(parent => selected.has(parent))
      || !group.parents.some(parent => directory.clusters[parent].error > maxError)) continue;
    for (const parent of group.parents) selected.delete(parent);
    for (const child of group.children) selected.add(child);
  }
  return [...selected].sort((a, b) => a - b);
}
function metric(directory: GeometryStreamDirectory, ids: readonly number[]): number {
  return ids.reduce((max, id) => Math.max(max, directory.clusters[id].error), 0);
}
function digest(bytes: BufferSource): Promise<string> {
  return crypto.subtle.digest('SHA-256', bytes).then(hash =>
    Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join(''));
}
async function directoryDigest(directory: GeometryStreamDirectory): Promise<string> {
  return digest(new TextEncoder().encode(JSON.stringify(directory)));
}
function meshBounds(vertices: ArrayBuffer, count: number): Bounds {
  const view = new DataView(vertices);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let vertex = 0; vertex < count; vertex++) for (let word = 0; word < 20; word++) {
    if (word >= 8 && word <= 11) continue; // joints are integers
    const value = view.getFloat32(vertex * 80 + word * 4, true);
    if (!Number.isFinite(value)) fail('REST_FLOAT');
    if (word < 3) { min[word] = Math.min(min[word], value); max[word] = Math.max(max[word], value); }
  }
  return { min, max };
}
function freezeProxy(proxy: GeometryQueryProxy): GeometryQueryProxy {
  Object.freeze(proxy.selectedClusters);
  Object.freeze(proxy.mesh.bounds.min); Object.freeze(proxy.mesh.bounds.max); Object.freeze(proxy.mesh.bounds);
  Object.freeze(proxy.mesh);
  return Object.freeze(proxy);
}
type ProxyHeaderInput = Pick<GeometryQueryProxy, 'primitiveKey' | 'directorySha256' | 'topologyVersion'
  | 'requestedError' | 'approximateObjectSpaceError' | 'selectedClusters'> & {
    mesh: { vertexCount: number; indices: { length: number }; uvs?: unknown; tangents?: unknown;
      colors?: unknown; bounds: Bounds };
  };
function metadata(proxy: ProxyHeaderInput) {
  return { primitiveKey: proxy.primitiveKey, directorySha256: proxy.directorySha256,
    topologyVersion: proxy.topologyVersion, requestedError: proxy.requestedError,
    approximateObjectSpaceError: proxy.approximateObjectSpaceError, selectedClusters: proxy.selectedClusters,
    vertexCount: proxy.mesh.vertexCount, triangleCount: proxy.mesh.indices.length / 3,
    hasUvs: !!proxy.mesh.uvs, hasTangents: !!proxy.mesh.tangents,
    ...(proxy.mesh.colors ? { hasColors: true } : {}), bounds: proxy.mesh.bounds };
}
function wireBytes(proxy: GeometryQueryProxy): number {
  const header = new TextEncoder().encode(JSON.stringify(metadata(proxy)));
  if (header.byteLength > MAX_METADATA_BYTES) fail('METADATA_BUDGET');
  return HEADER_BYTES + header.byteLength + rawBytes(proxy.mesh.vertexCount,
    proxy.mesh.indices.length / 3, !!proxy.mesh.uvs, !!proxy.mesh.tangents, !!proxy.mesh.colors);
}

/** Validate asset identity, deterministic cut, geometry layout, and retained budgets. */
export async function validateGeometryQueryProxy(directoryInput: GeometryStreamDirectory,
  proxy: GeometryQueryProxy, options: GeometryQueryProxyOptions = {}): Promise<void> {
  const directory = validateGeometryStreamDirectory(directoryInput), budget = limits(options);
  if (!proxy || proxy.version !== 1 || proxy.primitiveKey !== directory.primitiveKey
    || proxy.directorySha256 !== await directoryDigest(directory)) fail('IDENTITY');
  integer(proxy.topologyVersion, 0, UINT32_MAX, 'TOPOLOGY_VERSION');
  finite(proxy.requestedError, 'MAX_ERROR');
  if (options.topologyVersion !== undefined && proxy.topologyVersion !== budget.topologyVersion) fail('TOPOLOGY_VERSION');
  if (proxy.requestedError > budget.maxError) fail('MAX_ERROR');
  if (!Array.isArray(proxy.selectedClusters)) fail('CUT');
  const expected = selectedCut(directory, proxy.requestedError);
  if (proxy.selectedClusters.length !== expected.length || expected.some((id, i) => proxy.selectedClusters[i] !== id)
    || proxy.approximateObjectSpaceError !== metric(directory, expected)) fail('CUT');
  const mesh = proxy.mesh;
  if (!mesh || !(mesh.vertices instanceof ArrayBuffer) || !(mesh.indices instanceof Uint32Array)) fail('MESH');
  const vertices = integer(mesh.vertexCount, 1, budget.maxVertices, 'VERTEX_BUDGET');
  const triangles = expected.reduce((sum, id) => sum + directory.clusters[id].triangleCount, 0);
  if (triangles > budget.maxTriangles || mesh.indices.length !== triangles * 3) fail('TRIANGLE_BUDGET');
  const { hasUvs, hasTangents, hasColors } = flags(directory);
  if (!!mesh.uvs !== hasUvs || !!mesh.tangents !== hasTangents || !!mesh.colors !== hasColors
    || mesh.vertices.byteLength !== vertices * 80
    || (mesh.uvs && (!(mesh.uvs instanceof Float32Array) || mesh.uvs.length !== vertices * 2 || !mesh.uvs.every(Number.isFinite)))
    || (mesh.tangents && (!(mesh.tangents instanceof Float32Array) || mesh.tangents.length !== vertices * 4 || !mesh.tangents.every(Number.isFinite)))
    || (mesh.colors && (!(mesh.colors instanceof Float32Array) || mesh.colors.length !== vertices * 4
      || !mesh.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1)))) fail('MESH_LAYOUT');
  if (mesh.indices.some(id => id >= vertices)) fail('MESH_INDEX');
  const computed = meshBounds(mesh.vertices, vertices);
  if (!mesh.bounds || !Array.isArray(mesh.bounds.min) || !Array.isArray(mesh.bounds.max)
    || mesh.bounds.min.length !== 3 || mesh.bounds.max.length !== 3
    || computed.min.some((value, axis) => mesh.bounds.min[axis] !== value || mesh.bounds.max[axis] !== computed.max[axis])) fail('BOUNDS');
  if (wireBytes(proxy) > budget.maxBytes) fail('BYTE_BUDGET');
}

/** Decode only selected, independently hashed pages; no full source mesh is held. */
export async function buildGeometryQueryProxy(directoryInput: GeometryStreamDirectory,
  provider: GeometryPageProvider, options: GeometryQueryProxyOptions = {}): Promise<GeometryQueryProxy> {
  const directory = validateGeometryStreamDirectory(directoryInput), budget = limits(options);
  if (typeof provider !== 'function') fail('PROVIDER');
  const selectedClusters = selectedCut(directory, budget.maxError);
  const triangles = selectedClusters.reduce((sum, id) => sum + directory.clusters[id].triangleCount, 0);
  if (triangles > budget.maxTriangles) fail('TRIANGLE_BUDGET');
  const { hasUvs, hasTangents, hasColors } = flags(directory);
  if (rawBytes(1, triangles, hasUvs, hasTangents, hasColors) + HEADER_BYTES > budget.maxBytes) fail('BYTE_BUDGET');
  const wanted = new Set(selectedClusters), pageIds = [...new Set(selectedClusters.map(id => directory.clusterPages[id]))].sort((a, b) => a - b);
  const sourceToProxy = new Map<number, number>();
  const records: Uint8Array[] = [], uvRecords: number[] = [], tangentRecords: number[] = [], colorRecords: number[] = [], indices: number[] = [];
  const sourceAttributes = new Map<number, { rest: Uint8Array; uv?: number[]; tangent?: number[]; color?: number[] }>();
  for (const pageId of pageIds) {
    const page = directory.pages[pageId], supplied = await provider(page, new AbortController().signal);
    if (!(supplied instanceof Uint8Array) || supplied.byteLength !== page.encodedBytes) fail('PAGE_BYTES');
    const owned = new Uint8Array(page.encodedBytes); owned.set(supplied);
    const decoded = await decodeVerifiedGeometryPage(page, owned);
    if (decoded.vertexStride !== directory.vertexStride || (decoded.hasColors === true) !== hasColors
      || decoded.sourceVertexCount !== directory.sourceVertexCount
      || decoded.hierarchyClusterCount !== directory.clusters.length || decoded.clusters.length !== page.clusterIds.length
      || decoded.clusters.some((cluster, i) => cluster.id !== page.clusterIds[i])) fail('PAGE_LAYOUT');
    for (const cluster of decoded.clusters) {
      if (!wanted.has(cluster.id)) continue;
      const expected = directory.clusters[cluster.id];
      if (cluster.sourceVertexIds.length !== expected.vertexCount || cluster.localTriangles.length !== expected.triangleCount * 3
        || !!cluster.uvs !== hasUvs || !!cluster.tangents !== hasTangents || !!cluster.colors !== hasColors) fail('CLUSTER_LAYOUT');
      const local = new Uint32Array(expected.vertexCount);
      for (let i = 0; i < expected.vertexCount; i++) {
        const id = cluster.sourceVertexIds[i], rest = cluster.rest.slice(i * 80, (i + 1) * 80);
        const uv = cluster.uvs ? Array.from(cluster.uvs.subarray(i * 2, i * 2 + 2)) : undefined;
        const tangent = cluster.tangents ? Array.from(cluster.tangents.subarray(i * 4, i * 4 + 4)) : undefined;
        const color = cluster.colors ? Array.from(cluster.colors.subarray(i * 4, i * 4 + 4)) : undefined;
        const prior = sourceAttributes.get(id);
        if (prior) {
          if (prior.rest.some((byte, offset) => byte !== rest[offset])
            || uv?.some((value, axis) => !Object.is(value, prior.uv?.[axis]))
            || tangent?.some((value, axis) => !Object.is(value, prior.tangent?.[axis]))
            || color?.some((value, axis) => !Object.is(value, prior.color?.[axis]))) fail('SEAM_ATTRIBUTES');
        } else {
          const next = sourceToProxy.size;
          if (next >= budget.maxVertices || rawBytes(next + 1, triangles, hasUvs, hasTangents, hasColors) + HEADER_BYTES > budget.maxBytes) fail('VERTEX_BUDGET');
          sourceToProxy.set(id, next); sourceAttributes.set(id, { rest, uv, tangent, color });
          records.push(rest); if (uv) uvRecords.push(...uv); if (tangent) tangentRecords.push(...tangent);
          if (color) colorRecords.push(...color);
        }
        local[i] = sourceToProxy.get(id)!;
      }
      for (const id of cluster.localTriangles) indices.push(local[id]);
    }
  }
  if (indices.length !== triangles * 3) fail('CUT_CONTENT');
  const bounds: Bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const record of records) {
    const view = new DataView(record.buffer, record.byteOffset, record.byteLength);
    for (let axis = 0; axis < 3; axis++) {
      const value = view.getFloat32(axis * 4, true);
      bounds.min[axis] = Math.min(bounds.min[axis], value);
      bounds.max[axis] = Math.max(bounds.max[axis], value);
    }
  }
  const directorySha256 = await directoryDigest(directory);
  const planned = { version: 1 as const, primitiveKey: directory.primitiveKey, directorySha256,
    topologyVersion: budget.topologyVersion, requestedError: budget.maxError,
    approximateObjectSpaceError: metric(directory, selectedClusters), selectedClusters,
    mesh: { vertexCount: records.length, indices: { length: triangles * 3 },
      uvs: hasUvs ? true : undefined, tangents: hasTangents ? true : undefined,
      colors: hasColors ? true : undefined, bounds } };
  const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata(planned))).byteLength;
  if (metadataBytes > MAX_METADATA_BYTES || HEADER_BYTES + metadataBytes
    + rawBytes(records.length, triangles, hasUvs, hasTangents, hasColors) > budget.maxBytes) fail('BYTE_BUDGET');
  const vertices = new ArrayBuffer(records.length * 80), raw = new Uint8Array(vertices);
  records.forEach((record, i) => raw.set(record, i * 80));
  const mesh: MeshData = { vertices, vertexCount: records.length, indices: Uint32Array.from(indices),
    ...(hasUvs ? { uvs: Float32Array.from(uvRecords) } : {}),
    ...(hasTangents ? { tangents: Float32Array.from(tangentRecords) } : {}),
    ...(hasColors ? { colors: Float32Array.from(colorRecords) } : {}),
    bounds };
  const proxy = freezeProxy({ version: 1, primitiveKey: directory.primitiveKey,
    directorySha256, topologyVersion: budget.topologyVersion,
    requestedError: budget.maxError, approximateObjectSpaceError: metric(directory, selectedClusters),
    selectedClusters, mesh });
  await validateGeometryQueryProxy(directory, proxy, options);
  return proxy;
}

/** Little-endian header, bounded JSON metadata, then rest/index/UV/tangent/color payloads. */
export function encodeGeometryQueryProxy(proxy: GeometryQueryProxy): Uint8Array<ArrayBuffer> {
  if (!proxy || proxy.version !== 1 || !proxy.mesh || !(proxy.mesh.vertices instanceof ArrayBuffer)
    || !(proxy.mesh.indices instanceof Uint32Array)) fail('MESH');
  if (typeof proxy.primitiveKey !== 'string' || !proxy.primitiveKey.length || proxy.primitiveKey.length > 1024
    || !/^[0-9a-f]{64}$/.test(proxy.directorySha256)) fail('IDENTITY');
  integer(proxy.topologyVersion, 0, UINT32_MAX, 'TOPOLOGY_VERSION');
  finite(proxy.requestedError, 'MAX_ERROR');
  if (!Number.isFinite(proxy.approximateObjectSpaceError) || proxy.approximateObjectSpaceError < 0) fail('ERROR_METRIC');
  if (!Array.isArray(proxy.selectedClusters) || !proxy.selectedClusters.length || proxy.selectedClusters.length > 65_536
    || proxy.selectedClusters.some((id, i) => !Number.isSafeInteger(id) || id < 0 || id >= 65_536
      || (i > 0 && id <= proxy.selectedClusters[i - 1]))) fail('CUT');
  const vertexCount = integer(proxy.mesh.vertexCount, 1, HARD_VERTICES, 'VERTEX_BUDGET');
  const triangleCount = integer(proxy.mesh.indices.length / 3, 1, HARD_TRIANGLES, 'TRIANGLE_BUDGET');
  if (proxy.mesh.vertices.byteLength !== vertexCount * 80 || proxy.mesh.indices.some(id => id >= vertexCount)
    || (proxy.mesh.uvs && (!(proxy.mesh.uvs instanceof Float32Array) || proxy.mesh.uvs.length !== vertexCount * 2 || !proxy.mesh.uvs.every(Number.isFinite)))
    || (proxy.mesh.tangents && (!(proxy.mesh.tangents instanceof Float32Array) || proxy.mesh.tangents.length !== vertexCount * 4 || !proxy.mesh.tangents.every(Number.isFinite)))
    || (proxy.mesh.colors && (!(proxy.mesh.colors instanceof Float32Array) || proxy.mesh.colors.length !== vertexCount * 4
      || !proxy.mesh.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1)))) fail('MESH_LAYOUT');
  const expectedBounds = meshBounds(proxy.mesh.vertices, vertexCount);
  if (!proxy.mesh.bounds || !Array.isArray(proxy.mesh.bounds.min) || !Array.isArray(proxy.mesh.bounds.max)
    || proxy.mesh.bounds.min.length !== 3 || proxy.mesh.bounds.max.length !== 3
    || expectedBounds.min.some((value, axis) => value !== proxy.mesh.bounds.min[axis]
      || expectedBounds.max[axis] !== proxy.mesh.bounds.max[axis])) fail('BOUNDS');
  const header = new TextEncoder().encode(JSON.stringify(metadata(proxy)));
  if (header.byteLength > MAX_METADATA_BYTES) fail('METADATA_BUDGET');
  const size = HEADER_BYTES + header.byteLength + rawBytes(proxy.mesh.vertexCount,
    proxy.mesh.indices.length / 3, !!proxy.mesh.uvs, !!proxy.mesh.tangents, !!proxy.mesh.colors);
  if (size > HARD_BYTES) fail('BYTE_BUDGET');
  const result = new Uint8Array(size), view = new DataView(result.buffer);
  view.setUint32(0, MAGIC, true); view.setUint32(4, 1, true);
  view.setUint32(8, header.byteLength, true); view.setUint32(12, size, true);
  result.set(header, HEADER_BYTES);
  let offset = HEADER_BYTES + header.byteLength;
  for (const stream of [new Uint8Array(proxy.mesh.vertices),
    new Uint8Array(proxy.mesh.indices.buffer, proxy.mesh.indices.byteOffset, proxy.mesh.indices.byteLength),
    ...(proxy.mesh.uvs ? [new Uint8Array(proxy.mesh.uvs.buffer, proxy.mesh.uvs.byteOffset, proxy.mesh.uvs.byteLength)] : []),
    ...(proxy.mesh.tangents ? [new Uint8Array(proxy.mesh.tangents.buffer, proxy.mesh.tangents.byteOffset, proxy.mesh.tangents.byteLength)] : []),
    ...(proxy.mesh.colors ? [new Uint8Array(proxy.mesh.colors.buffer, proxy.mesh.colors.byteOffset, proxy.mesh.colors.byteLength)] : [])]) {
    result.set(stream, offset); offset += stream.byteLength;
  }
  return result;
}

/** Verify caller-provided asset digest before parsing or allocating payloads. */
export async function decodeGeometryQueryProxy(directoryInput: GeometryStreamDirectory,
  bytes: Uint8Array<ArrayBuffer>, sha256: string, options: GeometryQueryProxyOptions = {}): Promise<GeometryQueryProxy> {
  const directory = validateGeometryStreamDirectory(directoryInput), budget = limits(options);
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < HEADER_BYTES || bytes.byteLength > budget.maxBytes
    || !/^[0-9a-f]{64}$/.test(sha256)) fail('ENCODING');
  const owned = bytes.slice();
  if (await digest(owned) !== sha256) fail('HASH');
  const view = new DataView(owned.buffer, owned.byteOffset, owned.byteLength);
  const length = view.getUint32(8, true);
  if (view.getUint32(0, true) !== MAGIC || view.getUint32(4, true) !== 1
    || view.getUint32(12, true) !== owned.byteLength || length < 2 || length > MAX_METADATA_BYTES
    || HEADER_BYTES + length > owned.byteLength) fail('HEADER');
  let meta: Record<string, unknown>;
  try { meta = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(owned.subarray(HEADER_BYTES, HEADER_BYTES + length))) as Record<string, unknown>; }
  catch { fail('METADATA'); }
  if (!meta || typeof meta !== 'object' || typeof meta.primitiveKey !== 'string' || meta.primitiveKey !== directory.primitiveKey
    || typeof meta.directorySha256 !== 'string' || meta.directorySha256 !== await directoryDigest(directory)) fail('IDENTITY');
  const vertexCount = integer(meta.vertexCount, 1, budget.maxVertices, 'VERTEX_BUDGET');
  const triangleCount = integer(meta.triangleCount, 1, budget.maxTriangles, 'TRIANGLE_BUDGET');
  const { hasUvs, hasTangents, hasColors } = flags(directory);
  if (meta.hasUvs !== hasUvs || meta.hasTangents !== hasTangents || (meta.hasColors === true) !== hasColors
    || (meta.hasColors !== undefined && meta.hasColors !== true)
    || HEADER_BYTES + length + rawBytes(vertexCount, triangleCount, hasUvs, hasTangents, hasColors) !== owned.byteLength) fail('MESH_LAYOUT');
  let offset = HEADER_BYTES + length;
  const copy = (size: number): ArrayBuffer => {
    const result = new ArrayBuffer(size);
    new Uint8Array(result).set(owned.subarray(offset, offset + size)); offset += size; return result;
  };
  const vertices = copy(vertexCount * 80), indices = new Uint32Array(copy(triangleCount * 12));
  const uvs = hasUvs ? new Float32Array(copy(vertexCount * 8)) : undefined;
  const tangents = hasTangents ? new Float32Array(copy(vertexCount * 16)) : undefined;
  const colors = hasColors ? new Float32Array(copy(vertexCount * 16)) : undefined;
  const proxy: GeometryQueryProxy = { version: 1, primitiveKey: meta.primitiveKey,
    directorySha256: meta.directorySha256, topologyVersion: meta.topologyVersion as number,
    requestedError: meta.requestedError as number,
    approximateObjectSpaceError: meta.approximateObjectSpaceError as number,
    selectedClusters: meta.selectedClusters as number[],
    mesh: { vertices, indices, vertexCount, ...(uvs ? { uvs } : {}), ...(tangents ? { tangents } : {}),
      ...(colors ? { colors } : {}), bounds: meta.bounds as Bounds } };
  await validateGeometryQueryProxy(directory, proxy, options);
  return freezeProxy(proxy);
}
