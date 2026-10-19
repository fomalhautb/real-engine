import type { DecodedGeometryPage } from './geometry-pages';
import type { GeometryStreamDirectory } from './geometry-stream-types';

/** Physical upload record: rest[0..79], UV[80..87], flags[88], padding[92],
 * tangent[96..111], source/cluster/level/debug identity[112..127], linear RGBA[128..143]. */
export const GEOMETRY_GPU_VERTEX_BYTES = 144;

export interface PackedGeometryGpuPage {
  vertices: ArrayBuffer;
  indices: Uint32Array<ArrayBuffer>;
  clusters: { id: number; vertexOffset: number; vertexCount: number; indexOffset: number; indexCount: number }[];
}

const REST_BYTES = 80;
const MAX_DECODED_PAGE_BYTES = 4 * 1024 * 1024;
const MAX_UPLOAD_BYTES = 16 * 1024 * 1024;

function fail(reason: string): never { throw new RangeError(`GEOMETRY_GPU_PAGE_${reason}`); }

/** Same FNV-1a identity used by packGeometryLod for the cluster debug color. */
function stableSeed(primitiveKey: string, clusterId: number): number {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(primitiveKey)) hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
  hash = Math.imul(hash, 0x01000193) >>> 0;
  for (let shift = 0; shift < 32; shift += 8) hash = Math.imul(hash ^ ((clusterId >>> shift) & 255), 0x01000193) >>> 0;
  return hash;
}

/** Pack one already validated directory page into bounded physical GPU upload data.
 * The decoded page is checked again because callers may provide mutable or spoofed data. */
export function packGeometryGpuPage(
  directory: GeometryStreamDirectory, pageId: number, decoded: DecodedGeometryPage,
): PackedGeometryGpuPage {
  if (!Number.isSafeInteger(pageId) || pageId < 0 || pageId >= directory.pages.length) fail('PAGE_ID');
  const page = directory.pages[pageId];
  const hasUvs = decoded?.hasUvs === true, hasTangents = decoded?.hasTangents === true;
  const hasColors = decoded?.hasColors === true;
  const stride = REST_BYTES + (hasUvs ? 8 : 0) + (hasTangents ? 16 : 0) + (hasColors ? 16 : 0);
  if (!decoded || decoded.version !== 1 || typeof decoded.hasUvs !== 'boolean'
    || typeof decoded.hasTangents !== 'boolean' || (decoded.hasColors !== undefined && typeof decoded.hasColors !== 'boolean')
    || hasColors !== (directory.hasColors === true) || decoded.vertexStride !== stride
    || stride !== directory.vertexStride || decoded.sourceVertexCount !== directory.sourceVertexCount
    || decoded.hierarchyClusterCount !== directory.clusters.length
    || !Array.isArray(decoded.clusters) || decoded.clusters.length !== page.clusterIds.length) fail('IDENTITY');
  if (page.decodedBytes > MAX_DECODED_PAGE_BYTES) fail('PAGE_BUDGET');

  // The trusted directory determines every allocation and offset. Check all mutable
  // decoded arrays and values before allocating the expanded upload buffers.
  let vertexCount = 0, indexCount = 0, decodedBytes = 0;
  for (let ordinal = 0; ordinal < page.clusterIds.length; ordinal++) {
    const id = page.clusterIds[ordinal], metadata = directory.clusters[id], cluster = decoded.clusters[ordinal];
    const expectedIndices = metadata.triangleCount * 3;
    if (!cluster || cluster.id !== id || !(cluster.sourceVertexIds instanceof Uint32Array)
      || !(cluster.rest instanceof Uint8Array) || !(cluster.localTriangles instanceof Uint8Array)
      || cluster.sourceVertexIds.length !== metadata.vertexCount
      || cluster.rest.length !== metadata.vertexCount * REST_BYTES
      || cluster.localTriangles.length !== expectedIndices
      || (hasUvs ? !(cluster.uvs instanceof Float32Array) || cluster.uvs.length !== metadata.vertexCount * 2
        : cluster.uvs !== undefined)
      || (hasTangents ? !(cluster.tangents instanceof Float32Array) || cluster.tangents.length !== metadata.vertexCount * 4
        : cluster.tangents !== undefined)
      || (hasColors ? !(cluster.colors instanceof Float32Array) || cluster.colors.length !== metadata.vertexCount * 4
        : cluster.colors !== undefined)) fail('CLUSTER_LAYOUT');
    decodedBytes += 16 + metadata.vertexCount * (4 + stride) + ((expectedIndices + 3) & ~3);
    vertexCount += metadata.vertexCount;
    indexCount += expectedIndices;
    const rest = new DataView(cluster.rest.buffer, cluster.rest.byteOffset, cluster.rest.byteLength);
    const sourceIds = new Set<number>();
    for (let vertex = 0; vertex < metadata.vertexCount; vertex++) {
      const sourceId = cluster.sourceVertexIds[vertex];
      if (sourceId >= directory.sourceVertexCount || sourceIds.has(sourceId)) fail('SOURCE_ID');
      sourceIds.add(sourceId);
      const base = vertex * REST_BYTES;
      for (let word = 0; word < 20; word++) {
        if (word >= 8 && word <= 11) continue; // Joint IDs are raw u32 values.
        const value = rest.getFloat32(base + word * 4, true);
        if (!Number.isFinite(value)) fail('REST_FLOAT');
        if (word < 3 && (value < metadata.bounds.min[word] || value > metadata.bounds.max[word])) fail('BOUNDS');
      }
      if (cluster.uvs) for (let axis = 0; axis < 2; axis++) {
        if (!Number.isFinite(cluster.uvs[vertex * 2 + axis])) fail('UV');
      }
      if (cluster.tangents) for (let axis = 0; axis < 4; axis++) {
        if (!Number.isFinite(cluster.tangents[vertex * 4 + axis])) fail('TANGENT');
      }
      if (cluster.colors) for (let axis = 0; axis < 4; axis++) {
        if (!Number.isFinite(cluster.colors[vertex * 4 + axis]) || cluster.colors[vertex * 4 + axis] < 0
          || cluster.colors[vertex * 4 + axis] > 1) fail('COLOR');
      }
    }
    for (const localIndex of cluster.localTriangles) {
      if (localIndex >= metadata.vertexCount) fail('INDEX');
    }
  }
  if (decodedBytes !== page.decodedBytes) fail('DECODED_BYTES');
  if (vertexCount * GEOMETRY_GPU_VERTEX_BYTES + indexCount * 4 > MAX_UPLOAD_BYTES) fail('UPLOAD_BUDGET');

  const vertices = new ArrayBuffer(vertexCount * GEOMETRY_GPU_VERTEX_BYTES);
  const vertexBytes = new Uint8Array(vertices), vertexView = new DataView(vertices);
  const indices = new Uint32Array(indexCount);
  const clusters: PackedGeometryGpuPage['clusters'] = [];
  const flags = (hasUvs ? 1 : 0) | (hasTangents ? 2 : 0) | (hasColors ? 4 : 0);
  let vertexOffset = 0, indexOffset = 0;
  for (const cluster of decoded.clusters) {
    const metadata = directory.clusters[cluster.id], seed = stableSeed(directory.primitiveKey, cluster.id);
    const count = metadata.vertexCount, localIndices = cluster.localTriangles.length;
    clusters.push({ id: cluster.id, vertexOffset, vertexCount: count, indexOffset, indexCount: localIndices });
    for (let local = 0; local < count; local++) {
      const offset = (vertexOffset + local) * GEOMETRY_GPU_VERTEX_BYTES;
      vertexBytes.set(cluster.rest.subarray(local * REST_BYTES, (local + 1) * REST_BYTES), offset);
      if (cluster.uvs) for (let axis = 0; axis < 2; axis++) {
        vertexView.setFloat32(offset + 80 + axis * 4, cluster.uvs[local * 2 + axis], true);
      }
      vertexView.setUint32(offset + 88, flags, true);
      if (cluster.tangents) for (let axis = 0; axis < 4; axis++) {
        vertexView.setFloat32(offset + 96 + axis * 4, cluster.tangents[local * 4 + axis], true);
      }
      vertexView.setUint32(offset + 112, cluster.sourceVertexIds[local], true);
      vertexView.setUint32(offset + 116, cluster.id, true);
      vertexView.setUint32(offset + 120, metadata.level, true);
      vertexView.setUint32(offset + 124, seed, true);
      for (let axis = 0; axis < 4; axis++) {
        vertexView.setFloat32(offset + 128 + axis * 4, cluster.colors?.[local * 4 + axis] ?? 1, true);
      }
    }
    for (let local = 0; local < localIndices; local++) indices[indexOffset + local] = vertexOffset + cluster.localTriangles[local];
    vertexOffset += count;
    indexOffset += localIndices;
  }
  return { vertices, indices, clusters };
}
