import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';
import type { GeometryStreamDirectory } from './geometry-stream-types';

export const GEOMETRY_GPU_LOCATION_BYTES = 32;

/** Fixed-slot allocation layout. Callers provide a validated directory. Both
 * the physical allocator and scene planner must charge the same padding. */
export function geometryGpuPoolLayout(directory: GeometryStreamDirectory) {
  let verticesPerSlot = 0, indicesPerSlot = 0;
  for (const page of directory.pages) {
    let vertices = 0, indices = 0;
    for (const id of page.clusterIds) { vertices += directory.clusters[id].vertexCount; indices += directory.clusters[id].triangleCount * 3; }
    verticesPerSlot = Math.max(verticesPerSlot, vertices); indicesPerSlot = Math.max(indicesPerSlot, indices);
  }
  const bytesPerSlot = verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES + indicesPerSlot * 4;
  const metadataBytes = directory.clusters.length * GEOMETRY_GPU_LOCATION_BYTES + directory.pages.length * 4;
  return { verticesPerSlot, indicesPerSlot, bytesPerSlot, metadataBytes,
    rootBytes: directory.rootPages.length * bytesPerSlot + metadataBytes,
    fullBytes: directory.pages.length * bytesPerSlot + metadataBytes };
}
