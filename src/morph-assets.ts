import type { MeshData } from './geometry';
import type { MorphTarget } from './deformation';

/** Asset-build operation. Reconstruct the target normals using the mesh's
 * authored vertex splits; never smooth across distinct hard-edge vertices. */
export function buildMorphTarget(mesh: MeshData, displace: (position: [number, number, number], index: number) => [number, number, number]): MorphTarget {
  const positions = new Float32Array(mesh.vertexCount * 3);
  const normals = new Float32Array(mesh.vertexCount * 3);
  const target = new Float32Array(mesh.vertexCount * 3);
  const source = new DataView(mesh.vertices);
  for (let i = 0; i < mesh.vertexCount; i++) {
    const p: [number, number, number] = [0, 1, 2].map(axis => source.getFloat32(i * 80 + axis * 4, true)) as [number, number, number];
    const delta = displace(p, i);
    if (!delta.every(Number.isFinite)) throw new Error('MORPH_ASSET: nonfinite displacement');
    for (let axis = 0; axis < 3; axis++) {
      positions[i * 3 + axis] = delta[axis]; target[i * 3 + axis] = p[axis] + delta[axis];
    }
  }
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const [a, b, c] = [mesh.indices[i] * 3, mesh.indices[i + 1] * 3, mesh.indices[i + 2] * 3];
    const ux = target[b] - target[a], uy = target[b + 1] - target[a + 1], uz = target[b + 2] - target[a + 2];
    const vx = target[c] - target[a], vy = target[c + 1] - target[a + 1], vz = target[c + 2] - target[a + 2];
    const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
    for (const vertex of [a, b, c]) for (let axis = 0; axis < 3; axis++) normals[vertex + axis] += n[axis];
  }
  for (let i = 0; i < mesh.vertexCount; i++) {
    const length = Math.hypot(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
    if (length < 1e-10) throw new Error('MORPH_ASSET: target has degenerate vertex normal');
    for (let axis = 0; axis < 3; axis++) normals[i * 3 + axis] = normals[i * 3 + axis] / length - source.getFloat32(i * 80 + 16 + axis * 4, true);
  }
  return { positions, normals };
}

export function authoredMorph(mesh: MeshData): MorphTarget {
  const source = new DataView(mesh.vertices);
  return buildMorphTarget(mesh, (_, i) => [0, 1, 2].map(axis => source.getFloat32(i * 80 + 64 + axis * 4, true)) as [number, number, number]);
}
