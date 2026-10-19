import type { Bounds3 } from './deformation';
import type { MeshData } from './geometry';

const MAX_VERTICES = 1_048_576;
const MAX_TRIANGLES = 1_048_576;
const VERTEX_BYTES = 80;

/**
 * A conservative local-space position change per UV0 unit for an immutable mesh.
 * Cache this asset-level result; an absent or singular UV map cannot prove that a
 * coarse mip is sufficient and therefore requests the finest available mip.
 */
export function meshWorldUnitsPerUv(mesh: MeshData): number {
  const count = mesh.vertexCount;
  if (!Number.isSafeInteger(count) || count < 1 || count > MAX_VERTICES
    || !(mesh.vertices instanceof ArrayBuffer) || mesh.vertices.byteLength !== count * VERTEX_BYTES
    || !(mesh.indices instanceof Uint32Array) || mesh.indices.length < 3
    || mesh.indices.length % 3 !== 0 || mesh.indices.length / 3 > MAX_TRIANGLES) {
    throw new RangeError('invalid or oversized mesh for texture demand');
  }
  const uvs = mesh.uvs;
  if (uvs !== undefined && (!(uvs instanceof Float32Array) || uvs.length !== count * 2)) {
    throw new RangeError('invalid UV0 stream for texture demand');
  }
  const positions = new DataView(mesh.vertices);
  const indices = mesh.indices;
  for (let vertex = 0; vertex < count; vertex++) {
    for (let axis = 0; axis < 3; axis++) {
      if (!Number.isFinite(positions.getFloat32(vertex * VERTEX_BYTES + axis * 4, true))) return Infinity;
    }
  }
  for (let index = 0; index < indices.length; index++) {
    if (indices[index] >= count) throw new RangeError('invalid mesh index for texture demand');
  }
  if (uvs === undefined) return Infinity;
  let bound = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
    const u0 = uvs[2 * i0], v0 = uvs[2 * i0 + 1];
    const du1 = uvs[2 * i1] - u0, dv1 = uvs[2 * i1 + 1] - v0;
    const du2 = uvs[2 * i2] - u0, dv2 = uvs[2 * i2 + 1] - v0;
    if (![u0, v0, du1, dv1, du2, dv2].every(Number.isFinite)) return Infinity;
    const det = du1 * dv2 - dv1 * du2;
    if (!Number.isFinite(det) || det === 0) return Infinity;
    let duSquared = 0, dvSquared = 0;
    const edge1: number[] = [], edge2: number[] = [];
    for (let axis = 0; axis < 3; axis++) {
      const p0 = positions.getFloat32(i0 * VERTEX_BYTES + axis * 4, true);
      const p1 = positions.getFloat32(i1 * VERTEX_BYTES + axis * 4, true);
      const p2 = positions.getFloat32(i2 * VERTEX_BYTES + axis * 4, true);
      if (![p0, p1, p2].every(Number.isFinite)) return Infinity;
      const e1 = p1 - p0, e2 = p2 - p0;
      edge1.push(e1); edge2.push(e2);
      duSquared += ((e1 * dv2 - e2 * dv1) / det) ** 2;
      dvSquared += ((e2 * du1 - e1 * du2) / det) ** 2;
    }
    const crossX = edge1[1] * edge2[2] - edge1[2] * edge2[1];
    const crossY = edge1[2] * edge2[0] - edge1[0] * edge2[2];
    const crossZ = edge1[0] * edge2[1] - edge1[1] * edge2[0];
    if (Math.hypot(crossX, crossY, crossZ) === 0) return Infinity;
    const magnitude = Math.sqrt(duSquared + dvSquared);
    if (!Number.isFinite(magnitude)) return Infinity;
    bound = Math.max(bound, magnitude);
  }
  return bound;
}

/** Upper bound on the stretch of an affine model's 3x3 linear part. */
export function modelStretch(model: ArrayLike<number>): number {
  if (model.length !== 16 || !Array.from(model).every(Number.isFinite)) {
    throw new RangeError('model must be a finite 4x4 matrix');
  }
  let norm1 = 0, normInf = 0;
  for (let col = 0; col < 3; col++) {
    let sum = 0;
    for (let row = 0; row < 3; row++) sum += Math.abs(model[col * 4 + row]);
    norm1 = Math.max(norm1, sum);
  }
  for (let row = 0; row < 3; row++) {
    let sum = 0;
    for (let col = 0; col < 3; col++) sum += Math.abs(model[col * 4 + row]);
    normInf = Math.max(normInf, sum);
  }
  return Math.sqrt(norm1) * Math.sqrt(normInf);
}

export interface TextureViewDemandArgs {
  bounds: Bounds3;
  viewProjection: ArrayLike<number>;
  eye: readonly number[];
  forward: readonly number[];
  focalPixels: number;
  worldUnitsPerUv: number;
  width: number;
  height: number;
  maxMip: number;
}

/** Conservative camera demand for one world-space AABB and static UV scale. */
export function textureViewDemand(args: TextureViewDemandArgs): { mip: number; priority: number; visible: boolean } {
  const { bounds, viewProjection: m, eye, forward, focalPixels, worldUnitsPerUv, width, height, maxMip } = args;
  if (m.length !== 16 || !Array.from(m).every(Number.isFinite)
    || eye.length !== 3 || forward.length !== 3 || !eye.every(Number.isFinite) || !forward.every(Number.isFinite)
    || bounds.min.length !== 3 || bounds.max.length !== 3
    || !bounds.min.every(Number.isFinite) || !bounds.max.every(Number.isFinite)
    || bounds.min.some((value, axis) => value > bounds.max[axis])
    || !Number.isFinite(focalPixels) || focalPixels <= 0
    || !Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0
    || !(worldUnitsPerUv >= 0) || Number.isNaN(worldUnitsPerUv)
    || !Number.isInteger(maxMip) || maxMip < 0 || maxMip > 31) {
    throw new RangeError('invalid texture view demand input');
  }
  const forwardLength = Math.hypot(...forward);
  if (!Number.isFinite(forwardLength) || forwardLength === 0) throw new RangeError('forward must be nonzero');
  const outside = [true, true, true, true, true, true];
  let nearCrossing = false, minDepth = Infinity, maxLateral = 0;
  for (let corner = 0; corner < 8; corner++) {
    const x = bounds[corner & 1 ? 'max' : 'min'][0];
    const y = bounds[corner & 2 ? 'max' : 'min'][1];
    const z = bounds[corner & 4 ? 'max' : 'min'][2];
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cz) || !Number.isFinite(cw)) {
      return { mip: 0, priority: 1e9, visible: true };
    }
    if (cx >= -cw) outside[0] = false;
    if (cx <= cw) outside[1] = false;
    if (cy >= -cw) outside[2] = false;
    if (cy <= cw) outside[3] = false;
    if (cz >= 0) outside[4] = false;
    if (cz <= cw) outside[5] = false;
    if (cz <= 0) nearCrossing = true;
    const fromEyeX = x - eye[0], fromEyeY = y - eye[1], fromEyeZ = z - eye[2];
    const depth = (fromEyeX * forward[0] + fromEyeY * forward[1] + fromEyeZ * forward[2]) / forwardLength;
    if (!Number.isFinite(depth) || depth <= 0) nearCrossing = true;
    else minDepth = Math.min(minDepth, depth);
    const lateral = Math.hypot(
      fromEyeY * forward[2] - fromEyeZ * forward[1],
      fromEyeZ * forward[0] - fromEyeX * forward[2],
      fromEyeX * forward[1] - fromEyeY * forward[0],
    ) / forwardLength;
    if (!Number.isFinite(lateral)) nearCrossing = true;
    else maxLateral = Math.max(maxLateral, lateral);
  }
  if (outside.some(Boolean)) return { mip: maxMip, priority: 0, visible: false };
  const depth = nearCrossing ? 0 : minDepth;
  const halfX = (bounds.max[0] - bounds.min[0]) / 2;
  const halfY = (bounds.max[1] - bounds.min[1]) / 2;
  const halfZ = (bounds.max[2] - bounds.min[2]) / 2;
  const radius = Math.hypot(halfX, halfY, halfZ);
  const projectedExtent = focalPixels * Math.max(radius, 1e-6) / Math.max(depth, 1e-6);
  const priority = Number.isFinite(projectedExtent) ? Math.min(1e9, projectedExtent) : 1e9;
  if (nearCrossing || worldUnitsPerUv === Infinity) return { mip: 0, priority, visible: true };
  // The perspective Jacobian has norm (f / depth) * sqrt(1 + (lateral / depth)^2).
  // Max lateral and min depth over the box bound every visible point, including
  // points on clipped edges between corners.
  const perspectiveStretch = Math.hypot(1, maxLateral / depth);
  const ratio = Math.min(width, height) * depth / (focalPixels * worldUnitsPerUv * perspectiveStretch);
  const mip = ratio === Infinity ? maxMip : ratio <= 1 || Number.isNaN(ratio) ? 0
    : Math.min(maxMip, Math.max(0, Math.floor(Math.log2(ratio))));
  return { mip, priority, visible: true };
}
