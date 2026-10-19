import { validateMeshColors, type MeshData } from './geometry';
import { validateMeshTangents } from './tangent-frame';

export interface GeometryCleanOptions {
  removeZeroAreaTriangles?: boolean;
  maxVertices?: number;
  maxTriangles?: number;
}

export interface CleanedStaticGeometry {
  mesh: MeshData;
  /** Output vertex index to input vertex index. */
  sourceVertices: Uint32Array;
  /** Output triangle index to input triangle index. */
  sourceTriangles: Uint32Array;
  /** Input triangle indices removed for exact zero area, in input order. */
  removedTriangles: Uint32Array;
}

const REST_BYTES = 80;
const DEFAULT_LIMIT = 1_000_000;
const HARD_LIMIT = 4_000_000;
const KNOWN_MESH_KEYS = new Set(['vertices', 'vertexCount', 'indices', 'bounds', 'uvs', 'tangents', 'colors']);

function budget(value: number | undefined, label: string): number {
  const result = value ?? DEFAULT_LIMIT;
  if (!Number.isSafeInteger(result) || result < 1 || result > HARD_LIMIT) {
    throw new RangeError(`GEOMETRY_CLEAN_${label}`);
  }
  return result;
}

function validateBounds(bounds: MeshData['bounds']): void {
  if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max)
    || bounds.min.length !== 3 || bounds.max.length !== 3) throw new RangeError('GEOMETRY_CLEAN_BOUNDS');
  for (let axis = 0; axis < 3; axis++) {
    if (!Number.isFinite(bounds.min[axis]) || !Number.isFinite(bounds.max[axis])
      || bounds.min[axis] > bounds.max[axis]) throw new RangeError('GEOMETRY_CLEAN_BOUNDS');
  }
}

/** Every finite f32 is an integer multiple of 2^-149. This fallback avoids a
 * rounded JS subtraction/cross product discarding a real, very small area. */
function exactZeroArea(view: DataView, a: number, b: number, c: number): boolean {
  const coordinate = (vertex: number, axis: number): bigint => {
    const bits = view.getUint32(vertex * REST_BYTES + axis * 4, true);
    const exponent = (bits >>> 23) & 255, fraction = bits & 0x7f_ffff;
    const magnitude = exponent === 0 ? BigInt(fraction)
      : BigInt(0x80_0000 | fraction) << BigInt(exponent - 1);
    return bits & 0x8000_0000 ? -magnitude : magnitude;
  };
  const ab = [0, 1, 2].map(axis => coordinate(b, axis) - coordinate(a, axis));
  const ac = [0, 1, 2].map(axis => coordinate(c, axis) - coordinate(a, axis));
  return ab[1] * ac[2] - ab[2] * ac[1] === 0n
    && ab[2] * ac[0] - ab[0] * ac[2] === 0n
    && ab[0] * ac[1] - ab[1] * ac[0] === 0n;
}

/** Prepare an undeformed static mesh for offline hierarchy building. Vertex and
 * triangle provenance remains explicit; no visual duplicates or wedges merge. */
export function cleanStaticGeometry(mesh: MeshData, options: GeometryCleanOptions = {}): CleanedStaticGeometry {
  if (!options || typeof options !== 'object' || Array.isArray(options)
    || (options.removeZeroAreaTriangles !== undefined && typeof options.removeZeroAreaTriangles !== 'boolean')) {
    throw new RangeError('GEOMETRY_CLEAN_OPTIONS');
  }
  const maxVertices = budget(options.maxVertices, 'MAX_VERTICES');
  const maxTriangles = budget(options.maxTriangles, 'MAX_TRIANGLES');
  if (!mesh || typeof mesh !== 'object') throw new RangeError('GEOMETRY_CLEAN_MESH');
  // A future MeshData extension needs an explicit compaction rule; dropping it
  // would silently detach its values from the surviving vertices or triangles.
  if (Reflect.ownKeys(mesh).some(key => typeof key !== 'string' || !KNOWN_MESH_KEYS.has(key))) {
    throw new RangeError('GEOMETRY_CLEAN_UNSUPPORTED_ATTRIBUTE');
  }
  if (!Number.isSafeInteger(mesh.vertexCount) || mesh.vertexCount < 1 || mesh.vertexCount > maxVertices
    || !(mesh.vertices instanceof ArrayBuffer) || mesh.vertices.byteLength !== mesh.vertexCount * REST_BYTES) {
    throw new RangeError('GEOMETRY_CLEAN_VERTEX_LAYOUT');
  }
  if (!(mesh.indices instanceof Uint32Array) || mesh.indices.length % 3 !== 0
    || mesh.indices.length / 3 > maxTriangles) throw new RangeError('GEOMETRY_CLEAN_TRIANGLE_LAYOUT');
  const triangleCount = mesh.indices.length / 3;
  if (triangleCount === 0) throw new RangeError('GEOMETRY_CLEAN_EMPTY');
  validateBounds(mesh.bounds);
  if (mesh.uvs !== undefined && (!(mesh.uvs instanceof Float32Array)
    || mesh.uvs.length !== mesh.vertexCount * 2 || !mesh.uvs.every(Number.isFinite))) {
    throw new RangeError('GEOMETRY_CLEAN_UVS');
  }
  validateMeshTangents(mesh.tangents, mesh.vertexCount);
  validateMeshColors(mesh.colors, mesh.vertexCount);

  const view = new DataView(mesh.vertices);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    for (let word = 0; word < REST_BYTES / 4; word++) {
      if (word >= 8 && word < 12) continue; // Joint IDs are uint32, not floats.
      if (!Number.isFinite(view.getFloat32(vertex * REST_BYTES + word * 4, true))) {
        throw new RangeError('GEOMETRY_CLEAN_NONFINITE_REST');
      }
    }
  }
  for (const index of mesh.indices) if (index >= mesh.vertexCount) throw new RangeError('GEOMETRY_CLEAN_INDEX');

  const kept = new Uint32Array(triangleCount), removed = new Uint32Array(triangleCount);
  let keptCount = 0, removedCount = 0;
  const coordinate = (id: number, axis: number): number => view.getFloat32(id * REST_BYTES + axis * 4, true);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const offset = triangle * 3, a = mesh.indices[offset], b = mesh.indices[offset + 1], c = mesh.indices[offset + 2];
    const abx = coordinate(b, 0) - coordinate(a, 0), aby = coordinate(b, 1) - coordinate(a, 1),
      abz = coordinate(b, 2) - coordinate(a, 2);
    const acx = coordinate(c, 0) - coordinate(a, 0), acy = coordinate(c, 1) - coordinate(a, 1),
      acz = coordinate(c, 2) - coordinate(a, 2);
    const zeroAreaCandidate = aby * acz - abz * acy === 0
      && abz * acx - abx * acz === 0
      && abx * acy - aby * acx === 0;
    const zeroArea = zeroAreaCandidate && exactZeroArea(view, a, b, c);
    if (zeroArea) {
      if (!options.removeZeroAreaTriangles) throw new RangeError('GEOMETRY_CLEAN_ZERO_AREA_TRIANGLE');
      removed[removedCount++] = triangle;
    } else kept[keptCount++] = triangle;
  }
  if (keptCount === 0) throw new RangeError('GEOMETRY_CLEAN_EMPTY');

  const remap = new Int32Array(mesh.vertexCount).fill(-1);
  const sourceIds = new Uint32Array(mesh.vertexCount), indices = new Uint32Array(keptCount * 3);
  let vertexCount = 0;
  for (let triangle = 0; triangle < keptCount; triangle++) {
    const original = kept[triangle] * 3;
    for (let corner = 0; corner < 3; corner++) {
      const source = mesh.indices[original + corner];
      let target = remap[source];
      if (target === -1) {
        target = vertexCount;
        remap[source] = target;
        sourceIds[vertexCount++] = source;
      }
      indices[triangle * 3 + corner] = target;
    }
  }
  const vertices = new ArrayBuffer(vertexCount * REST_BYTES), targetRest = new Uint8Array(vertices),
    sourceRest = new Uint8Array(mesh.vertices);
  const uvs = mesh.uvs ? new Float32Array(vertexCount * 2) : undefined;
  const tangents = mesh.tangents ? new Float32Array(vertexCount * 4) : undefined;
  const colors = mesh.colors ? new Float32Array(vertexCount * 4) : undefined;
  const bounds: MeshData['bounds'] = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let target = 0; target < vertexCount; target++) {
    const source = sourceIds[target];
    targetRest.set(sourceRest.subarray(source * REST_BYTES, (source + 1) * REST_BYTES), target * REST_BYTES);
    if (uvs) uvs.set(mesh.uvs!.subarray(source * 2, source * 2 + 2), target * 2);
    if (tangents) tangents.set(mesh.tangents!.subarray(source * 4, source * 4 + 4), target * 4);
    if (colors) colors.set(mesh.colors!.subarray(source * 4, source * 4 + 4), target * 4);
    for (let axis = 0; axis < 3; axis++) {
      const value = coordinate(source, axis);
      bounds.min[axis] = Math.min(bounds.min[axis], value);
      bounds.max[axis] = Math.max(bounds.max[axis], value);
    }
  }
  return { mesh: { vertices, vertexCount, indices, bounds,
    ...(uvs ? { uvs } : {}), ...(tangents ? { tangents } : {}), ...(colors ? { colors } : {}) },
  sourceVertices: sourceIds.slice(0, vertexCount), sourceTriangles: kept.slice(0, keptCount),
  removedTriangles: removed.slice(0, removedCount) };
}
