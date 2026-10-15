/** A tangent stores its direction in xyz and the signed orientation of cross(normal, tangent) in w. */
export type Tangent4 = readonly [number, number, number, number];

export function validateTangent(tangent: readonly number[], label = 'tangent'): void {
  if (tangent.length !== 4) throw new RangeError(`${label} requires four components`);
  for (let component = 0; component < 4; component++) {
    if (!Number.isFinite(tangent[component]) || !Number.isFinite(Math.fround(tangent[component]))) {
      throw new RangeError(`${label}[${component}] must be a finite f32`);
    }
  }
  if (tangent[3] !== 1 && tangent[3] !== -1) throw new RangeError(`${label}.w must be +1 or -1`);
}

/** Validate authored data once before upload; absence is a distinct derivative-fallback contract. */
export function validateMeshTangents(tangents: Float32Array | undefined, vertexCount: number): void {
  if (!Number.isSafeInteger(vertexCount) || vertexCount <= 0) throw new RangeError('vertexCount must be a positive integer');
  if (tangents === undefined) return;
  if (!(tangents instanceof Float32Array) || tangents.length !== vertexCount * 4) {
    throw new RangeError('mesh tangents must contain vertexCount packed xyzw components');
  }
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const offset = vertex * 4;
    validateTangent([tangents[offset], tangents[offset + 1], tangents[offset + 2], tangents[offset + 3]], `tangent[${vertex}]`);
  }
}

/** Detached upload snapshot. Missing tangents use w=0 so shaders select UV-derived frames. */
export function snapshotMeshTangents(tangents: Float32Array | undefined, vertexCount: number): Float32Array<ArrayBuffer> {
  validateMeshTangents(tangents, vertexCount);
  return tangents === undefined ? new Float32Array(vertexCount * 4) : new Float32Array(tangents);
}
