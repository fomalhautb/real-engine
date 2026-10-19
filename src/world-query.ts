/** CPU-side, renderer-independent wire contract for bounded world-space ray queries. */
export interface WorldQueryOptions {
  maxRays?: number;
  maxBytes?: number;
  maxTrianglesPerMesh?: number;
  maxInstancesPerBatch?: number;
  /** Historical name: counts instance, node, and triangle work units per ray. */
  maxNodeVisits?: number;
  /** Shared cap across every ray and batch in one query. */
  maxTotalWorkUnits?: number;
  /** Distinct count/textureLod parameter snapshots retained within one GPU encoder; identical snapshots reuse a slot. */
  maxTraceParameterSetsPerEncoder?: number;
  /** Batch-local instance BVH for capacities >= 8; linear retains the reference traversal. */
  instanceAcceleration?: 'bvh' | 'linear';
  /** GPU batch candidates from a scene BVH when at least 8 batches are registered. */
  sceneAcceleration?: 'bvh' | 'linear';
}

export interface WorldRay {
  origin: readonly [number, number, number];
  direction: readonly [number, number, number];
  tMin?: number;
  tMax?: number;
  ignore?: { entityId: number; generation?: number };
}

/** Triangle indices are local to the named representation, never interchangeable. */
export type WorldHitGeometry = { kind: 'source' } | {
  kind: 'fixed-query-proxy';
  primitiveKey: string;
  directorySha256: string;
  topologyVersion: number;
  approximateObjectSpaceError: number;
  errorMetric: 'accumulated-simplification';
};

export interface WorldHit {
  batch: string;
  geometry: WorldHitGeometry;
  entityId: number;
  entityGeneration: number;
  /** Index into geometry's triangle namespace. Proxy indices are not source IDs. */
  triangleIndex: number;
  distance: number;
  position: [number, number, number];
  normal: [number, number, number];
  /** Oriented triangle normal; independent of smoothed vertices and material detail. */
  geometricNormal: [number, number, number];
  previousPosition: [number, number, number];
  uv: [number, number];
  baseColor: [number, number, number];
  opacity: number;
  metallic: number;
  roughness: number;
  emissive: [number, number, number];
  shading: 'pbr' | 'unlit';
  historyVersion: number;
  /** Caller queryIdentity version; proxy topology version is in geometry. */
  topologyVersion: number;
}

export interface WorldQueryResult {
  /** Most recently committed geometry frame used by the query. */
  frame: number;
  /** Material version atomically published for this query; parameter changes can advance it without another render. */
  materialRevision: number;
  hits: (WorldHit | null)[];
}

export const WORLD_RAY_BYTES = 48;
export const WORLD_HIT_BYTES = 160;
// Geometry normal is octahedrally encoded in the two formerly reserved f32s:
// position.w (word 11) and uvInfo.w (word 35). No hit stride/budget increase.

const UINT32_MAX = 0xffff_ffff;
const MAX_RAYS = 65_536;
type NumericOptions = Required<Omit<WorldQueryOptions, 'instanceAcceleration' | 'sceneAcceleration'>>;
const DEFAULT_OPTIONS: NumericOptions = {
  maxRays: 1_024,
  maxBytes: 128 * 1024 * 1024,
  maxTrianglesPerMesh: 65_536,
  maxInstancesPerBatch: 256,
  maxNodeVisits: 262_144,
  maxTotalWorkUnits: 16_777_216,
  maxTraceParameterSetsPerEncoder: 8,
};
const HARD_OPTIONS: NumericOptions = {
  maxRays: MAX_RAYS,
  maxBytes: 512 * 1024 * 1024,
  maxTrianglesPerMesh: 262_144,
  maxInstancesPerBatch: 4_096,
  maxNodeVisits: 1_048_576,
  maxTotalWorkUnits: 67_108_864,
  maxTraceParameterSetsPerEncoder: 32,
};

function boundedInteger(value: number, label: string, min: number, max: number): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${label} must be an integer from ${min} to ${max}`);
  }
  return value;
}

function uint32(value: number, label: string): number {
  return boundedInteger(value, label, 0, UINT32_MAX);
}

/** Defaults and hard caps apply before any CPU or GPU allocation. */
export function resolveWorldQueryOptions(options: WorldQueryOptions = {}): Required<WorldQueryOptions> {
  const result = {} as Required<WorldQueryOptions>;
  for (const key of Object.keys(DEFAULT_OPTIONS) as (keyof NumericOptions)[]) {
    result[key] = boundedInteger(options[key] ?? DEFAULT_OPTIONS[key], key, 1, HARD_OPTIONS[key]);
  }
  const acceleration = options.instanceAcceleration ?? 'bvh';
  if (acceleration !== 'bvh' && acceleration !== 'linear') throw new RangeError('instanceAcceleration must be bvh or linear');
  result.instanceAcceleration = acceleration;
  const sceneAcceleration = options.sceneAcceleration ?? 'bvh';
  if (sceneAcceleration !== 'bvh' && sceneAcceleration !== 'linear') throw new RangeError('sceneAcceleration must be bvh or linear');
  result.sceneAcceleration = sceneAcceleration;
  return result;
}

function float32(value: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite`);
  }
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded) || (value !== 0 && rounded === 0)) {
    throw new RangeError(`${label} cannot be represented by float32`);
  }
  return rounded;
}

function vector3(value: readonly [number, number, number], label: string): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new RangeError(`${label} must contain three numbers`);
  }
  return [float32(value[0], `${label}[0]`), float32(value[1], `${label}[1]`), float32(value[2], `${label}[2]`)];
}

/** Layout: three vec4 values, little-endian. Direction is unit length before float32 storage. */
export function packWorldRays(rays: readonly WorldRay[], maxRays: number): ArrayBuffer {
  boundedInteger(maxRays, 'maxRays', 1, MAX_RAYS);
  if (!Array.isArray(rays) || rays.length > maxRays) {
    throw new RangeError('WORLD_QUERY_RAY_BUDGET: ray count exceeds maxRays');
  }
  const data = new ArrayBuffer(rays.length * WORLD_RAY_BYTES);
  const view = new DataView(data);
  for (let i = 0; i < rays.length; i++) {
    const ray = rays[i];
    if (ray === null || typeof ray !== 'object') throw new RangeError(`rays[${i}] must be a ray`);
    const origin = vector3(ray.origin, `rays[${i}].origin`);
    const direction = vector3(ray.direction, `rays[${i}].direction`);
    const length = Math.hypot(...direction);
    if (!(length > 0) || !Number.isFinite(length)) {
      throw new RangeError(`rays[${i}].direction must be nonzero`);
    }
    const normal = direction.map((component) => float32(component / length, `rays[${i}].direction`));
    const tMin = float32(ray.tMin ?? 0.0001, `rays[${i}].tMin`);
    const tMax = float32(ray.tMax ?? 1e30, `rays[${i}].tMax`);
    if (tMin < 0 || tMax <= tMin) {
      throw new RangeError(`rays[${i}] requires 0 <= tMin < tMax after float32 conversion`);
    }
    const ignoreEnabled = ray.ignore !== undefined;
    const ignoreId = ignoreEnabled ? uint32(ray.ignore!.entityId, `rays[${i}].ignore.entityId`) : 0;
    const generation = ignoreEnabled ? uint32(ray.ignore!.generation ?? 0, `rays[${i}].ignore.generation`) : 0;
    const byte = i * WORLD_RAY_BYTES;
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(byte + axis * 4, origin[axis], true);
      view.setFloat32(byte + 16 + axis * 4, normal[axis], true);
    }
    view.setFloat32(byte + 12, tMin, true);
    view.setFloat32(byte + 28, tMax, true);
    view.setUint32(byte + 32, ignoreId, true);
    view.setUint32(byte + 36, generation, true);
    view.setUint32(byte + 40, ignoreEnabled ? 1 : 0, true);
    view.setUint32(byte + 44, 0, true);
  }
  return data;
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
}

function unitInterval(value: number, label: string): number {
  if (value < 0 || value > 1) throw new RangeError(`${label} must be in [0, 1]`);
  return value;
}

/** Decode complete, version-matched GPU records; a zero batch ID is the only miss marker. */
export function decodeWorldHits(
  data: ArrayBuffer,
  batchNames: readonly string[],
  count: number,
  frame: number,
  materialRevision: number,
  batchGeometry?: readonly WorldHitGeometry[],
): WorldQueryResult {
  boundedInteger(count, 'count', 0, MAX_RAYS);
  uint32(frame, 'frame');
  uint32(materialRevision, 'materialRevision');
  if (data.byteLength !== count * WORLD_HIT_BYTES) {
    throw new RangeError('World hit byte length does not match count');
  }
  if (batchGeometry && (!Array.isArray(batchGeometry) || batchGeometry.length !== batchNames.length)) {
    throw new RangeError('World hit geometry metadata must match batch names');
  }
  const view = new DataView(data);
  const hits: (WorldHit | null)[] = [];
  for (let i = 0; i < count; i++) {
    const byte = i * WORLD_HIT_BYTES;
    const u32 = (word: number): number => view.getUint32(byte + word * 4, true);
    const f32 = (word: number): number => finite(view, byte + word * 4, `hits[${i}] word ${word}`);
    const batchId = u32(0);
    const entityId = u32(1);
    const generation = u32(2);
    const triangleIndex = u32(3);
    const distance = f32(4);
    const baryU = f32(5);
    const baryV = f32(6);
    const frontFace = f32(7);
    const position: [number, number, number] = [f32(8), f32(9), f32(10)];
    const geometryX = f32(11);
    const normal: [number, number, number] = [f32(12), f32(13), f32(14)];
    const roughness = f32(15);
    const previousPosition: [number, number, number] = [f32(16), f32(17), f32(18)];
    const metallic = f32(19);
    const baseColor: [number, number, number] = [f32(20), f32(21), f32(22)];
    const opacity = f32(23);
    const emissive: [number, number, number] = [f32(24), f32(25), f32(26)];
    const isUnlit = f32(27);
    const recordFrame = u32(28);
    const historyVersion = u32(29);
    const topologyVersion = u32(30);
    const recordMaterialRevision = u32(31);
    const uv: [number, number] = [f32(32), f32(33)];
    f32(34);
    const geometryY = f32(35);
    const overflow = u32(38);
    if (overflow !== 0) throw new RangeError(`WORLD_QUERY_WORK_BUDGET: hits[${i}] exceeded traversal budget`);
    if (recordFrame !== frame || recordMaterialRevision !== materialRevision) {
      throw new RangeError(`hits[${i}] has a stale frame or material revision`);
    }
    if (batchId === 0) {
      if (entityId !== 0 || generation !== 0 || triangleIndex !== 0) {
        throw new RangeError(`hits[${i}] has invalid miss identity`);
      }
      hits.push(null);
      continue;
    }
    if (batchId > batchNames.length || typeof batchNames[batchId - 1] !== 'string') {
      throw new RangeError(`hits[${i}] batch ID is out of range`);
    }
    if (!(distance >= 0) || (baryU < 0 || baryV < 0 || baryU + baryV > 1.000001)
      || (frontFace !== 0 && frontFace !== 1) || (isUnlit !== 0 && isUnlit !== 1)) {
      throw new RangeError(`hits[${i}] has invalid intersection data`);
    }
    unitInterval(roughness, `hits[${i}].roughness`);
    unitInterval(metallic, `hits[${i}].metallic`);
    unitInterval(opacity, `hits[${i}].opacity`);
    for (const component of baseColor) {
      if (isUnlit === 0) unitInterval(component, `hits[${i}].baseColor`);
      else if (component < 0) throw new RangeError(`hits[${i}].baseColor must be nonnegative`);
    }
    for (const component of emissive) {
      if (component < 0) throw new RangeError(`hits[${i}].emissive must be nonnegative`);
    }
    if (Math.abs(Math.hypot(...normal) - 1) > 0.01) {
      throw new RangeError(`hits[${i}].normal must have unit length`);
    }
    if (Math.abs(geometryX) > 1 || Math.abs(geometryY) > 1) {
      throw new RangeError(`hits[${i}].geometricNormal encoding must be in [-1, 1]`);
    }
    const z = 1 - Math.abs(geometryX) - Math.abs(geometryY);
    const x = z < 0 ? (1 - Math.abs(geometryY)) * (geometryX >= 0 ? 1 : -1) : geometryX;
    const y = z < 0 ? (1 - Math.abs(geometryX)) * (geometryY >= 0 ? 1 : -1) : geometryY;
    const geometryLength = Math.hypot(x, y, z);
    const geometricNormal: [number, number, number] = [x / geometryLength, y / geometryLength, z / geometryLength];
    hits.push({
      batch: batchNames[batchId - 1], geometry: { ...(batchGeometry?.[batchId - 1] ?? { kind: 'source' }) },
      entityId, entityGeneration: generation, triangleIndex,
      distance, position, normal, geometricNormal, previousPosition, uv, baseColor, opacity, metallic,
      roughness, emissive, shading: isUnlit === 1 ? 'unlit' : 'pbr',
      historyVersion, topologyVersion,
    });
  }
  return { frame, materialRevision, hits };
}
