/** CPU reference for the directional distance map stored with each GI probe. */
export const GI_DISTANCE_RESOLUTION = 8;
export const GI_DISTANCE_CELLS = GI_DISTANCE_RESOLUTION * GI_DISTANCE_RESOLUTION;

type Vec3 = readonly [number, number, number];
type Vec2 = readonly [number, number];

function checkedUnitDirection(direction: Vec3): Vec3 {
  if (!Array.isArray(direction) || direction.length !== 3
    || !Number.isFinite(direction[0]) || !Number.isFinite(direction[1])
    || !Number.isFinite(direction[2])) {
    throw new RangeError('GI_VISIBILITY_DIRECTION');
  }
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  if (!Number.isFinite(length) || Math.abs(length - 1) > 1e-4) {
    throw new RangeError('GI_VISIBILITY_DIRECTION');
  }
  return direction;
}

function signNotZero(value: number): number {
  return value < 0 ? -1 : 1;
}

/** Map a unit direction to [0, 1]^2 using the standard folded octahedron. */
export function octaEncode(unitvec: Vec3): [number, number] {
  const [x, y, z] = checkedUnitDirection(unitvec);
  const inverseL1 = 1 / (Math.abs(x) + Math.abs(y) + Math.abs(z));
  let u = x * inverseL1;
  let v = y * inverseL1;
  if (z < 0) {
    const oldU = u;
    u = (1 - Math.abs(v)) * signNotZero(oldU);
    v = (1 - Math.abs(oldU)) * signNotZero(v);
  }
  return [(u + 1) * 0.5, (v + 1) * 0.5];
}

/** Decode a point in [0, 1]^2 to a unit direction, including the folded hemisphere. */
export function octaDecode(uv: Vec2): [number, number, number] {
  if (!Array.isArray(uv) || uv.length !== 2
    || !Number.isFinite(uv[0]) || !Number.isFinite(uv[1])
    || uv[0] < 0 || uv[0] > 1 || uv[1] < 0 || uv[1] > 1) {
    throw new RangeError('GI_VISIBILITY_UV');
  }
  let x = uv[0] * 2 - 1;
  let y = uv[1] * 2 - 1;
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const oldX = x;
    x = (1 - Math.abs(y)) * signNotZero(oldX);
    y = (1 - Math.abs(oldX)) * signNotZero(y);
  }
  const inverseLength = 1 / Math.hypot(x, y, z);
  return [x * inverseLength, y * inverseLength, z * inverseLength];
}

/** One-texel bilinear padding mirrors the opposite half of the same octahedral edge. */
export function wrapGiDistanceTexel(x: number, y: number): number {
  const size = GI_DISTANCE_RESOLUTION;
  if (!Number.isInteger(x) || !Number.isInteger(y) || x < -1 || x > size || y < -1 || y > size) {
    throw new RangeError('GI_VISIBILITY_TEXEL');
  }
  if (x < 0) {
    x = 0;
    y = size - 1 - y;
  } else if (x >= size) {
    x = size - 1;
    y = size - 1 - y;
  }
  if (y < 0) {
    y = 0;
    x = size - 1 - x;
  } else if (y >= size) {
    y = size - 1;
    x = size - 1 - x;
  }
  return y * size + x;
}

/** Row-major cosine^32 capture weights: [distance cell][ray index]. */
export function buildGiDistanceWeights(directions: readonly Vec3[]): Float32Array {
  if (!Array.isArray(directions) || directions.length < 16 || directions.length > 256) {
    throw new RangeError('GI_VISIBILITY_RAYS');
  }
  const rays: Vec3[] = [];
  for (let index = 0; index < directions.length; index++) {
    rays.push(checkedUnitDirection(directions[index]));
  }
  const weights = new Float32Array(GI_DISTANCE_CELLS * rays.length);
  const rowWeights = new Float64Array(rays.length);
  const size = GI_DISTANCE_RESOLUTION;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cell = y * size + x;
      const normal = octaDecode([(x + 0.5) / size, (y + 0.5) / size]);
      const offset = cell * rays.length;
      let total = 0;
      let nearestRay = 0;
      let nearestDot = -Infinity;
      for (let ray = 0; ray < rays.length; ray++) {
        const direction = rays[ray];
        const dot = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
        if (dot > nearestDot) {
          nearestDot = dot;
          nearestRay = ray;
        }
        const cosine = Math.max(0, Math.min(1, dot));
        const weight = cosine ** 32;
        rowWeights[ray] = weight;
        total += weight;
      }
      // A valid but poorly distributed input may have no ray in a cell's
      // positive hemisphere. Pick its nearest ray so every row stays defined.
      if (total === 0) {
        weights[offset + nearestRay] = 1;
        continue;
      }
      // Normalize before Float32 storage: a valid narrow lobe can be far
      // below Float32's subnormal range while its normalized share is not.
      for (let ray = 0; ray < rays.length; ray++) weights[offset + ray] = rowWeights[ray] / total;
    }
  }
  return weights;
}
