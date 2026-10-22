/** CPU contracts and bounded screen grid planning for diffuse final gathering. */
export interface FinalGatherOptions {
  pixelSpacing?: number;
  maxAnchors?: number;
  raysPerAnchor?: number;
  /** Exact diffuse segments before querying the world irradiance cache (1–4).
   * Single-bounce mode always traces one segment. */
  pathSegments?: number;
  maxHistory?: number;
  normalBias?: number;
  maxBytes?: number;
  /** The camera receiver can come from world queries or the completed raster prepass. */
  receiver?: 'world-query' | 'raster';
  /** Renderer-owned full-resolution receiver attachments and metadata, separate from maxBytes. */
  maxReceiverBytes?: number;
}

export interface ResolvedFinalGatherOptions {
  readonly pixelSpacing: number;
  readonly maxAnchors: number;
  readonly raysPerAnchor: number;
  readonly pathSegments: number;
  readonly maxHistory: number;
  readonly normalBias: number;
  readonly maxBytes: number;
  readonly receiver: 'world-query' | 'raster';
  readonly maxReceiverBytes: number;
}

export interface FinalGatherGridPlan {
  readonly spacing: number;
  readonly columns: number;
  readonly rows: number;
  readonly anchorCount: number;
  readonly rayCount: number;
}

function boundedInteger(value: unknown, minimum: number, maximum: number, name: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`FINAL_GATHER_${name}`);
  }
  return value;
}

export function resolveFinalGatherOptions(options: FinalGatherOptions = {}): ResolvedFinalGatherOptions {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new RangeError('FINAL_GATHER_OPTIONS');
  }
  const pixelSpacing = boundedInteger(options.pixelSpacing === undefined ? 8 : options.pixelSpacing, 1, 128, 'PIXEL_SPACING');
  const maxAnchors = boundedInteger(options.maxAnchors === undefined ? 16_384 : options.maxAnchors, 1, 65_536, 'MAX_ANCHORS');
  const raysPerAnchor = boundedInteger(options.raysPerAnchor === undefined ? 2 : options.raysPerAnchor, 1, 16, 'RAYS_PER_ANCHOR');
  const pathSegments = boundedInteger(options.pathSegments === undefined ? 1 : options.pathSegments, 1, 4, 'PATH_SEGMENTS');
  const maxHistory = boundedInteger(options.maxHistory === undefined ? 64 : options.maxHistory, 1, 256, 'MAX_HISTORY');
  const normalBias = options.normalBias === undefined ? 0.0001 : options.normalBias;
  if (typeof normalBias !== 'number' || !Number.isFinite(normalBias) || normalBias < 0 || normalBias > 0.01) {
    throw new RangeError('FINAL_GATHER_NORMAL_BIAS');
  }
  const maxBytes = boundedInteger(options.maxBytes === undefined ? 16 * 1024 * 1024 : options.maxBytes,
    1, 64 * 1024 * 1024, 'MAX_BYTES');
  const receiver = options.receiver === undefined ? 'world-query' : options.receiver;
  if (receiver !== 'world-query' && receiver !== 'raster') throw new RangeError('FINAL_GATHER_RECEIVER');
  const maxReceiverBytes = boundedInteger(options.maxReceiverBytes === undefined ? 128 * 1024 * 1024 : options.maxReceiverBytes,
    1, 512 * 1024 * 1024, 'MAX_RECEIVER_BYTES');
  return Object.freeze({ pixelSpacing, maxAnchors, raysPerAnchor, pathSegments, maxHistory, normalBias, maxBytes, receiver, maxReceiverBytes });
}

/** Bounded double-precision inversion. Keep doubles until camera-relative ray
 * coefficients have been formed; rounding inverse translation first amplifies
 * cancellation at distant cameras. */
export function invertFinalGatherProjection(matrix: ArrayLike<number>): Float64Array<ArrayBuffer> {
  if (matrix.length !== 16 || !Array.from(matrix).every(Number.isFinite)) throw new Error('FINAL_GATHER_VIEW_PROJECTION');
  const rows = Array.from({ length: 4 }, (_, row) => Array.from({ length: 8 }, (_, column) =>
    column < 4 ? matrix[column * 4 + row] : Number(column - 4 === row)));
  for (let column = 0; column < 4; column++) {
    let pivot = column;
    for (let row = column + 1; row < 4; row++) if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    if (rows[pivot][column] === 0) throw new Error('FINAL_GATHER_VIEW_PROJECTION');
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const divisor = rows[column][column];
    for (let i = 0; i < 8; i++) rows[column][i] /= divisor;
    for (let row = 0; row < 4; row++) {
      if (row === column) continue;
      const multiplier = rows[row][column];
      for (let i = 0; i < 8; i++) rows[row][i] -= multiplier * rows[column][i];
    }
  }
  const output = new Float64Array(16);
  for (let column = 0; column < 4; column++) for (let row = 0; row < 4; row++) output[column * 4 + row] = rows[row][column + 4];
  if (!output.every(Number.isFinite)) throw new Error('FINAL_GATHER_VIEW_PROJECTION');
  for (const depth of [0, 1]) for (const x of [-1, 1]) for (const y of [-1, 1]) {
    const w = output[3] * x + output[7] * y + output[11] * depth + output[15];
    if (!Number.isFinite(w) || Math.abs(w) < 1e-12) throw new Error('FINAL_GATHER_FINITE_CAMERA_RANGE');
  }
  return output;
}

/** Reconstruct P = origin + clipW * (dx * ndcX + dy * ndcY + dz).
 * Solve the inverse-projection homogeneous equation in doubles, so no GPU
 * subtraction of translated near/far world points is needed. clipW must be the
 * actual current raster clip.w, not nonlinear depth or Euclidean ray distance. */
export function finalGatherRasterRayBasis(inverse: ArrayLike<number>) {
  if (inverse.length !== 16 || !Array.from(inverse).every(Number.isFinite) || Math.abs(inverse[11]) < 1e-12) {
    throw new Error('FINAL_GATHER_RASTER_PERSPECTIVE');
  }
  const origin = [0, 1, 2].map(axis => inverse[8 + axis] / inverse[11]);
  const basis = (column: number) => [0, 1, 2].map(axis => inverse[column * 4 + axis] - origin[axis] * inverse[column * 4 + 3]);
  const dx = basis(0), dy = basis(1), dz = basis(3);
  if (![...origin, ...dx, ...dy, ...dz].every(v => Number.isFinite(v) && Number.isFinite(Math.fround(v)))) {
    throw new Error('FINAL_GATHER_RASTER_PERSPECTIVE');
  }
  return { origin, dx, dy, dz };
}

/** Find the finest integer spacing satisfying every per-frame ray and anchor budget. */
export function planFinalGatherGrid(
  width: number, height: number, options: FinalGatherOptions | ResolvedFinalGatherOptions,
  maxRays: number, maxWorkUnits: number,
): FinalGatherGridPlan {
  boundedInteger(width, 1, 32_768, 'WIDTH');
  boundedInteger(height, 1, 32_768, 'HEIGHT');
  boundedInteger(maxRays, 1, 0xffff_ffff, 'MAX_RAYS');
  boundedInteger(maxWorkUnits, 1, 0xffff_ffff, 'MAX_WORK_UNITS');
  const resolved = resolveFinalGatherOptions(options);
  const anchorBudget = Math.min(resolved.maxAnchors,
    Math.floor(maxRays / resolved.raysPerAnchor),
    Math.floor(maxWorkUnits / resolved.raysPerAnchor));
  if (anchorBudget < 1) throw new RangeError('FINAL_GATHER_QUERY_RAY_BUDGET');

  const fits = (spacing: number): boolean =>
    Math.ceil(width / spacing) * Math.ceil(height / spacing) <= anchorBudget;
  let low = resolved.pixelSpacing;
  let high = Math.max(low, width, height);
  // Monotonic count and a bounded search keep resize independent of viewport area.
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (fits(middle)) high = middle;
    else low = middle + 1;
  }
  const spacing = low;
  const columns = Math.ceil(width / spacing);
  const rows = Math.ceil(height / spacing);
  const anchorCount = columns * rows;
  return Object.freeze({ spacing, columns, rows, anchorCount,
    rayCount: anchorCount * resolved.raysPerAnchor });
}
