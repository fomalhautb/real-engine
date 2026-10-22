/** Bounded CPU contract for optional Final Gather reconstruction diagnostics. */
export const FINAL_GATHER_DIAGNOSTIC_MAX_POINTS = 64;
export const FINAL_GATHER_DIAGNOSTIC_WORDS_PER_POINT = 8;

export interface FinalGatherDiagnosticPoint {
  readonly x: number;
  readonly y: number;
}

export type FinalGatherDiagnosticBranch = 'disabled' | 'primary' | 'ring' | 'uncovered' | 'background' | 'unlit' | 'supplemental';
export type FinalGatherCandidateRejection =
  'invalid-or-stale' | 'identity' | 'shading-normal' | 'receiver-plane' | 'anchor-plane'
  | 'distance' | 'out-of-bounds' | 'not-visited';

export interface FinalGatherDiagnosticRecord {
  readonly pixel: FinalGatherDiagnosticPoint;
  readonly branch: FinalGatherDiagnosticBranch;
  readonly primaryAccepted: number;
  readonly ringAccepted: number;
  readonly supplementalAccepted: number;
  readonly totalWeight: number;
  readonly selectedLuminance: number;
  readonly worldLuminance: number;
  /** Four primary candidates followed by twelve recovery-ring candidates. */
  readonly candidateRejections: readonly {
    readonly mask: number;
    readonly reasons: readonly FinalGatherCandidateRejection[];
  }[];
}

export interface FinalGatherDiagnosticSurface {
  readonly position: readonly [number, number, number];
  readonly normal: readonly [number, number, number];
  readonly geometricNormal: readonly [number, number, number];
  readonly sourceId: number;
  readonly entityId: number;
  readonly representationToken: number;
}

/** The same signed-octahedral 2x16-bit encoding used by raster receivers. */
function decodeNormal(packed: number): [number, number, number] {
  const low = (packed << 16) >> 16, high = packed >> 16;
  const x = Math.max(-1, low / 32767), y = Math.max(-1, high / 32767);
  const z = 1 - Math.abs(x) - Math.abs(y);
  const nx = z < 0 ? (1 - Math.abs(y)) * (x >= 0 ? 1 : -1) : x;
  const ny = z < 0 ? (1 - Math.abs(x)) * (y >= 0 ? 1 : -1) : y;
  const length = Math.hypot(nx, ny, z);
  return [nx / length, ny / length, z / length];
}

/** Decode the second explicit redraw, retaining integer scene identity. */
export function decodeFinalGatherDiagnosticSurfaces(words: Uint32Array, points: readonly FinalGatherDiagnosticPoint[]):
  (FinalGatherDiagnosticSurface | null)[] {
  if (!(words instanceof Uint32Array)) throw new TypeError('FINAL_GATHER_DIAGNOSTIC_WORDS');
  const pixels = validateFinalGatherDiagnosticPoints(points, 32_768, 32_768);
  if (words.length !== pixels.length * 8) throw new RangeError('FINAL_GATHER_DIAGNOSTIC_LENGTH');
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  return pixels.map((_, index) => {
    const base = index * 8;
    if (words[base + 3] === 0) {
      if (words.subarray(base, base + 8).some(value => value !== 0)) throw new RangeError('FINAL_GATHER_DIAGNOSTIC_BACKGROUND');
      return null;
    }
    const position = [floats[base], floats[base + 1], floats[base + 2]] as const;
    if (!position.every(value => Number.isFinite(value))) throw new RangeError('FINAL_GATHER_DIAGNOSTIC_POSITION');
    return { position, sourceId: words[base + 3], normal: decodeNormal(words[base + 4]),
      geometricNormal: decodeNormal(words[base + 5]), entityId: words[base + 6], representationToken: words[base + 7] };
  });
}

const BRANCHES: readonly FinalGatherDiagnosticBranch[] = ['disabled', 'primary', 'ring', 'uncovered', 'background', 'unlit', 'supplemental'];

/** Full-frame diagnostic census, never a runtime readback or sampling policy.
 * One byte per pixel is retained for a lossless coverage visualization. */
export function decodeFinalGatherCoverage(words: Uint32Array, width: number, height: number, wordsPerRow: number,
  tileSize = 4) {
  if (!(words instanceof Uint32Array) || !Number.isSafeInteger(width) || width < 1
    || !Number.isSafeInteger(height) || height < 1 || width * height > 4_194_304
    || !Number.isSafeInteger(wordsPerRow) || wordsPerRow < width * 4
    || words.length !== wordsPerRow * height || !Number.isSafeInteger(tileSize) || tileSize < 1 || tileSize > 128) {
    throw new RangeError('FINAL_GATHER_COVERAGE_LAYOUT');
  }
  const branches = new Uint8Array(width * height);
  const counts = Object.fromEntries(BRANCHES.map(branch => [branch, 0])) as Record<FinalGatherDiagnosticBranch, number>;
  const columns = Math.ceil(width / tileSize), rows = Math.ceil(height / tileSize);
  const gaps = new Uint32Array(columns * rows);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const branch = words[y * wordsPerRow + x * 4] & 0xff;
    if (branch >= BRANCHES.length) throw new RangeError('FINAL_GATHER_COVERAGE_BRANCH');
    branches[y * width + x] = branch; counts[BRANCHES[branch]]++;
    if (branch === 3) gaps[Math.floor(y / tileSize) * columns + Math.floor(x / tileSize)]++;
  }
  const uncoveredPixelsPerTile = Array<number>(tileSize * tileSize + 1).fill(0);
  for (const count of gaps) uncoveredPixelsPerTile[count]++;
  return { width, height, counts, tileSize, columns, rows, uncoveredPixelsPerTile,
    tilesWithUncoveredPixels: gaps.length - uncoveredPixelsPerTile[0], branches };
}
const REASONS: readonly [number, FinalGatherCandidateRejection][] = [
  [1, 'invalid-or-stale'], [2, 'identity'], [4, 'shading-normal'], [8, 'receiver-plane'],
  [16, 'anchor-plane'], [32, 'distance'], [64, 'out-of-bounds'], [128, 'not-visited'],
];

export function validateFinalGatherDiagnosticPoints(
  points: readonly FinalGatherDiagnosticPoint[], width: number, height: number,
): FinalGatherDiagnosticPoint[] {
  if (!Number.isSafeInteger(width) || width < 1 || width > 32_768
    || !Number.isSafeInteger(height) || height < 1 || height > 32_768) {
    throw new RangeError('FINAL_GATHER_DIAGNOSTIC_FRAME_SIZE');
  }
  if (!Array.isArray(points) || points.length < 1 || points.length > FINAL_GATHER_DIAGNOSTIC_MAX_POINTS) {
    throw new RangeError('FINAL_GATHER_DIAGNOSTIC_POINT_COUNT');
  }
  const seen = new Set<number>();
  return points.map(point => {
    if (!point || !Number.isSafeInteger(point.x) || !Number.isSafeInteger(point.y)
      || point.x < 0 || point.x >= width || point.y < 0 || point.y >= height) {
      throw new RangeError('FINAL_GATHER_DIAGNOSTIC_POINT');
    }
    const key = point.y * width + point.x;
    if (seen.has(key)) throw new RangeError('FINAL_GATHER_DIAGNOSTIC_DUPLICATE_POINT');
    seen.add(key);
    return { x: point.x, y: point.y };
  });
}

/** Decode one little-endian GPU u32 record per requested point. */
export function decodeFinalGatherDiagnostics(
  words: Uint32Array, points: readonly FinalGatherDiagnosticPoint[],
): FinalGatherDiagnosticRecord[] {
  if (!(words instanceof Uint32Array)) throw new TypeError('FINAL_GATHER_DIAGNOSTIC_WORDS');
  const pixels = validateFinalGatherDiagnosticPoints(points, 32_768, 32_768);
  if (words.length !== pixels.length * FINAL_GATHER_DIAGNOSTIC_WORDS_PER_POINT) {
    throw new RangeError('FINAL_GATHER_DIAGNOSTIC_LENGTH');
  }
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  return pixels.map((pixel, index) => {
    const base = index * FINAL_GATHER_DIAGNOSTIC_WORDS_PER_POINT;
    const packed = words[base];
    const branchId = packed & 0xff, primaryAccepted = (packed >>> 8) & 0xff;
    const ringAccepted = (packed >>> 16) & 0xff;
    const supplementalAccepted = packed >>> 24;
    if (supplementalAccepted > 9 || (branchId !== 6 && supplementalAccepted !== 0)
      || (branchId === 6 && supplementalAccepted === 0) || branchId >= BRANCHES.length || primaryAccepted > 4 || ringAccepted > 12) {
      throw new RangeError('FINAL_GATHER_DIAGNOSTIC_HEADER');
    }
    const totalWeight = floats[base + 1], selectedLuminance = floats[base + 2];
    const worldLuminance = floats[base + 3];
    if (![totalWeight, selectedLuminance, worldLuminance].every(value => Number.isFinite(value) && value >= 0)) {
      throw new RangeError('FINAL_GATHER_DIAGNOSTIC_FLOAT');
    }
    const candidateRejections = Array.from({ length: 16 }, (_, candidate) => {
      const mask = (words[base + 4 + (candidate >>> 2)] >>> ((candidate & 3) * 8)) & 0xff;
      if ((mask & 0xc0) && mask !== 64 && mask !== 128) {
        throw new RangeError('FINAL_GATHER_DIAGNOSTIC_REJECTION_MASK');
      }
      return { mask, reasons: REASONS.filter(([bit]) => (mask & bit) !== 0).map(([, reason]) => reason) };
    });
    if ((branchId === 4 || branchId === 5) && (primaryAccepted !== 0 || ringAccepted !== 0
      || totalWeight !== 0 || selectedLuminance !== 0 || worldLuminance !== 0
      || candidateRejections.some(candidate => candidate.mask !== 128))) {
      throw new RangeError('FINAL_GATHER_DIAGNOSTIC_NON_SURFACE');
    }
    return { pixel, branch: BRANCHES[branchId], primaryAccepted, ringAccepted, supplementalAccepted,
      totalWeight, selectedLuminance, worldLuminance, candidateRejections };
  });
}
