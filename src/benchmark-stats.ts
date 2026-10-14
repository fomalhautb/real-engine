/** One observed rAF frame. Times are milliseconds; memory and upload values are bytes. */
export interface BenchmarkFrame {
  frame: number;
  elapsedMs: number;
  frameIntervalMs: number;
  updateMs: number;
  eventMs: number;
  cpuSubmitMs: number;
  ownedGpuBytes: number;
  uploadedBytes: number;
  triangles: number;
  deformedVertices: number;
  physicsBodies: number;
}

/** Optional GPU timings arrive after the corresponding CPU frame. Times are milliseconds. */
export interface BenchmarkGpuSample {
  frame: number;
  deformation: number;
  shadow: number;
  surface: number;
  /** Optional for recordings made before resolve and query-refit timing existed. */
  resolve?: number;
  worldQueryRefit?: number;
}

export interface BenchmarkQuantiles {
  count: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

export interface BenchmarkMemorySummary {
  firstBytes: number;
  lastBytes: number;
  minBytes: number;
  maxBytes: number;
  growthBytes: number;
  /** Consecutive thirds by sample order; nearest-rank median. Empty thirds are null. */
  thirdMedianBytes: { first: number | null; middle: number | null; last: number | null };
  thirdMedianGrowthBytes: number | null;
}

export interface BenchmarkSummary {
  frameCount: number;
  gpuSampleCount: number;
  frameBudgetMs: number;
  framesOverBudget: number;
  overBudgetFraction: number | null;
  /** Last elapsedMs minus first elapsedMs; null when no frames were recorded. */
  elapsedMs: number | null;
  frameIntervalMs: BenchmarkQuantiles | null;
  updateMs: BenchmarkQuantiles | null;
  eventMs: BenchmarkQuantiles | null;
  cpuSubmitMs: BenchmarkQuantiles | null;
  /** Sparse GPU samples. These timings are not added to CPU timings. */
  gpuMs: {
    deformation: BenchmarkQuantiles | null;
    shadow: BenchmarkQuantiles | null;
    surface: BenchmarkQuantiles | null;
    resolve: BenchmarkQuantiles | null;
    worldQueryRefit: BenchmarkQuantiles | null;
    total: BenchmarkQuantiles | null;
  };
  ownedGpuBytes: BenchmarkMemorySummary | null;
  uploadedBytesTotal: number | null;
  workload: {
    triangles: { mean: number; max: number; last: number };
    deformedVertices: { mean: number; max: number; last: number };
    physicsBodies: { mean: number; max: number; last: number };
  } | null;
}

const CHUNK_FRAMES = 4096;
const STRIDE = 17;
const FIELD = Object.freeze({
  frame: 0,
  elapsedMs: 1,
  frameIntervalMs: 2,
  updateMs: 3,
  eventMs: 4,
  cpuSubmitMs: 5,
  ownedGpuBytes: 6,
  uploadedBytes: 7,
  triangles: 8,
  deformedVertices: 9,
  physicsBodies: 10,
  gpuDeformation: 11,
  gpuShadow: 12,
  gpuSurface: 13,
  gpuResolve: 14,
  gpuWorldQueryRefit: 15,
  gpuTotal: 16,
});
type Field = (typeof FIELD)[keyof typeof FIELD];

function nonnegative(value: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be finite and nonnegative`);
  }
  return value;
}

function count(value: number, label: string): number {
  nonnegative(value, label);
  if (!Number.isSafeInteger(value)) throw new RangeError(`${label} must be a safe integer`);
  return value;
}

/** Nearest-rank percentile: sorted[ceil(p * count) - 1]. */
function nearestRank(sorted: Float64Array, fraction: number): number {
  return sorted[Math.ceil(fraction * sorted.length) - 1];
}

function quantiles(values: Float64Array): BenchmarkQuantiles | null {
  if (values.length === 0) return null;
  let mean = 0, seen = 0;
  for (const value of values) {
    seen++;
    mean += (value - mean) / seen;
  }
  values.sort();
  return {
    count: values.length,
    mean,
    p50: nearestRank(values, 0.5),
    p95: nearestRank(values, 0.95),
    p99: nearestRank(values, 0.99),
    max: values[values.length - 1],
  };
}

/** Bounded telemetry storage. Each 4,096-frame chunk is allocated only when needed. */
export class BenchmarkRecorder {
  private readonly chunks: Float64Array[] = [];
  private frames = 0;
  private gpuSamples = 0;
  private lastGpuFrame = -1;
  private uploadedTotal = 0;

  constructor(readonly maxFrames: number, readonly frameBudgetMs: number) {
    if (!Number.isSafeInteger(maxFrames) || maxFrames < 1 || maxFrames > 1_000_000) {
      throw new RangeError('maxFrames must be an integer from 1 to 1,000,000');
    }
    if (!Number.isFinite(frameBudgetMs) || frameBudgetMs <= 0) {
      throw new RangeError('frameBudgetMs must be finite and positive');
    }
  }

  get frameCount(): number { return this.frames; }
  get gpuSampleCount(): number { return this.gpuSamples; }

  private get(index: number, field: Field): number {
    return this.chunks[Math.floor(index / CHUNK_FRAMES)][(index % CHUNK_FRAMES) * STRIDE + field];
  }

  private set(index: number, field: Field, value: number): void {
    this.chunks[Math.floor(index / CHUNK_FRAMES)][(index % CHUNK_FRAMES) * STRIDE + field] = value;
  }

  record(frame: BenchmarkFrame): void {
    if (this.frames >= this.maxFrames) throw new RangeError('benchmark frame limit reached');
    count(frame.frame, 'frame');
    nonnegative(frame.elapsedMs, 'elapsedMs');
    nonnegative(frame.frameIntervalMs, 'frameIntervalMs');
    nonnegative(frame.updateMs, 'updateMs');
    nonnegative(frame.eventMs, 'eventMs');
    nonnegative(frame.cpuSubmitMs, 'cpuSubmitMs');
    count(frame.ownedGpuBytes, 'ownedGpuBytes');
    count(frame.uploadedBytes, 'uploadedBytes');
    count(frame.triangles, 'triangles');
    count(frame.deformedVertices, 'deformedVertices');
    count(frame.physicsBodies, 'physicsBodies');
    if (this.frames > 0 && (frame.frame <= this.get(this.frames - 1, FIELD.frame)
      || frame.elapsedMs <= this.get(this.frames - 1, FIELD.elapsedMs))) {
      throw new RangeError('benchmark frame IDs and elapsed times must strictly increase');
    }
    const nextUploadedTotal = this.uploadedTotal + frame.uploadedBytes;
    if (!Number.isSafeInteger(nextUploadedTotal)) throw new RangeError('uploaded byte total exceeds safe integer range');

    const index = this.frames;
    if (index % CHUNK_FRAMES === 0) this.chunks.push(new Float64Array(CHUNK_FRAMES * STRIDE));
    this.set(index, FIELD.frame, frame.frame);
    this.set(index, FIELD.elapsedMs, frame.elapsedMs);
    this.set(index, FIELD.frameIntervalMs, frame.frameIntervalMs);
    this.set(index, FIELD.updateMs, frame.updateMs);
    this.set(index, FIELD.eventMs, frame.eventMs);
    this.set(index, FIELD.cpuSubmitMs, frame.cpuSubmitMs);
    this.set(index, FIELD.ownedGpuBytes, frame.ownedGpuBytes);
    this.set(index, FIELD.uploadedBytes, frame.uploadedBytes);
    this.set(index, FIELD.triangles, frame.triangles);
    this.set(index, FIELD.deformedVertices, frame.deformedVertices);
    this.set(index, FIELD.physicsBodies, frame.physicsBodies);
    for (const field of [FIELD.gpuDeformation, FIELD.gpuShadow, FIELD.gpuSurface,
      FIELD.gpuResolve, FIELD.gpuWorldQueryRefit, FIELD.gpuTotal]) this.set(index, field, NaN);
    this.uploadedTotal = nextUploadedTotal;
    this.frames++;
  }

  recordGpu(sample: BenchmarkGpuSample): void {
    count(sample.frame, 'GPU sample frame');
    nonnegative(sample.deformation, 'GPU deformation');
    nonnegative(sample.shadow, 'GPU shadow');
    nonnegative(sample.surface, 'GPU surface');
    const resolve = nonnegative(sample.resolve === undefined ? 0 : sample.resolve, 'GPU resolve');
    const worldQueryRefit = nonnegative(sample.worldQueryRefit === undefined ? 0 : sample.worldQueryRefit, 'GPU worldQueryRefit');
    if (sample.frame <= this.lastGpuFrame) throw new RangeError('GPU sample frames must strictly increase');
    const total = sample.deformation + sample.shadow + sample.surface + resolve + worldQueryRefit;
    if (!Number.isFinite(total)) throw new RangeError('GPU timing total must be finite');
    let low = 0, high = this.frames - 1;
    while (low <= high) {
      const middle = low + Math.floor((high - low) / 2);
      const frame = this.get(middle, FIELD.frame);
      if (frame === sample.frame) {
        this.set(middle, FIELD.gpuDeformation, sample.deformation);
        this.set(middle, FIELD.gpuShadow, sample.shadow);
        this.set(middle, FIELD.gpuSurface, sample.surface);
        this.set(middle, FIELD.gpuResolve, resolve);
        this.set(middle, FIELD.gpuWorldQueryRefit, worldQueryRefit);
        this.set(middle, FIELD.gpuTotal, total);
        this.lastGpuFrame = sample.frame;
        this.gpuSamples++;
        return;
      }
      if (frame < sample.frame) low = middle + 1;
      else high = middle - 1;
    }
    throw new RangeError('GPU sample must refer to a recorded frame');
  }

  private values(field: Field, gpuOnly = false): Float64Array {
    const values = new Float64Array(gpuOnly ? this.gpuSamples : this.frames);
    let output = 0;
    for (let index = 0; index < this.frames; index++) {
      if (!gpuOnly || !Number.isNaN(this.get(index, FIELD.gpuTotal))) values[output++] = this.get(index, field);
    }
    return values;
  }

  private memory(): BenchmarkMemorySummary | null {
    if (this.frames === 0) return null;
    const values = this.values(FIELD.ownedGpuBytes);
    const firstBytes = values[0], lastBytes = values[values.length - 1];
    let minBytes = Infinity, maxBytes = -Infinity;
    for (const value of values) { minBytes = Math.min(minBytes, value); maxBytes = Math.max(maxBytes, value); }
    const firstEnd = Math.ceil(values.length / 3), middleEnd = Math.ceil(2 * values.length / 3);
    const median = (start: number, end: number): number | null => {
      if (start === end) return null;
      return nearestRank(values.slice(start, end).sort(), 0.5);
    };
    const thirds = {
      first: median(0, firstEnd),
      middle: median(firstEnd, middleEnd),
      last: median(middleEnd, values.length),
    };
    return { firstBytes, lastBytes, minBytes, maxBytes, growthBytes: lastBytes - firstBytes,
      thirdMedianBytes: thirds,
      thirdMedianGrowthBytes: thirds.first === null || thirds.last === null ? null : thirds.last - thirds.first };
  }

  private workload(field: Field): { mean: number; max: number; last: number } {
    let sum = 0, max = 0;
    for (let index = 0; index < this.frames; index++) {
      const value = this.get(index, field);
      sum += value;
      max = Math.max(max, value);
    }
    return { mean: sum / this.frames, max, last: this.get(this.frames - 1, field) };
  }

  summary(): BenchmarkSummary {
    let framesOverBudget = 0;
    for (let index = 0; index < this.frames; index++) {
      if (this.get(index, FIELD.frameIntervalMs) > this.frameBudgetMs) framesOverBudget++;
    }
    return {
      frameCount: this.frames,
      gpuSampleCount: this.gpuSamples,
      frameBudgetMs: this.frameBudgetMs,
      framesOverBudget,
      overBudgetFraction: this.frames === 0 ? null : framesOverBudget / this.frames,
      elapsedMs: this.frames === 0 ? null : this.get(this.frames - 1, FIELD.elapsedMs) - this.get(0, FIELD.elapsedMs),
      frameIntervalMs: quantiles(this.values(FIELD.frameIntervalMs)),
      updateMs: quantiles(this.values(FIELD.updateMs)),
      eventMs: quantiles(this.values(FIELD.eventMs)),
      cpuSubmitMs: quantiles(this.values(FIELD.cpuSubmitMs)),
      gpuMs: {
        deformation: quantiles(this.values(FIELD.gpuDeformation, true)),
        shadow: quantiles(this.values(FIELD.gpuShadow, true)),
        surface: quantiles(this.values(FIELD.gpuSurface, true)),
        resolve: quantiles(this.values(FIELD.gpuResolve, true)),
        worldQueryRefit: quantiles(this.values(FIELD.gpuWorldQueryRefit, true)),
        total: quantiles(this.values(FIELD.gpuTotal, true)),
      },
      ownedGpuBytes: this.memory(),
      uploadedBytesTotal: this.frames === 0 ? null : this.uploadedTotal,
      workload: this.frames === 0 ? null : {
        triangles: this.workload(FIELD.triangles),
        deformedVertices: this.workload(FIELD.deformedVertices),
        physicsBodies: this.workload(FIELD.physicsBodies),
      },
    };
  }

  toCsv(): string {
    const rows = ['frame,elapsedMs,frameIntervalMs,updateMs,eventMs,cpuSubmitMs,ownedGpuBytes,uploadedBytes,triangles,deformedVertices,physicsBodies,gpuDeformationMs,gpuShadowMs,gpuSurfaceMs,gpuResolveMs,gpuWorldQueryRefitMs,gpuTotalMs'];
    for (let index = 0; index < this.frames; index++) {
      const cells = Array.from({ length: STRIDE }, (_, field) => this.get(index, field as Field));
      rows.push(cells.map(value => Number.isNaN(value) ? '' : String(value)).join(','));
    }
    return rows.join('\n') + '\n';
  }
}
