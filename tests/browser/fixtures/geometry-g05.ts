/**
 * G05 geometry record: micro-triangles, overlapping thin sheets and multi-material
 * batches, rendered through the real Renderer from fixed procedural content and
 * fixed seeds.
 *
 * Correctness compares HDR identity colours with CPU analytic references; the spec
 * asserts those results. Every timing is DIAGNOSTIC: the GPU is shared with other
 * agents, timings are recorded and never gated, and a software adapter proves
 * function only. HDR, LOD-probe and timestamp readbacks here are test-only
 * diagnostics; no runtime path reads geometry or timings back to choose work.
 */
import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import type { GeometryHierarchy } from '../../../src/geometry-hierarchy-types';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import type { GeometryPageProvider } from '../../../src/geometry-stream-types';
import { identity, normalMatrix } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer, type RendererOptions } from '../../../src/renderer';
import type { TextureAsset } from '../../../src/texture-assets';

type Vec3 = [number, number, number];
type Color = [number, number, number, number];
type View = { readonly eye: Vec3; readonly target: Vec3 };
type HdrImage = { readonly width: number; readonly height: number; readonly rgb: Float32Array };
/** Test-only access to renderer internals: device, HDR target and the pass timestamp query set. */
type RendererInternals = { device: GPUDevice; hdr?: GPUTexture; querySet?: GPUQuerySet; timestampCount: number };

export interface G05FramePlan {
  /** Frames rendered before a capture; at least 3 so committed batches settle. */
  readonly warmup: number;
  /** Pipelined frames whose pass timestamps are resolved after every frame. */
  readonly frames: number;
  /** Frames each awaited to queue completion (wall-clock frame time). */
  readonly serialized: number;
}
interface G05Common { readonly profile: 'default' | 'large'; readonly frames: G05FramePlan }
export interface G05MicroOptions extends G05Common {
  /** Cube-sphere grid cells per face edge; triangles = 12 n². */
  readonly subdivisions: number;
  readonly width: number;
  readonly height: number;
  /** Projected sphere diameters in pixels. */
  readonly diametersPx: readonly number[];
  /** Include the paged stream path (settled before capture and timing). */
  readonly stream: boolean;
  readonly maxSettleFrames: number;
  readonly stableFrames: number;
}
export interface G05SheetOptions extends G05Common {
  readonly width: number;
  readonly height: number;
  /** Ascending sheet counts, each 1..64. */
  readonly layers: readonly number[];
  /** Cutout texture seed. */
  readonly seed: number;
}
export interface G05MaterialOptions extends G05Common {
  readonly width: number;
  readonly height: number;
  /** Wall quads per side; the same total for every material count. */
  readonly quads: number;
  /** Ascending powers of two, each 1..256. */
  readonly materials: readonly number[];
}

export const G05_DIAGNOSTIC = 'DIAGNOSTIC: shared GPU (other agents active); recorded, never gated; not a performance result';
const SOFTWARE_NOTE = 'software adapter: functional evidence only, not hardware performance';
const CPU_TIMER_NOTE = 'CPU times use performance.now(); Chrome coarsens it (about 0.1 ms) unless the page is cross-origin isolated';
const TAN_HALF = Math.tan(Math.PI / 6.4);
const CLEAR: Vec3 = [0.0035, 0.0055, 0.0065];
const LEVELS = [0.15, 0.2833, 0.4167, 0.55, 0.6833, 0.8167, 0.95];
const BACKGROUND = -2, UNCLASSIFIED = -1;
const MiB = 1024 * 1024;
/** Renderer query-set slot pairs without world queries, GI, reflections, occlusion, local lights or fog. */
const PASSES = [['deformation', 0], ['shadow', 2], ['surface', 4], ['resolve', 6], ['geometrySelection', 8]] as const;
type PassName = typeof PASSES[number][0];

const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a: readonly number[]): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const round = (value: number, digits = 4) => Number(value.toFixed(digits));

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function integer(value: number, min: number, max: number, code: string): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(code);
  return value;
}
function validateCommon(options: G05Common & { width: number; height: number }): void {
  if (options.profile !== 'default' && options.profile !== 'large') throw new RangeError('G05_PROFILE');
  integer(options.frames.warmup, 3, 240, 'G05_WARMUP_FRAMES');
  integer(options.frames.frames, 4, 1000, 'G05_TIMED_FRAMES');
  integer(options.frames.serialized, 1, 240, 'G05_SERIALIZED_FRAMES');
  integer(options.width, 64, 4096, 'G05_WIDTH');
  integer(options.height, 64, 4096, 'G05_HEIGHT');
}
function ascending(values: readonly number[], min: number, max: number, code: string): number[] {
  if (!values.length || values.length > 8 || values.some((value, index) => !Number.isSafeInteger(value) || value < min || value > max
    || (index > 0 && value <= values[index - 1]))) throw new RangeError(code);
  return [...values];
}

function instance(entityId: number, model: Float32Array, color: Color = [1, 1, 1, 1], roughness = 0.6): InstanceState {
  return { entityId, entityGeneration: 1, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [], color,
    metalness: 0, roughness, historyVersion: 0 };
}
function staticBatch(name: string, mesh: MeshData, instances: InstanceState[], material: MaterialBinding, capacity = instances.length): SceneBatch {
  return { name, mesh, capacity, inputMode: 'committed', instances, deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material };
}

// ---------------------------------------------------------------------------
// Identity colours: 343 colours on a 7-level lattice (minimum L1 distance 0.133).

export function g05IdentityColor(index: number): Color {
  integer(index, 0, 342, 'G05_IDENTITY_INDEX');
  const r = index % 7, g = Math.floor(index / 7) % 7, b = Math.floor(index / 49) % 7;
  return [LEVELS[r], LEVELS[(g + 3 * r) % 7], LEVELS[(b + 2 * g + 5 * r) % 7], 1];
}
function levelOf(value: number): number {
  for (let level = 0; level < LEVELS.length; level++) if (Math.abs(value - LEVELS[level]) <= 0.01) return level;
  return -1;
}
function identityOf(r: number, g: number, b: number): number {
  const lr = levelOf(r), lg = levelOf(g), lb = levelOf(b);
  if (lr < 0 || lg < 0 || lb < 0) return -1;
  const gi = (((lg - 3 * lr) % 7) + 7) % 7, bi = (((lb - 2 * gi - 5 * lr) % 7) + 7) % 7;
  return lr + 7 * gi + 49 * bi;
}
/** Maps identity index to object label; -1 marks identities that are not objects. */
function labelTable(identities: readonly number[]): Int16Array {
  const table = new Int16Array(343).fill(-1);
  identities.forEach((id, label) => { table[id] = label; });
  return table;
}
/** Per-pixel object label, BACKGROUND (clear colour) or UNCLASSIFIED. */
function classify(image: HdrImage, table: Int16Array): Int16Array {
  const labels = new Int16Array(image.width * image.height);
  for (let p = 0; p < labels.length; p++) {
    const r = image.rgb[p * 3], g = image.rgb[p * 3 + 1], b = image.rgb[p * 3 + 2];
    if (Math.abs(r - CLEAR[0]) + Math.abs(g - CLEAR[1]) + Math.abs(b - CLEAR[2]) < 0.01) { labels[p] = BACKGROUND; continue; }
    const id = identityOf(r, g, b);
    labels[p] = id >= 0 && table[id] >= 0 ? table[id] : UNCLASSIFIED;
  }
  return labels;
}
/** Pixels whose expected label equals all eight neighbours' (sampled one pixel apart): stable
 * under sub-pixel rasterization and interpolation differences. */
function robustMask(expected: Int16Array, width: number, height: number): Uint8Array {
  const robust = new Uint8Array(width * height);
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const p = y * width + x, label = expected[p];
    let same = true;
    for (let dy = -1; dy <= 1 && same; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (expected[p + dy * width + dx] !== label) { same = false; break; }
    }
    if (same) robust[p] = 1;
  }
  return robust;
}

// ---------------------------------------------------------------------------
// Test-only readbacks and evidence.

function viewportCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas'), ratio = Math.min(devicePixelRatio, 1.5);
  canvas.dataset.g05Fixture = 'true';
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${width / ratio}px;height:${height / ratio}px;z-index:10000`;
  document.body.appendChild(canvas);
  return canvas;
}
async function readHdr(renderer: Renderer): Promise<HdrImage> {
  const r = renderer as unknown as RendererInternals, width = renderer.stats.width, height = renderer.stats.height;
  if (!r.hdr) throw new Error('G05_NO_HDR');
  const bytesPerRow = Math.ceil(width * 8 / 256) * 256;
  const buffer = r.device.createBuffer({ label: 'G05 HDR readback', size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = r.device.createCommandEncoder({ label: 'G05 HDR copy' });
    encoder.copyTextureToBuffer({ texture: r.hdr }, { buffer, bytesPerRow }, { width, height });
    r.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint16Array(buffer.getMappedRange()), rgb = new Float32Array(width * height * 3);
    const half = (bits: number) => { const e = (bits >>> 10) & 31, f = bits & 1023, v = e ? (1 + f / 1024) * 2 ** (e - 15) : f * 2 ** -24; return bits & 0x8000 ? -v : v; };
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) rgb[(y * width + x) * 3 + c] = half(words[y * bytesPerRow / 2 + x * 4 + c]);
    }
    buffer.unmap();
    return { width, height, rgb };
  } finally { buffer.destroy(); }
}
/** Square-root tone-mapped PNG; marked pixels are drawn pure red. */
function encodePng(image: HdrImage, marked?: Uint8Array): string {
  const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
  const context = canvas.getContext('2d')!, data = context.createImageData(image.width, image.height);
  for (let p = 0; p < image.width * image.height; p++) {
    const red = marked?.[p] === 1;
    for (let c = 0; c < 3; c++) data.data[p * 4 + c] = red ? (c === 0 ? 255 : 0) : Math.min(255, Math.round(Math.sqrt(Math.max(0, image.rgb[p * 3 + c])) * 255));
    data.data[p * 4 + 3] = 255;
  }
  context.putImageData(data, 0, 0);
  return canvas.toDataURL('image/png');
}
function header(renderer: Renderer) {
  const info = renderer.info, adapter = info.adapterInfo;
  const software = adapter.isFallbackAdapter === true || /swiftshader/i.test(`${adapter.vendor} ${adapter.architecture} ${adapter.description}`);
  return { adapter, software, rendererTiming: info.timing, timestampQuery: info.features.includes('timestamp-query'),
    userAgent: info.userAgent, crossOriginIsolated: globalThis.crossOriginIsolated === true, cpuTimerNote: CPU_TIMER_NOTE,
    labels: software ? [G05_DIAGNOSTIC, SOFTWARE_NOTE] : [G05_DIAGNOSTIC] };
}
type Header = ReturnType<typeof header>;
type Cleanup = { label: string; errors: string[]; ownedGpuBytesAfterDispose: number };

/** Creates, uses and always disposes one renderer and its canvas. */
async function withRenderer<T>(label: string, width: number, height: number, batches: SceneBatch[], options: RendererOptions,
  cleanup: Cleanup[], use: (renderer: Renderer) => Promise<T>): Promise<T> {
  const canvas = viewportCanvas(width, height);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches, options);
    return await use(renderer);
  } finally {
    renderer?.dispose();
    // Paged-stream disposal settles asynchronously and reports into errors.
    await new Promise(resolve => setTimeout(resolve, 0));
    canvas.remove();
    if (renderer) cleanup.push({ label, errors: [...renderer.errors], ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes });
  }
}

// ---------------------------------------------------------------------------
// Frame loop and diagnostic timing.

async function idle(device: GPUDevice): Promise<void> {
  await device.queue.onSubmittedWorkDone();
  await new Promise(resolve => setTimeout(resolve, 0));
}
/** One rendered frame. A paged stream refuses frames under frames-in-flight
 * backpressure; wait for the queue and retry, bounded. */
async function renderFrame(renderer: Renderer, view: View): Promise<{ cpuSubmitMs: number; frame: number; retries: number; deformationDispatches: number }> {
  const device = (renderer as unknown as RendererInternals).device;
  for (let retries = 0; retries <= 64; retries++) {
    const before = renderer.stats.frames;
    renderer.render(view.eye, view.target);
    if (renderer.errors.length) throw new Error('G05_RENDER_ERROR: ' + renderer.errors.join(' | '));
    if (renderer.stats.frames === before + 1) {
      return { cpuSubmitMs: renderer.stats.cpuSubmitMs, frame: renderer.stats.frames, retries, deformationDispatches: renderer.stats.deformationDispatches };
    }
    await idle(device);
  }
  throw new Error('G05_FRAME_BACKPRESSURE');
}
/** mean is reported because coarsened CPU timers quantize single samples. */
export interface G05Summary { n: number; median: number; p95: number; mean: number; min: number; max: number }
/** Nearest-rank median and p95 (the src/benchmark-stats convention). */
function summarize(values: readonly number[]): G05Summary | null {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const rank = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))];
  return { n: sorted.length, median: round(rank(0.5), 5), p95: round(rank(0.95), 5),
    mean: round(sorted.reduce((sum, value) => sum + value, 0) / sorted.length, 5), min: round(sorted[0], 5), max: round(sorted[sorted.length - 1], 5) };
}
function bigGcd(a: bigint, b: bigint): bigint { while (b) [a, b] = [b, a % b]; return a; }

type SampleState = 'fresh' | 'stale' | 'invalid' | 'noDispatch';
/** Per-pass slot integrity over the timed window. */
interface PassIntegrity {
  fresh: number; stale: number; invalid: number;
  /** Frames in which the pass encoded no dispatch (deformation only): excluded whatever its slots hold. */
  noDispatch: number;
  /** No-dispatch frames whose slots still changed: an empty-pass timestamp, or an old value converted again. */
  noDispatchRewritten: number;
  /** Frames in which this pass contributed to the frame span. */
  spanFrames: number;
  /** One character per timed frame: f fresh, s stale, i invalid, n no dispatch with unchanged slots, r no dispatch with changed slots. */
  pattern: string;
  /** Duration of the r samples and their age (shadow begin minus sample end, same resolve). Not measurements of work. */
  noDispatchRewrittenMs: { duration: G05Summary | null; ageBeforeShadow: G05Summary | null } | null;
  /** First sample that is not fresh: raw nanosecond values as decimal strings, with the pair resolved after the previous frame. */
  example: { state: Exclude<SampleState, 'fresh'>; frame: number; begin: string; end: string; previousBegin: string; previousEnd: string } | null;
}
const TIMESTAMP_RULE = 'per frame and pass: noDispatch when the deformation pass encoded no dispatch (renderer.stats.deformationDispatches is 0), '
  + 'excluded whatever its slots hold; otherwise invalid when an endpoint is zero or end < begin, stale when an endpoint equals the value resolved '
  + 'after the previous frame (the slot was not rewritten), else fresh; only fresh samples enter pass summaries, frame spans, pass sums and the granularity';
const SPAN_RULE = 'per frame: earliest fresh begin to latest fresh end; requires fresh shadow, surface and resolve; other passes are '
  + 'excluded, and inter-pass gaps are included';
const STALE_NOTE = 'some passes were not fresh in every frame. A compute pass that encodes no dispatch may keep old slot values (pattern n), '
  + 'or report changed values without work (pattern r: a real empty-pass timestamp has a tiny duration and age; an old value converted again '
  + 'keeps the old duration and has a large age). Zero slots were never written. renderer.stats.gpuMs reads the same slots and inherits the same values';
const SPAN_REQUIRED: readonly PassName[] = ['shadow', 'surface', 'resolve'];

/** Diagnostic timing of a static view. Each frame's query set is resolved right
 * after its submit, so a resolve reads exactly that frame. A slot that the
 * adapter did not rewrite keeps its previous value; region 0 holds a baseline
 * resolved before the window so every timed frame can be checked against the
 * frame before it. No timestamp writes are added, so the Apple partial-write
 * hazard of src/gpu-timestamp-span.ts does not arise. */
async function measureFrames(renderer: Renderer, view: View, plan: G05FramePlan) {
  const internals = renderer as unknown as RendererInternals, device = internals.device, querySet = internals.querySet;
  const count = internals.timestampCount;
  if (querySet && count !== 8 && count !== 10) throw new Error('G05_TIMESTAMP_LAYOUT:' + count);
  const passes = PASSES.filter(([, slot]) => slot + 1 < count);
  let retries = 0;
  // Re-warm after capture readbacks so timed frames start from a busy queue.
  for (let i = 0; i < 3; i++) retries += (await renderFrame(renderer, view)).retries;
  await idle(device);
  const stride = Math.ceil(count * 8 / 256) * 256, bytes = stride * (plan.frames + 1);
  const resolve = querySet && device.createBuffer({ label: 'G05 timestamp resolve', size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = querySet && device.createBuffer({ label: 'G05 timestamp readback', size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const resolveInto = (region: number) => {
    if (!querySet || !resolve) return;
    const encoder = device.createCommandEncoder({ label: 'G05 per-frame timestamp resolve' });
    encoder.resolveQuerySet(querySet, 0, count, resolve, region * stride);
    device.queue.submit([encoder.finish()]);
  };
  const cpuSubmitMs: number[] = [], frameIds: number[] = [], dispatches: number[] = [];
  let raw: BigUint64Array | null = null;
  try {
    resolveInto(0);
    for (let k = 0; k < plan.frames; k++) {
      const frame = await renderFrame(renderer, view);
      retries += frame.retries; cpuSubmitMs.push(frame.cpuSubmitMs); frameIds.push(frame.frame); dispatches.push(frame.deformationDispatches);
      resolveInto(k + 1);
      // Bounded queue depth: the CPU runs at most three frames ahead.
      if (k % 3 === 2) await device.queue.onSubmittedWorkDone();
    }
    if (resolve && read) {
      const encoder = device.createCommandEncoder({ label: 'G05 timestamp copy' });
      encoder.copyBufferToBuffer(resolve, 0, read, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      raw = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
    }
  } finally { resolve?.destroy(); read?.destroy(); }
  await idle(device);
  const sampleFrame = renderer.stats.gpuSampleFrame, sample = renderer.stats.gpuMs;
  const serialized: number[] = [];
  for (let k = 0; k < plan.serialized; k++) {
    const started = performance.now();
    retries += (await renderFrame(renderer, view)).retries;
    await device.queue.onSubmittedWorkDone();
    serialized.push(performance.now() - started);
  }
  const perPass = new Map<PassName, number[]>(passes.map(([name]) => [name, []]));
  const integrity = new Map<PassName, PassIntegrity>(passes.map(([name]) => [name, { fresh: 0, stale: 0, invalid: 0, noDispatch: 0,
    noDispatchRewritten: 0, spanFrames: 0, pattern: '', noDispatchRewrittenMs: null, example: null }]));
  const rewrittenDurations = new Map<PassName, number[]>(passes.map(([name]) => [name, []]));
  const rewrittenAges = new Map<PassName, number[]>(passes.map(([name]) => [name, []]));
  const spans: number[] = [], sums: number[] = [];
  let gcd = 0n, origin: bigint | null = null, crossCheck: { frame: number; maxAbsDiffMs: number } | null = null;
  const values = raw;
  if (values) {
    for (let k = 0; k < plan.frames; k++) {
      const base = (k + 1) * stride / 8, previous = k * stride / 8;
      const fresh: { name: PassName; begin: bigint; end: bigint }[] = [], rewritten: { name: PassName; end: bigint }[] = [];
      for (const [name, slot] of passes) {
        const a = values[base + slot], b = values[base + slot + 1], pa = values[previous + slot], pb = values[previous + slot + 1];
        const changed = a !== pa && b !== pb, valid = a !== 0n && b !== 0n && b >= a;
        // The dispatch count is a CPU-side fact; slot equality alone misses old values that
        // the implementation converts again (observed on apple/metal-3: same duration, new value).
        const state: SampleState = name === 'deformation' && dispatches[k] === 0 ? 'noDispatch' : !valid ? 'invalid' : changed ? 'fresh' : 'stale';
        const record = integrity.get(name)!;
        record[state]++;
        record.pattern += state === 'noDispatch' ? (changed ? 'r' : 'n') : state[0];
        if (state !== 'fresh') {
          record.example ??= { state, frame: frameIds[k], begin: String(a), end: String(b), previousBegin: String(pa), previousEnd: String(pb) };
          if (state === 'noDispatch' && changed) {
            record.noDispatchRewritten++;
            if (valid) { rewrittenDurations.get(name)!.push(Number(b - a) / 1e6); rewritten.push({ name, end: b }); }
          }
          continue;
        }
        fresh.push({ name, begin: a, end: b });
        perPass.get(name)!.push(Number(b - a) / 1e6);
        for (const value of [a, b]) {
          if (origin === null) origin = value;
          else gcd = bigGcd(gcd, value > origin ? value - origin : origin - value);
        }
      }
      const shadow = fresh.find(entry => entry.name === 'shadow');
      if (shadow) for (const entry of rewritten) rewrittenAges.get(entry.name)!.push(Number(shadow.begin - entry.end) / 1e6);
      if (!SPAN_REQUIRED.every(name => fresh.some(entry => entry.name === name))) continue;
      let begin = fresh[0].begin, end = fresh[0].end, sum = 0;
      for (const entry of fresh) {
        if (entry.begin < begin) begin = entry.begin;
        if (entry.end > end) end = entry.end;
        sum += Number(entry.end - entry.begin) / 1e6;
        integrity.get(entry.name)!.spanFrames++;
      }
      spans.push(Number(end - begin) / 1e6); sums.push(sum);
    }
    // Harness self-check of slot indexing: the renderer's own every-30-frames sample
    // reads the same slots, so it must equal ours for that frame (stale or not).
    const k = sampleFrame === null ? -1 : frameIds.indexOf(sampleFrame);
    if (sample && k >= 0) {
      const base = (k + 1) * stride / 8, ms = (slot: number) => Number(values[base + slot + 1] - values[base + slot]) / 1e6;
      const pairs = [[sample.deformation, ms(0)], [sample.shadow, ms(2)], [sample.surface, ms(4)], [sample.resolve, ms(6)],
        ...(count === 10 ? [[sample.geometryLod, ms(8)]] : [])];
      crossCheck = { frame: sampleFrame!, maxAbsDiffMs: Math.max(...pairs.map(([a, b]) => Math.abs(a - b))) };
    }
  }
  for (const [name, record] of integrity) if (record.noDispatchRewritten) {
    record.noDispatchRewrittenMs = { duration: summarize(rewrittenDurations.get(name)!), ageBeforeShadow: summarize(rewrittenAges.get(name)!) };
  }
  const byPass = Object.fromEntries(integrity) as Partial<Record<PassName, PassIntegrity>>;
  const records = [...integrity.values()], nonFresh = records.some(record => record.fresh < plan.frames);
  return {
    label: G05_DIAGNOSTIC,
    timestamps: querySet ? 'per-pass timestamp-query' : 'unavailable: CPU submit and serialized frame time only',
    timestampGranularityNs: values && gcd > 0n ? Number(gcd) : null,
    plan, measuredFrames: frameIds.length, backpressureRetries: retries,
    invalidPassSamples: records.reduce((total, record) => total + record.invalid, 0),
    stalePassSamples: records.reduce((total, record) => total + record.stale, 0),
    /** Passes without a single fresh sample: no GPU time is reported for them. */
    unavailablePasses: values ? [...integrity].filter(([, record]) => record.fresh === 0).map(([name]) => name) : [],
    timestampIntegrity: values ? { rule: TIMESTAMP_RULE, spanRule: SPAN_RULE, notes: nonFresh ? [STALE_NOTE] : [], byPass } : null,
    gpuPassMs: Object.fromEntries([...perPass].map(([name, list]) => [name, summarize(list)])) as Partial<Record<PassName, G05Summary | null>>,
    gpuFrameSpanMs: summarize(spans), gpuPassSumMs: summarize(sums),
    cpuSubmitMs: summarize(cpuSubmitMs), serializedFrameMs: summarize(serialized),
    samples: { surfaceMs: (perPass.get('surface') ?? []).map(value => round(value, 5)), frameSpanMs: spans.map(value => round(value, 5)) },
    crossCheck,
  };
}
export type G05Timing = Awaited<ReturnType<typeof measureFrames>>;
const median = (timing: G05Timing, pass: PassName) => timing.gpuPassMs[pass]?.median ?? null;

function fit(points: readonly (readonly [number, number | null])[]) {
  const valid = points.filter((point): point is [number, number] => point[1] !== null);
  if (valid.length < 2) return null;
  const mx = valid.reduce((s, [x]) => s + x, 0) / valid.length, my = valid.reduce((s, [, y]) => s + y, 0) / valid.length;
  const sxx = valid.reduce((s, [x]) => s + (x - mx) ** 2, 0), sxy = valid.reduce((s, [x, y]) => s + (x - mx) * (y - my), 0);
  const slope = sxy / sxx, intercept = my - slope * mx;
  const total = valid.reduce((s, [, y]) => s + (y - my) ** 2, 0), residual = valid.reduce((s, [x, y]) => s + (y - intercept - slope * x) ** 2, 0);
  return { slope: round(slope, 6), intercept: round(intercept, 5), r2: total > 0 ? round(1 - residual / total, 4) : null };
}

// ---------------------------------------------------------------------------
// Case 1: micro-triangles on a dense closed sphere.

const SPHERE = 'g05-sphere';
const SPHERE_IDENTITY = 200;
const HIERARCHY_OPTIONS = { maxClusterTriangles: 128, groupSize: 8, simplifyRatio: 0.5, maxLevels: 16,
  maxVertices: 4_000_000, maxTriangles: 4_000_000, maxOutputTriangles: 8_000_000, maxClusters: 262_144, maxGroups: 262_144 };

/** Welded unit cube-sphere with tangent-warped face grids: 6n² + 2 vertices, 12n² outward triangles. */
function cubeSphere(n: number): MeshData {
  const side = n + 1, vertexCount = 6 * n * n + 2, vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
  const indices = new Uint32Array(36 * n * n), positions = new Float64Array(vertexCount * 3), ids = new Map<number, number>();
  const warp = (coordinate: number) => Math.tan((2 * coordinate / n - 1) * Math.PI / 4);
  let next = 0, written = 0;
  // Integer lattice coordinates on the cube surface weld shared face edges exactly.
  const vertex = (lx: number, ly: number, lz: number) => {
    const key = (lx * side + ly) * side + lz;
    let id = ids.get(key);
    if (id === undefined) {
      id = next++;
      const x = warp(lx), y = warp(ly), z = warp(lz), length = Math.hypot(x, y, z), p = [x / length, y / length, z / length];
      for (let axis = 0; axis < 3; axis++) {
        positions[id * 3 + axis] = p[axis];
        view.setFloat32(id * 80 + axis * 4, p[axis], true);
        view.setFloat32(id * 80 + 16 + axis * 4, p[axis], true);
      }
      view.setFloat32(id * 80 + 12, 1, true); view.setFloat32(id * 80 + 48, 1, true);
      ids.set(key, id);
    }
    return id;
  };
  // Orients each triangle outward: its normal must point away from the centre.
  const emit = (a: number, b: number, c: number) => {
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const e1x = positions[b * 3] - ax, e1y = positions[b * 3 + 1] - ay, e1z = positions[b * 3 + 2] - az;
    const e2x = positions[c * 3] - ax, e2y = positions[c * 3 + 1] - ay, e2z = positions[c * 3 + 2] - az;
    const outward = (e1y * e2z - e1z * e2y) * ax + (e1z * e2x - e1x * e2z) * ay + (e1x * e2y - e1y * e2x) * az > 0;
    indices[written++] = a; indices[written++] = outward ? b : c; indices[written++] = outward ? c : b;
  };
  const grid = new Uint32Array(side * side), lattice = [0, 0, 0];
  for (let face = 0; face < 6; face++) {
    const normalAxis = face >> 1, a = (normalAxis + 1) % 3, b = (normalAxis + 2) % 3;
    lattice[normalAxis] = face & 1 ? n : 0;
    for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) {
      lattice[a] = i; lattice[b] = j; grid[j * side + i] = vertex(lattice[0], lattice[1], lattice[2]);
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v00 = grid[j * side + i], v10 = grid[j * side + i + 1], v01 = grid[(j + 1) * side + i], v11 = grid[(j + 1) * side + i + 1];
      emit(v00, v10, v11); emit(v00, v11, v01);
    }
  }
  if (next !== vertexCount || written !== indices.length) throw new Error('G05_SPHERE_TOPOLOGY');
  return { vertices, indices, vertexCount, bounds: { min: [-1, -1, -1], max: [1, 1, 1] } };
}
/** Camera at a fixed oblique direction, aimed at the sphere centre: the exact
 * silhouette is a circle about the principal point. */
function sphereView(diameterPx: number, height: number) {
  const radiusPx = diameterPx / 2, focal = height / (2 * TAN_HALF), distance = 1 / Math.sin(Math.atan(radiusPx / focal));
  const direction = unit([0.31, 0.23, 0.92]);
  const view: View = { eye: [direction[0] * distance, direction[1] * distance, direction[2] * distance], target: [0, 0, 0] };
  return { view, radiusPx, distance };
}
function sourcePositions(mesh: MeshData): Float32Array {
  const view = new DataView(mesh.vertices), positions = new Float32Array(mesh.vertexCount * 3);
  for (let v = 0; v < mesh.vertexCount; v++) for (let axis = 0; axis < 3; axis++) positions[v * 3 + axis] = view.getFloat32(v * 80 + axis * 4, true);
  return positions;
}
function frontFacing(positions: Float32Array, eye: Vec3, a: number, b: number, c: number): boolean {
  const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
  const e1x = positions[b * 3] - ax, e1y = positions[b * 3 + 1] - ay, e1z = positions[b * 3 + 2] - az;
  const e2x = positions[c * 3] - ax, e2y = positions[c * 3 + 1] - ay, e2z = positions[c * 3 + 2] - az;
  const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
  return nx * (eye[0] - ax) + ny * (eye[1] - ay) + nz * (eye[2] - az) > 0;
}
/** Rendered sphere mask against the analytic circle; mismatches farther than the band from it are failures. */
function silhouette(labels: Int16Array, width: number, height: number, radiusPx: number, bandPx: number) {
  const cx = width / 2, cy = height / 2, marked = new Uint8Array(width * height), drawnMask = new Uint8Array(width * height);
  let covered = 0, analytic = 0, intersection = 0, union = 0, mismatched = 0, beyond = 0, maxDistance = 0, unclassified = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const p = y * width + x, label = labels[p];
    if (label === UNCLASSIFIED) { unclassified++; marked[p] = 1; continue; }
    const r = Math.hypot(x + 0.5 - cx, y + 0.5 - cy), inside = r < radiusPx, drawn = label === 0;
    if (drawn) { covered++; drawnMask[p] = 1; }
    if (inside) analytic++;
    if (drawn && inside) intersection++;
    if (drawn || inside) union++;
    if (drawn !== inside) {
      mismatched++;
      const distance = Math.abs(r - radiusPx);
      maxDistance = Math.max(maxDistance, distance);
      if (distance > bandPx) { beyond++; marked[p] = 1; }
    }
  }
  return { marked, drawnMask, result: { bandPx, coveredPixels: covered, analyticPixels: analytic, mismatchedPixels: mismatched,
    beyondBand: beyond, maxMismatchDistancePx: round(maxDistance, 3), iouAnalytic: round(intersection / Math.max(1, union), 6),
    unclassifiedPixels: unclassified } };
}
async function settleStream(renderer: Renderer, view: View, maxFrames: number, stableFrames: number) {
  const device = (renderer as unknown as RendererInternals).device, started = performance.now();
  let stable = 0, key = '', frames = 0;
  while (frames < maxFrames) {
    await renderFrame(renderer, view); frames++;
    await idle(device);
    const stream = renderer.diagnostics().geometryStreams[0];
    if (!stream) throw new Error('G05_STREAM_MISSING');
    if (stream.error) throw new Error('G05_STREAM_ERROR:' + stream.error);
    const resident = new Set(stream.pool.residentPages);
    // Settled: nothing loading, uploading or migrating, every admitted page resident, unchanged for stableFrames.
    const busy = stream.pool.pendingPages.length > 0 || stream.allocation.pages.some(page => !resident.has(page))
      || !!stream.cache.activeRequests || !!stream.cache.queuedRequests || !!stream.capacity.preparing || !!stream.capacity.retiringOwners;
    const next = stream.pool.residentPages.join(',');
    if (busy || next !== key) { stable = 0; key = next; } else if (++stable >= stableFrames) {
      return { frames, ms: round(performance.now() - started, 1), residentPages: resident.size, admittedPages: stream.allocation.pages.length,
        requestedPages: stream.allocation.requestedPages.length, budgetDeferredGroups: stream.allocation.budgetDeferredGroups.length,
        uploads: stream.pool.uploads, evictions: stream.pool.evictions };
    }
  }
  throw new Error('G05_STREAM_DID_NOT_SETTLE:' + frames);
}
const streamUploads = (renderer: Renderer) => renderer.diagnostics().geometryStreams[0]?.pool.uploads ?? 0;

type MicroVariant = 'full' | 'gpu-lod' | 'stream';
type MicroContext = { readonly options: G05MicroOptions; readonly hierarchy: GeometryHierarchy; readonly positions: Float32Array;
  readonly roots: ReadonlySet<number>; readonly sourceTriangles: number; readonly table: Int16Array;
  readonly fullMasks: Map<number, Uint8Array>; readonly evidence: Record<string, string>; readonly largestDiameter: number };

/** One sphere view: settle (paged), capture, silhouette check, drawn-cut statistics, diagnostic timing. */
async function sphereRun(renderer: Renderer, context: MicroContext, variant: MicroVariant, diameterPx: number, maxErrorPixels: number | null) {
  const { options, hierarchy, positions } = context, { width, height } = options, source = hierarchy.source;
  const { view, radiusPx, distance } = sphereView(diameterPx, height);
  if (maxErrorPixels !== null) renderer.setGeometryLod({ maxErrorPixels, frustumCulling: true });
  const settle = variant === 'stream' ? await settleStream(renderer, view, options.maxSettleFrames, options.stableFrames) : null;
  for (let i = 0; i < options.frames.warmup; i++) await renderFrame(renderer, view);
  if (renderer.stats.width !== width || renderer.stats.height !== height) throw new Error(`G05_VIEWPORT:${renderer.stats.width}x${renderer.stats.height}`);
  const probe = maxErrorPixels === null ? null : await renderer.probeGeometryLod(SPHERE);
  const image = await readHdr(renderer);
  const bandPx = maxErrorPixels ? Math.ceil(1.5 * maxErrorPixels) + 2 : 1;
  const check = silhouette(classify(image, context.table), width, height, radiusPx, bandPx);
  if (variant === 'full') context.fullMasks.set(diameterPx, check.drawnMask);
  const full = context.fullMasks.get(diameterPx);
  let versusFull: { differingPixels: number; iou: number } | null = null;
  if (full && variant !== 'full') {
    let differing = 0, both = 0, either = 0;
    for (let p = 0; p < full.length; p++) {
      const a = full[p] === 1, b = check.drawnMask[p] === 1;
      if (a !== b) differing++;
      if (a && b) both++;
      if (a || b) either++;
    }
    versusFull = { differingPixels: differing, iou: round(both / Math.max(1, either), 6) };
  }
  let facing = 0, selected = 0, rootSelected = 0, fallback = 0;
  const levels: Record<number, number> = {};
  if (probe) {
    for (const cluster of probe.selectedClusters) {
      if (cluster.instanceIndex !== 0) continue;
      const data = hierarchy.clusters[cluster.clusterId];
      selected++; levels[data.level] = (levels[data.level] ?? 0) + 1;
      if (context.roots.has(cluster.clusterId)) rootSelected++;
      if (cluster.fallback) fallback++;
      for (let t = 0; t < data.triangles.length; t += 3) {
        if (frontFacing(positions, view.eye, data.vertices[data.triangles[t]], data.vertices[data.triangles[t + 1]], data.vertices[data.triangles[t + 2]])) facing++;
      }
    }
  } else {
    for (let t = 0; t < source.indices.length; t += 3) if (frontFacing(positions, view.eye, source.indices[t], source.indices[t + 1], source.indices[t + 2])) facing++;
  }
  const drawnTriangles = probe ? probe.triangles : context.sourceTriangles, covered = Math.max(1, check.result.coveredPixels);
  const label = `${variant}${maxErrorPixels === null ? '' : `-${maxErrorPixels}px`}-d${diameterPx}`;
  if ((diameterPx === context.largestDiameter || check.result.beyondBand || check.result.unclassifiedPixels)
    && Object.keys(context.evidence).length < 6) context.evidence[label] = encodePng(image, check.marked);
  const uploadsBefore = streamUploads(renderer);
  const timing = await measureFrames(renderer, view, options.frames);
  return { label, variant, maxErrorPixels, diameterPx, silhouetteRadiusPx: radiusPx, distance: round(distance, 5),
    eye: view.eye.map(value => round(value, 5)),
    drawnTriangles, frontFacingTriangles: facing,
    trianglesPerCoveredPixel: round(drawnTriangles / covered, 4), frontFacingTrianglesPerCoveredPixel: round(facing / covered, 4),
    coveredPixelsPerFrontFacingTriangle: round(covered / Math.max(1, facing), 4),
    probe: probe && { frame: probe.frame, triangles: probe.triangles, overflow: probe.overflow, selectedClusters: selected,
      rootClustersSelected: rootSelected, fallbackClusters: fallback, levels },
    silhouette: { ...check.result, versusFull },
    stream: settle && { settle, uploadsDuringMeasurement: streamUploads(renderer) - uploadsBefore,
      backpressureFrames: renderer.stats.streamBackpressureFrames },
    drawCalls: renderer.stats.drawCalls, timing };
}
type MicroRun = Awaited<ReturnType<typeof sphereRun>>;

export async function runG05MicroTriangles(options: G05MicroOptions) {
  validateCommon(options);
  const n = integer(options.subdivisions, 8, 512, 'G05_SPHERE_SUBDIVISIONS'), { width, height } = options;
  const diameters = ascending(options.diametersPx, 16, Math.min(width, height) - 16, 'G05_SPHERE_DIAMETERS');
  integer(options.maxSettleFrames, 30, 5000, 'G05_SETTLE_FRAMES'); integer(options.stableFrames, 4, 240, 'G05_STABLE_FRAMES');
  let started = performance.now();
  const mesh = cubeSphere(n);
  const meshMs = performance.now() - started; started = performance.now();
  const hierarchy: GeometryHierarchy = await buildGeometryHierarchy({ primitiveKey: `g05-cube-sphere-${n}`, mesh, coverage: 'opaque' }, HIERARCHY_OPTIONS);
  const hierarchyMs = performance.now() - started;
  const source = hierarchy.source, sourceTriangles = hierarchy.sourceTriangleCount, positions = sourcePositions(source);
  const roots = new Set(hierarchy.roots);
  let stream: { layout: ReturnType<typeof geometryGpuPoolLayout>; poolBytes: number; batch: () => SceneBatch; pages: number; rootPages: number; buildMs: number } | null = null;
  if (options.stream) {
    started = performance.now();
    const encoded = await encodeGeometryPages(hierarchy, { maxTotalBytes: 512 * MiB });
    const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
    const layout = geometryGpuPoolLayout(directory);
    // Large enough for the settled 1 px cut of every tested view; never below the pinned roots.
    const poolBytes = Math.min(256 * MiB, layout.fullBytes, Math.max(64 * MiB, 2 * layout.rootBytes));
    if (poolBytes < layout.rootBytes) throw new RangeError('G05_STREAM_ROOTS_EXCEED_POOL');
    const payloads = encoded.pages.map(page => page.bytes);
    const provider: GeometryPageProvider = async (page, signal) => {
      signal.throwIfAborted();
      const bytes = payloads[page.id];
      if (!bytes) throw new RangeError('G05_STREAM_PAGE');
      return bytes;
    };
    stream = { layout, poolBytes, pages: directory.pages.length, rootPages: directory.rootPages.length, buildMs: round(performance.now() - started, 1),
      batch: () => ({ ...sphereBatch(), geometryStream: { directory, provider, maxRequestsPerFrame: 64,
        cpuCache: { maxResidentBytes: 128 * MiB, maxInFlightBytes: 64 * MiB, maxConcurrentRequests: 8 },
        gpuPool: { maxResidentBytes: poolBytes, maxStagingBytes: 64 * MiB, maxQueuedPages: 1024, maxUploadBytesPerFrame: 16 * MiB,
          maxUploadWritesPerFrame: 1024, maxUploadPagesPerFrame: 64, maxFramesInFlight: 4 } } }) };
  }
  function sphereBatch(): SceneBatch {
    return staticBatch(SPHERE, source, [instance(501, identity())], { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.parameter(0) },
      parameters: new Float32Array(g05IdentityColor(SPHERE_IDENTITY)) });
  }
  const variants: { variant: MicroVariant; batch: () => SceneBatch; errors: (number | null)[] }[] = [
    { variant: 'full', batch: sphereBatch, errors: [null] },
    { variant: 'gpu-lod', batch: () => ({ ...sphereBatch(), geometryLod: hierarchy }), errors: [0, 1] },
    ...(stream ? [{ variant: 'stream' as const, batch: stream.batch, errors: [1] }] : []),
  ];
  const context: MicroContext = { options, hierarchy, positions, roots, sourceTriangles, table: labelTable([SPHERE_IDENTITY]),
    fullMasks: new Map(), evidence: {}, largestDiameter: diameters[diameters.length - 1] };
  const cleanup: Cleanup[] = [], runs: MicroRun[] = [], meta: { head?: Header } = {};
  for (const entry of variants) {
    await withRenderer(entry.variant, width, height, [entry.batch()], {}, cleanup, async renderer => {
      meta.head ??= header(renderer);
      for (const diameterPx of diameters) for (const maxErrorPixels of entry.errors) {
        runs.push(await sphereRun(renderer, context, entry.variant, diameterPx, maxErrorPixels));
      }
    });
  }
  const rootTriangles = hierarchy.roots.reduce((sum, id) => sum + hierarchy.clusters[id].triangles.length / 3, 0);
  const packedTriangles = hierarchy.clusters.reduce((sum, cluster) => sum + cluster.triangles.length / 3, 0);
  return { case: 'micro-triangles' as const, profile: options.profile, ...meta.head!, viewport: { width, height }, frames: options.frames,
    mesh: { kind: 'welded tangent-warped cube-sphere, radius 1', subdivisions: n, sourceTriangles, sourceVertices: source.vertexCount,
      meshMs: round(meshMs, 1), hierarchyMs: round(hierarchyMs, 1), hierarchyOptions: HIERARCHY_OPTIONS,
      hierarchy: { clusters: hierarchy.clusters.length, groups: hierarchy.groups.length, roots: hierarchy.roots.length, rootTriangles,
        packedTriangles, packedTrianglesOverSource: round(packedTriangles / sourceTriangles, 4),
        rootTriangleFraction: round(rootTriangles / sourceTriangles, 5), levels: Math.max(...hierarchy.clusters.map(cluster => cluster.level)) + 1 } },
    stream: stream && { pages: stream.pages, rootPages: stream.rootPages, poolBytes: stream.poolBytes, rootBytes: stream.layout.rootBytes,
      fullBytes: stream.layout.fullBytes, bytesPerSlot: stream.layout.bytesPerSlot, buildMs: stream.buildMs },
    notes: [
      'Silhouette reference: the exact sphere circle (the camera aims at the centre); mismatches within the band of it are allowed rasterization or LOD deviation.',
      'Band: 1 px for exact detail, ceil(1.5 x maxErrorPixels) + 2 px for a 1 px LOD or paged cut (the streamed-gallery nominal band).',
      'Shadows always draw full detail (shared mesh); GPU LOD and paging change only the surface pass.',
      'Front-facing triangles are counted on the CPU from the drawn cut; per-pixel ratios divide by rendered sphere pixels.',
    ],
    runs, evidence: context.evidence, cleanup };
}

// ---------------------------------------------------------------------------
// Case 2: overlapping thin sheets along the view axis.

const SHEETS = 'g05-sheets';
const MAX_SHEETS = 64;
const ORDERS = ['front-to-back', 'back-to-front'] as const;
type Order = typeof ORDERS[number];
const QUARTER_TURNS: readonly [number, number][] = [[1, 0], [0, 1], [-1, 0], [0, -1]];

/** Unit quad in the XY plane facing +Z, counter-clockwise; v runs downward. */
function sheetMesh(): MeshData {
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices), uvs = new Float32Array(8);
  [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].forEach(([x, y], i) => {
    view.setFloat32(i * 80, x, true); view.setFloat32(i * 80 + 4, y, true); view.setFloat32(i * 80 + 12, 1, true);
    view.setFloat32(i * 80 + 24, 1, true); view.setFloat32(i * 80 + 48, 1, true);
    uvs[i * 2] = x + 0.5; uvs[i * 2 + 1] = 0.5 - y;
  });
  return { vertices, uvs, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4, bounds: { min: [-0.5, -0.5, 0], max: [0.5, 0.5, 0] } };
}
/** Sheet i: one of the eight square symmetries (exact quarter turns, optional mirror) at z = -i·spacing. */
function sheetPlacement(index: number, size: number, spacing: number) {
  const [c, s] = QUARTER_TURNS[index % 4], sx = (index >> 2) & 1 ? -size : size, sy = size, z = -index * spacing;
  return { c, s, sx, sy, z, model: new Float32Array([c * sx, s * sx, 0, 0, -s * sy, c * sy, 0, 0, 0, 0, 1, 0, 0, 0, z, 1]) };
}
type Sheet = ReturnType<typeof sheetPlacement>;
function cutoutTexture(seed: number, texels: number, holeFraction: number) {
  const random = mulberry32(seed), solid = new Uint8Array(texels * texels), pixels = new Uint8Array(texels * texels * 4);
  for (let t = 0; t < solid.length; t++) {
    solid[t] = random() >= holeFraction ? 1 : 0;
    pixels[t * 4] = 255; pixels[t * 4 + 1] = 255; pixels[t * 4 + 2] = 255; pixels[t * 4 + 3] = solid[t] ? 255 : 0;
  }
  const asset: TextureAsset = { width: texels, height: texels, pixels, colorSpace: 'linear' };
  return { asset, solid, texels, holes: solid.length - solid.reduce((sum, value) => sum + value, 0) };
}
/** Expected label per pixel centre plus test-only analytic fragment estimates. Sheets
 * are centred and equally sized, so deeper sheets project strictly inside nearer ones. */
function sheetReference(width: number, height: number, distance: number, sheets: readonly Sheet[], layers: number,
  masked: boolean, texture: ReturnType<typeof cutoutTexture>) {
  const expected = new Int16Array(width * height), aspect = width / height, texels = texture.texels;
  let rasterized = 0, covered = 0, frontToBack = 0;
  for (let y = 0; y < height; y++) {
    const dy = (1 - ((y + 0.5) / height) * 2) * TAN_HALF;
    for (let x = 0; x < width; x++) {
      const dx = (((x + 0.5) / width) * 2 - 1) * TAN_HALF * aspect;
      let label = BACKGROUND, covering = 0, shaded = 0;
      for (let i = 0; i < layers; i++) {
        const sheet = sheets[i], t = distance - sheet.z, wx = t * dx, wy = t * dy;
        const lx = (sheet.c * wx + sheet.s * wy) / sheet.sx, ly = (-sheet.s * wx + sheet.c * wy) / sheet.sy;
        if (Math.abs(lx) > 0.5 || Math.abs(ly) > 0.5) break;
        covering++;
        if (label !== BACKGROUND) continue;
        // Front-to-back with ideal early depth: this fragment is shaded because no nearer sheet wrote depth here.
        shaded++;
        if (!masked) { label = i; continue; }
        const tx = Math.min(texels - 1, Math.max(0, Math.floor((lx + 0.5) * texels)));
        const ty = Math.min(texels - 1, Math.max(0, Math.floor((0.5 - ly) * texels)));
        if (texture.solid[ty * texels + tx]) label = i;
      }
      expected[y * width + x] = label; rasterized += covering; frontToBack += shaded;
      if (covering) covered++;
    }
  }
  return { expected, estimate: { label: 'test-only analytic estimate at pixel centres; not a GPU counter', rasterizedFragments: rasterized,
    coveredPixels: covered, depthComplexity: round(rasterized / Math.max(1, covered), 3),
    idealEarlyZShadedFragments: { 'front-to-back': frontToBack, 'back-to-front': rasterized } as Record<Order, number> } };
}

type SheetContext = { readonly options: G05SheetOptions; readonly batch: SceneBatch; readonly states: readonly InstanceState[];
  readonly table: Int16Array; readonly view: View; readonly evidence: Record<string, string>; readonly maxLayers: number };
type SheetVariant = 'opaque' | 'masked';

/** One sheet stack in one submission order: capture, label check against the reference, diagnostic timing. */
async function sheetRun(renderer: Renderer, context: SheetContext, variant: SheetVariant, layers: number, order: Order,
  reference: ReturnType<typeof sheetReference>) {
  const { options, batch, states, view } = context, { width, height } = options;
  batch.instances = order === 'front-to-back' ? states.slice(0, layers) : states.slice(0, layers).reverse();
  for (let i = 0; i < options.frames.warmup; i++) await renderFrame(renderer, view);
  if (renderer.stats.width !== width || renderer.stats.height !== height) throw new Error(`G05_VIEWPORT:${renderer.stats.width}x${renderer.stats.height}`);
  const robust = robustMask(reference.expected, width, height);
  const image = await readHdr(renderer), labels = classify(image, context.table), marked = new Uint8Array(labels.length);
  let robustPixels = 0, expectedNonFront = 0, mismatches = 0, unclassified = 0, front = 0, nonFront = 0, background = 0, deepest = -1;
  for (let p = 0; p < labels.length; p++) {
    const label = labels[p];
    robustPixels += robust[p];
    if (reference.expected[p] > 0) expectedNonFront++;
    if (label === UNCLASSIFIED) { unclassified++; marked[p] = 1; }
    else if (label === BACKGROUND) background++;
    else if (label === 0) front++;
    else { nonFront++; deepest = Math.max(deepest, label); }
    if (robust[p] && label !== reference.expected[p]) { mismatches++; marked[p] = 1; }
  }
  const key = `${variant}-n${layers}-${order}`;
  const showcase = (variant === 'masked' && layers === 8 && order === 'front-to-back') || (layers === context.maxLayers && order === 'back-to-front');
  if ((showcase || mismatches || unclassified) && Object.keys(context.evidence).length < 6) context.evidence[key] = encodePng(image, marked);
  const timing = await measureFrames(renderer, view, options.frames);
  const run = { label: key, variant, layers, order, triangles: layers * 2,
    correctness: { robustPixels, robustMismatches: mismatches, unclassifiedPixels: unclassified, frontSheetPixels: front,
      nonFrontSheetPixels: nonFront, backgroundPixels: background, deepestVisibleSheet: deepest, expectedNonFrontPixels: expectedNonFront,
      /** Filled once both orders have rendered. */
      differsFromOtherOrder: -1 },
    estimate: { label: reference.estimate.label, rasterizedFragments: reference.estimate.rasterizedFragments,
      coveredPixels: reference.estimate.coveredPixels, depthComplexity: reference.estimate.depthComplexity,
      idealEarlyZShadedFragments: reference.estimate.idealEarlyZShadedFragments[order] },
    drawCalls: renderer.stats.drawCalls, timing };
  return { run, labels };
}
type SheetRun = Awaited<ReturnType<typeof sheetRun>>['run'];

export async function runG05ThinSheets(options: G05SheetOptions) {
  validateCommon(options);
  const { width, height } = options, layerCounts = ascending(options.layers, 1, MAX_SHEETS, 'G05_SHEET_LAYERS');
  integer(options.seed, 0, 0xffffffff, 'G05_SEED');
  const distance = 3, spacing = 0.01, size = 0.8 * 2 * distance * TAN_HALF, holeFraction = 0.35;
  const texture = cutoutTexture(options.seed, 16, holeFraction);
  const sheets = Array.from({ length: MAX_SHEETS }, (_, i) => sheetPlacement(i, size, spacing));
  const identities = sheets.map((_, i) => (97 * i + 13) % 343), table = labelTable(identities);
  const states = sheets.map((sheet, i) => instance(9000 + i, sheet.model, g05IdentityColor(identities[i])));
  const view: View = { eye: [0, 0, distance], target: [0, 0, 0] };
  const sampler = { magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' } as const;
  const material = (alphaMode: 'opaque' | 'mask'): MaterialBinding => ({ definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', alphaMode,
    doubleSided: true, alphaCutoff: 0.5, baseColor: m.multiply(m.texture(0), m.instanceColor) }, textures: [texture.asset], samplers: [sampler] });
  const batch = staticBatch(SHEETS, sheetMesh(), states.slice(0, layerCounts[0]), material('opaque'), MAX_SHEETS);
  const context: SheetContext = { options, batch, states, table, view, evidence: {}, maxLayers: layerCounts[layerCounts.length - 1] };
  const cleanup: Cleanup[] = [], runs: SheetRun[] = [], meta: { head?: Header } = {};
  await withRenderer('sheets', width, height, [batch], {}, cleanup, async renderer => {
    meta.head = header(renderer);
    for (const variant of ['opaque', 'masked'] as const) {
      if (variant === 'masked') await renderer.setMaterial(SHEETS, material('mask'));
      for (const layers of layerCounts) {
        const reference = sheetReference(width, height, distance, sheets, layers, variant === 'masked', texture);
        const pair: Awaited<ReturnType<typeof sheetRun>>[] = [];
        for (const order of ORDERS) pair.push(await sheetRun(renderer, context, variant, layers, order, reference));
        // Depth testing makes the result order-independent: both submissions must give identical labels.
        const [a, b] = pair.map(entry => entry.labels);
        let differing = 0;
        for (let p = 0; p < a.length; p++) if (a[p] !== b[p]) differing++;
        for (const { run } of pair) { run.correctness.differsFromOtherOrder = differing; runs.push(run); }
      }
    }
  });
  const evidence = context.evidence;
  const scaling = (['opaque', 'masked'] as const).flatMap(variant => ORDERS.map(order => {
    const selected = runs.filter(run => run.variant === variant && run.order === order);
    const surface = selected.map(run => [run.layers, median(run.timing, 'surface')] as const);
    const first = selected[0], last = selected[selected.length - 1], firstMs = median(first.timing, 'surface'), lastMs = median(last.timing, 'surface');
    return { variant, order, label: G05_DIAGNOSTIC,
      surfaceMedianMsByLayers: Object.fromEntries(surface),
      frameSpanMedianMsByLayers: Object.fromEntries(selected.map(run => [run.layers, run.timing.gpuFrameSpanMs?.median ?? null])),
      shadowMedianMsByLayers: Object.fromEntries(selected.map(run => [run.layers, median(run.timing, 'shadow')])),
      surfaceFitMsPerLayer: fit(surface),
      surfaceMaxOverMinLayers: firstMs && lastMs ? round(lastMs / firstMs, 3) : null,
      surfaceNsPerEstimatedShadedFragmentAtMaxLayers: lastMs === null ? null
        : round(lastMs * 1e6 / Math.max(1, last.estimate.idealEarlyZShadedFragments), 4),
      surfaceNsPerRasterizedFragmentAtMaxLayers: lastMs === null ? null : round(lastMs * 1e6 / Math.max(1, last.estimate.rasterizedFragments), 4) };
  }));
  const at = (variant: string, order: string) => scaling.find(entry => entry.variant === variant && entry.order === order)!;
  const ratio = (a: number | null | undefined, b: number | null | undefined) => a && b ? round(a / b, 3) : null;
  const maxLayers = layerCounts[layerCounts.length - 1];
  const surfaceAt = (variant: string, order: string) => (at(variant, order).surfaceMedianMsByLayers as Record<number, number | null>)[maxLayers];
  return { case: 'thin-sheets' as const, profile: options.profile, ...meta.head!, viewport: { width, height }, frames: options.frames,
    sheets: { size: round(size, 5), spacing, distance, frontCoverageOfHeight: 0.8, symmetries: 'exact quarter turns x optional mirror (8 patterns)',
      texture: { texels: 16, seed: options.seed, holeFraction, holes: texture.holes, sampling: 'nearest, clamp-to-edge' } },
    renderer: { source: 'src/renderer.ts and src/shaders/surface.wgsl inspection, not measured', depthPrepass: false, depthCompare: 'less',
      depthWrite: true, maskCoverage: 'fragment discard when baseColor alpha < cutoff', submission: 'one instanced draw; instance order is the submission order' },
    scaling,
    comparisonsAtMaxLayers: { layers: maxLayers, label: G05_DIAGNOSTIC,
      opaqueBackToFrontOverFrontToBack: ratio(surfaceAt('opaque', 'back-to-front'), surfaceAt('opaque', 'front-to-back')),
      maskedBackToFrontOverFrontToBack: ratio(surfaceAt('masked', 'back-to-front'), surfaceAt('masked', 'front-to-back')),
      maskedOverOpaqueFrontToBack: ratio(surfaceAt('masked', 'front-to-back'), surfaceAt('opaque', 'front-to-back')),
      maskedOverOpaqueBackToFront: ratio(surfaceAt('masked', 'back-to-front'), surfaceAt('opaque', 'back-to-front')) },
    runs, evidence, cleanup };
}

// ---------------------------------------------------------------------------
// Case 3: one wall split into M materials/batches, shadows on.

const WALL = { x0: -2.4, x1: 2.4, y0: 1, y1: 4 } as const;
const WALL_VIEW: View = { eye: [0, 2.5, 6], target: [0, 1.6, 0] };
/** Grid of quads in the z = 0 plane facing +Z. */
function wallTile(x0: number, x1: number, y0: number, y1: number, qx: number, qy: number): MeshData {
  const vertexCount = (qx + 1) * (qy + 1), vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
  const indices = new Uint32Array(qx * qy * 6);
  for (let j = 0; j <= qy; j++) for (let i = 0; i <= qx; i++) {
    const v = j * (qx + 1) + i;
    view.setFloat32(v * 80, x0 + (x1 - x0) * i / qx, true); view.setFloat32(v * 80 + 4, y0 + (y1 - y0) * j / qy, true);
    view.setFloat32(v * 80 + 12, 1, true); view.setFloat32(v * 80 + 24, 1, true); view.setFloat32(v * 80 + 48, 1, true);
  }
  let o = 0;
  for (let j = 0; j < qy; j++) for (let i = 0; i < qx; i++) {
    const a = j * (qx + 1) + i, b = a + 1, c = a + qx + 1, d = c + 1;
    indices[o++] = a; indices[o++] = b; indices[o++] = d; indices[o++] = a; indices[o++] = d; indices[o++] = c;
  }
  return { vertices, indices, vertexCount, bounds: { min: [x0, y0, 0], max: [x1, y1, 0] } };
}
/** Lit receiver at y = 0 facing +Y. */
function floorMesh(): MeshData {
  const corners: Vec3[] = [[-8, 0, 9], [8, 0, 9], [8, 0, -4], [-8, 0, -4]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  corners.forEach((corner, i) => {
    for (let axis = 0; axis < 3; axis++) view.setFloat32(i * 80 + axis * 4, corner[axis], true);
    view.setFloat32(i * 80 + 12, 1, true); view.setFloat32(i * 80 + 20, 1, true); view.setFloat32(i * 80 + 48, 1, true);
  });
  return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), vertexCount: 4, bounds: { min: [-8, 0, -4], max: [8, 0, 9] } };
}
function tileGrid(materials: number): [number, number] {
  const mx = 2 ** Math.ceil(Math.log2(materials) / 2);
  return [mx, materials / mx];
}
/** Expected tile per pixel (BACKGROUND off the wall) and floor samples: 1 lit (the sun
 * ray passes under the wall), 2 shadowed (the sun ray hits the wall interior). */
function wallReference(width: number, height: number, mx: number, my: number) {
  const { eye, target } = WALL_VIEW, forward = unit(sub(target, eye)), right = unit(cross(forward, [0, 1, 0])), up = cross(right, forward);
  const expected = new Int16Array(width * height), floor = new Uint8Array(width * height), aspect = width / height;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = (((x + 0.5) / width) * 2 - 1) * TAN_HALF * aspect, sy = (1 - ((y + 0.5) / height) * 2) * TAN_HALF;
    const d = [0, 1, 2].map(axis => forward[axis] + right[axis] * sx + up[axis] * sy);
    let label = BACKGROUND;
    if (d[2] < 0) {
      const t = -eye[2] / d[2], wx = eye[0] + t * d[0], wy = eye[1] + t * d[1];
      if (wx >= WALL.x0 && wx <= WALL.x1 && wy >= WALL.y0 && wy <= WALL.y1) {
        const tx = Math.min(mx - 1, Math.floor((wx - WALL.x0) / (WALL.x1 - WALL.x0) * mx)), ty = Math.min(my - 1, Math.floor((wy - WALL.y0) / (WALL.y1 - WALL.y0) * my));
        label = ty * mx + tx;
      }
    }
    expected[y * width + x] = label;
    if (label === BACKGROUND && d[1] < 0) {
      const t = -eye[1] / d[1], fx = eye[0] + t * d[0], fz = eye[2] + t * d[2];
      // Sun direction (0, 1, -1): a floor point at depth z reaches the wall plane at height z.
      if (Math.abs(fx) <= 2 && fz >= 0.2 && fz <= 0.8) floor[y * width + x] = 1;
      else if (Math.abs(fx) <= 2 && fz >= 1.2 && fz <= 2.5) floor[y * width + x] = 2;
    }
  }
  return { expected, floor };
}

type MaterialLayout = { materials: number; mx: number; my: number; qx: number; qy: number; identities: readonly number[];
  batches: number; createMs: number; showcase: boolean };

/** One material count: capture, per-tile identity and floor-shadow checks, diagnostic timing. */
async function materialRun(renderer: Renderer, options: G05MaterialOptions, layout: MaterialLayout, evidence: Record<string, string>) {
  const { width, height } = options, { materials, mx, my, qx, qy } = layout;
  renderer.setSun({ direction: [0, 1, -1], color: [3, 3, 3] });
  for (let i = 0; i < options.frames.warmup; i++) await renderFrame(renderer, WALL_VIEW);
  if (renderer.stats.width !== width || renderer.stats.height !== height) throw new Error(`G05_VIEWPORT:${renderer.stats.width}x${renderer.stats.height}`);
  const image = await readHdr(renderer), labels = classify(image, labelTable(layout.identities)), reference = wallReference(width, height, mx, my);
  const robust = robustMask(reference.expected, width, height), perTile = new Uint32Array(materials), marked = new Uint8Array(labels.length);
  let robustWall = 0, mismatches = 0, lit = 0, shadow = 0, litSum = 0, shadowSum = 0;
  for (let p = 0; p < labels.length; p++) {
    const expected = reference.expected[p];
    if (robust[p] && expected >= 0) {
      robustWall++; perTile[expected]++;
      if (labels[p] !== expected) { mismatches++; marked[p] = 1; }
    }
    if (reference.floor[p]) {
      const luminance = 0.2126 * image.rgb[p * 3] + 0.7152 * image.rgb[p * 3 + 1] + 0.0722 * image.rgb[p * 3 + 2];
      if (reference.floor[p] === 1) { lit++; litSum += luminance; } else { shadow++; shadowSum += luminance; }
    }
  }
  const missing = [...perTile].flatMap((count, tile) => count ? [] : [tile]);
  if ((layout.showcase || mismatches) && Object.keys(evidence).length < 4) evidence[`materials-${materials}`] = encodePng(image, marked);
  const diagnostics = renderer.diagnostics();
  const timing = await measureFrames(renderer, WALL_VIEW, options.frames);
  const litLuminance = litSum / Math.max(1, lit), shadowLuminance = shadowSum / Math.max(1, shadow);
  return { label: `materials-${materials}`, materials, tiles: [mx, my], batches: layout.batches, wallTriangles: options.quads * options.quads * 2,
    trianglesPerBatch: qx * qy * 2, drawCalls: renderer.stats.drawCalls, materialVariants: diagnostics.materialVariants,
    createMs: round(layout.createMs, 1),
    correctness: { robustWallPixels: robustWall, robustMismatches: mismatches, tilesWithoutRobustPixels: missing,
      minRobustPixelsPerTile: Math.min(...perTile),
      shadow: { litPixels: lit, shadowPixels: shadow, litLuminance: round(litLuminance, 5), shadowLuminance: round(shadowLuminance, 5),
        shadowOverLit: round(shadowLuminance / Math.max(1e-9, litLuminance), 4) } },
    timing };
}
type MaterialRun = Awaited<ReturnType<typeof materialRun>>;

export async function runG05MultiMaterial(options: G05MaterialOptions) {
  validateCommon(options);
  const { width, height } = options, quads = integer(options.quads, 16, 1024, 'G05_WALL_QUADS');
  const counts = ascending(options.materials, 1, 256, 'G05_MATERIAL_COUNTS');
  if (counts.some(count => (count & (count - 1)) !== 0 || tileGrid(count).some(cells => quads % cells !== 0))) throw new RangeError('G05_MATERIAL_TILING');
  const cleanup: Cleanup[] = [], runs: MaterialRun[] = [], evidence: Record<string, string> = {}, meta: { head?: Header } = {};
  for (const materials of counts) {
    const [mx, my] = tileGrid(materials), qx = quads / mx, qy = quads / my;
    const identities = Array.from({ length: materials }, (_, t) => (37 * t + 5) % 343);
    const batches: SceneBatch[] = [];
    for (let ty = 0; ty < my; ty++) for (let tx = 0; tx < mx; tx++) {
      const index = ty * mx + tx, w = (WALL.x1 - WALL.x0) / mx, h = (WALL.y1 - WALL.y0) / my;
      const mesh = wallTile(WALL.x0 + w * tx, WALL.x0 + w * (tx + 1), WALL.y0 + h * ty, WALL.y0 + h * (ty + 1), qx, qy);
      // One shared unlit definition; each batch owns a material instance with its identity parameter.
      batches.push(staticBatch(`g05-tile-${index}`, mesh, [instance(20000 + index, identity())],
        { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.parameter(0) },
          parameters: new Float32Array(g05IdentityColor(identities[index])) }));
    }
    batches.push(staticBatch('g05-floor', floorMesh(), [instance(19999, identity(), [0.5, 0.5, 0.5, 1], 0.8)], { definition: DEFAULT_MATERIAL_DEFINITION }));
    const createdAt = performance.now();
    await withRenderer(`materials-${materials}`, width, height, batches, { maxBatches: 512,
      sunShadow: { resolution: 2048, focusBounds: { min: [-4, -0.1, -2], max: [4, 4.2, 6] } } }, cleanup, async renderer => {
      const createMs = performance.now() - createdAt;
      meta.head ??= header(renderer);
      const showcase = materials === counts[0] || materials === counts[counts.length - 1];
      runs.push(await materialRun(renderer, options, { materials, mx, my, qx, qy, identities, batches: batches.length, createMs, showcase }, evidence));
    });
  }
  const scaling = runs.map(run => ({ materials: run.materials, batches: run.batches, drawCalls: run.drawCalls,
    cpuSubmitMs: run.timing.cpuSubmitMs, gpuShadowMedianMs: median(run.timing, 'shadow'), gpuSurfaceMedianMs: median(run.timing, 'surface'),
    gpuFrameSpanMedianMs: run.timing.gpuFrameSpanMs?.median ?? null, serializedFrameMedianMs: run.timing.serializedFrameMs?.median ?? null }));
  return { case: 'multi-material' as const, profile: options.profile, ...meta.head!, viewport: { width, height }, frames: options.frames,
    wall: { quadsPerSide: quads, triangles: quads * quads * 2, bounds: WALL, view: WALL_VIEW,
      sun: { direction: [0, 1, -1], color: [3, 3, 3], shadowResolution: 2048 } },
    scaling: { label: G05_DIAGNOSTIC, byMaterials: scaling,
      cpuSubmitMedianFitMsPerBatch: fit(runs.map(run => [run.batches, run.timing.cpuSubmitMs?.median ?? null] as const)),
      // Coarsened CPU timers quantize medians to about 0.1 ms; the mean keeps sub-step resolution.
      cpuSubmitMeanFitMsPerBatch: fit(runs.map(run => [run.batches, run.timing.cpuSubmitMs?.mean ?? null] as const)),
      gpuFrameSpanFitMsPerBatch: fit(runs.map(run => [run.batches, run.timing.gpuFrameSpanMs?.median ?? null] as const)) },
    runs, evidence, cleanup };
}
