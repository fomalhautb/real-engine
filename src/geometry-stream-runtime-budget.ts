import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import { geometryGpuPoolLayout } from './geometry-gpu-pool-layout';
import { admitGeometryStreamGroups } from './geometry-stream-admission';
import { geometryStreamSelectionResources } from './gpu-geometry-stream-selection';
import type { GeometryStreamDirectory } from './geometry-stream-types';
import type { GeometryGroupDemand } from './geometry-stream-working-set';
import { DEFORMED_VERTEX_BYTES } from './geometry';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';

export interface GeometryStreamRuntimeBudgetInput {
  readonly directory: GeometryStreamDirectory;
  readonly capacity: number;
  readonly maxPoolBytes?: number;
  readonly maxSelectionBytes?: number;
}
export interface GeometryStreamRuntimeBudgetOptions {
  /** All live page buffers/metadata, selection including maximum readback,
   * paged current/previous output and its 32-byte parameter buffer. */
  readonly maxResidentBytes: number;
  /** Same resources including other streams, candidate and retiring owners.
   * Does not include source/query proxies, textures, CPU caches or driver memory. */
  readonly maxPeakBytes: number;
}
export type GeometryStreamRuntimeLimits = Pick<GPUSupportedLimits, 'maxBufferSize' | 'maxStorageBufferBindingSize'
  | 'maxComputeWorkgroupsPerDimension' | 'minUniformBufferOffsetAlignment'>;

function budget(value: number | undefined, fallback: number, maximum: number, name: string) {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1 || result > maximum) throw new RangeError('GEOMETRY_STREAM_RUNTIME_' + name);
  return result;
}

/** Stateful cross-stream admission and bounded migration choice. The
 * Renderer owns preparation/commit/retirement; this scheduler accounts for all
 * their declared GPU dependents rather than just moving logical page quotas. */
export class GeometryStreamRuntimeBudget {
  private readonly inputs;
  private previous: Set<number>[];
  readonly options: Readonly<GeometryStreamRuntimeBudgetOptions>;
  readonly rootBytes: number;

  constructor(inputs: readonly GeometryStreamRuntimeBudgetInput[], options: GeometryStreamRuntimeBudgetOptions,
    limits: GeometryStreamRuntimeLimits) {
    const maxResidentBytes = budget(options?.maxResidentBytes, 0, 8 * 1024 ** 3, 'RESIDENT_BUDGET');
    const maxPeakBytes = budget(options?.maxPeakBytes, 0, 8 * 1024 ** 3, 'PEAK_BUDGET');
    if (maxPeakBytes < maxResidentBytes) throw new RangeError('GEOMETRY_STREAM_RUNTIME_PEAK_BUDGET');
    if (!Array.isArray(inputs) || !inputs.length || inputs.length > 4096) throw new RangeError('GEOMETRY_STREAM_RUNTIME_INPUTS');
    for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension', 'minUniformBufferOffsetAlignment'] as const) {
      const value = limits?.[name];
      if (!Number.isSafeInteger(value) || value < 1) throw new RangeError('GEOMETRY_STREAM_RUNTIME_DEVICE_LIMITS');
    }
    let clusters = 0, groups = 0, pages = 0;
    this.inputs = Array.from(inputs, input => {
      const directory = validateGeometryStreamDirectory(input?.directory);
      clusters += directory.clusters.length; groups += directory.groups.length; pages += directory.pages.length;
      if (Math.max(clusters, groups, pages) > 262144) throw new RangeError('GEOMETRY_STREAM_RUNTIME_WORK_BUDGET');
      const layout = geometryGpuPoolLayout(directory);
      const selection = geometryStreamSelectionResources(directory, input.capacity, limits.minUniformBufferOffsetAlignment);
      const maxPoolBytes = budget(input.maxPoolBytes, 256 * 1024 ** 2, 256 * 1024 ** 2, 'POOL_BUDGET');
      const maxSelectionBytes = budget(input.maxSelectionBytes, 128 * 1024 ** 2, 256 * 1024 ** 2, 'SELECTION_BUDGET');
      const maxBuffer = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
      if (selection.maxBytes > maxSelectionBytes || selection.sizes.some(bytes => bytes > maxBuffer)
        || selection.feedbackBytes > limits.maxBufferSize || selection.pairs > 262144
        || directory.groups.length * input.capacity > 262144 || Math.ceil(selection.pairs / 64) > limits.maxComputeWorkgroupsPerDimension) {
        throw new RangeError('GEOMETRY_STREAM_RUNTIME_SELECTION_BUDGET');
      }
      const outputPerSlot = layout.verticesPerSlot * input.capacity * DEFORMED_VERTEX_BYTES;
      const maxPages = Math.min(directory.pages.length, Math.floor((maxPoolBytes - layout.metadataBytes) / layout.bytesPerSlot),
        Math.floor(maxBuffer / (layout.verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES)), Math.floor(maxBuffer / (layout.indicesPerSlot * 4)),
        Math.floor(maxBuffer / outputPerSlot),
        Math.floor(limits.maxComputeWorkgroupsPerDimension * 64 / (layout.verticesPerSlot * input.capacity)));
      if (maxPages < directory.rootPages.length) throw new RangeError('GEOMETRY_STREAM_RUNTIME_ROOT_CAPACITY');
      return { directory, layout, capacity: input.capacity, selectionBytes: selection.maxBytes, outputPerSlot,
        pageBytes: layout.bytesPerSlot + outputPerSlot, fixedBytes: layout.metadataBytes + selection.maxBytes + 32, maxPages };
    });
    this.rootBytes = this.inputs.reduce((sum, input) => sum + input.fixedBytes + input.pageBytes * input.directory.rootPages.length, 0);
    if (this.rootBytes > maxResidentBytes) throw new RangeError('GEOMETRY_STREAM_RUNTIME_ROOT_BUDGET');
    this.options = Object.freeze({ maxResidentBytes, maxPeakBytes });
    this.previous = this.inputs.map(() => new Set());
  }
  /** Actual dependent allocations at this slot count, with maximum feedback
   * reserved even when no diagnostic buffer currently happens to be mapped. */
  state(index: number, slots: number) {
    const input = this.inputs[index];
    if (!input || !Number.isSafeInteger(index) || !Number.isSafeInteger(slots)
      || slots < input.directory.rootPages.length || slots > input.maxPages) throw new RangeError('GEOMETRY_STREAM_RUNTIME_CAPACITY');
    return { index, slots, poolBytes: input.layout.metadataBytes + slots * input.layout.bytesPerSlot,
      outputBytes: slots * input.outputPerSlot, parameterBytes: 32, selectionBytes: input.selectionBytes,
      totalBytes: input.fixedBytes + slots * input.pageBytes };
  }
  plan(demands: readonly (readonly GeometryGroupDemand[])[]) {
    if (!Array.isArray(demands) || demands.length !== this.inputs.length) throw new RangeError('GEOMETRY_STREAM_RUNTIME_DEMAND');
    const admitted = admitGeometryStreamGroups(this.inputs.map((input, index) => ({ ...input,
      demands: demands[index], previousAdmitted: this.previous[index] })), this.options.maxResidentBytes - this.rootBytes);
    const streams = admitted.map((plan, index) => ({ ...this.state(index, plan.pages.length), ...plan }));
    this.previous = admitted.map(plan => new Set(plan.admittedGroups));
    return { streams, allocatedBytes: streams.reduce((sum, stream) => sum + stream.totalBytes, 0), rootBytes: this.rootBytes };
  }
  /** Choose one migration. Shrinks free space before growth. A smaller temporary
   * root-only target is allowed when the desired shrink cannot fit in overlap.
   * A null action reports headroom pressure without overcommitting or stalling
   * rendering. reservedBytes includes all candidates and retiring owners. Growth
   * already admitted but not committed must also reserve its positive live delta:
   * asynchronous preparation/commit order cannot borrow an uncommitted shrink. */
  next(plan: ReturnType<GeometryStreamRuntimeBudget['plan']>, currentSlots: readonly number[], reservedBytes = 0,
    pending: { excluded?: readonly number[]; growthBytes?: number } = {}) {
    const growthBytes = pending.growthBytes ?? 0;
    if (!Array.isArray(currentSlots) || currentSlots.length !== this.inputs.length
      || !Number.isSafeInteger(reservedBytes) || reservedBytes < 0
      || !Number.isSafeInteger(growthBytes) || growthBytes < 0
      || pending.excluded !== undefined && (!Array.isArray(pending.excluded) || pending.excluded.length > this.inputs.length)
      || !Array.isArray(plan?.streams) || plan.streams.length !== this.inputs.length) throw new RangeError('GEOMETRY_STREAM_RUNTIME_STATE');
    const excluded = new Set(pending.excluded ?? []);
    if ([...excluded].some(index => !Number.isSafeInteger(index) || index < 0 || index >= this.inputs.length)) throw new RangeError('GEOMETRY_STREAM_RUNTIME_STATE');
    const current = currentSlots.map((slots, index) => this.state(index, slots));
    const liveBytes = current.reduce((sum, state) => sum + state.totalBytes, 0);
    const ownedBytes = liveBytes + reservedBytes;
    if (ownedBytes > this.options.maxPeakBytes) throw new RangeError('GEOMETRY_STREAM_RUNTIME_EXISTING_PEAK');
    if (liveBytes + growthBytes > this.options.maxResidentBytes) throw new RangeError('GEOMETRY_STREAM_RUNTIME_EXISTING_LIVE');
    const targets = plan.streams.map((stream, index) => {
      const state = this.state(index, stream.slots);
      if (stream.index !== index || stream.totalBytes !== state.totalBytes || stream.poolBytes !== state.poolBytes
        || stream.pages.length !== stream.slots || new Set(stream.pages).size !== stream.pages.length
        || stream.pages.some(id => !Number.isSafeInteger(id) || id < 0 || id >= this.inputs[index].directory.pages.length)
        || this.inputs[index].directory.rootPages.some(id => !stream.pages.includes(id))) throw new RangeError('GEOMETRY_STREAM_RUNTIME_PLAN');
      return { ...state, pages: stream.pages };
    });
    if (targets.reduce((sum, state) => sum + state.totalBytes, 0) > this.options.maxResidentBytes) throw new RangeError('GEOMETRY_STREAM_RUNTIME_PLAN');
    const candidates = targets.filter(target => !excluded.has(target.index) && target.slots !== current[target.index].slots).sort((a, b) => {
      const da = current[a.index].totalBytes - a.totalBytes, db = current[b.index].totalBytes - b.totalBytes;
      return Number(db > 0) - Number(da > 0) || db - da || a.index - b.index;
    });
    for (const target of candidates) {
      let desired = target, temporary = false;
      if (ownedBytes + desired.totalBytes > this.options.maxPeakBytes && desired.slots < current[desired.index].slots) {
        const input = this.inputs[desired.index];
        desired = { ...this.state(desired.index, input.directory.rootPages.length), pages: [...input.directory.rootPages] };
        temporary = desired.slots !== target.slots;
      }
      if (ownedBytes + desired.totalBytes > this.options.maxPeakBytes) continue;
      if (liveBytes + growthBytes + Math.max(0, desired.totalBytes - current[desired.index].totalBytes) > this.options.maxResidentBytes) continue;
      return { liveBytes, retiringBytes: reservedBytes, action: { ...desired, pages: [...desired.pages], temporary,
        expectedSlots: current[desired.index].slots, poolPeakBytes: current[desired.index].poolBytes + desired.poolBytes,
        peakBytes: ownedBytes + desired.totalBytes }, deferred: false };
    }
    return { liveBytes, retiringBytes: reservedBytes, action: null, deferred: candidates.length > 0 };
  }
}
