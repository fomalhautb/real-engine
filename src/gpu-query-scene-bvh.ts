import shaderSource from './shaders/query-scene-bvh.wgsl?raw';
import { buildInstanceBvhTopology, orderInstanceBounds } from './instance-bvh';
import type { Bounds3 } from './deformation';
import type { WorldQueryTraceBudgetPlan } from './gpu-world-query';

export type QueryBufferAllocator = (label: string, size: number, usage: GPUBufferUsageFlags,
  data?: ArrayBuffer) => GPUBuffer;

type RefitLevel = { count: number; bindGroup: GPUBindGroup };

const MAX_CAPACITY = 4096;
const MAX_RAYS = 65_536;
const MAX_PARAMETER_SLOTS = 32;
const PARAM_BYTES = 48;

function integer(value: number, low: number, high: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < low || value > high) {
    throw new RangeError(`${label} must be an integer in [${low}, ${high}]`);
  }
}

/** A bounded fixed-capacity batch tree. The allocator accounts for every buffer. */
export class GpuQuerySceneBvh {
  readonly maskBuffer: GPUBuffer;
  readonly boundsBuffer: GPUBuffer;
  readonly levelCount: number;
  readonly nodeCount: number;
  readonly maskWords: number;

  private readonly dataBuffer: GPUBuffer;
  private readonly candidateParams: GPUBuffer;
  private readonly candidateGroups: readonly GPUBindGroup[];
  private readonly refitLevels: readonly RefitLevel[];
  private readonly refitPipeline: GPUComputePipeline;
  private readonly candidatePipeline: GPUComputePipeline;
  private readonly parameterStride: number;
  private readonly maxRays: number;

  private constructor(
    private readonly device: GPUDevice,
    maskBuffer: GPUBuffer,
    boundsBuffer: GPUBuffer,
    dataBuffer: GPUBuffer,
    candidateParams: GPUBuffer,
    candidateGroups: readonly GPUBindGroup[],
    refitLevels: readonly RefitLevel[],
    refitPipeline: GPUComputePipeline,
    candidatePipeline: GPUComputePipeline,
    parameterStride: number,
    nodeCount: number,
    maxRays: number,
    private readonly capacity: number,
  ) {
    this.maskBuffer = maskBuffer;
    this.boundsBuffer = boundsBuffer;
    this.dataBuffer = dataBuffer;
    this.candidateParams = candidateParams;
    this.candidateGroups = candidateGroups;
    this.refitLevels = refitLevels;
    this.refitPipeline = refitPipeline;
    this.candidatePipeline = candidatePipeline;
    this.parameterStride = parameterStride;
    this.levelCount = refitLevels.length;
    this.nodeCount = nodeCount;
    this.maxRays = maxRays;
    this.maskWords = Math.ceil(capacity / 32);
  }

  static async create(device: GPUDevice, allocate: QueryBufferAllocator, rays: GPUBuffer, hits: GPUBuffer,
    maxRays: number, parameterSlots: number, capacity = 256): Promise<GpuQuerySceneBvh> {
    integer(maxRays, 1, MAX_RAYS, 'maxRays');
    integer(parameterSlots, 1, MAX_PARAMETER_SLOTS, 'parameterSlots');
    integer(capacity, 1, MAX_CAPACITY, 'capacity');
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    const parameterStride = Math.ceil(PARAM_BYTES / alignment) * alignment;
    if (!Number.isSafeInteger(parameterStride) || parameterStride < PARAM_BYTES) {
      throw new RangeError('scene BVH uniform alignment is invalid');
    }

    const topology = buildInstanceBvhTopology(capacity);
    const maskWords = Math.ceil(capacity / 32);
    const sceneLayout = [capacity, topology.nodeCount, topology.nodeCount * 4, maskWords];
    const nodeWords = topology.nodes.length;
    const orderOffset = nodeWords;
    const levelOffsets: number[] = [];
    let words = orderOffset + capacity;
    for (const level of topology.levels) { levelOffsets.push(words); words += level.length; }
    const packed = new Uint32Array(words);
    packed.set(topology.nodes);
    packed.fill(0xffff_ffff, orderOffset, orderOffset + capacity);
    topology.levels.forEach((level, depth) => packed.set(level, levelOffsets[depth]));

    const module = device.createShaderModule({ code: shaderSource, label: 'world query scene BVH' });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error =>
      `world query scene BVH:${error.lineNum}: ${error.message}`).join('\n'));
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const refitPipeline = await device.createComputePipelineAsync({ layout: pipelineLayout,
      compute: { module, entryPoint: 'refitScene' } });
    const candidatePipeline = await device.createComputePipelineAsync({ layout: pipelineLayout,
      compute: { module, entryPoint: 'selectBatches' } });

    const dataBuffer = allocate('world query scene BVH topology and order', packed.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, packed.buffer);
    // The first capacity Bounds are copied from the per-batch BLAS roots.
    const boundsBuffer = allocate('world query scene BVH roots and nodes',
      (capacity + topology.nodeCount) * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const maskBuffer = allocate('world query scene candidate masks', maxRays * maskWords * 4, GPUBufferUsage.STORAGE);
    const candidateParams = allocate('world query scene candidate parameter slots',
      parameterStride * parameterSlots, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const bindGroup = (params: GPUBuffer, offset = 0): GPUBindGroup => device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: params, offset, size: PARAM_BYTES } },
      { binding: 1, resource: { buffer: dataBuffer } },
      { binding: 2, resource: { buffer: boundsBuffer } },
      { binding: 3, resource: { buffer: rays } },
      { binding: 4, resource: { buffer: hits } },
      { binding: 5, resource: { buffer: maskBuffer } },
    ] });
    const refitLevels = topology.levels.map((level, depth) => {
      const params = allocate('world query scene BVH depth parameters', PARAM_BYTES,
        GPUBufferUsage.UNIFORM, new Uint32Array([levelOffsets[depth], level.length, 0, 0, ...sceneLayout, 0, 0, 0, 0]).buffer);
      return { count: level.length, bindGroup: bindGroup(params) };
    });
    const candidateGroups = Array.from({ length: parameterSlots }, (_, slot) =>
      bindGroup(candidateParams, slot * parameterStride));
    return new GpuQuerySceneBvh(device, maskBuffer, boundsBuffer, dataBuffer, candidateParams,
      candidateGroups, refitLevels, refitPipeline, candidatePipeline, parameterStride, topology.nodeCount, maxRays, capacity);
  }

  /** CPU boxes only sort candidates; the GPU refit reads the copied BLAS roots. */
  updateOrder(bounds: readonly (Bounds3 | null)[]): number {
    if (!Array.isArray(bounds) || bounds.length > this.capacity) throw new RangeError('scene BVH batch budget');
    const active: { index: number; bounds: Bounds3 }[] = [];
    bounds.forEach((entry, index) => { if (entry !== null) active.push({ index, bounds: entry }); });
    const localOrder = orderInstanceBounds(active.map(entry => entry.bounds), this.capacity);
    const order = new Uint32Array(this.capacity);
    order.fill(0xffff_ffff);
    for (let i = 0; i < active.length; i++) order[i] = active[localOrder[i]].index;
    this.device.queue.writeBuffer(this.dataBuffer, this.nodeCount * 4 * 4, order);
    return order.byteLength;
  }

  /** Each depth is a separate pass so child bounds are visible before parents. */
  encodeRefit(encoder: GPUCommandEncoder,
    timestampForPass: () => GPUComputePassTimestampWrites | undefined): void {
    for (let depth = this.refitLevels.length - 1; depth >= 0; depth--) {
      const level = this.refitLevels[depth];
      const pass = encoder.beginComputePass({ label: `world scene BVH refit depth ${depth}`,
        timestampWrites: timestampForPass() });
      pass.setPipeline(this.refitPipeline);
      pass.setBindGroup(0, level.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(level.count / 64));
      pass.end();
    }
  }

  /** Clears ceil(capacity/32) mask words per ray, then bounds-tests the tree. */
  encodeCandidates(encoder: GPUCommandEncoder, slot: number, count: number, workPerRay: number,
    upload: boolean, partition?: Pick<WorldQueryTraceBudgetPlan, 'raysPerBlock' | 'primaryRaysPerBlock' | 'supplementalWorkPerRay'>): number {
    integer(slot, 0, this.candidateGroups.length - 1, 'slot');
    integer(count, 0, this.maxRays, 'count');
    integer(workPerRay, 1, 1_048_576, 'workPerRay');
    if (partition !== undefined) {
      if (!partition || typeof partition !== 'object') throw new RangeError('scene BVH work partition');
      integer(partition.raysPerBlock, 2, count, 'raysPerBlock');
      integer(partition.primaryRaysPerBlock, 1, partition.raysPerBlock - 1, 'primaryRaysPerBlock');
      integer(partition.supplementalWorkPerRay, 1, 1_048_576, 'supplementalWorkPerRay');
      if (count % partition.raysPerBlock !== 0) throw new RangeError('scene BVH work partition');
    }
    if (count === 0) return 0;
    if (upload) this.device.queue.writeBuffer(this.candidateParams, slot * this.parameterStride,
      new Uint32Array([count, workPerRay, 0, 0, this.capacity, this.nodeCount, this.nodeCount * 4, this.maskWords,
        partition?.raysPerBlock ?? 0, partition?.primaryRaysPerBlock ?? 0, partition?.supplementalWorkPerRay ?? 0, 0]));
    const pass = encoder.beginComputePass({ label: 'world scene BVH candidates' });
    pass.setPipeline(this.candidatePipeline);
    pass.setBindGroup(0, this.candidateGroups[slot]);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    return upload ? PARAM_BYTES : 0;
  }
}
