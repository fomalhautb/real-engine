import traverseSource from './shaders/query-shared-traverse.wgsl?raw';
import poolSource from './shaders/query-shared-pool.wgsl?raw';
import tlasSource from './shaders/query-shared-tlas.wgsl?raw';
import { buildInstanceBvhTopology } from './instance-bvh';
import type { QueryBufferAllocator } from './gpu-query-scene-bvh';
import { SHARED_LEAF_WORDS, SHARED_MAX_LEAVES, sharedTopologyWords, type SharedStaticPlan } from './query-shared-static';

const TRAVERSE_PARAMETER_BYTES = 32;
const UPDATE_PARAMETER_BYTES = 32;
const REFIT_PARAMETER_BYTES = 16;

type Layout = { entries: ('uniform' | 'read' | 'write')[] };
const TRAVERSE_LAYOUT: Layout = { entries: ['uniform', 'read', 'write', 'read', 'read', 'read', 'read', 'write'] };
const UPDATE_LAYOUT: Layout = { entries: ['uniform', 'read', 'read', 'read', 'write', 'write'] };
const REFIT_LAYOUT: Layout = { entries: ['uniform', 'read', 'read', 'write'] };

function layout(device: GPUDevice, description: Layout): GPUBindGroupLayout {
  return device.createBindGroupLayout({ entries: description.entries.map((kind, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: kind === 'uniform' ? 'uniform' : kind === 'write' ? 'storage' : 'read-only-storage' },
  })) });
}

async function compile(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code, label });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(errors.map(error => `${label}:${error.lineNum}: ${error.message}`).join('\n'));
  return module;
}

/** Device pipelines, compiled once when a query enables shared static traversal. */
export class GpuQuerySharedStaticPipelines {
  private constructor(
    readonly traverseLayout: GPUBindGroupLayout, readonly updateLayout: GPUBindGroupLayout,
    readonly refitLayout: GPUBindGroupLayout, readonly traverse: GPUComputePipeline,
    readonly update: GPUComputePipeline, readonly refit: GPUComputePipeline,
  ) {}
  static async create(device: GPUDevice): Promise<GpuQuerySharedStaticPipelines> {
    const traverseLayout = layout(device, TRAVERSE_LAYOUT), updateLayout = layout(device, UPDATE_LAYOUT);
    const refitLayout = layout(device, REFIT_LAYOUT);
    const [traverseModule, poolModule, tlasModule] = await Promise.all([
      compile(device, traverseSource, 'world query shared static traversal'),
      compile(device, poolSource, 'world query shared static pool update'),
      compile(device, tlasSource, 'world query shared static TLAS refit')]);
    const pipeline = (bindGroupLayout: GPUBindGroupLayout, module: GPUShaderModule, entryPoint: string, label: string) =>
      device.createComputePipelineAsync({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module, entryPoint } });
    const [traverse, update, refit] = await Promise.all([
      pipeline(traverseLayout, traverseModule, 'traverse', 'world query shared static traversal'),
      pipeline(updateLayout, poolModule, 'updatePool', 'world query shared static pool update'),
      pipeline(refitLayout, tlasModule, 'refitTlas', 'world query shared static TLAS refit')]);
    return new GpuQuerySharedStaticPipelines(traverseLayout, updateLayout, refitLayout, traverse, update, refit);
  }
}

export interface SharedPoolBatchBuffers {
  /** Shared deformation output, instance-major Deformed records. */
  readonly output: GPUBuffer;
  /** Batch query data whose first words are the BVH topology. */
  readonly data: GPUBuffer;
  /** Per-instance BLAS bounds written by the batch refit. */
  readonly bounds: GPUBuffer;
  /** Leaf-ordered local vertex indices, uploaded once. */
  readonly triangles: Uint32Array<ArrayBuffer>;
}

/** Pool buffers for one admitted plan. Every allocation goes through the caller's ledger. */
export class GpuQuerySharedStatic {
  readonly nodes: GPUBuffer;
  readonly triangles: GPUBuffer;
  readonly positions: GPUBuffer;
  readonly leaves: GPUBuffer;
  readonly winners: GPUBuffer;
  readonly levelCount: number;
  private readonly topology: GPUBuffer;
  private readonly traverseParameters: GPUBuffer;
  private readonly updateParameters: GPUBuffer;
  private readonly traverseGroups: readonly GPUBindGroup[];
  private readonly updateGroups: readonly GPUBindGroup[];
  private readonly refitLevels: readonly { count: number; group: GPUBindGroup }[];
  private readonly traverseStride: number;
  private readonly updateStride: number;
  private readonly orderOffset: number;

  constructor(private readonly device: GPUDevice, private readonly pipelines: GpuQuerySharedStaticPipelines,
    readonly plan: SharedStaticPlan, allocate: QueryBufferAllocator, rays: GPUBuffer, hits: GPUBuffer,
    private readonly maxRays: number, parameterSlots: number, batchSlots: number,
    batches: readonly SharedPoolBatchBuffers[]) {
    if (!plan.leafCount || plan.leafCount > SHARED_MAX_LEAVES || batches.length !== plan.batches.length) {
      throw new Error('WORLD_QUERY_SHARED_STATIC_PLAN');
    }
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    this.traverseStride = Math.ceil(TRAVERSE_PARAMETER_BYTES / alignment) * alignment;
    this.updateStride = Math.ceil(UPDATE_PARAMETER_BYTES / alignment) * alignment;
    const topology = buildInstanceBvhTopology(plan.leafCount, SHARED_MAX_LEAVES);
    const words = new Uint32Array(sharedTopologyWords(plan.leafCount));
    words.set(topology.nodes);
    this.orderOffset = topology.nodes.length;
    words.fill(0xffff_ffff, this.orderOffset, this.orderOffset + plan.leafCount);
    let cursor = this.orderOffset + plan.leafCount;
    const levelStarts = topology.levels.map(level => { const start = cursor; words.set(level, start); cursor += level.length; return start; });
    const triangleWords = new Uint32Array(plan.triangleCount * 3);
    plan.batches.forEach((batch, index) => {
      if (batches[index].triangles.length !== batch.triangleCount * 3) throw new Error('WORLD_QUERY_SHARED_STATIC_TRIANGLES');
      triangleWords.set(batches[index].triangles, batch.triangleBase * 3);
    });
    const storage = GPUBufferUsage.STORAGE;
    this.nodes = allocate('world query shared static nodes', plan.bytes.nodes, storage);
    this.triangles = allocate('world query shared static triangles', plan.bytes.triangles,
      storage | GPUBufferUsage.COPY_DST, triangleWords.buffer);
    this.positions = allocate('world query shared static positions', plan.bytes.positions, storage);
    this.leaves = allocate('world query shared static leaves and batch flags',
      (plan.leafCount * SHARED_LEAF_WORDS + batchSlots) * 4, storage | GPUBufferUsage.COPY_DST);
    this.topology = allocate('world query shared static TLAS topology and order', words.byteLength,
      storage | GPUBufferUsage.COPY_DST, words.buffer);
    this.winners = allocate('world query shared static winners', maxRays * 4, storage);
    this.traverseParameters = allocate('world query shared static traversal parameter slots',
      this.traverseStride * parameterSlots, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.updateParameters = allocate('world query shared static update parameters',
      this.updateStride * plan.batches.length, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const refitParameters = topology.levels.map((level, depth) => allocate('world query shared static TLAS depth parameters',
      REFIT_PARAMETER_BYTES, GPUBufferUsage.UNIFORM, new Uint32Array([levelStarts[depth], level.length, this.orderOffset, 0]).buffer));
    this.levelCount = topology.levels.length;
    const whole = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer });
    const group = (layoutValue: GPUBindGroupLayout, resources: GPUBufferBinding[]) =>
      device.createBindGroup({ layout: layoutValue, entries: resources.map((resource, binding) => ({ binding, resource })) });
    this.traverseGroups = Array.from({ length: parameterSlots }, (_, slot) => group(pipelines.traverseLayout, [
      { buffer: this.traverseParameters, offset: slot * this.traverseStride, size: TRAVERSE_PARAMETER_BYTES },
      ...[rays, hits, this.nodes, this.triangles, this.positions, this.leaves, this.winners].map(whole)]));
    this.updateGroups = batches.map((buffers, index) => group(pipelines.updateLayout, [
      { buffer: this.updateParameters, offset: index * this.updateStride, size: UPDATE_PARAMETER_BYTES },
      ...[buffers.output, buffers.data, buffers.bounds, this.nodes, this.positions].map(whole)]));
    this.refitLevels = topology.levels.map((level, depth) => ({ count: level.length,
      group: group(pipelines.refitLayout, [refitParameters[depth], this.topology, this.leaves, this.nodes].map(whole)) }));
  }

  /** Leaf records for one pooled batch (all capacity slots). */
  writeLeaves(poolIndex: number, words: Uint32Array<ArrayBuffer>): number {
    const batch = this.plan.batches[poolIndex];
    if (!batch || words.length !== batch.capacity * SHARED_LEAF_WORDS) throw new Error('WORLD_QUERY_SHARED_STATIC_LEAVES');
    this.device.queue.writeBuffer(this.leaves, batch.leafBase * SHARED_LEAF_WORDS * 4, words);
    return words.byteLength;
  }
  writeFlags(flags: Uint32Array<ArrayBuffer>): number {
    this.device.queue.writeBuffer(this.leaves, this.plan.leafCount * SHARED_LEAF_WORDS * 4, flags);
    return flags.byteLength;
  }
  writeOrder(order: Uint32Array<ArrayBuffer>): number {
    if (order.length !== this.plan.leafCount) throw new Error('WORLD_QUERY_SHARED_STATIC_ORDER');
    this.device.queue.writeBuffer(this.topology, this.orderOffset * 4, order);
    return order.byteLength;
  }
  /** Update parameters change only with the active instance count. */
  writeUpdate(poolIndex: number, count: number): number {
    const batch = this.plan.batches[poolIndex];
    if (!batch || !Number.isSafeInteger(count) || count < 0 || count > batch.capacity) throw new Error('WORLD_QUERY_SHARED_STATIC_UPDATE');
    this.device.queue.writeBuffer(this.updateParameters, poolIndex * this.updateStride, new Uint32Array([
      count, batch.vertexCount, batch.nodeCount, 0, batch.nodeBase, batch.positionBase, 0, 0]));
    return UPDATE_PARAMETER_BYTES;
  }
  /** Copies refit batches into the pool, then refits the TLAS bottom-up; dispatch order provides visibility. */
  encodeUpdate(pass: GPUComputePassEncoder, batches: readonly { poolIndex: number; count: number }[]): number {
    let dispatches = 0;
    pass.setPipeline(this.pipelines.update);
    for (const { poolIndex, count } of batches) {
      if (!count) continue;
      const batch = this.plan.batches[poolIndex];
      pass.setBindGroup(0, this.updateGroups[poolIndex]);
      pass.dispatchWorkgroups(Math.ceil(count * (batch.vertexCount + batch.nodeCount) / 64));
      dispatches++;
    }
    pass.setPipeline(this.pipelines.refit);
    for (let depth = this.refitLevels.length - 1; depth >= 0; depth--) {
      pass.setBindGroup(0, this.refitLevels[depth].group);
      pass.dispatchWorkgroups(Math.ceil(this.refitLevels[depth].count / 64));
      dispatches++;
    }
    return dispatches;
  }
  encodeTraverse(pass: GPUComputePassEncoder, slot: number, count: number, workPerRay: number, upload: boolean,
    budget: { raysPerBlock: number; primaryRaysPerBlock: number; supplementalWorkPerRay: number }, reversed: boolean): number {
    if (!Number.isSafeInteger(slot) || slot < 0 || slot >= this.traverseGroups.length) throw new Error('WORLD_QUERY_SHARED_STATIC_SLOT');
    if (!Number.isSafeInteger(count) || count < 1 || count > this.maxRays) throw new Error('WORLD_QUERY_RAY_BUDGET');
    if (upload) this.device.queue.writeBuffer(this.traverseParameters, slot * this.traverseStride, new Uint32Array([
      count, workPerRay, this.plan.tlasNodeCount, reversed ? 1 : 0,
      budget.raysPerBlock, budget.primaryRaysPerBlock, budget.supplementalWorkPerRay, this.plan.leafCount]));
    pass.setPipeline(this.pipelines.traverse);
    pass.setBindGroup(0, this.traverseGroups[slot]);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    return upload ? TRAVERSE_PARAMETER_BYTES : 0;
  }
}
