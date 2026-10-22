import common from './shaders/reflections-common.wgsl?raw';
import updateSource from './shaders/reflections-update.wgsl?raw';
import sampleSource from './shaders/reflections-sample.wgsl?raw';
import surfaceFrame from './shaders/surface-frame.wgsl?raw';
import surfaceBrdf from './shaders/surface-brdf.wgsl?raw';
import giCommon from './shaders/diffuse-gi-common.wgsl?raw';
import giInterpolate from './shaders/diffuse-gi-interpolate.wgsl?raw';
import { reflectionFogSource, type GpuVolumetricFog } from './gpu-volumetric-fog';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from './gi-visibility';
import { lookAt } from './math';
import { planReflectionGrid, resolveReflectionsOptions, type ReflectionsOptions, type ReflectionGridPlan } from './reflections';
import type { GpuWorldQuery } from './gpu-world-query';
import type { GpuDiffuseGi } from './gpu-diffuse-gi';
import { diffuseGiQueryParameterSlots } from './diffuse-gi';
import { WORLD_HIT_BYTES } from './world-query';

export const reflectionSurfaceSource = common + '\n' + sampleSource;
const PARAMETER_BYTES = 176, ANCHOR_BYTES = 192, SAMPLE_BYTES = 32, PROBE_LIMIT = 1024;
type Entry = 'generateCamera' | 'generateReflections' | 'generateShadows' | 'integrate';
export interface ReflectionUpdate {
  frame: number; materialRevision: number; width: number; height: number;
  eye: readonly number[]; target: readonly number[]; fovY: number; near: number; far: number;
  sunDirection: readonly number[]; sunColor: readonly number[];
}
export interface ReflectionProbeResult {
  frame: number;
  anchors: { index: number; identity: number[]; position: number[]; normal: number[]; radiance: number[];
    valid: boolean; failedQueries: number }[];
}

/** Single reflected segment through the shared world query and material path.
 * Samples and reconstruction are bounded; no per-frame geometry readback or hidden history. */
export class GpuReflections {
  readonly options;
  private buffers: GPUBuffer[] = [];
  private bytes = 0;
  private disposed = false;
  private enabled = true;
  private pendingReadback = false;
  private frame = 0;
  private encodedFrame = 0;
  private materialRevision = 0;
  private grid?: ReflectionGridPlan;
  private parameters!: GPUBuffer;
  private cameraHits!: GPUBuffer;
  private secondaryHits!: GPUBuffer;
  private samples!: GPUBuffer;
  private anchors!: GPUBuffer;
  private readback!: GPUBuffer;
  private pipelines!: Record<Entry, GPUComputePipeline>;
  private groups!: Record<Exclude<Entry, 'integrate'>, GPUBindGroup>;
  private integrateGroups: GPUBindGroup[] = [];
  private fogGroup?: GPUBindGroup;
  private readonly data = new Float32Array(PARAMETER_BYTES / 4);

  get surfaceParameters(): GPUBuffer { return this.parameters; }
  get surfaceAnchors(): GPUBuffer { return this.anchors; }

  static async create(device: GPUDevice, queries: GpuWorldQuery, options: ReflectionsOptions,
    account: (bytes: number) => void, gi?: GpuDiffuseGi, fog?: GpuVolumetricFog, shadow?: GPUTexture): Promise<GpuReflections> {
    const resolved = resolveReflectionsOptions(options);
    const capacity = Math.min(resolved.maxAnchors, Math.floor(queries.options.maxRays / resolved.samplesPerAnchor),
      Math.floor(queries.options.maxTotalWorkUnits / resolved.samplesPerAnchor));
    if (capacity < 1) throw new Error('REFLECTIONS_QUERY_RAY_BUDGET');
    const parameterSlots = (gi ? Math.max(2, diffuseGiQueryParameterSlots(gi.options)) : 0) + 2;
    if (queries.options.maxTraceParameterSetsPerEncoder < parameterSlots) throw new Error('REFLECTIONS_QUERY_PARAMETER_BUDGET');
    if (device.limits.maxStorageBuffersPerShaderStage < 5 + Number(!!gi) + Number(!!fog)) throw new Error('REFLECTIONS_STORAGE_BINDING_LIMIT');
    if (fog && !shadow) throw new Error('REFLECTIONS_FOG_SHADOW_REQUIRED');
    const result = new GpuReflections(device, queries, resolved, capacity, account, gi, fog, shadow);
    try { await result.initialize(); return result; } catch (error) { result.dispose(); throw error; }
  }
  private constructor(private device: GPUDevice, private queries: GpuWorldQuery,
    options: ReturnType<typeof resolveReflectionsOptions>, private capacity: number,
    private account: (bytes: number) => void, private gi?: GpuDiffuseGi, private fog?: GpuVolumetricFog,
    private shadow?: GPUTexture) { this.options = options; }
  private assertAlive(): void { if (this.disposed) throw new Error('REFLECTIONS_DISPOSED'); }
  private buffer(label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size, usage });
    this.buffers.push(buffer); this.bytes += size; this.account(size); return buffer;
  }
  private bind(pipeline: GPUComputePipeline, entries: readonly [number, GPUBuffer][]): GPUBindGroup {
    return this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  }
  private async initialize(): Promise<void> {
    const cap = this.capacity, rays = cap * this.options.samplesPerAnchor;
    const sizes = [PARAMETER_BYTES, cap * WORLD_HIT_BYTES, rays * WORLD_HIT_BYTES,
      rays * SAMPLE_BYTES, cap * ANCHOR_BYTES, Math.min(cap, PROBE_LIMIT) * ANCHOR_BYTES];
    if (sizes.reduce((a, b) => a + b, 0) > this.options.maxBytes) throw new Error('REFLECTIONS_MEMORY_BUDGET');
    if (sizes.some(size => size > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize))) throw new Error('REFLECTIONS_BUFFER_LIMIT');
    if (Math.ceil(rays / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('REFLECTIONS_DISPATCH_LIMIT');
    this.parameters = this.buffer('reflection parameters', sizes[0], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.cameraHits = this.buffer('reflection camera hits', sizes[1], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.secondaryHits = this.buffer('reflection secondary hits', sizes[2], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.samples = this.buffer('reflection directions and BSDF weights', sizes[3], GPUBufferUsage.STORAGE);
    this.anchors = this.buffer('reflection radiance anchors', sizes[4], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.readback = this.buffer('reflection explicit diagnostic readback', sizes[5], GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const giSource = this.gi ? giCommon.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`)
      + '\n@group(0) @binding(7) var<uniform> gi: GiParameters;\n@group(0) @binding(8) var<storage, read> giProbes: array<GiProbe>;\n'
      + giInterpolate + '\nfn reflectionDiffuse(hit:ReflectionHit)->vec3f { return giSampleSurfaceIrradiance(hit.position.xyz,hit.normal.xyz,reflectionGeometry(hit),gi.state.x) * hit.baseColor.xyz * (1.0-hit.previous.w) * (0.96/3.14159265); }'
      : 'fn reflectionDiffuse(hit:ReflectionHit)->vec3f { return hit.baseColor.xyz * (0.045+0.065*max(hit.normal.y,0.0)) * (1.0-hit.previous.w*0.6); }';
    const fogSource = this.fog ? reflectionFogSource : 'fn reflectionApplyFog(origin:vec3f,direction:vec3f,distance:f32,incoming:vec3f)->vec3f {return incoming;}';
    const module = this.device.createShaderModule({ label: 'world-space reflections',
      code: surfaceFrame + '\n' + surfaceBrdf + '\n' + common + '\n' + updateSource.replace('// REFLECTION_GI', giSource) + '\n' + fogSource });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `reflections:${error.lineNum}: ${error.message}`).join('\n'));
    const names: Entry[] = ['generateCamera', 'generateReflections', 'generateShadows', 'integrate'];
    const pipelines = await Promise.all(names.map(entryPoint => this.device.createComputePipelineAsync({
      label: 'reflections ' + entryPoint, layout: 'auto', compute: { module, entryPoint } })));
    this.pipelines = Object.fromEntries(names.map((name, index) => [name, pipelines[index]])) as Record<Entry, GPUComputePipeline>;
    const p = this.pipelines;
    this.groups = {
      generateCamera: this.bind(p.generateCamera, [[0, this.parameters], [1, this.queries.rayBuffer]]),
      generateReflections: this.bind(p.generateReflections, [[0, this.parameters], [1, this.queries.rayBuffer], [3, this.cameraHits], [5, this.samples]]),
      generateShadows: this.bind(p.generateShadows, [[0, this.parameters], [1, this.queries.rayBuffer], [4, this.secondaryHits], [5, this.samples]]),
    };
    const base: [number, GPUBuffer][] = [[0, this.parameters], [2, this.queries.hitBuffer], [3, this.cameraHits],
      [4, this.secondaryHits], [5, this.samples], [6, this.anchors]];
    this.integrateGroups = this.gi ? this.gi.lightingProbeBuffers.map(probes =>
      this.bind(p.integrate, [...base, [7, this.gi!.lightingParameters], [8, probes]])) : [this.bind(p.integrate, base)];
    if (this.fog) this.fogGroup = this.device.createBindGroup({ layout: p.integrate.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: this.fog.worldParameters } }, { binding: 1, resource: { buffer: this.fog.worldVolumes } },
      { binding: 2, resource: this.shadow!.createView() },
      { binding: 3, resource: this.device.createSampler({ compare: 'less-equal', minFilter: 'linear', magFilter: 'linear' }) },
    ] });
  }
  setEnabled(enabled: boolean): void {
    this.assertAlive(); if (typeof enabled !== 'boolean') throw new Error('REFLECTIONS_ENABLED'); this.enabled = enabled;
  }
  update(input: ReflectionUpdate): number {
    this.assertAlive();
    for (const value of [input.frame, input.materialRevision]) if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) throw new Error('REFLECTIONS_VERSION');
    if (!input.frame || [input.eye, input.target, input.sunDirection, input.sunColor].some(v => v.length !== 3 || !v.every(Number.isFinite))
      || input.sunColor.some(value => value < 0 || value > 1e20) || !(input.near > 0 && input.far > input.near && input.far <= 10000)
      || !(input.fovY > 0 && input.fovY < Math.PI)) throw new Error('REFLECTIONS_UPDATE');
    const view = lookAt(input.eye, input.target, [0, 1, 0]);
    if (!view.every(Number.isFinite) || Math.hypot(view[2], view[6], view[10]) < 0.99
      || input.eye.some(value => Math.abs(value) > 1e6) || input.target.some(value => Math.abs(value) > 1e6)) throw new Error('REFLECTIONS_CAMERA');
    const sunLength = Math.hypot(...input.sunDirection);
    if (!(sunLength > 0) || !Number.isFinite(sunLength)) throw new Error('REFLECTIONS_SUN_DIRECTION');
    const grid = planReflectionGrid(input.width, input.height, this.options, this.queries.options.maxRays, this.queries.options.maxTotalWorkUnits);
    const tangent = Math.tan(input.fovY / 2), horizontal = tangent * input.width / input.height;
    const data = this.data, uints = new Uint32Array(data.buffer);
    data.set([...input.eye, this.options.maxDistance], 0);
    data.set([view[0] * horizontal, view[4] * horizontal, view[8] * horizontal, input.near], 4);
    data.set([view[1] * tangent, view[5] * tangent, view[9] * tangent, input.far], 8);
    data.set([-view[2], -view[6], -view[10], 0], 12);
    data.set(input.sunDirection.map(value => value / sunLength), 16); data.set(input.sunColor, 20); data.set(this.options.skyRadiance, 24);
    uints.set([input.width, input.height, grid.columns, grid.rows], 28);
    uints.set([grid.spacing, grid.anchorCount, this.options.samplesPerAnchor, Number(this.enabled)], 32);
    uints.set([input.frame, input.materialRevision, this.options.fogSteps, 0], 36);
    data.set([this.options.normalBias, this.options.textureLod, this.options.maxGatherDistance, this.options.planeTolerance], 40);
    this.device.queue.writeBuffer(this.parameters, 0, data);
    this.grid = grid; this.frame = input.frame; this.materialRevision = input.materialRevision;
    return PARAMETER_BYTES;
  }
  encode(encoder: GPUCommandEncoder, timing?: GPUComputePassTimestampWrites): number {
    this.assertAlive(); if (!this.grid || !this.frame) throw new Error('REFLECTIONS_NO_FRAME');
    if (!this.enabled) {
      const pass = encoder.beginComputePass({ label: 'reflections disabled', timestampWrites: timing }); pass.end(); return 0;
    }
    const grid = this.grid;
    const dispatch = (entry: Entry, count: number, group: GPUBindGroup, timestamps?: GPUComputePassTimestampWrites) => {
      const pass = encoder.beginComputePass({ label: 'reflections ' + entry, timestampWrites: timestamps });
      pass.setPipeline(this.pipelines[entry]); pass.setBindGroup(0, group);
      if (entry === 'integrate' && this.fogGroup) pass.setBindGroup(1, this.fogGroup);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    };
    dispatch('generateCamera', grid.anchorCount, this.groups.generateCamera,
      timing && { querySet: timing.querySet, beginningOfPassWriteIndex: timing.beginningOfPassWriteIndex });
    let uploaded = this.queries.encodeTrace(encoder, grid.anchorCount, this.options.textureLod, this.materialRevision);
    encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.cameraHits, 0, grid.anchorCount * WORLD_HIT_BYTES);
    dispatch('generateReflections', grid.rayCount, this.groups.generateReflections);
    uploaded += this.queries.encodeTrace(encoder, grid.rayCount, this.options.textureLod, this.materialRevision);
    encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.secondaryHits, 0, grid.rayCount * WORLD_HIT_BYTES);
    dispatch('generateShadows', grid.rayCount, this.groups.generateShadows);
    uploaded += this.queries.encodeTrace(encoder, grid.rayCount, this.options.textureLod, this.materialRevision);
    dispatch('integrate', grid.anchorCount, this.integrateGroups[this.gi?.lightingIndex ?? 0],
      timing && { querySet: timing.querySet, endOfPassWriteIndex: timing.endOfPassWriteIndex });
    this.encodedFrame = this.frame; return uploaded;
  }
  async probe(indices: readonly number[]): Promise<ReflectionProbeResult> {
    this.assertAlive();
    if (!this.enabled || !this.grid || !this.encodedFrame || this.encodedFrame !== this.frame) throw new Error('REFLECTIONS_FRAME_NOT_ENCODED');
    if (this.pendingReadback) throw new Error('REFLECTIONS_READBACK_PENDING');
    if (!Array.isArray(indices) || indices.length > PROBE_LIMIT || new Set(indices).size !== indices.length
      || indices.some(index => !Number.isSafeInteger(index) || index < 0 || index >= this.grid!.anchorCount)) throw new Error('REFLECTIONS_PROBE_INDICES');
    const requested = [...indices], frame = this.encodedFrame;
    if (!requested.length) return { frame, anchors: [] };
    this.pendingReadback = true;
    try {
      const encoder = this.device.createCommandEncoder({ label: 'reflection explicit diagnostics' });
      requested.forEach((index, i) => encoder.copyBufferToBuffer(this.anchors, index * ANCHOR_BYTES, this.readback, i * ANCHOR_BYTES, ANCHOR_BYTES));
      this.device.queue.submit([encoder.finish()]);
      await this.readback.mapAsync(GPUMapMode.READ, 0, requested.length * ANCHOR_BYTES); this.assertAlive();
      const floats = new Float32Array(this.readback.getMappedRange(0, requested.length * ANCHOR_BYTES));
      const uints = new Uint32Array(floats.buffer, floats.byteOffset, floats.length);
      return { frame, anchors: requested.map((index, i) => {
        const base = i * 48;
        if (uints[base + 45] !== frame) throw new Error('REFLECTIONS_FRAME_MISMATCH');
        const radiance = Array.from(floats.subarray(base + 40, base + 43));
        if (radiance.some(value => !Number.isFinite(value) || value < 0 || value > 65504)) throw new Error('REFLECTIONS_INVALID_RADIANCE');
        return { index, identity: Array.from(uints.subarray(base, base + 4)), position: Array.from(floats.subarray(base + 8, base + 11)),
          normal: Array.from(floats.subarray(base + 12, base + 15)), radiance, valid: uints[base + 44] === 1, failedQueries: uints[base + 46] };
      }) };
    } finally { if (this.readback.mapState === 'mapped') this.readback.unmap(); this.pendingReadback = false; }
  }
  diagnostics() {
    return { enabled: this.enabled, frame: this.encodedFrame, grid: this.grid ? { ...this.grid } : null,
      ownedGpuBytes: this.bytes, capacity: this.capacity, options: this.options,
      diffuseGi: !!this.gi, worldMedia: !!this.fog, reflectedSegments: 1, temporalHistory: false };
  }
  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy(); this.buffers = [];
    this.account(-this.bytes); this.bytes = 0;
  }
}
