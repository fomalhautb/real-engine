import deformationSource from './shaders/deform.wgsl?raw';
import surfaceSource from './shaders/surface.wgsl?raw';
import surfaceFrameSource from './shaders/surface-frame.wgsl?raw';
import surfaceBrdfSource from './shaders/surface-brdf.wgsl?raw';
import resolveSource from './shaders/resolve.wgsl?raw';
import { identity, lookAt, multiply, perspective, transformPoint, type Mat4 } from './math';
import { residentMesh, type InstanceState, type RenderBatch, type MaterialBinding } from './contracts';
import { MaterialResources, type GpuMaterial } from './gpu-material';
import { TEXTURE_COMPRESSION_FEATURES } from './texture-storage';
import { compileDeformation, conservativeDeformationBounds, evaluateDeformation,
  validateDeformationVertex, validateMorphTargets, type Bounds3 } from './deformation';
import { compileRenderGraph } from './render-graph';
import { GpuWorldQuery } from './gpu-world-query';
import type { WorldQueryOptions, WorldRay, WorldQueryResult } from './world-query';
import { GpuGeometryLod, type GeometryLodBinding, type GeometryLodSettings, type GeometryLodProbe } from './gpu-geometry-lod';
import { GpuDiffuseGi, diffuseGiSampleSource, diffuseGiFinalGatherSampleSource, type DiffuseGiProbeResult } from './gpu-diffuse-gi';
import { resolveDiffuseGiOptions, type DiffuseGiOptions } from './diffuse-gi';
import { resolveSunShadowOptions, sunShadowProjection, type SunShadowOptions, type ResolvedSunShadowOptions } from './sun-shadow';
import type { FinalGatherProbeResult } from './gpu-final-gather';
import { DEFORMED_VERTEX_BYTES, validateMeshColors, type MeshData } from './geometry';
import { snapshotMeshTangents } from './tangent-frame';
import { GpuGeometryStream } from './gpu-geometry-stream';
import { GpuGeometryPagePool, type GeometryGpuFrameLease } from './gpu-geometry-page-pool';
import { GeometryDecodeWorkerPool, type GeometryDecodeWorkerOptions } from './geometry-decode-worker-pool';
import { GpuVolumetricFog, type FogHistoryCell } from './gpu-volumetric-fog';
import { resolveVolumetricFogOptions, type VolumetricFogOptions } from './volumetric-fog';
import { GpuReflections, reflectionSurfaceSource, type ReflectionProbeResult } from './gpu-reflections';
import { resolveReflectionsOptions, type ReflectionsOptions } from './reflections';
import { GpuOcclusion } from './gpu-occlusion';
import { resolveOcclusionOptions, type OcclusionOptions } from './occlusion';

type Previous = { model: Mat4; bones: Mat4[]; scalars: number[]; version: number; generation: number; source?: InstanceState; settled: boolean };
function differsF32(values: ArrayLike<number>, old: ArrayLike<number>, offset = 0, length = values.length): boolean {
  for (let c = 0; c < length; c++) if (Math.fround(values[c]) !== old[offset + c]) return true;
  return false;
}
type MaterialPipelines = { surface: GPURenderPipeline; surfaceLod?: GPURenderPipeline; surfaceStream?: GPURenderPipeline; shadow: GPURenderPipeline; query?: GPUComputePipeline };
type Batch = {
  sourceId: number; scene: RenderBatch; mesh: MeshData; rest: GPUBuffer; indices: GPUBuffer; params: GPUBuffer;
  instances: GPUBuffer; palettes: GPUBuffer; output: GPUBuffer; scalars: GPUBuffer;
  compute: GPUBindGroup; render: GPUBindGroup;
  geometryLod?: GeometryLodBinding;
  stream?: { runtime: GpuGeometryStream; output: GPUBuffer; params: GPUBuffer; compute: GPUBindGroup; render: GPUBindGroup; pipeline: GPUComputePipeline };
  program: ReturnType<typeof compileDeformation>; pipeline: GPUComputePipeline;
  instanceData: Float32Array<ArrayBuffer>; instanceWords: Uint32Array<ArrayBuffer>; paletteData: Float32Array<ArrayBuffer>; scalarData: Float32Array<ArrayBuffer>;
  tangentData: Float32Array<ArrayBuffer>;
  morphBounds: Bounds3[]; bounds: Map<number, Bounds3>;
  history: Map<number, Previous>;
  dirty: boolean; fogSceneChanged: boolean; lastInstances: InstanceState[]; lastInputMode?: RenderBatch['inputMode'];
  /** Reused [start, end) slot ranges, bounded to 16 writes per input stream. */
  uploadRanges: number[];
  material: GpuMaterial; materialPipelines: MaterialPipelines; materialRequest: number;
};
export interface FrameStats {
  /** triangles is source instance-expanded geometry; use probeGeometryLod for actual selected triangles. */
  frames: number; drawCalls: number; triangles: number; deformedVertices: number;
  ownedGpuBytes: number; width: number; height: number; cpuSubmitMs: number;
  gpuMs: { deformation: number; shadow: number; surface: number; resolve: number; worldQueryRefit: number; geometryLod: number; diffuseGi: number; finalGather: number; reflections: number; occlusion: number; volumetricFog: number } | null;
  /** Render frame that produced gpuMs; unchanged while asynchronous readback is pending. */
  gpuSampleFrame: number | null;
  deformationDispatches: number; uploadedBytes: number;
  streamBackpressureFrames: number;
}
export interface DeformationVerification {
  vertices: number; maxError: number; maxNormalError: number; maxTangentError: number; boundsViolations: number;
  samples: { batch: string; entityId: number; vertexIndex: number; current: number[]; previous: number[] }[];
}
export interface RendererOptions {
  /** Dedicated module Workers shared by every streamed batch; no inline runtime fallback. */
  geometryDecoding?: GeometryDecodeWorkerOptions;
  /** Single-frame display-space edge filtering. HDR and motion/depth attachments stay unfiltered. */
  antiAliasing?: 'none' | 'fxaa';
  /** Directional shadow budget and world bounds, including off-screen casters. Defaults to the legacy 1024 / 50 m map. */
  sunShadow?: SunShadowOptions;
  /** Includes pending/resident definitions. Two render pipelines, plus a LOD variant when enabled. */
  materialVariantLimit?: number;
  /** Shared material images, all mips. Atomic replacements require overlap with their old images. */
  textureBudgetBytes?: number;
  /** Optional GPU world-space ray queries; disabled unless explicitly configured. */
  worldQueries?: WorldQueryOptions;
  /** Fixed world-space probe grid; requires worldQueries with sufficient ray capacity. */
  diffuseGi?: DiffuseGiOptions;
  /** Bounded view-space integration of world media and shared directional shadows. */
  volumetricFog?: VolumetricFogOptions;
  /** World-query specular reflection; requires worldQueries. */
  reflections?: ReflectionsOptions;
  /** Conservative two-phase Hi-Z for resident/paged virtual geometry. */
  occlusion?: OcclusionOptions;
}

export interface RendererAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly description: string;
  /** Null when neither the current nor legacy browser API reports fallback status. */
  readonly isFallbackAdapter: boolean | null;
}

/** Stage 1 scheduler: one deformation dispatch per mesh batch, shared by both views. */
export class Renderer {
  readonly errors: string[] = [];
  readonly stats: FrameStats = { frames: 0, drawCalls: 0, triangles: 0, deformedVertices: 0,
    ownedGpuBytes: 0, width: 0, height: 0, cpuSubmitMs: 0, gpuMs: null, gpuSampleFrame: null, deformationDispatches: 0, uploadedBytes: 0, streamBackpressureFrames: 0 };
  readonly info: { adapter: string; adapterInfo: RendererAdapterInfo; features: string[]; userAgent: string; timing: string; limits: Record<string, number> };
  private readonly batches: Batch[] = [];
  private readonly buffers: GPUBuffer[] = [];
  private readonly frameData = new Float32Array(64);
  private readonly frameBuffer: GPUBuffer;
  private readonly shadow: GPUTexture;
  private sunShadowOptions: ResolvedSunShadowOptions;
  private sunShadowProjection!: ReturnType<typeof sunShadowProjection>;
  private readonly frameBindGroup: GPUBindGroup;
  private readonly shadowFrameBindGroup: GPUBindGroup;
  private depth?: GPUTexture;
  private velocity?: GPUTexture;
  private hdr?: GPUTexture;
  private resolveGroup?: GPUBindGroup;
  private resolvePipeline!: GPURenderPipeline;
  private resolveLayout!: GPUBindGroupLayout;
  private readonly exposureBuffer: GPUBuffer;
  private exposureEV = 0;
  private antiAliasing: 'none' | 'fxaa' = 'none';
  private lastViewProjection?: Mat4;
  private lightMatrix: Mat4 = identity();
  private sunDirection: [number, number, number] = [-18, 30, 19];
  private sunColor: [number, number, number] = [3.5, 3.2, 2.8];
  private querySet?: GPUQuerySet;
  private queryResolve?: GPUBuffer;
  private queryRead?: GPUBuffer;
  private readonly timestampCount: number;
  private timingPending = false;
  private disposed = false;
  private readonly deformationPipelines = new Map<string, GPUComputePipeline>();
  private graph!: ReturnType<typeof compileRenderGraph<GPUCommandEncoder>>;
  private readonly materialResources: MaterialResources;
  private readonly materialPipelines = new Map<string, Promise<MaterialPipelines>>();
  private readonly compilingMaterials = new Set<string>();
  private readonly materialVariantLimit: number;
  private readonly worldQueryOptions?: WorldQueryOptions;
  private worldQueries?: GpuWorldQuery;
  private geometryLod?: GpuGeometryLod;
  private geometryDecoder?: GeometryDecodeWorkerPool;
  private readonly geometryDecodeOptions?: GeometryDecodeWorkerOptions;
  private diffuseGi?: GpuDiffuseGi;
  private readonly diffuseGiOptions?: DiffuseGiOptions;
  private readonly volumetricFogOptions?: VolumetricFogOptions;
  private volumetricFog?: GpuVolumetricFog;
  private readonly reflectionsOptions?: ReflectionsOptions;
  private reflections?: GpuReflections;
  private readonly occlusionOptions?: OcclusionOptions;
  private occlusion?: GpuOcclusion;
  private lightingLayout?: GPUBindGroupLayout;
  private lightingGroups: GPUBindGroup[] = [];
  private defaultClusterVertices!: GPUBuffer;
  private materialRevision = 0;
  private format!: GPUTextureFormat;
  private computeLayout!: GPUBindGroupLayout;
  private geometryLayout!: GPUBindGroupLayout;

  static async create(canvas: HTMLCanvasElement, scene: RenderBatch[], options: RendererOptions = {}): Promise<Renderer> {
    if (scene.filter(batch => !!batch.geometryStream).length > 256) throw new Error('GEOMETRY_STREAM_BATCH_BUDGET');
    if (options.antiAliasing !== undefined && options.antiAliasing !== 'none' && options.antiAliasing !== 'fxaa') throw new Error('INVALID_ANTI_ALIASING');
    resolveSunShadowOptions(options.sunShadow);
    if (options.volumetricFog) resolveVolumetricFogOptions(options.volumetricFog);
    if (options.reflections) {
      resolveReflectionsOptions(options.reflections);
      if (!options.worldQueries) throw new Error('REFLECTIONS_WORLD_QUERIES_REQUIRED');
    }
    if (options.occlusion) {
      resolveOcclusionOptions(options.occlusion);
      if (!scene.some(batch => batch.geometryLod || batch.geometryStream)) throw new Error('OCCLUSION_REQUIRES_VIRTUAL_GEOMETRY');
    }
    const limit = options.materialVariantLimit ?? 128;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 4096) throw new Error('INVALID_MATERIAL_VARIANT_LIMIT');
    if (options.textureBudgetBytes !== undefined && (!Number.isSafeInteger(options.textureBudgetBytes) || options.textureBudgetBytes < 0)) throw new Error('INVALID_TEXTURE_BUDGET');
    if (options.diffuseGi) {
      const gi = resolveDiffuseGiOptions(options.diffuseGi);
      if (!options.worldQueries || (options.worldQueries.maxRays ?? 1024) < gi.rayCount) throw new Error('DIFFUSE_GI_QUERY_RAY_BUDGET');
      if (gi.emissiveSamplesPerProbe && (options.worldQueries.maxTraceParameterSetsPerEncoder ?? 8) < 2) {
        throw new Error('DIFFUSE_GI_QUERY_PARAMETER_BUDGET');
      }
      if (gi.finalGather && (options.worldQueries.maxTraceParameterSetsPerEncoder ?? 8) < 6) {
        throw new Error('FINAL_GATHER_QUERY_PARAMETER_BUDGET');
      }
    }
    if (!navigator.gpu) throw new Error('WEBGPU_UNAVAILABLE: use a WebGPU browser on HTTPS or localhost.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('ADAPTER_UNAVAILABLE: no compatible WebGPU adapter.');
    const timing = adapter.features.has('timestamp-query');
    const requiredFeatures: GPUFeatureName[] = timing ? ['timestamp-query'] : [];
    for (const feature of TEXTURE_COMPRESSION_FEATURES) if (adapter.features.has(feature)) requiredFeatures.push(feature);
    const device = await adapter.requestDevice({ requiredFeatures });
    let context: GPUCanvasContext | null = null;
    let renderer: Renderer | undefined;
    try {
      context = canvas.getContext('webgpu');
      if (!context) throw new Error('CANVAS_UNAVAILABLE: cannot create WebGPU context.');
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });
      renderer = new Renderer(canvas, context, device, adapter, timing, options,
        scene.some(batch => !!batch.geometryLod || !!batch.geometryStream), scene.some(batch => !!batch.geometryStream));
      await renderer.initialize(format, scene);
      return renderer;
    } catch (error) {
      if (renderer) renderer.dispose();
      else { context?.unconfigure(); device.destroy(); }
      throw error;
    }
  }

  private constructor(
    private canvas: HTMLCanvasElement, private context: GPUCanvasContext,
    private device: GPUDevice, adapter: GPUAdapter, timing: boolean, options: RendererOptions, private hasGeometryLod: boolean,
    private hasGeometryStream: boolean,
  ) {
    const legacyFallback = 'isFallbackAdapter' in adapter && typeof adapter.isFallbackAdapter === 'boolean'
      ? adapter.isFallbackAdapter : null;
    const adapterInfo: RendererAdapterInfo = Object.freeze({
      vendor: adapter.info.vendor ?? '', architecture: adapter.info.architecture ?? '',
      description: adapter.info.description ?? '',
      isFallbackAdapter: typeof adapter.info.isFallbackAdapter === 'boolean'
        ? adapter.info.isFallbackAdapter : legacyFallback,
    });
    this.info = {
      adapter: [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.description].filter(Boolean).join(' / ') || 'Unreported adapter',
      adapterInfo,
      features: Array.from(device.features), userAgent: navigator.userAgent,
      timing: timing ? 'GPU timestamp-query (adapter-dependent precision)' : 'GPU timestamps unavailable',
      limits: { maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        maxTextureDimension2D: device.limits.maxTextureDimension2D, maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage, maxSampledTexturesPerShaderStage: device.limits.maxSampledTexturesPerShaderStage },
    };
    device.addEventListener('uncapturederror', event => this.errors.push(event.error.message));
    void device.lost.then(loss => {
      if (!this.disposed) {
        this.errors.push('DEVICE_LOST: ' + loss.reason + ' ' + loss.message);
        for (const batch of this.batches) if (batch.stream) void batch.stream.runtime.dispose().catch(error => this.errors.push(String(error)));
        this.geometryDecoder?.dispose();
      }
    });
    this.materialVariantLimit = options.materialVariantLimit ?? 128;
    this.worldQueryOptions = options.worldQueries;
    this.geometryDecodeOptions = options.geometryDecoding;
    this.diffuseGiOptions = options.diffuseGi && resolveDiffuseGiOptions(options.diffuseGi);
    this.volumetricFogOptions = options.volumetricFog && resolveVolumetricFogOptions(options.volumetricFog);
    this.reflectionsOptions = options.reflections && resolveReflectionsOptions(options.reflections);
    this.occlusionOptions = options.occlusion && resolveOcclusionOptions(options.occlusion);
    this.timestampCount = 8 + (options.worldQueries ? 2 : 0) + (hasGeometryLod ? 2 : 0) + (options.diffuseGi ? 2 : 0)
      + (this.diffuseGiOptions?.finalGather ? 2 : 0) + (options.reflections ? 2 : 0) + (options.occlusion ? 6 : 0)
      + (options.volumetricFog ? (options.volumetricFog.temporal ? 8 : 6) : 0);
    this.materialResources = new MaterialResources(device, bytes => { this.stats.ownedGpuBytes += bytes; }, options.textureBudgetBytes, adapter.info);
    this.frameBuffer = this.buffer('frame uniforms', 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.exposureBuffer = this.buffer('exposure uniforms', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.antiAliasing = options.antiAliasing ?? 'none';
    this.setExposure(0);
    this.sunShadowOptions = resolveSunShadowOptions(options.sunShadow, device.limits.maxTextureDimension2D);
    const shadowResolution = this.sunShadowOptions.resolution;
    this.shadow = device.createTexture({ label: 'sun shadow', size: [shadowResolution, shadowResolution], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.stats.ownedGpuBytes += this.sunShadowOptions.bytes;
    const frameLayout = device.createBindGroupLayout({ label: 'frame and lighting contract', entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ] });
    this.frameBindGroup = device.createBindGroup({ layout: frameLayout, entries: [
      { binding: 0, resource: { buffer: this.frameBuffer } },
      { binding: 1, resource: this.shadow.createView() },
      { binding: 2, resource: device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' }) },
    ] });
    // The depth pass must not bind its own render attachment as a sampled texture.
    const shadowFrameLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ] });
    this.shadowFrameBindGroup = device.createBindGroup({ layout: shadowFrameLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } }] });
    this.frameLayout = frameLayout;
    this.shadowFrameLayout = shadowFrameLayout;
    this.setSun({ direction: this.sunDirection, color: this.sunColor });
    if (timing) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: this.timestampCount });
      this.queryResolve = this.buffer('timestamp resolve', this.timestampCount * 8, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      this.queryRead = this.buffer('timestamp readback', this.timestampCount * 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    }
  }
  private frameLayout: GPUBindGroupLayout;
  private shadowFrameLayout: GPUBindGroupLayout;

  private buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer {
    if (!Number.isSafeInteger(size) || size < 0 || Math.max(4, Math.ceil(size / 4) * 4) > this.device.limits.maxBufferSize
      || (usage & GPUBufferUsage.STORAGE) && Math.max(4, Math.ceil(size / 4) * 4) > this.device.limits.maxStorageBufferBindingSize) {
      throw new Error('BUFFER_BUDGET: ' + label);
    }
    const buffer = this.device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4), usage, mappedAtCreation: !!data });
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data)); buffer.unmap(); }
    this.buffers.push(buffer);
    this.stats.ownedGpuBytes += buffer.size;
    return buffer;
  }

  private async shader(code: string, label: string): Promise<GPUShaderModule> {
    const shader = this.device.createShaderModule({ code, label });
    const info = await shader.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(e => label + ':' + e.lineNum + ': ' + e.message).join('\n'));
    return shader;
  }

  private async initialize(format: GPUTextureFormat, scene: RenderBatch[]): Promise<void> {
    if (this.hasGeometryStream) this.geometryDecoder = new GeometryDecodeWorkerPool(this.geometryDecodeOptions);
    this.format = format;
    this.resolveLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ] });
    const resolveModule = await this.shader(resolveSource, 'HDR resolve');
    this.resolvePipeline = await this.device.createRenderPipelineAsync({ label: 'HDR exposure and display transfer',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.resolveLayout] }),
      vertex: { module: resolveModule, entryPoint: 'vertexMain' },
      fragment: { module: resolveModule, entryPoint: 'fragmentMain', targets: [{ format: this.format }] },
    });
    this.computeLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6, 7].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 4 ? 'storage' : 'read-only-storage' },
    })) });
    this.geometryLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4].map(binding => ({
      binding, visibility: GPUShaderStage.VERTEX,
      buffer: { type: binding === 2 ? 'uniform' : 'read-only-storage' },
    })) });
    this.defaultClusterVertices = this.buffer('unused cluster identity', 16, GPUBufferUsage.STORAGE);
    if (new Set(scene.map(batch => batch.name)).size !== scene.length) throw new Error('DUPLICATE_BATCH_NAME');
    if (this.occlusionOptions) this.occlusion = await GpuOcclusion.create(this.device, this.occlusionOptions, bytes => { this.stats.ownedGpuBytes += bytes; });
    if (this.hasGeometryLod) this.geometryLod = await GpuGeometryLod.create(this.device, bytes => { this.stats.ownedGpuBytes += bytes; }, this.occlusion);
    if (this.worldQueryOptions) {
      this.worldQueries = await GpuWorldQuery.create(this.device, this.materialResources.layout,
        bytes => { this.stats.ownedGpuBytes += bytes; }, this.worldQueryOptions);
    }
    if (this.diffuseGiOptions) this.diffuseGi = await GpuDiffuseGi.create(this.device, this.worldQueries!, this.diffuseGiOptions,
      bytes => { this.stats.ownedGpuBytes += bytes; });
    if (this.volumetricFogOptions) this.volumetricFog = await GpuVolumetricFog.create(this.device, this.volumetricFogOptions,
      bytes => { this.stats.ownedGpuBytes += bytes; });
    if (this.reflectionsOptions) this.reflections = await GpuReflections.create(this.device, this.worldQueries!, this.reflectionsOptions,
      bytes => { this.stats.ownedGpuBytes += bytes; }, this.diffuseGi, this.volumetricFog, this.shadow);
    if (this.diffuseGi || this.reflections) {
      this.lightingLayout = this.device.createBindGroupLayout({ label: 'surface GI and reflections', entries: [
        ...(this.diffuseGi?.surfaceBindingLayoutEntries ?? []),
        ...(this.reflections ? [
          { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' as const } },
          { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' as const } },
        ] : []),
      ] });
      this.lightingGroups = (this.diffuseGi?.surfaceBindingVariants ?? [[]]).map(entries => this.device.createBindGroup({
        layout: this.lightingLayout!, entries: [...entries, ...(this.reflections ? [
          { binding: 4, resource: { buffer: this.reflections.surfaceParameters } },
          { binding: 5, resource: { buffer: this.reflections.surfaceAnchors } },
        ] : [])],
      }));
    }
    for (const entry of scene) {
      const batch = await this.createBatch(entry);
      this.batches.push(batch);
      this.worldQueries?.addBatch({ scene: batch.scene, output: batch.output, instances: batch.instances,
        getMaterial: () => batch.material, getPipeline: () => batch.materialPipelines.query!, getDirty: () => batch.dirty,
        getInstanceBounds: () => batch.scene.instances.map(state => batch.bounds.get(state.entityId)!) });
      this.diffuseGi?.registerEmissiveBatch(batch.scene, batch.output, this.batches.length);
    }
    this.graph = compileRenderGraph<GPUCommandEncoder>([
      { name: 'scene-input', kind: 'buffer', source: 'imported' },
      ...(this.hasGeometryStream ? [{ name: 'stream-pages', kind: 'buffer' as const, source: 'imported' as const }] : []),
      { name: 'shared-geometry', kind: 'buffer', source: 'imported' },
      ...(this.worldQueries ? [{ name: 'world-query-bounds', kind: 'buffer' as const, source: 'imported' as const }] : []),
      ...(this.geometryLod ? [{ name: 'selected-geometry', kind: 'buffer' as const, source: 'transient' as const }] : []),
      ...(this.diffuseGi ? [{ name: 'diffuse-gi-probes', kind: 'buffer' as const, source: 'imported' as const }] : []),
      ...(this.diffuseGi || this.reflections ? [{ name: 'world-query-scratch', kind: 'buffer' as const, source: 'imported' as const }] : []),
      ...(this.reflections ? [{ name: 'reflection-anchors', kind: 'buffer' as const, source: 'imported' as const }] : []),
      ...(this.diffuseGiOptions?.finalGather ? [{ name: 'final-gather-anchors', kind: 'buffer' as const, source: 'imported' as const }] : []),
      { name: 'sun-shadow', kind: 'texture', source: 'transient' },
      { name: 'canvas', kind: 'texture', source: 'imported' },
      { name: 'scene-hdr', kind: 'texture', source: 'transient' },
      { name: 'depth', kind: 'texture', source: 'transient' },
      { name: 'velocity', kind: 'texture', source: 'transient' },
      ...(this.occlusion ? [
        { name: 'occlusion-previous', kind: 'texture' as const, source: 'imported' as const },
        { name: 'occlusion-current', kind: 'texture' as const, source: 'transient' as const },
      ] : []),
      ...(this.volumetricFog ? [
        { name: 'fog-media', kind: 'buffer' as const, source: 'imported' as const },
        { name: 'fog-source', kind: 'buffer' as const, source: 'transient' as const },
        ...(this.volumetricFog.temporalConfigured ? [{ name: 'fog-history', kind: 'buffer' as const, source: 'imported' as const }] : []),
        { name: 'fog-integrated', kind: 'buffer' as const, source: 'transient' as const },
        { name: 'fog-hdr', kind: 'texture' as const, source: 'transient' as const },
      ] : []),
    ], [
      { name: 'deformation', uses: [{ resource: 'scene-input', access: 'read' },
        ...(this.hasGeometryStream ? [{ resource: 'stream-pages', access: 'read' as const }] : []),
        { resource: 'shared-geometry', access: 'write' }], execute: encoder => this.encodeDeformation(encoder) },
      ...(this.worldQueries ? [{ name: 'world-query-refit', uses: [
        { resource: 'shared-geometry' as const, access: 'read' as const },
        { resource: 'world-query-bounds' as const, access: 'write' as const },
      ], after: ['deformation'], execute: (encoder: GPUCommandEncoder) => this.worldQueries!.encodeRefit(encoder, this.timestamp(8)) }] : []),
      ...(this.geometryLod ? [{ name: 'geometry-lod', uses: [
        { resource: 'scene-input' as const, access: 'read' as const },
        ...(this.occlusion ? [{ resource: 'occlusion-previous', access: 'read' as const }] : []),
        ...(this.hasGeometryStream ? [{ resource: 'stream-pages', access: 'read' as const }] : []),
        { resource: 'selected-geometry' as const, access: 'write' as const },
      ], execute: (encoder: GPUCommandEncoder) => this.encodeGeometrySelection(encoder) }] : []),
      ...(this.diffuseGi ? [{ name: 'diffuse-gi', uses: [
        { resource: 'world-query-bounds', access: 'read' as const },
        { resource: 'shared-geometry', access: 'read' as const },
        { resource: 'world-query-scratch', access: 'readWrite' as const },
        { resource: 'diffuse-gi-probes', access: 'readWrite' as const },
      ], execute: (encoder: GPUCommandEncoder) => {
        this.stats.uploadedBytes += this.diffuseGi!.encode(encoder, this.stats.frames + 1, this.materialRevision,
          this.timestamp(10 + (this.geometryLod ? 2 : 0)));
      } }] : []),
      ...(this.diffuseGiOptions?.finalGather ? [{ name: 'final-gather', uses: [
        { resource: 'world-query-bounds', access: 'read' as const },
        { resource: 'shared-geometry', access: 'read' as const },
        { resource: 'world-query-scratch', access: 'readWrite' as const },
        { resource: 'diffuse-gi-probes', access: 'read' as const },
        { resource: 'final-gather-anchors', access: 'readWrite' as const },
      ], execute: (encoder: GPUCommandEncoder) => {
        this.stats.uploadedBytes += this.diffuseGi!.encodeFinalGather(encoder, this.materialRevision,
          this.timestamp(12 + (this.geometryLod ? 2 : 0)));
      } }] : []),
      { name: 'shadow', uses: [{ resource: 'shared-geometry', access: 'read' }, { resource: 'sun-shadow', access: 'write' }], execute: encoder => this.encodeShadow(encoder) },
      ...(this.reflections ? [{ name: 'reflections', uses: [
        { resource: 'world-query-bounds', access: 'read' as const }, { resource: 'shared-geometry', access: 'read' as const },
        { resource: 'world-query-scratch', access: 'readWrite' as const }, { resource: 'reflection-anchors', access: 'write' as const },
        ...(this.diffuseGi ? [{ resource: 'diffuse-gi-probes', access: 'read' as const }] : []),
        ...(this.volumetricFog ? [{ resource: 'fog-media', access: 'read' as const }, { resource: 'sun-shadow', access: 'read' as const }] : []),
      ], execute: (encoder: GPUCommandEncoder) => { this.stats.uploadedBytes += this.reflections!.encode(encoder, this.timestamp(this.reflectionTimestampIndex)); } }] : []),
      { name: 'surface', uses: [{ resource: 'shared-geometry', access: 'read' }, { resource: 'sun-shadow', access: 'read' },
        ...(this.geometryLod ? [{ resource: 'selected-geometry', access: 'read' as const }] : []),
        ...(this.diffuseGi ? [{ resource: 'diffuse-gi-probes', access: 'read' as const }] : []),
        ...(this.diffuseGiOptions?.finalGather ? [{ resource: 'final-gather-anchors', access: 'read' as const }] : []),
        ...(this.reflections ? [{ resource: 'reflection-anchors', access: 'read' as const }] : []),
        { resource: 'scene-hdr', access: 'write' }, { resource: 'depth', access: 'write' }, { resource: 'velocity', access: 'write' }], execute: encoder => this.encodeSurface(encoder) },
      ...(this.occlusion ? [
        { name: 'occlusion-pyramid', uses: [{ resource: 'depth', access: 'read' as const }, { resource: 'occlusion-current', access: 'write' as const }],
          execute: (encoder: GPUCommandEncoder) => this.occlusion!.encodePyramid(encoder, this.timestamp(this.occlusionTimestampIndex)) },
        { name: 'occlusion-recover', uses: [{ resource: 'occlusion-current', access: 'read' as const }, { resource: 'scene-input', access: 'read' as const },
          ...(this.hasGeometryStream ? [{ resource: 'stream-pages', access: 'read' as const }] : []),
          { resource: 'selected-geometry', access: 'readWrite' as const }], execute: (encoder: GPUCommandEncoder) => this.encodeGeometryRecovery(encoder) },
        { name: 'surface-recover', uses: [{ resource: 'shared-geometry', access: 'read' as const }, { resource: 'sun-shadow', access: 'read' as const },
          { resource: 'selected-geometry', access: 'read' as const },
          ...(this.diffuseGi ? [{ resource: 'diffuse-gi-probes', access: 'read' as const }] : []),
          ...(this.diffuseGiOptions?.finalGather ? [{ resource: 'final-gather-anchors', access: 'read' as const }] : []),
          ...(this.reflections ? [{ resource: 'reflection-anchors', access: 'read' as const }] : []),
          { resource: 'scene-hdr', access: 'attachmentLoad' as const }, { resource: 'depth', access: 'attachmentLoad' as const },
          { resource: 'velocity', access: 'attachmentLoad' as const }], execute: (encoder: GPUCommandEncoder) => this.encodeSurface(encoder, true) },
      ] : []),
      ...(this.volumetricFog ? [
        { name: 'fog-inject', uses: [{ resource: 'fog-media', access: 'read' as const }, { resource: 'sun-shadow', access: 'read' as const }, { resource: 'fog-source', access: 'write' as const }],
          execute: (encoder: GPUCommandEncoder) => this.volumetricFog!.encodeInject(encoder, this.timestamp(this.timestampCount - (this.volumetricFog!.temporalConfigured ? 8 : 6))) },
        ...(this.volumetricFog.temporalConfigured ? [{ name: 'fog-temporal', uses: [
          { resource: 'fog-source', access: 'read' as const }, { resource: 'fog-history', access: 'readWrite' as const }],
          execute: (encoder: GPUCommandEncoder) => this.volumetricFog!.encodeTemporal(encoder, this.timestamp(this.timestampCount - 6)) }] : []),
        { name: 'fog-integrate', uses: [{ resource: 'fog-source', access: 'read' as const },
          ...(this.volumetricFog.temporalConfigured ? [{ resource: 'fog-history', access: 'read' as const }] : []),
          { resource: 'fog-integrated', access: 'write' as const }],
          execute: (encoder: GPUCommandEncoder) => this.volumetricFog!.encodeIntegrate(encoder, this.timestamp(this.timestampCount - 4)) },
        { name: 'fog-compose', uses: [{ resource: 'fog-source', access: 'read' as const }, { resource: 'fog-integrated', access: 'read' as const },
          ...(this.volumetricFog.temporalConfigured ? [{ resource: 'fog-history', access: 'read' as const }] : []),
          { resource: 'scene-hdr', access: 'read' as const }, { resource: 'depth', access: 'read' as const }, { resource: 'fog-hdr', access: 'write' as const }],
          execute: (encoder: GPUCommandEncoder) => this.volumetricFog!.encodeCompose(encoder, this.timestamp(this.timestampCount - 2)) },
      ] : []),
      { name: 'resolve', uses: [{ resource: this.volumetricFog ? 'fog-hdr' : 'scene-hdr', access: 'read' }, { resource: 'canvas', access: 'write' }], execute: encoder => this.encodeResolve(encoder) },
    ]);
  }

  private async prepareMaterial(binding?: MaterialBinding): Promise<{ material: GpuMaterial; materialPipelines: MaterialPipelines }> {
    const material = this.materialResources.create(binding);
    const program = material.program;
    try {
      let pending = this.materialPipelines.get(program.key);
      if (!pending) {
        if (this.materialPipelines.size >= this.materialVariantLimit) {
          const unused = Array.from(this.materialPipelines.keys()).find(key => !this.compilingMaterials.has(key) && !this.materialResources.hasProgram(key));
          if (!unused) throw new Error('MATERIAL_VARIANT_BUDGET: all cached variants are referenced or compiling');
          this.materialPipelines.delete(unused);
        }
        this.compilingMaterials.add(program.key);
        pending = (async () => {
          const module = await this.shader((surfaceFrameSource + '\n' + surfaceBrdfSource + '\n' + surfaceSource)
            .replace('// REFLECTION_PROGRAM', this.reflections ? reflectionSurfaceSource : '')
            .replace('// REFLECTED_LIGHTING', this.reflections
              ? 'let reflected = reflectionRadiance(input.position.xy, input.world, n, geometricNormal, base, metalness, roughness, input.surfaceIdentity);'
              : 'let reflected = vec3f(0.0);').replace('// DIFFUSE_GI_PROGRAM', this.diffuseGi
            ? this.diffuseGiOptions?.finalGather ? diffuseGiFinalGatherSampleSource : diffuseGiSampleSource : '')
            .replace('// INDIRECT_LIGHTING', this.diffuseGi
              ? this.diffuseGiOptions?.finalGather
                ? 'let ambient = finalGatherIrradiance(input.position.xy, input.world, n, geometricNormal) * base * (1.0 - metalness) * (0.96 / 3.14159265);'
                : 'let ambient = diffuseGiSurfaceIrradiance(input.world, n, geometricNormal) * base * (1.0 - metalness) * (0.96 / 3.14159265);'
              : 'let ambient = base * (0.045 + 0.065 * max(n.y, 0.0)) * (1.0 - metalness * 0.6);')
            .replace('// MATERIAL_PROGRAM',
            `const MATERIAL_UNLIT = ${program.shading === 'unlit'};\nconst MATERIAL_NORMAL_MAPPED = ${program.normalMapped};\n` + program.wgsl), 'material');
          const primitive: GPUPrimitiveState = { topology: 'triangle-list', cullMode: program.doubleSided ? 'none' : 'back' };
          const surface = await this.device.createRenderPipelineAsync({ label: 'material surface and velocity',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.geometryLayout, this.materialResources.layout,
              ...(this.lightingLayout ? [this.lightingLayout] : [])] }),
            vertex: { module, entryPoint: 'vertexMain' },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }, { format: 'rg16float' }] },
            primitive, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
          });
          const surfaceLod = this.hasGeometryLod ? await this.device.createRenderPipelineAsync({ label: 'material GPU LOD surface and velocity',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.geometryLayout, this.materialResources.layout,
              ...(this.lightingLayout ? [this.lightingLayout] : [])] }),
            vertex: { module, entryPoint: 'clusterVertexMain' },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }, { format: 'rg16float' }] },
            primitive, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
          }) : undefined;
          const shadow = await this.device.createRenderPipelineAsync({ label: 'shared material shadow coverage',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.shadowFrameLayout, this.geometryLayout, this.materialResources.layout] }),
            vertex: { module, entryPoint: 'shadowMain' }, fragment: { module, entryPoint: 'shadowFragment', targets: [] },
            primitive, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
          });
          const surfaceStream = this.hasGeometryStream ? await this.device.createRenderPipelineAsync({ label: 'material paged geometry surface and velocity',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.geometryLayout, this.materialResources.layout,
              ...(this.lightingLayout ? [this.lightingLayout] : [])] }),
            vertex: { module, entryPoint: 'pagedClusterVertexMain' },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba16float' }, { format: 'rg16float' }] },
            primitive, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
          }) : undefined;
          const query = this.worldQueries ? await this.worldQueries.createPipeline(program) : undefined;
          return { surface, surfaceLod, surfaceStream, shadow, query };
        })();
        this.materialPipelines.set(program.key, pending);
        // Permit a corrected retry after compilation/device errors; do not cache rejected work.
        void pending.then(() => this.compilingMaterials.delete(program.key), () => {
          this.compilingMaterials.delete(program.key);
          if (this.materialPipelines.get(program.key) === pending) this.materialPipelines.delete(program.key);
        });
      } else {
        // Successful reuse becomes most recently used; only unreferenced variants are evictable.
        this.materialPipelines.delete(program.key); this.materialPipelines.set(program.key, pending);
      }
      const pipelines = await pending;
      if (this.disposed) throw new Error('RENDERER_DISPOSED');
      return { material, materialPipelines: pipelines };
    } catch (error) {
      this.materialResources.release(material);
      if (this.disposed) throw new Error('RENDERER_DISPOSED');
      throw error;
    }
  }

  /** Prepare asynchronously, then publish the complete replacement atomically. Last request wins. */
  async setMaterial(batchName: string, binding: MaterialBinding): Promise<void> {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    const batch = this.batches.find(item => item.scene.name === batchName);
    if (!batch) throw new Error('BATCH_NOT_FOUND: ' + batchName);
    if (batch.scene.geometryStream?.queryProxy && binding.definition.alphaMode !== 'opaque') {
      throw new Error('GEOMETRY_PROXY_MASK_REQUIRES_FULL_SOURCE');
    }
    const request = ++batch.materialRequest;
    const replacement = await this.prepareMaterial(binding);
    if (this.disposed) { this.materialResources.release(replacement.material); throw new Error('RENDERER_DISPOSED'); }
    if (request !== batch.materialRequest) { this.materialResources.release(replacement.material); return; }
    const previous = batch.material;
    Object.assign(batch, replacement);
    this.materialResources.release(previous);
    this.materialRevision++;
  }

  setMaterialParameters(batchName: string, values: Float32Array): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    const batch = this.batches.find(item => item.scene.name === batchName);
    if (!batch) throw new Error('BATCH_NOT_FOUND: ' + batchName);
    this.materialResources.updateParameters(batch.material, values);
    this.materialRevision++;
  }

  setGeometryLod(settings: GeometryLodSettings): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.geometryLod) throw new Error('GEOMETRY_LOD_DISABLED');
    this.geometryLod.setSettings(settings);
    for (const batch of this.batches) batch.stream?.runtime.selection.setSettings(settings);
  }
  setOcclusionEnabled(enabled: boolean): void {
    if (this.disposed || this.errors.length) throw new Error('RENDERER_UNAVAILABLE');
    if (!this.occlusion) throw new Error('OCCLUSION_DISABLED');
    this.occlusion.setEnabled(enabled);
  }

  /** Explicit asynchronous diagnostic, never required to choose or draw geometry. */
  async probeGeometryLod(batchName: string): Promise<GeometryLodProbe> {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (this.errors.length) throw new Error(this.errors.join('\n'));
    if (!this.geometryLod) throw new Error('GEOMETRY_LOD_DISABLED');
    const stream = this.batches.find(batch => batch.scene.name === batchName)?.stream;
    if (stream) {
      const result = await stream.runtime.probe();
      return { frame: result.frame, indices: result.indices, triangles: result.indices / 3, overflow: result.overflow,
        occlusion: result.occlusion,
        selectedClusters: (result.selectedClusters ?? []).filter(cluster => cluster.visible).map(({ clusterId, instanceIndex, level }) => ({ clusterId, instanceIndex, level })) };
    }
    return this.geometryLod.probe(batchName);
  }

  retryGeometryPage(batchName: string, pageId: number): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    const stream = this.batches.find(batch => batch.scene.name === batchName)?.stream;
    if (!stream) throw new Error('GEOMETRY_STREAM_BATCH_NOT_FOUND'); stream.runtime.retry(pageId);
  }

  /** Disable reflected transport while keeping its bounded resources allocated. */
  setReflectionsEnabled(enabled: boolean): void {
    if (this.disposed || this.errors.length) throw new Error('RENDERER_UNAVAILABLE');
    if (!this.reflections) throw new Error('REFLECTIONS_DISABLED');
    this.reflections.setEnabled(enabled);
  }

  async probeReflections(indices: readonly number[]): Promise<ReflectionProbeResult> {
    if (this.disposed || this.errors.length) throw new Error('RENDERER_UNAVAILABLE');
    if (!this.reflections) throw new Error('REFLECTIONS_DISABLED');
    return this.reflections.probe(indices);
  }

  /** Disabled media leave surface radiance unchanged; resources remain allocated. */
  setVolumetricFogEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.volumetricFog) throw new Error('VOLUMETRIC_FOG_DISABLED');
    this.volumetricFog.setEnabled(enabled);
  }

  setVolumetricFogMedia(media: Pick<VolumetricFogOptions, 'global' | 'volumes'>): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.volumetricFog) throw new Error('VOLUMETRIC_FOG_DISABLED');
    this.volumetricFog.setMedia(media);
  }

  /** Toggle configured history without reallocating the view grid. */
  setVolumetricFogTemporalEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.volumetricFog) throw new Error('VOLUMETRIC_FOG_DISABLED');
    this.volumetricFog.setTemporalEnabled(enabled);
  }

  async probeVolumetricFogHistory(cells: readonly FogHistoryCell[]) {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.volumetricFog || !this.lastViewProjection) throw new Error('FOG_HISTORY_UNAVAILABLE');
    return this.volumetricFog.probeHistory(cells);
  }

  /** Disabling GI removes indirect lighting; it does not restore the legacy ambient fill. */
  setDiffuseGiEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.diffuseGi) throw new Error('DIFFUSE_GI_DISABLED');
    this.diffuseGi.setEnabled(enabled);
  }
  setDiffuseGiMultiBounce(enabled: boolean): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.diffuseGi) throw new Error('DIFFUSE_GI_DISABLED');
    this.diffuseGi.setMultiBounce(enabled);
  }
  /** Toggle configured surface gathering; disabled gathering uses the world probe field. */
  setDiffuseGiFinalGatherEnabled(enabled: boolean): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.diffuseGi) throw new Error('DIFFUSE_GI_DISABLED');
    this.diffuseGi.setFinalGatherEnabled(enabled);
  }
  /** Explicit reset for discontinuities outside the scene/material identity contracts. */
  invalidateDiffuseGiHistory(): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.diffuseGi) throw new Error('DIFFUSE_GI_DISABLED');
    this.diffuseGi.invalidateHistory();
  }
  async probeDiffuseGi(): Promise<DiffuseGiProbeResult> {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (this.errors.length) throw new Error(this.errors.join('\n'));
    if (!this.diffuseGi) throw new Error('DIFFUSE_GI_DISABLED');
    return this.diffuseGi.probe();
  }
  /** Explicit bounded readback of raw and filtered surface-lighting anchors. */
  async probeDiffuseGiFinalGather(indices: readonly number[]): Promise<FinalGatherProbeResult> {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (this.errors.length) throw new Error(this.errors.join('\n'));
    if (!this.diffuseGi) throw new Error('DIFFUSE_GI_DISABLED');
    return this.diffuseGi.probeFinalGather(indices);
  }

  /** Exposure in stops, applied only during final scene-linear HDR resolve. */
  setExposure(ev: number): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!Number.isFinite(ev) || ev < -16 || ev > 16) throw new Error('INVALID_EXPOSURE');
    this.exposureEV = ev;
    this.writeOutputSettings();
  }

  setAntiAliasing(mode: 'none' | 'fxaa'): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (mode !== 'none' && mode !== 'fxaa') throw new Error('INVALID_ANTI_ALIASING');
    this.antiAliasing = mode;
    this.writeOutputSettings();
  }

  private writeOutputSettings(): void {
    this.device.queue.writeBuffer(this.exposureBuffer, 0, new Float32Array([2 ** this.exposureEV, this.antiAliasing === 'fxaa' ? 1 : 0, 0, 0]));
  }

  /** A directional light points from the surface towards the light. Linear RGB radiance. */
  setSun(sun: { direction: [number, number, number]; color: [number, number, number] }): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (sun.direction.length !== 3 || sun.color.length !== 3 || ![...sun.direction, ...sun.color].every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value))) ||
        Math.hypot(...sun.direction) < 1e-6 || sun.color.some(value => value < 0 || !Number.isFinite(Math.fround(value)))) throw new Error('INVALID_SUN');
    const length = Math.hypot(...sun.direction);
    if (this.volumetricFogOptions) GpuVolumetricFog.validateSunColor(sun.color);
    const d = sun.direction.map(value => value / length) as [number, number, number];
    const projection = sunShadowProjection(d, this.sunShadowOptions);
    this.sunDirection = d; this.sunColor = [...sun.color];
    this.sunShadowProjection = projection;
    this.lightMatrix = projection.matrix;
  }

  /** Update world bounds without reallocating the map. Include all casters that can shadow the receivers. */
  setSunShadowFocus(focusBounds: SunShadowOptions['focusBounds']): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    const options = resolveSunShadowOptions({ ...this.sunShadowOptions, focusBounds }, this.device.limits.maxTextureDimension2D);
    const projection = sunShadowProjection(this.sunDirection, options);
    this.sunShadowOptions = options;
    this.sunShadowProjection = projection;
    this.lightMatrix = projection.matrix;
  }

  private async createBatch(scene: RenderBatch): Promise<Batch> {
    const capacity = scene.capacity;
    if (scene.geometryStream?.queryProxy && scene.mesh) throw new Error('GEOMETRY_STREAM_AMBIGUOUS_REPRESENTATION');
    if (scene.geometryStream?.queryProxy && scene.material && scene.material.definition.alphaMode !== 'opaque') {
      throw new Error('GEOMETRY_PROXY_MASK_REQUIRES_FULL_SOURCE');
    }
    const mesh = residentMesh(scene);
    if (!Number.isSafeInteger(capacity) || capacity < 1) throw new Error('INVALID_BATCH_CAPACITY');
    if (!Number.isSafeInteger(mesh.vertexCount) || mesh.vertexCount < 1
      || Math.ceil(capacity * mesh.vertexCount / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('DEFORMATION_DISPATCH_BUDGET');
    if (capacity * mesh.vertexCount * DEFORMED_VERTEX_BYTES > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize)) {
      throw new Error('BUFFER_BUDGET: shared deformation capacity');
    }
    const program = compileDeformation(scene.deformation);
    const morphBounds = validateMorphTargets(program, mesh.vertexCount, scene.morphTargets);
    const source = new DataView(mesh.vertices);
    for (let i = 0; i < mesh.vertexCount; i++) validateDeformationVertex(program, this.readVertex(source, i));
    let pipeline = this.deformationPipelines.get(program.key);
    if (!pipeline) {
      const module = await this.shader(deformationSource.replace('// OPERATOR_BODY', program.wgslBody), 'deformation:' + program.key);
      pipeline = await this.device.createComputePipelineAsync({ label: 'compiled deformation',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }), compute: { module, entryPoint: 'main' } });
      this.deformationPipelines.set(program.key, pipeline);
    }
    const rest = this.buffer(scene.name + ' source', mesh.vertices.byteLength, GPUBufferUsage.STORAGE, mesh.vertices);
    const indices = this.buffer(scene.name + ' indices', mesh.indices.byteLength, GPUBufferUsage.INDEX, new Uint32Array(mesh.indices).buffer);
    if (mesh.uvs && (mesh.uvs.length !== mesh.vertexCount * 2 || !mesh.uvs.every(Number.isFinite))) throw new Error('INVALID_MESH_UVS');
    validateMeshColors(mesh.colors, mesh.vertexCount);
    // One binding for static surface attributes. Uncolored assets retain their
    // original UV-only allocation; color is not duplicated per deformed instance.
    const uvData = new Float32Array(mesh.vertexCount * (mesh.colors ? 6 : 2));
    if (mesh.uvs) uvData.set(mesh.uvs);
    if (mesh.colors) uvData.set(mesh.colors, mesh.vertexCount * 2);
    const uvs = this.buffer(scene.name + ' UVs and colors', uvData.byteLength, GPUBufferUsage.STORAGE, uvData.buffer);
    const params = this.buffer(scene.name + ' params', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const instances = this.buffer(scene.name + ' instances', capacity * 224, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const palettes = this.buffer(scene.name + ' palettes', Math.max(1, capacity * program.jointCount * 2) * 64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const scalarStride = Math.max(1, program.scalarCount);
    const scalars = this.buffer(scene.name + ' scalars', capacity * scalarStride * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const tangentData = snapshotMeshTangents(mesh.tangents, mesh.vertexCount);
    const tangents = this.buffer(scene.name + ' source tangents', tangentData.byteLength, GPUBufferUsage.STORAGE, tangentData.buffer);
    const morphData = new Float32Array(Math.max(1, program.morphTargetCount * mesh.vertexCount) * 12);
    for (let t = 0; t < program.morphTargetCount; t++) for (let i = 0; i < mesh.vertexCount; i++) {
      morphData.set(scene.morphTargets[t].positions.subarray(i * 3, i * 3 + 3), (t * mesh.vertexCount + i) * 12);
      morphData.set(scene.morphTargets[t].normals.subarray(i * 3, i * 3 + 3), (t * mesh.vertexCount + i) * 12 + 4);
      const delta = scene.morphTargets[t].tangents;
      if (delta) morphData.set(delta.subarray(i * 3, i * 3 + 3), (t * mesh.vertexCount + i) * 12 + 8);
    }
    const morphs = this.buffer(scene.name + ' morph targets', morphData.byteLength, GPUBufferUsage.STORAGE, morphData.buffer);
    const output = this.buffer(scene.name + ' shared deformed', capacity * mesh.vertexCount * DEFORMED_VERTEX_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const material = await this.prepareMaterial(scene.material);
    const instanceData = new Float32Array(capacity * 56);
    const batch: Batch = {
      ...material, materialRequest: 0, sourceId: this.batches.length + 1,
      scene, mesh, rest, indices, params, instances, palettes, output, scalars, program, pipeline, morphBounds, bounds: new Map(), tangentData,
      compute: this.device.createBindGroup({ layout: this.computeLayout,
        entries: [params, rest, instances, palettes, output, morphs, scalars, tangents].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      render: undefined as unknown as GPUBindGroup,
      instanceData, instanceWords: new Uint32Array(instanceData.buffer), paletteData: new Float32Array(capacity * program.jointCount * 32),
      scalarData: new Float32Array(capacity * scalarStride * 2), history: new Map(), dirty: true, fogSceneChanged: true, lastInstances: [], uploadRanges: [],
    };
    try {
      if (scene.geometryLod) batch.geometryLod = this.geometryLod!.addBatch(scene, instances, () => batch.material.program.alphaMode === 'opaque');
      batch.render = this.device.createBindGroup({ layout: this.geometryLayout,
        entries: [output, instances, params, uvs, batch.geometryLod?.vertices ?? this.defaultClusterVertices].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      if (scene.geometryStream) {
        const runtime = await GpuGeometryStream.create(this.device, scene, instances,
          bytes => { this.stats.ownedGpuBytes += bytes; }, this.geometryDecoder, this.occlusion);
        try {
          const vertexCount = runtime.selection.physicalVertexCount;
          if (Math.ceil(vertexCount * capacity / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('GEOMETRY_STREAM_DEFORMATION_BUDGET');
          const pagedOutput = this.buffer(scene.name + ' paged shared state', vertexCount * capacity * DEFORMED_VERTEX_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const pagedParams = this.buffer(scene.name + ' paged deformation parameters', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          const key = program.key + ':physical-pages';
          let pagedPipeline = this.deformationPipelines.get(key);
          if (!pagedPipeline) {
            const code = deformationSource.replace('// GEOMETRY_VERTEX_FIELDS', 'uv:vec2f, flags:u32, padding:u32, tangent:vec4f, identity:vec4u, color:vec4f,')
              .replace('var t = restTangents[vertexIndex];', 'var t = vertex.tangent;').replace('// OPERATOR_BODY', program.wgslBody);
            const module = await this.shader(code, 'paged shared deformation');
            pagedPipeline = await this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }),
              compute: { module, entryPoint: 'main' } }); this.deformationPipelines.set(key, pagedPipeline);
          }
          batch.stream = { runtime, output: pagedOutput, params: pagedParams, pipeline: pagedPipeline,
            compute: this.device.createBindGroup({ layout: this.computeLayout,
              entries: [pagedParams, runtime.pool.vertices, instances, palettes, pagedOutput, morphs, scalars, tangents].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
            render: this.device.createBindGroup({ layout: this.geometryLayout,
              entries: [pagedOutput, instances, pagedParams, uvs, runtime.pool.vertices].map((buffer, binding) => ({ binding, resource: { buffer } })) }) };
        } catch (error) { await runtime.dispose(); throw error; }
      }
      return batch;
    } catch (error) { this.materialResources.release(material.material); throw error; }
  }

  private upload(batch: Batch): void {
    const states = batch.scene.instances;
    batch.fogSceneChanged = states.length !== batch.lastInstances.length;
    if (states.length > batch.scene.capacity) throw new Error('INSTANCE_CAPACITY: ' + batch.scene.name);
    const committed = batch.scene.inputMode === 'committed' && batch.lastInputMode === 'committed';
    const reusable = (state: InstanceState, i: number, previous: Previous | undefined): boolean =>
      committed && state === batch.lastInstances[i] && previous?.settled === true && previous.source === state
      && previous.version === state.historyVersion && previous.generation === (state.entityGeneration ?? 0);
    // Inputs are committed frame objects. After one stationary frame has settled
    // previous=current, keep the shared output until this batch changes again.
    if (committed && states.length === batch.lastInstances.length && states.every((state, i) =>
      reusable(state, i, batch.history.get(state.entityId)))) {
      batch.dirty = false;
      return;
    }
    batch.dirty = true;
    const alive = new Set<number>();
    // Validate identity before updating any history. A duplicate would otherwise
    // overwrite the previous pose used by a second slot in the same batch.
    for (const state of states) {
      if (!Number.isSafeInteger(state.entityId) || state.entityId < 0 || state.entityId > 0xffffffff) throw new Error('INVALID_ENTITY_ID');
      const generation = state.entityGeneration ?? 0;
      if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffffffff) throw new Error('INVALID_ENTITY_GENERATION');
      if (alive.has(state.entityId)) throw new Error('DUPLICATE_ENTITY_ID: ' + batch.scene.name);
      alive.add(state.entityId);
    }
    const ranges = batch.uploadRanges;
    ranges.length = 0;
    let coalesced = false;
    const joints = batch.program.jointCount, scalarCount = Math.max(1, batch.program.scalarCount);
    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      const generation = state.entityGeneration ?? 0;
      let previous = batch.history.get(state.entityId);
      if (reusable(state, i, previous)) continue;
      // Upload dirtiness also includes mutable inputs and settling previous pose.
      // Fog history depends on actual rendered changes, not a CPU upload request.
      // Compare bounded inputs before overwriting the existing render snapshot;
      // no second geometry copy or per-frame GPU readback is needed.
      if (this.volumetricFog?.temporalConfigured && !batch.fogSceneChanged) {
        batch.fogSceneChanged = !previous || previous.version !== state.historyVersion || previous.generation !== generation
          || differsF32(state.model, batch.instanceData, i * 56)
          || differsF32(state.normalMatrix, batch.instanceData, i * 56 + 32)
          || differsF32(state.color, batch.instanceData, i * 56 + 48)
          || Math.fround(state.metalness) !== batch.instanceData[i * 56 + 52]
          || Math.fround(state.roughness) !== batch.instanceData[i * 56 + 53]
          || state.bones.some((bone, j) => differsF32(bone, previous!.bones[j]))
          || differsF32(state.deformScalars, batch.scalarData, i * scalarCount * 2, batch.program.scalarCount);
      }
      // Adjacent changes share a write. Highly fragmented updates use one
      // envelope instead of issuing an unbounded number of tiny queue writes.
      if (coalesced) ranges[1] = i + 1;
      else if (ranges.length && ranges[ranges.length - 1] === i) ranges[ranges.length - 1] = i + 1;
      else if (ranges.length < 32) ranges.push(i, i + 1);
      else { ranges[1] = i + 1; ranges.length = 2; coalesced = true; }
      const bounds = conservativeDeformationBounds(batch.program, batch.mesh.bounds, batch.morphBounds, state.bones, state.deformScalars, state.model);
      if (!previous || previous.version !== state.historyVersion || previous.generation !== generation) {
        previous = { model: state.model.slice(), bones: state.bones.map(b => b.slice()), scalars: [...state.deformScalars], version: state.historyVersion, generation, settled: true };
        batch.history.set(state.entityId, previous);
      }
      const offset = i * 56;
      batch.instanceData.set(state.model, offset);
      batch.instanceData.set(previous.model, offset + 16);
      batch.instanceData.set(state.normalMatrix, offset + 32);
      batch.instanceData.set(state.color, offset + 48);
      batch.instanceData.set([state.metalness, state.roughness, 0, 0], offset + 52);
      // Reserved material words carry exact integer identity, not float numeric IDs.
      batch.instanceWords[offset + 54] = batch.sourceId; batch.instanceWords[offset + 55] = state.entityId;
      for (let j = 0; j < joints; j++) {
        batch.paletteData.set(state.bones[j], i * joints * 32 + j * 16);
        batch.paletteData.set(previous.bones[j], i * joints * 32 + joints * 16 + j * 16);
        previous.bones[j].set(state.bones[j]);
      }
      for (let s = 0; s < batch.program.scalarCount; s++) {
        batch.scalarData[i * scalarCount * 2 + s] = state.deformScalars[s];
        batch.scalarData[i * scalarCount * 2 + scalarCount + s] = previous.scalars[s];
      }
      batch.bounds.set(state.entityId, bounds);
      previous.settled = previous.source === undefined || previous.source === state;
      previous.source = state;
      previous.model.set(state.model); previous.scalars = [...state.deformScalars]; previous.version = state.historyVersion; previous.generation = generation;
    }
    for (const id of batch.history.keys()) if (!alive.has(id)) { batch.history.delete(id); batch.bounds.delete(id); }
    if (states.length !== batch.lastInstances.length) {
      this.device.queue.writeBuffer(batch.params, 0, new Uint32Array([batch.mesh.vertexCount, states.length, joints, scalarCount]));
      this.stats.uploadedBytes += 16;
    }
    for (let r = 0; r < ranges.length; r += 2) {
      const start = ranges[r], count = ranges[r + 1] - start;
      this.device.queue.writeBuffer(batch.instances, start * 224, batch.instanceData, start * 56, count * 56);
      if (joints) this.device.queue.writeBuffer(batch.palettes, start * joints * 128, batch.paletteData, start * joints * 32, count * joints * 32);
      this.device.queue.writeBuffer(batch.scalars, start * scalarCount * 8, batch.scalarData, start * scalarCount * 2, count * scalarCount * 2);
      this.stats.uploadedBytes += count * (224 + joints * 128 + scalarCount * 8);
    }
    batch.lastInstances = [...states];
    batch.lastInputMode = batch.scene.inputMode;
  }

  private resize(): void {
    const ratio = Math.min(devicePixelRatio, 1.5);
    const width = Math.max(1, Math.min(this.device.limits.maxTextureDimension2D, Math.round(this.canvas.clientWidth * ratio)));
    const height = Math.max(1, Math.min(this.device.limits.maxTextureDimension2D, Math.round(this.canvas.clientHeight * ratio)));
    if (width === this.stats.width && height === this.stats.height) return;
    this.volumetricFog?.planResize(width, height);
    this.occlusion?.planResize(width, height);
    if (this.depth) {
      this.stats.ownedGpuBytes -= this.stats.width * this.stats.height * 16;
      this.depth.destroy(); this.velocity!.destroy(); this.hdr!.destroy();
    }
    this.canvas.width = width; this.canvas.height = height;
    this.depth = this.device.createTexture({ size: [width, height], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
    this.velocity = this.device.createTexture({ size: [width, height], format: 'rg16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.hdr = this.device.createTexture({ label: 'scene-linear HDR', size: [width, height], format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
    this.volumetricFog?.resize(width, height, this.depth, this.hdr, this.shadow);
    this.resolveGroup = this.device.createBindGroup({ layout: this.resolveLayout, entries: [
      { binding: 0, resource: (this.volumetricFog?.colorTexture ?? this.hdr).createView() }, { binding: 1, resource: { buffer: this.exposureBuffer } },
    ] });
    this.stats.width = width; this.stats.height = height;
    this.stats.ownedGpuBytes += width * height * 16;
    this.invalidateHistory();
  }

  /** Reject camera and object history for a cut, teleport, or topology change. */
  invalidateHistory(): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    this.lastViewProjection = undefined;
    this.volumetricFog?.invalidateHistory();
    for (const batch of this.batches) batch.history.clear();
  }

  render(eye: [number, number, number], target: [number, number, number]): void {
    if (this.disposed || this.errors.length) return;
    for (const batch of this.batches) if (batch.stream) {
      batch.stream.runtime.validateSources();
      const state = batch.stream.runtime.pool.diagnostics();
      if (state.framesInFlight >= state.limits.maxFramesInFlight) { this.stats.streamBackpressureFrames++; return; }
    }
    const started = performance.now();
    this.stats.uploadedBytes = 256; this.stats.deformationDispatches = 0;
    this.resize();
    this.worldQueries?.validateSources();
    this.geometryLod?.validateSources();
    for (const batch of this.batches) this.upload(batch);
    this.stats.uploadedBytes += this.worldQueries?.updateSnapshot(this.stats.frames + 1, this.materialRevision) ?? 0;
    const vp = multiply(perspective(Math.PI / 3.2, this.stats.width / this.stats.height, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
    this.stats.uploadedBytes += this.occlusion?.update(this.stats.frames + 1, this.stats.width, this.stats.height, vp, this.depth!,
      !this.geometryLod!.settings.frozen) ?? 0;
    this.frameData.set(vp); this.frameData.set(this.lastViewProjection ?? vp, 16); this.frameData.set(this.lightMatrix, 32);
    this.frameData.set([...eye, 1], 48); this.frameData.set([...this.sunDirection, 0], 52); this.frameData.set([...this.sunColor, 1], 56);
    const debug = this.geometryLod?.settings.debug;
    new Uint32Array(this.frameData.buffer)[60] = debug === 'clusters' ? 1 : debug === 'levels' ? 2 : 0;
    this.device.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    this.stats.uploadedBytes += this.volumetricFog?.update(eye, target, Math.PI / 3.2, 0.1, 150,
      this.sunDirection, this.sunColor, this.lightMatrix, vp,
      { cameraCut: !this.lastViewProjection, sceneChanged: this.batches.some(batch => batch.fogSceneChanged), materialRevision: this.materialRevision }) ?? 0;
    this.stats.uploadedBytes += this.geometryLod?.update(this.stats.frames + 1, vp, eye, this.stats.height / (2 * Math.tan(Math.PI / 6.4))) ?? 0;
    for (const batch of this.batches) if (batch.stream) {
      const count = batch.material.program.alphaMode === 'opaque' ? batch.scene.instances.length : 0;
      this.stats.uploadedBytes += batch.stream.runtime.update(this.stats.frames + 1, count, vp, eye, this.stats.height / (2 * Math.tan(Math.PI / 6.4)));
      this.device.queue.writeBuffer(batch.stream.params, 0, new Uint32Array([batch.stream.runtime.selection.physicalVertexCount, count, 0, 0]));
      this.stats.uploadedBytes += 16;
    }
    this.stats.uploadedBytes += this.diffuseGi?.update(this.stats.frames + 1, this.sunDirection, this.sunColor,
      this.materialRevision, this.worldQueries!.historyRevision) ?? 0;
    this.stats.uploadedBytes += this.diffuseGi?.updateFinalGather(this.stats.frames + 1, this.stats.width, this.stats.height,
      eye, vp, this.lastViewProjection, this.batches.some(batch => batch.dirty), this.materialRevision) ?? 0;
    this.stats.uploadedBytes += this.reflections?.update({ frame: this.stats.frames + 1, materialRevision: this.materialRevision,
      width: this.stats.width, height: this.stats.height, eye, target, fovY: Math.PI / 3.2, near: 0.1, far: 150,
      sunDirection: this.sunDirection, sunColor: this.sunColor }) ?? 0;
    this.lastViewProjection = vp;
    const encoder = this.device.createCommandEncoder({ label: 'deform → shadow → HDR surface → resolve' });
    const readTiming = !!this.querySet && !this.timingPending && this.stats.frames % 30 === 0;
    const leases: { pool: GpuGeometryPagePool; lease: GeometryGpuFrameLease }[] = [];
    try {
      for (const batch of this.batches) if (batch.stream) {
        const pool = batch.stream.runtime.pool; leases.push({ pool, lease: pool.acquireFrame([...pool.residentPageIds()]) });
      }
      this.graph.execute(encoder);
      if (readTiming) {
        encoder.resolveQuerySet(this.querySet!, 0, this.timestampCount, this.queryResolve!, 0);
        encoder.copyBufferToBuffer(this.queryResolve!, 0, this.queryRead!, 0, this.timestampCount * 8);
        this.timingPending = true;
      }
      if (leases.length) void GpuGeometryPagePool.submitFrames(leases, [encoder.finish()]).then(result => {
        if (result.error && !this.disposed) this.errors.push(result.error);
      });
      else this.device.queue.submit([encoder.finish()]);
    } catch (error) {
      for (const { pool, lease } of leases) try { pool.cancelFrame(lease); } catch { /* Already submitted or released. */ }
      throw error;
    }
    for (const batch of this.batches) batch.stream?.runtime.afterSubmit();
    this.volumetricFog?.afterSubmit();
    if (readTiming) void this.readTiming(this.stats.frames + 1);
    this.stats.frames++;
    this.stats.drawCalls = this.batches.filter(batch => batch.scene.instances.length).length * 2 + 1 + (this.volumetricFog ? 1 : 0)
      + (this.occlusion ? this.batches.filter(batch => batch.scene.instances.length && batch.material.program.alphaMode === 'opaque'
        && (batch.geometryLod || batch.stream)).length : 0);
    this.stats.triangles = this.batches.reduce((sum, batch) => sum + (batch.scene.geometryStream?.directory.sourceTriangleCount ?? batch.mesh.indices.length / 3) * batch.scene.instances.length, 0);
    this.stats.deformedVertices = this.batches.reduce((sum, batch) => sum + (batch.dirty ? batch.mesh.vertexCount * batch.scene.instances.length : 0)
      + (batch.stream && batch.material.program.alphaMode === 'opaque' ? batch.stream.runtime.selection.physicalVertexCount * batch.scene.instances.length : 0), 0);
    this.stats.cpuSubmitMs = performance.now() - started;
  }
  /** Query the last submitted deformation snapshot; material changes use the current revision. */
  async traceRays(rays: readonly WorldRay[], options: { textureLod?: number } = {}): Promise<WorldQueryResult> {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (!this.worldQueries) throw new Error('WORLD_QUERY_DISABLED');
    if (this.stats.frames === 0) throw new Error('WORLD_QUERY_NO_FRAME');
    if (this.errors.length) throw new Error(this.errors.join('\n'));
    return this.worldQueries.traceRays(rays, options, this.materialRevision);
  }
  private get reflectionTimestampIndex(): number {
    return 8 + (this.worldQueries ? 2 : 0) + (this.geometryLod ? 2 : 0) + (this.diffuseGi ? 2 : 0) + (this.diffuseGiOptions?.finalGather ? 2 : 0);
  }
  private get occlusionTimestampIndex(): number { return this.reflectionTimestampIndex + (this.reflections ? 2 : 0); }
  private timestamp(index: number): GPUComputePassTimestampWrites | undefined {
    return this.querySet ? { querySet: this.querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } : undefined;
  }
  private encodeDeformation(encoder: GPUCommandEncoder): void {
    const compute = encoder.beginComputePass({ label: 'shared deformation', timestampWrites: this.timestamp(0) });
    for (const batch of this.batches) {
      if (batch.stream && batch.scene.instances.length && batch.material.program.alphaMode === 'opaque') {
        compute.setPipeline(batch.stream.pipeline); compute.setBindGroup(0, batch.stream.compute);
        compute.dispatchWorkgroups(Math.ceil(batch.stream.runtime.selection.physicalVertexCount * batch.scene.instances.length / 64));
        this.stats.deformationDispatches++;
      }
      if (!batch.scene.instances.length || !batch.dirty) continue;
      compute.setPipeline(batch.pipeline); compute.setBindGroup(0, batch.compute);
      compute.dispatchWorkgroups(Math.ceil(batch.mesh.vertexCount * batch.scene.instances.length / 64));
      this.stats.deformationDispatches++;
    }
    compute.end();
  }
  private encodeGeometrySelection(encoder: GPUCommandEncoder): void {
    const timestamp = this.timestamp(this.worldQueries ? 10 : 8);
    if (!this.hasGeometryStream) { this.geometryLod!.encode(encoder, timestamp); return; }
    const begin = encoder.beginComputePass({ label: 'all geometry selection begin', timestampWrites: timestamp
      ? { querySet: timestamp.querySet, beginningOfPassWriteIndex: timestamp.beginningOfPassWriteIndex } : undefined }); begin.end();
    this.geometryLod!.encode(encoder);
    for (const batch of this.batches) batch.stream?.runtime.selection.encode(encoder);
    const end = encoder.beginComputePass({ label: 'all geometry selection end', timestampWrites: timestamp
      ? { querySet: timestamp.querySet, endOfPassWriteIndex: timestamp.endOfPassWriteIndex } : undefined }); end.end();
  }
  private encodeShadow(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginRenderPass({ label: 'sun shadow', colorAttachments: [], timestampWrites: this.timestamp(2),
      depthStencilAttachment: { view: this.shadow.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    pass.setBindGroup(0, this.shadowFrameBindGroup);
    this.draw(pass, 'shadow'); pass.end();
  }
  private encodeGeometryRecovery(encoder: GPUCommandEncoder): void {
    const timestamp = this.timestamp(this.occlusionTimestampIndex + 2);
    const begin = encoder.beginComputePass({ label: 'geometry visibility recovery begin', timestampWrites: timestamp
      ? { querySet: timestamp.querySet, beginningOfPassWriteIndex: timestamp.beginningOfPassWriteIndex } : undefined }); begin.end();
    this.geometryLod!.encodeRecovery(encoder);
    for (const batch of this.batches) batch.stream?.runtime.selection.encodeRecovery(encoder);
    const end = encoder.beginComputePass({ label: 'geometry visibility recovery end', timestampWrites: timestamp
      ? { querySet: timestamp.querySet, endOfPassWriteIndex: timestamp.endOfPassWriteIndex } : undefined }); end.end();
  }
  private encodeSurface(encoder: GPUCommandEncoder, recovery = false): void {
    const pass = encoder.beginRenderPass({ label: recovery ? 'recovered PBR and velocity' : 'PBR and velocity',
      timestampWrites: this.timestamp(recovery ? this.occlusionTimestampIndex + 4 : 4), colorAttachments: [
      { view: this.hdr!.createView(), clearValue: [0.0035, 0.0055, 0.0065, 1], loadOp: recovery ? 'load' : 'clear', storeOp: 'store' },
      { view: this.velocity!.createView(), clearValue: [0, 0, 0, 0], loadOp: recovery ? 'load' : 'clear', storeOp: 'store' },
    ], depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: recovery ? 'load' : 'clear', depthStoreOp: 'store' } });
    pass.setBindGroup(0, this.frameBindGroup);
    if (this.lightingLayout) pass.setBindGroup(3, this.lightingGroups[this.diffuseGi?.surfaceBindingIndex ?? 0]);
    this.draw(pass, 'surface', recovery); pass.end();
  }
  private encodeResolve(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginRenderPass({ label: 'HDR exposure and display transfer', timestampWrites: this.timestamp(6), colorAttachments: [
      { view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
    ] });
    pass.setPipeline(this.resolvePipeline); pass.setBindGroup(0, this.resolveGroup!); pass.draw(3); pass.end();
  }
  diagnostics() {
    return { deformationVariants: this.deformationPipelines.size, materialVariants: this.materialPipelines.size,
      worldQueries: this.worldQueries?.diagnostics() ?? null,
      geometryLod: this.geometryLod?.diagnostics() ?? null,
      geometryStreams: this.batches.filter(batch => batch.stream).map(batch => ({ name: batch.scene.name, ...batch.stream!.runtime.diagnostics() })),
      geometryDecoding: this.geometryDecoder?.diagnostics() ?? null,
      diffuseGi: this.diffuseGi?.diagnostics() ?? null,
      volumetricFog: this.volumetricFog?.diagnostics() ?? null,
      reflections: this.reflections?.diagnostics() ?? null,
      occlusion: this.occlusion?.diagnostics() ?? null,
      sunShadow: { resolution: this.sunShadowOptions.resolution, bytes: this.sunShadowOptions.bytes,
        focusBounds: this.sunShadowOptions.focusBounds ?? null, width: this.sunShadowProjection.width,
        depth: this.sunShadowProjection.depth, texelWorldSize: this.sunShadowProjection.texelWorldSize,
        projection: this.sunShadowOptions.focusBounds ? 'world-bounds-snapped' : 'legacy-fixed', cascades: 1 },
      output: { sceneFormat: 'rgba16float', exposureEV: this.exposureEV, antiAliasing: this.antiAliasing,
        transfer: 'sRGB', toneMap: 'filmic-fit', maxSceneRadiance: 65504 },
      materials: this.materialResources.diagnostics(), materialTexturePreparation: this.materialResources.preparationDiagnostics(),
      materialRevision: this.materialRevision, passes: this.graph.order,
      lifetimes: Object.fromEntries(this.graph.lifetimes),
      bounds: this.batches.flatMap(batch => Array.from(batch.bounds, ([entityId, bounds]) => ({ entityId, ...bounds }))) };
  }
  private readVertex(source: DataView, vertex: number, tangents?: Float32Array) {
    const floats = (offset: number, count: number) => Array.from({ length: count }, (_, axis) => source.getFloat32(vertex * 80 + offset + axis * 4, true));
    return { position: floats(0, 3) as [number, number, number], normal: floats(16, 3) as [number, number, number],
      joints: Array.from({ length: 4 }, (_, axis) => source.getUint32(vertex * 80 + 32 + axis * 4, true)) as [number, number, number, number],
      weights: floats(48, 4) as [number, number, number, number],
      ...(tangents?.[vertex * 4 + 3] ? { tangent: Array.from(tangents.subarray(vertex * 4, vertex * 4 + 4)) as [number, number, number, number] } : {}) };
  }
  private draw(pass: GPURenderPassEncoder, kind: 'surface' | 'shadow', recovery = false): void {
    for (const batch of this.batches) {
      if (!batch.scene.instances.length) continue;
      if (kind === 'surface' && batch.stream && batch.material.program.alphaMode === 'opaque') {
        pass.setPipeline(batch.materialPipelines.surfaceStream!); pass.setBindGroup(2, batch.material.bindGroup);
        pass.setBindGroup(1, batch.stream.render); pass.setIndexBuffer(batch.stream.runtime.selection.indices, 'uint32');
        pass.drawIndexedIndirect(batch.stream.runtime.selection.arguments, recovery ? 32 : 0); continue;
      }
      const lod = kind === 'surface' && batch.material.program.alphaMode === 'opaque' ? batch.geometryLod : undefined;
      if (recovery && !lod) continue;
      pass.setPipeline(lod ? batch.materialPipelines.surfaceLod! : batch.materialPipelines[kind]); pass.setBindGroup(2, batch.material.bindGroup);
      if (lod) {
        pass.setBindGroup(1, batch.render); pass.setIndexBuffer(lod.indices, 'uint32');
        pass.drawIndexedIndirect(lod.arguments, recovery ? 32 : 0);
        continue;
      }
      pass.setBindGroup(1, batch.render); pass.setIndexBuffer(batch.indices, 'uint32');
      pass.drawIndexed(batch.mesh.indices.length, batch.scene.instances.length);
    }
  }
  private async readTiming(frame: number): Promise<void> {
    try {
      await this.queryRead!.mapAsync(GPUMapMode.READ);
      const values = new BigUint64Array(this.queryRead!.getMappedRange());
      this.stats.gpuMs = { deformation: Number(values[1] - values[0]) / 1e6,
        shadow: Number(values[3] - values[2]) / 1e6, surface: Number(values[5] - values[4]) / 1e6
          + (this.occlusion ? Number(values[this.occlusionTimestampIndex + 5] - values[this.occlusionTimestampIndex + 4]) / 1e6 : 0),
        resolve: Number(values[7] - values[6]) / 1e6,
        worldQueryRefit: this.worldQueries ? Number(values[9] - values[8]) / 1e6 : 0,
        geometryLod: this.geometryLod ? Number(values[this.worldQueries ? 11 : 9] - values[this.worldQueries ? 10 : 8]) / 1e6 : 0,
        diffuseGi: this.diffuseGi ? Number(values[this.geometryLod ? 13 : 11] - values[this.geometryLod ? 12 : 10]) / 1e6 : 0,
        finalGather: this.diffuseGiOptions?.finalGather ? Number(values[this.geometryLod ? 15 : 13] - values[this.geometryLod ? 14 : 12]) / 1e6 : 0,
        reflections: this.reflections ? Number(values[this.reflectionTimestampIndex + 1] - values[this.reflectionTimestampIndex]) / 1e6 : 0,
        occlusion: this.occlusion ? Number(values[this.occlusionTimestampIndex + 1] - values[this.occlusionTimestampIndex]
          + values[this.occlusionTimestampIndex + 3] - values[this.occlusionTimestampIndex + 2]) / 1e6 : 0,
        volumetricFog: this.volumetricFog ? (this.volumetricFog.temporalConfigured ? [8, 6, 4, 2] : [6, 4, 2]).reduce((sum, offset) => sum
          + Number(values[this.timestampCount - offset + 1] - values[this.timestampCount - offset]) / 1e6, 0) : 0 };
      this.stats.gpuSampleFrame = frame;
      this.queryRead!.unmap();
    } catch (error) { if (!this.disposed) this.errors.push('TIMESTAMP_READBACK: ' + String(error)); }
    finally { this.timingPending = false; }
  }

  /** Diagnostic only. WebGPU requires entire depth subresources in texture-to-buffer copies. */
  async probeDepth(points: readonly [number, number, number][]): Promise<{ surface: number; shadow: number }[]> {
    if (this.disposed || !this.depth || !this.lastViewProjection) throw new Error('NO_RENDERED_DEPTH');
    if (points.length > 4096 || points.some(point => point.length !== 3 || !point.every(Number.isFinite))) throw new Error('INVALID_DEPTH_PROBES');
    if (!points.length) return [];
    const encoder = this.device.createCommandEncoder({ label: 'diagnostic depth readback' });
    const sources = [
      { texture: this.depth, width: this.stats.width, height: this.stats.height, matrix: this.frameData.slice(0, 16) },
      { texture: this.shadow, width: this.sunShadowOptions.resolution, height: this.sunShadowOptions.resolution, matrix: this.frameData.slice(32, 48) },
    ];
    const reads = sources.map(source => {
      const stride = Math.ceil(source.width * 4 / 256) * 256;
      const buffer = this.device.createBuffer({ size: stride * source.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyTextureToBuffer({ texture: source.texture, aspect: 'depth-only' }, { buffer, bytesPerRow: stride }, [source.width, source.height]);
      return { ...source, stride, buffer };
    });
    this.device.queue.submit([encoder.finish()]);
    try {
      // Wait for every mapping even after one failure before releasing temporary resources.
      const results = await Promise.allSettled(reads.map(async read => {
        await read.buffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(read.buffer.getMappedRange());
        return points.map(point => {
          const m = read.matrix, w = m[3] * point[0] + m[7] * point[1] + m[11] * point[2] + m[15];
          const ndc = transformPoint(m, point);
          if (w <= 0 || ndc.some(value => !Number.isFinite(value)) || Math.abs(ndc[0]) >= 1 || Math.abs(ndc[1]) >= 1 || ndc[2] < 0 || ndc[2] > 1) throw new Error('DEPTH_PROBE_OUTSIDE_VIEW');
          const x = Math.floor((ndc[0] * 0.5 + 0.5) * read.width), y = Math.floor((0.5 - ndc[1] * 0.5) * read.height);
          return data[y * read.stride / 4 + x];
        });
      }));
      const rejected = results.find(result => result.status === 'rejected');
      if (rejected?.status === 'rejected') throw rejected.reason;
      const values = results.map(result => (result as PromiseFulfilledResult<number[]>).value);
      return points.map((_, i) => ({ surface: values[0][i], shadow: values[1][i] }));
    } finally { for (const read of reads) read.buffer.destroy(); }
  }

  /** Diagnostic only: exact top-left-origin pixel indices, returning current-minus-previous UV deltas. */
  async probeVelocity(points: readonly { x: number; y: number }[]): Promise<number[][]> {
    if (this.disposed || !this.velocity || !this.lastViewProjection) throw new Error('NO_RENDERED_VELOCITY');
    return this.readFloat16Pixels(this.velocity, points, 2, 'VELOCITY');
  }

  /** Diagnostic only: linear scene radiance including media, before exposure/display transfer. */
  async probeHdr(points: readonly { x: number; y: number }[]): Promise<number[][]> {
    if (this.disposed || !this.hdr || !this.lastViewProjection) throw new Error('NO_RENDERED_HDR');
    return this.readFloat16Pixels(this.volumetricFog?.colorTexture ?? this.hdr, points, 4, 'HDR');
  }

  /** Diagnostic only: surface/sky radiance before participating media. */
  async probeSurfaceHdr(points: readonly { x: number; y: number }[]): Promise<number[][]> {
    if (this.disposed || !this.hdr || !this.lastViewProjection) throw new Error('NO_RENDERED_HDR');
    return this.readFloat16Pixels(this.hdr, points, 4, 'HDR');
  }

  private async readFloat16Pixels(texture: GPUTexture, points: readonly { x: number; y: number }[],
    channels: number, label: string): Promise<number[][]> {
    const width = this.stats.width, height = this.stats.height;
    if (points.length > 1024 || points.some(point => !Number.isSafeInteger(point.x) || !Number.isSafeInteger(point.y)
      || point.x < 0 || point.y < 0 || point.x >= width || point.y >= height)) throw new Error('INVALID_' + label + '_PROBES');
    if (!points.length) return [];
    // A one-pixel copy occupies one 256-byte row. This keeps readback bounded by
    // the requested sample count instead of copying the entire viewport texture.
    const read = this.device.createBuffer({ label: 'diagnostic ' + label + ' readback', size: points.length * 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder({ label: 'diagnostic ' + label + ' readback' });
      for (let i = 0; i < points.length; i++) {
        encoder.copyTextureToBuffer({ texture, origin: { x: points[i].x, y: points[i].y } },
          { buffer: read, offset: i * 256, bytesPerRow: 256, rowsPerImage: 1 }, { width: 1, height: 1 });
      }
      this.device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      try {
        const words = new Uint16Array(read.getMappedRange());
        const half = (bits: number): number => {
          const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, fraction = bits & 1023;
          if (exponent === 0) return sign * fraction * 2 ** -24;
          if (exponent === 31) return fraction ? NaN : sign * Infinity;
          return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
        };
        return points.map((_, i) => {
          const result = Array.from({ length: channels }, (_, channel) => half(words[i * 128 + channel]));
          if (!result.every(Number.isFinite)) throw new Error('NONFINITE_' + label);
          return result;
        });
      } finally { read.unmap(); }
    } finally { read.destroy(); }
  }

  /** Explicit diagnostic readback only; never used by the production frame loop. */
  async verifyDeformation(options: { allInstances?: boolean; sampleVertices?: readonly number[] } = {}): Promise<DeformationVerification> {
    if (options.sampleVertices && (options.sampleVertices.length > 1024 || options.sampleVertices.some(index => !Number.isSafeInteger(index) || index < 0))) throw new Error('INVALID_SAMPLE_VERTICES');
    const sampleVertices = new Set([0, ...options.sampleVertices ?? []]);
    let maxError = 0, vertices = 0, maxNormalError = 0, maxTangentError = 0, boundsViolations = 0;
    const samples: DeformationVerification['samples'] = [];
    for (const batch of this.batches) {
      if (!batch.scene.instances.length) continue;
      const instanceCount = options.allInstances ? batch.scene.instances.length : 1;
      const count = batch.mesh.vertexCount * instanceCount;
      const read = this.device.createBuffer({ size: count * DEFORMED_VERTEX_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const joints = batch.program.jointCount, scalarCount = Math.max(1, batch.program.scalarCount);
      const instances = batch.instanceData.slice(0, instanceCount * 56), palette = batch.paletteData.slice(0, instanceCount * joints * 32), scalars = batch.scalarData.slice(0, instanceCount * scalarCount * 2);
      const entityIds = batch.scene.instances.slice(0, instanceCount).map(state => state.entityId);
      const instanceBounds = entityIds.map(id => structuredClone(batch.bounds.get(id)!));
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(batch.output, 0, read, 0, count * DEFORMED_VERTEX_BYTES);
      this.device.queue.submit([encoder.finish()]);
      try {
        await read.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(read.getMappedRange()), source = new DataView(batch.mesh.vertices);
        for (let vertex = 0; vertex < count; vertex++) {
          const instanceIndex = Math.floor(vertex / batch.mesh.vertexCount), vertexIndex = vertex % batch.mesh.vertexCount;
          const instance = instances.subarray(instanceIndex * 56, instanceIndex * 56 + 56), bounds = instanceBounds[instanceIndex];
          if (sampleVertices.has(vertexIndex)) samples.push({ batch: batch.scene.name, entityId: entityIds[instanceIndex], vertexIndex,
            current: Array.from(result.subarray(vertex * 16, vertex * 16 + 3)), previous: Array.from(result.subarray(vertex * 16 + 8, vertex * 16 + 11)) });
          for (let history = 0; history < 2; history++) {
            const paletteOffset = instanceIndex * joints * 32 + history * joints * 16;
            const bones = Array.from({ length: joints }, (_, j) => palette.subarray(paletteOffset + j * 16, paletteOffset + j * 16 + 16));
            const reference = evaluateDeformation(batch.program, this.readVertex(source, vertexIndex, batch.tangentData), vertexIndex, batch.scene.morphTargets, bones, scalars.subarray(instanceIndex * scalarCount * 2 + history * scalarCount));
            const expected = transformPoint(instance.subarray(history * 16, history * 16 + 16), reference.position);
            for (let axis = 0; axis < 3; axis++) {
              const actual = result[vertex * 16 + history * 8 + axis];
              if (!Number.isFinite(actual)) throw new Error('NONFINITE_DEFORMATION');
              maxError = Math.max(maxError, Math.abs(actual - expected[axis]));
              if (history === 0 && (actual < bounds.min[axis] - 0.0002 || actual > bounds.max[axis] + 0.0002)) boundsViolations++;
            }
            if (history === 0) {
              const m = instance.subarray(32, 48), n = reference.normal;
              const worldNormal = [0, 1, 2].map(row => m[row] * n[0] + m[4 + row] * n[1] + m[8 + row] * n[2]);
              const length = Math.hypot(...worldNormal);
              for (let axis = 0; axis < 3; axis++) {
                const actual = result[vertex * 16 + 4 + axis];
                if (!Number.isFinite(actual)) throw new Error('NONFINITE_NORMAL');
                maxNormalError = Math.max(maxNormalError, Math.abs(actual - worldNormal[axis] / length));
              }
              const actualTangent = Array.from(result.subarray(vertex * 16 + 12, vertex * 16 + 16));
              if (!actualTangent.every(Number.isFinite)) throw new Error('NONFINITE_TANGENT');
              let expectedTangent = [0, 0, 0, 0];
              if (reference.tangent) {
                const model = instance.subarray(0, 16), t = reference.tangent;
                const direction = [0, 1, 2].map(row => model[row] * t[0] + model[row + 4] * t[1] + model[row + 8] * t[2]);
                const nUnit = worldNormal.map(value => value / length);
                const projection = direction.reduce((sum, value, axis) => sum + value * nUnit[axis], 0);
                let projected = direction.map((value, axis) => value - nUnit[axis] * projection);
                if (Math.hypot(...projected) < 1e-12) {
                  const axis = nUnit.map(Math.abs).indexOf(Math.min(...nUnit.map(Math.abs)));
                  projected = nUnit.map((value, i) => Number(i === axis) - value * nUnit[axis]);
                }
                const magnitude = Math.hypot(...projected);
                const determinant = model[0] * (model[5] * model[10] - model[9] * model[6])
                  - model[4] * (model[1] * model[10] - model[9] * model[2]) + model[8] * (model[1] * model[6] - model[5] * model[2]);
                expectedTangent = [...projected.map(value => value / magnitude), t[3] * (determinant < 0 ? -1 : 1)];
              }
              for (let axis = 0; axis < 4; axis++) maxTangentError = Math.max(maxTangentError, Math.abs(actualTangent[axis] - expectedTangent[axis]));
            }
          }
          vertices++;
        }
        read.unmap();
      } finally { read.destroy(); }
    }
    return { vertices, maxError, maxNormalError, maxTangentError, boundsViolations, samples };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.reflections?.dispose();
    this.volumetricFog?.dispose();
    this.diffuseGi?.dispose();
    this.worldQueries?.dispose();
    this.geometryLod?.dispose();
    this.occlusion?.dispose();
    for (const batch of this.batches) if (batch.stream) void batch.stream.runtime.dispose().catch(error => this.errors.push(String(error)));
    this.geometryDecoder?.dispose();
    this.materialResources.dispose(); this.materialPipelines.clear();
    for (const buffer of this.buffers) buffer.destroy();
    this.shadow.destroy(); this.depth?.destroy(); this.velocity?.destroy(); this.hdr?.destroy(); this.querySet?.destroy();
    this.context.unconfigure(); this.device.destroy();
    this.stats.ownedGpuBytes = 0;
  }
}
