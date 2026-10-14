import deformationSource from './shaders/deform.wgsl?raw';
import surfaceSource from './shaders/surface.wgsl?raw';
import { identity, lookAt, multiply, perspective, transformPoint, type Mat4 } from './math';
import type { InstanceState, SceneBatch, MaterialBinding } from './contracts';
import { MaterialResources, type GpuMaterial } from './gpu-material';
import { compileDeformation, conservativeDeformationBounds, evaluateDeformation,
  validateDeformationVertex, validateMorphTargets, type Bounds3 } from './deformation';
import { compileRenderGraph } from './render-graph';

type Previous = { model: Mat4; bones: Mat4[]; scalars: number[]; version: number; source?: InstanceState; settled: boolean };
type MaterialPipelines = { surface: GPURenderPipeline; shadow: GPURenderPipeline };
type Batch = {
  scene: SceneBatch; rest: GPUBuffer; indices: GPUBuffer; params: GPUBuffer;
  instances: GPUBuffer; palettes: GPUBuffer; output: GPUBuffer; scalars: GPUBuffer;
  compute: GPUBindGroup; render: GPUBindGroup;
  program: ReturnType<typeof compileDeformation>; pipeline: GPUComputePipeline;
  instanceData: Float32Array<ArrayBuffer>; paletteData: Float32Array<ArrayBuffer>; scalarData: Float32Array<ArrayBuffer>;
  morphBounds: Bounds3[]; bounds: Map<number, Bounds3>;
  history: Map<number, Previous>;
  dirty: boolean; lastInstances: InstanceState[];
  material: GpuMaterial; materialPipelines: MaterialPipelines; materialRequest: number;
};
export interface FrameStats {
  frames: number; drawCalls: number; triangles: number; deformedVertices: number;
  ownedGpuBytes: number; width: number; height: number; cpuSubmitMs: number;
  gpuMs: { deformation: number; shadow: number; surface: number } | null;
  deformationDispatches: number; uploadedBytes: number;
}
export interface DeformationVerification {
  vertices: number; maxError: number; maxNormalError: number; boundsViolations: number;
  samples: { batch: string; entityId: number; vertexIndex: number; current: number[]; previous: number[] }[];
}
export interface RendererOptions {
  /** Includes pending compilation and resident cached definitions. Two render pipelines per definition. */
  materialVariantLimit?: number;
  /** Shared material images, all mips. Atomic replacements require overlap with their old images. */
  textureBudgetBytes?: number;
}

/** Stage 1 scheduler: one deformation dispatch per mesh batch, shared by both views. */
export class Renderer {
  readonly errors: string[] = [];
  readonly stats: FrameStats = { frames: 0, drawCalls: 0, triangles: 0, deformedVertices: 0,
    ownedGpuBytes: 0, width: 0, height: 0, cpuSubmitMs: 0, gpuMs: null, deformationDispatches: 0, uploadedBytes: 0 };
  readonly info: { adapter: string; features: string[]; userAgent: string; timing: string };
  private readonly batches: Batch[] = [];
  private readonly buffers: GPUBuffer[] = [];
  private readonly frameData = new Float32Array(64);
  private readonly frameBuffer: GPUBuffer;
  private readonly shadow: GPUTexture;
  private readonly frameBindGroup: GPUBindGroup;
  private readonly shadowFrameBindGroup: GPUBindGroup;
  private depth?: GPUTexture;
  private velocity?: GPUTexture;
  private lastViewProjection?: Mat4;
  private lightMatrix: Mat4 = identity();
  private sunDirection: [number, number, number] = [-18, 30, 19];
  private sunColor: [number, number, number] = [3.5, 3.2, 2.8];
  private querySet?: GPUQuerySet;
  private queryResolve?: GPUBuffer;
  private queryRead?: GPUBuffer;
  private timingPending = false;
  private disposed = false;
  private readonly deformationPipelines = new Map<string, GPUComputePipeline>();
  private graph!: ReturnType<typeof compileRenderGraph<GPUCommandEncoder>>;
  private readonly materialResources: MaterialResources;
  private readonly materialPipelines = new Map<string, Promise<MaterialPipelines>>();
  private readonly compilingMaterials = new Set<string>();
  private readonly materialVariantLimit: number;
  private materialRevision = 0;
  private format!: GPUTextureFormat;
  private computeLayout!: GPUBindGroupLayout;
  private geometryLayout!: GPUBindGroupLayout;

  static async create(canvas: HTMLCanvasElement, scene: SceneBatch[], options: RendererOptions = {}): Promise<Renderer> {
    const limit = options.materialVariantLimit ?? 128;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 4096) throw new Error('INVALID_MATERIAL_VARIANT_LIMIT');
    if (options.textureBudgetBytes !== undefined && (!Number.isSafeInteger(options.textureBudgetBytes) || options.textureBudgetBytes < 0)) throw new Error('INVALID_TEXTURE_BUDGET');
    if (!navigator.gpu) throw new Error('WEBGPU_UNAVAILABLE: use a WebGPU browser on HTTPS or localhost.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('ADAPTER_UNAVAILABLE: no compatible WebGPU adapter.');
    const timing = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({ requiredFeatures: timing ? ['timestamp-query'] : [] });
    let context: GPUCanvasContext | null = null;
    let renderer: Renderer | undefined;
    try {
      context = canvas.getContext('webgpu');
      if (!context) throw new Error('CANVAS_UNAVAILABLE: cannot create WebGPU context.');
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });
      renderer = new Renderer(canvas, context, device, adapter, timing, options);
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
    private device: GPUDevice, adapter: GPUAdapter, timing: boolean, options: RendererOptions,
  ) {
    this.info = {
      adapter: [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(' / ') || 'Unreported adapter',
      features: Array.from(device.features), userAgent: navigator.userAgent,
      timing: timing ? 'GPU timestamp-query (adapter-dependent precision)' : 'GPU timestamps unavailable',
    };
    device.addEventListener('uncapturederror', event => this.errors.push(event.error.message));
    void device.lost.then(loss => {
      if (!this.disposed) this.errors.push('DEVICE_LOST: ' + loss.reason + ' ' + loss.message);
    });
    this.materialVariantLimit = options.materialVariantLimit ?? 128;
    this.materialResources = new MaterialResources(device, bytes => { this.stats.ownedGpuBytes += bytes; }, options.textureBudgetBytes);
    this.frameBuffer = this.buffer('frame uniforms', 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.shadow = device.createTexture({ label: 'sun shadow', size: [1024, 1024], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.stats.ownedGpuBytes += 1024 * 1024 * 4;
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
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 6 });
      this.queryResolve = this.buffer('timestamp resolve', 48, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      this.queryRead = this.buffer('timestamp readback', 48, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    }
  }
  private frameLayout: GPUBindGroupLayout;
  private shadowFrameLayout: GPUBindGroupLayout;

  private buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer {
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

  private async initialize(format: GPUTextureFormat, scene: SceneBatch[]): Promise<void> {
    this.format = format;
    this.computeLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 4 ? 'storage' : 'read-only-storage' },
    })) });
    this.geometryLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({
      binding, visibility: GPUShaderStage.VERTEX,
      buffer: { type: binding === 2 ? 'uniform' : 'read-only-storage' },
    })) });
    if (new Set(scene.map(batch => batch.name)).size !== scene.length) throw new Error('DUPLICATE_BATCH_NAME');
    for (const entry of scene) this.batches.push(await this.createBatch(entry));
    this.graph = compileRenderGraph<GPUCommandEncoder>([
      { name: 'scene-input', kind: 'buffer', source: 'imported' },
      { name: 'shared-geometry', kind: 'buffer', source: 'imported' },
      { name: 'sun-shadow', kind: 'texture', source: 'transient' },
      { name: 'canvas', kind: 'texture', source: 'imported' },
      { name: 'depth', kind: 'texture', source: 'transient' },
      { name: 'velocity', kind: 'texture', source: 'transient' },
    ], [
      { name: 'deformation', uses: [{ resource: 'scene-input', access: 'read' }, { resource: 'shared-geometry', access: 'write' }], execute: encoder => this.encodeDeformation(encoder) },
      { name: 'shadow', uses: [{ resource: 'shared-geometry', access: 'read' }, { resource: 'sun-shadow', access: 'write' }], execute: encoder => this.encodeShadow(encoder) },
      { name: 'surface', uses: [{ resource: 'shared-geometry', access: 'read' }, { resource: 'sun-shadow', access: 'read' },
        { resource: 'canvas', access: 'write' }, { resource: 'depth', access: 'write' }, { resource: 'velocity', access: 'write' }], execute: encoder => this.encodeSurface(encoder) },
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
          const module = await this.shader(surfaceSource.replace('// MATERIAL_PROGRAM',
            `const MATERIAL_UNLIT = ${program.shading === 'unlit'};\n` + program.wgsl), 'material');
          const primitive: GPUPrimitiveState = { topology: 'triangle-list', cullMode: program.doubleSided ? 'none' : 'back' };
          const surface = await this.device.createRenderPipelineAsync({ label: 'material surface and velocity',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.geometryLayout, this.materialResources.layout] }),
            vertex: { module, entryPoint: 'vertexMain' },
            fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: this.format }, { format: 'rg16float' }] },
            primitive, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
          });
          const shadow = await this.device.createRenderPipelineAsync({ label: 'shared material shadow coverage',
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.shadowFrameLayout, this.geometryLayout, this.materialResources.layout] }),
            vertex: { module, entryPoint: 'shadowMain' }, fragment: { module, entryPoint: 'shadowFragment', targets: [] },
            primitive, depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
          });
          return { surface, shadow };
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

  /** A directional light points from the surface towards the light. Linear RGB radiance. */
  setSun(sun: { direction: [number, number, number]; color: [number, number, number] }): void {
    if (this.disposed) throw new Error('RENDERER_DISPOSED');
    if (sun.direction.length !== 3 || sun.color.length !== 3 || ![...sun.direction, ...sun.color].every(value => Number.isFinite(value) && Number.isFinite(Math.fround(value))) ||
        Math.hypot(...sun.direction) < 1e-6 || sun.color.some(value => value < 0 || !Number.isFinite(Math.fround(value)))) throw new Error('INVALID_SUN');
    const length = Math.hypot(...sun.direction);
    const d = sun.direction.map(value => value / length) as [number, number, number];
    this.sunDirection = d; this.sunColor = [...sun.color];
    const target: [number, number, number] = [0, 0, -3];
    const eye = target.map((value, axis) => value + d[axis] * 40) as [number, number, number];
    const ortho = identity();
    ortho[0] = 1 / 25; ortho[5] = 1 / 25; ortho[10] = -1 / 70; ortho[14] = -0.1 / 70;
    this.lightMatrix = multiply(ortho, lookAt(eye, target, Math.abs(d[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0]));
  }

  private async createBatch(scene: SceneBatch): Promise<Batch> {
    const capacity = scene.capacity;
    const mesh = scene.mesh;
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
    const uvData = mesh.uvs ? new Float32Array(mesh.uvs) : new Float32Array(mesh.vertexCount * 2);
    const uvs = this.buffer(scene.name + ' UVs', uvData.byteLength, GPUBufferUsage.STORAGE, uvData.buffer);
    const params = this.buffer(scene.name + ' params', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const instances = this.buffer(scene.name + ' instances', capacity * 224, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const palettes = this.buffer(scene.name + ' palettes', Math.max(1, capacity * program.jointCount * 2) * 64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const scalarStride = Math.max(1, program.scalarCount);
    const scalars = this.buffer(scene.name + ' scalars', capacity * scalarStride * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const morphData = new Float32Array(Math.max(1, program.morphTargetCount * mesh.vertexCount) * 8);
    for (let t = 0; t < program.morphTargetCount; t++) for (let i = 0; i < mesh.vertexCount; i++) {
      morphData.set(scene.morphTargets[t].positions.subarray(i * 3, i * 3 + 3), (t * mesh.vertexCount + i) * 8);
      morphData.set(scene.morphTargets[t].normals.subarray(i * 3, i * 3 + 3), (t * mesh.vertexCount + i) * 8 + 4);
    }
    const morphs = this.buffer(scene.name + ' morph targets', morphData.byteLength, GPUBufferUsage.STORAGE, morphData.buffer);
    const output = this.buffer(scene.name + ' shared deformed', capacity * mesh.vertexCount * 48, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const material = await this.prepareMaterial(scene.material);
    return {
      ...material, materialRequest: 0,
      scene, rest, indices, params, instances, palettes, output, scalars, program, pipeline, morphBounds, bounds: new Map(),
      compute: this.device.createBindGroup({ layout: this.computeLayout,
        entries: [params, rest, instances, palettes, output, morphs, scalars].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      render: this.device.createBindGroup({ layout: this.geometryLayout,
        entries: [output, instances, params, uvs].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      instanceData: new Float32Array(capacity * 56), paletteData: new Float32Array(capacity * program.jointCount * 32),
      scalarData: new Float32Array(capacity * scalarStride * 2), history: new Map(), dirty: true, lastInstances: [],
    };
  }

  private upload(batch: Batch): void {
    const states = batch.scene.instances;
    if (states.length > batch.scene.capacity) throw new Error('INSTANCE_CAPACITY: ' + batch.scene.name);
    // Inputs are committed frame objects. After one stationary frame has settled
    // previous=current, keep the shared output until this batch changes again.
    if (batch.scene.inputMode === 'committed' && states.length === batch.lastInstances.length && states.every((state, i) =>
      state === batch.lastInstances[i] && batch.history.get(state.entityId)?.settled)) {
      batch.dirty = false;
      return;
    }
    batch.dirty = true;
    const alive = new Set<number>();
    const joints = batch.program.jointCount, scalarCount = Math.max(1, batch.program.scalarCount);
    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      const bounds = conservativeDeformationBounds(batch.program, batch.scene.mesh.bounds, batch.morphBounds, state.bones, state.deformScalars, state.model);
      let previous = batch.history.get(state.entityId);
      if (!previous || previous.version !== state.historyVersion) {
        previous = { model: state.model.slice(), bones: state.bones.map(b => b.slice()), scalars: [...state.deformScalars], version: state.historyVersion, settled: true };
        batch.history.set(state.entityId, previous);
      }
      const offset = i * 56;
      batch.instanceData.set(state.model, offset);
      batch.instanceData.set(previous.model, offset + 16);
      batch.instanceData.set(state.normalMatrix, offset + 32);
      batch.instanceData.set(state.color, offset + 48);
      batch.instanceData.set([state.metalness, state.roughness, 0, 0], offset + 52);
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
      previous.model.set(state.model); previous.scalars = [...state.deformScalars]; previous.version = state.historyVersion;
      alive.add(state.entityId);
    }
    for (const id of batch.history.keys()) if (!alive.has(id)) { batch.history.delete(id); batch.bounds.delete(id); }
    this.device.queue.writeBuffer(batch.params, 0, new Uint32Array([batch.scene.mesh.vertexCount, states.length, joints, scalarCount]));
    this.stats.uploadedBytes += 16;
    if (states.length) {
      this.device.queue.writeBuffer(batch.instances, 0, batch.instanceData, 0, states.length * 56);
      if (joints) this.device.queue.writeBuffer(batch.palettes, 0, batch.paletteData, 0, states.length * joints * 32);
      this.device.queue.writeBuffer(batch.scalars, 0, batch.scalarData, 0, states.length * scalarCount * 2);
      this.stats.uploadedBytes += states.length * (224 + joints * 128 + scalarCount * 8);
    }
    batch.lastInstances = [...states];
  }

  private resize(): void {
    const ratio = Math.min(devicePixelRatio, 1.5);
    const width = Math.max(1, Math.min(this.device.limits.maxTextureDimension2D, Math.round(this.canvas.clientWidth * ratio)));
    const height = Math.max(1, Math.min(this.device.limits.maxTextureDimension2D, Math.round(this.canvas.clientHeight * ratio)));
    if (width === this.stats.width && height === this.stats.height) return;
    if (this.depth) {
      this.stats.ownedGpuBytes -= this.stats.width * this.stats.height * 8;
      this.depth.destroy(); this.velocity!.destroy();
    }
    this.canvas.width = width; this.canvas.height = height;
    this.depth = this.device.createTexture({ size: [width, height], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.velocity = this.device.createTexture({ size: [width, height], format: 'rg16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.stats.width = width; this.stats.height = height;
    this.stats.ownedGpuBytes += width * height * 8;
    this.lastViewProjection = undefined;
    for (const batch of this.batches) batch.history.clear();
  }

  render(eye: [number, number, number], target: [number, number, number]): void {
    if (this.disposed || this.errors.length) return;
    const started = performance.now();
    this.stats.uploadedBytes = 256; this.stats.deformationDispatches = 0;
    this.resize();
    for (const batch of this.batches) this.upload(batch);
    const vp = multiply(perspective(Math.PI / 3.2, this.stats.width / this.stats.height, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
    this.frameData.set(vp); this.frameData.set(this.lastViewProjection ?? vp, 16); this.frameData.set(this.lightMatrix, 32);
    this.frameData.set([...eye, 1], 48); this.frameData.set([...this.sunDirection, 0], 52); this.frameData.set([...this.sunColor, 1], 56);
    this.device.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    this.lastViewProjection = vp;
    const encoder = this.device.createCommandEncoder({ label: 'deform → shadow → surface' });
    this.graph.execute(encoder);
    const readTiming = !!this.querySet && !this.timingPending && this.stats.frames % 30 === 0;
    if (readTiming) {
      encoder.resolveQuerySet(this.querySet!, 0, 6, this.queryResolve!, 0);
      encoder.copyBufferToBuffer(this.queryResolve!, 0, this.queryRead!, 0, 48);
      this.timingPending = true;
    }
    this.device.queue.submit([encoder.finish()]);
    if (readTiming) void this.readTiming();
    this.stats.frames++;
    this.stats.drawCalls = this.batches.filter(batch => batch.scene.instances.length).length * 2;
    this.stats.triangles = this.batches.reduce((sum, batch) => sum + batch.scene.mesh.indices.length / 3 * batch.scene.instances.length, 0);
    this.stats.deformedVertices = this.batches.reduce((sum, batch) => sum + (batch.dirty ? batch.scene.mesh.vertexCount * batch.scene.instances.length : 0), 0);
    this.stats.cpuSubmitMs = performance.now() - started;
  }
  private timestamp(index: number): GPUComputePassTimestampWrites | undefined {
    return this.querySet ? { querySet: this.querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } : undefined;
  }
  private encodeDeformation(encoder: GPUCommandEncoder): void {
    const compute = encoder.beginComputePass({ label: 'shared deformation', timestampWrites: this.timestamp(0) });
    for (const batch of this.batches) {
      if (!batch.scene.instances.length || !batch.dirty) continue;
      compute.setPipeline(batch.pipeline); compute.setBindGroup(0, batch.compute);
      compute.dispatchWorkgroups(Math.ceil(batch.scene.mesh.vertexCount * batch.scene.instances.length / 64));
      this.stats.deformationDispatches++;
    }
    compute.end();
  }
  private encodeShadow(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginRenderPass({ label: 'sun shadow', colorAttachments: [], timestampWrites: this.timestamp(2),
      depthStencilAttachment: { view: this.shadow.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    pass.setBindGroup(0, this.shadowFrameBindGroup);
    this.draw(pass, 'shadow'); pass.end();
  }
  private encodeSurface(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginRenderPass({ label: 'PBR and velocity', timestampWrites: this.timestamp(4), colorAttachments: [
      { view: this.context.getCurrentTexture().createView(), clearValue: [0.035, 0.055, 0.065, 1], loadOp: 'clear', storeOp: 'store' },
      { view: this.velocity!.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' },
    ], depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    pass.setBindGroup(0, this.frameBindGroup);
    this.draw(pass, 'surface'); pass.end();
  }
  diagnostics() {
    return { deformationVariants: this.deformationPipelines.size, materialVariants: this.materialPipelines.size,
      materials: this.materialResources.diagnostics(), materialRevision: this.materialRevision, passes: this.graph.order,
      lifetimes: Object.fromEntries(this.graph.lifetimes),
      bounds: this.batches.flatMap(batch => Array.from(batch.bounds, ([entityId, bounds]) => ({ entityId, ...bounds }))) };
  }
  private readVertex(source: DataView, vertex: number) {
    const floats = (offset: number, count: number) => Array.from({ length: count }, (_, axis) => source.getFloat32(vertex * 80 + offset + axis * 4, true));
    return { position: floats(0, 3) as [number, number, number], normal: floats(16, 3) as [number, number, number],
      joints: Array.from({ length: 4 }, (_, axis) => source.getUint32(vertex * 80 + 32 + axis * 4, true)) as [number, number, number, number],
      weights: floats(48, 4) as [number, number, number, number] };
  }
  private draw(pass: GPURenderPassEncoder, kind: keyof MaterialPipelines): void {
    for (const batch of this.batches) {
      if (!batch.scene.instances.length) continue;
      pass.setPipeline(batch.materialPipelines[kind]); pass.setBindGroup(2, batch.material.bindGroup);
      pass.setBindGroup(1, batch.render); pass.setIndexBuffer(batch.indices, 'uint32');
      pass.drawIndexed(batch.scene.mesh.indices.length, batch.scene.instances.length);
    }
  }
  private async readTiming(): Promise<void> {
    try {
      await this.queryRead!.mapAsync(GPUMapMode.READ);
      const values = new BigUint64Array(this.queryRead!.getMappedRange());
      this.stats.gpuMs = { deformation: Number(values[1] - values[0]) / 1e6,
        shadow: Number(values[3] - values[2]) / 1e6, surface: Number(values[5] - values[4]) / 1e6 };
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
      { texture: this.shadow, width: 1024, height: 1024, matrix: this.frameData.slice(32, 48) },
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

  /** Explicit diagnostic readback only; never used by the production frame loop. */
  async verifyDeformation(options: { allInstances?: boolean; sampleVertices?: readonly number[] } = {}): Promise<DeformationVerification> {
    if (options.sampleVertices && (options.sampleVertices.length > 1024 || options.sampleVertices.some(index => !Number.isSafeInteger(index) || index < 0))) throw new Error('INVALID_SAMPLE_VERTICES');
    const sampleVertices = new Set([0, ...options.sampleVertices ?? []]);
    let maxError = 0, vertices = 0, maxNormalError = 0, boundsViolations = 0;
    const samples: DeformationVerification['samples'] = [];
    for (const batch of this.batches) {
      if (!batch.scene.instances.length) continue;
      const instanceCount = options.allInstances ? batch.scene.instances.length : 1;
      const count = batch.scene.mesh.vertexCount * instanceCount;
      const read = this.device.createBuffer({ size: count * 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const joints = batch.program.jointCount, scalarCount = Math.max(1, batch.program.scalarCount);
      const instances = batch.instanceData.slice(0, instanceCount * 56), palette = batch.paletteData.slice(0, instanceCount * joints * 32), scalars = batch.scalarData.slice(0, instanceCount * scalarCount * 2);
      const entityIds = batch.scene.instances.slice(0, instanceCount).map(state => state.entityId);
      const instanceBounds = entityIds.map(id => structuredClone(batch.bounds.get(id)!));
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(batch.output, 0, read, 0, count * 48);
      this.device.queue.submit([encoder.finish()]);
      try {
        await read.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(read.getMappedRange()), source = new DataView(batch.scene.mesh.vertices);
        for (let vertex = 0; vertex < count; vertex++) {
          const instanceIndex = Math.floor(vertex / batch.scene.mesh.vertexCount), vertexIndex = vertex % batch.scene.mesh.vertexCount;
          const instance = instances.subarray(instanceIndex * 56, instanceIndex * 56 + 56), bounds = instanceBounds[instanceIndex];
          if (sampleVertices.has(vertexIndex)) samples.push({ batch: batch.scene.name, entityId: entityIds[instanceIndex], vertexIndex,
            current: Array.from(result.subarray(vertex * 12, vertex * 12 + 3)), previous: Array.from(result.subarray(vertex * 12 + 8, vertex * 12 + 11)) });
          for (let history = 0; history < 2; history++) {
            const paletteOffset = instanceIndex * joints * 32 + history * joints * 16;
            const bones = Array.from({ length: joints }, (_, j) => palette.subarray(paletteOffset + j * 16, paletteOffset + j * 16 + 16));
            const reference = evaluateDeformation(batch.program, this.readVertex(source, vertexIndex), vertexIndex, batch.scene.morphTargets, bones, scalars.subarray(instanceIndex * scalarCount * 2 + history * scalarCount));
            const expected = transformPoint(instance.subarray(history * 16, history * 16 + 16), reference.position);
            for (let axis = 0; axis < 3; axis++) {
              const actual = result[vertex * 12 + history * 8 + axis];
              if (!Number.isFinite(actual)) throw new Error('NONFINITE_DEFORMATION');
              maxError = Math.max(maxError, Math.abs(actual - expected[axis]));
              if (history === 0 && (actual < bounds.min[axis] - 0.0002 || actual > bounds.max[axis] + 0.0002)) boundsViolations++;
            }
            if (history === 0) {
              const m = instance.subarray(32, 48), n = reference.normal;
              const worldNormal = [0, 1, 2].map(row => m[row] * n[0] + m[4 + row] * n[1] + m[8 + row] * n[2]);
              const length = Math.hypot(...worldNormal);
              for (let axis = 0; axis < 3; axis++) {
                const actual = result[vertex * 12 + 4 + axis];
                if (!Number.isFinite(actual)) throw new Error('NONFINITE_NORMAL');
                maxNormalError = Math.max(maxNormalError, Math.abs(actual - worldNormal[axis] / length));
              }
            }
          }
          vertices++;
        }
        read.unmap();
      } finally { read.destroy(); }
    }
    return { vertices, maxError, maxNormalError, boundsViolations, samples };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.materialResources.dispose(); this.materialPipelines.clear();
    for (const buffer of this.buffers) buffer.destroy();
    this.shadow.destroy(); this.depth?.destroy(); this.velocity?.destroy(); this.querySet?.destroy();
    this.context.unconfigure(); this.device.destroy();
    this.stats.ownedGpuBytes = 0;
  }
}
