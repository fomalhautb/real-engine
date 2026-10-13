import deformationSource from './shaders/deform.wgsl?raw';
import surfaceSource from './shaders/surface.wgsl?raw';
import { identity, lookAt, multiply, perspective, type Mat4 } from './math';
import type { SceneBatch } from './contracts';

type Previous = { model: Mat4; bones: Mat4[]; morph: number; version: number };
type Batch = {
  scene: SceneBatch; rest: GPUBuffer; indices: GPUBuffer; params: GPUBuffer;
  instances: GPUBuffer; palettes: GPUBuffer; output: GPUBuffer;
  compute: GPUBindGroup; render: GPUBindGroup;
  instanceData: Float32Array<ArrayBuffer>; paletteData: Float32Array<ArrayBuffer>;
  history: Map<number, Previous>;
};
export interface FrameStats {
  frames: number; drawCalls: number; triangles: number; deformedVertices: number;
  ownedGpuBytes: number; width: number; height: number; cpuSubmitMs: number;
  gpuMs: { deformation: number; shadow: number; surface: number } | null;
}

/** Stage 1 scheduler: one deformation dispatch per mesh batch, shared by both views. */
export class Renderer {
  readonly errors: string[] = [];
  readonly stats: FrameStats = { frames: 0, drawCalls: 0, triangles: 0, deformedVertices: 0,
    ownedGpuBytes: 0, width: 0, height: 0, cpuSubmitMs: 0, gpuMs: null };
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
  private readonly lightMatrix: Mat4;
  private querySet?: GPUQuerySet;
  private queryResolve?: GPUBuffer;
  private queryRead?: GPUBuffer;
  private timingPending = false;
  private disposed = false;
  private computePipeline!: GPUComputePipeline;
  private surfacePipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;
  private computeLayout!: GPUBindGroupLayout;
  private geometryLayout!: GPUBindGroupLayout;

  static async create(canvas: HTMLCanvasElement, scene: SceneBatch[]): Promise<Renderer> {
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
      renderer = new Renderer(canvas, context, device, adapter, timing);
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
    private device: GPUDevice, adapter: GPUAdapter, timing: boolean,
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
    this.frameBuffer = this.buffer('frame uniforms', 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.shadow = device.createTexture({ label: 'sun shadow', size: [1024, 1024], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
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
    const ortho = identity();
    ortho[0] = 1 / 25; ortho[5] = 1 / 25; ortho[10] = -1 / 70; ortho[14] = -0.1 / 70;
    this.lightMatrix = multiply(ortho, lookAt([-18, 30, 16], [0, 0, -3], [0, 1, 0]));
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
    const [deform, surface] = await Promise.all([
      this.shader(deformationSource, 'shared-deformation'), this.shader(surfaceSource, 'surface'),
    ]);
    this.computeLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? 'uniform' : binding === 4 ? 'storage' : 'read-only-storage' },
    })) });
    this.geometryLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2].map(binding => ({
      binding, visibility: GPUShaderStage.VERTEX,
      buffer: { type: binding === 2 ? 'uniform' : 'read-only-storage' },
    })) });
    this.computePipeline = await this.device.createComputePipelineAsync({
      label: 'morph then shared skinning', layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }),
      compute: { module: deform, entryPoint: 'main' },
    });
    this.surfacePipeline = await this.device.createRenderPipelineAsync({
      label: 'opaque metal rough PBR and velocity',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.geometryLayout] }),
      vertex: { module: surface, entryPoint: 'vertexMain' },
      fragment: { module: surface, entryPoint: 'fragmentMain', targets: [{ format }, { format: 'rg16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.shadowPipeline = await this.device.createRenderPipelineAsync({
      label: 'shadow consumes shared geometry',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.shadowFrameLayout, this.geometryLayout] }),
      vertex: { module: surface, entryPoint: 'shadowMain' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
    });
    for (const entry of scene) this.batches.push(this.createBatch(entry));
  }

  private createBatch(scene: SceneBatch): Batch {
    const capacity = scene.capacity;
    const mesh = scene.mesh;
    const rest = this.buffer(scene.name + ' source', mesh.vertices.byteLength, GPUBufferUsage.STORAGE, mesh.vertices);
    const indices = this.buffer(scene.name + ' indices', mesh.indices.byteLength, GPUBufferUsage.INDEX, new Uint32Array(mesh.indices).buffer);
    const params = this.buffer(scene.name + ' params', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const instances = this.buffer(scene.name + ' instances', capacity * 224, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const palettes = this.buffer(scene.name + ' palettes', capacity * 512, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const output = this.buffer(scene.name + ' shared deformed', capacity * mesh.vertexCount * 48, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    return {
      scene, rest, indices, params, instances, palettes, output,
      compute: this.device.createBindGroup({ layout: this.computeLayout,
        entries: [params, rest, instances, palettes, output].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      render: this.device.createBindGroup({ layout: this.geometryLayout,
        entries: [output, instances, params].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
      instanceData: new Float32Array(capacity * 56), paletteData: new Float32Array(capacity * 128), history: new Map(),
    };
  }

  private upload(batch: Batch): void {
    const states = batch.scene.instances;
    if (states.length > batch.scene.capacity) throw new Error('INSTANCE_CAPACITY: ' + batch.scene.name);
    const alive = new Set<number>();
    for (let i = 0; i < states.length; i++) {
      const state = states[i];
      if (state.bones.length !== 4) throw new Error('DEFORMATION_CONTRACT: expected four joints');
      let previous = batch.history.get(state.entityId);
      if (!previous || previous.version !== state.historyVersion) {
        previous = { model: state.model.slice(), bones: state.bones.map(b => b.slice()), morph: state.morph, version: state.historyVersion };
        batch.history.set(state.entityId, previous);
      }
      const offset = i * 56;
      batch.instanceData.set(state.model, offset);
      batch.instanceData.set(previous.model, offset + 16);
      batch.instanceData.set(state.normalMatrix, offset + 32);
      batch.instanceData.set(state.color, offset + 48);
      batch.instanceData.set([state.metalness, state.roughness, state.morph, previous.morph], offset + 52);
      for (let j = 0; j < 4; j++) {
        batch.paletteData.set(state.bones[j], i * 128 + j * 16);
        batch.paletteData.set(previous.bones[j], i * 128 + 64 + j * 16);
        previous.bones[j].set(state.bones[j]);
      }
      previous.model.set(state.model); previous.morph = state.morph; previous.version = state.historyVersion;
      alive.add(state.entityId);
    }
    for (const id of batch.history.keys()) if (!alive.has(id)) batch.history.delete(id);
    this.device.queue.writeBuffer(batch.params, 0, new Uint32Array([batch.scene.mesh.vertexCount, states.length, 4, 0]));
    if (states.length) {
      this.device.queue.writeBuffer(batch.instances, 0, batch.instanceData, 0, states.length * 56);
      this.device.queue.writeBuffer(batch.palettes, 0, batch.paletteData, 0, states.length * 128);
    }
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
    this.depth = this.device.createTexture({ size: [width, height], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
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
    this.resize();
    for (const batch of this.batches) this.upload(batch);
    const vp = multiply(perspective(Math.PI / 3.2, this.stats.width / this.stats.height, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
    this.frameData.set(vp); this.frameData.set(this.lastViewProjection ?? vp, 16); this.frameData.set(this.lightMatrix, 32);
    this.frameData.set([...eye, 1], 48); this.frameData.set([-18, 30, 19, 0], 52); this.frameData.set([3.5, 3.2, 2.8, 1], 56);
    this.device.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    this.lastViewProjection = vp;
    const encoder = this.device.createCommandEncoder({ label: 'deform → shadow → surface' });
    const timestamp = (index: number): GPUComputePassTimestampWrites | undefined => this.querySet
      ? { querySet: this.querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } : undefined;
    const compute = encoder.beginComputePass({ label: 'shared deformation', timestampWrites: timestamp(0) });
    compute.setPipeline(this.computePipeline);
    for (const batch of this.batches) {
      if (!batch.scene.instances.length) continue;
      compute.setBindGroup(0, batch.compute);
      compute.dispatchWorkgroups(Math.ceil(batch.scene.mesh.vertexCount * batch.scene.instances.length / 64));
    }
    compute.end();
    const shadow = encoder.beginRenderPass({ label: 'sun shadow', colorAttachments: [], timestampWrites: timestamp(2),
      depthStencilAttachment: { view: this.shadow.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    shadow.setPipeline(this.shadowPipeline); shadow.setBindGroup(0, this.shadowFrameBindGroup);
    this.draw(shadow); shadow.end();
    const surface = encoder.beginRenderPass({ label: 'PBR and velocity', timestampWrites: timestamp(4), colorAttachments: [
      { view: this.context.getCurrentTexture().createView(), clearValue: [0.035, 0.055, 0.065, 1], loadOp: 'clear', storeOp: 'store' },
      { view: this.velocity!.createView(), clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' },
    ], depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' } });
    surface.setPipeline(this.surfacePipeline); surface.setBindGroup(0, this.frameBindGroup);
    this.draw(surface); surface.end();
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
    this.stats.deformedVertices = this.batches.reduce((sum, batch) => sum + batch.scene.mesh.vertexCount * batch.scene.instances.length, 0);
    this.stats.cpuSubmitMs = performance.now() - started;
  }
  private draw(pass: GPURenderPassEncoder): void {
    for (const batch of this.batches) {
      if (!batch.scene.instances.length) continue;
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

  /** Explicit diagnostic readback only; never used by the production frame loop. */
  async verifyDeformation(): Promise<{ vertices: number; maxError: number }> {
    let maxError = 0, vertices = 0;
    for (const batch of this.batches.filter(b => b.scene.name === 'character' || b.scene.name === 'trees')) {
      if (!batch.scene.instances.length) continue;
      const count = batch.scene.mesh.vertexCount;
      const read = this.device.createBuffer({ size: count * 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const instance = batch.instanceData.slice(0, 56), palette = batch.paletteData.slice(0, 128);
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(batch.output, 0, read, 0, count * 48);
      this.device.queue.submit([encoder.finish()]);
      try {
        await read.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(read.getMappedRange()), source = new DataView(batch.scene.mesh.vertices);
        const transform = (matrix: ArrayLike<number>, p: number[]) => [0, 1, 2, 3].map(row =>
          matrix[row] * p[0] + matrix[4 + row] * p[1] + matrix[8 + row] * p[2] + matrix[12 + row] * p[3]);
        for (let vertex = 0; vertex < count; vertex++) {
          for (let history = 0; history < 2; history++) {
            const p = [0, 1, 2, 3].map(axis => source.getFloat32(vertex * 80 + axis * 4, true)
              + source.getFloat32(vertex * 80 + 64 + axis * 4, true) * instance[54 + history]);
            const skinned = [0, 0, 0, 0];
            for (let j = 0; j < 4; j++) {
              const joint = source.getUint32(vertex * 80 + 32 + j * 4, true);
              const weight = source.getFloat32(vertex * 80 + 48 + j * 4, true);
              const transformed = transform(palette.subarray(history * 64 + joint * 16, history * 64 + joint * 16 + 16), p);
              for (let axis = 0; axis < 4; axis++) skinned[axis] += transformed[axis] * weight;
            }
            const expected = transform(instance.subarray(history * 16, history * 16 + 16), skinned);
            for (let axis = 0; axis < 4; axis++) {
              const actual = result[vertex * 12 + history * 8 + axis];
              if (!Number.isFinite(actual)) throw new Error('NONFINITE_DEFORMATION');
              maxError = Math.max(maxError, Math.abs(actual - expected[axis]));
            }
          }
          vertices++;
        }
        read.unmap();
      } finally { read.destroy(); }
    }
    return { vertices, maxError };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy();
    this.shadow.destroy(); this.depth?.destroy(); this.velocity?.destroy(); this.querySet?.destroy();
    this.context.unconfigure(); this.device.destroy();
    this.stats.ownedGpuBytes = 0;
  }
}
