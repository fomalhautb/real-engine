import coreSource from './shaders/local-lights.wgsl?raw';
import directSource from './shaders/local-lights-direct.wgsl?raw';
import fogSource from './shaders/local-lights-fog.wgsl?raw';
import fogCullSource from './shaders/local-lights-fog-cull.wgsl?raw';
import type { Bounds3 } from './deformation';
import {
  LOCAL_SHADOW_VIEW_MATRIX_OFFSET, LOCAL_SHADOW_VIEW_UNIFORM_STRIDE, LocalLightSet, localLightChangeRegions, resolveLocalLightOptions,
  type LocalLightOptions, type ResolvedLocalLight, type ResolvedLocalLightOptions,
} from './local-lights';
import { timestampSpanEnd, timestampSpanStart } from './gpu-timestamp-span';

export interface LocalLightShaderOptions {
  /** Bind group index and the three consecutive bindings: uniform, atlas, comparison sampler. */
  readonly group: number;
  readonly firstBinding?: number;
  /** Include PBR direct lighting (requires surface-brdf.wgsl in the same module). */
  readonly direct?: boolean;
  /** Include media scattering; 'froxel-block' adds the workgroup light list for fog injection. */
  readonly fog?: 'froxel-block' | 'all-lights';
}

/** WGSL for one consumer module. Every path uses the same functions and data. */
export function localLightsShaderSource(options: ResolvedLocalLightOptions, shader: LocalLightShaderOptions): string {
  const first = shader.firstBinding ?? 0, group = shader.group;
  const words = Math.ceil(options.maxLights / 32);
  const bindings = `@group(${group}) @binding(${first}) var<uniform> localLights: LocalLightBlock;\n`
    + `@group(${group}) @binding(${first + 1}) var localShadowAtlas: texture_depth_2d_array;\n`
    + `@group(${group}) @binding(${first + 2}) var localShadowSampler: sampler_comparison;`;
  let source = `const LOCAL_LIGHT_MASK_WORDS: u32 = ${words}u;\n` + coreSource
    .replace('LOCAL_LIGHT_CAPACITY', String(options.maxLights))
    .replace('LOCAL_SHADOW_VIEW_CAPACITY', String(options.maxShadowViews))
    .replace('// LOCAL_LIGHT_BINDINGS', bindings);
  if (shader.direct) source += '\n' + directSource;
  if (shader.fog) {
    source += '\n' + fogSource + '\n' + (shader.fog === 'froxel-block' ? fogCullSource
      : 'fn fogLocalCandidateWord(word: u32) -> u32 {\n'
        + '    let first = word * 32u;\n    let count = localLights.header.x;\n'
        + '    if (count <= first) { return 0u; }\n    let remaining = count - first;\n'
        + '    return select((1u << (remaining & 31u)) - 1u, 0xffffffffu, remaining >= 32u);\n}');
  }
  return source;
}

/** Media hook for volumetric-fog-world.wgsl, replacing `// FOG_LOCAL_LIGHTS`. */
export const FOG_LOCAL_LIGHTS_HOOK = 'source = vec4f(source.xyz + fogLocalLightsScattering(world, direction), source.w);';

export interface LocalLightShadowEncoding {
  /** Depth attachment of one atlas layer. */
  readonly view: GPUTextureView;
  /** Shadow frame contract group whose lightViewProjection is this view's matrix. */
  readonly frame: GPUBindGroup;
  readonly layer: number;
}

/** GPU resources for bounded local lights: one uniform block shared by all
 * consumers, a depth32float 2D-array shadow atlas and per-view uniforms that
 * feed the shared material shadow pipeline. Owns and releases all of them. */
export class GpuLocalLights {
  readonly set: LocalLightSet;
  readonly options: ResolvedLocalLightOptions;
  readonly uniform: GPUBuffer;
  readonly atlas: GPUTexture;
  readonly atlasView: GPUTextureView;
  readonly sampler: GPUSampler;
  readonly computeLayout: GPUBindGroupLayout;
  private readonly viewUniform: GPUBuffer;
  private readonly layerViews: GPUTextureView[];
  private readonly viewGroups: GPUBindGroup[];
  private readonly packed: Float32Array<ArrayBuffer>;
  private readonly viewData: Float32Array<ArrayBuffer>;
  private uploadedRevision = -1;
  private renderedLayers: number[] = [];
  private uploadedLights: ResolvedLocalLight[] = [];
  private changed: Bounds3[] = [];
  private bytes = 0;
  private disposed = false;
  private uploads = 0;
  private shadowViewsRendered = 0;

  static create(device: GPUDevice, options: LocalLightOptions, shadowFrameLayout: GPUBindGroupLayout,
    account: (bytes: number) => void): GpuLocalLights {
    const resolved = resolveLocalLightOptions(options, {
      maxTextureDimension2D: device.limits.maxTextureDimension2D,
      maxTextureArrayLayers: device.limits.maxTextureArrayLayers,
      maxUniformBufferBindingSize: device.limits.maxUniformBufferBindingSize,
    });
    if (resolved.viewUniformBytes > device.limits.maxBufferSize) throw new Error('LOCAL_LIGHT_BUDGET: shadow view uniforms');
    return new GpuLocalLights(device, resolved, shadowFrameLayout, account);
  }

  private constructor(private readonly device: GPUDevice, options: ResolvedLocalLightOptions,
    shadowFrameLayout: GPUBindGroupLayout, private readonly account: (bytes: number) => void) {
    this.options = options;
    this.set = new LocalLightSet(options);
    this.packed = new Float32Array(options.uniformBytes / 4);
    this.viewData = new Float32Array(options.viewUniformBytes / 4);
    const created: { destroy(): void }[] = [];
    try {
      this.uniform = device.createBuffer({ label: 'local light block', size: options.uniformBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      created.push(this.uniform);
      this.viewUniform = device.createBuffer({ label: 'local light shadow view frames', size: options.viewUniformBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      created.push(this.viewUniform);
      const edge = options.maxShadowedLights ? options.shadowResolution : 1;
      const layers = options.maxShadowedLights ? options.maxShadowViews : 1;
      this.atlas = device.createTexture({ label: 'local light shadow atlas', size: [edge, edge, layers],
        format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      created.push(this.atlas);
    } catch (error) { for (const resource of created) resource.destroy(); throw error; }
    this.bytes = options.uniformBytes + options.viewUniformBytes + options.atlasBytes;
    account(this.bytes);
    this.atlasView = this.atlas.createView({ dimension: '2d-array' });
    const layers = this.atlas.depthOrArrayLayers;
    this.layerViews = Array.from({ length: layers }, (_, layer) => this.atlas.createView({
      dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 }));
    this.viewGroups = Array.from({ length: options.maxShadowViews }, (_, layer) => device.createBindGroup({
      label: 'local light shadow view ' + layer, layout: shadowFrameLayout, entries: [{ binding: 0,
        resource: { buffer: this.viewUniform, offset: layer * LOCAL_SHADOW_VIEW_UNIFORM_STRIDE, size: LOCAL_SHADOW_VIEW_UNIFORM_STRIDE } }] }));
    this.sampler = device.createSampler({ label: 'local light shadow comparison', compare: 'less-equal',
      magFilter: 'linear', minFilter: 'linear' });
    this.computeLayout = device.createBindGroupLayout({ label: 'local lights (compute)', entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'comparison' } },
    ] });
  }

  private assertAlive(): void { if (this.disposed) throw new Error('LOCAL_LIGHTS_DISPOSED'); }

  /** Monotonic set revision for explicit history invalidation by dependents. */
  get revision(): number { return this.set.revision; }

  /** Range-sphere boxes (before and after) of the lights whose shading changed in the last update(); empty when
   * nothing changed. Dependents with region-based invalidation (GI) consume them. */
  get changedRegions(): readonly Bounds3[] { return this.changed; }

  /** Shadow views rendered this frame (enabled shadowed lights), as of the last update(). */
  get activeShadowViews(): number { return this.renderedLayers.length; }

  /** Fragment entries appended to the renderer's frame contract. */
  frameLayoutEntries(firstBinding: number): GPUBindGroupLayoutEntry[] {
    return [
      { binding: firstBinding, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: firstBinding + 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
      { binding: firstBinding + 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ];
  }
  frameGroupEntries(firstBinding: number): GPUBindGroupEntry[] {
    return [
      { binding: firstBinding, resource: { buffer: this.uniform } },
      { binding: firstBinding + 1, resource: this.atlasView },
      { binding: firstBinding + 2, resource: this.sampler },
    ];
  }
  /** Bind group for an explicit or 'auto' compute layout using bindings 0–2. */
  bindGroup(layout: GPUBindGroupLayout = this.computeLayout): GPUBindGroup {
    this.assertAlive();
    return this.device.createBindGroup({ label: 'local lights', layout, entries: this.frameGroupEntries(0) });
  }

  /** Upload the block only when the set changed. Returns uploaded bytes. */
  update(): number {
    this.assertAlive();
    if (this.uploadedRevision === this.set.revision) { this.changed = []; return 0; }
    const lights = this.set.list();
    this.changed = localLightChangeRegions(this.uploadedLights, lights);
    this.uploadedLights = lights;
    const packed = this.set.pack(this.packed);
    this.viewData.fill(0);
    for (const view of packed.views) {
      this.viewData.set(view.matrix, (view.layer * LOCAL_SHADOW_VIEW_UNIFORM_STRIDE + LOCAL_SHADOW_VIEW_MATRIX_OFFSET) / 4);
    }
    this.device.queue.writeBuffer(this.uniform, 0, this.packed);
    this.device.queue.writeBuffer(this.viewUniform, 0, this.viewData);
    this.renderedLayers = packed.views.map(view => view.layer);
    this.uploadedRevision = this.set.revision;
    this.uploads++;
    return this.packed.byteLength + this.viewData.byteLength;
  }

  /** Views of enabled shadowed lights, re-rendered every frame from the
   * current shared deformation. Camera visibility never removes a view. */
  shadowEncodings(): LocalLightShadowEncoding[] {
    this.assertAlive();
    return this.renderedLayers.map(layer => ({ view: this.layerViews[layer], frame: this.viewGroups[layer], layer }));
  }

  /** Encode every active view with the caller's shared shadow draw. The span's timestamps sit on the first and
   * last view passes themselves: empty compute marker passes around the views read 0 ms on Apple/Metal (measured
   * with 7 views that read 0.26-0.92 ms this way). Every timed pass writes a whole pair (unowned end to the
   * scratch slot), since partial writes lose the device there; without views one empty pass writes the span. */
  encodeShadows(encoder: GPUCommandEncoder, draw: (pass: GPURenderPassEncoder) => void,
    timing?: GPUComputePassTimestampWrites): void {
    const encodings = this.shadowEncodings();
    if (timing && !encodings.length) encoder.beginComputePass({ label: 'local light shadows (no views)', timestampWrites: timing }).end();
    encodings.forEach((encoding, index) => {
      const first = index === 0, last = index === encodings.length - 1;
      const timestampWrites = !timing ? undefined : first && last ? timing
        : first ? timestampSpanStart(timing) : last ? timestampSpanEnd(timing) : undefined;
      const pass = encoder.beginRenderPass({ label: 'local light shadow view ' + encoding.layer, colorAttachments: [],
        depthStencilAttachment: { view: encoding.view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        ...(timestampWrites ? { timestampWrites } : {}) });
      pass.setBindGroup(0, encoding.frame);
      draw(pass);
      pass.end();
    });
    this.shadowViewsRendered = encodings.length;
  }

  diagnostics() {
    const lights = this.disposed ? [] : this.set.list();
    return {
      revision: this.set.revision, lights: lights.length,
      enabledLights: lights.filter(light => light.enabled).length,
      shadowedLights: lights.filter(light => light.castShadows).length,
      reservedShadowViews: this.disposed ? 0 : this.set.shadowAssignments().reduce((sum, item) => sum + item.viewCount, 0),
      renderedShadowViews: this.shadowViewsRendered,
      limits: { maxLights: this.options.maxLights, maxShadowedLights: this.options.maxShadowedLights,
        maxShadowViews: this.options.maxShadowViews, shadowResolution: this.options.shadowResolution },
      shadowAtlasBytes: this.options.atlasBytes, ownedGpuBytes: this.bytes, uploads: this.uploads,
      shading: 'per-pixel-range-culled-loop' as const, fogLightList: 'froxel-block-sphere-aabb' as const,
      shadowProjection: 'perspective-spot-and-six-face-point' as const, shadowFilter: 'normal-offset-3x3-tent-pcf' as const,
      cameraCulling: false,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.uniform.destroy(); this.viewUniform.destroy(); this.atlas.destroy();
    this.account(-this.bytes); this.bytes = 0;
  }
}
