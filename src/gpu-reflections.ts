import common from './shaders/reflections-common.wgsl?raw';
import updateSource from './shaders/reflections-update.wgsl?raw';
import temporalSource from './shaders/reflections-temporal.wgsl?raw';
import screenSource from './shaders/reflections-screen.wgsl?raw';
import sampleSource from './shaders/reflections-sample.wgsl?raw';
import surfaceFrame from './shaders/surface-frame.wgsl?raw';
import surfaceBrdf from './shaders/surface-brdf.wgsl?raw';
import giCommon from './shaders/diffuse-gi-common.wgsl?raw';
import giInterpolate from './shaders/diffuse-gi-interpolate.wgsl?raw';
import { reflectionFogShaderSource, type GpuVolumetricFog } from './gpu-volumetric-fog';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from './gi-visibility';
import { lookAt } from './math';
import { planReflectionGrid, REFLECTION_SCREEN_BYTES_PER_PIXEL, resolveReflectionsOptions, type ReflectionsOptions,
  type ReflectionGridPlan } from './reflections';
import type { GpuWorldQuery } from './gpu-world-query';
import type { GpuDiffuseGi } from './gpu-diffuse-gi';
import { diffuseGiQueryParameterSlots } from './diffuse-gi';
import { WORLD_HIT_BYTES } from './world-query';
import { timestampSpanEnd, timestampSpanStart } from './gpu-timestamp-span';

export const reflectionSurfaceSource = common + '\n' + sampleSource;
export const REFLECTION_PARAMETER_BYTES = 304;
export const REFLECTION_ANCHOR_BYTES = 256;
export const REFLECTION_HISTORY_BYTES = 96;
const PARAMETER_BYTES = REFLECTION_PARAMETER_BYTES, ANCHOR_BYTES = REFLECTION_ANCHOR_BYTES;
const SAMPLE_BYTES = 32, PROBE_LIMIT = 1024, ANCHOR_WORDS = ANCHOR_BYTES / 4;
type Entry = 'generateCamera' | 'generateReflections' | 'generateShadows' | 'integrate' | 'temporal';
export type ReflectionHistoryMode = 'none' | 'stationary' | 'surface' | 'parallax' | 'rejected';
/** Why the screen-space supplement is or is not used this frame. */
export type ReflectionScreenState = 'not-configured' | 'disabled' | 'awaiting-resize' | 'over-budget'
  | 'awaiting-capture' | 'camera-cut' | 'active';
const SCREEN_SAMPLE = `let screen = reflectionScreenTrace(camera.position.xyz + reflectionGeometry(camera) * reflection.limits.x,
            sample.direction.xyz, hit, distance);
        if (screen.reason < 4u) { screenCounts[screen.reason] += 1u; }
        incoming = mix(incoming, screen.color, screen.confidence);`;
const MODES: readonly ReflectionHistoryMode[] = ['none', 'stationary', 'surface', 'parallax', 'rejected'];
export interface ReflectionUpdate {
  frame: number; materialRevision: number; width: number; height: number;
  eye: readonly number[]; target: readonly number[]; fovY: number; near: number; far: number;
  sunDirection: readonly number[]; sunColor: readonly number[];
  /** Current view-projection; required for reprojected history. */
  viewProjection?: ArrayLike<number>;
  /** Previous rendered view-projection; omission is an explicit camera cut. */
  previousViewProjection?: ArrayLike<number>;
  /** Any batch changed this frame (pose, deformation or instance parameters). */
  sceneMotion?: boolean;
  /** World-query identity/history revision; a change rejects all history. */
  historyRevision?: number;
}
export interface ReflectionProbeAnchor {
  index: number; identity: number[]; position: number[]; normal: number[];
  /** Resolved radiance used by the surface pass (temporal output when active). */
  radiance: number[];
  /** This frame's estimate before history. */
  rawRadiance: number[];
  valid: boolean; failedQueries: number;
  historyCount: number; historyMode: ReflectionHistoryMode; rejectedHistory: number; historyClamped: boolean;
  meanReflectedDistance: number;
  /** Dominant reflected sample: [batch, entity, generation]; batch 0 is a miss. */
  reflectedIdentity: number[]; reflectedSample: number | null;
  /** Screen-trace outcomes of this anchor's samples; the rest were skipped (sky,
   * beyond range, metallic or moving hit, or the supplement inactive). */
  screen: { accepted: number; offscreen: number; unresolved: number; disagree: number };
}
export interface ReflectionProbeResult { frame: number; anchors: ReflectionProbeAnchor[] }

const LIGHT_EPSILON = 0;

/** Single reflected segment through the shared world query and material path.
 * Samples and reconstruction are bounded; no per-frame geometry readback. The
 * optional history is explicit, budgeted and reset on identity, version,
 * material, lighting, media and camera discontinuities. */
export class GpuReflections {
  readonly options;
  private buffers: GPUBuffer[] = [];
  private bytes = 0;
  private historyBytes = 0;
  private disposed = false;
  private enabled = true;
  private temporalEnabled: boolean;
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
  private history: GPUBuffer[] = [];
  private historyIndex = 0;
  private historyValid = false;
  private historyFrames = 0;
  private resetReason = 'initial';
  private pendingReset: string | null = 'initial';
  private temporalActive = false;
  private previous?: { frame: number; width: number; height: number; spacing: number; viewProjection: Float32Array;
    materialRevision: number; light: number[]; media: number; historyRevision: number };
  private pipelines!: Record<Entry, GPUComputePipeline>;
  private groups!: Record<Exclude<Entry, 'integrate' | 'temporal'>, GPUBindGroup>;
  private integrateGroups: GPUBindGroup[] = [];
  private temporalGroups: GPUBindGroup[] = [];
  private fogGroup?: GPUBindGroup;
  private readonly data = new Float32Array(PARAMETER_BYTES / 4);
  private screenEnabled: boolean;
  private screenTextures: GPUTexture[] = [];
  private screenBytes = 0;
  private screenSize: [number, number] = [0, 0];
  private screenSources?: { hdr: GPUTexture; depth: GPUTexture };
  private screenOverBudget = false;
  private screenCapture?: Float32Array;
  private screenState: ReflectionScreenState;
  private screenActive = false;
  private currentViewProjection?: Float32Array;

  get surfaceParameters(): GPUBuffer { return this.parameters; }
  get surfaceAnchors(): GPUBuffer { return this.anchors; }
  get temporalConfigured(): boolean { return !!this.options.temporal; }
  get screenTraceConfigured(): boolean { return !!this.options.screenTrace; }

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
    private shadow?: GPUTexture) {
    this.options = options; this.temporalEnabled = !!options.temporal; this.screenEnabled = !!options.screenTrace;
    this.screenState = options.screenTrace ? 'awaiting-resize' : 'not-configured';
  }
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
    const historySize = this.options.temporal ? cap * REFLECTION_HISTORY_BYTES : 0;
    const sizes = [PARAMETER_BYTES, cap * WORLD_HIT_BYTES, rays * WORLD_HIT_BYTES,
      rays * SAMPLE_BYTES, cap * ANCHOR_BYTES, Math.min(cap, PROBE_LIMIT) * ANCHOR_BYTES];
    const placeholder = this.options.screenTrace ? REFLECTION_SCREEN_BYTES_PER_PIXEL : 0;
    if (sizes.reduce((a, b) => a + b, 0) + 2 * historySize + placeholder > this.options.maxBytes) throw new Error('REFLECTIONS_MEMORY_BUDGET');
    if ([...sizes, historySize].some(size => size > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize))) throw new Error('REFLECTIONS_BUFFER_LIMIT');
    if (Math.ceil(rays / 64) > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('REFLECTIONS_DISPATCH_LIMIT');
    this.parameters = this.buffer('reflection parameters', sizes[0], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.cameraHits = this.buffer('reflection camera hits', sizes[1], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.secondaryHits = this.buffer('reflection secondary hits', sizes[2], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.samples = this.buffer('reflection directions and BSDF weights', sizes[3], GPUBufferUsage.STORAGE);
    this.anchors = this.buffer('reflection radiance anchors', sizes[4], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.readback = this.buffer('reflection explicit diagnostic readback', sizes[5], GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    if (historySize) {
      this.history = [0, 1].map(index => this.buffer('reflection reprojected history ' + index, historySize, GPUBufferUsage.STORAGE));
      this.historyBytes = 2 * historySize;
    }
    const giSource = this.gi ? giCommon.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`)
      + '\n@group(0) @binding(7) var<uniform> gi: GiParameters;\n@group(0) @binding(8) var<storage, read> giProbes: array<GiProbe>;\n'
      + giInterpolate + '\nfn reflectionDiffuse(hit:ReflectionHit)->vec3f { return giSampleSurfaceIrradiance(hit.position.xyz,hit.normal.xyz,reflectionGeometry(hit),gi.state.x) * hit.baseColor.xyz * (1.0-hit.previous.w) * (0.96/3.14159265); }'
      : 'fn reflectionDiffuse(hit:ReflectionHit)->vec3f { return hit.baseColor.xyz * (0.045+0.065*max(hit.normal.y,0.0)) * (1.0-hit.previous.w*0.6); }';
    const fogSource = this.fog ? reflectionFogShaderSource(this.fog, !!this.gi) : 'fn reflectionApplyFog(origin:vec3f,direction:vec3f,distance:f32,incoming:vec3f)->vec3f {return incoming;}';
    const module = this.device.createShaderModule({ label: 'world-space reflections',
      code: surfaceFrame + '\n' + surfaceBrdf + '\n' + common + '\n' + updateSource.replace('// REFLECTION_GI', giSource)
        .replace('// REFLECTION_SCREEN_SAMPLE', this.options.screenTrace ? SCREEN_SAMPLE : '') + '\n' + fogSource
        + (this.options.temporal ? '\n' + temporalSource : '') + (this.options.screenTrace ? '\n' + screenSource : '') });
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => `reflections:${error.lineNum}: ${error.message}`).join('\n'));
    const names: Entry[] = ['generateCamera', 'generateReflections', 'generateShadows', 'integrate',
      ...(this.options.temporal ? ['temporal' as const] : [])];
    const pipelines = await Promise.all(names.map(entryPoint => this.device.createComputePipelineAsync({
      label: 'reflections ' + entryPoint, layout: 'auto', compute: { module, entryPoint } })));
    this.pipelines = Object.fromEntries(names.map((name, index) => [name, pipelines[index]])) as Record<Entry, GPUComputePipeline>;
    const p = this.pipelines;
    this.groups = {
      generateCamera: this.bind(p.generateCamera, [[0, this.parameters], [1, this.queries.rayBuffer]]),
      generateReflections: this.bind(p.generateReflections, [[0, this.parameters], [1, this.queries.rayBuffer], [3, this.cameraHits], [5, this.samples]]),
      generateShadows: this.bind(p.generateShadows, [[0, this.parameters], [1, this.queries.rayBuffer], [4, this.secondaryHits], [5, this.samples]]),
    };
    // One-texel placeholders keep the bindings valid until the first resize.
    if (this.options.screenTrace) this.allocateScreen(1, 1);
    this.createIntegrateGroups();
    if (this.fog) this.fogGroup = this.device.createBindGroup({ layout: p.integrate.getBindGroupLayout(1), entries: [
      { binding: 0, resource: { buffer: this.fog.worldParameters } }, { binding: 1, resource: { buffer: this.fog.worldVolumes } },
      { binding: 2, resource: this.shadow!.createView() },
      { binding: 3, resource: this.device.createSampler({ compare: 'less-equal', minFilter: 'linear', magFilter: 'linear' }) },
    ] });
    // Group i writes history[i] from history[1 - i].
    if (this.options.temporal) this.temporalGroups = [0, 1].map(index => this.bind(p.temporal,
      [[0, this.parameters], [6, this.anchors], [9, this.history[1 - index]], [10, this.history[index]]]));
  }
  private createIntegrateGroups(): void {
    const pipeline = this.pipelines.integrate;
    const base: [number, GPUBuffer][] = [[0, this.parameters], [2, this.queries.hitBuffer], [3, this.cameraHits],
      [4, this.secondaryHits], [5, this.samples], [6, this.anchors]];
    const screen: GPUBindGroupEntry[] = this.options.screenTrace ? [{ binding: 11, resource: this.screenTextures[0].createView() },
      { binding: 12, resource: this.screenTextures[1].createView() }] : [];
    const group = (entries: readonly [number, GPUBuffer][]) => this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [...entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })), ...screen] });
    this.integrateGroups = this.gi ? this.gi.lightingProbeBuffers.map(probes =>
      group([...base, [7, this.gi!.lightingParameters], [8, probes]])) : [group(base)];
  }
  private allocateScreen(width: number, height: number): void {
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    this.screenTextures = [
      this.device.createTexture({ label: 'reflection previous HDR', size: [width, height], format: 'rgba16float', usage }),
      this.device.createTexture({ label: 'reflection previous depth', size: [width, height], format: 'depth32float', usage }),
    ];
    this.screenBytes = width * height * REFLECTION_SCREEN_BYTES_PER_PIXEL; this.account(this.screenBytes);
    this.screenSize = [width, height];
  }
  private releaseScreen(): void {
    if (!this.screenTextures.length) return;
    for (const texture of this.screenTextures) texture.destroy();
    this.screenTextures = []; this.account(-this.screenBytes); this.screenBytes = 0; this.screenSize = [0, 0];
  }
  /** Reallocate the owned previous-frame copies for a new HDR/depth pair. A size
   * beyond maxBytes or the device limit leaves the supplement inactive
   * ('over-budget'); world-query reflections are unaffected. */
  resize(width: number, height: number, hdr: GPUTexture, depth: GPUTexture): void {
    this.assertAlive();
    if (!this.options.screenTrace) return;
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) throw new Error('REFLECTIONS_SCREEN_SIZE');
    const bytes = width * height * REFLECTION_SCREEN_BYTES_PER_PIXEL;
    const fits = this.bytes + bytes <= this.options.maxBytes && Math.max(width, height) <= this.device.limits.maxTextureDimension2D;
    this.releaseScreen();
    if (fits) this.allocateScreen(width, height); else this.allocateScreen(1, 1);
    this.screenSources = { hdr, depth }; this.screenOverBudget = !fits; this.screenCapture = undefined;
    this.createIntegrateGroups();
  }
  /** Copy this frame's final scene-linear HDR and depth for the next frame's trace. */
  encodeCapture(encoder: GPUCommandEncoder): void {
    this.assertAlive();
    if (!this.options.screenTrace || !this.screenSources || this.screenOverBudget || !this.currentViewProjection) {
      this.screenCapture = undefined; return;
    }
    const size: [number, number] = [this.screenSize[0], this.screenSize[1]];
    encoder.copyTextureToTexture({ texture: this.screenSources.hdr }, { texture: this.screenTextures[0] }, size);
    encoder.copyTextureToTexture({ texture: this.screenSources.depth }, { texture: this.screenTextures[1] }, size);
    this.screenCapture = this.currentViewProjection;
  }
  /** Runtime switch for a configured screen-space supplement; world queries continue. */
  setScreenTraceEnabled(enabled: boolean): void {
    this.assertAlive(); if (typeof enabled !== 'boolean') throw new Error('REFLECTIONS_SCREEN_TRACE_ENABLED');
    if (enabled && !this.options.screenTrace) throw new Error('REFLECTIONS_SCREEN_TRACE_NOT_CONFIGURED');
    this.screenEnabled = enabled;
  }
  setEnabled(enabled: boolean): void {
    this.assertAlive(); if (typeof enabled !== 'boolean') throw new Error('REFLECTIONS_ENABLED');
    if (enabled !== this.enabled) this.invalidateHistory('reflections-toggle');
    this.enabled = enabled;
  }
  /** Runtime switch for configured history; disabled uses the fixed sample set. */
  setTemporalEnabled(enabled: boolean): void {
    this.assertAlive(); if (typeof enabled !== 'boolean') throw new Error('REFLECTIONS_TEMPORAL_ENABLED');
    if (enabled && !this.options.temporal) throw new Error('REFLECTIONS_TEMPORAL_NOT_CONFIGURED');
    if (enabled !== this.temporalEnabled) this.invalidateHistory('temporal-toggle');
    this.temporalEnabled = enabled;
  }
  /** Explicit discontinuity outside the identity/material/lighting contracts. */
  invalidateHistory(reason = 'explicit'): void {
    this.assertAlive();
    if (typeof reason !== 'string' || !reason) throw new Error('REFLECTIONS_HISTORY_REASON');
    this.pendingReset ??= reason;
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
    const matrix = (value: ArrayLike<number> | undefined, name: string): Float32Array | undefined => {
      if (value === undefined) return undefined;
      const values = Float32Array.from(Array.from(value));
      if (values.length !== 16 || !values.every(Number.isFinite)) throw new Error('REFLECTIONS_' + name);
      return values;
    };
    const viewProjection = matrix(input.viewProjection, 'VIEW_PROJECTION');
    const previousViewProjection = matrix(input.previousViewProjection, 'PREVIOUS_VIEW_PROJECTION');
    if (input.sceneMotion !== undefined && typeof input.sceneMotion !== 'boolean') throw new Error('REFLECTIONS_SCENE_MOTION');
    if (input.historyRevision !== undefined && (!Number.isSafeInteger(input.historyRevision) || input.historyRevision < 0
      || input.historyRevision > 0xffff_ffff)) throw new Error('REFLECTIONS_HISTORY_REVISION');
    const grid = planReflectionGrid(input.width, input.height, this.options, this.queries.options.maxRays, this.queries.options.maxTotalWorkUnits);
    const tangent = Math.tan(input.fovY / 2), horizontal = tangent * input.width / input.height;
    const data = this.data, uints = new Uint32Array(data.buffer);
    data.fill(0);
    data.set([...input.eye, this.options.maxDistance], 0);
    data.set([view[0] * horizontal, view[4] * horizontal, view[8] * horizontal, input.near], 4);
    data.set([view[1] * tangent, view[5] * tangent, view[9] * tangent, input.far], 8);
    data.set([-view[2], -view[6], -view[10], 0], 12);
    const sunDirection = input.sunDirection.map(value => value / sunLength);
    data.set(sunDirection, 16); data.set(input.sunColor, 20); data.set(this.options.skyRadiance, 24);
    uints.set([input.width, input.height, grid.columns, grid.rows], 28);
    uints.set([grid.spacing, grid.anchorCount, this.options.samplesPerAnchor, Number(this.enabled)], 32);
    uints.set([input.frame, input.materialRevision, this.options.fogSteps, 0], 36);
    data.set([this.options.normalBias, this.options.textureLod, this.options.maxGatherDistance, this.options.planeTolerance], 40);
    const temporal = this.options.temporal;
    this.temporalActive = !!temporal && this.temporalEnabled && this.enabled;
    if (temporal) {
      if (this.temporalActive && !viewProjection) throw new Error('REFLECTIONS_TEMPORAL_VIEW_PROJECTION');
      const light = [...sunDirection, ...input.sunColor].map(Math.fround);
      const media = this.fog?.diagnostics().revision ?? 0, historyRevision = input.historyRevision ?? 0;
      const old = this.previous;
      let reason = this.pendingReset;
      if (!reason && this.temporalActive) {
        reason = !old ? 'initial' : old.frame + 1 !== input.frame ? 'frame-discontinuity'
          : this.encodedFrame !== old.frame ? 'unrendered-history'
          : !previousViewProjection || previousViewProjection.some((value, i) => value !== old.viewProjection[i]) ? 'camera-cut'
          : old.width !== input.width || old.height !== input.height || old.spacing !== grid.spacing ? 'grid-changed'
          : old.materialRevision !== input.materialRevision ? 'material-changed'
          : old.light.some((value, i) => Math.abs(value - light[i]) > LIGHT_EPSILON) ? 'lighting-changed'
          : old.media !== media ? 'media-changed'
          : old.historyRevision !== historyRevision ? 'scene-identity-changed' : null;
      }
      if (reason && this.temporalActive) {
        this.historyValid = false; this.historyFrames = 0; this.resetReason = reason; this.pendingReset = null;
      }
      const cameraMoved = !!old && !!viewProjection && viewProjection.some((value, i) => value !== old.viewProjection[i]);
      data.set([temporal.minHistoryFrames, temporal.maxHistoryFrames, temporal.clampSigma, temporal.parallaxRoughness], 60);
      uints.set([Number(this.temporalActive), Number(this.temporalActive && this.historyValid),
        Number(cameraMoved), Number(input.sceneMotion ?? true)], 64);
      this.previous = viewProjection ? { frame: input.frame, width: input.width, height: input.height, spacing: grid.spacing,
        viewProjection, materialRevision: input.materialRevision, light, media, historyRevision } : undefined;
    }
    data.set(previousViewProjection ?? viewProjection ?? new Float32Array(16), 44);
    this.currentViewProjection = viewProjection;
    const screen = this.options.screenTrace;
    if (screen) {
      const captured = this.screenCapture;
      this.screenState = !this.screenEnabled || !this.enabled ? 'disabled' : this.screenOverBudget ? 'over-budget'
        : !this.screenSources || this.screenSize[0] !== input.width || this.screenSize[1] !== input.height ? 'awaiting-resize'
        : !captured ? 'awaiting-capture'
        : !previousViewProjection || previousViewProjection.some((value, i) => value !== captured[i]) ? 'camera-cut' : 'active';
      this.screenActive = this.screenState === 'active';
      data.set([screen.maxDistance, screen.thickness, screen.distanceTolerance, screen.edgeFade], 68);
      uints.set([Number(this.screenActive), screen.steps, this.screenSize[0], this.screenSize[1]], 72);
    }
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
      timestampSpanStart(timing));
    let uploaded = this.queries.encodeTrace(encoder, grid.anchorCount, this.options.textureLod, this.materialRevision);
    encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.cameraHits, 0, grid.anchorCount * WORLD_HIT_BYTES);
    dispatch('generateReflections', grid.rayCount, this.groups.generateReflections);
    uploaded += this.queries.encodeTrace(encoder, grid.rayCount, this.options.textureLod, this.materialRevision);
    encoder.copyBufferToBuffer(this.queries.hitBuffer, 0, this.secondaryHits, 0, grid.rayCount * WORLD_HIT_BYTES);
    dispatch('generateShadows', grid.rayCount, this.groups.generateShadows);
    uploaded += this.queries.encodeTrace(encoder, grid.rayCount, this.options.textureLod, this.materialRevision);
    // The span ends on the last encoded pass: temporal history when active.
    dispatch('integrate', grid.anchorCount, this.integrateGroups[this.gi?.lightingIndex ?? 0],
      this.temporalActive ? undefined : timestampSpanEnd(timing));
    if (this.temporalActive) {
      dispatch('temporal', grid.anchorCount, this.temporalGroups[this.historyIndex], timestampSpanEnd(timing));
      this.historyIndex = 1 - this.historyIndex; this.historyValid = true; this.historyFrames++;
    }
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
        const base = i * ANCHOR_WORDS;
        if (uints[base + 45] !== frame) throw new Error('REFLECTIONS_FRAME_MISMATCH');
        const radiance = Array.from(floats.subarray(base + 40, base + 43));
        const rawRadiance = Array.from(floats.subarray(base + 48, base + 51));
        if ([...radiance, ...rawRadiance].some(value => !Number.isFinite(value) || value < 0 || value > 65504)) throw new Error('REFLECTIONS_INVALID_RADIANCE');
        const mode = MODES[uints[base + 60]];
        if (!mode) throw new Error('REFLECTIONS_INVALID_HISTORY_MODE');
        return { index, identity: Array.from(uints.subarray(base, base + 4)), position: Array.from(floats.subarray(base + 8, base + 11)),
          normal: Array.from(floats.subarray(base + 12, base + 15)), radiance, rawRadiance,
          valid: uints[base + 44] === 1, failedQueries: uints[base + 46],
          historyCount: this.temporalActive ? floats[base + 43] : 1, historyMode: this.temporalActive ? mode : 'none',
          rejectedHistory: this.temporalActive ? uints[base + 61] : 0, historyClamped: this.temporalActive && uints[base + 62] === 1,
          meanReflectedDistance: floats[base + 51],
          reflectedIdentity: Array.from(uints.subarray(base + 56, base + 59)),
          reflectedSample: uints[base + 59] === 0xffff_ffff ? null : uints[base + 59],
          screen: { accepted: uints[base + 47] & 255, offscreen: (uints[base + 47] >>> 8) & 255,
            unresolved: (uints[base + 47] >>> 16) & 255, disagree: uints[base + 47] >>> 24 } };
      }) };
    } finally { if (this.readback.mapState === 'mapped') this.readback.unmap(); this.pendingReadback = false; }
  }
  diagnostics() {
    return { enabled: this.enabled, frame: this.encodedFrame, grid: this.grid ? { ...this.grid } : null,
      ownedGpuBytes: this.bytes, capacity: this.capacity, options: this.options,
      diffuseGi: !!this.gi, worldMedia: !!this.fog, reflectedSegments: 1, temporalHistory: this.temporalActive,
      temporal: { configured: !!this.options.temporal, enabled: this.temporalEnabled, active: this.temporalActive,
        valid: this.temporalActive && this.historyValid, framesSinceReset: this.historyFrames,
        lastResetReason: this.resetReason, historyBytes: this.historyBytes,
        minHistoryFrames: this.options.temporal?.minHistoryFrames ?? 0, maxHistoryFrames: this.options.temporal?.maxHistoryFrames ?? 0 },
      screenTrace: { configured: !!this.options.screenTrace, enabled: this.screenEnabled, active: this.screenActive,
        state: this.screenState, captureBytes: this.screenBytes, width: this.screenSize[0], height: this.screenSize[1],
        maxDistance: this.options.screenTrace?.maxDistance ?? 0, steps: this.options.screenTrace?.steps ?? 0 } };
  }
  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    for (const buffer of this.buffers) buffer.destroy(); this.buffers = []; this.history = [];
    this.account(-this.bytes); this.bytes = 0;
    this.releaseScreen(); this.screenSources = undefined; this.screenCapture = undefined; this.screenActive = false;
  }
}
