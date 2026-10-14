import { BenchmarkRecorder } from './benchmark-stats';
import { sampleBenchmarkScenario } from './benchmark-scenario';
import { DemoScene } from './scene';
import { Renderer } from './renderer';
import { loadGltf } from './gltf-load';
import { m } from './material';
import type { MaterialBinding } from './contracts';

export interface BenchmarkOptions {
  trees?: number;
  seed?: number;
  warmupFrames?: number;
  frames?: number;
  /** Measure wall time instead of a fixed sample count. Never accelerates the 30-minute gate. */
  durationSeconds?: number;
  stress?: boolean;
  width?: number;
  height?: number;
  frameBudgetMs?: number;
}
export interface BenchmarkProgress { phase: string; frames: number; elapsedMs: number; gpuBytes: number }
const integer = (value: number, low: number, high: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < low || value > high) throw new Error('BENCHMARK_OPTION: ' + label);
  return value;
};
function config(options: BenchmarkOptions) {
  if (options.frames !== undefined && options.durationSeconds !== undefined) throw new Error('BENCHMARK_OPTION: choose frames or durationSeconds');
  if (options.stress !== undefined && typeof options.stress !== 'boolean') throw new Error('BENCHMARK_OPTION: stress');
  const frameBudgetMs = options.frameBudgetMs ?? 1000 / 30;
  if (!Number.isFinite(frameBudgetMs) || frameBudgetMs <= 0) throw new Error('BENCHMARK_OPTION: frameBudgetMs');
  return { trees: integer(options.trees ?? 128, 1, 256, 'trees'), seed: integer(options.seed ?? 104729, 0, 0xffffffff, 'seed'),
    warmupFrames: integer(options.warmupFrames ?? 300, 0, 10000, 'warmupFrames'),
    frames: options.durationSeconds === undefined ? integer(options.frames ?? 1800, 1, 1_000_000, 'frames') : null,
    durationSeconds: options.durationSeconds === undefined ? null : integer(options.durationSeconds, 1, 3600, 'durationSeconds'),
    stress: options.stress ?? false, width: integer(options.width ?? 1920, 64, 4096, 'width'),
    height: integer(options.height ?? 1080, 64, 4096, 'height'), frameBudgetMs };
}

/** Reproducible foundation workload. Deliberately separate from the interactive page's rAF loop. */
export class FoundationBenchmark {
  readonly options: ReturnType<typeof config>;
  readonly progress: BenchmarkProgress = { phase: 'created', frames: 0, elapsedMs: 0, gpuBytes: 0 };
  readonly recorder: BenchmarkRecorder;
  private scene?: DemoScene;
  private renderer?: Renderer;
  private disposed = false;
  private started = false;
  private raf = 0;
  private rejectFrame?: (reason: Error) => void;

  constructor(private readonly canvas: HTMLCanvasElement, options: BenchmarkOptions = {}) {
    this.options = config(options);
    this.recorder = new BenchmarkRecorder(this.options.frames ?? 1_000_000, this.options.frameBudgetMs);
  }
  get ownedGpuBytes(): number { return this.renderer?.stats.ownedGpuBytes ?? 0; }

  private nextFrame(): Promise<number> {
    if (this.disposed) return Promise.reject(new Error('BENCHMARK_DISPOSED'));
    if (document.visibilityState !== 'visible') return Promise.reject(new Error('BENCHMARK_BACKGROUND: cadence would be throttled'));
    return new Promise((resolve, reject) => {
      this.rejectFrame = reject;
      this.raf = requestAnimationFrame(now => {
        this.rejectFrame = undefined;
        if (document.visibilityState !== 'visible') reject(new Error('BENCHMARK_BACKGROUND'));
        else resolve(now);
      });
    });
  }
  private size(small = false): void {
    const scale = small ? 0.75 : 1;
    this.canvas.style.width = Math.round(this.options.width * scale) + 'px';
    this.canvas.style.height = Math.round(this.options.height * scale) + 'px';
  }

  async run() {
    if (this.started || this.disposed) throw new Error('BENCHMARK_ALREADY_STARTED_OR_DISPOSED');
    this.started = true;
    const startedAt = new Date().toISOString(), start = performance.now();
    this.progress.phase = 'loading';
    try {
      const assets = await Promise.all([
        loadGltf('/assets/gltf/SimpleSkin/glTF/SimpleSkin.gltf'),
        loadGltf('/assets/gltf/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf'),
      ]);
      const assetsDone = performance.now();
      if (this.disposed) throw new Error('BENCHMARK_DISPOSED');
      this.scene = await DemoScene.create(this.options.trees, assets);
      const scene = this.scene, sceneDone = performance.now();
      if (this.disposed) throw new Error('BENCHMARK_DISPOSED');
      this.size();
      this.renderer = await Renderer.create(this.canvas, scene.renderBatches);
      const renderer = this.renderer, rendererDone = performance.now();
      if (this.disposed) throw new Error('BENCHMARK_DISPOSED');
      const base = scene.renderBatches.find(batch => batch.name === 'material-card')!.material!;
      const alternate: MaterialBinding = { ...base, definition: { ...base.definition, shading: 'pbr', emissive: m.parameter(1) },
        parameters: new Float32Array([1, 0.65, 0.3, 1, 0.1, 0.3, 0.7, 0]) };
      // Compile both variants outside measured intervals; swaps still measure allocation/upload costs.
      if (this.options.stress) { await renderer.setMaterial('material-card', alternate); await renderer.setMaterial('material-card', base); }
      const prewarmDone = performance.now();
      this.progress.phase = 'warmup';
      let previousNow = await this.nextFrame();
      for (let i = 0; i < this.options.warmupFrames; i++) {
        previousNow = await this.nextFrame();
        const sample = sampleBenchmarkScenario(i, this.options.seed, false);
        scene.advance(1 / 60); scene.commitFrame(); renderer.setSun(sample.sun); renderer.render(sample.eye, sample.target);
        if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
      }
      scene.reset(); scene.commitFrame(); renderer.invalidateHistory();
      const initial = sampleBenchmarkScenario(0, this.options.seed, this.options.stress);
      renderer.setSun(initial.sun); renderer.render(initial.eye, initial.target);
      // This baseline frame is unmeasured; intervals begin at its next rAF boundary.
      previousNow = await this.nextFrame();
      const measuredStart = previousNow, rendererFrameOffset = renderer.stats.frames;
      let measuredDurationMs = 0, lastGpuFrame = rendererFrameOffset, baselineBytes = 0, lastWidth = 0, lastHeight = 0;
      const events: { frame: number; elapsedMs: number; action: string }[] = [];
      const resolutions: { frame: number; width: number; height: number }[] = [];
      const cycles: { frame: number; elapsedMs: number; gpuBytes: number; bodies: number; materials: number; images: number }[] = [];
      const captureGpu = () => {
        const { gpuSampleFrame, gpuMs } = renderer.stats;
        if (gpuSampleFrame !== null && gpuMs && gpuSampleFrame > lastGpuFrame) {
          this.recorder.recordGpu({ frame: gpuSampleFrame, deformation: gpuMs.deformation,
            shadow: gpuMs.shadow, surface: gpuMs.surface, resolve: gpuMs.resolve,
            worldQueryRefit: gpuMs.worldQueryRefit, geometryLod: gpuMs.geometryLod, diffuseGi: gpuMs.diffuseGi });
          lastGpuFrame = gpuSampleFrame;
        }
      };
      this.progress.phase = 'measuring';
      for (let frame = 0; frame < (this.options.frames ?? 1_000_000); frame++) {
        const now = await this.nextFrame(), interval = now - previousNow;
        measuredDurationMs = now - measuredStart;
        previousNow = now;
        const sample = sampleBenchmarkScenario(frame, this.options.seed, this.options.stress);
        const eventStart = performance.now();
        for (const action of sample.actions) {
          events.push({ frame, elapsedMs: measuredDurationMs, action });
          switch (action) {
            case 'spawn': if (!scene.spawnBox()) throw new Error('BENCHMARK_SPAWN_FAILED'); break;
            case 'remove': if (!scene.removeBox()) throw new Error('BENCHMARK_REMOVE_FAILED'); break;
            case 'reset': scene.reset(); break;
            case 'material-alt': await renderer.setMaterial('material-card', alternate); break;
            case 'material-base': await renderer.setMaterial('material-card', base); break;
            case 'resize-small': this.size(true); break;
            case 'resize-base': this.size(); break;
            case 'camera-cut': renderer.invalidateHistory(); break;
          }
        }
        const eventMs = performance.now() - eventStart, updateStart = performance.now();
        scene.advance(1 / 60); scene.commitFrame(); renderer.setSun(sample.sun);
        const updateMs = performance.now() - updateStart;
        renderer.render(sample.eye, sample.target);
        const stats = renderer.stats;
        if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
        this.recorder.record({ frame: stats.frames, elapsedMs: measuredDurationMs, frameIntervalMs: interval,
          updateMs, eventMs, cpuSubmitMs: stats.cpuSubmitMs, ownedGpuBytes: stats.ownedGpuBytes,
          uploadedBytes: stats.uploadedBytes, triangles: stats.triangles, deformedVertices: stats.deformedVertices, physicsBodies: scene.physics.bodyCount });
        captureGpu();
        if (stats.width !== lastWidth || stats.height !== lastHeight) {
          resolutions.push({ frame, width: stats.width, height: stats.height }); lastWidth = stats.width; lastHeight = stats.height;
        }
        if (frame === 0) baselineBytes = stats.ownedGpuBytes;
        if (this.options.stress && frame % 1800 === 1799) {
          const diagnostic = renderer.diagnostics();
          cycles.push({ frame, elapsedMs: measuredDurationMs, gpuBytes: stats.ownedGpuBytes,
            bodies: scene.physics.bodyCount, materials: diagnostic.materials.materials, images: diagnostic.materials.images });
          if (stats.ownedGpuBytes !== baselineBytes || scene.physics.bodyCount !== 6 || diagnostic.materials.materials !== scene.renderBatches.length || diagnostic.materials.images !== 2) {
            throw new Error('BENCHMARK_CYCLE_RESOURCE_DRIFT: ' + JSON.stringify(cycles.at(-1)));
          }
        }
        Object.assign(this.progress, { frames: frame + 1, elapsedMs: measuredDurationMs, gpuBytes: stats.ownedGpuBytes });
        if (this.options.durationSeconds !== null && measuredDurationMs >= this.options.durationSeconds * 1000) break;
        if (frame === 999_999 && this.options.durationSeconds !== null) throw new Error('BENCHMARK_FRAME_LIMIT_BEFORE_DURATION');
      }
      this.progress.phase = 'validating';
      // Diagnostic stalls and image export are outside the measured sequence.
      const verification = await renderer.verifyDeformation({ allInstances: true });
      captureGpu();
      if (!verification.vertices || verification.maxError >= 0.0002 || verification.maxNormalError >= 0.0002 || verification.boundsViolations) throw new Error('BENCHMARK_DEFORMATION_FAILED');
      if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
      this.progress.phase = 'complete';
      return { schema: 'real-engine-foundation-benchmark-v1', startedAt, options: this.options, environment: renderer.info,
        devicePixelRatio, simulationStepSeconds: 1 / 60, rendererFrameOffset, measuredDurationMs,
        startupMs: { assets: assetsDone - start, scene: sceneDone - assetsDone, renderer: rendererDone - sceneDone,
          prewarm: prewarmDone - rendererDone, total: prewarmDone - start },
        summary: this.recorder.summary(), events, resolutions, cycles,
        verification: { vertices: verification.vertices, maxError: verification.maxError, maxNormalError: verification.maxNormalError, boundsViolations: verification.boundsViolations },
        errors: [...renderer.errors], diagnostics: renderer.diagnostics(),
        notes: ['rAF intervals are browser callback cadence, not display scanout or isolated GPU execution time.',
          'GPU timings are sparse asynchronous samples; missing samples remain absent.',
          'Owned GPU bytes exclude driver, browser, JS, WASM, and swapchain allocations.',
          'Scenario advances exactly 1/60 simulated second per rendered frame; durationSeconds uses real wall time.',
          'This foundation scene does not validate large-scene, virtual geometry, GI, or reflection performance.'] };
    } catch (error) { this.progress.phase = 'failed'; this.dispose(); throw error; }
  }

  dispose(): void {
    this.disposed = true; cancelAnimationFrame(this.raf);
    this.rejectFrame?.(new Error('BENCHMARK_DISPOSED')); this.rejectFrame = undefined;
    // Repeat disposal also releases resources whose async initialization finished after cancellation.
    this.renderer?.dispose(); this.scene?.dispose();
  }
}
