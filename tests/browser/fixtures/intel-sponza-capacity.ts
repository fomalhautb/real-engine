import { Renderer } from '../../../src/renderer';
import { GeometryStreamRuntimeBudget, type GeometryStreamRuntimeBudgetOptions } from '../../../src/geometry-stream-runtime-budget';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import type { GpuGeometryStream } from '../../../src/gpu-geometry-stream';
import { lookAt, multiply, perspective, transformPoint } from '../../../src/math';
import { loadIntelSponzaAssets, type IntelSponzaAssetOptions } from './intel-sponza-assets';
import { assertCapacityCut, assertCapacityLedger, capacityExchange, interpolateCapacityCamera, type CapacityCamera } from './intel-sponza-capacity-checks';

const MiB = 1024 ** 2, width = 1280, height = 720, projectionScale = height / (2 * Math.tan(Math.PI / 6.4));
// Renderer requests standard device limits. This preflight never requests an
// extra device; the real Renderer independently validates its actual limits.
const standardLimits = { maxBufferSize: 256 * MiB, maxStorageBufferBindingSize: 128 * MiB,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };
const A: CapacityCamera = { eye: [9, 1.8, 0], target: [-6, 3.5, 0] };
const B: CapacityCamera = { eye: [-9, 1.8, 0], target: [6, 3.5, 0] };
const turned: CapacityCamera = { eye: [...A.eye], target: [24, 3.5, 0] };
const trajectory = [{ label: 'a-initial', camera: A }, { label: 'b', camera: B },
  { label: 'a-return', camera: A }, { label: 'a-turn', camera: turned }, { label: 'a-final', camera: A }];

export interface IntelSponzaCapacityOptions extends Omit<IntelSponzaAssetOptions, 'sourceRaster'> {
  /** Complete live geometry component, not just physical pools. Default is the
   * original quarter-page allocation INCLUDING selection and rendered output. */
  residentMiB?: number;
  peakMiB?: number;
  maxErrorPixels?: number;
  settleFrames?: number;
  travelFrames?: number;
  stableFrames?: number;
  intervalFrames?: number;
  maxConcurrentMigrations?: number;
  growthOrder?: GeometryStreamRuntimeBudgetOptions['growthOrder'];
  exposureEV?: number;
}

type Diagnostics = ReturnType<Renderer['diagnostics']>;
type CapacityCapture = { label: string; png: string; frame: number; debug: 'material' | 'clusters'; exposureEV: number;
  transition?: { stage: string; settleFrame: number; phase: string; migrations: number; actualTriangles: number;
    missingAllocatedPages: number; trianglesSource: 'latest-completed-feedback'; feedbackFrameRange: [number, number] | null } };
type PrivateRenderer = { device: GPUDevice; geometryCapacity: { dispose(): Promise<void> };
  batches: { instanceData: Float32Array; stream: { runtime: GpuGeometryStream; bindings: { output: GPUBuffer } } }[] };

/** Explicit local-asset correctness/capture run, never part of the small browser
 * suite. Diagnostic readbacks and awaits deliberately invalidate FPS claims. */
export async function captureIntelSponzaCapacity(options: IntelSponzaCapacityOptions = {},
  onCapture?: (capture: CapacityCapture) => Promise<void>) {
  const settleFrames = options.settleFrames ?? 1200, travelFrames = options.travelFrames ?? 48,
    stableFrames = options.stableFrames ?? 24, intervalFrames = options.intervalFrames ?? 2,
    maxConcurrentMigrations = options.maxConcurrentMigrations ?? 4;
  const growthOrder = options.growthOrder === undefined ? 'large-first' : options.growthOrder;
  if (growthOrder !== 'large-first' && growthOrder !== 'small-first') throw Error('SPONZA_CAPACITY_GROWTH_ORDER');
  for (const [value, minimum, maximum, name] of [[settleFrames, 90, 3600, 'SETTLE'], [travelFrames, 1, 240, 'TRAVEL'],
    [stableFrames, 12, 120, 'STABLE'], [intervalFrames, 1, 60, 'INTERVAL'], [maxConcurrentMigrations, 1, 16, 'CONCURRENT']] as const) {
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw Error('SPONZA_CAPACITY_' + name);
  }
  if (stableFrames < intervalFrames * 2 || settleFrames < stableFrames) throw Error('SPONZA_CAPACITY_SETTLE_WINDOW');
  const maxErrorPixels = options.maxErrorPixels ?? 2;
  if (!Number.isFinite(maxErrorPixels) || maxErrorPixels < 0 || maxErrorPixels > 16) throw Error('SPONZA_CAPACITY_LOD_ERROR');
  for (const value of [options.residentMiB, options.peakMiB]) if (value !== undefined
    && (!Number.isFinite(value) || value <= 0 || value > 8192 || !Number.isSafeInteger(value * MiB))) throw Error('SPONZA_CAPACITY_BYTE_BUDGET');
  const started = performance.now();
  const progress = (phase: string, data: unknown = {}) => console.log('SPONZA_CAPACITY_PROGRESS ' + JSON.stringify({ phase, elapsedMs: performance.now() - started, data }));
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  canvas.style.cssText = 'width:1280px;height:720px;display:block'; document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    const assets = await loadIntelSponzaAssets(options, progress), { batches, accounting, bounds, sources } = assets;
    const inputs = batches.map(batch => ({ directory: batch.geometryStream.directory, capacity: batch.capacity }));
    const cuts = inputs.map(input => new GeometryStreamCut(input.directory));
    const sizing = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 8 * 1024 ** 3, maxPeakBytes: 8 * 1024 ** 3 }, standardLimits);
    const baseline = accounting.map((row, index) => sizing.state(index, row.slots));
    const maxResidentBytes = options.residentMiB === undefined ? baseline.reduce((sum, row) => sum + row.totalBytes, 0) : options.residentMiB * MiB;
    const demands = (camera: CapacityCamera) => {
      const viewProjection = multiply(perspective(Math.PI / 3.2, width / height, 0.1, 150), lookAt(camera.eye, camera.target, [0, 1, 0]));
      return cuts.map((cut, index) => cut.demands(batches[index].instances.map(instance => ({ eye: camera.eye, near: 0.1,
        projectionScale, maxErrorPixels, model: instance.model, viewProjection }))));
    };
    const initialPlanner = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes, maxPeakBytes: 8 * 1024 ** 3 }, standardLimits);
    const initialPlan = initialPlanner.plan(demands(A));
    // A usable automatic overlap budget covers the largest target in the fixed
    // route plus one slot. Explicit lower --peak-mib remains an acceptance test.
    const plans = trajectory.map(stage => initialPlanner.plan(demands(stage.camera)));
    const largestTarget = Math.max(...plans.flatMap(plan => plan.streams.map(stream => stream.totalBytes)));
    const largestSlot = Math.max(...inputs.map((input, index) => {
      const root = sizing.state(index, input.directory.rootPages.length);
      return geometryGpuPoolLayout(input.directory).bytesPerSlot + root.outputBytes / input.directory.rootPages.length;
    }));
    const maxPeakBytes = options.peakMiB === undefined ? maxResidentBytes + Math.ceil(largestTarget + largestSlot) : options.peakMiB * MiB;
    const budgetModel = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes, maxPeakBytes }, standardLimits);
    for (const planned of initialPlan.streams) {
      const stream = batches[planned.index].geometryStream;
      batches[planned.index].geometryStream = { ...stream, gpuPool: { ...stream.gpuPool, maxResidentBytes: planned.poolBytes } };
    }
    progress('preflight', { batches: batches.length, rootBytes: sizing.rootBytes, maxResidentBytes, maxPeakBytes,
      initialBytes: initialPlan.allocatedBytes, largestTarget, baselineQuarterPageBytes: baseline.reduce((n, s) => n + s.totalBytes, 0) });
    renderer = await Renderer.create(canvas, batches, { maxBatches: 512, antiAliasing: 'fxaa', textureBudgetBytes: 512 * MiB,
      textureStreaming: { sources, maxCpuBytes: 256 * MiB, maxInFlightBytes: 128 * MiB, maxUploadBytesPerFrame: 2 * MiB },
      sunShadow: { resolution: 2048, focusBounds: bounds as { min: [number, number, number]; max: [number, number, number] } },
      geometryRuntimeBudget: { maxResidentBytes, maxPeakBytes, intervalFrames, maxConcurrentMigrations, growthOrder,
        maxCopyPages: 4096, maxCopyBytes: 64 * MiB } });
    renderer.setSun({ direction: [0.35, 0.8, 0.4], color: [4, 3.8, 3.5] });
    renderer.setExposure(options.exposureEV ?? 3);
    renderer.setGeometryLod({ maxErrorPixels, frustumCulling: true, debug: 'material' });
    const active = renderer, internals = active as unknown as PrivateRenderer;
    let frame = 0, maximumOwnedGpuBytes = active.stats.ownedGpuBytes, peakAccountedBytes = 0;
    const seenMissing = inputs.map(() => new Set<number>()), pendingRecovery = inputs.map(() => new Set<number>());
    let recoveredPages = 0, maximumMissingAllocatedPages = 0;
    let epochs = inputs.map(() => 0);
    const frames: Record<string, unknown>[] = [], migrations: { frame: number; stage: string;
      streams: Awaited<ReturnType<typeof auditCuts>>; slots: number[] }[] = [];
    const historyChecks: { frame: number; streams: number[]; samples: number; currentError: number; previousError: number }[] = [];

    const checkDiagnostics = () => {
      const diagnostics = active.diagnostics(), budget = diagnostics.geometryRuntimeBudget;
      if (!budget.enabled) throw Error('SPONZA_CAPACITY_NOT_ENABLED');
      assertCapacityLedger(budget);
      if (active.errors.length) throw Error('SPONZA_CAPACITY_RENDER: ' + active.errors.join('; '));
      let liveBytes = 0, missingAllocatedPages = 0, desiredGroups = 0, actualTriangles = 0;
      for (const [index, stream] of diagnostics.geometryStreams.entries()) {
        if (stream.error || stream.pool.error || stream.selection.error || stream.cache.failedPages.length || !stream.pool.rootsReady || stream.feedback?.overflow) {
          throw Error('SPONZA_CAPACITY_STREAM: ' + JSON.stringify({ index, error: stream.error, failed: stream.cache.failedPages, rootsReady: stream.pool.rootsReady }));
        }
        const resident = new Set(stream.pool.residentPages);
        if (inputs[index].directory.rootPages.some(page => !resident.has(page))) throw Error('SPONZA_CAPACITY_ROOT_PAGE');
        const expectedBytes = budgetModel.state(index, stream.pool.slotCount);
        if (expectedBytes.poolBytes !== stream.pool.allocatedGpuBytes || stream.selection.gpuBytes > expectedBytes.selectionBytes
          || internals.batches[index].stream.bindings.output.size !== expectedBytes.outputBytes) {
          throw Error('SPONZA_CAPACITY_OWNER_BYTES: ' + index);
        }
        liveBytes += expectedBytes.totalBytes;
        for (const page of stream.allocation.pages) if (!resident.has(page)) {
          seenMissing[index].add(page); pendingRecovery[index].add(page); missingAllocatedPages++;
        }
        for (const page of pendingRecovery[index]) if (resident.has(page)) { recoveredPages++; pendingRecovery[index].delete(page); }
        desiredGroups += stream.feedback?.desiredGroups.length ?? 0;
        actualTriangles += (stream.feedback?.indices ?? 0) / 3;
      }
      if (liveBytes !== budget.liveBytes) throw Error('SPONZA_CAPACITY_LIVE_SUM');
      for (const source of diagnostics.textureStreaming?.sources ?? []) if (source.failure) throw Error('SPONZA_CAPACITY_TEXTURE: ' + source.id);
      maximumOwnedGpuBytes = Math.max(maximumOwnedGpuBytes, active.stats.ownedGpuBytes);
      peakAccountedBytes = Math.max(peakAccountedBytes, budget.accountedBytes);
      maximumMissingAllocatedPages = Math.max(maximumMissingAllocatedPages, missingAllocatedPages);
      return { diagnostics, budget, missingAllocatedPages, desiredGroups, actualTriangles };
    };

    const auditCuts = async (indices: readonly number[], camera: CapacityCamera) => {
      const rows = [];
      for (const index of indices) {
        const runtime = internals.batches[index].stream.runtime, directory = inputs[index].directory;
        const feedback = await runtime.probe(), resident = new Set(runtime.pool.residentPageIds());
        const expected = batches[index].instances.map(instance => cuts[index].plan(resident,
          { eye: camera.eye, near: 0.1, projectionScale, maxErrorPixels, model: instance.model }).selectedClusters);
        let counts: ReturnType<typeof assertCapacityCut>;
        try { counts = assertCapacityCut(directory, resident, expected, feedback); }
        catch (error) { throw Error('SPONZA_CAPACITY_CUT: ' + JSON.stringify({ index, name: batches[index].name, frame,
          epoch: runtime.capacityEpoch, cause: String(error), resident: [...resident], expected, actual: feedback.selectedClusters })); }
        rows.push({ index, name: batches[index].name, epoch: runtime.capacityEpoch, frame: feedback.frame, ...counts });
      }
      return rows;
    };
    // Test-only bounded readback of first/last resident pages and instances.
    // Copy local source and both rendered positions while holding a frame lease.
    const auditHistory = async (indices: readonly number[]) => {
      let samples = 0, currentError = 0, previousError = 0;
      for (const index of indices) {
        const batch = internals.batches[index], { runtime, bindings } = batch.stream, pool = runtime.pool;
        const resident = [...pool.residentPageIds()], pages = [...new Set([resident[0], resident.at(-1)!])];
        const instanceIds = [...new Set([0, batches[index].instances.length - 1])];
        const rows = pages.flatMap(page => instanceIds.map(instance => ({ page, instance,
          location: pool.clusterLocation(inputs[index].directory.pages[page].clusterIds[0])!,
          models: batch.instanceData.slice(instance * 56, instance * 56 + 32) })));
        const read = internals.device.createBuffer({ size: rows.length * 80, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const lease = pool.acquireFrame(pages); let submitted = false;
        try {
          const encoder = internals.device.createCommandEncoder();
          rows.forEach((row, rowIndex) => {
            const vertex = row.location.baseVertex + row.location.firstVertex;
            encoder.copyBufferToBuffer(pool.vertices, vertex * 144, read, rowIndex * 80, 16);
            encoder.copyBufferToBuffer(bindings.output, (row.instance * runtime.selection.physicalVertexCount + vertex) * 64, read, rowIndex * 80 + 16, 64);
          });
          const commands = [encoder.finish()]; submitted = true;
          const completion = await pool.submitFrame(lease, commands);
          if (completion.error) throw Error(completion.error);
          await read.mapAsync(GPUMapMode.READ);
          const values = new Float32Array(read.getMappedRange());
          rows.forEach((row, rowIndex) => {
            const offset = rowIndex * 20, local = Array.from(values.subarray(offset, offset + 3));
            const current = transformPoint(row.models.subarray(0, 16), local), previous = transformPoint(row.models.subarray(16, 32), local);
            for (let axis = 0; axis < 3; axis++) {
              currentError = Math.max(currentError, Math.abs(current[axis] - values[offset + 4 + axis]));
              previousError = Math.max(previousError, Math.abs(previous[axis] - values[offset + 12 + axis]));
            }
            samples++;
          });
          read.unmap();
        } finally { if (!submitted && !pool.diagnostics().disposed) pool.cancelFrame(lease); read.destroy(); }
      }
      if (!Number.isFinite(currentError) || !Number.isFinite(previousError) || currentError > 0.0001 || previousError > 0.0001) {
        throw Error('SPONZA_CAPACITY_HISTORY: ' + JSON.stringify({ currentError, previousError }));
      }
      historyChecks.push({ frame, streams: [...indices], samples, currentError, previousError });
    };

    const tick = async (camera: CapacityCamera, stage: string, capture = false) => {
      const before = active.stats.frames; active.render(camera.eye, camera.target);
      const png = capture ? canvas.toDataURL('image/png') : undefined;
      await active.probeHdr([{ x: width / 2, y: height / 2 }]);
      await new Promise<void>(resolve => setTimeout(resolve, 0));
      if (active.stats.frames !== before + 1) throw Error('SPONZA_CAPACITY_FRAME_NOT_RENDERED');
      frame++;
      const state = checkDiagnostics();
      const changed = state.diagnostics.geometryStreams.flatMap((stream, index) => stream.capacity.epoch !== epochs[index] ? [index] : []);
      if (changed.length) {
        const selected = await auditCuts(changed, camera); await auditHistory(changed);
        migrations.push({ frame, stage, streams: selected, slots: changed.map(index => state.diagnostics.geometryStreams[index].pool.slotCount) });
        epochs = state.diagnostics.geometryStreams.map(stream => stream.capacity.epoch);
      }
      const row = { frame, stage, renderFrame: active.stats.frames, liveBytes: state.budget.liveBytes, reservedBytes: state.budget.reservedBytes,
        accountedBytes: state.budget.accountedBytes, phase: state.budget.phase, migrations: state.budget.migrations,
        inFlightMigrations: state.budget.inFlightMigrations, pendingGrowthBytes: state.budget.pendingGrowthBytes,
        projectedLiveBytes: state.budget.projectedLiveBytes, copyReservations: state.budget.copyReservations,
        missingAllocatedPages: state.missingAllocatedPages, desiredGroups: state.desiredGroups, actualTriangles: state.actualTriangles,
        residentPages: state.diagnostics.geometryStreams.reduce((n, stream) => n + stream.pool.residentPages.length, 0) };
      frames.push(row); if (frame % 30 === 0) progress('frames', row);
      return { ...state, png };
    };
    const settled = (d: Diagnostics) => d.geometryRuntimeBudget.enabled && d.geometryRuntimeBudget.phase === 'idle'
      && !d.textureStreaming?.active && (d.textureStreaming?.sources ?? []).every(s => s.currentMip === s.plannedMip)
      && d.geometryStreams.every(stream => !stream.pool.pendingPages.length && !stream.cache.activeRequests && !stream.cache.queuedRequests
        && !stream.capacity.retiringOwners && stream.allocation.pages.every(page => stream.pool.residentPages.includes(page)));
    const signature = (d: Diagnostics) => JSON.stringify({ capacity: d.geometryRuntimeBudget.enabled
      ? [d.geometryRuntimeBudget.migrations, d.geometryRuntimeBudget.deferred] : null,
      textures: d.textureStreaming?.sources.map(s => [s.id, s.currentMip]),
      geometry: d.geometryStreams.map(s => [s.capacity.epoch, s.pool.slotCount, s.pool.residentPages, s.allocation.pages]) });

    type Endpoint = { label: string; camera: CapacityCamera; frame: number; framesToSettle: number; stableFrames: number;
      slots: number[]; exchange: ReturnType<typeof capacityExchange> | null; budget: ReturnType<typeof checkDiagnostics>['budget'];
      selected: Awaited<ReturnType<typeof auditCuts>>; actualTriangles: number; visibleClusters: number; rootsReady: boolean;
      missingAllocatedPages: number };
    const captures: CapacityCapture[] = [], endpoints: Endpoint[] = [];
    const transitionFrames = new Set([16, 32, 64, 128, 256]);
    for (const [stageIndex, stage] of trajectory.entries()) {
      if (stageIndex) for (let step = 1; step <= travelFrames; step++) {
        await tick(interpolateCapacityCamera(trajectory[stageIndex - 1].camera, stage.camera, step / travelFrames), stage.label + '-travel');
      }
      let stable = 0, prior = '', didSettle = false, framesToSettle = 0;
      for (let step = 1; step <= settleFrames; step++) {
        // Save the already-rendered material frame. No debug switch, additional
        // render or extra contribution to the stability counter is introduced.
        const state = await tick(stage.camera, stage.label, transitionFrames.has(step)), next = signature(state.diagnostics);
        stable = settled(state.diagnostics) && next === prior ? stable + 1 : 0; prior = next;
        if (state.png) {
          const feedbackFrames = state.diagnostics.geometryStreams.flatMap(stream => stream.feedback ? [stream.feedback.frame] : []);
          const image: CapacityCapture = { label: `${stage.label}-step-${step}-material`, png: state.png, frame, debug: 'material',
            exposureEV: options.exposureEV ?? 3, transition: { stage: stage.label, settleFrame: step,
              phase: state.budget.phase, migrations: state.budget.migrations, actualTriangles: state.actualTriangles,
              missingAllocatedPages: state.missingAllocatedPages, trianglesSource: 'latest-completed-feedback',
              feedbackFrameRange: feedbackFrames.length ? [Math.min(...feedbackFrames), Math.max(...feedbackFrames)] : null } };
          captures.push(image); await onCapture?.(image);
        }
        if (stable >= stableFrames) { didSettle = true; framesToSettle = step; break; }
      }
      if (!didSettle) throw Error('SPONZA_CAPACITY_DID_NOT_SETTLE: ' + JSON.stringify({ stage: stage.label, state: frames.at(-1), budget: active.diagnostics().geometryRuntimeBudget }));
      const selected = await auditCuts(inputs.map((_, index) => index), stage.camera);
      const state = checkDiagnostics(), slots = state.diagnostics.geometryStreams.map(s => s.pool.slotCount);
      await auditHistory(inputs.map((_, index) => index));
      const beforeCapture = signature(state.diagnostics);
      try {
        for (const debug of ['material', 'clusters'] as const) {
          // Diagnostic colors are already scene-linear display colors. Match
          // the other cluster demos' EV0 instead of applying the scene's EV3.
          const exposureEV = debug === 'clusters' ? 0 : options.exposureEV ?? 3;
          active.setGeometryLod({ debug }); active.setExposure(exposureEV);
          const capture = await tick(stage.camera, stage.label + '-' + debug, true);
          if (signature(capture.diagnostics) !== beforeCapture) throw Error('SPONZA_CAPACITY_CAPTURE_CHANGED_RESIDENCY');
          const image = { label: stage.label + '-' + debug, png: capture.png!, frame, debug, exposureEV };
          captures.push(image); await onCapture?.(image);
        }
      } finally {
        active.setGeometryLod({ debug: 'material' }); active.setExposure(options.exposureEV ?? 3);
      }
      const previousSlots: number[] | undefined = endpoints.at(-1)?.slots;
      endpoints.push({ label: stage.label, camera: stage.camera, frame, framesToSettle, stableFrames: stable, slots,
        exchange: previousSlots ? capacityExchange(previousSlots, slots) : null, budget: state.budget,
        selected, actualTriangles: selected.reduce((n, stream) => n + stream.triangles, 0),
        visibleClusters: selected.reduce((n, stream) => n + stream.visibleClusters, 0),
        rootsReady: state.diagnostics.geometryStreams.every(s => s.pool.rootsReady),
        missingAllocatedPages: state.missingAllocatedPages });
      progress('endpoint', { label: stage.label, framesToSettle, budget: state.budget, exchange: endpoints.at(-1)!.exchange,
        triangles: endpoints.at(-1)!.actualTriangles });
    }
    const exchangeAB = capacityExchange(endpoints[0].slots, endpoints[1].slots), exchangeBA = capacityExchange(endpoints[1].slots, endpoints[2].slots);
    const exchangeTurn = capacityExchange(endpoints[2].slots, endpoints[3].slots), exchangeTurnBack = capacityExchange(endpoints[3].slots, endpoints[4].slots);
    const failures = [];
    if (!exchangeAB.exchanged || !exchangeBA.exchanged) failures.push('A_B_A_DID_NOT_EXCHANGE_CAPACITY');
    if (!exchangeTurn.grown.length && !exchangeTurn.shrunk.length || !exchangeTurnBack.grown.length && !exchangeTurnBack.shrunk.length) failures.push('TURN_DID_NOT_CHANGE_CAPACITY');
    if (!recoveredPages) failures.push('NO_MISSING_PAGE_RECOVERY_OBSERVED');
    if (!migrations.length || !historyChecks.some(check => check.samples > 0)) failures.push('NO_MIGRATION_HISTORY_EVIDENCE');
    const finalState = checkDiagnostics(), info = active.info, errors = [...active.errors];
    // Also dispose during a fresh turn, exercising asynchronous candidate cleanup.
    for (let step = 0; step < intervalFrames + 6; step++) {
      const state = await tick(B, 'dispose-pending'); if (state.budget.phase !== 'idle') break;
    }
    const pendingDispose = active.diagnostics().geometryRuntimeBudget;
    active.dispose(); await internals.geometryCapacity.dispose(); await internals.device.queue.onSubmittedWorkDone();
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    const afterDispose = active.diagnostics().geometryRuntimeBudget, ownedAfterDispose = active.stats.ownedGpuBytes;
    if (!afterDispose.enabled || afterDispose.liveBytes || afterDispose.reservedBytes || afterDispose.accountedBytes || ownedAfterDispose !== 0) failures.push('DISPOSE_NOT_ZERO');
    const auditedMigrations = migrations.reduce((sum, batch) => sum + batch.streams.length, 0);
    if (!afterDispose.enabled || auditedMigrations !== afterDispose.migrations) failures.push('MIGRATION_EPOCH_COUNT_MISMATCH');
    if (active.errors.length) failures.push('RENDERER_ERRORS');
    return { schemaVersion: 1, fixture: 'intel-sponza-runtime-capacity', passed: failures.length === 0, failures,
      info, errors: [...new Set([...errors, ...active.errors])], elapsedMs: performance.now() - started,
      content: { assetIdentity: assets.manifest.assetIdentity, geometryRoot: assets.root, sourceInstances: assets.sourceInstances,
        sourceTriangles: assets.sourceTriangles, batches: batches.length, omitted: assets.omitted, conversion: assets.conversion, bounds },
      scope: { budget: 'page pools + metadata + maximum selection/readback + paged current/previous output + parameters; excludes textures, source/query proxies, CPU caches and driver memory',
        lighting: 'PBR direct lighting and fixed coarse shadow proxies; no GI in this geometry acceptance run',
        history: 'static world transforms, moving cameras; sampled current/previous world positions after every committed migration',
        performance: 'diagnostic readbacks and serial waits; elapsed time and FPS are not hardware benchmark results' },
      limits: { maxResidentBytes, maxPeakBytes, intervalFrames, maxConcurrentMigrations, growthOrder,
        maxErrorPixels, settleFrames, travelFrames, stableFrames },
      preflight: { rootBytes: sizing.rootBytes, initialBytes: initialPlan.allocatedBytes, initialSlots: initialPlan.streams.map(s => s.slots) },
      peakAccountedBytes, maximumOwnedGpuBytes, frames, migrations, migrationBatches: migrations.length, auditedMigrations, historyChecks, endpoints, captures,
      recovery: { uniqueMissingPages: seenMissing.reduce((n, pages) => n + pages.size, 0), recoveredPages, maximumMissingAllocatedPages,
        finalMissingAllocatedPages: finalState.missingAllocatedPages, pendingOrNoLongerDemanded: pendingRecovery.reduce((n, pages) => n + pages.size, 0) },
      exchangeAB, exchangeBA, exchangeTurn, exchangeTurnBack, pendingDispose, afterDispose, ownedAfterDispose };
  } finally {
    renderer?.dispose();
    // Preserve the original failure while still waiting for asynchronous owner
    // retirement; no page/selection/output is stranded by a failed oracle.
    if (renderer) await (renderer as unknown as PrivateRenderer).geometryCapacity?.dispose().catch(() => undefined);
    canvas.remove();
  }
}
