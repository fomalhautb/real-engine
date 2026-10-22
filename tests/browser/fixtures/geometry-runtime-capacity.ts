import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import { GeometryStreamRuntimeBudget } from '../../../src/geometry-stream-runtime-budget';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { GpuGeometryPagePool } from '../../../src/gpu-geometry-page-pool';
import type { GpuGeometryStream } from '../../../src/gpu-geometry-stream';
import { Renderer } from '../../../src/renderer';
import type { SceneBatch } from '../../../src/contracts';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { normalMatrix, transformPoint, translation } from '../../../src/math';

/** Real Renderer scheduling: two isolated camera views trade a shared physical
 * budget. Geometry readback below is test-only, never part of runtime admission. */
export async function runGeometryRuntimeCapacityFixture() {
  const scene = await GeometryDemoScene.create(), hierarchy = scene.hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded), layout = geometryGpuPoolLayout(directory);
  const roots = directory.rootPages.length, pages = directory.pages.length;
  if (pages <= roots) throw new Error('CAPACITY_FIXTURE_NEEDS_DETAIL');
  const inputs = [0, 1].map(() => ({ directory, capacity: 1 }));
  const sizing = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 256 * 1024 ** 2, maxPeakBytes: 512 * 1024 ** 2 },
    { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2,
      maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 });
  const maxResidentBytes = sizing.state(0, pages).totalBytes + sizing.state(1, roots).totalBytes;
  const maxPeakBytes = maxResidentBytes + sizing.state(1, roots).totalBytes;
  const makeBatches = (streamed: boolean): SceneBatch[] => [0, 1].map(index => {
    const model = translation(index ? 8 : -8, 0, 0);
    return { ...scene.batches[0], name: 'capacity-' + index, capacity: 1,
      instances: [{ ...scene.batches[0].instances[index], model, normalMatrix: normalMatrix(model) }],
      material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.instanceColor } },
      geometryLod: streamed ? undefined : hierarchy,
      geometryStream: streamed ? { directory, provider: async descriptor => encoded.pages[descriptor.id].bytes,
        gpuPool: { maxResidentBytes: index ? layout.rootBytes : layout.fullBytes, maxUploadBytesPerFrame: 1024 * 1024,
          maxUploadPagesPerFrame: 16, maxUploadWritesPerFrame: 256 }, maxRequestsPerFrame: 32 } : undefined };
  });
  const referenceBatches = makeBatches(false), streamBatches = makeBatches(true);
  const panel = document.createElement('section'); panel.id = 'geometry-runtime-capacity-visual';
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#101820;color:white;padding:24px;font:16px system-ui;overflow:auto';
  panel.innerHTML = '<h1>相机驱动的跨流几何容量交换</h1><p>同一共享预算从左侧雕塑迁移到右侧，再移回。旧资源在 GPU 完成后退休，根簇始终可画。</p>';
  const canvases = [0, 1].map(() => { const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width:420px;height:360px;display:inline-block;margin:8px'; panel.appendChild(canvas); return canvas; });
  const gallery = document.createElement('div'); panel.appendChild(gallery); document.body.appendChild(panel);
  let reference: Renderer | undefined, streamed: Renderer | undefined;
  try {
    reference = await Renderer.create(canvases[0], referenceBatches, { sunShadow: { resolution: 256 } });
    streamed = await Renderer.create(canvases[1], streamBatches, { sunShadow: { resolution: 256 },
      geometryRuntimeBudget: { maxResidentBytes, maxPeakBytes, intervalFrames: 2, maxCopyPages: 4096, maxCopyBytes: 64 * 1024 ** 2 } });
    const a = reference, b = streamed;
    a.setGeometryLod({ maxErrorPixels: 0, frustumCulling: true }); b.setGeometryLod({ maxErrorPixels: 0, frustumCulling: true });
    const privateRenderer = b as unknown as { device: GPUDevice;
      batches: { instanceData: Float32Array; stream: { runtime: GpuGeometryStream; bindings: { output: GPUBuffer } } }[] };
    const cutReference = new GeometryStreamCut(directory), transitionChecks: { current: number; previous: number; samples: number }[] = [];
    const states: { slots: number[]; migrations: number; reservedBytes: number; accountedBytes: number; phase: string }[] = [];
    let previousEpochs = [0, 0], frame = 0, peakAccounted = 0, minimumOwned = Infinity;
    const historyCheck = async () => {
      let current = 0, previous = 0, samples = 0;
      for (const batch of privateRenderer.batches) {
        const { runtime, bindings } = batch.stream, pool = runtime.pool;
        const ids = [...pool.residentPageIds()], locations = ids.map(id => pool.clusterLocation(directory.pages[id].clusterIds[0])!);
        const models = batch.instanceData.slice(0, 32);
        const read = privateRenderer.device.createBuffer({ size: ids.length * 80, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const lease = pool.acquireFrame(ids), encoder = privateRenderer.device.createCommandEncoder();
        try {
          for (const [index, location] of locations.entries()) {
            const vertex = location.baseVertex + location.firstVertex;
            encoder.copyBufferToBuffer(pool.vertices, vertex * 144, read, index * 80, 16);
            encoder.copyBufferToBuffer(bindings.output, vertex * 64, read, index * 80 + 16, 64);
          }
          const completion = await pool.submitFrame(lease, [encoder.finish()]);
          if (completion.error) throw new Error(completion.error);
          await read.mapAsync(GPUMapMode.READ);
          const values = new Float32Array(read.getMappedRange());
          for (let index = 0; index < ids.length; index++) {
            const offset = index * 20, local = Array.from(values.subarray(offset, offset + 3));
            const expected = transformPoint(models.subarray(0, 16), local), last = transformPoint(models.subarray(16, 32), local);
            for (let axis = 0; axis < 3; axis++) {
              current = Math.max(current, Math.abs(expected[axis] - values[offset + 4 + axis]));
              previous = Math.max(previous, Math.abs(last[axis] - values[offset + 12 + axis]));
            }
            samples++;
          }
          read.unmap();
        } finally { read.destroy(); }
      }
      transitionChecks.push({ current, previous, samples });
    };
    const tick = async (view: number) => {
      frame++;
      for (const batches of [referenceBatches, streamBatches]) for (const [index, batch] of batches.entries()) {
        const model = translation((index ? 8 : -8) + Math.sin(frame * 0.17) * 0.035, 0, 0);
        batch.instances[0] = { ...batch.instances[0], model, normalMatrix: normalMatrix(model) };
      }
      const center = view ? 8 : -8, eye: [number, number, number] = [center, 4, 6], target: [number, number, number] = [center, 0, 0];
      a.render(eye, target); b.render(eye, target);
      await Promise.all([a.probeHdr([{ x: 0, y: 0 }]), b.probeHdr([{ x: 0, y: 0 }])]);
      const diagnostics = b.diagnostics(), budget = diagnostics.geometryRuntimeBudget;
      if (!budget.enabled || budget.error || a.errors.length || b.errors.length) throw new Error('CAPACITY_FIXTURE_RUNTIME: ' + JSON.stringify({ budget, errors: [...a.errors, ...b.errors] }));
      if (diagnostics.geometryStreams.some(stream => !stream.pool.rootsReady)) throw new Error('CAPACITY_FIXTURE_MISSING_ROOT');
      peakAccounted = Math.max(peakAccounted, budget.accountedBytes); minimumOwned = Math.min(minimumOwned, b.stats.ownedGpuBytes);
      const epochs = diagnostics.geometryStreams.map(stream => stream.capacity.epoch);
      if (epochs.some((epoch, index) => epoch !== previousEpochs[index])) { await historyCheck(); previousEpochs = epochs; }
      const cut = await b.probeGeometryLod('capacity-' + view);
      const expected = cutReference.plan(new Set(diagnostics.geometryStreams[view].pool.residentPages), {
        eye, projectionScale: b.stats.height / (2 * Math.tan(Math.PI / 6.4)), near: 0.1, maxErrorPixels: 0,
        model: streamBatches[view].instances[0].model });
      if (cut.overflow || JSON.stringify(cut.selectedClusters.map(cluster => cluster.clusterId).sort((x, y) => x - y))
        !== JSON.stringify(expected.selectedClusters)) throw new Error('CAPACITY_FIXTURE_INCOMPLETE_CUT');
      states.push({ slots: diagnostics.geometryStreams.map(stream => stream.pool.slotCount), migrations: budget.migrations,
        reservedBytes: budget.reservedBytes, accountedBytes: budget.accountedBytes, phase: budget.phase });
      await new Promise(resolve => setTimeout(resolve, 0));
      return { diagnostics, cut, eye, target };
    };
    const samples = Array.from({ length: 480 }, (_, id) => ({ x: Math.floor((id % 24 + 0.5) * 420 / 24), y: Math.floor((Math.floor(id / 24) + 0.5) * 360 / 20) }));
    const endpoints = [];
    for (const view of [0, 1, 0]) {
      let settled: Awaited<ReturnType<typeof tick>> | undefined;
      for (let step = 0; step < 240; step++) {
        const value = await tick(view), budget = value.diagnostics.geometryRuntimeBudget;
        if (budget.enabled && budget.phase === 'idle' && value.cut.triangles === hierarchy.sourceTriangleCount
          && value.diagnostics.geometryStreams[1 - view].pool.slotCount === roots) { settled = value; break; }
      }
      if (!settled) throw new Error('CAPACITY_FIXTURE_DID_NOT_SETTLE: ' + JSON.stringify(b.diagnostics().geometryRuntimeBudget));
      await historyCheck();
      const [expected, actual, expectedVelocity, actualVelocity] = await Promise.all([
        a.probeHdr(samples), b.probeHdr(samples), a.probeVelocity(samples), b.probeVelocity(samples)]);
      let maxError = 0, velocityError = 0, foreground = 0;
      actual.forEach((pixel, index) => { if (pixel.slice(0, 3).some(value => value > 0.02)) foreground++;
        pixel.slice(0, 3).forEach((value, channel) => { maxError = Math.max(maxError, Math.abs(value - expected[index][channel])); });
        actualVelocity[index].forEach((value, channel) => { velocityError = Math.max(velocityError, Math.abs(value - expectedVelocity[index][channel])); }); });
      endpoints.push({ view, slots: settled.diagnostics.geometryStreams.map(stream => stream.pool.slotCount), maxError, velocityError, foreground });
      b.setGeometryLod({ debug: 'clusters' }); b.render(settled.eye, settled.target);
      const picture = document.createElement('img'); picture.src = canvases[1].toDataURL('image/png'); picture.style.cssText = 'width:320px;margin:6px';
      await picture.decode(); gallery.appendChild(picture); b.setGeometryLod({ debug: 'material' });
    }
    const diagnostics = b.diagnostics(), adapter = b.info.adapterInfo;
    // Teardown while a new camera migration is preparing, then await its owner
    // cleanup to expose late tracker decrements after Renderer sets its ledger to zero.
    for (let step = 0; step < 6; step++) {
      await tick(1); const budget = b.diagnostics().geometryRuntimeBudget;
      if (budget.enabled && budget.phase !== 'idle') break;
    }
    const coordinator = (b as unknown as { geometryCapacity: { dispose(): Promise<void> } }).geometryCapacity;
    const pendingPhase = b.diagnostics().geometryRuntimeBudget;
    a.dispose(); b.dispose(); await coordinator.dispose(); await new Promise(resolve => setTimeout(resolve, 0));
    return { adapter, roots, pages, maxResidentBytes, maxPeakBytes, peakAccounted, minimumOwned, endpoints, transitionChecks,
      states, diagnostics: diagnostics.geometryRuntimeBudget, pendingPhase, afterDispose: b.diagnostics().geometryRuntimeBudget,
      errors: [...a.errors, ...b.errors], ownedAfterDispose: [a.stats.ownedGpuBytes, b.stats.ownedGpuBytes] };
  } finally { reference?.dispose(); streamed?.dispose(); }
}
