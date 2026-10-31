import { StreamedGalleryScene, type StreamedGalleryOptions, type StreamedGalleryPose } from '../../../src/geometry-demo-streamed-scene';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import { geometryCutCoverage, geometryCutWatertightness } from '../../../src/geometry-stream-coverage';
import { diagnoseGeometryStreamQualityDebt } from '../../../src/geometry-stream-quality-debt';
import type { GpuGeometryStream } from '../../../src/gpu-geometry-stream';
import { Renderer } from '../../../src/renderer';

export interface StreamedGalleryFixtureOptions {
  readonly scene?: StreamedGalleryOptions;
  readonly width?: number;
  readonly height?: number;
  /** Frames a settled signature must stay unchanged; must exceed the admission interval. */
  readonly stableFrames?: number;
  readonly maxSettleFrames?: number;
  /** Quality checks every this many frames during transitions; 0 measures recovery without probes. */
  readonly checkEvery?: number;
  readonly maxErrorPixels?: number;
}

type Vec3 = [number, number, number];
type Phase = { label: string; pose: StreamedGalleryPose };
type Check = Awaited<ReturnType<typeof checkFrame>>;
type PrivateRenderer = { device: GPUDevice; hdr: GPUTexture; batches: { scene: { name: string }; stream?: { runtime: GpuGeometryStream } }[] };

const BACKGROUND = [0.0035, 0.0055, 0.0065];

async function readHdr(renderer: Renderer): Promise<{ width: number; height: number; rgb: Float32Array }> {
  const r = renderer as unknown as PrivateRenderer, width = renderer.stats.width, height = renderer.stats.height;
  const bytesPerRow = Math.ceil(width * 8 / 256) * 256;
  const buffer = r.device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = r.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: r.hdr }, { buffer, bytesPerRow }, { width, height });
    r.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint16Array(buffer.getMappedRange()), rgb = new Float32Array(width * height * 3);
    const half = (bits: number) => { const e = (bits >>> 10) & 31, f = bits & 1023; return e ? (1 + f / 1024) * 2 ** (e - 15) : f * 2 ** -24; };
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) rgb[(y * width + x) * 3 + c] = half(words[y * bytesPerRow / 2 + x * 4 + c]);
    }
    buffer.unmap();
    return { width, height, rgb };
  } finally { buffer.destroy(); }
}

/** Nearest identity label: objects 0..n-1, n = architecture or background, -1 = unclassified. */
function labels(image: { rgb: Float32Array }, palette: readonly number[][], other: readonly number[][]): Int16Array {
  const result = new Int16Array(image.rgb.length / 3);
  for (let p = 0; p < result.length; p++) {
    let best = -1, distance = 0.03;
    const r = image.rgb[p * 3], g = image.rgb[p * 3 + 1], b = image.rgb[p * 3 + 2];
    for (let i = 0; i < palette.length; i++) {
      const d = Math.abs(r - palette[i][0]) + Math.abs(g - palette[i][1]) + Math.abs(b - palette[i][2]);
      if (d < distance) { distance = d; best = i; }
    }
    if (best < 0) for (const color of other) {
      const d = Math.abs(r - color[0]) + Math.abs(g - color[1]) + Math.abs(b - color[2]);
      if (d < distance) { distance = d; best = palette.length; }
    }
    result[p] = best;
  }
  return result;
}

/** Pixels deep inside a reference sculpture (every pixel within radius has the
 * same label) that the streamed frame shows as anything else. LOD silhouette
 * error is excluded by the radius; topology holes, cracks and mis-culling are not. */
function holes(reference: Int16Array, streamed: Int16Array, width: number, height: number, radii: readonly number[], objects: number) {
  const maxRadius = Math.max(...radii);
  // Separable erosion: distance to the nearest different label along rows, then columns (Chebyshev).
  const row = new Int32Array(width * height), interior = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    let last = -Infinity;
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      if (x === 0 || reference[p] !== reference[p - 1]) last = x;
      row[p] = x - last;
    }
    last = Infinity;
    for (let x = width - 1; x >= 0; x--) {
      const p = y * width + x;
      if (x === width - 1 || reference[p] !== reference[p + 1]) last = x;
      row[p] = Math.min(row[p], last - x);
    }
  }
  for (let x = 0; x < width; x++) for (let y = 1; y < height - 1; y++) {
    const p = y * width + x;
    if (reference[p] < 0 || reference[p] >= objects) continue;
    const radius = radii[reference[p]];
    if (row[p] < radius || y < radius || y >= height - radius) continue;
    let ok = true;
    for (let dy = -radius; dy <= radius && ok; dy++) {
      const q = p + dy * width;
      if (reference[q] !== reference[p] || row[q] < radius) ok = false;
    }
    if (ok) interior[p] = 1;
  }
  let count = 0, tested = 0;
  const byObject = new Map<number, number>(), visibleInStream = new Set<number>(), interiorObjects = new Set<number>();
  const holeSample: [number, number, number, number][] = [];
  for (let p = 0; p < interior.length; p++) {
    if (streamed[p] >= 0 && streamed[p] < objects) visibleInStream.add(streamed[p]);
    if (!interior[p]) continue;
    tested++; interiorObjects.add(reference[p]);
    if (streamed[p] !== reference[p]) {
      count++; byObject.set(reference[p], (byObject.get(reference[p]) ?? 0) + 1);
      if (holeSample.length < 8) holeSample.push([p % width, Math.floor(p / width), reference[p], streamed[p]]);
    }
  }
  const missingObjects = [...interiorObjects].filter(object => !visibleInStream.has(object));
  return { minRadius: Math.min(...radii), maxRadius, testedPixels: tested, holePixels: count, objectsTested: interiorObjects.size,
    holesByObject: [...byObject].sort((a, b) => b[1] - a[1]).slice(0, 8), holeSample, missingObjects, interior };
}

async function checkFrame(context: ReturnType<typeof makeContext>, label: string, settled: boolean) {
  const { scene, streamed, reference, runtimes, planners, view, maxErrorPixels } = context;
  const eyeTarget = context.current;
  reference.render(eyeTarget.eye, eyeTarget.target);
  // The streamed frame being checked is the last one rendered; probes read its cut.
  const cuts = await Promise.all(runtimes.map(runtime => runtime.probe()));
  let uncovered = 0, overlapping = 0, partialGroups = 0, visible = 0, fallback = 0, roots = 0, triangles = 0, cpuMismatches = 0;
  let boundaryEdges = 0, repeatedEdges = 0;
  let band = maxErrorPixels;
  const mismatchExamples: string[] = [], radii: number[] = [];
  for (let index = 0; index < runtimes.length; index++) {
    const object = scene.objects[index], cut = cuts[index], selected = (cut.selectedClusters ?? []).filter(cluster => cluster.instanceIndex === 0);
    const coverage = geometryCutCoverage(object.directory, selected.map(cluster => cluster.clusterId));
    uncovered += coverage.uncovered; overlapping += coverage.overlapping; partialGroups += coverage.partialGroups;
    const edges = geometryCutWatertightness(object.hierarchy, selected.map(cluster => cluster.clusterId));
    boundaryEdges += edges.boundaryEdges; repeatedEdges += edges.repeatedEdges;
    for (const cluster of selected) {
      if (cluster.visible) { visible++; triangles += object.directory.clusters[cluster.clusterId].triangleCount; }
      if (cluster.fallback) fallback++;
      if (cluster.root) roots++;
    }
    const resident = runtimes[index].pool.residentPageIds(), streamView = view(object.model);
    const debt = diagnoseGeometryStreamQualityDebt({ directory: object.directory, views: [streamView], residentPages: resident,
      admittedPages: new Set(runtimes[index].diagnostics().allocation.pages) });
    // Each sculpture's band is its own CPU-predicted fallback error (not a certified bound).
    const objectBand = Math.max(maxErrorPixels, debt.maxProjectedApproximateErrorPixels);
    band = Math.max(band, objectBand);
    radii.push(Math.min(48, Math.ceil(1.5 * objectBand) + 2));
    if (settled) {
      const expected = planners[index].plan(resident, streamView).selectedClusters;
      const actual = selected.map(cluster => cluster.clusterId).sort((a, b) => a - b);
      if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        cpuMismatches++;
        if (mismatchExamples.length < 3) mismatchExamples.push(`${object.name}:${actual.length}/${expected.length}`);
      }
    }
  }
  const [a, b] = await Promise.all([readHdr(reference), readHdr(streamed)]);
  const palette = scene.objects.map(object => object.identityColor.slice(0, 3));
  const other = [BACKGROUND, [0.05, 0.05, 0.05], [0.02, 0.02, 0.1], [0.1, 0.02, 0.02]];
  const ref = labels(a, palette, other), str = labels(b, palette, other);
  const result = holes(ref, str, a.width, a.height, radii, palette.length);
  const { interior, ...pixels } = result;
  let evidence: { reference: string; streamed: string } | null = null;
  if (pixels.holePixels > 0 && context.evidenceBudget > 0) {
    context.evidenceBudget--;
    const encode = (image: { rgb: Float32Array }, mark: boolean) => {
      const canvas = document.createElement('canvas'); canvas.width = a.width; canvas.height = a.height;
      const g = canvas.getContext('2d')!, data = g.createImageData(a.width, a.height);
      for (let p = 0; p < a.width * a.height; p++) {
        const hole = mark && interior[p] && str[p] !== ref[p];
        for (let c = 0; c < 3; c++) data.data[p * 4 + c] = hole ? (c === 0 ? 255 : 0) : Math.min(255, Math.round(Math.sqrt(image.rgb[p * 3 + c]) * 255));
        data.data[p * 4 + 3] = 255;
      }
      g.putImageData(data, 0, 0); return canvas.toDataURL('image/png');
    };
    evidence = { reference: encode(a, false), streamed: encode(b, true) };
  }
  return { label, frame: streamed.stats.frames, settled, eye: eyeTarget.eye.map(value => +value.toFixed(3)), evidence, radii,
    coverage: { uncovered, overlapping, partialGroups }, edges: { boundaryEdges, repeatedEdges }, visibleClusters: visible, fallbackClusters: fallback, rootClusters: roots,
    drawnTriangles: triangles, cpuCutMismatches: settled ? cpuMismatches : null, mismatchExamples,
    projectedErrorBandPixels: +band.toFixed(3), nominalRadiusObjects: radii.filter(radius => radius === Math.ceil(1.5 * maxErrorPixels) + 2).length,
    pixels,
    unclassifiedPixels: str.reduce((sum, value) => sum + (value < 0 ? 1 : 0), 0) };
}

function makeContext(scene: StreamedGalleryScene, streamed: Renderer, reference: Renderer, maxErrorPixels: number) {
  const r = streamed as unknown as PrivateRenderer;
  const runtimes = scene.objects.map(object => {
    const batch = r.batches.find(entry => entry.scene.name === object.name);
    if (!batch?.stream) throw new Error('GALLERY_FIXTURE_STREAM_MISSING:' + object.name);
    return batch.stream.runtime;
  });
  const context = {
    // Capacity migrations replace a runtime's pool and selection, not the runtime itself.
    scene, streamed, reference, maxErrorPixels, runtimes, evidenceBudget: 4,
    planners: scene.objects.map(object => new GeometryStreamCut(object.directory)),
    current: { eye: [0, 1.65, 0] as Vec3, target: [0, 1.2, 6] as Vec3 },
    view: (model: Float32Array): GeometryStreamView => ({ eye: context.current.eye, near: 0.1, maxErrorPixels,
      projectionScale: streamed.stats.height / (2 * Math.tan(Math.PI / 6.4)), model }),
  };
  return context;
}

function signature(renderer: Renderer) {
  const diagnostics = renderer.diagnostics(), budget = diagnostics.geometryRuntimeBudget;
  const streams = diagnostics.geometryStreams;
  // Requested pages may remain while capacity is exhausted; settling means every
  // admitted page is resident and nothing is still loading or migrating.
  const busy = !('phase' in budget) || budget.phase !== 'idle' || streams.some(stream => stream.pool.pendingPages.length
    || stream.allocation.pages.some(page => !stream.pool.residentPages.includes(page)) || stream.cache.activeRequests
    || stream.cache.queuedRequests || stream.capacity.preparing || stream.capacity.retiringOwners);
  const key = streams.map(stream => `${stream.pool.slotCount}:${stream.pool.residentPages.join(',')}`).join('|');
  const totals = streams.reduce((sum, stream) => ({ resident: sum.resident + stream.pool.residentPages.length,
    uploads: sum.uploads + stream.pool.uploads, evictions: sum.evictions + stream.pool.evictions,
    deferred: sum.deferred + stream.allocation.budgetDeferredGroups.length, requested: sum.requested + stream.allocation.requestedPages.length }),
  { resident: 0, uploads: 0, evictions: 0, deferred: 0, requested: 0 });
  return { busy, key, totals, budget, error: streams.find(stream => stream.error)?.error ?? ('error' in budget ? budget.error : null) };
}

/** Real Renderer streaming over an oversubscribed procedural gallery. Geometry
 * readback here is test-only diagnostics; runtime admission never reads it. */
export async function runStreamedGalleryFixture(options: StreamedGalleryFixtureOptions = {}) {
  const width = options.width ?? 400, height = options.height ?? 260, stableFrames = options.stableFrames ?? 75;
  const maxSettleFrames = options.maxSettleFrames ?? 1500, checkEvery = options.checkEvery ?? 6, maxErrorPixels = options.maxErrorPixels ?? 1;
  const built = performance.now();
  const scene = await StreamedGalleryScene.create({ objects: 16, around: 96, across: 32, ...options.scene, shading: 'identity' });
  const buildMs = performance.now() - built;
  const panel = document.createElement('section');
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#0b1418;color:#e8f1ef;padding:18px;font:14px system-ui;overflow:auto';
  panel.innerHTML = '<h1 style="font-size:18px;margin:0 0 8px">流送画廊：完整细节参考（左）与超额订阅流送（右）</h1>';
  const canvases = [0, 1].map(() => {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `width:${width}px;height:${height}px;display:inline-block;margin:6px`;
    panel.appendChild(canvas); return canvas;
  });
  document.body.appendChild(panel);
  let reference: Renderer | undefined, streamed: Renderer | undefined;
  try {
    reference = await Renderer.create(canvases[0], scene.referenceBatches);
    streamed = await Renderer.create(canvases[1], scene.batches, { geometryRuntimeBudget: scene.runtimeBudget, occlusion: {} });
    streamed.setGeometryLod({ maxErrorPixels, frustumCulling: true });
    const context = makeContext(scene, streamed, reference, maxErrorPixels);
    const s = streamed;
    const pace = () => new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    const render = async (pose: { eye: Vec3; target: Vec3 }) => {
      context.current = { eye: [...pose.eye], target: [...pose.target] };
      s.render(context.current.eye, context.current.target);
      if (s.errors.length) throw new Error(s.errors.join('\n'));
      await pace();
    };
    // Settle: idle admission, no outstanding pages or owners, and an unchanged
    // signature for stableFrames (longer than the admission interval).
    const settle = async (label: string, pose: StreamedGalleryPose, checks: Check[]) => {
      const started = performance.now(), startFrame = s.stats.frames;
      let stable = 0, lastKey = '', lastChangeFrame = startFrame, lastChangeMs = 0, frames = 0, maxFallback = 0;
      const residentSets = () => new Map(s.diagnostics().geometryStreams.map(stream => [stream.name, new Set(stream.pool.residentPages)]));
      const before = residentSets();
      while (frames < maxSettleFrames) {
        await render(pose); frames++;
        const state = signature(s);
        if (state.error) throw new Error(`GALLERY_FIXTURE_STREAM_ERROR:${label}:${state.error}`);
        if (state.busy || state.key !== lastKey) { stable = 0; lastKey = state.key; lastChangeFrame = s.stats.frames; lastChangeMs = performance.now() - started; }
        else if (++stable >= stableFrames) break;
        if (checkEvery && frames % checkEvery === 0 && stable === 0) {
          const check = await checkFrame(context, `${label}:transition`, false);
          maxFallback = Math.max(maxFallback, check.fallbackClusters); checks.push(check);
        }
      }
      const end = signature(s);
      if (stable < stableFrames) {
        const d = s.diagnostics(), b = d.geometryRuntimeBudget;
        throw new Error(`GALLERY_FIXTURE_DID_NOT_SETTLE:${label}:${JSON.stringify({ totals: end.totals,
          budget: 'phase' in b ? { phase: b.phase, migrations: b.migrations, deferred: b.deferred, inFlight: b.inFlightMigrations,
            live: b.liveBytes, reserved: b.reservedBytes, limits: b.limits } : b,
          streams: d.geometryStreams.map(stream => [stream.pool.slotCount, stream.pool.residentPages.length, stream.allocation.pages.length,
            stream.allocation.requestedPages.length, stream.pool.pendingPages.length, stream.cache.activeRequests, stream.cache.queuedRequests,
            stream.allocation.budgetDeferredGroups.length, stream.capacity.preparing, stream.capacity.retiringOwners]) })}`);
      }
      const check = await checkFrame(context, `${label}:settled`, true);
      checks.push(check);
      // Capacity migrations replace pools, so per-pool counters restart; compare resident sets instead.
      let loaded = 0, dropped = 0, reloaded = 0;
      for (const [name, pages] of residentSets()) {
        const previous = before.get(name) ?? new Set<number>(), history = droppedEver.get(name) ?? new Set<number>();
        for (const page of pages) if (!previous.has(page)) { loaded++; if (history.has(page)) reloaded++; }
        for (const page of previous) if (!pages.has(page)) { dropped++; history.add(page); }
        droppedEver.set(name, history);
      }
      return { label, framesToLastChange: lastChangeFrame - startFrame, msToLastChange: Math.round(lastChangeMs), framesRendered: frames,
        loadedPages: loaded, droppedPages: dropped, reloadedPages: reloaded,
        residentPages: end.totals.resident, budgetDeferredGroups: end.totals.deferred, maxTransitionFallback: maxFallback,
        migrations: 'migrations' in end.budget ? end.budget.migrations : null, settledCheck: check };
    };
    const checks: Check[] = [], phases: Awaited<ReturnType<typeof settle>>[] = [], firstFrames: Check[] = [];
    const droppedEver = new Map<string, Set<number>>();
    const sequence: Phase[] = [
      { label: 'cold-room-0', pose: scene.pose('teleport', 0) },
      { label: 'teleport-last-room', pose: scene.pose('teleport', 3) },
      { label: 'teleport-back-room-0', pose: scene.pose('teleport', 6) },
      { label: 'turn-180-into-previous-room', pose: scene.pose('turn', 4.0) },
      { label: 'turn-180-back', pose: scene.pose('turn', 8.0) },
    ];
    for (const [index, phase] of sequence.entries()) {
      if (index === 3) {
        // Arrive at the doorway facing forward and settle, then turn within 0.25 s.
        phases.push(await settle('doorway-before-turn', scene.pose('turn', 3.75), checks));
        for (let t = 3.75 + 1 / 60; t < 4 - 1e-6; t += 1 / 60) await render(scene.pose('turn', t));
      } else if (index === 4) {
        for (let t = 7.75 + 1 / 60; t < 8 - 1e-6; t += 1 / 60) await render(scene.pose('turn', t));
      }
      if (index > 0) {
        // The first frame after the jump, with Hi-Z built from the previous view.
        await render(phase.pose);
        firstFrames.push(await checkFrame(context, `${phase.label}:first-frame`, false));
      }
      phases.push(await settle(phase.label, phase.pose, checks));
    }
    // Continuous walk through the doorways: checks while pages stream.
    const walkChecks: Check[] = [];
    for (let t = 0, frame = 0; t < (scene.rooms * 12 + 1) / 1.6; t += 1 / 30, frame++) {
      await render(scene.pose('walk', t));
      const state = signature(s);
      if (state.error) throw new Error('GALLERY_FIXTURE_STREAM_ERROR:walk:' + state.error);
      if (frame % 10 === 0) walkChecks.push(await checkFrame(context, `walk:${t.toFixed(2)}`, false));
    }
    const diagnostics = s.diagnostics();
    return { adapter: s.info.adapterInfo, timing: s.info.timing, viewport: { width: s.stats.width, height: s.stats.height },
      accounting: scene.accounting, sceneOptions: scene.options, buildMs: Math.round(buildMs), maxErrorPixels,
      phases: phases.map(({ settledCheck, ...rest }) => ({ ...rest, settled: settledCheck })),
      firstFrames, transitionChecks: checks.filter(check => !check.settled), walkChecks,
      budget: diagnostics.geometryRuntimeBudget, backpressureFrames: s.stats.streamBackpressureFrames,
      peakOwnedBytes: s.stats.ownedGpuBytes, errors: [...s.errors, ...reference.errors] };
  } finally {
    streamed?.dispose(); reference?.dispose();
    await new Promise(resolve => setTimeout(resolve, 0));
    panel.remove();
  }
}
