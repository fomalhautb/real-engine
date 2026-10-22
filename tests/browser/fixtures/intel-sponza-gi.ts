import { Renderer } from '../../../src/renderer';
import { multiply, perspective, lookAt } from '../../../src/math';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { planGeometryStreamBudgets } from '../../../src/geometry-stream-budget';
import type { SceneBatch } from '../../../src/contracts';
import { intelSponzaReference } from './intel-sponza-reference';
import { loadIntelSponzaAssets } from './intel-sponza-assets';
import { FINAL_GATHER_DIAGNOSTIC_MAX_POINTS } from '../../../src/final-gather-diagnostics';
import type { GiVisibilityDirection } from '../../../src/diffuse-gi';

const MiB = 1024 ** 2;

/** Local Intel Sponza GI reproduction only. It is deliberately partial: four
 * BLEND primitives are omitted; coarse query proxies are not a visibility
 * reference. Referenced source documents, directories, pages and proxies are
 * verified against manifest hashes; the local manifests are caller inputs. */
export interface IntelSponzaGiOptions {
  geometryRoot?: string;
  queryRoot?: string;
  textureGpuRoot?: string;
  textureReferenceRoot?: string;
  uvAuditUrl?: string;
  exposureEV?: number;
  reference?: boolean;
  finalGather?: boolean;
  rasterReceiver?: boolean;
  /** Explicit actual-fragment reconstruction decisions; no extra GI frames. */
  reconstructionDiagnostics?: boolean;
  coverageDiagnostics?: boolean;
  supplementalGather?: boolean;
  /** Explicit quality experiment controls; omitted values preserve the baseline. */
  gatherPixelSpacing?: number;
  gatherCapacity?: number;
  supplementalCapacity?: number;
  supplementalWork?: number;
  diagnosticPoints?: readonly { x: number; y: number }[];
  /** Bounded single-bounce irradiance reference at actual visible fragments. */
  localReference?: boolean;
  sourceRaster?: boolean;
  fineGather?: boolean;
  anisotropy?: number;
  sceneBudget?: boolean;
  visibilityDirection?: GiVisibilityDirection;
  /** Exact rendered warmup frame count; residency must be settled at its end. */
  warmupFrames?: number;
}

export async function captureIntelSponzaGi(settings: IntelSponzaGiOptions = {}) {
  const warmupFrames = settings.warmupFrames ?? 600;
  if (settings.visibilityDirection !== undefined && settings.visibilityDirection !== 'probe-to-receiver'
    && settings.visibilityDirection !== 'receiver-to-probe') throw Error('SPONZA_VISIBILITY_DIRECTION');
  if (!Number.isSafeInteger(warmupFrames) || warmupFrames < 90 || warmupFrames > 3600) throw Error('SPONZA_WARMUP_FRAMES');
  if (settings.rasterReceiver && !settings.finalGather) throw Error('SPONZA_RASTER_RECEIVER_REQUIRES_FINAL_GATHER');
  if (settings.reconstructionDiagnostics && !settings.finalGather) throw Error('SPONZA_RECONSTRUCTION_REQUIRES_FINAL_GATHER');
  if (settings.localReference && !settings.reconstructionDiagnostics) throw Error('SPONZA_LOCAL_REFERENCE_REQUIRES_RECONSTRUCTION');
  for (const [value, maximum, label] of [[settings.gatherPixelSpacing, 128, 'GATHER_SPACING'],
    [settings.gatherCapacity, 65536, 'GATHER_CAPACITY'], [settings.supplementalCapacity, 16384, 'SUPPLEMENTAL_CAPACITY'],
    [settings.supplementalWork, 67108864, 'SUPPLEMENTAL_WORK']] as const) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 1 || value > maximum)) throw Error('SPONZA_' + label);
    if (value !== undefined && (!settings.finalGather || (label.startsWith('SUPPLEMENTAL') && !settings.supplementalGather))) throw Error('SPONZA_QUALITY_REQUIRES_ENABLED_PATH');
  }
  if (settings.diagnosticPoints && !settings.reconstructionDiagnostics) throw Error('SPONZA_DIAGNOSTIC_POINTS_REQUIRE_RECONSTRUCTION');
  if (settings.diagnosticPoints && (!Array.isArray(settings.diagnosticPoints) || settings.diagnosticPoints.length > 128
    || settings.diagnosticPoints.some(p => !p || !Number.isSafeInteger(p.x) || !Number.isSafeInteger(p.y)
      || p.x < 0 || p.x >= 1280 || p.y < 0 || p.y >= 720))) throw Error('SPONZA_DIAGNOSTIC_POINTS');
  if (settings.sourceRaster && !settings.queryRoot) throw Error('SPONZA_SOURCE_RASTER_REQUIRES_FINE_QUERY');
  const points = [{x:640,y:620},{x:480,y:560},{x:260,y:400},{x:1020,y:400},{x:640,y:300},{x:900,y:140},
    {x:80,y:475},{x:300,y:260},{x:960,y:265},{x:590,y:185}];
  if (settings.reconstructionDiagnostics) {
    for (const center of [{x:80,y:475},{x:285,y:480},{x:957,y:350},{x:300,y:250}]) {
      for (const dx of [-6,-4,-2,0,2,4,6]) {
        const pixel = { x: center.x + dx, y: center.y };
        if (!points.some(p => p.x === pixel.x && p.y === pixel.y)) points.push(pixel);
      }
    }
  }

  for (const point of settings.diagnosticPoints ?? []) {
    if (!points.some(p => p.x === point.x && p.y === point.y)) points.push({ ...point });
  }
  if (points.length > FINAL_GATHER_DIAGNOSTIC_MAX_POINTS) throw Error('SPONZA_DIAGNOSTIC_POINT_BUDGET');
  const started = performance.now();
  const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
  canvas.style.cssText = 'width:1280px;height:720px;display:block'; document.body.append(canvas);
  let renderer: Renderer | undefined;
  const progress = (phase: string, data: unknown = {}) => console.log('SPONZA_PROGRESS ' + JSON.stringify({ phase, elapsedMs: performance.now() - started, data }));
  try {
    const { root, manifest, batches, accounting, bounds, sources, sourceInstances, sourceTriangles, conversion, omitted }
      = await loadIntelSponzaAssets(settings, progress);
    const eye: [number, number, number] = [9, 1.8, 0], target: [number, number, number] = [-6, 3.5, 0];
    const viewProjection = multiply(perspective(Math.PI / 3.2, 1280 / 720, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
    const capacityPlan = settings.sceneBudget && !settings.sourceRaster ? planGeometryStreamBudgets(batches.map(batch => ({
      directory: batch.geometryStream.directory,
      demands: new GeometryStreamCut(batch.geometryStream.directory).demands(batch.instances.map(instance => ({
        eye, near: 0.1, projectionScale: 720 / (2 * Math.tan(Math.PI / 6.4)), maxErrorPixels: 2,
        model: instance.model, viewProjection }))),
    })), { maxResidentBytes: accounting.reduce((sum, row) => sum + row.gpuBytes, 0) }) : null;
    if (capacityPlan) for (const plan of capacityPlan.streams) {
      const stream = batches[plan.index].geometryStream;
      batches[plan.index].geometryStream = { ...stream, gpuPool: { ...stream.gpuPool, maxResidentBytes: plan.maxResidentBytes } };
    }
    const appliedAccounting = settings.sourceRaster ? [] : accounting.map((row, index) => ({ ...row,
      slots: capacityPlan?.streams[index].slots ?? row.slots,
      gpuBytes: capacityPlan?.streams[index].maxResidentBytes ?? row.gpuBytes }));
    progress('renderer-initializing', { batches: batches.length, sourceInstances, sourceTriangles, bounds,
      geometryGpuPoolBytes: appliedAccounting.reduce((n, p) => n + p.gpuBytes, 0) });
    const giCounts: [number, number, number] = [8, 6, 8];
    const giSpacing = bounds.max.map((v, i) => (v - bounds.min[i]) / giCounts[i]) as [number, number, number];
    const giOptions = { origin: bounds.min.map((v, i) => v + giSpacing[i] / 2) as [number, number, number],
      spacing: giSpacing, counts: giCounts, raysPerProbe: settings.fineGather ? 128 : 64, maxDistance: 80, normalBias: 0.03,
      skyRadiance: [0.2, 0.25, 0.35] as [number, number, number], multiBounce: false, maxBytes: 128 * MiB,
      visibilityDirection: settings.visibilityDirection ?? 'probe-to-receiver',
      maxVisibilityWorkUnits: settings.visibilityDirection === 'receiver-to-probe' ? 67_108_864 : 0,
      ...(settings.finalGather ? { finalGather: { pixelSpacing: settings.gatherPixelSpacing ?? (settings.fineGather ? 4 : 8),
        maxAnchors: settings.gatherCapacity ?? (settings.fineGather ? 65536 : 32768), raysPerAnchor: 1,
        pathSegments: 2, maxHistory: settings.fineGather ? 256 : 64, maxBytes: 64 * MiB,
        maxSupplementalAnchors: settings.supplementalGather ? settings.supplementalCapacity ?? 8192 : 0,
        maxSupplementalWorkUnits: settings.supplementalGather ? settings.supplementalWork ?? 8_388_608 : 0,
        receiver: settings.rasterReceiver ? 'raster' as const : 'world-query' as const } } : {}) };
    const renderBatches = settings.sourceRaster ? batches.map(({ geometryStream, ...batch }): SceneBatch => ({ ...batch, mesh: geometryStream.queryProxy.mesh })) : batches;
    renderer = await Renderer.create(canvas, renderBatches, { maxBatches: 512, antiAliasing: 'fxaa', textureBudgetBytes: 512 * MiB,
      worldQueries: { maxRays: settings.fineGather ? 65536 : 32768, maxBatches: 512, maxTrianglesPerMesh: 262144,
        maxBytes: settings.queryRoot ? 512 * MiB : 256 * MiB, maxTotalWorkUnits: 67108864 }, diffuseGi: giOptions,
      textureStreaming: { sources, maxCpuBytes: 256 * MiB, maxInFlightBytes: 128 * MiB, maxUploadBytesPerFrame: 2 * MiB },
      sunShadow: { resolution: 2048, focusBounds: bounds as any } });
    renderer.setSun({ direction: [0.35, 0.8, 0.4], color: [4, 3.8, 3.5] });
    renderer.setDiffuseGiEnabled(false);
    if (settings.finalGather) renderer.setDiffuseGiFinalGatherEnabled(false);
    renderer.setExposure(settings.exposureEV ?? 3);
    if (!settings.sourceRaster) renderer.setGeometryLod({ maxErrorPixels: 2, frustumCulling: true, debug: 'material' });
    progress('renderer-ready', { ownedGpuBytes: renderer.stats.ownedGpuBytes });
    const frames = [];
    let stableFrames = 0, firstSettledFrame: number | null = null, priorSignature = '', renderedFrame = 0;
    const residencySignature = () => {
      const d = renderer!.diagnostics();
      return JSON.stringify({ textures: d.textureStreaming?.sources.map(s => [s.id, s.currentMip]),
        geometry: d.geometryStreams.map(s => s.pool.residentPages), images: d.materials.imageBytes });
    };
    const residencySettled = (d: ReturnType<Renderer['diagnostics']>) => {
      for (const source of d.textureStreaming?.sources ?? []) {
        if (source.failure) throw Error('SPONZA_TEXTURE_FAILED: ' + source.id + ': ' + source.failure.message);
      }
      for (const stream of d.geometryStreams) {
        if (stream.cache.failedPages.length) throw Error('SPONZA_GEOMETRY_FAILED: ' + JSON.stringify(stream.cache.failedPages));
      }
      return !d.textureStreaming?.active && (d.textureStreaming?.sources ?? []).every(s => s.currentMip === s.plannedMip)
        && d.geometryStreams.every(s => !s.pool.pendingPages.length && !s.cache.activeRequests && !s.cache.queuedRequests
          && s.allocation.pages.every(page => s.pool.residentPages.includes(page)));
    };
    for (let frame = 0; frame < warmupFrames; frame++) {
      renderer.render(eye, target);
      renderedFrame++;
      await renderer.probeHdr([{ x: 640, y: 360 }]);
      await new Promise(requestAnimationFrame);
      if (renderer.errors.length) throw Error('SPONZA_RENDER: ' + renderer.errors.join('; '));
      const d = renderer.diagnostics();
      frames.push({ frame, renderFrame: renderedFrame, imageBytes: d.materials.imageBytes, ownedGpuBytes: renderer.stats.ownedGpuBytes,
        textureActive: d.textureStreaming?.active, geometryResidentPages: d.geometryStreams.reduce((n, stream) => n + stream.pool.residentPages.length, 0) });
      if (frame % 15 === 0) progress('warmup', frames.at(-1));
      const signature = residencySignature();
      const idle = residencySettled(d);
      stableFrames = idle && signature === priorSignature ? stableFrames + 1 : 0; priorSignature = signature;
      if (stableFrames >= 12 && firstSettledFrame === null) firstSettledFrame = renderedFrame;
    }
    if (stableFrames < 12) throw Error('SPONZA_RESIDENCY_NOT_SETTLED');
    const warmup = { requestedFrames: warmupFrames, renderedFrames: renderedFrame, firstSettledFrame, stableFramesAtEnd: stableFrames };
    let giAudit;
    {
      const captures: any[] = [];
      const capture = async (label: string, count: number) => {
        const signatureBefore = residencySignature(), hdrFrames = [], gpuSamples = [];
        const firstFrame = renderedFrame + 1;
        for (let i = 0; i < count; i++) {
          renderer!.render(eye, target); renderedFrame++; hdrFrames.push(await renderer!.probeHdr(points));
          if (renderer!.stats.gpuMs) gpuSamples.push({ frame: renderer!.stats.gpuSampleFrame, ...renderer!.stats.gpuMs });
        }
        const probe = label === 'direct-only' ? null : await renderer!.probeDiffuseGi();
        const diagnostics = renderer!.diagnostics();
        if (!residencySettled(diagnostics)) throw Error('SPONZA_CAPTURE_RESIDENCY_NOT_SETTLED');
        const grid = diagnostics.diffuseGi?.finalGather;
        const candidates = label.startsWith('gather-') && grid ? points.map(pixel => {
          const x = Math.floor((pixel.x - Math.floor(grid.actualSpacing / 2)) / grid.actualSpacing);
          const y = Math.floor((pixel.y - Math.floor(grid.actualSpacing / 2)) / grid.actualSpacing);
          const primary: number[] = [], recovery: number[] = [];
          for (let dy = -1; dy <= 2; dy++) for (let dx = -1; dx <= 2; dx++) {
            if (x + dx >= 0 && x + dx < grid.columns && y + dy >= 0 && y + dy < grid.rows) {
              (dx >= 0 && dx <= 1 && dy >= 0 && dy <= 1 ? primary : recovery).push(x + dx + (y + dy) * grid.columns);
            }
          }
          const supplemental: number[] = [], supplementalTiles: number[] = [];
          if (grid.supplementalTileSpacing) {
            const pitch = grid.supplementalTileSpacing, columns = Math.ceil(canvas.width / pitch), rows = Math.ceil(canvas.height / pitch);
            const sx = Math.floor(pixel.x / pitch), sy = Math.floor(pixel.y / pitch);
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
              if (sx + dx >= 0 && sy + dy >= 0 && sx + dx < columns && sy + dy < rows) {
                supplementalTiles.push(sx + dx + (sy + dy) * columns);
              }
            }
          }
          return { pixel, primary, recovery, supplemental, supplementalTiles, indices: [...primary, ...recovery] };
        }) : [];
        const tiles = [...new Set(candidates.flatMap(row => row.supplementalTiles))];
        const tileIndices = new Map<number, number[]>();
        for (let offset = 0; offset < tiles.length; offset += 256) {
          const map = await renderer!.probeDiffuseGiFinalGatherCoverageTiles(tiles.slice(offset, offset + 256));
          for (const entry of map.tiles) tileIndices.set(entry.tile, entry.indices);
        }
        for (const candidate of candidates) {
          candidate.supplemental = candidate.supplementalTiles.flatMap(tile => tileIndices.get(tile) ?? []);
          candidate.indices.push(...candidate.supplemental);
        }
        const indices = [...new Set(candidates.flatMap(row => row.indices))];
        const anchorProbes = [];
        for (let offset = 0; offset < indices.length; offset += 1024) {
          anchorProbes.push(await renderer!.probeDiffuseGiFinalGather(indices.slice(offset, offset + 1024)));
        }
        const finalGather = candidates.length ? { candidates,
          probe: { frame: anchorProbes[0]?.frame, anchors: anchorProbes.flatMap(probe => probe.anchors) },
          note: 'Candidate anchors only; inclusion does not prove surface compatibility or acceptance by reconstruction.' } : null;
        const diagnostic = settings.reconstructionDiagnostics ? await renderer!.probeDiffuseGiReconstruction(points,
          { coverage: settings.coverageDiagnostics }) : null;
        let coveragePng: string | null = null;
        if (diagnostic?.coverage) {
          const map = document.createElement('canvas'); map.width = diagnostic.width; map.height = diagnostic.height;
          const context = map.getContext('2d')!, image = context.createImageData(map.width, map.height);
          const palette = [[90,90,90], [36,150,80], [30,185,225], [255,55,45], [8,10,12], [95,80,210], [255,190,25]];
          diagnostic.coverage.branches.forEach((branch, i) => { image.data.set([...palette[branch], 255], i * 4); });
          context.putImageData(image, 0, 0); coveragePng = map.toDataURL();
        }
        const reconstruction = diagnostic ? { ...diagnostic, coverage: diagnostic.coverage ? {
          ...diagnostic.coverage, branches: undefined,
          legend: { primary: 'green', ring: 'cyan', uncovered: 'red', disabled: 'gray', background: 'black', unlit: 'purple', supplemental: 'amber' },
        } : null } : null;
        let localReference = null;
        if (settings.localReference && label === 'gather-single-bounce' && reconstruction) {
          const wanted = new Set(['640,620', '260,400', '80,475', '279,480', '281,480', '283,480', '291,480',
            '955,350', '957,350', '959,350', '961,350', '963,350', '640,300', '300,250', '900,140']);
          for (const point of settings.diagnosticPoints ?? []) wanted.add(`${point.x},${point.y}`);
          const receivers = reconstruction.points.flatMap(point => point.surface && wanted.has(`${point.pixel.x},${point.pixel.y}`)
            ? [{ pixel: point.pixel, ...point.surface }] : []);
          const { intelSponzaLocalReference } = await import('./intel-sponza-local-reference');
          const chunks: Awaited<ReturnType<typeof intelSponzaLocalReference>>[] = [];
          for (let offset = 0; offset < receivers.length; offset += 16) chunks.push(await intelSponzaLocalReference(
            renderer!, receivers.slice(offset, offset + 16), giOptions.skyRadiance, [4, 3.8, 3.5], [.35, .8, .4], giOptions.maxDistance));
          if (chunks.some(chunk => JSON.stringify(chunk.snapshot) !== JSON.stringify(chunks[0].snapshot))) throw Error('SPONZA_REFERENCE_SNAPSHOT_CHANGED');
          localReference = chunks.length ? { ...chunks[0], rows: chunks.flatMap(chunk => chunk.rows),
            budget: { perBatch: chunks[0].budget, batches: chunks.length,
              maxTotalRays: chunks.reduce((sum, chunk) => sum + chunk.budget.maxTotalRays, 0) },
            actual: { points: receivers.length, queryCalls: chunks.reduce((sum, chunk) => sum + chunk.actual.queryCalls, 0),
              primaryRays: chunks.reduce((sum, chunk) => sum + chunk.actual.primaryRays, 0),
              shadowRays: chunks.reduce((sum, chunk) => sum + chunk.actual.shadowRays, 0) } } : null;
        }
        const signatureAfter = residencySignature();
        const result = { label, firstFrame, lastFrame: renderedFrame, renderedFrames: count,
          png: canvas.toDataURL(), coveragePng, points, hdrFrames, gpuSamples, probe, finalGather, reconstruction, localReference,
          residencyUnchanged: signatureBefore === signatureAfter, diagnostics };
        captures.push(result);
        progress('gi-capture', { label, invalidProbes: probe?.invalidProbes, residencyUnchanged: result.residencyUnchanged, hdr: hdrFrames.at(-1) });
      };
      await capture('direct-only', 1);
      renderer.setDiffuseGiEnabled(true); renderer.setDiffuseGiMultiBounce(false);
      await capture('single-bounce', 3);
      const reference = settings.reference ? await intelSponzaReference(renderer, points, eye, target,
        giOptions.skyRadiance, [4,3.8,3.5], [.35,.8,.4], giOptions.maxDistance) : null;
      renderer.setDiffuseGiMultiBounce(true); await capture('multi-bounce', 16);
      if (settings.finalGather) {
        renderer.setDiffuseGiFinalGatherEnabled(true); renderer.setDiffuseGiMultiBounce(false);
        await capture('gather-single-bounce', settings.fineGather ? 256 : 64);
        renderer.setDiffuseGiMultiBounce(true); await capture('gather-multi-bounce', settings.fineGather ? 256 : 64);
      }
      giAudit = { options: giOptions, exposureEV: settings.exposureEV ?? 3, queryRoot: settings.queryRoot, sourceRaster: settings.sourceRaster ?? false,
        anisotropy: settings.anisotropy ?? 1, receiver: settings.rasterReceiver ? 'raster' : 'world-query', warmup, captures, reference,
        limitations: ['Fixed regular probe grid has no automatic relocation.', 'Coarse query proxies are not a visibility-error reference; zero-error proxies retain the cleaned LOD0 topology.',
          'Direct-only uses zero indirect term; the older unconfigured-GI capture included a constant ambient fill.',
          'Sky radiance is an explicit constant environment; no HDRI, reflection or volumetric fog.'] };
    }
    const diagnostics = renderer.diagnostics();
    const report = { root, buildConfig: manifest.config, info: renderer.info, bounds, sourceInstances, batches: batches.length, sourceTriangles,
      baselineAccounting: accounting, appliedAccounting, capacityPlan, frames, warmup, conversion, omitted, giAudit, errors: [...renderer.errors], diagnostics,
      limitations: ['Partial integration preview: four BLEND source primitives explicitly excluded.',
        'World diffuse GI preview; reflections and fog are not enabled.',
        'Fixed query proxies cast shadows; proxy quality remains unaccepted.',
        'Manual finite warmup, not a frame-rate benchmark or complete scene acceptance.'] };
    renderer.dispose(); return { ...report, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
