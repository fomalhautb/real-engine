import { Renderer } from '../../../src/renderer';
import { AnimationRig } from '../../../src/animation';
import { parseGltfNodeLayout } from '../../../src/gltf-node-layout';
import { createGltfMaterialLibrary } from '../../../src/gltf-material';
import { loadTextureStreamSource } from '../../../src/texture-stream-manifest';
import { validateGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { decodeGeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { normalMatrix, transformPoint, multiply, perspective, lookAt } from '../../../src/math';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { planGeometryStreamBudgets } from '../../../src/geometry-stream-budget';
import type { StreamedSceneBatch, SceneBatch } from '../../../src/contracts';
import type { TextureStreamSource } from '../../../src/texture-stream-types';
import { intelSponzaReference } from './intel-sponza-reference';

const MiB = 1024 ** 2;
async function checked(url: string, bytes: number, sha?: string, signal?: AbortSignal) {
  const response = await fetch(url, { signal });
  if (!response.ok || !response.body) throw Error('SPONZA_HTTP: ' + url);
  const reader = response.body.getReader(), data = new Uint8Array(bytes); let offset = 0;
  try {
    while (true) {
      const next = await reader.read(); if (next.done) break;
      if (next.value.length > bytes - offset) throw Error('SPONZA_BYTE_BUDGET: ' + url);
      data.set(next.value, offset); offset += next.value.length;
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; }
  finally { reader.releaseLock(); }
  if (sha) {
    if (offset !== bytes) throw Error('SPONZA_BYTE_LENGTH: ' + url);
    const digest = await crypto.subtle.digest('SHA-256', data);
    if (Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('') !== sha) throw Error('SPONZA_HASH: ' + url);
  }
  return data.subarray(0, offset);
}
const json = async (url: string, limit: number) => JSON.parse(new TextDecoder().decode(await checked(url, limit)));

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
  sourceRaster?: boolean;
  fineGather?: boolean;
  anisotropy?: number;
  sceneBudget?: boolean;
  /** Exact rendered warmup frame count; residency must be settled at its end. */
  warmupFrames?: number;
}

export async function captureIntelSponzaGi(settings: IntelSponzaGiOptions = {}) {
  const warmupFrames = settings.warmupFrames ?? 600;
  if (!Number.isSafeInteger(warmupFrames) || warmupFrames < 90 || warmupFrames > 3600) throw Error('SPONZA_WARMUP_FRAMES');
  if (settings.rasterReceiver && !settings.finalGather) throw Error('SPONZA_RASTER_RECEIVER_REQUIRES_FINAL_GATHER');
  const root = settings.geometryRoot ?? '/artifacts/sponza-group16-pages-20261022/';
  const textureGpuRoot = settings.textureGpuRoot ?? '/artifacts/sponza-textures-gpu-20261020/';
  const textureReferenceRoot = settings.textureReferenceRoot ?? '/artifacts/sponza-textures-reference-20261020/';
  const uvAuditUrl = settings.uvAuditUrl ?? '/artifacts/sponza-uv-demand-20261020.json';
  if (settings.sourceRaster && !settings.queryRoot) throw Error('SPONZA_SOURCE_RASTER_REQUIRES_FINE_QUERY');
  const started = performance.now();
  const canvas = document.createElement('canvas'); canvas.width = 1280; canvas.height = 720;
  canvas.style.cssText = 'width:1280px;height:720px;display:block'; document.body.append(canvas);
  let renderer: Renderer | undefined;
  const progress = (phase: string, data: unknown = {}) => console.log('SPONZA_PROGRESS ' + JSON.stringify({ phase, elapsedMs: performance.now() - started, data }));
  try {
    const manifest = await json(root + 'manifest.json', 32 * MiB);
    const queryManifest = settings.queryRoot ? await json(settings.queryRoot + 'manifest.json', 32 * MiB) : null;
    if (queryManifest && queryManifest.sourcePackage.assetIdentity !== manifest.assetIdentity) throw Error('SPONZA_FINE_PROXY_SOURCE');
    const queryEntries = new Map<string, any>((queryManifest?.primitives ?? []).map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p]));
    const doc = JSON.parse(new TextDecoder().decode(await checked(root + manifest.source.file, manifest.source.bytes, manifest.source.sha256)));
    if (doc.animations?.length || doc.skins?.length || doc.meshes.some((m: any) => m.primitives.some((p: any) => p.targets?.length))) throw Error('SPONZA_PREVIEW_EXPECTED_STATIC_SOURCE');
    const conversion = { path: 'materials[12].emissiveFactor[2]', before: doc.materials[12].emissiveFactor[2], after: 1 };
    if (Math.abs(conversion.before - 1) > 2 ** -22) throw Error('SPONZA_FACTOR_CONVERSION_BOUND');
    doc.materials[12].emissiveFactor[2] = 1;
    const layout = parseGltfNodeLayout(doc, doc.meshes.map(() => []));
    const pose = new AnimationRig(layout.rigNodes, []).sample(0);
    const roots = new Set(layout.scenes[layout.scene].roots);
    const nodesByMesh = new Map<number, number[]>();
    layout.nodes.forEach((node, i) => {
      let top = i; while (layout.nodes[top].parent !== null) top = layout.nodes[top].parent!;
      if (!roots.has(top) || node.mesh === undefined) return;
      const list = nodesByMesh.get(node.mesh) ?? []; list.push(i); nodesByMesh.set(node.mesh, list);
    });
    const textureManifest = await json(textureGpuRoot + 'manifest.json', MiB);
    const images = new Map(), sources: TextureStreamSource[] = [];
    for (const entry of textureManifest.images) {
      const reference = entry.roles.includes('normal:linear');
      const source = await loadTextureStreamSource((reference ? textureReferenceRoot : textureGpuRoot) + `image-${entry.image}/manifest.json`);
      images.set(entry.image, source.fallback); sources.push(source);
    }
    progress('textures-ready', { count: sources.length, referenceNormals: 23 });
    const materials = createGltfMaterialLibrary(doc, { images });
    const uvAudit = await json(uvAuditUrl, MiB);
    if (uvAudit.sourceSha256 !== manifest.source.sha256) throw Error('SPONZA_UV_SOURCE_MISMATCH');
    const uv = new Map(uvAudit.primitives.map((p: any) => [`${p.meshIndex}:${p.primitiveIndex}`, p.textureUvSpan]));
    const batches: StreamedSceneBatch[] = [], omitted: unknown[] = [], accounting = [];
    const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    let entityId = 1, sourceTriangles = 0, sourceInstances = 0;
    for (const entry of manifest.primitives) {
      const nodes = nodesByMesh.get(entry.meshIndex) ?? [];
      sourceInstances += nodes.length;
      if (entry.route !== 'paged') {
        omitted.push({ mesh: entry.meshIndex, primitive: entry.primitiveIndex, material: entry.materialIndex,
          route: entry.route, coverage: entry.coverage, nodes, sourceTriangles: entry.sourceTriangleCount,
          reason: 'BLEND remains unsupported; explicit partial integration preview, not a completed scene' });
        continue;
      }
      if (!nodes.length) continue;
      const directory = validateGeometryStreamDirectory(JSON.parse(new TextDecoder().decode(await checked(root + entry.streamDirectoryFile,
        entry.streamDirectoryBytes, entry.streamDirectorySha256))));
      const proxyEntry = queryEntries.get(`${entry.meshIndex}:${entry.primitiveIndex}`)?.queryProxy ?? entry.queryProxy;
      if (settings.queryRoot && proxyEntry === entry.queryProxy) throw Error('SPONZA_FINE_PROXY_MISSING');
      const queryProxy = await decodeGeometryQueryProxy(directory, await checked((settings.queryRoot ?? root) + proxyEntry.file,
        proxyEntry.bytes, proxyEntry.sha256), proxyEntry.sha256,
        { maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB });
      if (settings.sourceRaster && queryProxy.requestedError !== 0) throw Error('SPONZA_SOURCE_RASTER_REQUIRES_LOD0');
      if (settings.sourceRaster && queryProxy.mesh.indices.length / 3 !== entry.triangleCount) throw Error('SPONZA_SOURCE_RASTER_TRIANGLES');
      const verticesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].vertexCount, 0)));
      const indicesPerSlot = Math.max(...directory.pages.map(p => p.clusterIds.reduce((n, id) => n + directory.clusters[id].triangleCount * 3, 0)));
      const slotBytes = verticesPerSlot * 144 + indicesPerSlot * 4;
      const slots = Math.max(directory.rootPages.length, Math.ceil(directory.pages.length / 4));
      const gpuBytes = slots * slotBytes + directory.clusters.length * 32 + directory.pages.length * 4;
      const rootDecoded = directory.rootPages.reduce((n, id) => n + directory.pages[id].decodedBytes, 0);
      const largestPage = Math.max(...directory.pages.map(p => p.decodedBytes));
      const largestWork = Math.max(...directory.pages.map(p => 2 * p.encodedBytes + 2 * p.decodedBytes));
      const span = uv.get(`${entry.meshIndex}:${entry.primitiveIndex}`);
      const instances = nodes.map(node => {
        const model = pose.world[node];
        const determinant = model[0] * (model[5] * model[10] - model[9] * model[6]) - model[4] * (model[1] * model[10] - model[9] * model[2]) + model[8] * (model[1] * model[6] - model[5] * model[2]);
        if (determinant <= 0) throw Error('SPONZA_REFLECTED_INSTANCE: ' + node);
        for (const id of directory.roots) {
          const b = directory.clusters[id].lodBounds;
          for (let corner = 0; corner < 8; corner++) {
            const p = transformPoint(model, [b[(corner & 1) ? 'max' : 'min'][0], b[(corner & 2) ? 'max' : 'min'][1], b[(corner & 4) ? 'max' : 'min'][2]]);
            for (let axis = 0; axis < 3; axis++) { bounds.min[axis] = Math.min(bounds.min[axis], p[axis]); bounds.max[axis] = Math.max(bounds.max[axis], p[axis]); }
          }
        }
        return { entityId: entityId++, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
          color: [1, 1, 1, 1] as [number, number, number, number], metalness: 1, roughness: 1, historyVersion: 0 };
      });
      const pageFiles = new Map(entry.pages.map((p: any) => [p.id, p]));
      const material = materials.get(entry.materialIndex);
      const previewMaterial = settings.anisotropy ? { ...material, samplers: material.textures?.map((_, slot) => {
        const sampler = material.samplers?.[slot];
        const allLinear = [sampler?.magFilter ?? 'linear', sampler?.minFilter ?? 'linear', sampler?.mipmapFilter ?? 'linear'].every(filter => filter === 'linear');
        return allLinear ? { ...sampler, maxAnisotropy: settings.anisotropy } : sampler;
      }) } : material;
      batches.push({ name: `sponza:${entry.meshIndex}:${entry.primitiveIndex}`, capacity: instances.length, instances,
        inputMode: 'committed', deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
        material: previewMaterial, ...(typeof span === 'number' ? { textureUvSpan: span } : {}),
        queryIdentity: { assetId: manifest.assetIdentity, primitiveId: batches.length, topologyVersion: 0 },
        geometryStream: { directory, queryProxy,
          ...(settings.queryRoot ? { queryProxyLimits: { maxVertices: 1048576, maxTriangles: 262144, maxBytes: 128 * MiB } } : {}),
          provider: async (page, signal) => { const p: any = pageFiles.get(page.id); return checked(root + p.file, page.encodedBytes, page.sha256, signal); },
          cpuCache: { maxResidentBytes: rootDecoded + 2 * largestPage, maxInFlightBytes: 2 * largestWork, maxConcurrentRequests: 2 },
          gpuPool: { maxResidentBytes: gpuBytes, maxStagingBytes: 2 * slotBytes + 65536,
            maxUploadBytesPerFrame: 256 * 1024, maxUploadPagesPerFrame: 1, maxQueuedPages: 8 }, maxRequestsPerFrame: 2 } });
      sourceTriangles += entry.triangleCount * instances.length;
      accounting.push({ mesh: entry.meshIndex, primitive: entry.primitiveIndex, instances: instances.length, slots, pages: directory.pages.length, gpuBytes });
      if (batches.length % 50 === 0) progress('geometry-metadata', { batches: batches.length });
    }
    if (batches.length !== 401 || omitted.length !== 4) throw Error('SPONZA_CONTENT_COVERAGE');
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
      ...(settings.finalGather ? { finalGather: { pixelSpacing: settings.fineGather ? 4 : 8, maxAnchors: settings.fineGather ? 65536 : 32768, raysPerAnchor: 1,
        pathSegments: 2, maxHistory: settings.fineGather ? 256 : 64, maxBytes: 64 * MiB,
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
      const points = [{x:640,y:620},{x:480,y:560},{x:260,y:400},{x:1020,y:400},{x:640,y:300},{x:900,y:140},
        {x:80,y:475},{x:300,y:260},{x:960,y:265},{x:590,y:185}];
      const captures: any[] = [];
      const capture = async (label: string, count: number) => {
        const signatureBefore = residencySignature(), hdrFrames = [];
        const firstFrame = renderedFrame + 1;
        for (let i = 0; i < count; i++) { renderer!.render(eye, target); renderedFrame++; hdrFrames.push(await renderer!.probeHdr(points)); }
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
          return { pixel, primary, recovery, indices: [...primary, ...recovery] };
        }) : [];
        const finalGather = candidates.length ? { candidates,
          probe: await renderer!.probeDiffuseGiFinalGather([...new Set(candidates.flatMap(row => row.indices))]),
          note: 'Candidate anchors only; inclusion does not prove surface compatibility or acceptance by reconstruction.' } : null;
        const signatureAfter = residencySignature();
        const result = { label, firstFrame, lastFrame: renderedFrame, renderedFrames: count,
          png: canvas.toDataURL(), points, hdrFrames, probe, finalGather,
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
