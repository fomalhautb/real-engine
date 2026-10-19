import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { buildGeometryQueryProxy, decodeGeometryQueryProxy, encodeGeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { Renderer, type RendererOptions } from '../../../src/renderer';
import type { MaterialBinding, RenderBatch, SceneBatch, StreamedSceneBatch } from '../../../src/contracts';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { multiply, normalMatrix, scaling, transformPoint, translation } from '../../../src/math';
import type { WorldHit, WorldRay } from '../../../src/world-query';

/** Three independent Renderers: legacy streamed source, source-free streamed
 * surface, and resident proxy reference. Test-only source data is held for the
 * reference; it is never part of the source-free RenderBatch/provider closure. */
export async function runGeometryQueryProxyFixture() {
  const scene = await GeometryDemoScene.create(), mesh = scene.hierarchy.source;
  const data = new DataView(mesh.vertices), uvs = new Float32Array(mesh.vertexCount * 2), tangents = new Float32Array(mesh.vertexCount * 4);
  for (let v = 0; v < mesh.vertexCount; v++) {
    const n = [0, 1, 2].map(a => data.getFloat32(v * 80 + 16 + a * 4, true));
    const axis = Math.abs(n[1]) < 0.95 ? [0, 1, 0] : [1, 0, 0];
    const t = [axis[1] * n[2] - axis[2] * n[1], axis[2] * n[0] - axis[0] * n[2], axis[0] * n[1] - axis[1] * n[0]];
    const length = Math.hypot(...t); tangents.set([...t.map(x => x / length), 1], v * 4);
    uvs.set([data.getFloat32(v * 80, true) * 0.3 + 0.5, data.getFloat32(v * 80 + 8, true) * 0.3 + 0.5], v * 2);
  }
  const attributed = { ...mesh, uvs, tangents };
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'fixed-proxy-attributed', mesh: attributed, coverage: 'opaque' },
    { maxClusterTriangles: 96, simplifyRatio: 0.5, maxLevels: 8 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const pageBytes = encoded.pages.map(page => page.bytes);
  const provider = async (page: { id: number }, signal: AbortSignal) => { signal.throwIfAborted(); return pageBytes[page.id]; };
  const built = await buildGeometryQueryProxy(directory, provider, { topologyVersion: 7 });
  const binary = encodeGeometryQueryProxy(built);
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', binary)), x => x.toString(16).padStart(2, '0')).join('');
  const proxy = await decodeGeometryQueryProxy(directory, binary, digest);
  const material: MaterialBinding = { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
    baseColor: m.multiply(m.texture(0), m.instanceColor), normal: m.texture(1), normalScale: m.component(m.parameter(0), 0),
    emissive: m.constant([0.12, 0.08, 0.04, 1]) }, parameters: new Float32Array([0.8, 0, 0, 0]),
    textures: [
      { width: 2, height: 2, colorSpace: 'srgb', pixels: new Uint8Array([210, 225, 235, 255, 95, 130, 150, 255, 95, 130, 150, 255, 210, 225, 235, 255]) },
      { width: 2, height: 1, colorSpace: 'linear', pixels: new Uint8Array([166, 128, 248, 255, 100, 155, 250, 255]) },
    ], samplers: [{ addressModeU: 'repeat', addressModeV: 'repeat' }, { addressModeU: 'repeat', addressModeV: 'repeat' }] };
  const { mesh: _source, geometryLod: _lod, geometryStream: _stream, ...common } = scene.batches[0];
  // Capacity above the instance-BVH threshold also exercises inactive leaves.
  const base = { ...common, capacity: 8, material, queryIdentity: { assetId: directory.primitiveKey, primitiveId: 0, topologyVersion: 3 },
    emissiveSampling: 'triangle-area' as const };
  const streamedSource = { directory, provider, gpuPool: { maxUploadBytesPerFrame: 16384, maxUploadWritesPerFrame: 16 } };
  const legacy: SceneBatch = { ...base, mesh: attributed, geometryStream: streamedSource,
    instances: base.instances.map(state => ({ ...state })) };
  const paged: StreamedSceneBatch = { ...base, geometryStream: { ...streamedSource, queryProxy: proxy },
    instances: base.instances.map(state => ({ ...state })) };
  const fixed: SceneBatch = { ...base, mesh: proxy.mesh, instances: base.instances.map(state => ({ ...state })) };
  const batchSets: RenderBatch[][] = [legacy, paged, fixed].map(first => [first,
    ...scene.batches.slice(1).map(batch => ({ ...batch, instances: batch.instances.map(state => ({ ...state })),
      material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const } } }))]);
  // Register empty slots to exercise scene-tree invalid leaves alongside the
  // real paged proxy. These slots add no visible or lighting geometry.
  for (const batches of batchSets) while (batches.length < 8) batches.push({
    ...scene.batches[1], name: `unused-query-batch-${batches.length}`, capacity: 1, instances: [], inputMode: 'committed',
  });
  const panel = document.createElement('section'); panel.id = 'query-proxy-visual';
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#11171d;color:#eaf1f6;padding:24px;font:16px system-ui;overflow:auto';
  panel.innerHTML = '<h1 style="font-size:26px">分页细节与固定光照代理</h1><p>中间：主画面读取压缩页，阴影和 GI 使用固定代理。右侧显示光照实际使用的较粗几何。</p>';
  const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:16px'; panel.appendChild(row); document.body.appendChild(panel);
  const canvases = ['完整源网格参与光照', '分页画面＋固定代理', '固定代理的几何形状'].map(label => {
    const card = document.createElement('div'); card.style.cssText = 'flex:1;min-width:0';
    const title = document.createElement('h2'); title.textContent = label; title.style.fontSize = '18px';
    const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:420px;height:380px;max-width:100%;display:block';
    card.append(title, canvas); row.appendChild(card); return canvas;
  });
  const options: RendererOptions = { worldQueries: { maxRays: 1024 }, sunShadow: { resolution: 512,
    focusBounds: { min: [-9, -2, -5], max: [9, 4, 5] } }, diffuseGi: {
      origin: [-8, 0, -4], spacing: [16, 4, 8], counts: [2, 2, 2], raysPerProbe: 32,
      maxDistance: 30, skyRadiance: [0.12, 0.18, 0.25], emissiveSamplesPerProbe: 8,
      finalGather: { pixelSpacing: 32, maxAnchors: 512, raysPerAnchor: 2, maxHistory: 8 } } };
  const renderers: Renderer[] = [];
  const eye: [number, number, number] = [8, 9, 16], target: [number, number, number] = [0, -0.5, 0];
  try {
    for (let i = 0; i < 3; i++) renderers.push(await Renderer.create(canvases[i], batchSets[i], i === 2
      ? { ...options, worldQueries: { ...options.worldQueries, instanceAcceleration: 'linear', sceneAcceleration: 'linear' } } : options));
    const [full, stream, reference] = renderers;
    for (const r of [full, stream]) r.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    for (const r of renderers) r.setDiffuseGiEnabled(false);
    const tick = async () => {
      for (const r of renderers) r.render(eye, target);
      await Promise.all(renderers.map(r => r.probeHdr([{ x: 0, y: 0 }])));
      if (renderers.some(r => r.errors.length)) throw new Error(renderers.flatMap(r => r.errors).join('\n'));
    };
    // Select an upward-facing proxy triangle per affine instance. A vertical
    // ray through an arbitrary source-space point can miss the coarser mesh.
    const proxyRest = new DataView(proxy.mesh.vertices);
    const rays: WorldRay[] = base.instances.map(state => {
      let highest: [number, number, number] = [0, -Infinity, 0];
      for (let t = 0; t < proxy.mesh.indices.length; t += 3) {
        const p = [0, 1, 2].map(c => transformPoint(state.model, [0, 1, 2].map(a =>
          proxyRest.getFloat32(proxy.mesh.indices[t + c] * 80 + a * 4, true))));
        const up = (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]);
        const center = [0, 1, 2].map(a => (p[0][a] + p[1][a] + p[2][a]) / 3) as [number, number, number];
        if (up > 0 && center[1] > highest[1]) highest = center;
      }
      return { origin: [highest[0], highest[1] + 5, highest[2]], direction: [0, -1, 0], tMax: 15 };
    });
    const strip = (hits: (WorldHit | null)[]) => hits.map(hit => {
      if (!hit) return null; const { geometry: _geometry, ...value } = hit; return value;
    });
    await tick(); const coarse = await stream.probeGeometryLod(base.name);
    const queryCoarse = await stream.traceRays(rays);
    let loadedFrames = 0;
    for (; loadedFrames < 240; loadedFrames++) {
      await tick();
      const cuts = await Promise.all([full, stream].map(r => r.probeGeometryLod(base.name)));
      if (cuts.every(cut => cut.triangles === hierarchy.sourceTriangleCount * 3)) break;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (loadedFrames === 240) throw new Error('PROXY_SURFACE_DID_NOT_REFINE');
    const samples = Array.from({ length: 768 }, (_, id) => ({ x: Math.floor((id % 32 + 0.5) * stream.stats.width / 32),
      y: Math.floor((Math.floor(id / 32) + 0.5) * stream.stats.height / 24) }));
    const hdr = await Promise.all([full, stream].map(r => r.probeHdr(samples)));
    let surfaceMaxError = 0, foreground = 0;
    hdr[1].forEach((pixel, id) => { if (pixel.slice(0, 3).some(v => v > 0.02)) foreground++;
      pixel.slice(0, 3).forEach((v, channel) => { surfaceMaxError = Math.max(surfaceMaxError, Math.abs(v - hdr[0][id][channel])); }); });
    const queryDetailed = await stream.traceRays(rays), queryReference = await reference.traceRays(rays);
    const sameQuery = JSON.stringify(strip(queryDetailed.hits)) === JSON.stringify(strip(queryReference.hits));
    const points: [number, number, number][] = Array.from({ length: 99 }, (_, id) => [-5 + id % 11, -1.34, -4 + Math.floor(id / 11)]);
    const depth = await Promise.all([stream, reference].map(r => r.probeDepth(points)));
    const shadowEqual = depth[0].every((value, i) => value.shadow === depth[1][i].shadow);
    const ledger = { full: full.stats.ownedGpuBytes, proxy: stream.stats.ownedGpuBytes };
    const beforeRevision = stream.diagnostics().materialRevision;
    let maskError = '';
    try { await stream.setMaterial(base.name, { ...material, definition: { ...material.definition, alphaMode: 'mask' } }); }
    catch (error) { maskError = String(error); }
    const afterRevision = stream.diagnostics().materialRevision;
    const savedSource = paged.geometryStream;
    paged.geometryStream = { ...savedSource, queryProxy: { ...proxy, topologyVersion: 8 } };
    let replacementError = '';
    try { stream.render(eye, target); } catch (error) { replacementError = String(error); }
    paged.geometryStream = savedSource;
    for (const batches of batchSets) {
      const state = batches[0].instances[1], model = multiply(translation(0.2, 0.1, 0), multiply(state.model, scaling(1.1, 0.9, 1)));
      batches[0].instances[1] = { ...state, model, normalMatrix: normalMatrix(model) };
    }
    await tick();
    const velocity = await Promise.all([full, stream].map(r => r.probeVelocity(samples)));
    const movingQuery = await Promise.all([stream, reference].map(r => r.traceRays(rays)));
    const movingQueryEqual = JSON.stringify(strip(movingQuery[0].hits)) === JSON.stringify(strip(movingQuery[1].hits));
    for (const r of renderers) {
      await r.setMaterial(base.name, { ...material, definition: { ...material.definition, shading: 'pbr' } });
      for (const batch of scene.batches.slice(1)) await r.setMaterial(batch.name, { definition: DEFAULT_MATERIAL_DEFINITION });
      r.setDiffuseGiEnabled(true);
    }
    for (let frame = 0; frame < 12; frame++) await tick();
    const gi = await Promise.all([stream, reference].map(r => r.probeDiffuseGi()));
    const anchorCount = stream.diagnostics().diffuseGi!.finalGather!.anchorsPerUpdate;
    const gather = await Promise.all([stream, reference].map(r => r.probeDiffuseGiFinalGather(Array.from({ length: anchorCount }, (_, id) => id))));
    const diagnostics = stream.diagnostics();
    const footer = document.createElement('p'); footer.textContent = `每资产 ${mesh.vertexCount} → ${proxy.mesh.vertexCount} 顶点，${hierarchy.sourceTriangleCount} → ${proxy.mesh.indices.length / 3} 个光照代理三角形；主视图细节仍为 ${hierarchy.sourceTriangleCount} 三角形。代理误差是累计简化指标，不是阴影或光照误差保证。`;
    panel.appendChild(footer);
    const visiblePixels: number[] = [];
    for (let i = 0; i < 3; i++) {
      renderers[i].render(eye, target);
      const image = document.createElement('img'); image.src = canvases[i].toDataURL('image/png'); image.style.cssText = canvases[i].style.cssText;
      await image.decode();
      const check = document.createElement('canvas'); check.width = canvases[i].width; check.height = canvases[i].height;
      const context = check.getContext('2d')!; context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, check.width, check.height).data;
      let count = 0; for (let at = 0; at < pixels.length; at += 4) if (pixels[at + 3] && pixels[at] + pixels[at + 1] + pixels[at + 2] > 80) count++;
      visiblePixels.push(count); canvases[i].replaceWith(image);
    }
    const adapter = stream.info.adapterInfo;
    for (const r of renderers) r.dispose();
    return { adapter, sourceVertices: mesh.vertexCount, sourceTriangles: hierarchy.sourceTriangleCount,
      proxyVertices: proxy.mesh.vertexCount, proxyTriangles: proxy.mesh.indices.length / 3, proxyError: proxy.approximateObjectSpaceError,
      noSourceMesh: !('mesh' in paged), coarseTriangles: coarse.triangles, loadedFrames, surfaceMaxError, foreground,
      queryCoarse, queryDetailed, sameQuery, shadowEqual, ledger, maskError, beforeRevision, afterRevision, replacementError,
      velocityEqual: JSON.stringify(velocity[0]) === JSON.stringify(velocity[1]),
      velocityNonzero: velocity[1].some(v => Math.abs(v[0]) + Math.abs(v[1]) > 0.0001), movingQueryEqual,
      giEqual: JSON.stringify(gi[0].probes) === JSON.stringify(gi[1].probes), giInvalid: gi.map(value => value.invalidProbes),
      gatherEqual: JSON.stringify(gather[0].anchors) === JSON.stringify(gather[1].anchors),
      gatherValid: gather[0].anchors.filter(anchor => anchor.valid).length,
      diagnostics, visiblePixels, errors: renderers.flatMap(r => r.errors), afterDispose: renderers.map(r => r.stats.ownedGpuBytes) };
  } finally { for (const r of renderers) r.dispose(); }
}
