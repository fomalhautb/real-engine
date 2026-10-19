import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { Renderer, type RendererOptions } from '../../../src/renderer';
import type { MaterialBinding, SceneBatch } from '../../../src/contracts';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { multiply, normalMatrix, scaling, transformPoint, translation } from '../../../src/math';
import type { WorldRay } from '../../../src/world-query';

export async function runGeometryStreamRendererFixture() {
  const scene = await GeometryDemoScene.create(), mesh = scene.hierarchy.source;
  const source = new DataView(mesh.vertices), uvs = new Float32Array(mesh.vertexCount * 2), tangents = new Float32Array(mesh.vertexCount * 4);
  for (let vertex = 0; vertex < mesh.vertexCount; vertex++) {
    const at = vertex * 80;
    const n = [source.getFloat32(at + 16, true), source.getFloat32(at + 20, true), source.getFloat32(at + 24, true)];
    const axis = Math.abs(n[1]) < 0.95 ? [0, 1, 0] : [1, 0, 0];
    const t = [axis[1] * n[2] - axis[2] * n[1], axis[2] * n[0] - axis[0] * n[2], axis[0] * n[1] - axis[1] * n[0]];
    const length = Math.hypot(...t); tangents.set([...t.map(value => value / length), 1], vertex * 4);
    uvs.set([source.getFloat32(at, true) * 0.3 + 0.5, source.getFloat32(at + 8, true) * 0.3 + 0.5], vertex * 2);
  }
  const attributed = { ...mesh, uvs, tangents };
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'stream-renderer-attributed-sculpture', mesh: attributed, coverage: 'opaque' },
    { maxClusterTriangles: 96, simplifyRatio: 0.5, maxLevels: 8 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const material: MaterialBinding = { definition: { ...DEFAULT_MATERIAL_DEFINITION,
    baseColor: m.multiply(m.texture(0), m.instanceColor), normal: m.texture(1), normalScale: m.component(m.parameter(0), 0) },
    parameters: new Float32Array([0.8, 0, 0, 0]), textures: [
      { width: 2, height: 2, colorSpace: 'srgb', pixels: new Uint8Array([210, 225, 235, 255, 95, 130, 150, 0, 95, 130, 150, 0, 210, 225, 235, 255]) },
      { width: 2, height: 1, colorSpace: 'linear', pixels: new Uint8Array([166, 128, 248, 255, 100, 155, 250, 255]) },
    ], samplers: [{ addressModeU: 'repeat', addressModeV: 'repeat' }, { addressModeU: 'repeat', addressModeV: 'repeat' }] };
  let detailsEnabled = false;
  const waiting = new Set<() => void>();
  const requested = new Set<number>();
  const provider = async (page: { id: number }, signal: AbortSignal) => {
    requested.add(page.id);
    if (!detailsEnabled && !directory.rootPages.includes(page.id)) await new Promise<void>((resolve, reject) => {
      const finish = () => { signal.removeEventListener('abort', abort); waiting.delete(finish); resolve(); };
      const abort = () => { waiting.delete(finish); reject(new DOMException('Aborted', 'AbortError')); };
      waiting.add(finish); signal.addEventListener('abort', abort, { once: true }); if (signal.aborted) abort();
    });
    return encoded.pages[page.id].bytes;
  };
  const sculpture = scene.batches[0];
  const makeBatches = (streamed: boolean): SceneBatch[] => [0, 1].map<SceneBatch>((part) => ({ ...sculpture,
    name: 'stream-sculpture-' + part, capacity: part ? 2 : 1, mesh: hierarchy.source, material,
    instances: (part ? sculpture.instances.slice(1) : sculpture.instances.slice(0, 1)).map(state => ({ ...state })),
    geometryLod: streamed ? undefined : hierarchy,
    geometryStream: streamed ? { directory, provider, gpuPool: { maxUploadBytesPerFrame: 16384, maxUploadWritesPerFrame: 16, maxUploadPagesPerFrame: 2 } } : undefined,
  })).concat(scene.batches.slice(1).map(batch => ({ ...batch, instances: batch.instances.map(state => ({ ...state })) })));
  const referenceBatches = makeBatches(false), streamBatches = makeBatches(true);
  const panel = document.createElement('section'); panel.id = 'stream-renderer-visual';
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#11171d;color:#eaf1f6;padding:28px;font:16px system-ui;overflow:auto';
  panel.innerHTML = '<h1 style="font-size:27px;margin:0 0 10px">主渲染器：分页几何 × 材质 × 阴影 × 世界 GI</h1><p>右侧主视图由真实压缩页、GPU 选簇与间接绘制生成；阴影和查询暂时保留完整源网格。</p>';
  const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:20px'; panel.appendChild(row); document.body.appendChild(panel);
  const canvases = ['完整驻留参考', '分页主视图'].map(label => {
    const section = document.createElement('div'); section.style.cssText = 'flex:1;min-width:0';
    const title = document.createElement('h2'); title.textContent = label; title.style.fontSize = '19px';
    const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:600px;height:480px;max-width:100%;display:block;border-radius:8px';
    section.append(title, canvas); row.appendChild(section); return canvas;
  });
  const options: RendererOptions = { worldQueries: { maxRays: 256 }, sunShadow: { resolution: 512,
    focusBounds: { min: [-9, -2, -5], max: [9, 4, 5] } },
    diffuseGi: { origin: [-8, 0, -4], spacing: [16, 4, 8], counts: [2, 2, 2], raysPerProbe: 16,
      maxDistance: 30, skyRadiance: [0.12, 0.18, 0.25] } };
  let reference: Renderer | undefined, streamed: Renderer | undefined;
  try {
    reference = await Renderer.create(canvases[0], referenceBatches, options);
    streamed = await Renderer.create(canvases[1], streamBatches, options);
    const a = reference, b = streamed;
    a.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false }); b.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    a.setDiffuseGiEnabled(false); b.setDiffuseGiEnabled(false);
    const eye: [number, number, number] = [8, 9, 16], target: [number, number, number] = [0, -0.5, 0];
    const tick = async () => {
      a.render(eye, target); b.render(eye, target);
      await Promise.all([a.probeHdr([{ x: 0, y: 0 }]), b.probeHdr([{ x: 0, y: 0 }])]);
      if (a.errors.length || b.errors.length) throw new Error([...a.errors, ...b.errors].join('\n'));
    };
    await tick();
    const samples = Array.from({ length: 768 }, (_, id) => ({ x: Math.floor((id % 32 + 0.5) * a.stats.width / 32),
      y: Math.floor((Math.floor(id / 32) + 0.5) * a.stats.height / 24) }));
    const compare = async () => {
      const [expected, actual] = await Promise.all([a.probeHdr(samples), b.probeHdr(samples)]);
      let maxError = 0, totalError = 0, foreground = 0;
      actual.forEach((pixel, id) => { if (pixel.slice(0, 3).some(value => value > 0.015)) foreground++;
        pixel.slice(0, 3).forEach((value, channel) => { const delta = Math.abs(value - expected[id][channel]); maxError = Math.max(maxError, delta); totalError += delta; }); });
      return { maxError, meanError: totalError / (samples.length * 3), foreground };
    };
    const probes = async () => Promise.all([b.probeGeometryLod('stream-sculpture-0'), b.probeGeometryLod('stream-sculpture-1')]);
    const coarse = await probes();
    const rayOrigins = sculpture.instances.map(state => transformPoint(state.model, [1.4, 5, 0]));
    const rays: WorldRay[] = rayOrigins.map(origin => ({ origin, direction: [0, -1, 0], tMax: 15 }));
    const queryCoarse = await b.traceRays(rays);
    const rootsOnly = b.diagnostics().geometryStreams.map(stream => stream.pool.residentPages);
    detailsEnabled = true; for (const release of [...waiting]) release();
    let loadedFrames = 0;
    for (; loadedFrames < 240; loadedFrames++) {
      await tick();
      const cuts = await probes();
      if (cuts.reduce((sum, cut) => sum + cut.triangles, 0) === hierarchy.sourceTriangleCount * 3) break;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (loadedFrames === 240) throw new Error('RENDERER_STREAM_DID_NOT_REFINE: ' + JSON.stringify(b.diagnostics().geometryStreams));
    const detailed = await probes(), opaque = await compare();
    const queryDetailed = await b.traceRays(rays), queryReference = await a.traceRays(rays);
    const depthPoints: [number, number, number][] = [[-4, -1.34, 0], [0, -1.34, 0], [4, -1.34, 0]];
    const depth = { reference: await a.probeDepth(depthPoints), streamed: await b.probeDepth(depthPoints) };
    const variants = b.diagnostics().materialVariants;
    for (const renderer of [a, b]) for (const name of ['stream-sculpture-0', 'stream-sculpture-1']) renderer.setMaterialParameters(name, new Float32Array([0, 0, 0, 0]));
    await tick(); const normalUpdate = await compare();
    const variantsAfterParameters = b.diagnostics().materialVariants;
    for (const batches of [referenceBatches, streamBatches]) {
      const state = batches[1].instances[0], model = multiply(translation(0.25, 0.1, 0), multiply(state.model, scaling(1.1, 0.9, 1)));
      batches[1].instances[0] = { ...state, model, normalMatrix: normalMatrix(model) };
    }
    await tick(); const moving = await compare();
    const velocity = { reference: await a.probeVelocity(samples), streamed: await b.probeVelocity(samples) };
    const masked = { ...material, definition: { ...material.definition, alphaMode: 'mask' as const, alphaCutoff: 0.5 } };
    for (const renderer of [a, b]) for (const name of ['stream-sculpture-0', 'stream-sculpture-1']) await renderer.setMaterial(name, masked);
    await tick(); const mask = await compare(), maskCuts = await probes();
    const maskQuery = { reference: await a.traceRays(rays), streamed: await b.traceRays(rays) };
    for (const renderer of [a, b]) for (const name of ['stream-sculpture-0', 'stream-sculpture-1']) await renderer.setMaterial(name, material);
    a.setDiffuseGiEnabled(true); b.setDiffuseGiEnabled(true);
    for (let frame = 0; frame < 4; frame++) await tick();
    const gi = await compare();
    const diagnostics = b.diagnostics(), adapter = b.info.adapterInfo;
    const footer = document.createElement('p'); footer.textContent = `${directory.pages.length} 页/资产 · 两个独立页池同帧提交 · PBR/UV/法线贴图、刚体变换、阴影、世界查询、Masked 回退和基础 GI 对照`; panel.appendChild(footer);
    // Preserve actual rendered pixels before destroying the owned WebGPU contexts.
    const visiblePixels: number[] = [];
    for (const [index, canvas] of canvases.entries()) {
      [a, b][index].render(eye, target);
      // Capture synchronously in the render task, before WebGPU presents and
      // expires the current swapchain texture at the next task boundary.
      const png = canvas.toDataURL('image/png'), saved = document.createElement('img');
      saved.src = png; saved.style.cssText = canvas.style.cssText; await saved.decode();
      const check = document.createElement('canvas'); check.width = canvas.width; check.height = canvas.height;
      const context = check.getContext('2d')!; context.drawImage(saved, 0, 0);
      const pixels = context.getImageData(0, 0, check.width, check.height).data;
      let count = 0; for (let p = 0; p < pixels.length; p += 4) if (pixels[p + 3] && pixels[p] + pixels[p + 1] + pixels[p + 2] > 80) count++;
      visiblePixels.push(count); canvas.replaceWith(saved);
    }
    a.dispose(); b.dispose();
    return { adapter, pages: directory.pages.length, rootPages: [...directory.rootPages], sourceTriangles: hierarchy.sourceTriangleCount,
      rootsOnly, requestedPages: [...requested], loadedFrames, coarse, detailed, opaque, normalUpdate, moving, mask, gi,
      variants, variantsAfterParameters, queryCoarse, queryDetailed, queryReference, depth, velocity, maskCuts, maskQuery,
      diagnostics, decoderAfterDispose: b.diagnostics().geometryDecoding, visiblePixels,
      errors: [...a.errors, ...b.errors], ownedAfterDispose: [a.stats.ownedGpuBytes, b.stats.ownedGpuBytes] };
  } finally { reference?.dispose(); streamed?.dispose(); }
}
