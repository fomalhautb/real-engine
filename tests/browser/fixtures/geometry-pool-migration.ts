import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { decodeGeometryPage, encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { GeometryStreamWorkingSet } from '../../../src/geometry-stream-working-set';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import { GpuGeometryPagePool } from '../../../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../../../src/gpu-geometry-stream-selection';
import { identity, lookAt, multiply, perspective } from '../../../src/math';

const shader = `
struct Vertex { position:vec4f, normal:vec4f, joints:vec4u, weights:vec4f, morph:vec4f,
  uv:vec2f, flags:u32, padding:u32, tangent:vec4f, identity:vec4u, color:vec4f }
@group(0) @binding(0) var<uniform> vp:mat4x4f;
@group(0) @binding(1) var<storage,read> vertices:array<Vertex>;
struct Varyings { @builtin(position) clip:vec4f, @location(0) normal:vec3f,
  @location(1) @interpolate(flat) seed:u32 }
@vertex fn vertexMain(@builtin(vertex_index) id:u32)->Varyings {
  let v=vertices[id]; var out:Varyings; out.clip=vp*vec4f(v.position.xyz,1);
  out.normal=v.normal.xyz; out.seed=v.identity.w; return out;
}
@fragment fn fragmentMain(in:Varyings)->@location(0) vec4f {
  var seed=in.seed^(in.seed>>16u); seed*=0x7feb352du; seed^=seed>>15u;
  seed*=0x846ca68bu; seed^=seed>>16u;
  let color=vec3f(0.14)+vec3f(f32(seed&255u),f32((seed>>8u)&255u),f32((seed>>16u)&255u))/255.0*0.8;
  let light=0.4+0.6*max(0.0,dot(normalize(in.normal),normalize(vec3f(0.4,0.8,0.6))));
  return vec4f(sqrt(color*light),1);
}`;

/** Exercises real copies, new selection bindings and indirect draws. This does
 * not integrate migration into Renderer or choose a scene-wide runtime budget. */
export async function runGeometryPoolMigrationFixture() {
  const scene = await GeometryDemoScene.create();
  const encoded = await encodeGeometryPages(scene.hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(scene.hierarchy, encoded);
  const decoded = await Promise.all(encoded.pages.map(page => decodeGeometryPage(page.bytes, { maxPageBytes: 49152 })));
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('MIGRATION_FIXTURE_NO_ADAPTER');
  const device = await adapter.requestDevice(), errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const pools: GpuGeometryPagePool[] = [], selections: GpuGeometryStreamSelection[] = [];
  const buffers: GPUBuffer[] = [], textures: GPUTexture[] = [];
  let poolOwned = 0, poolPeak = 0, selectionOwned = 0;
  const trackPool = (bytes: number) => { poolOwned += bytes; poolPeak = Math.max(poolPeak, poolOwned); };
  const allocate = (size: number, usage: GPUBufferUsageFlags, data?: Float32Array<ArrayBuffer>) => {
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: !!data }); buffers.push(buffer);
    if (data) { new Float32Array(buffer.getMappedRange()).set(data); buffer.unmap(); } return buffer;
  };
  const width = 256, eye: [number, number, number] = [3.8, 3.2, 5];
  const vpData = multiply(perspective(Math.PI / 3.2, 1, 0.1, 30), lookAt(eye, [0, 0, 0], [0, 1, 0]));
  const view = { eye, near: 0.1, projectionScale: width / (2 * Math.tan(Math.PI / 6.4)), maxErrorPixels: 0, viewProjection: vpData };
  const layout = geometryGpuPoolLayout(directory), planner = new GeometryStreamCut(directory);
  const initialBudget = layout.metadataBytes + (directory.rootPages.length + 2) * layout.bytesPerSlot;
  const uploadFrames = new Map<GpuGeometryPagePool, number>();
  const upload = (pool: GpuGeometryPagePool, ids: readonly number[]) => {
    for (const id of ids) pool.enqueue(id, decoded[id]);
    for (let step = 0; step < 2048; step++) {
      if (ids.every(id => pool.residentPageIds().has(id))) return;
      const frame = (uploadFrames.get(pool) ?? 0) + 1; uploadFrames.set(pool, frame);
      const result = pool.uploadFrame(frame);
      if (result.blocked && !result.bytes) throw new Error('MIGRATION_FIXTURE_UPLOAD_BLOCKED');
    }
    throw new Error('MIGRATION_FIXTURE_UPLOAD_LIMIT');
  };
  const panel = document.createElement('section'); panel.id = 'geometry-pool-migration-visual';
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#11171d;color:#eaf1f6;padding:24px;font:16px system-ui;overflow:auto';
  panel.innerHTML = '<h1>GPU 几何池容量迁移</h1><p>保持旧池可用，GPU 内复制后切换新池；本测试独立验证迁移，尚未接入全场景动态预算。</p>';
  const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:16px'; panel.appendChild(row); document.body.appendChild(panel);
  try {
    const vp = allocate(64, GPUBufferUsage.UNIFORM, new Float32Array(vpData));
    const instanceData = new Float32Array(56);
    for (const base of [0, 16, 32]) instanceData.set(identity(), base);
    const instances = allocate(instanceData.byteLength, GPUBufferUsage.STORAGE, instanceData);
    const output = device.createTexture({ size: [width, width], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [width, width], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    textures.push(output, depth);
    const module = device.createShaderModule({ code: shader });
    const compilation = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (compilation.length) throw new Error(compilation.map(message => message.message).join('\n'));
    const pipeline = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module, entryPoint: 'vertexMain' },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] }, primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' } });
    const capture = async (pool: GpuGeometryPagePool, label: string) => {
      const selection = await GpuGeometryStreamSelection.create(device, pool, instances, 1, bytes => { selectionOwned += bytes; });
      selections.push(selection); selection.setSettings({ maxErrorPixels: 0, frustumCulling: false });
      const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [vp, pool.vertices]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const readback = device.createBuffer({ size: width * width * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const lease = pool.acquireFrame([...pool.residentPageIds()]);
        const encoder = device.createCommandEncoder(); selection.update(1, 1, vpData, eye, view.projectionScale); selection.encode(encoder);
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: output.createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
          depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
        pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.setIndexBuffer(selection.indices, 'uint32');
        pass.drawIndexedIndirect(selection.arguments, 0); pass.end();
        encoder.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow: width * 4 }, [width, width]);
        const completion = await pool.submitFrame(lease, [encoder.finish()]); if (completion.error) throw new Error(completion.error);
        await readback.mapAsync(GPUMapMode.READ); const pixels = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap();
        const feedback = await selection.feedback(true), reference = planner.plan(pool.residentPageIds(), view);
        const triangles = reference.selectedClusters.reduce((sum, id) => sum + directory.clusters[id].triangleCount, 0);
        if (feedback.overflow || feedback.indices !== triangles * 3) throw new Error('MIGRATION_FIXTURE_CUT_MISMATCH');
        const frame = document.createElement('div'), caption = document.createElement('p'), canvas = document.createElement('canvas');
        canvas.width = width; canvas.height = width; caption.textContent = `${label} · ${triangles} 三角形`;
        canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(pixels), width, width), 0, 0);
        frame.append(caption, canvas); row.appendChild(frame);
        const foreground = pixels.reduce((count, channel, index) => count + Number(index % 4 === 0 && channel > 0), 0);
        return { pixels, triangles, foreground, selected: feedback.selectedClusters?.filter(cluster => cluster.visible).map(cluster => cluster.clusterId) };
      } finally { readback.destroy(); selection.dispose(); }
    };
    const source = await GpuGeometryPagePool.create(device, directory, { maxResidentBytes: initialBudget,
      maxUploadBytesPerFrame: 1024 * 1024, maxQueuedPages: 4096, maxStagingBytes: 64 * 1024 * 1024 }, trackPool);
    pools.push(source);
    const intent = new GeometryStreamWorkingSet(directory, source.slotCount).plan(planner.demands([view]));
    upload(source, intent.pages); const before = await capture(source, '迁移前');
    const grown = await source.migrateResident({ maxResidentBytes: layout.fullBytes,
      maxPeakBytes: initialBudget + layout.fullBytes, maxCopyBytes: 256 * 1024 * 1024, maxCopyPages: 4096 });
    pools.push(grown.pool);
    const afterCopy = await capture(grown.pool, '保留原始页后');
    const unchangedBytes = before.pixels.reduce((count, value, index) => count + Number(value !== afterCopy.pixels[index]), 0);
    const sourceStillResident = [...source.residentPageIds()].sort((a, b) => a - b);
    source.dispose();
    upload(grown.pool, directory.pages.map(page => page.id)); const detailed = await capture(grown.pool, '新容量装入细节');
    const shrunk = await grown.pool.migrateResident({ maxResidentBytes: layout.rootBytes,
      maxPeakBytes: layout.fullBytes + layout.rootBytes, retainedPages: [], maxCopyBytes: 256 * 1024 * 1024, maxCopyPages: 4096 });
    pools.push(shrunk.pool); grown.pool.dispose(); const fallback = await capture(shrunk.pool, '缩容保留根簇');
    const summary = { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, isFallbackAdapter: adapter.info.isFallbackAdapter },
      initialBudget, fullBudget: layout.fullBytes, rootBudget: layout.rootBytes, poolPeak, unchangedBytes,
      sourceStillResident, initiallyRetained: intent.pages, copiedPages: grown.copiedPages, copiedBytes: grown.copiedBytes,
      stages: [before, afterCopy, detailed, fallback].map(({ triangles, foreground, selected }) => ({ triangles, foreground, selected })),
      shrinkRetained: [...shrunk.pool.residentPageIds()].sort((a, b) => a - b), roots: [...directory.rootPages].sort((a, b) => a - b), errors };
    for (const pool of pools) pool.dispose();
    return { ...summary, poolOwnedAfterDispose: poolOwned, selectionOwnedAfterDispose: selectionOwned };
  } finally {
    for (const selection of selections) selection.dispose(); for (const pool of pools) pool.dispose();
    for (const buffer of buffers) buffer.destroy(); for (const texture of textures) texture.destroy(); device.destroy();
  }
}
