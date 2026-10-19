import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryPageCache } from '../../../src/geometry-stream-cache';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { GpuGeometryPagePool } from '../../../src/gpu-geometry-page-pool';
import { GpuGeometryStreamSelection } from '../../../src/gpu-geometry-stream-selection';
import { packGeometryGpuPage, GEOMETRY_GPU_VERTEX_BYTES } from '../../../src/geometry-gpu-page';
import { packGeometryLod } from '../../../src/geometry-lod-data';
import { lookAt, multiply, perspective } from '../../../src/math';

const size = 512;
const eye: [number, number, number] = [3.8, 3.2, 5.0];
const view = { eye, near: 0.1, projectionScale: size / (2 * Math.tan(Math.PI / 6.4)), maxErrorPixels: 0 };
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

/** Native GPU storage/draw diagnostic. Selection is still the CPU reference;
 * this intentionally does not claim Renderer, GI or material integration. */
export async function runGeometryPagePoolFixture(gpuSelection = false) {
  const scene = await GeometryDemoScene.create(), hierarchy = scene.hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('GEOMETRY_POOL_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let owned = 0, peak = 0, pool: GpuGeometryPagePool | undefined;
  let selection: GpuGeometryStreamSelection | undefined, selectionOwned = 0, selectionFrame = 0;
  let cache: GeometryPageCache | undefined;
  const buffers: GPUBuffer[] = [], textures: GPUTexture[] = [];
  const panel = document.createElement('section');
  panel.id = 'geometry-page-pool-visual';
  panel.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#11171d;color:#eaf1f6;padding:36px;font:16px system-ui;overflow:auto';
  panel.innerHTML = '<h1 style="font-size:28px;margin:0 0 10px">GPU 物理页池 · 分簇绘制验证</h1><p>同一资产的压缩页实际上传到有限 GPU 槽位；CPU 参考选簇。本图独立验证页池，主渲染器兼容性另有用例。</p>';
  if (gpuSelection) panel.innerHTML = '<h1 style="font-size:28px;margin:0 0 10px">GPU 驻留判断 · 整组回退 · 间接绘制</h1><p>GPU 按实际页驻留选择层级、生成缺页请求并压紧索引；CPU 完整切分独立核对；主渲染器兼容性另有用例。</p>';
  const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:20px'; panel.appendChild(row); document.body.appendChild(panel);
  const allocate = (label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer => {
    const buffer = device.createBuffer({ label, size, usage, mappedAtCreation: !!data }); buffers.push(buffer);
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data)); buffer.unmap(); } return buffer;
  };
  try {
    cache = await GeometryPageCache.create(directory, async page => encoded.pages[page.id].bytes);
    const verticesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].vertexCount, 0)));
    const indicesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].triangleCount * 3, 0)));
    const slotCount = directory.rootPages.length + 2;
    const budget = slotCount * (verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES + indicesPerSlot * 4) + directory.clusters.length * 32 + directory.pages.length * 4;
    pool = await GpuGeometryPagePool.create(device, directory, { maxResidentBytes: budget,
      maxUploadBytesPerFrame: 8192, maxUploadWritesPerFrame: 8, maxUploadPagesPerFrame: 1 }, bytes => { owned += bytes; peak = Math.max(peak, owned); });
    const currentPool = pool;
    const planner = new GeometryStreamCut(directory);
    let frame = 0, maxUploadBytes = 0, maxUploadWrites = 0;
    const upload = () => {
      const result = currentPool.uploadFrame(++frame);
      maxUploadBytes = Math.max(maxUploadBytes, result.bytes); maxUploadWrites = Math.max(maxUploadWrites, result.writes);
      return result;
    };
    const flush = (ids: number[]) => {
      for (let iteration = 0; iteration < 2048; iteration++) {
        if (ids.every(id => currentPool.residentPageIds().has(id))) return;
        const step = upload(); if (step.blocked && step.bytes === 0) throw new Error('GEOMETRY_POOL_FIXTURE_CAPACITY');
      }
      throw new Error('GEOMETRY_POOL_FIXTURE_UPLOAD_LIMIT');
    };
    const read = async (source: GPUBuffer, offset: number, bytes: number): Promise<Uint8Array<ArrayBuffer>> => {
      const target = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(source, offset, target, 0, bytes);
        device.queue.submit([encoder.finish()]); await target.mapAsync(GPUMapMode.READ);
        return new Uint8Array(target.getMappedRange()).slice();
      } finally { target.destroy(); }
    };
    for (const id of directory.rootPages) pool.enqueue(id, cache.get(id)!);
    const firstUpload = upload();
    const unpublished = pool.residentPageIds().size === 0;
    const unpublishedFlags = Array.from(new Uint32Array((await read(pool.pageResidency, 0, directory.pages.length * 4)).buffer));
    flush([...directory.rootPages]);
    const rootPlan = planner.plan(pool.residentPageIds(), view);
    const triangles = (ids: number[]) => ids.reduce((sum, id) => sum + directory.clusters[id].triangleCount, 0);
    const leaves = directory.pages.map(page => page.id).filter(id => !directory.rootPages.includes(id));
    let best = [leaves[0], leaves[1]], bestTriangles = 0;
    for (let a = 0; a < leaves.length; a++) for (let b = a + 1; b < leaves.length; b++) {
      const test = planner.plan(new Set([...directory.rootPages, leaves[a], leaves[b]]), view);
      const count = triangles(test.selectedClusters);
      if (count > bestTriangles) { best = [leaves[a], leaves[b]]; bestTriangles = count; }
    }

    // Independently assemble a fully resident reference from the original hierarchy,
    // never from the page decoder or GPU page packing/locations.
    const identities = packGeometryLod(hierarchy).vertices;
    const referenceVertices = new ArrayBuffer(identities.length / 4 * GEOMETRY_GPU_VERTEX_BYTES);
    const referenceBytes = new Uint8Array(referenceVertices), referenceView = new DataView(referenceVertices);
    const referenceIndices: number[] = [], referenceDraws: { firstIndex: number; indexCount: number }[] = [];
    const original = new Uint8Array(hierarchy.source.vertices);
    let referenceVertex = 0;
    for (const cluster of hierarchy.clusters) {
      referenceDraws.push({ firstIndex: referenceIndices.length, indexCount: cluster.triangles.length });
      for (let local = 0; local < cluster.vertices.length; local++) {
        const sourceId = cluster.vertices[local], at = (referenceVertex + local) * GEOMETRY_GPU_VERTEX_BYTES;
        referenceBytes.set(original.subarray(sourceId * 80, sourceId * 80 + 80), at);
        for (let word = 0; word < 4; word++) referenceView.setUint32(at + 112 + word * 4, identities[(referenceVertex + local) * 4 + word], true);
        for (let channel = 0; channel < 4; channel++) referenceView.setFloat32(at + 128 + channel * 4, 1, true);
      }
      for (const local of cluster.triangles) referenceIndices.push(referenceVertex + local);
      referenceVertex += cluster.vertices.length;
    }
    const referenceVertexBuffer = allocate('independent full hierarchy reference vertices', referenceVertices.byteLength, GPUBufferUsage.STORAGE, referenceVertices);
    const referenceIndexData = Uint32Array.from(referenceIndices);
    const referenceIndexBuffer = allocate('independent full hierarchy reference indices', referenceIndexData.byteLength, GPUBufferUsage.INDEX, referenceIndexData.buffer);
    const vpData = multiply(perspective(Math.PI / 3.2, 1, 0.1, 30), lookAt(eye, [0, 0, 0], [0, 1, 0]));
    if (gpuSelection) {
      const instanceData = new Float32Array(56);
      for (const base of [0, 16, 32]) for (let axis = 0; axis < 4; axis++) instanceData[base + axis * 5] = 1;
      const instances = allocate('GPU selection instance', instanceData.byteLength, GPUBufferUsage.STORAGE, instanceData.buffer);
      selection = await GpuGeometryStreamSelection.create(device, pool, instances, 1, bytes => { selectionOwned += bytes; });
      selection.setSettings({ maxErrorPixels: 0, frustumCulling: false });
    }
    const vp = allocate('pool test camera', 64, GPUBufferUsage.UNIFORM, new Float32Array(vpData).buffer);
    const module = device.createShaderModule({ code: shader });
    const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (messages.length) throw new Error(messages.map(message => message.message).join('\n'));
    const pipeline = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module, entryPoint: 'vertexMain' },
      fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] }, primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' } });
    const bind = (vertices: GPUBuffer) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: vp } }, { binding: 1, resource: { buffer: vertices } }] });
    const pagedGroup = bind(pool.vertices), referenceGroup = bind(referenceVertexBuffer);
    const output = device.createTexture({ size: [size, size], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [size, size], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    textures.push(output, depth);
    const encodeDraw = (ids: number[], reference: boolean) => {
      const encoder = device.createCommandEncoder();
      if (selection && !reference) {
        selection.update(++selectionFrame, 1, vpData, eye, view.projectionScale); selection.encode(encoder);
      }
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: output.createView(), clearValue: [0.025, 0.035, 0.045, 1], loadOp: 'clear', storeOp: 'store' }],
        depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
      pass.setPipeline(pipeline); pass.setBindGroup(0, reference ? referenceGroup : pagedGroup);
      pass.setIndexBuffer(reference ? referenceIndexBuffer : selection?.indices ?? currentPool.indices, 'uint32');
      if (selection && !reference) pass.drawIndexedIndirect(selection.arguments, 0);
      else for (const id of ids) {
        if (reference) { const loc = referenceDraws[id]; pass.drawIndexed(loc.indexCount, 1, loc.firstIndex); }
        else { const loc = currentPool.clusterLocation(id); if (!loc) throw new Error('GPU_DRAW_PAGE_MISSING'); pass.drawIndexed(loc.indexCount, 1, loc.firstIndex, loc.baseVertex); }
      }
      pass.end(); return encoder.finish();
    };
    const pixels = async () => {
      const buffer = device.createBuffer({ size: size * size * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device.createCommandEncoder(); encoder.copyTextureToBuffer({ texture: output }, { buffer, bytesPerRow: size * 4 }, [size, size]);
        device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ); return new Uint8Array(buffer.getMappedRange()).slice();
      } finally { buffer.destroy(); }
    };
    const capture = async (label: string) => {
      const plan = planner.plan(currentPool.residentPageIds(), view);
      const token = currentPool.acquireFrame(selection ? [...currentPool.residentPageIds()] : plan.selectedPages);
      const completion = await currentPool.submitFrame(token, [encodeDraw(plan.selectedClusters, false)]);
      if (completion.error) throw new Error(completion.error);
      const gpuPlan = selection ? await selection.feedback(true) : undefined;
      if (gpuPlan) {
        if (gpuPlan.overflow || gpuPlan.indices !== triangles(plan.selectedClusters) * 3
          || JSON.stringify(gpuPlan.selectedClusters?.map(cluster => cluster.clusterId)) !== JSON.stringify(plan.selectedClusters)
          || JSON.stringify(gpuPlan.requestedPages) !== JSON.stringify(plan.requestedPages)) throw new Error('GPU_STREAM_SELECTION_REFERENCE_MISMATCH');
      }
      const actual = await pixels();
      device.queue.submit([encodeDraw(plan.selectedClusters, true)]); const expected = await pixels();
      let maxError = 0, differentBytes = 0, foregroundPixels = 0;
      for (let i = 0; i < actual.length; i++) { const error = Math.abs(actual[i] - expected[i]); maxError = Math.max(maxError, error); if (error) differentBytes++; }
      for (let i = 0; i < actual.length; i += 4) if (actual[i] > 30 || actual[i + 1] > 30 || actual[i + 2] > 30) foregroundPixels++;
      const section = document.createElement('div'); section.style.cssText = 'flex:1;min-width:0';
      const canvas = document.createElement('canvas'); canvas.width = size; canvas.height = size; canvas.style.cssText = 'width:100%;height:auto;border-radius:8px';
      canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(actual.buffer), size, size), 0, 0);
      const title = document.createElement('h2'); title.style.fontSize = '18px'; title.textContent = label;
      const note = document.createElement('p'); note.textContent = `${triangles(plan.selectedClusters)} 三角形 · ${plan.selectedClusters.length} 簇 · ${currentPool.residentPageIds().size}/${directory.pages.length} 页驻留`;
      section.append(title, canvas, note); row.appendChild(section);
      return { triangles: triangles(plan.selectedClusters), clusters: plan.selectedClusters.length,
        maxError, differentBytes, foregroundPixels, selectedPages: plan.selectedPages, fallbackGroups: plan.fallbackGroups, gpuPlan };
    };
    const coarse = await capture('根页：完整粗层级');
    for (const id of best) cache.request(id); await cache.idle();
    for (const id of best) pool.enqueue(id, cache.get(id)!); flush(best);
    const detailed = await capture('细节页上传后');
    let pageByteMismatches = 0;
    for (const id of pool.residentPageIds()) {
      const packed = packGeometryGpuPage(directory, id, cache.get(id)!);
      const loc = pool.clusterLocation(directory.pages[id].clusterIds[0])!;
      const actualVertices = await read(pool.vertices, loc.baseVertex * GEOMETRY_GPU_VERTEX_BYTES, packed.vertices.byteLength);
      const actualIndices = await read(pool.indices, loc.firstIndex * 4, packed.indices.byteLength);
      for (const [actual, expected] of [[actualVertices, new Uint8Array(packed.vertices)], [actualIndices, new Uint8Array(packed.indices.buffer)]]) {
        for (let byte = 0; byte < actual.length; byte++) if (actual[byte] !== expected[byte]) pageByteMismatches++;
      }
    }
    const beforeEviction = pool.diagnostics();
    const incoming = leaves.find(id => !best.includes(id))!; cache.request(incoming); await cache.idle(); pool.enqueue(incoming, cache.get(incoming)!);
    const plan = planner.plan(pool.residentPageIds(), view);
    const held = pool.acquireFrame([...pool.residentPageIds()]);
    const commands = encodeDraw(plan.selectedClusters, false);
    const heldBlocked = upload();
    const completion = pool.submitFrame(held, [commands]);
    const submittedBlocked = upload();
    const completed = await completion; if (completed.error) throw new Error(completed.error);
    flush([incoming]);
    const evicted = best.filter(id => !pool!.residentPageIds().has(id));
    const reused = pool.diagnostics();
    const recovered = await capture('槽位复用：缺页回退');
    const flags = Array.from(new Uint32Array((await read(pool.pageResidency, 0, directory.pages.length * 4)).buffer));
    const publishedIds = flags.flatMap((flag, id) => flag ? [id] : []);
    const residentIds = [...pool.residentPageIds()].sort((a, b) => a - b);
    const footer = document.createElement('p'); footer.textContent = `GPU 页池 ${slotCount} 个槽位 / ${directory.pages.length} 个资产页；每帧上传上限 8 KiB。三幅 GPU 图像均与同一选簇的完整驻留参考逐像素比对。`; panel.appendChild(footer);
    selection?.dispose(); pool.dispose(); await cache.dispose(); await device.queue.onSubmittedWorkDone();
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, isFallbackAdapter: adapter.info.isFallbackAdapter },
      sourceTriangles: hierarchy.sourceTriangleCount, pages: directory.pages.length, roots: directory.rootPages.length, slotCount, budget,
      peak, ownedAfterDispose: owned, firstUpload, unpublished, unpublishedFlags, maxUploadBytes, maxUploadWrites,
      coarse, detailed, recovered, best, incoming, beforeEviction, heldBlocked, submittedBlocked, evicted, reused,
      pageByteMismatches, publishedIds, residentIds, cleanup: pool.diagnostics(), selectionOwned, selection: selection?.diagnostics(), errors };
  } finally {
    selection?.dispose(); pool?.dispose(); await cache?.dispose(); for (const buffer of buffers) buffer.destroy(); for (const texture of textures) texture.destroy(); device.destroy();
  }
}
