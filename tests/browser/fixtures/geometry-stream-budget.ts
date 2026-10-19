import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { buildGeometryQueryProxy, type GeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { Renderer } from '../../../src/renderer';
import type { RenderBatch } from '../../../src/contracts';
import type { WorldRay } from '../../../src/world-query';
import { identity } from '../../../src/math';
import { GEOMETRY_GPU_VERTEX_BYTES } from '../../../src/geometry-gpu-page';

function proxyRays(proxy: GeometryQueryProxy): WorldRay[] {
  const view = new DataView(proxy.mesh.vertices), indices = proxy.mesh.indices;
  const point = (id: number): [number, number, number] => [0, 4, 8].map(offset =>
    view.getFloat32(id * 80 + offset, true)) as [number, number, number];
  return [0, Math.floor(indices.length / 6), indices.length / 3 - 1].map(triangle => {
    const [a, b, c] = [0, 1, 2].map(corner => point(indices[triangle * 3 + corner]));
    const u = b.map((value, axis) => value - a[axis]), v = c.map((value, axis) => value - a[axis]);
    const cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const length = Math.hypot(...cross);
    if (!(length > 0)) throw new Error('WORKING_SET_PROXY_DEGENERATE_TRIANGLE');
    const normal = cross.map(value => value / length);
    const center = a.map((value, axis) => (value + b[axis] + c[axis]) / 3);
    return { origin: center.map((value, axis) => value + normal[axis] * 0.02) as [number, number, number],
      direction: normal.map(value => -value) as [number, number, number], tMax: 0.1 };
  });
}

/** Real Renderer, small physical pool and one-page CPU transfer window. */
export async function runGeometryStreamBudget(sourceFree = false) {
  // The source mesh is scoped to offline construction. The source-free branch
  // passes only compressed pages, the directory, and a fixed proxy to Renderer.
  const assets = await (async () => {
    const scene = await GeometryDemoScene.create(), hierarchy = scene.hierarchy;
    const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
    const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
    const proxy = sourceFree ? await buildGeometryQueryProxy(directory,
      async page => encoded.pages[page.id].bytes.slice(), { maxError: 1e30, topologyVersion: 1 }) : undefined;
    const { mesh, geometryLod: _geometryLod, ...base } = scene.batches[0];
    return { encoded, directory, proxy, base, sourceMesh: sourceFree ? undefined : mesh };
  })();
  const { encoded, directory, proxy } = assets;
  const planner = new GeometryStreamCut(directory);
  const roots = new Set(directory.rootPages);
  const verticesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].vertexCount, 0)));
  const indicesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].triangleCount * 3, 0)));
  // Six slots can only hold the common coarse ancestry. Eight slots permit
  // competing finer branches: view A needs page 8, view B needs page 7.
  const slots = roots.size + 4;
  const poolBytes = slots * (verticesPerSlot * GEOMETRY_GPU_VERTEX_BYTES + indicesPerSlot * 4) + directory.clusters.length * 32 + directory.pages.length * 4;
  const cpuBytes = directory.rootPages.reduce((sum, id) => sum + directory.pages[id].decodedBytes, 0)
    + Math.max(...directory.pages.filter(page => !roots.has(page.id)).map(page => page.decodedBytes));
  const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:512px;height:512px;display:block';
  document.body.appendChild(canvas);
  const model = identity();
  const common = { ...assets.base,
    capacity: 1, instances: [{ ...assets.base.instances[0], model, normalMatrix: identity() }],
    geometryStream: { directory, provider: async (page: { id: number }, signal: AbortSignal) => {
      signal.throwIfAborted(); return encoded.pages[page.id].bytes;
    }, maxRequestsPerFrame: 2, cpuCache: { maxResidentBytes: cpuBytes }, gpuPool: {
      maxResidentBytes: poolBytes, maxUploadBytesPerFrame: 8192, maxUploadWritesPerFrame: 8, maxUploadPagesPerFrame: 1,
    } } };
  const batch: RenderBatch = sourceFree
    ? { ...common, geometryStream: { ...common.geometryStream, queryProxy: proxy! },
      queryIdentity: { assetId: 'stream-budget-proxy', primitiveId: 0, topologyVersion: proxy!.topologyVersion } }
    : { ...common, mesh: assets.sourceMesh! };
  if (sourceFree && 'mesh' in batch) throw new Error('WORKING_SET_SOURCE_MESH_RETAINED');
  const renderer = await Renderer.create(canvas, [batch], { sunShadow: { resolution: 256,
    focusBounds: { min: [-4, -4, -4], max: [4, 4, 4] } },
  ...(sourceFree ? { worldQueries: { maxRays: 4 } } : {}) });
  const gpu = renderer;
  let peakCpu = 0, checks = 0;
  const snapshots: { label: string; png: string; triangles: number; resident: number[]; uploads: number; evictions: number }[] = [];
  const eyeA: [number, number, number] = [3.8, 3.2, 5];
  const eyeB: [number, number, number] = [-3.8, 3.2, -5];
  const rays = proxy ? proxyRays(proxy) : [];
  const querySnapshots: { label: string; hits: string; metadata: string }[] = [];
  const checkQuery = async (label: string) => {
    if (!proxy) return;
    const result = await gpu.traceRays(rays);
    if (result.hits.some(hit => !hit)) throw new Error('WORKING_SET_PROXY_QUERY_MISS');
    const hits = JSON.stringify(result.hits.map(hit => ({ entityId: hit!.entityId, triangleIndex: hit!.triangleIndex,
      distance: hit!.distance, position: hit!.position, normal: hit!.normal,
      uv: hit!.uv, topologyVersion: hit!.topologyVersion, historyVersion: hit!.historyVersion })));
    const state = gpu.diagnostics().geometryStreams[0];
    if (state.sourceRepresentation !== 'fixed-shadow-query-proxy' || !state.queryProxy
      || state.queryProxy.cameraDependent || state.queryProxy.exactSourceRetained) throw new Error('WORKING_SET_PROXY_REPRESENTATION');
    const metadata = JSON.stringify(state.queryProxy);
    if (querySnapshots.length && (hits !== querySnapshots[0].hits || metadata !== querySnapshots[0].metadata)) {
      throw new Error('WORKING_SET_PROXY_CHANGED_WITH_RESIDENCY');
    }
    querySnapshots.push({ label, hits, metadata });
  };
  const step = async (eye: [number, number, number], label?: string) => {
    gpu.render(eye, [0, 0, 0]);
    const png = label ? canvas.toDataURL('image/png') : '';
    const cut = await gpu.probeGeometryLod(batch.name);
    const state = gpu.diagnostics().geometryStreams[0];
    const reference = planner.plan(new Set(state.pool.residentPages), { eye, near: 0.1,
      projectionScale: gpu.stats.height / (2 * Math.tan(Math.PI / 6.4)), maxErrorPixels: 0, model });
    const ids = cut.selectedClusters.map(cluster => cluster.clusterId).sort((a, b) => a - b);
    if (JSON.stringify(ids) !== JSON.stringify(reference.selectedClusters)) throw new Error('WORKING_SET_INCOMPLETE_CUT');
    if (cut.overflow || state.pool.allocatedGpuBytes !== poolBytes || state.allocation.pages.length > slots
      || state.cache.residentBytes + state.cache.reservedBytes > cpuBytes
      || !directory.rootPages.every(id => state.pool.residentPages.includes(id))) throw new Error('WORKING_SET_BUDGET_OR_ROOTS');
    if (gpu.errors.length) throw new Error(gpu.errors.join('\n'));
    peakCpu = Math.max(peakCpu, state.cache.residentBytes + state.cache.reservedBytes); checks++;
    if (label) snapshots.push({ label, png, triangles: cut.triangles, resident: state.pool.residentPages,
      uploads: state.pool.uploads, evictions: state.pool.evictions });
    return { cut, state };
  };
  const settle = async (eye: [number, number, number]) => {
    for (let attempt = 0; attempt < 240; attempt++) {
      const result = await step(eye), { state } = result;
      if (state.allocation.pages.every(id => state.pool.residentPages.includes(id)) && !state.pool.pendingPages.length
        && !state.cache.activeRequests && !state.cache.queuedRequests) {
        // Repeated unchanged views must not upload or refetch the same pages.
        const before = [state.pool.uploads, state.pool.evictions, state.cache.requestsStarted];
        let stable = result;
        for (let i = 0; i < 12; i++) stable = await step(eye);
        const after = [stable.state.pool.uploads, stable.state.pool.evictions, stable.state.cache.requestsStarted];
        if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('WORKING_SET_STEADY_THRASH');
        return stable;
      }
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    throw new Error('WORKING_SET_DID_NOT_SETTLE: ' + JSON.stringify(gpu.diagnostics().geometryStreams[0]));
  };
  try {
    gpu.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false, debug: 'clusters' });
    const first = await step(eyeA, '根页：完整粗层级');
    const steadyA = await settle(eyeA); await step(eyeA, '视点 A：整组细节稳定驻留');
    await checkQuery('A1');
    const pagesA = steadyA.state.allocation.pages.join(',');
    const changed = await settle(eyeB);
    await checkQuery('B1');
    if (changed.state.allocation.pages.join(',') === pagesA) throw new Error('WORKING_SET_VIEW_DID_NOT_CHANGE');
    await step(eyeB, '视点 B：预算内换页后稳定');
    const transitions = [steadyA, changed].map(result => ({ pages: result.state.allocation.pages,
      uploads: result.state.pool.uploads, evictions: result.state.pool.evictions }));
    // Repeated returns exercise eviction/reload, not just one initial admission.
    for (const eye of [eyeA, eyeB, eyeA]) {
      const result = await settle(eye);
      await checkQuery(eye === eyeA ? `A${querySnapshots.length}` : `B${querySnapshots.length}`);
      const expected = eye === eyeA ? pagesA : changed.state.allocation.pages.join(',');
      if (result.state.allocation.pages.join(',') !== expected) throw new Error('WORKING_SET_RETURN_CHANGED');
      const previous = transitions[transitions.length - 1];
      if (result.state.pool.evictions <= previous.evictions || result.state.pool.uploads <= previous.uploads) {
        throw new Error('WORKING_SET_RETURN_DID_NOT_REPLACE_PAGE');
      }
      transitions.push({ pages: result.state.allocation.pages, uploads: result.state.pool.uploads, evictions: result.state.pool.evictions });
    }
    const final = gpu.diagnostics().geometryStreams[0];
    if (steadyA.cut.triangles <= first.cut.triangles || !steadyA.state.allocation.budgetDeferredGroups.length
      || !final.pool.evictions) throw new Error('WORKING_SET_NOT_EXERCISING_PRESSURE');
    const panel = document.createElement('section'); panel.id = sourceFree ? 'stream-budget-proxy-visual' : 'stream-budget-visual';
    panel.style.cssText = 'position:fixed;inset:0;background:#11171d;color:#eaf1f6;padding:28px;z-index:10000;font:16px system-ui';
    panel.innerHTML = sourceFree
      ? '<h1>源网格不驻留 · 固定查询代理 · 小池换页</h1><p>表面细节页交替驻留，阴影与世界查询使用同一固定代理。</p>'
      : '<h1>整组页面预算 · 祖先保留 · 主渲染器</h1><p>GPU 给出细节需求；调度器只加载预算内的完整组。每帧实际切分均与独立 CPU 参考核对。</p>';
    const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:16px'; panel.appendChild(row);
    for (const snapshot of snapshots) {
      const card = document.createElement('div'); card.style.cssText = 'flex:1;min-width:0';
      const label = document.createElement('p'); label.textContent = snapshot.label;
      const image = document.createElement('img'); image.src = snapshot.png; image.style.cssText = 'width:100%;display:block;border-radius:8px';
      const caption = document.createElement('p'); caption.textContent = `${snapshot.triangles} 三角形 · ${snapshot.resident.length}/${directory.pages.length} 页驻留`;
      card.append(label, image, caption); row.appendChild(card); await image.decode();
    }
    const footer = document.createElement('p'); footer.textContent = `物理页池 ${slots} 槽 / ${directory.pages.length} 个资产页 · ${poolBytes.toLocaleString()} 字节 · 每帧上传 ≤8 KiB · 稳定视点无重复上传/请求`;
    panel.appendChild(footer); document.body.appendChild(panel);
    const adapter = gpu.info.adapterInfo, errors = [...gpu.errors];
    gpu.dispose(); canvas.remove();
    return { adapter, errors, checks, pages: directory.pages.length, slots, poolBytes, cpuBytes, peakCpu, sourceFree,
      firstTriangles: first.cut.triangles, steadyTriangles: steadyA.cut.triangles, viewBTriangles: changed.cut.triangles,
      firstAllocation: steadyA.state.allocation, viewBAllocation: changed.state.allocation,
      finalPool: final.pool, transitions, querySnapshots, proxyMetadata: final.queryProxy,
      proxyVertices: proxy?.mesh.vertexCount ?? null, proxyTriangles: proxy ? proxy.mesh.indices.length / 3 : null,
      sourceVertices: directory.sourceVertexCount, sourceTriangles: directory.sourceTriangleCount,
      afterDispose: gpu.stats.ownedGpuBytes };
  } finally { gpu.dispose(); canvas.remove(); }
}
