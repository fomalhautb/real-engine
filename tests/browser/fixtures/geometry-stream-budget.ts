import { GeometryDemoScene } from '../../../src/geometry-demo-scene';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { GeometryStreamCut } from '../../../src/geometry-stream-cut';
import { Renderer } from '../../../src/renderer';
import { identity } from '../../../src/math';

/** Real Renderer, small physical pool and one-page CPU transfer window. */
export async function runGeometryStreamBudget() {
  const scene = await GeometryDemoScene.create(), hierarchy = scene.hierarchy;
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 49152 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const planner = new GeometryStreamCut(directory);
  const roots = new Set(directory.rootPages);
  const verticesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].vertexCount, 0)));
  const indicesPerSlot = Math.max(...directory.pages.map(page => page.clusterIds.reduce((sum, id) => sum + directory.clusters[id].triangleCount * 3, 0)));
  // Six slots can only hold the common coarse ancestry. Eight slots permit
  // competing finer branches: view A needs page 8, view B needs page 7.
  const slots = roots.size + 4;
  const poolBytes = slots * (verticesPerSlot * 128 + indicesPerSlot * 4) + directory.clusters.length * 32 + directory.pages.length * 4;
  const cpuBytes = directory.rootPages.reduce((sum, id) => sum + directory.pages[id].decodedBytes, 0)
    + Math.max(...directory.pages.filter(page => !roots.has(page.id)).map(page => page.decodedBytes));
  const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:512px;height:512px;display:block';
  document.body.appendChild(canvas);
  const source = scene.batches[0], model = identity();
  const batch = { ...source, mesh: hierarchy.source, geometryLod: undefined,
    capacity: 1, instances: [{ ...source.instances[0], model, normalMatrix: identity() }],
    geometryStream: { directory, provider: async (page: { id: number }, signal: AbortSignal) => {
      signal.throwIfAborted(); return encoded.pages[page.id].bytes;
    }, maxRequestsPerFrame: 2, cpuCache: { maxResidentBytes: cpuBytes }, gpuPool: {
      maxResidentBytes: poolBytes, maxUploadBytesPerFrame: 8192, maxUploadWritesPerFrame: 8, maxUploadPagesPerFrame: 1,
    } } };
  const renderer = await Renderer.create(canvas, [batch], { sunShadow: { resolution: 256,
    focusBounds: { min: [-4, -4, -4], max: [4, 4, 4] } } });
  const gpu = renderer;
  let peakCpu = 0, checks = 0;
  const snapshots: { label: string; png: string; triangles: number; resident: number[]; uploads: number; evictions: number }[] = [];
  const eyeA: [number, number, number] = [3.8, 3.2, 5];
  const eyeB: [number, number, number] = [-3.8, 3.2, -5];
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
    const pagesA = steadyA.state.allocation.pages.join(',');
    const changed = await settle(eyeB);
    if (changed.state.allocation.pages.join(',') === pagesA) throw new Error('WORKING_SET_VIEW_DID_NOT_CHANGE');
    await step(eyeB, '视点 B：预算内换页后稳定');
    const transitions = [steadyA, changed].map(result => ({ pages: result.state.allocation.pages,
      uploads: result.state.pool.uploads, evictions: result.state.pool.evictions }));
    // Repeated returns exercise eviction/reload, not just one initial admission.
    for (const eye of [eyeA, eyeB, eyeA]) {
      const result = await settle(eye);
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
    const panel = document.createElement('section'); panel.id = 'stream-budget-visual';
    panel.style.cssText = 'position:fixed;inset:0;background:#11171d;color:#eaf1f6;padding:28px;z-index:10000;font:16px system-ui';
    panel.innerHTML = '<h1>整组页面预算 · 祖先保留 · 主渲染器</h1><p>GPU 给出细节需求；调度器只加载预算内的完整组。每帧实际切分均与独立 CPU 参考核对。</p>';
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
    return { adapter, errors, checks, pages: directory.pages.length, slots, poolBytes, cpuBytes, peakCpu,
      firstTriangles: first.cut.triangles, steadyTriangles: steadyA.cut.triangles, viewBTriangles: changed.cut.triangles,
      firstAllocation: steadyA.state.allocation, viewBAllocation: changed.state.allocation,
      finalPool: final.pool, transitions, afterDispose: gpu.stats.ownedGpuBytes };
  } finally { gpu.dispose(); canvas.remove(); }
}
