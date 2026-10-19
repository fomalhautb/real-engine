import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { translation, identity } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import { GpuQuerySceneBvh } from '../../../src/gpu-query-scene-bvh';
import { packWorldRays, WORLD_HIT_BYTES, type WorldRay, type WorldQueryResult } from '../../../src/world-query';
import type { Bounds3 } from '../../../src/deformation';

const COUNT = 405;
const LAST = COUNT - 1;
const SHIFT = 0.9;
type Vec3 = [number, number, number];
type Mode = 'bvh' | 'linear';

function quad(): MeshData {
  const vertices = new ArrayBuffer(4 * 80);
  const view = new DataView(vertices);
  const points = [[-0.35, -0.35], [0.35, -0.35], [0.35, 0.35], [-0.35, 0.35]];
  points.forEach(([x, y], index) => {
    const byte = index * 80;
    view.setFloat32(byte, x, true);
    view.setFloat32(byte + 4, y, true);
    view.setFloat32(byte + 12, 1, true);
    view.setFloat32(byte + 24, 1, true);
    view.setFloat32(byte + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    bounds: { min: [-0.35, -0.35, 0], max: [0.35, 0.35, 0] } };
}

function center(index: number): Vec3 {
  return [((index % 27) - 13) * 1.5, (Math.floor(index / 27) - 7) * 1.5, 0];
}

function color(index: number): Float32Array {
  return new Float32Array([(index + 1) / 512, 0.25, 0.6, 1]);
}

function pose(index: number, generation = 1, moved = false): InstanceState {
  const [x, y] = center(index);
  return { entityId: 10000 + index, entityGeneration: generation,
    historyVersion: generation - 1, model: translation(x + (moved ? SHIFT : 0), y, 0),
    normalMatrix: identity(), bones: [], deformScalars: [], color: [1, 1, 1, 1],
    metalness: 0, roughness: 0.5 };
}

function scene(mesh: MeshData): SceneBatch[] {
  return Array.from({ length: COUNT }, (_, index) => ({
    name: `capacity-${index}`, mesh, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'capacity-triangle', primitiveId: index, topologyVersion: 1 },
    instances: [pose(index)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.parameter(0) }, parameters: color(index) },
  }));
}

function ray(index: number, moved = false): WorldRay {
  const [x, y] = center(index);
  return { origin: [x + (moved ? SHIFT : 0), y, 5], direction: [0, 0, -1] };
}

function compact(result: WorldQueryResult) {
  return { frame: result.frame, materialRevision: result.materialRevision,
    hits: result.hits.map(hit => hit && ({ batch: hit.batch, entityId: hit.entityId,
      entityGeneration: hit.entityGeneration, historyVersion: hit.historyVersion,
      topologyVersion: hit.topologyVersion, triangleIndex: hit.triangleIndex,
      position: hit.position, distance: hit.distance, baseColor: hit.baseColor })) };
}

function diagnostics(renderer: Renderer) {
  const query = renderer.diagnostics().worldQueries;
  if (!query) throw new Error('CAPACITY_QUERY_DIAGNOSTICS_MISSING');
  return query;
}

async function reject(operation: () => Promise<unknown>): Promise<string> {
  try { await operation(); return ''; } catch (error) { return String(error); }
}

export async function runWorldQueryCapacityFixture() {
  const canvas = document.createElement('canvas');
  canvas.width = 32; canvas.height = 32;
  canvas.style.width = '32px'; canvas.style.height = '32px';
  document.body.appendChild(canvas);
  const mesh = quad();
  const batches = scene(mesh);
  const invalidBudgets = await Promise.all([0, 1.5, 4097, Number.NaN].map(maxBatches =>
    reject(() => Renderer.create(canvas, [], { maxBatches }))));
  const defaultOverflow = await reject(() => Renderer.create(canvas, batches.slice(0, 257)));
  const rendererOverflow = await reject(() => Renderer.create(canvas, [...batches, batches[0]],
    { maxBatches: COUNT }));
  const queryOverflow = await reject(() => Renderer.create(canvas, batches,
    { maxBatches: COUNT, worldQueries: { maxBatches: COUNT - 1 } }));

  const results = [];
  try {
    for (const mode of ['bvh', 'linear'] as Mode[]) {
      const current = scene(mesh);
      let renderer: Renderer | undefined;
      try {
        renderer = await Renderer.create(canvas, current, { maxBatches: COUNT,
          sunShadow: { resolution: 256 }, worldQueries: { maxRays: 512,
            maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1, sceneAcceleration: mode } });
        const adapter = renderer.info.adapterInfo;
        const view: Vec3 = [0, 0, 45], target: Vec3 = [0, 0, 0];
        renderer.render(view, target);
        const initial = compact(await renderer.traceRays(Array.from({ length: COUNT }, (_, i) => ray(i))));
        const initialDiagnostics = diagnostics(renderer);

        current[LAST].instances = [pose(LAST, 1, true)];
        renderer.render(view, target);
        const moved = compact(await renderer.traceRays([ray(LAST), ray(LAST, true), ray(383), ray(384)]));
        current[LAST].instances = [];
        renderer.render(view, target);
        const empty = compact(await renderer.traceRays([ray(LAST), ray(LAST, true), ray(383), ray(384)]));
        current[LAST].instances = [pose(LAST, 2, true)];
        renderer.render(view, target);
        const restored = compact(await renderer.traceRays([ray(LAST), ray(LAST, true), ray(383), ray(384)]));
        renderer.setMaterialParameters('capacity-256', new Float32Array([0.9, 0.1, 0.3, 1]));
        const materialChanged = compact(await renderer.traceRays([ray(255), ray(256), ray(257)]));

        const errors = [...renderer.errors];
        renderer.dispose();
        results.push({ mode, adapter, initial, initialDiagnostics, moved, empty, restored,
          materialChanged, errors, ownedAfterDispose: renderer.stats.ownedGpuBytes });
        renderer = undefined;
      } finally { renderer?.dispose(); }
    }
    return { count: COUNT, invalidBudgets, defaultOverflow, rendererOverflow, queryOverflow, results };
  } finally { canvas.remove(); }
}

/** Directly exercises the advertised 4096-slot tree without constructing 4096 render batches. */
export async function runSceneBvhMaskCapacityFixture() {
  if (!navigator.gpu) throw new Error('WEBGPU_UNAVAILABLE');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('ADAPTER_UNAVAILABLE');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const owned: GPUBuffer[] = [];
  let ownedBytes = 0;
  const allocate = (label: string, size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer): GPUBuffer => {
    // Only this fixture can copy candidate masks back; production usage is unchanged.
    const buffer = device.createBuffer({ label, size,
      usage: usage | (label === 'world query scene candidate masks' ? GPUBufferUsage.COPY_SRC : 0),
      mappedAtCreation: !!data });
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data)); buffer.unmap(); }
    owned.push(buffer); ownedBytes += buffer.size;
    return buffer;
  };
  const rays = [0, 10, 20, 100].map((x): WorldRay =>
    ({ origin: [x, 0, 5], direction: [0, 0, -1] }));
  const rayData = packWorldRays(rays, rays.length);
  const records = [];
  try {
    for (const capacity of [1, 33, 4096]) {
      const firstOwned = owned.length;
      const last = capacity - 1;
      const rayBuffer = allocate('capacity rays', rayData.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, rayData);
      const hitBuffer = allocate('capacity hits', rays.length * WORLD_HIT_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const tree = await GpuQuerySceneBvh.create(device, allocate, rayBuffer, hitBuffer,
        rays.length, 1, capacity);
      const maskBytes = rays.length * tree.maskWords * 4;
      const readback = allocate('capacity candidate readback', maskBytes,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

      async function sample(stage: 'initial' | 'moved' | 'deleted' | 'restored'): Promise<number[]> {
        const positions = new Map<number, number>();
        if (stage !== 'deleted' || last !== 0) positions.set(0, 0);
        if (stage === 'initial' || stage === 'restored') positions.set(last, last === 0 ? 0 : 10);
        if (stage === 'moved') positions.set(last, 20);
        const ordered: (Bounds3 | null)[] = Array(capacity).fill(null);
        const roots = new Float32Array(capacity * 8);
        for (const [index, x] of positions) {
          ordered[index] = { min: [x - 0.25, -0.25, -0.25], max: [x + 0.25, 0.25, 0.25] };
          roots.set([x - 0.25, -0.25, -0.25, 1,
            x + 0.25, 0.25, 0.25, 0], index * 8);
        }
        device.queue.writeBuffer(tree.boundsBuffer, 0, roots);
        tree.updateOrder(ordered);
        device.queue.writeBuffer(hitBuffer, 0, new Uint8Array(rays.length * WORLD_HIT_BYTES));
        const encoder = device.createCommandEncoder();
        tree.encodeRefit(encoder, () => undefined);
        tree.encodeCandidates(encoder, 0, rays.length, 16384, true);
        encoder.copyBufferToBuffer(tree.maskBuffer, 0, readback, 0, maskBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const words = Array.from(new Uint32Array(readback.getMappedRange().slice(0)));
        readback.unmap();
        return words;
      }

      const initial = await sample('initial');
      const moved = await sample('moved');
      const deleted = await sample('deleted');
      const restored = await sample('restored');
      records.push({ capacity, nodeCount: tree.nodeCount, maskWords: tree.maskWords,
        initial, moved, deleted, restored });
      for (const buffer of owned.splice(firstOwned)) { ownedBytes -= buffer.size; buffer.destroy(); }
    }
    const legacyFallback = 'isFallbackAdapter' in adapter && typeof adapter.isFallbackAdapter === 'boolean'
      ? adapter.isFallbackAdapter : null;
    return { adapter: { vendor: adapter.info.vendor ?? '', architecture: adapter.info.architecture ?? '',
      description: adapter.info.description ?? '',
      isFallbackAdapter: typeof adapter.info.isFallbackAdapter === 'boolean'
        ? adapter.info.isFallbackAdapter : legacyFallback }, records, errors, ownedAfterDispose: ownedBytes };
  } finally {
    for (const buffer of owned) buffer.destroy();
    device.destroy();
  }
}
