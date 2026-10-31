import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { MaterialResources, type GpuMaterial } from '../../../src/gpu-material';
import { GpuWorldQuery } from '../../../src/gpu-world-query';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity } from '../../../src/math';
import { packWorldRays, WORLD_HIT_BYTES, type WorldQueryOptions, type WorldRay } from '../../../src/world-query';

/**
 * Rays exactly on shared edges and vertices of closed triangle grids must hit
 * on every adapter and on both static traversal paths. Each grid has a split
 * seam column (equal positions, different vertex IDs), alternating diagonals
 * and per-triangle rotated vertex order; batches are rotated and offset far
 * from the origin. Straight axis rays on the axis-aligned grids pass exactly
 * through float32 vertices and edge points.
 */
type Vec3 = [number, number, number];
const N = 6, SPACING = 0.5;
const ARMS: { name: string; options: WorldQueryOptions }[] = [
  { name: 'per-batch', options: { staticTraversal: 'per-batch' } },
  { name: 'shared', options: { staticTraversal: 'shared', dispatchEncoding: 'single-pass' } },
];
const f = Math.fround;

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}

/** Grid vertices (with a duplicated seam column) and triangles. */
function grid(): { local: Vec3[]; indices: number[]; points: Vec3[]; edges: [number, number][] } {
  const local: Vec3[] = [], id = (i: number, j: number, right: boolean) => {
    const seam = i === N / 2;
    return seam && right ? (N + 1) * (N + 1) + j : j * (N + 1) + i;
  };
  for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) local.push([(i - N / 2) * SPACING, (j - N / 2) * SPACING, 0]);
  for (let j = 0; j <= N; j++) local.push([0, (j - N / 2) * SPACING, 0]); // Seam copies.
  const indices: number[] = [], edges: [number, number][] = [];
  let rotation = 0;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const right = i >= N / 2;
    const a = id(i, j, right), b = id(i + 1, j, right), c = id(i + 1, j + 1, right), d = id(i, j + 1, right);
    const pair = (i + j) % 2 ? [[a, b, c], [a, c, d]] : [[a, b, d], [b, c, d]];
    for (const triangle of pair) {
      const r = rotation++ % 3;
      indices.push(triangle[r], triangle[(r + 1) % 3], triangle[(r + 2) % 3]);
      for (let k = 0; k < 3; k++) edges.push([triangle[k], triangle[(k + 1) % 3]]);
    }
  }
  return { local, indices, points: local, edges };
}

function rotationMatrix(next: () => number): number[] {
  const axis = (() => { const v: Vec3 = [next() - 0.5, next() - 0.5, next() - 0.5]; const l = Math.hypot(...v); return v.map(x => x / l); })();
  const angle = next() * Math.PI * 2, c = Math.cos(angle), s = Math.sin(angle), t = 1 - c, [x, y, z] = axis;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}

export async function runWorldQueryWatertightFixture() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let ownedBytes = 0;
  const account = (delta: number) => { ownedBytes += delta; };
  const owned: GPUBuffer[] = [];
  const buffer = (size: number, usage: GPUBufferUsageFlags, source?: ArrayBuffer) => {
    const created = device.createBuffer({ size, usage, mappedAtCreation: !!source });
    if (source) { new Uint8Array(created.getMappedRange()).set(new Uint8Array(source)); created.unmap(); }
    owned.push(created); account(size); return created;
  };
  const materials = new MaterialResources(device, account);
  const queries: GpuWorldQuery[] = [];
  const cleanup = { ownedBytesAfterDispose: -1 };
  try {
    const { local, indices, edges } = grid();
    const vertexCount = local.length;
    const vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
    local.forEach((point, vertex) => {
      point.forEach((value, axis) => view.setFloat32(vertex * 80 + axis * 4, value, true));
      view.setFloat32(vertex * 80 + 12, 1, true); view.setFloat32(vertex * 80 + 24, 1, true); view.setFloat32(vertex * 80 + 48, 1, true);
    });
    const mesh: MeshData = { vertices, vertexCount, indices: Uint32Array.from(indices), uvs: new Float32Array(vertexCount * 2),
      bounds: { min: [-N / 2 * SPACING, -N / 2 * SPACING, 0], max: [N / 2 * SPACING, N / 2 * SPACING, 0] } };
    // Eight batches 30+ m apart: four axis-aligned at dyadic offsets (exact axis rays), four rotated at arbitrary offsets.
    const next = random(2026);
    const rotatedOffsets: Vec3[] = [[60.3, -40.7, 10.1], [-50.9, -70.3, -15.2], [90.1, 80.7, 5.3], [-20.6, 110.4, -8.8]];
    const placements = Array.from({ length: 8 }, (_, index) => index < 4
      ? { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], offset: [-120.5 + index * 32, 64.25, 0.75 * index] as Vec3 }
      : { rotation: rotationMatrix(next), offset: rotatedOffsets[index - 4] });
    const world = placements.map(({ rotation: r, offset }) => local.map(([x, y, z]) => [0, 1, 2].map(row =>
      f(r[row * 3] * x + r[row * 3 + 1] * y + r[row * 3 + 2] * z + offset[row])) as Vec3));
    const scenes: SceneBatch[] = placements.map((_, index) => ({ name: `grid-${index}`, mesh, capacity: 1, inputMode: 'committed',
      instances: [{ entityId: 900 + index, entityGeneration: 1, historyVersion: 0, model: identity(), normalMatrix: identity(),
        bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1 } as InstanceState],
      deformation: { jointCount: 0, nodes: [] }, morphTargets: [], queryIdentity: { assetId: 'watertight-grid', primitiveId: index, topologyVersion: 1 } }));
    const outputs = world.map(points => {
      const data = new Float32Array(vertexCount * 16);
      points.forEach((position, vertex) => { data.set([...position, 1], vertex * 16); data.set([0, 0, 1, 0], vertex * 16 + 4);
        data.set([...position, 1], vertex * 16 + 8); data.set([1, 0, 0, 1], vertex * 16 + 12); });
      return buffer(data.byteLength, GPUBufferUsage.STORAGE, data.buffer);
    });
    const instanceData = new Float32Array(56);
    for (const offset of [0, 16, 32]) instanceData.set(identity(), offset);
    instanceData.set([1, 1, 1, 1, 0, 1, 0, 0], 48);
    const instances = buffer(instanceData.byteLength, GPUBufferUsage.STORAGE, instanceData.buffer);
    const material: GpuMaterial = materials.create({ definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, baseColor: m.parameter(0) },
      parameters: new Float32Array([0.5, 0.5, 0.5, 1]) } as MaterialBinding);
    // Targets: every vertex (seam copies included), edge midpoints and random edge points, in float32 world space.
    const rays: WorldRay[] = [], kinds: string[] = [], targetBatch: number[] = [];
    const rayNext = random(77);
    // Boundary targets only take exact axis rays: a rounded oblique ray may legitimately pass just outside the mesh.
    const boundary = (vertex: number) => Math.abs(local[vertex][0]) === N / 2 * SPACING || Math.abs(local[vertex][1]) === N / 2 * SPACING;
    world.forEach((points, batch) => {
      const axisAligned = batch < 4;
      const targets: { point: Vec3; kind: string; interior: boolean }[] = points.map((point, vertex) => ({ point, kind: 'vertex', interior: !boundary(vertex) }));
      for (const [p, q] of edges) for (const t of [0.5, rayNext(), rayNext()]) {
        targets.push({ point: [0, 1, 2].map(axis => f(points[p][axis] + (points[q][axis] - points[p][axis]) * t)) as Vec3, kind: 'edge',
          interior: !(boundary(p) && boundary(q) && (local[p][0] === local[q][0] || local[p][1] === local[q][1])) });
      }
      for (const { point, kind, interior } of targets) {
        const directions: Vec3[] = axisAligned ? [[0, 0, -1], [0, 0, 1]] : [];
        const random3 = () => { const v: Vec3 = [rayNext() - 0.5, rayNext() - 0.5, rayNext() - 0.5]; const l = Math.hypot(...v); return v.map(x => x / l) as Vec3; };
        if (interior) directions.push(random3());
        for (const direction of directions) {
          const distance = 1 + rayNext() * 9;
          const exact = axisAligned && direction[2] !== 0 && direction[0] === 0 && direction[1] === 0;
          const origin = exact ? [point[0], point[1], f(point[2] - direction[2] * distance)] as Vec3
            : point.map((value, axis) => f(value - direction[axis] * distance)) as Vec3;
          rays.push({ origin, direction, tMin: 0, tMax: distance * 2 });
          kinds.push(`${exact ? 'exact-' : ''}${kind}`); targetBatch.push(batch);
        }
      }
    });
    const count = rays.length;
    for (const arm of ARMS) {
      const query = await GpuWorldQuery.create(device, materials.layout, account, { maxRays: count, maxBatches: 8, maxInstancesPerBatch: 1,
        maxTrianglesPerMesh: indices.length / 3, ...arm.options });
      const pipeline = await query.createPipeline(material.program);
      scenes.forEach((scene, index) => query.addBatch({ scene, output: outputs[index], instances,
        getMaterial: () => material, getPipeline: () => pipeline, getDirty: () => true }));
      queries.push(query);
    }
    const packed = packWorldRays(rays, count);
    const readback = buffer(count * WORLD_HIT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const records: Uint32Array[] = [];
    for (const query of queries) {
      query.validateSources(); query.updateSnapshot(1, 0);
      device.queue.writeBuffer(query.rayBuffer, 0, packed);
      const encoder = device.createCommandEncoder();
      query.encodeRefit(encoder);
      query.encodeTrace(encoder, count, 0, 0);
      encoder.copyBufferToBuffer(query.hitBuffer, 0, readback, 0, count * WORLD_HIT_BYTES);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      records.push(new Uint32Array(readback.getMappedRange().slice(0)));
      readback.unmap();
    }
    const words = WORLD_HIT_BYTES / 4;
    const arms = ARMS.map((arm, index) => {
      const misses: Record<string, number> = {}, wrongBatch: number[] = [], examples: unknown[] = [];
      let hits = 0, overflows = 0;
      for (let ray = 0; ray < count; ray++) {
        const at = ray * words, batch = records[index][at];
        if (records[index][at + 38]) overflows++;
        if (!batch) {
          misses[kinds[ray]] = (misses[kinds[ray]] ?? 0) + 1;
          if (examples.length < 6) examples.push({ ray, kind: kinds[ray], batch: targetBatch[ray], origin: rays[ray].origin, direction: rays[ray].direction });
          continue;
        }
        hits++;
        if (batch - 1 !== targetBatch[ray]) wrongBatch.push(ray);
      }
      return { arm: arm.name, hits, misses, overflows, wrongBatch: wrongBatch.length, examples };
    });
    let mismatches = 0;
    for (let ray = 0; ray < count; ray++) {
      for (let w = 0; w < words; w++) if ((w < 36 || w > 38) && records[0][ray * words + w] !== records[1][ray * words + w]) { mismatches++; break; }
    }
    const kindsCount = kinds.reduce<Record<string, number>>((total, kind) => ({ ...total, [kind]: (total[kind] ?? 0) + 1 }), {});
    await device.queue.onSubmittedWorkDone();
    const info = adapter.info as GPUAdapterInfo & { isFallbackAdapter?: boolean };
    return { adapter: { vendor: info.vendor, architecture: info.architecture, isFallbackAdapter: info.isFallbackAdapter ?? null },
      rays: count, kinds: kindsCount, triangles: indices.length / 3, arms, mismatches,
      sharedStatic: queries[1].diagnostics().sharedStatic, errors, cleanup };
  } finally {
    for (const query of queries) query.dispose();
    materials.dispose();
    for (const created of owned) { created.destroy(); account(-created.size); }
    cleanup.ownedBytesAfterDispose = ownedBytes;
    device.destroy();
  }
}
