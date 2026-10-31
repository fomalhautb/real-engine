import { describe, expect, it } from 'vitest';
import { buildBvhTopology, type BvhTopology } from '../src/bvh-topology';
import { buildInstanceBvhTopology } from '../src/instance-bvh';
import {
  buildCpuSharedPool, cpuIntersectTriangle, nearestTriangleCpu, orderSharedLeaves, packLeafOrderedTriangles, packSharedFlags, packSharedLeaves,
  planSharedStaticPool, sharedTopologyWords, traceSharedPoolCpu, SHARED_FLAG_DOUBLE_SIDED, SHARED_FLAG_MEMBER,
  SHARED_LEAF_WORDS, SHARED_MAX_LEAVES, SHARED_NODE_BYTES, SHARED_POSITION_BYTES, SHARED_TRIANGLE_BYTES,
  type CpuRay, type CpuReferenceTriangle, type SharedStaticCandidate, type SharedStaticLimits,
} from '../src/query-shared-static';

type V3 = [number, number, number];
const f = Math.fround;

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function limits(overrides: Partial<SharedStaticLimits> = {}): SharedStaticLimits {
  return { maxBytes: 1 << 30, maxBindingBytes: 128 << 20, maxBufferBytes: 256 << 20, maxWorkgroupsPerDimension: 65535,
    batchSlots: 16, fixedBytes: 0, ...overrides };
}
function candidate(index: number, overrides: Partial<SharedStaticCandidate> = {}): SharedStaticCandidate {
  return { index, capacity: 4, vertexCount: 6, nodeCount: 3, triangleCount: 4, ineligible: null, ...overrides };
}

function meshFrom(positions: readonly V3[], indices: readonly number[]) {
  const vertices = new ArrayBuffer(positions.length * 80);
  const view = new DataView(vertices);
  positions.forEach((position, vertex) => position.forEach((value, axis) => view.setFloat32(vertex * 80 + axis * 4, value, true)));
  return { vertices, indices: new Uint32Array(indices), vertexCount: positions.length };
}

/** CPU mirror of query-refit.wgsl: per-node bounds with the same conservative padding. */
function refitBounds(topology: BvhTopology, indices: Uint32Array, world: Float32Array): Float32Array {
  const bounds = new Float32Array(topology.nodeCount * 6);
  for (let depth = topology.levels.length - 1; depth >= 0; depth--) for (const node of topology.levels[depth]) {
    const t = node * 4, first = topology.nodes[t], second = topology.nodes[t + 1];
    const low = [f(3.402823466e38), f(3.402823466e38), f(3.402823466e38)], high = low.map(value => -value);
    if (topology.nodes[t + 3] === 1) {
      for (let i = 0; i < second; i++) for (let corner = 0; corner < 3; corner++) {
        const vertex = indices[topology.triangles[first + i] * 3 + corner];
        for (let axis = 0; axis < 3; axis++) {
          low[axis] = Math.min(low[axis], world[vertex * 3 + axis]);
          high[axis] = Math.max(high[axis], world[vertex * 3 + axis]);
        }
      }
    } else {
      for (let axis = 0; axis < 3; axis++) {
        low[axis] = Math.min(bounds[first * 6 + axis], bounds[second * 6 + axis]);
        high[axis] = Math.max(bounds[first * 6 + 3 + axis], bounds[second * 6 + 3 + axis]);
      }
    }
    for (let axis = 0; axis < 3; axis++) {
      const pad = Math.max(f(1e-6), f(Math.max(Math.abs(low[axis]), Math.abs(high[axis])) * f(2e-7)));
      bounds[node * 6 + axis] = f(low[axis] - pad);
      bounds[node * 6 + 3 + axis] = f(high[axis] + pad);
    }
  }
  return bounds;
}

interface TestBatch {
  positions: V3[]; indices: number[]; capacity: number; translations: V3[]; flags: number; entityBase: number;
}

function buildScene(batches: readonly TestBatch[]) {
  const prepared = batches.map(batch => {
    const mesh = meshFrom(batch.positions, batch.indices);
    return { batch, mesh, topology: buildBvhTopology(mesh) };
  });
  const plan = planSharedStaticPool(prepared.map(({ batch, mesh, topology }, index) => ({ index, capacity: batch.capacity,
    vertexCount: mesh.vertexCount, nodeCount: topology.nodeCount, triangleCount: topology.triangleCount, ineligible: null })),
  limits({ batchSlots: batches.length }));
  expect(plan.batches.map(entry => entry.index)).toEqual(batches.map((_, index) => index));
  const reference: CpuReferenceTriangle[] = [];
  const inputs = plan.batches.map(entry => {
    const { batch, mesh, topology } = prepared[entry.index];
    const instances = batch.translations.map((offset, slot) => {
      const world = new Float32Array(batch.positions.flatMap(position => position.map((value, axis) => value + offset[axis])));
      const entityId = batch.entityBase + slot, generation = slot % 2;
      if (batch.flags & SHARED_FLAG_MEMBER) for (let position = 0; position < topology.triangleCount; position++) {
        const corner = (index: number): V3 => {
          const vertex = mesh.indices[topology.triangles[position] * 3 + index];
          return [world[vertex * 3], world[vertex * 3 + 1], world[vertex * 3 + 2]];
        };
        reference.push({ batch: entry.index, instance: slot, position, a: corner(0), b: corner(1), c: corner(2),
          doubleSided: (batch.flags & SHARED_FLAG_DOUBLE_SIDED) !== 0, entityId, generation });
      }
      return { positions: world, bounds: refitBounds(topology, mesh.indices, world), entityId, generation };
    });
    return { topology: topology.nodes, leafTriangles: packLeafOrderedTriangles(mesh.indices, topology.triangles, mesh.vertexCount),
      instances, flags: batch.flags };
  });
  const pool = buildCpuSharedPool(plan, inputs, batches.length, buildInstanceBvhTopology(plan.leafCount, SHARED_MAX_LEAVES));
  return { plan, pool, reference };
}

function soup(seed: number, count: number, scale: number): { positions: V3[]; indices: number[] } {
  const next = random(seed), positions: V3[] = [];
  for (let i = 0; i < count * 3; i++) positions.push([f((next() - 0.5) * scale), f((next() - 0.5) * scale), f((next() - 0.5) * scale)]);
  return { positions, indices: positions.map((_, i) => i) };
}
/** Axis-aligned quad at z = 0 (exact float arithmetic for axis rays). */
function quad(x0: number, y0: number, x1: number, y1: number): { positions: V3[]; indices: number[] } {
  return { positions: [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], indices: [0, 1, 2, 0, 2, 3] };
}

const MEMBER = SHARED_FLAG_MEMBER, DOUBLE = SHARED_FLAG_DOUBLE_SIDED;
function testScene() {
  return buildScene([
    // General-position triangles, kept clear of the axis-ray tie grid below.
    { ...soup(11, 40, 3), capacity: 5, translations: [[5.5, 0, 0], [-4.5, 2, -1], [-4, -3.5, 1.5]], flags: MEMBER, entityBase: 100 },
    // Exact duplicates in slots 0 and 1, then stacked copies: in-batch ties keep the lower slot.
    { ...quad(-2, -2, 2, 2), capacity: 4, translations: [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2]], flags: MEMBER | DOUBLE, entityBase: 200 },
    // A coplanar quad with different bounds and Morton key: cross-batch ties keep the lower batch.
    { ...quad(0, 0, 3, 3), capacity: 2, translations: [[0, 0, 0]], flags: MEMBER | DOUBLE, entityBase: 300 },
    // Not a member (e.g. Masked now): never traversed by the pool.
    { ...soup(12, 24, 3), capacity: 3, translations: [[0, 0, 0], [1, 1, 1]], flags: DOUBLE, entityBase: 400 },
    // Pooled but empty: every slot is an explicitly invalid TLAS leaf.
    { ...soup(13, 8, 2), capacity: 3, translations: [], flags: MEMBER, entityBase: 500 },
    { ...quad(-1, -1, 1, 1), capacity: 1, translations: [[0.5, 0.5, -0.5]], flags: MEMBER, entityBase: 600 },
  ]);
}

function rays(seed: number): CpuRay[] {
  const next = random(seed), result: CpuRay[] = [];
  for (let i = 0; i < 600; i++) {
    const origin: V3 = [f((next() - 0.5) * 12), f((next() - 0.5) * 12), f((next() - 0.5) * 12)];
    const target: V3 = [f((next() - 0.5) * 6), f((next() - 0.5) * 6), f((next() - 0.5) * 6)];
    const delta = target.map((value, axis) => value - origin[axis]);
    const length = Math.hypot(...delta);
    const ray: CpuRay = { origin, direction: delta.map(value => f(value / length)) as V3,
      tMin: i % 3 === 0 ? f(0.0001) : 0, tMax: i % 5 === 0 ? f(4 + next() * 8) : f(1e30) };
    if (i % 7 === 0) ray.ignore = { entityId: [100, 201, 300, 600][i % 4], generation: [0, 1, 0, 0][i % 4] };
    result.push(ray);
  }
  // Exact coplanar ties on axis rays, from both sides (double-sided quads).
  for (let x = -2.5; x <= 3.5; x += 0.5) for (let y = -2.5; y <= 3.5; y += 0.5) {
    result.push({ origin: [x + 0.25, y + 0.25, -5], direction: [0, 0, 1], tMin: 0, tMax: f(1e30) });
    result.push({ origin: [x + 0.25, y + 0.25, 5], direction: [0, 0, -1], tMin: 0, tMax: f(1e30) });
  }
  return result;
}

describe('shared static pool planning', () => {
  it('admits a deterministic first-fit subset and records every exclusion reason', () => {
    const bytesFor = (leaves: number, records: number, triangles: number, vertices: number, slots: number, fixed: number) =>
      (2 * leaves - 1 + records) * SHARED_NODE_BYTES + triangles * SHARED_TRIANGLE_BYTES + vertices * SHARED_POSITION_BYTES
      + (leaves * SHARED_LEAF_WORDS + slots) * 4 + sharedTopologyWords(leaves) * 4 + fixed;
    const small = bytesFor(4, 12, 4, 24, 8, 100);
    const plan = planSharedStaticPool([
      candidate(0, { ineligible: 'deformation' }), candidate(1), candidate(2, { ineligible: 'mutable-input' }),
      candidate(3, { capacity: 64, vertexCount: 1000 }), candidate(5),
    ], limits({ batchSlots: 8, fixedBytes: 100, maxBytes: bytesFor(8, 24, 8, 48, 8, 100) }));
    expect(small).toBeLessThan(plan.bytes.total);
    expect(plan.batches.map(batch => batch.index)).toEqual([1, 5]);
    expect(plan.excluded).toEqual([{ index: 0, reason: 'deformation' }, { index: 2, reason: 'mutable-input' },
      { index: 3, reason: 'budget' }]);
    expect(plan.leafCount).toBe(8);
    expect(plan.tlasNodeCount).toBe(15);
    expect(plan.batches.map(({ leafBase, nodeBase, triangleBase, positionBase }) => ({ leafBase, nodeBase, triangleBase, positionBase })))
      .toEqual([{ leafBase: 0, nodeBase: 15, triangleBase: 0, positionBase: 0 }, { leafBase: 4, nodeBase: 27, triangleBase: 4, positionBase: 24 }]);
    expect(plan.nodeRecords).toBe(15 + 24);
    expect(plan.bytes.total).toBe(bytesFor(8, 24, 8, 48, 8, 100));
    expect(plan.bytes.total).toBe(plan.bytes.nodes + plan.bytes.triangles + plan.bytes.positions + plan.bytes.leaves
      + plan.bytes.topology + plan.bytes.fixed);
  });
  it('excludes batches that exceed the TLAS leaf, update dispatch or binding limits', () => {
    const plan = planSharedStaticPool([
      candidate(0, { capacity: SHARED_MAX_LEAVES - 2 }), candidate(1, { capacity: 4 }), candidate(2, { capacity: 2 }),
    ], limits({ batchSlots: 3 }));
    expect(plan.batches.map(batch => batch.index)).toEqual([0, 2]);
    expect(plan.excluded).toEqual([{ index: 1, reason: 'leaf-limit' }]);
    expect(plan.leafCount).toBe(SHARED_MAX_LEAVES);
    const dispatch = planSharedStaticPool([candidate(0, { capacity: 64, vertexCount: 1000, nodeCount: 24 })],
      limits({ batchSlots: 1, maxWorkgroupsPerDimension: 1023 }));
    expect(dispatch.excluded).toEqual([{ index: 0, reason: 'dispatch-limit' }]);
    expect(dispatch.bytes.total).toBe(0);
    const binding = planSharedStaticPool([candidate(0, { capacity: 16, vertexCount: 1000 }), candidate(1)],
      limits({ batchSlots: 2, maxBindingBytes: 16 * 1000 * SHARED_POSITION_BYTES - 4 }));
    expect(binding.excluded).toEqual([{ index: 0, reason: 'binding-limit' }]);
    expect(binding.batches.map(batch => batch.index)).toEqual([1]);
  });
  it('rejects malformed candidates and limits before planning', () => {
    expect(() => planSharedStaticPool([candidate(1), candidate(1)], limits())).toThrow(/candidate.index/);
    expect(() => planSharedStaticPool([candidate(16)], limits())).toThrow(/candidate.index/);
    expect(() => planSharedStaticPool([candidate(0, { capacity: 0 })], limits())).toThrow(/capacity/);
    expect(() => planSharedStaticPool([candidate(0, { vertexCount: 1.5 })], limits())).toThrow(/vertexCount/);
    expect(() => planSharedStaticPool([], limits({ batchSlots: 0 }))).toThrow(/batchSlots/);
    expect(() => planSharedStaticPool([], limits({ maxBytes: -1 }))).toThrow(/maxBytes/);
    expect(planSharedStaticPool([], limits()).batches).toEqual([]);
  });
});

describe('shared static pool packing', () => {
  it('stores triangles in BLAS leaf order and leaf records for every capacity slot', () => {
    expect(Array.from(packLeafOrderedTriangles(new Uint32Array([0, 1, 2, 2, 3, 0, 3, 4, 5]), new Uint32Array([2, 0, 1]), 6)))
      .toEqual([3, 4, 5, 0, 1, 2, 2, 3, 0]);
    expect(() => packLeafOrderedTriangles(new Uint32Array([0, 1, 2]), new Uint32Array([1]), 3)).toThrow('SHARED_STATIC_TRIANGLE_LAYOUT');
    expect(() => packLeafOrderedTriangles(new Uint32Array([0, 1, 9]), new Uint32Array([0]), 3)).toThrow('SHARED_STATIC_TRIANGLE_LAYOUT');
    const batch = { index: 7, capacity: 3, vertexCount: 10, nodeCount: 5, triangleCount: 4, leafBase: 20, nodeBase: 100, triangleBase: 30, positionBase: 40 };
    const words = packSharedLeaves(batch, [{ entityId: 0xffff_ffff, generation: 2 }, { entityId: 9, generation: 0 }]);
    expect(Array.from(words)).toEqual([
      7, 0, 0xffff_ffff, 2, 100, 30, 40, 5,
      7, 1, 9, 0, 105, 30, 50, 5,
      7, 2, 0, 0, 110, 30, 60, 0,
    ]);
    expect(() => packSharedLeaves(batch, Array(4).fill({ entityId: 1, generation: 0 }))).toThrow('SHARED_STATIC_LEAF_CAPACITY');
    expect(packSharedFlags(true, false)).toBe(SHARED_FLAG_MEMBER);
    expect(packSharedFlags(false, true)).toBe(SHARED_FLAG_DOUBLE_SIDED);
  });
  it('orders active leaves by Morton key with leaf-ID ties and keeps inactive slots explicit', () => {
    const box = (x: number) => ({ min: [x, 0, 0] as V3, max: [x, 0, 0] as V3 });
    const order = orderSharedLeaves(6, [box(5), null, box(1), box(5), null, box(3)]);
    expect(Array.from(order)).toEqual([2, 5, 0, 3, 0xffff_ffff, 0xffff_ffff]);
    expect(Array.from(orderSharedLeaves(2, [null, null]))).toEqual([0xffff_ffff, 0xffff_ffff]);
    expect(() => orderSharedLeaves(3, [null])).toThrow('SHARED_STATIC_ORDER_LENGTH');
    expect(sharedTopologyWords(1)).toBe(4 + 1 + 1);
    expect(sharedTopologyWords(8)).toBe(15 * 4 + 8 + 15);
  });
});

describe('shared static CPU traversal reference', () => {
  it('matches an independent brute force on every ray, tie, ignore, face policy, empty and non-member batch', () => {
    const { pool, reference } = testScene();
    const cases = rays(21);
    let hits = 0, ties = 0;
    for (const reversed of [false, true]) for (const ray of cases) {
      const traced = traceSharedPoolCpu(pool, ray, 1e9, reversed);
      const expected = nearestTriangleCpu(reference, ray, reversed);
      expect(traced.overflow).toBe(false);
      expect(traced.hit).toEqual(expected);
      if (expected) hits++;
      if (expected && reference.filter(triangle => triangle.batch !== expected.batch || triangle.instance !== expected.instance)
        .some(triangle => nearestTriangleCpu([triangle], ray, reversed)?.distance === expected.distance)) ties++;
    }
    expect(hits).toBeGreaterThan(400);
    expect(ties).toBeGreaterThan(100);
    // Non-member and empty batches never win.
    expect(reference.some(triangle => triangle.batch === 3 || triangle.batch === 4)).toBe(false);
  });
  it('resolves exact cross-batch and in-batch ties to the lowest batch, then slot, then BLAS order', () => {
    const { pool } = testScene();
    const trace = (origin: V3, direction: V3, ignore?: CpuRay['ignore'], reversed = false) =>
      traceSharedPoolCpu(pool, { origin, direction, tMin: 0, tMax: f(1e30), ignore }, 1e9, reversed).hit;
    // (0.75, 0.75) is on the quad diagonal: both BLAS triangles, both duplicate slots and batch 2 tie.
    expect(trace([0.75, 0.75, -5], [0, 0, 1])).toMatchObject({ batch: 1, instance: 0, distance: 5 });
    expect(trace([0.75, 0.75, -5], [0, 0, 1], { entityId: 200, generation: 0 })).toMatchObject({ batch: 1, instance: 1, distance: 5 });
    // Ignore matches entity and generation together.
    expect(trace([0.75, 0.75, -5], [0, 0, 1], { entityId: 200, generation: 1 })).toMatchObject({ batch: 1, instance: 0 });
    expect(trace([0.75, 0.75, 5], [0, 0, -1])).toMatchObject({ batch: 1, instance: 3, distance: 3 });
    expect(trace([2.5, 2.5, -5], [0, 0, 1])).toMatchObject({ batch: 2, instance: 0, distance: 5 });
    expect(trace([2.5, 2.5, -5], [0, 0, 1], { entityId: 300, generation: 0 })).toBeNull();
    // Reversed support makes the single-sided batch 5 eligible from below only.
    expect(trace([0.75, 0.75, -5], [0, 0, 1], undefined, true)).toMatchObject({ batch: 5, instance: 0, distance: 4.5 });
    expect(trace([0.75, 0.75, -5], [0, 0, 1])).not.toMatchObject({ batch: 5 });
  });
  it('charges exact work units and stops at the per-ray quota with an explicit overflow', () => {
    const { pool } = testScene();
    let overflowed = 0;
    for (const ray of rays(22).slice(0, 200)) {
      const full = traceSharedPoolCpu(pool, ray, 1e9);
      expect(full.workUnits).toBeGreaterThan(0);
      const exact = traceSharedPoolCpu(pool, ray, full.workUnits);
      expect(exact).toEqual(full);
      if (full.workUnits > 1) {
        const short = traceSharedPoolCpu(pool, ray, full.workUnits - 1);
        expect(short.overflow).toBe(true);
        expect(short.workUnits).toBe(full.workUnits - 1);
        overflowed++;
      }
    }
    expect(overflowed).toBeGreaterThan(150);
  });
  it('treats an all-inactive pool as an invalid TLAS root', () => {
    const { pool } = buildScene([{ ...soup(14, 6, 2), capacity: 2, translations: [], flags: MEMBER, entityBase: 1 }]);
    expect(pool.nodes[7]).toBe(0);
    expect(traceSharedPoolCpu(pool, { origin: [0, 0, -5], direction: [0, 0, 1], tMin: 0, tMax: f(1e30) }, 1e9))
      .toEqual({ hit: null, workUnits: 1, triangleTests: 0, overflow: false });
  });
});

describe('watertight ray-triangle crossing', () => {
  // Seeded LCG so failures reproduce.
  const lcg = (seed: number) => () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const add = (a: V3, b: V3, scale = 1): V3 => [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
  const unit = (a: V3): V3 => { const length = Math.hypot(...a); return [f(a[0] / length), f(a[1] / length), f(a[2] / length)]; };
  const frame = (next: () => number) => {
    const axisU = unit([next() - 0.5, next() - 0.5, next() - 0.5]);
    const axisV = unit([axisU[1] - axisU[2], axisU[2] - axisU[0], axisU[0] - axisU[1]]);
    return { axisU, axisV, normal: unit([axisU[1] * axisV[2] - axisU[2] * axisV[1], axisU[2] * axisV[0] - axisU[0] * axisV[2],
      axisU[0] * axisV[1] - axisU[1] * axisV[0]]) };
  };

  it('never lets a ray through the shared edge of two triangles with different base vertices', () => {
    const next = lcg(91);
    let rays = 0, both = 0;
    for (let quad = 0; quad < 400; quad++) {
      const center: V3 = [(next() - 0.5) * 100, (next() - 0.5) * 100, (next() - 0.5) * 20];
      const size = 0.01 + next() * 2, { axisU, axisV, normal } = frame(next);
      const corner = (x: number, y: number): V3 => add(add(center, axisU, x * size), axisV, y * size).map(f) as V3;
      const p = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
      for (let sample = 0; sample < 20; sample++) {
        // A float32 target on the shared diagonal p0-p2; the triangles start at different vertices.
        const t = next(), target = [0, 1, 2].map(axis => f(p[0][axis] + (p[2][axis] - p[0][axis]) * t)) as V3;
        const direction = unit([next() - 0.5, next() - 0.5, next() - 0.5]), distance = 1 + next() * 15;
        const ray: CpuRay = { origin: target.map((value, axis) => f(value - direction[axis] * distance)) as V3, direction, tMin: 0, tMax: 64 };
        const first = cpuIntersectTriangle(ray, p[0], p[1], p[2], true, false);
        const second = cpuIntersectTriangle(ray, p[2], p[3], p[0], true, false);
        expect(first ?? second, `quad ${quad} sample ${sample}`).not.toBeNull();
        for (const hit of [first, second]) {
          if (!hit) continue;
          expect(hit.u).toBeGreaterThanOrEqual(0); expect(hit.v).toBeGreaterThanOrEqual(0);
          expect(hit.u + hit.v).toBeLessThanOrEqual(1 + 1e-6);
          // float32 targets sit up to an ulp off the plane; grazing rays stretch that along the ray.
          const cosine = Math.abs(normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2]);
          expect(Math.abs(hit.distance - distance)).toBeLessThan(2e-5 * (100 + distance) / cosine);
        }
        if (first && second) both++;
        rays++;
      }
    }
    expect(rays).toBe(8000);
    expect(both).toBeLessThan(rays); // Mostly exactly one side: no tolerance band thickens edges.
  });

  it('closes every vertex of a triangle fan, exactly on the vertex and from arbitrary directions', () => {
    const next = lcg(7);
    for (let fan = 0; fan < 200; fan++) {
      const center: V3 = [f((next() - 0.5) * 100), f((next() - 0.5) * 100), f((next() - 0.5) * 20)];
      const sides = 3 + Math.floor(next() * 6), radius = 0.02 + next();
      const { axisU, axisV } = frame(next);
      const ring = Array.from({ length: sides }, (_, i) => {
        const angle = (i + next() * 0.5) / sides * Math.PI * 2;
        return add(add(center, axisU, Math.cos(angle) * radius), axisV, Math.sin(angle) * radius).map(f) as V3;
      });
      // Rotate each triangle's vertex order so shared edges appear in different roles.
      const triangles = ring.map((_, i) => [center, ring[i], ring[(i + 1) % sides]]).map((tri, i) => [0, 1, 2].map(k => tri[(k + i) % 3]));
      const hits = (ray: CpuRay) => triangles.filter(([a, b, c]) => cpuIntersectTriangle(ray, a, b, c, true, false)).length;
      // Exactly through the shared vertex: an axis ray whose other coordinates equal the vertex's.
      for (const axis of [0, 1, 2]) {
        const direction: V3 = [0, 0, 0]; direction[axis] = next() < 0.5 ? -1 : 1;
        const origin = [...center] as V3; origin[axis] = f(center[axis] - direction[axis] * 5);
        expect(hits({ origin, direction, tMin: 0, tMax: 64 }), `fan ${fan} axis ${axis}`).toBe(sides);
      }
      for (let sample = 0; sample < 20; sample++) {
        const direction = unit([next() - 0.5, next() - 0.5, next() - 0.5]), distance = 1 + next() * 15;
        const origin = center.map((value, axis) => f(value - direction[axis] * distance)) as V3;
        expect(hits({ origin, direction, tMin: 0, tMax: 64 }), `fan ${fan} sample ${sample}`).toBeGreaterThan(0);
      }
    }
  });

  it('still rejects rays a small fraction of the triangle outside its edges', () => {
    const a: V3 = [-30.15, 10.35, 2], b: V3 = [-29.45, 10.35, 2], c: V3 = [-29.45, 11.05, 2];
    const down: V3 = [0, 0, -1];
    const at = (x: number, y: number): CpuRay => ({ origin: [f(x), f(y), 7], direction: down, tMin: 0, tMax: 16 });
    expect(cpuIntersectTriangle(at(-29.8, 10.35 + 1e-4), a, b, c, true, false)).not.toBeNull();
    expect(cpuIntersectTriangle(at(-29.8, 10.35 - 1e-4), a, b, c, true, false)).toBeNull();
    expect(cpuIntersectTriangle(at(-29.45 + 1e-4, 10.7), a, b, c, true, false)).toBeNull();
    expect(cpuIntersectTriangle(at(-29.8 - 1e-4, 10.35 + 0.35), a, b, c, true, false)).toBeNull();
    // Exactly on an edge and a vertex counts as inside.
    expect(cpuIntersectTriangle(at(-29.8, 10.35), a, b, c, true, false)).not.toBeNull();
    expect(cpuIntersectTriangle(at(-29.45, 10.35), a, b, c, true, false)).not.toBeNull();
  });
});
