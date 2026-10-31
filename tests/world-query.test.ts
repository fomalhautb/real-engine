import { describe, expect, it } from 'vitest';
import {
  WORLD_HIT_BYTES, WORLD_RAY_BYTES, decodeWorldHits, packWorldRays, resolveWorldQueryOptions,
} from '../src/world-query';

const littleEndian = true;

function hitRecord(): ArrayBuffer {
  const data = new ArrayBuffer(WORLD_HIT_BYTES);
  const view = new DataView(data);
  const u = (word: number, value: number): void => view.setUint32(word * 4, value, littleEndian);
  const f = (word: number, value: number): void => view.setFloat32(word * 4, value, littleEndian);
  u(0, 2); u(1, 42); u(2, 3); u(3, 7);
  f(4, 5); f(5, 0.25); f(6, 0.5); f(7, 1);
  f(8, 1); f(9, 2); f(10, 3);
  f(12, 0); f(13, 1); f(14, 0); f(15, 0.4);
  f(16, 0.5); f(17, 2); f(18, 3); f(19, 0.6);
  f(20, 0.2); f(21, 0.3); f(22, 0.4); f(23, 0.8);
  f(24, 2); f(25, 1); f(26, 0); f(27, 1);
  u(28, 19); u(29, 23); u(30, 29); u(31, 31);
  f(32, 0.75); f(33, 0.125); f(34, 2); f(35, 1);
  u(36, 14); u(37, 6);
  return data;
}

describe('world query wire contract', () => {
  it('decodes independent geometric normals from reserved octahedral lanes without expanding the hit', () => {
    const cases = [
      { encoded: [0, 0], expected: [0, 0, 1] },
      { encoded: [1, 0], expected: [1, 0, 0] },
      { encoded: [0, 1], expected: [0, 1, 0] },
      { encoded: [-1, 0], expected: [-1, 0, 0] },
      { encoded: [0, -1], expected: [0, -1, 0] },
      { encoded: [1, 1], expected: [0, 0, -1] },
      { encoded: [-0.5, 0.5], expected: [-Math.SQRT1_2, Math.SQRT1_2, 0] },
    ];
    for (const { encoded, expected } of cases) {
      const data = hitRecord(), view = new DataView(data);
      view.setFloat32(11 * 4, encoded[0], true);
      view.setFloat32(35 * 4, encoded[1], true);
      const hit = decodeWorldHits(data, ['ground', 'character'], 1, 19, 31).hits[0]!;
      expect(data.byteLength).toBe(160);
      expect(hit.normal).toEqual([0, 1, 0]);
      expected.forEach((value, axis) => expect(hit.geometricNormal[axis]).toBeCloseTo(value, 7));
    }
    for (const invalid of [1.01, -1.01, Infinity, NaN]) {
      const data = hitRecord();
      new DataView(data).setFloat32(11 * 4, invalid, true);
      expect(() => decodeWorldHits(data, ['ground', 'character'], 1, 19, 31)).toThrow(/geometricNormal|finite/);
    }
  });
  it('resolves defaults and rejects invalid or excessive allocations', () => {
    expect(resolveWorldQueryOptions()).toEqual({
      maxRays: 1024, maxBytes: 128 * 1024 * 1024, maxTrianglesPerMesh: 65536,
      maxInstancesPerBatch: 256, maxBatches: 256, maxNodeVisits: 262144, maxTotalWorkUnits: 16777216,
      maxTraceParameterSetsPerEncoder: 8, instanceAcceleration: 'bvh', sceneAcceleration: 'bvh',
      maxSharedStaticBytes: 64 * 1024 * 1024, staticTraversal: 'per-batch', dispatchEncoding: 'pass-per-batch',
    });
    expect(resolveWorldQueryOptions({ staticTraversal: 'shared' }).staticTraversal).toBe('shared');
    expect(resolveWorldQueryOptions({ dispatchEncoding: 'single-pass' }).dispatchEncoding).toBe('single-pass');
    expect(resolveWorldQueryOptions({ maxSharedStaticBytes: 512 * 1024 * 1024 }).maxSharedStaticBytes).toBe(512 * 1024 * 1024);
    expect(() => resolveWorldQueryOptions({ staticTraversal: 'pooled' as 'shared' })).toThrow(/staticTraversal/);
    expect(() => resolveWorldQueryOptions({ dispatchEncoding: 'one' as 'single-pass' })).toThrow(/dispatchEncoding/);
    for (const invalid of [0, 1.5, 512 * 1024 * 1024 + 1, NaN]) {
      expect(() => resolveWorldQueryOptions({ maxSharedStaticBytes: invalid })).toThrow(/maxSharedStaticBytes/);
    }
    expect(resolveWorldQueryOptions({ maxRays: 2048, maxInstancesPerBatch: 16 }).maxRays).toBe(2048);
    for (const maxBatches of [1, 31, 32, 33, 256, 405, 4096]) {
      expect(resolveWorldQueryOptions({ maxBatches }).maxBatches).toBe(maxBatches);
    }
    for (const maxBatches of [0, -1, 4097, 1.5, Infinity, NaN]) {
      expect(() => resolveWorldQueryOptions({ maxBatches })).toThrow(/maxBatches/);
    }
    expect(resolveWorldQueryOptions({ instanceAcceleration: 'linear' }).instanceAcceleration).toBe('linear');
    expect(() => resolveWorldQueryOptions({ instanceAcceleration: 'unknown' as 'bvh' })).toThrow(/instanceAcceleration/);
    expect(resolveWorldQueryOptions({ sceneAcceleration: 'linear' }).sceneAcceleration).toBe('linear');
    expect(() => resolveWorldQueryOptions({ sceneAcceleration: 'unknown' as 'bvh' })).toThrow(/sceneAcceleration/);
    expect(resolveWorldQueryOptions({ maxTotalWorkUnits: 67108864 }).maxTotalWorkUnits).toBe(67108864);
    expect(resolveWorldQueryOptions({ maxTraceParameterSetsPerEncoder: 1 }).maxTraceParameterSetsPerEncoder).toBe(1);
    expect(resolveWorldQueryOptions({ maxTraceParameterSetsPerEncoder: 32 }).maxTraceParameterSetsPerEncoder).toBe(32);
    expect(() => resolveWorldQueryOptions({ maxRays: 0 })).toThrow(/maxRays/);
    expect(() => resolveWorldQueryOptions({ maxRays: 65537 })).toThrow(/maxRays/);
    expect(() => resolveWorldQueryOptions({ maxBytes: 512 * 1024 * 1024 + 1 })).toThrow(/maxBytes/);
    expect(() => resolveWorldQueryOptions({ maxTrianglesPerMesh: 1.5 })).toThrow(/maxTrianglesPerMesh/);
    expect(() => resolveWorldQueryOptions({ maxInstancesPerBatch: Infinity })).toThrow(/maxInstancesPerBatch/);
    expect(() => resolveWorldQueryOptions({ maxNodeVisits: -1 })).toThrow(/maxNodeVisits/);
    expect(() => resolveWorldQueryOptions({ maxTotalWorkUnits: 0 })).toThrow(/maxTotalWorkUnits/);
    expect(() => resolveWorldQueryOptions({ maxTotalWorkUnits: 67108865 })).toThrow(/maxTotalWorkUnits/);
    for (const invalid of [0, 33, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => resolveWorldQueryOptions({ maxTraceParameterSetsPerEncoder: invalid }))
        .toThrow(/maxTraceParameterSetsPerEncoder/);
    }
  });

  it('packs normalized rays, defaults, and optional ignore identity in 48-byte records', () => {
    const data = packWorldRays([
      { origin: [1, -2, 3], direction: [0, 3, 4] },
      { origin: [5, 6, 7], direction: [-2, 0, 0], tMin: 0, tMax: 20,
        ignore: { entityId: 0xffff_ffff, generation: 7 } },
    ], 2);
    expect(data.byteLength).toBe(2 * WORLD_RAY_BYTES);
    const view = new DataView(data);
    expect([0, 4, 8].map((offset) => view.getFloat32(offset, littleEndian))).toEqual([1, -2, 3]);
    expect(view.getFloat32(12, littleEndian)).toBeCloseTo(0.0001, 8);
    expect(view.getFloat32(16, littleEndian)).toBe(0);
    expect(view.getFloat32(20, littleEndian)).toBeCloseTo(0.6, 6);
    expect(view.getFloat32(24, littleEndian)).toBeCloseTo(0.8, 6);
    expect(view.getFloat32(28, littleEndian)).toBe(Math.fround(1e30));
    expect([32, 36, 40, 44].map((offset) => view.getUint32(offset, littleEndian))).toEqual([0, 0, 0, 0]);
    expect(view.getFloat32(48 + 16, littleEndian)).toBe(-1);
    expect(view.getFloat32(48 + 28, littleEndian)).toBe(20);
    expect([32, 36, 40, 44].map((offset) => view.getUint32(48 + offset, littleEndian)))
      .toEqual([0xffff_ffff, 7, 1, 0]);
    expect(packWorldRays([], 1).byteLength).toBe(0);
  });

  it('rejects rays that would overflow, underflow, collapse bounds, or exceed budgets', () => {
    const ray = { origin: [0, 0, 0] as [number, number, number], direction: [1, 0, 0] as [number, number, number] };
    expect(() => packWorldRays([ray, ray], 1)).toThrow(/BUDGET/);
    expect(() => packWorldRays([ray], 0)).toThrow(/maxRays/);
    expect(() => packWorldRays([{ ...ray, direction: [0, 0, 0] }], 1)).toThrow(/nonzero/);
    expect(() => packWorldRays([{ ...ray, origin: [Infinity, 0, 0] }], 1)).toThrow(/finite/);
    expect(() => packWorldRays([{ ...ray, origin: [Number.MAX_VALUE, 0, 0] }], 1)).toThrow(/float32/);
    expect(() => packWorldRays([{ ...ray, direction: [1e-50, 0, 0] }], 1)).toThrow(/float32/);
    expect(() => packWorldRays([{ ...ray, tMin: -1 }], 1)).toThrow(/tMin/);
    expect(() => packWorldRays([{ ...ray, tMin: 1, tMax: 1 + Number.EPSILON }], 1)).toThrow(/tMax/);
    expect(() => packWorldRays([{ ...ray, ignore: { entityId: -1 } }], 1)).toThrow(/entityId/);
    expect(() => packWorldRays([{ ...ray, ignore: { entityId: 1, generation: 2 ** 32 } }], 1)).toThrow(/generation/);
  });

  it('decodes a full hit and a versioned miss without losing material or history data', () => {
    const data = new ArrayBuffer(2 * WORLD_HIT_BYTES);
    new Uint8Array(data).set(new Uint8Array(hitRecord()));
    const miss = new DataView(data, WORLD_HIT_BYTES);
    miss.setUint32(28 * 4, 19, littleEndian);
    miss.setUint32(31 * 4, 31, littleEndian);
    miss.setUint32(36 * 4, 12, littleEndian);
    const result = decodeWorldHits(data, ['ground', 'character'], 2, 19, 31);
    expect(result.frame).toBe(19);
    expect(result.materialRevision).toBe(31);
    expect(result.hits[0]).toEqual({
      batch: 'character', geometry: { kind: 'source' }, entityId: 42, entityGeneration: 3, triangleIndex: 7, distance: 5,
      position: [1, 2, 3], normal: [0, 1, 0], geometricNormal: [0, 1, 0], previousPosition: [0.5, 2, 3],
      uv: [0.75, 0.125], baseColor: [Math.fround(0.2), Math.fround(0.3), Math.fround(0.4)],
      opacity: Math.fround(0.8), metallic: Math.fround(0.6), roughness: Math.fround(0.4),
      emissive: [2, 1, 0], shading: 'unlit', historyVersion: 23, topologyVersion: 29,
    });
    expect(result.hits[1]).toBeNull();
    expect(decodeWorldHits(new ArrayBuffer(0), [], 0, 0, 0).hits).toEqual([]);
  });

  it('labels proxy triangle numbering and clones geometry metadata for each hit', () => {
    const data = new ArrayBuffer(2 * WORLD_HIT_BYTES);
    new Uint8Array(data).set(new Uint8Array(hitRecord()));
    new Uint8Array(data, WORLD_HIT_BYTES).set(new Uint8Array(hitRecord()));
    const geometry = { kind: 'fixed-query-proxy' as const, primitiveKey: 'asset/primitive',
      directorySha256: 'a'.repeat(64), topologyVersion: 12, approximateObjectSpaceError: 0.25,
      errorMetric: 'accumulated-simplification' as const };
    const hits = decodeWorldHits(data, ['ground', 'character'], 2, 19, 31,
      [{ kind: 'source' }, geometry]).hits;
    expect(hits.map(hit => hit?.geometry)).toEqual([geometry, geometry]);
    expect(hits[0]?.geometry).not.toBe(geometry);
    expect(hits[1]?.geometry).not.toBe(geometry);
    expect(hits[0]?.geometry).not.toBe(hits[1]?.geometry);
    expect(hits[0]?.triangleIndex).toBe(7);
    expect(hits[0]?.topologyVersion).toBe(29);
    expect(() => decodeWorldHits(data, ['ground', 'character'], 2, 19, 31, [geometry]))
      .toThrow(/geometry metadata/);
  });

  it('preserves finite nonnegative Unlit HDR color while keeping PBR reflectance bounded', () => {
    const hdr = hitRecord();
    const view = new DataView(hdr);
    view.setFloat32(20 * 4, 4.5, littleEndian);
    view.setFloat32(21 * 4, 1.25, littleEndian);
    view.setFloat32(22 * 4, 0, littleEndian);
    expect(decodeWorldHits(hdr, ['ground', 'character'], 1, 19, 31).hits[0]?.baseColor)
      .toEqual([4.5, 1.25, 0]);
    view.setFloat32(27 * 4, 0, littleEndian);
    expect(() => decodeWorldHits(hdr, ['ground', 'character'], 1, 19, 31)).toThrow(/baseColor/);
    view.setFloat32(27 * 4, 1, littleEndian);
    view.setFloat32(20 * 4, -0.1, littleEndian);
    expect(() => decodeWorldHits(hdr, ['ground', 'character'], 1, 19, 31)).toThrow(/baseColor/);
  });

  it('rejects corrupt identity, nonfinite payloads, stale versions, and traversal overflow', () => {
    expect(() => decodeWorldHits(new ArrayBuffer(4), [], 1, 19, 31)).toThrow(/byte length/);
    const badBatch = hitRecord();
    expect(() => decodeWorldHits(badBatch, ['only one'], 1, 19, 31)).toThrow(/batch ID/);
    const overflow = hitRecord();
    new DataView(overflow).setUint32(38 * 4, 1, littleEndian);
    expect(() => decodeWorldHits(overflow, ['ground', 'character'], 1, 19, 31)).toThrow(/WORLD_QUERY_WORK_BUDGET/);
    const nonfinite = hitRecord();
    new DataView(nonfinite).setFloat32(8 * 4, NaN, littleEndian);
    expect(() => decodeWorldHits(nonfinite, ['ground', 'character'], 1, 19, 31)).toThrow(/finite/);
    const invalidMaterial = hitRecord();
    new DataView(invalidMaterial).setFloat32(23 * 4, 1.5, littleEndian);
    expect(() => decodeWorldHits(invalidMaterial, ['ground', 'character'], 1, 19, 31)).toThrow(/opacity/);
    const invalidNormal = hitRecord();
    new DataView(invalidNormal).setFloat32(13 * 4, 2, littleEndian);
    expect(() => decodeWorldHits(invalidNormal, ['ground', 'character'], 1, 19, 31)).toThrow(/unit length/);
    const stale = hitRecord();
    expect(() => decodeWorldHits(stale, ['ground', 'character'], 1, 20, 31)).toThrow(/stale/);
    const missWithIdentity = hitRecord();
    new DataView(missWithIdentity).setUint32(0, 0, littleEndian);
    expect(() => decodeWorldHits(missWithIdentity, ['ground', 'character'], 1, 19, 31)).toThrow(/miss identity/);
  });
});
