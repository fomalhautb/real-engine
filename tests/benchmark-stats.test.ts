import { describe, expect, it } from 'vitest';
import { BenchmarkRecorder, type BenchmarkFrame } from '../src/benchmark-stats';

function frame(id: number, overrides: Partial<BenchmarkFrame> = {}): BenchmarkFrame {
  return {
    frame: id,
    elapsedMs: id * 16,
    frameIntervalMs: id,
    updateMs: id / 2,
    eventMs: id / 4,
    cpuSubmitMs: id / 8,
    ownedGpuBytes: id * 10,
    uploadedBytes: id,
    triangles: id * 100,
    deformedVertices: id * 2,
    physicsBodies: id,
    ...overrides,
  };
}

describe('BenchmarkRecorder', () => {
  it('uses nearest-rank percentiles and keeps GPU timings separate from CPU timings', () => {
    const recorder = new BenchmarkRecorder(100, 16.67);
    for (let id = 1; id <= 100; id++) recorder.record(frame(id));
    recorder.recordGpu({ frame: 2, deformation: 2, shadow: 3, surface: 1 });
    recorder.recordGpu({ frame: 4, deformation: 4, shadow: 5, surface: 2 });
    const result = recorder.summary();
    expect(result.frameCount).toBe(100);
    expect(result.gpuSampleCount).toBe(2);
    expect(result.elapsedMs).toBe(99 * 16);
    expect(result.frameBudgetMs).toBe(16.67);
    expect(result.framesOverBudget).toBe(84);
    expect(result.overBudgetFraction).toBe(0.84);
    expect(result.frameIntervalMs).toEqual({ count: 100, mean: 50.5, p50: 50, p95: 95, p99: 99, max: 100 });
    expect(result.updateMs?.p95).toBe(47.5);
    expect(result.eventMs?.p99).toBe(24.75);
    expect(result.cpuSubmitMs?.max).toBe(12.5);
    expect(result.gpuMs.total).toEqual({ count: 2, mean: 8.5, p50: 6, p95: 11, p99: 11, max: 11 });
    expect(result.gpuMs.deformation?.mean).toBe(3);
    expect(result.gpuMs.resolve?.max).toBe(0);
    expect(result.gpuMs.worldQueryRefit?.max).toBe(0);
    expect(result.gpuMs.geometryLod?.max).toBe(0);
    expect(result.gpuMs.diffuseGi?.max).toBe(0);
    expect(result.gpuMs.finalGather?.max).toBe(0);
    expect(result.gpuMs.reflections?.max).toBe(0);
    expect(result.gpuMs.volumetricFog?.max).toBe(0);
    expect(result.uploadedBytesTotal).toBe(5050);
    expect(result.workload?.triangles).toEqual({ mean: 5050, max: 10000, last: 10000 });
  });

  it('reports memory extrema and consecutive third medians/growth', () => {
    const recorder = new BenchmarkRecorder(6, 20);
    for (let id = 1; id <= 6; id++) recorder.record(frame(id));
    expect(recorder.summary().ownedGpuBytes).toEqual({
      firstBytes: 10, lastBytes: 60, minBytes: 10, maxBytes: 60, growthBytes: 50,
      thirdMedianBytes: { first: 10, middle: 30, last: 50 }, thirdMedianGrowthBytes: 40,
    });
  });

  it('distinguishes no samples from measured zeros and emits stable sparse CSV rows', () => {
    const recorder = new BenchmarkRecorder(2, 16);
    const empty = recorder.summary();
    expect(empty.frameCount).toBe(0);
    expect(empty.framesOverBudget).toBe(0);
    expect(empty.overBudgetFraction).toBeNull();
    expect(empty.elapsedMs).toBeNull();
    expect(empty.frameIntervalMs).toBeNull();
    expect(empty.gpuMs.total).toBeNull();
    expect(empty.gpuMs.resolve).toBeNull();
    expect(empty.gpuMs.worldQueryRefit).toBeNull();
    expect(empty.gpuMs.geometryLod).toBeNull();
    expect(empty.gpuMs.diffuseGi).toBeNull();
    expect(empty.gpuMs.finalGather).toBeNull();
    expect(empty.gpuMs.reflections).toBeNull();
    expect(empty.gpuMs.volumetricFog).toBeNull();
    expect(empty.ownedGpuBytes).toBeNull();
    expect(empty.uploadedBytesTotal).toBeNull();
    recorder.record(frame(1, { elapsedMs: 0, frameIntervalMs: 0, updateMs: 0, eventMs: 0, cpuSubmitMs: 0,
      ownedGpuBytes: 0, uploadedBytes: 0, triangles: 0, deformedVertices: 0, physicsBodies: 0 }));
    recorder.record(frame(2));
    recorder.recordGpu({ frame: 2, deformation: 0, shadow: 1, surface: 2 });
    expect(recorder.summary().frameIntervalMs?.p50).toBe(0);
    expect(recorder.toCsv()).toBe(
      'frame,elapsedMs,frameIntervalMs,updateMs,eventMs,cpuSubmitMs,ownedGpuBytes,uploadedBytes,triangles,deformedVertices,physicsBodies,gpuDeformationMs,gpuShadowMs,gpuSurfaceMs,gpuResolveMs,gpuWorldQueryRefitMs,gpuGeometryLodMs,gpuDiffuseGiMs,gpuFinalGatherMs,gpuReflectionsMs,gpuVolumetricFogMs,gpuTotalMs\n'
      + '1,0,0,0,0,0,0,0,0,0,0,,,,,,,,,,,\n'
      + '2,32,2,1,0.5,0.25,20,2,200,4,2,0,1,2,0,0,0,0,0,0,0,3\n',
    );
  });

  it('includes resolve, world-query refit, geometry LOD and diffuse GI in sparse GPU totals, quantiles, and CSV', () => {
    const recorder = new BenchmarkRecorder(3, 16);
    recorder.record(frame(1));
    recorder.record(frame(2));
    recorder.record(frame(3));
    recorder.recordGpu({ frame: 1, deformation: 1, shadow: 2, surface: 3, resolve: 4, worldQueryRefit: 5, geometryLod: 6, diffuseGi: 7 });
    recorder.recordGpu({ frame: 3, deformation: 0, shadow: 0, surface: 1, resolve: 2, worldQueryRefit: 3, geometryLod: 4, diffuseGi: 2 });
    const { gpuMs } = recorder.summary();
    expect(gpuMs.resolve).toEqual({ count: 2, mean: 3, p50: 2, p95: 4, p99: 4, max: 4 });
    expect(gpuMs.worldQueryRefit).toEqual({ count: 2, mean: 4, p50: 3, p95: 5, p99: 5, max: 5 });
    expect(gpuMs.geometryLod).toEqual({ count: 2, mean: 5, p50: 4, p95: 6, p99: 6, max: 6 });
    expect(gpuMs.diffuseGi).toEqual({ count: 2, mean: 4.5, p50: 2, p95: 7, p99: 7, max: 7 });
    expect(gpuMs.total).toEqual({ count: 2, mean: 20, p50: 12, p95: 28, p99: 28, max: 28 });
    const rows = recorder.toCsv().trimEnd().split('\n');
    expect(rows[0]).toContain('gpuSurfaceMs,gpuResolveMs,gpuWorldQueryRefitMs,gpuGeometryLodMs,gpuDiffuseGiMs,gpuFinalGatherMs,gpuReflectionsMs,gpuVolumetricFogMs,gpuTotalMs');
    expect(rows[1]).toMatch(/,1,2,3,4,5,6,7,0,0,0,28$/);
    expect(rows[2].split(',').slice(11)).toEqual(Array(11).fill('')); // No GPU sample on frame 2.
    expect(rows[3]).toMatch(/,0,0,1,2,3,4,2,0,0,0,12$/);
  });

  it('includes final gather in the measured GPU total and CSV', () => {
    const recorder = new BenchmarkRecorder(1, 16);
    recorder.record(frame(1));
    recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, finalGather: 5 });
    expect(recorder.summary().gpuMs.finalGather?.max).toBe(5);
    expect(recorder.summary().gpuMs.total?.max).toBe(8);
    const [header, sample] = recorder.toCsv().trimEnd().split('\n');
    const columns = header.split(',');
    const values = sample.split(',');
    expect(values[columns.indexOf('gpuFinalGatherMs')]).toBe('5');
    expect(values[columns.indexOf('gpuTotalMs')]).toBe('8');
  });

  it('includes volumetric fog in GPU totals and preserves zero for older samples', () => {
    const recorder = new BenchmarkRecorder(2, 16);
    recorder.record(frame(1));
    recorder.record(frame(2));
    recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, volumetricFog: 5 });
    recorder.recordGpu({ frame: 2, deformation: 1, shadow: 1, surface: 1 });
    expect(recorder.summary().gpuMs.volumetricFog).toEqual({ count: 2, mean: 2.5,
      p50: 0, p95: 5, p99: 5, max: 5 });
    expect(recorder.summary().gpuMs.total).toEqual({ count: 2, mean: 5.5,
      p50: 3, p95: 8, p99: 8, max: 8 });
    const [header, first, second] = recorder.toCsv().trimEnd().split('\n');
    const fogColumn = header.split(',').indexOf('gpuVolumetricFogMs');
    expect(first.split(',')[fogColumn]).toBe('5');
    expect(second.split(',')[fogColumn]).toBe('0');
  });

  it('includes reflections in GPU totals and exports zero for older samples', () => {
    const recorder = new BenchmarkRecorder(2, 16);
    recorder.record(frame(1));
    recorder.record(frame(2));
    recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, reflections: 5 });
    recorder.recordGpu({ frame: 2, deformation: 1, shadow: 1, surface: 1 });
    expect(recorder.summary().gpuMs.reflections).toEqual({ count: 2, mean: 2.5,
      p50: 0, p95: 5, p99: 5, max: 5 });
    expect(recorder.summary().gpuMs.total).toEqual({ count: 2, mean: 5.5,
      p50: 3, p95: 8, p99: 8, max: 8 });
    const [header, first, second] = recorder.toCsv().trimEnd().split('\n');
    const column = header.split(',').indexOf('gpuReflectionsMs');
    expect(first.split(',')[column]).toBe('5');
    expect(second.split(',')[column]).toBe('0');
  });

  it('rejects invalid limits, frames, GPU samples, and byte-total overflow', () => {
    expect(() => new BenchmarkRecorder(0, 16)).toThrow(/maxFrames/);
    expect(() => new BenchmarkRecorder(1_000_001, 16)).toThrow(/maxFrames/);
    expect(() => new BenchmarkRecorder(1, Infinity)).toThrow(/frameBudgetMs/);
    const recorder = new BenchmarkRecorder(2, 16);
    expect(() => recorder.record(frame(1, { updateMs: NaN }))).toThrow(/updateMs/);
    expect(() => recorder.record(frame(1, { triangles: -1 }))).toThrow(/triangles/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1 })).toThrow(/recorded frame/);
    recorder.record(frame(1));
    expect(() => recorder.record(frame(1))).toThrow(/strictly increase/);
    expect(() => recorder.record(frame(2, { elapsedMs: 16 }))).toThrow(/strictly increase/);
    expect(() => recorder.recordGpu({ frame: 2, deformation: 1, shadow: 1, surface: 1 })).toThrow(/recorded frame/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, resolve: -1 })).toThrow(/GPU resolve/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, worldQueryRefit: Infinity })).toThrow(/GPU worldQueryRefit/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, geometryLod: -1 })).toThrow(/GPU geometryLod/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, geometryLod: Infinity })).toThrow(/GPU geometryLod/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, diffuseGi: -1 })).toThrow(/GPU diffuseGi/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, diffuseGi: Infinity })).toThrow(/GPU diffuseGi/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, finalGather: -1 })).toThrow(/GPU finalGather/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, reflections: -1 })).toThrow(/GPU reflections/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, reflections: Infinity })).toThrow(/GPU reflections/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, volumetricFog: -1 })).toThrow(/GPU volumetricFog/);
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1, volumetricFog: Infinity })).toThrow(/GPU volumetricFog/);
    recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1 });
    expect(() => recorder.recordGpu({ frame: 1, deformation: 1, shadow: 1, surface: 1 })).toThrow(/strictly increase/);
    recorder.record(frame(2));
    expect(() => recorder.record(frame(3))).toThrow(/frame limit/);
    const bytes = new BenchmarkRecorder(2, 16);
    bytes.record(frame(1, { uploadedBytes: Number.MAX_SAFE_INTEGER }));
    expect(() => bytes.record(frame(2, { uploadedBytes: 1 }))).toThrow(/safe integer/);
    expect(bytes.frameCount).toBe(1);
  });
});
