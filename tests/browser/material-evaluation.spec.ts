import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import type { runMaterialEvaluationFixture } from './fixtures/material-evaluation';

test('material IR matches its CPU reference in a compute backend with explicit texture LOD', async ({ page }, info) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') console.log('MATERIAL_COMPUTE_BROWSER_ERROR ' + message.text()); });
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const module = await import('/tests/browser/fixtures/' + 'material-evaluation.ts') as {
      runMaterialEvaluationFixture: typeof runMaterialEvaluationFixture;
    };
    return module.runMaterialEvaluationFixture();
  });
  await writeFile(info.outputPath('material-compute.json'), JSON.stringify(result, null, 2));
  console.log('MATERIAL_COMPUTE_DIAGNOSTIC ' + JSON.stringify({ adapter: result.adapterInfo,
    features: result.deviceFeatures, rows: result.rows.length, unavailable: result.unavailable,
    maxError: Math.max(...result.rows.map(row => row.maxError)), errors: result.errors,
    texturePreparation: result.texturePreparation }));
  const originalNames = [
    'expression-mask-equality', 'expression-mask-parameter-update',
    'expression-clamped-alpha-cutoff', 'expression-mask-cutoff-above-one',
    'unlit-hdr-opaque-zero-alpha', 'srgb-128', 'linear-128',
    'srgb-mip-zero', 'srgb-mip-one', 'per-slot-sampler-1.25', 'per-slot-sampler--0.25',
  ];
  expect(result.rows.slice(0, 11).map(row => row.name)).toEqual(originalNames);
  expect(result.adapterInfo).toEqual(expect.objectContaining({ vendor: expect.any(String),
    architecture: expect.any(String), device: expect.any(String), description: expect.any(String) }));
  expect(result.adapterFeatures).toEqual(expect.any(Array));
  expect(result.deviceFeatures).toEqual(expect.any(Array));
  const compressed = [
    { format: 'bc7', feature: 'texture-compression-bc', opaque: [128, 64, 32, 254],
      mask: [128, 0, 0, 64], last: [0, 128, 0, 254] },
    { format: 'etc2-rgba8', feature: 'texture-compression-etc2', opaque: [138, 70, 36, 255],
      mask: [138, 70, 36, 64], last: [138, 70, 36, 192] },
    { format: 'astc-4x4', feature: 'texture-compression-astc', opaque: [64, 128, 192, 255],
      mask: [64, 128, 192, 64], last: [192, 64, 32, 255] },
  ] as const;
  const available = compressed.filter(entry => result.deviceFeatures.includes(entry.feature));
  const swiftshader = /swiftshader/i.test(result.adapterInfo.architecture + ' ' + result.adapterInfo.description);
  expect(result.texturePreparation).toEqual({ mode: swiftshader ? 'swiftshader-render' : 'none',
    imagesPrepared: swiftshader ? 4 * available.length : 0,
    ownedGpuBytes: swiftshader && available.length ? 4 : 0 });
  expect(result.rows).toHaveLength(11 + 12 + 4 * available.length);
  expect(result.unavailable).toHaveLength(3 - available.length);
  for (const entry of compressed) {
    expect(result.adapterFeatures.includes(entry.feature)).toBe(result.deviceFeatures.includes(entry.feature));
    const failure = result.unavailable.find(value => value.format === entry.format);
    if (available.includes(entry)) {
      expect(failure).toBeUndefined();
    } else {
      expect(failure).toMatchObject({ feature: entry.feature, unchanged: true });
      expect(failure?.error).toContain('MATERIAL_TEXTURE_FEATURE');
    }
  }
  for (const row of result.rows) {
    expect(row.gpu, row.name).toHaveLength(16);
    expect(row.gpu.every(Number.isFinite), row.name).toBe(true);
    expect(row.maxError, row.name + ' CPU/GPU parity').toBeLessThan(row.tolerance);
    expect(row.gpu.slice(11), row.name + ' output padding').toEqual([0, 0, 0, 0, 0]);
    expect(row.gpu[7], row.name + ' exact coverage predicate').toBe(row.cpu[7]);
  }
  const row = (name: string) => result.rows.find(candidate => candidate.name === name)!.gpu;
  const equality = row('expression-mask-equality');
  expect(equality.slice(0, 3)).toEqual([1, 0.375, 0]);
  expect(equality[3]).toBe(0.5);
  expect(equality[4]).toBe(1);
  expect(equality[5]).toBeCloseTo(0.04, 6);
  expect(equality[7]).toBe(1);
  expect(equality.slice(8, 11)).toEqual([2, 3, 2.5]);
  expect(row('expression-mask-parameter-update')[7]).toBe(0);
  expect(row('expression-clamped-alpha-cutoff')[3]).toBe(0);
  expect(row('expression-clamped-alpha-cutoff')[6]).toBe(0);
  expect(row('expression-clamped-alpha-cutoff')[7]).toBe(1);
  expect(row('expression-mask-cutoff-above-one')[3]).toBe(1);
  expect(row('expression-mask-cutoff-above-one')[6]).toBe(1.25);
  expect(row('expression-mask-cutoff-above-one')[7]).toBe(0);
  expect(row('unlit-hdr-opaque-zero-alpha').slice(0, 4)).toEqual([0, 3, 9, 0]);
  expect(row('unlit-hdr-opaque-zero-alpha')[7]).toBe(1);
  expect(row('unlit-hdr-opaque-zero-alpha').slice(8, 11)).toEqual([4, 0, 2]);

  const srgbExpected = ((128 / 255 + 0.055) / 1.055) ** 2.4;
  expect(row('srgb-128')[0]).toBeCloseTo(srgbExpected, 2);
  expect(row('linear-128')[0]).toBeCloseTo(128 / 255, 2);
  expect(row('linear-128')[0] - row('srgb-128')[0]).toBeGreaterThan(0.25);
  expect(row('srgb-128')[3]).toBeCloseTo(128 / 255, 2); // Alpha is not sRGB-decoded.
  expect(row('srgb-mip-zero')[0]).toBeLessThan(0.004);
  expect(Math.abs(row('srgb-mip-one')[0] - 0.5)).toBeLessThan(0.007);
  for (const u of [1.25, -0.25]) {
    expect(row('per-slot-sampler-' + u).slice(0, 3)).toEqual([1, 1, 0]);
    expect(row('per-slot-sampler-' + u).slice(8, 11)).toEqual([1, 1, 0]);
  }
  expect(result.baselinePipelineCompilations).toBe(4);
  expect(result.pipelineCompilations).toBeGreaterThanOrEqual(4);
  expect(result.parameterUpdate.sameProgram).toBe(true);
  expect(result.parameterUpdate.compilationsAfter).toBe(result.parameterUpdate.compilationsBefore);
  expect(result.parameterUpdate.bytesAfter).toBe(result.parameterUpdate.bytesBefore);
  const expectedByte = (value: number) => value / 255;
  const srgbByte = (value: number) => value / 255 <= 0.04045
    ? value / 255 / 12.92 : ((value / 255 + 0.055) / 1.055) ** 2.4;
  const expectSample = (name: string, rgba: readonly number[], srgb = false) => {
    const actual = row(name);
    for (let channel = 0; channel < 4; channel++) {
      const expected = srgb && channel < 3 ? srgbByte(rgba[channel]) : expectedByte(rgba[channel]);
      expect(Math.abs(actual[channel] - expected), `${name} channel ${channel}`).toBeLessThan(0.012);
    }
  };
  const mipColors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 64]];
  for (let lod = 0; lod < 4; lod++) {
    expectSample(`prepared-rgba-full-${lod}`, mipColors[lod]);
    expectSample(`prepared-rgba-tail-${lod}`, mipColors[Math.max(1, lod)]);
  }
  expectSample('prepared-rgba-tail-min-clamp', mipColors[2]);
  expectSample('prepared-rgba-tail-max-clamp', mipColors[1]);
  expectSample('prepared-rgba-before-parameter-update', mipColors[3]);
  expect(row('prepared-rgba-after-parameter-update')[0]).toBeCloseTo(0.5, 2);
  expect(row('prepared-rgba-after-parameter-update')[1]).toBeCloseTo(0.25, 2);
  expect(row('prepared-rgba-after-parameter-update')[2]).toBeLessThan(0.005);
  expect(row('prepared-rgba-after-parameter-update')[3]).toBeCloseTo(64 / 255, 2);
  for (const entry of available) {
    expectSample(`prepared-${entry.format}-4x4`, entry.opaque);
    expectSample(`prepared-${entry.format}-2x2`, entry.mask);
    expectSample(`prepared-${entry.format}-1x1`, entry.last);
    expectSample(`prepared-${entry.format}-srgb`, entry.opaque, true);
    expect(row(`prepared-${entry.format}-4x4`)[7]).toBe(1);
    expect(row(`prepared-${entry.format}-2x2`)[7]).toBe(0);
  }
  expect(result.released).toEqual({ materials: 0, images: 0, imageBytes: 0 });
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
