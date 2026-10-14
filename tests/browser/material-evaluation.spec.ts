import { expect, test } from '@playwright/test';
import type { runMaterialEvaluationFixture } from './fixtures/material-evaluation';

test('material IR matches its CPU reference in a compute backend with explicit texture LOD', async ({ page }) => {
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
  console.log('MATERIAL_COMPUTE_DIAGNOSTIC ' + JSON.stringify(result));
  expect(result.rows).toHaveLength(11);
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
  expect(result.pipelineCompilations).toBe(4);
  expect(result.parameterUpdate.sameProgram).toBe(true);
  expect(result.parameterUpdate.compilationsAfter).toBe(result.parameterUpdate.compilationsBefore);
  expect(result.parameterUpdate.bytesAfter).toBe(result.parameterUpdate.bytesBefore);
  expect(result.released).toEqual({ materials: 0, images: 0, imageBytes: 0 });
  expect(result.ownedGpuBytesAfterDispose).toBe(0);
  expect(result.errors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
