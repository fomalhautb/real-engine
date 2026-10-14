import { writeFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type { runGiEmissiveFixture, runGiEmissiveOverflowFixture, runGiEmissiveBlocksFixture } from './fixtures/gi-emissive';

type Result = Awaited<ReturnType<typeof runGiEmissiveFixture>>;
type OverflowResult = Awaited<ReturnType<typeof runGiEmissiveOverflowFixture>>;
type ProbeSet = Result['initial'];
type Vec3 = readonly [number, number, number];

/** Independent midpoint area integral. For each surface point,
 * dOmega = cos(emitter normal, emitter-to-probe) dArea / distance squared.
 * The literal real SH basis is stated here rather than imported from the GI code. */
function areaReference(grid: Result['grid'], width: number, height: number,
  centerY: number, radiance: Vec3): number[][][] {
  const side = 64;
  const cellArea = width * height / (side * side);
  const probes: number[][][] = [];
  for (let probe = 0; probe < 8; probe++) {
    const position: Vec3 = [grid.origin[0] + (probe % 2) * grid.spacing[0],
      grid.origin[1] + (Math.floor(probe / 2) % 2) * grid.spacing[1],
      grid.origin[2] + Math.floor(probe / 4) * grid.spacing[2]];
    const coefficients = Array.from({ length: 9 }, () => [0, 0, 0]);
    for (let iy = 0; iy < side; iy++) for (let ix = 0; ix < side; ix++) {
      const surfaceX = ((ix + 0.5) / side - 0.5) * width;
      const surfaceY = centerY + ((iy + 0.5) / side - 0.5) * height;
      const dx = surfaceX - position[0], dy = surfaceY - position[1], dz = -2 - position[2];
      const distanceSquared = dx * dx + dy * dy + dz * dz;
      const inverseDistance = 1 / Math.sqrt(distanceSquared);
      const x = dx * inverseDistance, y = dy * inverseDistance, z = dz * inverseDistance;
      // The plane normal is +Z. Probe-to-emitter z is negative, so -z is
      // the positive emitter-to-probe cosine on this side of the plane.
      const solidAngle = Math.max(0, -z) * cellArea / distanceSquared;
      const basis = [0.28209479, 0.48860251 * y, 0.48860251 * z, 0.48860251 * x,
        1.09254843 * x * y, 1.09254843 * y * z,
        0.31539157 * (3 * z * z - 1), 1.09254843 * x * z,
        0.546274215 * (x * x - y * y)];
      for (let term = 0; term < 9; term++) for (let channel = 0; channel < 3; channel++) {
        coefficients[term][channel] += radiance[channel] * basis[term] * solidAngle;
      }
    }
    probes.push(coefficients);
  }
  return probes;
}

function coefficients(snapshot: ProbeSet): number[][][] {
  expect(snapshot.probes).toHaveLength(8);
  expect(snapshot.invalidProbes).toBe(0);
  return snapshot.probes.map((probe, index) => {
    expect(probe.valid, `probe ${index}`).toBe(true);
    expect(probe.primaryOverflows, `probe ${index} primary`).toBe(0);
    expect(probe.shadowOverflows, `probe ${index} shadow`).toBe(0);
    expect(probe.emissiveOverflows, `probe ${index} emissive`).toBe(0);
    expect(probe.coefficients).toHaveLength(9);
    expect(probe.coefficients.flat().every(Number.isFinite), `probe ${index} finite`).toBe(true);
    return probe.coefficients;
  });
}

function relativeRms(actual: readonly number[][][], reference: readonly number[][][]): number {
  let error = 0, signal = 0;
  for (let probe = 0; probe < reference.length; probe++) {
    for (let term = 0; term < 9; term++) for (let channel = 0; channel < 3; channel++) {
      const expected = reference[probe][term][channel];
      const delta = actual[probe][term][channel] - expected;
      error += delta * delta;
      signal += expected * expected;
    }
  }
  expect(signal).toBeGreaterThan(1e-8);
  return Math.sqrt(error / signal);
}

function maximumMagnitude(values: readonly number[][][]): number {
  return Math.max(...values.flat(2).map(Math.abs));
}

/** Establish that the large control occupies regular GI ray directions.
 * This audits the authored scene; the radiometric reference above has no ray sampling. */
function regularSphereRayHits(grid: Result['grid'], width: number, height: number): number[] {
  const hits: number[] = [];
  for (let probe = 0; probe < 8; probe++) {
    let probeHits = 0;
    const px = grid.origin[0] + (probe % 2) * grid.spacing[0];
    const py = grid.origin[1] + (Math.floor(probe / 2) % 2) * grid.spacing[1];
    const pz = grid.origin[2] + Math.floor(probe / 4) * grid.spacing[2];
    for (let ray = 0; ray < grid.raysPerProbe; ray++) {
      const y = 1 - 2 * (ray + 0.5) / grid.raysPerProbe;
      const radius = Math.sqrt(1 - y * y), phase = ray * 2.399963229728653;
      const dx = Math.cos(phase) * radius, dz = Math.sin(phase) * radius;
      if (dz >= 0) continue;
      const travel = (-2 - pz) / dz;
      if (Math.abs(px + travel * dx) < width / 2 &&
          Math.abs(py + travel * y - 1) < height / 2) probeHits++;
    }
    hits.push(probeHits);
  }
  return hits;
}

test('emissive area samples match an independent SH integral and update with scene state', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'gi-emissive.ts') as {
      runGiEmissiveFixture: typeof runGiEmissiveFixture;
    };
    return fixture.runGiEmissiveFixture();
  }) as Result;

  const initial = coefficients(result.initial);
  const enlarged = coefficients(result.enlarged);
  const reflective = coefficients(result.reflective);
  const occluded = coefficients(result.occluded);
  const masked = coefficients(result.masked);
  const doubled = coefficients(result.doubled);
  const moved = coefficients(result.moved);
  const removed = coefficients(result.removed);
  const smallReference = areaReference(result.grid, ...result.tinySize, 1, result.radiance);
  const largeReference = areaReference(result.grid, ...result.largeSize, 1, result.radiance);
  const doubledReference = areaReference(result.grid, ...result.tinySize, 1,
    result.radiance.map(value => 2 * value) as [number, number, number]);
  const movedReference = areaReference(result.grid, ...result.tinySize, 2,
    result.radiance.map(value => 2 * value) as [number, number, number]);
  const primaryHits = regularSphereRayHits(result.grid, ...result.largeSize);
  const metrics = { initialRelativeRms: relativeRms(initial, smallReference),
    enlargedRelativeRms: relativeRms(enlarged, largeReference),
    doubledRelativeRms: relativeRms(doubled, doubledReference),
    doubledScalingRms: relativeRms(doubled, initial.map(probe =>
      probe.map(term => term.map(value => value * 2)))),
    movedRelativeRms: relativeRms(moved, movedReference),
    motionDifferenceRms: relativeRms(moved, doubledReference),
    occludedMax: maximumMagnitude(occluded), maskedMax: maximumMagnitude(masked),
    removedMax: maximumMagnitude(removed),
    largePrimaryHits: primaryHits.reduce((sum, count) => sum + count, 0),
    primaryHits, reflectiveMaxRadiance: result.reflective.probes.map(probe => probe.maxRadiance),
    reflectiveMaxCoefficient: maximumMagnitude(reflective) };
  const path = testInfo.outputPath('gi-emissive-measurements.json');
  await writeFile(path, JSON.stringify({ metrics, result, references: {
    small: smallReference, large: largeReference, doubled: doubledReference, moved: movedReference,
  } }, null, 2));
  await testInfo.attach('gi-emissive-measurements', { path, contentType: 'application/json' });
  console.log('GI_EMISSIVE_GPU_DIAGNOSTIC ' + JSON.stringify(metrics));

  expect(metrics.initialRelativeRms).toBeLessThan(0.01);
  expect(metrics.largePrimaryHits).toBeGreaterThan(10);
  expect(metrics.enlargedRelativeRms).toBeLessThan(0.1);
  // Sphere rays retain diffuse reflection while area sampling adds emitted
  // radiance. A per-path maximum would clip the combined PBR contribution.
  const baseColor = [0.9, 0.8, 0.7];
  expect(primaryHits.every(count => count > 0)).toBe(true);
  for (let probe = 0; probe < primaryHits.length; probe++) {
    if (primaryHits[probe] === 0) continue;
    for (let channel = 0; channel < 3; channel++) {
      const expectedBound = result.radiance[channel] + baseColor[channel] * 0.96 * 3 / Math.PI;
      expect(result.reflective.probes[probe].maxRadiance[channel],
        `PBR combined radiance bound probe=${probe} channel=${channel}`)
        .toBeGreaterThan(expectedBound - 0.03);
    }
  }
  expect(metrics.occludedMax).toBeLessThan(1e-4);
  expect(metrics.maskedMax).toBeLessThan(1e-4);
  expect(metrics.doubledRelativeRms).toBeLessThan(0.01);
  expect(metrics.doubledScalingRms).toBeLessThan(0.02);
  expect(metrics.movedRelativeRms).toBeLessThan(0.01);
  expect(metrics.motionDifferenceRms).toBeGreaterThan(0.1);
  expect(metrics.removedMax).toBeLessThan(1e-4);
  expect(result.diagnostics?.options.emissiveSamplesPerProbe).toBe(32);
  expect(result.diagnostics?.emissiveSampling).not.toBeNull();
  expect(result.frames).toBe(8);
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('area query overflow is reported separately from primary and sun rays', async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true);
    window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'gi-emissive.ts') as {
      runGiEmissiveOverflowFixture: typeof runGiEmissiveOverflowFixture;
    };
    return fixture.runGiEmissiveOverflowFixture();
  }) as OverflowResult;

  const regularHits = regularSphereRayHits(result.grid, ...result.tinySize);
  const diagnostic = { regularHits, invalidProbes: result.probe.invalidProbes,
    overflows: result.probe.probes.map(probe => ({ primary: probe.primaryOverflows,
      shadow: probe.shadowOverflows, emissive: probe.emissiveOverflows })),
    errors: result.errors, afterDispose: result.afterDispose };
  const path = testInfo.outputPath('gi-emissive-overflow.json');
  await writeFile(path, JSON.stringify(diagnostic, null, 2));
  await testInfo.attach('gi-emissive-overflow', { path, contentType: 'application/json' });
  console.log('GI_EMISSIVE_OVERFLOW_GPU_DIAGNOSTIC ' + JSON.stringify(diagnostic));

  expect(regularHits).toEqual(Array(8).fill(0));
  expect(result.probe.probes).toHaveLength(8);
  expect(result.probe.invalidProbes).toBe(8);
  for (const probe of result.probe.probes) {
    expect(probe.valid).toBe(false);
    expect(probe.primaryOverflows).toBe(0);
    expect(probe.shadowOverflows).toBe(0);
    expect(probe.emissiveOverflows).toBe(32);
  }
  expect(result.errors).toEqual([]);
  expect(result.afterDispose).toBe(0);
  expect(pageErrors).toEqual([]);
});

test('emissive query blocks preserve superposition through holes, deletion and restoration', async ({ page }, testInfo) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?trees=1');
  await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const result = await page.evaluate(async () => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'gi-emissive.ts') as {
      runGiEmissiveBlocksFixture: typeof runGiEmissiveBlocksFixture;
    };
    return fixture.runGiEmissiveBlocksFixture();
  });
  const both = coefficients(result.both.probe), a = coefficients(result.firstOnly.probe);
  const b = coefficients(result.secondOnly.probe), restored = coefficients(result.restored.probe);
  const sum = a.map((probe, i) => probe.map((term, j) => term.map((v, c) => v + b[i][j][c])));
  const cases = [result.both, result.firstOnly, result.secondOnly, result.empty, result.restored];
  const metrics = { superpositionError: relativeRms(both, sum), restorationError: relativeRms(restored, both),
    emptyMagnitude: maximumMagnitude(coefficients(result.empty.probe)),
    rays: cases.map(value => value.diagnostics!.emissiveSampling!.queryRaysPerUpdate),
    holeRays: result.secondOnly.diagnostics!.emissiveSampling!.inactiveBlockRaysPerUpdate,
    giQueries: cases.map(value => value.diagnostics!.queriesPerUpdate) };
  const path = testInfo.outputPath('gi-emissive-blocks.json');
  await writeFile(path, JSON.stringify({ metrics, result }, null, 2));
  await testInfo.attach('gi-emissive-blocks', { path, contentType: 'application/json' });
  console.log('GI_EMISSIVE_BLOCKS ' + JSON.stringify(metrics));
  expect(metrics.superpositionError).toBeLessThan(1e-5);
  expect(metrics.restorationError).toBeLessThan(1e-6);
  expect(metrics.emptyMagnitude).toBe(0);
  expect(metrics.rays).toEqual([256, 128, 256, 0, 256]);
  expect(metrics.giQueries).toEqual([1280, 1152, 1280, 1024, 1280]);
  expect(metrics.holeRays).toBe(128);
  expect(result.parameterBudgetError).toContain('DIFFUSE_GI_QUERY_PARAMETER_BUDGET');
  expect(result.unexpectedlyCreated).toBe(false);
  expect(result.disabled!.queriesPerUpdate).toBe(0);
  expect(result.disabled!.emissiveSampling!.queryRaysPerUpdate).toBe(0);
  expect(result.disabled!.emissiveSampling!.samplesPerUpdate).toBe(0);
  expect(result.disabled!.emissiveSampling!.activeSamplesPerUpdate).toBe(0);
  expect(result.disabled!.emissiveSampling!.inactiveBlockRaysPerUpdate).toBe(0);
  expect(result.disabled!.emissiveSampling!.sampleCapacity).toBe(256);
  expect(result.disabled!.emissiveSampling!.enabled).toBe(false);
  expect(result.errors).toEqual([]); expect(pageErrors).toEqual([]);
  expect(result.afterDispose).toBe(0);
});
