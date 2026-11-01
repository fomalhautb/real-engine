import { spawn, type ChildProcess } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { expect, test } from '@playwright/test';
import type { G01FixtureOptions, runG01Fixture } from './fixtures/geometry-g01';

// G01 (docs/benchmarks.md): one closed high-poly mesh, slow zoom then orbits at
// three distances, fixed seed and fixed 1/60 s steps, on the resident GPU LOD
// path and the paged stream path against a full-detail reference.
//
// Sizes: hardware runs the 524,288-triangle mesh (1024 x 256 grid); SwiftShader
// runs the same seeded surface on a 256 x 64 grid (32,768 triangles) so the
// fixed 1921-step trajectory fits its per-test timeout. Both use 480 x 320.
// Environment (all optional):
//   G01_RECORD_DIR=<dir>  copy the JSON record and evidence PNGs there.
//   G01_LARGE=1           hardware only: 960 x 640 viewport, same mesh.
//   G01_VIDEO=1           hardware only, needs G01_RECORD_DIR: H.264 clip through
//                         FFMPEG (default /opt/homebrew/bin/ffmpeg) of reference |
//                         resident | stream with mismatches marked; never committed.
//   G01_STEPS=<n>         development only: stop early (the record is then partial
//                         and the full-trajectory assertion is skipped).
const hardware = process.env.PLAYWRIGHT_GPU === 'hardware';
const large = hardware && process.env.G01_LARGE === '1';
const recordDirectory = process.env.G01_RECORD_DIR;
const video = hardware && process.env.G01_VIDEO === '1' && !!recordDirectory;
const ffmpeg = process.env.FFMPEG ?? '/opt/homebrew/bin/ffmpeg';
const developmentSteps = process.env.G01_STEPS ? Number(process.env.G01_STEPS) : undefined;
const FULL_STEPS = 1921;
const EVIDENCE_STEPS = [0, 300, 599, 780, 1020, 1260, 1500, 1740, 1920];

const options: G01FixtureOptions = {
  ...(hardware ? { around: 1024, across: 256 } : { around: 256, across: 64 }),
  ...(large ? { width: 960, height: 640 } : { width: 480, height: 320 }),
  maxErrorPixels: 1, depthEvery: 8, depthSamples: 32, crossCheckEvery: 240, evidenceSteps: EVIDENCE_STEPS, video,
  ...(developmentSteps ? { steps: developmentSteps } : {}),
};

test('G01 high-poly zoom and orbits: crack-free cuts within declared error on resident and streamed LOD', async ({ page }, testInfo) => {
  // Measured on the M3 Max: about 2 min on SwiftShader, 1.5 min on Apple, 3 min for G01_LARGE.
  test.setTimeout(900_000);
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  let encoder: ChildProcess | undefined, encoderError: Error | undefined, encoderExit: Promise<number | null> | undefined;
  const clip = recordDirectory ? join(recordDirectory, `g01-${hardware ? 'apple' : 'swiftshader'}${large ? '-large' : ''}.mp4`) : '';
  if (video) {
    if (!existsSync(ffmpeg)) throw new Error('G01_VIDEO_FFMPEG_MISSING:' + ffmpeg);
    mkdirSync(recordDirectory!, { recursive: true });
    encoder = spawn(ffmpeg, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', '60', '-i', '-',
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', clip], { stdio: ['pipe', 'ignore', 'pipe'] });
    encoder.on('error', error => { encoderError = error; });
    encoderExit = new Promise(resolve => encoder!.on('close', resolve));
    await page.exposeFunction('__g01VideoFrame', async (_step: number, url: string) => {
      const bytes = Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');
      if (!encoder!.stdin!.write(bytes)) await new Promise(resolve => encoder!.stdin!.once('drain', resolve));
    });
  }
  await page.goto('/?trees=1'); await page.waitForFunction(() => window.__REAL_ENGINE__ || !document.getElementById('error')!.hidden);
  await expect(page.locator('#error')).toBeHidden();
  const r = await page.evaluate(async fixtureOptions => {
    window.__REAL_ENGINE__!.pause(true); window.__REAL_ENGINE__!.dispose();
    const fixture = await import('/tests/browser/fixtures/' + 'geometry-g01.ts') as { runG01Fixture: typeof runG01Fixture };
    return fixture.runG01Fixture(fixtureOptions);
  }, options);
  if (encoder) {
    encoder.stdin!.end();
    const code = await encoderExit;
    if (encoderError || code !== 0) throw new Error('G01_VIDEO_ENCODE_FAILED:' + (encoderError?.message ?? code));
  }

  const files: string[] = [];
  for (const entry of r.evidence) {
    const path = testInfo.outputPath(`g01-step${String(entry.step).padStart(4, '0')}-${entry.name}.png`);
    writeFileSync(path, Buffer.from(entry.png.slice(entry.png.indexOf(',') + 1), 'base64'));
    files.push(path);
  }
  const record = { record: 'G01', variant: hardware ? (large ? 'hardware-large' : 'hardware') : 'swiftshader', partial: !!developmentSteps,
    clip: video ? basename(clip) : null, ...r, evidence: r.evidence.map(entry => ({ step: entry.step, name: entry.name })) };
  const json = testInfo.outputPath('g01.json');
  writeFileSync(json, JSON.stringify(record));
  files.push(json);
  await testInfo.attach('g01.json', { path: json, contentType: 'application/json' });
  if (recordDirectory) {
    const target = join(recordDirectory, record.variant);
    mkdirSync(target, { recursive: true });
    for (const file of files) copyFileSync(file, join(target, basename(file)));
  }
  const brief = (name: 'resident' | 'stream') => {
    const p = r.paths[name];
    return { frames: p.frames, cut: p.cut, silhouette: { ...p.silhouette, worst: p.silhouette.worst.slice(0, 3), violationDiagnoses: p.silhouette.violationDiagnoses.slice(0, 6) },
      depth: { ...p.depth, worst: p.depth.worst.slice(0, 3) }, popping: { ...p.popping, frozenChecks: undefined },
      cracks: { ...p.cracks, crossChecks: p.cracks.crossChecks.filter(check => check.repeatedOrigins || !check.coverageEqual || !check.edgesEqual) },
      cpuCut: p.cpuCut, ...('streaming' in p ? { streaming: p.streaming } : {}) };
  };
  console.log('GEOMETRY_G01', JSON.stringify({ variant: record.variant, adapter: r.adapter, viewport: r.viewport, source: r.source, hierarchy: r.hierarchy,
    settle: r.streamSettle, end: r.streamEnd, unknownPixels: r.unknownPixels, backpressureFrames: r.backpressureFrames,
    diagnosticTimingMs: r.diagnosticTimingMs, resident: brief('resident'), stream: brief('stream') }));

  expect(r.errors).toEqual([]); expect(pageErrors).toEqual([]);
  expect(r.source.triangles).toBe(2 * options.around * options.across);
  if (hardware) {
    expect(r.source.triangles).toBeGreaterThanOrEqual(500_000);
    expect(r.adapter.isFallbackAdapter).toBe(false);
    if (process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR) expect(r.adapter.vendor.toLowerCase()).toContain(process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR.toLowerCase());
  }
  expect(r.viewport).toMatchObject({ width: options.width, height: options.height });
  if (!developmentSteps) expect(r.parameters.trajectory.measuredSteps).toBe(FULL_STEPS);
  expect(r.unknownPixels).toBe(0);
  for (const name of ['resident', 'stream'] as const) {
    const p = r.paths[name];
    expect(p.frames, name).toBe(r.parameters.trajectory.measuredSteps);
    // Cracks and holes: exact structure and welded topology of every probed cut.
    expect(p.cracks.maxBoundaryEdges, name).toBe(0);
    expect({ uncovered: p.cracks.maxUncovered, overlapping: p.cracks.maxOverlapping, partialGroups: p.cracks.maxPartialGroups }, name)
      .toEqual({ uncovered: 0, overlapping: 0, partialGroups: 0 });
    // Repeated directed edges (folds or duplicated surface) are stricter than the
    // engine's crack definition (boundary edges only; tests/geometry-stream-coverage.test.ts
    // accepts pinched coarse levels). They are a known builder gap, recorded but not
    // gated: the group boundary check compares net signed edge balance, so a
    // simplification may fold triangles over a locked border edge (Apple, 524k
    // triangles: 2-18 per frame at levels 1-3; SwiftShader's 32k mesh: 0).
    testInfo.annotations.push({ type: 'known-gap', description: `${name}: up to ${p.cracks.maxRepeatedEdges} repeated directed edges (folds) per frame` });
    // The per-frame counters reproduce the reference functions wherever both ran.
    expect(p.cracks.crossChecks.length, name).toBeGreaterThan(0);
    for (const check of p.cracks.crossChecks) expect({ step: check.step, coverage: check.coverageEqual, edges: check.edgesEqual }, name)
      .toEqual({ step: check.step, coverage: true, edges: true });
    expect(p.cut.gpuTrianglesEqualCutTriangles, name).toBe(true);
    // Silhouette displacement within eps * sec^2(theta) + sqrt(2) px and surface
    // gaps within eps * sec(theta) + 0.001 px, eps being the cut's declared error.
    // The 2D bound is a distance to the label boundary, which assumes that boundary
    // is the only silhouette. The torus also has internal silhouettes (the near tube
    // over the hole) where LOD error within eps opens one-pixel see-through slivers
    // deep inside the label (Apple 480x320: 17 per path, the full-detail point
    // 0.02-0.38 px from the cut). So a violation is explained when the surface point
    // the pixel shows lies within the 3D surface bound of the other surface, or when
    // the CPU ray misses that surface as well (a raster tie at the exact silhouette).
    // Raw violations stay in the record; all must be classified and none unexplained.
    // Soft, so one failing bound still reports the others.
    expect(p.silhouette.violationsClassified, `${name} classified silhouette violations`).toBe(p.silhouette.violations);
    expect.soft(p.silhouette.violations - p.silhouette.violationsWithinSurfaceBound - p.silhouette.violationsCpuMiss,
      `${name} silhouette pixels beyond the bound and unexplained by the surface bound`).toBe(0);
    expect(p.depth.samples, name).toBeGreaterThan(0);
    expect.soft(p.depth.violations, `${name} surface gaps beyond the bound`).toBe(0);
    // Popping is a measured value with no gate.
  }
  // The resident cut is a pure function of the camera: it equals the CPU
  // reference except threshold decisions within float rounding, and a frozen
  // selection camera draws exactly the previous cut.
  expect(r.paths.resident.cpuCut.unexplainedClusters).toBe(0);
  expect(r.paths.resident.popping.frozenChecks?.every(check => check.equalToPreviousCut)).toBe(true);
});
