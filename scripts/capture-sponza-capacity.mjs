import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// Explicit hardware correctness run against already-built, local hashed assets.
// No source assets are downloaded or rebuilt, and this is not an FPS benchmark.
const options = { geometryRoot: '/artifacts/sponza-group16-pages-20261022/',
  textureGpuRoot: '/artifacts/sponza-textures-gpu-20261020/',
  textureReferenceRoot: '/artifacts/sponza-textures-reference-20261020/',
  uvAuditUrl: '/artifacts/sponza-uv-demand-20261020.json', anisotropy: 8,
  output: 'artifacts/sponza-capacity-capture', settleFrames: 1200, travelFrames: 48, stableFrames: 24,
  intervalFrames: 2, maxConcurrentMigrations: 4, maxErrorPixels: 2, exposureEV: 3, timeoutMs: 1800000 };
const flags = new Map([['--geometry-root', 'geometryRoot'], ['--query-root', 'queryRoot'], ['--output', 'output'],
  ['--texture-gpu-root', 'textureGpuRoot'], ['--texture-reference-root', 'textureReferenceRoot'], ['--uv-audit', 'uvAuditUrl'],
  ['--resident-mib', 'residentMiB'], ['--peak-mib', 'peakMiB'], ['--settle-frames', 'settleFrames'],
  ['--travel-frames', 'travelFrames'], ['--stable-frames', 'stableFrames'], ['--interval-frames', 'intervalFrames'], ['--concurrent-migrations', 'maxConcurrentMigrations'],
  ['--lod-error', 'maxErrorPixels'], ['--anisotropy', 'anisotropy'], ['--exposure', 'exposureEV'], ['--timeout-ms', 'timeoutMs']]);
if (process.argv.includes('--help')) {
  console.log('Usage: node scripts/capture-sponza-capacity.mjs [--output artifacts/name] [--resident-mib N --peak-mib N]\n'
    + '  --settle-frames 1200 --travel-frames 48 --stable-frames 24 --interval-frames 2 --concurrent-migrations 4 --lod-error 2\n'
    + '  --geometry-root /local/path/ --texture-gpu-root /local/path/ --texture-reference-root /local/path/ --uv-audit /local/file.json\n'
    + 'Requires an existing Vite server at http://127.0.0.1:5173. Runs Edge with the high-performance hardware adapter.\n'
    + 'Default live budget equals original quarter-page pools plus ALL selection and current/previous output allocations.\n'
    + 'Default peak adds the largest planned route candidate plus one slot. JSON reports exact bytes; these are not total VRAM limits.');
  process.exit(0);
}
for (let i = 2; i < process.argv.length; i++) {
  const flag = process.argv[i], key = flags.get(flag), value = process.argv[++i];
  if (!key || !value || value.startsWith('--')) throw Error('SPONZA_CAPACITY_ARGUMENT: ' + flag);
  options[key] = value;
}
for (const key of ['geometryRoot', 'queryRoot', 'textureGpuRoot', 'textureReferenceRoot']) if (options[key] !== undefined
  && (!options[key].startsWith('/') || options[key].startsWith('//') || !options[key].endsWith('/'))) throw Error('SPONZA_CAPACITY_LOCAL_ROOT: ' + key);
if (!options.uvAuditUrl.startsWith('/') || options.uvAuditUrl.startsWith('//')) throw Error('SPONZA_CAPACITY_LOCAL_UV_AUDIT');
for (const [key, minimum, maximum, integral] of [['residentMiB', 0.000001, 8192, false], ['peakMiB', 0.000001, 8192, false],
  ['settleFrames', 90, 3600, true], ['travelFrames', 1, 240, true], ['stableFrames', 12, 120, true], ['intervalFrames', 1, 60, true],
  ['maxConcurrentMigrations', 1, 16, true],
  ['maxErrorPixels', 0, 16, false], ['anisotropy', 1, 16, true], ['exposureEV', -10, 10, false], ['timeoutMs', 60000, 3600000, true]]) {
  if (options[key] === undefined) continue;
  options[key] = Number(options[key]);
  if (!Number.isFinite(options[key]) || options[key] < minimum || options[key] > maximum || integral && !Number.isSafeInteger(options[key])) {
    throw Error('SPONZA_CAPACITY_ARGUMENT: ' + key);
  }
}
if (options.stableFrames < 2 * options.intervalFrames || options.settleFrames < options.stableFrames) throw Error('SPONZA_CAPACITY_SETTLE_WINDOW');
if (options.peakMiB !== undefined && options.residentMiB !== undefined && options.peakMiB < options.residentMiB) throw Error('SPONZA_CAPACITY_PEAK_LESS_THAN_LIVE');
await mkdir(options.output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--force-high-performance-gpu'] });
const pageErrors = [], progress = [], savedCaptures = [];
let deadline;
try {
  const page = await browser.newPage({ viewport: { width: 1320, height: 800 } });
  page.setDefaultTimeout(60000);
  page.on('pageerror', error => { pageErrors.push(error.message); console.log('PAGE_ERROR ' + error.message); });
  page.on('console', message => {
    const text = message.text(); if (!text.startsWith('SPONZA_CAPACITY_PROGRESS ')) return;
    console.log(text); progress.push(JSON.parse(text.slice('SPONZA_CAPACITY_PROGRESS '.length)));
  });
  // Persist each image immediately, so a later failed endpoint preserves evidence.
  await page.exposeFunction('__saveSponzaCapacityCapture', async capture => {
    if (!/^[a-z0-9-]+$/.test(capture.label) || !capture.png.startsWith('data:image/png;base64,')) throw Error('SPONZA_CAPACITY_CAPTURE_FORMAT');
    const file = capture.label + '.png';
    await writeFile(join(options.output, file), Buffer.from(capture.png.split(',')[1], 'base64'));
    savedCaptures.push({ label: capture.label, frame: capture.frame, debug: capture.debug, exposureEV: capture.exposureEV, file });
  });
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await Promise.race([page.evaluate(async settings => {
    window.__REAL_ENGINE__.pause(true); window.__REAL_ENGINE__.dispose();
    const { captureIntelSponzaCapacity } = await import('/tests/browser/fixtures/intel-sponza-capacity.ts');
    return captureIntelSponzaCapacity(settings, capture => window.__saveSponzaCapacityCapture(capture));
  }, options), new Promise((_, reject) => { deadline = setTimeout(() => reject(Error('SPONZA_CAPACITY_DEADLINE')), options.timeoutMs); })]);
  for (const capture of result.captures) { delete capture.png; capture.file = capture.label + '.png'; }
  await writeFile(join(options.output, 'report.json'), JSON.stringify({ ...result, options, pageErrors }, null, 2));
  if (!result.passed || result.errors.length || pageErrors.length || result.ownedAfterDispose !== 0
    || result.info.adapterInfo.isFallbackAdapter !== false) throw Error('SPONZA_CAPACITY_VALIDATION: ' + result.failures.join('; '));
  console.log(JSON.stringify({ passed: true, adapter: result.info.adapterInfo, limits: result.limits,
    endpoints: result.endpoints.map(endpoint => ({ label: endpoint.label, triangles: endpoint.actualTriangles,
      clusters: endpoint.visibleClusters, framesToSettle: endpoint.framesToSettle, exchange: endpoint.exchange })),
    migrationBatches: result.migrationBatches, migrations: result.auditedMigrations,
    diagnosticMigrations: result.afterDispose.migrations, recovery: result.recovery, ownedAfterDispose: result.ownedAfterDispose }));
} catch (error) {
  await writeFile(join(options.output, 'failure.json'), JSON.stringify({ error: String(error), options, pageErrors, savedCaptures, progress }, null, 2));
  throw error;
} finally {
  clearTimeout(deadline);
  await writeFile(join(options.output, 'progress.json'), JSON.stringify(progress, null, 2));
  await browser.close();
}
