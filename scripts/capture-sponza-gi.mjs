import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// Local, already-built Intel Sponza assets only. This capture does not fetch,
// rebuild, or accept unhashed geometry pages/proxies. Coarse proxies are not a
// visibility reference; use --query-root with --source-raster for exact LOD0
// topology comparisons. Four BLEND primitives remain explicitly omitted.
const defaults = {
  geometryRoot: '/artifacts/sponza-group16-pages-20261022/',
  textureGpuRoot: '/artifacts/sponza-textures-gpu-20261020/',
  textureReferenceRoot: '/artifacts/sponza-textures-reference-20261020/',
  uvAuditUrl: '/artifacts/sponza-uv-demand-20261020.json',
  output: 'artifacts/sponza-gi-capture',
  exposureEV: 3,
  anisotropy: 8,
  warmupFrames: 600,
};
const valueFlags = new Map([
  ['--geometry-root', 'geometryRoot'], ['--query-root', 'queryRoot'], ['--output', 'output'],
  ['--texture-gpu-root', 'textureGpuRoot'], ['--texture-reference-root', 'textureReferenceRoot'],
  ['--uv-audit', 'uvAuditUrl'],
  ['--anisotropy', 'anisotropy'],
  ['--warmup-frames', 'warmupFrames'],
  ['--gather-spacing', 'gatherPixelSpacing'], ['--gather-capacity', 'gatherCapacity'],
  ['--supplemental-capacity', 'supplementalCapacity'], ['--supplemental-work', 'supplementalWork'],
  ['--diagnostic-points', 'diagnosticPoints'],
  ['--visibility-direction', 'visibilityDirection'],
  ['--visibility-filter', 'visibilityFilter'],
]);
const switches = new Map([
  ['--source-raster', 'sourceRaster'], ['--fine-gather', 'fineGather'],
  ['--reference', 'reference'], ['--final-gather', 'finalGather'],
  ['--raster-receiver', 'rasterReceiver'],
  ['--reconstruction-diagnostics', 'reconstructionDiagnostics'],
  ['--coverage-diagnostics', 'coverageDiagnostics'],
  ['--supplemental-gather', 'supplementalGather'],
  ['--local-reference', 'localReference'],
  ['--scene-budget', 'sceneBudget'],
]);
const options = { ...defaults };
for (let i = 2; i < process.argv.length; i++) {
  const flag = process.argv[i];
  if (valueFlags.has(flag)) {
    const value = process.argv[++i];
    if (!value || value.startsWith('--')) throw Error(`SPONZA_GI_ARGUMENT:${flag}`);
    options[valueFlags.get(flag)] = value;
  } else if (switches.has(flag)) options[switches.get(flag)] = true;
  else throw Error(`SPONZA_GI_ARGUMENT:${flag}`);
}
for (const key of ['geometryRoot', 'queryRoot', 'textureGpuRoot', 'textureReferenceRoot']) {
  if (options[key] !== undefined && (!options[key].startsWith('/') || options[key].startsWith('//') || !options[key].endsWith('/')))
    throw Error(`SPONZA_GI_LOCAL_ROOT:${key}`);
}
if (!options.uvAuditUrl.startsWith('/') || options.uvAuditUrl.startsWith('//')) throw Error('SPONZA_GI_LOCAL_UV_AUDIT');
if (options.visibilityDirection !== undefined && !['probe-to-receiver', 'receiver-to-probe'].includes(options.visibilityDirection)) {
  throw Error('SPONZA_GI_VISIBILITY_DIRECTION');
}
if (options.visibilityFilter !== undefined && !['moment-bilinear', 'per-tap-chebyshev-bilinear'].includes(options.visibilityFilter)) {
  throw Error('SPONZA_GI_VISIBILITY_FILTER');
}
if (options.sourceRaster && !options.queryRoot) throw Error('SPONZA_GI_SOURCE_RASTER_REQUIRES_FINE_QUERY');
if (options.sourceRaster && options.sceneBudget) throw Error('SPONZA_GI_SOURCE_RASTER_HAS_NO_PAGE_POOLS');
if (options.fineGather) options.finalGather = true;
if (options.rasterReceiver && !options.finalGather) throw Error('SPONZA_GI_RASTER_RECEIVER_REQUIRES_FINAL_GATHER');
if (options.reconstructionDiagnostics && !options.finalGather) throw Error('SPONZA_GI_RECONSTRUCTION_REQUIRES_FINAL_GATHER');
if (options.coverageDiagnostics && !options.reconstructionDiagnostics) throw Error('SPONZA_GI_COVERAGE_REQUIRES_RECONSTRUCTION');
if (options.supplementalGather && !options.rasterReceiver) throw Error('SPONZA_GI_SUPPLEMENTAL_REQUIRES_RASTER');
if (options.localReference && !options.reconstructionDiagnostics) throw Error('SPONZA_GI_LOCAL_REFERENCE_REQUIRES_RECONSTRUCTION');
for (const [key, maximum] of [['gatherPixelSpacing', 128], ['gatherCapacity', 65536],
  ['supplementalCapacity', 16384], ['supplementalWork', 67108864]]) {
  if (options[key] === undefined) continue;
  options[key] = Number(options[key]);
  if (!Number.isSafeInteger(options[key]) || options[key] < 1 || options[key] > maximum) throw Error('SPONZA_GI_' + key);
  if (!options.finalGather || (key.startsWith('supplemental') && !options.supplementalGather)) throw Error('SPONZA_GI_QUALITY_REQUIRES_ENABLED_PATH');
}
if (options.diagnosticPoints !== undefined) {
  if (!options.reconstructionDiagnostics || !/^\d+,\d+(;\d+,\d+){0,127}$/.test(options.diagnosticPoints)) throw Error('SPONZA_GI_DIAGNOSTIC_POINTS');
  options.diagnosticPoints = options.diagnosticPoints.split(';').map(pair => {
    const [x, y] = pair.split(',').map(Number);
    if (x < 0 || x >= 1280 || y < 0 || y >= 720) throw Error('SPONZA_GI_DIAGNOSTIC_POINTS');
    return { x, y };
  });
}
if (options.anisotropy !== undefined) {
  options.anisotropy = Number(options.anisotropy);
  if (!Number.isInteger(options.anisotropy) || options.anisotropy < 1 || options.anisotropy > 16) throw Error('SPONZA_GI_ANISOTROPY');
}
options.warmupFrames = Number(options.warmupFrames);
if (!Number.isSafeInteger(options.warmupFrames) || options.warmupFrames < 90 || options.warmupFrames > 3600)
  throw Error('SPONZA_GI_WARMUP_FRAMES');

await mkdir(options.output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--force-high-performance-gpu'] });
try {
  const page = await browser.newPage({ viewport: { width: 1320, height: 800 } });
  page.setDefaultTimeout(600000);
  const pageErrors = [];
  page.on('pageerror', error => { pageErrors.push(error.message); console.log('PAGE_ERROR ' + error.message); });
  page.on('console', message => { if (message.text().startsWith('SPONZA_PROGRESS')) console.log(message.text()); });
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => window.__REAL_ENGINE__);
  const result = await page.evaluate(async settings => {
    window.__REAL_ENGINE__.pause(true);
    window.__REAL_ENGINE__.dispose();
    const { captureIntelSponzaGi } = await import('/tests/browser/fixtures/intel-sponza-gi.ts');
    return captureIntelSponzaGi(settings);
  }, options);
  for (const capture of result.giAudit.captures) {
    await writeFile(join(options.output, `${capture.label}.png`), Buffer.from(capture.png.split(',')[1], 'base64'));
    delete capture.png;
    if (capture.coveragePng) {
      await writeFile(join(options.output, `${capture.label}-coverage.png`), Buffer.from(capture.coveragePng.split(',')[1], 'base64'));
      delete capture.coveragePng;
    }
  }
  await writeFile(join(options.output, 'report.json'), JSON.stringify({ ...result, captureOptions: options, pageErrors }, null, 2));
  if (pageErrors.length || result.errors.length || result.afterDispose !== 0 || result.info.adapterInfo.isFallbackAdapter ||
    result.giAudit.captures.some(c => !c.residencyUnchanged || c.probe?.invalidProbes > 0)) throw Error('SPONZA_GI_VALIDATION');
  console.log(JSON.stringify({ captures: result.giAudit.captures.map(c => ({ label: c.label,
    invalidProbes: c.probe?.invalidProbes, residencyUnchanged: c.residencyUnchanged })),
    adapter: result.info.adapterInfo, pageErrors, afterDispose: result.afterDispose }));
} finally {
  await browser.close();
}
