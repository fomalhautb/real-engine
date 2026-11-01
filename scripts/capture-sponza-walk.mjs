import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { assetSetRoots, captureBaseUrl, captureChannel, launchCaptureBrowser, requireAssetRoots } from './capture-target.mjs';

// Oversubscribed Intel Sponza walk against already-built, local hashed assets:
// teleports, 180-degree turns and the asset set's camera path under a runtime
// geometry budget whose pool is 1/ratio of the unique data. Modes: verify (cut
// and pixel checks against a full-detail reference), recovery (no probes; the
// streaming recovery timings) and present (textured clip through ffmpeg).
// Nothing is downloaded or rebuilt; timings are diagnostic, never benchmarks.
const options = { mode: 'verify', ratio: 4, output: 'artifacts/sponza-walk', anisotropy: 8, timeoutMs: 3600000,
  ffmpeg: '/opt/homebrew/bin/ffmpeg', fps: 30 };
const flags = new Map([['--asset-set', 'assetSet'], ['--geometry-root', 'geometryRoot'], ['--texture-gpu-root', 'textureGpuRoot'],
  ['--texture-reference-root', 'textureReferenceRoot'], ['--uv-audit', 'uvAuditUrl'], ['--fine-query-root', 'fineQueryRoot'],
  ['--camera-path', 'cameraPathUrl'], ['--mode', 'mode'], ['--ratio', 'ratio'], ['--output', 'output'], ['--base-url', 'baseUrl'],
  ['--channel', 'channel'], ['--width', 'width'], ['--height', 'height'], ['--lod-error', 'maxErrorPixels'],
  ['--stable-frames', 'stableFrames'], ['--max-settle-frames', 'maxSettleFrames'], ['--check-every', 'checkEvery'],
  ['--walk-check-every', 'walkCheckEvery'], ['--walk-step', 'walkStep'], ['--turn-seconds', 'turnSeconds'],
  ['--interval-frames', 'intervalFrames'], ['--concurrent-migrations', 'maxConcurrentMigrations'], ['--exposure', 'exposureEV'],
  ['--debug-split', 'debugSplit'], ['--anisotropy', 'anisotropy'], ['--timeout-ms', 'timeoutMs'], ['--ffmpeg', 'ffmpeg'], ['--fps', 'fps']]);
if (process.argv.includes('--help')) {
  console.log('Usage: node scripts/with-dev-server.mjs --port N -- node scripts/capture-sponza-walk.mjs --asset-set artifacts/<set>/asset-set.json\n'
    + '  --mode verify|recovery|present --ratio 4|8 --output artifacts/<dir> [--lod-error 1] [--width W --height H]\n'
    + '  [--stable-frames 45] [--max-settle-frames 1800] [--check-every 15] [--walk-check-every 30] [--walk-step 0.0333] [--turn-seconds 0.25]\n'
    + '  [--interval-frames 30] [--concurrent-migrations 4] [--debug-split 0.5] [--exposure 3] [--fps 30] [--ffmpeg /opt/homebrew/bin/ffmpeg]\n'
    + 'The asset set supplies the package roots, fineQueryRoot (verify reference) and cameraPathUrl; flags override them.\n'
    + 'present writes <output>/walk.mp4 (H.264). Clips, logs and reports are local evidence and never committed.');
  process.exit(0);
}
for (let i = 2; i < process.argv.length; i++) {
  const flag = process.argv[i], key = flags.get(flag), value = process.argv[++i];
  if (!key || value === undefined || value.startsWith('--')) throw Error('SPONZA_WALK_ARGUMENT: ' + flag);
  options[key] = value;
}
for (const [key, value] of Object.entries(await assetSetRoots(options.assetSet))) options[key] ??= value;
if (options.assetSet) {
  // The single-set reader only returns the four package roots; the walk also needs these two.
  const file = resolve(options.assetSet);
  if ((await stat(file)).size > 1024 * 1024) throw Error('SPONZA_WALK_ASSET_SET_BYTES');
  const set = JSON.parse(await readFile(file, 'utf8'));
  options.fineQueryRoot ??= set.roots?.fineQueryRoot;
  options.cameraPathUrl ??= set.roots?.cameraPathUrl;
}
requireAssetRoots(options, 'SPONZA_WALK');
for (const key of ['geometryRoot', 'textureGpuRoot', 'textureReferenceRoot', 'fineQueryRoot']) if (options[key] !== undefined
  && (typeof options[key] !== 'string' || !options[key].startsWith('/') || options[key].startsWith('//') || !options[key].endsWith('/'))) throw Error('SPONZA_WALK_LOCAL_ROOT: ' + key);
for (const key of ['uvAuditUrl', 'cameraPathUrl']) if (typeof options[key] !== 'string' || !options[key].startsWith('/') || options[key].startsWith('//')) {
  throw Error('SPONZA_WALK_LOCAL_FILE: ' + key);
}
if (!['verify', 'recovery', 'present'].includes(options.mode)) throw Error('SPONZA_WALK_ARGUMENT: mode');
if (options.mode === 'verify' && !options.fineQueryRoot) throw Error('SPONZA_WALK_VERIFY_REQUIRES_FINE_QUERY');
for (const [key, minimum, maximum, integral] of [['ratio', 1, 64, false], ['width', 64, 3840, true], ['height', 64, 2160, true],
  ['maxErrorPixels', 0.0625, 16, false], ['stableFrames', 2, 600, true], ['maxSettleFrames', 10, 20000, true], ['checkEvery', 0, 600, true],
  ['walkCheckEvery', 1, 3600, true], ['walkStep', 0.001, 1, false], ['turnSeconds', 0.01, 2, false], ['intervalFrames', 1, 600, true],
  ['maxConcurrentMigrations', 1, 16, true], ['exposureEV', -10, 10, false], ['debugSplit', 0, 1, false], ['anisotropy', 1, 16, true],
  ['timeoutMs', 60000, 14400000, true], ['fps', 1, 120, true]]) {
  if (options[key] === undefined) continue;
  options[key] = Number(options[key]);
  if (!Number.isFinite(options[key]) || options[key] < minimum || options[key] > maximum || integral && !Number.isSafeInteger(options[key])) {
    throw Error('SPONZA_WALK_ARGUMENT: ' + key);
  }
}
options.baseUrl = captureBaseUrl(options.baseUrl);
options.channel = captureChannel(options.channel);
await mkdir(options.output, { recursive: true });

let encoder = null, encoderExit = null;
if (options.mode === 'present') {
  encoder = spawn(options.ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(options.fps),
    '-c:v', 'mjpeg', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    join(options.output, 'walk.mp4')], { stdio: ['pipe', 'inherit', 'inherit'] });
  encoderExit = once(encoder, 'exit');
}
const browser = await launchCaptureBrowser(options.channel);
const pageErrors = [], progress = [], saved = [];
let deadline, frames = 0;
try {
  const page = await browser.newPage({ viewport: { width: 1360, height: 820 } });
  page.setDefaultTimeout(120000);
  page.on('pageerror', error => { pageErrors.push(error.message); console.log('PAGE_ERROR ' + error.message); });
  page.on('console', message => {
    const text = message.text(); if (!text.startsWith('SPONZA_WALK_PROGRESS ')) return;
    console.log(text); progress.push(JSON.parse(text.slice('SPONZA_WALK_PROGRESS '.length)));
  });
  await page.exposeFunction('__saveSponzaWalkCapture', async capture => {
    if (!/^[a-z0-9.:-]+$/.test(capture.label) || !capture.png.startsWith('data:image/png;base64,')) throw Error('SPONZA_WALK_CAPTURE_FORMAT');
    const file = capture.label.replace(/[^a-z0-9-]/g, '-') + '.png';
    await writeFile(join(options.output, file), Buffer.from(capture.png.split(',')[1], 'base64'));
    saved.push(file);
  });
  await page.exposeFunction('__sponzaWalkFrame', async frame => {
    if (!encoder || !frame.jpeg.startsWith('data:image/jpeg;base64,')) throw Error('SPONZA_WALK_FRAME_FORMAT');
    if (!encoder.stdin.write(Buffer.from(frame.jpeg.split(',')[1], 'base64'))) await once(encoder.stdin, 'drain');
    frames++;
  });
  await page.goto(options.baseUrl + '/');
  await page.waitForFunction(() => window.__REAL_ENGINE__);
  const settings = { ...options };
  for (const key of ['assetSet', 'output', 'baseUrl', 'channel', 'timeoutMs', 'ffmpeg', 'fps']) delete settings[key];
  const result = await Promise.race([page.evaluate(async settings => {
    window.__REAL_ENGINE__.pause(true); window.__REAL_ENGINE__.dispose();
    const { runIntelSponzaWalk } = await import('/tests/browser/fixtures/intel-sponza-walk.ts');
    return runIntelSponzaWalk(settings, {
      capture: capture => window.__saveSponzaWalkCapture(capture),
      ...(settings.mode === 'present' ? { frame: frame => window.__sponzaWalkFrame(frame) } : {}) });
  }, settings), new Promise((_, reject) => { deadline = setTimeout(() => reject(Error('SPONZA_WALK_DEADLINE')), options.timeoutMs); })]);
  await writeFile(join(options.output, 'report.json'), JSON.stringify({ ...result, options, pageErrors, savedCaptures: saved, videoFrames: frames }, null, 2));
  const summary = { passed: result.passed, failures: result.failures, mode: result.mode, adapter: result.adapter, budget: {
    uniqueToPool: result.budget.uniqueToPool, poolBudgetBytes: result.budget.poolBudgetBytes, maxResidentBytes: result.budget.maxResidentBytes,
    maxPeakBytes: result.budget.maxPeakBytes, uniqueResidentBytes: result.budget.uniqueResidentBytes },
    phases: result.phases.map(phase => ({ label: phase.label, framesToLastChange: phase.framesToLastChange, msToLastChange: phase.msToLastChange,
      framesToDemandMet: phase.framesToDemandMet, msToDemandMet: phase.msToDemandMet, loaded: phase.loaded, dropped: phase.dropped })),
    walk: { frames: result.walk.frames, wallMs: result.walk.wallMs, peakMissing: result.walk.peakMissingAllocatedPages,
      loaded: result.walk.loaded, dropped: result.walk.dropped }, summary: result.summary, streaming: result.streaming, videoFrames: frames };
  console.log(JSON.stringify(summary));
  if (result.adapter.isFallbackAdapter !== false) throw Error('SPONZA_WALK_FALLBACK_ADAPTER');
  if (!result.passed || result.errors.length || pageErrors.length) throw Error('SPONZA_WALK_VALIDATION: ' + result.failures.join('; '));
} catch (error) {
  await writeFile(join(options.output, 'failure.json'), JSON.stringify({ error: String(error), options, pageErrors, saved, progress }, null, 2));
  throw error;
} finally {
  clearTimeout(deadline);
  await writeFile(join(options.output, 'progress.json'), JSON.stringify(progress, null, 2));
  await browser.close();
  if (encoder) { encoder.stdin.end(); const [code] = await encoderExit; if (code !== 0) process.exitCode = 1; }
}
