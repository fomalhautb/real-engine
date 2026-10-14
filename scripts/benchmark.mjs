import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { cp, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { platform, release, arch, cpus, totalmem } from 'node:os';

const args = process.argv.slice(2), options = {}, settings = { channel: 'chrome', repeats: 3, headed: false, software: false,
  highPerformanceGpu: false, expectVendor: null };
let output;
const numeric = new Map([['--frames', 'frames'], ['--seconds', 'durationSeconds'], ['--warmup', 'warmupFrames'],
  ['--trees', 'trees'], ['--seed', 'seed'], ['--width', 'width'], ['--height', 'height'], ['--budget-ms', 'frameBudgetMs']]);
for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (numeric.has(arg)) { const value = Number(args[++i]); if (!Number.isFinite(value)) throw new Error('Invalid ' + arg); options[numeric.get(arg)] = value; }
  else if (arg === '--stress') options.stress = true;
  else if (arg === '--software') settings.software = true;
  else if (arg === '--high-performance-gpu') settings.highPerformanceGpu = true;
  else if (arg === '--expect-vendor') settings.expectVendor = args[++i];
  else if (arg === '--headed') settings.headed = true;
  else if (arg === '--channel') settings.channel = args[++i];
  else if (arg === '--repeat') settings.repeats = Number(args[++i]);
  else if (arg === '--out') output = args[++i];
  else throw new Error('Unknown benchmark argument: ' + arg);
}
if (!Number.isInteger(settings.repeats) || settings.repeats < 1 || settings.repeats > 10) throw new Error('--repeat must be 1..10');
if (!['chrome', 'msedge', 'chromium'].includes(settings.channel)) throw new Error('--channel must be chrome, msedge, or chromium');
if (settings.software && settings.highPerformanceGpu) throw new Error('Cannot combine software and high-performance GPU selection');
if (settings.expectVendor !== null && !['nvidia', 'intel', 'amd', 'apple', 'qualcomm', 'arm'].includes(settings.expectVendor)) {
  throw new Error('--expect-vendor must be nvidia, intel, amd, apple, qualcomm, or arm');
}
const stamp = new Date().toISOString().replaceAll(/[:.]/g, '-');
const directory = resolve(output ?? `artifacts/benchmarks/${stamp}`);
if (directory === resolve('.')) throw new Error('Benchmark output directory cannot be the repository root');
await mkdir(directory, { recursive: true });
// A unique copied production build prevents concurrent source edits/HMR from changing a long run.
const snapshot = resolve(directory, 'build-' + stamp);
const git = args => execFileSync('git', args, { encoding: 'utf8', windowsHide: true }).trim();
const status = git(['status', '--porcelain']);
const sourcePaths = () => execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'],
  { encoding: 'utf8', windowsHide: true }).split('\0').filter(path => path && !resolve(path).startsWith(directory + sep)).sort();
const hashFiles = async (paths, absolutePath) => {
  const aggregate = createHash('sha256');
  const entries = [];
  for (const path of paths) {
    const digest = createHash('sha256').update(await readFile(absolutePath(path))).digest('hex');
    entries.push([path, digest]);
    aggregate.update(path + '\0' + digest + '\n');
  }
  return { sha256: aggregate.digest('hex'), files: Object.fromEntries(entries) };
};
const listBuildFiles = async (root, relative = '') => {
  const paths = [];
  for (const entry of (await readdir(resolve(root, relative), { withFileTypes: true })).sort((a, b) =>
    a.name < b.name ? -1 : a.name > b.name ? 1 : 0)) {
    const path = relative ? relative + '/' + entry.name : entry.name;
    if (entry.isDirectory()) paths.push(...await listBuildFiles(root, path));
    else if (entry.isFile()) paths.push(path);
    else throw new Error('BUILD_SNAPSHOT_UNSUPPORTED_ENTRY: ' + path);
  }
  return paths;
};
const sources = sourcePaths();
const sourceHashes = await hashFiles(sources, path => path);
if (options.frames !== undefined && options.durationSeconds !== undefined) throw new Error('Choose --frames or --seconds');
if (options.durationSeconds !== undefined && (!Number.isSafeInteger(options.durationSeconds) || options.durationSeconds < 1 || options.durationSeconds > 3600)) {
  throw new Error('--seconds must be 1..3600');
}
if (options.frames !== undefined && (!Number.isSafeInteger(options.frames) || options.frames < 1 || options.frames > 1_000_000)) {
  throw new Error('--frames must be 1..1000000');
}
// Warmup and startup get ten minutes beyond the requested measured duration.
// Frame-count runs allow at least ten minutes, with a two-hour upper bound.
const runTimeoutMs = 1000 * (options.durationSeconds !== undefined
  ? options.durationSeconds + 600
  : Math.min(7200, Math.max(600, (options.frames ?? 1800) / 10 + 600)));
const metadata = { commit: git(['rev-parse', 'HEAD']), dirty: status !== '', status,
  sourceSha256: sourceHashes.sha256, sourceFiles: sourceHashes.files, startedAt: new Date().toISOString(),
  os: { platform: platform(), release: release(), architecture: arch(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
  browserSettings: settings, requestedOptions: options, runTimeoutMs, gpuDrivers: null };
if (platform() === 'win32') {
  try { metadata.gpuDrivers = JSON.parse(execFileSync('powershell.exe', ['-NoProfile', '-Command',
    'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json -Compress'],
  { encoding: 'utf8', windowsHide: true })); } catch { /* Explicit null when platform driver metadata is unavailable. */ }
}
await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(metadata, null, 2) + '\n');
execFileSync(process.execPath, ['node_modules/vite/bin/vite.js', 'build'], { stdio: 'inherit', windowsHide: true });
const postBuildSources = sourcePaths();
const postBuildSourceHashes = await hashFiles(postBuildSources, path => path);
if (postBuildSourceHashes.sha256 !== sourceHashes.sha256) {
  throw new Error('BENCHMARK_SOURCE_CHANGED_DURING_BUILD');
}
await cp('dist', snapshot, { recursive: true, errorOnExist: true, force: false });
const buildPaths = await listBuildFiles(snapshot);
if (!buildPaths.length) throw new Error('BENCHMARK_EMPTY_BUILD_SNAPSHOT');
const buildHashes = await hashFiles(buildPaths, path => resolve(snapshot, ...path.split('/')));
metadata.buildSha256 = buildHashes.sha256;
metadata.buildFiles = buildHashes.files;
await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(metadata, null, 2) + '\n');

async function evaluateWithTimeout(action, timeoutMs, repetition) {
  let timer;
  try {
    return await Promise.race([
      action(),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`BENCHMARK_RUN_TIMEOUT: repetition ${repetition} exceeded ${timeoutMs} ms`)), timeoutMs);
      }),
    ]);
  } finally { clearTimeout(timer); }
}
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.gltf': 'model/gltf+json',
  '.json': 'application/json', '.bin': 'application/octet-stream', '.png': 'image/png', '.wasm': 'application/wasm' };
const server = createServer(async (request, response) => {
  try {
    const relative = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    const path = resolve(snapshot, '.' + (relative === '/' ? '/benchmark.html' : relative));
    if (!path.startsWith(snapshot + sep) || !(await stat(path)).isFile()) { response.writeHead(404); response.end(); return; }
    response.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    createReadStream(path).pipe(response);
  } catch { response.writeHead(404); response.end(); }
});
await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
const baseUrl = 'http://127.0.0.1:' + server.address().port;
let browser;
const results = [];
try {
  const flags = settings.software ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader',
    ...(platform() === 'linux' ? ['--enable-gpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan',
      '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--use-vulkan=swiftshader'] : [])]
    : settings.highPerformanceGpu ? ['--force-high-performance-gpu'] : [];
  browser = await chromium.launch({ channel: settings.channel, headless: !settings.headed, args: flags });
  metadata.browserVersion = browser.version(); metadata.browserFlags = flags;
  await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(metadata, null, 2) + '\n');
  for (let repetition = 1; repetition <= settings.repeats; repetition++) {
    const context = await browser.newContext({ viewport: { width: options.width ?? 1920, height: options.height ?? 1080 }, deviceScaleFactor: 1 });
    const page = await context.newPage(), errors = [], hostSamples = [];
    page.on('pageerror', error => errors.push(error.message));
    let cdp;
    try { cdp = await context.newCDPSession(page); await cdp.send('Performance.enable'); } catch { /* Non-Chromium ports may omit host metrics. */ }
    let interval, sampling = false;
    try {
      await page.goto(baseUrl + '/benchmark.html');
      await page.waitForFunction(() => window.__REAL_BENCHMARK__);
      const adapterSelection = await page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) return null;
        const info = adapter.info;
        return { vendor: info.vendor, architecture: info.architecture, description: info.description,
          isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null };
      });
      // This second request is only corroborating evidence. Renderer.create
      // obtains its own adapter, whose identity is checked after the run.
      (metadata.adapterSelections ??= []).push({ repetition, adapter: adapterSelection });
      await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(metadata, null, 2) + '\n');
      console.log('BENCHMARK_PREFLIGHT_ADAPTER ' + JSON.stringify({ repetition, adapter: adapterSelection }));
      interval = setInterval(async () => {
        if (sampling) return;
        sampling = true;
        try {
          const progress = await page.evaluate(() => window.__REAL_BENCHMARK__.progress());
          const metrics = cdp ? await cdp.send('Performance.getMetrics') : null;
          const memory = metrics ? Object.fromEntries(metrics.metrics.filter(metric => ['JSHeapUsedSize', 'JSHeapTotalSize', 'Nodes', 'Documents', 'JSEventListeners'].includes(metric.name)).map(metric => [metric.name, metric.value])) : null;
          hostSamples.push({ observedAt: new Date().toISOString(), progress, memory });
          await writeFile(resolve(directory, `run-${repetition}-progress.json`), JSON.stringify(hostSamples, null, 2) + '\n');
          console.log('BENCHMARK_PROGRESS ' + JSON.stringify({ repetition, ...progress, memory }));
        } catch (error) { console.log('BENCHMARK_OBSERVATION ' + String(error)); }
        finally { sampling = false; }
      }, 15000);
      const result = await evaluateWithTimeout(() => page.evaluate(options => window.__REAL_BENCHMARK__.run(options), options),
        runTimeoutMs, repetition);
      const rendererAdapter = result.environment?.adapterInfo;
      (metadata.rendererAdapters ??= []).push({ repetition, adapter: rendererAdapter ?? null });
      await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(metadata, null, 2) + '\n');
      console.log('BENCHMARK_RENDERER_ADAPTER ' + JSON.stringify({ repetition, adapter: rendererAdapter ?? null }));
      if (!rendererAdapter || (!settings.software && rendererAdapter.isFallbackAdapter !== false)) {
        throw new Error('BENCHMARK_RENDERER_ADAPTER_UNCONFIRMED: ' + JSON.stringify(rendererAdapter));
      }
      if (settings.expectVendor !== null && rendererAdapter.vendor.toLowerCase() !== settings.expectVendor) {
        throw new Error('BENCHMARK_RENDERER_ADAPTER_MISMATCH: expected ' + settings.expectVendor + ', actual ' + rendererAdapter.vendor);
      }
      clearInterval(interval);
      await page.locator('#benchmark').screenshot({ path: resolve(directory, `run-${repetition}.png`) });
      const csv = await page.evaluate(() => window.__REAL_BENCHMARK__.csv());
      const cleanup = await page.evaluate(() => window.__REAL_BENCHMARK__.dispose());
      if (errors.length || cleanup.ownedGpuBytes !== 0) throw new Error('BENCHMARK_CLEANUP_OR_PAGE_ERROR: ' + JSON.stringify({ errors, cleanup }));
      const report = { ...result, repetition, runTimeoutMs,
        adapterEvidence: { preflight: adapterSelection, renderer: rendererAdapter },
        cleanup, pageErrors: errors, hostSamples, manifest: 'manifest.json' };
      await writeFile(resolve(directory, `run-${repetition}.json`), JSON.stringify(report, null, 2) + '\n');
      await writeFile(resolve(directory, `run-${repetition}.csv`), csv);
      results.push({ repetition, measuredDurationMs: result.measuredDurationMs, summary: result.summary, verification: result.verification, cleanup });
      console.log('BENCHMARK_RESULT ' + JSON.stringify(results.at(-1)));
    } finally { clearInterval(interval); await context.close(); }
  }
  await writeFile(resolve(directory, 'summary.json'), JSON.stringify({ manifest: metadata, runs: results }, null, 2) + '\n');
  console.log('BENCHMARK_OUTPUT ' + directory);
} catch (error) {
  await writeFile(resolve(directory, 'failure.json'), JSON.stringify({ error: String(error), completedRuns: results, time: new Date().toISOString() }, null, 2) + '\n');
  throw error;
} finally { await browser?.close(); await new Promise(resolveClose => server.close(resolveClose)); }
