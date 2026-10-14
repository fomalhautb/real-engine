import { chromium } from '@playwright/test';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { cp, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { arch, cpus, platform, release, totalmem } from 'node:os';
import { extname, relative, resolve, sep } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const dist = resolve(root, 'dist');
const artifacts = resolve(root, 'artifacts');
const stamp = new Date().toISOString().replaceAll(/[:.]/g, '-');
const defaults = { page: 'gi', channel: 'chrome', warmup: 60, frames: 300, repeat: 1,
  width: 1280, height: 720, highPerformanceGpu: false, expectVendor: null, parseOnly: false, finalGather: false, camera: 'fixed',
  out: resolve(artifacts, 'benchmarks', `demo-${stamp}`) };
const bounds = { warmup: [0, 10_000], frames: [30, 36_000], repeat: [1, 3],
  width: [320, 3840], height: [240, 2160] };

function parseArgs(argv) {
  const options = { ...defaults };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === '--high-performance-gpu') { options.highPerformanceGpu = true; continue; }
    if (key === '--final-gather') { options.finalGather = true; continue; }
    if (key === '--parse-only') { options.parseOnly = true; continue; }
    if (!['--page', '--channel', '--warmup', '--frames', '--repeat', '--width', '--height', '--out', '--expect-vendor', '--camera'].includes(key)) {
      throw new Error(`Unknown benchmark argument: ${key}`);
    }
    const raw = argv[++i];
    if (raw === undefined || raw.startsWith('--')) throw new Error(`Missing value for ${key}`);
    const name = key === '--expect-vendor' ? 'expectVendor' : key.slice(2);
    options[name] = name in bounds ? Number(raw) : raw;
  }
  if (!['gi', 'geometry'].includes(options.page)) throw new Error('--page must be gi or geometry');
  if (options.finalGather && options.page !== 'gi') throw new Error('--final-gather requires --page gi');
  if (!['fixed', 'orbit'].includes(options.camera)) throw new Error('--camera must be fixed or orbit');
  if (options.camera === 'orbit' && options.page !== 'gi') throw new Error('--camera orbit requires --page gi');
  if (!['msedge', 'chrome', 'chromium'].includes(options.channel)) throw new Error('--channel must be msedge, chrome, or chromium');
  if (options.expectVendor !== null && !/^[a-z][a-z0-9-]{0,31}$/.test(options.expectVendor)) {
    throw new Error('--expect-vendor must be an exact lowercase vendor token (1..32 characters)');
  }
  for (const [name, [min, max]] of Object.entries(bounds)) {
    if (!Number.isSafeInteger(options[name]) || options[name] < min || options[name] > max) {
      throw new Error(`--${name} must be an integer in ${min}..${max}`);
    }
  }
  options.out = resolve(root, options.out);
  const rel = relative(artifacts, options.out);
  if (!rel || rel === '..' || rel.startsWith('..' + sep) || resolve(artifacts, rel) !== options.out) {
    throw new Error('--out must be a directory inside ignored artifacts/');
  }
  return options;
}

function git(args) {
  return execFileSync('git', args, { cwd: root, encoding: 'utf8', windowsHide: true }).trim();
}

function sourcePaths() {
  return execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'],
    { cwd: root, encoding: 'utf8', windowsHide: true }).split('\0').filter(Boolean).sort();
}

async function hashFiles(paths, locate) {
  const aggregate = createHash('sha256');
  const files = {};
  for (const path of paths) {
    const bytes = await readFile(locate(path));
    const digest = createHash('sha256').update(bytes).digest('hex');
    files[path] = digest;
    aggregate.update(path).update('\0').update(digest).update('\n');
  }
  return { sha256: aggregate.digest('hex'), files };
}

async function listFiles(directory, prefix = '') {
  const files = [];
  for (const entry of (await readdir(resolve(directory, prefix), { withFileTypes: true }))
    .sort((a, b) => a.name.localeCompare(b.name, 'en'))) {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...await listFiles(directory, path));
    else if (entry.isFile()) files.push(path);
    else throw new Error(`BUILD_UNSUPPORTED_ENTRY: ${path}`);
  }
  return files;
}

const buildHash = async directory => {
  const paths = await listFiles(directory);
  if (!paths.length) throw new Error('BUILD_EMPTY');
  return hashFiles(paths, path => resolve(directory, ...path.split('/')));
};

function summary(values) {
  if (!values.length) return { count: 0, p50: null, p95: null, p99: null, min: null, max: null };
  if (values.some(value => !Number.isFinite(value) || value < 0)) throw new Error('BENCHMARK_INVALID_SAMPLE');
  const sorted = values.toSorted((a, b) => a - b);
  const percentile = p => {
    const index = (sorted.length - 1) * p;
    const low = Math.floor(index), high = Math.ceil(index);
    return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
  };
  return { count: sorted.length, p50: percentile(.5), p95: percentile(.95), p99: percentile(.99),
    min: sorted[0], max: sorted.at(-1) };
}

function canonicalCamera(camera) {
  const result = { eye: camera.eye, target: camera.target, phase: camera.phase,
    yaw: camera.yaw, radius: camera.radius ?? camera.distance, elevation: camera.elevation };
  if (!Array.isArray(result.eye) || result.eye.length !== 3 || !Array.isArray(result.target) ||
      result.target.length !== 3 || ![...result.eye, ...result.target, result.phase, result.yaw,
        result.radius, result.elevation].every(Number.isFinite)) throw new Error('BENCHMARK_CAMERA_INVALID');
  return result;
}

function csv(rows, columns) {
  return [columns.join(','), ...rows.map(row => columns.map(column => {
    const value = row[column];
    return value === null || value === undefined ? '' : String(value);
  }).join(','))].join('\n') + '\n';
}

function createStaticServer(snapshot, pageName) {
  const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.bin': 'application/octet-stream',
    '.png': 'image/png', '.wasm': 'application/wasm' };
  return createServer(async (request, response) => {
    try {
      if (request.method !== 'GET' && request.method !== 'HEAD') throw new Error('BAD_METHOD');
      const raw = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
      if (raw === '/favicon.ico') { response.writeHead(204, { 'cache-control': 'no-store' }); response.end(); return; }
      const path = resolve(snapshot, '.' + (raw === '/' ? `/${pageName}.html` : raw));
      if (!path.startsWith(snapshot + sep) || !(await stat(path)).isFile()) throw new Error('NOT_FOUND');
      response.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream',
        'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
      if (request.method === 'HEAD') response.end();
      else createReadStream(path).pipe(response);
    } catch { response.writeHead(404); response.end(); }
  });
}

async function withTimeout(promise, milliseconds, label) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} exceeded ${milliseconds} ms`)), milliseconds);
    })]);
  } finally { clearTimeout(timer); }
}

// Runs inside a paused demo. One deterministic camera pose is rendered per rAF.
async function measureDemo({ pageName, warmup, frames, width, height, expectVendor, expectedCamera, cameraMode }) {
  const demo = pageName === 'gi' ? window.__GI_DEMO__ : window.__GEOMETRY_DEMO__;
  if (!demo?.renderer) throw new Error(`${pageName.toUpperCase()}_DEMO_RENDERER_UNAVAILABLE`);
  const initial = demo.snapshot();
  if (initial.controls.paused !== true || initial.controls.diagnosticsEnabled !== false) {
    throw new Error('BENCHMARK_DEMO_MODE_CHANGED');
  }
  const initialCamera = { eye: initial.camera.eye, target: initial.camera.target,
    phase: initial.camera.phase, yaw: initial.camera.yaw,
    radius: initial.camera.radius ?? initial.camera.distance, elevation: initial.camera.elevation };
  if (JSON.stringify(initialCamera) !== JSON.stringify(expectedCamera)) throw new Error('BENCHMARK_CAMERA_CHANGED_BEFORE_WARMUP');
  const eye = [...initial.camera.eye], target = [...initial.camera.target];
  const renderer = demo.renderer;
  const canvas = document.querySelector('#viewport');
  if (canvas.clientWidth !== width || canvas.clientHeight !== height) {
    throw new Error(`BENCHMARK_CANVAS_SIZE: ${canvas.clientWidth}x${canvas.clientHeight} vs ${width}x${height}`);
  }
  const poseAt = index => {
    if (cameraMode === 'fixed') return { eye, target };
    const phase = index / 60;
    const yaw = initialCamera.yaw + 0.1 * Math.sin(phase * 0.3);
    const elevation = initialCamera.elevation + 0.02 * Math.sin(phase * 0.2);
    const radius = initialCamera.radius;
    return { eye: [target[0] + Math.sin(yaw) * Math.cos(elevation) * radius,
      target[1] + Math.sin(elevation) * radius, target[2] + Math.cos(yaw) * Math.cos(elevation) * radius], target };
  };
  const result = { initial, camera: { eye, target, mode: cameraMode,
    trajectory: cameraMode === 'orbit' ? { nominalStepSeconds: 1 / 60,
      yawAmplitude: 0.1, yawFrequency: 0.3, elevationAmplitude: 0.02, elevationFrequency: 0.2 } : null },
    warmup, frames, frameSamples: [], gpuSamples: [],
    memoryChanges: [], measuredDurationMs: 0, final: null, adapterCheck: null };
  const actual = renderer.info.adapterInfo;
  const adapterText = String(renderer.info.adapter);
  if (!actual || !actual.vendor || actual.isFallbackAdapter !== false ||
      /swiftshader|software|llvmpipe|lavapipe|\bwarp\b|basic render|fallback/i.test(adapterText)) {
    throw new Error(`NATIVE_ADAPTER_UNCONFIRMED: ${adapterText}; actual=${JSON.stringify(actual)}`);
  }
  const rendererVendor = String(actual.vendor).trim().toLowerCase();
  if (expectVendor !== null && rendererVendor !== expectVendor) {
    throw new Error(`UNEXPECTED_GPU_VENDOR: expected=${expectVendor}, actual=${rendererVendor}, adapter=${adapterText}`);
  }
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance', forceFallbackAdapter: false });
  result.adapterCheck = adapter ? { vendor: adapter.info?.vendor ?? null,
    architecture: adapter.info?.architecture ?? null, description: adapter.info?.description ?? null,
    isFallbackAdapter: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null } : null;
  result.adapterCheckMatchesActual = result.adapterCheck?.vendor?.toLowerCase() === rendererVendor;
  if (pageName === 'gi') { demo.toggleGi(true); demo.setMultiBounce(true); }
  const waitFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
  let lastGpuFrame = renderer.stats.gpuSampleFrame;
  result.initialOwnedGpuBytes = renderer.stats.ownedGpuBytes;
  let lastMemory = renderer.stats.ownedGpuBytes;
  const captureGpu = () => {
    const stats = renderer.stats;
    if (stats.gpuSampleFrame === null || stats.gpuSampleFrame === lastGpuFrame) return;
    lastGpuFrame = stats.gpuSampleFrame;
    if (stats.gpuSampleFrame <= result.warmupLastRenderFrame ||
        stats.gpuSampleFrame > result.lastRenderFrame || !stats.gpuMs) return;
    const values = { ...stats.gpuMs };
    const passNames = ['deformation', 'shadow', 'surface', 'resolve', 'worldQueryRefit', 'geometryLod', 'diffuseGi', 'finalGather'];
    if (passNames.some(name => !Number.isFinite(values[name]) || values[name] < 0)) throw new Error('INVALID_GPU_TIMING');
    result.gpuSamples.push({ renderFrame: stats.gpuSampleFrame, ...values,
      fullGpuMs: passNames.reduce((total, name) => total + values[name], 0) });
  };
  for (let index = 0; index < warmup; index++) {
    await waitFrame();
    const pose = poseAt(index);
    renderer.render(pose.eye, pose.target);
    if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
  }
  result.warmupLastRenderFrame = renderer.stats.frames;
  result.warmupEndOwnedGpuBytes = renderer.stats.ownedGpuBytes;
  lastMemory = result.warmupEndOwnedGpuBytes;
  if (renderer.stats.width !== width || renderer.stats.height !== height) {
    throw new Error(`BENCHMARK_RENDER_SIZE: ${renderer.stats.width}x${renderer.stats.height} vs ${width}x${height}`);
  }
  result.firstRenderFrame = result.warmupLastRenderFrame + 1;
  let lastRaf = null;
  const started = performance.now();
  for (let index = 0; index < frames; index++) {
    const rafTime = await waitFrame();
    const pose = poseAt(warmup + index);
    renderer.render(pose.eye, pose.target);
    const stats = renderer.stats;
    if (stats.width !== width || stats.height !== height) throw new Error('BENCHMARK_RENDER_SIZE_CHANGED');
    result.lastRenderFrame = stats.frames;
    if (renderer.errors.length) throw new Error(renderer.errors.join('\n'));
    result.frameSamples.push({ index, renderFrame: stats.frames,
      rafDeltaMs: lastRaf === null ? null : rafTime - lastRaf,
      cpuSubmitMs: stats.cpuSubmitMs, ownedGpuBytes: stats.ownedGpuBytes,
      drawCalls: stats.drawCalls, triangles: stats.triangles, uploadedBytes: stats.uploadedBytes,
      eyeX: pose.eye[0], eyeY: pose.eye[1], eyeZ: pose.eye[2] });
    if (stats.ownedGpuBytes !== lastMemory) {
      result.memoryChanges.push({ index, renderFrame: stats.frames, from: lastMemory, to: stats.ownedGpuBytes });
      lastMemory = stats.ownedGpuBytes;
    }
    captureGpu();
    lastRaf = rafTime;
  }
  result.measuredDurationMs = performance.now() - started;
  // Let the asynchronous timestamp readback complete without submitting extra measured frames.
  for (let i = 0; i < 120; i++) {
    await waitFrame();
    captureGpu();
    if (renderer.stats.gpuSampleFrame === result.lastRenderFrame) break;
  }
  result.final = demo.snapshot();
  result.camera.lastRenderedEye = [...poseAt(warmup + frames - 1).eye];
  result.camera.lastRenderedTarget = [...target];
  if (result.final.camera.eye.some((value, i) => value !== eye[i]) ||
      result.final.camera.target.some((value, i) => value !== target[i])) throw new Error('BENCHMARK_CAMERA_MOVED');
  if (result.final.errors.length) throw new Error(result.final.errors.join('\n'));
  return result;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.parseOnly) { console.log(JSON.stringify(options)); return; }
  const output = options.out;
  const snapshot = resolve(output, `build-${stamp}`);
  const sourceBefore = await hashFiles(sourcePaths(), path => resolve(root, path));
  const status = git(['status', '--porcelain']);
  const manifest = { createdAt: new Date().toISOString(), commit: git(['rev-parse', 'HEAD']),
    dirty: status !== '', status, source: sourceBefore,
    buildSourceCorrespondence: 'Vite production build from sourceSha256; source verified unchanged after build and snapshot',
    snapshot: `build-${stamp}`, options, scope: 'tiny production demo fixture; not large-scene acceptance',
    timingInterpretation: 'sparse GPU timestamp pass spans; fullGpuMs is their sum, not CPU+GPU or presented-frame latency',
    pacing: 'headless requestAnimationFrame, one selected deterministic camera pose per callback; rAF interval recorded separately',
    os: { platform: platform(), release: release(), architecture: arch(),
      cpu: cpus()[0]?.model ?? null, logicalCpus: cpus().length, memoryBytes: totalmem() },
    diagnosticReadback: 'benchmark=1 starts paused with diagnostics disabled; no initial automatic probe; explicit disable is reasserted before warmup' };
  await mkdir(output, { recursive: true });
  if ((await readdir(output)).length) throw new Error('BENCHMARK_OUTPUT_NOT_EMPTY');
  await writeFile(resolve(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  execFileSync(process.execPath, ['node_modules/vite/bin/vite.js', 'build'],
    { cwd: root, stdio: 'inherit', windowsHide: true });
  const sourceAfterBuild = await hashFiles(sourcePaths(), path => resolve(root, path));
  if (sourceBefore.sha256 !== sourceAfterBuild.sha256) throw new Error('SOURCE_CHANGED_DURING_BUILD');
  const buildBefore = await buildHash(dist);
  if (!buildBefore.files[`${options.page}.html`]) throw new Error(`BUILD_PAGE_MISSING: ${options.page}.html`);
  manifest.dist = buildBefore;
  await cp(dist, snapshot, { recursive: true, errorOnExist: true, force: false });
  const [sourceAfter, distAfter, snapshotHash] = await Promise.all([
    hashFiles(sourcePaths(), path => resolve(root, path)), buildHash(dist), buildHash(snapshot),
  ]);
  if (sourceBefore.sha256 !== sourceAfter.sha256) throw new Error('SOURCE_CHANGED_DURING_SNAPSHOT');
  if (buildBefore.sha256 !== distAfter.sha256 || buildBefore.sha256 !== snapshotHash.sha256) {
    throw new Error('DIST_CHANGED_DURING_SNAPSHOT');
  }
  manifest.snapshotSha256 = snapshotHash.sha256;
  await writeFile(resolve(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

  const server = createStaticServer(snapshot, options.page);
  await new Promise((resolveListen, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  let browser;
  const runs = [];
  let referenceCamera = null;
  try {
    const browserFlags = options.highPerformanceGpu ? ['--force-high-performance-gpu'] : [];
    browser = await chromium.launch({ channel: options.channel, headless: true, args: browserFlags });
    manifest.browser = { version: browser.version(), channel: options.channel, headless: true,
      flags: browserFlags };
    await writeFile(resolve(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
    for (let repeat = 1; repeat <= options.repeat; repeat++) {
      const context = await browser.newContext({ viewport: { width: options.width, height: options.height },
        deviceScaleFactor: 1 });
      const page = await context.newPage();
      const pageErrors = [];
      page.on('pageerror', error => pageErrors.push(`pageerror: ${error.message}`));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        const { url, lineNumber, columnNumber } = message.location();
        pageErrors.push(`console: ${message.text()} @ ${url || 'unknown'}:${lineNumber}:${columnNumber}`);
      });
      page.on('response', response => {
        if (response.status() >= 400) pageErrors.push(`HTTP ${response.status()}: ${response.url()}`);
      });
      page.on('crash', () => pageErrors.push('page crashed'));
      try {
        await page.goto(`http://127.0.0.1:${server.address().port}/${options.page}.html?benchmark=1${options.finalGather ? '&finalGather=1' : ''}`);
        const harness = options.page === 'gi' ? '__GI_DEMO__' : '__GEOMETRY_DEMO__';
        await page.waitForFunction(name => Boolean(window[name]?.renderer), harness, { timeout: 120_000 });
        const firstSnapshot = await page.evaluate(name => window[name].snapshot(), harness);
        if (firstSnapshot.controls.paused !== true || firstSnapshot.controls.diagnosticsEnabled !== false) {
          throw new Error('BENCHMARK_QUERY_MODE_UNAVAILABLE');
        }
        const runCamera = canonicalCamera(firstSnapshot.camera);
        if (runCamera.phase !== 0) throw new Error(`BENCHMARK_CAMERA_PHASE: ${runCamera.phase}`);
        if (referenceCamera === null) {
          referenceCamera = runCamera;
          manifest.canonicalCamera = runCamera;
          await writeFile(resolve(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
        } else if (JSON.stringify(runCamera) !== JSON.stringify(referenceCamera)) {
          throw new Error(`BENCHMARK_CAMERA_REPETITION_MISMATCH: ${JSON.stringify({ expected: referenceCamera, actual: runCamera })}`);
        }
        await page.evaluate(({ name, width, height }) => {
          const demo = window[name];
          if (demo.snapshot().controls.paused !== true) throw new Error('BENCHMARK_DEMO_NOT_PAUSED');
          demo.setDiagnosticsEnabled(false);
          document.querySelector(name === '__GI_DEMO__' ? '#gi-app' : '#geometry-app')?.classList.add('ui-hidden');
          const canvas = document.querySelector('#viewport');
          canvas.style.position = 'fixed';
          canvas.style.left = '0';
          canvas.style.top = '0';
          canvas.style.width = `${width}px`;
          canvas.style.height = `${height}px`;
          canvas.style.maxWidth = 'none';
          canvas.style.maxHeight = 'none';
          canvas.style.zIndex = '1000';
        }, { name: harness, width: options.width, height: options.height });
        const timeoutMs = Math.min(30 * 60_000, Math.max(120_000,
          (options.warmup + options.frames) * 100 + 120_000));
        const result = await withTimeout(page.evaluate(measureDemo, {
          pageName: options.page, warmup: options.warmup, frames: options.frames,
          width: options.width, height: options.height, expectVendor: options.expectVendor,
          expectedCamera: referenceCamera,
          cameraMode: options.camera,
        }), timeoutMs, `BENCHMARK_RUN_${repeat}`);
        if (pageErrors.length) throw new Error('BENCHMARK_PAGE_ERROR: ' + JSON.stringify(pageErrors));
        const passes = ['deformation', 'shadow', 'surface', 'resolve', 'worldQueryRefit', 'geometryLod', 'diffuseGi', 'finalGather'];
        const metrics = { cpuSubmitMs: summary(result.frameSamples.map(row => row.cpuSubmitMs)),
          rafDeltaMs: summary(result.frameSamples.map(row => row.rafDeltaMs).filter(value => value !== null)),
          fullGpuMs: summary(result.gpuSamples.map(row => row.fullGpuMs)),
          gpuPasses: Object.fromEntries(passes.map(name => [name, summary(result.gpuSamples.map(row => row[name]))])),
          gpuTimingAvailable: result.gpuSamples.length > 0,
          gpuTimestampQuerySupported: result.final.adapter.features.includes('timestamp-query'),
          gpuSamples: result.gpuSamples.length, gpuSampledFrames: result.gpuSamples.map(row => row.renderFrame),
          ownedGpuBytes: { first: result.frameSamples[0].ownedGpuBytes,
            last: result.frameSamples.at(-1).ownedGpuBytes,
            initialBeforeWarmup: result.initialOwnedGpuBytes,
            afterWarmup: result.warmupEndOwnedGpuBytes,
            min: Math.min(...result.frameSamples.map(row => row.ownedGpuBytes)),
            max: Math.max(...result.frameSamples.map(row => row.ownedGpuBytes)),
            changes: result.memoryChanges } };
        const geometryActual = options.page === 'geometry' ? await page.evaluate(async name => {
          const renderer = window[name].renderer;
          const batches = renderer.diagnostics().geometryLod?.batches ?? [];
          const measured = [];
          for (const batch of batches) {
            if (!batch.active) continue;
            const probe = await renderer.probeGeometryLod(batch.name);
            if (probe.overflow) throw new Error(`GEOMETRY_LOD_PROBE_OVERFLOW: ${batch.name}`);
            const levels = {};
            for (const cluster of probe.selectedClusters) levels[cluster.level] = (levels[cluster.level] ?? 0) + 1;
            measured.push({ name: batch.name, frame: probe.frame, actualSelectedIndices: probe.indices,
              actualSelectedTriangles: probe.triangles, selectedClusterCount: probe.selectedClusters.length,
              selectedClusterLevels: levels, sourceTrianglesPerInstance: batch.sourceTriangles,
              instances: batch.instances, sourceExpandedTriangles: batch.sourceTriangles * batch.instances });
          }
          return { batches: measured,
            actualSelectedTriangles: measured.reduce((sum, batch) => sum + batch.actualSelectedTriangles, 0),
            sourceExpandedTriangles: measured.reduce((sum, batch) => sum + batch.sourceExpandedTriangles, 0) };
        }, harness) : null;
        if (geometryActual && geometryActual.batches.some(batch => batch.frame !== result.lastRenderFrame)) {
          throw new Error('GEOMETRY_LOD_PROBE_FRAME_MISMATCH');
        }
        const screenshot = resolve(output, `run-${repeat}.png`);
        await page.locator('#viewport').screenshot({ path: screenshot });
        const cleanup = await page.evaluate(name => {
          const demo = window[name], renderer = demo.renderer;
          demo.dispose();
          return { ownedGpuBytes: renderer.stats.ownedGpuBytes, errors: [...renderer.errors] };
        }, harness);
        if (cleanup.ownedGpuBytes !== 0 || cleanup.errors.length || pageErrors.length) {
          throw new Error('BENCHMARK_CLEANUP_ERROR: ' + JSON.stringify({ cleanup, pageErrors }));
        }
        const report = { repeat, page: options.page, scope: manifest.scope,
          diagnosticReadback: manifest.diagnosticReadback, camera: result.camera,
          adapter: result.final.adapter, adapterCheck: result.adapterCheck,
          adapterCheckMatchesActual: result.adapterCheckMatchesActual,
          initialCamera: runCamera,
          scene: result.final.scene ?? result.final.source, controls: result.final.controls,
          sourceExpandedTriangles: result.frameSamples.at(-1).triangles,
          geometryActual,
          metrics, measuredDurationMs: result.measuredDurationMs,
          firstRenderFrame: result.firstRenderFrame, lastRenderFrame: result.lastRenderFrame,
          frameSamples: result.frameSamples, gpuSamples: result.gpuSamples,
          finalDiagnostics: result.final.diagnostics, pageErrors, cleanup,
          screenshot: `run-${repeat}.png`, manifest: 'manifest.json' };
        await writeFile(resolve(output, `run-${repeat}.json`), JSON.stringify(report, null, 2) + '\n');
        await writeFile(resolve(output, `run-${repeat}-frames.csv`), csv(result.frameSamples,
          ['index', 'renderFrame', 'rafDeltaMs', 'cpuSubmitMs', 'ownedGpuBytes', 'drawCalls', 'triangles', 'uploadedBytes', 'eyeX', 'eyeY', 'eyeZ']));
        await writeFile(resolve(output, `run-${repeat}-gpu.csv`), csv(result.gpuSamples,
          ['renderFrame', ...passes, 'fullGpuMs']));
        runs.push({ repeat, metrics, adapter: result.final.adapter, cleanup });
        console.log('BENCHMARK_DEMO_RESULT ' + JSON.stringify({ repeat, metrics, adapter: result.final.adapter }));
      } finally { await context.close(); }
    }
    await writeFile(resolve(output, 'summary.json'), JSON.stringify({ manifest: 'manifest.json', runs }, null, 2) + '\n');
    console.log('BENCHMARK_DEMO_OUTPUT ' + output);
  } catch (error) {
    await writeFile(resolve(output, 'failure.json'), JSON.stringify({ error: String(error), completedRuns: runs,
      at: new Date().toISOString() }, null, 2) + '\n');
    throw error;
  } finally {
    await browser?.close();
    await new Promise(resolveClose => server.close(resolveClose));
  }
}

await main();
