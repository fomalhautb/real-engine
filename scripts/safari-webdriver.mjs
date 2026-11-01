#!/usr/bin/env node
// Runs a defined set of engine checks in Safari over W3C WebDriver (safaridriver).
// Playwright cannot drive Safari, so this script replaces the Playwright `page` with plain
// WebDriver commands. It starts a private Vite dev server for this checkout and its own
// safaridriver, writes screenshots and a JSON report, and always deletes the session, stops the
// driver and closes the server, also on failure or Ctrl-C.
//   node scripts/safari-webdriver.mjs --out artifacts/<run>/safari [--port 5267] [--driver-port 5268] [--only id,id] [--list]
// Remote automation must already be enabled by the user (Safari > Develop > Allow Remote
// Automation). The script never changes Safari or system settings and navigates only to its own
// dev server. Results are functional evidence for Safari; timings are diagnostic only.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SAFARIDRIVER = '/usr/bin/safaridriver';
const ELEMENT = 'element-6066-11e4-a52e-4f735466cecf';
// Window rects in screen points, toolbar included. engine.spec.ts renders a 1440 x 1000 viewport and
// resizes it to 1100 x 820. Safari sizes the whole window, and a 1512 x 982 pt laptop screen has no
// room for a 1000 pt viewport, so the runner opens every check in LARGE_WINDOW at the top-left of
// the screen, the resize check narrows it to SMALL_WINDOW, and results record the rects and
// viewports Safari actually used.
const LARGE_WINDOW = { x: 0, y: 0, width: 1440, height: 940 };
const SMALL_WINDOW = { x: 0, y: 0, width: 1100, height: 820 };

// ---------------------------------------------------------------------------------------------
// Arguments

function parseArguments(argv) {
  const options = { out: undefined, port: Number(process.env.PLAYWRIGHT_PORT ?? 5173), driverPort: 5268, only: undefined, list: false };
  for (let i = 0; i < argv.length; i++) {
    const name = argv[i], value = argv[i + 1];
    if (name === '--list') { options.list = true; continue; }
    if (value === undefined || value.startsWith('--')) throw new Error(`SAFARI_RUNNER_ARGUMENTS: ${name} needs a value`);
    i++;
    if (name === '--out') options.out = resolve(value);
    else if (name === '--port') options.port = Number(value);
    else if (name === '--driver-port') options.driverPort = Number(value);
    else if (name === '--only') options.only = value.split(',').map(id => id.trim()).filter(Boolean);
    else throw new Error('SAFARI_RUNNER_ARGUMENTS: unknown option ' + name);
  }
  for (const [label, port] of [['--port', options.port], ['--driver-port', options.driverPort]]) {
    if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error(`SAFARI_RUNNER_ARGUMENTS: ${label} must be an integer in 1024..65535`);
  }
  if (options.port === options.driverPort) throw new Error('SAFARI_RUNNER_ARGUMENTS: --port and --driver-port must differ');
  if (!options.list && !options.out) throw new Error('SAFARI_RUNNER_ARGUMENTS: --out <directory> is required');
  return options;
}

// ---------------------------------------------------------------------------------------------
// Minimal W3C WebDriver client

class WebDriverError extends Error {
  constructor(status, error, message) { super(`WEBDRIVER ${status} ${error}: ${message}`); this.error = error; }
}

class WebDriver {
  constructor(base) { this.base = base; this.session = undefined; this.capabilities = undefined; }

  async command(method, path, body, timeoutMs = 600_000) {
    const response = await fetch(this.base + path, {
      method, headers: body === undefined ? undefined : { 'content-type': 'application/json; charset=utf-8' },
      body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await response.text();
    let payload;
    try { payload = text ? JSON.parse(text) : {}; } catch { throw new WebDriverError(response.status, 'invalid response', text.slice(0, 500)); }
    const value = payload.value;
    if (!response.ok || (value && typeof value === 'object' && typeof value.error === 'string')) {
      throw new WebDriverError(response.status, value?.error ?? 'unknown error', value?.message ?? text.slice(0, 500));
    }
    return value;
  }

  sessionCommand(method, path, body, timeoutMs) {
    if (!this.session) throw new Error('WEBDRIVER_NO_SESSION');
    return this.command(method, `/session/${this.session}${path}`, body, timeoutMs);
  }

  async status() { return this.command('GET', '/status', undefined, 2_000); }

  async newSession() {
    // Only standard capabilities: the session must not change Safari preferences.
    const value = await this.command('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } }, 60_000);
    this.session = value.sessionId;
    this.capabilities = value.capabilities;
    await this.sessionCommand('POST', '/timeouts', { script: 600_000, pageLoad: 120_000, implicit: 0 });
    return value;
  }

  async deleteSession() {
    if (!this.session) return;
    const session = this.session;
    this.session = undefined;
    await this.command('DELETE', `/session/${session}`, undefined, 30_000);
  }

  navigate(url) { return this.sessionCommand('POST', '/url', { url }); }
  setWindowRect({ x, y, width, height }) { return this.sessionCommand('POST', '/window/rect', { x, y, width, height }); }

  async screenshot() { return Buffer.from(await this.sessionCommand('GET', '/screenshot'), 'base64'); }

  async elementScreenshot(selector) {
    const element = await this.sessionCommand('POST', '/element', { using: 'css selector', value: selector });
    return Buffer.from(await this.sessionCommand('GET', `/element/${element[ELEMENT]}/screenshot`), 'base64');
  }

  /**
   * Runs `fn` (a self-contained page function, possibly async) with JSON arguments and returns its
   * JSON result. Typed arrays become arrays and non-finite numbers become strings, so a NaN in a
   * result fails a numeric check instead of turning into null.
   */
  async execute(fn, ...args) {
    const script = `const done = arguments[arguments.length - 1];
const args = Array.prototype.slice.call(arguments, 0, -1);
const fn = ${fn.toString()};
const replacer = (key, value) => ArrayBuffer.isView(value) ? Array.from(value)
  : typeof value === 'number' && !Number.isFinite(value) ? String(value) : value;
Promise.resolve().then(() => fn(...args)).then(
  value => done({ ok: true, json: JSON.stringify(value === undefined ? null : value, replacer) }),
  error => done({ ok: false, error: error && error.stack ? String(error) + '\\n' + error.stack : String(error) }));`;
    const result = await this.sessionCommand('POST', '/execute/async', { script, args });
    if (!result?.ok) throw new Error('PAGE_SCRIPT_FAILED: ' + (result?.error ?? 'no result'));
    return JSON.parse(result.json);
  }

  /** Polls a synchronous page predicate until it returns a truthy value. */
  async waitFor(label, fn, timeoutMs = 60_000, ...args) {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const value = await this.execute(fn, ...args);
      if (value) return value;
      if (Date.now() > deadline) throw new Error(`WAIT_TIMEOUT: ${label} after ${timeoutMs} ms`);
      await new Promise(resolveWait => setTimeout(resolveWait, 100));
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Assertions (each failure names the value that broke the check)

function check(condition, message) { if (!condition) throw new Error('CHECK_FAILED: ' + message); }
const equal = (actual, expected, label) => check(Object.is(actual, expected) || JSON.stringify(actual) === JSON.stringify(expected),
  `${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
const finite = (value, label) => check(typeof value === 'number' && Number.isFinite(value), `${label}: expected a finite number, got ${JSON.stringify(value)}`);
const below = (value, limit, label) => { finite(value, label); check(value < limit, `${label}: ${value} is not below ${limit}`); };
const above = (value, limit, label) => { finite(value, label); check(value > limit, `${label}: ${value} is not above ${limit}`); };
const atLeast = (value, limit, label) => { finite(value, label); check(value >= limit, `${label}: ${value} is below ${limit}`); };
const empty = (list, label) => check(Array.isArray(list) && list.length === 0, `${label}: expected [], got ${JSON.stringify(list)?.slice(0, 400)}`);
const contains = (text, part, label) => check(typeof text === 'string' && text.includes(part), `${label}: ${JSON.stringify(text)} lacks ${JSON.stringify(part)}`);

// ---------------------------------------------------------------------------------------------
// Page functions. They run in Safari and must be self-contained (serialized with toString()).

const engineReady = () => window.__REAL_ENGINE__ || !document.getElementById('error').hidden;
const framesAtLeast = count => (window.__REAL_ENGINE__?.snapshot().stats?.frames ?? 0) >= count || !document.getElementById('error').hidden;
const pageError = () => { const node = document.getElementById('error'); return node && !node.hidden ? node.textContent : ''; };
const snapshot = () => window.__REAL_ENGINE__.snapshot();
const viewport = () => ({ innerWidth, innerHeight, devicePixelRatio });
/**
 * The drawing-buffer size once it follows the canvas the way Renderer.resize() sizes it (CSS size
 * times min(devicePixelRatio, 1.5)); with `narrowerThan`, only once it is narrower than that width.
 */
const settledDrawingBuffer = narrowerThan => {
  const stats = window.__REAL_ENGINE__?.snapshot().stats, canvas = document.getElementById('viewport');
  if (!stats || !canvas) return null;
  const ratio = Math.min(devicePixelRatio, 1.5);
  if (stats.width !== Math.max(1, Math.round(canvas.clientWidth * ratio))
    || stats.height !== Math.max(1, Math.round(canvas.clientHeight * ratio))) return null;
  if (narrowerThan != null && stats.width >= narrowerThan) return null;
  return { width: stats.width, height: stats.height, innerWidth, innerHeight, devicePixelRatio };
};

/** Counts blue character and green tree pixels of a PNG, with the thresholds of engine.spec.ts. */
async function countScenePixels(base64) {
  const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
  const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
  const probe = document.createElement('canvas');
  probe.width = bitmap.width; probe.height = bitmap.height;
  const context = probe.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
  let blue = 0, green = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    if (b > r * 1.3 && b > g * 1.15 && b > 45) blue++;
    if (g > r * 1.15 && g > b * 1.15 && g > 40) green++;
  }
  return { blue, green, width: probe.width, height: probe.height };
}

/**
 * Imports a browser-spec fixture module from the dev server and calls one export. Playwright runs
 * every spec at deviceScaleFactor 1, and several fixtures size their canvases from it; WebDriver
 * cannot set it, so the page reports 1 to match (the home-page checks keep Safari's own ratio).
 */
async function runFixture(path, exportName, settings, disposeMain) {
  if (disposeMain && window.__REAL_ENGINE__) { window.__REAL_ENGINE__.pause(true); window.__REAL_ENGINE__.dispose(); }
  Object.defineProperty(window, 'devicePixelRatio', { configurable: true, get: () => 1 });
  const module = await import(path);
  return settings === null ? module[exportName]() : module[exportName](settings);
}

/** First-frame readiness of a demo page harness, or its visible error. */
function demoReady(harness) {
  const error = document.getElementById('error');
  if (error && !error.hidden) return { error: error.textContent };
  const demo = window[harness];
  if (!demo) return null;
  const state = demo.snapshot();
  const frames = state.stats?.frames ?? demo.renderer?.stats.frames ?? 0;
  return frames > 0 ? { frames, errors: state.errors ?? demo.renderer?.errors ?? [], adapter: state.adapter?.adapterInfo ?? null } : null;
}

// ---------------------------------------------------------------------------------------------
// Checks. `spec` names the Playwright spec whose fixture or flow a check reuses; the checks keep a
// subset of that spec's assertions, listed in `covers`, and never loosen a threshold.

async function openHome(context, trees) {
  const { driver } = context;
  await driver.navigate(`${context.baseUrl}/?trees=${trees}`);
  await driver.waitFor('engine start', engineReady, 60_000);
  const error = await driver.execute(pageError);
  check(!error, 'page error after start: ' + error);
}

async function fixtureCheck(context, path, exportName, settings = null) {
  const { driver } = context;
  await openHome(context, 1);
  return driver.execute(runFixture, path, exportName, settings, true);
}

const checks = [
  {
    id: 'initialization', spec: 'tests/browser/engine.spec.ts',
    covers: 'WebGPU adapter and device creation through Renderer.create, adapter report, features, limits, no GPU errors',
    async run(context) {
      const { driver } = context;
      await openHome(context, 12);
      await driver.waitFor('5 frames', framesAtLeast, 60_000, 5);
      const first = await driver.execute(snapshot);
      context.environment = first.environment;
      check(first.environment, 'renderer environment missing');
      equal(first.environment.adapterInfo.isFallbackAdapter === true, false, 'adapter is not a fallback adapter');
      empty(first.errors, 'renderer errors');
      context.screenshots.push(await context.save('initialization', await driver.screenshot()));
      return { environment: first.environment, deviceRecovery: first.deviceRecovery, viewport: await driver.execute(viewport) };
    },
  },
  {
    id: 'home-render', spec: 'tests/browser/engine.spec.ts',
    covers: 'render graph and draw counts, GPU deformation readback against CPU, cached frame uploads, rigid-body controls, reset, raycast, visible scene pixels',
    async run(context) {
      const { driver } = context;
      await openHome(context, 12);
      await driver.waitFor('5 frames', framesAtLeast, 60_000, 5);
      const first = await driver.execute(snapshot);
      equal(first.stats.drawCalls, 15, 'draw calls (seven batches in two views plus resolve)');
      equal(first.physics.bodies, 6, 'rigid bodies');
      empty(first.errors, 'renderer errors');
      equal(first.renderGraph.passes, ['deformation', 'shadow', 'surface', 'resolve'], 'render graph passes');
      equal(first.renderGraph.deformationVariants, 5, 'deformation variants');
      equal(first.renderGraph.bounds.length, 23, 'bounds');
      equal(first.renderGraph.materialVariants, 4, 'material variants');
      equal(first.renderGraph.materials.images, 2, 'material images');
      const verified = await driver.execute(async () => {
        const engine = window.__REAL_ENGINE__;
        engine.pause(true);
        engine.step(1 / 60);
        return engine.verify();
      });
      above(verified.vertices, 100, 'verified vertices');
      equal(verified.passed, true, 'deformation verification');
      below(verified.maxNormalError, 0.0002, 'max normal error');
      equal(verified.boundsViolations, 0, 'bounds violations');
      await driver.waitFor('cached frame', () => window.__REAL_ENGINE__.snapshot().stats.deformationDispatches === 0, 30_000);
      const cached = await driver.execute(snapshot);
      equal(cached.stats.uploadedBytes, 256, 'uploaded bytes of a cached frame (camera/light uniform only)');
      equal(cached.sceneChanges.lastFrame, { create: 0, update: 0, remove: 0 }, 'scene changes of a cached frame');
      const controls = await driver.execute(async () => {
        const engine = window.__REAL_ENGINE__;
        const spawned = engine.spawn(), afterSpawn = engine.snapshot().physics.bodies;
        const removed = engine.remove(), afterRemove = engine.snapshot().physics.bodies;
        engine.reset();
        const reset = await engine.verify();
        const hit = engine.raycast([0, 10, -2.5], [0, -1, 0], 20);
        return { spawned, afterSpawn, removed, afterRemove, reset, hit };
      });
      equal(controls.afterSpawn, 7, 'bodies after spawn');
      equal(controls.afterRemove, 6, 'bodies after remove');
      equal(controls.reset.passed, true, 'verification after reset');
      equal(typeof controls.hit?.entityId, 'number', 'raycast hit entity');
      above(controls.hit.normal[1], 0.99, 'raycast hit normal y');
      await driver.execute(() => { window.__REAL_ENGINE__.pause(false); });
      await driver.waitFor('frames after resume', framesAtLeast, 30_000, cached.stats.frames + 5);
      const png = await driver.elementScreenshot('#viewport');
      context.screenshots.push(await context.save('home-viewport', png));
      const pixels = await driver.execute(countScenePixels, png.toString('base64'));
      above(pixels.blue, 50, 'visible blue character pixels');
      above(pixels.green, 50, 'visible green tree pixels');
      const final = await driver.execute(snapshot);
      empty(final.errors, 'renderer errors after controls');
      return { drawCalls: first.stats.drawCalls, verified: { vertices: verified.vertices, maxError: verified.maxError,
        maxNormalError: verified.maxNormalError }, pixels, ownedGpuBytes: final.stats.ownedGpuBytes, gpuMs: final.stats.gpuMs };
    },
  },
  {
    id: 'resize', spec: 'tests/browser/engine.spec.ts',
    covers: 'window resize reallocates size-dependent targets, deformation still verifies, the ledger returns to its first value',
    async run(context) {
      const { driver } = context;
      await openHome(context, 12);
      await driver.waitFor('5 frames', framesAtLeast, 60_000, 5);
      const large = await driver.waitFor('settled drawing buffer', settledDrawingBuffer, 30_000);
      const before = await driver.execute(snapshot);
      const smallRect = await driver.setWindowRect(SMALL_WINDOW);
      const small = await driver.waitFor('narrower drawing buffer', settledDrawingBuffer, 30_000, before.stats.width);
      const resized = await driver.execute(snapshot);
      const verification = await driver.execute(() => window.__REAL_ENGINE__.verify());
      equal(verification.passed, true, 'verification after resize');
      context.screenshots.push(await context.save('resize-small', await driver.screenshot()));
      const largeRect = await driver.setWindowRect(LARGE_WINDOW);
      await driver.waitFor('original drawing buffer', size => {
        const stats = window.__REAL_ENGINE__.snapshot().stats;
        return stats.width === size[0] && stats.height === size[1];
      }, 30_000, [before.stats.width, before.stats.height]);
      const after = await driver.execute(snapshot);
      empty(after.errors, 'renderer errors after resize');
      equal(after.stats.ownedGpuBytes, before.stats.ownedGpuBytes, 'owned GPU bytes after restoring the size');
      check(resized.stats.ownedGpuBytes < before.stats.ownedGpuBytes, 'smaller targets must own fewer bytes');
      return { windows: { small: smallRect, large: largeRect }, viewports: { large, small },
        sizes: [[before.stats.width, before.stats.height], [resized.stats.width, resized.stats.height], [after.stats.width, after.stats.height]],
        ownedGpuBytes: [before.stats.ownedGpuBytes, resized.stats.ownedGpuBytes, after.stats.ownedGpuBytes] };
    },
  },
  {
    id: 'dispose-and-device-loss', spec: 'tests/browser/device-loss.spec.ts',
    covers: 'dispose destroys the device and zeroes the ledger; an explicit GPUDevice.destroy() stops rendering, rejects queries and releases resources',
    async run(context) {
      const result = await fixtureCheck(context, '/tests/browser/fixtures/device-loss.ts', 'runDeviceLossFixture');
      equal(result.normal.beforeDispose, 1, 'frames before dispose');
      equal(result.normal.lossReason, 'destroyed', 'loss reason after dispose');
      empty(result.normal.errors, 'errors after dispose');
      equal(result.normal.ownedGpuBytesAfterDispose, 0, 'owned GPU bytes after dispose');
      equal(result.loss.framesBeforeLoss, 1, 'frames before loss');
      above(result.loss.bytesBeforeLoss, 0, 'owned GPU bytes before loss');
      equal(result.loss.lossReason, 'destroyed', 'loss reason');
      check(result.loss.errorsAfterLoss.some(error => error.includes('DEVICE_LOST')), 'DEVICE_LOST reported after loss');
      equal(result.loss.framesAfterAttempt, result.loss.framesBeforeLoss, 'no frame rendered after loss');
      contains(result.loss.queryRejection, 'DEVICE_LOST', 'query rejection after loss');
      equal(result.loss.ownedGpuBytesAfterDispose, 0, 'owned GPU bytes after disposing the lost renderer');
      return result;
    },
  },
  {
    id: 'device-recovery', spec: 'tests/browser/device-loss.spec.ts',
    covers: 'the home page rebuilds its Renderer on a new adapter and device after GPUDevice.destroy(), keeps the retained scene, then disposes without recovering',
    async run(context) {
      const { driver } = context;
      await openHome(context, 4);
      await driver.waitFor('5 frames', framesAtLeast, 60_000, 5);
      // Playwright's addInitScript has no WebDriver equivalent, so the live device is captured from
      // the next per-frame createCommandEncoder call; the prototype is restored immediately.
      const before = await driver.execute(async () => {
        const engine = window.__REAL_ENGINE__;
        if (!engine.spawn()) throw new Error('spawn failed');
        const prototype = GPUDevice.prototype, original = prototype.createCommandEncoder;
        const device = await new Promise((resolveDevice, reject) => {
          const timer = setTimeout(() => { prototype.createCommandEncoder = original; reject(new Error('no frame encoded within 5 s')); }, 5000);
          prototype.createCommandEncoder = function (...args) {
            prototype.createCommandEncoder = original;
            clearTimeout(timer);
            resolveDevice(this);
            return original.apply(this, args);
          };
        });
        window.__SAFARI_LOST_DEVICE__ = device;
        const state = engine.snapshot();
        device.destroy();
        return state;
      });
      equal(before.physics.bodies, 7, 'bodies before loss');
      equal(before.deviceRecovery.state, 'running', 'recovery state before loss');
      await driver.waitFor('recovered renderer', () => {
        const state = window.__REAL_ENGINE__.snapshot();
        return (state.deviceRecovery.recoveries === 1 && (state.stats?.frames ?? 0) >= 5) || !document.getElementById('error').hidden;
      }, 30_000);
      const error = await driver.execute(pageError);
      check(!error, 'page error after recovery: ' + error);
      const after = await driver.execute(async () => {
        const engine = window.__REAL_ENGINE__;
        const lost = await window.__SAFARI_LOST_DEVICE__.lost;
        delete window.__SAFARI_LOST_DEVICE__;
        return { snapshot: engine.snapshot(), verification: await engine.verify(), oldReason: lost.reason };
      });
      const recovery = after.snapshot.deviceRecovery;
      equal(recovery.state, 'running', 'recovery state');
      equal(recovery.losses, 1, 'losses');
      equal(recovery.events.map(event => event.kind), ['started', 'lost', 'attempt', 'recovered'], 'recovery events');
      equal(recovery.events[1].reason, 'destroyed', 'loss reason');
      equal(after.oldReason, 'destroyed', 'old device loss reason');
      equal(after.snapshot.physics.bodies, 7, 'retained bodies after recovery');
      empty(after.snapshot.errors, 'renderer errors after recovery');
      equal(after.snapshot.stats.ownedGpuBytes, before.stats.ownedGpuBytes, 'owned GPU bytes after recovery');
      equal(after.verification.passed, true, 'deformation verification after recovery');
      equal(after.verification.boundsViolations, 0, 'bounds violations after recovery');
      context.screenshots.push(await context.save('device-recovery', await driver.screenshot()));
      const disposed = await driver.execute(async () => {
        window.__REAL_ENGINE__.dispose();
        await new Promise(resolveWait => setTimeout(resolveWait, 300));
        const state = window.__REAL_ENGINE__.snapshot();
        return { state: state.deviceRecovery.state, stats: state.stats };
      });
      equal(disposed, { state: 'disposed', stats: null }, 'state after intentional dispose');
      return { events: recovery.events.map(({ kind, reason }) => ({ kind, reason })), ownedGpuBytes: after.snapshot.stats.ownedGpuBytes,
        adapter: after.snapshot.environment?.adapterInfo };
    },
  },
  {
    id: 'fixture-deformation', spec: 'tests/browser/deformation.spec.ts',
    covers: 'six-bone skin, ordered multi-Morph/wave graphs and per-entity previous positions against CPU references; ledger zero after dispose',
    async run(context) {
      const result = await fixtureCheck(context, '/tests/browser/deformation-fixture.ts', 'runDeformationFixture');
      for (const [name, verification] of Object.entries({ first: result.first, second: result.second, reordered: result.reordered })) {
        equal(verification.vertices, result.expectedVerticesPerVerification, `${name} verified vertices`);
        below(verification.maxError, 0.0002, `${name} max position error`);
        below(verification.maxNormalError, 0.0002, `${name} max normal error`);
        equal(verification.boundsViolations, 0, `${name} bounds violations`);
      }
      for (const distance of [...result.historyMovement, ...result.reorderedHistoryMovement]) above(distance, 0.1, 'history movement');
      below(result.historyMaxError, 0.0002, 'previous-position error');
      below(result.currentMaxError, 0.0002, 'current-position error');
      above(result.orderDifference, 0.05, 'operator order difference');
      equal(result.diagnostics.deformationVariants, 2, 'deformation variants');
      equal(result.diagnostics.bounds.length, 4, 'bounds');
      equal(result.frames, 3, 'frames');
      empty(result.errors, 'renderer errors');
      equal(result.ownedGpuBytesAfterDispose, 0, 'owned GPU bytes after dispose');
      return { maxError: Math.max(result.first.maxError, result.second.maxError, result.reordered.maxError), historyMaxError: result.historyMaxError };
    },
  },
  {
    id: 'fixture-hdr', spec: 'tests/browser/hdr.spec.ts',
    covers: 'linear HDR radiance through exposure changes and resize, ACES/sRGB resolve, velocity after exposure, size-dependent ledger',
    async run(context) {
      const result = await fixtureCheck(context, '/tests/browser/fixtures/hdr.ts', 'runHdrFixture');
      const radiance = [5, 0.75, 0.125, 1];
      const aces = value => Math.max(0, Math.min(1, value * (2.51 * value + 0.03) / (value * (2.43 * value + 0.59) + 0.14)));
      const srgb = value => value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
      const resolved = ev => [...radiance.slice(0, 3).map(value => srgb(aces(value * 2 ** ev))), 1];
      const near = (actual, expected, tolerance, label) => {
        equal(actual.length, expected.length, label + ' length');
        expected.forEach((value, i) => below(Math.abs(actual[i] - value), tolerance, `${label} channel ${i}`));
      };
      for (const [name, values] of Object.entries({ first: result.firstHdr, second: result.secondHdr, resized: result.resizedHdr })) {
        equal(values.length, 1, name + ' HDR samples');
        near(values[0], radiance, 0.004, name + ' HDR');
      }
      near(result.firstOutput, resolved(0), 0.015, 'output at EV 0');
      near(result.secondOutput, resolved(-2), 0.015, 'output at EV -2');
      near(result.resizedOutput, resolved(-2), 0.015, 'resized output at EV -2');
      above(result.velocityAfterExposure[0][0], 0.01, 'velocity x after exposure');
      below(Math.abs(result.velocityAfterExposure[0][1]), 0.001, 'velocity y after exposure');
      equal(result.secondMaterialVariants, result.materialVariants, 'exposure does not add material variants');
      above(result.resizedSize.width, result.firstSize.width, 'resized width');
      equal(result.resizedBytes - result.firstBytes,
        (result.resizedSize.width * result.resizedSize.height - result.firstSize.width * result.firstSize.height) * 16, 'size-dependent bytes');
      equal(result.frames, 3, 'frames');
      empty(result.errors, 'renderer errors');
      equal(result.ownedGpuBytesAfterDispose, 0, 'owned GPU bytes after dispose');
      return { firstHdr: result.firstHdr[0], firstOutput: result.firstOutput, sizes: [result.firstSize, result.resizedSize] };
    },
  },
  {
    id: 'fixture-gltf', spec: 'tests/browser/gltf.spec.ts',
    covers: 'imported glTF skin and Morph clips keep independent GPU history; GPU readback against CPU per vertex',
    async run(context) {
      const result = await fixtureCheck(context, '/tests/browser/fixtures/gltf.ts', 'runGltfFixture');
      for (const [name, verification] of Object.entries({ first: result.first, second: result.second })) {
        equal(verification.vertices, result.expectedVertices, `${name} verified vertices`);
        below(verification.maxError, 0.0002, `${name} max position error`);
        below(verification.maxNormalError, 0.0002, `${name} max normal error`);
        equal(verification.boundsViolations, 0, `${name} bounds violations`);
      }
      const sample = (verification, batch, entityId, vertexIndex) => {
        const matches = verification.samples.filter(value => value.batch === batch && value.entityId === entityId && value.vertexIndex === vertexIndex);
        equal(matches.length, 1, `one sample for ${batch} ${entityId} ${vertexIndex}`);
        return matches[0];
      };
      const close = (actual, expected, tolerance, label) => expected.forEach((value, axis) => below(Math.abs(actual[axis] - value), tolerance, `${label} axis ${axis}`));
      const leftFirst = sample(result.first, result.names.firstSkin, 701, result.topVertex);
      const leftSecond = sample(result.second, result.names.firstSkin, 701, result.topVertex);
      close(leftFirst.current, [-3.5, 2, 0], 0.002, 'first skin current');
      close(leftSecond.current, [-5, 1.5, 0], 0.002, 'first skin second current');
      close(leftSecond.previous, leftFirst.current, 0.0002, 'first skin second previous');
      above(result.greatestMotion, 0.1, 'greatest motion');
      equal(result.frames, 2, 'frames');
      empty(result.errors, 'renderer errors');
      equal(result.ownedGpuBytesAfterDispose, 0, 'owned GPU bytes after dispose');
      return { maxError: Math.max(result.first.maxError, result.second.maxError) };
    },
  },
  {
    id: 'fixture-velocity', spec: 'tests/browser/velocity.spec.ts',
    covers: 'velocity attachment for camera, object, skin and Morph motion, cuts, masked holes and resize',
    async run(context) {
      const result = await fixtureCheck(context, '/tests/browser/fixtures/velocity.ts', 'runVelocityFixture');
      contains(result.noFrameRejection, 'NO_RENDERED_VELOCITY', 'no-frame rejection');
      contains(result.invalidProbeRejection, 'INVALID_VELOCITY_PROBES', 'invalid-probe rejection');
      const near = (actual, expected, label) => [0, 1].forEach(axis => below(Math.abs(actual[axis] - expected[axis]), 0.0001, `${label} axis ${axis}`));
      for (const [name, capture] of Object.entries({ first: result.first, camera: result.camera, moved: result.moved, settled: result.settled,
        entityReset: result.entityReset, cut: result.cut, resized: result.resized })) {
        equal(capture.values.length, 6, name + ' probes');
        for (let i = 0; i < 4; i++) near(capture.values[i], capture.expected[i], `${name} probe ${i}`);
        near(capture.values[4], [0, 0], name + ' masked hole');
        near(capture.values[5], [0, 0], name + ' background');
      }
      below(result.camera.values[0][0], -0.005, 'camera motion shifts the static world left');
      above(result.moved.values[1][1], 0.005, 'object moved down');
      below(result.moved.values[2][1], -0.003, 'skin moved up');
      above(result.moved.values[3][1], 0.003, 'Morph moved down');
      equal(result.frames, 7, 'frames');
      empty(result.errors, 'renderer errors');
      equal(result.ownedGpuBytesAfterDispose, 0, 'owned GPU bytes after dispose');
      return { camera: result.camera.values, moved: result.moved.values };
    },
  },
  {
    id: 'fixture-world-query-watertight', spec: 'tests/browser/world-query-watertight.spec.ts',
    covers: 'rays exactly on shared edges and vertices hit on both static traversal paths, no overflows, ledger zero',
    async run(context) {
      const result = await fixtureCheck(context, '/tests/browser/fixtures/world-query-watertight.ts', 'runWorldQueryWatertightFixture');
      above(result.rays, 5000, 'rays');
      above(result.kinds['exact-vertex'], 400, 'exact-vertex rays');
      above(result.kinds['exact-edge'], 2000, 'exact-edge rays');
      for (const arm of result.arms) {
        equal(arm.misses, {}, arm.arm + ' misses');
        equal(arm.hits, result.rays, arm.arm + ' hits');
        equal(arm.wrongBatch, 0, arm.arm + ' wrong batch');
        equal(arm.overflows, 0, arm.arm + ' overflows');
      }
      equal(result.mismatches, 0, 'arm mismatches');
      equal(result.sharedStatic.pooledBatches, 8, 'pooled batches');
      empty(result.errors, 'errors');
      equal(result.cleanup.ownedBytesAfterDispose, 0, 'owned GPU bytes after dispose');
      return { rays: result.rays, kinds: result.kinds };
    },
  },
  {
    id: 'fixture-world-query-instances', spec: 'tests/browser/world-query-instances.spec.ts',
    covers: 'fixture page: batch instance BVH matches linear queries and GI through move, reorder, delete and restore, with less GPU work',
    async run(context) {
      const { driver } = context;
      await driver.navigate(`${context.baseUrl}/tests/browser/fixtures/world-query-instances.html`);
      const result = await driver.execute(runFixture, '/tests/browser/fixtures/world-query-instances.ts', 'runWorldQueryInstancesFixture', null, false);
      for (const [name, stage] of Object.entries({ initial: result.initial, moved: result.moved, reordered: result.reordered,
        empty: result.empty, restored: result.restored })) {
        equal(stage.accelerated.hits, stage.linear.hits, name + ' accelerated hits equal linear hits');
        equal(stage.acceleratedDiagnostics.instanceAcceleration, 'batch-bvh', name + ' acceleration');
        equal(stage.acceleratedDiagnostics.lastReadbackQuery?.overflows, 0, name + ' accelerated overflows');
        equal(stage.linearDiagnostics.lastReadbackQuery?.overflows, 0, name + ' linear overflows');
      }
      equal(result.pictureEqual, true, 'accelerated and linear pictures are equal');
      for (const count of [...result.initialVisiblePixels, ...result.finalVisiblePixels]) above(count, 50, 'visible pixels');
      equal(result.initial.accelerated.hits.slice(0, 7).map(value => value?.entityId), [1000, 1001, 1017, 1063, 1127, 1200, 1255], 'initial hit entities');
      equal(result.empty.accelerated.hits, [null, null, null], 'hits after deleting every instance');
      equal(result.acceleratedGi.invalidProbes, 0, 'accelerated GI invalid probes');
      equal(result.linearGi.invalidProbes, 0, 'linear GI invalid probes');
      context.screenshots.push(await context.save('fixture-world-query-instances', await driver.screenshot()));
      return { pictureEqual: result.pictureEqual, visiblePixels: [...result.initialVisiblePixels, ...result.finalVisiblePixels] };
    },
  },
  ...[['demo-gi', '/gi.html', '__GI_DEMO__'], ['demo-fog', '/fog.html', '__FOG_DEMO__'], ['demo-reflection', '/reflection.html', '__REFLECTION_DEMO__'],
    ['demo-geometry', '/geometry.html', '__GEOMETRY_DEMO__']].map(([id, path, harness]) => ({
    id, spec: 'demo page ' + path,
    covers: `${path} starts, renders frames without renderer errors and is captured; no numeric assertions`,
    async run(context) {
      const { driver } = context;
      await driver.navigate(context.baseUrl + path);
      const ready = await driver.waitFor(path + ' first frames', demoReady, 120_000, harness);
      check(!ready.error, path + ' page error: ' + ready.error);
      // Let a running page settle for a few frames before the capture; paused pages stay as they are.
      await driver.waitFor(path + ' more frames', (name, frames) => {
        const demo = window[name], state = demo.snapshot();
        return (state.stats?.frames ?? demo.renderer?.stats.frames ?? 0) >= frames || !document.getElementById('error').hidden;
      }, 5_000, harness, ready.frames + 30).catch(() => null);
      const settled = await driver.execute(demoReady, harness);
      check(settled && !settled.error, path + ' page error after frames: ' + settled?.error);
      empty(settled.errors, path + ' renderer errors');
      context.screenshots.push(await context.save(id, await driver.screenshot()));
      await driver.execute(name => { window[name].dispose?.(); }, harness);
      return settled;
    },
  })),
];

// ---------------------------------------------------------------------------------------------
// Run

async function startDriver(port, logPath) {
  const probe = new WebDriver(`http://127.0.0.1:${port}`);
  // Never attach to a driver someone else started: the port must be free.
  const occupied = await probe.status().then(() => true, () => false);
  if (occupied) throw new Error(`SAFARIDRIVER_PORT_IN_USE: something already answers on ${port}`);
  const log = [];
  const child = spawn(SAFARIDRIVER, ['-p', String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.on('data', chunk => log.push(chunk));
  child.stderr.on('data', chunk => log.push(chunk));
  let exited = false;
  const exit = new Promise(resolveExit => child.once('exit', (code, signal) => { exited = true; resolveExit({ code, signal }); }));
  child.once('error', error => log.push(Buffer.from(String(error))));
  const deadline = Date.now() + 10_000;
  for (;;) {
    if (exited) throw new Error('SAFARIDRIVER_EXITED: ' + Buffer.concat(log).toString().trim());
    const ready = await probe.status().then(value => value?.ready !== false, () => false);
    if (ready) break;
    if (Date.now() > deadline) { child.kill('SIGTERM'); throw new Error('SAFARIDRIVER_TIMEOUT: not ready within 10 s'); }
    await new Promise(resolveWait => setTimeout(resolveWait, 100));
  }
  return {
    driver: probe,
    async stop() {
      if (!exited) {
        child.kill('SIGTERM');
        const stopped = await Promise.race([exit, new Promise(resolveWait => setTimeout(() => resolveWait(null), 5000))]);
        if (!stopped) { child.kill('SIGKILL'); await exit; }
      }
      writeFileSync(logPath, Buffer.concat(log));
    },
  };
}

async function main(argv) {
  const options = parseArguments(argv);
  const selected = options.only ? options.only.map(id => {
    const found = checks.find(candidate => candidate.id === id);
    if (!found) throw new Error('SAFARI_RUNNER_ARGUMENTS: unknown check ' + id + '; known: ' + checks.map(item => item.id).join(', '));
    return found;
  }) : checks;
  if (options.list) {
    for (const item of checks) console.log(`${item.id}\t${item.spec}\t${item.covers}`);
    return 0;
  }
  mkdirSync(options.out, { recursive: true });
  const report = { runner: 'scripts/safari-webdriver.mjs', startedAt: new Date().toISOString(), baseUrl: undefined,
    browser: null, window: null, environment: null, checks: [], counts: { passed: 0, failed: 0, skipped: 0 }, cleanup: [] };
  const cleanup = [];
  let interrupted = false;
  const runCleanup = async () => {
    while (cleanup.length) {
      const [label, step] = cleanup.pop();
      try { await step(); report.cleanup.push({ step: label, ok: true }); } catch (error) { report.cleanup.push({ step: label, ok: false, error: String(error) }); }
    }
  };
  const onSignal = signal => {
    if (interrupted) return;
    interrupted = true;
    console.error(`safari-webdriver: ${signal}, cleaning up`);
    void runCleanup().finally(() => process.exit(130));
  };
  process.on('SIGINT', onSignal); process.on('SIGTERM', onSignal);
  try {
    const server = await createServer({ root: ROOT, logLevel: 'warn',
      server: { port: options.port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null } });
    await server.listen();
    cleanup.push(['close dev server', () => server.close()]);
    report.baseUrl = `http://127.0.0.1:${options.port}`;
    const safari = await startDriver(options.driverPort, join(options.out, 'safaridriver.log'));
    cleanup.push(['stop safaridriver', () => safari.stop()]);
    const { driver } = safari;
    const session = await driver.newSession();
    cleanup.push(['delete session', () => driver.deleteSession()]);
    report.browser = { name: session.capabilities.browserName, version: session.capabilities.browserVersion,
      platform: session.capabilities.platformName };
    report.window = { requested: LARGE_WINDOW, rect: await driver.setWindowRect(LARGE_WINDOW) };
    console.log(`safari-webdriver: ${report.browser.name} ${report.browser.version} on ${report.browser.platform}, ${selected.length} checks`);
    const context = {
      driver, baseUrl: report.baseUrl, environment: null, screenshots: [],
      async save(name, png) { const path = join(options.out, name + '.png'); writeFileSync(path, png); return path; },
    };
    for (const item of selected) {
      if (interrupted) break;
      context.screenshots = [];
      const started = performance.now();
      const entry = { id: item.id, spec: item.spec, covers: item.covers, status: 'passed', durationMs: 0, screenshots: [], result: null, error: null };
      try {
        entry.result = await item.run(context);
      } catch (error) {
        entry.status = 'failed';
        entry.error = error && error.stack ? String(error.stack) : String(error);
        try {
          entry.pageError = await driver.execute(pageError).catch(() => null);
          context.screenshots.push(await context.save(item.id + '-failure', await driver.screenshot()));
        } catch { /* the failure itself may have closed the page */ }
      }
      entry.durationMs = Math.round(performance.now() - started);
      entry.screenshots = context.screenshots.map(path => path.slice(ROOT.length + 1));
      report.checks.push(entry);
      report.counts[entry.status]++;
      console.log(`safari-webdriver: ${entry.status.toUpperCase()} ${item.id} (${entry.durationMs} ms)${entry.error ? '\n  ' + entry.error.split('\n')[0] : ''}`);
    }
    report.environment = context.environment;
  } catch (error) {
    report.fatal = error && error.stack ? String(error.stack) : String(error);
    console.error('safari-webdriver: ' + report.fatal);
  } finally {
    await runCleanup();
    report.finishedAt = new Date().toISOString();
    writeFileSync(join(options.out, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    process.off('SIGINT', onSignal); process.off('SIGTERM', onSignal);
  }
  const { passed, failed, skipped } = report.counts;
  console.log(`safari-webdriver: ${passed} passed, ${failed} failed, ${skipped} skipped; report ${join(options.out, 'results.json')}`);
  return report.fatal || failed || report.cleanup.some(step => !step.ok) ? 1 : 0;
}

process.exitCode = await main(process.argv.slice(2));
