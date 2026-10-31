// Shared browser target and local asset-set resolution for the Sponza capture
// scripts. Nothing here downloads: the base URL must be a loopback dev server
// and asset roots are server paths of already-built local packages.
import { chromium } from '@playwright/test';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const LOOPBACK = new Set(['127.0.0.1', 'localhost', '[::1]']);
const CHANNELS = ['chrome', 'msedge', 'chromium'];
export const ASSET_SET_KEYS = ['geometryRoot', 'textureGpuRoot', 'textureReferenceRoot', 'uvAuditUrl'];

export function defaultChannel(platform = process.platform) {
  return platform === 'darwin' ? 'chrome' : platform === 'win32' ? 'msedge' : 'chromium';
}

/** --base-url, else SPONZA_BASE_URL, else the stream's PLAYWRIGHT_PORT (5173 when unset). */
export function captureBaseUrl(flag, env = process.env) {
  const text = flag ?? env.SPONZA_BASE_URL ?? `http://127.0.0.1:${env.PLAYWRIGHT_PORT ?? 5173}`;
  let url;
  try { url = new URL(text); } catch { throw Error('CAPTURE_BASE_URL'); }
  if (!['http:', 'https:'].includes(url.protocol) || !LOOPBACK.has(url.hostname) || url.username || url.password
    || (url.pathname !== '/' && url.pathname !== '') || url.search || url.hash) throw Error('CAPTURE_BASE_URL_NOT_LOCAL');
  return url.origin;
}

/** --channel, else SPONZA_CHANNEL, else chrome on macOS, msedge on Windows, chromium elsewhere. */
export function captureChannel(flag, env = process.env, platform = process.platform) {
  const channel = flag ?? env.SPONZA_CHANNEL ?? defaultChannel(platform);
  if (!CHANNELS.includes(channel)) throw Error('CAPTURE_CHANNEL');
  return channel;
}

/**
 * Bounds one long page evaluation: rejects with `${label}_DEADLINE` after `ms`
 * and with `${label}_BROWSER_DISCONNECTED` as soon as the browser goes away
 * (killed, crashed or out of memory), so a capture never waits forever on a
 * browser that no longer exists. `browser` only needs once/off('disconnected').
 */
export function boundedCapture(browser, work, ms, label) {
  if (!Number.isSafeInteger(ms) || ms < 1) throw Error('CAPTURE_DEADLINE');
  let timer, onDisconnected;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(() => reject(Error(`${label}_DEADLINE: ${ms} ms`)), ms);
    onDisconnected = () => reject(Error(`${label}_BROWSER_DISCONNECTED`));
    browser.once('disconnected', onDisconnected);
  });
  return Promise.race([work, guard]).finally(() => { clearTimeout(timer); browser.off('disconnected', onDisconnected); });
}

/** Hardware adapter launch, matching the previous Edge captures except for the channel. */
export function launchCaptureBrowser(channel) {
  return chromium.launch({ channel, headless: true, args: ['--force-high-performance-gpu'] });
}

function localRoot(value, key, directory) {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || value.includes('..')
    || (directory && !value.endsWith('/'))) throw Error('CAPTURE_ASSET_SET_ROOT: ' + key);
  return value;
}

/**
 * Reads an asset-set description (--asset-set, else SPONZA_ASSET_SET) and
 * returns the server roots it declares. Explicit per-root flags win.
 */
export async function assetSetRoots(flag, env = process.env) {
  const path = flag ?? env.SPONZA_ASSET_SET;
  if (!path) return {};
  const file = resolve(path);
  if ((await stat(file)).size > 1024 * 1024) throw Error('CAPTURE_ASSET_SET_BYTES');
  const set = JSON.parse(await readFile(file, 'utf8'));
  if (set.version !== 1 || set.format !== 'real-engine-asset-set' || typeof set.roots !== 'object') throw Error('CAPTURE_ASSET_SET_FORMAT');
  const roots = {};
  for (const key of ASSET_SET_KEYS) if (set.roots[key] !== undefined) roots[key] = localRoot(set.roots[key], key, key !== 'uvAuditUrl');
  return roots;
}

/** Fails before launching a browser when no source supplied a required root. */
export function requireAssetRoots(options, label) {
  const missing = ASSET_SET_KEYS.filter(key => options[key] === undefined);
  if (missing.length) throw Error(`${label}_ASSET_ROOTS_REQUIRED: ${missing.join(',')} (pass --asset-set file.json or SPONZA_ASSET_SET, or the per-root flags)`);
}
