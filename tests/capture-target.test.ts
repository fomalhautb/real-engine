import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { EventEmitter } from 'node:events';
import { describe, expect, it } from 'vitest';

type Env = Record<string, string | undefined>;
const target = await import(pathToFileURL(resolve('scripts/capture-target.mjs')).href) as {
  captureBaseUrl: (flag: string | undefined, env?: Env) => string;
  captureChannel: (flag: string | undefined, env?: Env, platform?: string) => string;
  assetSetRoots: (flag: string | undefined, env?: Env) => Promise<Record<string, string>>;
  requireAssetRoots: (options: Record<string, unknown>, label: string) => void;
  boundedCapture: <T>(browser: EventEmitter, work: Promise<T>, ms: number, label: string) => Promise<T>;
};

describe('capture target', () => {
  it('derives a loopback base URL from the flag, SPONZA_BASE_URL or PLAYWRIGHT_PORT', () => {
    expect(target.captureBaseUrl(undefined, {})).toBe('http://127.0.0.1:5173');
    expect(target.captureBaseUrl(undefined, { PLAYWRIGHT_PORT: '5208' })).toBe('http://127.0.0.1:5208');
    expect(target.captureBaseUrl(undefined, { SPONZA_BASE_URL: 'http://localhost:6000/', PLAYWRIGHT_PORT: '5208' })).toBe('http://localhost:6000');
    expect(target.captureBaseUrl('http://[::1]:7000', {})).toBe('http://[::1]:7000');
    for (const bad of ['http://example.com:5173', 'file:///tmp', 'http://127.0.0.1:5173/sub', 'http://user@127.0.0.1:1', 'not a url']) {
      expect(() => target.captureBaseUrl(bad, {}), bad).toThrow(/CAPTURE_BASE_URL/);
    }
  });

  it('defaults the channel per platform and rejects unknown channels', () => {
    expect(target.captureChannel(undefined, {}, 'darwin')).toBe('chrome');
    expect(target.captureChannel(undefined, {}, 'win32')).toBe('msedge');
    expect(target.captureChannel(undefined, {}, 'linux')).toBe('chromium');
    expect(target.captureChannel(undefined, { SPONZA_CHANNEL: 'msedge' }, 'darwin')).toBe('msedge');
    expect(() => target.captureChannel('firefox', {}, 'darwin')).toThrow('CAPTURE_CHANNEL');
  });

  it('reads server roots from an asset set and reports missing roots before launch', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'real-engine-asset-set-'));
    try {
      const file = join(directory, 'asset-set.json');
      await writeFile(file, JSON.stringify({ version: 1, format: 'real-engine-asset-set', roots: {
        geometryRoot: '/artifacts/g/', textureGpuRoot: '/artifacts/t/', textureReferenceRoot: '/artifacts/t/', uvAuditUrl: '/artifacts/uv.json' } }));
      expect(await target.assetSetRoots(undefined, { SPONZA_ASSET_SET: file })).toEqual({
        geometryRoot: '/artifacts/g/', textureGpuRoot: '/artifacts/t/', textureReferenceRoot: '/artifacts/t/', uvAuditUrl: '/artifacts/uv.json' });
      expect(await target.assetSetRoots(undefined, {})).toEqual({});
      await writeFile(file, JSON.stringify({ version: 1, format: 'real-engine-asset-set', roots: { geometryRoot: 'https://cdn.example/g/' } }));
      await expect(target.assetSetRoots(file, {})).rejects.toThrow('CAPTURE_ASSET_SET_ROOT: geometryRoot');
      expect(() => target.requireAssetRoots({ geometryRoot: '/g/' }, 'SPONZA_GI')).toThrow(/SPONZA_GI_ASSET_ROOTS_REQUIRED: textureGpuRoot,textureReferenceRoot,uvAuditUrl/);
    } finally { await rm(directory, { recursive: true, force: true }); }
  });

  it('bounds a capture by a deadline and by browser disconnection', async () => {
    const browser = new EventEmitter();
    await expect(target.boundedCapture(browser, Promise.resolve(7), 1000, 'X')).resolves.toBe(7);
    expect(browser.listenerCount('disconnected')).toBe(0);
    await expect(target.boundedCapture(browser, new Promise(() => {}), 5, 'X')).rejects.toThrow('X_DEADLINE: 5 ms');
    const pending = target.boundedCapture(browser, new Promise(() => {}), 60000, 'X');
    browser.emit('disconnected');
    await expect(pending).rejects.toThrow('X_BROWSER_DISCONNECTED');
    expect(browser.listenerCount('disconnected')).toBe(0);
    expect(() => target.boundedCapture(browser, Promise.resolve(1), 0, 'X')).toThrow('CAPTURE_DEADLINE');
  });
});
