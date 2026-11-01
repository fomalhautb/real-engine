import { describe, expect, it } from 'vitest';
import { resolveBrowserEngine } from './browser/browser-engine';

describe('Playwright browser engine selection', () => {
  it('defaults to Chromium and leaves every Chromium variable to playwright.config.ts', () => {
    expect(resolveBrowserEngine({})).toBe('chromium');
    expect(resolveBrowserEngine({ PLAYWRIGHT_BROWSER: 'chromium', PLAYWRIGHT_CHANNEL: 'chrome', PLAYWRIGHT_GPU: 'hardware',
      PLAYWRIGHT_HIGH_PERFORMANCE_GPU: '1' })).toBe('chromium');
  });

  it('selects WebKit only on the hardware adapter and without Chromium channel or flags', () => {
    expect(resolveBrowserEngine({ PLAYWRIGHT_BROWSER: 'webkit', PLAYWRIGHT_GPU: 'hardware' })).toBe('webkit');
    expect(() => resolveBrowserEngine({ PLAYWRIGHT_BROWSER: 'webkit' })).toThrow('needs PLAYWRIGHT_GPU=hardware');
    expect(() => resolveBrowserEngine({ PLAYWRIGHT_BROWSER: 'webkit', PLAYWRIGHT_GPU: 'hardware', PLAYWRIGHT_CHANNEL: 'chrome' }))
      .toThrow('PLAYWRIGHT_CHANNEL selects a Chromium channel');
    expect(() => resolveBrowserEngine({ PLAYWRIGHT_BROWSER: 'webkit', PLAYWRIGHT_GPU: 'hardware', PLAYWRIGHT_HIGH_PERFORMANCE_GPU: '1' }))
      .toThrow('PLAYWRIGHT_HIGH_PERFORMANCE_GPU is a Chromium flag');
  });

  it('rejects engines the suite cannot launch', () => {
    for (const value of ['firefox', 'safari', 'WebKit', '']) {
      expect(() => resolveBrowserEngine({ PLAYWRIGHT_BROWSER: value, PLAYWRIGHT_GPU: 'hardware' })).toThrow('PLAYWRIGHT_BROWSER must be one of chromium, webkit');
    }
  });
});
