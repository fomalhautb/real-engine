import { expect } from '@playwright/test';
import { browserEngine } from './browser-engine';

export interface ReportedAdapter { vendor: string; architecture: string; isFallbackAdapter: boolean | null }

/**
 * Adapter checks for the run's environment report. Chromium keeps its checks unchanged: a hardware
 * run must not use a fallback adapter, and PLAYWRIGHT_EXPECT_GPU_VENDOR names the vendor. WebKit
 * has the same checks plus its own report: on Apple GPUs it says `apple / apple` where Chrome says
 * `apple / metal-3`, so a WebKit result is labelled by what WebKit reports and is never Chrome
 * target evidence. The user agent must match the engine that playwright.config.ts launched.
 */
export function expectRunAdapter(adapter: ReportedAdapter, userAgent: string): void {
  if (process.env.PLAYWRIGHT_GPU === 'hardware') expect(adapter.isFallbackAdapter).toBe(false);
  const vendor = process.env.PLAYWRIGHT_EXPECT_GPU_VENDOR;
  if (vendor) expect(adapter.vendor.toLowerCase()).toBe(vendor);
  if (browserEngine === 'webkit') {
    expect(userAgent).toMatch(/AppleWebKit\/[\d.]+ \(KHTML, like Gecko\) Version\/[\d.]+ Safari\//);
    expect(userAgent).not.toContain('Chrome/');
    if (vendor === 'apple') expect(`${adapter.vendor} / ${adapter.architecture}`).toBe('apple / apple');
  }
}
