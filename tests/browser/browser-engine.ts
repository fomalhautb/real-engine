/** Browser engines playwright.config.ts can launch. Chromium is the default and the target. */
export const BROWSER_ENGINES = ['chromium', 'webkit'] as const;
export type BrowserEngine = typeof BROWSER_ENGINES[number];

type Environment = Readonly<Record<string, string | undefined>>;

/**
 * Validates PLAYWRIGHT_BROWSER and the variables that only make sense for one engine.
 * WebKit always runs on the system Metal adapter: it has no software adapter to select,
 * so it requires PLAYWRIGHT_GPU=hardware and rejects Chromium channel and flag selections.
 */
export function resolveBrowserEngine(env: Environment): BrowserEngine {
  const value = env.PLAYWRIGHT_BROWSER ?? 'chromium';
  if (!(BROWSER_ENGINES as readonly string[]).includes(value)) {
    throw new Error(`PLAYWRIGHT_BROWSER must be one of ${BROWSER_ENGINES.join(', ')}, not ${value}`);
  }
  const engine = value as BrowserEngine;
  if (engine === 'webkit') {
    if (env.PLAYWRIGHT_GPU !== 'hardware') {
      throw new Error('PLAYWRIGHT_BROWSER=webkit needs PLAYWRIGHT_GPU=hardware: WebKit has no software WebGPU adapter');
    }
    if (env.PLAYWRIGHT_CHANNEL) throw new Error('PLAYWRIGHT_CHANNEL selects a Chromium channel; unset it for WebKit');
    if (env.PLAYWRIGHT_HIGH_PERFORMANCE_GPU) throw new Error('PLAYWRIGHT_HIGH_PERFORMANCE_GPU is a Chromium flag; unset it for WebKit');
  }
  return engine;
}

/** Engine of the current run; playwright.config.ts has already validated the environment. */
export const browserEngine: BrowserEngine = resolveBrowserEngine(process.env);
