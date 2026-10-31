import { defineConfig } from '@playwright/test';

const hardware = process.env.PLAYWRIGHT_GPU === 'hardware';
const highPerformance = process.env.PLAYWRIGHT_HIGH_PERFORMANCE_GPU === '1';
if (highPerformance && !hardware) throw new Error('High-performance GPU selection requires PLAYWRIGHT_GPU=hardware');
// Parallel checkouts must not reuse each other's dev server, which would test
// another worktree's sources under this run's name.
const port = Number(process.env.PLAYWRIGHT_PORT ?? 5173);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('PLAYWRIGHT_PORT must be an integer in 1024..65535');

export default defineConfig({
  testDir: './tests/browser',
  timeout: 60_000,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 1000 },
    headless: true,
    channel: process.env.PLAYWRIGHT_CHANNEL ?? 'chromium',
    launchOptions: {
      args: hardware ? (highPerformance ? ['--force-high-performance-gpu'] : []) : [
        '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader',
        // Linux CI also runs under Xvfb: Dawn and the headless compositor need
        // compatible Vulkan presentation, not just a working compute adapter.
        ...(process.platform === 'linux' ? [
          '--enable-gpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan',
          '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--use-vulkan=swiftshader',
        ] : []),
      ],
    },
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `npm run dev -- --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
