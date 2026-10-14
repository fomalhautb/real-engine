import { defineConfig } from '@playwright/test';

const hardware = process.env.PLAYWRIGHT_GPU === 'hardware';
const highPerformance = process.env.PLAYWRIGHT_HIGH_PERFORMANCE_GPU === '1';
if (highPerformance && !hardware) throw new Error('High-performance GPU selection requires PLAYWRIGHT_GPU=hardware');

export default defineConfig({
  testDir: './tests/browser',
  timeout: 60_000,
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:5173',
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
    command: 'npm run dev -- --port 5173 --strictPort',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
