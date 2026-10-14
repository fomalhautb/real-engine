import { defineConfig } from 'vite';

export default defineConfig({
  // Geometry demos load meshoptimizer lazily. Prebundle it before page startup
  // so the first dynamic import cannot trigger a dependency-driven full reload.
  optimizeDeps: { include: ['meshoptimizer', '@dimforge/rapier3d-compat'] },
  build: { rolldownOptions: { input: { main: 'index.html', benchmark: 'benchmark.html', geometry: 'geometry.html', gi: 'gi.html' } } },
});
