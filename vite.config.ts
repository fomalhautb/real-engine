import { defineConfig } from 'vite';

export default defineConfig({
  build: { rolldownOptions: { input: { main: 'index.html', benchmark: 'benchmark.html', geometry: 'geometry.html' } } },
});
