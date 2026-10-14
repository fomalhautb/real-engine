import { FoundationBenchmark, type BenchmarkOptions } from './benchmark';

let active: FoundationBenchmark | undefined;
const api = {
  async run(options: BenchmarkOptions) {
    active?.dispose();
    active = new FoundationBenchmark(document.getElementById('benchmark') as HTMLCanvasElement, options);
    return active.run();
  },
  progress() { return active ? { ...active.progress } : null; },
  csv() { return active?.recorder.toCsv() ?? ''; },
  dispose() { active?.dispose(); return { ownedGpuBytes: active?.ownedGpuBytes ?? 0 }; },
};
declare global { interface Window { __REAL_BENCHMARK__: typeof api } }
window.__REAL_BENCHMARK__ = api;
window.addEventListener('pagehide', () => api.dispose(), { once: true });
if (import.meta.hot) import.meta.hot.dispose(() => api.dispose());
