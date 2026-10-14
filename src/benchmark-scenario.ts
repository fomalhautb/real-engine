export type BenchmarkAction =
  | 'spawn' | 'remove' | 'reset'
  | 'material-alt' | 'material-base'
  | 'resize-small' | 'resize-base' | 'camera-cut';

export interface BenchmarkScenarioSample {
  eye: [number, number, number];
  target: [number, number, number];
  sun: { direction: [number, number, number]; color: [number, number, number] };
  actions: BenchmarkAction[];
}

const FRAMES_PER_CYCLE = 1800;
const FRAMES_PER_SECOND = 60;
const TWO_PI = 2 * Math.PI;
const TARGET: [number, number, number] = [0, 1.2, -1.5];

/** Pure, repeatable foundation workload for the current demo scene.
 * Its small fixed action schedule is not a large-scene performance result. */
export function sampleBenchmarkScenario(frame: number, seed: number, stress: boolean): BenchmarkScenarioSample {
  if (!Number.isSafeInteger(frame) || frame < 0 || frame > 1_000_000) throw new RangeError('benchmark frame must be an integer in [0, 1000000]');
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new RangeError('benchmark seed must be uint32');
  if (typeof stress !== 'boolean') throw new TypeError('benchmark stress must be boolean');

  const seconds = frame / FRAMES_PER_SECOND;
  const phase = (seed / 0x1_0000_0000) * TWO_PI;
  const cycleFrame = frame % FRAMES_PER_CYCLE;
  const baseAngle = phase + seconds * (TWO_PI / (FRAMES_PER_CYCLE / FRAMES_PER_SECOND));
  const cut = stress && cycleFrame >= 1600;
  const angle = baseAngle + (cut ? Math.PI : 0);
  const radius = 20 + 2 * Math.sin(baseAngle * 2 + phase);
  const eye: [number, number, number] = [
    TARGET[0] + radius * Math.sin(angle),
    6 + Math.sin(baseAngle + 0.7),
    TARGET[2] + radius * Math.cos(angle),
  ];

  const sunAngle = phase * 0.3 + seconds * (TWO_PI / 120);
  const rawSun: [number, number, number] = [
    -0.45 + 0.18 * Math.sin(sunAngle),
    0.72 + 0.05 * Math.cos(sunAngle),
    0.53 + 0.14 * Math.cos(sunAngle),
  ];
  const sunLength = Math.hypot(...rawSun);
  const direction: [number, number, number] = rawSun.map(value => value / sunLength) as [number, number, number];
  const color: [number, number, number] = [
    3.5 + 0.12 * Math.sin(sunAngle),
    3.2 + 0.10 * Math.sin(sunAngle + 0.5),
    2.8 + 0.08 * Math.cos(sunAngle),
  ];

  const actions: BenchmarkAction[] = [];
  if (stress) {
    if (cycleFrame >= 180 && cycleFrame <= 315 && (cycleFrame - 180) % 9 === 0) actions.push('spawn');
    if (cycleFrame >= 600 && cycleFrame <= 735 && (cycleFrame - 600) % 9 === 0) actions.push('remove');
    if (cycleFrame === 900) actions.push('material-alt');
    if (cycleFrame === 1050) actions.push('resize-small');
    if (cycleFrame === 1200) actions.push('material-base');
    if (cycleFrame === 1350) actions.push('resize-base');
    if (cycleFrame === 1500) actions.push('reset');
    if (cycleFrame === 0 || cycleFrame === 1600) actions.push('camera-cut');
  }
  return { eye, target: [...TARGET], sun: { direction, color }, actions };
}
