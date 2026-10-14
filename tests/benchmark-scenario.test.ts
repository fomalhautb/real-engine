import { describe, expect, it } from 'vitest';
import { sampleBenchmarkScenario, type BenchmarkAction } from '../src/benchmark-scenario';

const distance = (a: readonly number[], b: readonly number[]) => Math.hypot(...a.map((value, i) => value - b[i]));

describe('deterministic benchmark scenario', () => {
  it('has a continuous foundation orbit and slowly varying valid sun without stress events', () => {
    for (const frame of [0, 1, 300, 900, 1599, 1600, 1799, 1800, 1_000_000]) {
      const sample = sampleBenchmarkScenario(frame, 0xffffffff, false);
      expect(sample.actions).toEqual([]);
      expect(sample.target).toEqual([0, 1.2, -1.5]);
      expect([...sample.eye, ...sample.sun.direction, ...sample.sun.color].every(Number.isFinite)).toBe(true);
      expect(Math.hypot(sample.eye[0], sample.eye[2] + 1.5)).toBeGreaterThanOrEqual(18 - 1e-10);
      expect(Math.hypot(sample.eye[0], sample.eye[2] + 1.5)).toBeLessThanOrEqual(22 + 1e-10);
      expect(Math.hypot(...sample.sun.direction)).toBeCloseTo(1, 10);
      expect(sample.sun.color.every(channel => channel > 0)).toBe(true);
    }
    for (const frame of [1, 1600, 1800]) {
      expect(distance(sampleBenchmarkScenario(frame, 42, false).eye, sampleBenchmarkScenario(frame - 1, 42, false).eye)).toBeLessThan(0.2);
    }
    expect(distance(sampleBenchmarkScenario(60, 42, false).sun.direction, sampleBenchmarkScenario(0, 42, false).sun.direction)).toBeLessThan(0.02);
  });

  it('emits the exact balanced 1800-frame stress action schedule without duplicates', () => {
    const events = new Map<BenchmarkAction, number[]>();
    for (let frame = 0; frame < 3600; frame++) {
      const actions = sampleBenchmarkScenario(frame, 1234, true).actions;
      expect(new Set(actions).size).toBe(actions.length);
      for (const action of actions) events.set(action, [...(events.get(action) ?? []), frame]);
    }
    const repeat = (frames: number[]) => [...frames, ...frames.map(frame => frame + 1800)];
    expect(events.get('spawn')).toEqual(repeat(Array.from({ length: 16 }, (_, i) => 180 + i * 9)));
    expect(events.get('remove')).toEqual(repeat(Array.from({ length: 16 }, (_, i) => 600 + i * 9)));
    for (const [action, frame] of [
      ['material-alt', 900], ['resize-small', 1050], ['material-base', 1200],
      ['resize-base', 1350], ['reset', 1500],
    ] as const) expect(events.get(action)).toEqual(repeat([frame]));
    expect(events.get('camera-cut')).toEqual([0, 1600, 1800, 3400]);
    expect([...events.keys()].sort()).toEqual([
      'camera-cut', 'material-alt', 'material-base', 'remove', 'reset', 'resize-base', 'resize-small', 'spawn',
    ]);
  });

  it('marks actual camera discontinuities and restores the normal orbit at each cycle boundary', () => {
    const sample = (frame: number) => sampleBenchmarkScenario(frame, 7, true);
    expect(distance(sample(1599).eye, sample(1600).eye)).toBeGreaterThan(30);
    expect(distance(sample(1799).eye, sample(1800).eye)).toBeGreaterThan(30);
    expect(distance(sample(1600).eye, sample(1601).eye)).toBeLessThan(0.2);
    expect(distance(sample(1800).eye, sample(0).eye)).toBeLessThan(1e-10);
    expect(sample(1800).actions).toEqual(['camera-cut']);
  });

  it('depends only on frame, seed, and stress mode, regardless of sampling order', () => {
    const expected = sampleBenchmarkScenario(612, 0x7fffffff, true);
    sampleBenchmarkScenario(999, 0, true);
    sampleBenchmarkScenario(3, 0xffffffff, false);
    expect(sampleBenchmarkScenario(612, 0x7fffffff, true)).toEqual(expected);
    expect(sampleBenchmarkScenario(612, 0x80000000, true).eye).not.toEqual(expected.eye);
  });

  it('rejects noninteger or out-of-range frames and seeds', () => {
    for (const frame of [-1, 1.5, 1_000_001, Infinity, NaN]) {
      expect(() => sampleBenchmarkScenario(frame, 0, false)).toThrow(/frame/);
    }
    for (const seed of [-1, 0.5, 0x1_0000_0000, Infinity, NaN]) {
      expect(() => sampleBenchmarkScenario(0, seed, false)).toThrow(/seed/);
    }
    expect(() => sampleBenchmarkScenario(0, 0, 1 as unknown as boolean)).toThrow(/stress/);
    expect(sampleBenchmarkScenario(1_000_000, 0xffffffff, true).eye.every(Number.isFinite)).toBe(true);
  });
});
