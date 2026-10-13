import { describe, expect, it } from 'vitest';
import { EntityRegistry, FixedStepper } from '../src/core';
import { PhysicsWorld } from '../src/physics';

describe('EntityRegistry', () => {
  it('never reuses an ID after removal', () => {
    const registry = new EntityRegistry();
    const first = registry.create();
    registry.remove(first);
    const second = registry.create();
    expect(second).toBeGreaterThan(first);
    expect(registry.has(first)).toBe(false);
    expect(registry.has(second)).toBe(true);
  });
});

describe('FixedStepper', () => {
  it('runs the same number of fixed updates at different frame rates', () => {
    const run = (frames: number, elapsed: number) => {
      const stepper = new FixedStepper();
      let updates = 0;
      for (let frame = 0; frame < frames; frame++) {
        stepper.advance(elapsed, () => updates++);
      }
      return updates;
    };
    expect(run(120, 1 / 120)).toBe(60);
    expect(run(60, 1 / 60)).toBe(60);
  });

  it('reports discarded backlog and leaves no phantom update after a capped frame', () => {
    const stepper = new FixedStepper(0.1, 3);
    expect(stepper.advance(0.05, () => {})).toMatchObject({ steps: 0, alpha: 0.5 });
    const result = stepper.advance(0.7, () => {});
    expect(result.steps).toBe(3);
    expect(result.droppedSeconds).toBeCloseTo(0.45);
    expect(result.alpha).toBeCloseTo(0);
    expect(stepper.advance(0, () => {}).steps).toBe(0);
  });

  it('does not accumulate paused time and clears interpolation on reset', () => {
    const stepper = new FixedStepper(0.1);
    stepper.advance(0.05, () => {});
    stepper.pause();
    expect(stepper.advance(10, () => {})).toEqual({ steps: 0, alpha: 0.5, droppedSeconds: 0 });
    stepper.resume();
    expect(stepper.advance(0.05, () => {}).steps).toBe(1);
    stepper.advance(0.05, () => {});
    stepper.reset();
    expect(stepper.advance(0, () => {}).alpha).toBe(0);
    expect(() => stepper.advance(Number.NaN, () => {})).toThrow(RangeError);
    expect(() => stepper.advance(-1, () => {})).toThrow(RangeError);
  });
});

describe('PhysicsWorld', () => {
  it('interpolates fixed-step snapshots and resets them on teleport', async () => {
    const world = await PhysicsWorld.create();
    try {
      world.createBox(1, [0, 2, 0], [0.5, 0.5, 0.5]);
      expect(world.bodyCount).toBe(1);
      expect(world.activeCount).toBe(1);
      world.step(1 / 60);
      const previous = world.pose(1, 0)!;
      const current = world.pose(1, 1)!;
      const halfway = world.pose(1, 0.5)!;
      expect(previous.position[1]).toBeCloseTo(2);
      expect(current.position[1]).toBeLessThan(2);
      expect(halfway.position[1]).toBeCloseTo((previous.position[1] + current.position[1]) / 2);

      world.teleportBox(1, [3, 4, 5]);
      expect(world.pose(1, 0)?.position).toEqual([3, 4, 5]);
      expect(world.pose(1, 1)?.position).toEqual([3, 4, 5]);
      world.step(1 / 60);
      expect(world.pose(1, 0)?.position).toEqual([3, 4, 5]);
      for (let i = 0; i < 360; i++) world.step(1 / 60);
      expect(world.pose(1, 1)!.position[1]).toBeCloseTo(0.5, 1);

      world.remove(1);
      expect(world.bodyCount).toBe(0);
      expect(world.activeCount).toBe(0);
      expect(world.pose(1, 0.5)).toBeNull();
      world.step(1 / 60);
      expect(world.bodyCount).toBe(0);
    } finally {
      world.dispose();
    }
  });

  it('rejects duplicate and invalid boxes without leaking bodies', async () => {
    const world = await PhysicsWorld.create();
    try {
      world.createBox(4, [0, 1, 0], [1, 1, 1]);
      expect(() => world.createBox(4, [0, 1, 0], [1, 1, 1])).toThrow();
      expect(() => world.createBox(5, [0, 1, 0], [0, 1, 1])).toThrow(RangeError);
      expect(world.bodyCount).toBe(1);
    } finally {
      world.dispose();
    }
  });
});
