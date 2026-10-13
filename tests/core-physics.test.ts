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

  it('raycasts the nearest box in world units, filters an entity, and identifies the ground', async () => {
    const world = await PhysicsWorld.create();
    try {
      world.createBox(11, [0, 2, 0], [0.5, 0.5, 0.5]);
      world.createBox(12, [0, 4, 0], [0.5, 0.5, 0.5]);
      const nearest = world.raycast([0, 6, 0], [0, -4, 0], 10);
      expect(nearest?.entityId).toBe(12);
      expect(nearest?.distance).toBeCloseTo(1.5);
      expect(nearest?.point).toEqual([0, 4.5, 0]);
      expect(nearest?.normal).toEqual([0, 1, 0]);

      const excluded = world.raycast([0, 6, 0], [0, -4, 0], 10, 12);
      expect(excluded?.entityId).toBe(11);
      expect(excluded?.distance).toBeCloseTo(3.5);
      world.step(1 / 60);
      expect(world.raycast([0, 6, 0], [0, -4, 0], 10)?.entityId).toBe(12);
      expect(world.raycast([0, 6, 0], [0, -4, 0], 10, 12)?.entityId).toBe(11);
      world.remove(11);
      expect(world.raycast([0, 6, 0], [0, -1, 0], 10, 12)).toMatchObject({
        entityId: null, point: [0, 0, 0], distance: 6,
      });
      world.remove(12);
      expect(world.raycast([0, 6, 0], [0, -1, 0], 5)).toBeNull();
    } finally {
      world.dispose();
    }
  });

  it('validates ray inputs and rejects queries after disposal', async () => {
    const world = await PhysicsWorld.create();
    try {
      expect(() => world.raycast([0, 1, 0], [0, 0, 0], 10)).toThrow(RangeError);
      expect(() => world.raycast([0, 1, 0], [0, -1, 0], 0)).toThrow(RangeError);
      expect(() => world.raycast([0, Number.NaN, 0], [0, -1, 0], 10)).toThrow(RangeError);
      expect(() => world.raycast([0, 1, 0], [0, -1, 0], 10, -1)).toThrow(RangeError);
    } finally {
      world.dispose();
    }
    expect(() => world.raycast([0, 1, 0], [0, -1, 0], 10)).toThrow('disposed');
  });

  it('queries the current collider position after a step and a teleport', async () => {
    const world = await PhysicsWorld.create();
    try {
      world.createBox(21, [0, 3, 0], [0.5, 0.5, 0.5]);
      world.step(1 / 60);
      expect(world.raycast([0, 6, 0], [0, -1, 0], 10)?.entityId).toBe(21);

      world.teleportBox(21, [5, 3, 0]);
      expect(world.raycast([0, 6, 0], [0, -1, 0], 10)?.entityId).toBeNull();
      expect(world.raycast([5, 6, 0], [0, -1, 0], 10)?.entityId).toBe(21);
    } finally {
      world.dispose();
    }
  });

  it('finds sphere overlaps in current collider state and omits the ground', async () => {
    const world = await PhysicsWorld.create();
    try {
      world.createBox(31, [0, 2, 0], [0.5, 0.5, 0.5]);
      world.createBox(32, [5, 2, 0], [0.5, 0.5, 0.5]);
      expect(world.overlapSphere([0, 2, 0], 1)).toEqual([31]);
      expect(world.overlapSphere([0, 0, 0], 0.25)).toEqual([]);
      world.step(1 / 60);
      expect(world.overlapSphere([0, 2, 0], 1)).toEqual([31]);
      expect(world.overlapSphere([5, 2, 0], 1)).toEqual([32]);
      expect(world.overlapSphere([0, 0, 0], 0.25)).toEqual([]);
      world.teleportBox(31, [5, 2, 0]);
      expect(world.overlapSphere([5, 2, 0], 1)).toEqual([31, 32]);
      world.remove(31);
      expect(world.overlapSphere([5, 2, 0], 1)).toEqual([32]);
      expect(() => world.overlapSphere([0, 0, 0], 0)).toThrow(RangeError);
    } finally {
      world.dispose();
    }
    expect(() => world.overlapSphere([0, 0, 0], 1)).toThrow('disposed');
  });
});
