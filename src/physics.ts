import RAPIER, { type RigidBody, type World } from '@dimforge/rapier3d-compat';

export type Pose = {
  position: [number, number, number];
  rotation: [number, number, number, number];
};

type DynamicBox = {
  body: RigidBody;
  previous: Pose;
  current: Pose;
};

function snapshot(body: RigidBody): Pose {
  const position = body.translation();
  const rotation = body.rotation();
  return {
    position: [position.x, position.y, position.z],
    rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
  };
}

function finiteVector(values: readonly number[], length: number, name: string): void {
  if (values.length !== length || values.some((value) => !Number.isFinite(value))) {
    throw new RangeError(`${name} must contain ${length} finite numbers`);
  }
}

function interpolate(previous: Pose, current: Pose, alpha: number): Pose {
  const position = previous.position.map((value, index) =>
    value + (current.position[index] - value) * alpha,
  ) as Pose['position'];

  const a = previous.rotation;
  const b = current.rotation;
  // q and -q represent the same orientation; choose the shorter arc.
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const sign = dot < 0 ? -1 : 1;
  const mixed = a.map((value, index) => value + (sign * b[index] - value) * alpha);
  const magnitude = Math.hypot(...mixed);
  const rotation = mixed.map((value) => value / magnitude) as Pose['rotation'];
  return { position, rotation };
}

/** A small Rapier adapter with no renderer dependency. The caller owns fixed timing. */
export class PhysicsWorld {
  private static initialization?: Promise<void>;
  private readonly world: World;
  private readonly boxes = new Map<number, DynamicBox>();
  private disposed = false;

  static async create(): Promise<PhysicsWorld> {
    PhysicsWorld.initialization ??= RAPIER.init().catch((error: unknown) => {
      PhysicsWorld.initialization = undefined;
      throw error;
    });
    await PhysicsWorld.initialization;
    return new PhysicsWorld();
  }

  private constructor() {
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    const ground = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -0.1, 0));
    this.world.createCollider(RAPIER.ColliderDesc.cuboid(40, 0.1, 40), ground);
  }

  get bodyCount(): number {
    this.assertAlive();
    return this.boxes.size;
  }

  get activeCount(): number {
    this.assertAlive();
    let count = 0;
    for (const { body } of this.boxes.values()) {
      if (!body.isSleeping()) count++;
    }
    return count;
  }

  createBox(
    entityId: number,
    position: [number, number, number],
    halfExtents: [number, number, number],
  ): void {
    this.assertAlive();
    if (!Number.isSafeInteger(entityId) || entityId < 1) {
      throw new RangeError('entityId must be a positive safe integer');
    }
    if (this.boxes.has(entityId)) {
      throw new Error(`Entity ${entityId} already has a physics box`);
    }
    finiteVector(position, 3, 'position');
    finiteVector(halfExtents, 3, 'halfExtents');
    if (halfExtents.some((value) => value <= 0)) {
      throw new RangeError('halfExtents must be positive');
    }

    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(...position),
    );
    try {
      this.world.createCollider(RAPIER.ColliderDesc.cuboid(...halfExtents), body);
      const initial = snapshot(body);
      this.boxes.set(entityId, { body, previous: initial, current: initial });
    } catch (error) {
      this.world.removeRigidBody(body);
      throw error;
    }
  }

  step(dt: number): void {
    this.assertAlive();
    if (!Number.isFinite(dt) || dt <= 0) {
      throw new RangeError('dt must be finite and positive');
    }
    this.world.timestep = dt;
    for (const box of this.boxes.values()) box.previous = box.current;
    this.world.step();
    for (const box of this.boxes.values()) box.current = snapshot(box.body);
  }

  pose(entityId: number, alpha: number): Pose | null {
    this.assertAlive();
    if (!Number.isFinite(alpha)) {
      throw new RangeError('alpha must be finite');
    }
    const box = this.boxes.get(entityId);
    if (!box) return null;
    return interpolate(box.previous, box.current, Math.max(0, Math.min(1, alpha)));
  }

  teleportBox(entityId: number, position: [number, number, number]): void {
    this.assertAlive();
    finiteVector(position, 3, 'position');
    const box = this.boxes.get(entityId);
    if (!box) throw new Error(`Entity ${entityId} has no physics box`);
    box.body.setTranslation({ x: position[0], y: position[1], z: position[2] }, true);
    box.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    box.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    box.body.wakeUp();
    const pose = snapshot(box.body);
    box.previous = pose;
    box.current = pose;
  }

  remove(entityId: number): void {
    this.assertAlive();
    const box = this.boxes.get(entityId);
    if (!box) return;
    this.world.removeRigidBody(box.body);
    this.boxes.delete(entityId);
  }

  dispose(): void {
    if (this.disposed) return;
    this.boxes.clear();
    this.world.free();
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('PhysicsWorld has been disposed');
  }
}
