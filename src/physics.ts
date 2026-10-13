import RAPIER, { type Collider, type RigidBody, type World } from '@dimforge/rapier3d-compat';

export type Pose = {
  position: [number, number, number];
  rotation: [number, number, number, number];
};

/** A hit in the physics world. The fixed ground has no entity ID. */
export type RayHit = {
  entityId: number | null;
  point: [number, number, number];
  normal: [number, number, number];
  distance: number;
};

type DynamicBox = {
  body: RigidBody;
  collider: Collider;
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
  private readonly groundCollider: Collider;
  private readonly boxes = new Map<number, DynamicBox>();
  private readonly colliderEntities = new Map<number, number>();
  private queryDirty = true;
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
    this.groundCollider = this.world.createCollider(RAPIER.ColliderDesc.cuboid(40, 0.1, 40), ground);
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
      const collider = this.world.createCollider(RAPIER.ColliderDesc.cuboid(...halfExtents), body);
      const initial = snapshot(body);
      this.boxes.set(entityId, { body, collider, previous: initial, current: initial });
      this.colliderEntities.set(collider.handle, entityId);
      this.queryDirty = true;
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
    this.queryDirty = false;
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

  /** Cast against collider state, independent of rendering interpolation or camera LOD. */
  raycast(
    origin: [number, number, number],
    direction: [number, number, number],
    maxDistance: number,
    excludeEntityId?: number,
  ): RayHit | null {
    this.assertAlive();
    finiteVector(origin, 3, 'origin');
    finiteVector(direction, 3, 'direction');
    if (!Number.isFinite(maxDistance) || maxDistance <= 0) {
      throw new RangeError('maxDistance must be finite and positive');
    }
    if (excludeEntityId !== undefined && (!Number.isSafeInteger(excludeEntityId) || excludeEntityId < 1)) {
      throw new RangeError('excludeEntityId must be a positive safe integer');
    }
    const length = Math.hypot(...direction);
    if (!Number.isFinite(length) || length === 0) {
      throw new RangeError('direction must have a finite nonzero length');
    }
    const unit: [number, number, number] = [direction[0] / length, direction[1] / length, direction[2] / length];
    const ray = new RAPIER.Ray(
      { x: origin[0], y: origin[1], z: origin[2] },
      { x: unit[0], y: unit[1], z: unit[2] },
    );
    let nearest: { timeOfImpact: number; normal: { x: number; y: number; z: number } } | null;
    let entityId: number | null;
    if (this.queryDirty) {
      // Rapier's world query acceleration structure is refreshed by step().
      // Direct collider queries keep edits visible without advancing simulation.
      this.world.propagateModifiedBodyPositionsToColliders();
      nearest = this.groundCollider.castRayAndGetNormal(ray, maxDistance, true);
      entityId = null;
      for (const [candidateId, box] of this.boxes) {
        if (candidateId === excludeEntityId) continue;
        const hit = box.collider.castRayAndGetNormal(ray, nearest?.timeOfImpact ?? maxDistance, true);
        if (hit && (!nearest || hit.timeOfImpact < nearest.timeOfImpact)) {
          nearest = hit;
          entityId = candidateId;
        }
      }
    } else {
      const excludedBody = excludeEntityId === undefined ? undefined : this.boxes.get(excludeEntityId)?.body;
      const hit = this.world.castRayAndGetNormal(ray, maxDistance, true, undefined, undefined, undefined, excludedBody);
      nearest = hit;
      entityId = hit ? (this.colliderEntities.get(hit.collider.handle) ?? null) : null;
    }
    if (!nearest) return null;
    const distance = nearest.timeOfImpact;
    return {
      entityId,
      point: [
        origin[0] + unit[0] * distance,
        origin[1] + unit[1] * distance,
        origin[2] + unit[2] * distance,
      ],
      normal: [nearest.normal.x, nearest.normal.y, nearest.normal.z],
      distance,
    };
  }

  /** Return dynamic entity IDs whose colliders intersect the sphere; ground is omitted. */
  overlapSphere(center: [number, number, number], radius: number): number[] {
    this.assertAlive();
    finiteVector(center, 3, 'center');
    if (!Number.isFinite(radius) || radius <= 0) {
      throw new RangeError('radius must be finite and positive');
    }
    const shape = new RAPIER.Ball(radius);
    const position = { x: center[0], y: center[1], z: center[2] };
    const rotation = { x: 0, y: 0, z: 0, w: 1 };
    const entities: number[] = [];
    if (this.queryDirty) {
      this.world.propagateModifiedBodyPositionsToColliders();
      for (const [entityId, box] of this.boxes) {
        if (box.collider.intersectsShape(shape, position, rotation)) entities.push(entityId);
      }
    } else {
      this.world.intersectionsWithShape(position, rotation, shape, (collider) => {
        const entityId = this.colliderEntities.get(collider.handle);
        if (entityId !== undefined) entities.push(entityId);
        return true;
      });
    }
    return entities.sort((a, b) => a - b);
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
    this.queryDirty = true;
  }

  remove(entityId: number): void {
    this.assertAlive();
    const box = this.boxes.get(entityId);
    if (!box) return;
    this.colliderEntities.delete(box.collider.handle);
    this.world.removeRigidBody(box.body);
    this.boxes.delete(entityId);
    this.queryDirty = true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.boxes.clear();
    this.colliderEntities.clear();
    this.world.free();
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('PhysicsWorld has been disposed');
  }
}
