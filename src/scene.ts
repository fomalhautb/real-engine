import { EntityRegistry, FixedStepper } from './core';
import { createCharacter, createCube, createPlane, createTree } from './geometry';
import type { InstanceState, SceneBatch } from './contracts';
import {
  identity, multiply, quaternionMatrix, rotationX, rotationY, rotationZ,
  scaling, translation, type Mat4,
} from './math';
import { PhysicsWorld } from './physics';

const pivotRotation = (pivot: readonly number[], rotation: Mat4): Mat4 =>
  multiply(
    multiply(translation(pivot[0], pivot[1], pivot[2]), rotation),
    translation(-pivot[0], -pivot[1], -pivot[2]),
  );

const initialBoxPosition = (index: number): [number, number, number] => [
  [0, -0.72, 0.72][index % 3],
  2 + Math.floor(index / 3) * 0.76 + (index % 3) * 0.03,
  -2.5,
];

function instance(
  entityId: number, model: Mat4, normalMatrix: Mat4,
  color: [number, number, number, number], metalness: number, roughness: number,
): InstanceState {
  return {
    entityId, model, normalMatrix,
    bones: [identity(), identity(), identity(), identity()],
    morph: 0, color, metalness, roughness, historyVersion: 0,
  };
}

/** A deterministic demo scene; physics, animation, and render state stay renderer independent. */
export class DemoScene {
  readonly batches: SceneBatch[];
  readonly physics: PhysicsWorld;
  time = 0;
  droppedSeconds = 0;

  private readonly ids = new EntityRegistry();
  private readonly stepper = new FixedStepper();
  private readonly treePositions: [number, number, number][] = [];
  private nextBoxOrdinal = 0;
  private disposed = false;
  private isPaused = false;

  private constructor(physics: PhysicsWorld) {
    this.physics = physics;
    this.batches = [
      { name: 'character', mesh: createCharacter(), capacity: 1, instances: [] },
      { name: 'trees', mesh: createTree(), capacity: 256, instances: [] },
      { name: 'ground', mesh: createPlane(), capacity: 1, instances: [] },
      { name: 'boxes', mesh: createCube(), capacity: 64, instances: [] },
    ];
  }

  static async create(treeCount = 32): Promise<DemoScene> {
    if (!Number.isSafeInteger(treeCount) || treeCount < 0 || treeCount > 256) {
      throw new RangeError('treeCount must be an integer from 0 to 256');
    }
    const physics = await PhysicsWorld.create();
    const scene = new DemoScene(physics);
    try {
      scene.initialize(treeCount);
      return scene;
    } catch (error) {
      physics.dispose();
      throw error;
    }
  }

  get paused(): boolean { return this.isPaused; }
  set paused(value: boolean) {
    this.assertAlive();
    this.isPaused = value;
    if (value) this.stepper.pause();
    else this.stepper.resume();
  }

  private initialize(treeCount: number): void {
    this.batches[0].instances.push(instance(
      this.ids.create(), translation(-2, 0, 0), identity(), [0.24, 0.48, 0.83, 1], 0.08, 0.7,
    ));
    for (let i = 0; i < treeCount; i++) {
      const position: [number, number, number] = i === 0
        ? [2.4, 0, 0]
        : [-15 + ((i - 1) % 16) * 2, 0, -16 + Math.floor((i - 1) / 16) * 0.8];
      this.treePositions.push(position);
      this.batches[1].instances.push(instance(
        this.ids.create(), translation(...position), identity(), [0.24, 0.55, 0.28, 1], 0, 0.88,
      ));
    }
    this.batches[2].instances.push(instance(
      this.ids.create(), scaling(80, 1, 80), scaling(1 / 80, 1, 1 / 80),
      [0.12, 0.19, 0.20, 0], 0, 1,
    ));
    for (let i = 0; i < 6; i++) this.spawnBox();
    this.updateAnimated();
    this.updateBoxes(0);
  }

  advance(elapsed: number): void {
    this.assertAlive();
    if (!Number.isFinite(elapsed) || elapsed < 0) throw new RangeError('elapsed must be finite and nonnegative');
    if (this.paused) return;
    const result = this.stepper.advance(elapsed, (dt) => this.physics.step(dt));
    this.droppedSeconds += result.droppedSeconds;
    this.time += Math.min(elapsed, 0.1);
    this.updateAnimated();
    this.updateBoxes(result.alpha);
  }

  spawnBox(): boolean {
    this.assertAlive();
    const boxes = this.batches[3].instances;
    if (boxes.length >= this.batches[3].capacity) return false;
    const entityId = this.ids.create();
    const position = initialBoxPosition(this.nextBoxOrdinal++);
    try {
      this.physics.createBox(entityId, position, [0.35, 0.35, 0.35]);
    } catch (error) {
      this.ids.remove(entityId);
      throw error;
    }
    boxes.push(instance(
      entityId, multiply(translation(...position), scaling(0.7, 0.7, 0.7)),
      scaling(1 / 0.7, 1 / 0.7, 1 / 0.7), [0.95, 0.55, 0.12, 1], 0.36, 0.5,
    ));
    return true;
  }

  removeBox(): boolean {
    this.assertAlive();
    const removed = this.batches[3].instances.pop();
    if (!removed) return false;
    this.physics.remove(removed.entityId);
    this.ids.remove(removed.entityId);
    return true;
  }

  reset(): void {
    this.assertAlive();
    this.time = 0;
    this.droppedSeconds = 0;
    this.stepper.reset();
    this.updateAnimated();
    this.batches[0].instances[0].historyVersion++;
    for (const tree of this.batches[1].instances) tree.historyVersion++;
    for (const [index, box] of this.batches[3].instances.entries()) {
      this.physics.teleportBox(box.entityId, initialBoxPosition(index));
      box.historyVersion++;
    }
    this.updateBoxes(0);
  }

  dispose(): void {
    if (this.disposed) return;
    this.physics.dispose();
    this.disposed = true;
  }

  private updateAnimated(): void {
    const t = this.time;
    const character = this.batches[0].instances[0];
    character.bones = [
      identity(),
      pivotRotation([0, 2.3, 0], multiply(rotationY(Math.sin(t * 1.3) * 0.22), rotationZ(Math.sin(t * 0.8) * 0.08))),
      pivotRotation([-0.55, 1.95, 0], rotationX(Math.sin(t * 2.2) * 0.48)),
      pivotRotation([0.55, 1.95, 0], rotationX(-Math.sin(t * 2.2) * 0.48)),
    ];
    character.morph = 0.5 + 0.5 * Math.sin(t * 1.7);
    for (let i = 0; i < this.batches[1].instances.length; i++) {
      const tree = this.batches[1].instances[i];
      const sway = Math.sin(t * 1.6 + i * 0.47) * 0.08;
      tree.bones = [
        identity(),
        pivotRotation([0, 1.2, 0], rotationZ(sway)),
        pivotRotation([-0.6, 2, 0], rotationZ(sway * 1.8)),
        pivotRotation([0.6, 2.5, 0], rotationZ(-sway * 1.5)),
      ];
    }
  }

  private updateBoxes(alpha: number): void {
    for (const box of this.batches[3].instances) {
      const pose = this.physics.pose(box.entityId, alpha);
      if (!pose) throw new Error(`Physics box ${box.entityId} is missing`);
      const orientation = quaternionMatrix(...pose.rotation);
      box.model = multiply(multiply(translation(...pose.position), orientation), scaling(0.7, 0.7, 0.7));
      box.normalMatrix = multiply(orientation, scaling(1 / 0.7, 1 / 0.7, 1 / 0.7));
    }
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('DemoScene has been disposed');
  }
}
