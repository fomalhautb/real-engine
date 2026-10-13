import { EntityRegistry, FixedStepper } from './core';
import { createCharacter, createCube, createPlane, createTree } from './geometry';
import type { InstanceState, SceneBatch } from './contracts';
import {
  identity, multiply, quaternionMatrix, rotationX, rotationY, rotationZ,
  scaling, translation, type Mat4,
} from './math';
import { PhysicsWorld } from './physics';
import { authoredMorph, buildMorphTarget } from './morph-assets';
import { SceneStore, type ChangeKind, type EntityHandle } from './scene-store';
import { createCardMaterial, createGroundMaterial } from './demo-materials';
import { parseObj } from './mesh-import';
import materialCardObj from './assets/material-card.obj?raw';

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
    deformScalars: [0, 0], color, metalness, roughness, historyVersion: 0,
  };
}

/** A deterministic demo scene; physics, animation, and render state stay renderer independent. */
export class DemoScene {
  /** Mutable simulation state. The renderer consumes renderBatches after commitFrame. */
  readonly batches: SceneBatch[];
  readonly renderBatches: SceneBatch[];
  readonly physics: PhysicsWorld;
  time = 0;
  droppedSeconds = 0;
  committedRevision = 0;
  changeCounts: Readonly<{ create: number; update: number; remove: number }> = Object.freeze({ create: 0, update: 0, remove: 0 });

  private readonly ids = new EntityRegistry();
  private readonly store = new SceneStore<InstanceState>();
  private readonly handlesByEntity = new Map<number, EntityHandle>();
  private readonly batchByIdentity = new Map<string, number>();
  private readonly renderLocations = new Map<string, { batchIndex: number; instanceIndex: number; entityId: number }>();
  private readonly renderKeysByBatch: string[][];
  private readonly stepper = new FixedStepper();
  private readonly treePositions: [number, number, number][] = [];
  private nextBoxOrdinal = 0;
  private disposed = false;
  private isPaused = false;

  private constructor(physics: PhysicsWorld) {
    this.physics = physics;
    const character = createCharacter();
    const ground = createPlane();
    // The ground spans 80 metres, so forty UV repeats make each tile 2 metres.
    for (let i = 0; i < ground.uvs!.length; i++) ground.uvs![i] *= 40;
    const cardAsset = parseObj(materialCardObj);
    if (cardAsset.primitives.length !== 1 || cardAsset.primitives[0].materialName !== 'signal-ring') {
      throw new Error('Demo card OBJ must contain one signal-ring primitive');
    }
    this.batches = [
      { name: 'character', mesh: character, capacity: 1, instances: [],
        deformation: { jointCount: 4, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'morph', target: 1, weight: 1 }, { kind: 'skin' }] },
        morphTargets: [authoredMorph(character), buildMorphTarget(character, p => [p[1] > 2.3 ? 0.035 * Math.sin(p[1] * 5) : 0, 0, 0])] },
      { name: 'trees', mesh: createTree(), capacity: 256, instances: [],
        deformation: { jointCount: 4, nodes: [{ kind: 'skin' }, { kind: 'wave', amplitude: 0.035, frequency: 1.3, phase: 0 }] }, morphTargets: [] },
      { name: 'ground', mesh: ground, capacity: 1, instances: [], deformation: { jointCount: 4, nodes: [] }, morphTargets: [],
        material: createGroundMaterial() },
      { name: 'boxes', mesh: createCube(), capacity: 64, instances: [], deformation: { jointCount: 4, nodes: [] }, morphTargets: [] },
      { name: 'material-card', mesh: cardAsset.primitives[0].mesh, capacity: 1, instances: [],
        deformation: { jointCount: 4, nodes: [] }, morphTargets: [], material: createCardMaterial() },
    ];
    this.renderBatches = this.batches.map((batch) => ({ ...batch, instances: [], inputMode: 'committed' }));
    this.renderKeysByBatch = this.batches.map(() => []);
  }

  static async create(treeCount = 32): Promise<DemoScene> {
    if (!Number.isSafeInteger(treeCount) || treeCount < 0 || treeCount > 256) {
      throw new RangeError('treeCount must be an integer from 0 to 256');
    }
    const physics = await PhysicsWorld.create();
    const scene = new DemoScene(physics);
    try {
      scene.initialize(treeCount);
      scene.commitFrame();
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

  /** Returns the current generation-safe identity for a simulation entity. */
  handleForEntity(entityId: number): EntityHandle | undefined {
    const handle = this.handlesByEntity.get(entityId);
    return handle && { ...handle };
  }

  /** Applies only committed entity changes to the renderer-owned frame state. */
  commitFrame(): void {
    this.assertAlive();
    const counts = { create: 0, update: 0, remove: 0 };
    for (const change of this.store.drainChanges()) {
      counts[change.kind]++;
      const key = `${change.handle.index}:${change.handle.generation}`;
      if (change.kind === 'create') {
        const batchIndex = this.batchByIdentity.get(key);
        if (batchIndex === undefined) throw new Error(`Missing batch for scene entity ${key}`);
        const item = change.value;
        item.historyVersion = change.historyVersion;
        const instances = this.renderBatches[batchIndex].instances;
        const instanceIndex = instances.length;
        instances.push(item);
        this.renderKeysByBatch[batchIndex].push(key);
        this.renderLocations.set(key, { batchIndex, instanceIndex, entityId: item.entityId });
      } else if (change.kind === 'update') {
        const location = this.renderLocations.get(key);
        if (!location) throw new Error(`Missing rendered scene entity ${key}`);
        const instances = this.renderBatches[location.batchIndex].instances;
        if (instances[location.instanceIndex]?.entityId !== location.entityId) {
          throw new Error(`Missing rendered instance ${location.entityId}`);
        }
        const item = change.value;
        item.historyVersion = change.historyVersion;
        instances[location.instanceIndex] = item;
      } else {
        const location = this.renderLocations.get(key);
        if (!location) throw new Error(`Missing rendered scene entity ${key}`);
        const instances = this.renderBatches[location.batchIndex].instances;
        if (instances[location.instanceIndex]?.entityId !== location.entityId) {
          throw new Error(`Missing rendered instance ${location.entityId}`);
        }
        instances.splice(location.instanceIndex, 1);
        const keys = this.renderKeysByBatch[location.batchIndex];
        keys.splice(location.instanceIndex, 1);
        this.renderLocations.delete(key);
        this.batchByIdentity.delete(key);
        for (let index = location.instanceIndex; index < keys.length; index++) {
          const shifted = this.renderLocations.get(keys[index]);
          if (!shifted) throw new Error(`Missing rendered scene entity ${keys[index]}`);
          shifted.instanceIndex = index;
        }
      }
    }
    this.committedRevision = this.store.revision;
    this.changeCounts = Object.freeze(counts);
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
      [1, 1, 1, 1], 0, 1,
    ));
    for (let i = 0; i < 6; i++) this.spawnBox();
    this.batches[4].instances.push(instance(
      this.ids.create(), translation(0, 1.5, 1), identity(), [1, 1, 1, 1], 0, 1,
    ));
    this.updateAnimated();
    this.updateBoxes(0);
    for (const [batchIndex, batch] of this.batches.entries()) {
      for (const item of batch.instances) {
        if (!this.handlesByEntity.has(item.entityId)) this.register(item, batchIndex);
      }
    }
    for (const item of this.batches[3].instances) this.updateCommitted(item, ['transform']);
  }

  advance(elapsed: number): void {
    this.assertAlive();
    if (!Number.isFinite(elapsed) || elapsed < 0) throw new RangeError('elapsed must be finite and nonnegative');
    if (this.paused || elapsed === 0) return;
    const result = this.stepper.advance(elapsed, (dt) => this.physics.step(dt));
    this.droppedSeconds += result.droppedSeconds;
    this.time += Math.min(elapsed, 0.1);
    this.updateAnimated();
    this.updateBoxes(result.alpha);
    for (const item of this.batches[0].instances) this.updateCommitted(item, ['deformation']);
    for (const item of this.batches[1].instances) this.updateCommitted(item, ['deformation']);
    for (const item of this.batches[3].instances) this.updateCommitted(item, ['transform']);
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
    const box = instance(
      entityId, multiply(translation(...position), scaling(0.7, 0.7, 0.7)),
      scaling(1 / 0.7, 1 / 0.7, 1 / 0.7), [0.95, 0.55, 0.12, 1], 0.36, 0.5,
    );
    boxes.push(box);
    this.register(box, 3);
    return true;
  }

  removeBox(): boolean {
    this.assertAlive();
    const removed = this.batches[3].instances.pop();
    if (!removed) return false;
    this.physics.remove(removed.entityId);
    this.ids.remove(removed.entityId);
    const handle = this.handlesByEntity.get(removed.entityId);
    if (!handle) throw new Error(`Missing scene handle for ${removed.entityId}`);
    this.store.remove(handle);
    this.handlesByEntity.delete(removed.entityId);
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
    for (const item of this.batches[0].instances) this.updateCommitted(item, ['deformation', 'teleport']);
    for (const item of this.batches[1].instances) this.updateCommitted(item, ['deformation', 'teleport']);
    for (const item of this.batches[3].instances) this.updateCommitted(item, ['transform', 'teleport']);
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
    character.deformScalars = [0.5 + 0.5 * Math.sin(t * 1.7), 0.5 + 0.5 * Math.sin(t * 2.1)];
    for (let i = 0; i < this.batches[1].instances.length; i++) {
      const tree = this.batches[1].instances[i];
      const sway = Math.sin(t * 1.6 + i * 0.47) * 0.08;
      tree.deformScalars[0] = t * 1.6 + i * 0.47;
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

  private register(item: InstanceState, batchIndex: number): void {
    const handle = this.store.create(item);
    this.handlesByEntity.set(item.entityId, handle);
    this.batchByIdentity.set(`${handle.index}:${handle.generation}`, batchIndex);
  }

  private updateCommitted(item: InstanceState, changes: readonly ChangeKind[]): void {
    const handle = this.handlesByEntity.get(item.entityId);
    if (!handle) throw new Error(`Missing scene handle for ${item.entityId}`);
    this.store.update(handle, changes, item);
  }
}
