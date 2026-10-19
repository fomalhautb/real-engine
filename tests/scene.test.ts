import { describe, expect, it } from 'vitest';
import { DemoScene } from '../src/scene';
import { transformPoint } from '../src/math';
import { compileMaterial, evaluateMaterial } from '../src/material';

describe('DemoScene', () => {
  it('creates bounded batches with four compatible bone transforms', async () => {
    const scene = await DemoScene.create();
    try {
      expect(scene.batches.map((batch) => batch.name)).toEqual(['character', 'trees', 'ground', 'boxes', 'material-card']);
      expect(scene.batches.map((batch) => batch.capacity)).toEqual([1, 256, 1, 64, 1]);
      expect(scene.batches.map((batch) => batch.instances.length)).toEqual([1, 32, 1, 6, 1]);
      expect(scene.physics.bodyCount).toBe(6);
      for (const batch of scene.batches) {
        for (const item of batch.instances) {
          expect(item.bones).toHaveLength(4);
          for (const bone of item.bones) expect(bone).toHaveLength(16);
        }
      }
      expect(transformPoint(scene.batches[0].instances[0].model, [0, 0, 0])).toEqual([-2, 0, 0]);
      expect(transformPoint(scene.batches[1].instances[0].model, [0, 0, 0])).toEqual([2.4000000953674316, 0, 0]);
      expect(scene.batches[2].instances[0].color[3]).toBe(1);
      expect(scene.batches[2].instances[0].normalMatrix[0]).toBeCloseTo(1 / 80);
      expect(scene.batches[2].instances[0].normalMatrix[5]).toBe(1);
    } finally {
      scene.dispose();
    }
  });

  it('registers an imported masked card and repeats a textured ground', async () => {
    const scene = await DemoScene.create(0);
    try {
      const card = scene.batches[4];
      expect(card.mesh.vertexCount).toBe(4);
      expect([...card.mesh.indices]).toEqual([0, 1, 2, 0, 2, 3]);
      expect(card.mesh.bounds).toEqual({ min: [-0.800000011920929, -0.800000011920929, 0], max: [0.800000011920929, 0.800000011920929, 0] });
      expect([...card.mesh.uvs!]).toEqual([0, 1, 1, 1, 1, 0, 0, 0]);
      expect(transformPoint(card.instances[0].model, [0, 0, 0])).toEqual([0, 1.5, 1]);
      expect(scene.handleForEntity(card.instances[0].entityId)).toBeDefined();
      expect(scene.renderBatches[4].instances[0].entityId).toBe(card.instances[0].entityId);
      expect(scene.batches[2].mesh.uvs).toEqual(new Float32Array([0, 0, 40, 0, 40, 40, 0, 40]));
      expect(card.deformation.nodes).toEqual([]);

      const binding = card.material!;
      const program = compileMaterial(binding.definition);
      expect([program.shading, program.alphaMode, program.doubleSided]).toEqual(['unlit', 'mask', true]);
      expect(program.textureSlots).toEqual([0]);
      const texture = binding.textures![0]!;
      if (!('pixels' in texture)) throw new Error('Expected decoded demo card texture');
      const sample = (x: number, y: number) => Array.from(texture.pixels.slice((y * texture.width + x) * 4, (y * texture.width + x) * 4 + 4), v => v / 255) as [number, number, number, number];
      const input = { parameters: binding.parameters, instanceColor: [1, 1, 1, 1] as const, instanceMetalRough: [0, 1] as const };
      expect(evaluateMaterial(program, { ...input, textures: () => sample(32, 32) }).covered).toBe(false);
      expect(evaluateMaterial(program, { ...input, textures: () => sample(55, 32) }).covered).toBe(true);

      const ground = scene.batches[2].material!;
      expect(ground.definition.alphaMode).toBe('opaque');
      const groundTexture = ground.textures![0]!;
      if (!('pixels' in groundTexture)) throw new Error('Expected decoded demo ground texture');
      expect(groundTexture.pixels[3]).toBe(255);
      expect(groundTexture.pixels[(40 * groundTexture.width + 40) * 4 + 3]).toBe(255);
    } finally {
      scene.dispose();
    }
  });

  it('steps physics with bounded animation time and freezes while paused', async () => {
    const scene = await DemoScene.create(2);
    try {
      const character = scene.batches[0].instances[0];
      const tree = scene.batches[1].instances[0];
      const originalCharacterBone = [...character.bones[2]];
      const originalTreeBone = [...tree.bones[1]];
      scene.advance(0.5);
      expect(scene.time).toBeCloseTo(0.1);
      expect(scene.droppedSeconds).toBeGreaterThan(0);
      expect([...character.bones[2]]).not.toEqual(originalCharacterBone);
      expect([...tree.bones[1]]).not.toEqual(originalTreeBone);
      const heldModel = [...scene.batches[3].instances[0].model];
      const heldTime = scene.time;
      const heldDropped = scene.droppedSeconds;
      scene.paused = true;
      scene.advance(10);
      expect(scene.time).toBe(heldTime);
      expect(scene.droppedSeconds).toBe(heldDropped);
      expect([...scene.batches[3].instances[0].model]).toEqual(heldModel);
      scene.paused = false;
      scene.advance(1 / 60);
      expect(scene.time).toBeGreaterThan(heldTime);
    } finally {
      scene.dispose();
    }
  });

  it('preserves entity identity through removal and resets teleport history', async () => {
    const scene = await DemoScene.create(0);
    try {
      const boxes = scene.batches[3].instances;
      const initialPosition = scene.physics.pose(boxes[0].entityId, 0)!.position;
      const ids = boxes.map((box) => box.entityId);
      expect(scene.removeBox()).toBe(true);
      expect(boxes.map((box) => box.entityId)).toEqual(ids.slice(0, -1));
      expect(scene.spawnBox()).toBe(true);
      expect(boxes.at(-1)!.entityId).toBeGreaterThan(ids.at(-1)!);
      scene.advance(1 / 6);
      const version = boxes[0].historyVersion;
      scene.reset();
      expect(scene.time).toBe(0);
      expect(scene.droppedSeconds).toBe(0);
      expect(boxes[0].historyVersion).toBe(version + 1);
      expect(scene.batches[0].instances[0].historyVersion).toBe(1);
      expect(scene.physics.pose(boxes[0].entityId, 0)?.position).toEqual(initialPosition);
      expect(boxes[0].model[13]).toBeCloseTo(initialPosition[1]);
      expect(scene.physics.bodyCount).toBe(boxes.length);
    } finally {
      scene.dispose();
      scene.dispose();
    }
  });

  it('replays spawned body positions after a reset', async () => {
    const scene = await DemoScene.create(0);
    try {
      const positions: number[][] = [];
      const firstIds: number[] = [];
      for (let i = 0; i < 16; i++) {
        expect(scene.spawnBox()).toBe(true);
        const id = scene.batches[3].instances.at(-1)!.entityId;
        firstIds.push(id);
        positions.push([...scene.physics.pose(id, 0)!.position]);
      }
      for (let i = 0; i < 16; i++) expect(scene.removeBox()).toBe(true);
      scene.advance(1 / 10);
      scene.reset();
      for (let i = 0; i < 16; i++) {
        expect(scene.spawnBox()).toBe(true);
        const id = scene.batches[3].instances.at(-1)!.entityId;
        expect(id).toBeGreaterThan(firstIds.at(-1)!);
        expect(scene.physics.pose(id, 0)!.position).toEqual(positions[i]);
      }
      expect(scene.physics.bodyCount).toBe(22);
    } finally { scene.dispose(); }
  });

  it('enforces instance capacities and validates tree counts', async () => {
    await expect(DemoScene.create(257)).rejects.toThrow(RangeError);
    const scene = await DemoScene.create(256);
    try {
      expect(scene.batches[1].instances).toHaveLength(256);
      for (let i = 6; i < 64; i++) expect(scene.spawnBox()).toBe(true);
      expect(scene.spawnBox()).toBe(false);
      expect(scene.batches[3].instances).toHaveLength(64);
      expect(scene.physics.bodyCount).toBe(64);
      for (let i = 0; i < 64; i++) expect(scene.removeBox()).toBe(true);
      expect(scene.removeBox()).toBe(false);
      expect(scene.physics.bodyCount).toBe(0);
    } finally {
      scene.dispose();
    }
  });

  it('commits animation and physics to a separate render frame', async () => {
    const scene = await DemoScene.create(2);
    try {
      const initialRevision = scene.committedRevision;
      const simulationCharacter = scene.batches[0].instances[0];
      const renderedCharacter = scene.renderBatches[0].instances[0];
      const renderedBox = scene.renderBatches[3].instances[0];
      const oldBone = [...renderedCharacter.bones[2]];
      const oldModel = [...renderedBox.model];
      expect(renderedCharacter).not.toBe(simulationCharacter);
      expect(renderedCharacter.bones[2]).not.toBe(simulationCharacter.bones[2]);
      expect(renderedCharacter.historyVersion).toBe(1);
      expect(simulationCharacter.historyVersion).toBe(0);

      scene.advance(1 / 10);
      expect([...simulationCharacter.bones[2]]).not.toEqual(oldBone);
      expect([...renderedCharacter.bones[2]]).toEqual(oldBone);
      expect([...renderedBox.model]).toEqual(oldModel);
      expect(scene.committedRevision).toBe(initialRevision);

      scene.commitFrame();
      expect(scene.committedRevision).toBeGreaterThan(initialRevision);
      expect(scene.changeCounts).toEqual({ create: 0, update: 9, remove: 0 });
      expect([...scene.renderBatches[0].instances[0].bones[2]]).not.toEqual(oldBone);
      expect([...renderedCharacter.bones[2]]).toEqual(oldBone);
      expect([...renderedBox.model]).toEqual(oldModel);
      expect(scene.renderBatches[3].instances[0]).not.toBe(scene.batches[3].instances[0]);

      const committedRevision = scene.committedRevision;
      scene.paused = true;
      scene.advance(1);
      scene.commitFrame();
      expect(scene.changeCounts).toEqual({ create: 0, update: 0, remove: 0 });
      expect(scene.committedRevision).toBe(committedRevision);
    } finally {
      scene.dispose();
    }
  });

  it('uses generation-safe handles when removed render slots are reused', async () => {
    const scene = await DemoScene.create(0);
    try {
      const old = scene.batches[3].instances.at(-1)!;
      const oldHandle = scene.handleForEntity(old.entityId)!;
      expect(scene.removeBox()).toBe(true);
      expect(scene.spawnBox()).toBe(true);
      const next = scene.batches[3].instances.at(-1)!;
      const nextHandle = scene.handleForEntity(next.entityId)!;
      expect(next.entityId).toBeGreaterThan(old.entityId);
      expect(nextHandle.index).toBe(oldHandle.index);
      expect(nextHandle.generation).toBe(oldHandle.generation + 1);
      expect(scene.handleForEntity(old.entityId)).toBeUndefined();
      expect(scene.renderBatches[3].instances.some((item) => item.entityId === old.entityId)).toBe(true);
      scene.commitFrame();
      expect(scene.changeCounts).toEqual({ create: 1, update: 0, remove: 1 });
      expect(scene.renderBatches[3].instances.some((item) => item.entityId === old.entityId)).toBe(false);
      expect(scene.renderBatches[3].instances.some((item) => item.entityId === next.entityId)).toBe(true);
    } finally {
      scene.dispose();
    }
  });

  it('commits reset as explicit history-invalidating updates', async () => {
    const scene = await DemoScene.create(1);
    try {
      scene.advance(1 / 10);
      scene.commitFrame();
      const before = scene.renderBatches[0].instances[0];
      const treeBefore = scene.renderBatches[1].instances[0];
      const boxBefore = scene.renderBatches[3].instances[0];
      scene.reset();
      expect(scene.renderBatches[0].instances[0]).toBe(before);
      expect(scene.renderBatches[1].instances[0]).toBe(treeBefore);
      expect(scene.renderBatches[3].instances[0]).toBe(boxBefore);
      scene.commitFrame();
      expect(scene.changeCounts).toEqual({ create: 0, update: 8, remove: 0 });
      expect(scene.renderBatches[0].instances[0].historyVersion).toBe(before.historyVersion + 1);
      expect(scene.renderBatches[1].instances[0].historyVersion).toBe(treeBefore.historyVersion + 1);
      expect(scene.renderBatches[3].instances[0].historyVersion).toBe(boxBefore.historyVersion + 1);
    } finally {
      scene.dispose();
    }
  });

  it('repairs render indices after middle removal and takes topology history from store events', async () => {
    const scene = await DemoScene.create(0);
    try {
      const boxes = scene.renderBatches[3].instances;
      const removed = boxes[2];
      const shifted = boxes[4];
      const removedHandle = scene.handleForEntity(removed.entityId)!;
      const shiftedHandle = scene.handleForEntity(shifted.entityId)!;
      const oldHistory = shifted.historyVersion;

      // Inject generic scene changes to exercise an operation beyond the demo's last-box UI.
      scene['store'].remove(removedHandle);
      scene.commitFrame();
      expect(scene.renderBatches[3].instances[3].entityId).toBe(shifted.entityId);

      const replacement = structuredClone(shifted);
      replacement.color = [0.9, 0.1, 0.2, 1];
      scene['store'].update(shiftedHandle, ['topology'], replacement);
      scene.commitFrame();
      const rendered = scene.renderBatches[3].instances[3];
      expect(rendered.entityId).toBe(shifted.entityId);
      expect(rendered.color).toEqual([0.9, 0.1, 0.2, 1]);
      expect(rendered.historyVersion).toBe(oldHistory + 1);
      expect(scene.batches[3].instances[4].historyVersion).toBe(0);
    } finally {
      scene.dispose();
    }
  });
});
