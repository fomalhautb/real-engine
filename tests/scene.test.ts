import { describe, expect, it } from 'vitest';
import { DemoScene } from '../src/scene';
import { transformPoint } from '../src/math';

describe('DemoScene', () => {
  it('creates bounded batches with four compatible bone transforms', async () => {
    const scene = await DemoScene.create();
    try {
      expect(scene.batches.map((batch) => batch.name)).toEqual(['character', 'trees', 'ground', 'boxes']);
      expect(scene.batches.map((batch) => batch.capacity)).toEqual([1, 256, 1, 64]);
      expect(scene.batches.map((batch) => batch.instances.length)).toEqual([1, 32, 1, 6]);
      expect(scene.physics.bodyCount).toBe(6);
      for (const batch of scene.batches) {
        for (const item of batch.instances) {
          expect(item.bones).toHaveLength(4);
          for (const bone of item.bones) expect(bone).toHaveLength(16);
        }
      }
      expect(transformPoint(scene.batches[0].instances[0].model, [0, 0, 0])).toEqual([-2, 0, 0]);
      expect(transformPoint(scene.batches[1].instances[0].model, [0, 0, 0])).toEqual([2.4000000953674316, 0, 0]);
      expect(scene.batches[2].instances[0].color[3]).toBe(0);
      expect(scene.batches[2].instances[0].normalMatrix[0]).toBeCloseTo(1 / 80);
      expect(scene.batches[2].instances[0].normalMatrix[5]).toBe(1);
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
});
