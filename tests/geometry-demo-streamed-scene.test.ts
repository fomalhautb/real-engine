import { describe, expect, it } from 'vitest';
import { StreamedGalleryScene, STREAMED_GALLERY_TRAJECTORIES, streamedGalleryIdentityColor, streamedGalleryMesh } from '../src/geometry-demo-streamed-scene';

describe('streamed gallery scene', () => {
  it('generates closed, consistently oriented and seed-unique sculptures', () => {
    const mesh = streamedGalleryMesh(3, 32, 12), again = streamedGalleryMesh(3, 32, 12), other = streamedGalleryMesh(4, 32, 12);
    expect(new Uint8Array(mesh.vertices)).toEqual(new Uint8Array(again.vertices));
    expect(new Uint8Array(mesh.vertices)).not.toEqual(new Uint8Array(other.vertices));
    // Every directed edge appears once and its reverse once: watertight 2-manifold.
    const edges = new Map<string, number>();
    for (let i = 0; i < mesh.indices.length; i += 3) for (let k = 0; k < 3; k++) {
      const key = `${mesh.indices[i + k]}>${mesh.indices[i + (k + 1) % 3]}`;
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
    for (const [key, count] of edges) {
      const [a, b] = key.split('>');
      expect(count).toBe(1);
      expect(edges.get(`${b}>${a}`)).toBe(1);
    }
    expect(mesh.indices.length / 3).toBe(32 * 12 * 2);
    expect(() => streamedGalleryMesh(1, 4, 12)).toThrow('STREAMED_GALLERY_MESH_RESOLUTION');
  });

  it('plans a runtime budget whose geometry pool is oversubscribed by unique resident-format data', async () => {
    const scene = await StreamedGalleryScene.create({ objects: 3, around: 96, across: 24 });
    const a = scene.accounting;
    expect(a.objects).toBe(3);
    expect(a.uniqueToPool).toBeGreaterThanOrEqual(4);
    expect(a.poolBudgetBytes).toBe(Math.floor(a.uniqueResidentBytes / 4));
    // Any slot split inside maxResidentBytes keeps the pool within its bound.
    expect(a.uniqueToResidentBudget).toBeLessThan(a.uniqueToPool);
    expect(a.uniqueToPeakBudget).toBeLessThan(a.uniqueToResidentBudget);
    expect(a.slotPaddedBytes).toBeGreaterThan(a.uniqueResidentBytes);
    expect(a.maxResidentBytes).toBeGreaterThan(a.rootBytes);
    expect(a.rootPoolBytes).toBeLessThan(a.poolBudgetBytes);
    expect(a.maxPeakBytes).toBeGreaterThanOrEqual(a.maxResidentBytes);
    expect(a.fullResidencyBytes).toBeGreaterThan(a.maxResidentBytes);
    expect(scene.runtimeBudget).toMatchObject({ maxResidentBytes: a.maxResidentBytes, maxPeakBytes: a.maxPeakBytes, intervalFrames: 30 });
    // Streamed sculptures are source-free with fixed proxies; the reference keeps exact meshes.
    const streamed = scene.batches.filter(batch => batch.geometryStream);
    expect(streamed.map(batch => batch.name)).toEqual(scene.streamBatchNames);
    expect(streamed.every(batch => !batch.mesh && batch.geometryStream!.queryProxy)).toBe(true);
    expect(scene.referenceBatches.every(batch => !batch.geometryStream && batch.mesh)).toBe(true);
    expect(new Set(scene.objects.map(object => object.primitiveKey)).size).toBe(3);
    const page = scene.objects[0].directory.pages[0];
    const bytes = await streamed[0].geometryStream!.provider(page, new AbortController().signal);
    expect(bytes.byteLength).toBe(page.encodedBytes);
  });

  it('rejects unsatisfiable or unbounded configurations', async () => {
    await expect(StreamedGalleryScene.create({ objects: 0 })).rejects.toThrow('STREAMED_GALLERY_OBJECTS');
    await expect(StreamedGalleryScene.create({ objects: 1, providerLatencyMs: 5000 })).rejects.toThrow('STREAMED_GALLERY_LATENCY');
    await expect(StreamedGalleryScene.create({ objects: 1, targetUniqueToPoolRatio: 0.5 })).rejects.toThrow('STREAMED_GALLERY_RATIO');
    // Coarse meshes cannot oversubscribe a budget that must still hold every root.
    await expect(StreamedGalleryScene.create({ objects: 1, around: 16, across: 8, targetUniqueToPoolRatio: 64 }))
      .rejects.toThrow('STREAMED_GALLERY_ROOTS_EXCEED_BUDGET');
  });

  it('defines deterministic finite trajectories with explicit teleport discontinuities', async () => {
    const scene = await StreamedGalleryScene.create({ objects: 9, around: 64, across: 16, targetUniqueToPoolRatio: 1 });
    expect(scene.rooms).toBe(2);
    for (const trajectory of STREAMED_GALLERY_TRAJECTORIES) for (const t of [0, 0.5, 3.9, 4, 7.99, 12.5, 30]) {
      const pose = scene.pose(trajectory, t);
      expect([...pose.eye, ...pose.target].every(Number.isFinite)).toBe(true);
      expect(scene.pose(trajectory, t)).toEqual(pose);
    }
    expect(scene.pose('teleport', 2.9).discontinuity).toBe(false);
    expect(scene.pose('teleport', 3).discontinuity).toBe(true);
    expect(scene.pose('teleport', 3).eye[2]).toBeGreaterThan(scene.pose('teleport', 2.9).eye[2] + 5);
    // The fast turn completes 180 degrees within 0.25 s.
    const before = scene.pose('turn', 3.75), after = scene.pose('turn', 4);
    const forward = (pose: typeof before) => Math.sign(pose.target[2] - pose.eye[2]);
    expect(forward(before)).toBe(1);
    expect(forward(after)).toBe(-1);
    expect(() => scene.pose('walk', -1)).toThrow('STREAMED_GALLERY_TIME');
    // The walk reaches its far-wall hold exactly at walkSeconds.
    expect(scene.pose('walk', scene.walkSeconds).eye[2]).toBeCloseTo(scene.rooms * 12 - 2, 9);
    expect(scene.pose('walk', scene.walkSeconds - 0.5).eye[2]).toBeLessThan(scene.rooms * 12 - 2);
    expect(scene.pose('walk', scene.walkSeconds + 5).eye[2]).toBeCloseTo(scene.rooms * 12 - 2, 9);
    const colors = new Set(Array.from({ length: 48 }, (_, index) => streamedGalleryIdentityColor(index).join(',')));
    expect(colors.size).toBe(48);
  });
});
