import { describe, expect, it } from 'vitest';
import { GeometryDemoScene } from '../src/geometry-demo-scene';
import { packGeometryLod } from '../src/geometry-lod-data';

describe('geometry LOD demo scene', () => {
  it('builds a valid reduced hierarchy with outward-facing sculpture triangles', async () => {
    const scene = await GeometryDemoScene.create();
    const hierarchy = scene.hierarchy;
    const packed = packGeometryLod(hierarchy);
    const source = hierarchy.source;
    const sourceTriangles = source.indices.length / 3;
    const leafTriangles = hierarchy.clusters.filter(cluster => cluster.level === 0)
      .reduce((sum, cluster) => sum + cluster.triangles.length / 3, 0);
    const rootTriangles = hierarchy.roots.reduce((sum, id) => sum + hierarchy.clusters[id].triangles.length / 3, 0);

    expect(sourceTriangles).toBe(scene.sourceTriangles);
    expect(sourceTriangles).toBeGreaterThan(1000);
    expect(sourceTriangles).toBeLessThan(10_000);
    expect(leafTriangles).toBe(sourceTriangles);
    expect(hierarchy.clusters.some(cluster => cluster.level >= 2)).toBe(true);
    expect(rootTriangles).toBeLessThan(sourceTriangles);
    expect(packed.sourceTriangleCount).toBe(sourceTriangles);
    expect(packed.clusterCount).toBe(hierarchy.clusters.length);
    expect(packed.indices.length).toBeGreaterThan(source.indices.length);

    const view = new DataView(source.vertices);
    const xyz = (vertex: number, offset: number): [number, number, number] => [0, 1, 2]
      .map(axis => view.getFloat32(vertex * 80 + offset + axis * 4, true)) as [number, number, number];
    for (let index = 0; index < source.indices.length; index += 3) {
      const ids = [source.indices[index], source.indices[index + 1], source.indices[index + 2]];
      const [a, b, c] = ids.map(id => xyz(id, 0));
      const ab = b.map((value, axis) => value - a[axis]);
      const ac = c.map((value, axis) => value - a[axis]);
      const cross = [ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0]];
      const meanNormal = [0, 1, 2].map(axis => ids.reduce((sum, id) => sum + xyz(id, 16)[axis], 0));
      expect(cross.reduce((sum, value, axis) => sum + value * meanNormal[axis], 0)).toBeGreaterThan(0);
    }

    const sculpture = scene.batches.find(batch => batch.name === 'sculpture');
    expect(sculpture).toBeDefined();
    expect(sculpture!.mesh).toBe(hierarchy.source);
    expect(sculpture!.geometryLod).toBe(hierarchy);
    expect(sculpture!.capacity).toBe(scene.instanceCount);
    expect(sculpture!.inputMode).toBe('committed');
    expect(sculpture!.instances).toHaveLength(3);
    expect(new Set(sculpture!.instances.map(instance => instance.entityId)).size).toBe(3);
    expect(sculpture!.deformation).toEqual({ jointCount: 0, nodes: [] });
    expect(sculpture!.morphTargets).toEqual([]);
    for (const instance of sculpture!.instances) {
      expect(instance.bones).toEqual([]);
      expect(instance.deformScalars).toEqual([]);
      expect(instance.historyVersion).toBe(0);
      expect(instance.model.length).toBe(16);
      expect([...instance.model, ...instance.normalMatrix].every(Number.isFinite)).toBe(true);
    }
  }, 30_000);
});
