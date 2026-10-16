import { describe, expect, it } from 'vitest';
import { packGeometryStreamSelection } from '../src/geometry-stream-selection-data';
import type { GeometryStreamDirectory } from '../src/geometry-stream-types';

const box = (min: [number, number, number], max: [number, number, number]) => ({ min, max });

function directory(): GeometryStreamDirectory {
  const left = box([0.25, 0, 0], [1, 1, 1]);
  const right = box([1, 0, 0], [2, 1, 1]);
  const joined = box([0, 0, 0], [2, 1, 1]);
  const rootLod = box([0, -1, -1], [2, 2, 2]);
  const separate = box([10, 0, 0], [11, 1, 1]);
  return {
    version: 1, primitiveKey: 'selection/fixture', sourceVertexCount: 12,
    sourceTriangleCount: 3, vertexStride: 80,
    clusters: [
      { level: 0, group: 1, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds: left, lodBounds: left, error: 0 },
      { level: 0, group: 1, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds: right, lodBounds: right, error: 0 },
      { level: 1, group: 2, refinedGroup: 1, vertexCount: 3, triangleCount: 1,
        bounds: joined, lodBounds: rootLod, error: 1.5 },
      { level: 0, group: 0, refinedGroup: null, vertexCount: 3, triangleCount: 1,
        bounds: separate, lodBounds: separate, error: 0 },
    ],
    groups: [
      { level: 0, children: [3], parents: [], bounds: separate, error: 0, terminal: true },
      { level: 0, children: [0, 1], parents: [2], bounds: joined, error: 1.5, terminal: false },
      { level: 1, children: [2], parents: [], bounds: rootLod, error: 1.5, terminal: true },
    ],
    roots: [2, 3],
    pages: [
      { id: 0, clusterIds: [0, 1], encodedBytes: 64, decodedBytes: 544, sha256: 'a'.repeat(64) },
      { id: 1, clusterIds: [2, 3], encodedBytes: 64, decodedBytes: 544, sha256: 'b'.repeat(64) },
    ],
    clusterPages: [0, 0, 1, 1], rootPages: [1],
  };
}

describe('stream selection metadata packing', () => {
  it('packs cluster float fields, pages, root bits, levels, and refined group IDs', () => {
    const packed = packGeometryStreamSelection(directory());
    const view = new DataView(packed.words.buffer);
    const f = (word: number) => view.getFloat32(word * 4, true);
    expect(packed.clusterOffset).toBe(0);
    expect(packed.groupOffset).toBe(4 * 16);
    expect(packed.edgeOffset).toBe(4 * 16 + 3 * 12);
    expect(packed.words.length).toBe(packed.edgeOffset + 5);
    expect([f(0), f(1), f(2), f(4), f(5), f(6)]).toEqual([0.25, 0, 0, 1, 1, 1]);
    expect([f(2 * 16 + 8), f(2 * 16 + 9), f(2 * 16 + 10),
      f(2 * 16 + 12), f(2 * 16 + 13), f(2 * 16 + 14)])
      .toEqual([0, -1, -1, 2, 2, 2]);
    expect([0, 1, 2, 3].map(id => packed.words[id * 16 + 3])).toEqual([0, 0, 1, 1]);
    expect([0, 1, 2, 3].map(id => packed.words[id * 16 + 7])).toEqual([0, 0, 1, 1]);
    expect([0, 1, 2, 3].map(id => packed.words[id * 16 + 11])).toEqual([0, 0, 1, 0]);
    expect([0, 1, 2, 3].map(id => packed.words[id * 16 + 15]))
      .toEqual([0xffff_ffff, 0xffff_ffff, 1, 0xffff_ffff]);
  });

  it('sorts every group by descending level then ID and packs parent/child edge spans', () => {
    const packed = packGeometryStreamSelection(directory());
    const view = new DataView(packed.words.buffer);
    expect(packed.groupIds).toEqual([2, 0, 1]);
    expect(packed.levels).toEqual([
      { level: 1, firstGroup: 0, groupCount: 1 },
      { level: 0, firstGroup: 1, groupCount: 2 },
    ]);
    const record = (ordinal: number) => Array.from(packed.words.subarray(
      packed.groupOffset + ordinal * 12, packed.groupOffset + (ordinal + 1) * 12));
    const top = record(0), terminal = record(1), refining = record(2);
    expect([view.getFloat32((packed.groupOffset + 2 * 12) * 4, true),
      view.getFloat32((packed.groupOffset + 2 * 12 + 3) * 4, true),
      view.getFloat32((packed.groupOffset + 2 * 12 + 4) * 4, true)])
      .toEqual([0, 1.5, 2]);
    expect(top.slice(7)).toEqual([1, 0, 0, 0, 1]);
    expect(terminal.slice(7)).toEqual([1, 1, 0, 1, 1]);
    expect(refining.slice(7)).toEqual([0, 2, 1, 3, 2]);
    expect(Array.from(packed.words.subarray(packed.edgeOffset))).toEqual([2, 3, 2, 0, 1]);
  });

  it('takes an immutable directory snapshot and produces deterministic bytes', () => {
    const source = directory();
    const first = packGeometryStreamSelection(source);
    const second = packGeometryStreamSelection(source);
    expect(first.words).toEqual(second.words);
    expect(first.groupIds).toEqual(second.groupIds);
    (source.roots as number[])[0] = 0;
    (source.groups[1].children as number[])[0] = 3;
    expect(first.directory.roots).toEqual([2, 3]);
    expect(first.directory.groups[1].children).toEqual([0, 1]);
    expect(Object.isFrozen(first.directory)).toBe(true);
    expect(Object.isFrozen(first.directory.groups[1].children)).toBe(true);
  });

  it('rejects malformed directories before allocating packed records', () => {
    const missingRoot = directory();
    (missingRoot.roots as number[]).pop();
    expect(() => packGeometryStreamSelection(missingRoot)).toThrow('ROOT_COVERAGE');
    const badPage = directory();
    (badPage.clusterPages as number[])[0] = 1;
    expect(() => packGeometryStreamSelection(badPage)).toThrow('PAGE_OWNERSHIP');
  });
});
