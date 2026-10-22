import { describe, expect, it } from 'vitest';
import { AnimationRig } from '../src/animation';
import { parseGltfNodeLayout } from '../src/gltf-node-layout';
import { transformPoint } from '../src/math';

const translation = (x: number, y: number, z: number): number[] => [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1,
];

const parse = (doc: Record<string, unknown>, meshWeights: readonly (readonly number[])[]) =>
  parseGltfNodeLayout({ asset: { version: '2.0' }, ...doc }, meshWeights);

describe('glTF node layout without geometry or binary buffers', () => {
  it('keeps nested TRS and matrix transforms, skin references, weight overrides, and selected scenes', () => {
    const meshWeights = [[0.2, 0.8], []] as const;
    const doc = {
      asset: { version: '2.0' }, meshes: [{}, {}], skins: [{}],
      nodes: [
        { name: 'root', translation: [2, 3, 0], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
          scale: [2, 1, 1], children: [1] },
        { name: 'child', mesh: 0, skin: 0, weights: [0.75, 0.25], matrix: translation(4, 0, 0) },
        { mesh: 0, translation: [-3, 0, 0] },
      ],
      scenes: [{ name: 'first', nodes: [0] }, { name: 'second', nodes: [2] }], scene: 1,
    };
    const before = JSON.stringify(doc), weightsBefore = JSON.stringify(meshWeights);
    const layout = parse(doc, meshWeights);
    expect(layout.nodes).toEqual([
      { name: 'root', parent: null, mesh: undefined, skin: undefined },
      { name: 'child', parent: 0, mesh: 0, skin: 0 },
      { name: 'node-2', parent: null, mesh: 0, skin: undefined },
    ]);
    expect(layout.scenes).toEqual([{ name: 'first', roots: [0] }, { name: 'second', roots: [2] }]);
    expect(layout.scene).toBe(1);
    expect(layout.rigNodes[1].weights).toEqual([0.75, 0.25]);
    expect(layout.rigNodes[2].weights).toEqual([0.2, 0.8]);
    const pose = new AnimationRig(layout.rigNodes, []).sample(0);
    // Child local (1,0,0) -> matrix (5,0,0) -> root scale (10,0,0)
    // -> quarter turn (0,10,0) -> translation (2,13,0).
    const point = transformPoint(pose.world[1], [1, 0, 0]);
    point.forEach((value, axis) => expect(value).toBeCloseTo([2, 13, 0][axis]));
    expect(transformPoint(pose.world[2], [0, 0, 0])).toEqual([-3, 0, 0]);
    expect(JSON.stringify(doc)).toBe(before);
    expect(JSON.stringify(meshWeights)).toBe(weightsBefore);
    expect(layout.rigNodes[1].weights).not.toBe(doc.nodes[1].weights);
    expect(layout.rigNodes[2].weights).not.toBe(meshWeights[0]);
  });

  it('synthesizes a default scene from all parentless nodes', () => {
    const layout = parse({ meshes: [{}], nodes: [
      { children: [1] }, { mesh: 0 }, { translation: [0, 4, 0] },
    ] }, [[]]);
    expect(layout.scenes).toEqual([{ name: 'implicit-scene', roots: [0, 2] }]);
    expect(layout.scene).toBe(0);
    expect(layout.rigNodes.map(node => node.parent)).toEqual([null, 0, null]);
    expect(layout.rigNodes.map(node => node.weights)).toEqual([[], [], []]);
  });

  it('rejects ambiguous hierarchy, scene roots, transform modes, and invalid references', () => {
    expect(() => parse({ nodes: [{ children: [2] }, { children: [2] }, {}] }, []))
      .toThrow(/GLTF_MULTIPLE_PARENTS/);
    expect(() => parse({ nodes: [{ children: [1] }, {}], scenes: [{ nodes: [1] }] }, []))
      .toThrow(/GLTF_SCENE_ROOTS/);
    expect(() => parse({ nodes: [{}], scenes: [{ nodes: [1] }] }, []))
      .toThrow(/GLTF_INDEX.*scene.node/);
    expect(() => parse({ nodes: [{ matrix: translation(0, 0, 0), scale: [1, 1, 1] }] }, []))
      .toThrow(/GLTF_MATRIX_WITH_TRS/);
    expect(() => parse({ meshes: [{}], nodes: [{ mesh: 1 }] }, [[]]))
      .toThrow(/GLTF_INDEX.*node.mesh/);
    expect(() => parse({ nodes: [{ skin: 0 }] }, []))
      .toThrow(/GLTF_INDEX.*node.skin/);
    expect(() => parse({ nodes: [{ children: [1] }] }, []))
      .toThrow(/GLTF_INDEX.*child/);
    expect(() => parse({ scenes: [{}], scene: 1 }, []))
      .toThrow(/GLTF_INDEX.*scene/);
    expect(() => parse({ meshes: [{}] }, []))
      .toThrow(/GLTF_MESH_WEIGHTS_COUNT/);
    expect(() => parse({ meshes: [{}], nodes: [{ mesh: 0, weights: [1] }] }, [[0.3, 0.7]]))
      .toThrow(/GLTF_VECTOR.*node.weights/);
    expect(() => parse({ nodes: Array.from({ length: 4097 }, () => ({})) }, []))
      .toThrow(/GLTF_ARRAY_LIMIT.*nodes/);
  });

  it('rejects cycles before scene roots and leaves non-affine matrix validation to AnimationRig', () => {
    expect(() => parse({ nodes: [{ children: [1] }, { children: [0] }], scenes: [{ nodes: [0] }] }, []))
      .toThrow(/cycle/i);
    const projective = translation(0, 0, 0);
    projective[3] = 0.5;
    const layout = parse({ nodes: [{ matrix: projective }] }, []);
    expect(() => new AnimationRig(layout.rigNodes, [])).toThrow(/affine/i);
  });

  it('requires a glTF 2.0 asset declaration at the standalone entry point', () => {
    expect(() => parseGltfNodeLayout({ nodes: [] }, [])).toThrow(/GLTF_OBJECT: asset/);
    for (const asset of [{ version: '1.0' }, { version: '2.1' },
      { version: '2.0', minVersion: '1.0' }, { version: '2.0', minVersion: '2.1' }]) {
      expect(() => parseGltfNodeLayout({ asset, nodes: [] }, [])).toThrow(/GLTF_VERSION/);
    }
    expect(parseGltfNodeLayout({ asset: { version: '2.0', minVersion: '2.0' } }, []).scene).toBe(0);
  });

  it('bounds and validates each supplied mesh weight array before constructing nodes', () => {
    const doc = { meshes: [{}], nodes: [{ mesh: 0 }] };
    expect(parse(doc, [Array(32).fill(0.25)]).rigNodes[0].weights).toHaveLength(32);
    for (const invalid of [null, {}, new Float32Array([0.5]), Array(33).fill(0)]) {
      expect(() => parse(doc, [invalid] as unknown as readonly (readonly number[])[]))
        .toThrow(/GLTF_ARRAY_LIMIT.*mesh.weights/);
    }
    for (const invalid of [[NaN], [Infinity], [-Infinity], [1e39], [undefined]]) {
      expect(() => parse(doc, [invalid] as unknown as readonly (readonly number[])[]))
        .toThrow(/GLTF_NUMBER.*mesh.weights/);
    }
  });
});
