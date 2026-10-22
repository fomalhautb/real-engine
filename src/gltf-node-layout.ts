import type { AnimationNode } from './animation';
import { identity } from './math';

type Json = Record<string, unknown>;

export interface GltfNodeLayout {
  nodes: { mesh?: number; skin?: number; name: string; parent: number | null }[];
  rigNodes: AnimationNode[];
  scenes: { name: string; roots: number[] }[];
  scene: number;
}

function object(value: unknown, label: string): Json {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('GLTF_OBJECT: ' + label);
  return value as Json;
}
function array(value: unknown, label: string, limit = 65536): unknown[] {
  if (!Array.isArray(value) || value.length > limit) throw new Error('GLTF_ARRAY_LIMIT: ' + label);
  return value;
}
function index(value: unknown, length: number, label: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value >= length) throw new Error('GLTF_INDEX: ' + label);
  return value;
}
function optionalIndex(value: unknown, list: readonly unknown[], label: string): number | undefined {
  return value === undefined ? undefined : index(value, list.length, label);
}
function number(value: unknown, fallback: number, label: string): number {
  if (value === undefined) return fallback;
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value))) throw new Error('GLTF_NUMBER: ' + label);
  return value;
}
function vector(value: unknown, defaults: readonly number[], label: string): number[] {
  if (value === undefined) return [...defaults];
  const values = array(value, label, defaults.length);
  if (values.length !== defaults.length) throw new Error('GLTF_VECTOR: ' + label);
  return values.map(v => number(v, 0, label));
}

/** Reads scene and animation node metadata without decoding mesh, image, or binary data.
 * AnimationRig validates transforms; this parser also rejects cycles before checking scene roots. */
export function parseGltfNodeLayout(json: unknown, meshWeights: readonly (readonly number[])[]): GltfNodeLayout {
  const doc = object(json, 'root');
  const header = object(doc.asset, 'asset');
  if (header.version !== '2.0' || (header.minVersion !== undefined && header.minVersion !== '2.0')) throw new Error('GLTF_VERSION: requires 2.0');
  const meshJson = array(doc.meshes ?? [], 'meshes');
  if (!Array.isArray(meshWeights) || meshWeights.length !== meshJson.length) throw new Error('GLTF_MESH_WEIGHTS_COUNT');
  for (const weights of meshWeights) {
    for (const weight of array(weights, 'mesh.weights', 32)) {
      if (typeof weight !== 'number' || !Number.isFinite(Math.fround(weight))) throw new Error('GLTF_NUMBER: mesh.weights');
    }
  }
  const nodeJson = array(doc.nodes ?? [], 'nodes', 4096).map((node, i) => object(node, 'node' + i));
  const skinJson = array(doc.skins ?? [], 'skins');
  const parents: (number | null)[] = new Array(nodeJson.length).fill(null);
  for (let parent = 0; parent < nodeJson.length; parent++) for (const childValue of array(nodeJson[parent].children ?? [], 'children')) {
    const child = index(childValue, nodeJson.length, 'child');
    if (child === parent) throw new RangeError(`node ${child} has an invalid parent`);
    if (parents[child] !== null) throw new Error('GLTF_MULTIPLE_PARENTS');
    parents[child] = parent;
  }
  // Scene roots cannot be diagnosed reliably until the parent links form a forest.
  // Each node changes state at most twice, so even a deep hierarchy takes linear work.
  const visited = new Uint8Array(parents.length);
  for (let start = 0; start < parents.length; start++) {
    let cursor: number | null = start;
    while (cursor !== null && visited[cursor] === 0) { visited[cursor] = 1; cursor = parents[cursor]; }
    if (cursor !== null && visited[cursor] === 1) throw new RangeError('animation node hierarchy contains a cycle');
    cursor = start;
    while (cursor !== null && visited[cursor] === 1) { visited[cursor] = 2; cursor = parents[cursor]; }
  }
  const nodes = nodeJson.map((node, i) => ({ parent: parents[i], mesh: optionalIndex(node.mesh, meshWeights, 'node.mesh'), skin: optionalIndex(node.skin, skinJson, 'node.skin'),
    name: typeof node.name === 'string' ? node.name : 'node-' + i }));
  const rigNodes: AnimationNode[] = nodeJson.map((node, i) => {
    if (node.matrix !== undefined && [node.translation, node.rotation, node.scale].some(value => value !== undefined)) throw new Error('GLTF_MATRIX_WITH_TRS');
    const meshIndex = nodes[i].mesh;
    if (nodes[i].skin !== undefined && meshIndex === undefined) throw new Error('GLTF_SKIN_REQUIRES_MESH');
    if (node.weights !== undefined && meshIndex === undefined) throw new Error('GLTF_NODE_WEIGHTS_REQUIRES_MESH');
    return { parent: parents[i], translation: vector(node.translation, [0, 0, 0], 'translation') as [number, number, number],
      rotation: vector(node.rotation, [0, 0, 0, 1], 'rotation') as [number, number, number, number],
      scale: vector(node.scale, [1, 1, 1], 'scale') as [number, number, number],
      matrix: node.matrix === undefined ? undefined : new Float32Array(vector(node.matrix, Array.from(identity()), 'matrix')),
      weights: vector(node.weights, meshIndex === undefined ? [] : meshWeights[meshIndex], 'node.weights') };
  });
  const scenes = array(doc.scenes ?? [], 'scenes').map((value, i) => {
    const source = object(value, 'scene'), roots = array(source.nodes ?? [], 'scene.nodes').map(node => index(node, nodes.length, 'scene.node'));
    if (roots.some(root => parents[root] !== null) || new Set(roots).size !== roots.length) throw new Error('GLTF_SCENE_ROOTS');
    return { name: typeof source.name === 'string' ? source.name : 'scene-' + i, roots };
  });
  if (!scenes.length) scenes.push({ name: 'implicit-scene', roots: parents.flatMap((parent, i) => parent === null ? [i] : []) });
  return { nodes, rigNodes, scenes, scene: doc.scene === undefined ? 0 : index(doc.scene, scenes.length, 'scene') };
}
