import type { MeshoptTangents } from 'meshoptimizer/tangents';
import type { MorphTarget } from './deformation';
import type { MeshData } from './geometry';

type Generator = typeof MeshoptTangents;
let generator: Generator | undefined;
let preparation: Promise<void> | undefined;
let generatorFailure: Error | undefined;

/** Lazy shared WASM initialization; synchronous importers must await this before generating tangents. */
export function prepareGltfTangents(): Promise<void> {
  if (generatorFailure) return Promise.reject(generatorFailure);
  return preparation ??= (async () => {
    try {
      const { MeshoptTangents: api } = await import('meshoptimizer/tangents');
      if (!api.supported) throw new Error('WebAssembly is unavailable');
      await api.ready;
      generator = api;
    } catch (cause) {
      generatorFailure = new Error('GLTF_TANGENT_GENERATOR_INITIALIZATION_FAILED', { cause });
      throw generatorFailure;
    }
  })();
}

export interface GltfTangentOptions {
  /** Remaining cumulative output vertex allowance, including seam duplicates. */
  maxVertices: number;
  /** Conservative preparation-work byte estimate, excluding input assets and JS/WASM runtime overhead. */
  maxTangentBytes: number;
}

const EMPTY = 0xffffffff;
const MAX_HASH_PROBES = 256;
const MAX_ADJACENCY_PAIRS = 32_000_000;

function tableSize(count: number): number {
  let size = 1;
  while (size < count * 2) size *= 2;
  return size;
}

function mix(hash: number, word: number): number {
  // Canonicalize -0 without rounding or merging distinct finite float values.
  return Math.imul(hash ^ (word === 0x80000000 ? 0 : word), 16777619) >>> 0;
}

/** Translation and one positive uniform scale preserve the tangent construction.
 * Packing in a local unit box avoids overflow/underflow for uniformly huge/tiny assets. */
function normalizeCoordinates(values: Float32Array, components: number): void {
  const min = new Array<number>(components).fill(Infinity);
  const max = new Array<number>(components).fill(-Infinity);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error('GLTF_TANGENT_NONFINITE_INPUT');
    const axis = i % components;
    min[axis] = Math.min(min[axis], values[i]);
    max[axis] = Math.max(max[axis], values[i]);
  }
  const extent = Math.max(...max.map((value, axis) => value - min[axis])) || 1;
  for (let i = 0; i < values.length; i++) values[i] = (values[i] - min[i % components]) / extent;
}

/** Bound the library's quadratic adjacency merge using its position/normal/UV equivalence.
 * The hash scratch is typed and every collision chain also has a finite work limit. */
function adjacencyPairs(positions: Float32Array, normals: Float32Array, uvs: Float32Array,
  indices: Uint32Array, table: Uint32Array, groups: Uint32Array, counts: Uint32Array,
  remainingPairs: number): number {
  const p = new Uint32Array(positions.buffer), n = new Uint32Array(normals.buffer), uv = new Uint32Array(uvs.buffer);
  table.fill(EMPTY); counts.fill(0);
  for (let vertex = 0; vertex < groups.length; vertex++) {
    let hash = 2166136261;
    for (let axis = 0; axis < 3; axis++) hash = mix(mix(hash, p[vertex * 3 + axis]), n[vertex * 3 + axis]);
    hash = mix(mix(hash, uv[vertex * 2]), uv[vertex * 2 + 1]);
    let bucket = hash & (table.length - 1), found = false;
    for (let probe = 0; probe < MAX_HASH_PROBES; probe++) {
      const candidate = table[bucket];
      if (candidate === EMPTY) { table[bucket] = vertex; groups[vertex] = vertex; found = true; break; }
      let equal = true;
      for (let axis = 0; axis < 3; axis++) {
        equal &&= positions[vertex * 3 + axis] === positions[candidate * 3 + axis]
          && normals[vertex * 3 + axis] === normals[candidate * 3 + axis];
      }
      equal &&= uvs[vertex * 2] === uvs[candidate * 2] && uvs[vertex * 2 + 1] === uvs[candidate * 2 + 1];
      if (equal) { groups[vertex] = candidate; found = true; break; }
      bucket = (bucket + probe + 1) & (table.length - 1);
    }
    if (!found) throw new Error('GLTF_TANGENT_HASH_WORK_BUDGET');
  }
  let pairs = 0;
  for (const vertex of indices) {
    pairs += counts[groups[vertex]]++;
    if (pairs > remainingPairs) throw new Error('GLTF_TANGENT_ADJACENCY_WORK_BUDGET');
  }
  return pairs;
}

/** Generate MikkTSpace-compatible corner frames, splitting only incompatible source vertices.
 * Base and every Morph target participate in the split key; source bytes are never mutated.
 * This is asset preparation, not a per-frame deformation or readback path. */
export function generateGltfTangents(mesh: MeshData, morphTargets: readonly MorphTarget[],
  options: GltfTangentOptions): { mesh: MeshData; morphTargets: MorphTarget[] } {
  if (!Number.isSafeInteger(options.maxVertices) || options.maxVertices < 1 || options.maxVertices > 10_000_000) {
    throw new Error('GLTF_VERTEX_BUDGET');
  }
  if (!Number.isSafeInteger(options.maxTangentBytes) || options.maxTangentBytes < 1 || options.maxTangentBytes > 512 * 1024 * 1024) {
    throw new Error('GLTF_TANGENT_WORK_BUDGET');
  }
  const vertices = mesh.vertexCount, corners = mesh.indices.length, targets = morphTargets.length;
  if (!Number.isSafeInteger(vertices) || vertices < 1 || vertices > 10_000_000
    || !(mesh.vertices instanceof ArrayBuffer) || mesh.vertices.byteLength !== vertices * 80
    || !(mesh.indices instanceof Uint32Array) || !corners || corners % 3 || corners > 30_000_000
    || targets > 32) throw new Error('GLTF_TANGENT_LAYOUT');
  if (!(mesh.uvs instanceof Float32Array) || mesh.uvs.length !== vertices * 2) {
    throw new Error('GLTF_NORMAL_TEXTURE_REQUIRES_UV0');
  }
  if (mesh.colors && (!(mesh.colors instanceof Float32Array) || mesh.colors.length !== vertices * 4
    || !mesh.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1))) {
    throw new Error('GLTF_COLOR_LAYOUT');
  }
  // Includes retained corner frames, packed inputs, typed hash/remap scratch, worst-case
  // output streams and conservative library workspace. It is not a process-memory ceiling:
  // the shared WASM module retains its high-water heap and exposes no shrinking API.
  const vertexTableSize = tableSize(vertices), cornerTableSize = tableSize(corners);
  const workBytes = 256 * 1024 + vertices * (96 + (targets ? 24 : 0) + (mesh.colors ? 16 : 0)) + vertexTableSize * 8
    + corners * (72 + 16 * (targets + 1)) + cornerTableSize * 4
    + Math.min(corners, options.maxVertices) * (104 + 36 * targets + (mesh.colors ? 16 : 0));
  if (workBytes > options.maxTangentBytes) throw new Error('GLTF_TANGENT_WORK_BUDGET');
  if (!generator) throw generatorFailure ?? new Error('GLTF_TANGENT_GENERATOR_NOT_READY: await prepareGltfTangents()');
  for (const index of mesh.indices) if (index >= vertices) throw new Error('GLTF_TANGENT_INDEX');
  for (const target of morphTargets) {
    for (const stream of [target.positions, target.normals, target.tangents]) {
      if (stream !== undefined && (!(stream instanceof Float32Array) || stream.length !== vertices * 3
        || !stream.every(Number.isFinite))) throw new Error('GLTF_TANGENT_MORPH_LAYOUT');
    }
    if (!target.positions || !target.normals) throw new Error('GLTF_TANGENT_MORPH_LAYOUT');
  }

  const source = new DataView(mesh.vertices), positions = new Float32Array(vertices * 3);
  const normals = new Float32Array(vertices * 3), uvs = new Float32Array(mesh.uvs);
  const groupTable = new Uint32Array(vertexTableSize), groups = new Uint32Array(vertices), counts = new Uint32Array(vertices);
  normalizeCoordinates(uvs, 2);
  let pairTotal = 0;
  const frames: Float32Array[] = [];
  // Reuse target input scratch. Each generated output is retained for joint seam remapping.
  for (let variant = 0; variant <= targets; variant++) {
    const target = variant ? morphTargets[variant - 1] : undefined;
    for (let vertex = 0; vertex < vertices; vertex++) {
      for (let axis = 0; axis < 3; axis++) {
        const at = vertex * 3 + axis;
        positions[at] = source.getFloat32(vertex * 80 + axis * 4, true) + (target?.positions[at] ?? 0);
        normals[at] = source.getFloat32(vertex * 80 + 16 + axis * 4, true) + (target?.normals[at] ?? 0);
      }
      const at = vertex * 3, length = Math.hypot(normals[at], normals[at + 1], normals[at + 2]);
      if (!(length > 0) || !Number.isFinite(length)) throw new Error('GLTF_TANGENT_DEGENERATE_NORMAL');
      for (let axis = 0; axis < 3; axis++) normals[at + axis] /= length;
    }
    normalizeCoordinates(positions, 3);
    pairTotal += adjacencyPairs(positions, normals, uvs, mesh.indices, groupTable, groups, counts, MAX_ADJACENCY_PAIRS - pairTotal);
    let frame: Float32Array;
    try {
      frame = generator.generateTangents(mesh.indices, positions, 3, normals, 3, uvs, 2, ['Compatible']);
    } catch (cause) {
      // The upstream wrapper cannot rewind its temporary WASM arena after a trap.
      // Refuse reuse rather than repeatedly consuming an uncertain arena.
      generator = undefined;
      generatorFailure = new Error('GLTF_TANGENT_GENERATION_FAILED', { cause });
      throw generatorFailure;
    }
    if (frame.length !== corners * 4 || !frame.every(Number.isFinite)) throw new Error('GLTF_TANGENT_INVALID_RESULT');
    for (let corner = 0; corner < corners; corner++) {
      const at = corner * 4, handedness = frame[at + 3];
      if ((handedness !== 1 && handedness !== -1) || Math.abs(Math.hypot(frame[at], frame[at + 1], frame[at + 2]) - 1) > 1e-3) {
        throw new Error('GLTF_TANGENT_INVALID_RESULT');
      }
      if (variant && handedness !== frames[0][at + 3]) throw new Error('GLTF_TANGENT_HANDEDNESS_CHANGE');
    }
    frames.push(frame);
  }

  const bits = frames.map(frame => new Uint32Array(frame.buffer, frame.byteOffset, frame.length));
  const table = new Uint32Array(cornerTableSize).fill(EMPTY);
  const representatives = new Uint32Array(Math.min(corners, options.maxVertices));
  const indices = new Uint32Array(corners);
  let outputCount = 0;
  for (let corner = 0; corner < corners; corner++) {
    let hash = mix(2166136261, mesh.indices[corner]);
    for (const frame of bits) for (let axis = 0; axis < 4; axis++) hash = mix(hash, frame[corner * 4 + axis]);
    let bucket = hash & (table.length - 1), found = false;
    for (let probe = 0; probe < MAX_HASH_PROBES; probe++) {
      const candidate = table[bucket];
      if (candidate === EMPTY) {
        if (outputCount >= options.maxVertices) throw new Error('GLTF_VERTEX_BUDGET');
        table[bucket] = corner; representatives[outputCount] = corner;
        indices[corner] = outputCount++; found = true; break;
      }
      let equal = mesh.indices[corner] === mesh.indices[candidate];
      for (const frame of frames) for (let axis = 0; axis < 4; axis++) {
        equal &&= frame[corner * 4 + axis] === frame[candidate * 4 + axis];
      }
      if (equal) { indices[corner] = indices[candidate]; found = true; break; }
      bucket = (bucket + probe + 1) & (table.length - 1);
    }
    if (!found) throw new Error('GLTF_TANGENT_HASH_WORK_BUDGET');
  }

  const output = new Uint8Array(outputCount * 80), input = new Uint8Array(mesh.vertices);
  const outputUvs = new Float32Array(outputCount * 2), tangents = new Float32Array(outputCount * 4);
  const outputColors = mesh.colors ? new Float32Array(outputCount * 4) : undefined;
  const outputMorphs: MorphTarget[] = morphTargets.map(() => ({ positions: new Float32Array(outputCount * 3),
    normals: new Float32Array(outputCount * 3), tangents: new Float32Array(outputCount * 3) }));
  const bounds: MeshData['bounds'] = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (let vertex = 0; vertex < outputCount; vertex++) {
    const corner = representatives[vertex], sourceVertex = mesh.indices[corner];
    output.set(input.subarray(sourceVertex * 80, (sourceVertex + 1) * 80), vertex * 80);
    for (let axis = 0; axis < 2; axis++) outputUvs[vertex * 2 + axis] = mesh.uvs[sourceVertex * 2 + axis];
    if (outputColors) for (let channel = 0; channel < 4; channel++) outputColors[vertex * 4 + channel] = mesh.colors![sourceVertex * 4 + channel];
    for (let axis = 0; axis < 4; axis++) tangents[vertex * 4 + axis] = frames[0][corner * 4 + axis];
    for (let axis = 0; axis < 3; axis++) {
      const value = source.getFloat32(sourceVertex * 80 + axis * 4, true);
      bounds.min[axis] = Math.min(bounds.min[axis], value); bounds.max[axis] = Math.max(bounds.max[axis], value);
      for (let target = 0; target < targets; target++) {
        outputMorphs[target].positions[vertex * 3 + axis] = morphTargets[target].positions[sourceVertex * 3 + axis];
        outputMorphs[target].normals[vertex * 3 + axis] = morphTargets[target].normals[sourceVertex * 3 + axis];
        outputMorphs[target].tangents![vertex * 3 + axis] = frames[target + 1][corner * 4 + axis] - frames[0][corner * 4 + axis];
      }
    }
  }
  return { mesh: { vertices: output.buffer, vertexCount: outputCount, indices, bounds, uvs: outputUvs, colors: outputColors, tangents },
    morphTargets: outputMorphs };
}
