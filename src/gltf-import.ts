import { readAccessor, type GltfBinaryDocument } from './gltf-binary';
import { AnimationRig, type AnimationTrack, type AnimationClip } from './animation';
import type { MeshData } from './geometry';
import type { MorphTarget } from './deformation';
import type { MaterialBinding } from './contracts';
import { createGltfMaterialLibrary, type GltfMaterialOptions } from './gltf-material';
import { identity, inverseAffine, type Mat4 } from './math';
import { buildMorphTarget } from './morph-assets';
import { generateGltfTangents } from './gltf-tangents';
import { parseGltfNodeLayout } from './gltf-node-layout';

export { prepareGltfTangents } from './gltf-tangents';

type Json = Record<string, unknown>;
export interface GltfPrimitive { mesh: MeshData; morphTargets: MorphTarget[]; material: MaterialBinding; hasSkin: boolean }
export interface GltfMesh { name: string; primitives: GltfPrimitive[]; weights: number[] }
export interface GltfSkin { joints: number[]; inverseBindMatrices: Mat4[] }
export interface GltfAsset {
  rig: AnimationRig; meshes: GltfMesh[]; skins: GltfSkin[];
  nodes: { mesh?: number; skin?: number; name: string; parent: number | null }[];
  scenes: { name: string; roots: number[] }[]; scene: number;
  warnings: string[];
}
export interface GltfImportOptions extends GltfMaterialOptions {
  maxVertices?: number; maxIndices?: number;
  /** Conservative tangent preparation-work estimate per primitive, not a runtime RSS ceiling (default 256 MiB, hard limit 512 MiB). */
  maxTangentBytes?: number;
  /** Cumulative animation channels (default 65,536, matching AnimationRig). */
  maxAnimationTracks?: number;
  /** Cumulative timestamp and output scalar references, including reused accessors (default 16 million). */
  maxAnimationScalars?: number;
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
function number(value: unknown, fallback: number, label: string, min = -Infinity, max = Infinity): number {
  if (value === undefined) return fallback;
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)) || value < min || value > max) throw new Error('GLTF_NUMBER: ' + label);
  return value;
}
function vector(value: unknown, defaults: readonly number[], label: string): number[] {
  if (value === undefined) return [...defaults];
  const values = array(value, label, defaults.length);
  if (values.length !== defaults.length) throw new Error('GLTF_VECTOR: ' + label);
  return values.map(v => number(v, 0, label));
}
/** Imports the supported core glTF subset. Unsupported rendering semantics fail explicitly.
 * Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html */
export function importGltf(json: unknown, buffers: readonly ArrayBuffer[], options: GltfImportOptions = {}): GltfAsset {
  const doc = object(json, 'root'), header = object(doc.asset, 'asset');
  if (header.version !== '2.0' || (header.minVersion !== undefined && header.minVersion !== '2.0')) throw new Error('GLTF_VERSION: requires 2.0');
  const supportedExtensions = new Set(['KHR_materials_unlit']);
  for (const extension of array(doc.extensionsRequired ?? [], 'extensionsRequired')) {
    if (typeof extension !== 'string' || !supportedExtensions.has(extension)) throw new Error('GLTF_REQUIRED_EXTENSION: ' + String(extension));
  }
  const binary = doc as unknown as GltfBinaryDocument;
  // Reject oversized or malformed node tables before decoding any mesh data.
  array(doc.nodes ?? [], 'nodes', 4096).forEach((node, i) => object(node, 'node' + i));
  const meshJson = array(doc.meshes ?? [], 'meshes').map((mesh, i) => object(mesh, 'mesh' + i));
  const skinJson = array(doc.skins ?? [], 'skins').map((skin, i) => object(skin, 'skin' + i));
  array(doc.materials ?? [], 'materials').forEach((value, i) => object(value, 'material' + i));
  array(doc.textures ?? [], 'textures').forEach(value => object(value, 'texture'));
  array(doc.samplers ?? [], 'samplers').forEach(value => object(value, 'sampler'));
  const materialLibrary = createGltfMaterialLibrary(json, options);
  for (let i = 0; i < materialLibrary.materialCount; i++) materialLibrary.get(i);
  const defaultMaterial = materialLibrary.get(null);
  const accessorDefinitions = array(doc.accessors ?? [], 'accessors');
  const accessorCount = accessorDefinitions.length;
  const warnings: string[] = [];
  if (doc.cameras !== undefined) warnings.push('glTF cameras are asset metadata; the application supplies the render camera.');
  let vertexTotal = 0, indexTotal = 0;
  const maxVertices = number(options.maxVertices, 1_000_000, 'maxVertices', 1, 10_000_000);
  const maxIndices = number(options.maxIndices, 3_000_000, 'maxIndices', 3, 30_000_000);
  const maxTangentBytes = number(options.maxTangentBytes, 256 * 1024 * 1024, 'maxTangentBytes', 1, 512 * 1024 * 1024);
  if (![maxVertices, maxIndices, maxTangentBytes].every(Number.isSafeInteger)) throw new Error('GLTF_IMPORT_BUDGET_INTEGER');
  const access = (value: unknown, type: string, label: string) => {
    const result = readAccessor(binary, buffers, index(value, accessorCount, label));
    if (result.type !== type) throw new Error('GLTF_ACCESSOR_TYPE: ' + label + ' must be ' + type);
    return result;
  };
  const meshes: GltfMesh[] = meshJson.map((source, meshIndex) => {
    const primitives = array(source.primitives, 'primitives', 4096).map((primitiveValue, primitiveIndex): GltfPrimitive => {
      const primitive = object(primitiveValue, 'primitive');
      if ((primitive.mode ?? 4) !== 4) throw new Error('GLTF_UNSUPPORTED_PRIMITIVE: triangles required');
      if (primitive.extensions !== undefined) throw new Error('GLTF_UNSUPPORTED_PRIMITIVE_EXTENSION');
      const binding = primitive.material === undefined ? defaultMaterial : materialLibrary.get(index(primitive.material, materialLibrary.materialCount, 'primitive.material'));
      const attrs = object(primitive.attributes, 'attributes');
      for (const attribute of Object.keys(attrs)) if (!['POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'COLOR_0', 'JOINTS_0', 'WEIGHTS_0'].includes(attribute)) throw new Error('GLTF_UNSUPPORTED_ATTRIBUTE: ' + attribute);
      const position = access(attrs.POSITION, 'VEC3', 'POSITION');
      if (position.componentType !== 5126) throw new Error('GLTF_POSITION_FLOAT_REQUIRED');
      const count = position.count;
      const attribute = (value: unknown, type: string, label: string) => {
        if (value === undefined) return undefined;
        const result = access(value, type, label);
        if (result.count !== count) throw new Error('GLTF_ATTRIBUTE_COUNT: ' + label);
        return result;
      };
      const normal = attribute(attrs.NORMAL, 'VEC3', 'NORMAL'), tangent = attribute(attrs.TANGENT, 'VEC4', 'TANGENT');
      const uv = attribute(attrs.TEXCOORD_0, 'VEC2', 'TEXCOORD_0');
      const color = attrs.COLOR_0 === undefined ? undefined : readAccessor(binary, buffers, index(attrs.COLOR_0, accessorCount, 'COLOR_0'));
      if (color) {
        if (color.count !== count || (color.type !== 'VEC3' && color.type !== 'VEC4')
          || !(color.componentType === 5126 && !color.normalized
            || (color.componentType === 5121 || color.componentType === 5123) && color.normalized)) {
          throw new Error('GLTF_COLOR_LAYOUT');
        }
        for (const value of color.values) if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error('GLTF_COLOR_RANGE');
      }
      const generateTangents = binding.definition.normal !== undefined && !(normal && tangent);
      if (binding.definition.normal !== undefined && !uv) throw new Error('GLTF_NORMAL_TEXTURE_REQUIRES_UV0');
      const joints = attribute(attrs.JOINTS_0, 'VEC4', 'JOINTS_0'), weights = attribute(attrs.WEIGHTS_0, 'VEC4', 'WEIGHTS_0');
      if (!!joints !== !!weights) throw new Error('GLTF_SKIN_ATTRIBUTES_PAIRED');
      if (normal && normal.componentType !== 5126) throw new Error('GLTF_NORMAL_FLOAT_REQUIRED');
      if (tangent && tangent.componentType !== 5126) throw new Error('GLTF_TANGENT_FLOAT_REQUIRED');
      if (tangent) {
        for (let vertex = 0; vertex < count; vertex++) {
          const handedness = tangent.values[vertex * 4 + 3];
          if (handedness !== 1 && handedness !== -1) throw new Error('GLTF_TANGENT_HANDEDNESS');
        }
        if (!normal) warnings.push(`mesh ${meshIndex} primitive ${primitiveIndex}: source TANGENT and any morph TANGENT deltas are ignored because NORMAL is absent.`);
      }
      if (joints && (![5121, 5123].includes(joints.componentType) || joints.normalized)) throw new Error('GLTF_JOINT_COMPONENT_TYPE');
      if (weights && !(weights.componentType === 5126 || ([5121, 5123].includes(weights.componentType) && weights.normalized))) throw new Error('GLTF_WEIGHT_COMPONENT_TYPE');
      if (uv && !(uv.componentType === 5126 || ([5121, 5123].includes(uv.componentType) && uv.normalized))) throw new Error('GLTF_UV_COMPONENT_TYPE');
      let indices: Uint32Array;
      if (primitive.indices === undefined) indices = Uint32Array.from({ length: count }, (_, i) => i);
      else {
        const data = access(primitive.indices, 'SCALAR', 'indices');
        if (data.normalized || !(data.values instanceof Uint32Array)) throw new Error('GLTF_INDEX_COMPONENT_TYPE');
        const restart = data.componentType === 5121 ? 255 : data.componentType === 5123 ? 65535 : 4294967295;
        if (data.values.includes(restart)) throw new Error('GLTF_INDEX_PRIMITIVE_RESTART');
        indices = new Uint32Array(data.values);
      }
      indexTotal += indices.length;
      if (indexTotal > maxIndices || !indices.length || indices.length % 3 !== 0 || indices.some(i => i >= count)) throw new Error('GLTF_INDICES_OR_BUDGET');
      // Flat normals require one vertex per triangle corner: shared indexed vertices
      // cannot carry distinct face normals. The mapping also remaps every side stream.
      const deindexed = !normal;
      const sourceVertices = deindexed ? indices : Uint32Array.from({ length: count }, (_, i) => i);
      const outputCount = sourceVertices.length;
      vertexTotal += outputCount;
      if (vertexTotal > maxVertices) throw new Error('GLTF_VERTEX_BUDGET');
      if (deindexed) indices = Uint32Array.from({ length: outputCount }, (_, i) => i);
      const remap = (data: Float32Array | Uint32Array, components: number): Float32Array => {
        const output = new Float32Array(outputCount * components);
        for (let vertex = 0; vertex < outputCount; vertex++) {
          const sourceOffset = sourceVertices[vertex] * components, destinationOffset = vertex * components;
          for (let k = 0; k < components; k++) output[destinationOffset + k] = data[sourceOffset + k];
        }
        return output;
      };
      const vertices = new ArrayBuffer(outputCount * 80), view = new DataView(vertices);
      const bounds: MeshData['bounds'] = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
      const generatedNormals = new Float32Array(outputCount * 3);
      if (!normal) {
        for (let tri = 0; tri < outputCount; tri += 3) {
          const [a, b, c] = [sourceVertices[tri], sourceVertices[tri + 1], sourceVertices[tri + 2]];
          const p = position.values;
          const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
          const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
          const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
          const length = Math.hypot(nx, ny, nz);
          if (length === 0 || !Number.isFinite(length)) throw new Error('GLTF_DEGENERATE_NORMAL');
          for (let corner = 0; corner < 3; corner++) {
            const destination = (tri + corner) * 3;
            generatedNormals[destination] = nx / length;
            generatedNormals[destination + 1] = ny / length;
            generatedNormals[destination + 2] = nz / length;
          }
        }
        warnings.push('mesh ' + meshIndex + ': missing normals generated as flat face normals.');
      }
      for (let vertex = 0; vertex < outputCount; vertex++) {
        const sourceVertex = sourceVertices[vertex];
        const base = vertex * 80;
        for (let k = 0; k < 3; k++) {
          const value = position.values[sourceVertex * 3 + k];
          view.setFloat32(base + k * 4, value, true); bounds.min[k] = Math.min(bounds.min[k], value); bounds.max[k] = Math.max(bounds.max[k], value);
        }
        view.setFloat32(base + 12, 1, true);
        const normals = normal?.values ?? generatedNormals;
        const normalOffset = normal ? sourceVertex * 3 : vertex * 3;
        const length = Math.hypot(normals[normalOffset], normals[normalOffset + 1], normals[normalOffset + 2]);
        if (length === 0 || !Number.isFinite(length)) throw new Error('GLTF_DEGENERATE_NORMAL');
        for (let k = 0; k < 3; k++) view.setFloat32(base + 16 + k * 4, normals[normalOffset + k] / length, true);
        let weightSum = 0;
        for (let k = 0; k < 4; k++) {
          const weight = weights?.values[sourceVertex * 4 + k] ?? Number(k === 0);
          if (weight < 0) throw new Error('GLTF_NEGATIVE_WEIGHT');
          weightSum += weight;
        }
        if (weightSum <= 0) throw new Error('GLTF_ZERO_WEIGHT_SUM');
        for (let k = 0; k < 4; k++) {
          view.setUint32(base + 32 + k * 4, joints?.values[sourceVertex * 4 + k] ?? 0, true);
          view.setFloat32(base + 48 + k * 4, (weights?.values[sourceVertex * 4 + k] ?? Number(k === 0)) / weightSum, true);
        }
      }
      const mesh: MeshData = { vertices, indices, vertexCount: outputCount, bounds,
        uvs: uv ? remap(uv.values, 2) : undefined,
        colors: color ? (() => {
          const output = new Float32Array(outputCount * 4);
          for (let vertex = 0; vertex < outputCount; vertex++) {
            const source = sourceVertices[vertex] * color.components, destination = vertex * 4;
            for (let channel = 0; channel < color.components; channel++) output[destination + channel] = color.values[source + channel];
            if (color.components === 3) output[destination + 3] = 1;
          }
          return output;
        })() : undefined,
        tangents: normal && tangent ? remap(tangent.values, 4) : undefined };
      let ignoredAuthoredMorphTangents = false;
      const morphTargets = array(primitive.targets ?? [], 'targets', joints ? 31 : 32).map((targetValue): MorphTarget => {
        const target = object(targetValue, 'target');
        if (generateTangents && target.TANGENT !== undefined) ignoredAuthoredMorphTangents = true;
        if (Object.keys(target).some(key => key !== 'POSITION' && key !== 'NORMAL' && key !== 'TANGENT')) throw new Error('GLTF_UNSUPPORTED_MORPH_ATTRIBUTE');
        if (target.NORMAL !== undefined && !normal) throw new Error('GLTF_MORPH_NORMAL_REQUIRES_BASE_NORMAL');
        if (target.TANGENT !== undefined && !tangent && !generateTangents) throw new Error('GLTF_MORPH_TANGENT_REQUIRES_BASE_TANGENT');
        const positions = attribute(target.POSITION, 'VEC3', 'target.POSITION');
        const normals = attribute(target.NORMAL, 'VEC3', 'target.NORMAL');
        const tangents = attribute(target.TANGENT, 'VEC3', 'target.TANGENT');
        if ([positions, normals, tangents].some(data => data && data.componentType !== 5126)) throw new Error('GLTF_MORPH_FLOAT_REQUIRED');
        const delta = positions ? remap(positions.values, 3) : new Float32Array(outputCount * 3);
        const tangentDeltas = !generateTangents && normal && tangent
          ? tangents ? remap(tangents.values, 3) : new Float32Array(outputCount * 3)
          : undefined;
        if (normal) return { positions: delta,
          normals: normals ? remap(normals.values, 3) : new Float32Array(outputCount * 3),
          tangents: tangentDeltas };
        if (!positions) return { positions: delta, normals: new Float32Array(outputCount * 3), tangents: tangentDeltas };
        return { ...buildMorphTarget(mesh, (_, i) => [delta[i * 3], delta[i * 3 + 1], delta[i * 3 + 2]]), tangents: tangentDeltas };
      });
      if (generateTangents) {
        if (ignoredAuthoredMorphTangents) {
          warnings.push(`mesh ${meshIndex} primitive ${primitiveIndex}: authored morph TANGENT deltas ignored while generating tangents.`);
        }
        const generated = generateGltfTangents(mesh, morphTargets, {
          maxVertices: maxVertices - vertexTotal + outputCount, maxTangentBytes,
        });
        vertexTotal += generated.mesh.vertexCount - outputCount;
        if (vertexTotal > maxVertices) throw new Error('GLTF_VERTEX_BUDGET');
        warnings.push(`mesh ${meshIndex} primitive ${primitiveIndex}: MikkTSpace-compatible tangents generated for normal texture${generated.mesh.vertexCount !== outputCount ? ' with vertex splits' : ''}.`);
        return { mesh: generated.mesh, morphTargets: generated.morphTargets, hasSkin: !!joints, material: binding };
      }
      return { mesh, morphTargets, hasSkin: !!joints, material: binding };
    });
    if (!primitives.length) throw new Error('GLTF_EMPTY_MESH');
    const targetCount = primitives[0].morphTargets.length;
    if (primitives.some(primitive => primitive.morphTargets.length !== targetCount)) throw new Error('GLTF_MORPH_TARGET_COUNT');
    return { name: typeof source.name === 'string' ? source.name : 'mesh-' + meshIndex, primitives,
      weights: vector(source.weights, new Array(targetCount).fill(0), 'mesh.weights') };
  });
  const { nodes, rigNodes, scenes, scene } = parseGltfNodeLayout(json, meshes.map(mesh => mesh.weights));
  const skins: GltfSkin[] = skinJson.map(skin => {
    const joints = array(skin.joints, 'skin.joints', 1024).map(value => index(value, nodes.length, 'skin.joint'));
    if (!joints.length || new Set(joints).size !== joints.length) throw new Error('GLTF_SKIN_JOINTS');
    if (skin.skeleton !== undefined) index(skin.skeleton, nodes.length, 'skin.skeleton');
    let matrices: Mat4[];
    if (skin.inverseBindMatrices === undefined) matrices = joints.map(() => identity());
    else {
      const data = access(skin.inverseBindMatrices, 'MAT4', 'inverseBindMatrices');
      if (data.componentType !== 5126 || data.count < joints.length) throw new Error('GLTF_INVERSE_BIND_LAYOUT');
      matrices = joints.map((_, i) => new Float32Array(data.values.slice(i * 16, i * 16 + 16)));
      matrices.forEach(matrix => inverseAffine(matrix));
    }
    return { joints, inverseBindMatrices: matrices };
  });
  const animationLimit = (value: number | undefined, fallback: number, label: string): number => {
    const limit = value ?? fallback;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > fallback) throw new Error('GLTF_ANIMATION_BUDGET_OPTION: ' + label);
    return limit;
  };
  const maxAnimationTracks = animationLimit(options.maxAnimationTracks, 65_536, 'maxAnimationTracks');
  const maxAnimationScalars = animationLimit(options.maxAnimationScalars, 16_000_000, 'maxAnimationScalars');
  const animations = array(doc.animations ?? [], 'animations', 4096).map((value, i) => object(value, 'animation' + i));
  let trackTotal = 0, scalarTotal = 0;
  const accessorScalars = (value: unknown, type: string, components: number, label: string): number => {
    const accessor = object(accessorDefinitions[index(value, accessorCount, label)], label);
    if (accessor.type !== type || typeof accessor.count !== 'number' || !Number.isSafeInteger(accessor.count) || accessor.count < 1) {
      throw new Error('GLTF_ANIMATION_ACCESSOR_LAYOUT: ' + label);
    }
    const scalars = accessor.count * components;
    if (!Number.isSafeInteger(scalars)) throw new Error('GLTF_ANIMATION_SCALAR_BUDGET');
    return scalars;
  };
  // Charge each channel, even when it reuses a sampler: each becomes an owned track.
  // This completes before any animation accessor is decoded or copied.
  for (const animation of animations) {
    const channels = array(animation.channels, 'animation.channels');
    const samplers = array(animation.samplers, 'animation.samplers');
    trackTotal += channels.length;
    if (trackTotal > maxAnimationTracks) throw new Error('GLTF_ANIMATION_TRACK_BUDGET');
    for (const channelValue of channels) {
      const channel = object(channelValue, 'animation.channel');
      const target = object(channel.target, 'animation.target');
      const path = target.path;
      if (path !== 'translation' && path !== 'rotation' && path !== 'scale' && path !== 'weights') throw new Error('GLTF_ANIMATION_PATH');
      const sampler = object(samplers[index(channel.sampler, samplers.length, 'channel.sampler')], 'animation.sampler');
      const type = path === 'weights' ? 'SCALAR' : path === 'rotation' ? 'VEC4' : 'VEC3';
      const components = path === 'weights' ? 1 : path === 'rotation' ? 4 : 3;
      const scalars = accessorScalars(sampler.input, 'SCALAR', 1, 'animation.input')
        + accessorScalars(sampler.output, type, components, 'animation.output');
      if (!Number.isSafeInteger(scalars) || scalars > maxAnimationScalars - scalarTotal) throw new Error('GLTF_ANIMATION_SCALAR_BUDGET');
      scalarTotal += scalars;
    }
  }
  const clips: AnimationClip[] = animations.map((value, clipIndex) => {
    const animation = object(value, 'animation'), samplers = array(animation.samplers, 'animation.samplers');
    const tracks: AnimationTrack[] = array(animation.channels, 'animation.channels').map(channelValue => {
      const channel = object(channelValue, 'channel'), target = object(channel.target, 'target');
      const node = index(target.node, nodes.length, 'target.node'), path = target.path;
      if (path !== 'translation' && path !== 'rotation' && path !== 'scale' && path !== 'weights') throw new Error('GLTF_ANIMATION_PATH');
      const sampler = object(samplers[index(channel.sampler, samplers.length, 'channel.sampler')], 'animation.sampler');
      const interpolation = sampler.interpolation ?? 'LINEAR';
      if (interpolation !== 'STEP' && interpolation !== 'LINEAR' && interpolation !== 'CUBICSPLINE') throw new Error('GLTF_INTERPOLATION');
      const times = access(sampler.input, 'SCALAR', 'animation.input');
      const values = access(sampler.output, path === 'weights' ? 'SCALAR' : path === 'rotation' ? 'VEC4' : 'VEC3', 'animation.output');
      if (times.componentType !== 5126 || values.componentType !== 5126) throw new Error('GLTF_ANIMATION_FLOAT_REQUIRED');
      return { node, path, interpolation, times: new Float32Array(times.values), values: new Float32Array(values.values),
        components: path === 'weights' ? rigNodes[node].weights.length : path === 'rotation' ? 4 : 3 };
    });
    return { name: typeof animation.name === 'string' ? animation.name : 'animation-' + clipIndex, tracks,
      duration: tracks.reduce((duration, track) => Math.max(duration, track.times.at(-1) ?? 0), 0) };
  });
  const rig = new AnimationRig(rigNodes, clips);
  return { rig, meshes, skins, nodes, scenes, scene, warnings };
}
