#!/usr/bin/env node
// Validates a local skinned, animated glTF with the engine's own loader, importer,
// animation rig, scene instance and CPU deformation reference, in Node through Vite SSR.
//   node scripts/validate-gltf-skinned-animation.mjs --gltf scene.gltf --output new-report.json
//     [--times 0,1] [--clip 0]
//     [--reference points.json --reference-fps 24 --reference-first-frame 1 [--reference-yaw-degrees A]
//      --reference-mesh MESH=PRIM+PRIM ... [--reference-mesh MESH=REST]]
// Checks, in order:
//  1. loadGltf through a local fetch, with its default budgets and with probes at
//     exact budget values. Node has no createImageBitmap, so decoding is emulated
//     from PNG headers: each image is charged width * height * 4 bytes exactly as
//     decodeTexture charges it, but its pixels stay zero. Failures are recorded.
//  2. The validated asset is loaded with maxDecodedImageBytes raised to the
//     package's decoded need. That is a caller option; no limit in src changes.
//  3. importGltf budget probes: each counted total passes at its exact value and
//     fails at one less, which proves this script counts as the importer does.
//  4. Totals against the importer, AnimationRig, loader and DEFORMATION_LIMITS limits.
//  5. GltfSceneInstance construction, which validates every vertex for its skin graph.
//  6. Rest pose: each joint's rest world matrix times its inverse bind matrix is ~I.
//  7. Joint palettes sampled at consecutive --times differ.
//  8. Two-key and STEP tracks hold one value (the exporter's animation-size optimisation).
//  9. Optional: positions skinned by evaluateDeformation against reference points
//     (metres, Y-up) at frame f, sampled at t = (f - first) / fps. --reference-yaw-degrees
//     rotates the skinned positions about +Y through the origin first, for a reference
//     whose world heading differs from the glTF's by a known constant.
//     Correspondence is unknown, so each direction reports nearest-neighbour
//     distances: a lower bound of the per-vertex displacement.
// The report holds no timestamps or absolute paths; wall times go to stdout only.
import { createHash } from 'node:crypto';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { exists } from './asset-output.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
const MAX_RESOURCE_BYTES = 1024 * MiB;
const MAX_REFERENCE_BYTES = 512 * MiB;
const MAX_REFERENCE_POINTS = 4_000_000;
const MAX_TIMES = 64;
const CELL = 0.01;
const MAX_RING = 5;
const CONSTANT_TOLERANCE = 1e-6;
const CHANGED_JOINT_TOLERANCE = 1e-6;
const ENGINE_SOURCES = ['src/gltf-load.ts', 'src/gltf-import.ts', 'src/gltf-node-layout.ts', 'src/gltf-material.ts',
  'src/gltf-scene.ts', 'src/animation.ts', 'src/deformation.ts', 'src/texture-assets.ts', 'src/math.ts'];
// Defaults applied when an option is omitted. Each is a literal in the named file and
// not exported; a change there shows up as a mismatch against the probes below.
const DEFAULTS = {
  loadMaxBytes: 128 * MiB, // src/gltf-load.ts maxBytes
  loadMaxDecodedImageBytes: 256 * MiB, // src/gltf-load.ts maxDecodedImageBytes
  importMaxVertices: 1_000_000, // src/gltf-import.ts
  importMaxIndices: 3_000_000, // src/gltf-import.ts
  importMaxAnimationTracks: 65_536, // src/gltf-import.ts, equal to AnimationRig MAX_TRACKS
  importMaxAnimationScalars: 16_000_000, // src/gltf-import.ts
  importMaxNodes: 4096, // src/gltf-import.ts node table, equal to AnimationRig MAX_NODES
  importMaxSkinJoints: 1024, // src/gltf-import.ts skin.joints
  importMaxMorphTargetsSkinned: 31, // src/gltf-import.ts primitive.targets with JOINTS_0
  importMaxMorphTargetsUnskinned: 32,
  rigMaxKeyValues: 16_000_000, // src/animation.ts MAX_KEY_VALUES
  textureMaxDimension: 8192, // src/texture-assets.ts MAX_TEXTURE_DIMENSION
};
const DEFORMATION_MEANINGS = {
  nodes: 'operators (morph, skin, wave) in one batch deformation graph',
  joints: 'joint palette size of one graph; every vertex joint index must be below it',
  scalars: 'per-instance deformation scalar slots (morph weights, wave phases) one graph can index',
  morphTargets: 'morph target slots one graph can reference (highest target index + 1)',
};

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const usage = 'usage: validate-gltf-skinned-animation --gltf scene.gltf --output new-report.json [--times 0,1] [--clip 0] '
  + '[--reference points.json --reference-fps 24 --reference-first-frame 1 [--reference-yaw-degrees A] --reference-mesh MESH=PRIM+PRIM|REST ...]';
const single = new Map(), referenceMeshArguments = [];
const singleKeys = ['--gltf', '--output', '--times', '--clip', '--reference', '--reference-fps', '--reference-first-frame', '--reference-yaw-degrees'];
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i], value = process.argv[i + 1];
  if (value === undefined) throw Error(usage);
  if (key === '--reference-mesh') referenceMeshArguments.push(value);
  else if (singleKeys.includes(key) && !single.has(key)) single.set(key, value);
  else throw Error(usage);
}
if (!single.has('--gltf') || !single.has('--output')) throw Error(usage);
const gltfPath = resolve(single.get('--gltf')), gltfDir = dirname(gltfPath), output = resolve(single.get('--output'));
if (await exists(output)) throw Error('VALIDATE_OUTPUT_EXISTS: ' + output);
const times = (single.get('--times') ?? '0,1').split(',').map(Number);
if (times.length < 2 || times.length > MAX_TIMES || !times.every(Number.isFinite)) throw Error('VALIDATE_TIMES: two to 64 finite seconds');
const clip = Number(single.get('--clip') ?? 0);
if (!Number.isSafeInteger(clip) || clip < 0) throw Error('VALIDATE_CLIP');
const referenceRequested = ['--reference', '--reference-fps', '--reference-first-frame', '--reference-yaw-degrees'].some(key => single.has(key)) || referenceMeshArguments.length > 0;
if (referenceRequested && (!['--reference', '--reference-fps', '--reference-first-frame'].every(key => single.has(key)) || !referenceMeshArguments.length)) {
  throw Error('VALIDATE_REFERENCE_ARGUMENTS: --reference, --reference-fps, --reference-first-frame and --reference-mesh go together');
}

// Local resources: one read per file, reused by every probe. Paths stay inside the glTF directory.
const BASE = new URL('http://validate.invalid/');
const resources = new Map();
const readLocal = async relativePath => {
  const path = resolve(gltfDir, relativePath), back = relative(gltfDir, path);
  if (!relativePath || !back || isAbsolute(back) || back === '..' || back.startsWith('..' + sep)) throw Error('VALIDATE_LOCAL_RESOURCE_ONLY: ' + relativePath);
  let entry = resources.get(back);
  if (!entry) {
    const size = (await stat(path)).size;
    if (size > MAX_RESOURCE_BYTES) throw Error('VALIDATE_RESOURCE_BYTES: ' + back);
    const bytes = new Uint8Array(await readFile(path));
    entry = { uri: back.split(sep).join('/'), bytes, sha256: sha256(bytes) };
    resources.set(back, entry);
  }
  return entry;
};
let fetchedBytes = 0;
const localFetch = async target => {
  const url = new URL(target);
  if (url.origin !== BASE.origin || url.search || url.hash) throw Error('VALIDATE_LOCAL_RESOURCE_ONLY: ' + url.href);
  const entry = await readLocal(decodeURIComponent(url.pathname.slice(1)));
  fetchedBytes += entry.bytes.byteLength;
  return new Response(entry.bytes.slice(), { status: 200, headers: { 'content-length': String(entry.bytes.byteLength) } });
};

// PNG header emulation of createImageBitmap and a zero-pixel 2D context.
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const pngSize = bytes => {
  if (bytes.length < 24 || PNG_SIGNATURE.some((value, i) => bytes[i] !== value) || String.fromCharCode(...bytes.subarray(12, 16)) !== 'IHDR') {
    throw Error('VALIDATE_IMAGE_FORMAT: only PNG headers are emulated');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, 24);
  return { width: view.getUint32(16), height: view.getUint32(20) };
};
let decodeCalls = 0;
const emulatedDecoding = typeof globalThis.createImageBitmap !== 'function';
if (emulatedDecoding) {
  globalThis.createImageBitmap = async blob => {
    decodeCalls++;
    const { width, height } = pngSize(new Uint8Array(await blob.slice(0, 24).arrayBuffer()));
    return { width, height, close() {} };
  };
  globalThis.OffscreenCanvas ??= class {
    constructor(width, height) { this.width = width; this.height = height; }
    getContext() { return { drawImage() {}, getImageData: (x, y, width, height) => ({ data: new Uint8ClampedArray(width * height * 4) }) }; }
  };
}

const timings = [];
const timed = async (label, work) => {
  const started = performance.now();
  try { return await work(); } finally { timings.push([label, (performance.now() - started) / 1000]); }
};
const message = error => String(error?.message ?? error);

const server = await createServer({ root: ROOT, configFile: false, server: { middlewareMode: true, watch: null, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom', logLevel: 'error' });
let report;
try {
  const { loadGltf } = await server.ssrLoadModule('/src/gltf-load.ts');
  const { importGltf } = await server.ssrLoadModule('/src/gltf-import.ts');
  const { parseGltfNodeLayout } = await server.ssrLoadModule('/src/gltf-node-layout.ts');
  const { GltfSceneInstance } = await server.ssrLoadModule('/src/gltf-scene.ts');
  const { AnimationRig } = await server.ssrLoadModule('/src/animation.ts');
  const { DEFORMATION_LIMITS, compileDeformation, evaluateDeformation } = await server.ssrLoadModule('/src/deformation.ts');
  const { multiply, transformPoint } = await server.ssrLoadModule('/src/math.ts');

  const gltfEntry = await readLocal(basename(gltfPath));
  const json = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(gltfEntry.bytes));
  const list = value => Array.isArray(value) ? value : [];
  const isLocalUri = uri => typeof uri === 'string' && uri.length > 0 && !/^[a-z][a-z0-9+.-]*:/i.test(uri);
  for (const item of [...list(json.buffers), ...list(json.images)]) {
    if (item.uri !== undefined && !isLocalUri(item.uri)) throw Error('VALIDATE_LOCAL_RESOURCE_ONLY: ' + String(item.uri).slice(0, 32));
  }
  const buffers = [];
  for (const buffer of list(json.buffers)) buffers.push(await readLocal(decodeURIComponent(buffer.uri)));
  const referencedImages = new Set(list(json.textures).map(texture => texture.source).filter(Number.isSafeInteger));
  const images = [];
  for (const [index, image] of list(json.images).entries()) {
    if (image.uri === undefined) throw Error('VALIDATE_IMAGE_BUFFER_VIEW: only image URIs are emulated');
    const entry = await readLocal(decodeURIComponent(image.uri));
    const size = pngSize(entry.bytes);
    images.push({ index, uri: entry.uri, ...size, referenced: referencedImages.has(index), decodedBytes: size.width * size.height * 4 });
  }
  const decodedNeed = images.filter(image => image.referenced).reduce((sum, image) => sum + image.decodedBytes, 0);
  const packageBytes = gltfEntry.bytes.byteLength + buffers.reduce((sum, b) => sum + b.bytes.byteLength, 0)
    + [...new Set(images.filter(image => image.referenced).map(image => image.uri))].reduce((sum, uri) => sum + resources.get(uri.split('/').join(sep)).bytes.byteLength, 0);

  // 1-2. Loader runs. Each run drops its asset unless it is the validated one.
  const gltfUrl = new URL(encodeURIComponent(basename(gltfPath)), BASE);
  const loadRun = async (label, options) => timed('load ' + label, async () => {
    decodeCalls = 0; fetchedBytes = 0;
    try {
      const asset = await loadGltf(gltfUrl, { ...options, fetch: localFetch });
      return { record: { label, options, ok: true, imageDecodeCalls: decodeCalls, fetchedBytes }, asset };
    } catch (error) {
      return { record: { label, options, ok: false, error: message(error), imageDecodeCalls: decodeCalls, fetchedBytes } };
    }
  });
  const loads = [];
  loads.push((await loadRun('defaults', {})).record);
  loads.push((await loadRun('decoded image budget = need - 1', { maxDecodedImageBytes: decodedNeed - 1 })).record);
  loads.push((await loadRun('download budget = package bytes - 1', { maxDecodedImageBytes: decodedNeed, maxBytes: packageBytes - 1 })).record);
  const validated = await loadRun('decoded image budget = need', { maxDecodedImageBytes: decodedNeed });
  loads.push(validated.record);
  if (!validated.asset) throw Error('VALIDATE_LOAD_FAILED: ' + validated.record.error);
  const asset = validated.asset;

  // 3. Import budget probes on the same document, with 1x1 placeholder images.
  const placeholders = new Map(images.map(image => [image.index, { width: 1, height: 1, pixels: new Uint8Array(4), colorSpace: 'srgb' }]));
  const bufferData = buffers.map(b => b.bytes.buffer.slice(b.bytes.byteOffset, b.bytes.byteOffset + b.bytes.byteLength));
  const vertexTotal = asset.meshes.reduce((sum, mesh) => sum + mesh.primitives.reduce((s, p) => s + p.mesh.vertexCount, 0), 0);
  const indexTotal = asset.meshes.reduce((sum, mesh) => sum + mesh.primitives.reduce((s, p) => s + p.mesh.indices.length, 0), 0);
  const accessorCount = index => json.accessors[index].count;
  let trackTotal = 0, scalarTotal = 0;
  for (const animation of list(json.animations)) {
    for (const channel of animation.channels) {
      const sampler = animation.samplers[channel.sampler];
      const components = channel.target.path === 'rotation' ? 4 : channel.target.path === 'weights' ? 1 : 3;
      trackTotal++;
      scalarTotal += accessorCount(sampler.input) + accessorCount(sampler.output) * components;
    }
  }
  const counted = { maxVertices: vertexTotal, maxIndices: indexTotal, maxAnimationTracks: trackTotal, maxAnimationScalars: scalarTotal };
  const importProbe = options => { try { importGltf(json, bufferData, { images: placeholders, ...options }); return { ok: true }; } catch (error) { return { ok: false, error: message(error) }; } };
  const budgetProbes = await timed('import budget probes', async () => {
    const probes = [{ options: { ...counted }, ...importProbe(counted) }];
    for (const [option, value] of Object.entries(counted)) probes.push({ options: { [option]: value - 1 }, ...importProbe({ [option]: value - 1 }) });
    return probes;
  });
  const accountingMatches = budgetProbes[0].ok && budgetProbes.slice(1).every(probe => !probe.ok);

  // 4. Animation totals as the rig owns them.
  const clips = asset.rig.clips;
  const keyValues = clips.reduce((sum, c) => sum + c.tracks.reduce((s, t) => s + t.values.length, 0), 0);
  const clipReports = clips.map((c, clipIndex) => {
    const histogram = (key) => c.tracks.reduce((map, t) => { const k = String(key(t)); map[k] = (map[k] ?? 0) + 1; return map; }, {});
    let twoKey = 0, twoKeyConstant = 0, step = 0, stepConstant = 0, maxConstantDeviation = 0, nonConstantStep = [];
    for (const track of c.tracks) {
      const keys = track.times.length, stride = track.values.length / keys;
      if (keys !== 2 && track.interpolation !== 'STEP') continue;
      let deviation = 0;
      for (let k = 1; k < keys; k++) {
        let same = 0, flipped = 0;
        for (let j = 0; j < stride; j++) {
          same = Math.max(same, Math.abs(track.values[k * stride + j] - track.values[j]));
          flipped = Math.max(flipped, Math.abs(track.values[k * stride + j] + track.values[j]));
        }
        deviation = Math.max(deviation, track.path === 'rotation' ? Math.min(same, flipped) : same);
      }
      const constant = deviation <= CONSTANT_TOLERANCE;
      if (keys === 2) { twoKey++; twoKeyConstant += Number(constant); }
      if (track.interpolation === 'STEP') {
        step++; stepConstant += Number(constant);
        if (!constant) nonConstantStep.push({ node: asset.nodes[track.node].name, path: track.path, keys, deviation });
      }
      if (constant) maxConstantDeviation = Math.max(maxConstantDeviation, deviation);
    }
    return { index: clipIndex, name: c.name, duration: c.duration, tracks: c.tracks.length,
      paths: histogram(t => t.path), interpolation: histogram(t => t.interpolation), keys: histogram(t => t.times.length),
      keyValues: c.tracks.reduce((s, t) => s + t.values.length, 0),
      constantTracks: { twoKey, twoKeyConstant, step, stepConstant, maxDeviationOfConstantTracks: maxConstantDeviation,
        nonConstantStep: nonConstantStep.slice(0, 16), nonConstantStepCount: nonConstantStep.length } };
  });
  if (clip >= Math.max(1, clips.length)) throw Error('VALIDATE_CLIP: asset has ' + clips.length + ' clips');

  // 5. Scene instance: per-primitive deformation graphs.
  const instance = await timed('scene instance', async () => new GltfSceneInstance(asset, { name: 'validate' }));
  const programs = instance.batches.map(batch => compileDeformation(batch.deformation));
  const batchMesh = batch => asset.nodes[Number(batch.name.split(':')[1])].mesh;
  const batches = instance.batches.map((batch, i) => ({ name: batch.name, mesh: asset.meshes[batchMesh(batch)].name,
    vertices: batch.mesh.vertexCount, triangles: batch.mesh.indices.length / 3, morphTargets: batch.morphTargets.length,
    deformation: { jointCount: batch.deformation.jointCount, nodes: batch.deformation.nodes.map(node => ({ ...node })) },
    compiled: { nodes: programs[i].nodes.length, jointCount: programs[i].jointCount, scalarCount: programs[i].scalarCount, morphTargetCount: programs[i].morphTargetCount } }));
  const most = key => Math.max(0, ...programs.map(program => key(program)));

  // 6. Rest pose from the engine's node layout, without any clip.
  const layout = parseGltfNodeLayout(json, asset.meshes.map(mesh => mesh.weights));
  const rest = new AnimationRig(layout.rigNodes, []).sample(0);
  const restPose = asset.skins.map((skin, skinIndex) => {
    let worst = 0, worstJoint = null;
    skin.joints.forEach((joint, i) => {
      const product = multiply(rest.world[joint], skin.inverseBindMatrices[i]);
      for (let k = 0; k < 16; k++) {
        const deviation = Math.abs(product[k] - Number(k % 5 === 0));
        if (deviation > worst) { worst = deviation; worstJoint = asset.nodes[joint].name; }
      }
    });
    return { skin: skinIndex, joints: skin.joints.length, maxDeviationFromIdentity: worst, worstJoint };
  });

  // 7. Joint palettes at consecutive times.
  const skinned = instance.batches.map((batch, i) => i).filter(i => instance.batches[i].deformation.jointCount > 0);
  const palettesAt = time => { instance.sample(time, clip); return instance.batches.map(batch => batch.instances[0].bones.map(m => Float64Array.from(m))); };
  const sampled = await timed('palette sampling', async () => times.map(time => ({ time, palettes: palettesAt(time) })));
  const matrixDifference = (a, b) => { let d = 0; for (let k = 0; k < 16; k++) d = Math.max(d, Math.abs(a[k] - b[k])); return d; };
  const sampling = [];
  for (let i = 1; i < sampled.length; i++) {
    const before = sampled[i - 1], after = sampled[i];
    const first = skinned[0];
    let maxDifference = 0, changed = 0;
    if (first !== undefined) {
      before.palettes[first].forEach((matrix, joint) => {
        const d = matrixDifference(matrix, after.palettes[first][joint]);
        maxDifference = Math.max(maxDifference, d); changed += Number(d > CHANGED_JOINT_TOLERANCE);
      });
    }
    let batchDisagreement = 0;
    for (const s of sampled.slice(i - 1, i + 1)) for (const b of skinned) s.palettes[b].forEach((m, j) => { batchDisagreement = Math.max(batchDisagreement, matrixDifference(m, s.palettes[first][j])); });
    sampling.push({ from: before.time, to: after.time, joints: first === undefined ? 0 : before.palettes[first].length,
      changedJoints: changed, maxMatrixElementDifference: maxDifference, differs: changed > 0,
      maxDisagreementBetweenBatches: batchDisagreement });
  }

  // 8 (reported above). 9. Optional reference comparison.
  let reference = null;
  if (referenceRequested) {
    const referencePath = resolve(single.get('--reference'));
    if ((await stat(referencePath)).size > MAX_REFERENCE_BYTES) throw Error('VALIDATE_REFERENCE_BYTES');
    const referenceBytes = await readFile(referencePath);
    const data = JSON.parse(referenceBytes.toString('utf8'));
    const fps = Number(single.get('--reference-fps')), firstFrame = Number(single.get('--reference-first-frame'));
    if (!(fps > 0) || !Number.isFinite(firstFrame)) throw Error('VALIDATE_REFERENCE_TIME');
    const yawDegrees = Number(single.get('--reference-yaw-degrees') ?? 0);
    if (!Number.isFinite(yawDegrees) || Math.abs(yawDegrees) > 360) throw Error('VALIDATE_REFERENCE_YAW');
    const yawCos = Math.cos(yawDegrees * Math.PI / 180), yawSin = Math.sin(yawDegrees * Math.PI / 180);
    const prims = data?.prims && typeof data.prims === 'object' ? data.prims : {};
    const named = new Set(), mappings = [];
    let restMesh = null;
    for (const argument of referenceMeshArguments) {
      const split = argument.indexOf('=');
      const mesh = argument.slice(0, split), right = argument.slice(split + 1);
      if (split < 1 || !right || !asset.meshes.some(m => m.name === mesh) || mappings.some(m => m.mesh === mesh)) throw Error('VALIDATE_REFERENCE_MESH: ' + argument);
      if (right === 'REST') { if (restMesh !== null) throw Error('VALIDATE_REFERENCE_REST_TWICE'); restMesh = mesh; mappings.push({ mesh, prims: null }); continue; }
      const names = right.split('+');
      for (const name of names) { if (!Object.hasOwn(prims, name) || named.has(name)) throw Error('VALIDATE_REFERENCE_PRIM: ' + name); named.add(name); }
      mappings.push({ mesh, prims: names });
    }
    for (const mapping of mappings) if (mapping.prims === null) mapping.prims = Object.keys(prims).filter(name => !named.has(name)).sort();
    const frames = [...new Set(mappings.flatMap(m => m.prims.flatMap(name => Object.keys(prims[name].frames ?? {}))))].map(Number).sort((a, b) => a - b);
    let pointTotal = 0;
    const gather = (names, frame) => {
      const arrays = names.map(name => { const points = prims[name].frames?.[String(frame)]; if (!Array.isArray(points)) throw Error('VALIDATE_REFERENCE_FRAME: ' + name + ' ' + frame); return points; });
      const count = arrays.reduce((s, a) => s + a.length, 0);
      pointTotal += count;
      if (pointTotal > MAX_REFERENCE_POINTS) throw Error('VALIDATE_REFERENCE_POINTS');
      const out = new Float64Array(count * 3);
      let o = 0;
      for (const points of arrays) for (const p of points) {
        if (!Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite)) throw Error('VALIDATE_REFERENCE_POINT');
        out[o++] = p[0]; out[o++] = p[1]; out[o++] = p[2];
      }
      return out;
    };
    const skinnedPositions = () => {
      const byMesh = new Map();
      instance.batches.forEach((batch, b) => {
        const state = batch.instances[0], view = new DataView(batch.mesh.vertices), program = programs[b];
        const out = new Float64Array(batch.mesh.vertexCount * 3);
        for (let v = 0; v < batch.mesh.vertexCount; v++) {
          const at = (offset, count, read) => Array.from({ length: count }, (_, k) => read(v * 80 + offset + k * 4));
          const vertex = { position: at(0, 3, o => view.getFloat32(o, true)), normal: at(16, 3, o => view.getFloat32(o, true)),
            joints: at(32, 4, o => view.getUint32(o, true)), weights: at(48, 4, o => view.getFloat32(o, true)) };
          const deformed = evaluateDeformation(program, vertex, v, batch.morphTargets, state.bones, state.deformScalars);
          const world = transformPoint(state.model, deformed.position);
          // Rotation about +Y: x' = c x + s z, z' = -s x + c z.
          out[v * 3] = yawCos * world[0] + yawSin * world[2]; out[v * 3 + 1] = world[1]; out[v * 3 + 2] = -yawSin * world[0] + yawCos * world[2];
        }
        const name = asset.meshes[batchMesh(batch)].name;
        byMesh.set(name, [...(byMesh.get(name) ?? []), out]);
      });
      return new Map([...byMesh].map(([name, parts]) => {
        const all = new Float64Array(parts.reduce((s, p) => s + p.length, 0));
        let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
        return [name, all];
      }));
    };
    const comparisons = await timed('reference comparison', async () => frames.map(frame => {
      const time = (frame - firstFrame) / fps;
      instance.sample(time, clip);
      const positions = skinnedPositions();
      return { frame, time, meshes: mappings.map(mapping => {
        const points = gather(mapping.prims, frame), vertices = positions.get(mapping.mesh);
        return { mesh: mapping.mesh, referencePrims: mapping.prims.length, referencePoints: points.length / 3, vertices: vertices.length / 3,
          referenceBounds: bounds(points), meshBounds: bounds(vertices),
          referenceToMesh: distanceStatistics(nearestDistances(points, vertices)),
          meshToReference: distanceStatistics(nearestDistances(vertices, points)) };
      }) };
    }));
    reference = { file: basename(referencePath), sha256: sha256(referenceBytes), fps, firstFrame, yawDegreesAppliedToMesh: yawDegrees,
      timeMapping: 't = (frame - firstFrame) / fps', metric: 'nearest-neighbour distance in metres, a lower bound of per-vertex displacement',
      searchRadiusMeters: CELL * MAX_RING,
      mappings: mappings.map(m => ({ mesh: m.mesh, prims: m.prims.length, rest: m.mesh === restMesh, names: m.prims.length <= 16 ? m.prims : undefined })),
      comparisons };
  }

  const nodeCount = list(json.nodes).length;
  const maxJoints = Math.max(0, ...asset.skins.map(skin => skin.joints.length));
  const maxMorph = Math.max(0, ...asset.meshes.flatMap(mesh => mesh.primitives.map(p => p.morphTargets.length)));
  const anySkinnedMorph = asset.meshes.some(mesh => mesh.primitives.some(p => p.hasSkin && p.morphTargets.length));
  const maxImageDimension = Math.max(0, ...images.map(image => Math.max(image.width, image.height)));
  const limit = (owner, name, value, used, meaning) => ({ owner, name, value, used, headroom: value - used, ok: used <= value, ...(used > value ? { exceededBy: used - value, ratio: used / value } : {}), meaning });
  const limits = [
    limit('src/gltf-load.ts', 'maxBytes (default)', DEFAULTS.loadMaxBytes, packageBytes, 'cumulative downloaded bytes: JSON, buffers and referenced images'),
    limit('src/gltf-load.ts', 'maxDecodedImageBytes (default)', DEFAULTS.loadMaxDecodedImageBytes, decodedNeed, 'cumulative decoded RGBA8 bytes of referenced images'),
    limit('src/texture-assets.ts', 'MAX_TEXTURE_DIMENSION', DEFAULTS.textureMaxDimension, maxImageDimension, 'largest decoded image width or height'),
    limit('src/gltf-import.ts', 'maxVertices (default)', DEFAULTS.importMaxVertices, vertexTotal, 'cumulative imported vertices after any de-indexing or tangent splits'),
    limit('src/gltf-import.ts', 'maxIndices (default)', DEFAULTS.importMaxIndices, indexTotal, 'cumulative triangle indices'),
    limit('src/gltf-import.ts', 'nodes', DEFAULTS.importMaxNodes, nodeCount, 'node table length (also AnimationRig MAX_NODES)'),
    limit('src/gltf-import.ts', 'skin joints', DEFAULTS.importMaxSkinJoints, maxJoints, 'joints of one skin'),
    limit('src/gltf-import.ts', 'morph targets per primitive', anySkinnedMorph ? DEFAULTS.importMaxMorphTargetsSkinned : DEFAULTS.importMaxMorphTargetsUnskinned, maxMorph, 'targets of one primitive (31 when skinned, 32 otherwise)'),
    limit('src/gltf-import.ts', 'maxAnimationTracks (default)', DEFAULTS.importMaxAnimationTracks, trackTotal, 'cumulative animation channels (also AnimationRig MAX_TRACKS)'),
    limit('src/gltf-import.ts', 'maxAnimationScalars (default)', DEFAULTS.importMaxAnimationScalars, scalarTotal, 'cumulative input times plus output scalars over channels'),
    limit('src/animation.ts', 'MAX_KEY_VALUES', DEFAULTS.rigMaxKeyValues, keyValues, 'cumulative output key values owned by the rig'),
    ...Object.entries(DEFORMATION_LIMITS).map(([name, value]) => limit('src/deformation.ts', 'DEFORMATION_LIMITS.' + name, value,
      name === 'nodes' ? most(p => p.nodes.length) : name === 'joints' ? most(p => p.jointCount) : name === 'scalars' ? most(p => p.scalarCount) : most(p => p.morphTargetCount),
      DEFORMATION_MEANINGS[name] ?? 'see src/deformation.ts')),
  ];

  const engineSources = {};
  for (const source of ENGINE_SOURCES) engineSources[source] = sha256(await readFile(resolve(ROOT, source)));
  const scriptPath = fileURLToPath(import.meta.url);
  report = {
    format: 'real-engine-gltf-skinned-animation-validation', version: 1,
    tool: { script: relative(ROOT, scriptPath).split(sep).join('/'), sha256: sha256(await readFile(scriptPath)) },
    engineSources,
    input: { gltf: gltfEntry.uri, files: [...resources.values()].map(({ uri, bytes, sha256: hash }) => ({ uri, bytes: bytes.byteLength, sha256: hash })).sort((a, b) => a.uri.localeCompare(b.uri)),
      packageBytes, images, decodedImageBytesNeeded: decodedNeed },
    imageDecoding: emulatedDecoding ? 'png-header emulation: real dimensions and byte charges, zero pixels' : 'native createImageBitmap',
    loads, budgetProbes, importAccountingMatchesEngine: accountingMatches,
    limits, limitsExceeded: limits.filter(l => !l.ok).map(l => l.owner + ' ' + l.name),
    asset: {
      meshes: asset.meshes.map(mesh => ({ name: mesh.name, primitives: mesh.primitives.map(p => ({ vertices: p.mesh.vertexCount, triangles: p.mesh.indices.length / 3,
        hasSkin: p.hasSkin, morphTargets: p.morphTargets.length, tangents: !!p.mesh.tangents, uvs: !!p.mesh.uvs, colors: !!p.mesh.colors })) })),
      nodes: nodeCount, skins: asset.skins.map(skin => ({ joints: skin.joints.length, inverseBindMatrices: skin.inverseBindMatrices.length })),
      scenes: asset.scenes.map(scene => ({ name: scene.name, roots: scene.roots.length })), warnings: asset.warnings,
      totals: { vertices: vertexTotal, indices: indexTotal, triangles: indexTotal / 3, tracks: trackTotal, animationScalars: scalarTotal, rigKeyValues: keyValues },
    },
    animation: { clipCount: asset.rig.clipCount, sampledClip: clip, clips: clipReports },
    sceneInstance: { batches },
    restPose,
    sampling,
    reference,
  };
} finally { await server.close(); }

await writeFile(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
const line = (...parts) => process.stdout.write(parts.join(' ') + '\n');
line('validated', report.input.gltf, 'package bytes', report.input.packageBytes, 'decoded image bytes needed', report.input.decodedImageBytesNeeded);
for (const load of report.loads) line('load', JSON.stringify(load.label), load.ok ? 'ok' : 'failed: ' + load.error, 'decode calls', load.imageDecodeCalls, 'fetched', load.fetchedBytes);
line('import accounting matches engine', report.importAccountingMatchesEngine);
for (const l of report.limits) line('limit', l.owner, l.name, 'value', l.value, 'used', l.used, l.ok ? 'ok' : 'EXCEEDED by ' + l.exceededBy + ' (' + l.ratio.toFixed(3) + 'x)');
for (const c of report.animation.clips) line('clip', JSON.stringify(c.name), 'duration', c.duration, 'tracks', c.tracks, 'interpolation', JSON.stringify(c.interpolation), 'keys', JSON.stringify(c.keys), 'constant', JSON.stringify(c.constantTracks));
for (const r of report.restPose) line('rest pose skin', r.skin, 'joints', r.joints, 'max |rest*IBM - I|', r.maxDeviationFromIdentity, r.worstJoint);
for (const s of report.sampling) line('sampling', s.from, '->', s.to, 'changed joints', s.changedJoints, '/', s.joints, 'max element difference', s.maxMatrixElementDifference, 'batch disagreement', s.maxDisagreementBetweenBatches);
for (const c of report.reference?.comparisons ?? []) for (const m of c.meshes) {
  const box = b => JSON.stringify([b.min.map(v => +v.toFixed(4)), b.max.map(v => +v.toFixed(4))]);
  line('reference frame', c.frame, 't', c.time.toFixed(6), m.mesh, 'ref->mesh max', m.referenceToMesh.max, 'mean', m.referenceToMesh.mean, 'p99', m.referenceToMesh.p99,
    'beyond', m.referenceToMesh.beyondSearchRadius, '| mesh->ref max', m.meshToReference.max, 'mean', m.meshToReference.mean, 'p99', m.meshToReference.p99,
    'beyond', m.meshToReference.beyondSearchRadius, '| bounds ref', box(m.referenceBounds), 'mesh', box(m.meshBounds));
}
line('warnings', JSON.stringify(report.asset.warnings));
for (const [label, seconds] of timings) line('wall', label, seconds.toFixed(2) + ' s');

function nearestDistances(queries, targets) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const set of [queries, targets]) for (let i = 0; i < set.length; i += 3) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], set[i + k]); max[k] = Math.max(max[k], set[i + k]); }
  const dims = [0, 1, 2].map(k => Math.floor((max[k] - min[k]) / CELL) + 1);
  if (!dims.every(d => Number.isSafeInteger(d) && d <= 1 << 16)) throw Error('VALIDATE_REFERENCE_EXTENT');
  const cell = (value, k) => Math.min(dims[k] - 1, Math.floor((value - min[k]) / CELL));
  const grid = new Map();
  for (let i = 0; i < targets.length / 3; i++) {
    const key = cell(targets[i * 3], 0) + dims[0] * (cell(targets[i * 3 + 1], 1) + dims[1] * cell(targets[i * 3 + 2], 2));
    const bucket = grid.get(key);
    if (bucket) bucket.push(i); else grid.set(key, [i]);
  }
  const distances = new Float64Array(queries.length / 3);
  for (let q = 0; q < distances.length; q++) {
    const x = queries[q * 3], y = queries[q * 3 + 1], z = queries[q * 3 + 2];
    const cx = cell(x, 0), cy = cell(y, 1), cz = cell(z, 2);
    let best = Infinity;
    for (let r = 0; r <= MAX_RING; r++) {
      for (let dz = -r; dz <= r; dz++) for (let dy = -r; dy <= r; dy++) {
        const shell = Math.abs(dz) === r || Math.abs(dy) === r;
        for (let dx = -r; dx <= r; dx += shell || r === 0 ? 1 : 2 * r) {
          const ix = cx + dx, iy = cy + dy, iz = cz + dz;
          if (ix < 0 || iy < 0 || iz < 0 || ix >= dims[0] || iy >= dims[1] || iz >= dims[2]) continue;
          const bucket = grid.get(ix + dims[0] * (iy + dims[1] * iz));
          if (!bucket) continue;
          for (const t of bucket) {
            const d = Math.hypot(targets[t * 3] - x, targets[t * 3 + 1] - y, targets[t * 3 + 2] - z);
            if (d < best) best = d;
          }
        }
      }
      if (best <= r * CELL) break;
    }
    // Beyond the searched rings the nearest point is unknown: record it as infinite.
    distances[q] = best <= MAX_RING * CELL ? best : Infinity;
  }
  return distances;
}

function bounds(points) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < points.length; i += 3) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], points[i + k]); max[k] = Math.max(max[k], points[i + k]); }
  return { min, max };
}

function distanceStatistics(distances) {
  const sorted = Float64Array.from(distances).sort();
  const finite = sorted.filter(Number.isFinite);
  const at = fraction => finite.length ? finite[Math.min(finite.length - 1, Math.floor(fraction * (finite.length - 1)))] : null;
  return { count: sorted.length, beyondSearchRadius: sorted.length - finite.length,
    max: finite.length ? finite[finite.length - 1] : null, mean: finite.length ? finite.reduce((s, d) => s + d, 0) / finite.length : null,
    p50: at(0.5), p99: at(0.99), p999: at(0.999),
    over0_1mm: finite.filter(d => d > 1e-4).length, over1mm: finite.filter(d => d > 1e-3).length, over1cm: finite.filter(d => d > 1e-2).length };
}
