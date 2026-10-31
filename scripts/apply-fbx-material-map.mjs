#!/usr/bin/env node
// Second step of an FBX conversion (after scripts/blender/convert-fbx-scene.py):
// rewrites the exported glTF materials from the texture files Blender saw on
// each FBX material, using a declared texture-naming convention instead of
// Blender's shader guesses, and publishes a self-contained glTF directory:
//   scene.gltf, scene.bin (hard link), Textures/<file> (hard links to the
//   source files), copied license files, camera-path.json and conversion.json
//   (every material decision, alpha statistics and unsupported features).
//   node scripts/apply-fbx-material-map.mjs --input blender-output-dir \
//     --source-root extracted-source-dir --output new-directory \
//     [--convention orca-metal-rough] [--license file]...
// Convention orca-metal-rough (Amazon Lumberyard Bistro README): _BaseColor
// RGB + alpha opacity, _Specular R occlusion / G roughness / B metalness,
// _Normal tangent space with the DirectX green axis, _Emissive RGB; material
// names containing ".DoubleSided" are double-sided, all others single-sided.
// Texture files must be legacy DDS/PNG/TGA inside --source-root.
import { realpathSync } from 'node:fs';
import { mkdir, mkdtemp, link, copyFile, writeFile, realpath } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { basename, dirname, extname, join, relative, resolve, sep } from 'node:path';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';
import { boundedRead, sha256 } from './texture-package.mjs';
import { decodeLegacyDds } from '../src/texture-bcn.ts';

const MiB = 1024 * 1024;
const CONVENTIONS = {
  'orca-metal-rough': {
    roles: { BaseColor: 'baseColor', Specular: 'occlusionRoughnessMetalness', Normal: 'normal', Emissive: 'emissive' },
    normalConvention: 'directx',
    alpha: 'base color alpha is opacity; any texel below 255 makes the material MASK with cutoff 0.5 (alpha test, as Falcor rasterizes this asset)',
    doubleSidedNameMarker: '.DoubleSided',
  },
};

function argumentsOf(argv) {
  const values = new Map(), licenses = [];
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i], value = argv[i + 1];
    if (!['--input', '--source-root', '--output', '--convention', '--license'].includes(key) || !value || value.startsWith('--')
      || (key !== '--license' && values.has(key))) throw Error('FBX_MATERIAL_MAP_ARGUMENTS');
    if (key === '--license') licenses.push(resolve(value)); else values.set(key, value);
    i++;
  }
  if (!values.has('--input') || !values.has('--source-root') || !values.has('--output'))
    throw Error('usage: apply-fbx-material-map --input blender-dir --source-root dir --output new-dir [--convention orca-metal-rough] [--license file]...');
  const convention = values.get('--convention') ?? 'orca-metal-rough';
  if (!CONVENTIONS[convention]) throw Error('FBX_MATERIAL_MAP_CONVENTION');
  return { input: resolve(values.get('--input')), sourceRoot: resolve(values.get('--source-root')),
    output: resolve(values.get('--output')), convention, licenses };
}

const inside = (root, path) => { const r = relative(root, path); return r && !r.startsWith('..') && !r.split(sep).includes('..'); };

/** Role of a texture file from its stem suffix (`Name_BaseColor.dds` -> baseColor). */
export function textureRole(file, convention) {
  const stem = basename(file, extname(file)), suffix = stem.slice(stem.lastIndexOf('_') + 1);
  const role = CONVENTIONS[convention].roles[suffix];
  if (!role) throw Error('FBX_MATERIAL_TEXTURE_ROLE_UNKNOWN: ' + basename(file));
  return role;
}

/** Alpha statistics of a decoded RGBA8 top level and the resulting glTF alpha mode. */
export function alphaDecision(pixels) {
  let opaque = 0, cut = 0, min = 255;
  for (let i = 3; i < pixels.length; i += 4) {
    const a = pixels[i];
    if (a === 255) opaque++;
    if (a < 128) cut++;
    if (a < min) min = a;
  }
  const texels = pixels.length / 4;
  return { texels, opaqueTexels: opaque, cutTexels: cut, minAlpha: min,
    alphaMode: opaque === texels ? 'OPAQUE' : 'MASK', fullyCut: cut === texels };
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  const convention = CONVENTIONS[config.convention];
  if (await exists(config.output)) throw Error('FBX_MATERIAL_MAP_OUTPUT_EXISTS');
  const sourceRoot = await realpath(config.sourceRoot);
  const gltfFile = await boundedRead(join(config.input, 'scene.gltf'), 128 * MiB);
  const reportFile = await boundedRead(join(config.input, 'blender-report.json'), 256 * MiB);
  const document = JSON.parse(gltfFile.bytes.toString('utf8')), report = JSON.parse(reportFile.bytes.toString('utf8'));
  if (report.format !== 'real-engine-blender-fbx-report') throw Error('FBX_MATERIAL_MAP_REPORT_FORMAT');
  if (document.images?.length || document.textures?.length) throw Error('FBX_MATERIAL_MAP_EXPORT_HAS_IMAGES');
  if (document.buffers?.length !== 1 || document.buffers[0].uri !== 'scene.bin') throw Error('FBX_MATERIAL_MAP_BUFFER_LAYOUT');

  // Texture files per material, deduplicated, one file per role.
  const files = new Map(), decisions = [], unsupported = [];
  const alphaCache = new Map();
  const imageIndex = file => {
    if (!files.has(file)) files.set(file, files.size);
    return files.get(file);
  };
  const materials = [];
  for (const [index, exported] of (document.materials ?? []).entries()) {
    const source = report.materials[exported.name];
    if (!source) throw Error('FBX_MATERIAL_MAP_MATERIAL_UNKNOWN: ' + exported.name);
    const roles = {};
    for (const texture of source.textures) {
      const file = await realpath(texture.file).catch(() => { throw Error('FBX_MATERIAL_MAP_TEXTURE_MISSING: ' + texture.file); });
      if (!inside(sourceRoot, file)) throw Error('FBX_MATERIAL_MAP_TEXTURE_OUTSIDE_SOURCE: ' + texture.file);
      if (!['.dds', '.png', '.tga'].includes(extname(file).toLowerCase())) throw Error('FBX_MATERIAL_MAP_TEXTURE_FORMAT: ' + basename(file));
      const role = textureRole(file, config.convention);
      if (roles[role] && roles[role] !== file) throw Error(`FBX_MATERIAL_MAP_ROLE_CONFLICT: ${exported.name} ${role}`);
      roles[role] = file;
    }
    const material = { name: exported.name, pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] } };
    const decision = { material: index, name: exported.name, blenderBlendMethod: source.blendMethod, textures: {} };
    if (roles.baseColor) {
      material.pbrMetallicRoughness.baseColorTexture = { index: imageIndex(roles.baseColor) };
      decision.textures.baseColor = relative(sourceRoot, roles.baseColor);
      if (!alphaCache.has(roles.baseColor)) {
        if (extname(roles.baseColor).toLowerCase() !== '.dds') throw Error('FBX_MATERIAL_MAP_ALPHA_SOURCE: ' + basename(roles.baseColor));
        const bytes = await boundedRead(roles.baseColor, 384 * MiB);
        alphaCache.set(roles.baseColor, alphaDecision(decodeLegacyDds(new Uint8Array(bytes.bytes.buffer, bytes.bytes.byteOffset, bytes.bytes.byteLength)).pixels));
      }
      decision.alpha = alphaCache.get(roles.baseColor);
    } else {
      const color = source.principled?.['Base Color']?.value;
      if (color) material.pbrMetallicRoughness.baseColorFactor = color.slice(0, 3).concat(1);
      decision.alpha = null;
    }
    material.alphaMode = decision.alpha?.alphaMode ?? 'OPAQUE';
    if (material.alphaMode === 'MASK') material.alphaCutoff = 0.5;
    if (decision.alpha?.fullyCut) unsupported.push({ material: exported.name,
      feature: 'every base color texel is below the 0.5 cutoff: the material is fully cut out (transparency/transmission are not represented)' });
    if (roles.occlusionRoughnessMetalness) {
      const texture = imageIndex(roles.occlusionRoughnessMetalness);
      Object.assign(material.pbrMetallicRoughness, { metallicFactor: 1, roughnessFactor: 1, metallicRoughnessTexture: { index: texture } });
      material.occlusionTexture = { index: texture, strength: 1 };
      decision.textures.occlusionRoughnessMetalness = relative(sourceRoot, roles.occlusionRoughnessMetalness);
    } else {
      // No packed texture: the FBX scalar values Blender imported are kept.
      const metallic = source.principled?.Metallic?.value, roughness = source.principled?.Roughness?.value;
      Object.assign(material.pbrMetallicRoughness, { metallicFactor: typeof metallic === 'number' ? metallic : 1,
        roughnessFactor: typeof roughness === 'number' ? roughness : 1 });
      decision.factorsFromFbx = { metallic, roughness };
    }
    if (roles.normal) {
      material.normalTexture = { index: imageIndex(roles.normal), scale: 1 };
      decision.textures.normal = relative(sourceRoot, roles.normal);
    }
    if (roles.emissive) {
      material.emissiveTexture = { index: imageIndex(roles.emissive) };
      material.emissiveFactor = [1, 1, 1];
      decision.textures.emissive = relative(sourceRoot, roles.emissive);
    }
    // FBX materials carry no sidedness flag (Blender exports all as double-sided);
    // the convention's name marker decides instead.
    material.doubleSided = exported.name.includes(convention.doubleSidedNameMarker);
    decision.doubleSided = material.doubleSided;
    decision.alphaMode = material.alphaMode;
    materials.push(material);
    decisions.push(decision);
  }

  const names = new Map();
  const images = [...files.keys()].map(file => {
    const name = basename(file);
    if (names.has(name) && names.get(name) !== file) throw Error('FBX_MATERIAL_MAP_TEXTURE_NAME_COLLISION: ' + name);
    names.set(name, file);
    return { name: basename(file, extname(file)), uri: 'Textures/' + encodeURIComponent(name) };
  });
  const output = { ...document, materials, images, samplers: [{ magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 }],
    textures: images.map((_, i) => ({ source: i, sampler: 0 })) };
  const dropped = (document.extensionsUsed ?? []).filter(name => name !== 'KHR_lights_punctual');
  const lights = (document.extensionsUsed ?? []).includes('KHR_lights_punctual');
  if (lights) output.extensionsUsed = ['KHR_lights_punctual']; else delete output.extensionsUsed;
  // Punctual lights never change geometry or materials, so they are optional.
  delete output.extensionsRequired;
  for (const name of dropped) unsupported.push({ feature: 'exporter extension removed with the rewritten materials: ' + name });
  unsupported.push({ feature: 'second UV set (TEXCOORD_1) is exported but no material reads it' });
  unsupported.push({ feature: 'camera animation is not in the glTF; the sampled path is camera-path.json' });

  await mkdir(dirname(config.output), { recursive: true });
  const temporary = await mkdtemp(join(dirname(config.output), '.' + basename(config.output) + '.tmp-'));
  try {
    const gltfBytes = Buffer.from(JSON.stringify(output) + '\n');
    await writeFile(join(temporary, 'scene.gltf'), gltfBytes, { flag: 'wx' });
    await link(await realpath(join(config.input, 'scene.bin')), join(temporary, 'scene.bin'));
    await mkdir(join(temporary, 'Textures'));
    for (const [name, file] of names) await link(file, join(temporary, 'Textures', name));
    const licenses = [];
    for (const file of config.licenses) {
      const bytes = await boundedRead(file, 4 * MiB);
      await copyFile(bytes.canonical, join(temporary, basename(file)));
      licenses.push({ file: basename(file), bytes: bytes.bytes.length, sha256: sha256(bytes.bytes) });
    }
    const camera = report.cameras?.[0];
    if (camera) await writeFile(join(temporary, 'camera-path.json'), JSON.stringify({ version: 1, format: 'real-engine-camera-path',
      space: 'glTF (Y-up, meters as exported)', source: report.source, name: camera.name, fps: camera.fps,
      frameRange: camera.frameRange, verticalFovRadians: camera.verticalFovRadians, clip: camera.clip, samples: camera.path }) + '\n', { flag: 'wx' });
    const modes = decisions.reduce((count, d) => ({ ...count, [d.alphaMode]: (count[d.alphaMode] ?? 0) + 1 }), {});
    const manifest = { version: 1, format: 'real-engine-fbx-gltf-conversion', convention: config.convention, conventionRules: convention,
      source: report.source, blender: report.blender, blenderReport: { file: 'blender-report.json', sha256: sha256(reportFile.bytes) },
      gltf: { file: 'scene.gltf', bytes: gltfBytes.length, sha256: sha256(gltfBytes) },
      counts: { objects: report.objects, meshes: report.meshCount, primitives: output.meshes.reduce((n, m) => n + m.primitives.length, 0),
        nodes: output.nodes.length, uniqueTriangles: report.uniqueTriangles, instanceExpandedTriangles: report.instanceExpandedTriangles,
        sharedMeshes: report.sharedMeshes, materials: materials.length, images: images.length, alphaModes: modes,
        lights: report.lights.length, cameras: report.cameras.length },
      units: report.units, boundsGltf: report.boundsGltf, normalConvention: convention.normalConvention,
      licenses, materials: decisions, unsupported };
    await writeFile(join(temporary, 'conversion.json'), JSON.stringify(manifest, null, 1) + '\n', { flag: 'wx' });
    await copyFile(reportFile.canonical, join(temporary, 'blender-report.json'));
    await writeFile(join(temporary, 'manifest.json'), JSON.stringify({ version: 1, format: 'real-engine-fbx-gltf-conversion-complete',
      conversion: 'conversion.json', gltf: manifest.gltf }) + '\n', { flag: 'wx' });
    await publishDirectory(temporary, config.output, { hardLinkFiles: true });
    console.log(JSON.stringify({ done: config.output, ...manifest.counts, unsupported: unsupported.length }));
  } finally {
    await removeOwnTemporary(temporary, dirname(config.output), config.output);
  }
}

const invokedDirectly = () => { try { return realpathSync(process.argv[1]) === fileURLToPath(import.meta.url); } catch { return false; } };
if (invokedDirectly()) main().catch(error => { console.error(error instanceof Error ? error.stack ?? error.message : String(error)); process.exitCode = 1; });
