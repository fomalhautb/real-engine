#!/usr/bin/env node
// Builds texture stream packages for every image a glTF material references,
// using the deterministic builtin encoder. Image roles come from the material
// slots; normal maps stay on the RGBA8 linear reference path, all other images
// are BC7 in the color space their slots require (conflicts fail). Output:
// <output>/manifest.json (set summary with roles, sizes and quality), one
// <output>/image-<n>/ package per image, and an optional copied license file.
import { copyFile, mkdir, mkdtemp } from 'node:fs/promises';
import { basename, dirname, extname, join, relative, resolve, sep } from 'node:path';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';
import { BUILTIN_DEFAULTS, boundedRead, buildTextureInto, createTextureContext, sha256 } from './texture-package.mjs';
import { TextureCodecPool, defaultWorkerCount } from './texture-codec-pool.mjs';
import { writeFile } from 'node:fs/promises';

const MiB = 1024 * 1024;
const SLOT_ROLES = [
  [material => material.pbrMetallicRoughness?.baseColorTexture, 'baseColor:srgb'],
  [material => material.pbrMetallicRoughness?.metallicRoughnessTexture, 'metallicRoughness:linear'],
  [material => material.normalTexture, 'normal:linear'],
  [material => material.occlusionTexture, 'occlusion:linear'],
  [material => material.emissiveTexture, 'emissive:srgb'],
];

function argumentsOf(argv) {
  const keys = ['--input', '--output', '--workers', '--bc7-quality', '--bc7-early-out', '--fallback-size', '--license', '--images'];
  const values = new Map();
  for (let i = 0; i < argv.length; i++) {
    if (!keys.includes(argv[i]) || values.has(argv[i]) || !argv[i + 1] || argv[i + 1].startsWith('--')) throw Error('GLTF_TEXTURES_ARGUMENTS');
    values.set(argv[i], argv[++i]);
  }
  if (!values.has('--input') || !values.has('--output')) throw Error('usage: build-gltf-textures --input scene.gltf --output new-directory [--workers n] [--bc7-quality 0|1|2] [--bc7-early-out sse] [--fallback-size 64] [--license file] [--images 0,1,...]');
  const workers = Number(values.get('--workers') ?? defaultWorkerCount());
  const bc7Quality = Number(values.get('--bc7-quality') ?? BUILTIN_DEFAULTS.bc7Quality);
  const bc7EarlyOutError = Number(values.get('--bc7-early-out') ?? BUILTIN_DEFAULTS.bc7EarlyOutError);
  const fallbackSize = Number(values.get('--fallback-size') ?? 64);
  if (!Number.isSafeInteger(workers) || workers < 0 || workers > 64) throw Error('GLTF_TEXTURES_WORKERS');
  if (![0, 1, 2].includes(bc7Quality)) throw Error('GLTF_TEXTURES_BC7_QUALITY');
  if (!Number.isSafeInteger(bc7EarlyOutError) || bc7EarlyOutError < 0) throw Error('GLTF_TEXTURES_BC7_EARLY_OUT');
  if (!Number.isSafeInteger(fallbackSize) || fallbackSize < 1 || fallbackSize > 8192) throw Error('GLTF_TEXTURES_FALLBACK_SIZE');
  const images = values.has('--images') ? values.get('--images').split(',').map(Number) : null;
  if (images && images.some(i => !Number.isSafeInteger(i) || i < 0)) throw Error('GLTF_TEXTURES_IMAGES');
  return { input: resolve(values.get('--input')), output: resolve(values.get('--output')),
    license: values.has('--license') ? resolve(values.get('--license')) : null,
    workers, bc7Quality, bc7EarlyOutError, fallbackSize, images };
}

/** Image index -> sorted role list, from material texture slots only. */
export function imageRoles(document) {
  const roles = new Map();
  for (const material of document.materials ?? []) for (const [slot, role] of SLOT_ROLES) {
    const info = slot(material);
    if (!info) continue;
    const texture = document.textures?.[info.index];
    if (!texture || !Number.isSafeInteger(texture.source)) throw Error('GLTF_TEXTURES_TEXTURE_SOURCE');
    if (!roles.has(texture.source)) roles.set(texture.source, new Set());
    roles.get(texture.source).add(role);
  }
  return new Map([...roles].sort((a, b) => a[0] - b[0]).map(([image, set]) => [image, [...set].sort()]));
}

/** Format and color space for one image; mixed sRGB/linear use is rejected. */
export function imageEncoding(roles) {
  if (roles.includes('normal:linear')) {
    if (roles.length !== 1) throw Error('GLTF_TEXTURES_NORMAL_SHARED: ' + roles.join(','));
    return { format: 'rgba8', colorSpace: 'linear', path: 'reference' };
  }
  const spaces = new Set(roles.map(role => role.split(':')[1]));
  if (spaces.size !== 1) throw Error('GLTF_TEXTURES_COLOR_SPACE_CONFLICT: ' + roles.join(','));
  return { format: 'bc7', colorSpace: [...spaces][0], path: 'gpu' };
}

function imagePath(base, uri) {
  if (typeof uri !== 'string' || /^[a-z][a-z0-9+.-]*:/i.test(uri)) throw Error('GLTF_TEXTURES_IMAGE_URI');
  const path = resolve(base, decodeURIComponent(uri));
  const inside = relative(base, path);
  if (!inside || inside.startsWith('..') || inside.split(sep).includes('..')) throw Error('GLTF_TEXTURES_IMAGE_OUTSIDE_SOURCE');
  return path;
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) throw Error('GLTF_TEXTURES_OUTPUT_EXISTS');
  const source = await boundedRead(config.input, 64 * MiB);
  const document = JSON.parse(source.bytes.toString('utf8'));
  const base = dirname(source.canonical), roles = imageRoles(document);
  const selected = [...roles.keys()].filter(image => !config.images || config.images.includes(image));
  if (config.images && selected.length !== config.images.length) throw Error('GLTF_TEXTURES_IMAGES_NOT_REFERENCED');
  // Resolve every input before encoding anything, so unsupported content fails fast.
  const plan = selected.map(image => {
    const entry = document.images?.[image];
    if (!entry || entry.bufferView !== undefined || entry.uri === undefined) throw Error('GLTF_TEXTURES_EMBEDDED_IMAGE_UNSUPPORTED: ' + image);
    const input = imagePath(base, entry.uri), extension = extname(input).toLowerCase();
    if (!['.png', '.tga'].includes(extension)) throw Error(`GLTF_TEXTURES_IMAGE_FORMAT_UNSUPPORTED: ${image} ${entry.uri}`);
    return { image, uri: entry.uri, input, extension, roles: roles.get(image), ...imageEncoding(roles.get(image)) };
  });
  await mkdir(dirname(config.output), { recursive: true });
  const temporary = await mkdtemp(join(dirname(config.output), '.' + basename(config.output) + '.tmp-'));
  const context = await createTextureContext();
  const pool = config.workers > 0 ? new TextureCodecPool(config.workers) : null;
  const started = Date.now(), images = [];
  try {
    for (const item of plan) {
      const directory = join(temporary, `image-${item.image}`);
      await mkdir(directory);
      const result = await buildTextureInto({ input: item.input, extension: item.extension, id: `image-${item.image}`,
        fallbackSize: config.fallbackSize, format: item.format, colorSpace: item.colorSpace, encoder: 'builtin',
        bc7Quality: config.bc7Quality, bc7EarlyOutError: config.bc7EarlyOutError }, directory, context, pool);
      const manifestBytes = await boundedRead(join(directory, 'manifest.json'), MiB);
      const quality = result.quality ? { psnrRgbTop: result.quality.top.psnrRgb, rmseRgbTop: result.quality.top.rmseRgb,
        maxAbsTop: result.quality.top.maxAbs, psnrAlphaTop: result.quality.top.psnrAlpha, modesTop: result.quality.top.modes,
        psnrRgbAllLevels: result.quality.allLevels.psnrRgb, maxAbsAllLevels: result.quality.allLevels.maxAbs } : null;
      images.push({ image: item.image, uri: item.uri, roles: item.roles, path: item.path, format: result.format,
        colorSpace: result.colorSpace, width: result.width, height: result.height, levels: result.levels,
        fallbackMip: result.fallbackMip, nativeBytes: result.nativeBytes, fallbackBytes: result.fallbackBytes,
        packageBytes: result.outputBytes, manifest: `image-${item.image}/manifest.json`,
        manifestSha256: sha256(manifestBytes.bytes), quality });
      console.log(JSON.stringify({ progress: `${images.length}/${plan.length}`, image: item.image, uri: item.uri,
        format: result.format, colorSpace: result.colorSpace, size: `${result.width}x${result.height}`, quality, run: result.run }));
    }
    let license = null;
    if (config.license) {
      const bytes = await boundedRead(config.license, 4 * MiB);
      await copyFile(bytes.canonical, join(temporary, basename(config.license)));
      license = { file: basename(config.license), bytes: bytes.bytes.length, sha256: sha256(bytes.bytes) };
    }
    const sum = (list, key) => list.reduce((total, item) => total + item[key], 0);
    const bc7 = images.filter(item => item.format === 'bc7'), rgba8 = images.filter(item => item.format === 'rgba8');
    const psnr = bc7.map(item => item.quality.psnrRgbTop).filter(Number.isFinite);
    const manifest = { version: 1, format: 'real-engine-gltf-texture-set',
      source: { file: basename(source.canonical), bytes: source.bytes.length, sha256: sha256(source.bytes) },
      encoder: { name: 'real-engine builtin', node: process.version, bc7Quality: config.bc7Quality,
        bc7EarlyOutError: config.bc7EarlyOutError, fallbackSize: config.fallbackSize,
        normals: 'RGBA8 linear reference path (not block compressed)', otherImages: 'BC7, slot color space' },
      license, complete: !config.images, images,
      totals: { images: images.length, bc7Images: bc7.length, rgba8Images: rgba8.length,
        bc7NativeBytes: sum(bc7, 'nativeBytes'), rgba8NativeBytes: sum(rgba8, 'nativeBytes'),
        nativeBytes: sum(images, 'nativeBytes'), fallbackBytes: sum(images, 'fallbackBytes'), packageBytes: sum(images, 'packageBytes'),
        bc7PsnrRgbTop: psnr.length ? { min: Math.min(...psnr), max: Math.max(...psnr), mean: psnr.reduce((a, b) => a + b, 0) / psnr.length } : null,
        bc7MaxAbsTop: bc7.reduce((max, item) => max.map((v, c) => Math.max(v, item.quality.maxAbsTop[c])), [0, 0, 0, 0]) } };
    await writeFile(join(temporary, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
    await publishDirectory(temporary, config.output, { hardLinkFiles: true });
    console.log(JSON.stringify({ done: config.output, elapsedMs: Date.now() - started, workers: pool?.size ?? 0, totals: manifest.totals }));
  } finally {
    await pool?.close();
    await context.close();
    await removeOwnTemporary(temporary, dirname(config.output), config.output);
  }
}

main().catch(error => { console.error(error instanceof Error ? error.stack ?? error.message : String(error)); process.exitCode = 1; });
