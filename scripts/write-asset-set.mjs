#!/usr/bin/env node
// Writes an asset-set description that binds already-built packages of one
// source scene together, after checking that they really derive from the same
// source document (and the same geometry package where that applies). Capture
// scripts and the large-scene demo read the server roots from it.
//   node scripts/write-asset-set.mjs --name intel-sponza-main --geometry dir --textures dir
//     [--query dir] [--uv file.json] [--license file] [--camera-path file.json] --output new-directory
// A camera path (format real-engine-camera-path: samples of position, forward
// and up in glTF space) is copied into the set as camera-path.json.
//
// Compose mode binds finished asset sets of different sources that share one
// glTF scene frame (for example the Intel Sponza building and its add-ons):
//   node scripts/write-asset-set.mjs --compose main=path/asset-set.json --compose ivy=path/asset-set.json
//     --name intel-sponza-composed --license file [--camera-component main] --output new-directory
// It re-reads and hashes every component set, re-hashes every package manifest
// (geometry, textures, fine query, UV audit, camera path) and requires the
// hashes the component recorded, one source hash per component, distinct
// sources and names, and one license notice shared by every component. The
// result has format real-engine-asset-composition: single-set readers reject it.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';
import { parseCameraPath } from '../src/camera-path.ts';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const FLAGS = { '--name': 'name', '--geometry': 'geometry', '--textures': 'textures', '--query': 'query', '--uv': 'uv',
  '--license': 'license', '--output': 'output', '--artifacts-root': 'artifactsRoot', '--camera-path': 'cameraPath',
  '--compose': 'compose', '--camera-component': 'cameraComponent' };
const MiB = 1024 * 1024;
const COMPOSE_LIMITS = Object.freeze({ components: 64, setBytes: MiB, geometryManifestBytes: 64 * MiB,
  textureManifestBytes: 4 * MiB, queryManifestBytes: 64 * MiB, uvAuditBytes: 8 * MiB, cameraPathBytes: 8 * MiB, licenseBytes: 4 * MiB });

const options = {}, composes = [];
for (let i = 2; i < process.argv.length; i++) {
  const key = FLAGS[process.argv[i]], value = process.argv[i + 1];
  // --compose is the only repeatable flag.
  if (key === 'compose' && value && !value.startsWith('--')) { composes.push(value); i++; continue; }
  if (!key || options[key] !== undefined || !value || value.startsWith('--')) throw Error('ASSET_SET_ARGUMENTS: ' + process.argv[i]);
  options[key] = value; i++;
}
if (composes.length) {
  for (const flag of ['--geometry', '--textures', '--query', '--uv', '--camera-path']) {
    if (options[FLAGS[flag]] !== undefined) throw Error('ASSET_SET_COMPOSE_ARGUMENTS: ' + flag + ' belongs to single-source mode');
  }
  for (const key of ['name', 'license', 'output']) if (!options[key]) throw Error('ASSET_SET_REQUIRED: --' + key);
  if (composes.length > COMPOSE_LIMITS.components) throw Error('ASSET_SET_COMPOSE_COMPONENT_BUDGET');
} else {
  if (options.cameraComponent !== undefined) throw Error('ASSET_SET_ARGUMENTS: --camera-component');
  for (const key of ['name', 'geometry', 'textures', 'output']) if (!options[key]) throw Error('ASSET_SET_REQUIRED: --' + key);
}
if (!/^[a-z0-9][a-z0-9.-]{0,63}$/.test(options.name)) throw Error('ASSET_SET_NAME');
const artifacts = await realpath(resolve(options.artifactsRoot ?? join(ROOT, 'artifacts')));
const output = resolve(options.output);
if (await exists(output)) throw Error('ASSET_SET_OUTPUT_EXISTS');

/** Server path (served by the dev server under /artifacts/) of a local package. */
async function serverPath(path, directory) {
  const real = await realpath(resolve(path)), inside = relative(artifacts, real);
  if (!inside || inside.startsWith('..') || inside.split(sep).includes('..')) throw Error('ASSET_SET_OUTSIDE_ARTIFACTS: ' + path);
  return '/artifacts/' + inside.split(sep).join('/') + (directory ? '/' : '');
}
async function manifestOf(path, file = 'manifest.json') {
  const bytes = await readFile(join(resolve(path), file));
  return { bytes, sha256: sha256(bytes), json: JSON.parse(bytes.toString('utf8')) };
}

async function writeSingle() {
  const geometry = await manifestOf(options.geometry), textures = await manifestOf(options.textures);
  if (geometry.json.format !== 'real-engine-gltf-geometry') throw Error('ASSET_SET_GEOMETRY_FORMAT');
  if (textures.json.format !== 'real-engine-gltf-texture-set' || !textures.json.complete) throw Error('ASSET_SET_TEXTURE_FORMAT');
  const sourceSha256 = geometry.json.source.sha256;
  if (textures.json.source.sha256 !== sourceSha256) throw Error('ASSET_SET_TEXTURE_SOURCE_MISMATCH');
  const roots = { geometryRoot: await serverPath(options.geometry, true), textureGpuRoot: await serverPath(options.textures, true) };
  roots.textureReferenceRoot = roots.textureGpuRoot;
  const packages = {
    geometry: { root: roots.geometryRoot, manifestSha256: geometry.sha256, assetIdentity: geometry.json.assetIdentity },
    textures: { root: roots.textureGpuRoot, manifestSha256: textures.sha256, images: textures.json.totals.images,
      bc7Images: textures.json.totals.bc7Images, rgba8Images: textures.json.totals.rgba8Images, nativeBytes: textures.json.totals.nativeBytes },
  };
  if (options.query) {
    const query = await manifestOf(options.query);
    if (query.json.format !== 'real-engine-gltf-query-proxies' || query.json.sourcePackage.assetIdentity !== geometry.json.assetIdentity
      || query.json.sourcePackage.manifestSha256 !== geometry.sha256 || query.json.sourcePackage.sourceSha256 !== sourceSha256) throw Error('ASSET_SET_QUERY_PACKAGE_MISMATCH');
    roots.fineQueryRoot = await serverPath(options.query, true);
    packages.fineQuery = { root: roots.fineQueryRoot, manifestSha256: query.sha256, maxError: query.json.config.maxError, totals: query.json.totals };
  }
  if (options.uv) {
    const bytes = await readFile(resolve(options.uv)), uv = JSON.parse(bytes.toString('utf8'));
    if (uv.format !== 'real-engine-gltf-uv-demand-audit' || uv.sourceSha256 !== sourceSha256) throw Error('ASSET_SET_UV_SOURCE_MISMATCH');
    if (uv.geometryCheck && (uv.geometryCheck.manifestSha256 !== geometry.sha256 || !uv.geometryCheck.identical)) throw Error('ASSET_SET_UV_GEOMETRY_MISMATCH');
    roots.uvAuditUrl = await serverPath(options.uv, false);
    packages.uvAudit = { url: roots.uvAuditUrl, sha256: sha256(bytes), summary: uv.summary, geometryChecked: Boolean(uv.geometryCheck) };
  }
  let cameraPath = null;
  if (options.cameraPath) {
    // The same validation the demo applies when it loads the path.
    const bytes = await readFile(resolve(options.cameraPath));
    let path;
    try { path = parseCameraPath(JSON.parse(bytes.toString('utf8'))); } catch (error) { throw Error('ASSET_SET_CAMERA_PATH: ' + error.message); }
    cameraPath = { bytes, samples: path.samples.length, duration: path.duration };
  }
  await mkdir(dirname(output), { recursive: true });
  if (cameraPath) {
    // The set directory does not exist yet: resolve its parent, then append its name.
    const inside = relative(artifacts, join(await realpath(dirname(output)), basename(output)));
    if (!inside || inside.startsWith('..') || inside.split(sep).includes('..')) throw Error('ASSET_SET_OUTSIDE_ARTIFACTS: ' + output);
    roots.cameraPathUrl = '/artifacts/' + inside.split(sep).join('/') + '/camera-path.json';
  }
  const temporary = await mkdtemp(join(dirname(output), `.${basename(output)}.tmp-`));
  try {
    let license = null;
    if (options.license) {
      const bytes = await readFile(resolve(options.license));
      await copyFile(resolve(options.license), join(temporary, basename(options.license)));
      license = { file: basename(options.license), bytes: bytes.length, sha256: sha256(bytes) };
    }
    if (cameraPath) {
      await writeFile(join(temporary, 'camera-path.json'), cameraPath.bytes, { flag: 'wx' });
      packages.cameraPath = { url: roots.cameraPathUrl, sha256: sha256(cameraPath.bytes), samples: cameraPath.samples, durationSeconds: cameraPath.duration };
    }
    const set = { version: 1, format: 'real-engine-asset-set', name: options.name,
      source: { file: geometry.json.source.file, bytes: geometry.json.source.bytes, sha256: sourceSha256 },
      roots, packages, license,
      omitted: geometry.json.primitives.filter(p => p.route !== 'paged').map(p => ({ meshIndex: p.meshIndex, primitiveIndex: p.primitiveIndex,
        materialIndex: p.materialIndex, coverage: p.coverage, route: p.route })) };
    const text = JSON.stringify(set, null, 2) + '\n';
    await writeFile(join(temporary, 'asset-set.json'), text, { flag: 'wx' });
    await writeFile(join(temporary, 'manifest.json'), text, { flag: 'wx' });
    await publishDirectory(temporary, output);
    process.stdout.write(JSON.stringify({ output, roots }) + '\n');
  } finally { await removeOwnTemporary(temporary, dirname(output), output); }
}

/** Reads a whole regular file of at most `limit` bytes. */
async function boundedFile(path, limit, label) {
  const info = await stat(path).catch(error => { throw Error(`ASSET_SET_COMPOSE_MISSING: ${label} (${error.code ?? error.message})`); });
  if (!info.isFile() || info.size < 1 || info.size > limit) throw Error(`ASSET_SET_COMPOSE_BYTES: ${label} (${info.size} bytes, limit ${limit})`);
  const bytes = await readFile(path);
  if (bytes.length !== info.size) throw Error('ASSET_SET_COMPOSE_CHANGED: ' + label);
  return bytes;
}
/** Local file of a server path recorded by a component set, confined to the artifacts root. */
async function localOf(url, directory, label) {
  if (typeof url !== 'string' || !url.startsWith('/artifacts/') || url.includes('..') || url.includes('//') || url.includes('\\')
    || url.endsWith('/') !== directory) throw Error('ASSET_SET_COMPOSE_URL: ' + label);
  const real = await realpath(join(artifacts, ...url.slice('/artifacts/'.length).split('/').filter(Boolean)))
    .catch(error => { throw Error(`ASSET_SET_COMPOSE_MISSING: ${label} (${error.code ?? error.message})`); });
  const inside = relative(artifacts, real);
  if (!inside || isAbsolute(inside) || inside === '..' || inside.startsWith('..' + sep)) throw Error('ASSET_SET_OUTSIDE_ARTIFACTS: ' + label);
  return real;
}
async function hashedJson(path, limit, expected, label) {
  const bytes = await boundedFile(path, limit, label), hash = sha256(bytes);
  if (expected !== undefined && hash !== expected) throw Error('ASSET_SET_COMPOSE_STALE_PACKAGE: ' + label);
  let json;
  try { json = JSON.parse(bytes.toString('utf8')); } catch { throw Error('ASSET_SET_COMPOSE_JSON: ' + label); }
  if (json === null || typeof json !== 'object' || Array.isArray(json)) throw Error('ASSET_SET_COMPOSE_JSON: ' + label);
  return { bytes, sha256: hash, json };
}
const GEOMETRY_TOTALS = ['sourceTriangles', 'triangles', 'removedTriangles', 'pagedPrimitives', 'sourcePrimitives', 'clusters', 'pages',
  'encodedPageBytes', 'decodedPageBytes', 'rootEncodedBytes', 'rootDecodedBytes', 'proxyVertices', 'proxyTriangles', 'proxyBytes'];
const TEXTURE_TOTALS = ['images', 'bc7Images', 'rgba8Images', 'nativeBytes'];
/** The named counters of a manifest's totals; each must be a non-negative safe integer. */
function counts(object, keys, label) {
  const result = {};
  for (const key of keys) {
    const value = object?.[key];
    if (!Number.isSafeInteger(value) || value < 0) throw Error('ASSET_SET_COMPOSE_TOTALS: ' + label + ' ' + key);
    result[key] = value;
  }
  return result;
}

/** Re-verifies one finished single-source asset set and summarises it. */
async function component(entry, license) {
  const match = /^([a-z][a-z0-9-]{0,31})=(.+)$/.exec(entry);
  if (!match) throw Error('ASSET_SET_COMPOSE_COMPONENT: ' + entry + ' (expected name=path/asset-set.json)');
  const [, name, path] = match;
  const file = await realpath(resolve(path)).catch(error => { throw Error(`ASSET_SET_COMPOSE_MISSING: ${name} asset set (${error.code ?? error.message})`); });
  if (basename(file) !== 'asset-set.json') throw Error('ASSET_SET_COMPOSE_COMPONENT: ' + name + ' must name an asset-set.json');
  const url = await serverPath(file, false);
  const set = await hashedJson(file, COMPOSE_LIMITS.setBytes, undefined, `${name} asset set`);
  const json = set.json;
  if (json.format !== 'real-engine-asset-set' || json.version !== 1) throw Error('ASSET_SET_COMPOSE_COMPONENT_FORMAT: ' + name);
  // A published set carries manifest.json (its completion marker) with the same bytes.
  const marker = await boundedFile(join(dirname(file), 'manifest.json'), COMPOSE_LIMITS.setBytes, `${name} manifest`).catch(() => null);
  if (!marker || !marker.equals(set.bytes)) throw Error('ASSET_SET_COMPOSE_INCOMPLETE: ' + name);
  const roots = json.roots ?? {}, packages = json.packages ?? {}, source = json.source ?? {};
  const sourceSha256 = source.sha256;
  if (typeof sourceSha256 !== 'string' || !/^[0-9a-f]{64}$/.test(sourceSha256)) throw Error('ASSET_SET_COMPOSE_SOURCE: ' + name);
  if (!packages.geometry || !packages.textures || roots.geometryRoot !== packages.geometry.root || roots.textureGpuRoot !== packages.textures.root
    || roots.textureReferenceRoot !== roots.textureGpuRoot) throw Error('ASSET_SET_COMPOSE_ROOTS: ' + name);
  const geometry = await hashedJson(join(await localOf(packages.geometry.root, true, `${name} geometry`), 'manifest.json'),
    COMPOSE_LIMITS.geometryManifestBytes, packages.geometry.manifestSha256, `${name} geometry`);
  if (geometry.json.format !== 'real-engine-gltf-geometry' || geometry.json.assetIdentity !== packages.geometry.assetIdentity
    || geometry.json.source?.sha256 !== sourceSha256 || geometry.json.source.file !== source.file || geometry.json.source.bytes !== source.bytes) {
    throw Error('ASSET_SET_COMPOSE_SOURCE_MISMATCH: ' + name + ' geometry');
  }
  const textures = await hashedJson(join(await localOf(packages.textures.root, true, `${name} textures`), 'manifest.json'),
    COMPOSE_LIMITS.textureManifestBytes, packages.textures.manifestSha256, `${name} textures`);
  if (textures.json.format !== 'real-engine-gltf-texture-set' || !textures.json.complete || textures.json.source?.sha256 !== sourceSha256) {
    throw Error('ASSET_SET_COMPOSE_SOURCE_MISMATCH: ' + name + ' textures');
  }
  const summary = {
    geometry: { root: packages.geometry.root, manifestSha256: geometry.sha256, assetIdentity: geometry.json.assetIdentity,
      groupSize: geometry.json.config?.geometryBuild?.groupSize ?? null, removeZeroAreaTriangles: geometry.json.config?.removeZeroAreaTriangles ?? null,
      totals: counts(geometry.json.totals, GEOMETRY_TOTALS, `${name} geometry`) },
    textures: { root: packages.textures.root, manifestSha256: textures.sha256, ...counts(textures.json.totals, TEXTURE_TOTALS, `${name} textures`),
      channelPolicy: textures.json.encoder?.channelPolicy ?? null },
  };
  if (packages.fineQuery || roots.fineQueryRoot !== undefined) {
    if (!packages.fineQuery || roots.fineQueryRoot !== packages.fineQuery.root) throw Error('ASSET_SET_COMPOSE_ROOTS: ' + name + ' fineQuery');
    const query = await hashedJson(join(await localOf(packages.fineQuery.root, true, `${name} fineQuery`), 'manifest.json'),
      COMPOSE_LIMITS.queryManifestBytes, packages.fineQuery.manifestSha256, `${name} fineQuery`);
    const from = query.json.sourcePackage ?? {};
    if (query.json.format !== 'real-engine-gltf-query-proxies' || from.manifestSha256 !== geometry.sha256
      || from.assetIdentity !== geometry.json.assetIdentity || from.sourceSha256 !== sourceSha256) throw Error('ASSET_SET_COMPOSE_SOURCE_MISMATCH: ' + name + ' fineQuery');
    summary.fineQuery = { root: packages.fineQuery.root, manifestSha256: query.sha256, maxError: query.json.config?.maxError ?? null, totals: query.json.totals };
  }
  // The scene loader needs the UV demand audit, so a composed component must carry one.
  if (!packages.uvAudit || roots.uvAuditUrl !== packages.uvAudit.url) throw Error('ASSET_SET_COMPOSE_UV_REQUIRED: ' + name);
  const uv = await hashedJson(await localOf(packages.uvAudit.url, false, `${name} uvAudit`), COMPOSE_LIMITS.uvAuditBytes, packages.uvAudit.sha256, `${name} uvAudit`);
  if (uv.json.format !== 'real-engine-gltf-uv-demand-audit' || uv.json.sourceSha256 !== sourceSha256
    || (uv.json.geometryCheck && (uv.json.geometryCheck.manifestSha256 !== geometry.sha256 || !uv.json.geometryCheck.identical))) {
    throw Error('ASSET_SET_COMPOSE_SOURCE_MISMATCH: ' + name + ' uvAudit');
  }
  summary.uvAudit = { url: packages.uvAudit.url, sha256: uv.sha256, summary: uv.json.summary, geometryChecked: Boolean(uv.json.geometryCheck) };
  if (packages.cameraPath || roots.cameraPathUrl !== undefined) {
    if (!packages.cameraPath || roots.cameraPathUrl !== packages.cameraPath.url) throw Error('ASSET_SET_COMPOSE_ROOTS: ' + name + ' cameraPath');
    const bytes = await boundedFile(await localOf(packages.cameraPath.url, false, `${name} cameraPath`), COMPOSE_LIMITS.cameraPathBytes, `${name} cameraPath`);
    if (sha256(bytes) !== packages.cameraPath.sha256) throw Error('ASSET_SET_COMPOSE_STALE_PACKAGE: ' + name + ' cameraPath');
    let parsed;
    try { parsed = parseCameraPath(JSON.parse(bytes.toString('utf8'))); } catch (error) { throw Error('ASSET_SET_CAMERA_PATH: ' + name + ' ' + error.message); }
    if (parsed.samples.length !== packages.cameraPath.samples || parsed.duration !== packages.cameraPath.durationSeconds) {
      throw Error('ASSET_SET_COMPOSE_STALE_PACKAGE: ' + name + ' cameraPath samples');
    }
    summary.cameraPath = { url: packages.cameraPath.url, sha256: packages.cameraPath.sha256, samples: parsed.samples.length, durationSeconds: parsed.duration };
  }
  // Every component carries the composition's license notice, byte for byte.
  if (!json.license || typeof json.license.file !== 'string' || basename(json.license.file) !== json.license.file) throw Error('ASSET_SET_COMPOSE_LICENSE: ' + name + ' has no license file');
  const notice = await boundedFile(join(dirname(file), json.license.file), COMPOSE_LIMITS.licenseBytes, `${name} license`);
  if (notice.length !== json.license.bytes || sha256(notice) !== json.license.sha256) throw Error('ASSET_SET_COMPOSE_LICENSE: ' + name + ' license file changed');
  if (json.license.sha256 !== license.sha256) throw Error('ASSET_SET_COMPOSE_LICENSE_MISMATCH: ' + name);
  if (!Array.isArray(json.omitted)) throw Error('ASSET_SET_COMPOSE_OMITTED: ' + name);
  return { name, setName: json.name, assetSet: { url, bytes: set.bytes.length, sha256: set.sha256 },
    source: { file: source.file, bytes: source.bytes, sha256: sourceSha256 }, packages: summary,
    license: { file: json.license.file, bytes: json.license.bytes, sha256: json.license.sha256 }, omitted: json.omitted };
}

async function writeComposition() {
  const licenseBytes = await boundedFile(resolve(options.license), COMPOSE_LIMITS.licenseBytes, 'license');
  const license = { file: basename(options.license), bytes: licenseBytes.length, sha256: sha256(licenseBytes) };
  if (['asset-set.json', 'manifest.json'].includes(license.file)) throw Error('ASSET_SET_COMPOSE_LICENSE: reserved file name ' + license.file);
  const components = [], names = new Set(), sources = new Set();
  for (const entry of composes) {
    const item = await component(entry, license);
    if (names.has(item.name)) throw Error('ASSET_SET_COMPOSE_DUPLICATE_NAME: ' + item.name);
    if (sources.has(item.source.sha256)) throw Error('ASSET_SET_COMPOSE_DUPLICATE_SOURCE: ' + item.name);
    names.add(item.name); sources.add(item.source.sha256);
    components.push(item);
  }
  const withCamera = components.filter(item => item.packages.cameraPath);
  let camera = null;
  if (options.cameraComponent !== undefined) {
    camera = withCamera.find(item => item.name === options.cameraComponent);
    if (!camera) throw Error('ASSET_SET_COMPOSE_CAMERA: ' + options.cameraComponent + ' has no camera path');
  } else if (withCamera.length > 1) throw Error('ASSET_SET_COMPOSE_CAMERA_AMBIGUOUS: ' + withCamera.map(item => item.name).join(','));
  else camera = withCamera[0] ?? null;
  const total = (keys, read) => Object.fromEntries(keys.map(key => [key, components.reduce((sum, item) => sum + read(item)[key], 0)]));
  const composition = { version: 1, format: 'real-engine-asset-composition', name: options.name,
    frame: { componentTransform: 'identity',
      note: 'each component keeps its own glTF nodes, transforms and materials and gets no extra transform; that the sources share one scene space is declared by the composer, not checked by this tool' },
    components,
    cameraPath: camera ? { component: camera.name, ...camera.packages.cameraPath } : null,
    license,
    totals: { components: components.length,
      counting: 'sums of the component manifests (per source primitive, before node instancing)',
      geometry: total(GEOMETRY_TOTALS, item => item.packages.geometry.totals),
      textures: total(TEXTURE_TOTALS, item => item.packages.textures),
      omittedPrimitives: components.reduce((sum, item) => sum + item.omitted.length, 0) } };
  await mkdir(dirname(output), { recursive: true });
  const temporary = await mkdtemp(join(dirname(output), `.${basename(output)}.tmp-`));
  try {
    await writeFile(join(temporary, license.file), licenseBytes, { flag: 'wx' });
    const text = JSON.stringify(composition, null, 2) + '\n';
    await writeFile(join(temporary, 'asset-set.json'), text, { flag: 'wx' });
    await writeFile(join(temporary, 'manifest.json'), text, { flag: 'wx' });
    await publishDirectory(temporary, output);
    process.stdout.write(JSON.stringify({ output, components: components.map(item => item.name), cameraPath: composition.cameraPath?.component ?? null,
      totals: composition.totals }) + '\n');
  } finally { await removeOwnTemporary(temporary, dirname(output), output); }
}

if (composes.length) await writeComposition();
else await writeSingle();
