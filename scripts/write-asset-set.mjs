#!/usr/bin/env node
// Writes an asset-set description that binds already-built packages of one
// source scene together, after checking that they really derive from the same
// source document (and the same geometry package where that applies). Capture
// scripts and the large-scene demo read the server roots from it.
//   node scripts/write-asset-set.mjs --name intel-sponza-main --geometry dir --textures dir
//     [--query dir] [--uv file.json] [--license file] --output new-directory
import { createHash } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const FLAGS = { '--name': 'name', '--geometry': 'geometry', '--textures': 'textures', '--query': 'query', '--uv': 'uv',
  '--license': 'license', '--output': 'output', '--artifacts-root': 'artifactsRoot' };

const options = {};
for (let i = 2; i < process.argv.length; i++) {
  const key = FLAGS[process.argv[i]], value = process.argv[i + 1];
  if (!key || options[key] !== undefined || !value || value.startsWith('--')) throw Error('ASSET_SET_ARGUMENTS: ' + process.argv[i]);
  options[key] = value; i++;
}
for (const key of ['name', 'geometry', 'textures', 'output']) if (!options[key]) throw Error('ASSET_SET_REQUIRED: --' + key);
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
await mkdir(dirname(output), { recursive: true });
const temporary = await mkdtemp(join(dirname(output), `.${basename(output)}.tmp-`));
try {
  let license = null;
  if (options.license) {
    const bytes = await readFile(resolve(options.license));
    await copyFile(resolve(options.license), join(temporary, basename(options.license)));
    license = { file: basename(options.license), bytes: bytes.length, sha256: sha256(bytes) };
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
