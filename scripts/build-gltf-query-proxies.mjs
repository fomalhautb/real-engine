#!/usr/bin/env node
// Rebuilds fixed query proxies at an explicit object-space error from an
// existing, hash-verified build-gltf-geometry package. Pages are read from the
// package itself (never re-simplified) so the proxies stay bound to the same
// stream directories; a camera-selected rendering LOD is never involved.
import { createHash } from 'node:crypto';
import { open, mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
const LIMITS = Object.freeze({ manifestBytes: 64 * MiB, directoryBytes: 64 * MiB, pageBytes: 64 * MiB, outputBytes: 2048 * MiB });
const PROXY_OPTIONS = Object.freeze({ maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB });
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonBytes = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');

function argumentsOf(argv) {
  const values = new Map();
  for (let i = 0; i < argv.length; i++) {
    if (!['--package', '--output', '--query-error'].includes(argv[i]) || values.has(argv[i]) || !argv[i + 1]) {
      throw Error('usage: build-gltf-query-proxies --package geometry-package-dir --output new-directory --query-error 0');
    }
    values.set(argv[i], argv[++i]);
  }
  if (!values.has('--package') || !values.has('--output') || !values.has('--query-error')) throw Error('QUERY_PROXY_ARGUMENTS');
  const maxError = Number(values.get('--query-error'));
  if (!Number.isFinite(maxError) || maxError < 0 || maxError > 1e30) throw Error('query-error must be finite and in [0, 1e30]');
  return { packageRoot: resolve(values.get('--package')), output: resolve(values.get('--output')), maxError };
}

async function readChecked(path, limit, expected) {
  const handle = await open(path, 'r');
  try {
    const info = await handle.stat();
    if (!info.isFile() || info.size > limit || (expected && info.size !== expected.bytes)) throw Error('QUERY_PROXY_INPUT_BYTES: ' + path);
    const bytes = Buffer.alloc(info.size);
    for (let offset = 0; offset < bytes.length;) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.length - offset, offset);
      if (!bytesRead) throw Error('QUERY_PROXY_SOURCE_CHANGED: ' + path);
      offset += bytesRead;
    }
    if (expected && sha256(bytes) !== expected.sha256) throw Error('QUERY_PROXY_HASH: ' + path);
    return bytes;
  } finally { await handle.close(); }
}

async function buildInto(config, temporary) {
  const server = await createServer({ root: ROOT, configFile: false, server: { middlewareMode: true, watch: null, ws: false },
    optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
  try {
    const { validateGeometryStreamDirectory } = await server.ssrLoadModule('/src/geometry-stream-directory.ts');
    const { buildGeometryQueryProxy, encodeGeometryQueryProxy, decodeGeometryQueryProxy } = await server.ssrLoadModule('/src/geometry-query-proxy.ts');
    const manifestBytes = await readChecked(join(config.packageRoot, 'manifest.json'), LIMITS.manifestBytes);
    const manifest = JSON.parse(manifestBytes.toString('utf8'));
    if (manifest.format !== 'real-engine-gltf-geometry' || !Array.isArray(manifest.primitives)) throw Error('QUERY_PROXY_PACKAGE_FORMAT');
    const primitives = [];
    const totals = { primitives: 0, vertices: 0, triangles: 0, bytes: 0, sourceTriangles: 0 };
    let outputBytes = 0;
    for (const entry of manifest.primitives) {
      if (entry.route !== 'paged') continue;
      const directoryBytes = await readChecked(join(config.packageRoot, entry.streamDirectoryFile), LIMITS.directoryBytes,
        { bytes: entry.streamDirectoryBytes, sha256: entry.streamDirectorySha256 });
      const directory = validateGeometryStreamDirectory(JSON.parse(directoryBytes.toString('utf8')));
      const pages = new Map(entry.pages.map(page => [page.id, page]));
      const proxy = await buildGeometryQueryProxy(directory, async request => {
        const page = pages.get(request.id);
        if (!page) throw Error('QUERY_PROXY_PAGE_MISSING');
        return new Uint8Array(await readChecked(join(config.packageRoot, page.file), LIMITS.pageBytes, page));
      }, { ...PROXY_OPTIONS, maxError: config.maxError });
      const encoded = encodeGeometryQueryProxy(proxy);
      const proxySha = sha256(encoded);
      // Round trip through the production decoder before publishing.
      const decoded = await decodeGeometryQueryProxy(directory, encoded, proxySha, { ...PROXY_OPTIONS, maxError: config.maxError });
      if (decoded.mesh.indices.length !== proxy.mesh.indices.length) throw Error('QUERY_PROXY_ROUND_TRIP');
      const file = `${entry.directory}/query-proxy.bin`;
      outputBytes += encoded.byteLength;
      if (outputBytes > LIMITS.outputBytes) throw Error('QUERY_PROXY_OUTPUT_BYTES');
      await mkdir(join(temporary, entry.directory), { recursive: true });
      await writeFile(join(temporary, file), encoded, { flag: 'wx' });
      const triangleCount = proxy.mesh.indices.length / 3;
      primitives.push({ meshIndex: entry.meshIndex, primitiveIndex: entry.primitiveIndex, streamDirectorySha256: entry.streamDirectorySha256,
        sourceTriangleCount: entry.triangleCount,
        queryProxy: { file, bytes: encoded.byteLength, sha256: proxySha, directorySha256: proxy.directorySha256,
          topologyVersion: proxy.topologyVersion, requestedError: proxy.requestedError,
          approximateObjectSpaceError: proxy.approximateObjectSpaceError, errorMetric: 'accumulated-simplification',
          vertexCount: proxy.mesh.vertexCount, triangleCount } });
      totals.primitives++; totals.vertices += proxy.mesh.vertexCount; totals.triangles += triangleCount;
      totals.bytes += encoded.byteLength; totals.sourceTriangles += entry.triangleCount;
      if (totals.primitives % 50 === 0) process.stdout.write(`proxies ${totals.primitives}\n`);
    }
    await writeFile(join(temporary, 'manifest.json'), jsonBytes({ version: 1, format: 'real-engine-gltf-query-proxies',
      purpose: 'fixed query proxies built from hash-verified pages of the named geometry package; not camera LOD',
      sourcePackage: { root: basename(config.packageRoot), manifestSha256: sha256(manifestBytes), assetIdentity: manifest.assetIdentity,
        sourceSha256: manifest.source.sha256 },
      config: { maxError: config.maxError, limits: PROXY_OPTIONS }, totals, primitives }), { flag: 'wx' });
    return totals;
  } finally { await server.close(); }
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) throw Error('output already exists: ' + config.output);
  const parent = dirname(config.output);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(join(parent, `.${basename(config.output)}.tmp-`));
  try {
    const totals = await buildInto(config, temporary);
    await publishDirectory(temporary, config.output, { hardLinkFiles: true });
    process.stdout.write(JSON.stringify({ output: config.output, ...totals }) + '\n');
  } finally { await removeOwnTemporary(temporary, parent, config.output); }
}
try { await main(); }
catch (error) { process.stderr.write(`build-gltf-query-proxies: ${error.message}\n`); process.exitCode = 1; }
