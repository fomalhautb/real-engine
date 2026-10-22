#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { open, stat, realpath, mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, isAbsolute, relative, resolve, join, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
const LIMITS = Object.freeze({ jsonBytes: 128 * MiB, inputBytes: 1024 * MiB,
  singleFileBytes: 512 * MiB, outputBytes: 2048 * MiB, pageBytes: 1024 * MiB,
  buffers: 256, primitives: 4096 });
const PAGE_OPTIONS = Object.freeze({ maxPageBytes: 65536, maxTotalBytes: 256 * MiB });
const PROXY_OPTIONS = Object.freeze({ maxVertices: 1_048_576, maxTriangles: 262_144, maxBytes: 128 * MiB });
// Pass every default explicitly so a package remains reproducible if builder defaults change.
const GEOMETRY_BUILD_OPTIONS = Object.freeze({ maxClusterVertices: 64, maxClusterTriangles: 128,
  simplifyRatio: 0.5, maxSimplificationError: 1e30, maxLevels: 16, maxVertices: 1_000_000,
  maxTriangles: 1_000_000, maxOutputTriangles: 3_000_000, maxClusters: 65_536, maxGroups: 65_536 });
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonBytes = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');

function argumentsOf(argv) {
  const values = new Map();
  let removeZeroAreaTriangles = false;
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === '--remove-zero-area' && !removeZeroAreaTriangles) { removeZeroAreaTriangles = true; continue; }
    if (!['--input', '--output', '--query-error', '--max-page-bytes', '--group-size'].includes(key) || values.has(key) || !argv[i + 1]) throw new Error('invalid build arguments');
    values.set(key, argv[++i]);
  }
  if (!values.has('--input') || !values.has('--output')) {
    throw new Error('usage: build-gltf-geometry --input scene.gltf|glb --output new-directory [--remove-zero-area] [--query-error value] [--max-page-bytes value] [--group-size 2..16]');
  }
  const input = resolve(values.get('--input')), output = resolve(values.get('--output'));
  if (!['.gltf', '.glb'].includes(extname(input).toLowerCase()) || input === output) throw new Error('invalid input/output');
  const maxError = values.has('--query-error') ? Number(values.get('--query-error')) : 1e30;
  if (!Number.isFinite(maxError) || maxError < 0 || maxError > 1e30) throw new Error('query-error must be finite and in [0, 1e30]');
  const maxPageBytes = values.has('--max-page-bytes') ? Number(values.get('--max-page-bytes')) : LIMITS.pageBytes;
  if (!Number.isSafeInteger(maxPageBytes) || maxPageBytes < 1 || maxPageBytes > 4096 * MiB) throw new Error('max-page-bytes must be an integer in [1, 4294967296]');
  const groupSize = values.has('--group-size') ? Number(values.get('--group-size')) : 4;
  if (!Number.isSafeInteger(groupSize) || groupSize < 2 || groupSize > 16) throw new Error('group-size must be an integer in [2, 16]');
  return { input, output, maxError, maxPageBytes, groupSize, removeZeroAreaTriangles };
}

function list(value, label, max = 65536) {
  if (!Array.isArray(value) || value.length > max) throw new Error('GLTF_BUILD_ARRAY: ' + label);
  return value;
}
function record(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('GLTF_BUILD_OBJECT: ' + label);
  return value;
}
function within(parent, target) {
  const path = relative(parent, target);
  return path !== '' && !isAbsolute(path) && path !== '..' && !path.startsWith('..' + sep);
}
function uint32Bytes(values) {
  const result = Buffer.alloc(values.byteLength);
  for (let i = 0; i < values.length; i++) result.writeUInt32LE(values[i], i * 4);
  return result;
}

/** Local geometry only: no network, texture decoding, or implicit asset repair. */
async function loadSource(input, parseGlb) {
  let inputBytes = 0;
  const files = new Map();
  const load = async (path, allowedRoot) => {
    // Resolve first, then open the canonical path, so retargeting the URI's
    // symlink cannot redirect the subsequent read. Compare the opened identity
    // with the canonical pathname before reading any payload.
    const canonical = await realpath(path);
    if (allowedRoot && !within(allowedRoot, canonical)) throw new Error('GLTF_BUILD_BUFFER_OUTSIDE_ASSET');
    if (files.has(canonical)) return files.get(canonical);
    const handle = await open(canonical, 'r');
    try {
      const info = await handle.stat();
      if (!info.isFile() || info.size > LIMITS.singleFileBytes || inputBytes + info.size > LIMITS.inputBytes) throw new Error('GLTF_BUILD_INPUT_BYTE_BUDGET');
      if (await realpath(path) !== canonical || await realpath(canonical) !== canonical) throw new Error('GLTF_BUILD_SOURCE_CHANGED');
      const named = await stat(canonical);
      if (named.dev !== info.dev || named.ino !== info.ino) throw new Error('GLTF_BUILD_SOURCE_CHANGED');
      inputBytes += info.size;
      const data = Buffer.alloc(info.size);
      for (let offset = 0; offset < data.length;) {
        const { bytesRead } = await handle.read(data, offset, Math.min(4 * MiB, data.length - offset), offset);
        if (!bytesRead) throw new Error('GLTF_BUILD_SOURCE_CHANGED');
        offset += bytesRead;
      }
      const extra = await handle.read(Buffer.alloc(1), 0, 1, data.length);
      const after = await handle.stat();
      if (extra.bytesRead || after.size !== info.size || after.mtimeMs !== info.mtimeMs) throw new Error('GLTF_BUILD_SOURCE_CHANGED');
      files.set(canonical, data);
      return data;
    } finally { await handle.close(); }
  };
  const data = await load(input), glb = extname(input).toLowerCase() === '.glb';
  if (!glb && data.length > LIMITS.jsonBytes) throw new Error('GLTF_BUILD_JSON_BYTE_BUDGET');
  if (glb && data.length >= 20 && data.readUInt32LE(12) > LIMITS.jsonBytes) throw new Error('GLTF_BUILD_JSON_BYTE_BUDGET');
  const parsed = glb ? parseGlb(Uint8Array.from(data).buffer) : { json: JSON.parse(data.toString('utf8')) };
  const doc = record(parsed.json, 'root');
  if (doc.asset?.version !== '2.0') throw new Error('GLTF_BUILD_VERSION');
  const base = await realpath(dirname(input));
  const copied = [{ file: 'source/' + basename(input), data }];
  const buffers = [], descriptors = [];
  const definitions = list(doc.buffers ?? [], 'buffers', LIMITS.buffers);
  for (let i = 0; i < definitions.length; i++) {
    const definition = record(definitions[i], 'buffer');
    if (!Number.isSafeInteger(definition.byteLength) || definition.byteLength < 1 || definition.byteLength > LIMITS.singleFileBytes) throw new Error('GLTF_BUILD_BUFFER_LENGTH');
    let bytes, file;
    if (definition.uri === undefined && glb && i === 0 && parsed.bin) {
      bytes = new Uint8Array(parsed.bin); file = `embedded/buffer-${i}.bin`;
      copied.push({ file, data: bytes });
    } else if (typeof definition.uri === 'string' && definition.uri.startsWith('data:')) {
      const match = /^data:(?:application\/octet-stream|application\/gltf-buffer);base64,([A-Za-z0-9+/]*={0,2})$/.exec(definition.uri);
      if (!match || match[1].length % 4 !== 0) throw new Error('GLTF_BUILD_DATA_URI');
      const estimated = match[1].length / 4 * 3;
      if (estimated > LIMITS.singleFileBytes + 2 || inputBytes + estimated > LIMITS.inputBytes) throw new Error('GLTF_BUILD_INPUT_BYTE_BUDGET');
      bytes = Buffer.from(match[1], 'base64'); inputBytes += bytes.length;
      file = `embedded/buffer-${i}.bin`;
      copied.push({ file, data: bytes });
    } else {
      if (typeof definition.uri !== 'string') throw new Error('GLTF_BUILD_BUFFER_URI');
      const uri = decodeURIComponent(definition.uri);
      if (!uri || /[\\\u0000?#:]/.test(uri) || isAbsolute(uri)) throw new Error('GLTF_BUILD_LOCAL_BUFFER_URI');
      const target = resolve(dirname(input), uri);
      if (!within(dirname(input), target)) throw new Error('GLTF_BUILD_BUFFER_OUTSIDE_ASSET');
      bytes = await load(target, base); file = 'source/' + relative(dirname(input), target).split(sep).join('/');
      if (file === copied[0].file) throw new Error('GLTF_BUILD_BUFFER_SOURCE_COLLISION');
      if (!copied.some(item => item.file === file)) copied.push({ file, data: bytes });
    }
    if (bytes.byteLength < definition.byteLength || (definition.uri === undefined && bytes.byteLength > definition.byteLength + 3)) throw new Error('GLTF_BUILD_BUFFER_LENGTH');
    buffers.push(Uint8Array.from(bytes).buffer);
    descriptors.push({ index: i, file, bytes: bytes.byteLength, sha256: sha256(bytes), embedded: definition.uri === undefined || definition.uri.startsWith('data:') });
  }
  const source = { file: copied[0].file, bytes: data.length, sha256: sha256(data), buffers: descriptors };
  return { doc, buffers, source, copied, identity: sha256(jsonBytes([source.sha256, descriptors.map(item => item.sha256)])), inputBytes };
}

async function buildInto(config, temporary) {
  const geometryBuildOptions = { ...GEOMETRY_BUILD_OPTIONS, groupSize: config.groupSize };
  const server = await createServer({ root: ROOT, configFile: false,
    server: { middlewareMode: true, watch: null, ws: false },
    optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
  try {
    const { parseGlb } = await server.ssrLoadModule('/src/gltf-binary.ts');
    const { extractGltfStaticPrimitive } = await server.ssrLoadModule('/src/gltf-geometry-build.ts');
    const { cleanStaticGeometry } = await server.ssrLoadModule('/src/geometry-clean.ts');
    const { meshWorldUnitsPerUv } = await server.ssrLoadModule('/src/texture-stream-demand.ts');
    const { buildGeometryHierarchy } = await server.ssrLoadModule('/src/geometry-hierarchy.ts');
    const { encodeGeometryPages } = await server.ssrLoadModule('/src/geometry-pages.ts');
    const { buildGeometryStreamDirectory } = await server.ssrLoadModule('/src/geometry-stream-directory.ts');
    const { buildGeometryQueryProxy, encodeGeometryQueryProxy } = await server.ssrLoadModule('/src/geometry-query-proxy.ts');
    const textureUvSpanOf = mesh => {
      // Source-routed primitives may exceed the demand estimator's bounded
      // workload. Preserve them and mark their mip demand as unknown.
      if (mesh.vertexCount > 1_048_576 || mesh.indices.length > 3_145_728) return 'unknown';
      const span = meshWorldUnitsPerUv(mesh);
      return Number.isFinite(span) && span > 0 ? span : 'unknown';
    };
    const asset = await loadSource(config.input, parseGlb);
    const meshes = list(asset.doc.meshes ?? [], 'meshes');
    let count = 0;
    for (const mesh of meshes) count += list(record(mesh, 'mesh').primitives, 'primitives', LIMITS.primitives).length;
    if (!count || count > LIMITS.primitives) throw new Error('GLTF_BUILD_PRIMITIVE_BUDGET');
    let outputBytes = 0, pageBytes = 0;
    const write = async (file, bytes) => {
      if (outputBytes + bytes.byteLength > LIMITS.outputBytes) throw new Error('GLTF_BUILD_OUTPUT_BYTE_BUDGET');
      outputBytes += bytes.byteLength;
      const path = resolve(temporary, file);
      if (!within(temporary, path)) throw new Error('GLTF_BUILD_OUTPUT_PATH');
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, bytes, { flag: 'wx' });
      return { file, bytes: bytes.byteLength, sha256: sha256(bytes) };
    };
    for (const file of asset.copied) await write(file.file, file.data);
    const primitives = [];
    const totals = { sourceTriangles: 0, triangles: 0, removedTriangles: 0, pagedPrimitives: 0, sourcePrimitives: 0,
      clusters: 0, pages: 0, encodedPageBytes: 0, decodedPageBytes: 0, rootEncodedBytes: 0, rootDecodedBytes: 0,
      proxyVertices: 0, proxyTriangles: 0, proxyBytes: 0 };
    for (let meshIndex = 0; meshIndex < meshes.length; meshIndex++) {
      for (let primitiveIndex = 0; primitiveIndex < meshes[meshIndex].primitives.length; primitiveIndex++) {
        const extracted = extractGltfStaticPrimitive(asset.doc, asset.buffers, meshIndex, primitiveIndex);
        const sourceTriangleCount = extracted.mesh.indices.length / 3;
        const entry = { meshIndex, primitiveIndex, name: meshes[meshIndex].name ?? null,
          materialIndex: extracted.materialIndex, coverage: extracted.coverage,
          sourceVertexCount: extracted.mesh.vertexCount, sourceTriangleCount,
          ignoredAttributes: extracted.ignoredAttributes, warnings: extracted.warnings };
        totals.sourceTriangles += sourceTriangleCount;
        if (extracted.coverage !== 'opaque') {
          // Source-routed coverage keeps its original triangles, including any
          // degenerates; an unprovable UV bound requests finest-mip fallback.
          // Preserve the original primitive and optical definition. No transparent
          // geometry is relabelled opaque or silently absent from the manifest.
          primitives.push({ ...entry, route: 'source', textureUvSpan: textureUvSpanOf(extracted.mesh),
            triangleCount: sourceTriangleCount, removedTriangleCount: 0,
            reason: 'virtual geometry currently requires opaque coverage; source primitive and material retained' });
          totals.triangles += sourceTriangleCount; totals.sourcePrimitives++;
          continue;
        }
        const clean = cleanStaticGeometry(extracted.mesh, { removeZeroAreaTriangles: config.removeZeroAreaTriangles });
        // This is the full cleaned source, before hierarchy LOD or query-proxy
        // simplification. Its units are local position distance per UV0 unit.
        const textureUvSpan = textureUvSpanOf(clean.mesh);
        const primitiveKey = `gltf:${asset.identity}:mesh:${meshIndex}:primitive:${primitiveIndex}`;
        const hierarchy = await buildGeometryHierarchy({ primitiveKey, mesh: clean.mesh, coverage: 'opaque' }, geometryBuildOptions);
        const encoded = await encodeGeometryPages(hierarchy, PAGE_OPTIONS);
        const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
        const proxy = await buildGeometryQueryProxy(directory, async request => encoded.pages[request.id].bytes,
          { ...PROXY_OPTIONS, maxError: config.maxError });
        const folder = `mesh-${meshIndex}-primitive-${primitiveIndex}`;
        const stream = await write(`${folder}/stream-directory.json`, jsonBytes(directory));
        const proxyFile = await write(`${folder}/query-proxy.bin`, encodeGeometryQueryProxy(proxy));
        const provenance = {
          vertices: await write(`${folder}/source-vertices.bin`, uint32Bytes(clean.sourceVertices)),
          triangles: await write(`${folder}/source-triangles.bin`, uint32Bytes(clean.sourceTriangles)),
          removedTriangles: await write(`${folder}/removed-triangles.bin`, uint32Bytes(clean.removedTriangles)),
        };
        const pages = [];
        for (const page of encoded.pages) {
          if (pageBytes + page.bytes.byteLength + page.decodedBytes > config.maxPageBytes) throw new Error('GLTF_BUILD_TOTAL_PAGE_BUDGET');
          pageBytes += page.bytes.byteLength + page.decodedBytes;
          pages.push({ id: page.id, ...await write(`${folder}/page-${page.id}.bin`, page.bytes),
            decodedBytes: page.decodedBytes, clusterIds: page.clusterIds });
          totals.encodedPageBytes += page.bytes.byteLength; totals.decodedPageBytes += page.decodedBytes;
          if (directory.rootPages.includes(page.id)) {
            totals.rootEncodedBytes += page.bytes.byteLength; totals.rootDecodedBytes += page.decodedBytes;
          }
        }
        const triangleCount = clean.mesh.indices.length / 3;
        primitives.push({ ...entry, route: 'paged', textureUvSpan,
          directory: folder, primitiveKey,
          vertexCount: clean.mesh.vertexCount, triangleCount, removedTriangleCount: clean.removedTriangles.length,
          streamDirectoryFile: stream.file, streamDirectoryBytes: stream.bytes, streamDirectorySha256: stream.sha256,
          buildOptions: hierarchy.options, provenance, rootPages: directory.rootPages, pages,
          queryProxy: { ...proxyFile, directorySha256: proxy.directorySha256, topologyVersion: proxy.topologyVersion,
            requestedError: proxy.requestedError, approximateObjectSpaceError: proxy.approximateObjectSpaceError,
            errorMetric: 'accumulated-simplification', vertexCount: proxy.mesh.vertexCount, triangleCount: proxy.mesh.indices.length / 3 } });
        totals.triangles += triangleCount; totals.removedTriangles += clean.removedTriangles.length;
        totals.pagedPrimitives++; totals.clusters += directory.clusters.length; totals.pages += pages.length;
        totals.proxyVertices += proxy.mesh.vertexCount; totals.proxyTriangles += proxy.mesh.indices.length / 3; totals.proxyBytes += proxyFile.bytes;
        process.stdout.write(`built ${primitives.length}/${count}: mesh ${meshIndex} primitive ${primitiveIndex}, ${triangleCount} triangles, ${pages.length} pages\n`);
      }
    }
    await write('manifest.json', jsonBytes({ version: 1, format: 'real-engine-gltf-geometry', builder: 'meshoptimizer-1.3.0',
      purpose: 'offline geometry package; material compilation, images, scene batching and runtime loading remain separate',
      source: asset.source, assetIdentity: asset.identity, config: { removeZeroAreaTriangles: config.removeZeroAreaTriangles,
        geometryBuild: geometryBuildOptions,
        page: PAGE_OPTIONS, queryProxy: { ...PROXY_OPTIONS, maxError: config.maxError },
        limits: { ...LIMITS, pageBytes: config.maxPageBytes, primitiveVertices: 1_000_000, primitiveTriangles: 1_000_000 } },
      imageResourcesCopied: false, images: asset.doc.images ?? [],
      provenance: 'uint32 little-endian; output vertex/triangle to original primitive vertex/triangle; removed triangle IDs refer to original primitive',
      totals, primitives }));
  } finally { await server.close(); }
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) throw new Error('output already exists: ' + config.output);
  const parent = dirname(config.output);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(join(parent, `.${basename(config.output)}.tmp-`));
  try {
    await buildInto(config, temporary);
    await publishDirectory(temporary, config.output);
    process.stdout.write(config.output + '\n');
  } finally { await removeOwnTemporary(temporary, parent, config.output); }
}
try { await main(); }
catch (error) { process.stderr.write(`build-gltf-geometry: ${error.message}\n`); process.exitCode = 1; }
