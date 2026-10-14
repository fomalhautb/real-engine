#!/usr/bin/env node
import { createHash, randomUUID } from 'node:crypto';
import { cpSync, renameSync } from 'node:fs';
import { readFile, stat, lstat, mkdir, mkdtemp, writeFile, rename, rm } from 'node:fs/promises';
import { basename, dirname, extname, resolve, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MAX_SOURCE_BYTES = 32 * 1024 * 1024;
const MAX_PRIMITIVES = 4096;
const PAGE_OPTIONS = { maxPageBytes: 65_536, maxTotalBytes: 256 * 1024 * 1024 };

function argumentsOf(argv) {
  if (argv.length !== 6) throw new Error('usage: node scripts/build-geometry.mjs --input path.obj --output directory --coverage opaque');
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    if (!['--input', '--output', '--coverage'].includes(name) || values.has(name) || !argv[index + 1]) {
      throw new Error('expected exactly --input path.obj --output directory --coverage opaque');
    }
    values.set(name, argv[index + 1]);
  }
  if (!values.has('--input') || !values.has('--output') || values.get('--coverage') !== 'opaque') {
    throw new Error('OBJ geometry build requires explicit --coverage opaque; OBJ materials are not converted');
  }
  const input = resolve(values.get('--input'));
  const output = resolve(values.get('--output'));
  if (extname(input).toLowerCase() !== '.obj') throw new Error('only .obj input is supported');
  if (input === output) throw new Error('output must differ from input');
  return { input, output };
}

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

async function exists(path) {
  try { await lstat(path); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

async function removeOwnTemporary(temporary, targetParent, output) {
  if (!(await exists(temporary))) return;
  const absolute = resolve(temporary);
  if (dirname(absolute) !== resolve(targetParent) ||
      !basename(absolute).startsWith(`.${basename(output)}.tmp-`)) {
    throw new Error('refusing to remove a directory outside this build temporary root');
  }
  await rm(absolute, { recursive: true, force: true });
}

async function publishReplacement(replacement, output) {
  const retryDelays = [0, 50, 100, 200, 400, 800, 1600, 3200];
  for (const [attempt, waitMs] of retryDelays.entries()) {
    if (waitMs) await delay(waitMs);
    if (await exists(output)) throw new Error(`output already exists: ${output}`);
    try { renameSync(replacement, output); return; }
    catch (error) {
      if (attempt === retryDelays.length - 1 ||
          !['EPERM', 'EACCES', 'EBUSY'].includes(error.code)) throw error;
    }
  }
}

async function publishDirectory(temporary, output) {
  if (await exists(output)) throw new Error(`output already exists: ${output}`);
  try { await rename(temporary, output); return; }
  catch (error) {
    if (process.platform !== 'win32' || !['EPERM', 'EACCES'].includes(error.code)) throw error;
  }
  // If Windows denies the original directory rename, retry with a fresh private
  // copy. The final directory is still published by one rename, never file by file.
  const targetParent = dirname(output);
  const replacement = join(targetParent, `.${basename(output)}.tmp-${randomUUID()}`);
  try {
    cpSync(temporary, replacement, { recursive: true, force: false, errorOnExist: true });
    if (await exists(output)) throw new Error(`output already exists: ${output}`);
    await publishReplacement(replacement, output);
    await removeOwnTemporary(temporary, targetParent, output);
  } finally {
    await removeOwnTemporary(replacement, targetParent, output);
  }
}

function metadata(hierarchy) {
  const { source, ...rest } = hierarchy;
  return {
    ...rest,
    purpose: 'offline-audit; cluster arrays are duplicated in pages and are not a resident runtime manifest',
    source: { vertexCount: source.vertexCount, bounds: source.bounds,
      triangleCount: source.indices.length / 3 },
    clusters: hierarchy.clusters.map(cluster => ({
      ...cluster,
      vertices: Array.from(cluster.vertices),
      triangles: Array.from(cluster.triangles),
      ...(cluster.sourceTriangles ? { sourceTriangles: Array.from(cluster.sourceTriangles) } : {}),
    })),
  };
}

async function sourceOf(input) {
  const inputStat = await stat(input);
  if (!inputStat.isFile() || inputStat.size > MAX_SOURCE_BYTES) {
    throw new Error(`OBJ input must be a file of at most ${MAX_SOURCE_BYTES} bytes`);
  }
  const sourceBytes = await readFile(input);
  if (sourceBytes.byteLength > MAX_SOURCE_BYTES) throw new Error('OBJ input exceeds source byte budget');
  return sourceBytes;
}

async function buildInto(input, temporary, sourceBytes) {
  const sourceHash = sha256(sourceBytes);

  const server = await createServer({ root: REPOSITORY_ROOT, configFile: false,
    server: { middlewareMode: true }, appType: 'custom' });
  let parseObj, buildGeometryHierarchy, encodeGeometryPages;
  try {
    ({ parseObj } = await server.ssrLoadModule('/src/mesh-import.ts'));
    ({ buildGeometryHierarchy } = await server.ssrLoadModule('/src/geometry-hierarchy.ts'));
    ({ encodeGeometryPages } = await server.ssrLoadModule('/src/geometry-pages.ts'));
  } finally {
    await server.close();
  }

  const asset = parseObj(sourceBytes.toString('utf8'));
  if (asset.primitives.length > MAX_PRIMITIVES) throw new Error(`OBJ primitive count exceeds ${MAX_PRIMITIVES}`);
  const primitives = [];
  for (const [index, primitive] of asset.primitives.entries()) {
    const key = `obj:${sourceHash}:primitive:${index}`;
    const hierarchy = await buildGeometryHierarchy({
      primitiveKey: key, mesh: primitive.mesh, coverage: 'opaque',
    });
    const encoded = await encodeGeometryPages(hierarchy, PAGE_OPTIONS);
    const directory = `primitive-${index}`;
    await mkdir(join(temporary, directory));
    const hierarchyFile = `${directory}/hierarchy.json`;
    await writeFile(join(temporary, directory, 'hierarchy.json'), `${JSON.stringify(metadata(hierarchy), null, 2)}\n`);
    const pages = [];
    for (const page of encoded.pages) {
      const file = `${directory}/page-${page.id}.bin`;
      await writeFile(join(temporary, directory, `page-${page.id}.bin`), page.bytes);
      pages.push({ id: page.id, file, clusterIds: page.clusterIds,
        compressedBytes: page.bytes.byteLength, decodedBytes: page.decodedBytes,
        sha256: sha256(page.bytes) });
    }
    primitives.push({ index, name: primitive.name, materialName: primitive.materialName ?? null,
      sourceSha256: sourceHash, primitiveKey: key, hierarchyFile,
      buildOptions: hierarchy.options,
      vertexCount: primitive.mesh.vertexCount, triangleCount: primitive.mesh.indices.length / 3,
      clusterCount: hierarchy.clusters.length, groupCount: hierarchy.groups.length,
      rootClusters: hierarchy.roots, rootPages: encoded.rootPages,
      clusterPages: encoded.clusterPages, vertexStride: encoded.vertexStride,
      pageFormatVersion: encoded.version, pages });
  }
  const manifest = { version: 1, format: 'real-engine-geometry', builder: 'meshoptimizer-1.3.0',
    source: { name: basename(input), sha256: sourceHash, bytes: sourceBytes.byteLength },
    config: { inputFormat: 'obj', coverage: 'opaque', coverageContract: 'caller asserts opaque; OBJ materials are not converted',
      maxSourceBytes: MAX_SOURCE_BYTES, maxPrimitives: MAX_PRIMITIVES,
      page: PAGE_OPTIONS },
    materialLibraries: asset.materialLibraries, warnings: asset.warnings, primitives };
  await writeFile(join(temporary, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

async function build() {
  const { input, output } = argumentsOf(process.argv.slice(2));
  if (await exists(output)) throw new Error(`output already exists: ${output}`);
  const sourceBytes = await sourceOf(input);
  const targetParent = dirname(output);
  await mkdir(targetParent, { recursive: true });
  const temporary = await mkdtemp(join(targetParent, `.${basename(output)}.tmp-`));
  let published = false;
  try {
    await buildInto(input, temporary, sourceBytes);
    await publishDirectory(temporary, output);
    published = true;
    process.stdout.write(`${output}\n`);
  } finally {
    if (!published) await removeOwnTemporary(temporary, targetParent, output);
  }
}

try { await build(); }
catch (error) { process.stderr.write(`build-geometry: ${error.message}\n`); process.exitCode = 1; }
