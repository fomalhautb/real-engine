#!/usr/bin/env node
// Independent static UV0 demand audit for a local glTF. It decodes every
// primitive from the original document/BIN (not from built pages) and reports
// the conservative local-units-per-UV bound used by texture streaming, or the
// explicit string 'unknown'. Optionally cross-checks a build-gltf-geometry
// manifest, so a texture-demand input can never silently drift from geometry.
import { createHash } from 'node:crypto';
import { readFile, writeFile, stat } from 'node:fs/promises';
import { dirname, resolve, isAbsolute, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { exists } from './asset-output.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

const args = new Map();
let removeZeroArea = false;
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  if (key === '--remove-zero-area') { removeZeroArea = true; continue; }
  if (!['--input', '--output', '--geometry-manifest'].includes(key) || !process.argv[i + 1]) {
    throw Error('usage: audit-gltf-uv-demand --input scene.gltf --output new-file.json [--remove-zero-area] [--geometry-manifest manifest.json]');
  }
  args.set(key, process.argv[++i]);
}
if (!args.has('--input') || !args.has('--output')) throw Error('UV_AUDIT_ARGUMENTS');
const input = resolve(args.get('--input')), output = resolve(args.get('--output'));
if (await exists(output)) throw Error('UV_AUDIT_OUTPUT_EXISTS: ' + output);

const server = await createServer({ root: ROOT, configFile: false, server: { middlewareMode: true, watch: null, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
try {
  const { extractGltfStaticPrimitive } = await server.ssrLoadModule('/src/gltf-geometry-build.ts');
  const { cleanStaticGeometry } = await server.ssrLoadModule('/src/geometry-clean.ts');
  const { meshWorldUnitsPerUv } = await server.ssrLoadModule('/src/texture-stream-demand.ts');
  if ((await stat(input)).size > 128 * MiB) throw Error('UV_AUDIT_JSON_BYTES');
  const sourceBytes = await readFile(input);
  const doc = JSON.parse(sourceBytes.toString('utf8'));
  const buffers = [];
  for (const buffer of doc.buffers ?? []) {
    const uri = decodeURIComponent(buffer.uri ?? '');
    const path = resolve(dirname(input), uri);
    const rel = relative(dirname(input), path);
    if (!uri || /^[a-z][a-z0-9+.-]*:/i.test(uri) || isAbsolute(rel) || rel.startsWith('..' + sep) || rel === '..') throw Error('UV_AUDIT_LOCAL_BUFFER_ONLY');
    if ((await stat(path)).size > 1024 * MiB) throw Error('UV_AUDIT_BUFFER_BYTES');
    const bytes = await readFile(path);
    buffers.push({ uri, bytes: bytes.length, sha256: sha256(bytes), data: Uint8Array.from(bytes).buffer });
  }
  const spanOf = mesh => {
    if (mesh.vertexCount > 1_048_576 || mesh.indices.length > 3_145_728) return 'unknown';
    const value = meshWorldUnitsPerUv(mesh);
    return Number.isFinite(value) && value > 0 ? value : 'unknown';
  };
  const primitives = [];
  for (let meshIndex = 0; meshIndex < (doc.meshes ?? []).length; meshIndex++) {
    for (let primitiveIndex = 0; primitiveIndex < doc.meshes[meshIndex].primitives.length; primitiveIndex++) {
      const extracted = extractGltfStaticPrimitive(doc, buffers.map(b => b.data), meshIndex, primitiveIndex);
      const sourceSpan = spanOf(extracted.mesh);
      // Mirror build-gltf-geometry: opaque primitives are measured after the
      // explicit zero-area cleanup; source-routed coverage keeps its mesh.
      const cleaned = removeZeroArea && extracted.coverage === 'opaque'
        ? cleanStaticGeometry(extracted.mesh, { removeZeroAreaTriangles: true }) : null;
      const textureUvSpan = cleaned ? spanOf(cleaned.mesh) : sourceSpan;
      primitives.push({ meshIndex, primitiveIndex, materialIndex: extracted.materialIndex, coverage: extracted.coverage,
        triangles: extracted.mesh.indices.length / 3, sourceTextureUvSpan: sourceSpan, textureUvSpan,
        measuredOn: cleaned ? 'zero-area-cleaned-source' : 'source' });
    }
  }
  let geometryCheck = null;
  if (args.has('--geometry-manifest')) {
    const manifestBytes = await readFile(resolve(args.get('--geometry-manifest')));
    const manifest = JSON.parse(manifestBytes.toString('utf8'));
    if (manifest.source.sha256 !== sha256(sourceBytes)) throw Error('UV_AUDIT_GEOMETRY_SOURCE_MISMATCH');
    const built = new Map(manifest.primitives.map(p => [`${p.meshIndex}:${p.primitiveIndex}`, p.textureUvSpan]));
    const mismatches = primitives.filter(p => built.get(`${p.meshIndex}:${p.primitiveIndex}`) !== p.textureUvSpan)
      .map(p => ({ meshIndex: p.meshIndex, primitiveIndex: p.primitiveIndex, audit: p.textureUvSpan, built: built.get(`${p.meshIndex}:${p.primitiveIndex}`) }));
    if (mismatches.length || built.size !== primitives.length) throw Error('UV_AUDIT_GEOMETRY_MISMATCH: ' + JSON.stringify(mismatches.slice(0, 8)));
    geometryCheck = { manifestSha256: sha256(manifestBytes), assetIdentity: manifest.assetIdentity, primitives: built.size, identical: true };
  }
  const finite = primitives.filter(p => typeof p.textureUvSpan === 'number');
  const report = { version: 1, format: 'real-engine-gltf-uv-demand-audit', tool: 'scripts/audit-gltf-uv-demand.mjs',
    sourceSha256: sha256(sourceBytes), sourceFile: input.split(sep).at(-1), buffers: buffers.map(({ data, ...rest }) => rest),
    removeZeroArea, units: 'local position units per UV0 unit (conservative per-triangle upper bound)',
    summary: { primitives: primitives.length, finite: finite.length, unknown: primitives.length - finite.length,
      min: Math.min(...finite.map(p => p.textureUvSpan)), max: Math.max(...finite.map(p => p.textureUvSpan)) },
    geometryCheck, primitives };
  await writeFile(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  process.stdout.write(JSON.stringify({ output, ...report.summary, geometryCheck }) + '\n');
} finally { await server.close(); }

