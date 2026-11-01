#!/usr/bin/env node
// Offline exact crack audit of built glTF geometry packages.
//   node scripts/audit-geometry-boundaries.mjs --package artifacts/<pages> [--package ...] --output report.json
// For every paged primitive it verifies the stream directory hash, decodes and
// hash-verifies every page, then runs geometryGroupBoundaryAudit over the decoded
// payload bytes: each non-terminal group must keep its signed welded boundary.
// With that property every cut that geometryCutCoverage accepts at run time has
// exactly LOD0's net boundary, so the browser walk only needs exact coverage of
// each probed GPU cut. Reads only local files and writes one JSON report.
import { readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MiB = 1024 * 1024;
const LIMITS = Object.freeze({ manifestBytes: 64 * MiB, directoryBytes: 64 * MiB, pageBytes: 64 * MiB, packages: 16 });
const packages = [];
let output;
for (let i = 2; i < process.argv.length; i++) {
  const flag = process.argv[i], value = process.argv[++i];
  if (!value || value.startsWith('--')) throw Error('AUDIT_BOUNDARY_ARGUMENT: ' + flag);
  if (flag === '--package') packages.push(resolve(value));
  else if (flag === '--output' && output === undefined) output = resolve(value);
  else throw Error('AUDIT_BOUNDARY_ARGUMENT: ' + flag);
}
if (!packages.length || packages.length > LIMITS.packages || !output) {
  throw Error('usage: audit-geometry-boundaries --package artifacts/<pages> [--package ...] --output report.json');
}

async function bounded(path, limit) {
  const size = (await stat(path)).size;
  if (size > limit) throw Error('AUDIT_BOUNDARY_BYTES: ' + path);
  return new Uint8Array(await readFile(path));
}

const server = await createServer({ root: ROOT, configFile: false, logLevel: 'error', appType: 'custom',
  server: { middlewareMode: true, watch: null, ws: false }, optimizeDeps: { noDiscovery: true, include: [] } });
try {
  const { validateGeometryStreamDirectory } = await server.ssrLoadModule('/src/geometry-stream-directory.ts');
  const { decodeVerifiedGeometryPage } = await server.ssrLoadModule('/src/geometry-page-decode.ts');
  const { geometryGroupBoundaryAudit } = await server.ssrLoadModule('/src/geometry-stream-coverage.ts');
  const { createHash } = await import('node:crypto');
  const reports = [];
  for (const root of packages) {
    const started = performance.now();
    const manifestBytes = await bounded(join(root, 'manifest.json'), LIMITS.manifestBytes);
    const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
    if (manifest.format !== 'real-engine-gltf-geometry') throw Error('AUDIT_BOUNDARY_FORMAT: ' + root);
    const totals = { primitives: 0, groups: 0, checkedGroups: 0, mismatchedGroups: 0, clusters: 0, triangles: 0, pages: 0,
      lod0BoundaryEdges: 0, closedPrimitives: 0 };
    const mismatches = [], primitives = [];
    for (const entry of manifest.primitives) {
      if (entry.route !== 'paged') continue;
      const directoryBytes = await bounded(join(root, entry.streamDirectoryFile), LIMITS.directoryBytes);
      if (directoryBytes.byteLength !== entry.streamDirectoryBytes
        || createHash('sha256').update(directoryBytes).digest('hex') !== entry.streamDirectorySha256) throw Error('AUDIT_BOUNDARY_DIRECTORY_HASH: ' + entry.streamDirectoryFile);
      const directory = validateGeometryStreamDirectory(JSON.parse(new TextDecoder().decode(directoryBytes)));
      const files = new Map(entry.pages.map(page => [page.id, page.file]));
      const payloads = new Map();
      for (const page of directory.pages) {
        const file = files.get(page.id);
        if (typeof file !== 'string') throw Error('AUDIT_BOUNDARY_PAGE_FILE: ' + page.id);
        const bytes = await bounded(join(root, file), LIMITS.pageBytes);
        const decoded = await decodeVerifiedGeometryPage(page, bytes);
        for (const cluster of decoded.clusters) payloads.set(cluster.id, { rest: cluster.rest, localTriangles: cluster.localTriangles });
      }
      const audit = geometryGroupBoundaryAudit(directory, payloads);
      const row = { mesh: entry.meshIndex, primitive: entry.primitiveIndex, pages: directory.pages.length, ...audit };
      primitives.push(row);
      totals.primitives++; totals.pages += directory.pages.length;
      for (const key of ['groups', 'checkedGroups', 'mismatchedGroups', 'clusters', 'triangles', 'lod0BoundaryEdges']) totals[key] += audit[key];
      if (audit.lod0BoundaryEdges === 0) totals.closedPrimitives++;
      if (audit.mismatchedGroups) mismatches.push(row);
    }
    const report = { package: root.slice(ROOT.length + 1), manifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
      assetIdentity: manifest.assetIdentity, totals, mismatches, primitives, elapsedMs: Math.round(performance.now() - started) };
    reports.push(report);
    console.log(JSON.stringify({ package: report.package, totals, elapsedMs: report.elapsedMs }));
  }
  const passed = reports.every(report => report.totals.mismatchedGroups === 0 && report.totals.checkedGroups > 0);
  await writeFile(output, JSON.stringify({ format: 'real-engine-geometry-boundary-audit', version: 1, passed,
    method: 'per non-terminal group: signed undirected welded edge balance of parents equals that of children, over decoded, hash-verified page payloads',
    packages: reports }, null, 2) + '\n');
  if (!passed) process.exitCode = 1;
} finally { await server.close(); }
