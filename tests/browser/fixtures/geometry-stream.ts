import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { encodeGeometryPages, type EncodedGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { createHttpGeometryPageProvider } from '../../../src/geometry-page-provider';
import { GeometryPageCache } from '../../../src/geometry-stream-cache';
import { GeometryStreamCut, type GeometryStreamView } from '../../../src/geometry-stream-cut';
import type { GeometryStreamDirectory } from '../../../src/geometry-stream-types';

const SIZE = 12;
const view: GeometryStreamView = { eye: [6, 6, 5], projectionScale: 900, near: 0.1, maxErrorPixels: 0 };

function curvedGrid() {
  const side = SIZE + 1, vertexCount = side * side;
  const vertices = new ArrayBuffer(vertexCount * 80), rest = new DataView(vertices);
  const indices: number[] = [];
  let minZ = Infinity, maxZ = -Infinity;
  for (let y = 0; y <= SIZE; y++) for (let x = 0; x <= SIZE; x++) {
    const id = y * side + x, offset = id * 80;
    const z = 0.25 * Math.sin(x * 0.47) * Math.cos(y * 0.39);
    rest.setFloat32(offset, x, true); rest.setFloat32(offset + 4, y, true);
    rest.setFloat32(offset + 8, z, true); rest.setFloat32(offset + 12, 1, true);
    rest.setFloat32(offset + 24, 1, true); rest.setFloat32(offset + 48, 1, true);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
  }
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const a = y * side + x, b = a + 1, c = a + side, d = c + 1;
    indices.push(a, b, d, a, d, c);
  }
  return { vertices, vertexCount, indices: Uint32Array.from(indices),
    bounds: { min: [0, 0, minZ] as [number, number, number], max: [SIZE, SIZE, maxZ] as [number, number, number] } };
}

function base64(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

let prepared: { directory: GeometryStreamDirectory; encoded: EncodedGeometryPages } | undefined;

/** Prepare real encoded pages in the browser, then expose only bounded wire payloads to Playwright's route. */
export async function prepareGeometryStreamFixture() {
  if (!prepared) {
    const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'browser/geometry-stream-curved',
      mesh: curvedGrid(), coverage: 'opaque' }, {
      maxClusterVertices: 12, maxClusterTriangles: 6, groupSize: 2, maxLevels: 6,
      simplifyRatio: 0.5, maxSimplificationError: 2,
    });
    const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 2048 });
    const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
    prepared = { directory, encoded };
  }
  const { directory, encoded } = prepared;
  const initial = new GeometryStreamCut(directory).plan(new Set(directory.rootPages), view);
  return { pageCount: directory.pages.length, clusterCount: directory.clusters.length,
    sourceTriangles: directory.sourceTriangleCount,
    leafClusters: directory.clusters.flatMap((cluster, id) => cluster.level === 0 ? [id] : []),
    rootPages: [...directory.rootPages], initialRequestedPages: initial.requestedPages.map(page => page.pageId),
    pages: encoded.pages.map(page => ({ id: page.id, encodedBytes: page.bytes.byteLength, base64: base64(page.bytes) })) };
}

/** CPU reference: decoded cache IDs simulate availability for the planner; no GPU page upload or render occurs. */
export async function runGeometryStreamFixture(failedPageId: number, baseUrl: string) {
  if (!prepared) throw new Error('GEOMETRY_STREAM_FIXTURE_NOT_PREPARED');
  const { directory } = prepared;
  if (!directory.pages.some(page => page.id === failedPageId && !directory.rootPages.includes(page.id))) {
    throw new Error('GEOMETRY_STREAM_FIXTURE_FAILED_PAGE');
  }
  const provider = createHttpGeometryPageProvider(baseUrl,
    directory.pages.map(page => `page-${page.id}.bin`));
  const cache = await GeometryPageCache.create(directory, provider, { maxConcurrentRequests: 2 });
  try {
    const planner = new GeometryStreamCut(directory);
    const rootState = cache.diagnostics();
    const rootPlan = planner.plan(cache.residentPageIds(), view);
    for (const entry of directory.pages) if (!directory.rootPages.includes(entry.id)) cache.request(entry.id);
    await cache.idle();
    const partialState = cache.diagnostics();
    const partialPlan = planner.plan(cache.residentPageIds(), view);
    const atomicFallback = partialPlan.fallbackGroups.every(groupId => {
      const group = directory.groups[groupId];
      return group.parents.every(id => partialPlan.selectedClusters.includes(id))
        && group.children.every(id => !partialPlan.selectedClusters.includes(id));
    });
    cache.retry(failedPageId);
    await cache.idle();
    const fullState = cache.diagnostics();
    const fullPlan = planner.plan(cache.residentPageIds(), view);
    const selectedPagesAvailable = fullPlan.selectedPages.every(id => cache.residentPageIds().has(id));
    await cache.dispose();
    return { mode: 'CPU reference; cache availability simulates planner residency, no GPU upload',
      pageCount: directory.pages.length, rootPages: [...directory.rootPages],
      rootTriangles: rootPlan.selectedClusters.reduce((sum, id) => sum + directory.clusters[id].triangleCount, 0),
      fullTriangles: fullPlan.selectedClusters.reduce((sum, id) => sum + directory.clusters[id].triangleCount, 0),
      rootState, rootPlan, partialState, partialPlan, atomicFallback,
      fullState, fullPlan, selectedPagesAvailable, failedPageId, disposedState: cache.diagnostics() };
  } finally {
    await cache.dispose();
  }
}
