import { beforeAll, describe, expect, it, vi } from 'vitest';
import { buildGeometryHierarchy } from '../src/geometry-hierarchy';
import { encodeGeometryPages } from '../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../src/geometry-stream-directory';
import type { GeometryPageProvider, GeometryStreamDirectory } from '../src/geometry-stream-types';
import { buildGeometryQueryProxy, decodeGeometryQueryProxy, encodeGeometryQueryProxy,
  validateGeometryQueryProxy } from '../src/geometry-query-proxy';

const N = 16;
function grid(withColors = false) {
  const count = (N + 1) ** 2, vertices = new ArrayBuffer(count * 80), view = new DataView(vertices);
  const indices: number[] = [], uvs = new Float32Array(count * 2), tangents = new Float32Array(count * 4);
  const colors = withColors ? new Float32Array(count * 4) : undefined;
  let low = Infinity, high = -Infinity;
  for (let y = 0; y <= N; y++) for (let x = 0; x <= N; x++) {
    const id = y * (N + 1) + x, z = Math.fround(0.2 * Math.sin(x * 0.47) * Math.cos(y * 0.39));
    view.setFloat32(id * 80, x, true); view.setFloat32(id * 80 + 4, y, true);
    view.setFloat32(id * 80 + 8, z, true); view.setFloat32(id * 80 + 12, 1, true);
    view.setFloat32(id * 80 + 24, 1, true); view.setFloat32(id * 80 + 48, 1, true);
    uvs.set([x / N, y / N], id * 2); tangents.set([1, 0, 0, 1], id * 4);
    colors?.set([x / N, y / N, (x + y) / (2 * N), 1], id * 4);
    low = Math.min(low, z); high = Math.max(high, z);
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = y * (N + 1) + x, b = a + 1, c = a + N + 1, d = c + 1;
    indices.push(a, b, d, a, d, c);
  }
  return { vertices, vertexCount: count, indices: Uint32Array.from(indices), uvs, tangents,
    ...(colors ? { colors } : {}),
    bounds: { min: [0, 0, low] as [number, number, number], max: [N, N, high] as [number, number, number] } };
}

type Fixture = Awaited<ReturnType<typeof fixture>>;
async function fixture(withColors = false) {
  const source = grid(withColors);
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'proxy/curved-grid', mesh: source, coverage: 'opaque' },
    { maxClusterTriangles: 32, groupSize: 4, simplifyRatio: 0.5, maxSimplificationError: 2, maxLevels: 8 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 16_384 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const provider: GeometryPageProvider = async page => encoded.pages[page.id].bytes.slice();
  return { source, hierarchy, encoded, directory, provider };
}

let data: Fixture;
beforeAll(async () => { data = await fixture(); });

function assertCompleteCut(directory: GeometryStreamDirectory, ids: readonly number[]) {
  const selected = new Set(ids);
  for (const group of directory.groups) {
    const parentCount = group.parents.filter(id => selected.has(id)).length;
    const childCount = group.children.filter(id => selected.has(id)).length;
    if (group.terminal) expect(childCount === 0 || childCount === group.children.length).toBe(true);
    else expect(childCount === 0 || (parentCount === 0 && childCount === group.children.length)).toBe(true);
    expect(parentCount && childCount).toBeFalsy();
  }
}

describe('offline geometry query proxy', () => {
  it('preserves optional RGBA through the directory, reduced cut, and proxy wire format', async () => {
    const colored = await fixture(true);
    expect(colored.directory.hasColors).toBe(true);
    expect(colored.directory.vertexStride).toBe(120);
    const proxy = await buildGeometryQueryProxy(colored.directory, colored.provider);
    expect(proxy.mesh.colors?.length).toBe(proxy.mesh.vertexCount * 4);
    const bytes = encodeGeometryQueryProxy(proxy);
    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
      byte => byte.toString(16).padStart(2, '0')).join('');
    const decoded = await decodeGeometryQueryProxy(colored.directory, bytes, sha256);
    expect(decoded.mesh.colors).toEqual(proxy.mesh.colors);
    const view = new DataView(proxy.mesh.vertices);
    for (let id = 0; id < proxy.mesh.vertexCount; id++) {
      const x = view.getFloat32(id * 80, true), y = view.getFloat32(id * 80 + 4, true);
      const sourceId = y * (N + 1) + x;
      expect(proxy.mesh.colors!.slice(id * 4, id * 4 + 4)).toEqual(colored.source.colors!.slice(sourceId * 4, sourceId * 4 + 4));
    }
    const malformed = { ...proxy, mesh: { ...proxy.mesh, colors: proxy.mesh.colors!.slice() } };
    malformed.mesh.colors![0] = 1.1;
    await expect(validateGeometryQueryProxy(colored.directory, malformed)).rejects.toThrow('MESH_LAYOUT');
    expect(data.directory.hasColors).toBeUndefined();
  });
  it('constructs a reduced source-free root cut from verified pages', async () => {
    const calls = vi.fn(data.provider);
    const proxy = await buildGeometryQueryProxy(data.directory, calls);
    expect(proxy.primitiveKey).toBe(data.directory.primitiveKey);
    expect(proxy.selectedClusters).toEqual([...data.directory.roots].sort((a, b) => a - b));
    expect(proxy.mesh.indices.length).toBeLessThan(data.source.indices.length);
    expect(proxy.mesh.vertexCount).toBeLessThan(data.source.vertexCount);
    expect(proxy.approximateObjectSpaceError).toBe(Math.max(...proxy.selectedClusters.map(id => data.directory.clusters[id].error)));
    expect(calls.mock.calls.length).toBe(new Set(proxy.selectedClusters.map(id => data.directory.clusterPages[id])).size);
    expect(Object.isFrozen(proxy)).toBe(true);
    expect(Object.isFrozen(proxy.mesh)).toBe(true);
    expect(Object.isFrozen(proxy.selectedClusters)).toBe(true);
    expect('source' in proxy).toBe(false);
    assertCompleteCut(data.directory, proxy.selectedClusters);
  });

  it('refines whole groups to the zero-error leaf cut without losing triangles or attributes', async () => {
    const proxy = await buildGeometryQueryProxy(data.directory, data.provider, { maxError: 0 });
    expect(proxy.selectedClusters.every(id => data.directory.clusters[id].level === 0)).toBe(true);
    expect(proxy.mesh.indices.length).toBe(data.source.indices.length);
    expect(proxy.approximateObjectSpaceError).toBe(0);
    assertCompleteCut(data.directory, proxy.selectedClusters);
    const view = new DataView(proxy.mesh.vertices);
    for (let id = 0; id < proxy.mesh.vertexCount; id++) {
      const x = view.getFloat32(id * 80, true), y = view.getFloat32(id * 80 + 4, true);
      const sourceId = y * (N + 1) + x;
      expect(proxy.mesh.uvs!.slice(id * 2, id * 2 + 2)).toEqual(data.source.uvs.slice(sourceId * 2, sourceId * 2 + 2));
      expect(proxy.mesh.tangents!.slice(id * 4, id * 4 + 4)).toEqual(data.source.tangents.slice(sourceId * 4, sourceId * 4 + 4));
    }
  });

  it('roundtrips deterministic bytes and rejects corruption or mismatched identities', async () => {
    const proxy = await buildGeometryQueryProxy(data.directory, data.provider, { maxError: 0.05, topologyVersion: 7 });
    const first = encodeGeometryQueryProxy(proxy), second = encodeGeometryQueryProxy(proxy);
    expect(second).toEqual(first);
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', first)),
      byte => byte.toString(16).padStart(2, '0')).join('');
    const decoded = await decodeGeometryQueryProxy(data.directory, first, hash, { topologyVersion: 7 });
    expect(encodeGeometryQueryProxy(decoded)).toEqual(first);
    expect(decoded.mesh.vertices).toEqual(proxy.mesh.vertices);
    expect(decoded.mesh.indices).toEqual(proxy.mesh.indices);
    expect(decoded.mesh.uvs).toEqual(proxy.mesh.uvs);
    expect(decoded.mesh.tangents).toEqual(proxy.mesh.tangents);
    const corrupt = first.slice(); corrupt[corrupt.length - 1] ^= 1;
    await expect(decodeGeometryQueryProxy(data.directory, corrupt, hash)).rejects.toThrow('GEOMETRY_QUERY_PROXY_HASH');
    const mismatched = { ...data.directory, primitiveKey: 'another-source' };
    await expect(validateGeometryQueryProxy(mismatched, proxy)).rejects.toThrow('GEOMETRY_QUERY_PROXY_IDENTITY');
    await expect(decodeGeometryQueryProxy(data.directory, first, hash, { topologyVersion: 8 })).rejects.toThrow('TOPOLOGY_VERSION');
  });

  it('rejects budgets before fetching pages and malformed authenticated payloads', async () => {
    const provider = vi.fn(data.provider);
    await expect(buildGeometryQueryProxy(data.directory, provider, { maxVertices: 1_048_577 }))
      .rejects.toThrow('VERTEX_BUDGET');
    await expect(buildGeometryQueryProxy(data.directory, provider, { maxTriangles: 262_145 }))
      .rejects.toThrow('TRIANGLE_BUDGET');
    await expect(buildGeometryQueryProxy(data.directory, provider, { maxBytes: 128 * 1024 * 1024 + 1 }))
      .rejects.toThrow('BYTE_BUDGET');
    expect(provider).not.toHaveBeenCalled();
    await expect(buildGeometryQueryProxy(data.directory, provider, { maxTriangles: 1 })).rejects.toThrow('TRIANGLE_BUDGET');
    expect(provider).not.toHaveBeenCalled();
    await expect(buildGeometryQueryProxy(data.directory, data.provider, { maxVertices: 2 })).rejects.toThrow('VERTEX_BUDGET');
    await expect(buildGeometryQueryProxy(data.directory, data.provider, { maxBytes: 100 })).rejects.toThrow('BYTE_BUDGET');
    const proxy = await buildGeometryQueryProxy(data.directory, data.provider);
    await expect(validateGeometryQueryProxy(data.directory, proxy, { maxTriangles: 1 }))
      .rejects.toThrow('TRIANGLE_BUDGET');
    await expect(validateGeometryQueryProxy(data.directory, proxy, { maxVertices: 1_048_577 }))
      .rejects.toThrow('VERTEX_BUDGET');
    await expect(validateGeometryQueryProxy(data.directory, { ...proxy, selectedClusters: [proxy.selectedClusters[0]] }))
      .rejects.toThrow('CUT');
    const malformed = encodeGeometryQueryProxy(proxy).slice();
    new DataView(malformed.buffer).setUint32(8, 0xffff_ffff, true);
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', malformed)),
      byte => byte.toString(16).padStart(2, '0')).join('');
    await expect(decodeGeometryQueryProxy(data.directory, malformed, hash)).rejects.toThrow('HEADER');
  });

  it('rejects inconsistent seam attributes across selected pages', async () => {
    const base = await buildGeometryQueryProxy(data.directory, data.provider, { maxError: 0 });
    const selected = new Set(base.selectedClusters);
    const pages = data.encoded.pages.map(page => page.bytes.slice());
    const decodedPages = await Promise.all(data.directory.pages.map(async page => {
      const { decodeVerifiedGeometryPage } = await import('../src/geometry-page-decode');
      return decodeVerifiedGeometryPage(page, pages[page.id]);
    }));
    const seen = new Map<number, number>();
    let conflict: { pageId: number; sourceId: number } | undefined;
    for (const page of decodedPages) for (const cluster of page.clusters) {
      if (!selected.has(cluster.id)) continue;
      const pageId = data.directory.clusterPages[cluster.id];
      for (const id of cluster.sourceVertexIds) {
        const first = seen.get(id);
        if (first !== undefined && first !== pageId) { conflict = { pageId, sourceId: id }; break; }
        seen.set(id, pageId);
      }
      if (conflict) break;
    }
    expect(conflict).toBeDefined();
    const modifiedSource = grid();
    modifiedSource.uvs[conflict!.sourceId * 2] += 0.125;
    const alternate = await encodeGeometryPages({ ...data.hierarchy, source: modifiedSource }, { maxPageBytes: 16_384 });
    const changedPage = alternate.pages[conflict!.pageId].bytes;
    const sha = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(changedPage))),
      byte => byte.toString(16).padStart(2, '0')).join('');
    const directory = { ...data.directory, pages: data.directory.pages.map(page => page.id === conflict!.pageId
      ? { ...page, encodedBytes: changedPage.byteLength, sha256: sha } : page) };
    const mixed: GeometryPageProvider = async page => page.id === conflict!.pageId ? changedPage.slice() : pages[page.id].slice();
    await expect(buildGeometryQueryProxy(directory, mixed, { maxError: 0 })).rejects.toThrow('SEAM_ATTRIBUTES');
  });
});
