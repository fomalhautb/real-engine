import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';
import type { GeometryHierarchy } from './geometry-hierarchy-types';

/** Exact structural coverage of one instance's cut over directory metadata. */
export interface GeometryCutCoverage {
  readonly lod0Clusters: number;
  /** LOD0 clusters represented neither by themselves nor by complete ancestor replacement: holes. */
  readonly uncovered: number;
  /** LOD0 clusters represented more than once: overlapping surfaces. */
  readonly overlapping: number;
  /** Non-terminal groups whose parents are represented unequally: a split replacement. */
  readonly partialGroups: number;
  readonly selectedClusters: number;
}

/** Every LOD0 cluster must be represented exactly once, by itself or by a chain
 * of complete parent-group replacements. This checks topology only: it uses no
 * page payload, screen error or GPU state, so it complements (does not replace)
 * pixel comparison against a full-detail reference. Work is O(clusters + group
 * references), bounded by directory validation. */
export function geometryCutCoverage(directoryInput: GeometryStreamDirectory, selected: Iterable<number>): GeometryCutCoverage {
  const directory = validateGeometryStreamDirectory(directoryInput);
  const count = directory.clusters.length, chosen = new Uint8Array(count);
  let selectedClusters = 0;
  for (const id of selected) {
    if (!Number.isSafeInteger(id) || id < 0 || id >= count) throw new RangeError('GEOMETRY_CUT_COVERAGE_CLUSTER');
    if (chosen[id]) throw new RangeError('GEOMETRY_CUT_COVERAGE_DUPLICATE');
    chosen[id] = 1; selectedClusters++;
  }
  const order = Array.from({ length: count }, (_, id) => id)
    .sort((a, b) => directory.clusters[b].level - directory.clusters[a].level || a - b);
  // Representation count from the cluster itself and from complete ancestors.
  const represented = new Uint32Array(count);
  const groupState = new Int32Array(directory.groups.length).fill(-1);
  let partialGroups = 0, lod0Clusters = 0, uncovered = 0, overlapping = 0;
  for (const id of order) {
    const cluster = directory.clusters[id], group = directory.groups[cluster.group];
    let above = 0;
    if (!group.terminal) {
      if (groupState[cluster.group] < 0) {
        let low = Infinity, high = 0;
        for (const parent of group.parents) {
          if (directory.clusters[parent].level <= cluster.level) throw new RangeError('GEOMETRY_CUT_COVERAGE_LEVEL_ORDER');
          low = Math.min(low, represented[parent]); high = Math.max(high, represented[parent]);
        }
        if (low !== high) partialGroups++;
        // A split replacement leaves some child surface unrepresented: count it as a hole.
        groupState[cluster.group] = Number.isFinite(low) ? low : 0;
      }
      above = groupState[cluster.group];
    }
    represented[id] = above + chosen[id];
    if (cluster.level === 0) {
      lod0Clusters++;
      if (represented[id] === 0) uncovered++;
      else if (represented[id] > 1) overlapping++;
    }
  }
  return { lod0Clusters, uncovered, overlapping, partialGroups, selectedClusters };
}

/** Edge topology of the triangles a cut actually draws. */
export interface GeometryCutWatertightness {
  readonly triangles: number;
  /** Directed edges whose reverse is absent after welding equal positions: cracks or open borders. */
  readonly boundaryEdges: number;
  /** Directed edges used more than once: folds, overlaps or duplicated surface. */
  readonly repeatedEdges: number;
}

/** Welds vertices with bit-identical positions, then requires every directed
 * edge of the cut to appear exactly once with its reverse exactly once. For a
 * closed source surface this proves the drawn cut is crack-free regardless of
 * screen error; open sources report their own borders. Work is bounded by the
 * cut's triangle count. */
export function geometryCutWatertightness(hierarchy: Pick<GeometryHierarchy, 'source' | 'clusters'>, selected: Iterable<number>): GeometryCutWatertightness {
  const source = hierarchy.source, view = new DataView(source.vertices), canonical = new Map<string, number>();
  const weld = (vertex: number) => {
    const offset = vertex * 80, key = `${view.getUint32(offset, true)},${view.getUint32(offset + 4, true)},${view.getUint32(offset + 8, true)}`;
    let id = canonical.get(key);
    if (id === undefined) { id = canonical.size; canonical.set(key, id); }
    return id;
  };
  const edges = new Map<number, number>();
  let triangles = 0;
  for (const id of selected) {
    const cluster = hierarchy.clusters[id];
    if (!cluster) throw new RangeError('GEOMETRY_CUT_WATERTIGHT_CLUSTER');
    triangles += cluster.triangles.length / 3;
    if (triangles > 4_000_000) throw new RangeError('GEOMETRY_CUT_WATERTIGHT_BUDGET');
    for (let t = 0; t < cluster.triangles.length; t += 3) {
      const v = [0, 1, 2].map(k => weld(cluster.vertices[cluster.triangles[t + k]]));
      for (let k = 0; k < 3; k++) {
        const key = v[k] * 0x4000000 + v[(k + 1) % 3];
        edges.set(key, (edges.get(key) ?? 0) + 1);
      }
    }
  }
  let boundaryEdges = 0, repeatedEdges = 0;
  for (const [key, count] of edges) {
    if (count > 1) repeatedEdges++;
    const a = Math.floor(key / 0x4000000), b = key % 0x4000000;
    if (!edges.has(b * 0x4000000 + a)) boundaryEdges++;
  }
  return { triangles, boundaryEdges, repeatedEdges };
}

/** Decoded payload of one cluster: interleaved 80-byte rest records (position
 * first, as float32) and local triangle corners, as produced by page decoding. */
export interface GeometryClusterPayload {
  readonly rest: Uint8Array;
  readonly localTriangles: Uint8Array;
}

/** Signed welded boundary preservation of every non-terminal group. */
export interface GeometryGroupBoundaryAudit {
  readonly groups: number;
  /** Non-terminal groups whose children and parents were compared. */
  readonly checkedGroups: number;
  /** Groups whose parents' signed welded boundary differs from their children's. */
  readonly mismatchedGroups: number;
  readonly mismatchExamples: readonly number[];
  /** Undirected welded edges with a nonzero signed balance over all LOD0 clusters: the source's open borders. */
  readonly lod0BoundaryEdges: number;
  readonly clusters: number;
  readonly triangles: number;
}

const REST_RECORD_BYTES = 80;
const WELD_KEY_SCALE = 0x4000000;
const BOUNDARY_AUDIT_TRIANGLES = 16_000_000;

/** Exact crack audit of a streamed hierarchy over the payload bytes that stream.
 * Vertices are welded by equal float32 positions (the builder's position remap),
 * then every non-terminal group must have the same signed undirected edge
 * balance (opposite interior half edges cancel) in its parent clusters as in its
 * children. Because the balance is additive over disjoint triangle sets, every
 * cut that geometryCutCoverage accepts (each LOD0 cluster represented exactly
 * once through complete group replacements) then has exactly the net boundary of
 * LOD0, for open and closed sources alike: a crack or T-junction between
 * clusters of different levels would add a nonzero edge LOD0 does not have.
 * Folds that keep the net balance are not cracks and are not reported here.
 * Work is bounded: each cluster is visited at most three times (as a child, as a
 * parent and as LOD0) and the total is capped. */
export function geometryGroupBoundaryAudit(directoryInput: GeometryStreamDirectory,
  payloads: ReadonlyMap<number, GeometryClusterPayload> | ((clusterId: number) => GeometryClusterPayload | undefined)): GeometryGroupBoundaryAudit {
  const directory = validateGeometryStreamDirectory(directoryInput);
  const payload = typeof payloads === 'function' ? payloads : (id: number) => payloads.get(id);
  const canonical = new Map<string, number>(), welded = new Map<number, Uint32Array>();
  let visited = 0;
  const weld = (id: number) => {
    const known = welded.get(id);
    if (known) return known;
    const cluster = directory.clusters[id], data = payload(id);
    if (!data || data.rest.byteLength !== cluster.vertexCount * REST_RECORD_BYTES
      || data.localTriangles.length !== cluster.triangleCount * 3) throw new RangeError('GEOMETRY_GROUP_BOUNDARY_PAYLOAD: ' + id);
    const view = new DataView(data.rest.buffer, data.rest.byteOffset, data.rest.byteLength);
    const ids = new Uint32Array(cluster.vertexCount);
    for (let vertex = 0; vertex < cluster.vertexCount; vertex++) {
      const offset = vertex * REST_RECORD_BYTES;
      const key = `${view.getFloat32(offset, true)},${view.getFloat32(offset + 4, true)},${view.getFloat32(offset + 8, true)}`;
      let weldedId = canonical.get(key);
      if (weldedId === undefined) {
        weldedId = canonical.size;
        if (weldedId >= WELD_KEY_SCALE) throw new RangeError('GEOMETRY_GROUP_BOUNDARY_VERTEX_BUDGET');
        canonical.set(key, weldedId);
      }
      ids[vertex] = weldedId;
    }
    welded.set(id, ids);
    return ids;
  };
  const balance = (ids: readonly number[]) => {
    const edges = new Map<number, number>();
    for (const id of ids) {
      const vertices = weld(id), corners = payload(id)!.localTriangles;
      visited += corners.length / 3;
      if (visited > BOUNDARY_AUDIT_TRIANGLES) throw new RangeError('GEOMETRY_GROUP_BOUNDARY_BUDGET');
      for (let t = 0; t < corners.length; t += 3) for (let e = 0; e < 3; e++) {
        const local = corners[t + e], next = corners[t + (e + 1) % 3];
        if (local >= vertices.length || next >= vertices.length) throw new RangeError('GEOMETRY_GROUP_BOUNDARY_CORNER: ' + id);
        const a = vertices[local], b = vertices[next];
        if (a === b) continue;
        const key = a < b ? a * WELD_KEY_SCALE + b : b * WELD_KEY_SCALE + a;
        const count = (edges.get(key) ?? 0) + (a < b ? 1 : -1);
        if (count) edges.set(key, count); else edges.delete(key);
      }
    }
    return edges;
  };
  let checkedGroups = 0, mismatchedGroups = 0, triangles = 0;
  const mismatchExamples: number[] = [];
  for (const [index, group] of directory.groups.entries()) {
    if (group.terminal) continue;
    checkedGroups++;
    const before = balance(group.children), after = balance(group.parents);
    const same = before.size === after.size && [...before].every(([key, count]) => after.get(key) === count);
    if (!same) { mismatchedGroups++; if (mismatchExamples.length < 8) mismatchExamples.push(index); }
    // Each cluster is a child of exactly one group: its welded ids are no longer needed here.
    for (const id of group.children) welded.delete(id);
  }
  const lod0 = directory.clusters.flatMap((cluster, id) => cluster.level === 0 ? [id] : []);
  const lod0BoundaryEdges = balance(lod0).size;
  for (const cluster of directory.clusters) triangles += cluster.triangleCount;
  return { groups: directory.groups.length, checkedGroups, mismatchedGroups, mismatchExamples, lod0BoundaryEdges,
    clusters: directory.clusters.length, triangles };
}
