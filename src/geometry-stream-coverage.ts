import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';

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
