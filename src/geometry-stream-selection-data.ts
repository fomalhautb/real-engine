import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';

const CLUSTER_WORDS = 16;
const GROUP_WORDS = 12;
const NO_REFINED_GROUP = 0xffff_ffff;

export interface PackedGeometryStreamSelection {
  /** Owned, immutable snapshot of the input metadata. */
  directory: GeometryStreamDirectory;
  /** Cluster records, group records, then parent/child cluster IDs. */
  words: Uint32Array<ArrayBuffer>;
  /** Offsets in u32 words from the beginning of words. */
  clusterOffset: number;
  groupOffset: number;
  edgeOffset: number;
  /** Ranges of packed groups, including terminal groups, in descending level order. */
  levels: { level: number; firstGroup: number; groupCount: number }[];
  /** Original directory group ID for each packed group ordinal. */
  groupIds: number[];
}

/** Pack validated stream metadata for a bounded GPU selection pass. */
export function packGeometryStreamSelection(input: GeometryStreamDirectory): PackedGeometryStreamSelection {
  const directory = validateGeometryStreamDirectory(input);
  const clusterOffset = 0;
  const groupOffset = directory.clusters.length * CLUSTER_WORDS;
  const edgeOffset = groupOffset + directory.groups.length * GROUP_WORDS;
  const edgeWords = directory.groups.reduce((count, group) => count + group.parents.length + group.children.length, 0);
  const words = new Uint32Array(edgeOffset + edgeWords);
  const view = new DataView(words.buffer);
  const float = (word: number, value: number) => view.setFloat32(word * 4, value, true);
  const uint = (word: number, value: number) => view.setUint32(word * 4, value, true);
  const roots = new Set(directory.roots);

  for (let id = 0; id < directory.clusters.length; id++) {
    const cluster = directory.clusters[id];
    const base = clusterOffset + id * CLUSTER_WORDS;
    for (let axis = 0; axis < 3; axis++) {
      float(base + axis, cluster.bounds.min[axis]);
      float(base + 4 + axis, cluster.bounds.max[axis]);
      float(base + 8 + axis, cluster.lodBounds.min[axis]);
      float(base + 12 + axis, cluster.lodBounds.max[axis]);
    }
    uint(base + 3, roots.has(id) ? 1 : 0);
    uint(base + 7, directory.clusterPages[id]);
    uint(base + 11, cluster.level);
    uint(base + 15, cluster.refinedGroup ?? NO_REFINED_GROUP);
  }

  const groupIds = directory.groups.map((_, id) => id)
    .sort((a, b) => directory.groups[b].level - directory.groups[a].level || a - b);
  const levels: PackedGeometryStreamSelection['levels'] = [];
  let nextEdge = 0;
  for (let ordinal = 0; ordinal < groupIds.length; ordinal++) {
    const group = directory.groups[groupIds[ordinal]];
    const currentLevel = levels[levels.length - 1];
    if (currentLevel?.level === group.level) currentLevel.groupCount++;
    else levels.push({ level: group.level, firstGroup: ordinal, groupCount: 1 });

    const base = groupOffset + ordinal * GROUP_WORDS;
    for (let axis = 0; axis < 3; axis++) {
      float(base + axis, group.bounds.min[axis]);
      float(base + 4 + axis, group.bounds.max[axis]);
    }
    float(base + 3, group.error);
    uint(base + 7, group.terminal ? 1 : 0);
    uint(base + 8, nextEdge);
    uint(base + 9, group.parents.length);
    for (const id of group.parents) uint(edgeOffset + nextEdge++, id);
    uint(base + 10, nextEdge);
    uint(base + 11, group.children.length);
    for (const id of group.children) uint(edgeOffset + nextEdge++, id);
  }
  return { directory, words, clusterOffset, groupOffset, edgeOffset, levels, groupIds };
}
