/** Two-level scan: 512 items per block, at most 512 blocks. Each pair has at
 * most 512 triangles, so even the worst prefix fits in u32 (402,653,184). */
export function geometryStreamCompaction(capacityPairs: number) {
  if (!Number.isSafeInteger(capacityPairs) || capacityPairs < 1 || capacityPairs > 262144) {
    throw new Error('GEOMETRY_STREAM_WORK_BUDGET');
  }
  const blocks = Math.ceil(capacityPairs / 512);
  return { capacityPairs, blocks, scratchBytes: blocks > 1 ? (capacityPairs + blocks) * 4 : 0 };
}
