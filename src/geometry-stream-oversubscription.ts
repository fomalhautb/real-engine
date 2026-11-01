import { geometryGpuPoolLayout } from './geometry-gpu-pool-layout';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';
import { GeometryStreamRuntimeBudget, type GeometryStreamRuntimeLimits } from './geometry-stream-runtime-budget';
import type { GeometryStreamDirectory } from './geometry-stream-types';

/** Runtime budget whose steady-state geometry pool is a fixed fraction of the
 * unique resident-format data of a set of streams (the G03 ratio). */
export interface OversubscribedGeometryBudget {
  readonly streams: number;
  readonly instances: number;
  /** Sum over streams of every cluster's resident-format bytes (vertices plus triangle indices), shared by instances. */
  readonly uniqueResidentBytes: number;
  /** uniqueResidentBytes / targetUniqueToPoolRatio: page slots plus location metadata, all streams. */
  readonly poolBudgetBytes: number;
  readonly metadataBytes: number;
  /** Live bytes that do not scale with slot count (selection, parameters, per-stream fixed buffers). */
  readonly fixedBytes: number;
  /** Largest per-stream share of one slot's live cost (page storage over storage plus paged outputs). */
  readonly storageShare: number;
  /** Every stream with roots only, as the runtime budget accounts it. */
  readonly rootBytes: number;
  /** Every stream with every page resident. */
  readonly fullResidencyBytes: number;
  readonly maxResidentBytes: number;
  /** maxResidentBytes plus the full size of the largest maxConcurrentMigrations streams (candidate owners). */
  readonly maxPeakBytes: number;
  readonly uniqueToPool: number;
  readonly uniqueToResidentBudget: number;
  readonly uniqueToPeakBudget: number;
  readonly fullResidencyToResidentBudget: number;
}

/** The procedural gallery's accounting (src/geometry-demo-streamed-scene.ts)
 * for any set of streams: the live budget is the fixed live cost plus enough
 * slots, at the worst storage share, to hold poolBudgetBytes of page storage.
 * Roots must fit strictly inside it. Work is O(streams + clusters). */
export function oversubscribedGeometryBudget(inputs: readonly { directory: GeometryStreamDirectory; capacity: number }[],
  targetUniqueToPoolRatio: number, limits: GeometryStreamRuntimeLimits, maxConcurrentMigrations: number): OversubscribedGeometryBudget {
  if (!inputs.length) throw new RangeError('GEOMETRY_OVERSUBSCRIPTION_STREAMS');
  if (!Number.isFinite(targetUniqueToPoolRatio) || targetUniqueToPoolRatio < 1 || targetUniqueToPoolRatio > 64) {
    throw new RangeError('GEOMETRY_OVERSUBSCRIPTION_RATIO');
  }
  if (!Number.isSafeInteger(maxConcurrentMigrations) || maxConcurrentMigrations < 1 || maxConcurrentMigrations > 16) {
    throw new RangeError('GEOMETRY_OVERSUBSCRIPTION_MIGRATIONS');
  }
  const sizing = new GeometryStreamRuntimeBudget(inputs, { maxResidentBytes: 8 * 1024 ** 3, maxPeakBytes: 8 * 1024 ** 3 }, limits);
  const full = inputs.map((input, index) => sizing.state(index, input.directory.pages.length));
  const layouts = inputs.map(input => geometryGpuPoolLayout(input.directory));
  const uniqueResidentBytes = inputs.reduce((sum, input) => sum + input.directory.clusters.reduce((bytes, cluster) =>
    bytes + cluster.vertexCount * GEOMETRY_GPU_VERTEX_BYTES + cluster.triangleCount * 12, 0), 0);
  const fullResidencyBytes = full.reduce((sum, state) => sum + state.totalBytes, 0);
  const storageShare = Math.max(...layouts.map((layout, index) =>
    layout.bytesPerSlot / (layout.bytesPerSlot + full[index].outputBytes / full[index].slots)));
  const fixedBytes = fullResidencyBytes
    - full.reduce((sum, state, index) => sum + state.slots * (layouts[index].bytesPerSlot + state.outputBytes / state.slots), 0);
  const metadataBytes = layouts.reduce((sum, layout) => sum + layout.metadataBytes, 0);
  const poolBudgetBytes = Math.floor(uniqueResidentBytes / targetUniqueToPoolRatio);
  const maxResidentBytes = Math.floor(fixedBytes + Math.max(0, poolBudgetBytes - metadataBytes) / storageShare);
  if (sizing.rootBytes >= maxResidentBytes) {
    throw new RangeError(`GEOMETRY_OVERSUBSCRIPTION_ROOTS_EXCEED_BUDGET:${sizing.rootBytes}:${maxResidentBytes}`);
  }
  const largest = full.map(state => state.totalBytes).sort((a, b) => b - a);
  const maxPeakBytes = maxResidentBytes + largest.slice(0, maxConcurrentMigrations).reduce((sum, bytes) => sum + bytes, 0);
  return Object.freeze({ streams: inputs.length, instances: inputs.reduce((sum, input) => sum + input.capacity, 0),
    uniqueResidentBytes, poolBudgetBytes, metadataBytes, fixedBytes, storageShare, rootBytes: sizing.rootBytes, fullResidencyBytes,
    maxResidentBytes, maxPeakBytes, uniqueToPool: uniqueResidentBytes / poolBudgetBytes,
    uniqueToResidentBudget: uniqueResidentBytes / maxResidentBytes, uniqueToPeakBudget: uniqueResidentBytes / maxPeakBytes,
    fullResidencyToResidentBudget: fullResidencyBytes / maxResidentBytes });
}
