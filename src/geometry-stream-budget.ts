import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';
import type { GeometryGroupDemand } from './geometry-stream-working-set';
import { admitGeometryStreamGroups } from './geometry-stream-admission';
import { geometryGpuPoolLayout } from './geometry-gpu-pool-layout';

export interface GeometryStreamBudgetInput {
  readonly directory: GeometryStreamDirectory;
  /** Maximum projected error across relevant instances/views, keyed by group.
   * Omitted means unknown, [] means known but no visible refinement demand.
   * Both retain roots only; no invisible fine-detail reservation is implicit. */
  readonly demands?: readonly GeometryGroupDemand[];
  /** A device/caller may impose a smaller physical pool limit. */
  readonly maxPoolBytes?: number;
}
export interface GeometryStreamBudgetOptions {
  /** Sum of physical vertex/index slots AND per-stream GPU metadata. Does not
   * include selection buffers, CPU caches, upload staging or query proxies. */
  readonly maxResidentBytes: number;
}

/** Plan capacities before creating immutable per-stream GPU pools. This does
 * not resize live pools, borrow their slots, or promise that a future viewpoint
 * fits. The output can seed a later runtime allocator without changing demand
 * semantics; current callers must recreate pools to apply a new capacity plan.
 * Work is bounded by 4,096 streams and aggregate limits of 262,144 clusters,
 * 262,144 groups and 262,144 pages across all inputs. */
export function planGeometryStreamBudgets(inputs: readonly GeometryStreamBudgetInput[], options: GeometryStreamBudgetOptions) {
  const budget = options?.maxResidentBytes;
  if (!Number.isSafeInteger(budget) || budget < 1 || budget > 8 * 1024 ** 3) throw new RangeError('GEOMETRY_STREAM_SCENE_BUDGET');
  if (!Array.isArray(inputs) || !inputs.length || inputs.length > 4096) throw new RangeError('GEOMETRY_STREAM_SCENE_INPUTS');
  let clusters = 0, groups = 0, pages = 0;
  for (const input of inputs) {
    if (!input?.directory || !Array.isArray(input.directory.clusters) || !Array.isArray(input.directory.groups)
      || !Array.isArray(input.directory.pages)) throw new RangeError('GEOMETRY_STREAM_SCENE_INPUTS');
    clusters += input.directory.clusters.length; groups += input.directory.groups.length; pages += input.directory.pages.length;
    if (Math.max(clusters, groups, pages) > 262144) throw new RangeError('GEOMETRY_STREAM_SCENE_WORK_BUDGET');
  }
  const prepared = inputs.map(input => {
    const directory = validateGeometryStreamDirectory(input.directory), layout = geometryGpuPoolLayout(directory);
    const maxPoolBytes = input.maxPoolBytes ?? 256 * 1024 ** 2;
    if (!Number.isSafeInteger(maxPoolBytes) || maxPoolBytes < 1 || maxPoolBytes > 256 * 1024 ** 2) {
      throw new RangeError('GEOMETRY_STREAM_SCENE_POOL_BUDGET');
    }
    if (layout.rootBytes > maxPoolBytes) throw new RangeError('GEOMETRY_STREAM_SCENE_ROOT_POOL_BUDGET');
    return { directory, layout, demands: input.demands === undefined ? [] : input.demands, demandStatus: input.demands === undefined ? 'unknown' as const : 'known' as const,
      maxPages: Math.min(directory.pages.length, Math.floor((maxPoolBytes - layout.metadataBytes) / layout.bytesPerSlot)),
      pageBytes: layout.bytesPerSlot };
  });
  const rootBytes = prepared.reduce((sum, input) => sum + input.layout.rootBytes, 0);
  if (rootBytes > budget) throw new RangeError('GEOMETRY_STREAM_SCENE_ROOT_BUDGET');
  const plans = admitGeometryStreamGroups(prepared, budget - rootBytes);
  const streams = prepared.map((input, index) => {
    const plan = plans[index], slots = plan.pages.length;
    return { index, primitiveKey: input.directory.primitiveKey, demandStatus: input.demandStatus,
      slots, rootSlots: input.directory.rootPages.length, totalPages: input.directory.pages.length,
      maxResidentBytes: input.layout.metadataBytes + slots * input.layout.bytesPerSlot,
      bytesPerSlot: input.layout.bytesPerSlot, metadataBytes: input.layout.metadataBytes, ...plan };
  });
  const allocatedBytes = streams.reduce((sum, stream) => sum + stream.maxResidentBytes, 0);
  return { policy: 'projected-error-with-dependency-retention' as const,
    allocation: 'immutable-pool-capacity-plan' as const, maxResidentBytes: budget, allocatedBytes,
    unallocatedBytes: budget - allocatedBytes, rootBytes, streams };
}
