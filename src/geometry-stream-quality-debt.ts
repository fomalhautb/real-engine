import { GeometryStreamCut, type GeometryStreamView } from './geometry-stream-cut';
import type { GeometryStreamDirectory } from './geometry-stream-types';

export type GeometryStreamQualityDebtCause =
  'capacity-deferred' | 'admitted-not-resident' | 'failed' | 'mixed';

export interface GeometryStreamQualityDebtInput {
  readonly directory: GeometryStreamDirectory;
  /** One view per instance, with that instance's model. These must describe the
   * same intended frame; this helper cannot synchronize asynchronous GPU feedback. */
  readonly views: readonly GeometryStreamView[];
  /** GPU-published residency, not decoded CPU pages or queued uploads. */
  readonly residentPages: ReadonlySet<number>;
  /** Pages admitted by the capacity plan for the same intended frame. */
  readonly admittedPages: ReadonlySet<number>;
  readonly failedPages?: ReadonlySet<number>;
}

export interface GeometryStreamQualityDebtDetail {
  readonly instanceIndex: number;
  readonly groupId: number;
  /** Existing accumulated simplification metric projected to pixels, not a
   * certified screen-space or silhouette error bound. */
  readonly projectedApproximateErrorPixels: number;
  readonly requestedErrorPixels: number;
  readonly cause: GeometryStreamQualityDebtCause;
  readonly missingChildPages: readonly number[];
  readonly capacityDeferredPages: readonly number[];
  readonly admittedNotResidentPages: readonly number[];
  readonly failedPages: readonly number[];
}

export interface GeometryStreamQualityDebt {
  readonly basis: 'cpu-predicted-visible-cut';
  readonly instanceCount: number;
  readonly fallbackGroupInstances: number;
  readonly maxProjectedApproximateErrorPixels: number;
  /** Exclusive group-instance buckets; mixed means two or more causes. */
  readonly groupsByCause: Readonly<Record<GeometryStreamQualityDebtCause, number>>;
  readonly details: readonly GeometryStreamQualityDebtDetail[];
}

const MAX_VIEWS = 256;
const MAX_GROUP_VIEW_PAIRS = 262_144;
// Directory validation bounds total parent + child references by 2 * clusters.
// This gates adjacency-heavy replay; it is an estimate, not a count of every
// operation (plan() also iterates and sorts selected clusters).
const MAX_REPLAY_WORK_ESTIMATE = 1_048_576;

function checkedPages(pages: ReadonlySet<number> | undefined, count: number, name: string): ReadonlySet<number> {
  if (!pages || typeof pages.has !== 'function' || typeof pages[Symbol.iterator] !== 'function'
    || !Number.isSafeInteger(pages.size) || pages.size < 0 || pages.size > count) throw new RangeError(name);
  for (const id of pages) if (!Number.isSafeInteger(id) || id < 0 || id >= count) throw new RangeError(name);
  return pages;
}

/** Replay the visible CPU cut against caller-supplied residency and admission.
 * This is a bounded prediction, not drawn GPU statistics: Hi-Z, frozen GPU view,
 * and asynchronous frame alignment are outside this contract. No GPU readback
 * or geometry payload is used. */
export function diagnoseGeometryStreamQualityDebt(input: GeometryStreamQualityDebtInput): GeometryStreamQualityDebt {
  const cut = new GeometryStreamCut(input.directory);
  const { directory } = cut;
  const views = input.views;
  if (!Array.isArray(views) || views.length > MAX_VIEWS
    || views.length * directory.groups.length > MAX_GROUP_VIEW_PAIRS
    || views.length * (2 * directory.groups.length + 2 * directory.clusters.length) > MAX_REPLAY_WORK_ESTIMATE) {
    throw new RangeError('GEOMETRY_STREAM_QUALITY_DEBT_WORK_BUDGET');
  }
  const resident = checkedPages(input.residentPages, directory.pages.length, 'GEOMETRY_STREAM_QUALITY_DEBT_RESIDENCY');
  const admitted = checkedPages(input.admittedPages, directory.pages.length, 'GEOMETRY_STREAM_QUALITY_DEBT_ADMISSION');
  const failed = input.failedPages === undefined ? new Set<number>()
    : checkedPages(input.failedPages, directory.pages.length, 'GEOMETRY_STREAM_QUALITY_DEBT_FAILURES');
  for (const page of directory.rootPages) {
    if (!resident.has(page)) throw new Error(`GEOMETRY_STREAM_ROOT_PAGE_MISSING:${page}`);
    if (!admitted.has(page)) throw new RangeError('GEOMETRY_STREAM_QUALITY_DEBT_ADMISSION');
  }

  const details: GeometryStreamQualityDebtDetail[] = [];
  const groupsByCause = { 'capacity-deferred': 0, 'admitted-not-resident': 0, failed: 0, mixed: 0 };
  let maxProjectedApproximateErrorPixels = 0;
  for (const [instanceIndex, view] of views.entries()) {
    // plan() identifies the reachable fallback frontier; demand alone includes
    // descendants whose parents were never selected and cannot be drawn yet.
    const fallback = new Set(cut.plan(resident, view).fallbackGroups);
    if (!fallback.size) continue;
    for (const demand of cut.demands([view])) {
      if (!fallback.has(demand.groupId)) continue; // Exclude off-screen fallback.
      const group = directory.groups[demand.groupId];
      const missing = [...new Set(group.children.map(child => directory.clusterPages[child])
        .filter(page => !resident.has(page)))].sort((a, b) => a - b);
      const capacityDeferredPages: number[] = [], admittedNotResidentPages: number[] = [], failedPages: number[] = [];
      for (const page of missing) {
        if (failed.has(page)) failedPages.push(page);
        else if (admitted.has(page)) admittedNotResidentPages.push(page);
        else capacityDeferredPages.push(page);
      }
      const causeCount = Number(capacityDeferredPages.length > 0)
        + Number(admittedNotResidentPages.length > 0) + Number(failedPages.length > 0);
      const cause: GeometryStreamQualityDebtCause = causeCount > 1 ? 'mixed'
        : capacityDeferredPages.length ? 'capacity-deferred'
          : admittedNotResidentPages.length ? 'admitted-not-resident' : 'failed';
      groupsByCause[cause]++;
      maxProjectedApproximateErrorPixels = Math.max(maxProjectedApproximateErrorPixels, demand.projectedError);
      details.push({ instanceIndex, groupId: demand.groupId,
        projectedApproximateErrorPixels: demand.projectedError, requestedErrorPixels: view.maxErrorPixels,
        cause, missingChildPages: missing, capacityDeferredPages, admittedNotResidentPages, failedPages });
    }
  }
  return { basis: 'cpu-predicted-visible-cut', instanceCount: views.length,
    fallbackGroupInstances: details.length, maxProjectedApproximateErrorPixels, groupsByCause, details };
}
