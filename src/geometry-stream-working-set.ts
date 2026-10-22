import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';
import { admitGeometryStreamGroups } from './geometry-stream-admission';

export interface GeometryGroupDemand {
  groupId: number;
  projectedError: number;
}

export interface GeometryStreamWorkingSetPlan {
  pages: number[];
  admittedGroups: number[];
  budgetDeferredGroups: number[];
  dependencyDeferredGroups: number[];
  requestedPages: { pageId: number; priority: number }[];
}

/** Bounded page admission intent; actual GPU cut and residency remain separate. */
export class GeometryStreamWorkingSet {
  readonly directory: GeometryStreamDirectory;
  private readonly maxPages: number;
  private previousAdmitted = new Set<number>();

  constructor(directory: GeometryStreamDirectory, maxPages: number) {
    this.directory = validateGeometryStreamDirectory(directory);
    if (!Number.isSafeInteger(maxPages) || maxPages < this.directory.rootPages.length
      || maxPages > this.directory.pages.length) throw new RangeError('GEOMETRY_STREAM_WORKING_SET_BUDGET');
    this.maxPages = maxPages;
  }

  plan(demands: readonly GeometryGroupDemand[]): GeometryStreamWorkingSetPlan {
    const [plan] = admitGeometryStreamGroups([{ directory: this.directory, demands,
      maxPages: this.maxPages, pageBytes: 1, previousAdmitted: this.previousAdmitted }], this.maxPages - this.directory.rootPages.length);
    this.previousAdmitted = new Set(plan.admittedGroups);
    return plan;
  }
}
