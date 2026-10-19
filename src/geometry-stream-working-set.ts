import { validateGeometryStreamDirectory } from './geometry-stream-directory';
import type { GeometryStreamDirectory } from './geometry-stream-types';

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
  private readonly childPages: readonly (readonly number[])[];
  private previousAdmitted = new Set<number>();

  constructor(directory: GeometryStreamDirectory, maxPages: number) {
    this.directory = validateGeometryStreamDirectory(directory);
    if (!Number.isSafeInteger(maxPages) || maxPages < this.directory.rootPages.length
      || maxPages > this.directory.pages.length) throw new RangeError('GEOMETRY_STREAM_WORKING_SET_BUDGET');
    this.maxPages = maxPages;
    this.childPages = this.directory.groups.map(group => [...new Set(group.children.map(id => this.directory.clusterPages[id]))]
      .sort((a, b) => a - b));
  }

  plan(demands: readonly GeometryGroupDemand[]): GeometryStreamWorkingSetPlan {
    if (!Array.isArray(demands) || demands.length > this.directory.groups.length) {
      throw new RangeError('GEOMETRY_STREAM_WORKING_SET_DEMAND');
    }
    const seen = new Set<number>();
    const ordered: GeometryGroupDemand[] = [];
    for (let index = 0; index < demands.length; index++) {
      if (!Object.hasOwn(demands, index)) throw new RangeError('GEOMETRY_STREAM_WORKING_SET_DEMAND');
      const demand = demands[index];
      if (!demand || !Number.isSafeInteger(demand.groupId) || demand.groupId < 0
        || demand.groupId >= this.directory.groups.length || seen.has(demand.groupId)
        || this.directory.groups[demand.groupId].terminal
        || typeof demand.projectedError !== 'number' || !Number.isFinite(demand.projectedError)
        || demand.projectedError <= 0 || !Number.isFinite(Math.fround(demand.projectedError))) {
        throw new RangeError('GEOMETRY_STREAM_WORKING_SET_DEMAND');
      }
      seen.add(demand.groupId);
      ordered.push({ groupId: demand.groupId, projectedError: demand.projectedError });
    }

    // Prior admissions get a small deterministic preference only within their
    // own level. Dependencies are still evaluated from roots every frame.
    ordered.sort((a, b) => this.directory.groups[b.groupId].level - this.directory.groups[a.groupId].level
      || b.projectedError * (this.previousAdmitted.has(b.groupId) ? 1.1 : 1)
        - a.projectedError * (this.previousAdmitted.has(a.groupId) ? 1.1 : 1)
      || a.groupId - b.groupId);

    const rootPages = new Set(this.directory.rootPages);
    const pages = new Set(rootPages);
    const reachable = new Uint8Array(this.directory.clusters.length);
    for (const id of this.directory.roots) reachable[id] = 1;
    const admittedGroups: number[] = [], budgetDeferredGroups: number[] = [], dependencyDeferredGroups: number[] = [];
    const priorities = new Map<number, number>();
    for (const { groupId } of ordered) {
      const group = this.directory.groups[groupId];
      if (!group.parents.every(id => reachable[id] === 1)) {
        dependencyDeferredGroups.push(groupId);
        continue;
      }
      const childPages = this.childPages[groupId];
      const additional = childPages.reduce((count, id) => count + Number(!pages.has(id)), 0);
      if (pages.size + additional > this.maxPages) {
        budgetDeferredGroups.push(groupId);
        continue;
      }
      admittedGroups.push(groupId);
      for (const id of group.children) reachable[id] = 1;
      for (const pageId of childPages) {
        pages.add(pageId);
        if (!rootPages.has(pageId)) {
          priorities.set(pageId, Math.max(priorities.get(pageId) ?? -1, group.level));
        }
      }
    }
    this.previousAdmitted = new Set(admittedGroups);
    return { pages: [...pages].sort((a, b) => a - b), admittedGroups,
      budgetDeferredGroups, dependencyDeferredGroups,
      requestedPages: [...priorities].map(([pageId, priority]) => ({ pageId, priority }))
        .sort((a, b) => b.priority - a.priority || a.pageId - b.pageId) };
  }
}
