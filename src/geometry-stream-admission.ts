import type { GeometryStreamDirectory } from './geometry-stream-types';
import type { GeometryGroupDemand, GeometryStreamWorkingSetPlan } from './geometry-stream-working-set';

/** Internal admission input. Directories must already be validated snapshots. */
export interface GeometryAdmissionInput {
  directory: GeometryStreamDirectory;
  demands: readonly GeometryGroupDemand[];
  maxPages: number;
  pageBytes: number;
  previousAdmitted?: ReadonlySet<number>;
}
type Candidate = { stream: number; groupId: number; projectedError: number; score: number };

/** Same dependency-aware priority order for scene capacity planning and the
 * runtime working set. Each demand enters the heap at most once. Ancestors and
 * roots stay resident; a partial refinement group is never admitted. */
export function admitGeometryStreamGroups(inputs: readonly GeometryAdmissionInput[], availableBytes: number): GeometryStreamWorkingSetPlan[] {
  const heap: Candidate[] = [];
  const before = (a: Candidate, b: Candidate) => a.score > b.score
    || a.score === b.score && (a.stream < b.stream || a.stream === b.stream && a.groupId < b.groupId);
  const push = (value: Candidate) => {
    let at = heap.length; heap.push(value);
    while (at > 0) { const parent = (at - 1) >> 1; if (!before(value, heap[parent])) break;
      heap[at] = heap[parent]; at = parent; }
    heap[at] = value;
  };
  const pop = (): Candidate | undefined => {
    const value = heap[0], last = heap.pop();
    if (heap.length && last) {
      let at = 0;
      while (at * 2 + 1 < heap.length) {
        let child = at * 2 + 1;
        if (child + 1 < heap.length && before(heap[child + 1], heap[child])) child++;
        if (!before(heap[child], last)) break;
        heap[at] = heap[child]; at = child;
      }
      heap[at] = last;
    }
    return value;
  };
  // Validate every demand before scheduling or exposing any new admission.
  const states = inputs.map((input, stream) => {
    const { directory, demands } = input;
    if (!Array.isArray(demands) || demands.length > directory.groups.length) throw new RangeError('GEOMETRY_STREAM_WORKING_SET_DEMAND');
    const candidates = new Map<number, Candidate>();
    for (const demand of demands) {
      if (!demand || !Number.isSafeInteger(demand.groupId) || demand.groupId < 0
        || demand.groupId >= directory.groups.length || candidates.has(demand.groupId)
        || directory.groups[demand.groupId].terminal || !Number.isFinite(demand.projectedError)
        || demand.projectedError <= 0 || !Number.isFinite(Math.fround(demand.projectedError))) {
        throw new RangeError('GEOMETRY_STREAM_WORKING_SET_DEMAND');
      }
      candidates.set(demand.groupId, { stream, ...demand,
        score: demand.projectedError * (input.previousAdmitted?.has(demand.groupId) ? 1.1 : 1) });
    }
    const roots = new Set(directory.roots), rootPages = new Set(directory.rootPages);
    const dependencies = new Map<number, number>(), dependents = new Map<number, number[]>();
    for (const { groupId } of candidates.values()) {
      const parents = new Set(directory.groups[groupId].parents.filter(id => !roots.has(id)).map(id => directory.clusters[id].group));
      dependencies.set(groupId, parents.size);
      for (const parent of parents) {
        const children = dependents.get(parent) ?? []; children.push(groupId); dependents.set(parent, children);
      }
    }
    return { candidates, dependencies, dependents, rootPages, pages: new Set(rootPages),
      admittedGroups: [] as number[], budgetDeferredGroups: [] as number[], processed: new Set<number>(), priorities: new Map<number, number>() };
  });
  states.forEach(state => { for (const [id, count] of state.dependencies) if (count === 0) push(state.candidates.get(id)!); });
  let candidate: Candidate | undefined;
  while ((candidate = pop())) {
    const { stream, groupId } = candidate, input = inputs[stream], state = states[stream];
    const group = input.directory.groups[groupId];
    const childPages = new Set(group.children.map(id => input.directory.clusterPages[id]));
    let additional = 0;
    for (const id of childPages) if (!state.pages.has(id)) additional++;
    const bytes = additional * input.pageBytes;
    state.processed.add(groupId);
    if (state.pages.size + additional > input.maxPages || bytes > availableBytes) {
      state.budgetDeferredGroups.push(groupId); continue;
    }
    availableBytes -= bytes; state.admittedGroups.push(groupId);
    for (const page of childPages) {
      state.pages.add(page);
      if (!state.rootPages.has(page)) state.priorities.set(page, Math.max(state.priorities.get(page) ?? -1, group.level));
    }
    for (const child of state.dependents.get(groupId) ?? []) {
      const remaining = state.dependencies.get(child)! - 1; state.dependencies.set(child, remaining);
      if (!remaining) push(state.candidates.get(child)!);
    }
  }
  return states.map(state => ({ pages: [...state.pages].sort((a, b) => a - b), admittedGroups: state.admittedGroups,
    budgetDeferredGroups: state.budgetDeferredGroups,
    dependencyDeferredGroups: [...state.candidates.keys()].filter(id => !state.processed.has(id)).sort((a, b) => a - b),
    requestedPages: [...state.priorities].map(([pageId, priority]) => ({ pageId, priority }))
      .sort((a, b) => b.priority - a.priority || a.pageId - b.pageId) }));
}
