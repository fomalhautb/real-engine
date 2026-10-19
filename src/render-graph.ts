/** A named resource used for scheduling. GPU allocation remains the caller's responsibility. */
export interface RenderResource {
  name: string;
  kind: 'buffer' | 'texture';
  source: 'imported' | 'transient';
}

export interface RenderResourceUse {
  resource: string;
  /** attachmentLoad reads an existing texture attachment and stores its replacement. */
  access: 'read' | 'write' | 'readWrite' | 'attachmentLoad';
}

export interface RenderPass<Context> {
  name: string;
  uses: readonly RenderResourceUse[];
  /** Additional ordering constraints, named by pass. */
  after?: readonly string[];
  execute(context: Context): void;
}

/** Inclusive indices into the compiled pass order. Unused resources have a null lifetime. */
export interface ResourceLifetime {
  first: number;
  last: number;
}

export interface CompiledRenderGraph<Context> {
  readonly order: readonly string[];
  readonly lifetimes: ReadonlyMap<string, ResourceLifetime | null>;
  execute(context: Context): void;
}

/**
 * Compile explicit pass uses into a stable schedule. Conflicting accesses retain
 * declaration order; independent passes may move to satisfy `after` constraints.
 * Transient resources must be written before their first read in declaration order.
 */
export function compileRenderGraph<Context>(
  resources: readonly RenderResource[], inputPasses: readonly RenderPass<Context>[],
): CompiledRenderGraph<Context> {
  // Validation and execution must refer to the same immutable declarations.
  // Callers may reuse and edit their pass definitions for a later compilation.
  const passes: RenderPass<Context>[] = inputPasses.map(pass => ({
    name: pass.name,
    uses: pass.uses.map(use => ({ resource: use.resource, access: use.access })),
    after: [...(pass.after ?? [])],
    execute: pass.execute,
  }));
  const resourceByName = new Map<string, RenderResource>();
  for (const resource of resources) {
    if (!resource.name) throw new Error('Render graph resource name must not be empty.');
    if (resourceByName.has(resource.name)) throw new Error(`Duplicate render graph resource "${resource.name}".`);
    resourceByName.set(resource.name, resource);
  }

  const passByName = new Map<string, number>();
  passes.forEach((pass, index) => {
    if (!pass.name) throw new Error('Render graph pass name must not be empty.');
    if (passByName.has(pass.name)) throw new Error(`Duplicate render graph pass "${pass.name}".`);
    passByName.set(pass.name, index);
  });

  const edges = passes.map(() => new Set<number>());
  const indegree = passes.map(() => 0);
  const addEdge = (from: number, to: number): void => {
    if (from === to) throw new Error(`Render graph pass "${passes[from].name}" depends on itself.`);
    if (!edges[from].has(to)) {
      edges[from].add(to);
      indegree[to]++;
    }
  };

  passes.forEach((pass, index) => {
    for (const dependency of pass.after ?? []) {
      const predecessor = passByName.get(dependency);
      if (predecessor === undefined) throw new Error(`Render graph pass "${pass.name}" depends on missing pass "${dependency}".`);
      addEdge(predecessor, index);
    }
  });

  // A writer depends on every reader since the previous write, preserving WAR
  // hazards as well as the more obvious write-to-read and write-to-write edges.
  const history = new Map<string, { writer: number | null; readers: number[]; initialized: boolean }>();
  for (const resource of resources) {
    history.set(resource.name, { writer: null, readers: [], initialized: resource.source === 'imported' });
  }
  passes.forEach((pass, index) => {
    const used = new Set<string>();
    for (const use of pass.uses) {
      const resource = resourceByName.get(use.resource);
      if (!resource) throw new Error(`Render graph pass "${pass.name}" uses missing resource "${use.resource}".`);
      if (used.has(use.resource)) {
        throw new Error(`Render graph pass "${pass.name}" uses "${use.resource}" more than once; declare one explicit access.`);
      }
      used.add(use.resource);
      if (use.access === 'readWrite' && resource.kind !== 'buffer') {
        throw new Error(`Render graph pass "${pass.name}" cannot readWrite texture "${use.resource}" in one pass.`);
      }
      if (use.access === 'attachmentLoad' && resource.kind !== 'texture') {
        throw new Error(`Render graph pass "${pass.name}" cannot attachmentLoad buffer "${use.resource}".`);
      }
      const state = history.get(use.resource)!;
      if (use.access !== 'write' && !state.initialized) {
        throw new Error(`Render graph pass "${pass.name}" reads transient resource "${use.resource}" before its first write.`);
      }
      if (state.writer !== null) addEdge(state.writer, index);
      if (use.access === 'read') {
        state.readers.push(index);
      } else {
        for (const reader of state.readers) addEdge(reader, index);
        state.readers = [];
        state.writer = index;
        state.initialized = true;
      }
    }
  });

  const orderedIndices: number[] = [];
  const scheduled = passes.map(() => false);
  while (orderedIndices.length < passes.length) {
    // Scan in declaration order so ties are deterministic without a heap.
    const next = indegree.findIndex((degree, index) => degree === 0 && !scheduled[index]);
    if (next < 0) {
      const remaining = passes.filter((_, index) => !scheduled[index]).map(pass => pass.name);
      throw new Error(`Render graph dependency cycle involving: ${remaining.join(', ')}.`);
    }
    scheduled[next] = true;
    orderedIndices.push(next);
    for (const dependent of edges[next]) indegree[dependent]--;
  }

  const lifetimes = new Map<string, ResourceLifetime | null>();
  for (const resource of resources) lifetimes.set(resource.name, null);
  orderedIndices.forEach((passIndex, compiledIndex) => {
    for (const use of passes[passIndex].uses) {
      const lifetime = lifetimes.get(use.resource);
      if (lifetime) lifetime.last = compiledIndex;
      else lifetimes.set(use.resource, { first: compiledIndex, last: compiledIndex });
    }
  });

  const order = orderedIndices.map(index => passes[index].name);
  return {
    order,
    lifetimes,
    execute(context: Context): void {
      for (const index of orderedIndices) passes[index].execute(context);
    },
  };
}
