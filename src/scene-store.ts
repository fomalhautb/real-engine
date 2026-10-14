/** Stable runtime identity. A recycled index always receives a new generation. */
export interface EntityHandle {
  readonly index: number;
  readonly generation: number;
}

export const changeKinds = [
  'transform',
  'deformation',
  'topology',
  'materialParameters',
  'materialProgram',
  'coverage',
  'emissive',
  'resourcePages',
  'teleport',
] as const;

export type ChangeKind = typeof changeKinds[number];
export type SceneVersions = Readonly<Record<ChangeKind, number>>;

export interface SceneSnapshotEntry<T> {
  readonly handle: EntityHandle;
  /** A defensive copy. Changes to it cannot mutate this snapshot or the store. */
  readonly value: T;
  readonly versions: SceneVersions;
  /** Incremented whenever temporal history must be rejected. */
  readonly historyVersion: number;
}

export interface SceneSnapshot<T> {
  readonly revision: number;
  readonly entities: readonly SceneSnapshotEntry<T>[];
}

interface ChangeBase {
  readonly handle: EntityHandle;
  readonly changes: readonly ChangeKind[];
  readonly versions: SceneVersions;
  readonly historyVersion: number;
  readonly historyInvalidated: boolean;
}

export type SceneChange<T> =
  | (ChangeBase & { readonly kind: 'create' | 'update'; readonly value: T })
  | (ChangeBase & { readonly kind: 'remove' });

interface Slot<T> {
  generation: number;
  value?: T;
  occupied: boolean;
  versions: SceneVersions;
  historyVersion: number;
}

interface PendingChange<T> {
  kind: 'create' | 'update' | 'remove';
  handle: EntityHandle;
  value?: T;
  changes: Set<ChangeKind>;
  versions: SceneVersions;
  historyVersion: number;
  historyInvalidated: boolean;
}

function initialVersions(): SceneVersions {
  return Object.freeze({
    transform: 0,
    deformation: 0,
    topology: 0,
    materialParameters: 0,
    materialProgram: 0,
    coverage: 0,
    emissive: 0,
    resourcePages: 0,
    teleport: 0,
  });
}

function identityKey(handle: EntityHandle): string {
  return `${handle.index}:${handle.generation}`;
}

function copyHandle(handle: EntityHandle): EntityHandle {
  return Object.freeze({ index: handle.index, generation: handle.generation });
}

function copyVersions(versions: SceneVersions): SceneVersions {
  return Object.freeze({ ...versions });
}

function copyChanges(changes: ReadonlySet<ChangeKind>): readonly ChangeKind[] {
  return Object.freeze(changeKinds.filter((kind) => changes.has(kind)));
}

export function sameEntityHandle(a: EntityHandle, b: EntityHandle): boolean {
  return a.index === b.index && a.generation === b.generation;
}

/**
 * Category-independent scene state with explicit, coalesced incremental changes.
 * Values must be supported by structuredClone. Inputs are copied on commit and
 * read access returns a copy, so callers cannot silently edit committed state.
 */
export class SceneStore<T> {
  private readonly slots: Slot<T>[] = [];
  private readonly freeIndices: number[] = [];
  private readonly pending: PendingChange<T>[] = [];
  private readonly pendingByIdentity = new Map<string, PendingChange<T>>();
  private currentRevision = 0;

  get revision(): number { return this.currentRevision; }

  get size(): number { return this.slots.length - this.freeIndices.length; }

  create(value: T): EntityHandle {
    const committed = structuredClone(value);
    const index = this.freeIndices.pop() ?? this.slots.length;
    const slot = this.slots[index] ?? {
      generation: 0,
      occupied: false,
      versions: initialVersions(),
      historyVersion: 0,
    };
    const handle = copyHandle({ index, generation: slot.generation });
    slot.value = committed;
    slot.occupied = true;
    slot.versions = initialVersions();
    slot.historyVersion = 1;
    this.slots[index] = slot;
    this.record({
      kind: 'create', handle, value: committed, changes: new Set(),
      versions: slot.versions, historyVersion: slot.historyVersion,
      historyInvalidated: true,
    });
    this.currentRevision++;
    return handle;
  }

  /** Returns a copy so callers cannot bypass version tracking. */
  get(handle: EntityHandle): T {
    return structuredClone(this.requireSlot(handle).value as T);
  }

  /**
   * Commits either a replacement value or a mutation of a detached draft.
   * `changes` declares all affected dependencies; an empty list is rejected.
   */
  update(handle: EntityHandle, changes: readonly ChangeKind[], edit: T | ((draft: T) => T | void)): void {
    const slot = this.requireSlot(handle);
    const unique = new Set(changes);
    if (unique.size === 0) throw new RangeError('Scene update requires at least one change kind');
    for (const kind of unique) {
      if (!(changeKinds as readonly string[]).includes(kind)) {
        throw new RangeError(`Unknown scene change kind: ${String(kind)}`);
      }
    }

    let committed: T;
    if (typeof edit === 'function') {
      const draft = structuredClone(slot.value as T);
      const result = (edit as (draft: T) => T | void)(draft);
      // The callback can retain its draft or insert caller-owned nested objects.
      // Publish a detached value so neither can mutate committed state later.
      committed = structuredClone(result === undefined ? draft : result);
    } else {
      committed = structuredClone(edit);
    }
    const versions = { ...slot.versions };
    for (const kind of unique) versions[kind]++;
    slot.value = committed;
    slot.versions = Object.freeze(versions);
    if (unique.has('topology') || unique.has('teleport')) slot.historyVersion++;

    const key = identityKey(handle);
    const prior = this.pendingByIdentity.get(key);
    if (prior && prior.kind !== 'remove') {
      prior.value = committed;
      for (const kind of unique) prior.changes.add(kind);
      prior.versions = slot.versions;
      prior.historyVersion = slot.historyVersion;
      prior.historyInvalidated ||= unique.has('topology') || unique.has('teleport');
    } else {
      this.record({
        kind: 'update', handle: copyHandle(handle), value: committed, changes: unique,
        versions: slot.versions, historyVersion: slot.historyVersion,
        historyInvalidated: unique.has('topology') || unique.has('teleport'),
      });
    }
    this.currentRevision++;
  }

  remove(handle: EntityHandle): void {
    const slot = this.requireSlot(handle);
    const key = identityKey(handle);
    const prior = this.pendingByIdentity.get(key);
    // A removed entity needs no intermediate update, but an unseen create
    // remains observable so the log preserves each committed identity event.
    if (prior?.kind === 'update') {
      this.pending.splice(this.pending.indexOf(prior), 1);
    }
    this.pendingByIdentity.delete(key);
    const removed: PendingChange<T> = {
      kind: 'remove', handle: copyHandle(handle), changes: new Set(),
      versions: slot.versions, historyVersion: slot.historyVersion + 1,
      historyInvalidated: true,
    };
    this.record(removed);
    slot.value = undefined;
    slot.occupied = false;
    slot.generation++;
    slot.historyVersion++;
    this.freeIndices.push(handle.index);
    this.currentRevision++;
  }

  /** Captures a frame boundary; later commits cannot alter this snapshot. */
  snapshot(): SceneSnapshot<T> {
    const entities: SceneSnapshotEntry<T>[] = [];
    for (let index = 0; index < this.slots.length; index++) {
      const slot = this.slots[index];
      if (!slot.occupied) continue;
      const frameValue = structuredClone(slot.value as T);
      const handle = copyHandle({ index, generation: slot.generation });
      const versions = copyVersions(slot.versions);
      const historyVersion = slot.historyVersion;
      entities.push(Object.freeze({
        handle,
        get value(): T { return structuredClone(frameValue); },
        versions,
        historyVersion,
      }));
    }
    return Object.freeze({ revision: this.currentRevision, entities: Object.freeze(entities) });
  }

  /** Returns committed changes in order and clears the incremental log. */
  drainChanges(): readonly SceneChange<T>[] {
    const changes: SceneChange<T>[] = this.pending.map((item) => {
      const base = {
        handle: copyHandle(item.handle),
        changes: copyChanges(item.changes),
        versions: copyVersions(item.versions),
        historyVersion: item.historyVersion,
        historyInvalidated: item.historyInvalidated,
      };
      if (item.kind === 'remove') return Object.freeze({ ...base, kind: 'remove' as const });
      // create/update already detached this value from the caller. Later commits
      // replace it rather than mutating it, so only the public read needs a copy.
      const committed = item.value as T;
      return Object.freeze({
        ...base, kind: item.kind,
        get value(): T { return structuredClone(committed); },
      });
    });
    this.pending.length = 0;
    this.pendingByIdentity.clear();
    return Object.freeze(changes);
  }

  private requireSlot(handle: EntityHandle): Slot<T> {
    if (!Number.isSafeInteger(handle.index) || !Number.isSafeInteger(handle.generation)) {
      throw new RangeError('Invalid entity handle');
    }
    const slot = this.slots[handle.index];
    if (!slot?.occupied || slot.generation !== handle.generation) {
      throw new RangeError(`Stale entity handle ${handle.index}:${handle.generation}`);
    }
    return slot;
  }

  private record(change: PendingChange<T>): void {
    this.pending.push(change);
    this.pendingByIdentity.set(identityKey(change.handle), change);
  }
}
