/** Derive bounded GI change regions from renderer scene snapshots. Engine-owned
 * types only; no GPU data is read. Committed inputs change by object identity
 * (contracts.ts), mutable inputs are compared by value. */
import type { InstanceState } from './contracts';
import type { Bounds3 } from './deformation';
import type { GiChangeReason } from './gi-probe-schedule';

export interface GiObservedBatch {
  /** Stable for the renderer lifetime (one per primitive batch). */
  readonly key: object;
  readonly instances: readonly InstanceState[];
  readonly committed: boolean;
  /** Conservative current world bounds per active entity (shared deformation). */
  readonly bounds: ReadonlyMap<number, Bounds3>;
  /** Published material identity and binding identity (texture publication swaps it). */
  readonly material: object;
  readonly materialBinding: object;
  /** Monotonic per-batch count of material parameter edits. */
  readonly materialVersion: number;
  /** Query topology/representation version. */
  readonly topologyVersion: number;
}

export interface GiSceneChange {
  readonly region: Bounds3;
  readonly reason: GiChangeReason;
}

interface EntitySnapshot {
  state: InstanceState;
  generation: number;
  historyVersion: number;
  bounds: Bounds3;
  values?: Float32Array;
}
interface BatchSnapshot {
  material: object;
  materialBinding: object;
  materialVersion: number;
  topologyVersion: number;
  entities: Map<number, EntitySnapshot>;
}

function copyBounds(bounds: Bounds3): Bounds3 {
  return { min: [bounds.min[0], bounds.min[1], bounds.min[2]], max: [bounds.max[0], bounds.max[1], bounds.max[2]] };
}
function union(a: Bounds3 | undefined, b: Bounds3 | undefined): Bounds3 | undefined {
  if (!a) return b && copyBounds(b);
  if (!b) return copyBounds(a);
  return { min: [0, 1, 2].map(axis => Math.min(a.min[axis], b.min[axis])) as [number, number, number],
    max: [0, 1, 2].map(axis => Math.max(a.max[axis], b.max[axis])) as [number, number, number] };
}
function overlaps(a: Bounds3, b: Bounds3): boolean {
  for (let axis = 0; axis < 3; axis++) if (a.min[axis] > b.max[axis] || b.min[axis] > a.max[axis]) return false;
  return true;
}
function sameBounds(a: Bounds3, b: Bounds3): boolean {
  for (let axis = 0; axis < 3; axis++) {
    if (Math.fround(a.min[axis]) !== Math.fround(b.min[axis]) || Math.fround(a.max[axis]) !== Math.fround(b.max[axis])) return false;
  }
  return true;
}
/** Every rendered input of one instance, so in-place edits of mutable inputs are seen. */
function instanceValues(state: InstanceState): Float32Array {
  const values = new Float32Array(16 + 4 + 2 + state.deformScalars.length + state.bones.length * 16);
  values.set(state.model, 0); values.set(state.color, 16);
  values[20] = state.metalness; values[21] = state.roughness;
  values.set(state.deformScalars, 22);
  state.bones.forEach((bone, index) => values.set(bone, 22 + state.deformScalars.length + index * 16));
  return values;
}
function sameValues(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i] && !(Number.isNaN(a[i]) && Number.isNaN(b[i]))) return false;
  return true;
}

/** Observes once per rendered frame after instance bounds were computed. */
export class GiSceneObserver {
  private readonly batches = new Map<object, BatchSnapshot>();
  private observed = 0;

  get observations(): number { return this.observed; }

  /** Returns the changes since the previous call. The first observation of a
   * batch is its baseline (covered by the initial full refresh). */
  observe(batches: readonly GiObservedBatch[]): GiSceneChange[] {
    this.observed++;
    const changes: GiSceneChange[] = [];
    for (const batch of batches) {
      const previous = this.batches.get(batch.key);
      const entities = new Map<number, EntitySnapshot>();
      let currentUnion: Bounds3 | undefined;
      for (const state of batch.instances) {
        const bounds = batch.bounds.get(state.entityId);
        if (!bounds) throw new Error('DIFFUSE_GI_OBSERVED_BOUNDS_MISSING');
        currentUnion = union(currentUnion, bounds);
        const old = previous?.entities.get(state.entityId);
        const generation = state.entityGeneration ?? 0;
        // Committed inputs are immutable objects: an identical object is unchanged.
        if (batch.committed && old && old.state === state && old.generation === generation
          && old.historyVersion === state.historyVersion && !old.values) { entities.set(state.entityId, old); continue; }
        const snapshot: EntitySnapshot = { state, generation, historyVersion: state.historyVersion,
          bounds: copyBounds(bounds), values: batch.committed ? undefined : instanceValues(state) };
        entities.set(state.entityId, snapshot);
        if (!previous) continue;
        if (!old) { changes.push({ region: copyBounds(bounds), reason: 'instance' }); continue; }
        const changed = old.generation !== generation || old.historyVersion !== state.historyVersion
          || (batch.committed ? old.state !== state || !!old.values : !old.values || !sameValues(old.values, snapshot.values!))
          || !sameBounds(old.bounds, bounds);
        // Only the old and new placements affect the next field. Disjoint ones
        // (a teleport) stay separate so the space between is not marked.
        if (changed && overlaps(old.bounds, bounds)) changes.push({ region: union(old.bounds, bounds)!, reason: 'transform' });
        else if (changed) changes.push({ region: copyBounds(old.bounds), reason: 'transform' },
          { region: copyBounds(bounds), reason: 'transform' });
      }
      if (previous) {
        let previousUnion: Bounds3 | undefined;
        for (const [id, old] of previous.entities) {
          previousUnion = union(previousUnion, old.bounds);
          if (!entities.has(id)) changes.push({ region: copyBounds(old.bounds), reason: 'instance' });
        }
        const region = union(previousUnion, currentUnion);
        if (region && (previous.material !== batch.material || previous.materialBinding !== batch.materialBinding
          || previous.materialVersion !== batch.materialVersion)) changes.push({ region, reason: 'material' });
        if (region && previous.topologyVersion !== batch.topologyVersion) changes.push({ region, reason: 'topology' });
      }
      this.batches.set(batch.key, { material: batch.material, materialBinding: batch.materialBinding,
        materialVersion: batch.materialVersion, topologyVersion: batch.topologyVersion, entities });
    }
    return changes;
  }
}
