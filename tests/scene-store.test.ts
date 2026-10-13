import { describe, expect, it } from 'vitest';
import { SceneStore, sameEntityHandle } from '../src/scene-store';

interface State {
  transform: Float32Array;
  material: { color: number[] };
}

function state(x = 0): State {
  return {
    transform: new Float32Array([x, 1, 2]),
    material: { color: [0.2, 0.4, 0.6] },
  };
}

describe('SceneStore', () => {
  it('rejects stale generations after slot reuse', () => {
    const store = new SceneStore<State>();
    const old = store.create(state());
    store.drainChanges();
    store.remove(old);
    const next = store.create(state(7));

    expect(next.index).toBe(old.index);
    expect(next.generation).toBe(old.generation + 1);
    expect(sameEntityHandle(old, next)).toBe(false);
    expect(() => store.get(old)).toThrow(/Stale entity handle/);
    expect(() => store.update(old, ['transform'], state())).toThrow(/Stale entity handle/);
    expect(() => store.remove(old)).toThrow(/Stale entity handle/);
    expect(store.get(next).transform[0]).toBe(7);
    expect(store.size).toBe(1);
  });

  it('copies input, reads, snapshots, and change values including typed arrays', () => {
    const store = new SceneStore<State>();
    const input = state(3);
    const handle = store.create(input);
    input.transform[0] = 99;
    input.material.color[0] = 99;
    const read = store.get(handle);
    read.transform[0] = 88;
    read.material.color[0] = 88;
    expect(store.get(handle).transform[0]).toBe(3);
    expect(store.get(handle).material.color[0]).toBe(0.2);

    const frame = store.snapshot();
    const entry = frame.entities[0];
    const frameCopy = entry.value;
    frameCopy.transform[0] = 77;
    frameCopy.material.color[0] = 77;
    expect(entry.value.transform[0]).toBe(3);
    expect(entry.value.material.color[0]).toBe(0.2);
    expect(Object.isFrozen(frame.entities)).toBe(true);
    expect(Object.isFrozen(entry.handle)).toBe(true);
    expect(Object.isFrozen(entry.versions)).toBe(true);

    const create = store.drainChanges()[0];
    expect(create.kind).toBe('create');
    if (create.kind !== 'create') throw new Error('Expected create');
    create.value.transform[0] = 66;
    expect(create.value.transform[0]).toBe(3);

    store.update(handle, ['transform'], (draft) => { draft.transform[0] = 4; });
    expect(store.get(handle).transform[0]).toBe(4);
    expect(entry.value.transform[0]).toBe(3);
    expect(frame.revision).toBe(1);
    expect(entry.versions.transform).toBe(0);
    expect(entry.historyVersion).toBe(1);
  });

  it('coalesces updates by live identity and records independent revisions', () => {
    const store = new SceneStore<State>();
    const handle = store.create(state());
    const created = store.drainChanges();
    expect(created).toHaveLength(1);
    store.update(handle, ['transform'], (draft) => { draft.transform[0] = 1; });
    store.update(handle, ['transform', 'materialParameters'], (draft) => {
      draft.transform[0] = 2;
      draft.material.color[0] = 0.9;
    });
    store.update(handle, ['emissive'], state(5));
    const changes = store.drainChanges();

    expect(changes).toHaveLength(1);
    const event = changes[0];
    expect(event.kind).toBe('update');
    expect(event.changes).toEqual(['transform', 'materialParameters', 'emissive']);
    expect(event.versions.transform).toBe(2);
    expect(event.versions.materialParameters).toBe(1);
    expect(event.versions.emissive).toBe(1);
    expect(event.versions.topology).toBe(0);
    expect(event.historyInvalidated).toBe(false);
    if (event.kind !== 'update') throw new Error('Expected update');
    expect(event.value.transform[0]).toBe(5);
    expect(store.drainChanges()).toEqual([]);
  });

  it('keeps remove and recycled create distinct while coalescing create updates', () => {
    const store = new SceneStore<State>();
    const old = store.create(state(1));
    store.drainChanges();
    store.update(old, ['topology'], state(2));
    store.remove(old);
    const next = store.create(state(3));
    store.update(next, ['materialProgram'], state(4));
    const events = store.drainChanges();

    expect(events.map((event) => event.kind)).toEqual(['remove', 'create']);
    expect(events[0].handle).toEqual(old);
    expect(events[1].handle).toEqual(next);
    expect(events[0].historyInvalidated).toBe(true);
    expect(events[0].historyVersion).toBe(3);
    expect(events[0].versions.topology).toBe(1);
    expect(events[1].changes).toEqual(['materialProgram']);
    expect(events[1].versions.materialProgram).toBe(1);
    expect(events[1].historyVersion).toBe(1);
    expect(events[1].historyInvalidated).toBe(true);
    if (events[1].kind !== 'create') throw new Error('Expected create');
    expect(events[1].value.transform[0]).toBe(4);
  });

  it('invalidates history only for topology, teleport, create, and remove', () => {
    const store = new SceneStore<State>();
    const handle = store.create(state());
    store.drainChanges();
    store.update(handle, ['deformation', 'coverage'], state(1));
    expect(store.drainChanges()[0].historyInvalidated).toBe(false);
    store.update(handle, ['topology'], state(2));
    expect(store.drainChanges()[0].historyVersion).toBe(2);
    store.update(handle, ['teleport', 'transform'], state(3));
    const event = store.drainChanges()[0];
    expect(event.historyInvalidated).toBe(true);
    expect(event.historyVersion).toBe(3);
    expect(event.versions.teleport).toBe(1);
    expect(store.snapshot().entities[0].historyVersion).toBe(3);
  });

  it('leaves committed state unchanged when editing or validation fails', () => {
    const store = new SceneStore<State>();
    const handle = store.create(state());
    store.drainChanges();
    expect(() => store.update(handle, [], state(5))).toThrow(RangeError);
    expect(() => store.update(handle, ['transform'], () => { throw new Error('failed edit'); })).toThrow('failed edit');
    expect(store.get(handle).transform[0]).toBe(0);
    expect(store.revision).toBe(1);
    expect(store.drainChanges()).toEqual([]);
  });
});
