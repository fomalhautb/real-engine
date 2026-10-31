import { describe, expect, it } from 'vitest';
import type { InstanceState } from '../src/contracts';
import type { Bounds3 } from '../src/deformation';
import { GiSceneObserver, type GiObservedBatch } from '../src/gi-scene-changes';
import { normalMatrix, translation } from '../src/math';

function instance(entityId: number, x: number, extra: Partial<InstanceState> = {}): InstanceState {
  const model = translation(x, 0, 0);
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [], color: [1, 1, 1, 1],
    metalness: 0, roughness: 1, historyVersion: 0, ...extra };
}
function boundsOf(states: readonly InstanceState[]): Map<number, Bounds3> {
  return new Map(states.map(state => [state.entityId, {
    min: [state.model[12] - 0.5, -0.5, -0.5], max: [state.model[12] + 0.5, 0.5, 0.5] }]));
}
const key = {}, material = {}, binding = {};
function batch(instances: InstanceState[], overrides: Partial<GiObservedBatch> = {}): GiObservedBatch {
  return { key, instances, committed: true, bounds: boundsOf(instances), material, materialBinding: binding,
    materialVersion: 0, topologyVersion: 1, ...overrides };
}

describe('GI scene change sets', () => {
  it('treats the first observation as a baseline and identical committed objects as unchanged', () => {
    const observer = new GiSceneObserver();
    const states = [instance(1, 0), instance(2, 4)];
    expect(observer.observe([batch(states)])).toEqual([]);
    expect(observer.observe([batch(states)])).toEqual([]);
    expect(observer.observe([batch([...states])])).toEqual([]);
  });

  it('reports moved committed instances by their old and new bounds', () => {
    const observer = new GiSceneObserver();
    const states = [instance(1, 0), instance(2, 4)];
    observer.observe([batch(states)]);
    const moved = [states[0], instance(2, 5)];
    // Touching placements merge into one region.
    expect(observer.observe([batch(moved)])).toEqual([
      { region: { min: [3.5, -0.5, -0.5], max: [5.5, 0.5, 0.5] }, reason: 'transform' }]);
    // A replaced object with identical values is still a committed change.
    expect(observer.observe([batch([states[0], { ...moved[1] }])])).toHaveLength(1);
    // A teleport marks both placements, not the space between them.
    expect(observer.observe([batch([states[0], instance(2, 20)])])).toEqual([
      { region: { min: [4.5, -0.5, -0.5], max: [5.5, 0.5, 0.5] }, reason: 'transform' },
      { region: { min: [19.5, -0.5, -0.5], max: [20.5, 0.5, 0.5] }, reason: 'transform' }]);
  });

  it('reports additions, removals, generations and history versions', () => {
    const observer = new GiSceneObserver();
    observer.observe([batch([instance(1, 0)])]);
    expect(observer.observe([batch([instance(1, 0), instance(7, 3)])]).map(change => change.reason))
      .toEqual(['transform', 'instance']);
    const kept = instance(1, 0);
    observer.observe([batch([kept])]);
    expect(observer.observe([batch([{ ...kept, entityGeneration: 1 }])]).map(change => change.reason)).toEqual(['transform']);
    const regenerated = instance(1, 0, { entityGeneration: 1, historyVersion: 3 });
    expect(observer.observe([batch([regenerated])])).toHaveLength(1);
    expect(observer.observe([batch([])])).toEqual([
      { region: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] }, reason: 'instance' }]);
  });

  it('attributes material identity, binding and parameter edits to the batch bounds', () => {
    const observer = new GiSceneObserver();
    const states = [instance(1, 0), instance(2, 4)];
    observer.observe([batch(states)]);
    const region = { min: [-0.5, -0.5, -0.5], max: [4.5, 0.5, 0.5] };
    expect(observer.observe([batch(states, { materialVersion: 1 })])).toEqual([{ region, reason: 'material' }]);
    expect(observer.observe([batch(states, { materialVersion: 1, materialBinding: {} })])).toEqual([{ region, reason: 'material' }]);
    expect(observer.observe([batch(states, { materialVersion: 1, materialBinding: {}, material: {} })])).toHaveLength(1);
    expect(observer.observe([batch(states, { materialVersion: 1, materialBinding: {}, material: {}, topologyVersion: 2 })])
      .map(change => change.reason)).toEqual(['material', 'topology']);
  });

  it('compares mutable inputs by value so in-place edits are not missed', () => {
    const observer = new GiSceneObserver();
    const state = instance(1, 0);
    observer.observe([batch([state], { committed: false })]);
    expect(observer.observe([batch([state], { committed: false })])).toEqual([]);
    state.color = [1, 0, 0, 1];
    expect(observer.observe([batch([state], { committed: false })]).map(change => change.reason)).toEqual(['transform']);
    state.deformScalars.push(0.5);
    expect(observer.observe([batch([state], { committed: false })])).toHaveLength(1);
    expect(observer.observe([batch([state], { committed: false })])).toEqual([]);
  });

  it('requires bounds for every observed entity', () => {
    const observer = new GiSceneObserver();
    expect(() => observer.observe([{ ...batch([instance(1, 0)]), bounds: new Map() }])).toThrow('DIFFUSE_GI_OBSERVED_BOUNDS_MISSING');
  });
});
