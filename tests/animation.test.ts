import { describe, expect, it } from 'vitest';
import { AnimationRig, type AnimationClip, type AnimationNode, type AnimationTrack } from '../src/animation';
import { transformPoint } from '../src/math';

const node = (parent: number | null = null, weights: number[] = []): AnimationNode => ({
  parent, translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], weights,
});
const track = (nodeIndex: number, path: AnimationTrack['path'], interpolation: AnimationTrack['interpolation'],
  times: number[], values: number[], components: number): AnimationTrack => ({
  node: nodeIndex, path, interpolation, times: new Float32Array(times), values: new Float32Array(values), components,
});
const clip = (tracks: AnimationTrack[], duration = 2): AnimationClip => ({ name: 'test', tracks, duration });
const close = (actual: readonly number[], expected: readonly number[]) => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 5);
};

describe('AnimationRig', () => {
  it('evaluates unsorted hierarchy from bind transforms and composes T * R * S', () => {
    const child: AnimationNode = { ...node(1), translation: [1, 0, 0] };
    const root: AnimationNode = { ...node(), translation: [3, 4, 5], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2], scale: [2, 3, 4] };
    const rig = new AnimationRig([child, root], []);
    const pose = rig.sample(0);
    close(transformPoint(pose.local[0], [0, 0, 0]), [1, 0, 0]);
    close(transformPoint(pose.world[0], [0, 0, 0]), [3, 6, 5]);
    close(transformPoint(pose.world[1], [1, 0, 0]), [3, 6, 5]);
  });

  it('samples STEP and LINEAR channels with clamped endpoints and clip looping', () => {
    const rig = new AnimationRig([node(null, [0, 1])], [clip([
      track(0, 'translation', 'LINEAR', [0.5, 2], [1, 0, 0, 4, 0, 0], 3),
      track(0, 'weights', 'STEP', [0.5, 2], [0.1, 0.9, 0.8, 0.2], 2),
    ])]);
    close(transformPoint(rig.sample(0).local[0], [0, 0, 0]), [1, 0, 0]);
    close(transformPoint(rig.sample(1.25).local[0], [0, 0, 0]), [2.5, 0, 0]);
    close(rig.sample(1.25).weights[0], [0.1, 0.9]);
    close(rig.sample(2).weights[0], [0.8, 0.2]);
    close(transformPoint(rig.sample(3.25, 0, true).local[0], [0, 0, 0]), [2.5, 0, 0]);
    close(transformPoint(rig.sample(2, 0, true).local[0], [0, 0, 0]), [1, 0, 0]);
  });

  it('uses shortest-path quaternion interpolation and normalizes the result', () => {
    const rig = new AnimationRig([node()], [clip([
      track(0, 'rotation', 'LINEAR', [0, 2], [0, 0, 0, 1, 0, 0, -1, 0], 4),
    ])]);
    close(transformPoint(rig.sample(1).local[0], [1, 0, 0]), [0, -1, 0]);
  });

  it('scales cubic tangents by segment duration and normalizes cubic quaternions', () => {
    const rig = new AnimationRig([node()], [clip([
      track(0, 'translation', 'CUBICSPLINE', [0, 2], [
        0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
      ], 3),
      track(0, 'rotation', 'CUBICSPLINE', [0, 2], [
        0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
      ], 4),
    ])]);
    close(transformPoint(rig.sample(1).local[0], [0, 0, 0]), [0.5, 0, 0]);
    close(transformPoint(rig.sample(1).local[0], [1, 0, 0]), [0.5, 1, 0]);
  });

  it('keeps missing channels at bind values across separate clips', () => {
    const base: AnimationNode = { ...node(null, [0.25, 0.75]), translation: [5, 0, 0] };
    const rig = new AnimationRig([base], [
      clip([track(0, 'translation', 'LINEAR', [0, 2], [0, 0, 0, 2, 0, 0], 3)]),
      clip([track(0, 'weights', 'STEP', [0], [0.8, 0.2], 2)]),
    ]);
    close(transformPoint(rig.sample(1, 0).local[0], [0, 0, 0]), [1, 0, 0]);
    close(rig.sample(1, 0).weights[0], [0.25, 0.75]);
    close(transformPoint(rig.sample(1, 1).local[0], [0, 0, 0]), [5, 0, 0]);
    close(rig.sample(1, 1).weights[0], [0.8, 0.2]);
  });

  it('clones constructor input and shields validated keys from exposed clips', () => {
    const source = node(null, [0.25]);
    const keys = track(0, 'weights', 'STEP', [0], [0.75], 1);
    const rig = new AnimationRig([source], [clip([keys])]);
    (source.weights as number[])[0] = 99;
    keys.values[0] = 99;
    rig.clips[0].tracks[0].values[0] = 99;
    close(rig.sample(0).weights[0], [0.75]);
    close(new AnimationRig([source], []).sample(0).weights[0], [99]);
  });

  it('allows affine matrix nodes with weight animation', () => {
    const affine = new Float32Array([2, 0, 0, 0, 0, -3, 0, 0, 0, 0, 4, 0, 1, 2, 3, 1]);
    const rig = new AnimationRig([{ ...node(null, [0]), matrix: affine }], [clip([
      track(0, 'weights', 'STEP', [0], [1], 1),
    ])]);
    affine[12] = 99;
    close(transformPoint(rig.sample(0).world[0], [1, 1, 1]), [3, -1, 7]);
    close(rig.sample(0).weights[0], [1]);
  });

  it('rejects cycles, malformed keys, duplicate channels, and matrix TRS tracks', () => {
    expect(() => new AnimationRig([node(1), node(0)], [])).toThrow(/cycle/);
    expect(() => new AnimationRig([node()], [clip([track(0, 'translation', 'STEP', [1, 1], [0, 0, 0, 1, 0, 0], 3)])])).toThrow(/strictly increasing/);
    expect(() => new AnimationRig([node()], [clip([track(0, 'translation', 'STEP', [0], [1, 2], 3)])])).toThrow(/value count/);
    expect(() => new AnimationRig([node()], [clip([track(0, 'translation', 'STEP', [0], [1, 2, 3], 3), track(0, 'translation', 'STEP', [1], [1, 2, 3], 3)])])).toThrow(/duplicates/);
    expect(() => new AnimationRig([{ ...node(), matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) }], [clip([track(0, 'scale', 'STEP', [0], [1, 1, 1], 3)])])).toThrow(/matrix node/);
    expect(() => new AnimationRig([node()], [clip([track(0, 'rotation', 'LINEAR', [0], [0, 0, 0, 0], 4)])])).toThrow(/quaternion/);
  });
});
