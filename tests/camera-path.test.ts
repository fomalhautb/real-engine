import { describe, expect, it } from 'vitest';
import { cameraPathKeyframes, parseCameraPath, sampleCameraPath } from '../src/camera-path';

const sample = (time: number, x: number, forward: number[], key?: boolean) => ({ time, position: [x, 1, 0], forward, up: [0, 2, 0], ...(key ? { key } : {}) });
const doc = { format: 'real-engine-camera-path', name: 'walk', verticalFovRadians: 0.8,
  samples: [sample(0, 0, [1, 0, 0], true), sample(2, 2, [1, 0, 0]), sample(4, 4, [0, 0, -3], true), sample(6, 6, [-1, 0, 0])] };

describe('camera path', () => {
  it('validates documents and normalizes directions', () => {
    const path = parseCameraPath(doc);
    expect(path).toMatchObject({ name: 'walk', verticalFovRadians: 0.8, duration: 6 });
    expect(path.samples[2].forward).toEqual([0, 0, -1]);
    expect(path.samples[0].up).toEqual([0, 1, 0]);
    expect(() => parseCameraPath({ ...doc, format: 'x' })).toThrow('CAMERA_PATH_FORMAT');
    expect(() => parseCameraPath({ ...doc, samples: [doc.samples[0]] })).toThrow('CAMERA_PATH_SAMPLE_COUNT');
    expect(() => parseCameraPath({ ...doc, samples: [doc.samples[1], doc.samples[0]] })).toThrow('CAMERA_PATH_TIME_ORDER: 1');
    expect(() => parseCameraPath({ ...doc, samples: [doc.samples[0], { ...doc.samples[1], forward: [0, 0, 0] }] })).toThrow('CAMERA_PATH_DIRECTION');
    expect(() => parseCameraPath({ ...doc, verticalFovRadians: 4 })).toThrow('CAMERA_PATH_FOV');
  });

  it('passes through samples, interpolates between them and loops', () => {
    const path = parseCameraPath(doc);
    for (const [i, s] of path.samples.entries()) {
      const pose = sampleCameraPath(path, s.time, false);
      expect(pose.eye.map(v => Number(v.toFixed(9)))).toEqual(s.position);
      expect(pose.target).toEqual([s.position[0] + s.forward[0], s.position[1] + s.forward[1], s.position[2] + s.forward[2]]);
      expect(i).toBeGreaterThanOrEqual(0);
    }
    // Evenly spaced collinear samples: Catmull-Rom stays on the line at the linear position.
    expect(sampleCameraPath(path, 1, false).eye.map(v => Number(v.toFixed(9)))).toEqual([1, 1, 0]);
    const turn = sampleCameraPath(path, 3, false);
    expect(Math.hypot(turn.target[0] - turn.eye[0], turn.target[1] - turn.eye[1], turn.target[2] - turn.eye[2])).toBeCloseTo(1, 12);
    expect(sampleCameraPath(path, 7).eye).toEqual(sampleCameraPath(path, 1).eye);
    expect(sampleCameraPath(path, -5).eye).toEqual(sampleCameraPath(path, 1).eye);
    expect(sampleCameraPath(path, 99, false).eye.map(v => Number(v.toFixed(9)))).toEqual([6, 1, 0]);
    // Opposite directions (samples 1 -> 3 never adjacent here) still blend to a unit vector.
    const opposite = parseCameraPath({ ...doc, samples: [sample(0, 0, [1, 0, 0]), sample(1, 1, [-1, 0, 0])] });
    const mid = sampleCameraPath(opposite, 0.5, false);
    expect(Math.hypot(mid.target[0] - mid.eye[0], mid.target[1] - mid.eye[1], mid.target[2] - mid.eye[2])).toBeCloseTo(1, 12);
  });

  it('offers marked samples as keyframes, else evenly spaced ones', () => {
    expect(cameraPathKeyframes(parseCameraPath(doc))).toEqual([0, 2]);
    const unmarked = parseCameraPath({ ...doc, samples: Array.from({ length: 11 }, (_, i) => sample(i, i, [1, 0, 0])) });
    expect(cameraPathKeyframes(unmarked, 3)).toEqual([0, 5, 10]);
    expect(() => cameraPathKeyframes(unmarked, 0)).toThrow('CAMERA_PATH_KEYFRAME_COUNT');
  });
});
