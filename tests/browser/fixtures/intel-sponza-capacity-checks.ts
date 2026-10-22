import type { GeometryStreamDirectory } from '../../../src/geometry-stream-types';

export type CapacityCamera = { eye: [number, number, number]; target: [number, number, number] };

/** Interpolate heading instead of look-at points: opposite views must not pass
 * through a zero-length eye/target vector halfway through the walk. */
export function interpolateCapacityCamera(a: CapacityCamera, b: CapacityCamera, t: number): CapacityCamera {
  if (!Number.isFinite(t) || t < 0 || t > 1) throw Error('SPONZA_CAPACITY_CAMERA_TIME');
  const da = a.target.map((n, i) => n - a.eye[i]), db = b.target.map((n, i) => n - b.eye[i]);
  const start = Math.atan2(da[2], da[0]), end = Math.atan2(db[2], db[0]);
  const delta = ((end - start + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
  const yaw = start + delta * t, horizontal = Math.hypot(da[0], da[2]) * (1 - t) + Math.hypot(db[0], db[2]) * t;
  const eye = a.eye.map((n, i) => n + (b.eye[i] - n) * t) as CapacityCamera['eye'];
  return { eye, target: [eye[0] + Math.cos(yaw) * horizontal,
    eye[1] + da[1] * (1 - t) + db[1] * t, eye[2] + Math.sin(yaw) * horizontal] };
}

export function assertCapacityLedger(state: { liveBytes: number; reservedBytes: number; accountedBytes: number;
  peakBytes: number; error: string | null; limits: { maxResidentBytes: number; maxPeakBytes: number } }) {
  const values = [state.liveBytes, state.reservedBytes, state.accountedBytes, state.peakBytes];
  if (state.error || values.some(n => !Number.isSafeInteger(n) || n < 0)
    || state.accountedBytes !== state.liveBytes + state.reservedBytes
    || state.liveBytes > state.limits.maxResidentBytes || state.accountedBytes > state.limits.maxPeakBytes
    || state.peakBytes > state.limits.maxPeakBytes || state.peakBytes < state.accountedBytes) {
    throw Error('SPONZA_CAPACITY_LEDGER: ' + JSON.stringify(state));
  }
}

/** Compare the complete GPU cut, including frustum-culled members, to the
 * resident CPU reference. Visible flags then independently explain draw indices. */
export function assertCapacityCut(directory: Pick<GeometryStreamDirectory, 'clusters' | 'clusterPages'>,
  resident: ReadonlySet<number>, expected: readonly (readonly number[])[],
  feedback: { indices: number; overflow: boolean; selectedClusters?: readonly { clusterId: number; instanceIndex: number; visible: boolean }[] }) {
  if (feedback.overflow || !feedback.selectedClusters) throw Error('SPONZA_CAPACITY_CUT_UNAVAILABLE');
  const actual = expected.map(() => new Set<number>());
  let visibleClusters = 0, triangles = 0;
  for (const item of feedback.selectedClusters) {
    const cluster = directory.clusters[item.clusterId], cut = actual[item.instanceIndex];
    if (!cluster || !cut || cut.has(item.clusterId) || !resident.has(directory.clusterPages[item.clusterId])) {
      throw Error('SPONZA_CAPACITY_CUT_MEMBER');
    }
    cut.add(item.clusterId);
    if (item.visible) { visibleClusters++; triangles += cluster.triangleCount; }
  }
  for (let instance = 0; instance < expected.length; instance++) {
    if (actual[instance].size !== expected[instance].length || expected[instance].some(id => !actual[instance].has(id))) {
      throw Error('SPONZA_CAPACITY_INCOMPLETE_CUT: instance=' + instance);
    }
  }
  if (triangles * 3 !== feedback.indices) throw Error('SPONZA_CAPACITY_DRAW_INDICES');
  return { completeClusters: feedback.selectedClusters.length, visibleClusters, triangles, indices: feedback.indices };
}

export function capacityExchange(before: readonly number[], after: readonly number[]) {
  if (before.length !== after.length) throw Error('SPONZA_CAPACITY_SLOT_LENGTH');
  const grown: number[] = [], shrunk: number[] = [];
  after.forEach((slots, index) => { if (slots > before[index]) grown.push(index); else if (slots < before[index]) shrunk.push(index); });
  return { grown, shrunk, exchanged: grown.length > 0 && shrunk.length > 0 };
}
