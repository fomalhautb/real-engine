import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

const NAME = 'sparse-upload';
const COUNT = 40;
const STRIDE = { instances: 224, palettes: 128, scalars: 8 } as const;
type Write = { label: string; offset: number; bytes: number };
type Pose = { generation: number; historyVersion: number; point: [number, number, number] };

function triangle(): MeshData {
  const vertices = new ArrayBuffer(3 * 80), view = new DataView(vertices);
  const positions = [[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0]];
  for (let i = 0; i < 3; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, positions[i][axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setUint32(offset + 32, 0, true);
    view.setFloat32(offset + 48, 1, true);
  }
  return { vertices, indices: new Uint32Array([0, 1, 2]), vertexCount: 3,
    bounds: { min: [0, 0, 0], max: [0.3, 0.3, 0] } };
}

function state(entityId: number, x: number, boneY = 0, morph = 0, generation = 0, historyVersion = 0): InstanceState {
  return { entityId, entityGeneration: generation, model: translation(x, 0, 0), normalMatrix: identity(),
    bones: [translation(0, boneY, 0)], deformScalars: [morph],
    color: [0.8, 0.7, 0.6, 1], metalness: 0, roughness: 0.6, historyVersion };
}

function moved(source: InstanceState, delta: number): InstanceState {
  return state(source.entityId, source.model[12] + 0.17 * delta,
    source.bones[0][13] + 0.11 * delta, source.deformScalars[0] + 0.2 * delta,
    source.entityGeneration ?? 0, source.historyVersion);
}

/** Independent oracle for vertex zero: Morph X, then one translating bone, then model. */
function point(source: InstanceState): [number, number, number] {
  return [source.model[12] + 0.25 * source.deformScalars[0] + source.bones[0][12],
    source.model[13] + source.bones[0][13], source.model[14] + source.bones[0][14]];
}

function interceptWrites(renderer: Renderer, writes: Write[]): () => void {
  const queue = (renderer as unknown as { device: GPUDevice }).device.queue;
  const own = Object.getOwnPropertyDescriptor(queue, 'writeBuffer');
  const original = queue.writeBuffer.bind(queue);
  const wrapper = ((...args: Parameters<GPUQueue['writeBuffer']>) => {
    const [buffer, offset, data, dataOffset = 0, size] = args;
    const elementBytes = ArrayBuffer.isView(data) && 'BYTES_PER_ELEMENT' in data
      ? (data as ArrayBufferView & { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT : 1;
    writes.push({ label: buffer.label, offset,
      bytes: (size ?? (data.byteLength / elementBytes - dataOffset)) * elementBytes });
    original(...args);
    return undefined;
  }) as GPUQueue['writeBuffer'];
  Object.defineProperty(queue, 'writeBuffer', { configurable: true, value: wrapper });
  return () => {
    if (own) Object.defineProperty(queue, 'writeBuffer', own);
    else Reflect.deleteProperty(queue, 'writeBuffer');
  };
}

export async function runSparseUploadFixture() {
  const mesh = triangle();
  const morphPositions = new Float32Array(mesh.vertexCount * 3);
  for (let i = 0; i < mesh.vertexCount; i++) morphPositions[i * 3] = 0.25;
  const batch: SceneBatch = { name: NAME, mesh, capacity: COUNT, inputMode: 'committed',
    instances: Array.from({ length: COUNT }, (_, i) => state(1000 + i, (i - 20) * 0.2)),
    deformation: { jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' }] },
    morphTargets: [{ positions: morphPositions, normals: new Float32Array(mesh.vertexCount * 3) }] };
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  let restoreWrites: (() => void) | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch]);
    const writes: Write[] = [];
    restoreWrites = interceptWrites(renderer, writes);
    const previous = new Map<number, Pose>();
    const eye: [number, number, number] = [0, 0, 8], target: [number, number, number] = [0, 0, 0];
    const render = async (verify = true, resetHistory = false) => {
      if (resetHistory) previous.clear();
      const current = batch.instances.map(item => structuredClone(item));
      const expected = new Map<number, { current: [number, number, number]; previous: [number, number, number] }>();
      let historyMotion = 0;
      for (const item of current) {
        const prior = previous.get(item.entityId), now = point(item);
        const before = prior && prior.generation === (item.entityGeneration ?? 0) && prior.historyVersion === item.historyVersion
          ? prior.point : now;
        expected.set(item.entityId, { current: now, previous: before });
        historyMotion = Math.max(historyMotion, Math.hypot(...now.map((value, axis) => value - before[axis])));
      }
      writes.length = 0;
      renderer!.render(eye, target);
      const uploadedBytes = renderer!.stats.uploadedBytes;
      const deformationDispatches = renderer!.stats.deformationDispatches;
      const frameWrites = writes.filter(write => write.label.startsWith(NAME)).map(write => ({ ...write }));
      previous.clear();
      for (const item of current) previous.set(item.entityId, {
        generation: item.entityGeneration ?? 0, historyVersion: item.historyVersion, point: point(item),
      });
      let currentError = 0, previousError = 0, sampleCount = -1, referenceError = 0, boundsViolations = 0;
      if (verify) {
        const check = await renderer!.verifyDeformation({ allInstances: true, sampleVertices: [0] });
        sampleCount = check.samples.length;
        referenceError = Math.max(check.maxError, check.maxNormalError);
        boundsViolations = check.boundsViolations;
        const seen = new Set<number>();
        for (const sample of check.samples) {
          const pose = expected.get(sample.entityId);
          if (sample.batch !== NAME || sample.vertexIndex !== 0 || !pose || seen.has(sample.entityId)) {
            throw new Error('SPARSE_UPLOAD_SAMPLE_IDENTITY');
          }
          seen.add(sample.entityId);
          for (let axis = 0; axis < 3; axis++) {
            currentError = Math.max(currentError, Math.abs(sample.current[axis] - pose.current[axis]));
            previousError = Math.max(previousError, Math.abs(sample.previous[axis] - pose.previous[axis]));
          }
        }
        if (seen.size !== current.length || check.vertices !== current.length * mesh.vertexCount) {
          throw new Error('SPARSE_UPLOAD_SAMPLE_COVERAGE');
        }
      }
      return { uploadedBytes, deformationDispatches, writes: frameWrites, sampleCount,
        currentError, previousError, referenceError, boundsViolations, historyMotion,
        instances: current.length, width: renderer!.stats.width, height: renderer!.stats.height };
    };

    const first = await render();
    const still = await render();
    batch.instances[20] = moved(batch.instances[20], 1);
    const one = await render();
    const oneSettle = await render();
    const oneCached = await render(false);

    batch.instances[10] = moved(batch.instances[10], 2);
    batch.instances[11] = moved(batch.instances[11], 2);
    const adjacent = await render();
    await render(false); await render(false); // Let previous=current, then reach the cache path.
    batch.instances[5] = moved(batch.instances[5], 3);
    batch.instances[30] = moved(batch.instances[30], 3);
    const separated = await render();
    await render(false); await render(false);

    for (let i = 0; i < COUNT; i += 2) batch.instances[i] = moved(batch.instances[i], 4);
    const fragmented = await render();
    const fragmentedSettle = await render();
    await render(false);

    [batch.instances[0], batch.instances[39]] = [batch.instances[39], batch.instances[0]];
    const reordered = await render();
    await render(false); await render(false);
    const removed = batch.instances.splice(20, 1)[0];
    const deleted = await render();
    await render(false); await render(false);
    batch.instances[20] = state(removed.entityId, removed.model[12] + 0.4, 0.3, 0.5, 1);
    const newGeneration = await render();

    renderer.invalidateHistory();
    const cut = await render(true, true);
    canvas.style.width = '96px';
    const resized = await render(true, true);
    batch.instances = [];
    const empty = await render();
    batch.instances = [state(2000, 0.1, 0.2, 0.4)];
    const readded = await render();
    const live = batch.instances[0];
    batch.instances[0] = state(live.entityId, live.model[12] + 0.6,
      live.bones[0][13] + 0.3, live.deformScalars[0] + 0.2, (live.entityGeneration ?? 0) + 1);
    const liveGeneration = await render();

    batch.inputMode = 'mutable';
    batch.instances[0].bones[0][13] += 0.2;
    batch.instances[0].deformScalars[0] += 0.15;
    const mutable = await render();
    batch.inputMode = 'committed';
    const backToCommitted = await render();
    const committedCached = await render();

    const frameBeforeDuplicate = renderer.stats.frames;
    batch.instances = [batch.instances[0], { ...batch.instances[0] }];
    let duplicateError = '';
    try { renderer.render(eye, target); } catch (error) { duplicateError = String(error); }
    const frameAfterDuplicate = renderer.stats.frames;
    batch.instances = [batch.instances[0]];
    const afterDuplicate = await render();

    const errors = [...renderer.errors], frames = renderer.stats.frames;
    restoreWrites(); restoreWrites = undefined;
    renderer.dispose();
    return { first, still, one, oneSettle, oneCached, adjacent, separated, fragmented,
      fragmentedSettle, reordered, deleted, newGeneration, cut, resized, empty, readded,
      liveGeneration, mutable, backToCommitted, committedCached, afterDuplicate,
      frameBeforeDuplicate, frameAfterDuplicate, duplicateError,
      errors, frames, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes, stride: STRIDE };
  } finally { restoreWrites?.(); renderer?.dispose(); canvas.remove(); }
}
