import { FINAL_GATHER_ANCHOR_BYTES, FINAL_GATHER_PARAMETER_BYTES } from '../../../src/gpu-final-gather';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from '../../../src/gi-visibility';
import { WORLD_HIT_BYTES } from '../../../src/world-query';
import commonSource from '../../../src/shaders/diffuse-gi-common.wgsl?raw';
import interpolateSource from '../../../src/shaders/diffuse-gi-interpolate.wgsl?raw';
import updateSource from '../../../src/shaders/final-gather-update.wgsl?raw';
import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';

const anchorWords = FINAL_GATHER_ANCHOR_BYTES / 4;
const parameterWords = FINAL_GATHER_PARAMETER_BYTES / 4;
if (anchorWords !== 28 || parameterWords !== 80) throw new Error('GI_VARIANCE_LAYOUT');

type Anchor = { frame: number; value: number; count?: number; variance?: number;
  identity?: [number, number, number, number]; versions?: [number, number, number, number];
  position?: [number, number, number] };
type Estimate = { mean: number; count: number; variance: number };

/** Directly drives the production temporal pass with known frame estimates.
 * Transport sampling and denoise are intentionally outside this fixture. */
export async function runGiVarianceHistoryFixture() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('GI_VARIANCE_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  device.pushErrorScope('validation');
  let ownedBytes = 0, validationError = '';
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer); ownedBytes += size; return buffer;
  };
  let result: { estimates: number[]; stationary: Estimate[];
    discontinuities: (Estimate & { label: string })[]; moved: Estimate;
    compacted: { sourceSlot: number; targetSlot: number; mapRank: number; recovered: Estimate; withoutMap: Estimate;
      rejected: (Estimate & { label: string })[] } } | undefined;
  try {
    const source = surfaceFrameSource + '\n' + updateSource.replace('// GI_COMMON', commonSource.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`))
      .replace('// GI_INTERPOLATE', interpolateSource);
    const module = device.createShaderModule({ label: 'production final gather variance history', code: source });
    const compilation = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (compilation.length) throw new Error(compilation.map(error => error.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'accumulate' } });
    const parameters = allocate(FINAL_GATHER_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const cameraHits = allocate(4 * WORLD_HIT_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const current = allocate(4 * FINAL_GATHER_ANCHOR_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const previous = allocate(4 * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const historyCoverageMap = allocate(4 * 9 * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const readback = allocate(4 * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: parameters } }, { binding: 4, resource: { buffer: cameraHits } },
      { binding: 6, resource: { buffer: current } }, { binding: 7, resource: { buffer: previous } },
      { binding: 23, resource: { buffer: historyCoverageMap } },
    ] });
    const pack = (input: Anchor, slot = 0, records = new Float32Array(4 * anchorWords)) => {
      const base = slot * anchorWords, words = new Uint32Array(records.buffer);
      records.set([...(input.position ?? [0, 0, 0]), 0], base);
      records.set([0, 1, 0, 1], base + 4); // +Y material and oct-encoded geometric normal.
      words.set(input.identity ?? [1, 10, 1, 7], base + 8);
      words.set(input.versions ?? [input.frame, 2, 3, 4], base + 12);
      records.set([input.value, input.value, input.value, input.count ?? 1], base + 16);
      words.set([1, input.frame, 0, 0], base + 20);
      records[base + 24] = input.variance ?? 0;
      return records;
    };
    const dispatch = async (frame: number, currentAnchors: Anchor[], oldAnchors: Anchor[],
      moving = false, reset = false, maxHistory = 4, coverage?: Uint32Array) => {
      const count = currentAnchors.length, data = new Uint32Array(parameterWords), floats = new Float32Array(data.buffer);
      if (moving || coverage) {
        // Previous world origin projects to pixel (2.5,2.5), centered among four anchors.
        floats.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.25, -0.25, 0.5, 1], 16);
      }
      data.set([moving || coverage ? 4 : 1, moving || coverage ? 4 : 1, moving ? 2 : 1, moving ? 2 : 1], 36);
      data.set([2, count, 1, maxHistory], 40);
      data.set([frame, 1, Number(reset), 0], 44);
      data[55] = Number(moving);
      if (coverage) {
        data.set([1, 0, 2, 3], 72); // One primary anchor, three compacted extras, 2 px coverage tiles.
        data.set([4, 4, 1, 0], 76);
      }
      device.queue.writeBuffer(parameters, 0, data);
      device.queue.writeBuffer(historyCoverageMap, 0, coverage ?? new Uint32Array(4 * 9));
      const fresh = new Float32Array(4 * anchorWords), old = new Float32Array(4 * anchorWords);
      currentAnchors.forEach((anchor, slot) => pack(anchor, slot, fresh));
      oldAnchors.forEach((anchor, slot) => pack(anchor, slot, old));
      device.queue.writeBuffer(current, 0, fresh);
      device.queue.writeBuffer(previous, 0, old);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(current, 0, readback, 0, count * FINAL_GATHER_ANCHOR_BYTES);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ, 0, count * FINAL_GATHER_ANCHOR_BYTES);
      const output = new Float32Array(readback.getMappedRange(0, count * FINAL_GATHER_ANCHOR_BYTES).slice(0));
      readback.unmap();
      return Array.from({ length: count }, (_, slot) => ({ mean: output[slot * anchorWords + 16],
        count: output[slot * anchorWords + 19], variance: output[slot * anchorWords + 24] }));
    };
    const estimates = [0, 2, 4, 8, 16, 0];
    const stationary = [];
    let old: Anchor = { frame: 0, value: 999, variance: 999, count: 4 };
    for (let index = 0; index < estimates.length; index++) {
      const frame = index + 1;
      const output = (await dispatch(frame, [{ frame, value: estimates[index] }], [old], false, index === 0))[0];
      stationary.push(output);
      old = { frame, value: output.mean, count: output.count, variance: output.variance };
    }
    const baseline: Anchor = { frame: 6, value: stationary[5].mean,
      count: stationary[5].count, variance: stationary[5].variance };
    const fresh: Anchor = { frame: 7, value: 3 };
    const discontinuities = [];
    const scenarios: [string, Anchor, Anchor, boolean][] = [
      ['reset', fresh, baseline, true],
      ['source', { ...fresh, identity: [2, 10, 1, 7] }, baseline, false],
      ['entity', { ...fresh, identity: [1, 11, 1, 7] }, baseline, false],
      ['generation', { ...fresh, identity: [1, 10, 2, 7] }, baseline, false],
      ['representation-token', { ...fresh, identity: [1, 10, 1, 8] }, baseline, false],
      ['history-version', { ...fresh, versions: [7, 3, 3, 4] }, baseline, false],
      ['topology-version', { ...fresh, versions: [7, 2, 4, 4] }, baseline, false],
      ['material-revision', { ...fresh, versions: [7, 2, 3, 5] }, baseline, false],
    ];
    for (const [label, next, prior, reset] of scenarios) {
      discontinuities.push({ label, ...(await dispatch(7, [next], [prior], false, reset))[0] });
    }
    const oldMoving = [0, 2, 4, 6].map((value, slot): Anchor => ({ frame: 9, value, count: 4, variance: 1,
      identity: [1, 10, 1, slot + 20], position: [slot % 2 * 0.02, 0, Math.floor(slot / 2) * 0.02] }));
    const moved = (await dispatch(10, [{ frame: 10, value: 10, identity: [1, 10, 1, 99] },
      ...Array.from({ length: 3 }, (): Anchor => ({ frame: 10, value: 0 }))], oldMoving, true, false, 8))[0];
    // The stationary receiver moves from slot 1 to slot 3 as other tiles compact.
    // Slot 3's unrelated, bright history must fail the fast identity check. Only
    // rank 7 of its previous tile maps to the matching receiver in old slot 1.
    const sourceSlot = 1, targetSlot = 3, mapRank = 7;
    const coverage = new Uint32Array(4 * 9);
    coverage[3 * 8 + mapRank] = sourceSlot + 1;
    const oldCompacted: Anchor[] = [
      { frame: 9, value: 0, identity: [1, 100, 1, 7] },
      { frame: 9, value: 2, count: 4, variance: 1 },
      { frame: 9, value: 0, identity: [1, 102, 1, 7] },
      { frame: 9, value: 999, count: 4, variance: 999, identity: [1, 199, 1, 7] },
    ];
    const compactedFresh: Anchor = { frame: 10, value: 10 };
    const dispatchCompacted = async (next: Anchor, map: Uint32Array, reset = false,
      prior: Anchor[] = oldCompacted) => (await dispatch(10, [
      { frame: 10, value: 0, identity: [1, 100, 1, 7] },
      { frame: 10, value: 0, identity: [1, 101, 1, 7] },
      { frame: 10, value: 0, identity: [1, 102, 1, 7] }, next,
    ], prior, false, reset, 8, map))[targetSlot];
    const recovered = await dispatchCompacted(compactedFresh, coverage);
    const withoutMap = await dispatchCompacted(compactedFresh, new Uint32Array(4 * 9));
    const rejected = [];
    const compactedScenarios: [string, Anchor, boolean][] = [
      ['reset', compactedFresh, true],
      ['source', { ...compactedFresh, identity: [2, 10, 1, 7] }, false],
      ['entity', { ...compactedFresh, identity: [1, 11, 1, 7] }, false],
      ['generation', { ...compactedFresh, identity: [1, 10, 2, 7] }, false],
      ['representation-token', { ...compactedFresh, identity: [1, 10, 1, 8] }, false],
      ['history-version', { ...compactedFresh, versions: [10, 3, 3, 4] }, false],
      ['topology-version', { ...compactedFresh, versions: [10, 2, 4, 4] }, false],
      ['material-revision', { ...compactedFresh, versions: [10, 2, 3, 5] }, false],
      ['position', { ...compactedFresh, position: [0.0002, 0, 0] }, false],
    ];
    for (const [label, next, reset] of compactedScenarios) {
      rejected.push({ label, ...await dispatchCompacted(next, coverage, reset) });
    }
    const stale = oldCompacted.map((anchor, slot) => slot === sourceSlot ? { ...anchor, frame: 8 } : anchor);
    rejected.push({ label: 'stale-frame', ...await dispatchCompacted(compactedFresh, coverage, false, stale) });
    result = { estimates, stationary, discontinuities, moved,
      compacted: { sourceSlot, targetSlot, mapRank, recovered, withoutMap, rejected } };
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally { for (const buffer of buffers) { buffer.destroy(); ownedBytes -= buffer.size; } device.destroy(); }
  }
  return { ...result!, errors, validationError, ownedBytes };
}
