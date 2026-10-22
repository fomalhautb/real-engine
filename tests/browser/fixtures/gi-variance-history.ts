import { FINAL_GATHER_ANCHOR_BYTES, FINAL_GATHER_PARAMETER_BYTES } from '../../../src/gpu-final-gather';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from '../../../src/gi-visibility';
import { WORLD_HIT_BYTES } from '../../../src/world-query';
import commonSource from '../../../src/shaders/diffuse-gi-common.wgsl?raw';
import interpolateSource from '../../../src/shaders/diffuse-gi-interpolate.wgsl?raw';
import updateSource from '../../../src/shaders/final-gather-update.wgsl?raw';
import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';

const anchorWords = FINAL_GATHER_ANCHOR_BYTES / 4;
const parameterWords = FINAL_GATHER_PARAMETER_BYTES / 4;
if (anchorWords !== 28) throw new Error('GI_VARIANCE_ANCHOR_LAYOUT');

type Anchor = { frame: number; value: number; count?: number; variance?: number;
  identity?: [number, number, number, number]; versions?: [number, number, number, number];
  position?: [number, number, number] };

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
  let result: { estimates: number[]; stationary: { mean: number; count: number; variance: number }[];
    discontinuities: { label: string; mean: number; count: number; variance: number }[];
    moved: { mean: number; count: number; variance: number } } | undefined;
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
    const readback = allocate(4 * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: parameters } }, { binding: 4, resource: { buffer: cameraHits } },
      { binding: 6, resource: { buffer: current } }, { binding: 7, resource: { buffer: previous } },
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
      moving = false, reset = false, maxHistory = 4) => {
      const count = currentAnchors.length, data = new Uint32Array(parameterWords), floats = new Float32Array(data.buffer);
      if (moving) {
        // Previous world origin projects to pixel (2.5,2.5), centered among four anchors.
        floats.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.25, -0.25, 0.5, 1], 16);
      }
      data.set([moving ? 4 : 1, moving ? 4 : 1, moving ? 2 : 1, moving ? 2 : 1], 36);
      data.set([2, count, 1, maxHistory], 40);
      data.set([frame, 1, Number(reset), 0], 44);
      data[55] = Number(moving);
      device.queue.writeBuffer(parameters, 0, data);
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
    result = { estimates, stationary, discontinuities, moved };
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally { for (const buffer of buffers) { buffer.destroy(); ownedBytes -= buffer.size; } device.destroy(); }
  }
  return { ...result!, errors, validationError, ownedBytes };
}
