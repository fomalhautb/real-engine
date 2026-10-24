import commonSource from '../../../src/shaders/diffuse-gi-common.wgsl?raw';
import interpolateSource from '../../../src/shaders/diffuse-gi-interpolate.wgsl?raw';
import updateSource from '../../../src/shaders/final-gather-update.wgsl?raw';
import coverageSource from '../../../src/shaders/final-gather-coverage.wgsl?raw';
import cacheSource from '../../../src/shaders/final-gather-cache.wgsl?raw';
import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from '../../../src/gi-visibility';
import { FINAL_GATHER_ANCHOR_BYTES, FINAL_GATHER_PARAMETER_BYTES } from '../../../src/gpu-final-gather';
import { WORLD_HIT_BYTES } from '../../../src/world-query';

const CANDIDATES = 8;
const PRIMARY_CAPACITY = 9;
type Vec4 = [number, number, number, number];

function putU4(view: DataView, offset: number, values: readonly number[]) {
  for (let i = 0; i < 4; i++) view.setUint32(offset + i * 4, values[i], true);
}
function putF4(view: DataView, offset: number, values: readonly number[]) {
  for (let i = 0; i < 4; i++) view.setFloat32(offset + i * 4, values[i], true);
}
function anchor(view: DataView, lane: number, options: {
  frame: number; valid: boolean; gray: number; count: number; variance: number; age: number;
  generation?: number; topology?: number; material?: number;
}) {
  const at = lane * FINAL_GATHER_ANCHOR_BYTES, id = 100 + lane;
  putF4(view, at, [lane * 2, 0, 0, 0]);
  putF4(view, at + 16, [0, 1, 0, 0]);
  putU4(view, at + 32, [id, id, options.generation ?? 1, lane + 1]);
  putU4(view, at + 48, [options.frame, 11, options.topology ?? 1, options.material ?? 8]);
  putF4(view, at + 64, [options.gray, options.gray, options.gray, options.count]);
  putU4(view, at + 80, [Number(options.valid), options.frame, 0, 0]);
  putF4(view, at + 96, [options.variance, options.age, 0, 0]);
}
function candidate(view: DataView, lane: number) {
  const at = lane * WORLD_HIT_BYTES, id = 100 + lane;
  putU4(view, at, [id, id, 1, lane + 1]);
  putF4(view, at + 32, [lane * 2, 0, 0, 0]);
  putF4(view, at + 48, [0, 1, 0, 0]);
  putF4(view, at + 64, [lane * 2, 0, 0, 0]);
  putU4(view, at + 112, [10, 11, 1, 8]);
}
function decode(data: ArrayBuffer): {
  identity: Vec4; versions: Vec4; irradiance: Vec4; state: Vec4; moments: Vec4;
}[] {
  const view = new DataView(data);
  const u4 = (at: number) => Array.from({ length: 4 }, (_, i) => view.getUint32(at + i * 4, true)) as Vec4;
  const f4 = (at: number) => Array.from({ length: 4 }, (_, i) => view.getFloat32(at + i * 4, true)) as Vec4;
  return Array.from({ length: CANDIDATES }, (_, lane) => {
    const at = lane * FINAL_GATHER_ANCHOR_BYTES;
    return { identity: u4(at + 32), versions: u4(at + 48), irradiance: f4(at + 64),
      state: u4(at + 80), moments: f4(at + 96) };
  });
}

/** One-tile production cache update with no world query or renderer scene. */
export async function runGiSupplementalCacheValuesFixture() {
  if (WORLD_HIT_BYTES !== 160 || FINAL_GATHER_ANCHOR_BYTES !== 112 || FINAL_GATHER_PARAMETER_BYTES !== 320) {
    throw new Error('GI_SUPPLEMENTAL_CACHE_LAYOUT');
  }
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('GI_SUPPLEMENTAL_CACHE_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const errors: string[] = [], buffers: GPUBuffer[] = [];
  let ownedBytes = 0, validationError = '';
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer); ownedBytes += size; return buffer;
  };
  device.pushErrorScope('validation');
  let ordinary: ReturnType<typeof decode> = [], reset: ReturnType<typeof decode> = [];
  try {
    if (device.limits.maxStorageBuffersPerShaderStage < 5) throw new Error('GI_SUPPLEMENTAL_CACHE_BINDINGS');
    const parameters = allocate(FINAL_GATHER_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const rawNew = allocate(PRIMARY_CAPACITY * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const candidates = allocate(CANDIDATES * WORLD_HIT_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const map = allocate(CANDIDATES * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const previous = allocate(CANDIDATES * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const next = allocate(CANDIDATES * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(CANDIDATES * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const oldData = new ArrayBuffer(previous.size), oldView = new DataView(oldData);
    for (let lane = 0; lane < CANDIDATES; lane++) anchor(oldView, lane, {
      frame: 9, valid: lane !== 7, gray: 2, count: 4, variance: 2, age: lane === 1 ? 3 : lane === 2 ? 8 : 0,
      generation: lane === 5 ? 2 : 1, topology: lane === 3 ? 2 : 1, material: lane === 4 ? 7 : 8,
    });
    device.queue.writeBuffer(previous, 0, oldData);
    const candidateData = new ArrayBuffer(candidates.size), candidateView = new DataView(candidateData);
    for (let lane = 0; lane < CANDIDATES; lane++) candidate(candidateView, lane);
    device.queue.writeBuffer(candidates, 0, candidateData);
    const freshData = new ArrayBuffer(rawNew.size), freshView = new DataView(freshData);
    for (const lane of [0, 6, 7]) anchor(freshView, lane, {
      frame: 10, valid: lane !== 6, gray: lane === 6 ? 0 : 6, count: 1, variance: 0, age: 0,
    });
    device.queue.writeBuffer(rawNew, 0, freshData);
    const mapped = new Uint32Array(CANDIDATES); mapped[0] = 1; mapped[6] = 7; mapped[7] = 8;
    device.queue.writeBuffer(map, 0, mapped);

    const common = commonSource.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`);
    const source = surfaceFrameSource + '\n'
      + updateSource.replace('// GI_COMMON', common).replace('// GI_INTERPOLATE', interpolateSource)
      + '\n' + coverageSource + '\n'
      + cacheSource.replace('// CACHE_CONSTANTS', `const CACHE_PRIMARY_CAPACITY: u32 = ${PRIMARY_CAPACITY}u;`);
    const shader = device.createShaderModule({ label: 'GI supplemental cache values', code: source });
    const compileErrors = (await shader.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (compileErrors.length) throw new Error(compileErrors.map(error => `${error.lineNum}: ${error.message}`).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
      compute: { module: shader, entryPoint: 'updateSupplementalHistory' } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: parameters } }, { binding: 6, resource: { buffer: rawNew } },
      { binding: 20, resource: { buffer: candidates } }, { binding: 22, resource: { buffer: map } },
      { binding: 24, resource: { buffer: previous } }, { binding: 25, resource: { buffer: next } },
    ] });
    const run = async (resetHistory: boolean) => {
      const data = new Float32Array(FINAL_GATHER_PARAMETER_BYTES / 4), u32 = new Uint32Array(data.buffer);
      u32.set([1, 1, 1, 1], 36); // viewport
      u32.set([1, PRIMARY_CAPACITY, 1, 16], 40); // spacing, capacity, rays, max history
      u32.set([10, 1, Number(resetHistory), 0], 44); // frame, enabled, reset, bounce
      u32[54] = 8; // material revision (emissive.z)
      u32.set([1, 0, 1, 8], 72); // raster mode, metadata rows, tile pitch, budget
      u32.set([1, 1, 1, 8], 76); // capacity, tiles, permutation, cache age cap
      device.queue.writeBuffer(parameters, 0, data);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(next, 0, readback, 0, readback.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      try { return decode(readback.getMappedRange().slice(0)); }
      finally { readback.unmap(); }
    };
    ordinary = await run(false);
    reset = await run(true);
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally { for (const buffer of buffers) { if (buffer.mapState === 'mapped') buffer.unmap();
      buffer.destroy(); ownedBytes -= buffer.size; } device.destroy(); }
  }
  return { ordinary, reset, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    isFallbackAdapter: adapter.info.isFallbackAdapter }, errors, validationError, ownedBytes };
}
