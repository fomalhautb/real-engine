import commonSource from '../../../src/shaders/diffuse-gi-common.wgsl?raw';
import interpolateSource from '../../../src/shaders/diffuse-gi-interpolate.wgsl?raw';
import updateSource from '../../../src/shaders/final-gather-update.wgsl?raw';
import coverageSource from '../../../src/shaders/final-gather-coverage.wgsl?raw';
import cacheSource from '../../../src/shaders/final-gather-cache.wgsl?raw';
import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION } from '../../../src/gi-visibility';
import { FINAL_GATHER_PARAMETER_BYTES } from '../../../src/gpu-final-gather';
import { DIFFUSE_GI_PARAMETER_BYTES } from '../../../src/diffuse-gi';

/** Executes the production ray generator with injected, fully declared previous
 * history. No scene queries: read back the actual ray directions and cache state. */
export async function runProgressiveSequenceFixture() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw Error('PROGRESSIVE_NO_ADAPTER');
  const device = await adapter.requestDevice(), buffers: GPUBuffer[] = [], errors: string[] = [];
  let ownedBytes = 0, validationError = '';
  device.addEventListener('uncapturederror', e => errors.push(e.error.message));
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size, usage }); buffers.push(buffer); ownedBytes += size; return buffer;
  };
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const rows = [], recovery = [];
  device.pushErrorScope('validation');
  try {
    const parameters = allocate(FINAL_GATHER_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const gi = allocate(DIFFUSE_GI_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const path = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const rays = allocate(32 * 48, storage | GPUBufferUsage.COPY_SRC);
    const camera = allocate(2 * 160, storage), primary = allocate(32 * 160, storage);
    const throughput = allocate(32 * 16, storage), map = allocate(9 * 4, storage);
    const previous = allocate(8 * 112, storage), next = allocate(8 * 112, storage | GPUBufferUsage.COPY_SRC);
    const fresh = allocate(2 * 112, storage), candidates = allocate(8 * 160, storage);
    const samples = allocate(16 * 32, storage | GPUBufferUsage.COPY_SRC);
    const readback = allocate(rays.size + next.size + samples.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const common = commonSource.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`);
    const shader = device.createShaderModule({ code: surfaceFrameSource + '\n'
      + updateSource.replace('// GI_COMMON', common).replace('// GI_INTERPOLATE', interpolateSource)
        .replace('// GATHER_PROGRESSIVE_DIRECTION', 'uv = cacheProgressiveUv(anchor, sample, uv);')
      + '\n' + coverageSource + '\n' + cacheSource.replace('// CACHE_CONSTANTS', 'const CACHE_PRIMARY_CAPACITY: u32 = 2u;')
        .replace('const CACHE_PROGRESSIVE_SAMPLING: bool = false;', 'const CACHE_PROGRESSIVE_SAMPLING: bool = true;')
        .replace('let history = cacheResolveHistory(id.x, current, hit.previous.xyz);', 'let history = cachePreparedHistory(id.x);')
      + `\n@group(0) @binding(27) var<storage, read_write> sampledUv: array<vec4f>;
        @compute @workgroup_size(64) fn inspectSequence(@builtin(global_invocation_id) id: vec3u) {
          if (id.x >= gather.settings.z) { return; }
          let uv = cacheProgressiveUv(1u, id.x, vec2f(-1.0));
          let angle = 6.28318530718 * uv.y;
          sampledUv[id.x * 2u] = vec4f(uv, sqrt(uv.x), sqrt(1.0 - uv.x));
          let sequence = cacheSequenceState(cacheCurrentSurface(cameraHits[1u]), cachePreparedHistory(3u));
          sampledUv[id.x * 2u + 1u] = vec4f(cos(angle), sin(angle), angle, f32(sequence.y));
        }` });
    const compileErrors = (await shader.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (compileErrors.length) throw Error(compileErrors.map(m => `${m.lineNum}: ${m.message}`).join('\n'));
    const generate = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'generateHemisphere' } });
    const prepare = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'prepareSupplementalHistory' } });
    const update = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'updateSupplementalHistory' } });
    const inspect = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint: 'inspectSequence' } });
    const bind = (pipeline: GPUComputePipeline, entries: [number, GPUBuffer][]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
    const generateGroup = bind(generate, [[0, parameters], [1, gi], [2, rays], [4, camera], [5, primary],
      [14, path], [15, throughput], [22, map], [25, next]]);
    const prepareGroup = bind(prepare, [[0, parameters], [20, candidates], [24, previous], [25, next]]);
    const updateGroup = bind(update, [[0, parameters], [6, fresh], [20, candidates], [22, map], [25, next]]);
    const inspectGroup = bind(inspect, [[0, parameters], [4, camera], [14, path], [22, map], [25, next], [27, samples]]);
    const makeHit = () => {
      const bytes = new ArrayBuffer(160), f = new Float32Array(bytes), u = new Uint32Array(bytes);
      u.set([2, 3, 4, 5]); f.set([0, 0, 0, 0], 8); f.set([0, 1, 0, 0], 12);
      f.set([1, 1, 1, 1], 20); f[35] = 1; // geometric normal +Y in octahedral coordinates
      u.set([10, 11, 1, 8], 28); return bytes;
    };
    const hit = makeHit(); device.queue.writeBuffer(camera, 160, hit);
    for (let i = 0; i < 32; i++) device.queue.writeBuffer(primary, i * 160, hit);
    device.queue.writeBuffer(candidates, 3 * 160, hit);
    const mapping = new Uint32Array(9); mapping[3] = 2; mapping[8] = 0;
    const giData = new Float32Array(DIFFUSE_GI_PARAMETER_BYTES / 4); giData[7] = 80;
    device.queue.writeBuffer(gi, 0, giData);
    for (const count of [1, 255, 256, 1_048_575, 1_048_576]) for (const raysPerAnchor of [1, 4, 16]) {
      const old = new ArrayBuffer(112), f = new Float32Array(old), u = new Uint32Array(old);
      f.set([0, 0, 0, 0]); f.set([0, 1, 0, 1], 4); u.set([2, 3, 4, 5], 8);
      u.set([9, 11, 1, 8], 12); f.set([2, 2, 2, 16], 16); u.set([1, 9, 0, 0], 20);
      f.set([0, 0, count, 0x123456], 24); device.queue.writeBuffer(previous, 3 * 112, old);
      const newData = old.slice(0), nf = new Float32Array(newData), nu = new Uint32Array(newData);
      nu[12] = 10; nu[21] = 10; nf.set([4, 4, 4, 1], 16); nf.fill(0, 24);
      device.queue.writeBuffer(fresh, 112, newData);
      const data = new Float32Array(FINAL_GATHER_PARAMETER_BYTES / 4), ints = new Uint32Array(data.buffer);
      ints.set([1, 1, 1, 1], 36); ints.set([1, 2, raysPerAnchor, 16], 40); ints.set([10, 1, 0, 1], 44);
      data[48] = 0.0001; ints[54] = 8; ints.set([1, 0, 1, 1], 72); ints.set([1, 1, 1, 8], 76);
      device.queue.writeBuffer(parameters, 0, data); device.queue.writeBuffer(map, 0, mapping);
      for (const segment of [0, 1, 2, 3]) {
        device.queue.writeBuffer(path, 0, new Uint32Array([segment, 4, 0, 0]));
        device.queue.writeBuffer(throughput, 0, new Float32Array(32 * 4).fill(1));
        const encoder = device.createCommandEncoder();
        for (const [pipeline, group] of [[prepare, prepareGroup], [generate, generateGroup], [inspect, inspectGroup], [update, updateGroup]] as const) {
          const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(1); pass.end();
        }
        encoder.copyBufferToBuffer(rays, 0, readback, 0, rays.size);
        encoder.copyBufferToBuffer(next, 0, readback, rays.size, next.size);
        encoder.copyBufferToBuffer(samples, 0, readback, rays.size + next.size, samples.size);
        device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const output = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
        const at = rays.size / 4 + 3 * 28;
        rows.push({ count, raysPerAnchor, segment,
          sampling: Array.from({ length: raysPerAnchor }, (_, i) => Array.from(output.slice((rays.size + next.size) / 4 + i * 8, (rays.size + next.size) / 4 + i * 8 + 8))),
          directions: Array.from({ length: raysPerAnchor }, (_, i) => Array.from(output.slice((raysPerAnchor + i) * 12 + 4, (raysPerAnchor + i) * 12 + 7))),
          rayIntervals: Array.from({ length: raysPerAnchor }, (_, i) => [output[(raysPerAnchor + i) * 12 + 3], output[(raysPerAnchor + i) * 12 + 7]]),
          historyCount: output[at + 19], sequence: Array.from(output.slice(at + 26, at + 28)) });
      }
    }
    // Inject failed fresh estimates, as produced by an incomplete bounded query.
    // No failed light is reusable, but its replacement must not be locked to the
    // same first direction. A subsequent valid history must keep its sequence.
    device.queue.writeBuffer(previous, 0, new Uint8Array(previous.size));
    for (const frame of [10, 11, 12, 13]) {
      const valid = frame >= 12;
      const hit = makeHit(); new Uint32Array(hit)[28] = frame;
      device.queue.writeBuffer(camera, 160, hit); device.queue.writeBuffer(candidates, 3 * 160, hit);
      const freshBytes = new ArrayBuffer(112), f = new Float32Array(freshBytes), u = new Uint32Array(freshBytes);
      f.set([0, 0, 0, 0]); f.set([0, 1, 0, 1], 4); u.set([2, 3, 4, 5], 8);
      u.set([frame, 11, 1, 8], 12); f.set(valid ? [4, 4, 4, 1] : [60000, 60000, 60000, 1], 16);
      u.set([Number(valid), frame, Number(!valid), 0], 20); device.queue.writeBuffer(fresh, 112, freshBytes);
      const parametersData = new Float32Array(FINAL_GATHER_PARAMETER_BYTES / 4), integers = new Uint32Array(parametersData.buffer);
      integers.set([1, 1, 1, 1], 36); integers.set([1, 2, 1, 16], 40); integers.set([frame, 1, 0, 1], 44);
      parametersData[48] = 0.0001; integers[54] = 8; integers.set([1, 0, 1, 1], 72); integers.set([1, 1, 1, 8], 76);
      device.queue.writeBuffer(parameters, 0, parametersData); device.queue.writeBuffer(path, 0, new Uint32Array([0, 1, 0, 0]));
      const encoder = device.createCommandEncoder();
      for (const [pipeline, group] of [[prepare, prepareGroup], [generate, generateGroup], [inspect, inspectGroup], [update, updateGroup]] as const) {
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      }
      encoder.copyBufferToBuffer(rays, 0, readback, 0, rays.size);
      encoder.copyBufferToBuffer(next, 0, readback, rays.size, next.size);
      encoder.copyBufferToBuffer(samples, 0, readback, rays.size + next.size, samples.size);
      encoder.copyBufferToBuffer(next, 0, previous, 0, next.size);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange().slice(0), output = new Float32Array(bytes), words = new Uint32Array(bytes); readback.unmap();
      const at = rays.size / 4 + 3 * 28, sampleAt = (rays.size + next.size) / 4;
      recovery.push({ frame, valid, direction: Array.from(output.slice(16, 19)), uv: Array.from(output.slice(sampleAt, sampleAt + 2)),
        generatedKey: output[sampleAt + 7], sequence: Array.from(output.slice(at + 26, at + 28)),
        state: Array.from(words.slice(at + 20, at + 24)), irradiance: Array.from(output.slice(at + 16, at + 20)) });
    }
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally { for (const b of buffers) { if (b.mapState === 'mapped') b.unmap(); b.destroy(); ownedBytes -= b.size; } device.destroy(); }
  }
  return { rows, recovery, errors, validationError, ownedBytes, adapter: { vendor: adapter.info.vendor,
    architecture: adapter.info.architecture, isFallbackAdapter: adapter.info.isFallbackAdapter } };
}
