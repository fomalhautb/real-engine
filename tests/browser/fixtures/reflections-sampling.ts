import surfaceFrame from '../../../src/shaders/surface-frame.wgsl?raw';
import surfaceBrdf from '../../../src/shaders/surface-brdf.wgsl?raw';
import reflectionCommon from '../../../src/shaders/reflections-common.wgsl?raw';
import reflectionUpdate from '../../../src/shaders/reflections-update.wgsl?raw';

const SAMPLE_COUNT = 16;
const VIEW_COSINES = [0.0001, 0.005, 0.1, 1] as const;

export async function runReflectionSamplingFixture() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('REFLECTION_SAMPLING_NO_ADAPTER');
  const adapterInfo = adapter.info;
  const device = await adapter.requestDevice();
  const buffers: GPUBuffer[] = [];
  let ownedBytes = 0;
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer);
    ownedBytes += size;
    return buffer;
  };
  const uncaptured: string[] = [];
  device.addEventListener('uncapturederror', event => uncaptured.push(event.error.message));
  device.pushErrorScope('validation');
  let validationError = '';
  const cases: { nv: number; eye: number[]; samples: { direction: number[]; weight: number[] }[] }[] = [];
  try {
    const code = surfaceFrame + '\n' + surfaceBrdf + '\n' + reflectionCommon + '\n'
      + reflectionUpdate.replace('// REFLECTION_GI',
        'fn reflectionDiffuse(hit:ReflectionHit)->vec3f {return vec3f(0.0);}')
      + '\nfn reflectionApplyFog(origin:vec3f,direction:vec3f,distance:f32,incoming:vec3f)->vec3f {return incoming;}';
    const shader = device.createShaderModule({ code });
    const compilation = await shader.getCompilationInfo();
    const compileErrors = compilation.messages.filter(message => message.type === 'error');
    if (compileErrors.length) throw new Error(compileErrors.map(error => error.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: shader, entryPoint: 'generateReflections' },
    });
    const uniform = allocate(176, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const camera = allocate(160, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const rays = allocate(SAMPLE_COUNT * 48, GPUBufferUsage.STORAGE);
    const samples = allocate(SAMPLE_COUNT * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(SAMPLE_COUNT * 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: rays } },
      { binding: 3, resource: { buffer: camera } },
      { binding: 5, resource: { buffer: samples } },
    ] });
    const hit = new Float32Array(40);
    const hitU32 = new Uint32Array(hit.buffer);
    hitU32[0] = 1; // valid surface identity
    hit.set([0, 0, 0, 0], 8); // position and geometric octahedral X
    hit.set([0, 0, 1, 0.5], 12); // shading normal and roughness
    hit[19] = 1; // metallic
    hit.set([1, 1, 1, 1], 20); // base color
    hitU32[28] = 7; // frame version
    hitU32[31] = 11; // material version
    hit[35] = 0; // geometric octahedral Y
    device.queue.writeBuffer(camera, 0, hit);
    for (const nv of VIEW_COSINES) {
      const eye = [Math.sqrt(1 - nv * nv), 0, nv];
      const parameters = new Float32Array(44);
      const words = new Uint32Array(parameters.buffer);
      parameters.set([...eye, 20], 0);
      words.set([1, 1, 1, 1], 28); // viewport
      words.set([1, 1, SAMPLE_COUNT, 1], 32); // spacing, anchors, samples, enabled
      words.set([7, 11, 0, 0], 36); // matching frame and material versions
      parameters[40] = 0.0001; // ray normal bias
      device.queue.writeBuffer(uniform, 0, parameters);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(samples, 0, readback, 0, SAMPLE_COUNT * 32);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      let floats: Float32Array;
      try { floats = new Float32Array(readback.getMappedRange().slice(0)); }
      finally { readback.unmap(); }
      cases.push({ nv, eye, samples: Array.from({ length: SAMPLE_COUNT }, (_, index) => {
        const offset = index * 8;
        return { direction: Array.from(floats.subarray(offset, offset + 4)),
          weight: Array.from(floats.subarray(offset + 4, offset + 8)) };
      }) });
    }
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally {
      for (const buffer of buffers) { buffer.destroy(); ownedBytes -= buffer.size; }
      device.destroy();
    }
  }
  return { adapterInfo: {
    vendor: adapterInfo.vendor, architecture: adapterInfo.architecture,
    device: adapterInfo.device, description: adapterInfo.description,
    isFallbackAdapter: adapterInfo.isFallbackAdapter,
  }, cases, uncaptured, validationError, ownedBytes };
}
