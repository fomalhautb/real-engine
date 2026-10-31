import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';

// Non-uniform inputs (indexed by fragment position) written beside a float MRT target, as the
// raster receivers do. Values sit away from rounding boundaries, so f32 and FMA agree exactly.
const CASES = [
  [0.25, -5.0e-7, -0.3432662785, 0], [-0.1875, 0.1875, 1, -1], [0.75, -0.75, 0.5, -0.5], [0, 0, -0.25, 0.25],
  [2, -3, 0.3, -0.7], [0.999, -0.999, 0.0001, -0.0001], [0.125, 0.875, -0.625, 0.375], [-0.9, 0.6, 0.45, -0.15],
];

const hex = (word: number) => '0x' + word.toString(16).padStart(8, '0');
const reference = (x: number, y: number) => {
  const lane = (value: number) => Math.floor(0.5 + 32767 * Math.min(1, Math.max(-1, Math.fround(value)))) & 0xffff;
  return (lane(x) | (lane(y) << 16)) >>> 0;
};

const layouts = {
  receiver: { pack: 'surfacePackSnorm2x16', lanes: 'vec4u(index + 1u, 0xffffff01u, PACK(c.xy), PACK(c.zw))' },
  allLanes: { pack: 'surfacePackSnorm2x16', lanes: 'vec4u(PACK(c.xy), PACK(c.zw), PACK(-c.xy), PACK(-c.zw))' },
  builtinReceiver: { pack: 'pack2x16snorm', lanes: 'vec4u(index + 1u, 0xffffff01u, PACK(c.xy), PACK(c.zw))' },
} as const;

function expected(layout: keyof typeof layouts, index: number): number[] {
  const [a, b, c, d] = CASES[index];
  return layout === 'allLanes' ? [reference(a, b), reference(c, d), reference(-a, -b), reference(-c, -d)]
    : [index + 1, 0xffffff01, reference(a, b), reference(c, d)];
}

export async function runFragmentPackFixture() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice(), errors: string[] = [];
  device.onuncapturederror = event => errors.push(event.error.message);
  const width = CASES.length;
  const cases = device.createBuffer({ size: width * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(cases, 0, new Float32Array(CASES.flat()));
  const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
  const results: Record<string, { mismatches: unknown[], floatMismatches: number }> = {};
  for (const [name, layout] of Object.entries(layouts)) {
    const module = device.createShaderModule({ label: 'fragment pack ' + name, code: surfaceFrameSource + `
      @group(0) @binding(0) var<storage, read> cases: array<vec4f>;
      struct Out { @location(0) motion: vec4f, @location(1) identity: vec4u }
      @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        return vec4f(array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0))[i], 0.5, 1.0);
      }
      @fragment fn fs(@builtin(position) position: vec4f) -> Out {
        let index = u32(position.x); let c = cases[index];
        return Out(vec4f(c.xyz, 1.0 / position.w), ${layout.lanes.replaceAll('PACK', layout.pack)});
      }` });
    const pipeline = device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba32float' }, { format: 'rgba32uint' }] } });
    const targets = (['rgba32float', 'rgba32uint'] as const).map(format => device.createTexture({ size: [width, 1], format, usage }));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: targets.map(texture => ({ view: texture.createView(),
      clearValue: [0, 0, 0, 0], loadOp: 'clear' as const, storeOp: 'store' as const })) });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cases } }] }));
    pass.draw(3); pass.end();
    const readbacks = targets.map(texture => {
      const buffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: 256 }, [width, 1]);
      return buffer;
    });
    device.queue.submit([encoder.finish()]);
    await Promise.all(readbacks.map(buffer => buffer.mapAsync(GPUMapMode.READ)));
    const motion = new Float32Array(readbacks[0].getMappedRange().slice(0)), identity = new Uint32Array(readbacks[1].getMappedRange().slice(0));
    const mismatches: unknown[] = [];
    let floatMismatches = 0;
    for (let i = 0; i < width; i += 1) {
      const actual = Array.from(identity.subarray(i * 4, i * 4 + 4)), want = expected(name as keyof typeof layouts, i);
      if (actual.some((word, lane) => word !== want[lane])) mismatches.push({ index: i, actual: actual.map(hex), expected: want.map(hex) });
      if ([0, 1, 2].some(lane => motion[i * 4 + lane] !== Math.fround(CASES[i][lane])) || motion[i * 4 + 3] !== 1) floatMismatches += 1;
    }
    results[name] = { mismatches, floatMismatches };
    readbacks.forEach(buffer => buffer.destroy()); targets.forEach(texture => texture.destroy());
  }
  await device.queue.onSubmittedWorkDone();
  cases.destroy();
  const report = { adapter: adapter.info.vendor + ' / ' + adapter.info.architecture, cases: width, results, errors };
  device.destroy();
  return report;
}
