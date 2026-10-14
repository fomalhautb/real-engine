import type { MaterialBinding } from '../../../src/contracts';
import { MaterialResources, type GpuMaterial } from '../../../src/gpu-material';
import {
  DEFAULT_MATERIAL_DEFINITION, evaluateMaterial, m, type MaterialEvaluation,
  type MaterialEvaluationInputs, type Vec4,
} from '../../../src/material';
import type { TextureAsset } from '../../../src/texture-assets';

/** This fixture evaluates material parameters in compute. It does not implement GI or ray tracing. */
const materialBindingsWGSL = `
struct MaterialParameters { values: array<vec4f, 16> }
@group(0) @binding(0) var<uniform> materialParameters: MaterialParameters;
@group(0) @binding(1) var materialImage0: texture_2d<f32>;
@group(0) @binding(2) var materialImage1: texture_2d<f32>;
@group(0) @binding(3) var materialImage2: texture_2d<f32>;
@group(0) @binding(4) var materialImage3: texture_2d<f32>;
@group(0) @binding(5) var materialSampler0: sampler;
@group(0) @binding(6) var materialSampler1: sampler;
@group(0) @binding(7) var materialSampler2: sampler;
@group(0) @binding(8) var materialSampler3: sampler;
fn materialTexture0(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage0, materialSampler0, uv, lod); }
fn materialTexture1(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage1, materialSampler1, uv, lod); }
fn materialTexture2(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage2, materialSampler2, uv, lod); }
fn materialTexture3(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage3, materialSampler3, uv, lod); }
`;
const probeWGSL = `
struct ProbeInputs { instanceColor: vec4f, metalRoughUv: vec4f, lodAndPadding: vec4f }
@group(1) @binding(0) var<storage, read_write> result: array<vec4f, 4>;
@group(1) @binding(1) var<uniform> probe: ProbeInputs;
@compute @workgroup_size(1)
fn main() {
    let surface = evaluateMaterial(probe.metalRoughUv.zw, probe.lodAndPadding.x, probe.instanceColor, probe.metalRoughUv.xy);
    let covered = evaluateCoverage(probe.metalRoughUv.zw, probe.lodAndPadding.x, probe.instanceColor, probe.metalRoughUv.xy);
    // Packed floats: [base RGB, opacity], [metallic, roughness, cutoff, coverage],
    // [emissive RGB, 0], [0, 0, 0, 0]. CPU checks all 16 lanes, including padding.
    result[0] = vec4f(surface.baseColor, surface.opacity);
    result[1] = vec4f(surface.metallic, surface.roughness, surface.alphaCutoff, select(0.0, 1.0, covered));
    result[2] = vec4f(surface.emissive, 0.0);
    result[3] = vec4f(0.0);
}
`;

function packed(surface: MaterialEvaluation): number[] {
  return [...surface.baseColor, surface.opacity, surface.metallic, surface.roughness,
    surface.alphaCutoff, surface.covered ? 1 : 0, ...surface.emissive, 0, 0, 0, 0, 0];
}
function linearFromSrgbByte(value: number): number {
  const encoded = value / 255;
  return encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4;
}

export async function runMaterialEvaluationFixture() {
  if (!navigator.gpu) throw new Error('MATERIAL_COMPUTE_WEBGPU_UNAVAILABLE');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('MATERIAL_COMPUTE_ADAPTER_UNAVAILABLE');
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let disposed = false;
  void device.lost.then(loss => { if (!disposed) errors.push('DEVICE_LOST: ' + loss.reason + ' ' + loss.message); });
  const buffers: GPUBuffer[] = [];
  let resources: MaterialResources | undefined;
  let ownedGpuBytes = 0;
  let pipelineCompilations = 0;
  const rows: { name: string; gpu: number[]; cpu: number[]; maxError: number; tolerance: number }[] = [];
  let parameterUpdate = { sameProgram: false, compilationsBefore: 0, compilationsAfter: 0, bytesBefore: 0, bytesAfter: 0 };
  let released = { materials: -1, images: -1, imageBytes: -1 };
  try {
    resources = new MaterialResources(device, bytes => { ownedGpuBytes += bytes; }, 1024);
    const manager = resources;
    const buffer = (size: number, usage: GPUBufferUsageFlags, label: string) => {
      const created = device.createBuffer({ size, usage, label });
      buffers.push(created);
      return created;
    };
    const output = buffer(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'material compute result');
    const read = buffer(64, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'material compute diagnostic readback');
    const probe = buffer(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'material compute probe inputs');
    const probeLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage', minBindingSize: 64 } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: 48 } },
    ] });
    const probeGroup = device.createBindGroup({ layout: probeLayout, entries: [
      { binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: probe } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [manager.layout, probeLayout] });
    const pipelines = new Map<string, GPUComputePipeline>();
    const run = async (name: string, material: GpuMaterial, inputs: MaterialEvaluationInputs, tolerance = 0.00002) => {
      let pipeline = pipelines.get(material.program.key);
      if (!pipeline) {
        const shader = device.createShaderModule({ label: 'material compute parity: ' + name,
          code: materialBindingsWGSL + material.program.wgsl + probeWGSL });
        const compilation = await shader.getCompilationInfo();
        const compileErrors = compilation.messages.filter(message => message.type === 'error');
        if (compileErrors.length) throw new Error(compileErrors.map(message => `${name}:${message.lineNum}:${message.message}`).join('\n'));
        pipeline = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module: shader, entryPoint: 'main' } });
        pipelines.set(material.program.key, pipeline);
        pipelineCompilations++;
      }
      const uv = inputs.uv ?? [0, 0];
      device.queue.writeBuffer(probe, 0, new Float32Array([
        ...inputs.instanceColor, ...inputs.instanceMetalRough, ...uv, inputs.lod ?? 0, 0, 0, 0,
      ]));
      const encoder = device.createCommandEncoder({ label: 'material compute parity' });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, material.bindGroup); pass.setBindGroup(1, probeGroup);
      pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, 64);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      let gpu: number[];
      try { gpu = Array.from(new Float32Array(read.getMappedRange())); }
      finally { read.unmap(); }
      if (!gpu.every(Number.isFinite)) throw new Error('MATERIAL_COMPUTE_NONFINITE: ' + name);
      const cpu = packed(evaluateMaterial(material.program, inputs));
      if (!cpu.every(Number.isFinite)) throw new Error('MATERIAL_REFERENCE_NONFINITE: ' + name);
      rows.push({ name, gpu, cpu, maxError: Math.max(...gpu.map((value, i) => Math.abs(value - cpu[i]))), tolerance });
    };

    const parameters = new Float32Array([2, 0.5, 0.25, 1, 8, 6, -2, 0, 0.25, 0.5, 0, 0]);
    const expressionBinding: MaterialBinding = { definition: {
      shading: 'pbr', alphaMode: 'mask', doubleSided: false,
      baseColor: m.add(m.multiply(m.instanceColor, m.parameter(0)), m.constant([0.25, 0.125, -0.5, 0])),
      metallic: m.component(m.instanceMetalRough, 0), roughness: m.component(m.instanceMetalRough, 1),
      emissive: m.mix(m.constant([0, 2, 4, 0]), m.parameter(1), m.component(m.parameter(2), 0)),
      alphaCutoff: m.component(m.parameter(2), 1),
    }, parameters };
    const expression = manager.create(expressionBinding);
    try {
      const inputs: MaterialEvaluationInputs = { parameters, instanceColor: [0.75, 0.5, 0.25, 0.5],
        instanceMetalRough: [2, 0], uv: [0.25, 0.75], lod: 0 };
      await run('expression-mask-equality', expression, inputs);
      const originalProgram = expression.program;
      parameterUpdate = { sameProgram: false, compilationsBefore: pipelineCompilations, compilationsAfter: 0,
        bytesBefore: ownedGpuBytes, bytesAfter: 0 };
      parameters[3] = 0.8; parameters[8] = 0.75;
      manager.updateParameters(expression, parameters);
      await run('expression-mask-parameter-update', expression, inputs);
      parameters[3] = -4; parameters[9] = -1;
      manager.updateParameters(expression, parameters);
      await run('expression-clamped-alpha-cutoff', expression, inputs);
      parameters[3] = 2; parameters[9] = 1.25;
      manager.updateParameters(expression, parameters);
      await run('expression-mask-cutoff-above-one', expression, inputs);
      parameterUpdate.sameProgram = expression.program === originalProgram;
      parameterUpdate.compilationsAfter = pipelineCompilations;
      parameterUpdate.bytesAfter = ownedGpuBytes;
    } finally { manager.release(expression); }

    const unlit = manager.create({ definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.constant([-1, 3, 9, 0]), emissive: m.constant([4, 0, 2, 0]), metallic: 0, roughness: 0.5 } });
    try { await run('unlit-hdr-opaque-zero-alpha', unlit, { instanceColor: [1, 1, 1, 1], instanceMetalRough: [0, 0.5] }); }
    finally { manager.release(unlit); }

    // Expected samples are analytic, not produced by buildMipChain. A 50/50 black/white sRGB
    // level averages in linear light, encodes to byte188, then hardware decodes on sampling.
    const srgb128 = linearFromSrgbByte(128), srgb188 = linearFromSrgbByte(188);
    const solid = (colorSpace: TextureAsset['colorSpace']): TextureAsset => ({
      width: 1, height: 1, colorSpace, pixels: new Uint8Array([128, 128, 128, 128]),
    });
    const mipTexture: TextureAsset = { width: 2, height: 2, colorSpace: 'srgb', pixels: new Uint8Array([
      0, 0, 0, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 0, 0, 0, 255,
    ]) };
    const cases: { name: string; texture: TextureAsset; lod: number; sample: Vec4 }[] = [
      { name: 'srgb-128', texture: solid('srgb'), lod: 0, sample: [srgb128, srgb128, srgb128, 128 / 255] },
      { name: 'linear-128', texture: solid('linear'), lod: 0, sample: [128 / 255, 128 / 255, 128 / 255, 128 / 255] },
      { name: 'srgb-mip-zero', texture: mipTexture, lod: 0, sample: [0, 0, 0, 1] },
      { name: 'srgb-mip-one', texture: mipTexture, lod: 1, sample: [srgb188, srgb188, srgb188, 1] },
    ];
    for (const sample of cases) {
      const material = manager.create({ definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', alphaMode: 'mask',
        baseColor: m.texture(0), metallic: 0, roughness: 0.5 }, textures: [sample.texture] });
      try {
        await run(sample.name, material, { instanceColor: [1, 1, 1, 1], instanceMetalRough: [0, 0.5], uv: [0.25, 0.25],
          lod: sample.lod, textures: () => sample.sample }, 0.004);
      } finally { manager.release(material); }
    }
    const stripes: TextureAsset = { width: 2, height: 1, colorSpace: 'linear',
      pixels: new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]) };
    const filters = { magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest', lodMaxClamp: 0 } as const;
    const distinctSamplers = manager.create({ definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.add(m.texture(0), m.texture(1)), emissive: m.add(m.texture(2), m.texture(3)) },
      textures: [stripes, stripes, stripes, stripes], samplers: [
        { ...filters, addressModeU: 'repeat' }, { ...filters, addressModeU: 'clamp-to-edge' },
        { ...filters, addressModeU: 'mirror-repeat' }, { ...filters, addressModeU: 'repeat' },
      ] });
    try {
      for (const u of [1.25, -0.25]) {
        const red: Vec4 = [1, 0, 0, 1], green: Vec4 = [0, 1, 0, 1];
        const expected = u > 1 ? [red, green, green, red] : [green, red, red, green];
        await run('per-slot-sampler-' + u, distinctSamplers, { instanceColor: [1, 1, 1, 1],
          instanceMetalRough: [0, 0.5], uv: [u, 0.5], lod: 1, textures: slot => expected[slot] });
      }
    } finally { manager.release(distinctSamplers); }
    // Invalid sampler descriptors must fail before retaining images or allocating uniforms.
    for (const invalid of [{ compare: 'less' }, { lodMaxClamp: -1 }, { maxAnisotropy: 2, minFilter: 'nearest' }]) {
      const bytes = ownedGpuBytes;
      let rejected = false;
      try { manager.create({ definition: DEFAULT_MATERIAL_DEFINITION, samplers: [invalid as GPUSamplerDescriptor] }); }
      catch (error) { rejected = String(error).includes('MATERIAL_SAMPLER_'); }
      if (!rejected || bytes !== ownedGpuBytes) throw new Error('MATERIAL_SAMPLER_INVALID_NOT_ATOMIC');
    }
    await device.queue.onSubmittedWorkDone();
    // Give uncaptured GPU validation events a task boundary before returning diagnostics.
    await new Promise<void>(resolve => setTimeout(resolve, 0));
    released = manager.diagnostics();
  } finally {
    resources?.dispose();
    for (const buffer of buffers) buffer.destroy();
    disposed = true;
    device.destroy();
  }
  return { rows, pipelineCompilations, parameterUpdate, released, ownedGpuBytesAfterDispose: ownedGpuBytes, errors };
}
