import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { DIFFUSE_GI_PROBE_BYTES, DIFFUSE_GI_PROBE_FLOATS,
  DIFFUSE_GI_STATE_FLOAT_OFFSET, giRayDirection } from '../../../src/diffuse-gi';
import { createCube } from '../../../src/geometry';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION, buildGiDistanceWeights,
  octaDecode, octaEncode, wrapGiDistanceTexel } from '../../../src/gi-visibility';
import { diffuseGiSampleSource } from '../../../src/gpu-diffuse-gi';
import { multiply, normalMatrix, scaling, translation, type Mat4 } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const boxGrid = { origin: [-0.5, -0.5, -0.5] as Vec3,
  spacing: [1, 1, 1] as Vec3, counts: [2, 2, 2] as Vec3,
  raysPerProbe: 64, maxDistance: 10, skyRadiance: [0, 0, 0] as Vec3 };
const boxHalf: Vec3 = [3.3, 3.8, 4.2];

function canvas(): HTMLCanvasElement {
  const element = document.createElement('canvas');
  element.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(element);
  return element;
}

function instance(entityId: number, model: Mat4): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

function cube(name: string, entityId: number, position: Vec3, size: Vec3,
  material: SceneBatch['material']): SceneBatch {
  const model = multiply(translation(...position), scaling(...size));
  return { name, mesh: createCube(), capacity: 1, instances: [instance(entityId, model)],
    inputMode: 'committed', deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: `gi-visibility/${name}`, primitiveId: entityId, topologyVersion: 1 },
    material };
}

/** Distance to the first positive AABB face from a point strictly inside it.
 * This reference uses six analytic planes, independent of GPU triangle traversal. */
function boxExitDistance(origin: Vec3, direction: Vec3): number {
  let nearest = Infinity;
  for (let axis = 0; axis < 3; axis++) {
    const component = direction[axis];
    if (Math.abs(component) < 1e-12) continue;
    const face = component > 0 ? boxHalf[axis] : -boxHalf[axis];
    const distance = (face - origin[axis]) / component;
    if (distance > 0) nearest = Math.min(nearest, distance);
  }
  return Math.min(nearest, boxGrid.maxDistance);
}

function expectedBoxMoments(): number[][][] {
  const rays = Array.from({ length: boxGrid.raysPerProbe }, (_, index) =>
    giRayDirection(index, boxGrid.raysPerProbe));
  const weights = buildGiDistanceWeights(rays);
  return Array.from({ length: 8 }, (_, probe) => {
    const origin: Vec3 = [boxGrid.origin[0] + (probe & 1),
      boxGrid.origin[1] + ((probe >> 1) & 1), boxGrid.origin[2] + ((probe >> 2) & 1)];
    const distances = rays.map(direction => boxExitDistance(origin, direction));
    return Array.from({ length: GI_DISTANCE_CELLS }, (_, cell) => {
      let mean = 0, square = 0, total = 0;
      for (let ray = 0; ray < rays.length; ray++) {
        const weight = weights[cell * rays.length + ray];
        mean += distances[ray] * weight;
        square += distances[ray] ** 2 * weight;
        total += weight;
      }
      return [mean / total, square / total, total];
    });
  });
}

const epsilon = 1 / 4096;
const seamUv: [number, number][] = [
  [epsilon, 0.23], [epsilon, 0.77], [1 - epsilon, 0.23], [1 - epsilon, 0.77],
  [0.23, epsilon], [0.77, epsilon], [0.23, 1 - epsilon], [0.77, 1 - epsilon],
  [0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5],
  [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],
];
const seamPairs: [number, number][] = [[0, 1], [2, 3], [4, 5], [6, 7]];

function bilinearMoment(moments: number[][], direction: Vec3): [number, number] {
  const uv = octaEncode(direction);
  const px = uv[0] * GI_DISTANCE_RESOLUTION - 0.5;
  const py = uv[1] * GI_DISTANCE_RESOLUTION - 0.5;
  const x = Math.floor(px), y = Math.floor(py);
  const fx = px - x, fy = py - y;
  const taps = [wrapGiDistanceTexel(x, y), wrapGiDistanceTexel(x + 1, y),
    wrapGiDistanceTexel(x, y + 1), wrapGiDistanceTexel(x + 1, y + 1)];
  const weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  return [0, 1].map(channel => taps.reduce((sum, tap, index) =>
    sum + moments[tap][channel] * weights[index], 0)) as [number, number];
}

/** Dispatches the production giDistanceMoments helper with a separately owned
 * probe record. The 8x8 moments come from the actual Renderer readback. */
async function sampleProductionHelper(moments: number[][]) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('GI_VISIBILITY_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const buffers: GPUBuffer[] = [];
  let ownedBytes = 0;
  const allocate = (size: number, usage: GPUBufferUsageFlags): GPUBuffer => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer); ownedBytes += size;
    return buffer;
  };
  const uncaptured: string[] = [];
  device.addEventListener('uncapturederror', event => uncaptured.push(event.error.message));
  device.pushErrorScope('validation');
  let validationError = '';
  let samples: { uv: [number, number]; direction: Vec3; actual: number[]; expected: [number, number] }[] = [];
  try {
    const record = new Float32Array(DIFFUSE_GI_PROBE_FLOATS);
    for (let cell = 0; cell < GI_DISTANCE_CELLS; cell++) {
      record.set(moments[cell].slice(0, 3), 36 + cell * 4);
    }
    const state = new Uint32Array(record.buffer);
    state[DIFFUSE_GI_STATE_FLOAT_OFFSET] = 1;
    state[DIFFUSE_GI_STATE_FLOAT_OFFSET + 1] = 1;
    const probe = allocate(DIFFUSE_GI_PROBE_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(probe, 0, record);
    const uniform = allocate(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const parameters = new Float32Array(28);
    parameters[3] = 1;
    new Uint32Array(parameters.buffer).set([2, 2, 2, 64], 8);
    device.queue.writeBuffer(uniform, 0, parameters);
    const directions = seamUv.map(uv => octaDecode(uv));
    const inputData = new Float32Array(directions.length * 4);
    directions.forEach((direction, index) => inputData.set([...direction, 0], index * 4));
    const input = allocate(inputData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const output = allocate(inputData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(inputData.byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(input, 0, inputData);
    const source = diffuseGiSampleSource + `
@group(0) @binding(0) var<storage, read> testDirections: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> testMoments: array<vec4f>;
@compute @workgroup_size(64)
fn testDistanceMoments(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${directions.length}u) { return; }
  let sample = testDirections[id.x];
  testMoments[id.x] = vec4f(giDistanceMoments(u32(sample.w), normalize(sample.xyz)), 0.0, 0.0);
}`;
    const shader = device.createShaderModule({ code: source });
    const compilation = await shader.getCompilationInfo();
    const compileErrors = compilation.messages.filter(message => message.type === 'error');
    if (compileErrors.length) throw new Error(compileErrors.map(error => error.message).join('\n'));
    const ioLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });
    const emptyLayout = device.createBindGroupLayout({ entries: [] });
    const giLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ] });
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [ioLayout, emptyLayout, emptyLayout, giLayout] }),
      compute: { module: shader, entryPoint: 'testDistanceMoments' },
    });
    const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
    ] });
    const giGroup = device.createBindGroup({ layout: giLayout, entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: probe } },
    ] });
    const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, ioGroup);
    pass.setBindGroup(1, emptyGroup); pass.setBindGroup(2, emptyGroup); pass.setBindGroup(3, giGroup);
    pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, inputData.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    let values: Float32Array;
    try { values = new Float32Array(readback.getMappedRange().slice(0)); }
    finally { readback.unmap(); }
    samples = directions.map((direction, index) => ({ uv: seamUv[index], direction,
      actual: [values[index * 4], values[index * 4 + 1]],
      expected: bilinearMoment(moments, direction) }));
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally {
      for (const buffer of buffers) { buffer.destroy(); ownedBytes -= buffer.size; }
      device.destroy();
    }
  }
  return { samples, seamPairs, uncaptured, validationError, ownedBytes };
}

/** A separate, constant-radiance field checks the production eight-probe
 * interpolation, including the very small confidence at a grid fringe. */
export async function runGiUniformIrradianceFixture() {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('GI_IRRADIANCE_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const buffers: GPUBuffer[] = [];
  let ownedBytes = 0;
  const allocate = (size: number, usage: GPUBufferUsageFlags): GPUBuffer => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer); ownedBytes += size;
    return buffer;
  };
  const uncaptured: string[] = [];
  device.addEventListener('uncapturederror', event => uncaptured.push(event.error.message));
  device.pushErrorScope('validation');
  let validationError = '';
  const sky: Vec3 = [0.1, 0.2, 0.3];
  const diagonal = 1 / Math.sqrt(3);
  const queries: { label: string; position: Vec3; normal: Vec3 }[] = [
    { label: 'center +X', position: [0, 0, 0], normal: [1, 0, 0] },
    { label: 'center -X', position: [0, 0, 0], normal: [-1, 0, 0] },
    { label: 'center +Y', position: [0, 0, 0], normal: [0, 1, 0] },
    { label: 'center -Y', position: [0, 0, 0], normal: [0, -1, 0] },
    { label: 'center +Z', position: [0, 0, 0], normal: [0, 0, 1] },
    { label: 'center -Z', position: [0, 0, 0], normal: [0, 0, -1] },
    { label: 'positive fringe corner', position: [1, 1, 1], normal: [diagonal, diagonal, diagonal] },
    { label: 'negative fringe corner', position: [-1, -1, -1], normal: [-diagonal, -diagonal, -diagonal] },
    { label: 'probe center zero delta', position: [-0.5, -0.5, -0.5], normal: [0, 1, 0] },
  ];
  let samples: { label: string; position: Vec3; normal: Vec3; actual: number[]; expected: Vec3 }[] = [];
  try {
    const records = new Float32Array(8 * DIFFUSE_GI_PROBE_FLOATS);
    const state = new Uint32Array(records.buffer);
    const c0 = 4 * Math.PI * 0.28209479;
    for (let probe = 0; probe < 8; probe++) {
      const base = probe * DIFFUSE_GI_PROBE_FLOATS;
      records.set(sky.map(value => c0 * value), base);
      records[base + 3] = sky[0];
      records[base + 7] = sky[1];
      records[base + 11] = sky[2];
      for (let cell = 0; cell < GI_DISTANCE_CELLS; cell++) {
        records.set([10, 100, 1], base + 36 + cell * 4);
      }
      state[base + DIFFUSE_GI_STATE_FLOAT_OFFSET] = 1;
      state[base + DIFFUSE_GI_STATE_FLOAT_OFFSET + 1] = 1;
    }
    const probeBuffer = allocate(8 * DIFFUSE_GI_PROBE_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(probeBuffer, 0, records);
    const parameters = new Float32Array(28);
    parameters.set([-0.5, -0.5, -0.5, 1], 0);
    parameters.set([1, 1, 1, 10], 4);
    new Uint32Array(parameters.buffer).set([2, 2, 2, 64], 8);
    parameters.set([...sky, 0.2], 20);
    new Uint32Array(parameters.buffer)[24] = 1;
    const uniform = allocate(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uniform, 0, parameters);
    const inputs = new Float32Array(queries.length * 8);
    queries.forEach((query, index) => {
      inputs.set([...query.position, 0], index * 8);
      inputs.set([...query.normal, 0], index * 8 + 4);
    });
    const input = allocate(inputs.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const outputBytes = queries.length * 16;
    const output = allocate(outputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(outputBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(input, 0, inputs);
    const source = diffuseGiSampleSource + `
@group(0) @binding(0) var<storage, read> testInputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> testIrradiance: array<vec4f>;
@compute @workgroup_size(64)
fn testUniformIrradiance(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${queries.length}u) { return; }
  let position = testInputs[id.x * 2u].xyz;
  let normal = normalize(testInputs[id.x * 2u + 1u].xyz);
  testIrradiance[id.x] = vec4f(giSampleIrradiance(position, normal, 1u), 0.0);
}`;
    const shader = device.createShaderModule({ code: source });
    const compilation = await shader.getCompilationInfo();
    const compileErrors = compilation.messages.filter(message => message.type === 'error');
    if (compileErrors.length) throw new Error(compileErrors.map(error => error.message).join('\n'));
    const ioLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });
    const emptyLayout = device.createBindGroupLayout({ entries: [] });
    const giLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ] });
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [ioLayout, emptyLayout, emptyLayout, giLayout] }),
      compute: { module: shader, entryPoint: 'testUniformIrradiance' },
    });
    const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
    ] });
    const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
    const giGroup = device.createBindGroup({ layout: giLayout, entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: probeBuffer } },
    ] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, ioGroup);
    pass.setBindGroup(1, emptyGroup); pass.setBindGroup(2, emptyGroup); pass.setBindGroup(3, giGroup);
    pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    let values: Float32Array;
    try { values = new Float32Array(readback.getMappedRange().slice(0)); }
    finally { readback.unmap(); }
    samples = queries.map((query, index) => ({ ...query,
      actual: Array.from(values.slice(index * 4, index * 4 + 3)),
      expected: sky.map(value => value * Math.PI) as Vec3,
    }));
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally {
      for (const buffer of buffers) { buffer.destroy(); ownedBytes -= buffer.size; }
      device.destroy();
    }
  }
  return { samples, uncaptured, validationError, ownedBytes };
}

export async function runGiVisibilityMomentsFixture() {
  const element = canvas();
  let renderer: Renderer | undefined;
  try {
    const box = cube('closed-black-box', 1, [0, 0, 0], [6.6, 7.6, 8.4],
      { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
        baseColor: m.constant([0, 0, 0, 1]), emissive: m.constant([0, 0, 0, 1]) } });
    renderer = await Renderer.create(element, [box], { worldQueries: {
      maxRays: 512, maxTrianglesPerMesh: 12, maxInstancesPerBatch: 1,
    }, diffuseGi: boxGrid });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    renderer.render([0, 0, 0], [0, 0, -1]);
    const actual = await renderer.probeDiffuseGi();
    const expected = expectedBoxMoments();
    const helper = await sampleProductionHelper(actual.probes[0].moments);
    const errors = [...renderer.errors];
    renderer.dispose();
    return { actual, expected, helper, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

const floorGrid = { origin: [-0.6, 0.4, -0.6] as Vec3,
  spacing: [1.2, 1.2, 1.2] as Vec3, counts: [2, 2, 2] as Vec3,
  raysPerProbe: 64, maxDistance: 10, skyRadiance: [0, 0, 0] as Vec3 };
const floorEye: Vec3 = [-0.25, 2.8, 2];
const floorTarget: Vec3 = [-0.25, 0, 0];

export async function runGiVisibilityLeakFixture() {
  const element = canvas();
  let renderer: Renderer | undefined;
  try {
    const floor = cube('receiver-floor', 10, [0, -0.05, 0], [8, 0.1, 8],
      { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
        baseColor: m.constant([1, 1, 1, 1]) } });
    const wall = cube('solid-partition', 11, [0, 1.4, 0], [0.04, 3.2, 6],
      { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
        baseColor: m.constant([0, 0, 0, 1]) } });
    const wallInstance = wall.instances[0];
    wall.instances = [];
    const emitter = cube('bright-ceiling', 12, [1, 2, 0], [1.6, 0.02, 2],
      { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
        baseColor: m.constant([0, 0, 0, 1]), emissive: m.constant([20, 20, 20, 1]) } });
    renderer = await Renderer.create(element, [floor, wall, emitter], {
      worldQueries: { maxRays: 512, maxTrianglesPerMesh: 12, maxInstancesPerBatch: 1 },
      diffuseGi: floorGrid,
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    renderer.invalidateDiffuseGiHistory();
    renderer.render(floorEye, floorTarget);
    const center = { x: Math.floor(renderer.stats.width / 2), y: Math.floor(renderer.stats.height / 2) };
    const noWallProbe = await renderer.probeDiffuseGi();
    const noWallHdr = (await renderer.probeHdr([center]))[0];
    const sightLength = Math.hypot(0, -2.8, -2);
    const floorRay = (await renderer.traceRays([{
      origin: floorEye, direction: [0, -2.8 / sightLength, -2 / sightLength],
    }])).hits[0];
    wall.instances = [wallInstance];
    renderer.invalidateDiffuseGiHistory();
    renderer.render(floorEye, floorTarget);
    const withWallProbe = await renderer.probeDiffuseGi();
    const withWallHdr = (await renderer.probeHdr([center]))[0];
    const wallRay = (await renderer.traceRays([{
      origin: [-0.25, 1, 0], direction: [1, 0, 0],
    }])).hits[0];
    const errors = [...renderer.errors];
    renderer.dispose();
    return { noWallProbe, withWallProbe, noWallHdr, withWallHdr,
      floorRay, wallRay, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}
