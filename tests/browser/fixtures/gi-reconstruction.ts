import { DIFFUSE_GI_PROBE_BYTES, DIFFUSE_GI_PROBE_FLOATS, DIFFUSE_GI_STATE_FLOAT_OFFSET } from '../../../src/diffuse-gi';
import { GI_DISTANCE_CELLS } from '../../../src/gi-visibility';
import { diffuseGiSampleSource } from '../../../src/gpu-diffuse-gi';
import { FINAL_GATHER_ANCHOR_BYTES, FINAL_GATHER_PARAMETER_BYTES, finalGatherSampleSource } from '../../../src/gpu-final-gather';
import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';

type Vec3 = [number, number, number];
interface Query { position: Vec3; normal: Vec3; pixel?: [number, number]; }
interface Field { radiance: Vec3; blocked: (index: number) => boolean;
  invalid?: (index: number) => boolean; stale?: (index: number) => boolean;
  blockedMoments?: [number, number]; }
interface Anchor { position: Vec3; normal: Vec3; geometry?: [number, number]; irradiance: Vec3; frame?: number;
  identity?: [number, number]; }
interface Diagnostic { total: number; branch: number; primaryAccepted: number; ringAccepted: number;
  supplementalAccepted: number; rejections: number[]; }

/** Known inputs are supplied directly to the production reconstruction shaders.
 * This isolates interpolation from triangle traversal, ray sampling and textures. */
export async function runGiReconstructionFixture() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('GI_RECONSTRUCTION_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const buffers: GPUBuffer[] = [], errors: string[] = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  let ownedBytes = 0, validationError = '';
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer); ownedBytes += size; return buffer;
  };
  device.pushErrorScope('validation');
  const results: { label: string; actual: number[]; expected?: Vec3; upperFraction?: number }[] = [];
  const diagnostics: Diagnostic[] = [];
  try {
    const giParameters = allocate(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const probes = allocate(8 * DIFFUSE_GI_PROBE_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const gatherParameters = allocate(FINAL_GATHER_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const anchors = allocate(20 * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const coverageMap = allocate(4 * 9 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const input = allocate(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const outputBytes = 112;
    const output = allocate(outputBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(outputBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const source = surfaceFrameSource + diffuseGiSampleSource + finalGatherSampleSource + `
struct ReconstructionQuery { pixel: vec4f, position: vec4f, normal: vec4f, geometry: vec4f }
struct ReconstructionOutput {
  irradiance: vec4f, measures: vec4f, state: vec4u, rejections: array<u32, 16>,
}
@group(0) @binding(0) var<storage, read> query: ReconstructionQuery;
@group(0) @binding(1) var<storage, read_write> answer: ReconstructionOutput;
@compute @workgroup_size(1)
fn reconstruct() {
  answer.measures = vec4f(0.0);
  answer.state = vec4u(0u);
  for (var slot = 0u; slot < 16u; slot++) { answer.rejections[slot] = 128u; }
  if (query.pixel.z == 0.0) {
    answer.irradiance = vec4f(diffuseGiSurfaceIrradiance(query.position.xyz, query.normal.xyz, query.geometry.xyz), 1.0);
  } else {
    let result = finalGatherReconstruct(query.pixel.xy, query.position.xyz, query.normal.xyz, query.geometry.xyz,
      vec2u(1u, 1u));
    answer.irradiance = vec4f(result.irradiance, 1.0);
    answer.measures.x = result.total;
    answer.state = vec4u(result.branch, result.primaryAccepted, result.ringAccepted, result.supplementalAccepted);
    for (var slot = 0u; slot < 16u; slot++) { answer.rejections[slot] = result.rejections[slot]; }
  }
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
    const lightingLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ] });
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [ioLayout, emptyLayout, emptyLayout, lightingLayout] }),
      compute: { module: shader, entryPoint: 'reconstruct' },
    });
    const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
    ] });
    const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
    const lightingGroup = device.createBindGroup({ layout: lightingLayout, entries: [
      { binding: 0, resource: { buffer: giParameters } }, { binding: 1, resource: { buffer: probes } },
      { binding: 2, resource: { buffer: gatherParameters } }, { binding: 3, resource: { buffer: anchors } },
      { binding: 6, resource: { buffer: coverageMap } },
    ] });
    const gi = new Float32Array(28), giIntegers = new Uint32Array(gi.buffer);
    gi.set([-0.5, -0.5, -0.5, 1], 0); gi.set([1, 1, 1, 10], 4);
    giIntegers.set([2, 2, 2, 64], 8); gi[23] = 0.2; giIntegers[24] = 1;
    device.queue.writeBuffer(giParameters, 0, gi);

    const dispatch = async (field: Field, query: Query, anchorData?: Anchor[], viewport = 16, extra?: Anchor[],
      mappedIndices?: readonly number[]): Promise<number[]> => {
      const records = new Float32Array(8 * DIFFUSE_GI_PROBE_FLOATS), words = new Uint32Array(records.buffer);
      for (let index = 0; index < 8; index++) {
        const base = index * DIFFUSE_GI_PROBE_FLOATS;
        records.set(field.radiance.map(value => value * 4 * Math.PI * 0.28209479), base);
        records[base + 3] = field.radiance[0]; records[base + 7] = field.radiance[1]; records[base + 11] = field.radiance[2];
        for (let cell = 0; cell < GI_DISTANCE_CELLS; cell++) {
          // Blocked samples have mean .4 and variance .04; the query is farther
          // than the mean. The independent test bounds the Cantelli confidence.
          records.set(field.blocked(index) ? [...(field.blockedMoments ?? [0.4, 0.2]), 1]
            : [10, 100, 1], base + 36 + cell * 4);
        }
        words[base + DIFFUSE_GI_STATE_FLOAT_OFFSET] = field.invalid?.(index) ? 0 : 1;
        words[base + DIFFUSE_GI_STATE_FLOAT_OFFSET + 1] = field.stale?.(index) ? 0 : 1;
      }
      device.queue.writeBuffer(probes, 0, records);
      const parameters = new Float32Array(FINAL_GATHER_PARAMETER_BYTES / 4), integers = new Uint32Array(parameters.buffer);
      integers.set([viewport, viewport, 4, 4], 36); integers.set([4, 16 + (extra?.length ?? 0), 1, 64], 40);
      integers.set([1, 1, 0, 0], 44); parameters.set([0.0001, 0, 0.001, 0.25], 48);
      if (extra) {
        integers.set([1, 0, 8, extra.length], 72);
        integers.set([4, 4, 1, 0], 76);
      }
      device.queue.writeBuffer(gatherParameters, 0, parameters);
      const map = new Uint32Array(4 * 9);
      if (mappedIndices) map.set(mappedIndices);
      else extra?.forEach((_, tile) => { map[tile * 8] = 16 + tile + 1; });
      device.queue.writeBuffer(coverageMap, 0, map);
      const anchorFloats = FINAL_GATHER_ANCHOR_BYTES / Float32Array.BYTES_PER_ELEMENT;
      const packedAnchors = new Float32Array(20 * anchorFloats), anchorWords = new Uint32Array(packedAnchors.buffer);
      [...(anchorData ?? []), ...(extra ?? [])].forEach((anchor, index) => {
        const base = index * anchorFloats, geometry = anchor.geometry ?? [0, 1];
        packedAnchors.set([...anchor.position, geometry[0]], base);
        packedAnchors.set([...anchor.normal, geometry[1]], base + 4);
        anchorWords.set([anchor.identity?.[0] ?? 1, anchor.identity?.[1] ?? 1, 1, index], base + 8);
        anchorWords.set([1, 1, 1, 0], base + 12);
        packedAnchors.set([...anchor.irradiance, 64], base + 16);
        anchorWords.set([1, anchor.frame ?? 1, 0, 0], base + 20);
      });
      device.queue.writeBuffer(anchors, 0, packedAnchors);
      device.queue.writeBuffer(input, 0, new Float32Array([
        ...(query.pixel ?? [0, 0]), anchorData ? 1 : 0, 0,
        ...query.position, 0, ...query.normal, 0, ...query.normal, 0,
      ]));
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, ioGroup);
      pass.setBindGroup(1, emptyGroup); pass.setBindGroup(2, emptyGroup); pass.setBindGroup(3, lightingGroup);
      pass.dispatchWorkgroups(1); pass.end(); encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      try {
        const raw = readback.getMappedRange();
        const floats = new Float32Array(raw), words = new Uint32Array(raw);
        diagnostics.push({ total: floats[4], branch: words[8], primaryAccepted: words[9], ringAccepted: words[10], supplementalAccepted: words[11],
          rejections: Array.from(words.slice(12, 28)) });
        return Array.from(floats.slice(0, 3));
      }
      finally { readback.unmap(); }
    };
    const sky: Vec3 = [0.1, 0.2, 0.3], expected = sky.map(value => Math.PI * value) as Vec3;
    const center: Query = { position: [0, 0, 0], normal: [0, 1, 0] };
    const open: Field = { radiance: sky, blocked: () => false };
    const blocked: Field = { radiance: sky, blocked: () => true };
    const root = 1 / Math.sqrt(3);
    for (const [label, query] of [['visible-center', center], ['visible-grazing-fringe',
      { position: [1, 1, 1], normal: [root, root, root] }]] as [string, Query][]) {
      results.push({ label, actual: await dispatch(open, query), expected });
    }
    results.push({ label: 'all-probes-occluded', actual: await dispatch(blocked, center), expected, upperFraction: 0.005 });
    results.push({ label: 'one-visible-probe', actual: await dispatch({ radiance: sky, blocked: index => index !== 0 }, center), expected });
    results.push({ label: 'invalid-visible-probe', actual: await dispatch({ radiance: sky,
      blocked: index => index !== 0, invalid: index => index === 0 }, center), expected, upperFraction: 0.005 });
    results.push({ label: 'stale-visible-probe', actual: await dispatch({ radiance: sky,
      blocked: index => index !== 0, stale: index => index === 0 }, center), expected, upperFraction: 0.005 });
    results.push({ label: 'zero-weight-visible-probes', actual: await dispatch({ radiance: sky,
      blocked: index => (index & 1) === 0 }, { ...center, position: [-0.5, 0, 0] }), expected, upperFraction: 0.03 });
    results.push({ label: 'near-zero-weight-visible-probes', actual: await dispatch({ radiance: sky,
      blocked: index => (index & 1) === 0 }, { ...center, position: [-0.499999, 0, 0] }), expected, upperFraction: 0.03 });
    // At the cell center all eight probes are equally far away. Construct the
    // independent Cantelli variance for a known cubed confidence. These are
    // confidence fields, not partially transmitting physical walls.
    const confidenceMoments = (support: number): [number, number] => {
      const visibility = Math.cbrt(support), beyond = Math.sqrt(0.75) - 0.4;
      const variance = visibility * beyond * beyond / (1 - visibility);
      return [0.4, 0.16 + variance];
    };
    for (const [label, support] of [['medium-confidence-uniform', 0.25], ['high-confidence-uniform', 0.8],
      ['threshold-below', 0.2 * (1 - 1e-6)], ['threshold-above', 0.2 * (1 + 1e-6)]] as [string, number][]) {
      const supportWeight = Math.min(1, support / 0.2) ** 2;
      results.push({ label, actual: await dispatch({ ...blocked, blockedMoments: confidenceMoments(support) }, center),
        expected: expected.map(value => value * supportWeight) as Vec3 });
    }
    const recovered: Vec3 = [0.2, 0.4, 0.6];
    const gatherQuery: Query = { ...center, pixel: [8.5, 8.5] };
    const makeAnchors = (): Anchor[] => Array.from({ length: 16 }, (_, index) => ({
      position: [((index % 4) - 1.5) * 0.04, 0, (Math.floor(index / 4) - 1.5) * 0.04],
      normal: [1, 0, 0], irradiance: [10, 10, 10],
    }));
    const ring = makeAnchors(); ring[4].normal = [0, 1, 0]; ring[4].irradiance = recovered;
    results.push({ label: 'matching-normal-outside-primary-cell', actual: await dispatch(open, gatherQuery, ring), expected: recovered });
    const thinWall = makeAnchors(); thinWall[4] = { ...ring[4], position: [-0.06, 0.004, -0.02] };
    results.push({ label: 'thin-parallel-wall-rejected', actual: await dispatch(open, gatherQuery, thinWall), expected });
    const curved = makeAnchors(); curved[4] = { ...ring[4], geometry: [0.3660254, 0.6339746] };
    results.push({ label: 'different-geometric-plane-rejected', actual: await dispatch(open, gatherQuery, curved), expected });
    const stale = makeAnchors(); stale[4] = { ...ring[4], frame: 0 };
    results.push({ label: 'stale-ring-anchor-rejected', actual: await dispatch(open, gatherQuery, stale), expected });
    const far = makeAnchors(); far[4] = { ...ring[4], position: [-0.3, 0, 0] };
    results.push({ label: 'distant-ring-anchor-rejected', actual: await dispatch(open, gatherQuery, far), expected });
    const otherIdentity = makeAnchors(); otherIdentity[4] = { ...ring[4], identity: [1, 2] };
    results.push({ label: 'different-entity-ring-anchor-rejected', actual: await dispatch(open, gatherQuery, otherIdentity), expected });
    const primary = makeAnchors(); primary.forEach(anchor => { anchor.normal = [0, 1, 0]; anchor.irradiance = recovered; });
    primary[4].irradiance = [40, 0, 0];
    results.push({ label: 'covered-cell-does-not-use-ring', actual: await dispatch(open, gatherQuery, primary), expected: recovered });
    const partial = makeAnchors(); partial.forEach(anchor => { anchor.normal = [0, 1, 0]; anchor.irradiance = [10, 10, 10]; });
    partial[15].irradiance = recovered;
    results.push({ label: 'partial-last-cell-position', actual: await dispatch(open, { ...center, pixel: [13.5, 13.5] }, partial, 14), expected: recovered });
    results.push({ label: 'edge-out-of-bounds-order', actual: await dispatch(open,
      { ...center, pixel: [0.5, 0.5] }, makeAnchors()), expected });
    const extra = makeAnchors().slice(0, 4);
    extra[3] = { position: [0, 0, 0], normal: [0, 1, 0], irradiance: recovered };
    results.push({ label: 'supplemental-compatible', actual: await dispatch(open, gatherQuery, makeAnchors(), 16, extra), expected: recovered });
    const lastRank = Array<number>(32).fill(0); lastRank[3 * 8 + 7] = 20;
    results.push({ label: 'supplemental-last-rank', actual: await dispatch(open, gatherQuery, makeAnchors(), 16, extra, lastRank), expected: recovered });
    results.push({ label: 'supplemental-unmapped-ignored', actual: await dispatch(open, gatherQuery, makeAnchors(), 16, extra, []), expected });
    const paired = extra.map(anchor => ({ ...anchor }));
    paired[0] = { position: [0, 0, 0], normal: [0, 1, 0], irradiance: [0.6, 0.2, 0] };
    const sameTile = [...lastRank]; sameTile[3 * 8] = 17;
    results.push({ label: 'supplemental-two-ranks-same-tile', actual: await dispatch(open, gatherQuery, makeAnchors(), 16, paired, sameTile),
      expected: [0.4, 0.3, 0.3] });
    for (const [label, change] of [
      ['supplemental-four-mm-wall', { position: [0, 0.004, 0] }],
      ['supplemental-other-entity', { identity: [1, 2] }],
      ['supplemental-stale', { frame: 0 }],
    ] as [string, Partial<Anchor>][]) {
      const changed = extra.map((a, i) => i === 3 ? { ...a, ...change } : a);
      results.push({ label, actual: await dispatch(open, gatherQuery, makeAnchors(), 16, changed), expected });
    }
    extra[3].irradiance = [50, 0, 0];
    results.push({ label: 'primary-preserved-with-bright-supplemental', actual: await dispatch(open, gatherQuery, primary, 16, extra), expected: recovered });
  } finally {
    try { validationError = String((await device.popErrorScope())?.message ?? ''); }
    finally { for (const buffer of buffers) { buffer.destroy(); ownedBytes -= buffer.size; } device.destroy(); }
  }
  return { results: results.map((sample, index) => ({ ...sample, diagnostic: diagnostics[index] })),
    errors, validationError, ownedBytes };
}
