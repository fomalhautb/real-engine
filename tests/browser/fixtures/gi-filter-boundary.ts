import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { FINAL_GATHER_ANCHOR_BYTES, FINAL_GATHER_PARAMETER_BYTES,
  type FinalGatherProbeResult } from '../../../src/gpu-final-gather';
import { GI_DISTANCE_CELLS, GI_DISTANCE_RESOLUTION, octaEncode } from '../../../src/gi-visibility';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import commonSource from '../../../src/shaders/diffuse-gi-common.wgsl?raw';
import interpolateSource from '../../../src/shaders/diffuse-gi-interpolate.wgsl?raw';
import updateSource from '../../../src/shaders/final-gather-update.wgsl?raw';
import surfaceFrameSource from '../../../src/shaders/surface-frame.wgsl?raw';

type Vec3 = [number, number, number];
type GatherAnchor = FinalGatherProbeResult['anchors'][number];
const sky: Vec3 = [0.1, 0.2, 0.3];
const roofHeight = 0.03, normalBias = 0.0001, maxDistance = 12;
const width = 128, spacing = 4, columns = width / spacing;
const frameCount = 256, raysPerAnchor = 16;

/** Cosine-weighted sky visibility below an infinite black roof at y=h, x>=0.
 * The receiver has normal +Y. Its irradiance is pi * sky * this fraction. */
function skyFraction(x: number, h: number): number {
  return 0.5 * (1 - x / Math.hypot(x, h));
}

interface ProfilePoint { x: number; expected: number; raw: number; filtered: number; }
function mean(values: number[]): number { return values.reduce((sum, x) => sum + x, 0) / values.length; }
function crossing(profile: ProfilePoint[], key: 'expected' | 'raw' | 'filtered', level: number): number | null {
  const candidates: number[] = [];
  for (let i = 1; i < profile.length; i++) {
    const a = profile[i - 1], b = profile[i];
    if (a[key] >= level && b[key] < level) {
      candidates.push(a.x + (b.x - a.x) * (a[key] - level) / (a[key] - b[key]));
    }
  }
  // Monte Carlo noise can produce more than one crossing: report ambiguity
  // instead of silently choosing the crossing which makes a filter look best.
  return candidates.length === 1 ? candidates[0] : null;
}
function metrics(points: ProfilePoint[]) {
  const measure = (key: 'raw' | 'filtered') => ({
    rmse: Math.sqrt(mean(points.map(p => (p[key] - p.expected) ** 2))),
    meanSignedError: mean(points.map(p => p[key] - p.expected)),
    maxAbsoluteError: Math.max(...points.map(p => Math.abs(p[key] - p.expected))),
    darkSideSignedGain: mean(points.filter(p => p.x > 0).map(p => p[key] - p.expected)),
    brightSideSignedLoss: mean(points.filter(p => p.x < 0).map(p => p.expected - p[key])),
  });
  return { raw: measure('raw'), filtered: measure('filtered') };
}
function profileWidths(profile: ProfilePoint[]) {
  return Object.fromEntries((['expected', 'raw', 'filtered'] as const).map(key => {
    const x90 = crossing(profile, key, 0.9), x10 = crossing(profile, key, 0.1);
    return [key, { x90, x10, width: x90 !== null && x10 !== null ? x10 - x90 : null }];
  }));
}
function instance(id: number, x: number, y: number, sizeX: number, sizeZ: number): InstanceState {
  const model = multiply(translation(x, y, 0), scaling(sizeX, 1, sizeZ));
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}
function batch(name: string, id: number, state: InstanceState, black = false): SceneBatch {
  return { name, mesh: createPlane(), capacity: 1, inputMode: 'committed', instances: [state],
    queryIdentity: { assetId: 'gi-filter-boundary/' + name, primitiveId: id, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: black ? 'unlit' : 'pbr',
      doubleSided: true, baseColor: m.constant(black ? [0, 0, 0, 1] : [0.5, 0.5, 0.5, 1]),
      emissive: m.constant([0, 0, 0, 1]), metallic: 0, roughness: 1 } } };
}
function isReceiver(anchor: GatherAnchor): boolean {
  return anchor.valid && anchor.identity[1] === 1 && Math.abs(anchor.position[1]) < 1e-5
    && anchor.geometricNormal[1] > 0.9999 && anchor.normal[1] > 0.9999;
}

/** Full production raster receivers, triangle traversal, cosine sampling,
 * raw history and denoise. Every render call is explicit. */
export async function runGiFilterBoundaryFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;';
  document.body.appendChild(canvas);
  const receiver = batch('floor', 1, instance(1, 0, 0, 80, 80));
  // createPlane spans [-.5,.5]: this roof is x=[0,40], z=[-40,40].
  const roofState = instance(2, 20, roofHeight, 40, 80);
  const roof = batch('black-half-roof', 2, roofState, true);
  roof.instances = [];
  const eye: Vec3 = [0, 0.025, 0.5], target: Vec3 = [0, 0, 0.25];
  const indices = Array.from({ length: columns * columns }, (_, index) => index);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [receiver, roof], {
      worldQueries: { maxRays: indices.length * raysPerAnchor },
      diffuseGi: { origin: [-0.5, 0.1, -0.5], spacing: [1, 1, 1], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance, skyRadiance: sky, multiBounce: false,
        finalGather: { receiver: 'raster', pixelSpacing: spacing, maxAnchors: indices.length, raysPerAnchor,
          maxHistory: frameCount, normalBias, pathSegments: 1 } },
    });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    for (let frame = 0; frame < 2; frame++) renderer.render(eye, target);
    const constant = await renderer.probeDiffuseGiFinalGather(indices);
    roof.instances = [roofState];
    renderer.invalidateDiffuseGiHistory();
    const firstSamplingFrame = renderer.stats.frames + 1;
    for (let frame = 0; frame < frameCount; frame++) renderer.render(eye, target);
    const boundary = await renderer.probeDiffuseGiFinalGather(indices);
    const selected = boundary.anchors.filter(anchor => isReceiver(anchor)
      && Math.abs(anchor.position[0]) <= 0.16 && Math.abs(anchor.position[2]) <= 0.5);
    const samples = selected.map(anchor => {
      const h = roofHeight - anchor.position[1] - normalBias;
      const expectedFraction = skyFraction(anchor.position[0], h);
      // Any hit within R is inside the finite roof's far/side edges. Only
      // grazing directions whose infinite-plane intersection is beyond R can
      // differ. Their cosine-weighted measure is at most (h/R)^2. Thus the
      // finite-range estimator lies in [analytic, analytic + tail], in mean.
      const finiteRangeTailFraction = (h / maxDistance) ** 2;
      return { ...anchor, x: anchor.position[0], expected: expectedFraction,
        expectedIrradiance: sky.map(value => Math.PI * value * expectedFraction),
        finiteRangeTailFraction,
        farEdgeDistance: Math.min(40 - anchor.position[0], 40 - Math.abs(anchor.position[2])),
        raw: anchor.irradiance[0] / (Math.PI * sky[0]),
        filtered: anchor.filteredIrradiance[0] / (Math.PI * sky[0]) };
    });
    const rows = [...new Set(samples.map(anchor => Math.floor(anchor.index / columns)))];
    const profileRow = rows.sort((a, b) => {
      const distance = (row: number) => Math.abs(mean(samples.filter(p => Math.floor(p.index / columns) === row)
        .map(p => p.position[2])) - target[2]);
      return distance(a) - distance(b);
    })[0];
    const profile = samples.filter(p => Math.floor(p.index / columns) === profileRow).sort((a, b) => a.x - b.x);
    const diagnostics = renderer.diagnostics().diffuseGi?.finalGather;
    const errors = [...renderer.errors];
    const size = { width: renderer.stats.width, height: renderer.stats.height };
    renderer.dispose();
    return { sky, roofHeight, normalBias, maxDistance, eye, target, size,
      schedule: { constantFrames: 2, firstSamplingFrame, lastSamplingFrame: boundary.frame,
        frameCount, raysPerAnchor, samplesPerAnchor: frameCount * raysPerAnchor },
      method: 'production raster-receiver single-bounce final gather; analytic infinite half-plane with bounded finite-range tail',
      constant: { frame: constant.frame, anchors: constant.anchors.filter(isReceiver) },
      samples, metrics: metrics(samples), profileRow, profile, widths: profileWidths(profile),
      diagnostics, errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Deterministic denoise isolation. Inputs are exact analytic irradiance,
 * not synthetic renderer screenshots. This proves filter behavior only;
 * the separate Renderer fixture above covers actual light transport. */
export async function runGiFilterBoundaryDenoiseFixture() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('GI_FILTER_BOUNDARY_NO_ADAPTER');
  const device = await adapter.requestDevice();
  const buffers: GPUBuffer[] = [], errors: string[] = [];
  let ownedBytes = 0, validationError = '';
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  device.pushErrorScope('validation');
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size, usage });
    buffers.push(buffer); ownedBytes += size; return buffer;
  };
  const gridColumns = 64, gridRows = 5, count = gridColumns * gridRows;
  const gridStep = 0.01, h = 0.01, wallSeparation = 0.004;
  const anchorWords = FINAL_GATHER_ANCHOR_BYTES / Float32Array.BYTES_PER_ELEMENT;
  if (anchorWords !== 28) throw new Error('GI_FILTER_BOUNDARY_MOMENT_LAYOUT');
  // moments.x stores population variance of individual frame estimates. For
  // N=16, v=(N-1)*0.08^2 gives a known standard error of the mean of 0.08.
  const noiseAmplitude = 0.08, noiseHistory = 16;
  const noiseVariance = (noiseHistory - 1) * noiseAmplitude ** 2;
  const constantValue = 0.375, startupIndex = Math.floor(gridRows / 2) * gridColumns + gridColumns / 2;
  try {
    const parameters = allocate(FINAL_GATHER_PARAMETER_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const raw = allocate(count * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const filtered = allocate(count * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = allocate(count * FINAL_GATHER_ANCHOR_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const common = commonSource.replace('// GI_DISTANCE_CONSTANTS',
      `const GI_DISTANCE_RESOLUTION: u32 = ${GI_DISTANCE_RESOLUTION}u;\nconst GI_DISTANCE_CELLS: u32 = ${GI_DISTANCE_CELLS}u;`);
    const module = device.createShaderModule({ label: 'analytic boundary: production denoise',
      code: surfaceFrameSource + '\n' + updateSource.replace('// GI_COMMON', common).replace('// GI_INTERPOLATE', interpolateSource) });
    const compileErrors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (compileErrors.length) throw new Error(compileErrors.map(error => error.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'denoise' } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: parameters } }, { binding: 6, resource: { buffer: filtered } },
      { binding: 7, resource: { buffer: raw } },
    ] });
    const params = new Uint32Array(FINAL_GATHER_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT);
    params.set([gridColumns * 4, gridRows * 4, gridColumns, gridRows], 36);
    params.set([4, count, 1, 256], 40); params.set([1, 1, 0, 0], 44);
    device.queue.writeBuffer(parameters, 0, params);
    const encodedNormal = octaEncode([0, 1, 0]);
    const dispatch = async (mode: 'half-plane' | 'constant' | 'four-mm-wall' | 'noisy-constant'
      | 'startup-zero' | 'startup-recovered') => {
      const records = new Float32Array(count * anchorWords), words = new Uint32Array(records.buffer);
      const reference: ProfilePoint[] = [];
      for (let index = 0; index < count; index++) {
        const column = index % gridColumns, row = Math.floor(index / gridColumns);
        const x = (column - (gridColumns - 1) / 2) * gridStep;
        const expected = mode === 'half-plane' ? skyFraction(x, h)
          : mode === 'four-mm-wall' ? Number(x >= 0) : constantValue;
        const noise = mode === 'noisy-constant' ? (column + row) % 2 === 0 ? noiseAmplitude : -noiseAmplitude : 0;
        const value = mode === 'startup-zero' && index === startupIndex ? 0 : expected + noise;
        const offset = index * anchorWords;
        // Identical scene identity and material normal deliberately prevent
        // identity rejection from hiding a broken geometric plane check.
        records.set([x, mode === 'four-mm-wall' && x >= 0 ? wallSeparation : 0,
          (row - 2) * gridStep, encodedNormal[0]], offset);
        records.set([0, 1, 0, encodedNormal[1]], offset + 4);
        words.set([1, 1, 1, 0], offset + 8); words.set([1, 0, 1, 1], offset + 12);
        records.set([value, value, value,
          mode === 'startup-zero' ? 1 : mode === 'noisy-constant' ? noiseHistory : 256], offset + 16);
        words.set([1, 1, 0, 0], offset + 20);
        records.set([mode === 'noisy-constant' ? noiseVariance : 0, 0, 0, 0], offset + 24);
        reference.push({ x, expected, raw: records[offset + 16], filtered: 0 });
      }
      device.queue.writeBuffer(raw, 0, records);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
      encoder.copyBufferToBuffer(filtered, 0, readback, 0, count * FINAL_GATHER_ANCHOR_BYTES);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
      const samples = reference.map((point, index) => ({ ...point, index,
        filtered: values[index * anchorWords + 16],
        rgb: Array.from(values.slice(index * anchorWords + 16, index * anchorWords + 19)) }));
      const profile = samples.slice(gridColumns * 2, gridColumns * 3);
      return { mode, samples, profile, metrics: metrics(samples), widths: profileWidths(profile) };
    };
    const halfPlane = await dispatch('half-plane');
    const constant = await dispatch('constant');
    const thinWall = await dispatch('four-mm-wall');
    const noisyConstant = await dispatch('noisy-constant');
    const startupZero = await dispatch('startup-zero');
    const startupRecovered = await dispatch('startup-recovered');
    validationError = (await device.popErrorScope())?.message ?? '';
    for (const buffer of buffers) buffer.destroy(); ownedBytes = 0;
    return { method: 'production denoise entry point, exact analytic and deterministic-noise inputs; no trace or temporal-history claim',
      gridColumns, gridRows, gridStep, h, wallSeparation, noiseAmplitude, noiseHistory, noiseVariance, constantValue,
      startupIndex, halfPlane, constant, thinWall, noisyConstant, startupZero, startupRecovered,
      errors, validationError, ownedBytes };
  } finally {
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}
