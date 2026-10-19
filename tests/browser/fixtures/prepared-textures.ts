import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import { textureStorageLayout, type PreparedTextureAsset } from '../../../src/texture-storage';
import type { TextureStreamSource } from '../../../src/texture-stream-types';
import { compressedSolidBlockCases, type Rgba8 } from './compressed-texture-blocks';

type V3 = [number, number, number];
const COLORS: Rgba8[] = [[192, 32, 16, 255], [32, 192, 16, 255], [16, 32, 192, 255], [128, 64, 32, 255]];
const EYE: V3 = [0, 0, 4], TARGET: V3 = [0, 0, 0];

function rgbaTexture(width: number, height: number, baseMipLevel: number, colors = COLORS): PreparedTextureAsset {
  const levels = [];
  for (let mip = baseMipLevel; mip <= Math.floor(Math.log2(Math.max(width, height))); mip++) {
    const w = Math.max(1, Math.floor(width / 2 ** mip)), h = Math.max(1, Math.floor(height / 2 ** mip));
    const data = new Uint8Array(w * h * 4);
    for (let pixel = 0; pixel < w * h; pixel++) data.set(colors[mip], pixel * 4);
    levels.push({ width: w, height: h, data });
  }
  return { kind: 'prepared', format: 'rgba8', width, height, baseMipLevel, colorSpace: 'linear', levels };
}

function blockTexture(format: PreparedTextureAsset['format'], block: Uint8Array): PreparedTextureAsset {
  return { kind: 'prepared', format, width: 8, height: 8, baseMipLevel: 1, colorSpace: 'linear',
    levels: [4, 2, 1].map(size => ({ width: size, height: size, data: block })) };
}

function quad(uvSpan = 1): MeshData {
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]].forEach((point, i) => {
    point.forEach((value, axis) => view.setFloat32(i * 80 + axis * 4, value, true));
    view.setFloat32(i * 80 + 12, 1, true); view.setFloat32(i * 80 + 24, 1, true);
    view.setFloat32(i * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, uvSpan, 0, uvSpan, 0, 0, 0]),
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function instance(entityId: number, model = identity()): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

function binding(texture: PreparedTextureAsset, sampler?: GPUSamplerDescriptor): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
    alphaMode: 'mask', alphaCutoff: 0.5, baseColor: m.multiply(m.texture(0), m.parameter(0)) },
    parameters: new Float32Array([1, 1, 1, 1]), textures: [texture], samplers: [sampler] };
}

function batch(name: string, mesh: MeshData, material: MaterialBinding): SceneBatch {
  return { name, mesh, capacity: 1, inputMode: 'committed', instances: [instance(1)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material };
}

function canvasOf(size: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${size}px;height:${size}px;z-index:10000`;
  document.body.append(canvas);
  return canvas;
}

/** Uses production surface, shadow and world-query shaders, including implicit NPOT LOD. */
export async function runPreparedTextureSurfaceFixture() {
  const canvas = canvasOf(128), scene = batch('prepared-surface', quad(), binding(rgbaTexture(8, 8, 0)));
  let renderer: Renderer | undefined, npotRenderer: Renderer | undefined;
  let npotCanvas: HTMLCanvasElement | undefined;
  try {
    renderer = await Renderer.create(canvas, [scene], { antiAliasing: 'none', textureBudgetBytes: 424,
      worldQueries: { maxRays: 8 },
      sunShadow: { resolution: 256, focusBounds: { min: [-2, -2, -1], max: [2, 2, 1] } } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async (name: string, lod = 0) => {
      active.render(EYE, TARGET);
      const png = canvas.toDataURL('image/png');
      const [hdr, depth, query] = await Promise.all([
        active.probeHdr([{ x: 64, y: 64 }]), active.probeDepth([[0, 0, 0]]),
        active.traceRays([{ origin: EYE, direction: [0, 0, -1], tMax: 8 }], { textureLod: lod }),
      ]);
      return { name, hdr: hdr[0], depth: depth[0], hit: query.hits[0], png,
        materials: active.diagnostics().materials, revision: active.diagnostics().materialRevision };
    };
    const samples = [await capture('full')];
    await active.setMaterial(scene.name, binding(rgbaTexture(8, 8, 1)));
    samples.push(await capture('tail'));
    // A failed replacement must preserve both the old material and its revision.
    const beforeFailure = active.diagnostics();
    const failures: { name: string; error: string; unchanged: boolean }[] = [];
    const reject = async (name: string, asset: PreparedTextureAsset) => {
      let error = '';
      try { await active.setMaterial(scene.name, binding(asset)); }
      catch (failure) { error = String(failure); }
      const after = active.diagnostics();
      failures.push({ name, error, unchanged: after.materialRevision === beforeFailure.materialRevision
        && JSON.stringify(after.materials) === JSON.stringify(beforeFailure.materials) });
    };
    await reject('budget', { ...rgbaTexture(8, 8, 0), width: 16, height: 16,
      levels: [
        { width: 16, height: 16, data: new Uint8Array(16 * 16 * 4) },
        ...rgbaTexture(8, 8, 0).levels,
      ] });
    for (const sample of compressedSolidBlockCases.filter(value => value.expected[3] > 0.5)) {
      const asset = blockTexture(sample.format, sample.bytes), layout = textureStorageLayout(asset);
      if (!active.info.features.includes(layout.requiredFeature!)) await reject(layout.requiredFeature!, asset);
    }
    samples.push(await capture('tail-after-rejections'));
    samples.push(await capture('tail-explicit-lod2', 2));
    active.setMaterialParameters(scene.name, new Float32Array([0.5, 1, 1, 1]));
    samples.push(await capture('tail-parameter-update'));
    await active.setMaterial(scene.name, binding(rgbaTexture(8, 8, 1), { lodMinClamp: 2, lodMaxClamp: 2 }));
    samples.push(await capture('tail-sampler-lod2'));
    const alphaColors = COLORS.map(color => [color[0], color[1], color[2], 64] as Rgba8);
    await active.setMaterial(scene.name, binding(rgbaTexture(8, 8, 1, alphaColors)));
    samples.push(await capture('tail-mask-hole'));
    const compressed = [];
    for (const sample of compressedSolidBlockCases) {
      const asset = blockTexture(sample.format, sample.bytes), layout = textureStorageLayout(asset);
      if (!active.info.features.includes(layout.requiredFeature!)) continue;
      await active.setMaterial(scene.name, binding(asset));
      compressed.push({ ...await capture(sample.name), expected: sample.expected, tolerance: sample.tolerance });
    }
    // Known dU/dx gives source LOD=1.5. The 9x5 -> 4x2 tail needs distinct
    // derivative scale factors; subtracting baseMip alone in raster is wrong.
    const projectedWidth = 128 / (4 * Math.tan(Math.PI / 6.4));
    const uvSpan = 2 ** 1.5 * projectedWidth / 9;
    const npot = batch('npot-tail', quad(uvSpan), binding(rgbaTexture(9, 5, 0), { lodMinClamp: 1 }));
    npotCanvas = canvasOf(128);
    npotRenderer = await Renderer.create(npotCanvas, [npot], { antiAliasing: 'none', worldQueries: { maxRays: 8 } });
    const readNpot = async () => {
      npotRenderer!.render(EYE, TARGET);
      return { hdr: (await npotRenderer!.probeHdr([{ x: 64, y: 64 }]))[0],
        hit: (await npotRenderer!.traceRays([{ origin: EYE, direction: [0, 0, -1], tMax: 8 }], { textureLod: 1.5 })).hits[0],
        materials: npotRenderer!.diagnostics().materials };
    };
    const npotFull = await readNpot();
    await npotRenderer.setMaterial(npot.name, binding(rgbaTexture(9, 5, 1), { lodMinClamp: 1 }));
    const npotTail = await readNpot();
    const errors = [...active.errors, ...npotRenderer.errors];
    const adapter = active.info.adapterInfo, features = active.info.features;
    active.dispose(); npotRenderer.dispose();
    return { samples, failures, compressed, npotFull, npotTail, adapter, features, errors,
      afterDispose: [active.stats.ownedGpuBytes, npotRenderer.stats.ownedGpuBytes] };
  } finally { renderer?.dispose(); npotRenderer?.dispose(); canvas.remove(); npotCanvas?.remove(); }
}

/** Prepared mip-tail emitters remain present offscreen to both GI and reflections. */
export async function runPreparedTextureLightingFixture(staged = false) {
  const canvas = canvasOf(32);
  const mirror = batch('prepared-mirror', quad(), { definition: { ...DEFAULT_MATERIAL_DEFINITION,
    baseColor: m.constant([1, 1, 1, 1]), metallic: 1, roughness: 0.045 } });
  mirror.instances = [instance(11, scaling(3, 3, 1))];
  const tail = rgbaTexture(8, 8, 1);
  const emitterBinding = (asset: PreparedTextureAsset): MaterialBinding => ({ definition: {
    ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, alphaMode: 'mask', alphaCutoff: 0.5,
    baseColor: m.multiply(m.texture(0), m.constant([0, 0, 0, 1])), emissive: m.texture(0),
  }, textures: [asset] });
  const emitter = batch('prepared-emitter', quad(), emitterBinding(tail));
  const model = multiply(translation(0, 0, 5), scaling(20, 20, 1));
  emitter.instances = [instance(12, model)];
  const morph = new Float32Array(12);
  for (let i = 0; i < 4; i++) morph[i * 3] = 3;
  emitter.deformation = { jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] };
  emitter.morphTargets = [{ positions: morph, normals: new Float32Array(12) }];
  emitter.instances[0].deformScalars = [0];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [mirror, emitter], { antiAliasing: 'none',
      worldQueries: { maxRays: 4096 },
      diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance: 10, skyRadiance: [0, 0, 0],
        finalGather: { pixelSpacing: 4, maxAnchors: 64, raysPerAnchor: 4 } },
      reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4,
        maxDistance: 20, maxBytes: 16 * 1024 * 1024, skyRadiance: [0, 0, 0] },
      volumetricFog: { pixelSize: 4, slices: 16, maxViewDepth: 10,
        maxFroxels: 8 * 8 * 16, maxBytes: 1024 * 1024 } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    active.setDiffuseGiMultiBounce(false);
    let current: PreparedTextureAsset = tail;
    const stagedUploads: { name: string; bytes: number[]; total: number; layoutBytes: number;
      beforeRevision: number; afterRevision: number; beforeVariants: number; afterVariants: number;
      materials: number; slots: number }[] = [];
    const switchEmitter = async (name: string, asset: PreparedTextureAsset) => {
      if (!staged) {
        await active.setMaterial(emitter.name, emitterBinding(asset));
        current = asset;
        return;
      }
      if (current === asset) return;
      const layoutBytes = textureStorageLayout(asset).byteLength;
      const before = active.diagnostics();
      const handle = active.stageTexture(asset);
      const bytes: number[] = [];
      try {
        while (!handle.complete) {
          const written = handle.upload(16);
          if (written < 1) throw new Error('STAGED_TEXTURE_UPLOAD_STALLED');
          bytes.push(written);
          active.render([0, 0, 3], TARGET);
          if (active.diagnostics().materialRevision !== before.materialRevision) {
            throw new Error('STAGED_TEXTURE_PUBLISHED_BEFORE_REPLACE');
          }
        }
        const replacement = active.replaceTexture(current, asset);
        const after = active.diagnostics();
        stagedUploads.push({ name, bytes, total: bytes.reduce((sum, value) => sum + value, 0), layoutBytes,
          beforeRevision: before.materialRevision, afterRevision: after.materialRevision,
          beforeVariants: before.materialVariants, afterVariants: after.materialVariants,
          materials: replacement.materials, slots: replacement.slots });
        current = asset;
      } finally { handle.dispose(); }
    };
    const capture = async (name: string, expected: readonly number[]) => {
      for (let i = 0; i < 8; i++) active.render([0, 0, 3], TARGET);
      const [gi, reflections] = await Promise.all([active.probeDiffuseGi(), active.probeReflections([16 * 32 + 16])]);
      return { name, expected, giMax: [0, 1, 2].map(c => Math.max(...gi.probes.map(p => p.maxRadiance[c]))),
        anchor: reflections.anchors[0], materials: active.diagnostics().materials };
    };
    const expected = COLORS[1].slice(0, 3).map(value => value / 255);
    const samples = [await capture('rgba-tail', expected)];
    emitter.instances = [{ ...emitter.instances[0], deformScalars: [1] }];
    samples.push(await capture('morph-away', [0, 0, 0]));
    emitter.instances = [{ ...emitter.instances[0], deformScalars: [0] }];
    samples.push(await capture('morph-restored', expected));
    for (const sample of compressedSolidBlockCases) {
      const asset = blockTexture(sample.format, sample.bytes), layout = textureStorageLayout(asset);
      if (!active.info.features.includes(layout.requiredFeature!)) continue;
      await switchEmitter(sample.name, asset);
      samples.push(await capture(sample.name, sample.expected[3] < 0.5 ? [0, 0, 0] : sample.expected.slice(0, 3)));
    }
    await switchEmitter('rgba-restored', tail);
    samples.push(await capture('rgba-restored', expected));
    const errors = [...active.errors], adapter = active.info.adapterInfo, features = active.info.features;
    active.dispose();
    return { samples, stagedUploads, errors, adapter, features, afterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** The offscreen emitter remains available to GI and reflections as its managed mip changes. */
export async function runAutomaticTextureLightingFixture() {
  const canvas = canvasOf(32);
  const mirror = batch('automatic-mirror', quad(), { definition: { ...DEFAULT_MATERIAL_DEFINITION,
    baseColor: m.constant([1, 1, 1, 1]), metallic: 1, roughness: 0.045 } });
  mirror.instances = [instance(11, scaling(3, 3, 1))];
  const full = rgbaTexture(8, 8, 0), fallback = rgbaTexture(8, 8, 1);
  const mips = await Promise.all(full.levels.map(async (level, index) => {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', level.data.buffer as ArrayBuffer));
    return { level: index, width: level.width, height: level.height, byteLength: level.data.byteLength,
      sha256: Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('') };
  }));
  const source: TextureStreamSource = { id: 'automatic-emitter', width: 8, height: 8,
    format: 'rgba8', colorSpace: 'linear', mips, fallback,
    provider: async (mip, signal) => {
      if (signal.aborted) throw signal.reason;
      return full.levels[mip.level].data.slice();
    } };
  const emitter = batch('automatic-emitter', quad(), { definition: {
    ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true, alphaMode: 'mask', alphaCutoff: 0.5,
    baseColor: m.multiply(m.texture(0), m.constant([0, 0, 0, 1])), emissive: m.texture(0),
  }, textures: [fallback] });
  const model = multiply(translation(0, 0, 5), scaling(20, 20, 1));
  emitter.instances = [instance(12, model)];
  const morph = new Float32Array(12);
  for (let i = 0; i < 4; i++) morph[i * 3] = 3;
  emitter.deformation = { jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] };
  emitter.morphTargets = [{ positions: morph, normals: new Float32Array(12) }];
  emitter.instances[0].deformScalars = [0];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [mirror, emitter], { antiAliasing: 'none',
      textureBudgetBytes: 500, textureStreaming: { sources: [source], maxCpuBytes: 512,
        maxInFlightBytes: 512, maxUploadBytesPerFrame: 32, downgradeDelayFrames: 32 },
      worldQueries: { maxRays: 4096 },
      diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance: 10, skyRadiance: [0, 0, 0],
        finalGather: { pixelSpacing: 4, maxAnchors: 64, raysPerAnchor: 4 } },
      reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4,
        maxDistance: 20, maxBytes: 16 * 1024 * 1024, skyRadiance: [0, 0, 0] },
      volumetricFog: { pixelSize: 4, slices: 16, maxViewDepth: 10,
        maxFroxels: 8 * 8 * 16, maxBytes: 1024 * 1024 } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    active.setDiffuseGiMultiBounce(false);
    const frameDiagnostics: { streaming: NonNullable<ReturnType<Renderer['diagnostics']>['textureStreaming']>; imageBytes: number }[] = [];
    const render = (eye: V3, target: V3) => {
      active.render(eye, target);
      const diagnostic = active.diagnostics();
      frameDiagnostics.push({ streaming: diagnostic.textureStreaming!, imageBytes: diagnostic.materials.imageBytes });
      if (frameDiagnostics.length > 200) throw new Error('AUTOMATIC_LIGHTING_FRAME_LIMIT');
      return diagnostic;
    };
    const mirrorEye: V3 = [0, 0, 3], mirrorTarget: V3 = [0, 0, 0];
    const emitterTarget: V3 = [0, 0, 5];
    const capture = async (name: string) => {
      for (let frame = 0; frame < 8; frame++) render(mirrorEye, mirrorTarget);
      const [gi, reflections] = await Promise.all([active.probeDiffuseGi(), active.probeReflections([16 * 32 + 16])]);
      const diagnostics = active.diagnostics();
      return { name, giMax: [0, 1, 2].map(channel => Math.max(...gi.probes.map(probe => probe.maxRadiance[channel]))),
        anchor: reflections.anchors[0], mip: diagnostics.textureStreaming!.sources[0].currentMip,
        revision: diagnostics.materialRevision, variants: diagnostics.materialVariants };
    };
    const coarse = await capture('coarse-offscreen');
    let visibleFrames = 0;
    for (; visibleFrames < 100; visibleFrames++) {
      const diagnostics = render(mirrorEye, emitterTarget);
      await active.waitForTextureIo();
      if (diagnostics.textureStreaming!.sources[0].currentMip === 0 && !diagnostics.textureStreaming!.active) break;
    }
    if (visibleFrames === 100) throw new Error('AUTOMATIC_LIGHTING_DID_NOT_REFINE');
    const fine = await capture('fine-offscreen');
    let downgradeFrames = 0;
    for (; downgradeFrames < 100; downgradeFrames++) {
      const diagnostics = render(mirrorEye, mirrorTarget);
      await active.waitForTextureIo();
      if (diagnostics.textureStreaming!.sources[0].currentMip === 1 && !diagnostics.textureStreaming!.active) break;
    }
    if (downgradeFrames === 100) throw new Error('AUTOMATIC_LIGHTING_DID_NOT_DOWNGRADE');
    const coarseAgain = await capture('coarse-again');
    emitter.instances = [{ ...emitter.instances[0], deformScalars: [1] }];
    const morphAway = await capture('morph-away');
    emitter.instances = [{ ...emitter.instances[0], deformScalars: [0] }];
    const morphRestored = await capture('morph-restored');
    const adapter = active.info.adapterInfo, errors = [...active.errors];
    active.dispose();
    return { coarse, fine, coarseAgain, morphAway, morphRestored, visibleFrames, downgradeFrames,
      frames: frameDiagnostics, adapter, errors, afterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
