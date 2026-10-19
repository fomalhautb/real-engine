import type { MaterialBinding, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { PreparedTextureAsset } from '../../../src/texture-storage';
import type { TextureStreamSource } from '../../../src/texture-stream-types';
import { createHttpTextureMipProvider } from '../../../src/texture-stream-source';
import { loadTextureStreamSource } from '../../../src/texture-stream-manifest';

const COLORS = {
  a: [[192, 32, 16], [32, 192, 16], [16, 32, 192], [128, 64, 32]],
  b: [[16, 192, 96], [192, 16, 96], [96, 16, 192], [32, 128, 64]],
} as const;
type SourceName = keyof typeof COLORS;
const EYES = { a: [-4, 0, 4], b: [4, 0, 4] } as const;
const TARGETS = { a: [-4, 0, 0], b: [4, 0, 0] } as const;

function quad(): MeshData {
  const vertices = new ArrayBuffer(4 * 80), data = new DataView(vertices);
  [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([x, y], vertex) => {
    data.setFloat32(vertex * 80, x, true);
    data.setFloat32(vertex * 80 + 4, y, true);
    data.setFloat32(vertex * 80 + 12, 1, true);
    data.setFloat32(vertex * 80 + 24, 1, true);
    data.setFloat32(vertex * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function payload(name: SourceName, level: number): Uint8Array {
  const size = 8 >> level, data = new Uint8Array(size * size * 4), rgb = COLORS[name][level];
  for (let i = 0; i < data.length; i += 4) data.set([...rgb, 255], i);
  return data;
}

async function source(name: SourceName, gate?: Promise<void>, useManifest = false): Promise<TextureStreamSource> {
  if (useManifest) {
    const loaded = await loadTextureStreamSource(`/__texture-stream/${name}/manifest.json`);
    let first = true;
    return { ...loaded, provider: async (mip, signal) => {
      if (gate && first) { first = false; await gate; }
      if (signal.aborted) throw signal.reason;
      return loaded.provider(mip, signal);
    } };
  }
  const payloads = Array.from({ length: 4 }, (_, level) => payload(name, level));
  const mips = await Promise.all(payloads.map(async (data, level) => {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', data.buffer as ArrayBuffer));
    return { level, width: 8 >> level, height: 8 >> level, byteLength: data.byteLength,
      sha256: Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('') };
  }));
  const fallback: PreparedTextureAsset = { kind: 'prepared', width: 8, height: 8, format: 'rgba8',
    colorSpace: 'linear', baseMipLevel: 2, levels: [2, 3].map(level => ({ width: 8 >> level,
      height: 8 >> level, data: payloads[level].slice() })) };
  let first = true;
  const http = createHttpTextureMipProvider(mips.map(mip => `/__texture-stream/${name}/${mip.level}.bin`));
  return { id: name, width: 8, height: 8, format: 'rgba8', colorSpace: 'linear', mips, fallback,
    provider: async (mip, signal) => {
      if (gate && first) { first = false; await gate; }
      if (signal.aborted) throw signal.reason;
      return http(mip, signal);
    } };
}

function binding(asset: PreparedTextureAsset, tint: readonly [number, number, number, number], repeatU = false): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', alphaMode: 'mask',
    doubleSided: true, alphaCutoff: 0.5, baseColor: m.multiply(m.texture(0), m.parameter(0)) },
    parameters: new Float32Array(tint), textures: [asset],
    samplers: [{ minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest',
      ...(repeatU ? { addressModeU: 'repeat' } : {}) }] };
}

function batch(name: string, x: number, material: MaterialBinding, entityId: number): SceneBatch {
  const model = multiply(translation(x, 0, 0), scaling(0.35, 0.35, 1));
  return { name, mesh: quad(), capacity: 1, inputMode: 'committed', material,
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    instances: [{ entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
      color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 }] };
}

export async function runTextureStreamingFixture(useManifest = false) {
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 128;
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;z-index:10000';
  document.body.append(canvas);
  let releaseFirstA!: () => void;
  const gate = new Promise<void>(resolve => { releaseFirstA = resolve; });
  const [a, b] = await Promise.all([source('a', gate, useManifest), source('b', undefined, useManifest)]);
  const scene = [batch('a-left', -4.5, binding(a.fallback, [1, 1, 1, 1]), 1),
    batch('a-right', -3.5, binding(a.fallback, [1, 0.5, 1, 1], true), 2),
    batch('b', 4, binding(b.fallback, [1, 1, 1, 1]), 3)];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scene, { antiAliasing: 'none', textureBudgetBytes: 400,
      textureStreaming: { sources: [a, b], maxCpuBytes: 400, maxInFlightBytes: 512,
        maxUploadBytesPerFrame: 32, downgradeDelayFrames: 1 }, worldQueries: { maxRays: 8 },
      sunShadow: { resolution: 256, focusBounds: { min: [-5, -1, -1], max: [5, 1, 1] } } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const frames: ReturnType<Renderer['diagnostics']>[] = [];
    const render = (view: SourceName) => {
      active.render([...EYES[view]], [...TARGETS[view]]);
      const state = active.diagnostics();
      frames.push(state);
      if (frames.length > 300) throw new Error('TEXTURE_STREAM_FIXTURE_FRAME_LIMIT');
      return state;
    };
    const sample = async (name: string, view: SourceName) => {
      const state = active.diagnostics();
      const points = view === 'a' ? [{ x: 49, y: 64 }, { x: 79, y: 64 }] : [{ x: 64, y: 64 }];
      const [hdr, depth, query] = await Promise.all([
        active.probeHdr(points),
        active.probeDepth((view === 'a' ? [-4.5, -3.5] : [4]).map(x => [x, 0, 0] as [number, number, number])),
        active.traceRays([-4.5, -3.5, 4].map(x => ({ origin: [x, 0, 4] as [number, number, number],
          direction: [0, 0, -1] as [number, number, number], tMax: 8 })), { textureLod: 0 }),
      ]);
      return { name, view, hdr, depth, hits: query.hits, png: canvas.toDataURL(),
        streaming: state.textureStreaming!, imageBytes: state.materials.imageBytes,
        variants: state.materialVariants, revision: state.materialRevision };
    };
    const pump = async (view: SourceName, expectedA: number, expectedB: number) => {
      for (let frame = 0; frame < 100; frame++) {
        const state = render(view).textureStreaming!;
        await active.waitForTextureIo();
        if (state.sources[0].currentMip === expectedA && state.sources[1].currentMip === expectedB && !state.active) return;
      }
      throw new Error('TEXTURE_STREAM_FIXTURE_DID_NOT_SETTLE: ' + JSON.stringify(active.diagnostics().textureStreaming));
    };

    render('a');
    const coarse = await sample('coarse', 'a');
    releaseFirstA();
    await active.waitForTextureIo();
    render('a');
    const partial = await sample('partial', 'a');
    await pump('a', 0, 2);
    const aFine = await sample('a-fine', 'a');
    const managedError = (() => {
      try { active.replaceTexture(a.fallback, b.fallback); return ''; }
      catch (error) { return String(error); }
    })();
    await active.setMaterial('a-right', binding(a.fallback, [0.5, 1, 1, 1]));
    render('a');
    const aLateMaterial = await sample('a-late-material', 'a');

    await pump('b', 2, 0);
    const bFine = await sample('b-fine', 'b');
    await active.setMaterial('b', binding(b.fallback, [1, 1, 1, 1]));
    render('b');
    const bLateMaterial = await sample('b-late-material', 'b');
    await pump('a', 0, 2);
    const aReturn = await sample('a-return', 'a');
    const settledStart = active.diagnostics().textureStreaming!;
    for (let frame = 0; frame < 8; frame++) { render('a'); await active.waitForTextureIo(); }
    const settledEnd = active.diagnostics().textureStreaming!;
    const errors = [...active.errors], adapter = active.info.adapterInfo;
    active.dispose();
    return { coarse, partial, aFine, aLateMaterial, bFine, bLateMaterial, aReturn,
      frames: frames.map(frame => ({ streaming: frame.textureStreaming, imageBytes: frame.materials.imageBytes })),
      managedError, settledStart, settledEnd, errors, adapter, afterDispose: active.stats.ownedGpuBytes };
  } finally { releaseFirstA(); renderer?.dispose(); canvas.remove(); }
}
