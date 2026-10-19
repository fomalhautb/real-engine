import type { InstanceState, SceneBatch, StreamedSceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { buildGeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { identity, normalMatrix, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { PreparedTextureAsset } from '../../../src/texture-storage';

type V3 = [number, number, number];
const SIZE = 256;
const EYE: V3 = [0, 0, 4], TARGET: V3 = [0, 0, 0];
const XS = [-0.5, -0.25, 0.5] as const;
const POINTS = XS.map(x => [x, 0, 0] as V3);
const PIXELS = XS.map(x => ({ x: Math.floor(SIZE * (0.5 + x / (8 * Math.tan(Math.PI / 6.4)))), y: SIZE / 2 }));

function pose(entityId: number, shift = 0, morph = 0, phase = 0): InstanceState {
  const model = identity();
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [translation(shift, 0, 0)],
    deformScalars: [morph, phase], color: [0.3, 0.9, 0.4, 0.5],
    metalness: 0, roughness: 1, historyVersion: shift ? 1 : 0 };
}

/** Affine colour field: independent expected RGBA is [0.5+0.3x,0.4+0.2y,0.25,0.5+0.3x]. */
function colouredGrid(cells: number): MeshData {
  const side = cells + 1, vertices = new ArrayBuffer(side * side * 80);
  const view = new DataView(vertices), colors = new Float32Array(side * side * 4);
  const indices = new Uint32Array(cells * cells * 6);
  for (let y = 0; y <= cells; y++) for (let x = 0; x <= cells; x++) {
    const px = 2 * x / cells - 1, py = 2 * y / cells - 1;
    const v = y * side + x, at = v * 80;
    view.setFloat32(at, px, true); view.setFloat32(at + 4, py, true);
    view.setFloat32(at + 12, 1, true); view.setFloat32(at + 24, 1, true);
    // All four joints must be in range even where their weight is zero.
    view.setFloat32(at + 48, 1, true);
    colors.set([0.5 + 0.3 * px, 0.4 + 0.2 * py, 0.25, 0.5 + 0.3 * px], v * 4);
  }
  let at = 0;
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const a = y * side + x, b = a + 1, c = a + side, d = c + 1;
    indices.set([a, b, d, a, d, c], at); at += 6;
  }
  return { vertices, indices, vertexCount: side * side, colors,
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function batch(name: string, mesh: MeshData, mode: 'opaque' | 'mask', deform = false,
  texture?: PreparedTextureAsset): SceneBatch {
  const morph = new Float32Array(mesh.vertexCount * 3);
  for (let i = 0; i < mesh.vertexCount; i++) morph[i * 3] = 0.1;
  const color = texture ? m.multiply(m.vertexColor, m.texture(0)) : m.vertexColor;
  return { name, mesh, capacity: 1, inputMode: 'committed', instances: [pose(101)],
    deformation: deform ? { jointCount: 1, nodes: [
      { kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' },
      { kind: 'wave', amplitude: 0.25, frequency: Math.PI / 2, phase: 1 },
    ] } : { jointCount: 0, nodes: [] },
    morphTargets: deform ? [{ positions: morph, normals: new Float32Array(morph.length) }] : [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      alphaMode: mode, alphaCutoff: 0.5, doubleSided: true,
      baseColor: color, emissive: m.multiply(color, m.constant([0.2, 0.2, 0.2, 1])) },
    textures: texture ? [texture] : undefined } };
}

function mipTailTexture(): PreparedTextureAsset {
  const solid = (width: number, height: number, rgba: readonly [number, number, number, number]) => {
    const data = new Uint8Array(width * height * 4);
    for (let offset = 0; offset < data.length; offset += 4) data.set(rgba, offset);
    return { width, height, data };
  };
  return { kind: 'prepared', width: 8, height: 8, colorSpace: 'linear', format: 'rgba8', baseMipLevel: 1,
    levels: [solid(4, 4, [192, 128, 64, 255]), solid(2, 2, [32, 224, 48, 255]),
      solid(1, 1, [224, 32, 192, 255])] };
}

const rays = (shift = 0) => POINTS.map(([x, y]) => ({ origin: [x + shift, y, 4] as V3,
  direction: [0, 0, -1] as V3, tMax: 8 }));

export async function runVertexColorSurfaceFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px;z-index:10000`;
  document.body.append(canvas);
  const scene = batch('vertex-colour-surface', colouredGrid(2), 'opaque', true);
  let renderer: Renderer | undefined, whiteRenderer: Renderer | undefined;
  let whiteCanvas: HTMLCanvasElement | undefined;
  try {
    renderer = await Renderer.create(canvas, [scene], { antiAliasing: 'none',
      worldQueries: { maxRays: 256 }, sunShadow: { resolution: 256,
        focusBounds: { min: [-2, -2, -1], max: [2, 2, 1] } },
      diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
        raysPerProbe: 32, maxDistance: 5, skyRadiance: [0, 0, 0] } });
    const active = renderer;
    active.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async (shift = 0) => {
      active.render(EYE, TARGET);
      const png = canvas.toDataURL('image/png');
      const [hdr, depth, hits] = await Promise.all([
        active.probeHdr(PIXELS.map((p, i) => ({ ...p, x: p.x + Math.round(shift * SIZE / (8 * Math.tan(Math.PI / 6.4))) }))),
        active.probeDepth(POINTS.map(([x, y]) => [x + shift, y, 0] as V3)),
        active.traceRays(rays(shift)),
      ]);
      return { png, hdr, depth, hits: hits.hits };
    };
    const opaque = await capture();
    await active.setMaterial(scene.name, { definition: { ...scene.material!.definition,
      alphaMode: 'mask', alphaCutoff: 0.5 } });
    const masked = await capture();
    scene.instances = [pose(101, 0.2, 1, Math.PI / 2)];
    const moved = await capture(0.55);
    // PBR base colour does not itself emit into the world field. The neutral
    // emissive factor leaves all channel differences attributable to vertices.
    await active.setMaterial(scene.name, { definition: { ...scene.material!.definition,
      shading: 'pbr', alphaMode: 'mask' } });
    active.setDiffuseGiMultiBounce(false);
    for (let i = 0; i < 8; i++) active.render(EYE, TARGET);
    const giEmissive = await active.probeDiffuseGi();
    await active.setMaterial(scene.name, { definition: { ...scene.material!.definition,
      shading: 'pbr', alphaMode: 'mask', emissive: m.constant([0, 0, 0, 1]) } });
    for (let i = 0; i < 8; i++) active.render(EYE, TARGET);
    const giDark = await active.probeDiffuseGi();
    const whiteMesh = colouredGrid(2);
    whiteMesh.colors = undefined;
    whiteCanvas = document.createElement('canvas');
    whiteCanvas.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px`;
    document.body.append(whiteCanvas);
    whiteRenderer = await Renderer.create(whiteCanvas, [batch('vertex-colour-missing', whiteMesh, 'opaque')],
      { antiAliasing: 'none', worldQueries: { maxRays: 4 } });
    whiteRenderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    whiteRenderer.render(EYE, TARGET);
    const white = { png: whiteCanvas.toDataURL('image/png'),
      hdr: (await whiteRenderer.probeHdr([PIXELS[2]]))[0],
      hit: (await whiteRenderer.traceRays([rays()[2]])).hits[0] };
    const adapter = active.info.adapterInfo, errors = [...active.errors];
    errors.push(...whiteRenderer.errors);
    whiteRenderer.dispose();
    const whiteAfterDispose = whiteRenderer.stats.ownedGpuBytes;
    active.dispose();
    return { points: POINTS, pixels: PIXELS, opaque, masked, moved, white,
      giEmissive, giDark, adapter, errors,
      afterDispose: active.stats.ownedGpuBytes, whiteAfterDispose };
  } finally { renderer?.dispose(); whiteRenderer?.dispose(); canvas.remove(); whiteCanvas?.remove(); }
}

export async function runVertexColorHierarchyFixture(preparedMipTail = false) {
  const source = colouredGrid(16);
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'vertex-colour-grid', mesh: source,
    coverage: 'opaque' }, { maxClusterTriangles: 32, groupSize: 4, simplifyRatio: 0.5, maxLevels: 6 });
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 16384 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const provider = async (page: { id: number }, signal: AbortSignal) => {
    signal.throwIfAborted(); return encoded.pages[page.id].bytes;
  };
  const proxy = await buildGeometryQueryProxy(directory, provider, { topologyVersion: 1 });
  const texture = preparedMipTail ? mipTailTexture() : undefined;
  const full = batch('vertex-colour-full', hierarchy.source, 'opaque', false, texture);
  const resident = { ...batch('vertex-colour-resident', hierarchy.source, 'opaque', false, texture),
    geometryLod: hierarchy };
  const { mesh: _omittedMesh, geometryLod: _omittedLod, geometryStream: _omittedStream,
    ...pagedCommon } = batch('vertex-colour-paged', hierarchy.source, 'opaque', false, texture);
  const paged: StreamedSceneBatch = { ...pagedCommon,
    geometryStream: { directory, provider, queryProxy: proxy } };
  const scenes = [full, resident, paged], canvases: HTMLCanvasElement[] = [], renderers: Renderer[] = [];
  try {
    for (const scene of scenes) {
      const canvas = document.createElement('canvas');
      canvas.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px`;
      document.body.append(canvas); canvases.push(canvas);
      const renderer = await Renderer.create(canvas, [scene], { antiAliasing: 'none',
        worldQueries: { maxRays: 32 } });
      renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
      if (scene !== full) renderer.setGeometryLod({ maxErrorPixels: 10_000, frustumCulling: false });
      renderers.push(renderer);
    }
    for (let i = 0; i < 8; i++) {
      for (const renderer of renderers) renderer.render(EYE, TARGET);
      await Promise.all(renderers.map(renderer => renderer.probeHdr([{ x: 0, y: 0 }])));
    }
    const read = async () => Promise.all(renderers.map(async (renderer, i) => ({
      png: canvases[i].toDataURL('image/png'), hdr: await renderer.probeHdr(PIXELS),
      hits: (await renderer.traceRays(rays())).hits,
      lod: i ? await renderer.probeGeometryLod(scenes[i].name) : null,
    })));
    const coarse = await read();
    for (const renderer of renderers.slice(1)) renderer.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    let fine: Awaited<ReturnType<typeof read>> | undefined;
    for (let i = 0; i < 240; i++) {
      for (const renderer of renderers) renderer.render(EYE, TARGET);
      fine = await read();
      if (fine.slice(1).every(sample => sample.lod!.triangles === hierarchy.sourceTriangleCount)) break;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (!fine || fine.slice(1).some(sample => sample.lod!.triangles !== hierarchy.sourceTriangleCount)) {
      throw new Error('VERTEX_COLOR_PAGES_DID_NOT_REFINE');
    }
    const adapter = renderers[0].info.adapterInfo;
    const errors = renderers.flatMap(renderer => renderer.errors);
    for (const renderer of renderers) renderer.dispose();
    return { coarse, samples: fine, points: POINTS, sourceTriangles: hierarchy.sourceTriangleCount,
      preparedBaseMipLevel: texture?.baseMipLevel ?? null,
      sourceColors: Array.from(hierarchy.source.colors ?? []),
      proxyColors: Array.from(proxy.mesh.colors ?? []), pages: directory.pages.length,
      adapter, errors, afterDispose: renderers.map(renderer => renderer.stats.ownedGpuBytes) };
  } finally { for (const renderer of renderers) renderer.dispose(); for (const canvas of canvases) canvas.remove(); }
}
