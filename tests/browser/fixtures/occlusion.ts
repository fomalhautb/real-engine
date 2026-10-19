import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { createCube } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import { identity, normalMatrix, multiply, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer, type RendererOptions } from '../../../src/renderer';

type Vec3 = [number, number, number];
type Pixel = { x: number; y: number };
type Cut = Awaited<ReturnType<Renderer['probeGeometryLod']>> & {
  occlusion: { deferredClusters: number; culledClusters: number; recoveredClusters: number };
};
type OcclusionState = { enabled: boolean; historyValid: boolean; grid: unknown;
  ownedGpuBytes: number; algorithm: string };
type OcclusionRenderer = Renderer & { setOcclusionEnabled(enabled: boolean): void };

const EYE: Vec3 = [0, 0, 7];
const TARGET: Vec3 = [0, 0, 0];
const DEPTH_POINTS: Vec3[] = [[-1, -1, 0], [0, 0, 0], [1, 1, 0]];

function grid(cells = 32): MeshData {
  const side = cells + 1;
  const vertices = new ArrayBuffer(side * side * 80);
  const view = new DataView(vertices);
  const indices = new Uint32Array(cells * cells * 6);
  const min: Vec3 = [-2, -2, -0.035], max: Vec3 = [2, 2, 0.035];
  for (let y = 0; y <= cells; y++) for (let x = 0; x <= cells; x++) {
    const px = 4 * x / cells - 2, py = 4 * y / cells - 2;
    const pz = 0.03 * Math.sin(px * 2) * Math.cos(py * 2);
    const dx = 0.06 * Math.cos(px * 2) * Math.cos(py * 2);
    const dy = -0.06 * Math.sin(px * 2) * Math.sin(py * 2);
    const length = Math.hypot(dx, dy, 1);
    const offset = (y * side + x) * 80;
    view.setFloat32(offset, px, true); view.setFloat32(offset + 4, py, true);
    view.setFloat32(offset + 8, pz, true); view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 16, -dx / length, true);
    view.setFloat32(offset + 20, -dy / length, true);
    view.setFloat32(offset + 24, 1 / length, true);
    view.setFloat32(offset + 48, 1, true);
  }
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const a = y * side + x, b = a + 1, c = a + side, d = c + 1;
    indices.set([a, b, d, a, d, c], (y * cells + x) * 6);
  }
  return { vertices, indices, vertexCount: side * side, bounds: { min, max } };
}

function instance(id: number, model: Float32Array): InstanceState {
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
}

function material(color: [number, number, number, number], alphaMode: 'opaque' | 'mask' = 'opaque') {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const,
    alphaMode, alphaCutoff: 0.5, baseColor: m.parameter(0) }, parameters: new Float32Array(color) };
}

function maskHoleMaterial() {
  const pixels = new Uint8Array(8 * 8 * 4);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const offset = (y * 8 + x) * 4;
    pixels.set([255, 255, 255, x >= 2 && x < 6 && y >= 2 && y < 6 ? 0 : 255], offset);
  }
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const,
    alphaMode: 'mask' as const, alphaCutoff: 0.5,
    baseColor: m.multiply(m.parameter(0), m.texture(0)) },
    parameters: new Float32Array([0.03, 0.1, 0.75, 1]),
    textures: [{ width: 8, height: 8, colorSpace: 'srgb' as const, pixels }],
    samplers: [{ addressModeU: 'clamp-to-edge' as const, addressModeV: 'clamp-to-edge' as const }] };
}

function pixels(width: number, height: number): Pixel[] {
  return Array.from({ length: 16 * 16 }, (_, index) => ({
    x: Math.floor((index % 16 + 0.5) * width / 16),
    y: Math.floor((Math.floor(index / 16) + 0.5) * height / 16),
  }));
}

function diff(reference: number[][], actual: number[][]) {
  let max = 0, total = 0, visible = 0, referenceVisible = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i][0] + actual[i][1] + actual[i][2] > 0.08) visible++;
    if (reference[i][0] + reference[i][1] + reference[i][2] > 0.08) referenceVisible++;
    for (let c = 0; c < 3; c++) {
      const error = Math.abs(actual[i][c] - reference[i][c]);
      max = Math.max(max, error); total += error;
    }
  }
  return { max, mean: total / (actual.length * 3), visible, referenceVisible };
}

/** A virtual red backboard and independently rendered blue wall. Every capture
 * compares the same frame against a renderer with occlusion disabled. */
export async function runOcclusionFixture(mode: 'resident' | 'stream') {
  const mesh = grid();
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'occlusion-backboard-v1', mesh, coverage: 'opaque' },
    { maxClusterTriangles: 24, maxClusterVertices: 48, groupSize: 4, simplifyRatio: 0.5, maxLevels: 8 });
  const encoded = mode === 'stream' ? await encodeGeometryPages(hierarchy, { maxPageBytes: 8192 }) : undefined;
  const directory = encoded ? await buildGeometryStreamDirectory(hierarchy, encoded) : undefined;
  const container = document.createElement('div');
  container.id = `occlusion-${mode}-visual`;
  container.style.cssText = 'display:flex;flex-direction:column;gap:16px;position:relative;z-index:10000;color:white;background:#111;padding:12px;width:max-content';
  const canvases = [0, 1].map(() => {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width:256px;height:256px;display:block';
    container.append(canvas); return canvas;
  });
  document.body.append(container);
  const scenePairs = [0, 1].map(index => {
    const wallModel = multiply(translation(0, 0, 2), scaling(5, 5, 0.2));
    const backboard: SceneBatch = { name: 'backboard', mesh: hierarchy.source, capacity: 1, inputMode: 'committed',
      instances: [instance(100 + index * 10, identity())], deformation: { jointCount: 0, nodes: [] },
      morphTargets: [], material: material([0.8, 0.08, 0.03, 1]),
      ...(mode === 'resident' ? { geometryLod: hierarchy } : { geometryStream: {
        directory: directory!, provider: async (page: { id: number }, signal: AbortSignal) => {
          signal.throwIfAborted(); return encoded!.pages[page.id].bytes.slice();
        }, maxRequestsPerFrame: 8, gpuPool: { maxUploadBytesPerFrame: 1 << 20,
          maxUploadWritesPerFrame: 128, maxUploadPagesPerFrame: 64 } } }),
    };
    const virtualWall = mode === 'resident';
    const actualWallModel = virtualWall ? translation(0, 0, 2) : wallModel;
    const wallMesh = virtualWall ? hierarchy.source : createCube();
    if (!virtualWall) {
      const vertexData = new DataView(wallMesh.vertices);
      for (let vertex = 0; vertex < wallMesh.vertexCount; vertex++) {
        for (let joint = 0; joint < 4; joint++) vertexData.setUint32(vertex * 80 + 32 + joint * 4, 0, true);
      }
    }
    const morphPositions = new Float32Array(wallMesh.vertexCount * 3);
    if (!virtualWall) for (let vertex = 0; vertex < wallMesh.vertexCount; vertex++) morphPositions[vertex * 3] = 2;
    const wall: SceneBatch = { name: 'wall', mesh: wallMesh,
      ...(virtualWall ? { geometryLod: hierarchy } : {}), capacity: 1, inputMode: 'committed',
      instances: [{ ...instance(101 + index * 10, actualWallModel),
        bones: virtualWall ? [] : [identity()], deformScalars: virtualWall ? [] : [0] }],
      deformation: virtualWall ? { jointCount: 0, nodes: [] }
        : { jointCount: 1, nodes: [{ kind: 'morph' as const, target: 0, weight: 0 }, { kind: 'skin' as const }] },
      morphTargets: virtualWall ? [] : [{ positions: morphPositions, normals: new Float32Array(morphPositions.length) }],
      material: material([0.03, 0.1, 0.75, 1]) };
    return { batches: [backboard, wall], backboard, wall };
  });
  const options = { occlusion: { pixelSize: 4, maxBytes: 150_000 }, worldQueries: { maxRays: 4 },
    sunShadow: { resolution: 256, focusBounds: { min: [-5, -5, -2], max: [5, 5, 4] } } } as RendererOptions;
  let reference: Renderer | undefined, tested: Renderer | undefined;
  try {
    reference = await Renderer.create(canvases[0], scenePairs[0].batches, options);
    tested = await Renderer.create(canvases[1], scenePairs[1].batches, options);
    const a = reference, b = tested as OcclusionRenderer;
    a.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    b.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    a.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    b.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    (a as OcclusionRenderer).setOcclusionEnabled(false);
    b.setOcclusionEnabled(true);
    const visuals: { label: string; images: string[]; triangles: number; deferred: number; culled: number; recovered: number }[] = [];
    const render = async (eye: Vec3 = EYE, target: Vec3 = TARGET, visualLabel?: string) => {
      a.render(eye, target); b.render(eye, target);
      const images = visualLabel ? canvases.map(canvas => canvas.toDataURL('image/png')) : undefined;
      const samplePoints = pixels(b.stats.width, b.stats.height);
      if (b.stats.width % 4 || b.stats.height % 4) samplePoints.push(
        { x: b.stats.width - 1, y: b.stats.height - 1 },
        { x: b.stats.width - 1, y: Math.floor(b.stats.height / 2) },
        { x: Math.floor(b.stats.width / 2), y: b.stats.height - 1 });
      const [referenceHdr, actualHdr, referenceDepth, actualDepth, baseline, cut] = await Promise.all([
        a.probeHdr(samplePoints), b.probeHdr(samplePoints), a.probeDepth(DEPTH_POINTS), b.probeDepth(DEPTH_POINTS),
        a.probeGeometryLod('backboard'), b.probeGeometryLod('backboard'),
      ]);
      if (visualLabel && images) {
        const counted = cut as Cut;
        visuals.push({ label: visualLabel, images, triangles: counted.triangles,
          deferred: counted.occlusion.deferredClusters, culled: counted.occlusion.culledClusters,
          recovered: counted.occlusion.recoveredClusters });
      }
      return { error: diff(referenceHdr, actualHdr), referenceHdr, actualHdr,
        referenceDepth: referenceDepth.map(value => value.surface), actualDepth: actualDepth.map(value => value.surface),
        baseline, cut: cut as Cut, occlusion: (b.diagnostics() as unknown as { occlusion: OcclusionState }).occlusion };
    };
    // Stream pages and history both need time to settle. An unchanged first
    // frame is deliberately not treated as proof of effective occlusion.
    let stable = await render(), settledFrames = 0;
    for (; settledFrames < 80; settledFrames++) {
      stable = await render();
      if (stable.cut.occlusion?.culledClusters > 0
        && (mode === 'resident' || stable.baseline.triangles === hierarchy.sourceTriangleCount)) break;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    stable = await render(EYE, TARGET, 'Closed wall, stable occlusion');
    const queryHidden = await b.traceRays([{ origin: [0, 0, 1], direction: [0, 0, -1], tMax: 3 }]);
    const wall = scenePairs.map(pair => pair.wall);
    for (const batch of wall) {
      if (mode === 'resident') {
        const moved = translation(8, 0, 2);
        batch.instances[0] = { ...batch.instances[0], model: moved, normalMatrix: normalMatrix(moved) };
      } else batch.instances[0] = { ...batch.instances[0], deformScalars: [1] };
    }
    const movedFirst = await render(EYE, TARGET, 'Occluder moved, first recovery frame');
    const movedNext = await render();
    for (const batch of wall) {
      if (mode === 'resident') {
        const original = translation(0, 0, 2);
        batch.instances[0] = { ...batch.instances[0], model: original, normalMatrix: normalMatrix(original) };
      } else batch.instances[0] = { ...batch.instances[0], deformScalars: [0] };
    }
    await render(); await render();
    if (mode === 'stream') for (const renderer of [a, b]) {
      await renderer.setMaterial('wall', maskHoleMaterial());
    }
    const maskFirst = mode === 'stream' ? await render(EYE, TARGET, 'Mask hole, first recovery frame') : null;
    if (mode === 'stream') for (const renderer of [a, b]) {
      await renderer.setMaterial('wall', material([0.03, 0.1, 0.75, 1], 'opaque'));
    }
    await render(); await render();
    const turnedFirst = await render([4, 0, 7], TARGET);
    const nearPlane = await render([0, 0, 2.05], TARGET);
    b.setOcclusionEnabled(false);
    const disabled = await render();
    b.setOcclusionEnabled(true);
    const reenabled = await render();
    for (const canvas of canvases) canvas.style.cssText = 'width:319px;height:223px;display:block';
    const resized = await render();
    const queryAfterResize = await b.traceRays([{ origin: [0, 0, 1], direction: [0, 0, -1], tMax: 3 }]);
    const budgetState = () => ({
      stats: { width: b.stats.width, height: b.stats.height, frames: b.stats.frames,
        ownedGpuBytes: b.stats.ownedGpuBytes },
      occlusion: structuredClone((b.diagnostics() as unknown as { occlusion: OcclusionState }).occlusion),
    });
    const beforeBudgetFailure = budgetState();
    for (const canvas of canvases) canvas.style.cssText = 'width:1024px;height:1024px;display:block';
    let budgetError = '';
    try { b.render(EYE, TARGET); }
    catch (error) { budgetError = String(error); }
    const afterBudgetFailure = budgetState();
    for (const canvas of canvases) canvas.style.cssText = 'width:319px;height:223px;display:block';
    const recoveredAfterBudget = await render();
    for (const pair of scenePairs) { pair.backboard.instances = []; pair.wall.instances = []; }
    const empty = await render();
    for (const visual of visuals) {
      const figure = document.createElement('figure'); figure.style.cssText = 'margin:0';
      const caption = document.createElement('figcaption');
      caption.textContent = `${visual.label} — drawn triangles ${visual.triangles}, deferred ${visual.deferred}, culled ${visual.culled}, recovered ${visual.recovered}`;
      figure.append(caption);
      const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:12px';
      visual.images.forEach((src, index) => {
        const column = document.createElement('div');
        const title = document.createElement('div'); title.textContent = index ? 'Two-phase occlusion' : 'Occlusion disabled';
        const image = document.createElement('img'); image.src = src;
        image.width = 256; image.height = 256; image.style.cssText = 'display:block;image-rendering:auto';
        column.append(title, image); row.append(column);
      });
      figure.append(row); container.append(figure);
    }
    canvases.forEach(canvas => {
      canvas.remove();
    });
    await Promise.all(Array.from(container.querySelectorAll('img'), image => image.decode()));
    const diagnostics = (b.diagnostics() as unknown as { occlusion: OcclusionState }).occlusion;
    const adapter = b.info.adapterInfo;
    const errors = [...a.errors, ...b.errors];
    a.dispose(); b.dispose();
    return { mode, sourceTriangles: hierarchy.sourceTriangleCount, pages: directory?.pages.length ?? 0,
      settledFrames, stable, movedFirst, movedNext, maskFirst, turnedFirst, nearPlane, disabled, reenabled, resized,
      failedResize: { error: budgetError, before: beforeBudgetFailure, after: afterBudgetFailure },
      recoveredAfterBudget, empty,
      queryHidden, queryAfterResize,
      diagnostics, adapter, errors, ownedAfterDispose: [a.stats.ownedGpuBytes, b.stats.ownedGpuBytes] };
  } finally { reference?.dispose(); tested?.dispose(); }
}
