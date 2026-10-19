import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { createCube } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer, type RendererOptions } from '../../../src/renderer';

type Vec3 = [number, number, number];
type Pixel = { x: number; y: number };
const EYE: Vec3 = [0, 0, 7];
const TARGET: Vec3 = [0, 0, 0];
const CENTER: Pixel = { x: 32, y: 32 };
const DEPTH_POINTS: Vec3[] = [[-0.8, -0.8, 0], [0, 0, 0], [0.8, 0.8, 0]];
const PIXELS: Pixel[] = Array.from({ length: 64 }, (_, index) => ({
  x: (index % 8) * 8 + 4, y: Math.floor(index / 8) * 8 + 4,
}));

function backboardMesh(cells = 32): MeshData {
  const side = cells + 1, vertices = new ArrayBuffer(side * side * 80);
  const view = new DataView(vertices), indices = new Uint32Array(cells * cells * 6);
  for (let y = 0; y <= cells; y++) for (let x = 0; x <= cells; x++) {
    const px = 4 * x / cells - 2, py = 4 * y / cells - 2;
    const pz = 0.03 * Math.sin(px * 2) * Math.cos(py * 2);
    const dx = 0.06 * Math.cos(px * 2) * Math.cos(py * 2);
    const dy = -0.06 * Math.sin(px * 2) * Math.sin(py * 2);
    const length = Math.hypot(dx, dy, 1), offset = (y * side + x) * 80;
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
  return { vertices, indices, vertexCount: side * side,
    bounds: { min: [-2, -2, -0.035], max: [2, 2, 0.035] } };
}

function quad(): MeshData {
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([x, y], index) => {
    const at = index * 80;
    view.setFloat32(at, x, true); view.setFloat32(at + 4, y, true);
    view.setFloat32(at + 12, 1, true); view.setFloat32(at + 24, 1, true);
    view.setFloat32(at + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function instance(entityId: number, model: Float32Array, metalness: number, roughness: number): InstanceState {
  return { entityId, entityGeneration: 1, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness, roughness, historyVersion: 0 };
}

function scene(mesh: MeshData, hierarchy: Awaited<ReturnType<typeof buildGeometryHierarchy>>) {
  const board: SceneBatch = { name: 'combined-virtual-pbr-board', mesh, geometryLod: hierarchy,
    capacity: 1, inputMode: 'committed', instances: [instance(101, identity(), 0.45, 0.045)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      baseColor: m.constant([0.85, 0.7, 0.55, 1]) } } };
  const wallModel = multiply(translation(0, 0, 2), scaling(5, 5, 0.2));
  const wall: SceneBatch = { name: 'combined-occluding-wall', mesh: createCube(),
    capacity: 1, inputMode: 'committed', instances: [instance(102, wallModel, 0, 1)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.constant([0.03, 0.08, 0.16, 1]) } } };
  const lightModel = multiply(translation(0, 0, 10), scaling(20, 20, 1));
  const emitter: SceneBatch = { name: 'combined-offscreen-reflection-emitter', mesh: quad(),
    capacity: 1, inputMode: 'committed', instances: [instance(103, lightModel, 0, 1)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.constant([3, 1.5, 0.75, 1]) } } };
  return { batches: [board, wall, emitter], wall };
}

function difference(left: number[][], right: number[][]) {
  let max = 0, mean = 0, visible = 0;
  for (let pixel = 0; pixel < left.length; pixel++) {
    if (left[pixel][0] + left[pixel][1] + left[pixel][2] > 0.1) visible++;
    for (let channel = 0; channel < 3; channel++) {
      const delta = Math.abs(left[pixel][channel] - right[pixel][channel]);
      max = Math.max(max, delta); mean += delta;
    }
  }
  return { max, mean: mean / (left.length * 3), visible };
}

function energy(values: readonly number[]): number {
  return values[0] + values[1] + values[2];
}

/** Two equivalent scenes exercise both virtual-geometry passes while GI,
 * final gather, reflections and fog all consume the recovered PBR surface. */
export async function runOcclusionCombinedFixture() {
  const mesh = backboardMesh();
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'occlusion-combined-pbr-v1', mesh,
    coverage: 'opaque' }, { maxClusterTriangles: 24, maxClusterVertices: 48,
    groupSize: 4, simplifyRatio: 0.5, maxLevels: 8 });
  const canvases = [0, 1].map(() => {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width:64px;height:64px;display:block;image-rendering:pixelated';
    document.body.append(canvas); return canvas;
  });
  const scenes = [scene(hierarchy.source, hierarchy), scene(hierarchy.source, hierarchy)];
  const options: RendererOptions = { antiAliasing: 'none',
    occlusion: { pixelSize: 4, maxBytes: 200_000 },
    worldQueries: { maxRays: 4096, maxTotalWorkUnits: 16_777_216 },
    diffuseGi: { origin: [-2, -2, -1], spacing: [4, 4, 4], counts: [2, 2, 2],
      raysPerProbe: 16, maxDistance: 20, skyRadiance: [0.7, 0.5, 0.3], multiBounce: true,
      finalGather: { pixelSpacing: 8, maxAnchors: 64, raysPerAnchor: 2, pathSegments: 2, maxHistory: 8 } },
    reflections: { pixelSpacing: 2, maxAnchors: 1024, samplesPerAnchor: 4,
      maxDistance: 20, maxBytes: 16 * 1024 * 1024, skyRadiance: [0, 0, 0] },
    volumetricFog: { pixelSize: 4, slices: 24, maxViewDepth: 12,
      maxFroxels: 16 * 16 * 24, maxBytes: 2 * 1024 * 1024,
      global: { coefficients: { extinction: 0.12, albedo: [0, 0, 0], emission: [0, 0, 0] } } },
  };
  let reference: Renderer | undefined, tested: Renderer | undefined;
  try {
    reference = await Renderer.create(canvases[0], scenes[0].batches, options);
    tested = await Renderer.create(canvases[1], scenes[1].batches, options);
    const a = reference, b = tested;
    for (const renderer of [a, b]) {
      renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
      renderer.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false });
    }
    a.setOcclusionEnabled(false);
    b.setOcclusionEnabled(true);
    const capture = async () => {
      a.render(EYE, TARGET); b.render(EYE, TARGET);
      const [referenceHdr, actualHdr, referenceDepth, actualDepth, baseline, cut] = await Promise.all([
        a.probeHdr(PIXELS), b.probeHdr(PIXELS), a.probeDepth(DEPTH_POINTS), b.probeDepth(DEPTH_POINTS),
        a.probeGeometryLod('combined-virtual-pbr-board'), b.probeGeometryLod('combined-virtual-pbr-board'),
      ]);
      return { hdr: difference(referenceHdr, actualHdr),
        depthMax: Math.max(...referenceDepth.map((value, index) => Math.abs(value.surface - actualDepth[index].surface))),
        baselineTriangles: baseline.triangles, triangles: cut.triangles, occlusion: cut.occlusion,
        center: actualHdr.find((_, index) => index === 4 * 8 + 4),
      };
    };
    let closed = await capture(), settledFrames = 0;
    for (; settledFrames < 24 && !closed.occlusion?.culledClusters; settledFrames++) closed = await capture();
    const hiddenQuery = await b.traceRays([{ origin: [0, 0, 1], direction: [0, 0, -1], tMax: 3 }]);
    for (const pair of scenes) {
      const moved = multiply(translation(8, 0, 2), scaling(5, 5, 0.2));
      pair.wall.instances[0] = { ...pair.wall.instances[0], model: moved, normalMatrix: normalMatrix(moved) };
    }
    const recovered = await capture();
    let open = recovered;
    for (let frame = 0; frame < 8; frame++) open = await capture();
    const [surfaceOn, hdrOn, reflection, gi, gather] = await Promise.all([
      b.probeSurfaceHdr([CENTER]), b.probeHdr([CENTER]), b.probeReflections([16 * 32 + 16]),
      b.probeDiffuseGi(), b.probeDiffuseGiFinalGather([4 * 8 + 4]),
    ]);
    b.setReflectionsEnabled(false);
    b.render(EYE, TARGET);
    const reflectionOff = (await b.probeSurfaceHdr([CENTER]))[0];
    b.setReflectionsEnabled(true);
    b.setDiffuseGiEnabled(false);
    b.render(EYE, TARGET);
    const giOff = (await b.probeSurfaceHdr([CENTER]))[0];
    b.setDiffuseGiEnabled(true);
    b.setVolumetricFogEnabled(false);
    b.render(EYE, TARGET);
    const [fogOffSurface, fogOff] = await Promise.all([
      b.probeSurfaceHdr([CENTER]), b.probeHdr([CENTER]),
    ]);
    const diagnostics = b.diagnostics();
    const adapter = b.info.adapterInfo;
    const errors = [...a.errors, ...b.errors];
    a.dispose(); b.dispose();
    return { adapter, sourceTriangles: hierarchy.sourceTriangleCount, settledFrames,
      closed, recovered, open, hiddenQuery,
      contributions: { surfaceOn: surfaceOn[0], hdrOn: hdrOn[0], reflectionOff, giOff,
        fogOffSurface: fogOffSurface[0], fogOff: fogOff[0],
        reflection: reflection.anchors[0], giEnergy: gi.probes.reduce((sum, probe) =>
          sum + energy(probe.maxRadiance), 0),
        gather: gather.anchors[0] },
      diagnostics: { occlusion: diagnostics.occlusion, diffuseGi: diagnostics.diffuseGi,
        reflections: diagnostics.reflections, volumetricFog: diagnostics.volumetricFog },
      errors, ownedAfterDispose: [a.stats.ownedGpuBytes, b.stats.ownedGpuBytes] };
  } finally {
    reference?.dispose(); tested?.dispose(); canvases.forEach(canvas => canvas.remove());
  }
}
