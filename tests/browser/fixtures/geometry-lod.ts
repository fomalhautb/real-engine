import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { buildGeometryHierarchy, selectGeometryCut } from '../../../src/geometry-hierarchy';
import type { GeometryHierarchy } from '../../../src/geometry-hierarchy-types';
import { identity } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const NAME = 'lod-grid';
const eye: Vec3 = [0.7, 0.4, 6];
const target: Vec3 = [0, 0, 0];
const away: Vec3 = [0.7, 0.4, 12];
const baseColor: [number, number, number, number] = [0.18, 0.34, 0.63, 1];

/** Curvature makes the coarser cut a real approximation, not a second meshlet partition. */
function grid(cells = 16): MeshData {
  const side = cells + 1;
  const vertices = new ArrayBuffer(side * side * 80);
  const view = new DataView(vertices);
  const indices = new Uint32Array(cells * cells * 6);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let y = 0; y <= cells; y++) for (let x = 0; x <= cells; x++) {
    const px = 4 * x / cells - 2, py = 4 * y / cells - 2;
    const pz = 0.13 * Math.sin(2.1 * px) * Math.cos(1.7 * py);
    const dx = 0.13 * 2.1 * Math.cos(2.1 * px) * Math.cos(1.7 * py);
    const dy = -0.13 * 1.7 * Math.sin(2.1 * px) * Math.sin(1.7 * py);
    const length = Math.hypot(dx, dy, 1);
    const vertex = y * side + x, byte = vertex * 80;
    const position: Vec3 = [px, py, pz], normal: Vec3 = [-dx / length, -dy / length, 1 / length];
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(byte + axis * 4, position[axis], true);
      view.setFloat32(byte + 16 + axis * 4, normal[axis], true);
      min[axis] = Math.min(min[axis], position[axis]);
      max[axis] = Math.max(max[axis], position[axis]);
    }
    view.setFloat32(byte + 12, 1, true);
    view.setFloat32(byte + 48, 1, true);
  }
  let offset = 0;
  for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const a = y * side + x, b = a + 1, c = a + side, d = c + 1;
    indices.set([a, b, d, a, d, c], offset);
    offset += 6;
  }
  return { vertices, indices, vertexCount: side * side, bounds: { min, max } };
}

function scene(hierarchy: GeometryHierarchy): SceneBatch {
  const pose: InstanceState = { entityId: 101, entityGeneration: 1,
    model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
  return { name: NAME, mesh: hierarchy.source, geometryLod: hierarchy, capacity: 1, inputMode: 'committed',
    queryIdentity: { assetId: 'lod-grid-v1', primitiveId: 0, topologyVersion: 1 },
    instances: [pose], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: fixtureMaterial('opaque') };
}

function fixtureMaterial(alphaMode: 'opaque' | 'mask') {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit' as const,
    alphaMode, baseColor: m.parameter(0) }, parameters: new Float32Array(baseColor) };
}

function pixels(width: number, height: number): { x: number; y: number }[] {
  return [0.38, 0.5, 0.62].flatMap(v => [0.38, 0.5, 0.62].map(u =>
    ({ x: Math.floor(width * u), y: Math.floor(height * v) })));
}

let active: { canvas: HTMLCanvasElement; renderer: Renderer; hierarchy: GeometryHierarchy;
  samplePoints: { x: number; y: number }[] } | undefined;

/** These expected IDs come from the CPU reference, never the GPU's selected flags. */
export async function startGeometryLodFixture() {
  if (active) throw new Error('Geometry LOD fixture is already active');
  const canvas = document.createElement('canvas');
  canvas.dataset.geometryLodFixture = 'true';
  canvas.style.cssText = 'position:fixed;left:16px;top:16px;width:384px;height:384px;z-index:10000;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'lod-grid-v1', mesh: grid(), coverage: 'opaque' },
      { maxClusterTriangles: 24, maxClusterVertices: 48, groupSize: 4, simplifyRatio: 0.5, maxLevels: 8 });
    renderer = await Renderer.create(canvas, [scene(hierarchy)], {
      worldQueries: { maxRays: 1, maxTrianglesPerMesh: hierarchy.sourceTriangleCount, maxInstancesPerBatch: 1 },
    });
    renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const projectionScale = () => renderer!.stats.height / (2 * Math.tan(Math.PI / 6.4));
    const expected = (maxErrorPixels: number, viewEye: Vec3) => {
      const ids = selectGeometryCut(hierarchy, { eye: viewEye, projectionScale: projectionScale(), near: 0.1, maxErrorPixels });
      return { ids, indices: ids.reduce((sum, id) => sum + hierarchy.clusters[id].triangles.length, 0) };
    };
    const capture = async (maxErrorPixels: number, viewEye: Vec3 = eye) => {
      renderer!.setGeometryLod({ maxErrorPixels, frustumCulling: false, frozen: false, debug: 'material' });
      renderer!.render(viewEye, target);
      return { maxErrorPixels, viewEye, probe: await renderer!.probeGeometryLod(NAME),
        expected: expected(maxErrorPixels, viewEye) };
    };

    const fine = await capture(0);
    const intermediate = [];
    for (const threshold of [0.25, 1, 4, 16]) intermediate.push(await capture(threshold));
    const far = await capture(1, [eye[0] * 2, eye[1] * 2, eye[2] * 2]);
    const coarse = await capture(10_000);
    // Freeze only the selection camera. The draw camera can move without changing selected IDs.
    renderer.setGeometryLod({ frozen: true, frustumCulling: true });
    renderer.render(eye, away);
    const frozen = await renderer.probeGeometryLod(NAME);
    renderer.setGeometryLod({ frozen: false });
    renderer.render(eye, away);
    const culled = await renderer.probeGeometryLod(NAME);

    renderer.setGeometryLod({ maxErrorPixels: 0, frustumCulling: false, debug: 'material' });
    renderer.render(eye, target);
    const restored = await renderer.probeGeometryLod(NAME);
    active = { canvas, renderer, hierarchy, samplePoints: pixels(renderer.stats.width, renderer.stats.height) };
    return { sourceTriangles: hierarchy.sourceTriangleCount, roots: hierarchy.roots,
      levelsByCluster: hierarchy.clusters.map(cluster => cluster.level),
      fine, intermediate, far, coarse, frozen, culled, restored };
  } catch (error) { renderer?.dispose(); canvas.remove(); throw error; }
}

export async function captureGeometryLodFixtureDebug(debug: 'material' | 'clusters' | 'levels') {
  if (!active) throw new Error('Geometry LOD fixture is not active');
  const { renderer, samplePoints } = active;
  renderer.setGeometryLod({ debug });
  renderer.render(eye, target);
  const probe = await renderer.probeGeometryLod(NAME);
  const hdr = await renderer.probeHdr(samplePoints);
  const hit = await renderer.traceRays([{ origin: [0.17, 0.23, 6], direction: [0, 0, -1] }]);
  return { probe, hdr, hit, timing: renderer.stats.gpuMs, errors: [...renderer.errors] };
}

/** Masked primitives must use the exact full-detail draw while keeping the same material. */
export async function exerciseGeometryLodCoverageFallback() {
  if (!active) throw new Error('Geometry LOD fixture is not active');
  const { renderer, samplePoints } = active;
  renderer.setGeometryLod({ debug: 'material', maxErrorPixels: 0, frustumCulling: false });
  await renderer.setMaterial(NAME, fixtureMaterial('mask'));
  renderer.render(eye, target);
  const masked = { probe: await renderer.probeGeometryLod(NAME), hdr: await renderer.probeHdr(samplePoints),
    hit: await renderer.traceRays([{ origin: [0.17, 0.23, 6], direction: [0, 0, -1] }]) };
  await renderer.setMaterial(NAME, fixtureMaterial('opaque'));
  renderer.render(eye, target);
  const restored = { probe: await renderer.probeGeometryLod(NAME), hdr: await renderer.probeHdr(samplePoints),
    hit: await renderer.traceRays([{ origin: [0.17, 0.23, 6], direction: [0, 0, -1] }]) };
  return { masked, restored, errors: [...renderer.errors] };
}

export function disposeGeometryLodFixture() {
  if (!active) return null;
  const { canvas, renderer } = active;
  active = undefined;
  const errors = [...renderer.errors];
  renderer.dispose();
  canvas.remove();
  return { errors, bytesAfterDispose: renderer.stats.ownedGpuBytes };
}
