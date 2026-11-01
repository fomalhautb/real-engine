import type { InstanceState, MaterialBinding, RenderBatch, SceneBatch, StreamedSceneBatch } from '../../../src/contracts';
import { AnimationRig } from '../../../src/animation';
import { compileDeformation, evaluateDeformation, type DeformationGraph, type MorphTarget } from '../../../src/deformation';
import { createCube, createPlane, type MeshData } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import type { GeometryHierarchy } from '../../../src/geometry-hierarchy-types';
import { geometryGpuPoolLayout } from '../../../src/geometry-gpu-pool-layout';
import { encodeGeometryPages } from '../../../src/geometry-pages';
import { buildGeometryQueryProxy, type GeometryQueryProxy } from '../../../src/geometry-query-proxy';
import { buildGeometryStreamDirectory } from '../../../src/geometry-stream-directory';
import type { GeometryStreamDirectory } from '../../../src/geometry-stream-types';
import { GeometryStreamRuntimeBudget } from '../../../src/geometry-stream-runtime-budget';
import type { GeometryStreamSource } from '../../../src/gpu-geometry-stream';
import { identity, lookAt, multiply, normalMatrix, perspective, scaling, transformPoint, translation, type Mat4 } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { buildMorphTarget } from '../../../src/morph-assets';
import { Renderer, type RendererOptions } from '../../../src/renderer';
import { resolveSunShadowOptions, sunShadowProjection } from '../../../src/sun-shadow';
import type { WorldHit, WorldQueryOptions, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
type Rgb = [number, number, number];
type Kind = 'stream-proxy' | 'stream-source' | 'lod' | 'masked' | 'masked-lod' | 'masked-stream' | 'skinned' | 'morph';
type Grid = { width: number; height: number; cellsX: number; cellsY: number };
interface PanelSpec { name: string; kind: Kind; x: number; color: Rgb; entityId: number }
/** One world point of one batch: 'covered' must show that batch, 'empty' must not. */
interface Sample { batch: string; label: string; world: Vec3; expect: 'covered' | 'empty'; margin: number; core: boolean }
interface Failure { frame: string; check: string; batch: string; label: string; detail: string }
interface Pose { time: number; bones: Mat4[]; weight: number; dx: number }

const WIDTH = 960, HEIGHT = 400, NEAR = 0.1, FAR = 150, FOV = Math.PI / 3.2;
const EYE_A: Vec3 = [0, 6, 10], EYE_B: Vec3 = [30, 6, 10], TARGET: Vec3 = [0, 1.5, -1];
// Points from the surface towards the light, so shadows fall towards +z onto
// visible ground in front of the panels; the ground point of p is (x, 0, z + y).
const SUN: Vec3 = [0, Math.SQRT1_2, -Math.SQRT1_2], SUN_COLOR: Vec3 = [3, 3, 3];
const SHADOW = { resolution: 2048, focusBounds: { min: [-11, -0.5, -3.5] as Vec3, max: [11, 4.5, 4] as Vec3 } };
const PANEL_Z = -1, PANEL_BASE = 0.2, SKIN_PIVOT = 1.5, BUMP = 0.02;
// Hides every panel from EYE_B and lies outside the EYE_A frustum.
const OCCLUDER = { min: [13, 2.5, 5.9] as Vec3, max: [26, 7, 6.1] as Vec3, color: [0.35, 0.22, 0.1] as Rgb };
const PANELS: readonly PanelSpec[] = [
  { name: 'skinned', kind: 'skinned', x: -7.7, color: [0.95, 0.95, 0.1], entityId: 701 },
  { name: 'stream-proxy', kind: 'stream-proxy', x: -5.5, color: [0.9, 0.1, 0.1], entityId: 702 },
  { name: 'stream-source', kind: 'stream-source', x: -3.3, color: [0.95, 0.55, 0.05], entityId: 703 },
  { name: 'lod', kind: 'lod', x: -1.1, color: [0.1, 0.85, 0.15], entityId: 704 },
  { name: 'masked', kind: 'masked', x: 1.1, color: [0.1, 0.2, 0.95], entityId: 705 },
  { name: 'masked-lod', kind: 'masked-lod', x: 3.3, color: [0.05, 0.85, 0.9], entityId: 706 },
  { name: 'masked-stream', kind: 'masked-stream', x: 5.5, color: [0.9, 0.1, 0.85], entityId: 707 },
  { name: 'morph', kind: 'morph', x: 7.7, color: [0.55, 0.15, 0.95], entityId: 708 },
];
const VIRTUAL = ['stream-proxy', 'stream-source', 'lod', 'masked-lod', 'masked-stream'] as const;
const GRID = {
  panel: { width: 1.6, height: 2, cellsX: 32, cellsY: 40 },
  skin: { width: 1.6, height: 3, cellsX: 16, cellsY: 30 },
  morph: { width: 1.6, height: 2, cellsX: 16, cellsY: 20 },
} satisfies Record<string, Grid>;
const HIERARCHY_OPTIONS = { maxClusterTriangles: 24, maxClusterVertices: 48, groupSize: 4, simplifyRatio: 0.5, maxLevels: 8 };
const STATIC_GRAPH: DeformationGraph = { jointCount: 0, nodes: [] };
const SKIN_GRAPH: DeformationGraph = { jointCount: 2, nodes: [{ kind: 'skin' }] };
const MORPH_GRAPH: DeformationGraph = { jointCount: 0, nodes: [{ kind: 'morph', target: 0, weight: 0 }] };
const MIB = 1024 * 1024;

const smooth = (edge0: number, edge1: number, value: number) => {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
};

/** Vertical grid in the local xy-plane facing +z with UVs over the panel. With
 * skin, joint 1 owns the part above a smooth band centred on SKIN_PIVOT. */
function panel(grid: Grid, skin = false): MeshData {
  const columns = grid.cellsX + 1, rows = grid.cellsY + 1, count = columns * rows;
  const vertices = new ArrayBuffer(count * 80), view = new DataView(vertices);
  const uvs = new Float32Array(count * 2), indices = new Uint32Array(grid.cellsX * grid.cellsY * 6);
  for (let j = 0; j < rows; j++) for (let i = 0; i < columns; i++) {
    const x = (i / grid.cellsX - 0.5) * grid.width, y = j / grid.cellsY * grid.height, at = (j * columns + i) * 80;
    const dx = 3 * BUMP * Math.cos(3 * x) * Math.cos(3 * y), dy = -3 * BUMP * Math.sin(3 * x) * Math.sin(3 * y);
    const length = Math.hypot(dx, dy, 1), weight = skin ? smooth(SKIN_PIVOT - 0.3, SKIN_PIVOT + 0.3, y) : 0;
    [x, y, BUMP * Math.sin(3 * x) * Math.cos(3 * y), 1, -dx / length, -dy / length, 1 / length, 0]
      .forEach((value, word) => view.setFloat32(at + word * 4, value, true));
    view.setUint32(at + 36, skin ? 1 : 0, true);
    view.setFloat32(at + 48, 1 - weight, true); view.setFloat32(at + 52, weight, true);
    uvs.set([i / grid.cellsX, 1 - j / grid.cellsY], (j * columns + i) * 2);
  }
  for (let j = 0; j < grid.cellsY; j++) for (let i = 0; i < grid.cellsX; i++) {
    const a = j * columns + i, b = a + 1, c = a + columns, d = c + 1;
    indices.set([a, b, d, a, d, c], (j * grid.cellsX + i) * 6);
  }
  return { vertices, indices, vertexCount: count, uvs,
    bounds: { min: [-grid.width / 2, 0, -BUMP], max: [grid.width / 2, grid.height, BUMP] } };
}

function vertexAt(grid: Grid, x: number, y: number): number {
  const i = (x / grid.width + 0.5) * grid.cellsX, j = y / grid.height * grid.cellsY;
  if (Math.abs(i - Math.round(i)) > 1e-6 || Math.abs(j - Math.round(j)) > 1e-6) throw new Error('SILENT_DROP_SAMPLE_OFF_GRID');
  return Math.round(j) * (grid.cellsX + 1) + Math.round(i);
}

function restVertex(mesh: MeshData, index: number) {
  const view = new DataView(mesh.vertices), at = index * 80, f = (offset: number) => view.getFloat32(at + offset, true);
  return { position: [f(0), f(4), f(8)] as Vec3, normal: [f(16), f(20), f(24)] as Vec3,
    joints: [0, 1, 2, 3].map(k => view.getUint32(at + 32 + k * 4, true)) as [number, number, number, number],
    weights: [0, 1, 2, 3].map(k => f(48 + k * 4)) as [number, number, number, number] };
}

function instance(entityId: number, model: Mat4, extra: Partial<InstanceState> = {}): InstanceState {
  return { entityId, entityGeneration: 1, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0, ...extra };
}

/** Exact identity color: unlit output is base + emissive, with no lighting or fog. */
function unlit(color: Rgb, doubleSided = true): MaterialBinding {
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided, baseColor: m.parameter(0) },
    parameters: new Float32Array([...color, 1]) };
}

/** 8x8 white cutout whose centre 4x4 texels are transparent: local x in [-0.4, 0.4], y in [0.5, 1.5]. */
function cutout(color: Rgb): MaterialBinding {
  const pixels = new Uint8Array(8 * 8 * 4);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    pixels.set([255, 255, 255, x >= 2 && x < 6 && y >= 2 && y < 6 ? 0 : 255], (y * 8 + x) * 4);
  }
  return { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, alphaMode: 'mask', alphaCutoff: 0.5,
    baseColor: m.multiply(m.parameter(0), m.texture(0)) }, parameters: new Float32Array([...color, 1]),
  textures: [{ width: 8, height: 8, colorSpace: 'srgb', pixels }],
  samplers: [{ addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }] };
}

const panelModel = (x: number) => translation(x, PANEL_BASE, PANEL_Z);

/** Engine-owned rig: joint 1 slides the upper skin part left; node 2 drives the morph weight. */
function createRig(): AnimationRig {
  const node = (parent: number | null, t: Vec3, weights: number[] = []) => ({ parent, translation: t,
    rotation: [0, 0, 0, 1] as [number, number, number, number], scale: [1, 1, 1] as Vec3, weights });
  return new AnimationRig([node(null, [0, 0, 0]), node(0, [-1, SKIN_PIVOT, 0]), node(null, [0, 0, 0], [0.75])], [{
    name: 'silent-drop-pose', duration: 2, tracks: [
      { node: 1, path: 'translation', interpolation: 'LINEAR', components: 3, times: new Float32Array([0, 1, 2]),
        values: new Float32Array([-0.8, SKIN_PIVOT, 0, -1.2, SKIN_PIVOT, 0, -0.8, SKIN_PIVOT, 0]) },
      { node: 2, path: 'weights', interpolation: 'LINEAR', components: 1, times: new Float32Array([0, 1, 2]),
        values: new Float32Array([0.55, 0.95, 0.55]) },
    ] }]);
}

function samplePose(rig: AnimationRig, time: number): Pose {
  const sample = rig.sample(time, 0, true);
  const bones = [new Float32Array(sample.world[0]), multiply(sample.world[1], translation(0, -SKIN_PIVOT, 0))];
  return { time, bones, weight: sample.weights[2][0], dx: bones[1][12] };
}

interface Assets {
  panelMesh: MeshData; skinMesh: MeshData; morphMesh: MeshData; morphTarget: MorphTarget;
  hierarchy: GeometryHierarchy; directory: GeometryStreamDirectory; proxy: GeometryQueryProxy;
  provider: (page: { id: number }, signal: AbortSignal) => Promise<Uint8Array<ArrayBuffer>>;
  requested: Set<number>;
}

async function buildAssets(): Promise<Assets> {
  const panelMesh = panel(GRID.panel), skinMesh = panel(GRID.skin, true), morphMesh = panel(GRID.morph);
  const morphTarget = buildMorphTarget(morphMesh, ([, y]) => [1, 0.8 * y, 0]);
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'silent-drop-panel-v1', mesh: panelMesh, coverage: 'opaque' },
    HIERARCHY_OPTIONS);
  const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: 8192 });
  const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
  const requested = new Set<number>();
  // Every stream receives its own copy: decode workers may take ownership of the bytes.
  const provider = async (page: { id: number }, signal: AbortSignal) => {
    signal.throwIfAborted(); requested.add(page.id); return encoded.pages[page.id].bytes.slice();
  };
  const proxy = await buildGeometryQueryProxy(directory, provider, { topologyVersion: 1 });
  requested.clear();
  return { panelMesh, skinMesh, morphMesh, morphTarget, hierarchy, directory, proxy, provider, requested };
}

function streamSource(assets: Assets, residentBytes: number): GeometryStreamSource {
  return { directory: assets.directory, provider: assets.provider, maxRequestsPerFrame: 32,
    gpuPool: { maxResidentBytes: residentBytes, maxUploadBytesPerFrame: 1 << 20, maxUploadWritesPerFrame: 128, maxUploadPagesPerFrame: 64 } };
}

function sceneBatches(assets: Assets, pose: Pose): RenderBatch[] {
  const source = assets.hierarchy.source, roots = geometryGpuPoolLayout(assets.directory).rootBytes;
  const common = (spec: PanelSpec) => ({ name: spec.name, capacity: 2, inputMode: 'committed' as const,
    instances: [instance(spec.entityId, panelModel(spec.x))], deformation: STATIC_GRAPH, morphTargets: [] });
  const byKind = (spec: PanelSpec): RenderBatch => {
    switch (spec.kind) {
      case 'stream-proxy': return { ...common(spec), material: unlit(spec.color),
        geometryStream: { ...streamSource(assets, roots), queryProxy: assets.proxy } } satisfies StreamedSceneBatch;
      case 'stream-source': return { ...common(spec), mesh: source, material: unlit(spec.color), geometryStream: streamSource(assets, roots) };
      case 'lod': return { ...common(spec), mesh: source, material: unlit(spec.color), geometryLod: assets.hierarchy };
      case 'masked': return { ...common(spec), mesh: assets.panelMesh, material: cutout(spec.color) };
      case 'masked-lod': return { ...common(spec), mesh: source, material: cutout(spec.color), geometryLod: assets.hierarchy };
      case 'masked-stream': return { ...common(spec), mesh: source, material: cutout(spec.color), geometryStream: streamSource(assets, roots) };
      // Capacity 8 also routes queries through the batch-local instance BVH.
      case 'skinned': return { ...common(spec), capacity: 8, mesh: assets.skinMesh, material: unlit(spec.color), deformation: SKIN_GRAPH,
        instances: [instance(spec.entityId, panelModel(spec.x), { bones: pose.bones })] };
      case 'morph': return { ...common(spec), capacity: 8, mesh: assets.morphMesh, material: unlit(spec.color), deformation: MORPH_GRAPH,
        morphTargets: [assets.morphTarget], instances: [instance(spec.entityId, panelModel(spec.x), { deformScalars: [pose.weight] })] };
    }
  };
  const ground: SceneBatch = { name: 'ground', mesh: createPlane(), capacity: 1, inputMode: 'committed',
    instances: [instance(900, scaling(40, 1, 40), { color: [0.5, 0.5, 0.5, 1], roughness: 0.85 })],
    deformation: STATIC_GRAPH, morphTargets: [], material: { definition: DEFAULT_MATERIAL_DEFINITION } };
  const size = OCCLUDER.max.map((value, axis) => value - OCCLUDER.min[axis]) as Vec3;
  const center = OCCLUDER.max.map((value, axis) => (value + OCCLUDER.min[axis]) / 2) as Vec3;
  const occluder: SceneBatch = { name: 'occluder', mesh: createCube(), capacity: 1, inputMode: 'committed',
    instances: [instance(950, multiply(translation(...center), scaling(...size)))], deformation: STATIC_GRAPH,
    morphTargets: [], material: unlit(OCCLUDER.color, false) };
  return [ground, ...PANELS.map(byKind), occluder];
}

const OPAQUE_POINTS: [number, number][] = [-0.5, 0, 0.5].flatMap(x => [0.4, 1, 1.6].map(y => [x, y] as [number, number]));
const MASK_SOLID: [number, number, number][] = [[-0.6, 0.3, 0.18], [0.6, 0.3, 0.18], [-0.6, 1, 0.18], [0.6, 1, 0.18],
  [-0.6, 1.7, 0.18], [0.6, 1.7, 0.18], [0, 0.25, 0.22], [0, 1.75, 0.22]];
const MASK_HOLE: [number, number, number][] = [[0, 1, 0.38], [-0.2, 0.75, 0.18], [0.2, 1.25, 0.18]];

/** World samples for the batch state of one frame. Covered points are mesh
 * vertices, deformed through the independent CPU oracle. */
function samplesFor(assets: Assets, pose: Pose, maskedState: Record<string, boolean>): Sample[] {
  const skin = compileDeformation(SKIN_GRAPH), morph = compileDeformation(MORPH_GRAPH);
  const out: Sample[] = [];
  for (const spec of PANELS) {
    const model = panelModel(spec.x), world = (local: Vec3) => transformPoint(model, local) as Vec3;
    const add = (label: string, local: Vec3, expect: Sample['expect'], margin: number, core: boolean) =>
      out.push({ batch: spec.name, label, world: world(local), expect, margin, core });
    const vertex = (x: number, y: number) => restVertex(assets.panelMesh, vertexAt(GRID.panel, x, y)).position;
    if (spec.kind === 'stream-proxy' || spec.kind === 'stream-source' || spec.kind === 'lod') {
      for (const [x, y] of OPAQUE_POINTS) add(`solid ${x},${y}`, vertex(x, y), 'covered', 0.3, (x === 0 && y === 1) || (x === -0.5 && y === 0.4) || (x === 0.5 && y === 1.6));
    } else if (spec.kind === 'masked' || spec.kind === 'masked-lod' || spec.kind === 'masked-stream') {
      const masked = maskedState[spec.name];
      for (const [x, y, margin] of MASK_SOLID) add(`solid ${x},${y}`, vertex(x, y), 'covered', margin,
        (x === -0.6 && y === 1) || (x === 0 && y === 1.75) || (x === 0.6 && y === 0.3));
      for (const [x, y, margin] of MASK_HOLE) {
        if (masked) add(`hole ${x},${y}`, [x, y, 0], 'empty', margin, true);
        else add(`filled hole ${x},${y}`, vertex(x, y), 'covered', margin, x === 0);
      }
    } else if (spec.kind === 'skinned') {
      const deform = (x: number, y: number) => {
        const index = vertexAt(GRID.skin, x, y);
        return evaluateDeformation(skin, restVertex(assets.skinMesh, index), index, [], pose.bones, []).position;
      };
      for (const [x, y, core] of [[-0.4, 0.5, false], [0.4, 0.5, true], [0, 1, false]] as const) add(`lower ${x},${y}`, deform(x, y), 'covered', 0.3, core);
      for (const [x, y, core] of [[0, 2.4, true], [-0.3, 2.6, false], [0.3, 2.2, false]] as const) add(`upper ${x},${y}`, deform(x, y), 'covered', 0.3, core);
      add('deformed only -0.6,2.5', deform(-0.6, 2.5), 'covered', 0.18, true);
      add('rest only 0.5,2.5', [0.5, 2.5, 0], 'empty', 0.38, true);
    } else {
      const deform = (x: number, y: number) => {
        const index = vertexAt(GRID.morph, x, y);
        return evaluateDeformation(morph, restVertex(assets.morphMesh, index), index, [assets.morphTarget], [], [pose.weight]).position;
      };
      for (const [x, y, core] of [[-0.4, 0.5, true], [0, 1, false], [0.4, 1.5, false]] as const) add(`morphed ${x},${y}`, deform(x, y), 'covered', 0.3, core);
      add('deformed only 0.6,1.8', deform(0.6, 1.8), 'covered', 0.18, true);
      add('rest only -0.65,0.4', [-0.65, 0.4, 0], 'empty', 0.33, true);
    }
  }
  return out;
}

interface CoveragePoint { batch: string; world: Vec3; expect: 'covered' | 'empty' }

/** Dense 0.1 m grids per batch: surface points at least 0.1 m inside the expected
 * silhouette or cutout edge, and casters whose ground shadow lies at least 0.2 m
 * inside its edge. Rest positions vacated by the deformation must be empty. */
function coverageFor(assets: Assets, pose: Pose, maskedState: Record<string, boolean>) {
  const surface: CoveragePoint[] = [], shadow: CoveragePoint[] = [];
  const steps = (from: number, to: number) => Array.from({ length: Math.round((to - from) * 10) + 1 }, (_, i) => Number((from + i * 0.1).toFixed(6)));
  const skin = compileDeformation(SKIN_GRAPH), morph = compileDeformation(MORPH_GRAPH);
  // Signed L-infinity distance to the cutout hole: positive outside, negative inside.
  const hole = (x: number, y: number) => Math.max(Math.abs(x) - 0.4, 0.5 - y, y - 1.5);
  for (const spec of PANELS) {
    const model = panelModel(spec.x);
    const push = (list: CoveragePoint[], local: Vec3, expect: CoveragePoint['expect']) =>
      list.push({ batch: spec.name, world: transformPoint(model, local) as Vec3, expect });
    if (spec.kind === 'skinned' || spec.kind === 'morph') {
      const skinned = spec.kind === 'skinned', grid = skinned ? GRID.skin : GRID.morph, mesh = skinned ? assets.skinMesh : assets.morphMesh;
      const deform = (x: number, y: number) => {
        const index = vertexAt(grid, x, y), rest = restVertex(mesh, index);
        return (skinned ? evaluateDeformation(skin, rest, index, [], pose.bones, [])
          : evaluateDeformation(morph, rest, index, [assets.morphTarget], [], [pose.weight])).position;
      };
      const top = grid.height - 0.1;
      for (const x of steps(-0.7, 0.7)) for (const y of steps(0.1, top)) push(surface, deform(x, y), 'covered');
      for (const x of steps(-0.6, 0.6)) for (const y of steps(0.2, top - 0.1)) push(shadow, deform(x, y), 'covered');
      // The skin slides its upper part left by at least 0.8 m; the morph shifts right by at least 0.55 m.
      const vacated = skinned ? { x: [0.3, 0.7], y: [2.1, 2.8] } : { x: [-0.7, -0.5], y: [0.1, 1.9] };
      for (const x of steps(vacated.x[0], vacated.x[1])) for (const y of steps(vacated.y[0], vacated.y[1])) {
        push(surface, [x, y, 0], 'empty'); push(shadow, [x, y, 0], 'empty');
      }
      continue;
    }
    const masked = spec.kind.startsWith('masked') && maskedState[spec.name];
    const vertex = (x: number, y: number) => restVertex(assets.panelMesh, vertexAt(GRID.panel, x, y)).position;
    for (const [list, inset, xs, ys] of [[surface, 0.1, steps(-0.7, 0.7), steps(0.1, 1.9)], [shadow, 0.2, steps(-0.6, 0.6), steps(0.2, 1.8)]] as const) {
      for (const x of xs) for (const y of ys) {
        const distance = masked ? hole(x, y) : Infinity;
        if (distance >= inset - 1e-9) push(list, vertex(x, y), 'covered');
        else if (distance <= -inset + 1e-9) push(list, [x, y, 0], 'empty');
      }
    }
  }
  return { surface, shadow };
}

const viewProjection = (eye: Vec3, width: number, height: number) =>
  multiply(perspective(FOV, width / height, NEAR, FAR), lookAt(eye, TARGET, [0, 1, 0]));
const viewDistance = (depth: number) => NEAR / (1 - depth * (FAR - NEAR) / FAR);

function clip(vp: Mat4, p: readonly number[]) {
  const w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15], ndc = transformPoint(vp, p);
  return { ndc, inside: w > 0 && Math.abs(ndc[0]) < 1 && Math.abs(ndc[1]) < 1 && ndc[2] >= 0 && ndc[2] <= 1 };
}
const pixelOf = (ndc: readonly number[], width: number, height: number) =>
  ({ x: Math.floor((ndc[0] * 0.5 + 0.5) * width), y: Math.floor((0.5 - ndc[1] * 0.5) * height) });

/** Pixel of p when every in-plane neighbour at `margin` metres projects at least 1.5 pixels away. */
function marginPixel(vp: Mat4, p: Vec3, margin: number, axes: [Vec3, Vec3], width: number, height: number) {
  const center = clip(vp, p);
  if (!center.inside) return null;
  const sx = (center.ndc[0] * 0.5 + 0.5) * width, sy = (0.5 - center.ndc[1] * 0.5) * height;
  for (const axis of axes) for (const sign of [-1, 1]) {
    const q = clip(vp, p.map((value, i) => value + sign * margin * axis[i]));
    if (!q.inside) return null;
    if (Math.hypot((q.ndc[0] * 0.5 + 0.5) * width - sx, (0.5 - q.ndc[1] * 0.5) * height - sy) < 1.5) return null;
  }
  return pixelOf(center.ndc, width, height);
}

function segmentHitsBox(a: readonly number[], b: readonly number[], min: readonly number[], max: readonly number[]): boolean {
  let t0 = 0, t1 = 1;
  for (let axis = 0; axis < 3; axis++) {
    const d = b[axis] - a[axis];
    if (Math.abs(d) < 1e-9) { if (a[axis] < min[axis] || a[axis] > max[axis]) return false; continue; }
    let ta = (min[axis] - a[axis]) / d, tb = (max[axis] - a[axis]) / d;
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
    if (t0 > t1) return false;
  }
  return true;
}
/** true: behind the occluder, false: clear line of sight, null: within 5 cm of its silhouette. */
function occluded(eye: Vec3, p: Vec3): boolean | null {
  const grow = (sign: number) => [OCCLUDER.min.map(v => v - sign * 0.05), OCCLUDER.max.map(v => v + sign * 0.05)];
  const [innerMin, innerMax] = grow(-1), [outerMin, outerMax] = grow(1);
  if (segmentHitsBox(eye, p, innerMin, innerMax)) return true;
  return segmentHitsBox(eye, p, outerMin, outerMax) ? null : false;
}

const maxChannelError = (actual: readonly number[], expected: Rgb) => Math.max(...expected.map((value, c) => Math.abs(actual[c] - value)));
const luminance = (rgb: readonly number[]) => (rgb[0] + rgb[1] + rgb[2]) / 3;
const round = (value: number, digits = 4) => Number(value.toFixed(digits));
const PLANE_AXES: [Vec3, Vec3] = [[1, 0, 0], [0, 1, 0]];
const GROUND_AXES: [Vec3, Vec3] = [[1, 0, 0], [0, 0, 1]];
const COLOR = Object.fromEntries(PANELS.map(spec => [spec.name, spec.color])) as Record<string, Rgb>;
const ENTITY = Object.fromEntries(PANELS.map(spec => [spec.name, spec.entityId])) as Record<string, number>;

type ProbeSummary = { name: string; indices: number; triangles: number; overflow: boolean; clusters: number; fallbackClusters: number;
  occlusion: { deferredClusters: number; culledClusters: number; recoveredClusters: number } };

/** Masked, skinned and morph batches share one Renderer with paged streams (one
 * source-free proxy, one full-source), resident GPU LOD, an oversubscribed
 * capacity budget and two-phase occlusion. Every check compares against
 * independently computed geometry; readbacks are test-only diagnostics. */
export async function runGeometrySilentDropFixture(config: { staticTraversal: NonNullable<WorldQueryOptions['staticTraversal']> }) {
  const assets = await buildAssets(), rig = createRig();
  let pose = samplePose(rig, 0.25);
  const maskedState: Record<string, boolean> = { masked: true, 'masked-lod': true, 'masked-stream': true };
  const batches = sceneBatches(assets, pose);
  const byName = (name: string) => batches.find(batch => batch.name === name)!;
  const layout = geometryGpuPoolLayout(assets.directory);
  const streams = batches.filter(batch => batch.geometryStream);
  // The renderer sizes the identical inputs against the device; state() is limit-independent here.
  const sizing = new GeometryStreamRuntimeBudget(streams.map(batch => ({ directory: assets.directory, capacity: batch.capacity })),
    { maxResidentBytes: 256 * MIB, maxPeakBytes: 512 * MIB },
    { maxBufferSize: 256 * MIB, maxStorageBufferBindingSize: 128 * MIB, maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 });
  const roots = assets.directory.rootPages.length, pages = assets.directory.pages.length;
  if (pages <= roots * 2) throw new Error('SILENT_DROP_NEEDS_STREAMED_DETAIL');
  const rootState = sizing.state(0, roots).totalBytes, fullState = sizing.state(0, pages).totalBytes;
  // Room for every root plus half of one stream's detail, while two Opaque streams request all of theirs.
  const maxResidentBytes = streams.length * rootState + Math.floor((fullState - rootState) / 2), maxPeakBytes = 2 * maxResidentBytes;
  const oversubscription = { maxResidentBytes, maxPeakBytes, rootStateBytes: rootState, fullStateBytes: fullState,
    opaqueDemandBytes: 2 * fullState + rootState, allOpaqueDemandBytes: streams.length * fullState, pages, roots,
    poolLayout: { rootBytes: layout.rootBytes, fullBytes: layout.fullBytes } };

  const panelSection = document.createElement('section'); panelSection.id = 'silent-drop-visual';
  panelSection.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#101418;color:#e8eef3;padding:16px;font:14px system-ui;overflow:hidden';
  panelSection.innerHTML = '<h1 style="font-size:20px;margin:0 0 6px">Masked, skinned and morph batches beside paged streams, GPU LOD, budget and occlusion</h1>';
  const canvases = [0, 1].map(() => {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `width:${WIDTH}px;height:${HEIGHT}px;display:block;margin:4px 0`;
    panelSection.appendChild(canvas); return canvas;
  });
  document.body.appendChild(panelSection);

  const options: RendererOptions = { worldQueries: { maxRays: 512, staticTraversal: config.staticTraversal }, occlusion: {},
    sunShadow: SHADOW, geometryRuntimeBudget: { maxResidentBytes, maxPeakBytes, intervalFrames: 2, maxCopyPages: 4096, maxCopyBytes: 64 * MIB } };
  let main: Renderer | undefined, reference: Renderer | undefined;
  const failures: Failure[] = [];
  let failureCount = 0;
  const fail = (failure: Failure) => { failureCount++; if (failures.length < 40) failures.push(failure); };
  try {
    main = await Renderer.create(canvases[0], batches, options);
    const groundOnly = sceneBatches(assets, pose).filter(batch => batch.name === 'ground');
    reference = await Renderer.create(canvases[1], groundOnly, { sunShadow: SHADOW });
    const a = main, ref = reference;
    for (const renderer of [a, ref]) renderer.setSun({ direction: SUN, color: SUN_COLOR });
    a.setGeometryLod({ maxErrorPixels: 0, frustumCulling: true });
    const length = Math.hypot(...SUN), light = sunShadowProjection(SUN.map(value => value / length) as Vec3, resolveSunShadowOptions(SHADOW));
    const lightDepth = (p: readonly number[]) => transformPoint(light.matrix, p)[2];

    const setPose = (next: Pose) => {
      pose = next;
      const skinned = byName('skinned'), morph = byName('morph');
      skinned.instances = [instance(ENTITY.skinned, panelModel(-7.7), { bones: next.bones })];
      morph.instances = [instance(ENTITY.morph, panelModel(7.7), { deformScalars: [next.weight] })];
    };
    const expectedDrawCalls = () => batches.length * 2 + 1 + batches.filter(batch => (batch.geometryLod || batch.geometryStream)
      && !maskedState[batch.name]).length;
    let skippedFrames = 0;
    /** One rendered frame; a whole frame withheld by stream backpressure is retried, never probed. */
    const frame = (eye: Vec3, renderer: Renderer = a) => {
      const before = renderer.stats.frames;
      renderer.render(eye, TARGET);
      if (renderer.errors.length) throw new Error('SILENT_DROP_RENDER: ' + renderer.errors.join('\n'));
      return renderer.stats.frames === before + 1;
    };
    const renderFrame = async (eye: Vec3) => {
      for (let attempt = 0; attempt < 16; attempt++) {
        if (frame(eye)) return;
        skippedFrames++; await new Promise(resolve => setTimeout(resolve, 4));
      }
      throw new Error('SILENT_DROP_BACKPRESSURE');
    };
    const probeVirtual = async (): Promise<ProbeSummary[]> => {
      const out: ProbeSummary[] = [];
      for (const name of VIRTUAL) {
        const cut = await a.probeGeometryLod(name);
        out.push({ name, indices: cut.indices, triangles: cut.triangles, overflow: cut.overflow, clusters: cut.selectedClusters.length,
          fallbackClusters: cut.selectedClusters.filter(cluster => cluster.fallback).length, occlusion: { ...cut.occlusion } });
      }
      return out;
    };
    const budgetState = () => {
      const diagnostics = a.diagnostics(), budget = diagnostics.geometryRuntimeBudget as { enabled: boolean; phase?: string; error?: string | null;
        liveBytes?: number; accountedBytes?: number; peakBytes?: number; migrations?: number; deferred?: number };
      return { phase: budget.phase ?? 'disabled', error: budget.error ?? null, liveBytes: budget.liveBytes ?? 0,
        accountedBytes: budget.accountedBytes ?? 0, peakBytes: budget.peakBytes ?? 0, migrations: budget.migrations ?? 0,
        slots: diagnostics.geometryStreams.map(stream => stream.pool.slotCount),
        resident: diagnostics.geometryStreams.map(stream => stream.pool.residentPages.length),
        decoding: diagnostics.geometryDecoding ? { active: diagnostics.geometryDecoding.activeRequests, waiting: diagnostics.geometryDecoding.waitingRequests } : null };
    };

    // Settle the shared budget at camera A. Rendering correctness must not depend on it.
    let settledFrames = 0, stable = 0;
    for (; settledFrames < 300 && stable < 4; settledFrames++) {
      await renderFrame(EYE_A); await a.probeHdr([{ x: 0, y: 0 }]);
      const state = budgetState();
      stable = state.phase === 'idle' && state.decoding?.active === 0 && state.decoding.waiting === 0 ? stable + 1 : 0;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    frame(EYE_A, ref); await ref.probeHdr([{ x: 0, y: 0 }]);
    const width = a.stats.width, height = a.stats.height;
    if (width !== WIDTH || height !== HEIGHT || ref.stats.width !== WIDTH) throw new Error('SILENT_DROP_CANVAS_SIZE');

    // ---- Static audit at camera A: main view, depth, sun shadow map, shadowed ground, world queries.
    const staticAudit = async (frameLabel: string) => {
      const vp = viewProjection(EYE_A, width, height), samples = samplesFor(assets, pose, maskedState);
      const counts: Record<string, Record<string, number>> = {};
      const count = (batch: string, check: string) => { (counts[batch] ??= {})[check] = (counts[batch][check] ?? 0) + 1; };
      const pixels = samples.map(sample => marginPixel(vp, sample.world, sample.margin, PLANE_AXES, width, height));
      if (pixels.some(pixel => !pixel)) throw new Error('SILENT_DROP_STATIC_SAMPLE_UNRESOLVED: ' + samples.filter((_, i) => !pixels[i]).map(s => s.batch + ' ' + s.label).join(', '));
      const grounds = samples.map(sample => [sample.world[0], 0, sample.world[2] + sample.world[1]] as Vec3);
      const groundPixels = grounds.map((g, i) => marginPixel(vp, g, samples[i].margin, GROUND_AXES, width, height));
      if (groundPixels.some(pixel => !pixel)) throw new Error('SILENT_DROP_GROUND_SAMPLE_UNRESOLVED');
      const [hdr, depth, groundHdr, groundReference] = [await a.probeHdr(pixels as { x: number; y: number }[]),
        await a.probeDepth(samples.map(sample => sample.world)),
        await a.probeHdr(groundPixels as { x: number; y: number }[]), await ref.probeHdr(groundPixels as { x: number; y: number }[])];
      const rays: WorldRay[] = samples.map(sample => ({ origin: [sample.world[0], sample.world[1], sample.world[2] + 2.5], direction: [0, 0, -1], tMax: 5 }));
      const query = await a.traceRays(rays);
      const identityError: Record<string, number> = {};
      samples.forEach((sample, i) => {
        const color = COLOR[sample.batch], covered = sample.expect === 'covered';
        const error = maxChannelError(hdr[i], color), report = (check: string, detail: string) =>
          fail({ frame: frameLabel, check, batch: sample.batch, label: sample.label, detail });
        // Main view: exact identity color, or anything else where the batch must be absent.
        if (covered ? error > 0.02 : error < 0.15) report('main-view', `hdr ${hdr[i].slice(0, 3).map(v => round(v)).join(',')} pixel ${JSON.stringify(pixels[i])}`);
        if (covered) identityError[sample.batch] = Math.max(identityError[sample.batch] ?? 0, error);
        count(sample.batch, covered ? 'main-covered' : 'main-empty');
        const expected = viewDistance(transformPoint(vp, sample.world)[2]), actual = viewDistance(depth[i].surface);
        if (covered ? Math.abs(actual - expected) > 0.1 : actual - expected < 0.5) report('depth', `expected ${round(expected)} actual ${round(actual)}`);
        // Sun shadow map: the caster's own light depth, or the ground behind an empty point.
        const gap = (depth[i].shadow - lightDepth(sample.world)) * light.depth;
        if (covered ? Math.abs(gap) > 0.06 : gap < 0.5 * sample.world[1] * Math.SQRT2) report('shadow-map', `light-space gap ${round(gap)} m`);
        count(sample.batch, covered ? 'shadow-caster' : 'shadow-gap');
        // Shadowed ground against the same view of a ground-only renderer. The sun has
        // no x component, so panel shadow columns are disjoint and the reference is
        // the frame without this caster.
        const ratio = luminance(groundHdr[i]) / Math.max(1e-4, luminance(groundReference[i]));
        if (covered ? ratio > 0.5 : ratio < 0.85 || ratio > 1.15) report('ground-shadow', `ratio ${round(ratio)} at ${JSON.stringify(groundPixels[i])}`);
        count(sample.batch, covered ? 'ground-shadowed' : 'ground-lit');
        // World query: identity and deformed position, or a ray that passes through.
        const hit: WorldHit | null = query.hits[i];
        if (covered) {
          const kind = sample.batch === 'stream-proxy' ? 'fixed-query-proxy' : 'source';
          if (!hit || hit.batch !== sample.batch || hit.entityId !== ENTITY[sample.batch] || Math.abs(hit.distance - 2.5) > 0.06
            || maxChannelError(hit.baseColor, color) > 0.02 || hit.geometry.kind !== kind) {
            report('world-query', hit ? `${hit.batch}/${hit.entityId} d=${round(hit.distance)} ${hit.geometry.kind} base ${hit.baseColor.map(v => round(v)).join(',')}` : 'miss');
          }
        } else if (hit) report('world-query', `unexpected ${hit.batch}/${hit.entityId} d=${round(hit.distance)}`);
        count(sample.batch, covered ? 'query-hit' : 'query-pass');
      });
      return { counts, identityError, samples: samples.length, queryFrame: query.frame };
    };
    // ---- Dense coverage at camera A: whole surfaces and, optionally, their ground shadows.
    const probeMany = async (renderer: Renderer, pixels: readonly { x: number; y: number }[]) => {
      const out: number[][] = [];
      for (let start = 0; start < pixels.length; start += 1024) out.push(...await renderer.probeHdr(pixels.slice(start, start + 1024)));
      return out;
    };
    const coverageAudit = async (frameLabel: string, withShadow: boolean) => {
      const vp = viewProjection(EYE_A, width, height), grid = coverageFor(assets, pose, maskedState);
      const counts: Record<string, Record<string, number>> = {};
      const count = (batch: string, check: string) => { (counts[batch] ??= {})[check] = (counts[batch][check] ?? 0) + 1; };
      const project = (p: Vec3) => {
        const projected = clip(vp, p);
        if (!projected.inside) throw new Error('SILENT_DROP_COVERAGE_OUTSIDE_VIEW');
        return pixelOf(projected.ndc, width, height);
      };
      const label = (p: Vec3) => p.map(value => round(value, 2)).join(',');
      const surfacePixels = grid.surface.map(point => project(point.world)), surface = await probeMany(a, surfacePixels);
      grid.surface.forEach((point, i) => {
        const error = maxChannelError(surface[i], COLOR[point.batch]), covered = point.expect === 'covered';
        if (covered ? error > 0.02 : error < 0.15) fail({ frame: frameLabel, check: 'coverage-surface', batch: point.batch, label: label(point.world),
          detail: `${point.expect} hdr ${surface[i].slice(0, 3).map(v => round(v)).join(',')} pixel ${JSON.stringify(surfacePixels[i])}` });
        count(point.batch, covered ? 'surface-covered' : 'surface-empty');
      });
      if (withShadow) {
        const groundPixels = grid.shadow.map(point => project([point.world[0], 0, point.world[2] + point.world[1]]));
        const [lit, unshadowed] = [await probeMany(a, groundPixels), await probeMany(ref, groundPixels)];
        grid.shadow.forEach((point, i) => {
          const ratio = luminance(lit[i]) / Math.max(1e-4, luminance(unshadowed[i])), covered = point.expect === 'covered';
          if (covered ? ratio > 0.5 : ratio < 0.85 || ratio > 1.15) fail({ frame: frameLabel, check: 'coverage-shadow', batch: point.batch,
            label: label(point.world), detail: `${point.expect} ratio ${round(ratio)} pixel ${JSON.stringify(groundPixels[i])}` });
          count(point.batch, covered ? 'ground-shadowed' : 'ground-lit');
        });
      }
      return counts;
    };

    const captures: { label: string; src: string }[] = [];
    const capture = (label: string) => { captures.push({ label, src: canvases[0].toDataURL('image/png') }); };
    // A fresh render immediately before capturing keeps the swapchain image alive.
    await renderFrame(EYE_A); capture('Camera A after settling: every batch drawn, cutouts open, deformed poses');
    const staticResult = await staticAudit('static A');
    const staticCoverage = await coverageAudit('static A', true);
    const staticVirtual = await probeVirtual();
    const staticDiagnostics = a.diagnostics();
    const lodActive = Object.fromEntries((staticDiagnostics.geometryLod?.batches ?? []).map(batch => [batch.name, batch.active]));
    const representations = Object.fromEntries(staticDiagnostics.geometryStreams.map(stream => [stream.name, stream.sourceRepresentation]));

    // Shared deformed output, against an oracle that never reads the renderer's packed state.
    const skinTop = vertexAt(GRID.skin, -0.6, 2.5), morphTop = vertexAt(GRID.morph, 0.6, 1.8);
    const verification = await a.verifyDeformation({ sampleVertices: [skinTop, morphTop] });
    const oracle = (batch: string, vertex: number) => batch === 'skinned'
      ? transformPoint(panelModel(-7.7), evaluateDeformation(compileDeformation(SKIN_GRAPH), restVertex(assets.skinMesh, vertex), vertex, [], pose.bones, []).position)
      : transformPoint(panelModel(7.7), evaluateDeformation(compileDeformation(MORPH_GRAPH), restVertex(assets.morphMesh, vertex), vertex, [assets.morphTarget], [], [pose.weight]).position);
    const deformationSamples = [['skinned', skinTop], ['morph', morphTop]].map(([batch, vertex]) => {
      const sample = verification.samples.find(entry => entry.batch === batch && entry.vertexIndex === vertex);
      const expected = oracle(batch as string, vertex as number);
      return { batch, vertex, error: sample ? Math.max(...expected.map((value, axis) => Math.abs(value - sample.current[axis]))) : Infinity,
        restDistance: Math.hypot(...expected.map((value, axis) => value - transformPoint(panelModel(batch === 'skinned' ? -7.7 : 7.7),
          restVertex(batch === 'skinned' ? assets.skinMesh : assets.morphMesh, vertex as number).position)[axis])) };
    });

    // ---- Load: oversubscribed streaming, two-phase occlusion, occluding camera paths,
    // material switches and animated deformation, a few probes per frame.
    type Step = { label: string; eye: Vec3; key?: boolean; coverage?: boolean; capture?: string; before?: () => Promise<void> | void };
    const setMasked = (masked: boolean) => async () => {
      for (const name of ['masked-lod', 'masked-stream']) {
        const spec = PANELS.find(entry => entry.name === name)!;
        await a.setMaterial(name, masked ? cutout(spec.color) : unlit(spec.color));
        maskedState[name] = masked;
      }
    };
    const steps: Step[] = [];
    const add = (label: string, eye: Vec3, extra: Omit<Step, 'label' | 'eye'> = {}) => steps.push({ label, eye, ...extra });
    for (let i = 0; i < 4; i++) add(`A animated ${i}`, EYE_A);
    for (const x of [6, 12, 18, 24]) add(`sweep A to B x=${x}`, [x, 6, 10]);
    for (let i = 0; i < 3; i++) add(`hold B ${i}`, EYE_B, i === 2 ? { key: true, capture: 'Camera B: the occluder hides every panel' } : {});
    add('teleport to A', EYE_A, { key: true, coverage: true, capture: 'First frame after teleporting back to A (two-phase recovery)' });
    add('hold A', EYE_A);
    add('masked LOD/stream switched to Opaque', EYE_A, { before: setMasked(false), key: true, coverage: true, capture: 'Masked LOD and stream switched to Opaque' });
    add('opaque hold A', EYE_A);
    for (const x of [15, 30]) add(`opaque sweep x=${x}`, [x, 6, 10]);
    add('opaque hold B 0', EYE_B); add('opaque hold B 1', EYE_B, { key: true });
    add('opaque teleport to A', EYE_A, { key: true, coverage: true });
    add('switched back to Masked', EYE_A, { before: setMasked(true), key: true, coverage: true });
    add('occlusion disabled', EYE_A, { before: () => a.setOcclusionEnabled(false), coverage: true });
    add('occlusion re-enabled', EYE_A, { before: () => a.setOcclusionEnabled(true), coverage: true });
    add('occlusion re-enabled 1', EYE_A);
    add('back to B 0', EYE_B); add('back to B 1', EYE_B);
    for (const x of [24, 18, 12, 6, 0]) add(`sweep B to A x=${x}`, [x, 6, 10], x === 0 ? { capture: 'Gradual disocclusion back at A' } : {});

    const frames: { label: string; eye: Vec3; checked: Record<string, number>; occludedChecks: number; skipped: number; failures: number;
      queries: number; shadowChecks: number; drawCalls: number; expectedDrawCalls: number; deformationDispatches: number;
      budget: ReturnType<typeof budgetState>; sharedMembers: number | null; virtual?: ProbeSummary[]; coverage?: Record<string, Record<string, number>> }[] = [];
    let time = pose.time;
    for (const step of steps) {
      await step.before?.();
      time += 0.13; setPose(samplePose(rig, time));
      await renderFrame(step.eye);
      if (step.capture) capture(step.capture);
      const vp = viewProjection(step.eye, width, height), samples = samplesFor(assets, pose, maskedState).filter(sample => sample.core);
      const before = failureCount, checked: Record<string, number> = {};
      let skipped = 0, occludedChecks = 0;
      const probes: { sample: Sample; pixel: { x: number; y: number }; hidden: boolean }[] = [];
      for (const sample of samples) {
        const pixel = marginPixel(vp, sample.world, sample.margin, PLANE_AXES, width, height), hidden = occluded(step.eye, sample.world);
        if (!pixel || hidden === null) { skipped++; continue; }
        probes.push({ sample, pixel, hidden });
      }
      const hdr = await a.probeHdr(probes.map(probe => probe.pixel));
      probes.forEach(({ sample, pixel, hidden }, i) => {
        const error = maxChannelError(hdr[i], hidden ? OCCLUDER.color : COLOR[sample.batch]);
        const bad = hidden || sample.expect === 'covered' ? error > 0.02 : error < 0.15;
        if (bad) fail({ frame: step.label, check: hidden ? 'occluder' : 'main-view', batch: sample.batch, label: sample.label,
          detail: `hdr ${hdr[i].slice(0, 3).map(v => round(v)).join(',')} pixel ${JSON.stringify(pixel)}` });
        if (hidden) occludedChecks++; else checked[sample.batch] = (checked[sample.batch] ?? 0) + 1;
      });
      // A few camera-independent world queries and shadow-map lookups per frame.
      const querySamples = samples.filter(sample => sample.expect === 'empty' || sample.label.startsWith('deformed') || sample.label.startsWith('solid 0,1')
        || sample.label.startsWith('solid -0.6,1') || sample.label.startsWith('filled hole 0,'));
      const query = await a.traceRays(querySamples.map(sample => ({ origin: [sample.world[0], sample.world[1], sample.world[2] + 2.5] as Vec3,
        direction: [0, 0, -1] as Vec3, tMax: 5 })));
      querySamples.forEach((sample, i) => {
        const hit = query.hits[i], covered = sample.expect === 'covered';
        const bad = covered ? !hit || hit.batch !== sample.batch || hit.entityId !== ENTITY[sample.batch] || Math.abs(hit.distance - 2.5) > 0.06 : !!hit;
        if (bad) fail({ frame: step.label, check: 'world-query', batch: sample.batch, label: sample.label,
          detail: hit ? `${hit.batch}/${hit.entityId} d=${round(hit.distance)}` : 'miss' });
      });
      const shadowSamples = querySamples.filter(sample => clip(vp, sample.world).inside);
      const depth = shadowSamples.length ? await a.probeDepth(shadowSamples.map(sample => sample.world)) : [];
      shadowSamples.forEach((sample, i) => {
        const gap = (depth[i].shadow - lightDepth(sample.world)) * light.depth;
        const bad = sample.expect === 'covered' ? Math.abs(gap) > 0.06 : gap < 0.5 * sample.world[1] * Math.SQRT2;
        if (bad) fail({ frame: step.label, check: 'shadow-map', batch: sample.batch, label: sample.label, detail: `light-space gap ${round(gap)} m` });
      });
      const drawCalls = a.stats.drawCalls, expected = expectedDrawCalls();
      if (drawCalls !== expected) fail({ frame: step.label, check: 'draw-calls', batch: '*', label: 'stats.drawCalls', detail: `${drawCalls} != ${expected}` });
      const virtual = step.key ? await probeVirtual() : undefined;
      const coverage = step.coverage ? await coverageAudit(step.label, false) : undefined;
      frames.push({ label: step.label, eye: step.eye, checked, occludedChecks, skipped, failures: failureCount - before,
        queries: querySamples.length, shadowChecks: shadowSamples.length, drawCalls, expectedDrawCalls: expected,
        deformationDispatches: a.stats.deformationDispatches, budget: budgetState(),
        sharedMembers: a.diagnostics().worldQueries?.sharedStatic?.members ?? null, virtual, coverage });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    // After the load the static audit must still pass, at a new pose.
    const finalResult = await staticAudit('final A');
    const finalCoverage = await coverageAudit('final A', true);
    const finalVirtual = await probeVirtual();
    const finalDiagnostics = a.diagnostics();

    const gallery = document.createElement('div');
    gallery.style.cssText = 'display:grid;grid-template-columns:repeat(2,480px);gap:6px 12px';
    for (const shot of captures) {
      const figure = document.createElement('figure'); figure.style.cssText = 'margin:0';
      const image = document.createElement('img'); image.src = shot.src; image.width = 480; image.height = 200; image.style.cssText = 'display:block';
      const caption = document.createElement('figcaption'); caption.textContent = shot.label; caption.style.cssText = 'font-size:12px';
      figure.append(image, caption); gallery.appendChild(figure);
    }
    for (const canvas of canvases) canvas.remove();
    panelSection.appendChild(gallery);
    await Promise.all(Array.from(gallery.querySelectorAll('img'), image => image.decode()));

    const adapter = a.info.adapterInfo, errors = [...a.errors, ...ref.errors];
    const coordinator = (a as unknown as { geometryCapacity?: { dispose(): Promise<void> } }).geometryCapacity;
    a.dispose(); ref.dispose();
    await coordinator?.dispose(); await new Promise(resolve => setTimeout(resolve, 0));
    return {
      adapter, staticTraversal: config.staticTraversal, oversubscription, settledFrames, skippedFrames,
      static: staticResult, staticVirtual, final: finalResult, finalVirtual, lodActive, representations,
      coverage: { static: staticCoverage, final: finalCoverage },
      deformation: { maxError: verification.maxError, boundsViolations: verification.boundsViolations, vertices: verification.vertices, samples: deformationSamples },
      frames, failures, failureCount, requestedPages: assets.requested.size,
      budget: finalDiagnostics.geometryRuntimeBudget, occlusion: finalDiagnostics.occlusion,
      decoding: finalDiagnostics.geometryDecoding, decoderAfterDispose: a.diagnostics().geometryDecoding,
      worldQueries: { staticTraversal: finalDiagnostics.worldQueries?.staticTraversal ?? null,
        sharedStatic: finalDiagnostics.worldQueries?.sharedStatic ?? null,
        geometryRepresentations: finalDiagnostics.worldQueries?.geometryRepresentations ?? [] },
      errors, ownedAfterDispose: [a.stats.ownedGpuBytes, ref.stats.ownedGpuBytes],
    };
  } finally { main?.dispose(); reference?.dispose(); }
}

/** Each combination the renderer cannot draw must be refused when it is created
 * (or when the material is set), never accepted and then left undrawn. */
export async function runGeometrySilentDropRejections() {
  const assets = await buildAssets(), source = assets.hierarchy.source;
  const roots = geometryGpuPoolLayout(assets.directory).rootBytes;
  const restPose = samplePose(createRig(), 0.25);
  const skinned = (spec: Partial<SceneBatch>): SceneBatch => ({ name: 'deformed', capacity: 1, inputMode: 'committed', mesh: source,
    deformation: { jointCount: 1, nodes: [{ kind: 'skin' }] }, morphTargets: [], material: unlit([1, 1, 0]),
    instances: [instance(1, panelModel(0), { bones: [identity()] })], ...spec });
  const morphOf = (mesh: MeshData) => ({ deformation: MORPH_GRAPH, morphTargets: [buildMorphTarget(mesh, () => [0.2, 0, 0])],
    instances: [instance(1, panelModel(0), { deformScalars: [0.5] })] });
  const wave = { deformation: { jointCount: 0, nodes: [{ kind: 'wave' as const, amplitude: 0.1, frequency: 1, phase: 0 }] },
    instances: [instance(1, panelModel(0), { deformScalars: [0] })] };
  const proxyStream = { ...streamSource(assets, roots), queryProxy: assets.proxy };
  /** The same batch as a source-free paged surface whose fixed proxy is its only resident representation. */
  const sourceFree = (batch: SceneBatch, extra: Partial<StreamedSceneBatch> = {}): StreamedSceneBatch => {
    const { mesh: _mesh, geometryLod: _lod, geometryStream: _stream, ...shared } = batch;
    return { ...shared, ...extra, geometryStream: proxyStream };
  };
  const plainVirtual: SceneBatch = { name: 'virtual', capacity: 1, inputMode: 'committed', mesh: source, geometryLod: assets.hierarchy,
    deformation: STATIC_GRAPH, morphTargets: [], material: unlit([0, 1, 0]), instances: [instance(9, panelModel(3))] };
  const cases: { name: string; expected: string; scene: () => RenderBatch[]; options?: RendererOptions }[] = [
    { name: 'skinned + geometryLod', expected: 'GEOMETRY_LOD_REQUIRES_STATIC_SOURCE', scene: () => [skinned({ geometryLod: assets.hierarchy })] },
    { name: 'morph + geometryLod', expected: 'GEOMETRY_LOD_REQUIRES_STATIC_SOURCE', scene: () => [skinned({ ...morphOf(source), geometryLod: assets.hierarchy })] },
    { name: 'wave + geometryLod', expected: 'GEOMETRY_LOD_REQUIRES_STATIC_SOURCE', scene: () => [skinned({ ...wave, geometryLod: assets.hierarchy })] },
    { name: 'skinned + full-source stream', expected: 'GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE', scene: () => [skinned({ geometryStream: streamSource(assets, roots) })] },
    { name: 'morph + full-source stream', expected: 'GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE', scene: () => [skinned({ ...morphOf(source), geometryStream: streamSource(assets, roots) })] },
    { name: 'skinned + source-free proxy stream', expected: 'GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE', scene: () => [sourceFree(skinned({}))] },
    { name: 'morph + source-free proxy stream', expected: 'GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE',
      scene: () => [sourceFree(skinned(morphOf(assets.proxy.mesh)))] },
    { name: 'Masked + source-free proxy stream', expected: 'GEOMETRY_PROXY_MASK_REQUIRES_FULL_SOURCE',
      scene: () => [sourceFree(plainVirtual, { material: cutout([1, 0, 1]) })] },
    { name: 'geometryLod + geometryStream', expected: 'GEOMETRY_STREAM_REQUIRES_STATIC_SOURCE',
      scene: () => [{ ...plainVirtual, geometryStream: streamSource(assets, roots) }] },
    { name: 'occlusion without virtual geometry', expected: 'OCCLUSION_REQUIRES_VIRTUAL_GEOMETRY', options: { occlusion: {} },
      scene: () => [{ ...plainVirtual, name: 'masked', mesh: assets.panelMesh, geometryLod: undefined, material: cutout([0, 0, 1]) },
        { ...skinned({ mesh: assets.skinMesh, deformation: SKIN_GRAPH, instances: [instance(2, panelModel(-3), { bones: restPose.bones })] }), name: 'skin' },
        { ...skinned({ mesh: assets.morphMesh, ...morphOf(assets.morphMesh) }), name: 'morph' }] },
  ];
  const results: { name: string; expected: string; error: string | null; createdRenderer: boolean }[] = [];
  for (const entry of cases) {
    const canvas = document.createElement('canvas'); canvas.style.cssText = 'width:64px;height:64px;position:fixed;left:0;top:0';
    document.body.appendChild(canvas);
    let error: string | null = null, created = false;
    try { const renderer = await Renderer.create(canvas, entry.scene(), entry.options); created = true; renderer.dispose(); }
    catch (reason) { error = String(reason); }
    finally { canvas.remove(); }
    results.push({ name: entry.name, expected: entry.expected, error, createdRenderer: created });
  }
  // Offline builds refuse coverage and deformation that only the full-detail path can draw.
  const offline: { name: string; expected: string; error: string | null }[] = [];
  for (const [name, expected, input] of [
    ['Masked coverage hierarchy', 'GEOMETRY_UNSUPPORTED_COVERAGE', { primitiveKey: 'silent-drop-mask', mesh: assets.panelMesh, coverage: 'mask' as const }],
    ['skinned hierarchy', 'GEOMETRY_UNSUPPORTED_DEFORMATION', { primitiveKey: 'silent-drop-skin', mesh: assets.skinMesh, coverage: 'opaque' as const, deformation: SKIN_GRAPH }],
    ['morph hierarchy', 'GEOMETRY_UNSUPPORTED_DEFORMATION', { primitiveKey: 'silent-drop-morph', mesh: assets.morphMesh, coverage: 'opaque' as const, deformation: MORPH_GRAPH }],
  ] as const) {
    let error: string | null = null;
    try { await buildGeometryHierarchy(input, HIERARCHY_OPTIONS); } catch (reason) { error = String(reason); }
    offline.push({ name, expected, error });
  }
  // A refused Masked switch on a source-free proxy must leave the batch drawn as before.
  const canvas = document.createElement('canvas'); canvas.style.cssText = `width:${WIDTH}px;height:${HEIGHT}px;position:fixed;left:0;top:0`;
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    const proxyBatch = sourceFree(plainVirtual, { name: 'proxy', material: unlit([0.9, 0.1, 0.1]), instances: [instance(11, panelModel(0))] });
    renderer = await Renderer.create(canvas, [proxyBatch], { worldQueries: { maxRays: 4 }, occlusion: {} });
    renderer.setSun({ direction: SUN, color: SUN_COLOR });
    const center = transformPoint(panelModel(0), [0, 1, 0]) as Vec3;
    const look = async () => {
      renderer!.render(EYE_A, TARGET);
      const vp = viewProjection(EYE_A, renderer!.stats.width, renderer!.stats.height);
      const [hdr] = await renderer!.probeHdr([pixelOf(transformPoint(vp, center), renderer!.stats.width, renderer!.stats.height)]);
      const query = await renderer!.traceRays([{ origin: [center[0], center[1], center[2] + 2.5], direction: [0, 0, -1], tMax: 5 }]);
      const cut = await renderer!.probeGeometryLod('proxy');
      return { hdr: hdr.slice(0, 3), hit: query.hits[0] ? { batch: query.hits[0].batch, entityId: query.hits[0].entityId } : null,
        indices: cut.indices, materialRevision: renderer!.diagnostics().materialRevision };
    };
    const before = await look();
    let maskError: string | null = null;
    try { await renderer.setMaterial('proxy', cutout([0.9, 0.1, 0.1])); } catch (reason) { maskError = String(reason); }
    const after = await look();
    const errors = [...renderer.errors];
    renderer.dispose();
    return { adapter: renderer.info.adapterInfo, results, offline, proxySwitch: { before, after, maskError }, errors,
      ownedAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
