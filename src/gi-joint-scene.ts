/** Joint dynamic-response acceptance scenes (docs/benchmarks.md L01-L04, T01).
 * Engine-owned scene contracts only; shared by joint.html and the browser
 * acceptance fixtures so cameras, events and ROIs stay frozen in one place. */
import type { InstanceState, SceneBatch } from './contracts';
import type { DiffuseGiOptions } from './diffuse-gi';
import { createCube } from './geometry';
import { m, type MaterialDefinition } from './material';
import { multiply, normalMatrix, scaling, translation, type Mat4 } from './math';

type Vec3 = [number, number, number];
type Rgba = [number, number, number, number];

const emptyDeformation = { jointCount: 0, nodes: [] };
const diffuse = (baseColor: MaterialDefinition['baseColor']): MaterialDefinition => ({
  shading: 'pbr', alphaMode: 'opaque', doubleSided: true, baseColor, metallic: 0, roughness: 0.9,
  emissive: m.constant([0, 0, 0, 1]), alphaCutoff: 0 });
const panelMaterial: MaterialDefinition = {
  shading: 'unlit', alphaMode: 'opaque', doubleSided: true, baseColor: m.constant([0, 0, 0, 1]),
  metallic: 0, roughness: 1, emissive: m.parameter(0), alphaCutoff: 0 };

export const JOINT_WHITE: Rgba = [0.8, 0.8, 0.8, 1];
export const JOINT_RED: Rgba = [0.8, 0.1, 0.08, 1];
export const JOINT_GREEN: Rgba = [0.12, 0.62, 0.16, 1];
export const JOINT_PANEL_ON: Rgba = [6, 5, 3.6, 1];
export const JOINT_PANEL_OFF: Rgba = [0, 0, 0, 1];
/** Through the room A skylight (x -3..-1, z -1..1); room B never sees the sun directly. */
export const JOINT_SUN = Object.freeze({ direction: normalize([0.2, 0.95, 0.1]), color: [3.2, 3.0, 2.7] as Vec3 });
export const JOINT_SKY: Vec3 = [0.05, 0.06, 0.08];

function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}
function box(entityId: number, center: Vec3, size: Vec3, color: Rgba = JOINT_WHITE): InstanceState {
  const model: Mat4 = multiply(translation(...center), scaling(...size));
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [], color,
    metalness: 0, roughness: 0.9, historyVersion: 0 };
}
function batch(name: string, instances: InstanceState[], definition: MaterialDefinition, parameters?: Float32Array,
  capacity = Math.max(1, instances.length)): SceneBatch {
  return { name, mesh: createCube(), capacity, instances, inputMode: 'committed', deformation: emptyDeformation,
    morphTargets: [], material: { definition, parameters },
    queryIdentity: { assetId: `gi-joint/${name}/v1`, primitiveId: 0, topologyVersion: 1 } };
}

/** Two 4 x 3 x 5 m rooms joined by a 1.2 x 2.1 m door. Room A (x < 0) has a red
 * wall and a skylight; room B (x > 0) has a green wall, a movable emissive panel
 * and only indirect light. Every wall is a solid slab; the skylight is the only
 * opening to the sky. */
// The half-cell fringe of the 10 x 4 x 6 grid covers every interior surface
// (x +/-4.2, y -0.1..3.1, z +/-2.6); probes keep >= 0.1 m from all geometry.
export const JOINT_ROOM_GRID: DiffuseGiOptions = Object.freeze({
  origin: [-3.78, 0.3, -2.6 + 2.6 / 6] as Vec3, spacing: [0.84, 0.8, 5.2 / 6] as Vec3, counts: [10, 4, 6] as Vec3,
  raysPerProbe: 128, maxDistance: 14, normalBias: 0.03, skyRadiance: JOINT_SKY, multiBounce: true,
  emissiveSamplesPerProbe: 16 });

export const JOINT_DOOR_CLOSED = box(40, [0, 1.05, 0], [0.08, 2.1, 1.2]);
export const JOINT_DOOR_OPEN = box(40, [0.2, 1.05, 1.25], [0.08, 2.1, 1.2]);
export const JOINT_PANEL_START: Vec3 = [2.0, 1.7, -2.47];
export const JOINT_PANEL_END: Vec3 = [3.2, 1.7, -2.47];
export function jointPanel(center: Vec3): InstanceState { return box(50, center, [1.2, 0.5, 0.04], [1, 1, 1, 1]); }
export const JOINT_FOREGROUND = box(60, [-0.8, 0.35, 0.8], [0.5, 0.7, 0.5], [0.85, 0.82, 0.78, 1]);
/** Fast mover crossing the room A sun patch along z at x = -2.5. */
export function jointMover(z: number): InstanceState { return box(70, [-2.5, 0.25, z], [0.5, 0.5, 0.5], [0.85, 0.85, 0.85, 1]); }
export const JOINT_MOVER_PARKED_Z = -2.1;

export interface JointRoomsScene {
  readonly batches: SceneBatch[];
  readonly door: SceneBatch;
  readonly panel: SceneBatch;
  readonly foreground: SceneBatch;
  readonly mover: SceneBatch;
}

export function createJointRoomsScene(): JointRoomsScene {
  // Slabs overlap at every joint: an exact edge contact can let query rays slip through.
  const shell = batch('joint-shell', [
    box(1, [0, -0.1, 0], [8.4, 0.2, 5.4]),
    box(2, [0, 1.5, -2.6], [8.4, 3.2, 0.2]),
    box(3, [0, 1.5, 2.6], [8.4, 3.2, 0.2]),
    box(4, [2.1, 3.1, 0], [4.2, 0.2, 5.4]),
    // Room A ceiling around the 2 x 2 m skylight (x -3..-1, z -1..1).
    box(5, [-3.6, 3.1, 0], [1.2, 0.2, 5.4]),
    box(6, [-0.45, 3.1, 0], [1.1, 0.2, 5.4]),
    box(7, [-2, 3.1, -1.85], [2.1, 0.2, 1.7]),
    box(8, [-2, 3.1, 1.85], [2.1, 0.2, 1.7]),
    // Divider with the door opening z -0.6..0.6, y 0..2.1.
    box(9, [0, 1.5, -1.6], [0.2, 3.2, 2.0]),
    box(10, [0, 1.5, 1.6], [0.2, 3.2, 2.0]),
    box(11, [0, 2.6, 0], [0.2, 1.0, 1.4]),
  ], diffuse(m.instanceColor));
  const red = batch('joint-red-wall', [box(20, [-4.1, 1.5, 0], [0.2, 3.2, 5.4])], diffuse(m.parameter(0)),
    new Float32Array(JOINT_RED));
  const green = batch('joint-green-wall', [box(21, [4.1, 1.5, 0], [0.2, 3.2, 5.4])], diffuse(m.parameter(0)),
    new Float32Array(JOINT_GREEN));
  const props = batch('joint-props', [box(30, [-3.3, 0.7, 1.75], [0.5, 1.4, 0.5]),
    box(31, [2.5, 0.3, -0.85], [0.6, 0.6, 0.6])], diffuse(m.instanceColor));
  const door = batch('joint-door', [JOINT_DOOR_OPEN], diffuse(m.instanceColor));
  const panel = batch('joint-panel', [jointPanel(JOINT_PANEL_START)], panelMaterial, new Float32Array(JOINT_PANEL_ON));
  panel.emissiveSampling = 'triangle-area';
  const foreground = batch('joint-foreground', [JOINT_FOREGROUND], diffuse(m.instanceColor));
  const mover = batch('joint-mover', [jointMover(JOINT_MOVER_PARKED_Z)], diffuse(m.instanceColor));
  return { batches: [shell, red, green, props, door, panel, foreground, mover], door, panel, foreground, mover };
}

/** L04: closed boxes on a sunlit ground. Interiors receive no light except the
 * slit box, whose 4 cm roof slit admits a sun stripe. */
export const JOINT_LEAK_GRID: DiffuseGiOptions = Object.freeze({
  origin: [-4.0, 0.35, -1.25] as Vec3, spacing: [0.5, 0.5, 0.5] as Vec3, counts: [17, 4, 6] as Vec3,
  raysPerProbe: 64, maxDistance: 12, normalBias: 0.03, skyRadiance: JOINT_SKY, multiBounce: true });

/** Interior of each 1 m box: x center +/-0.5, y 0.1..1.1, z -0.5..0.5. Centers keep
 * probe positions at least 0.1 m from every wall surface. */
export const JOINT_LEAK_BOXES = Object.freeze({
  black: { center: -2.75, wall: 0.15 },
  thin: { center: 0.25, wall: 0.02 },
  slit: { center: 2.75, wall: 0.15, slit: 0.04 },
});

function enclosure(firstId: number, cx: number, wall: number, slit = 0): InstanceState[] {
  const inner = 1, half = inner / 2, floor = 0.1, top = floor + inner;
  const outer = inner + 2 * wall;
  // Walls overlap the floor slab, the roof and each other (no edge-contact cracks).
  const bottom = floor / 2, height = top + wall - bottom, y = bottom + height / 2;
  const parts = [
    box(firstId, [cx, floor / 2, 0], [outer, floor, outer]),
    box(firstId + 1, [cx - half - wall / 2, y, 0], [wall, height, outer]),
    box(firstId + 2, [cx + half + wall / 2, y, 0], [wall, height, outer]),
    box(firstId + 3, [cx, y, -half - wall / 2], [outer, height, wall]),
    box(firstId + 4, [cx, y, half + wall / 2], [outer, height, wall]),
  ];
  if (!slit) return [...parts, box(firstId + 5, [cx, top + wall / 2, 0], [outer, wall, outer])];
  // Roof in two halves leaving a slit along x at z = 0.
  const piece = (outer - slit) / 2;
  return [...parts, box(firstId + 5, [cx, top + wall / 2, -slit / 2 - piece / 2], [outer, wall, piece]),
    box(firstId + 6, [cx, top + wall / 2, slit / 2 + piece / 2], [outer, wall, piece])];
}

export function createJointLeakScene(): { batches: SceneBatch[] } {
  const ground = batch('leak-ground', [box(1, [0, -0.1, 0], [10, 0.2, 5])], diffuse(m.instanceColor));
  const walls = batch('leak-boxes', [
    ...enclosure(10, JOINT_LEAK_BOXES.black.center, JOINT_LEAK_BOXES.black.wall),
    ...enclosure(20, JOINT_LEAK_BOXES.thin.center, JOINT_LEAK_BOXES.thin.wall),
    ...enclosure(30, JOINT_LEAK_BOXES.slit.center, JOINT_LEAK_BOXES.slit.wall, JOINT_LEAK_BOXES.slit.slit),
  ], diffuse(m.instanceColor));
  return { batches: [ground, walls] };
}

export interface JointCamera { readonly eye: Vec3; readonly target: Vec3 }
export const JOINT_CAMERAS = Object.freeze({
  roomA: { eye: [-0.4, 1.8, 2.2], target: [-2.5, 0.6, -1.0] },
  roomB: { eye: [0.8, 1.6, 2.2], target: [2.8, 0.6, -1.6] },
  /** Same room B floor patch, green wall and panel off-screen (L02). */
  roomBAway: { eye: [3.4, 2.2, -0.2], target: [1.6, 0.0, 0.2] },
  blackBox: { eye: [JOINT_LEAK_BOXES.black.center, 0.6, 0.35], target: [JOINT_LEAK_BOXES.black.center, 0.5, -0.5] },
  thinBox: { eye: [JOINT_LEAK_BOXES.thin.center, 0.6, 0.35], target: [JOINT_LEAK_BOXES.thin.center, 0.5, -0.5] },
  slitBox: { eye: [JOINT_LEAK_BOXES.slit.center, 0.6, 0.35], target: [JOINT_LEAK_BOXES.slit.center, 0.5, -0.5] },
} satisfies Record<string, JointCamera>);
export type JointCameraName = keyof typeof JOINT_CAMERAS;

/** A rectangular patch on one surface, sampled on a fixed n x n grid. */
export interface JointRoi {
  readonly name: string;
  readonly camera: JointCameraName;
  readonly center: Vec3;
  readonly u: Vec3;
  readonly v: Vec3;
  readonly normal: Vec3;
  /** Batch and entity the samples must hit (verified with world queries). */
  readonly batch: string;
  readonly entityId: number;
  readonly kind: 'direct' | 'indirect' | 'leak';
}

const floorRoi = (name: string, camera: JointCameraName, center: Vec3, half: number, kind: JointRoi['kind'],
  batchName = 'joint-shell', entityId = 1): JointRoi =>
  ({ name, camera, center, u: [half, 0, 0], v: [0, 0, half], normal: [0, 1, 0], batch: batchName, entityId, kind });

export const JOINT_ROIS: readonly JointRoi[] = Object.freeze([
  floorRoi('A.sunPatch', 'roomA', [-2.6, 0, -0.3], 0.35, 'direct'),
  { name: 'A.backWallNearRed', camera: 'roomA', center: [-3.3, 1.0, -2.5], u: [0.35, 0, 0], v: [0, 0.35, 0],
    normal: [0, 0, 1], batch: 'joint-shell', entityId: 2, kind: 'indirect' },
  { name: 'A.redWall', camera: 'roomA', center: [-4.0, 1.0, -0.6], u: [0, 0, 0.35], v: [0, 0.35, 0],
    normal: [1, 0, 0], batch: 'joint-red-wall', entityId: 20, kind: 'indirect' },
  floorRoi('B.floorNearDoor', 'roomB', [0.9, 0, -0.3], 0.25, 'indirect'),
  floorRoi('B.floor', 'roomB', [2.4, 0, 0.1], 0.25, 'indirect'),
  { name: 'B.backWallBelowPanel', camera: 'roomB', center: [2.0, 0.85, -2.5], u: [0.45, 0, 0], v: [0, 0.25, 0],
    normal: [0, 0, 1], batch: 'joint-shell', entityId: 2, kind: 'indirect' },
  floorRoi('B.floorAway', 'roomBAway', [2.4, 0, 0.1], 0.25, 'indirect'),
]);

/** Interior views: the whole frame is the ROI, sampled on a fixed grid. */
export const JOINT_LEAK_VIEWS = Object.freeze(['blackBox', 'thinBox', 'slitBox'] as const);
