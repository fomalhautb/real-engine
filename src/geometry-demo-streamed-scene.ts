import { buildGeometryHierarchy } from './geometry-hierarchy';
import type { GeometryHierarchy } from './geometry-hierarchy-types';
import { encodeGeometryPages } from './geometry-pages';
import { buildGeometryStreamDirectory } from './geometry-stream-directory';
import { buildGeometryQueryProxy } from './geometry-query-proxy';
import { geometryGpuPoolLayout } from './geometry-gpu-pool-layout';
import { GeometryStreamRuntimeBudget, type GeometryStreamRuntimeLimits } from './geometry-stream-runtime-budget';
import type { GeometryStreamCapacityOptions } from './geometry-stream-capacity-coordinator';
import type { GeometryPageProvider, GeometryStreamDirectory } from './geometry-stream-types';
import { GEOMETRY_GPU_VERTEX_BYTES } from './geometry-gpu-page';
import { createCube, createPlane, type MeshData } from './geometry';
import { multiply, normalMatrix, rotationY, scaling, translation, type Mat4 } from './math';
import type { InstanceState, MaterialBinding, RenderBatch, SceneBatch, StreamedSceneBatch } from './contracts';
import { DEFAULT_MATERIAL_DEFINITION, m } from './material';

type Vec3 = [number, number, number];
type Color = [number, number, number, number];

export interface StreamedGalleryOptions {
  /** Unique sculptures, each with its own hierarchy, pages and stream. Default 24, maximum 48. */
  readonly objects?: number;
  /** Samples around the ring and around the tube. Defaults 192 x 48 (18,432 triangles). */
  readonly around?: number;
  readonly across?: number;
  readonly seed?: number;
  readonly maxPageBytes?: number;
  /** Artificial provider latency in milliseconds, 0..1000. Pages always come from memory. */
  readonly providerLatencyMs?: number;
  /** Required unique resident-format bytes per byte of steady-state geometry
   * pool (page slots plus location metadata, all streams). Default 4. */
  readonly targetUniqueToPoolRatio?: number;
  /** 'identity' renders unlit per-object colors for hole and mis-culling tests. */
  readonly shading?: 'lit' | 'identity';
  /** Default 30. Lower values revisit capacity admission more often. */
  readonly admissionIntervalFrames?: number;
  readonly maxConcurrentMigrations?: number;
  readonly maxRequestsPerFrame?: number;
}

export interface StreamedGalleryObject {
  readonly name: string;
  readonly primitiveKey: string;
  readonly directory: GeometryStreamDirectory;
  /** Offline build output, retained for diagnostics and the full-detail reference. */
  readonly hierarchy: GeometryHierarchy;
  readonly model: Mat4;
  readonly center: Vec3;
  readonly radius: number;
  readonly room: number;
  readonly identityColor: Color;
  readonly sourceTriangles: number;
  /** Exact packed GPU vertex records plus 32-bit indices of every cluster at every level. */
  readonly uniqueResidentBytes: number;
  /** Same data in fixed physical slots, including padding and location metadata. */
  readonly slotPaddedBytes: number;
}

/** Byte accounting for the G03 oversubscription claim. All values are planned
 * from directories before any GPU allocation; they are not measured GPU memory. */
export interface StreamedGalleryAccounting {
  readonly objects: number;
  readonly sourceTriangles: number;
  readonly clusters: number;
  readonly pages: number;
  readonly uniqueResidentBytes: number;
  readonly slotPaddedBytes: number;
  /** All streams, roots only: fixed selection/metadata plus root slots and their outputs. */
  readonly rootBytes: number;
  readonly rootPoolBytes: number;
  /** All streams fully resident under the runtime budget's own accounting. */
  readonly fullResidencyBytes: number;
  /** Upper bound of steady-state page slots plus metadata implied by maxResidentBytes. */
  readonly poolBudgetBytes: number;
  /** Runtime live budget: pool, selection with readback, paged outputs and parameters. */
  readonly maxResidentBytes: number;
  /** Adds candidate owners for maxConcurrentMigrations simultaneous full-size migrations. */
  readonly maxPeakBytes: number;
  /** The G03 ratio: unique resident-format bytes per steady-state geometry pool byte. */
  readonly uniqueToPool: number;
  /** Stricter ratios against everything the stream runtime may hold. */
  readonly uniqueToResidentBudget: number;
  readonly uniqueToPeakBudget: number;
  readonly fullResidencyToResidentBudget: number;
  readonly basis: 'directory-planned-bytes';
}

export interface StreamedGalleryPose { readonly eye: Vec3; readonly target: Vec3; readonly discontinuity: boolean }
export type StreamedGalleryTrajectory = 'walk' | 'zoom' | 'orbit' | 'turn' | 'teleport';
export const STREAMED_GALLERY_TRAJECTORIES: readonly StreamedGalleryTrajectory[] = Object.freeze(['walk', 'zoom', 'orbit', 'turn', 'teleport']);

const ROOM_DEPTH = 12;
const ROOM_WIDTH = 18;
const WALL_HEIGHT = 4.2;
const DOOR_WIDTH = 2.6;
const DOOR_HEIGHT = 3;
const PER_ROOM = 8;
// Offsets inside a room; z is measured from the room's entrance wall. The
// central aisle |x| < 1.2 stays clear for the walk and turn trajectories.
const SPOT_OFFSETS: readonly [number, number][] = [[-6.6, 3], [-2.6, 3], [2.6, 3], [6.6, 3], [-6.6, 8.6], [-2.6, 8.6], [2.6, 8.6], [6.6, 8.6]];

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Closed, watertight torus-topology surface with seeded integer harmonics, so
 * the u and v seams close exactly. Every seed yields unique geometry. */
export function streamedGalleryMesh(seed: number, around = 192, across = 48): MeshData {
  if (!Number.isSafeInteger(around) || !Number.isSafeInteger(across) || around < 8 || across < 6
    || around * across > 262_144) throw new RangeError('STREAMED_GALLERY_MESH_RESOLUTION');
  const random = mulberry32(seed);
  const pick = (low: number, high: number) => low + (high - low) * random();
  const integer = (low: number, high: number) => Math.floor(pick(low, high + 1));
  const major = pick(1.15, 1.45), tube = pick(0.34, 0.5), lobes = integer(2, 5), lobeDepth = pick(0.08, 0.2);
  const lift = pick(0.12, 0.36), liftFrequency = integer(2, 4), twist = integer(0, 2);
  const ridges = Array.from({ length: 3 }, () => ({ u: integer(3, 17), v: integer(1, 5) * (random() < 0.5 ? -1 : 1),
    amplitude: pick(0.02, 0.085), phase: pick(0, Math.PI * 2) }));
  const point = (u: number, v: number): Vec3 => {
    let ridge = 1;
    for (const r of ridges) ridge += r.amplitude * Math.cos(r.u * u + r.v * v + r.phase);
    const t = tube * ridge, w = v + twist * u;
    const radial = major * (1 + lobeDepth * Math.cos(lobes * u)) + t * Math.cos(w);
    return [radial * Math.cos(u), lift * Math.sin(liftFrequency * u) + t * Math.sin(w), radial * Math.sin(u)];
  };
  const vertexCount = around * across, vertices = new ArrayBuffer(vertexCount * 80), view = new DataView(vertices);
  const indices = new Uint32Array(vertexCount * 6);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  const h = 1e-4;
  for (let i = 0; i < around; i++) for (let j = 0; j < across; j++) {
    const u = i * Math.PI * 2 / around, v = j * Math.PI * 2 / across, p = point(u, v);
    const pu = point(u + h, v), mu = point(u - h, v), pv = point(u, v + h), mv = point(u, v - h);
    const du = [pu[0] - mu[0], pu[1] - mu[1], pu[2] - mu[2]], dv = [pv[0] - mv[0], pv[1] - mv[1], pv[2] - mv[2]];
    const n = [dv[1] * du[2] - dv[2] * du[1], dv[2] * du[0] - dv[0] * du[2], dv[0] * du[1] - dv[1] * du[0]];
    const length = Math.hypot(n[0], n[1], n[2]);
    if (!(length > 0)) throw new Error('STREAMED_GALLERY_DEGENERATE_NORMAL');
    const offset = (i * across + j) * 80;
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(offset + axis * 4, p[axis], true);
      view.setFloat32(offset + 16 + axis * 4, n[axis] / length, true);
      min[axis] = Math.min(min[axis], p[axis]); max[axis] = Math.max(max[axis], p[axis]);
    }
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 48, 1, true);
  }
  let cursor = 0;
  for (let i = 0; i < around; i++) for (let j = 0; j < across; j++) {
    const a = i * across + j, b = ((i + 1) % around) * across + j;
    const c = ((i + 1) % around) * across + (j + 1) % across, d = i * across + (j + 1) % across;
    indices.set([a, d, b, b, d, c], cursor); cursor += 6;
  }
  return { vertices, indices, vertexCount, bounds: { min, max } };
}

function instance(entityId: number, model: Mat4, color: Color, metalness: number, roughness: number): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [], color, metalness, roughness, historyVersion: 0 };
}

/** Distinct 8-bit identity colors, far apart in RGB so nearest-color decoding is exact. */
export function streamedGalleryIdentityColor(index: number): Color {
  const levels = [0.18, 0.42, 0.66, 0.9];
  const r = index % 4, g = Math.floor(index / 4) % 4, b = Math.floor(index / 16) % 4;
  return [levels[r], levels[g], levels[(b + r + g) % 4], 1];
}

/** WebGPU core minimum limits; the Renderer re-validates against its device. */
const CORE_LIMITS: GeometryStreamRuntimeLimits = { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2,
  maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 };

const emptyDeformation = { jointCount: 0, nodes: [] };

/** A procedural many-room gallery of unique streamed sculptures whose unique
 * resident-format geometry oversubscribes the whole stream GPU budget. */
export class StreamedGalleryScene {
  readonly objects: StreamedGalleryObject[];
  readonly accounting: StreamedGalleryAccounting;
  readonly runtimeBudget: GeometryStreamCapacityOptions;
  readonly rooms: number;
  readonly options: Readonly<Required<StreamedGalleryOptions>>;
  private readonly streamed: RenderBatch[];
  private readonly reference: RenderBatch[];

  private constructor(options: Readonly<Required<StreamedGalleryOptions>>, objects: StreamedGalleryObject[],
    streamed: RenderBatch[], reference: RenderBatch[], accounting: StreamedGalleryAccounting, runtimeBudget: GeometryStreamCapacityOptions) {
    this.options = options; this.objects = objects; this.streamed = streamed; this.reference = reference;
    this.accounting = accounting; this.runtimeBudget = runtimeBudget; this.rooms = Math.ceil(objects.length / PER_ROOM);
  }

  /** Source-free paged sculptures with fixed query proxies, plus resident walls. */
  get batches(): RenderBatch[] { return this.streamed; }
  /** Same scene with exact full-detail resident source meshes and no paging. */
  get referenceBatches(): RenderBatch[] { return this.reference; }
  get streamBatchNames(): string[] { return this.objects.map(object => object.name); }

  static async create(input: StreamedGalleryOptions = {}, onProgress?: (done: number, total: number) => void): Promise<StreamedGalleryScene> {
    const options: Required<StreamedGalleryOptions> = {
      objects: input.objects ?? 24, around: input.around ?? 192, across: input.across ?? 48, seed: input.seed ?? 20261030,
      maxPageBytes: input.maxPageBytes ?? 32768, providerLatencyMs: input.providerLatencyMs ?? 0,
      targetUniqueToPoolRatio: input.targetUniqueToPoolRatio ?? 4, shading: input.shading ?? 'lit',
      admissionIntervalFrames: input.admissionIntervalFrames ?? 30, maxConcurrentMigrations: input.maxConcurrentMigrations ?? 4,
      maxRequestsPerFrame: input.maxRequestsPerFrame ?? 16,
    };
    if (!Number.isSafeInteger(options.objects) || options.objects < 1 || options.objects > 48) throw new RangeError('STREAMED_GALLERY_OBJECTS');
    if (!Number.isFinite(options.providerLatencyMs) || options.providerLatencyMs < 0 || options.providerLatencyMs > 1000) {
      throw new RangeError('STREAMED_GALLERY_LATENCY');
    }
    if (!Number.isFinite(options.targetUniqueToPoolRatio) || options.targetUniqueToPoolRatio < 1 || options.targetUniqueToPoolRatio > 64) {
      throw new RangeError('STREAMED_GALLERY_RATIO');
    }
    if (options.shading !== 'lit' && options.shading !== 'identity') throw new RangeError('STREAMED_GALLERY_SHADING');
    const frozen = Object.freeze(options);
    const built: { hierarchy: GeometryHierarchy; directory: GeometryStreamDirectory; pages: Uint8Array[]; model: Mat4;
      center: Vec3; radius: number; room: number; key: string }[] = [];
    for (let index = 0; index < options.objects; index++) {
      const seed = (options.seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0;
      const mesh = streamedGalleryMesh(seed, options.around, options.across);
      const key = `streamed-gallery-v2/${options.seed}/${options.around}x${options.across}/${index}`;
      // Groups of 8 clusters leave fewer locked group borders than the default 4,
      // so far fewer groups terminate early: on these meshes roots drop from
      // ~15% to under 7% of source triangles, leaving the pool for detail.
      const hierarchy = await buildGeometryHierarchy({ primitiveKey: key, mesh, coverage: 'opaque' },
        { maxClusterTriangles: 96, groupSize: 8, simplifyRatio: 0.5, maxLevels: 16 });
      const encoded = await encodeGeometryPages(hierarchy, { maxPageBytes: options.maxPageBytes });
      const directory = await buildGeometryStreamDirectory(hierarchy, encoded);
      const room = Math.floor(index / PER_ROOM), [x, z] = SPOT_OFFSETS[index % PER_ROOM];
      const random = mulberry32(seed ^ 0x5bd1e995), scale = 0.5 + 0.12 * random();
      const center: Vec3 = [x, 1.95, room * ROOM_DEPTH + z];
      const model = multiply(translation(center[0], center[1], center[2]),
        multiply(rotationY(random() * Math.PI * 2), multiply(rotationXTilt(0.15 + 0.3 * random()), scaling(scale, scale, scale))));
      const extent = Math.max(...[0, 1, 2].map(axis => Math.max(Math.abs(mesh.bounds.min[axis]), Math.abs(mesh.bounds.max[axis]))));
      built.push({ hierarchy, directory, pages: encoded.pages.map(page => page.bytes), model, center, radius: extent * scale * Math.sqrt(3), room, key });
      onProgress?.(index + 1, options.objects);
      // Yield so a page stays responsive while building.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    // Plan the runtime budget with the same accounting the Renderer enforces.
    const sizing = new GeometryStreamRuntimeBudget(built.map(entry => ({ directory: entry.directory, capacity: 1 })),
      { maxResidentBytes: 8 * 1024 ** 3, maxPeakBytes: 8 * 1024 ** 3 }, CORE_LIMITS);
    const objects: StreamedGalleryObject[] = built.map((entry, index) => {
      const layout = geometryGpuPoolLayout(entry.directory);
      const uniqueResidentBytes = entry.directory.clusters.reduce((sum, cluster) =>
        sum + cluster.vertexCount * GEOMETRY_GPU_VERTEX_BYTES + cluster.triangleCount * 12, 0);
      return { name: `gallery-${index}`, primitiveKey: entry.key, directory: entry.directory, hierarchy: entry.hierarchy, model: entry.model, center: entry.center,
        radius: entry.radius, room: entry.room, identityColor: streamedGalleryIdentityColor(index),
        sourceTriangles: entry.directory.sourceTriangleCount, uniqueResidentBytes, slotPaddedBytes: layout.fullBytes };
    });
    const uniqueResidentBytes = objects.reduce((sum, object) => sum + object.uniqueResidentBytes, 0);
    const full = built.map((entry, index) => sizing.state(index, entry.directory.pages.length));
    const roots = built.map((entry, index) => sizing.state(index, entry.directory.rootPages.length));
    const fullResidencyBytes = full.reduce((sum, state) => sum + state.totalBytes, 0);
    const layouts = built.map(entry => geometryGpuPoolLayout(entry.directory));
    // Live cost of one slot is its page storage plus paged current/previous
    // output. Bounding the worst storage share bounds the geometry pool for any
    // slot distribution the runtime admission may choose.
    const storageShare = Math.max(...layouts.map((layout, index) => layout.bytesPerSlot / (layout.bytesPerSlot + full[index].outputBytes / full[index].slots)));
    const fixedBytes = full.reduce((sum, state) => sum + state.totalBytes, 0)
      - full.reduce((sum, state, index) => sum + state.slots * (layouts[index].bytesPerSlot + state.outputBytes / state.slots), 0);
    const metadataBytes = layouts.reduce((sum, layout) => sum + layout.metadataBytes, 0);
    const poolBudgetBytes = Math.floor(uniqueResidentBytes / options.targetUniqueToPoolRatio);
    const maxResidentBytes = Math.floor(fixedBytes + Math.max(0, poolBudgetBytes - metadataBytes) / storageShare);
    if (sizing.rootBytes >= maxResidentBytes) {
      throw new RangeError(`STREAMED_GALLERY_ROOTS_EXCEED_BUDGET:${sizing.rootBytes}:${maxResidentBytes}`);
    }
    // Migrations allocate a candidate owner while the old one is still live.
    const largest = full.map(state => state.totalBytes).sort((a, b) => b - a);
    const maxPeakBytes = maxResidentBytes + largest.slice(0, options.maxConcurrentMigrations).reduce((sum, bytes) => sum + bytes, 0);
    const accounting: StreamedGalleryAccounting = Object.freeze({
      objects: objects.length, sourceTriangles: objects.reduce((sum, object) => sum + object.sourceTriangles, 0),
      clusters: objects.reduce((sum, object) => sum + object.directory.clusters.length, 0),
      pages: objects.reduce((sum, object) => sum + object.directory.pages.length, 0),
      uniqueResidentBytes, slotPaddedBytes: objects.reduce((sum, object) => sum + object.slotPaddedBytes, 0),
      rootBytes: sizing.rootBytes, rootPoolBytes: roots.reduce((sum, state) => sum + state.poolBytes, 0),
      fullResidencyBytes, poolBudgetBytes, maxResidentBytes, maxPeakBytes,
      uniqueToPool: uniqueResidentBytes / poolBudgetBytes, uniqueToResidentBudget: uniqueResidentBytes / maxResidentBytes,
      uniqueToPeakBudget: uniqueResidentBytes / maxPeakBytes, fullResidencyToResidentBudget: fullResidencyBytes / maxResidentBytes,
      basis: 'directory-planned-bytes' as const,
    });
    const runtimeBudget: GeometryStreamCapacityOptions = Object.freeze({ maxResidentBytes, maxPeakBytes,
      intervalFrames: options.admissionIntervalFrames, maxConcurrentMigrations: options.maxConcurrentMigrations,
      maxCopyPages: 4096, maxCopyBytes: 64 * 1024 ** 2 });
    const material: MaterialBinding | undefined = options.shading === 'identity'
      ? { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.instanceColor } } : undefined;
    const sculptureColor = (index: number): Color => options.shading === 'identity' ? objects[index].identityColor
      : [0.55 + 0.4 * ((index * 37) % 11) / 10, 0.45 + 0.4 * ((index * 53) % 7) / 6, 0.5 + 0.4 * ((index * 29) % 5) / 4, 1];
    const latency = options.providerLatencyMs;
    const streamed: RenderBatch[] = [], reference: RenderBatch[] = [];
    for (let index = 0; index < built.length; index++) {
      const entry = built[index], object = objects[index], pages = entry.pages;
      const provider: GeometryPageProvider = async (page, signal) => {
        signal.throwIfAborted();
        if (latency > 0) await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, latency);
          signal.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason); }, { once: true });
        });
        const bytes = pages[page.id];
        if (!bytes) throw new RangeError('STREAMED_GALLERY_PAGE');
        return bytes;
      };
      const queryProxy = await buildGeometryQueryProxy(entry.directory, provider, { maxError: 0.02 });
      const layout = geometryGpuPoolLayout(entry.directory);
      const common = { name: object.name, capacity: 1, inputMode: 'committed' as const, deformation: emptyDeformation, morphTargets: [],
        material, queryIdentity: { assetId: entry.key, primitiveId: 0, topologyVersion: 0 },
        instances: [instance(100 + index, entry.model, sculptureColor(index), 0.35, 0.38)] };
      const batch: StreamedSceneBatch = { ...common, geometryStream: { directory: entry.directory, provider, queryProxy,
        gpuPool: { maxResidentBytes: layout.rootBytes, maxUploadBytesPerFrame: 1024 * 1024, maxUploadPagesPerFrame: 16, maxUploadWritesPerFrame: 256 },
        maxRequestsPerFrame: options.maxRequestsPerFrame } };
      streamed.push(batch);
      const full: SceneBatch = { ...common, mesh: entry.hierarchy.source };
      reference.push(full);
    }
    const architecture = galleryArchitecture(Math.ceil(objects.length / PER_ROOM), objects, options.shading);
    streamed.push(...architecture); reference.push(...architecture);
    return new StreamedGalleryScene(frozen, objects, streamed, reference, accounting, runtimeBudget);
  }

  /** Deterministic camera poses; t is seconds from the start of the trajectory. */
  pose(trajectory: StreamedGalleryTrajectory, t: number): StreamedGalleryPose {
    if (!Number.isFinite(t) || t < 0) throw new RangeError('STREAMED_GALLERY_TIME');
    const depth = this.rooms * ROOM_DEPTH, eyeHeight = 1.65;
    const focus = this.objects[Math.min(this.objects.length - 1, 1)];
    switch (trajectory) {
      case 'walk': {
        // Enter room 0 and walk the central aisle through every doorway at 1.6 m/s.
        const z = Math.min(depth - 2, -1.5 + 1.6 * t), sway = 0.9 * Math.sin(t * 0.7);
        return { eye: [0.4 * Math.sin(t * 0.45), eyeHeight, z], target: [sway * 3, 1.2, z + 6], discontinuity: false };
      }
      case 'zoom': {
        // Approach one sculpture from 8.5 m to 0.9 m and back over 16 s, inside its room.
        const phase = (t % 16) / 16, distance = 0.9 + 7.6 * (0.5 + 0.5 * Math.cos(phase * Math.PI * 2));
        const c = focus.center, length = Math.hypot(0.35, 0.12, 1);
        return { eye: [c[0] + 0.35 * distance / length, c[1] + 0.12 * distance / length, c[2] + distance / length], target: [...c], discontinuity: false };
      }
      case 'orbit': {
        const c = focus.center, angle = t * 0.5, distance = 2.4;
        return { eye: [c[0] + Math.sin(angle) * distance, c[1] + 0.9, c[2] - Math.cos(angle) * distance], target: [...c], discontinuity: false };
      }
      case 'turn': {
        // Stand in the last doorway; every 4 s turn 180 degrees within 0.25 s.
        const z = Math.max(0, (this.rooms - 1) * ROOM_DEPTH), cycle = Math.floor(t / 4), local = t - cycle * 4;
        const from = cycle % 2 ? Math.PI : 0, blend = Math.min(1, Math.max(0, (local - 3.75) / 0.25));
        const yaw = from + Math.PI * blend;
        return { eye: [0, eyeHeight, z + 0.5], target: [Math.sin(yaw) * 6, 1.2, z + 0.5 + Math.cos(yaw) * 6], discontinuity: false };
      }
      case 'teleport': {
        // Alternate between the first and last rooms every 3 s with no motion between.
        const far = Math.floor(t / 3) % 2 === 1, room = far ? this.rooms - 1 : 0, z = room * ROOM_DEPTH;
        const changed = t > 0 && Math.floor(t / 3) !== Math.floor(Math.max(0, t - 1e-6) / 3);
        return { eye: [-1.5, eyeHeight, z + 0.8], target: [1.5, 1.1, z + 7], discontinuity: changed };
      }
      default: throw new RangeError('STREAMED_GALLERY_TRAJECTORY');
    }
  }
}

function rotationXTilt(angle: number): Mat4 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}

function galleryArchitecture(rooms: number, objects: readonly StreamedGalleryObject[], shading: 'lit' | 'identity'): SceneBatch[] {
  const cube = createCube(), plane = createPlane(), depth = rooms * ROOM_DEPTH;
  const wallColor: Color = shading === 'identity' ? [0.05, 0.05, 0.05, 1] : [0.62, 0.6, 0.56, 1];
  const material: MaterialBinding | undefined = shading === 'identity'
    ? { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.instanceColor } } : undefined;
  // Box helper: createCube spans [-0.5, 0.5]^3.
  const box = (center: Vec3, size: Vec3) => multiply(translation(center[0], center[1], center[2]), scaling(size[0], size[1], size[2]));
  const walls: Mat4[] = [];
  const t = 0.3, side = (ROOM_WIDTH - DOOR_WIDTH) / 2;
  for (let room = 0; room <= rooms; room++) {
    const z = room * ROOM_DEPTH;
    if (room === 0 || room === rooms) { walls.push(box([0, WALL_HEIGHT / 2, z + (room ? t / 2 : -t / 2)], [ROOM_WIDTH, WALL_HEIGHT, t])); continue; }
    // Partition with a central doorway and a lintel.
    walls.push(box([-(DOOR_WIDTH + side) / 2, WALL_HEIGHT / 2, z], [side, WALL_HEIGHT, t]));
    walls.push(box([(DOOR_WIDTH + side) / 2, WALL_HEIGHT / 2, z], [side, WALL_HEIGHT, t]));
    walls.push(box([0, (WALL_HEIGHT + DOOR_HEIGHT) / 2, z], [DOOR_WIDTH, WALL_HEIGHT - DOOR_HEIGHT, t]));
  }
  // The entrance wall has a doorway too, so the walk starts outside room 0.
  walls[0] = box([-(DOOR_WIDTH + side) / 2, WALL_HEIGHT / 2, -t / 2], [side, WALL_HEIGHT, t]);
  walls.push(box([(DOOR_WIDTH + side) / 2, WALL_HEIGHT / 2, -t / 2], [side, WALL_HEIGHT, t]));
  walls.push(box([-ROOM_WIDTH / 2 - t / 2, WALL_HEIGHT / 2, depth / 2], [t, WALL_HEIGHT, depth + t]));
  walls.push(box([ROOM_WIDTH / 2 + t / 2, WALL_HEIGHT / 2, depth / 2], [t, WALL_HEIGHT, depth + t]));
  const plinths = objects.map(object => box([object.center[0], 0.2, object.center[2]], [1.9, 0.4, 1.9]));
  const plinthColor: Color = shading === 'identity' ? [0.02, 0.02, 0.1, 1] : [0.24, 0.27, 0.31, 1];
  const groundColor: Color = shading === 'identity' ? [0.1, 0.02, 0.02, 1] : [0.16, 0.17, 0.18, 1];
  return [
    { name: 'gallery-walls', mesh: cube, capacity: walls.length, inputMode: 'committed', deformation: emptyDeformation, morphTargets: [], material,
      instances: walls.map((model, index) => instance(1000 + index, model, wallColor, 0, 0.85)) },
    { name: 'gallery-plinths', mesh: cube, capacity: plinths.length, inputMode: 'committed', deformation: emptyDeformation, morphTargets: [], material,
      instances: plinths.map((model, index) => instance(2000 + index, model, plinthColor, 0.05, 0.6)) },
    { name: 'gallery-ground', mesh: plane, capacity: 1, inputMode: 'committed', deformation: emptyDeformation, morphTargets: [], material,
      instances: [instance(3000, box([0, 0, depth / 2], [ROOM_WIDTH + 2, 1, depth + 4]), groundColor, 0, 0.92)] },
  ];
}
