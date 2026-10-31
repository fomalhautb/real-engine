import type { Bounds3 } from './deformation';
import { lookAt, multiply, type Mat4 } from './math';

/**
 * Engine-owned bounded point and spot lights, shared by every lighting path.
 *
 * Radiometry. `Renderer.setSun` takes `color` as the irradiance E⊥ that the sun
 * delivers to a surface facing it, in scene-linear units; reflected radiance is
 * BRDF × E⊥ × cosθ. A local light's `color × intensity` is a radiant intensity I
 * in the same irradiance units times square metres (per steradian). At distance
 * d it delivers
 *
 *   E⊥(d) = color × intensity × W(d) × C / max(d², LOCAL_LIGHT_MIN_DISTANCE²),
 *   W(d)  = saturate(1 − (d / range)⁴)²,
 *
 * where C is the spot cone factor (1 for point lights). Therefore a light with
 * color 1 and intensity 3.5 at 1 m gives a facing surface the same irradiance as
 * a sun color of 3.5. If a project maps one irradiance unit to k lux, intensity
 * × k is the on-axis luminous intensity in candela; an isotropic point light then
 * emits Φ = 4π × intensity × k lumens. A spot uses on-axis intensity: narrowing
 * its cone does not concentrate power (it is not power-normalised).
 */
export type LocalLightVec3 = readonly [number, number, number];
export type LocalLightKind = 'point' | 'spot';

export interface LocalLightDescriptor {
  readonly kind: LocalLightKind;
  /** World position in metres. */
  readonly position: LocalLightVec3;
  /** Spot only: propagation direction from the light into the scene. Unlike
   * `Renderer.setSun`, this does not point toward the light. Normalised. */
  readonly direction?: LocalLightVec3;
  /** Spot only: half-angles in radians, 0 <= inner <= outer <= 80°. */
  readonly innerConeAngle?: number;
  readonly outerConeAngle?: number;
  /** Linear RGB, nonnegative. */
  readonly color: LocalLightVec3;
  /** Radiant intensity scale; see the module radiometry note. */
  readonly intensity: number;
  /** Finite influence radius in metres. Contribution is exactly zero for d >= range. */
  readonly range: number;
  /** Reserves shadow atlas views (spot 1, point 6) while the light exists. */
  readonly castShadows?: boolean;
  /** Disabled lights keep their slot and shadow reservation but cost no shading. */
  readonly enabled?: boolean;
}

export interface LocalLightHandle {
  /** Stable slot index for the light's lifetime. */
  readonly id: number;
  /** Incremented when a slot is released; stale handles are rejected. */
  readonly generation: number;
}

export interface ResolvedLocalLight {
  readonly handle: LocalLightHandle;
  readonly kind: LocalLightKind;
  readonly position: LocalLightVec3;
  /** Unit spot axis; [0, -1, 0] for point lights (unused). */
  readonly direction: LocalLightVec3;
  readonly innerConeAngle: number;
  readonly outerConeAngle: number;
  readonly color: LocalLightVec3;
  readonly intensity: number;
  readonly range: number;
  readonly castShadows: boolean;
  readonly enabled: boolean;
  /** Per-light version: 1 when added, incremented by every effective change. */
  readonly version: number;
}

export interface LocalLightOptions {
  /** Live light slots, including disabled lights: 1–256 (default 64). */
  readonly maxLights?: number;
  /** Lights with castShadows, enabled or not: 0–8 (default 4). */
  readonly maxShadowedLights?: number;
  /** depth32float atlas layer edge: power of two 128–2048 (default 512). */
  readonly shadowResolution?: number;
  /** Atlas layers; a spot uses 1 view and a point 6 cube faces: 1–48
   * (default min(48, 6 × maxShadowedLights), or 1 without shadowed lights). */
  readonly maxShadowViews?: number;
  /** Atlas byte budget checked before allocation (default 32 MiB). */
  readonly maxShadowBytes?: number;
  /** Receiver offset along the geometric normal, in shadow texels at the receiver: 0–16 (default 1.5). */
  readonly shadowNormalBias?: number;
  /** Receiver depth offset toward the light, in shadow texels at the receiver: 0–16 (default 1). */
  readonly shadowDepthBias?: number;
}

export interface ResolvedLocalLightOptions {
  readonly maxLights: number;
  readonly maxShadowedLights: number;
  readonly shadowResolution: number;
  readonly maxShadowViews: number;
  readonly maxShadowBytes: number;
  readonly shadowNormalBias: number;
  readonly shadowDepthBias: number;
  /** Bytes of the depth32float shadow atlas (a single 1×1 layer without shadows). */
  readonly atlasBytes: number;
  /** Bytes of the light uniform block shared by every consumer. */
  readonly uniformBytes: number;
  /** Bytes of the per-view uniforms used by the shared shadow pipeline. */
  readonly viewUniformBytes: number;
}

export const LOCAL_LIGHT_LIMITS = Object.freeze({
  maxLights: 256,
  maxShadowedLights: 8,
  maxShadowViews: 48,
  minShadowResolution: 128,
  maxShadowResolution: 2048,
  /** 80°: wider cones should be point lights; keeps the spot shadow frustum well conditioned. */
  maxSpotAngle: 80 * Math.PI / 180,
  minRange: 0.01,
  maxRange: 10_000,
  maxCoordinate: 1_000_000,
  /** Upper bound of color × intensity; keeps f32 sums finite at the 1 cm clamp. */
  maxRadiantIntensity: 1e12,
  maxBias: 16,
});
/** Inverse-square clamp radius in metres; irradiance within 1 cm is bounded. */
export const LOCAL_LIGHT_MIN_DISTANCE = 0.01;
/** One light record: five vec4f. */
export const LOCAL_LIGHT_RECORD_FLOATS = 20;
/** Header vec4u before the light records. */
export const LOCAL_LIGHT_HEADER_FLOATS = 4;
/** Per-view uniform stride of the shared shadow pipeline's frame contract. */
export const LOCAL_SHADOW_VIEW_UNIFORM_STRIDE = 256;
/** Byte offset of Frame.lightViewProjection inside one per-view uniform. */
export const LOCAL_SHADOW_VIEW_MATRIX_OFFSET = 128;
/** Shadow texels reserved on each side of a view for the 3×3 PCF footprint. */
export const LOCAL_SHADOW_BORDER_TEXELS = 2;
const DEFAULT_SHADOW_BYTES = 32 * 1024 * 1024;
const POINT_FACES: readonly { axis: LocalLightVec3; up: LocalLightVec3 }[] = Object.freeze([
  { axis: [1, 0, 0], up: [0, 1, 0] }, { axis: [-1, 0, 0], up: [0, 1, 0] },
  { axis: [0, 1, 0], up: [0, 0, 1] }, { axis: [0, -1, 0], up: [0, 0, 1] },
  { axis: [0, 0, 1], up: [0, 1, 0] }, { axis: [0, 0, -1], up: [0, 1, 0] },
]);

function fail(code: string, detail?: string): never {
  throw new RangeError(detail ? `${code}: ${detail}` : code);
}

function finiteF32(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && Number.isFinite(Math.fround(value));
}

function integer(value: unknown, min: number, max: number, code: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) fail(code);
  return value as number;
}

function vector(value: unknown, label: string, maximum: number, nonnegative = false): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(finiteF32)
    || value.some(component => Math.abs(component) > maximum || (nonnegative && component < 0))) {
    fail('INVALID_LOCAL_LIGHT', label);
  }
  return [value[0], value[1], value[2]];
}

/** Validate and budget the fixed allocation before any GPU object is created. */
export function resolveLocalLightOptions(options: LocalLightOptions = {}, limits: {
  maxTextureDimension2D?: number; maxTextureArrayLayers?: number; maxUniformBufferBindingSize?: number;
} = {}): ResolvedLocalLightOptions {
  if (!options || typeof options !== 'object') fail('INVALID_LOCAL_LIGHT_OPTIONS');
  const maxLights = integer(options.maxLights ?? 64, 1, LOCAL_LIGHT_LIMITS.maxLights, 'INVALID_LOCAL_LIGHT_OPTIONS');
  const maxShadowedLights = integer(options.maxShadowedLights ?? 4, 0, LOCAL_LIGHT_LIMITS.maxShadowedLights,
    'INVALID_LOCAL_LIGHT_OPTIONS');
  if (maxShadowedLights > maxLights) fail('INVALID_LOCAL_LIGHT_OPTIONS', 'maxShadowedLights exceeds maxLights');
  const shadowResolution = integer(options.shadowResolution ?? 512, LOCAL_LIGHT_LIMITS.minShadowResolution,
    LOCAL_LIGHT_LIMITS.maxShadowResolution, 'INVALID_LOCAL_LIGHT_SHADOW_RESOLUTION');
  if ((shadowResolution & (shadowResolution - 1)) !== 0
    || shadowResolution > (limits.maxTextureDimension2D ?? Infinity)) fail('INVALID_LOCAL_LIGHT_SHADOW_RESOLUTION');
  const defaultViews = maxShadowedLights ? Math.min(LOCAL_LIGHT_LIMITS.maxShadowViews, 6 * maxShadowedLights) : 1;
  const maxShadowViews = integer(options.maxShadowViews ?? defaultViews, 1, LOCAL_LIGHT_LIMITS.maxShadowViews,
    'INVALID_LOCAL_LIGHT_OPTIONS');
  if (maxShadowViews > (limits.maxTextureArrayLayers ?? Infinity)) fail('LOCAL_LIGHT_SHADOW_BUDGET', 'texture array layers');
  const maxShadowBytes = integer(options.maxShadowBytes ?? DEFAULT_SHADOW_BYTES, 1, Number.MAX_SAFE_INTEGER,
    'INVALID_LOCAL_LIGHT_OPTIONS');
  const bias = (value: number | undefined, fallback: number) => {
    const resolved = value ?? fallback;
    if (!finiteF32(resolved) || resolved < 0 || resolved > LOCAL_LIGHT_LIMITS.maxBias) fail('INVALID_LOCAL_LIGHT_OPTIONS', 'shadow bias');
    return resolved;
  };
  const shadowNormalBias = bias(options.shadowNormalBias, 1.5);
  const shadowDepthBias = bias(options.shadowDepthBias, 1);
  const atlasBytes = maxShadowedLights ? 4 * shadowResolution * shadowResolution * maxShadowViews : 4;
  if (atlasBytes > maxShadowBytes) fail('LOCAL_LIGHT_SHADOW_BUDGET', `atlas ${atlasBytes} > ${maxShadowBytes} bytes`);
  const uniformBytes = 4 * (LOCAL_LIGHT_HEADER_FLOATS + maxLights * LOCAL_LIGHT_RECORD_FLOATS + maxShadowViews * 16);
  if (uniformBytes > (limits.maxUniformBufferBindingSize ?? 65536)) fail('LOCAL_LIGHT_BUDGET', 'uniform binding size');
  return Object.freeze({ maxLights, maxShadowedLights, shadowResolution, maxShadowViews, maxShadowBytes,
    shadowNormalBias, shadowDepthBias, atlasBytes, uniformBytes,
    viewUniformBytes: maxShadowViews * LOCAL_SHADOW_VIEW_UNIFORM_STRIDE });
}

type Fields = Omit<ResolvedLocalLight, 'handle' | 'version'>;

/** Validate one complete descriptor; omitted spot angles and flags take defaults. */
export function validateLocalLight(descriptor: LocalLightDescriptor): Fields {
  if (!descriptor || typeof descriptor !== 'object') fail('INVALID_LOCAL_LIGHT', 'descriptor');
  const kind = descriptor.kind;
  if (kind !== 'point' && kind !== 'spot') fail('INVALID_LOCAL_LIGHT', 'kind');
  const position = vector(descriptor.position, 'position', LOCAL_LIGHT_LIMITS.maxCoordinate);
  const color = vector(descriptor.color, 'color', LOCAL_LIGHT_LIMITS.maxRadiantIntensity, true);
  const intensity = descriptor.intensity;
  if (!finiteF32(intensity) || intensity < 0) fail('INVALID_LOCAL_LIGHT', 'intensity');
  if (color.some(component => component * intensity > LOCAL_LIGHT_LIMITS.maxRadiantIntensity)) {
    fail('INVALID_LOCAL_LIGHT', 'radiant intensity exceeds 1e12');
  }
  const range = descriptor.range;
  if (!finiteF32(range) || range < LOCAL_LIGHT_LIMITS.minRange || range > LOCAL_LIGHT_LIMITS.maxRange) {
    fail('INVALID_LOCAL_LIGHT', 'range');
  }
  for (const flag of [descriptor.castShadows, descriptor.enabled]) {
    if (flag !== undefined && typeof flag !== 'boolean') fail('INVALID_LOCAL_LIGHT', 'flag');
  }
  let direction: LocalLightVec3 = [0, -1, 0], innerConeAngle = 0, outerConeAngle = 0;
  if (kind === 'spot') {
    const axis = vector(descriptor.direction, 'direction', LOCAL_LIGHT_LIMITS.maxCoordinate);
    const length = Math.hypot(...axis);
    if (!(length > 1e-6)) fail('INVALID_LOCAL_LIGHT', 'direction');
    direction = [axis[0] / length, axis[1] / length, axis[2] / length];
    outerConeAngle = descriptor.outerConeAngle ?? Math.PI / 4;
    innerConeAngle = descriptor.innerConeAngle ?? 0;
    if (!finiteF32(outerConeAngle) || !finiteF32(innerConeAngle) || outerConeAngle <= 0
      || outerConeAngle > LOCAL_LIGHT_LIMITS.maxSpotAngle + 1e-12 || innerConeAngle < 0 || innerConeAngle > outerConeAngle) {
      fail('INVALID_LOCAL_LIGHT', 'cone angles');
    }
  } else if (descriptor.direction !== undefined || descriptor.innerConeAngle !== undefined
    || descriptor.outerConeAngle !== undefined) fail('INVALID_LOCAL_LIGHT', 'point lights have no cone');
  return { kind, position: Object.freeze(position), direction: Object.freeze([...direction]) as unknown as LocalLightVec3,
    innerConeAngle, outerConeAngle, color: Object.freeze(color), intensity, range,
    castShadows: descriptor.castShadows ?? false, enabled: descriptor.enabled ?? true };
}

function sameFields(a: Fields, b: Fields): boolean {
  return a.kind === b.kind && a.position.every((v, i) => v === b.position[i]) && a.direction.every((v, i) => v === b.direction[i])
    && a.innerConeAngle === b.innerConeAngle && a.outerConeAngle === b.outerConeAngle
    && a.color.every((v, i) => v === b.color[i]) && a.intensity === b.intensity && a.range === b.range
    && a.castShadows === b.castShadows && a.enabled === b.enabled;
}

/** Number of atlas views a shadowed light reserves. */
export function localLightShadowViewCount(kind: LocalLightKind): number { return kind === 'point' ? 6 : 1; }

/** Smooth finite range window W(d) = saturate(1 − (d/r)⁴)². */
export function localLightRangeWindow(distance: number, range: number): number {
  const ratio = (distance * distance) / (range * range);
  const window = Math.min(1, Math.max(0, 1 - ratio * ratio));
  return window * window;
}

/** Squared linear ramp between the outer and inner cone in cosine space. */
export function localLightConeFactor(cosAngle: number, innerConeAngle: number, outerConeAngle: number): number {
  const cosOuter = Math.cos(outerConeAngle);
  const scale = 1 / Math.max(Math.cos(innerConeAngle) - cosOuter, 1e-4);
  const t = Math.min(1, Math.max(0, (cosAngle - cosOuter) * scale));
  return t * t;
}

/** CPU reference: unshadowed E⊥ (RGB) and the unit direction from `point` toward the light. */
export function localLightIrradiance(light: Pick<ResolvedLocalLight, 'kind' | 'position' | 'direction'
  | 'innerConeAngle' | 'outerConeAngle' | 'color' | 'intensity' | 'range'>, point: LocalLightVec3) {
  const delta = [light.position[0] - point[0], light.position[1] - point[1], light.position[2] - point[2]];
  const distanceSquared = delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2;
  const distance = Math.sqrt(distanceSquared);
  const direction = (distance > 0 ? delta.map(value => value / distance) : [0, 1, 0]) as [number, number, number];
  if (distanceSquared >= light.range * light.range) return { direction, distance, irradiance: [0, 0, 0] as [number, number, number] };
  let factor = localLightRangeWindow(distance, light.range) / Math.max(distanceSquared, LOCAL_LIGHT_MIN_DISTANCE ** 2);
  if (light.kind === 'spot') {
    const cosAngle = -(direction[0] * light.direction[0] + direction[1] * light.direction[1] + direction[2] * light.direction[2]);
    factor *= localLightConeFactor(cosAngle, light.innerConeAngle, light.outerConeAngle);
  }
  return { direction, distance,
    irradiance: light.color.map(channel => channel * light.intensity * factor) as [number, number, number] };
}

export interface LocalLightShadowFrustum {
  /** World → light clip, WebGPU depth [0, 1]; one matrix per view (spot 1, point +X −X +Y −Y +Z −Z). */
  readonly matrices: readonly Mat4[];
  readonly near: number;
  readonly far: number;
  /** tan of the half field of view, including the PCF border. */
  readonly tanHalfFov: number;
}

/** Perspective views covering the whole lit volume; independent of the camera. */
export function localLightShadowFrustum(light: Pick<ResolvedLocalLight, 'kind' | 'position' | 'direction'
  | 'outerConeAngle' | 'range'>, resolution: number): LocalLightShadowFrustum {
  const far = light.range;
  // Near-plane casters closer than this are not represented; it bounds depth precision.
  const near = Math.min(1, Math.max(0.005, far / 500));
  const border = 1 - 2 * LOCAL_SHADOW_BORDER_TEXELS / resolution;
  const tanHalfFov = (light.kind === 'spot' ? Math.tan(light.outerConeAngle) : 1) / border;
  const f = 1 / tanHalfFov, depthScale = far / (near - far);
  const projection = new Float32Array([f, 0, 0, 0, 0, f, 0, 0, 0, 0, depthScale, -1, 0, 0, near * depthScale, 0]);
  const p = light.position;
  const faces = light.kind === 'spot'
    ? [{ axis: light.direction, up: (Math.abs(light.direction[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0]) as LocalLightVec3 }]
    : POINT_FACES;
  const matrices = faces.map(face => multiply(projection,
    lookAt(p, [p[0] + face.axis[0], p[1] + face.axis[1], p[2] + face.axis[2]], face.up)));
  if (matrices.some(matrix => !matrix.every(Number.isFinite))) fail('LOCAL_LIGHT_SHADOW_PROJECTION_OUT_OF_RANGE');
  return { matrices, near, far, tanHalfFov };
}

export interface LocalShadowAssignment {
  readonly handle: LocalLightHandle;
  readonly firstView: number;
  readonly viewCount: number;
  readonly enabled: boolean;
}

export interface PackedLocalLights {
  /** Enabled lights written to the block, in slot order. */
  readonly lightCount: number;
  /** Atlas views that must be rendered this frame (enabled shadowed lights). */
  readonly views: readonly { readonly layer: number; readonly matrix: Mat4 }[];
}

/** World bounds of every light whose shading differs between two snapshots: the range-sphere box before and
 * after each added, removed or re-versioned light. Disabled lights shade nothing, so they contribute no box.
 * An in-place change (same position and range) reports its box once. */
export function localLightChangeRegions(before: readonly ResolvedLocalLight[], after: readonly ResolvedLocalLight[]): Bounds3[] {
  const box = (light: ResolvedLocalLight): Bounds3 => ({
    min: [light.position[0] - light.range, light.position[1] - light.range, light.position[2] - light.range],
    max: [light.position[0] + light.range, light.position[1] + light.range, light.position[2] + light.range] });
  const key = (light: ResolvedLocalLight) => light.handle.id;
  const same = (a: ResolvedLocalLight | undefined, b: ResolvedLocalLight | undefined) =>
    !!a && !!b && a.handle.generation === b.handle.generation && a.version === b.version;
  const previous = new Map(before.map(light => [key(light), light])), next = new Map(after.map(light => [key(light), light]));
  const regions: Bounds3[] = [], seen = new Set<string>();
  const push = (light: ResolvedLocalLight) => {
    const region = box(light), id = [...region.min, ...region.max].join();
    if (!seen.has(id)) { seen.add(id); regions.push(region); }
  };
  for (const light of before) if (light.enabled && !same(light, next.get(key(light)))) push(light);
  for (const light of after) if (light.enabled && !same(previous.get(key(light)), light)) push(light);
  return regions;
}

/** CPU owner of slots, generations, versions, budgets and GPU packing.
 * Has no GPU objects; `revision` drives explicit history invalidation. */
export class LocalLightSet {
  readonly options: ResolvedLocalLightOptions;
  private readonly slots: (ResolvedLocalLight | undefined)[];
  private readonly generations: number[];
  private revisionValue = 0;

  constructor(options: LocalLightOptions | ResolvedLocalLightOptions = {}) {
    this.options = 'atlasBytes' in options ? options : resolveLocalLightOptions(options);
    this.slots = Array.from({ length: this.options.maxLights }, () => undefined);
    this.generations = Array.from({ length: this.options.maxLights }, () => 0);
  }

  /** Incremented by every add, removal and effective update. */
  get revision(): number { return this.revisionValue; }
  get count(): number { return this.slots.reduce((sum, light) => sum + (light ? 1 : 0), 0); }

  add(descriptor: LocalLightDescriptor): LocalLightHandle {
    const fields = validateLocalLight(descriptor);
    const id = this.slots.findIndex(slot => !slot);
    if (id < 0) fail('LOCAL_LIGHT_BUDGET', `at most ${this.options.maxLights} lights`);
    this.checkShadowBudget(fields, -1);
    const handle = Object.freeze({ id, generation: this.generations[id] });
    this.slots[id] = Object.freeze({ ...fields, handle, version: 1 });
    this.revisionValue++;
    return handle;
  }

  /** Merge a partial change; unchanged values keep the per-light version. */
  update(handle: LocalLightHandle, patch: Partial<LocalLightDescriptor>): ResolvedLocalLight {
    const current = this.get(handle);
    if (!patch || typeof patch !== 'object') fail('INVALID_LOCAL_LIGHT', 'patch');
    const kind = patch.kind ?? current.kind;
    const spot = kind === 'spot';
    const merged: LocalLightDescriptor = {
      kind, position: patch.position ?? current.position, color: patch.color ?? current.color,
      intensity: patch.intensity ?? current.intensity, range: patch.range ?? current.range,
      castShadows: patch.castShadows ?? current.castShadows, enabled: patch.enabled ?? current.enabled,
      ...(spot ? {
        direction: patch.direction ?? (current.kind === 'spot' ? current.direction : undefined),
        innerConeAngle: patch.innerConeAngle ?? (current.kind === 'spot' ? current.innerConeAngle : undefined),
        outerConeAngle: patch.outerConeAngle ?? (current.kind === 'spot' ? current.outerConeAngle : undefined),
      } : {}),
    };
    if (!spot && (patch.direction !== undefined || patch.innerConeAngle !== undefined || patch.outerConeAngle !== undefined)) {
      fail('INVALID_LOCAL_LIGHT', 'point lights have no cone');
    }
    const fields = validateLocalLight(merged);
    if (sameFields(fields, current)) return current;
    this.checkShadowBudget(fields, handle.id);
    const next = Object.freeze({ ...fields, handle: current.handle, version: current.version + 1 });
    this.slots[handle.id] = next;
    this.revisionValue++;
    return next;
  }

  setEnabled(handle: LocalLightHandle, enabled: boolean): ResolvedLocalLight {
    if (typeof enabled !== 'boolean') fail('INVALID_LOCAL_LIGHT', 'enabled');
    return this.update(handle, { enabled });
  }

  remove(handle: LocalLightHandle): void {
    this.get(handle);
    this.slots[handle.id] = undefined;
    if (this.generations[handle.id] >= 0xffff_ffff) fail('LOCAL_LIGHT_GENERATION_EXHAUSTED');
    this.generations[handle.id]++;
    this.revisionValue++;
  }

  get(handle: LocalLightHandle): ResolvedLocalLight {
    if (!handle || typeof handle !== 'object' || !Number.isSafeInteger(handle.id) || handle.id < 0
      || handle.id >= this.slots.length || !Number.isSafeInteger(handle.generation)) fail('LOCAL_LIGHT_NOT_FOUND');
    const light = this.slots[handle.id];
    if (!light || light.handle.generation !== handle.generation) fail('LOCAL_LIGHT_NOT_FOUND', 'stale or released handle');
    return light;
  }

  list(): ResolvedLocalLight[] { return this.slots.filter((light): light is ResolvedLocalLight => !!light); }

  /** Atlas reservations in slot order; disabled shadowed lights keep their views. */
  shadowAssignments(): LocalShadowAssignment[] {
    let firstView = 0;
    return this.list().filter(light => light.castShadows).map(light => {
      const viewCount = localLightShadowViewCount(light.kind);
      const assignment = { handle: light.handle, firstView, viewCount, enabled: light.enabled };
      firstView += viewCount;
      return assignment;
    });
  }

  /** Fill the uniform block: header, enabled light records, then view matrices. */
  pack(target: Float32Array): PackedLocalLights {
    const options = this.options;
    if (target.length < options.uniformBytes / 4) fail('LOCAL_LIGHT_PACK_TARGET');
    target.fill(0, 0, options.uniformBytes / 4);
    const words = new Uint32Array(target.buffer, target.byteOffset, target.length);
    const shadows = new Map(this.shadowAssignments().map(assignment => [assignment.handle.id, assignment]));
    const views: { layer: number; matrix: Mat4 }[] = [];
    let count = 0, shadowed = 0;
    const viewBase = LOCAL_LIGHT_HEADER_FLOATS + options.maxLights * LOCAL_LIGHT_RECORD_FLOATS;
    for (const light of this.list()) {
      if (!light.enabled) continue;
      const assignment = shadows.get(light.handle.id);
      const frustum = localLightShadowFrustum(light, options.shadowResolution);
      const cosOuter = Math.cos(light.outerConeAngle);
      const base = LOCAL_LIGHT_HEADER_FLOATS + count * LOCAL_LIGHT_RECORD_FLOATS;
      target.set([...light.position, light.range], base);
      target.set([...light.direction, light.kind === 'spot' ? 1 : 0], base + 4);
      target.set([...light.color.map(channel => channel * light.intensity), assignment ? assignment.firstView : -1], base + 8);
      target.set([light.kind === 'spot' ? cosOuter : -2,
        light.kind === 'spot' ? 1 / Math.max(Math.cos(light.innerConeAngle) - cosOuter, 1e-4) : 0,
        frustum.near, frustum.far], base + 12);
      target.set([frustum.tanHalfFov, options.shadowNormalBias, options.shadowDepthBias, 0], base + 16);
      if (assignment) {
        frustum.matrices.forEach((matrix, face) => {
          const layer = assignment.firstView + face;
          target.set(matrix, viewBase + layer * 16);
          views.push({ layer, matrix });
        });
        shadowed++;
      }
      count++;
    }
    words[0] = count; words[1] = shadowed; words[2] = this.revisionValue >>> 0; words[3] = options.shadowResolution;
    return { lightCount: count, views };
  }

  private checkShadowBudget(fields: Fields, replacing: number): void {
    if (!fields.castShadows) return;
    const others = this.list().filter(light => light.castShadows && light.handle.id !== replacing);
    if (others.length + 1 > this.options.maxShadowedLights) {
      fail('LOCAL_LIGHT_SHADOW_BUDGET', `at most ${this.options.maxShadowedLights} shadowed lights`);
    }
    const views = others.reduce((sum, light) => sum + localLightShadowViewCount(light.kind), 0)
      + localLightShadowViewCount(fields.kind);
    if (views > this.options.maxShadowViews) fail('LOCAL_LIGHT_SHADOW_BUDGET', `${views} > ${this.options.maxShadowViews} atlas views`);
  }
}
