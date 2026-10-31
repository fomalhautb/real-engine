import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { DiffuseGiOptions } from '../../../src/diffuse-gi';
import type { MeshData } from '../../../src/geometry';
import type { LocalLightDescriptor } from '../../../src/local-lights';
import { DEFAULT_MATERIAL_DEFINITION } from '../../../src/material';
import { identity } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
type Vec4 = [number, number, number, number];
type Pixel = { x: number; y: number };

function mesh(triangles: readonly (readonly [Vec3, Vec3, Vec3])[], normals: readonly Vec3[]): MeshData {
  const vertices = new ArrayBuffer(triangles.length * 3 * 80), view = new DataView(vertices);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  triangles.forEach((triangle, face) => triangle.forEach((point, corner) => {
    const offset = (face * 3 + corner) * 80;
    for (let axis = 0; axis < 3; axis++) {
      view.setFloat32(offset + axis * 4, point[axis], true);
      view.setFloat32(offset + 16 + axis * 4, normals[face][axis], true);
      min[axis] = Math.min(min[axis], point[axis]); max[axis] = Math.max(max[axis], point[axis]);
    }
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 48, 1, true);
  }));
  return { vertices, vertexCount: triangles.length * 3, indices: Uint32Array.from({ length: triangles.length * 3 }, (_, i) => i),
    bounds: { min, max } };
}

/** Horizontal quad facing +Z. */
function quad(center: Vec3, half: number): MeshData {
  const [x, y, z] = center;
  const a: Vec3 = [x - half, y - half, z], b: Vec3 = [x + half, y - half, z], c: Vec3 = [x + half, y + half, z], d: Vec3 = [x - half, y + half, z];
  return mesh([[a, b, c], [a, c, d]], [[0, 0, 1], [0, 0, 1]]);
}

function state(entityId: number, color: Vec4): InstanceState {
  return { entityId, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [], color,
    metalness: 0, roughness: 1, historyVersion: 0 };
}

function batch(name: string, geometry: MeshData, entityId: number, instances: InstanceState[]): SceneBatch {
  return { name, mesh: geometry, capacity: 1, inputMode: 'committed', instances,
    queryIdentity: { assetId: 'local-light-gi/' + name, primitiveId: entityId, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}

function canvas(size: number): HTMLCanvasElement {
  const element = document.createElement('canvas');
  element.style.cssText = `position:fixed;left:0;top:0;width:${size}px;height:${size}px;`;
  document.body.appendChild(element);
  return element;
}

export interface LocalLightGiProbePlan {
  floorHalf: number;
  floorColor: Vec4;
  plate: { center: Vec3; half: number };
  grid: DiffuseGiOptions;
  light: LocalLightDescriptor;
  eye: Vec3;
  target: Vec3;
  pixels: Pixel[];
}

/** A floor lit only by an off-screen point light; an unlit black plate is the shadow occluder. */
export async function runLocalLightGiProbeFixture(plan: LocalLightGiProbePlan) {
  const element = canvas(96);
  const floor = batch('local-light-gi-floor', quad([0, 0, 0], plan.floorHalf), 1, [state(1, plan.floorColor)]);
  const plate = batch('local-light-gi-plate', quad(plan.plate.center, plan.plate.half), 2, []);
  // Double-sided: probes below the plate must hit its black back face, and it casts from either side.
  plate.material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(element, [floor, plate], { worldQueries: { maxRays: 2048 }, diffuseGi: plan.grid,
      localLights: { maxLights: 4, maxShadowedLights: 1, shadowResolution: 512 } });
    renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async () => {
      renderer!.render(plan.eye, plan.target);
      const [probes, hdr] = await Promise.all([renderer!.probeDiffuseGi(), renderer!.probeHdr(plan.pixels)]);
      const gi = renderer!.diagnostics().diffuseGi!;
      return { probes: probes.probes.map(probe => ({ coefficients: probe.coefficients, valid: probe.valid,
        primaryOverflows: probe.primaryOverflows })), hdr, resetReason: gi.historyResetReason };
    };
    const dark = await capture();
    const handle = renderer.addLocalLight({ ...plan.light, castShadows: false });
    const unshadowed = await capture();
    renderer.setDiffuseGiEnabled(false);
    const giOffLit = await capture();
    renderer.setLocalLightEnabled(handle, false);
    const giOffDark = await capture();
    renderer.setLocalLightEnabled(handle, true);
    renderer.setDiffuseGiEnabled(true);
    renderer.updateLocalLight(handle, { castShadows: true });
    const shadowedOpen = await capture();
    plate.instances = [state(2, [0, 0, 0, 1])];
    const shadowedPlate = await capture();
    plate.instances = [];
    const shadowedRestored = await capture();
    const lightDiagnostics = renderer.diagnostics().localLights;
    const errors = [...renderer.errors], adapter = renderer.info.adapterInfo;
    renderer.dispose();
    return { dark, unshadowed, giOffLit, giOffDark, shadowedOpen, shadowedPlate, shadowedRestored, lightDiagnostics,
      errors, adapter, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); element.remove(); }
}

/** Flat-shaded inward icosphere: every facet faces the centre. */
function icosphere(radius: number, subdivisions: number) {
  const t = (1 + Math.sqrt(5)) / 2;
  let points: Vec3[] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let faces: [number, number, number][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
    [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
    [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const unit = (p: readonly number[]): Vec3 => { const l = Math.hypot(p[0], p[1], p[2]); return [p[0] / l, p[1] / l, p[2] / l]; };
  points = points.map(unit);
  for (let level = 0; level < subdivisions; level++) {
    const cache = new Map<string, number>(), next: [number, number, number][] = [];
    const middle = (a: number, b: number) => {
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      let index = cache.get(key);
      if (index === undefined) {
        index = points.length;
        points.push(unit(points[a].map((value, axis) => value + points[b][axis])));
        cache.set(key, index);
      }
      return index;
    };
    for (const [a, b, c] of faces) {
      const ab = middle(a, b), bc = middle(b, c), ca = middle(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }
  const triangles: [Vec3, Vec3, Vec3][] = [], normals: Vec3[] = [];
  let minPlane = Infinity, maxPlane = 0;
  for (const face of faces) {
    let [a, b, c] = face.map(index => points[index].map(value => value * radius) as Vec3);
    const e1 = b.map((value, axis) => value - a[axis]), e2 = c.map((value, axis) => value - a[axis]);
    const outward = unit([e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]);
    const centroid = [0, 1, 2].map(axis => (a[axis] + b[axis] + c[axis]) / 3);
    let normal = outward;
    if (normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2] < 0) { normal = normal.map(v => -v) as Vec3; [b, c] = [c, b]; }
    // Reverse to counter-clockwise as seen from the inside and point the normal inward.
    triangles.push([a, c, b]);
    normals.push(normal.map(value => -value) as Vec3);
    const plane = Math.abs(normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2]);
    minPlane = Math.min(minPlane, plane); maxPlane = Math.max(maxPlane, plane);
  }
  return { mesh: mesh(triangles, normals), faces: faces.length, minPlane, maxPlane };
}

export interface LocalLightGatherPlan {
  radius: number;
  subdivisions: number;
  color: Vec4;
  light: LocalLightDescriptor;
  grid: DiffuseGiOptions;
  eye: Vec3;
  target: Vec3;
  size: number;
  anchors: number[];
}

/** A point light at the centre of a closed diffuse sphere makes the gathered radiance uniform,
 * so the Final Gather irradiance is π L and the GI probe SH is a pure DC term, independent of sampling. */
export async function runLocalLightGatherFixture(plan: LocalLightGatherPlan) {
  const element = canvas(plan.size);
  const sphere = icosphere(plan.radius, plan.subdivisions);
  const shell = batch('local-light-gather-shell', sphere.mesh, 1, [state(1, plan.color)]);
  let renderer: Renderer | undefined, stage = 'create';
  try {
    renderer = await Renderer.create(element, [shell], {
      worldQueries: { maxRays: 8192, maxTrianglesPerMesh: sphere.faces },
      diffuseGi: plan.grid, localLights: { maxLights: 2, maxShadowedLights: 1, shadowResolution: 512 } });
    renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    const capture = async (label: string, frames: number) => {
      stage = label;
      for (let frame = 0; frame < frames; frame++) renderer!.render(plan.eye, plan.target);
      const [probes, gather] = await Promise.all([renderer!.probeDiffuseGi(), renderer!.probeDiffuseGiFinalGather(plan.anchors)]);
      const gi = renderer!.diagnostics().diffuseGi!;
      return { probes: probes.probes.map(probe => ({ coefficients: probe.coefficients, valid: probe.valid })),
        anchors: gather.anchors.map(anchor => ({ irradiance: anchor.irradiance, filtered: anchor.filteredIrradiance,
          valid: anchor.valid, position: anchor.position, historyUpdates: anchor.historyUpdates })),
        giReset: gi.historyResetReason, gatherReset: (gi.finalGather as { historyResetReason?: string } | null)?.historyResetReason ?? null,
        gatherHistory: (gi.finalGather as { historyUpdates?: number } | null)?.historyUpdates ?? null };
    };
    const dark = await capture('dark', 2);
    const handle = renderer.addLocalLight({ ...plan.light, castShadows: false });
    const unshadowedFirst = await capture('unshadowed-first', 1);
    const unshadowed = await capture('unshadowed', 3);
    renderer.updateLocalLight(handle, { castShadows: true });
    const shadowedFirst = await capture('shadowed-first', 1);
    const shadowed = await capture('shadowed', 3);
    renderer.updateLocalLight(handle, { intensity: plan.light.intensity * 2 });
    const doubled = await capture('doubled', 3);
    const errors = [...renderer.errors], adapter = renderer.info.adapterInfo;
    renderer.dispose();
    return { geometry: { faces: sphere.faces, minPlane: sphere.minPlane, maxPlane: sphere.maxPlane, radius: plan.radius },
      dark, unshadowedFirst, unshadowed, shadowedFirst, shadowed, doubled, errors, adapter, afterDispose: renderer.stats.ownedGpuBytes };
  } catch (error) {
    // Device loss and readback failures keep their stage and the renderer's recorded loss reason.
    throw new Error(`LOCAL_LIGHT_GATHER_FIXTURE at ${stage}: ${String(error)}; renderer errors: ${JSON.stringify(renderer?.errors ?? [])}`);
  } finally { renderer?.dispose(); element.remove(); }
}
