import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createCube, createPlane, type MeshData } from '../../../src/geometry';
import { identity, multiply, normalMatrix, perspective, lookAt, scaling, transformPoint, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
type Pixel = { x: number; y: number };

const size = 512;
const eye: Vec3 = [0, 6, 5];
const target: Vec3 = [0, 0, 0];
const sunDirection: Vec3 = [0.6, 1, -0.5];
const sunColor: Vec3 = [3, 3, 3];
const casterCenter: Vec3 = [0.25, 1.2, -0.15];
const angle = 0.5;
const casterHalfLength = 1.45;
const casterHalfWidth = 0.55;
const focusBounds = { min: [-3.5, -0.25, -3.5] as Vec3,
  max: [3.5, 2, 3.5] as Vec3 };
const focusOptions = { resolution: 2048, focusBounds, padding: 0.15 };

function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];
}

/** Independent camera ray, intersected with the y=0 receiver at the pixel center. */
function floorAtPixel(pixel: Pixel): Vec3 {
  const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const right = normalize(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const half = Math.tan(Math.PI / 6.4);
  const nx = (2 * (pixel.x + 0.5) / size - 1) * half;
  const ny = (1 - 2 * (pixel.y + 0.5) / size) * half;
  const ray: Vec3 = [forward[0] + right[0] * nx + up[0] * ny,
    forward[1] + right[1] * nx + up[1] * ny,
    forward[2] + right[2] * nx + up[2] * ny];
  const t = -eye[1] / ray[1];
  return [eye[0] + ray[0] * t, 0, eye[2] + ray[2] * t];
}

function pixelAtWorld(point: Vec3): Pixel {
  const vp = multiply(perspective(Math.PI / 3.2, 1, 0.1, 150), lookAt(eye, target, [0, 1, 0]));
  const ndc = transformPoint(vp, point);
  return { x: Math.round((ndc[0] * 0.5 + 0.5) * size - 0.5),
    y: Math.round((1 - (ndc[1] * 0.5 + 0.5)) * size - 0.5) };
}

/** Signed distance to the long +v edge of the authored horizontal quad. */
function expectedEdge(pixel: Pixel) {
  const floor = floorAtPixel(pixel);
  const rayT = casterCenter[1] / sunDirection[1];
  const hitX = floor[0] + sunDirection[0] * rayT - casterCenter[0];
  const hitZ = floor[2] + sunDirection[2] * rayT - casterCenter[2];
  const u = Math.cos(angle) * hitX + Math.sin(angle) * hitZ;
  const v = -Math.sin(angle) * hitX + Math.cos(angle) * hitZ;
  return { floor, u, v, edgeDistance: v - casterHalfWidth,
    occluded: Math.abs(u) <= casterHalfLength && Math.abs(v) <= casterHalfWidth };
}

/** Slab ray test against the separate near-contact cube, independent of raster depth. */
function rayHitsContactCube(origin: Vec3, direction: Vec3, minT: number, maxT: number): boolean {
  const min: Vec3 = [-2.38, 0, -1.68], max: Vec3 = [-2.02, 0.46, -1.32];
  let enter = minT, exit = maxT;
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(direction[axis]) < 1e-12) {
      if (origin[axis] < min[axis] || origin[axis] > max[axis]) return false;
      continue;
    }
    const a = (min[axis] - origin[axis]) / direction[axis];
    const b = (max[axis] - origin[axis]) / direction[axis];
    enter = Math.max(enter, Math.min(a, b));
    exit = Math.min(exit, Math.max(a, b));
    if (enter > exit) return false;
  }
  return true;
}

function casterPoint(u: number, v: number): Vec3 {
  return [casterCenter[0] + Math.cos(angle) * u - Math.sin(angle) * v,
    casterCenter[1], casterCenter[2] + Math.sin(angle) * u + Math.cos(angle) * v];
}

function quad(points: Vec3[]): MeshData {
  const vertices = new ArrayBuffer(4 * 80);
  const data = new DataView(vertices);
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  points.forEach((point, vertex) => {
    point.forEach((value, axis) => {
      data.setFloat32(vertex * 80 + axis * 4, value, true);
      min[axis] = Math.min(min[axis], value);
      max[axis] = Math.max(max[axis], value);
    });
    data.setFloat32(vertex * 80 + 12, 1, true);
    data.setFloat32(vertex * 80 + 20, 1, true);
    data.setUint32(vertex * 80 + 32, 0, true);
    data.setFloat32(vertex * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min, max } };
}

function instance(entityId: number, model = identity()): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [0.85, 0.85, 0.85, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}
function batch(name: string, mesh: MeshData, instances: InstanceState[]): SceneBatch {
  return { name, mesh, capacity: 1, inputMode: 'committed', instances,
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true } } };
}

function samples() {
  const edgeOffsets = [-0.12, -0.09, -0.06, -0.04, -0.025, -0.012,
    0.012, 0.025, 0.04, 0.06, 0.09, 0.12];
  const points: Pixel[] = [];
  const tags: string[] = [];
  for (let along = 0; along < 56; along++) {
    const u = -1.13 + along * 2.26 / 55;
    for (const offset of edgeOffsets) {
      const caster = casterPoint(u, casterHalfWidth + offset);
      const floor: Vec3 = [caster[0] - casterCenter[1] * sunDirection[0] / sunDirection[1],
        0, caster[2] - casterCenter[1] * sunDirection[2] / sunDirection[1]];
      points.push(pixelAtWorld(floor)); tags.push('edge');
    }
    if (along % 4 === 0) {
      for (const offset of [-0.25, 0.25]) {
        const caster = casterPoint(u, casterHalfWidth + offset);
        points.push(pixelAtWorld([caster[0] - casterCenter[1] * sunDirection[0] / sunDirection[1],
          0, caster[2] - casterCenter[1] * sunDirection[2] / sunDirection[1]]));
        tags.push(offset < 0 ? 'deep-shadow' : 'clear-light');
      }
    }
  }
  const contactCenter: Vec3 = [-2.2, 0.23, -1.5];
  points.push(pixelAtWorld([-2.51, 0, -1.28])); tags.push('contact');
  points.push(pixelAtWorld([-2.9, 0, -1.1])); tags.push('contact-clear');
  return { points, tags, contactCenter };
}

export async function runShadowPrecisionFixture() {
  const canvas = document.createElement('canvas');
  canvas.id = 'shadow-precision-canvas';
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${size}px;height:${size}px;z-index:9999;pointer-events:none;`;
  document.body.appendChild(canvas);
  const { points, tags, contactCenter } = samples();
  const contactChecks = points.slice(-2).map((pixel, index) => {
    const floor = floorAtPixel(pixel);
    const cameraRay: Vec3 = [floor[0] - eye[0], floor[1] - eye[1], floor[2] - eye[2]];
    return { tag: tags[tags.length - 2 + index], pixel, floor,
      cameraHitsCube: rayHitsContactCube(eye, cameraRay, 0, 1),
      sunHitsCube: rayHitsContactCube(floor, sunDirection, 0, 5),
      sunHitsSlab: expectedEdge(pixel).occluded };
  });
  if (contactChecks[0].cameraHitsCube || !contactChecks[0].sunHitsCube ||
      contactChecks[0].sunHitsSlab || contactChecks[1].cameraHitsCube ||
      contactChecks[1].sunHitsCube || contactChecks[1].sunHitsSlab) {
    throw new Error(`SHADOW_PRECISION_CONTROL_RAYS ${JSON.stringify(contactChecks)}`);
  }
  const floor = batch('precision-floor', createPlane(), [instance(1, scaling(6, 1, 6))]);
  const blocker = batch('diagonal-sun-blocker', quad([
    casterPoint(-casterHalfLength, casterHalfWidth), casterPoint(casterHalfLength, casterHalfWidth),
    casterPoint(casterHalfLength, -casterHalfWidth), casterPoint(-casterHalfLength, -casterHalfWidth),
  ]), []);
  const contactModel = multiply(translation(...contactCenter), scaling(0.36, 0.46, 0.36));
  const contact = batch('near-contact-control', createCube(), []);
  const modes = [];
  try {
    for (const mode of ['default', 'focused'] as const) {
      let renderer: Renderer | undefined;
      try {
        blocker.instances = [];
        contact.instances = [];
        renderer = await Renderer.create(canvas, [floor, blocker, contact],
          mode === 'focused' ? { sunShadow: focusOptions } : undefined);
        const render = async (strength: number) => {
          renderer!.setSun({ direction: sunDirection,
            color: sunColor.map(value => value * strength) as Vec3 });
          renderer!.render(eye, target);
          if (renderer!.stats.width !== size || renderer!.stats.height !== size) {
            throw new Error(`SHADOW_PRECISION_SIZE ${renderer!.stats.width}x${renderer!.stats.height}`);
          }
          return renderer!.probeHdr(points);
        };
        const ambient = await render(0);
        const lit = await render(1);
        blocker.instances = [instance(2)];
        contact.instances = [instance(3, contactModel)];
        renderer.setSun({ direction: sunDirection, color: sunColor });
        renderer.render(eye, target);
        // Capture the just-presented canvas before any later task obtains a new swapchain image.
        const png = canvas.toDataURL('image/png');
        if (!png.startsWith('data:image/png;base64,')) throw new Error('SHADOW_PRECISION_PNG_CAPTURE');
        const blocked = await renderer.probeHdr(points);
        const diagnostics = renderer.diagnostics().sunShadow;
        const bytesBeforeFocusChange = renderer.stats.ownedGpuBytes;
        let focusChange: null | { original: typeof diagnostics; changed: typeof diagnostics;
          afterInvalid: typeof diagnostics; bytesAfter: number; invalidMessage: string } = null;
        if (mode === 'focused') {
          renderer.setSunShadowFocus({ min: [-3.3, -0.25, -3.3], max: [3.3, 2, 3.3] });
          const changed = renderer.diagnostics().sunShadow;
          let invalidMessage = '';
          try { renderer.setSunShadowFocus({ min: [1, 0, 0], max: [0, 1, 1] }); }
          catch (error) { invalidMessage = String(error); }
          const afterInvalid = renderer.diagnostics().sunShadow;
          focusChange = { original: diagnostics, changed, afterInvalid,
            bytesAfter: renderer.stats.ownedGpuBytes, invalidMessage };
        }
        const errors = [...renderer.errors];
        renderer.dispose();
        modes.push({ mode, ambient, lit, blocked, png, diagnostics,
          bytesBeforeFocusChange, focusChange, errors,
          ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes });
      } finally { renderer?.dispose(); }
    }
    return { size, points, tags, geometry: points.map(expectedEdge), contactChecks, modes };
  } finally { canvas.remove(); }
}
