import type { MeshData } from './geometry';

type Vec2 = [number, number];
type Vec3 = [number, number, number];

export interface ObjImportOptions {
  /** Maximum emitted vertices across all primitives. Defaults to 1,000,000. */
  maxVertices?: number;
  /** Maximum emitted indices across all primitives. Defaults to 3,000,000. */
  maxIndices?: number;
  /** Maximum corners in one convex face. Defaults to 256. */
  maxFaceVertices?: number;
}

export interface ObjPrimitive {
  name: string;
  /** Declared usemtl name. Material values and textures are not loaded here. */
  materialName?: string;
  mesh: MeshData;
}

export interface ObjAsset {
  primitives: ObjPrimitive[];
  materialLibraries: string[];
  warnings: string[];
}

interface Corner {
  position: number;
  uv?: number;
  normal?: number;
}

interface Section {
  name: string;
  materialName?: string;
  vertices: number[];
  uvs: number[];
  indices: number[];
  unique: Map<string, number>;
  min: Vec3;
  max: Vec3;
}

const DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const INTEGER = /^[+-]?\d+$/;

function fail(line: number, message: string): never {
  throw new Error(`OBJ line ${line}: ${message}`);
}

function scalar(token: string, line: number): number {
  if (!DECIMAL.test(token)) fail(line, `invalid number '${token}'`);
  const value = Math.fround(Number(token));
  if (!Number.isFinite(value)) fail(line, `number '${token}' is outside finite float32 range`);
  return value;
}

function budget(value: number | undefined, fallback: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 3 || result > 0xffffffff) {
    throw new Error(`OBJ ${label} must be an integer from 3 to 4294967295`);
  }
  return result;
}

function index(token: string, count: number, line: number, label: string): number {
  if (!INTEGER.test(token)) fail(line, `invalid ${label} index '${token}'`);
  const raw = Number(token);
  if (!Number.isSafeInteger(raw) || raw === 0) fail(line, `invalid ${label} index '${token}'`);
  const resolved = raw > 0 ? raw - 1 : count + raw;
  if (resolved < 0 || resolved >= count) fail(line, `${label} index '${token}' is out of range`);
  return resolved;
}

function corner(token: string, positionCount: number, uvCount: number, normalCount: number, line: number): Corner {
  const parts = token.split('/');
  if (parts.length > 3 || parts[0] === '' || (parts.length > 1 && parts[1] === '' && parts.length === 2)
    || (parts.length === 3 && parts[2] === '')) fail(line, `invalid face corner '${token}'`);
  const result: Corner = { position: index(parts[0], positionCount, line, 'position') };
  if (parts.length > 1 && parts[1] !== '') result.uv = index(parts[1], uvCount, line, 'texture');
  if (parts.length === 3) result.normal = index(parts[2], normalCount, line, 'normal');
  return result;
}

function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function faceNormal(corners: Corner[], positions: Vec3[], line: number): Vec3 {
  const points = corners.map(({ position }) => positions[position]);
  const origin = points[0];
  const area = cross(subtract(points[1], origin), subtract(points[2], origin));
  const length = Math.hypot(...area);
  if (!Number.isFinite(length) || length === 0) fail(line, 'degenerate face');
  const normal: Vec3 = [area[0] / length, area[1] / length, area[2] / length];
  if (points.length === 3) return normal;

  // Fan triangulation is valid only for simple, planar convex polygons. A bounded
  // all-vertices half-plane check rejects concave and self-crossing polygons.
  const drop = normal.map(Math.abs).indexOf(Math.max(...normal.map(Math.abs)));
  const axes = [0, 1, 2].filter((axis) => axis !== drop);
  const signedArea = (drop === 1 ? -normal[drop] : normal[drop]) > 0 ? 1 : -1;
  const span = Math.max(...points.flatMap((p) => p.map((v, axis) => Math.abs(v - origin[axis]))));
  for (const point of points) {
    const distance = normal[0] * (point[0] - origin[0])
      + normal[1] * (point[1] - origin[1]) + normal[2] * (point[2] - origin[2]);
    if (Math.abs(distance) > 1e-5 * span) fail(line, 'non-planar polygon');
  }
  for (let edge = 0; edge < points.length; edge++) {
    const a = points[edge], b = points[(edge + 1) % points.length];
    const dx = b[axes[0]] - a[axes[0]], dy = b[axes[1]] - a[axes[1]];
    if (dx === 0 && dy === 0) fail(line, 'degenerate polygon edge');
    for (let vertex = 0; vertex < points.length; vertex++) {
      if (vertex === edge || vertex === (edge + 1) % points.length) continue;
      const p = points[vertex];
      const side = signedArea * (dx * (p[axes[1]] - a[axes[1]]) - dy * (p[axes[0]] - a[axes[0]]));
      if (side < 0) fail(line, 'concave or self-crossing polygon');
    }
  }
  for (let triangle = 1; triangle < points.length - 1; triangle++) {
    const area = cross(subtract(points[triangle], origin), subtract(points[triangle + 1], origin));
    if (area[0] * normal[0] + area[1] * normal[1] + area[2] * normal[2] <= 0) {
      fail(line, 'degenerate polygon triangulation');
    }
  }
  return normal;
}

function makeSection(name: string, materialName?: string): Section {
  return {
    name, materialName, vertices: [], uvs: [], indices: [], unique: new Map(),
    min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity],
  };
}

function finishSection(section: Section): ObjPrimitive {
  const vertices = new ArrayBuffer(section.vertices.length * 4);
  const view = new DataView(vertices);
  for (let i = 0; i < section.vertices.length; i++) {
    if (i % 20 >= 8 && i % 20 < 12) view.setUint32(i * 4, section.vertices[i], true);
    else view.setFloat32(i * 4, section.vertices[i], true);
  }
  return {
    name: section.name,
    ...(section.materialName === undefined ? {} : { materialName: section.materialName }),
    mesh: {
      vertices,
      uvs: new Float32Array(section.uvs),
      indices: new Uint32Array(section.indices),
      vertexCount: section.vertices.length / 20,
      bounds: { min: section.min, max: section.max },
    },
  };
}

/**
 * Import the polygonal subset of Wavefront OBJ into engine-owned mesh data.
 * Face forms v, v/vt, v//vn and v/vt/vn, including negative relative indices,
 * are supported. N-gons must be planar and convex. Missing normals are flat
 * per source face; `s` smoothing directives are reported as ignored. Lines,
 * points, free-form surfaces, homogeneous positions, and 3D `vt` are rejected.
 */
export function parseObj(text: string, options: ObjImportOptions = {}): ObjAsset {
  const maxVertices = budget(options.maxVertices, 1_000_000, 'maxVertices');
  const maxIndices = budget(options.maxIndices, 3_000_000, 'maxIndices');
  const maxFaceVertices = budget(options.maxFaceVertices, 256, 'maxFaceVertices');
  const positions: Vec3[] = [], uvs: Vec2[] = [], normals: Vec3[] = [];
  const primitives: ObjPrimitive[] = [], materialLibraries: string[] = [], warnings: string[] = [];
  let objectName = 'default', groupName = '', materialName: string | undefined;
  let section = makeSection(objectName), vertexTotal = 0, indexTotal = 0, faceSerial = 0;
  let smoothingWarned = false;

  const flush = () => {
    if (section.indices.length) primitives.push(finishSection(section));
    section = makeSection(groupName ? `${objectName}/${groupName}` : objectName, materialName);
  };
  const lines = text.split(/\r\n|\n|\r/);
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lineIndex + 1;
    const source = lines[lineIndex].split('#', 1)[0].trim();
    if (!source) continue;
    if (source.endsWith('\\')) fail(line, 'continued lines are unsupported');
    const [directive, ...args] = source.split(/\s+/);
    switch (directive) {
      case 'v': {
        if (args.length !== 3) fail(line, 'v requires exactly x y z; homogeneous coordinates and vertex colors are unsupported');
        if (positions.length >= maxVertices) fail(line, 'position record budget exceeded');
        positions.push([scalar(args[0], line), scalar(args[1], line), scalar(args[2], line)]);
        break;
      }
      case 'vt': {
        if (args.length !== 2) fail(line, 'vt requires exactly u v; 1D and 3D texture coordinates are unsupported');
        if (uvs.length >= maxVertices) fail(line, 'texture record budget exceeded');
        uvs.push([scalar(args[0], line), scalar(args[1], line)]);
        break;
      }
      case 'vn': {
        if (args.length !== 3) fail(line, 'vn requires exactly x y z');
        if (normals.length >= maxVertices) fail(line, 'normal record budget exceeded');
        const n: Vec3 = [scalar(args[0], line), scalar(args[1], line), scalar(args[2], line)];
        const length = Math.hypot(...n);
        if (!Number.isFinite(length) || length === 0) fail(line, 'normal must be nonzero and finite');
        normals.push(n.map((value) => Math.fround(value / length)) as Vec3);
        break;
      }
      case 'f': {
        if (args.length < 3 || args.length > maxFaceVertices) fail(line, `f requires 3 to ${maxFaceVertices} corners`);
        const nextIndices = (args.length - 2) * 3;
        if (indexTotal + nextIndices > maxIndices) fail(line, 'index budget exceeded');
        const corners = args.map((token) => corner(token, positions.length, uvs.length, normals.length, line));
        const flatNormal = faceNormal(corners, positions, line);
        const faceVertices: number[] = [];
        for (const item of corners) {
          const key = `${item.position}/${item.uv ?? ''}/${item.normal ?? `f${faceSerial}`}`;
          let vertex = section.unique.get(key);
          if (vertex === undefined) {
            if (vertexTotal >= maxVertices) fail(line, 'vertex budget exceeded');
            vertex = section.vertices.length / 20;
            section.unique.set(key, vertex);
            const position = positions[item.position], normal = item.normal === undefined ? flatNormal : normals[item.normal];
            const uv = item.uv === undefined ? [0, 0] : uvs[item.uv];
            section.vertices.push(
              ...position, 1,
              ...normal, 0,
              0, 1, 2, 3,
              1, 0, 0, 0,
              0, 0, 0, 0,
            );
            section.uvs.push(uv[0], uv[1]);
            for (let axis = 0; axis < 3; axis++) {
              section.min[axis] = Math.min(section.min[axis], position[axis]);
              section.max[axis] = Math.max(section.max[axis], position[axis]);
            }
            vertexTotal++;
          }
          faceVertices.push(vertex);
        }
        for (let triangle = 1; triangle < faceVertices.length - 1; triangle++) {
          section.indices.push(faceVertices[0], faceVertices[triangle], faceVertices[triangle + 1]);
        }
        indexTotal += nextIndices;
        faceSerial++;
        break;
      }
      case 'o':
        if (!args.length) fail(line, 'o requires a name');
        objectName = args.join(' ');
        groupName = '';
        flush();
        break;
      case 'g':
        groupName = args.join(' ');
        flush();
        break;
      case 'usemtl':
        if (!args.length) fail(line, 'usemtl requires a name');
        materialName = args.join(' ');
        flush();
        break;
      case 'mtllib':
        if (!args.length) fail(line, 'mtllib requires a file name');
        materialLibraries.push(...args);
        break;
      case 's':
        if (args.length !== 1) fail(line, 's requires one value');
        if (!smoothingWarned) {
          warnings.push('OBJ smoothing groups are ignored; faces without vn receive flat normals.');
          smoothingWarned = true;
        }
        break;
      default:
        fail(line, `unsupported directive '${directive}'`);
    }
  }
  flush();
  if (!primitives.length) throw new Error('OBJ contains no faces');
  return { primitives, materialLibraries, warnings };
}
