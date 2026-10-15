import type { Mat4 } from './math';
import { validateTangent } from './tangent-frame';

export type Vec3 = readonly [number, number, number];
export interface Bounds3 { min: [number, number, number]; max: [number, number, number] }

/** Scalars are per-instance, per-frame values; node constants form the pipeline key. */
export type DeformationNode =
  | Readonly<{ kind: 'morph'; target: number; weight: number }>
  | Readonly<{ kind: 'skin'; normalMode?: 'rigid' | 'affine' }>
  | Readonly<{ kind: 'wave'; amplitude: number; frequency: number; phase: number }>;
export interface DeformationGraph { jointCount: number; nodes: readonly DeformationNode[] }
export interface MorphTarget { positions: Float32Array; normals: Float32Array; tangents?: Float32Array }
export interface DeformationVertex {
  position: Vec3;
  normal: Vec3;
  /** Authored local tangent; w is signed bitangent orientation, never a vector component. */
  tangent?: readonly [number, number, number, number];
  joints: readonly [number, number, number, number];
  weights: readonly [number, number, number, number];
}
export interface CompiledDeformation {
  readonly key: string;
  readonly jointCount: number;
  readonly scalarCount: number;
  readonly morphTargetCount: number;
  readonly nodes: readonly DeformationNode[];
  /** See compileDeformation for the surrounding WGSL interface. */
  readonly wgslBody: string;
}

export const DEFORMATION_LIMITS = Object.freeze({ nodes: 32, joints: 1024, scalars: 128, morphTargets: 32 });

function finite(value: number, label: string): void {
  if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new RangeError(`${label} must be a finite f32`);
  }
}
function index(value: number, limit: number, label: string): void {
  if (!Number.isInteger(value) || value < 0 || value >= limit) {
    throw new RangeError(`${label} must be an integer in [0, ${limit})`);
  }
}
function floatLiteral(value: number): string {
  const text = Math.fround(value).toString();
  return /[.e]/i.test(text) ? `${text}f` : `${text}.0f`;
}

/**
 * Compile a bounded, ordered deformation graph into one fused compute function body.
 * The caller supplies p/n (var vec3f), t (var vec4f), vertexIndex/paletteBase/scalarBase (u32),
 * vertex.joints (vec4u), vertex.weights (vec4f), params.vertexCount (u32), and storage
 * arrays palettes:mat4x4f, scalars:f32, morphs:MorphDelta {position/normal/tangent:vec4f},
 * and deformationOrthogonalTangent/deformationOrientation helpers from deform.wgsl.
 * Morph storage is target-major, with one delta per original vertex. A mesh's
 * instance/model transform is applied by the caller after this local-space graph.
 *
 * Four influences per vertex may address any of jointCount bones. Bones must be
 * rigid by default; affine mode uses the inverse transpose of each vertex's
 * weighted blended linear transform. Morph deltas
 * belong to the coordinate space at their node, so ordering is significant.
 * No intermediate normal normalization discards morph normal magnitude.
 */
export function compileDeformation(graph: DeformationGraph): CompiledDeformation {
  index(graph.jointCount, DEFORMATION_LIMITS.joints + 1, 'jointCount');
  if (!Array.isArray(graph.nodes) || graph.nodes.length > DEFORMATION_LIMITS.nodes) {
    throw new RangeError(`deformation supports at most ${DEFORMATION_LIMITS.nodes} nodes`);
  }
  let scalarCount = 0, morphTargetCount = 0;
  const nodes = graph.nodes.map((node): DeformationNode => {
    switch (node.kind) {
      case 'morph':
        index(node.target, DEFORMATION_LIMITS.morphTargets, 'morph.target');
        index(node.weight, DEFORMATION_LIMITS.scalars, 'morph.weight');
        scalarCount = Math.max(scalarCount, node.weight + 1);
        morphTargetCount = Math.max(morphTargetCount, node.target + 1);
        return Object.freeze({ kind: 'morph', target: node.target, weight: node.weight });
      case 'skin':
        if (!graph.jointCount) throw new RangeError('skin requires at least one joint');
        if (node.normalMode !== undefined && node.normalMode !== 'rigid' && node.normalMode !== 'affine') {
          throw new RangeError('skin.normalMode must be rigid or affine');
        }
        // Preserve existing pipeline keys for omitted and explicit rigid mode.
        return Object.freeze(node.normalMode === 'affine' ? { kind: 'skin', normalMode: 'affine' } : { kind: 'skin' });
      case 'wave':
        finite(node.amplitude, 'wave.amplitude');
        finite(node.frequency, 'wave.frequency');
        finite(node.amplitude * node.frequency, 'wave normal derivative');
        index(node.phase, DEFORMATION_LIMITS.scalars, 'wave.phase');
        scalarCount = Math.max(scalarCount, node.phase + 1);
        return Object.freeze({ kind: 'wave', amplitude: Math.fround(node.amplitude), frequency: Math.fround(node.frequency), phase: node.phase });
      default:
        throw new TypeError(`unknown deformation node: ${(node as { kind: string }).kind}`);
    }
  });
  const body = nodes.map((node) => {
    switch (node.kind) {
      case 'morph': return `{
    let delta = morphs[${node.target}u * params.vertexCount + vertexIndex];
    let weight = scalars[scalarBase + ${node.weight}u];
    p += delta.position.xyz * weight;
    n += delta.normal.xyz * weight;
    t = vec4f(t.xyz + delta.tangent.xyz * weight, t.w);
}`;
      case 'skin': return node.normalMode === 'affine' ? `{
    let weights = vertex.weights / dot(vertex.weights, vec4f(1.0));
    var skinnedPosition = vec3f(0.0);
    var rotatedNormal = vec3f(0.0);
    var skinnedTangent = vec3f(0.0);
    var c0 = vec3f(0.0);
    var c1 = vec3f(0.0);
    var c2 = vec3f(0.0);
    for (var influence = 0u; influence < 4u; influence++) {
        let bone = palettes[paletteBase + vertex.joints[influence]];
        let weight = weights[influence];
        skinnedPosition += (bone * vec4f(p, 1.0)).xyz * weight;
        rotatedNormal += (bone * vec4f(n, 0.0)).xyz * weight;
        skinnedTangent += (bone * vec4f(t.xyz, 0.0)).xyz * weight;
        c0 += bone[0].xyz * weight;
        c1 += bone[1].xyz * weight;
        c2 += bone[2].xyz * weight;
    }
    p = skinnedPosition;
    t = vec4f(skinnedTangent, t.w * deformationOrientation(c0, c1, c2));
    let matrixScale = max(
        max(max(abs(c0.x), abs(c0.y)), abs(c0.z)),
        max(max(max(abs(c1.x), abs(c1.y)), abs(c1.z)), max(max(abs(c2.x), abs(c2.y)), abs(c2.z))));
    var affineNormal = rotatedNormal;
    if (matrixScale > 0.0 && matrixScale <= 3.402823e38f) {
        let a = c0 / matrixScale;
        let b = c1 / matrixScale;
        let c = c2 / matrixScale;
        let cof0 = cross(b, c);
        let cof1 = cross(c, a);
        let cof2 = cross(a, b);
        let determinant = dot(a, cof0);
        if (abs(determinant) > 0.000001f) {
            let candidate = (cof0 * n.x + cof1 * n.y + cof2 * n.z) / (matrixScale * determinant);
            if (all(abs(candidate) <= vec3f(3.402823e38f))) { affineNormal = candidate; }
        }
    }
    if (all(abs(affineNormal) <= vec3f(3.402823e38f))) { n = affineNormal; }
    else { n = vec3f(0.0); }
}` : `{
    let weights = vertex.weights / dot(vertex.weights, vec4f(1.0));
    var skinnedPosition = vec3f(0.0);
    var skinnedNormal = vec3f(0.0);
    var skinnedTangent = vec3f(0.0);
    var c0 = vec3f(0.0);
    var c1 = vec3f(0.0);
    var c2 = vec3f(0.0);
    for (var influence = 0u; influence < 4u; influence++) {
        let bone = palettes[paletteBase + vertex.joints[influence]];
        skinnedPosition += (bone * vec4f(p, 1.0)).xyz * weights[influence];
        skinnedNormal += (bone * vec4f(n, 0.0)).xyz * weights[influence];
        skinnedTangent += (bone * vec4f(t.xyz, 0.0)).xyz * weights[influence];
        c0 += bone[0].xyz * weights[influence];
        c1 += bone[1].xyz * weights[influence];
        c2 += bone[2].xyz * weights[influence];
    }
    p = skinnedPosition;
    n = skinnedNormal;
    t = vec4f(skinnedTangent, t.w * deformationOrientation(c0, c1, c2));
}`;
      case 'wave': return `{
    let angle = p.y * ${floatLiteral(node.frequency)} + scalars[scalarBase + ${node.phase}u];
    let slope = ${floatLiteral(node.amplitude)} * ${floatLiteral(node.frequency)} * cos(angle);
    p.x += ${floatLiteral(node.amplitude)} * sin(angle);
    n.y -= slope * n.x;
    t.x += slope * t.y;
}`;
    }
  }).join('\n');
  // Degenerate blends have a deterministic finite fallback rather than normalize(0).
  const normalization = `
{
    let magnitude = max(abs(n.x), max(abs(n.y), abs(n.z)));
    if (magnitude > 0.0 && all(abs(n) <= vec3f(3.402823e38f))) { n = normalize(n / magnitude); }
    else { n = vec3f(0.0, 1.0, 0.0); }
    if (t.w != 0.0) { t = vec4f(deformationOrthogonalTangent(t.xyz, n), t.w); }
    else { t = vec4f(0.0); }
}`;
  return Object.freeze({
    key: `deformation-v1:${JSON.stringify({ jointCount: graph.jointCount, nodes })}`,
    jointCount: graph.jointCount, scalarCount, morphTargetCount,
    nodes: Object.freeze(nodes), wgslBody: body + normalization,
  });
}

function validateAffine(matrix: ArrayLike<number>, label: string): void {
  if (matrix.length !== 16) throw new RangeError(`${label} must have 16 components`);
  for (let i = 0; i < 16; i++) finite(matrix[i], `${label}[${i}]`);
  if (matrix[3] !== 0 || matrix[7] !== 0 || matrix[11] !== 0 || matrix[15] !== 1) {
    throw new RangeError(`${label} must be affine`);
  }
}

/** Validate pose inputs once on submission, not once per shader vertex. */
export function validateDeformationState(
  program: CompiledDeformation, palette: readonly Mat4[], scalars: ArrayLike<number>,
): void {
  if (palette.length < program.jointCount) throw new RangeError('palette has fewer matrices than jointCount');
  if (scalars.length < program.scalarCount) throw new RangeError('deformation scalar array is too short');
  for (let i = 0; i < program.scalarCount; i++) finite(scalars[i], `scalar[${i}]`);
  const usesSkin = program.nodes.some((node) => node.kind === 'skin');
  const needsRigid = program.nodes.some((node) => node.kind === 'skin' && node.normalMode !== 'affine');
  for (let i = 0; i < program.jointCount; i++) {
    const matrix = palette[i];
    validateAffine(matrix, `bone[${i}]`);
    if (!usesSkin) continue;
    if (needsRigid) {
      for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) {
        let dot = 0;
        for (let r = 0; r < 3; r++) dot += matrix[a * 4 + r] * matrix[b * 4 + r];
        if (Math.abs(dot - Number(a === b)) > 0.0001) {
          throw new RangeError(`bone[${i}] must be rigid (no scale or shear)`);
        }
      }
    }
    const determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
      - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
      + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2]);
    if (!Number.isFinite(determinant) || determinant === 0) throw new RangeError(`bone[${i}] must be invertible`);
    if (needsRigid && determinant < 0) throw new RangeError(`bone[${i}] must not reflect orientation`);
  }
}

/** Asset validation is performed when uploading/replacing data, independently of poses. */
export function validateDeformationVertex(program: CompiledDeformation, vertex: DeformationVertex): void {
  if (vertex.position.length !== 3 || vertex.normal.length !== 3 || vertex.joints.length !== 4 || vertex.weights.length !== 4) {
    throw new RangeError('vertex requires three position/normal components and four joint/weight components');
  }
  for (let i = 0; i < 3; i++) {
    finite(vertex.position[i], `position[${i}]`);
    finite(vertex.normal[i], `normal[${i}]`);
  }
  if (vertex.tangent !== undefined) validateTangent(vertex.tangent);
  if (!program.nodes.some((node) => node.kind === 'skin')) return;
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    index(vertex.joints[i], program.jointCount, `joint[${i}]`);
    finite(vertex.weights[i], `weight[${i}]`);
    if (vertex.weights[i] < 0) throw new RangeError('skin weights must be nonnegative');
    sum += vertex.weights[i];
  }
  if (Math.abs(sum - 1) > 0.0001) throw new RangeError('skin weights must sum to one');
}

/** Validates packed xyz arrays and returns signed position-delta envelopes for bounds. */
export function validateMorphTargets(
  program: CompiledDeformation, vertexCount: number, targets: readonly MorphTarget[],
): Bounds3[] {
  if (!Number.isSafeInteger(vertexCount) || vertexCount <= 0) throw new RangeError('vertexCount must be a positive integer');
  if (targets.length < program.morphTargetCount) throw new RangeError('morph target array is too short');
  return targets.slice(0, program.morphTargetCount).map((target, targetIndex) => {
    if (target.positions.length !== vertexCount * 3 || target.normals.length !== vertexCount * 3) {
      throw new RangeError(`morph target ${targetIndex} must contain vertexCount packed xyz position and normal deltas`);
    }
    if (target.tangents !== undefined && target.tangents.length !== vertexCount * 3) {
      throw new RangeError(`morph target ${targetIndex} tangents must contain vertexCount packed xyz deltas`);
    }
    const bounds: Bounds3 = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (let i = 0; i < target.positions.length; i++) {
      finite(target.positions[i], `morph[${targetIndex}].positions[${i}]`);
      finite(target.normals[i], `morph[${targetIndex}].normals[${i}]`);
      if (target.tangents !== undefined) finite(target.tangents[i], `morph[${targetIndex}].tangents[${i}]`);
      const axis = i % 3;
      bounds.min[axis] = Math.min(bounds.min[axis], target.positions[i]);
      bounds.max[axis] = Math.max(bounds.max[axis], target.positions[i]);
    }
    return bounds;
  });
}

function normalize(value: Vec3): [number, number, number] {
  const length = Math.hypot(...value);
  return length > 0 ? [value[0] / length, value[1] / length, value[2] / length] : [0, 1, 0];
}
function transform(matrix: ArrayLike<number>, value: Vec3, w: number): [number, number, number] {
  return [0, 1, 2].map((r) => matrix[r] * value[0] + matrix[4 + r] * value[1] + matrix[8 + r] * value[2] + matrix[12 + r] * w) as [number, number, number];
}

function blendOrientation(columns: readonly number[]): number {
  const magnitude = Math.max(...columns.map(Math.abs));
  if (!(magnitude > 0) || !columns.every(value => Number.isFinite(Math.fround(value)))) return 1;
  const [a, d, g, b, e, h, c, f, i] = columns.map(value => value / magnitude);
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g) < 0 ? -1 : 1;
}

function orthogonalTangent(value: Vec3, normal: Vec3): [number, number, number] {
  const magnitude = Math.max(...value.map(Math.abs));
  if (magnitude > 0 && value.every(component => Number.isFinite(Math.fround(component)))) {
    const scaled = value.map(component => component / magnitude);
    const projection = scaled.reduce((sum, component, axis) => sum + component * normal[axis], 0);
    const orthogonal = scaled.map((component, axis) => component - normal[axis] * projection) as [number, number, number];
    if (orthogonal.reduce((sum, component) => sum + component * component, 0) > 1e-12) return normalize(orthogonal);
  }
  // Select the least-aligned cardinal direction to avoid an unstable fallback near a pole.
  const axis = Math.abs(normal[0]) <= Math.abs(normal[1]) && Math.abs(normal[0]) <= Math.abs(normal[2]) ? 0
    : Math.abs(normal[1]) <= Math.abs(normal[2]) ? 1 : 2;
  return normalize(normal.map((component, componentAxis) => Number(componentAxis === axis) - component * normal[axis]) as [number, number, number]);
}

/** Inverse transpose of the vertex's weighted blend; a singular blend uses its finite rotated normal. */
function affineSkinNormal(columns: readonly number[], normal: Vec3, rotatedNormal: Vec3): [number, number, number] {
  const fallback: [number, number, number] = rotatedNormal.every((value) => Number.isFinite(Math.fround(value)))
    ? [...rotatedNormal] : [0, 0, 0];
  const matrixScale = Math.max(...columns.map(Math.abs));
  if (!(matrixScale > 0) || !Number.isFinite(Math.fround(matrixScale))) return fallback;
  const a = columns.slice(0, 3).map((value) => value / matrixScale);
  const b = columns.slice(3, 6).map((value) => value / matrixScale);
  const c = columns.slice(6, 9).map((value) => value / matrixScale);
  const cross = (x: readonly number[], y: readonly number[]): [number, number, number] => [
    x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0],
  ];
  const cof0 = cross(b, c), cof1 = cross(c, a), cof2 = cross(a, b);
  const determinant = a[0] * cof0[0] + a[1] * cof0[1] + a[2] * cof0[2];
  if (!(Math.abs(determinant) > 0.000001)) return fallback;
  const result: [number, number, number] = [0, 1, 2].map((row) =>
    (cof0[row] * normal[0] + cof1[row] * normal[1] + cof2[row] * normal[2]) / (matrixScale * determinant)) as [number, number, number];
  return result.every((value) => Number.isFinite(Math.fround(value))) ? result : fallback;
}

/** CPU reference for correctness tests/tools; production rendering evaluates the WGSL. */
export function evaluateDeformation(
  program: CompiledDeformation, vertex: DeformationVertex, vertexIndex: number,
  targets: readonly MorphTarget[], palette: readonly Mat4[], scalars: ArrayLike<number>,
): { position: [number, number, number]; normal: [number, number, number]; tangent?: [number, number, number, number] } {
  validateDeformationVertex(program, vertex);
  validateDeformationState(program, palette, scalars);
  if (!Number.isSafeInteger(vertexIndex) || vertexIndex < 0) throw new RangeError('vertexIndex must be a nonnegative integer');
  let p: [number, number, number] = [...vertex.position];
  let n: [number, number, number] = [...vertex.normal];
  let t: [number, number, number] | undefined = vertex.tangent ? vertex.tangent.slice(0, 3) as [number, number, number] : undefined;
  let orientation = vertex.tangent?.[3] ?? 0;
  for (const node of program.nodes) {
    switch (node.kind) {
      case 'morph': {
        const target = targets[node.target], offset = vertexIndex * 3;
        if (!target || target.positions.length <= offset + 2 || target.normals.length <= offset + 2) {
          throw new RangeError('morph target does not contain the requested vertex');
        }
        if (target.tangents !== undefined && target.tangents.length !== target.positions.length) {
          throw new RangeError('morph tangent deltas must contain packed xyz for every target vertex');
        }
        for (let axis = 0; axis < 3; axis++) {
          finite(target.positions[offset + axis], 'morph position');
          finite(target.normals[offset + axis], 'morph normal');
          if (target.tangents !== undefined) finite(target.tangents[offset + axis], 'morph tangent');
          p[axis] += target.positions[offset + axis] * scalars[node.weight];
          n[axis] += target.normals[offset + axis] * scalars[node.weight];
          if (t && target.tangents !== undefined) t[axis] += target.tangents[offset + axis] * scalars[node.weight];
        }
        break;
      }
      case 'skin': {
        const position: [number, number, number] = [0, 0, 0], normal: [number, number, number] = [0, 0, 0];
        const tangent: [number, number, number] = [0, 0, 0];
        const columns = node.normalMode === 'affine' || t ? new Array<number>(9).fill(0) : undefined;
        const sum = vertex.weights.reduce((total, weight) => total + weight, 0);
        for (let influence = 0; influence < 4; influence++) {
          const bone = palette[vertex.joints[influence]], weight = vertex.weights[influence] / sum;
          const bp = transform(bone, p, 1), bn = transform(bone, n, 0);
          const bt = t ? transform(bone, t, 0) : undefined;
          for (let axis = 0; axis < 3; axis++) {
            position[axis] += bp[axis] * weight;
            normal[axis] += bn[axis] * weight;
            if (bt) tangent[axis] += bt[axis] * weight;
          }
          if (columns) for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++) {
            columns[col * 3 + row] += bone[col * 4 + row] * weight;
          }
        }
        p = position; n = node.normalMode === 'affine' && columns ? affineSkinNormal(columns, n, normal) : normal;
        if (t && columns) { t = tangent; orientation *= blendOrientation(columns); }
        break;
      }
      case 'wave': {
        const angle = p[1] * node.frequency + scalars[node.phase];
        p[0] += node.amplitude * Math.sin(angle);
        n[1] -= node.amplitude * node.frequency * Math.cos(angle) * n[0];
        if (t) t[0] += node.amplitude * node.frequency * Math.cos(angle) * t[1];
        break;
      }
    }
  }
  for (const value of [...p, ...n]) finite(value, 'deformation result');
  const normal = normalize(n);
  return t ? { position: p, normal, tangent: [...orthogonalTangent(t, normal), orientation] } : { position: p, normal };
}

function validateBounds(bounds: Bounds3, label: string): void {
  for (let axis = 0; axis < 3; axis++) {
    finite(bounds.min[axis], `${label}.min[${axis}]`);
    finite(bounds.max[axis], `${label}.max[${axis}]`);
    if (bounds.min[axis] > bounds.max[axis]) throw new RangeError(`${label} has an inverted extent`);
  }
}
function transformedBounds(bounds: Bounds3, matrix: ArrayLike<number>): Bounds3 {
  const result: Bounds3 = { min: [0, 0, 0], max: [0, 0, 0] };
  for (let row = 0; row < 3; row++) {
    let min = matrix[12 + row], max = min;
    for (let axis = 0; axis < 3; axis++) {
      const a = bounds.min[axis] * matrix[axis * 4 + row], b = bounds.max[axis] * matrix[axis * 4 + row];
      min += Math.min(a, b); max += Math.max(a, b);
    }
    result.min[row] = min; result.max[row] = max;
  }
  return result;
}

/**
 * O(nodes * joints) production bound propagation, independent of vertex count.
 * Skin union is conservative for the validated nonnegative normalized weights;
 * signed morph scalars are supported. Full sine amplitude handles any phase.
 * These bounds deliberately trade tightness for guaranteed coverage; callers can
 * supply per-cluster base/morph envelopes for tighter culling. Optional model may
 * include affine scale/shear. Output includes an outward float roundoff margin.
 */
export function conservativeDeformationBounds(
  program: CompiledDeformation, baseBounds: Bounds3, morphBounds: readonly Bounds3[],
  palette: readonly Mat4[], scalars: ArrayLike<number>, model?: ArrayLike<number>,
): Bounds3 {
  validateBounds(baseBounds, 'base bounds');
  validateDeformationState(program, palette, scalars);
  let bounds: Bounds3 = { min: [...baseBounds.min], max: [...baseBounds.max] };
  for (const node of program.nodes) {
    if (node.kind === 'morph') {
      const delta = morphBounds[node.target];
      if (!delta) throw new RangeError('missing morph bounds');
      validateBounds(delta, 'morph bounds');
      for (let axis = 0; axis < 3; axis++) {
        const a = delta.min[axis] * scalars[node.weight], b = delta.max[axis] * scalars[node.weight];
        bounds.min[axis] += Math.min(a, b); bounds.max[axis] += Math.max(a, b);
      }
    } else if (node.kind === 'skin') {
      const union: Bounds3 = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
      for (let joint = 0; joint < program.jointCount; joint++) {
        const boneBounds = transformedBounds(bounds, palette[joint]);
        for (let axis = 0; axis < 3; axis++) {
          union.min[axis] = Math.min(union.min[axis], boneBounds.min[axis]);
          union.max[axis] = Math.max(union.max[axis], boneBounds.max[axis]);
        }
      }
      bounds = union;
    } else {
      bounds.min[0] -= Math.abs(node.amplitude); bounds.max[0] += Math.abs(node.amplitude);
    }
  }
  if (model) { validateAffine(model, 'model'); bounds = transformedBounds(bounds, model); }
  for (let axis = 0; axis < 3; axis++) {
    const margin = Math.max(1, Math.abs(bounds.min[axis]), Math.abs(bounds.max[axis])) * 0.00001;
    bounds.min[axis] -= margin; bounds.max[axis] += margin;
  }
  validateBounds(bounds, 'deformed bounds');
  return bounds;
}
