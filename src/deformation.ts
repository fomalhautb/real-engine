import type { Mat4 } from './math';

export type Vec3 = readonly [number, number, number];
export interface Bounds3 { min: [number, number, number]; max: [number, number, number] }

/** Scalars are per-instance, per-frame values; node constants form the pipeline key. */
export type DeformationNode =
  | Readonly<{ kind: 'morph'; target: number; weight: number }>
  | Readonly<{ kind: 'skin' }>
  | Readonly<{ kind: 'wave'; amplitude: number; frequency: number; phase: number }>;
export interface DeformationGraph { jointCount: number; nodes: readonly DeformationNode[] }
export interface MorphTarget { positions: Float32Array; normals: Float32Array }
export interface DeformationVertex {
  position: Vec3;
  normal: Vec3;
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
 * The caller supplies p/n (var vec3f), vertexIndex/paletteBase/scalarBase (u32),
 * vertex.joints (vec4u), vertex.weights (vec4f), params.vertexCount (u32), and storage
 * arrays palettes:mat4x4f, scalars:f32, morphs:MorphDelta {position/normal:vec4f}.
 * Morph storage is target-major, with one delta per original vertex. A mesh's
 * instance/model transform is applied by the caller after this local-space graph.
 *
 * Four influences per vertex may address any of jointCount bones. Bones must be
 * rigid; blended normal rotation is the declared LBS approximation. Morph deltas
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
        return Object.freeze({ kind: 'skin' });
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
}`;
      case 'skin': return `{
    let weights = vertex.weights / dot(vertex.weights, vec4f(1.0));
    var skinnedPosition = vec3f(0.0);
    var skinnedNormal = vec3f(0.0);
    for (var influence = 0u; influence < 4u; influence++) {
        let bone = palettes[paletteBase + vertex.joints[influence]];
        skinnedPosition += (bone * vec4f(p, 1.0)).xyz * weights[influence];
        skinnedNormal += (bone * vec4f(n, 0.0)).xyz * weights[influence];
    }
    p = skinnedPosition;
    n = skinnedNormal;
}`;
      case 'wave': return `{
    let angle = p.y * ${floatLiteral(node.frequency)} + scalars[scalarBase + ${node.phase}u];
    let slope = ${floatLiteral(node.amplitude)} * ${floatLiteral(node.frequency)} * cos(angle);
    p.x += ${floatLiteral(node.amplitude)} * sin(angle);
    n.y -= slope * n.x;
}`;
    }
  }).join('\n');
  // Degenerate blends have a deterministic finite fallback rather than normalize(0).
  const normalization = `
{
    let magnitude = max(abs(n.x), max(abs(n.y), abs(n.z)));
    if (magnitude > 0.0) { n = normalize(n / magnitude); }
    else { n = vec3f(0.0, 1.0, 0.0); }
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
  for (let i = 0; i < program.jointCount; i++) {
    const matrix = palette[i];
    validateAffine(matrix, `bone[${i}]`);
    if (!usesSkin) continue;
    for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) {
      let dot = 0;
      for (let r = 0; r < 3; r++) dot += matrix[a * 4 + r] * matrix[b * 4 + r];
      if (Math.abs(dot - Number(a === b)) > 0.0001) {
        throw new RangeError(`bone[${i}] must be rigid (no scale or shear)`);
      }
    }
    const determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
      - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
      + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2]);
    if (determinant < 0) throw new RangeError(`bone[${i}] must not reflect orientation`);
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
    const bounds: Bounds3 = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    for (let i = 0; i < target.positions.length; i++) {
      finite(target.positions[i], `morph[${targetIndex}].positions[${i}]`);
      finite(target.normals[i], `morph[${targetIndex}].normals[${i}]`);
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

/** CPU reference for correctness tests/tools; production rendering evaluates the WGSL. */
export function evaluateDeformation(
  program: CompiledDeformation, vertex: DeformationVertex, vertexIndex: number,
  targets: readonly MorphTarget[], palette: readonly Mat4[], scalars: ArrayLike<number>,
): { position: [number, number, number]; normal: [number, number, number] } {
  validateDeformationVertex(program, vertex);
  validateDeformationState(program, palette, scalars);
  if (!Number.isSafeInteger(vertexIndex) || vertexIndex < 0) throw new RangeError('vertexIndex must be a nonnegative integer');
  let p: [number, number, number] = [...vertex.position];
  let n: [number, number, number] = [...vertex.normal];
  for (const node of program.nodes) {
    switch (node.kind) {
      case 'morph': {
        const target = targets[node.target], offset = vertexIndex * 3;
        if (!target || target.positions.length <= offset + 2 || target.normals.length <= offset + 2) {
          throw new RangeError('morph target does not contain the requested vertex');
        }
        for (let axis = 0; axis < 3; axis++) {
          finite(target.positions[offset + axis], 'morph position');
          finite(target.normals[offset + axis], 'morph normal');
          p[axis] += target.positions[offset + axis] * scalars[node.weight];
          n[axis] += target.normals[offset + axis] * scalars[node.weight];
        }
        break;
      }
      case 'skin': {
        const position: [number, number, number] = [0, 0, 0], normal: [number, number, number] = [0, 0, 0];
        const sum = vertex.weights.reduce((total, weight) => total + weight, 0);
        for (let influence = 0; influence < 4; influence++) {
          const bone = palette[vertex.joints[influence]], weight = vertex.weights[influence] / sum;
          const bp = transform(bone, p, 1), bn = transform(bone, n, 0);
          for (let axis = 0; axis < 3; axis++) {
            position[axis] += bp[axis] * weight;
            normal[axis] += bn[axis] * weight;
          }
        }
        p = position; n = normal;
        break;
      }
      case 'wave': {
        const angle = p[1] * node.frequency + scalars[node.phase];
        p[0] += node.amplitude * Math.sin(angle);
        n[1] -= node.amplitude * node.frequency * Math.cos(angle) * n[0];
        break;
      }
    }
  }
  for (const value of [...p, ...n]) finite(value, 'deformation result');
  return { position: p, normal: normalize(n) };
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
