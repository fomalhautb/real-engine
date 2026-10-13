/** Bounded, side-effect-free material expressions shared by raster and world-space queries. */
export type Vec4 = readonly [number, number, number, number];
export type Vec4Expr =
  | Readonly<{ kind: 'constant'; value: Vec4 }>
  | Readonly<{ kind: 'parameter'; slot: number }>
  | Readonly<{ kind: 'texture'; slot: number }>
  | Readonly<{ kind: 'instanceColor' }>
  | Readonly<{ kind: 'instanceMetalRough' }>
  | Readonly<{ kind: 'add' | 'multiply'; a: Vec4Expr; b: Vec4Expr }>
  | Readonly<{ kind: 'mix'; a: Vec4Expr; b: Vec4Expr; factor: ScalarExpr }>;
export type ScalarExpr = number | Readonly<{ kind: 'component'; value: Vec4Expr; channel: 0 | 1 | 2 | 3 }>;
export interface MaterialDefinition {
  readonly shading: 'pbr' | 'unlit';
  readonly alphaMode: 'opaque' | 'mask';
  readonly doubleSided: boolean;
  readonly baseColor: Vec4Expr;
  readonly metallic: ScalarExpr;
  readonly roughness: ScalarExpr;
  readonly emissive: Vec4Expr;
  readonly alphaCutoff: ScalarExpr;
}
export const MATERIAL_LIMITS = Object.freeze({ nodes: 128, depth: 32, parameters: 16, textures: 4 });

/** Builders do not compile or allocate GPU objects; compileMaterial validates and snapshots them. */
export const m = Object.freeze({
  constant: (value: Vec4): Vec4Expr => ({ kind: 'constant', value }),
  parameter: (slot: number): Vec4Expr => ({ kind: 'parameter', slot }),
  texture: (slot: number): Vec4Expr => ({ kind: 'texture', slot }),
  instanceColor: Object.freeze({ kind: 'instanceColor' } as const),
  instanceMetalRough: Object.freeze({ kind: 'instanceMetalRough' } as const),
  add: (a: Vec4Expr, b: Vec4Expr): Vec4Expr => ({ kind: 'add', a, b }),
  multiply: (a: Vec4Expr, b: Vec4Expr): Vec4Expr => ({ kind: 'multiply', a, b }),
  mix: (a: Vec4Expr, b: Vec4Expr, factor: ScalarExpr): Vec4Expr => ({ kind: 'mix', a, b, factor }),
  component: (value: Vec4Expr, channel: 0 | 1 | 2 | 3): ScalarExpr => ({ kind: 'component', value, channel }),
});

export interface MaterialProgram {
  /** Collision-free canonical static graph key. Runtime parameters and texture contents are excluded. */
  readonly key: string;
  readonly wgsl: string;
  /** Standalone coverage function; no surface/BSDF/emissive-only dependencies. */
  readonly coverageWgsl: string;
  readonly parameterCount: number;
  readonly textureSlots: readonly number[];
  readonly coverageParameterCount: number;
  readonly coverageTextureSlots: readonly number[];
  readonly shading: MaterialDefinition['shading'];
  readonly alphaMode: MaterialDefinition['alphaMode'];
  readonly doubleSided: boolean;
}
type Node =
  | Readonly<{ kind: 'constant'; value: Vec4 }>
  | Readonly<{ kind: 'number'; value: number }>
  | Readonly<{ kind: 'parameter' | 'texture'; slot: number }>
  | Readonly<{ kind: 'instanceColor' | 'instanceMetalRough' }>
  | Readonly<{ kind: 'add' | 'multiply'; a: number; b: number }>
  | Readonly<{ kind: 'mix'; a: number; b: number; factor: number }>
  | Readonly<{ kind: 'component'; value: number; channel: number }>;
interface CompiledGraph {
  readonly nodes: readonly Node[];
  readonly baseColor: number;
  readonly metallic: number;
  readonly roughness: number;
  readonly emissive: number;
  readonly alphaCutoff: number;
}
const graphs = new WeakMap<MaterialProgram, CompiledGraph>();

function finite(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new RangeError(`${label} must be a finite f32`);
  }
  return Math.fround(value) || 0;
}
function index(value: unknown, limit: number, label: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value >= limit) {
    throw new RangeError(`${label} must be an integer in [0, ${limit})`);
  }
  return value;
}
function vector(value: unknown, length: number, label: string): number[] {
  if ((!Array.isArray(value) && !(value instanceof Float32Array)) || value.length !== length) {
    throw new TypeError(`${label} must contain exactly ${length} numbers`);
  }
  return Array.from({ length }, (_, i) => finite(value[i], `${label}[${i}]`));
}
function literal(value: number): string {
  const s = value.toString();
  return /[.e]/i.test(s) ? `${s}f` : `${s}.0f`;
}
function dependencies(node: Node): readonly number[] {
  switch (node.kind) {
    case 'add': case 'multiply': return [node.a, node.b];
    case 'mix': return [node.a, node.b, node.factor];
    case 'component': return [node.value];
    default: return [];
  }
}
function reachable(nodes: readonly Node[], roots: readonly number[]): Set<number> {
  const found = new Set<number>();
  const visit = (id: number): void => {
    if (found.has(id)) return;
    found.add(id);
    dependencies(nodes[id]).forEach(visit);
  };
  roots.forEach(visit);
  return found;
}
function resources(nodes: readonly Node[], selected: ReadonlySet<number>): { parameterCount: number; textureSlots: readonly number[] } {
  let parameterCount = 0;
  const textures = new Set<number>();
  for (const id of selected) {
    const node = nodes[id];
    if (node.kind === 'parameter') parameterCount = Math.max(parameterCount, node.slot + 1);
    if (node.kind === 'texture') textures.add(node.slot);
  }
  return { parameterCount, textureSlots: Object.freeze([...textures].sort((a, b) => a - b)) };
}
function expression(node: Node): string {
  switch (node.kind) {
    case 'constant': return `vec4f(${node.value.map(literal).join(', ')})`;
    case 'number': return literal(node.value);
    case 'parameter': return `materialParameters.values[${node.slot}u]`;
    case 'texture': return `materialTexture${node.slot}(uv, lod)`;
    case 'instanceColor': return 'instanceColor';
    case 'instanceMetalRough': return 'vec4f(instanceMR, 0.0, 0.0)';
    case 'add': return `v${node.a} + v${node.b}`;
    case 'multiply': return `v${node.a} * v${node.b}`;
    case 'mix': return `mix(v${node.a}, v${node.b}, vec4f(v${node.factor}))`;
    case 'component': return `v${node.value}[${node.channel}u]`;
  }
}
const argumentsWGSL = 'uv: vec2f, lod: f32, instanceColor: vec4f, instanceMR: vec2f';
function emitNodes(nodes: readonly Node[], selected: ReadonlySet<number>): string {
  // Node IDs are topological. Repeated object references and structurally equal nodes share one let.
  return nodes.flatMap((node, id) => selected.has(id) ? [`    let v${id} = ${expression(node)};`] : []).join('\n');
}

/**
 * Texture helpers materialTexture0..3(uv:vec2f,lod:f32)->vec4f are supplied by the backend.
 * Explicit LOD makes evaluation legal in fragment, compute, and ray-hit contexts; helpers own
 * address/filter/color-space policy. Parameters use materialParameters.values: array<vec4f,16>.
 * PBR base color is linear reflectance in [0,1]; Unlit base color and emissive retain nonnegative
 * HDR values. Alpha/metallic/cutoff clamp to [0,1], roughness to [0.04,1]. Mask equality passes.
 * Opaque coverage always passes, regardless of stored opacity.
 */
export function compileMaterial(definition: MaterialDefinition): MaterialProgram {
  if (!definition || typeof definition !== 'object') throw new TypeError('material definition must be an object');
  if (definition.shading !== 'pbr' && definition.shading !== 'unlit') throw new TypeError('material.shading must be pbr or unlit');
  if (definition.alphaMode !== 'opaque' && definition.alphaMode !== 'mask') throw new TypeError('material.alphaMode must be opaque or mask');
  if (typeof definition.doubleSided !== 'boolean') throw new TypeError('material.doubleSided must be boolean');
  const nodes: Node[] = [], interned = new Map<string, number>();
  const snapshots = new WeakMap<object, { id: number; height: number; scalar: boolean }>();
  const visiting = new WeakSet<object>();
  let nodeCount = 0;
  const intern = (node: Node): number => {
    const key = JSON.stringify(node), existing = interned.get(key);
    if (existing !== undefined) return existing;
    if (nodes.length >= MATERIAL_LIMITS.nodes) throw new RangeError(`material exceeds ${MATERIAL_LIMITS.nodes} expression nodes`);
    const id = nodes.length;
    nodes.push(Object.freeze(node));
    interned.set(key, id);
    return id;
  };
  const read = (value: unknown, scalar: boolean, depth: number, path: string): { id: number; height: number } => {
    if (depth > MATERIAL_LIMITS.depth) throw new RangeError(`material expression exceeds depth ${MATERIAL_LIMITS.depth} at ${path}`);
    if (scalar && typeof value === 'number') return { id: intern({ kind: 'number', value: finite(value, path) }), height: 1 };
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${path} must be a ${scalar ? 'scalar' : 'vec4'} expression`);
    if (visiting.has(value)) throw new TypeError(`material expression cycle at ${path}`);
    const previous = snapshots.get(value);
    if (previous) {
      if (previous.scalar !== scalar) throw new TypeError(`${path} expression has the wrong value type`);
      if (depth + previous.height - 1 > MATERIAL_LIMITS.depth) throw new RangeError(`material expression exceeds depth ${MATERIAL_LIMITS.depth} at ${path}`);
      return previous;
    }
    if (++nodeCount > MATERIAL_LIMITS.nodes) throw new RangeError(`material exceeds ${MATERIAL_LIMITS.nodes} expression nodes`);
    visiting.add(value);
    const source = value as Record<string, unknown>;
    let node: Node, height = 1;
    const child = (v: unknown, isScalar: boolean, suffix: string): number => {
      const result = read(v, isScalar, depth + 1, `${path}.${suffix}`);
      height = Math.max(height, result.height + 1);
      return result.id;
    };
    if (scalar) {
      if (source.kind !== 'component') throw new TypeError(`${path} must be a number or component expression`);
      node = { kind: 'component', value: child(source.value, false, 'value'), channel: index(source.channel, 4, `${path}.channel`) };
    } else {
      switch (source.kind) {
        case 'constant': node = { kind: 'constant', value: Object.freeze(vector(source.value, 4, `${path}.value`)) as unknown as Vec4 }; break;
        case 'parameter': node = { kind: 'parameter', slot: index(source.slot, MATERIAL_LIMITS.parameters, `${path}.slot`) }; break;
        case 'texture': node = { kind: 'texture', slot: index(source.slot, MATERIAL_LIMITS.textures, `${path}.slot`) }; break;
        case 'instanceColor': case 'instanceMetalRough': node = { kind: source.kind }; break;
        case 'add': case 'multiply': node = { kind: source.kind, a: child(source.a, false, 'a'), b: child(source.b, false, 'b') }; break;
        case 'mix': node = { kind: 'mix', a: child(source.a, false, 'a'), b: child(source.b, false, 'b'), factor: child(source.factor, true, 'factor') }; break;
        default: throw new TypeError(`unknown vec4 material expression at ${path}: ${String(source.kind)}`);
      }
    }
    const result = { id: intern(node), height, scalar };
    snapshots.set(value, result);
    visiting.delete(value);
    return result;
  };
  const graph: CompiledGraph = Object.freeze({
    nodes,
    baseColor: read(definition.baseColor, false, 1, 'baseColor').id,
    metallic: read(definition.metallic, true, 1, 'metallic').id,
    roughness: read(definition.roughness, true, 1, 'roughness').id,
    emissive: read(definition.emissive, false, 1, 'emissive').id,
    alphaCutoff: read(definition.alphaCutoff, true, 1, 'alphaCutoff').id,
  });
  Object.freeze(nodes);
  const surfaceNodes = reachable(nodes, [graph.baseColor, graph.metallic, graph.roughness, graph.emissive, graph.alphaCutoff]);
  const coverageNodes = reachable(nodes, definition.alphaMode === 'mask' ? [graph.baseColor, graph.alphaCutoff] : []);
  const coverageWgsl = `fn evaluateCoverage(${argumentsWGSL}) -> bool {\n${emitNodes(nodes, coverageNodes)}\n    return ${definition.alphaMode === 'opaque' ? 'true' : `clamp(v${graph.baseColor}.a, 0.0, 1.0) >= clamp(v${graph.alphaCutoff}, 0.0, 1.0)`};\n}`;
  const baseColorWgsl = definition.shading === 'pbr'
    ? `clamp(v${graph.baseColor}.rgb, vec3f(0.0), vec3f(1.0))`
    : `max(v${graph.baseColor}.rgb, vec3f(0.0))`;
  const wgsl = `struct MaterialSurface {\n    baseColor: vec3f,\n    opacity: f32,\n    metallic: f32,\n    roughness: f32,\n    emissive: vec3f,\n    alphaCutoff: f32,\n}\nfn evaluateMaterial(${argumentsWGSL}) -> MaterialSurface {\n${emitNodes(nodes, surfaceNodes)}\n    return MaterialSurface(\n        ${baseColorWgsl}, clamp(v${graph.baseColor}.a, 0.0, 1.0),\n        clamp(v${graph.metallic}, 0.0, 1.0), clamp(v${graph.roughness}, 0.04, 1.0),\n        max(v${graph.emissive}.rgb, vec3f(0.0)), clamp(v${graph.alphaCutoff}, 0.0, 1.0)\n    );\n}\n${coverageWgsl}`;
  const surfaceResources = resources(nodes, surfaceNodes), coverageResources = resources(nodes, coverageNodes);
  const program: MaterialProgram = Object.freeze({
    key: `material-v1:${JSON.stringify([definition.shading, definition.alphaMode, definition.doubleSided, graph])}`,
    wgsl, coverageWgsl, ...surfaceResources,
    coverageParameterCount: coverageResources.parameterCount,
    coverageTextureSlots: coverageResources.textureSlots,
    shading: definition.shading, alphaMode: definition.alphaMode, doubleSided: definition.doubleSided,
  });
  graphs.set(program, graph);
  return program;
}

export interface MaterialEvaluationInputs {
  readonly parameters?: Float32Array;
  readonly textures?: (slot: number, uv: readonly [number, number], lod: number) => Vec4;
  readonly uv?: readonly [number, number];
  readonly lod?: number;
  readonly instanceColor: Vec4;
  readonly instanceMetalRough: readonly [number, number];
}
export interface MaterialEvaluation {
  readonly baseColor: readonly [number, number, number];
  readonly opacity: number;
  readonly metallic: number;
  readonly roughness: number;
  readonly emissive: readonly [number, number, number];
  readonly alphaCutoff: number;
  readonly covered: boolean;
}

/** Deterministic f32 reference for diagnostics; production geometry does not read back for this. */
export function evaluateMaterial(program: MaterialProgram, inputs: MaterialEvaluationInputs): MaterialEvaluation {
  const graph = graphs.get(program);
  if (!graph) throw new TypeError('program must be returned by compileMaterial');
  const instanceColor = vector(inputs.instanceColor, 4, 'instanceColor');
  const instanceMR = vector(inputs.instanceMetalRough, 2, 'instanceMetalRough');
  const uv = Object.freeze(vector(inputs.uv ?? [0, 0], 2, 'uv')) as unknown as readonly [number, number];
  const lod = finite(inputs.lod ?? 0, 'lod');
  if (program.parameterCount && (!(inputs.parameters instanceof Float32Array) || inputs.parameters.length < program.parameterCount * 4)) {
    throw new RangeError(`material needs ${program.parameterCount * 4} packed parameter values`);
  }
  const values: (number | number[])[] = [];
  const vec = (id: number): number[] => values[id] as number[];
  const scalar = (id: number): number => values[id] as number;
  for (const node of graph.nodes) {
    let result: number | number[];
    switch (node.kind) {
      case 'constant': result = [...node.value]; break;
      case 'number': result = node.value; break;
      case 'parameter': result = Array.from({ length: 4 }, (_, i) => finite(inputs.parameters![node.slot * 4 + i], `parameters[${node.slot * 4 + i}]`)); break;
      case 'texture':
        if (typeof inputs.textures !== 'function') throw new TypeError(`material needs texture sampler slot ${node.slot}`);
        result = vector(inputs.textures(node.slot, uv, lod), 4, `texture[${node.slot}]`); break;
      case 'instanceColor': result = instanceColor; break;
      case 'instanceMetalRough': result = [instanceMR[0], instanceMR[1], 0, 0]; break;
      case 'component': result = vec(node.value)[node.channel]; break;
      case 'add': result = vec(node.a).map((v, i) => Math.fround(v + vec(node.b)[i])); break;
      case 'multiply': result = vec(node.a).map((v, i) => Math.fround(v * vec(node.b)[i])); break;
      case 'mix': result = vec(node.a).map((v, i) => Math.fround(Math.fround(v * Math.fround(1 - scalar(node.factor))) + Math.fround(vec(node.b)[i] * scalar(node.factor)))); break;
    }
    values.push(result);
  }
  const clamp = (v: number, low = 0, high = 1) => Math.min(high, Math.max(low, v));
  const color = vec(graph.baseColor), emissive = vec(graph.emissive);
  const opacity = clamp(color[3]), alphaCutoff = clamp(scalar(graph.alphaCutoff));
  return {
    baseColor: [clamp(color[0], 0, program.shading === 'pbr' ? 1 : Infinity),
      clamp(color[1], 0, program.shading === 'pbr' ? 1 : Infinity),
      clamp(color[2], 0, program.shading === 'pbr' ? 1 : Infinity)], opacity,
    metallic: clamp(scalar(graph.metallic)), roughness: clamp(scalar(graph.roughness), Math.fround(0.04)),
    emissive: [Math.max(emissive[0], 0), Math.max(emissive[1], 0), Math.max(emissive[2], 0)], alphaCutoff,
    covered: program.alphaMode === 'opaque' || opacity >= alphaCutoff,
  };
}

export const DEFAULT_MATERIAL_DEFINITION: MaterialDefinition = Object.freeze({
  shading: 'pbr', alphaMode: 'opaque', doubleSided: false,
  baseColor: m.instanceColor,
  metallic: Object.freeze({ kind: 'component', value: m.instanceMetalRough, channel: 0 }),
  roughness: Object.freeze({ kind: 'component', value: m.instanceMetalRough, channel: 1 }),
  emissive: Object.freeze({ kind: 'constant', value: Object.freeze([0, 0, 0, 0]) as Vec4 }),
  alphaCutoff: 0.5,
});
export const DEFAULT_MATERIAL_PROGRAM = compileMaterial(DEFAULT_MATERIAL_DEFINITION);
