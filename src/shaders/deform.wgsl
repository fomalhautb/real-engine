struct Vertex {
    position: vec4f,
    normal: vec4f,
    joints: vec4u,
    weights: vec4f,
    morph: vec4f,
}
struct Instance {
    model: mat4x4f,
    previousModel: mat4x4f,
    normalMatrix: mat4x4f,
    color: vec4f,
    material: vec4f,
}
struct Deformed {
    position: vec4f,
    normal: vec4f,
    previous: vec4f,
}
struct Parameters {
    vertexCount: u32,
    instanceCount: u32,
    jointCount: u32,
    scalarCount: u32,
}
struct MorphDelta { position: vec4f, normal: vec4f }
struct DeformationResult { position: vec3f, normal: vec3f }
@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> palettes: array<mat4x4f>;
@group(0) @binding(4) var<storage, read_write> output: array<Deformed>;
@group(0) @binding(5) var<storage, read> morphs: array<MorphDelta>;
@group(0) @binding(6) var<storage, read> scalars: array<f32>;

fn evaluate(vertexIndex: u32, paletteBase: u32, scalarBase: u32) -> DeformationResult {
    let vertex = vertices[vertexIndex];
    var p = vertex.position.xyz;
    var n = vertex.normal.xyz;
    // OPERATOR_BODY
    return DeformationResult(p, normalize(n));
}

// Generated operators share this current/previous execution and output contract.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3u) {
    let index = invocation.x;
    if (index >= params.vertexCount * params.instanceCount) { return; }
    let instanceIndex = index / params.vertexCount;
    let instance = instances[instanceIndex];
    let paletteBase = instanceIndex * params.jointCount * 2u;
    let scalarBase = instanceIndex * params.scalarCount * 2u;
    let current = evaluate(index % params.vertexCount, paletteBase, scalarBase);
    let previous = evaluate(index % params.vertexCount, paletteBase + params.jointCount, scalarBase + params.scalarCount);
    output[index].position = instance.model * vec4f(current.position, 1.0);
    output[index].previous = instance.previousModel * vec4f(previous.position, 1.0);
    output[index].normal = vec4f(normalize((instance.normalMatrix * vec4f(current.normal, 0.0)).xyz), 0.0);
}
