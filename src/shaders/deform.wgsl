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
    tangent: vec4f,
}
struct Parameters {
    vertexCount: u32,
    instanceCount: u32,
    jointCount: u32,
    scalarCount: u32,
}
struct MorphDelta { position: vec4f, normal: vec4f, tangent: vec4f }
struct DeformationResult { position: vec3f, normal: vec3f, tangent: vec4f }
@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> palettes: array<mat4x4f>;
@group(0) @binding(4) var<storage, read_write> output: array<Deformed>;
@group(0) @binding(5) var<storage, read> morphs: array<MorphDelta>;
@group(0) @binding(6) var<storage, read> scalars: array<f32>;
@group(0) @binding(7) var<storage, read> restTangents: array<vec4f>;

// Scale before normalization so finite authored f32 vectors cannot overflow a dot product.
fn deformationSafeNormal(value: vec3f) -> vec3f {
    let magnitude = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (magnitude > 0.0 && all(abs(value) <= vec3f(3.402823e38f))) {
        return normalize(value / magnitude);
    }
    return vec3f(0.0, 1.0, 0.0);
}

// A collapsed tangent still has a deterministic frame; missing authored tangents keep w=0.
fn deformationOrthogonalTangent(value: vec3f, normal: vec3f) -> vec3f {
    let magnitude = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (magnitude > 0.0 && all(abs(value) <= vec3f(3.402823e38f))) {
        let scaled = value / magnitude;
        let orthogonal = scaled - normal * dot(normal, scaled);
        if (dot(orthogonal, orthogonal) > 0.000000000001f) { return normalize(orthogonal); }
    }
    var axis = vec3f(0.0, 0.0, 1.0);
    if (abs(normal.x) <= abs(normal.y) && abs(normal.x) <= abs(normal.z)) { axis = vec3f(1.0, 0.0, 0.0); }
    else if (abs(normal.y) <= abs(normal.z)) { axis = vec3f(0.0, 1.0, 0.0); }
    return normalize(axis - normal * dot(axis, normal));
}

// Only orientation is needed; the scaled determinant is bounded even for large affine matrices.
// A singular weighted blend has no defined orientation and retains the previous handedness.
fn deformationOrientation(c0: vec3f, c1: vec3f, c2: vec3f) -> f32 {
    let magnitude = max(max(max(abs(c0.x), abs(c0.y)), abs(c0.z)),
        max(max(max(abs(c1.x), abs(c1.y)), abs(c1.z)), max(max(abs(c2.x), abs(c2.y)), abs(c2.z))));
    if (magnitude > 0.0 && magnitude <= 3.402823e38f &&
        all(abs(c0) <= vec3f(3.402823e38f)) && all(abs(c1) <= vec3f(3.402823e38f)) && all(abs(c2) <= vec3f(3.402823e38f))) {
        let determinant = dot(c0 / magnitude, cross(c1 / magnitude, c2 / magnitude));
        return select(1.0, -1.0, determinant < 0.0);
    }
    return 1.0;
}

fn evaluate(vertexIndex: u32, paletteBase: u32, scalarBase: u32) -> DeformationResult {
    let vertex = vertices[vertexIndex];
    var p = vertex.position.xyz;
    var n = vertex.normal.xyz;
    var t = restTangents[vertexIndex];
    // OPERATOR_BODY
    return DeformationResult(p, n, t);
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
    let worldNormal = deformationSafeNormal((instance.normalMatrix * vec4f(current.normal, 0.0)).xyz);
    output[index].normal = vec4f(worldNormal, 0.0);
    var worldTangent = vec4f(0.0);
    if (current.tangent.w != 0.0) {
        let direction = deformationOrthogonalTangent((instance.model * vec4f(current.tangent.xyz, 0.0)).xyz, worldNormal);
        let orientation = deformationOrientation(instance.model[0].xyz, instance.model[1].xyz, instance.model[2].xyz);
        worldTangent = vec4f(direction, current.tangent.w * orientation);
    }
    output[index].tangent = worldTangent;
}
