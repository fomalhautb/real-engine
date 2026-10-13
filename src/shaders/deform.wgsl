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
    padding: u32,
}
@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> palettes: array<mat4x4f>;
@group(0) @binding(4) var<storage, read_write> output: array<Deformed>;

// One implementation for every mesh. Pose generation never enters this kernel.
// The first slice supports rigid joint transforms and authored position morphs.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3u) {
    let index = invocation.x;
    if (index >= params.vertexCount * params.instanceCount) { return; }
    let instanceIndex = index / params.vertexCount;
    let vertex = vertices[index % params.vertexCount];
    let instance = instances[instanceIndex];
    let paletteBase = instanceIndex * params.jointCount * 2u;
    let currentPosition = vertex.position + vertex.morph * instance.material.z;
    let previousPosition = vertex.position + vertex.morph * instance.material.w;
    var current = vec4f(0.0);
    var previous = vec4f(0.0);
    var normal = vec4f(0.0);
    for (var joint = 0u; joint < 4u; joint++) {
        let weight = vertex.weights[joint];
        let bone = vertex.joints[joint];
        current += (palettes[paletteBase + bone] * currentPosition) * weight;
        previous += (palettes[paletteBase + params.jointCount + bone] * previousPosition) * weight;
        normal += (palettes[paletteBase + bone] * vertex.normal) * weight;
    }
    output[index].position = instance.model * current;
    output[index].previous = instance.previousModel * previous;
    output[index].normal = vec4f(normalize((instance.normalMatrix * normal).xyz), 0.0);
}
