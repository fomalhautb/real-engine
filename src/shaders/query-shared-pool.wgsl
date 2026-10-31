// Copies one refit batch into the shared static pool: world positions from the
// shared deformation output (bit-exact) and BLAS records that merge the batch's
// refit bounds with its fixed topology. Runs only for changed batches.
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
struct Bounds { low: vec4f, high: vec4f }
struct UpdateParams {
    shape: vec4u, // active instances, vertex count, BLAS node count, 0
    bases: vec4u, // node record of slot 0, pooled vertex of slot 0, 0, 0
}
@group(0) @binding(0) var<uniform> update: UpdateParams;
@group(0) @binding(1) var<storage, read> deformed: array<Deformed>;
// Batch topology words at offset 0: [left|first, right|count, escape, leaf].
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read> blasBounds: array<Bounds>;
@group(0) @binding(4) var<storage, read_write> poolNodes: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> poolPositions: array<f32>;

@compute @workgroup_size(64)
fn updatePool(@builtin(global_invocation_id) id: vec3u) {
    let vertexThreads = update.shape.x * update.shape.y;
    if (id.x < vertexThreads) {
        // Deformation output is instance-major, as is the pooled range.
        let p = deformed[id.x].position.xyz;
        let at = (update.bases.y + id.x) * 3u;
        poolPositions[at] = p.x; poolPositions[at + 1u] = p.y; poolPositions[at + 2u] = p.z;
        return;
    }
    let item = id.x - vertexThreads;
    if (item >= update.shape.x * update.shape.z) { return; }
    let node = item % update.shape.z;
    let box = blasBounds[item];
    let t = node * 4u;
    let info = select(0u, (source[t] << 4u) | source[t + 1u], source[t + 3u] == 1u);
    let record = (update.bases.x + item) * 2u;
    poolNodes[record] = vec4u(bitcast<vec3u>(box.low.xyz), source[t + 2u]);
    poolNodes[record + 1u] = vec4u(bitcast<vec3u>(box.high.xyz), info);
}
