struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
struct Bounds { low: vec4f, high: vec4f }
struct Params {
    shape: vec4u, // node count, vertex count, instance count, batch ID
    offsets: vec4u, // triangle IDs, indices, UVs, this depth's node IDs
    level: vec4u, // number of nodes at this depth
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<Deformed>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read_write> bounds: array<Bounds>;

// Each depth is a separate pass. Children are completely written before their
// parents read them; a workgroup barrier alone cannot provide this ordering.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.level.x * params.shape.z) { return; }
    let instance = id.x / params.level.x;
    let node = source[params.offsets.w + id.x % params.level.x];
    let base = instance * params.shape.x;
    let first = source[node * 4u];
    let second = source[node * 4u + 1u];
    var low = vec3f(3.402823466e+38);
    var high = -low;
    if (source[node * 4u + 3u] == 1u) {
        for (var i = 0u; i < second; i++) {
            let triangle = source[params.offsets.x + first + i];
            for (var corner = 0u; corner < 3u; corner++) {
                let vertex = source[params.offsets.y + triangle * 3u + corner];
                let p = vertices[instance * params.shape.y + vertex].position.xyz;
                low = min(low, p); high = max(high, p);
            }
        }
    } else {
        low = min(bounds[base + first].low.xyz, bounds[base + second].low.xyz);
        high = max(bounds[base + first].high.xyz, bounds[base + second].high.xyz);
    }
    // Conservative roundoff padding also keeps zero-thickness planes queryable.
    let pad = max(vec3f(1e-6), max(abs(low), abs(high)) * 2e-7);
    bounds[base + node] = Bounds(vec4f(low - pad, 0.0), vec4f(high + pad, 0.0));
}

// After triangle refit: one fixed leaf per capacity slot, with the active
// Morton order mapping leaves to current instances. Empty leaves are explicit;
// stale BLAS roots belonging to deleted instances are never included.
// offsets: instance nodes, instance order, instance bounds, depth node IDs.
@compute @workgroup_size(64)
fn refitInstances(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.level.x) { return; }
    let node = source[params.offsets.w + id.x];
    let start = params.offsets.x + node * 4u;
    let first = source[start];
    let second = source[start + 1u];
    var low = vec3f(3.402823466e+38);
    var high = -low;
    var valid = 0.0;
    if (source[start + 3u] == 1u) {
        let instance = source[params.offsets.y + first];
        if (instance < params.shape.z) {
            let root = bounds[instance * params.shape.x];
            low = root.low.xyz; high = root.high.xyz; valid = 1.0;
        }
    } else {
        let left = bounds[params.offsets.z + first];
        let right = bounds[params.offsets.z + second];
        if (left.low.w != 0.0) { low = left.low.xyz; high = left.high.xyz; valid = 1.0; }
        if (right.low.w != 0.0) { low = min(low, right.low.xyz); high = max(high, right.high.xyz); valid = 1.0; }
    }
    bounds[params.offsets.z + node] = Bounds(vec4f(low, valid), vec4f(high, 0.0));
}

// One current root per batch for the scene BVH. shape.w is the output record;
// offsets.xy are the optional instance-tree root offset and node count.
@compute @workgroup_size(1)
fn refitBatch() {
    var low = vec3f(3.402823466e+38);
    var high = -low;
    var valid = 0.0;
    if (params.shape.z != 0u) {
        if (params.offsets.y != 0u) {
            let root = bounds[params.offsets.x];
            low = root.low.xyz; high = root.high.xyz; valid = root.low.w;
        } else {
            // Linear-instance reference: bounded by the existing 4096 hard cap.
            for (var instance = 0u; instance < params.shape.z; instance++) {
                let root = bounds[instance * params.shape.x];
                low = min(low, root.low.xyz); high = max(high, root.high.xyz);
            }
            valid = 1.0;
        }
    }
    bounds[params.shape.w] = Bounds(vec4f(low, valid), vec4f(high, 0.0));
}
