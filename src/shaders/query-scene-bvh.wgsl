struct Bounds { low: vec4f, high: vec4f }
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}

// Refit: [depthNodeOffset, nodeCount, 0, 0].
// Candidate selection: [rayCount, perRayWorkLimit, 0, 0].
// Layout: [batch capacity, tree nodes, order word offset, mask words per ray].
struct Params {
    dispatch: vec4u,
    scene: vec4u,
    budget: vec4u, // repeating block size (0 = uniform), primary prefix, supplemental quota, reserved
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> bounds: array<Bounds>;
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> hits: array<Hit>;
@group(0) @binding(5) var<storage, read_write> masks: array<u32>;

@compute @workgroup_size(64)
fn refitScene(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.dispatch.y) { return; }
    let node = data[params.dispatch.x + id.x];
    let start = node * 4u;
    let first = data[start];
    let second = data[start + 1u];
    var low = vec3f(3.402823466e+38);
    var high = -low;
    var valid = 0.0;
    if (data[start + 3u] == 1u) {
        let batch = data[params.scene.z + first];
        if (batch < params.scene.x) {
            let root = bounds[batch];
            if (root.low.w != 0.0) {
                low = root.low.xyz; high = root.high.xyz; valid = 1.0;
            }
        }
    } else {
        let left = bounds[params.scene.x + first];
        let right = bounds[params.scene.x + second];
        if (left.low.w != 0.0) { low = left.low.xyz; high = left.high.xyz; valid = 1.0; }
        if (right.low.w != 0.0) { low = min(low, right.low.xyz); high = max(high, right.high.xyz); valid = 1.0; }
    }
    bounds[params.scene.x + node] = Bounds(vec4f(low, valid), vec4f(high, 0.0));
}

fn intersectsBox(ray: Ray, box: Bounds) -> bool {
    var near = ray.origin.w;
    var far = ray.direction.w;
    for (var axis = 0u; axis < 3u; axis++) {
        let direction = ray.direction[axis];
        if (direction == 0.0) {
            if (ray.origin[axis] < box.low[axis] || ray.origin[axis] > box.high[axis]) { return false; }
        } else {
            let a = (box.low[axis] - ray.origin[axis]) / direction;
            let b = (box.high[axis] - ray.origin[axis]) / direction;
            near = max(near, min(a, b)); far = min(far, max(a, b));
            if (far < near) { return false; }
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn selectBatches(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.dispatch.x) { return; }
    let maskBase = id.x * params.scene.w;
    for (var word = 0u; word < params.scene.w; word++) { masks[maskBase + word] = 0u; }
    let ray = rays[id.x];
    if (ray.direction.w < ray.origin.w) { return; }
    var workLimit = params.dispatch.y;
    if (params.budget.x != 0u) {
        var localRay = id.x;
        if (localRay >= params.budget.x) { localRay %= params.budget.x; }
        if (localRay >= params.budget.y) { workLimit = params.budget.z; }
    }
    var hit = hits[id.x];
    var node = 0u;
    loop {
        if (node >= params.scene.y) { break; }
        if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
        hit.work.x++;
        let start = node * 4u;
        let box = bounds[params.scene.x + node];
        if (box.low.w == 0.0 || !intersectsBox(ray, box)) {
            node = data[start + 2u]; continue;
        }
        node++;
        if (data[start + 3u] == 0u) { continue; }
        let batch = data[params.scene.z + data[start]];
        if (batch < params.scene.x) {
            let word = batch >> 5u;
            masks[maskBase + word] = masks[maskBase + word] | (1u << (batch & 31u));
        }
    }
    hits[id.x] = hit;
}
