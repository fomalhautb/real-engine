struct Deformed { position: vec4f, normal: vec4f, previous: vec4f }
struct Bounds { low: vec4f, high: vec4f }
struct Instance {
    model: mat4x4f, previousModel: mat4x4f, normalMatrix: mat4x4f,
    color: vec4f, material: vec4f,
}
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
struct Params {
    shape: vec4u, // node count, vertex count, instance count, batch ID
    offsets: vec4u, // triangle IDs, indices, UVs, unused
    limits: vec4u, // ray count, work limit, frame, topology version
    material: vec4u, // material revision
    textureLod: vec4f,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<Deformed>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read> bounds: array<Bounds>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> rays: array<Ray>;
@group(0) @binding(6) var<storage, read_write> hits: array<Hit>;
@group(0) @binding(7) var<storage, read> identities: array<vec4u>;
struct MaterialParameters { values: array<vec4f, 16> }
@group(1) @binding(0) var<uniform> materialParameters: MaterialParameters;
@group(1) @binding(1) var materialImage0: texture_2d<f32>;
@group(1) @binding(2) var materialImage1: texture_2d<f32>;
@group(1) @binding(3) var materialImage2: texture_2d<f32>;
@group(1) @binding(4) var materialImage3: texture_2d<f32>;
@group(1) @binding(5) var materialSampler0: sampler;
@group(1) @binding(6) var materialSampler1: sampler;
@group(1) @binding(7) var materialSampler2: sampler;
@group(1) @binding(8) var materialSampler3: sampler;
fn materialTexture0(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage0, materialSampler0, uv, lod); }
fn materialTexture1(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage1, materialSampler1, uv, lod); }
fn materialTexture2(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage2, materialSampler2, uv, lod); }
fn materialTexture3(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage3, materialSampler3, uv, lod); }
// MATERIAL_PROGRAM

fn intersectsBox(ray: Ray, box: Bounds, closest: f32) -> bool {
    var near = ray.origin.w;
    var far = closest;
    for (var axis = 0u; axis < 3u; axis++) {
        let direction = ray.direction[axis];
        // Parallel rays on a slab boundary must not evaluate 0 * infinity.
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
fn uvAt(vertex: u32) -> vec2f {
    let start = params.offsets.z + vertex * 2u;
    return vec2f(bitcast<f32>(source[start]), bitcast<f32>(source[start + 1u]));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.limits.x) { return; }
    let ray = rays[id.x];
    var hit = hits[id.x];
    if (hit.work.z != 0u) { return; }
    // Instance roots are currently scanned linearly; per-instance triangle
    // traversal is accelerated by a refitted, stackless preorder BVH.
    for (var instance = 0u; instance < params.shape.z; instance++) {
        if (hit.work.x >= params.limits.y) { hit.work.z = 1u; break; }
        hit.work.x++;
        let identity = identities[instance];
        if (ray.ignore.z != 0u && all(ray.ignore.xy == identity.xy)) { continue; }
        var node = 0u;
        loop {
            if (node >= params.shape.x) { break; }
            if (hit.work.x >= params.limits.y) { hit.work.z = 1u; break; }
            hit.work.x++;
            if (!intersectsBox(ray, bounds[instance * params.shape.x + node], hit.metric.x)) {
                node = source[node * 4u + 2u]; continue;
            }
            if (source[node * 4u + 3u] == 0u) { node++; continue; }
            let first = source[node * 4u];
            let count = source[node * 4u + 1u];
            for (var i = 0u; i < count; i++) {
                if (hit.work.x >= params.limits.y) { hit.work.z = 1u; break; }
                hit.work.x++; hit.work.y++;
                let triangle = source[params.offsets.x + first + i];
                let v0 = source[params.offsets.y + triangle * 3u];
                let v1 = source[params.offsets.y + triangle * 3u + 1u];
                let v2 = source[params.offsets.y + triangle * 3u + 2u];
                let a = vertices[instance * params.shape.y + v0];
                let b = vertices[instance * params.shape.y + v1];
                let c = vertices[instance * params.shape.y + v2];
                let edge1 = b.position.xyz - a.position.xyz;
                let edge2 = c.position.xyz - a.position.xyz;
                let crossDirection = cross(ray.direction.xyz, edge2);
                let determinant = dot(edge1, crossDirection);
                let epsilon = max(1e-30, length(edge1) * length(edge2) * 1e-7);
                if (abs(determinant) <= epsilon || (!MATERIAL_DOUBLE_SIDED && determinant < 0.0)) { continue; }
                let delta = ray.origin.xyz - a.position.xyz;
                let u = dot(delta, crossDirection) / determinant;
                if (u < 0.0 || u > 1.0) { continue; }
                let crossDelta = cross(delta, edge1);
                let v = dot(ray.direction.xyz, crossDelta) / determinant;
                if (v < 0.0 || u + v > 1.0) { continue; }
                let distance = dot(edge2, crossDelta) / determinant;
                if (distance < ray.origin.w || distance > ray.direction.w || distance > hit.metric.x
                    || (distance == hit.metric.x && hit.identity.x != 0u)) { continue; }
                let weights = vec3f(1.0 - u - v, u, v);
                let uv = uvAt(v0) * weights.x + uvAt(v1) * weights.y + uvAt(v2) * weights.z;
                let inst = instances[instance];
                if (!evaluateCoverage(uv, params.textureLod.x, inst.color, inst.material.xy)) { continue; }
                let surface = evaluateMaterial(uv, params.textureLod.x, inst.color, inst.material.xy);
                var normal = a.normal.xyz * weights.x + b.normal.xyz * weights.y + c.normal.xyz * weights.z;
                if (dot(normal, normal) < 1e-20) { normal = cross(edge1, edge2); }
                normal = normalize(normal) * select(-1.0, 1.0, determinant > 0.0);
                hit.identity = vec4u(params.shape.w, identity.x, identity.y, triangle);
                hit.metric = vec4f(distance, u, v, select(0.0, 1.0, determinant > 0.0));
                hit.position = vec4f(ray.origin.xyz + distance * ray.direction.xyz, 0.0);
                hit.normal = vec4f(normal, surface.roughness);
                hit.previous = vec4f(a.previous.xyz * weights.x + b.previous.xyz * weights.y + c.previous.xyz * weights.z, surface.metallic);
                hit.baseColor = vec4f(surface.baseColor, surface.opacity);
                hit.emissive = vec4f(surface.emissive, select(0.0, 1.0, MATERIAL_UNLIT));
                hit.versions = vec4u(params.limits.z, identity.z, params.limits.w, params.material.x);
                hit.uvInfo = vec4f(uv, params.textureLod.x, 0.0);
            }
            if (hit.work.z != 0u) { break; }
            node++;
        }
        if (hit.work.z != 0u) { break; }
    }
    hits[id.x] = hit;
}
