struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
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
    offsets: vec4u, // triangle IDs, indices, UVs, colors (0 = white)
    limits: vec4u, // ray count, work limit, frame, topology version
    material: vec4u, // material revision, GI area-sampling source flag, candidates enabled, words per ray
    textureLod: f32,
    budgetBlock: u32, // zero = uniform quota; otherwise repeating source-major block size
    budgetPrimary: u32, // primary prefix length within each block
    budgetSupplemental: u32, // work quota for each suffix ray
    instanceTree: vec4u, // node offset, order offset, bounds offset, node count (0 = linear)
    queryPolicy: vec4u, // x: 0 = authored material faces, 1 = reversed material support
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<Deformed>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read> bounds: array<Bounds>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> rays: array<Ray>;
@group(0) @binding(6) var<storage, read_write> hits: array<Hit>;
@group(0) @binding(7) var<storage, read> identities: array<vec4u>;
@group(0) @binding(8) var<storage, read> batchCandidates: array<u32>;
struct MaterialParameters { values: array<vec4f, 16>, textureInfo: array<vec4f, 4> }
@group(1) @binding(0) var<uniform> materialParameters: MaterialParameters;
@group(1) @binding(1) var materialImage0: texture_2d<f32>;
@group(1) @binding(2) var materialImage1: texture_2d<f32>;
@group(1) @binding(3) var materialImage2: texture_2d<f32>;
@group(1) @binding(4) var materialImage3: texture_2d<f32>;
@group(1) @binding(5) var materialSampler0: sampler;
@group(1) @binding(6) var materialSampler1: sampler;
@group(1) @binding(7) var materialSampler2: sampler;
@group(1) @binding(8) var materialSampler3: sampler;
fn materialTexture0(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage0, materialSampler0, uv, max(0.0, lod - materialParameters.textureInfo[0].x)); }
fn materialTexture1(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage1, materialSampler1, uv, max(0.0, lod - materialParameters.textureInfo[1].x)); }
fn materialTexture2(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage2, materialSampler2, uv, max(0.0, lod - materialParameters.textureInfo[2].x)); }
fn materialTexture3(uv: vec2f, lod: f32) -> vec4f { return textureSampleLevel(materialImage3, materialSampler3, uv, max(0.0, lod - materialParameters.textureInfo[3].x)); }
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
fn colorAt(vertex: u32) -> vec4f {
    if (params.offsets.w == 0u) { return vec4f(1.0); }
    let start = params.offsets.w + vertex * 4u;
    return vec4f(bitcast<f32>(source[start]), bitcast<f32>(source[start + 1u]),
        bitcast<f32>(source[start + 2u]), bitcast<f32>(source[start + 3u]));
}
// Barycentric interpolation with explicit fused operations. Implicit contraction
// or reassociation may differ between entry points (measured on Apple Metal),
// so both the per-batch traversal and the pooled resolve use this fixed form.
fn barycentric2(a: vec2f, b: vec2f, c: vec2f, w: vec3f) -> vec2f { return fma(c, vec2f(w.z), fma(b, vec2f(w.y), a * w.x)); }
fn barycentric3(a: vec3f, b: vec3f, c: vec3f, w: vec3f) -> vec3f { return fma(c, vec3f(w.z), fma(b, vec3f(w.y), a * w.x)); }
fn barycentric4(a: vec4f, b: vec4f, c: vec4f, w: vec3f) -> vec4f { return fma(c, vec4f(w.z), fma(b, vec4f(w.y), a * w.x)); }
// Shared by the per-batch traversal and the pooled-traversal resolve, so a
// winning triangle yields identical attributes on both paths. Returns false
// when Masked coverage rejects the point (the hit is left unchanged).
fn writeTriangleHit(hit: ptr<function, Hit>, ray: Ray, instance: u32, identity: vec4u, triangle: u32,
    v0: u32, v1: u32, v2: u32, a: Deformed, b: Deformed, c: Deformed,
    edge1: vec3f, edge2: vec3f, determinant: f32, distance: f32, u: f32, v: f32) -> bool {
    let weights = vec3f(1.0 - u - v, u, v);
    let uv = barycentric2(uvAt(v0), uvAt(v1), uvAt(v2), weights);
    let color = barycentric4(colorAt(v0), colorAt(v1), colorAt(v2), weights);
    let inst = instances[instance];
    if (!evaluateCoverage(uv, params.textureLod, inst.color, inst.material.xy, color)) { return false; }
    let surface = evaluateMaterial(uv, params.textureLod, inst.color, inst.material.xy, color);
    let geometricNormal = surfaceUnit(cross(edge1, edge2), vec3f(0.0, 1.0, 0.0)) * select(-1.0, 1.0, determinant > 0.0);
    let vertexNormal = barycentric3(a.normal.xyz, b.normal.xyz, c.normal.xyz, weights);
    var normal = surfaceUnit(vertexNormal, geometricNormal);
    normal *= select(-1.0, 1.0, dot(normal, geometricNormal) >= 0.0);
    if (MATERIAL_NORMAL_MAPPED) {
        let tangent = barycentric4(a.tangent, b.tangent, c.tangent, weights);
        normal = surfaceMappedNormal(vertexNormal, geometricNormal, tangent, edge1, edge2,
            uvAt(v1) - uvAt(v0), uvAt(v2) - uvAt(v0), surface.tangentNormal);
    }
    let encodedGeometry = surfaceOctEncode(geometricNormal);
    (*hit).identity = vec4u(params.shape.w, identity.x, identity.y, triangle);
    (*hit).metric = vec4f(distance, u, v, select(0.0, 1.0, determinant > 0.0));
    (*hit).position = vec4f(ray.origin.xyz + distance * ray.direction.xyz, encodedGeometry.x);
    (*hit).normal = vec4f(normal, surface.roughness);
    (*hit).previous = vec4f(barycentric3(a.previous.xyz, b.previous.xyz, c.previous.xyz, weights), surface.metallic);
    (*hit).baseColor = vec4f(surface.baseColor, surface.opacity);
    (*hit).emissive = vec4f(surface.emissive, select(0.0, 1.0, MATERIAL_UNLIT));
    (*hit).versions = vec4u(params.limits.z, identity.z, params.limits.w, params.material.x);
    (*hit).uvInfo = vec4f(uv, params.textureLod, encodedGeometry.y);
    (*hit).work.w = params.material.y;
    return true;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.limits.x) { return; }
    if (params.material.z != 0u) {
        let batch = params.shape.w - 1u;
        if ((batchCandidates[id.x * params.material.w + batch / 32u] & (1u << (batch % 32u))) == 0u) { return; }
    }
    let ray = rays[id.x];
    var hit = hits[id.x];
    if (hit.work.z != 0u) { return; }
    // GPU consumers use an empty interval for inactive work (e.g. sky rays
    // needing no sun-visibility query). Preserve the reset miss at zero cost.
    if (ray.direction.w < ray.origin.w) { return; }
    var workLimit = params.limits.y;
    if (params.budgetBlock != 0u) {
        var localRay = id.x;
        if (localRay >= params.budgetBlock) { localRay %= params.budgetBlock; }
        if (localRay >= params.budgetPrimary) { workLimit = params.budgetSupplemental; }
    }
    var instanceCursor = 0u;
    var winningInstance = 0xffffffffu;
    loop {
        let accelerated = params.instanceTree.w != 0u;
        if (instanceCursor >= select(params.shape.z, params.instanceTree.w, accelerated)) { break; }
        if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
        hit.work.x++;
        var instance = instanceCursor;
        if (accelerated) {
            let start = params.instanceTree.x + instanceCursor * 4u;
            let box = bounds[params.instanceTree.z + instanceCursor];
            if (box.low.w == 0.0 || !intersectsBox(ray, box, hit.metric.x)) {
                instanceCursor = source[start + 2u]; continue;
            }
            instanceCursor++;
            if (source[start + 3u] == 0u) { continue; }
            instance = source[params.instanceTree.y + source[start]];
            if (instance >= params.shape.z) { continue; }
        } else { instanceCursor++; }
        let identity = identities[instance];
        if (ray.ignore.z != 0u && all(ray.ignore.xy == identity.xy)) { continue; }
        var node = 0u;
        loop {
            if (node >= params.shape.x) { break; }
            if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
            hit.work.x++;
            if (!intersectsBox(ray, bounds[instance * params.shape.x + node], hit.metric.x)) {
                node = source[node * 4u + 2u]; continue;
            }
            if (source[node * 4u + 3u] == 0u) { node++; continue; }
            let first = source[node * 4u];
            let count = source[node * 4u + 1u];
            for (var i = 0u; i < count; i++) {
                if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
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
                // Reversing a finite segment negates its determinant. Support
                // captures use this opposite single-sided eligibility without
                // changing double-sided coverage or ordinary transport queries.
                let rejectedFace = select((determinant < 0.0), (determinant > 0.0), params.queryPolicy.x == 1u);
                if (abs(determinant) <= epsilon || (!MATERIAL_DOUBLE_SIDED && rejectedFace)) { continue; }
                let delta = ray.origin.xyz - a.position.xyz;
                let u = dot(delta, crossDirection) / determinant;
                if (u < 0.0 || u > 1.0) { continue; }
                let crossDelta = cross(delta, edge1);
                let v = dot(ray.direction.xyz, crossDelta) / determinant;
                if (v < 0.0 || u + v > 1.0) { continue; }
                let distance = dot(edge2, crossDelta) / determinant;
                if (distance < ray.origin.w || distance > ray.direction.w || distance > hit.metric.x
                    // Traversal order must not change equal-distance identity:
                    // earlier batch, then lower instance slot, then BLAS order.
                    // A pooled static hit may come from a later batch; ID order decides.
                    || (distance == hit.metric.x && hit.identity.x != 0u
                        && (hit.identity.x < params.shape.w
                            || (hit.identity.x == params.shape.w && instance >= winningInstance)))) { continue; }
                if (!writeTriangleHit(&hit, ray, instance, identity, triangle, v0, v1, v2, a, b, c,
                    edge1, edge2, determinant, distance, u, v)) { continue; }
                winningInstance = instance;
            }
            if (hit.work.z != 0u) { break; }
            node++;
        }
        if (hit.work.z != 0u) { break; }
    }
    hits[id.x] = hit;
}

// Material resolve for a pooled static batch: only rays whose shared traversal
// winner is this batch (binding 8 holds winner IDs here, not candidate masks).
// The temporary identity carries the instance slot and leaf-order triangle;
// distance and barycentrics come from the shared traversal.
@compute @workgroup_size(64)
fn resolveShared(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.limits.x) { return; }
    if (batchCandidates[id.x] != params.shape.w) { return; }
    var hit = hits[id.x];
    if (hit.identity.x != params.shape.w) { return; }
    let ray = rays[id.x];
    let instance = hit.identity.y;
    let triangle = source[params.offsets.x + hit.identity.w];
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
    // Pooled batches are Opaque, whose coverage is unconditional.
    _ = writeTriangleHit(&hit, ray, instance, identities[instance], triangle, v0, v1, v2, a, b, c,
        edge1, edge2, determinant, hit.metric.x, hit.metric.y, hit.metric.z);
    hits[id.x] = hit;
}
