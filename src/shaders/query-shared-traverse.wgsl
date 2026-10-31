// One dispatch traverses every pooled static batch: TLAS over (batch, instance)
// leaves, then that instance's BLAS records, with the same box/triangle tests,
// face policy and work units as query-trace.wgsl. Equal distances keep the
// lowest (batch, instance slot) and then the first triangle in BLAS order.
// Material, normals and identity are resolved later by the winning batch.
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
struct Params {
    dispatch: vec4u, // ray count, primary work quota, TLAS node count, face policy (1 = reversed support)
    budget: vec4u, // repeating block size (0 = uniform), primary prefix, supplemental quota, leaf count
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rays: array<Ray>;
@group(0) @binding(2) var<storage, read_write> hits: array<Hit>;
// Two vec4u per record: [low.xyz bits, escape] [high.xyz bits, info]. TLAS first.
@group(0) @binding(3) var<storage, read> nodes: array<vec4u>;
// Three local vertex indices per pooled triangle, in BLAS leaf order.
@group(0) @binding(4) var<storage, read> triangles: array<u32>;
// Packed xyz per pooled vertex; copied bit-exactly from the shared deformation output.
@group(0) @binding(5) var<storage, read> positions: array<f32>;
// Eight words per leaf, then one flag word per registered batch.
@group(0) @binding(6) var<storage, read> leaves: array<u32>;
// Winning batch ID (1-based) or zero for the per-batch material resolve.
@group(0) @binding(7) var<storage, read_write> winners: array<u32>;

// Watertight ray/triangle crossing (Woop, Benthin and Wald 2013). The ray is
// sheared onto +z once per ray. Each edge function is evaluated from the endpoint
// that comes first in sheared (x, y, z) order, with explicit fma, so two triangles
// sharing an edge (even across split seam vertices, which share positions but not
// IDs) compute exactly opposite values whatever contraction a backend applies; zero
// (on an edge or vertex) counts as inside: no ray passes between triangles that
// share vertices. Strict Moller-Trumbore edge tests let 213 of 405 quad-diagonal
// rays through on apple/metal-3. Same function as query-trace.wgsl, so both paths agree bit for bit.
struct RayShear { axes: vec3u, factors: vec3f }
fn rayShear(direction: vec3f) -> RayShear {
    let magnitude = abs(direction);
    var kz = 2u;
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) { kz = 0u; } else if (magnitude.y >= magnitude.z) { kz = 1u; }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (direction[kz] < 0.0) { let swap = kx; kx = ky; ky = swap; }
    return RayShear(vec3u(kx, ky, kz), vec3f(direction[kx] / direction[kz], direction[ky] / direction[kz], 1.0 / direction[kz]));
}
fn shearPoint(shear: RayShear, p: vec3f) -> vec3f {
    let z = p[shear.axes.z];
    return vec3f(fma(-shear.factors.x, z, p[shear.axes.x]), fma(-shear.factors.y, z, p[shear.axes.y]), shear.factors.z * z);
}
fn precedes(p: vec3f, q: vec3f) -> bool {
    return p.x < q.x || (p.x == q.x && (p.y < q.y || (p.y == q.y && p.z < q.z)));
}
fn edgeFunction(p: vec3f, q: vec3f) -> f32 {
    if (precedes(q, p)) { return -fma(p.x, q.y, -(p.y * q.x)); }
    return fma(q.x, p.y, -(q.y * p.x));
}
fn watertightTriangle(shear: RayShear, a: vec3f, b: vec3f, c: vec3f) -> vec4f {
    let sa = shearPoint(shear, a);
    let sb = shearPoint(shear, b);
    let sc = shearPoint(shear, c);
    let weightA = edgeFunction(sb, sc);
    let weightB = edgeFunction(sc, sa);
    let weightC = edgeFunction(sa, sb);
    if ((weightA < 0.0 || weightB < 0.0 || weightC < 0.0) && (weightA > 0.0 || weightB > 0.0 || weightC > 0.0)) { return vec4f(0.0); }
    let det = weightA + weightB + weightC;
    if (det == 0.0) { return vec4f(0.0); }
    let inverse = 1.0 / det;
    return vec4f(fma(weightA, sa.z, fma(weightB, sb.z, weightC * sc.z)) * inverse, weightB * inverse, weightC * inverse, det);
}
fn intersectsBox(ray: Ray, low: vec3f, high: vec3f, closest: f32) -> bool {
    var near = ray.origin.w;
    var far = closest;
    for (var axis = 0u; axis < 3u; axis++) {
        let direction = ray.direction[axis];
        if (direction == 0.0) {
            if (ray.origin[axis] < low[axis] || ray.origin[axis] > high[axis]) { return false; }
        } else {
            let a = (low[axis] - ray.origin[axis]) / direction;
            let b = (high[axis] - ray.origin[axis]) / direction;
            near = max(near, min(a, b)); far = min(far, max(a, b));
            if (far < near) { return false; }
        }
    }
    return true;
}
fn pooledPosition(vertex: u32) -> vec3f {
    let at = vertex * 3u;
    return vec3f(positions[at], positions[at + 1u], positions[at + 2u]);
}

@compute @workgroup_size(64)
fn traverse(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.dispatch.x) { return; }
    winners[id.x] = 0u;
    let ray = rays[id.x];
    var hit = hits[id.x];
    if (hit.work.z != 0u) { return; }
    if (ray.direction.w < ray.origin.w) { return; }
    let shear = rayShear(ray.direction.xyz);
    var workLimit = params.dispatch.y;
    if (params.budget.x != 0u) {
        var localRay = id.x;
        if (localRay >= params.budget.x) { localRay %= params.budget.x; }
        if (localRay >= params.budget.y) { workLimit = params.budget.z; }
    }
    let flagBase = params.budget.w * 8u;
    let reversed = params.dispatch.w == 1u;
    var closest = hit.metric.x;
    var found = false;
    var bestBatch = 0xffffffffu;
    var bestInstance = 0xffffffffu;
    var bestTriangle = 0u;
    var bestU = 0.0;
    var bestV = 0.0;
    var bestFront = 0.0;
    var tlas = 0u;
    loop {
        if (tlas >= params.dispatch.z) { break; }
        if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
        hit.work.x++;
        let low = nodes[tlas * 2u];
        let high = nodes[tlas * 2u + 1u];
        if ((high.w & 0x80000000u) == 0u
            || !intersectsBox(ray, bitcast<vec3f>(low.xyz), bitcast<vec3f>(high.xyz), closest)) {
            tlas = low.w; continue;
        }
        tlas++;
        let leafRef = high.w & 0x7fffffffu;
        if (leafRef == 0u) { continue; }
        let leaf = (leafRef - 1u) * 8u;
        let batch = leaves[leaf];
        let flags = leaves[flagBase + batch];
        if ((flags & 1u) == 0u) { continue; }
        if (ray.ignore.z != 0u && ray.ignore.x == leaves[leaf + 2u] && ray.ignore.y == leaves[leaf + 3u]) { continue; }
        let instance = leaves[leaf + 1u];
        let nodeBase = leaves[leaf + 4u];
        let triangleBase = leaves[leaf + 5u];
        let positionBase = leaves[leaf + 6u];
        let nodeCount = leaves[leaf + 7u];
        let doubleSided = (flags & 2u) != 0u;
        var node = 0u;
        loop {
            if (node >= nodeCount) { break; }
            if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
            hit.work.x++;
            let record = (nodeBase + node) * 2u;
            let lo = nodes[record];
            let hi = nodes[record + 1u];
            if (!intersectsBox(ray, bitcast<vec3f>(lo.xyz), bitcast<vec3f>(hi.xyz), closest)) {
                node = lo.w; continue;
            }
            node++;
            if (hi.w == 0u) { continue; }
            let first = hi.w >> 4u;
            let count = hi.w & 15u;
            for (var i = 0u; i < count; i++) {
                if (hit.work.x >= workLimit) { hit.work.z = 1u; break; }
                hit.work.x++; hit.work.y++;
                let t = (triangleBase + first + i) * 3u;
                let a = pooledPosition(positionBase + triangles[t]);
                let b = pooledPosition(positionBase + triangles[t + 1u]);
                let c = pooledPosition(positionBase + triangles[t + 2u]);
                let edge1 = b - a;
                let edge2 = c - a;
                let crossDirection = cross(ray.direction.xyz, edge2);
                let determinant = dot(edge1, crossDirection);
                let epsilon = max(1e-30, length(edge1) * length(edge2) * 1e-7);
                let rejectedFace = select((determinant < 0.0), (determinant > 0.0), reversed);
                if (abs(determinant) <= epsilon || (!doubleSided && rejectedFace)) { continue; }
                let crossing = watertightTriangle(shear, a - ray.origin.xyz, b - ray.origin.xyz, c - ray.origin.xyz);
                if (crossing.w == 0.0) { continue; }
                let distance = crossing.x;
                let u = crossing.y;
                let v = crossing.z;
                // A prior pass never writes pooled batches; only batch order decides its ties.
                let priorWins = !found && hit.identity.x != 0u && hit.identity.x - 1u < batch;
                let tieRejected = distance == closest && (priorWins || (found
                    && (batch > bestBatch || (batch == bestBatch && instance >= bestInstance))));
                if (distance < ray.origin.w || distance > ray.direction.w || distance > closest || tieRejected) { continue; }
                closest = distance;
                found = true;
                bestBatch = batch; bestInstance = instance; bestTriangle = first + i;
                bestU = u; bestV = v; bestFront = select(0.0, 1.0, determinant > 0.0);
            }
            if (hit.work.z != 0u) { break; }
        }
        if (hit.work.z != 0u) { break; }
    }
    if (found) {
        // Temporary identity until resolve: instance slot and leaf-order triangle.
        hit.identity = vec4u(bestBatch + 1u, bestInstance, 0u, bestTriangle);
        hit.metric = vec4f(closest, bestU, bestV, bestFront);
        winners[id.x] = bestBatch + 1u;
    }
    hits[id.x] = hit;
}
