// GI_COMMON
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
struct AreaParameters {
    shape: vec4u, // generation: vertices, triangles, active instances, source ID; integration: source count
    limits: vec4u, // used block ray count, captured probe slots, samples per probe/source, source block offset
}
struct AreaSample {
    expected: vec4u, // source, entity, generation, original triangle
    measure: vec4f, // endpoint distance, area-to-solid-angle Monte Carlo weight, endpoint tolerance, reserved
    direction: vec4f, // probe-to-emitter unit direction; w<0 records a numerical failure
}
@group(0) @binding(0) var<uniform> gi: GiParameters;
@group(0) @binding(1) var<uniform> area: AreaParameters;
@group(0) @binding(2) var<storage, read> vertices: array<Deformed>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> identities: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(6) var<storage, read_write> samples: array<AreaSample>;
@group(0) @binding(7) var<storage, read> hits: array<Hit>;
@group(0) @binding(8) var<storage, read_write> probes: array<GiProbe>;
// Probe captured by each slot of this GI update (identity for a full update).
@group(0) @binding(9) var<storage, read> schedule: array<u32>;

fn areaHash(value: u32) -> u32 {
    var v = value;
    v = (v ^ (v >> 16u)) * 0x7feb352du;
    v = (v ^ (v >> 15u)) * 0x846ca68bu;
    return v ^ (v >> 16u);
}
fn areaRandom(value: u32) -> f32 {
    // Twenty-four bits are exactly representable; this cannot round up to one.
    return f32(areaHash(value) >> 8u) * (1.0 / 16777216.0);
}
fn areaRadicalInverse(value: u32) -> f32 {
    return f32(reverseBits(value) >> 8u) * (1.0 / 16777216.0);
}
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= area.limits.x) { return; }
    rays[id.x] = Ray(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(0.0), vec4u(0u));
    var empty: AreaSample;
    samples[id.x] = empty;
}
@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= area.limits.y * area.limits.z || area.shape.z == 0u) { return; }
    // Seeds use the probe index, so a probe's samples do not depend on its slot.
    let probe = schedule[id.x / area.limits.z];
    let sample = id.x % area.limits.z;
    let output = area.limits.w + id.x;
    let seed = areaHash(probe ^ (area.shape.w * 0x9e3779b9u));
    let ordinal = areaHash(sample ^ seed);
    let triangle = ordinal % area.shape.y;
    let instance = areaHash(ordinal ^ 0x632be59bu) % area.shape.z;
    let offset = instance * area.shape.x;
    let a = vertices[offset + indices[triangle * 3u]].position.xyz;
    let b = vertices[offset + indices[triangle * 3u + 1u]].position.xyz;
    let c = vertices[offset + indices[triangle * 3u + 2u]].position.xyz;
    let crossEdges = cross(b - a, c - a);
    let twiceArea = length(crossEdges);
    // Collapsed triangles have zero measure and need no visibility work.
    if (twiceArea == 0.0) { return; }
    let u = fract((f32(sample) + 0.5) / f32(area.limits.z) + areaRandom(seed ^ 0xa511e9b3u));
    let v = fract(areaRadicalInverse(sample) + areaRandom(seed ^ 0x63d83595u));
    let root = sqrt(u);
    let point = a * (1.0 - root) + b * (root * (1.0 - v)) + c * (root * v);
    let coordinate = giCoordinate(probe, gi.counts.xyz);
    let origin = gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz;
    let delta = point - origin;
    let distance = length(delta);
    let tolerance = max(0.001, distance * 0.0001);
    if (distance < 0.0001 || distance > gi.spacing.w) { return; }
    let direction = delta / distance;
    // Query material compilation enforces sidedness. Taking the absolute
    // geometric cosine here supports both sides without a second material IR.
    let cosine = abs(dot(crossEdges / twiceArea, -direction));
    let weight = cosine * (0.5 * twiceArea) * f32(area.shape.y) * f32(area.shape.z)
        / (f32(area.limits.z) * distance * distance);
    if (!(twiceArea > 0.0 && twiceArea <= 1e30 && distance <= 1e30 && weight >= 0.0 && weight <= 1e30)) {
        samples[output].direction.w = -1.0;
        return;
    }
    if (weight == 0.0) { return; }
    let identity = identities[instance];
    samples[output] = AreaSample(vec4u(area.shape.w, identity.x, identity.y, triangle),
        vec4f(distance, weight, tolerance, 0.0), vec4f(direction, 1.0));
    // Trace through the target, then require the target triangle and distance.
    // Another triangle of this very entity remains a valid occluder.
    rays[output] = Ray(vec4f(origin, 0.0001), vec4f(direction, distance + tolerance), vec4u(0u));
}
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= area.limits.y) { return; }
    let probe = schedule[id.x];
    var coefficients: array<vec3f, 9>;
    let retainedMaximum = vec3f(probes[probe].coefficients[0].w, probes[probe].coefficients[1].w,
        probes[probe].coefficients[2].w);
    var emittedMaximum = vec3f(0.0);
    var failures = 0u;
    for (var source = 0u; source < area.shape.x; source++) {
        let block = source * area.limits.y * area.limits.z + id.x * area.limits.z;
        for (var sample = 0u; sample < area.limits.z; sample++) {
            let index = block + sample;
            let measure = samples[index];
            if (measure.direction.w < 0.0) { failures++; continue; }
            if (measure.expected.x == 0u) { continue; }
            let hit = hits[index];
            if (hit.work.z != 0u) { failures++; continue; }
            if (!all(hit.identity == measure.expected)
                || abs(hit.metric.x - measure.measure.x) > measure.measure.z) { continue; }
            // Unlit base is emitted radiance; PBR base is reflected energy and
            // was already integrated by the sphere capture, so add only Le.
            let emission = clamp(hit.emissive.xyz + select(vec3f(0.0), hit.baseColor.xyz, hit.emissive.w != 0.0),
                vec3f(0.0), vec3f(65504.0));
            emittedMaximum = max(emittedMaximum, emission);
            let radiance = emission * measure.measure.y;
            if (!all(radiance >= vec3f(0.0)) || !all(radiance <= vec3f(1e35))) { failures++; continue; }
            let basis = giBasis(measure.direction.xyz);
            for (var coefficient = 0u; coefficient < 9u; coefficient++) {
                coefficients[coefficient] += radiance * basis[coefficient];
            }
        }
    }
    // At most 256 samples per probe and coefficient magnitudes below 1e35,
    // so the sum stays finite. Keep visibility moments from the sphere rays.
    for (var coefficient = 0u; coefficient < 9u; coefficient++) {
        probes[probe].coefficients[coefficient] += vec4f(coefficients[coefficient], 0.0);
    }
    // Both retained reflection/sky and explicitly sampled emission can arrive
    // from the same direction. Their sum needs an additive upper bound.
    // Each component is bounded to 65504; their combined bound is at most 131008.
    let maximum = retainedMaximum + emittedMaximum;
    probes[probe].coefficients[0].w = maximum.x;
    probes[probe].coefficients[1].w = maximum.y;
    probes[probe].coefficients[2].w = maximum.z;
    if (failures != 0u) {
        probes[probe].state.x = 0u;
        probes[probe].moments[0].w = f32(failures);
    }
}
