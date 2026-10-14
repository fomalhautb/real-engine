// GI_COMMON
struct FinalGatherParameters {
    inverseViewProjection: mat4x4f,
    eye: vec4f,
    viewport: vec4u, // width, height, columns, rows
    settings: vec4u, // actual spacing, anchors, rays per anchor, maximum history
    state: vec4u, // frame, enabled, reset history, multiple diffuse bounces
    sampling: vec4f, // independent normal bias, reserved, plane tolerance, maximum filter distance
    emissive: vec4u, // active source prefix, area query count, reserved
}
struct FinalGatherAnchor {
    position: vec4f,
    normal: vec4f,
    identity: vec4u, // source, entity, generation, triangle
    versions: vec4u, // rendered frame, history, topology, material
    irradiance: vec4f, // RGB irradiance, accumulated sample updates
    state: vec4u, // valid, rendered frame, primary/area failures, shadow failures
}
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f }
struct FinalGatherAreaParameters {
    shape: vec4u, // vertices, triangles, instances, source
    limits: vec4u, // area prefix query count, anchors, reserved, source block offset
}
struct FinalGatherAreaSample {
    expected: vec4u,
    measure: vec4f, // distance, area-to-irradiance weight, endpoint tolerance, reserved
    direction: vec4f, // w < 0 denotes numerical failure
}
@group(0) @binding(0) var<uniform> gather: FinalGatherParameters;
@group(0) @binding(1) var<uniform> gi: GiParameters;
@group(0) @binding(2) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(3) var<storage, read> hits: array<Hit>;
@group(0) @binding(4) var<storage, read> cameraHits: array<Hit>;
@group(0) @binding(5) var<storage, read> primaryHits: array<Hit>;
@group(0) @binding(6) var<storage, read_write> anchors: array<FinalGatherAnchor>;
@group(0) @binding(7) var<storage, read> oldAnchors: array<FinalGatherAnchor>;
@group(0) @binding(8) var<storage, read> giProbes: array<GiProbe>;
@group(0) @binding(9) var<uniform> area: FinalGatherAreaParameters;
@group(0) @binding(10) var<storage, read> vertices: array<Deformed>;
@group(0) @binding(11) var<storage, read> indices: array<u32>;
@group(0) @binding(12) var<storage, read> identities: array<vec4u>;
@group(0) @binding(13) var<storage, read_write> areaSamples: array<FinalGatherAreaSample>;

// GI_INTERPOLATE

fn gatherHash(value: u32) -> u32 {
    var v = value;
    v = (v ^ (v >> 16u)) * 0x7feb352du;
    v = (v ^ (v >> 15u)) * 0x846ca68bu;
    return v ^ (v >> 16u);
}
fn gatherRandom(value: u32) -> f32 {
    return f32(gatherHash(value) >> 8u) * (1.0 / 16777216.0);
}
fn gatherDirection(anchor: u32, sample: u32, normal: vec3f) -> vec3f {
    let seed = gatherHash(anchor ^ (gather.state.x * 0x9e3779b9u));
    let u = (f32(sample) + gatherRandom(seed)) / f32(gather.settings.z);
    let v = gatherRandom(seed ^ (sample * 0x632be59bu) ^ 0xa511e9b3u);
    let radius = sqrt(u);
    let angle = 6.28318530718 * v;
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle))
        + normal * sqrt(max(0.0, 1.0 - u)));
}
fn gatherEmptyRay() -> Ray {
    return Ray(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(0.0), vec4u(0u));
}

@compute @workgroup_size(64)
fn generateCamera(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y) { return; }
    let coordinate = vec2u(id.x % gather.viewport.z, id.x / gather.viewport.z);
    let pixelOffset = f32(gather.settings.x / 2u) + 0.5;
    let pixel = min(vec2f(coordinate) * f32(gather.settings.x) + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let clip = vec2f(pixel.x / f32(gather.viewport.x) * 2.0 - 1.0,
        1.0 - pixel.y / f32(gather.viewport.y) * 2.0);
    let nearH = gather.inverseViewProjection * vec4f(clip, 0.0, 1.0);
    let farH = gather.inverseViewProjection * vec4f(clip, 1.0, 1.0);
    let start = nearH.xyz / nearH.w;
    let end = farH.xyz / farH.w;
    let delta = end - start;
    let distance = length(delta);
    rays[id.x] = Ray(vec4f(start, 0.00001), vec4f(delta / distance, distance), vec4u(0u));
}

@compute @workgroup_size(64)
fn generateHemisphere(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y * gather.settings.z) { return; }
    let anchor = id.x / gather.settings.z;
    let hit = cameraHits[anchor];
    var ray = gatherEmptyRay();
    if (hit.identity.x != 0u && hit.work.z == 0u) {
        ray = Ray(vec4f(hit.position.xyz + hit.normal.xyz * gather.sampling.x, 0.00001),
            vec4f(gatherDirection(anchor, id.x % gather.settings.z, hit.normal.xyz), gi.spacing.w), vec4u(0u));
    }
    rays[id.x] = ray;
}

@compute @workgroup_size(64)
fn generateShadows(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y * gather.settings.z) { return; }
    let hit = primaryHits[id.x];
    var ray = gatherEmptyRay();
    if (hit.identity.x != 0u && hit.work.z == 0u && hit.emissive.w == 0.0
        && dot(hit.normal.xyz, gi.sunDirection.xyz) > 0.0 && any(gi.sunColor.xyz > vec3f(0.0))) {
        ray = Ray(vec4f(hit.position.xyz + hit.normal.xyz * gather.sampling.x, 0.00001),
            vec4f(gi.sunDirection.xyz, gi.spacing.w), vec4u(0u));
    }
    rays[id.x] = ray;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y) { return; }
    let camera = cameraHits[id.x];
    var result: FinalGatherAnchor;
    result.position = camera.position;
    result.normal = camera.normal;
    result.identity = camera.identity;
    result.versions = camera.versions;
    let cameraSnapshotValid = camera.versions.x == gather.state.x && camera.versions.w == gather.emissive.z;
    var primaryFailures = select(0u, 1u, camera.work.z != 0u || !cameraSnapshotValid);
    var shadowFailures = 0u;
    var irradiance = vec3f(0.0);
    if (camera.identity.x != 0u && camera.work.z == 0u && cameraSnapshotValid) {
        for (var sample = 0u; sample < gather.settings.z; sample++) {
            let index = id.x * gather.settings.z + sample;
            let hit = primaryHits[index];
            let shadow = hits[index];
            primaryFailures += select(0u, 1u, hit.work.z != 0u
                || hit.versions.x != gather.state.x || hit.versions.w != gather.emissive.z);
            shadowFailures += select(0u, 1u, shadow.work.z != 0u
                || shadow.versions.x != gather.state.x || shadow.versions.w != gather.emissive.z);
            var incoming = gi.sky.xyz;
            if (hit.identity.x != 0u) {
                let includeEmission = gather.emissive.x == 0u || hit.work.w == 0u;
                incoming = select(vec3f(0.0), hit.emissive.xyz, includeEmission);
                if (hit.emissive.w != 0.0) {
                    incoming += select(vec3f(0.0), hit.baseColor.xyz, includeEmission);
                } else {
                    let visible = select(0.0, 1.0, shadow.identity.x == 0u && shadow.work.z == 0u);
                    var hitIrradiance = gi.sunColor.xyz * max(0.0, dot(hit.normal.xyz, gi.sunDirection.xyz)) * visible;
                    if (gather.state.w != 0u) {
                        hitIrradiance += giSampleIrradiance(hit.position.xyz, hit.normal.xyz, gi.state.x);
                    }
                    incoming += hit.baseColor.xyz * (1.0 - hit.previous.w) * (0.96 / 3.14159265359) * hitIrradiance;
                }
            }
            irradiance += clamp(incoming, vec3f(0.0), vec3f(65504.0)) * (3.14159265359 / f32(gather.settings.z));
        }
    }
    result.irradiance = vec4f(clamp(irradiance, vec3f(0.0), vec3f(65504.0)), 1.0);
    result.state = vec4u(select(0u, 1u, camera.identity.x != 0u && primaryFailures + shadowFailures == 0u),
        gather.state.x, primaryFailures, shadowFailures);
    anchors[id.x] = result;
}

@compute @workgroup_size(64)
fn clearArea(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.emissive.y) { return; }
    rays[id.x] = gatherEmptyRay();
    var sample: FinalGatherAreaSample;
    areaSamples[id.x] = sample;
}

@compute @workgroup_size(64)
fn generateArea(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y || area.shape.z == 0u) { return; }
    let camera = cameraHits[id.x];
    if (camera.identity.x == 0u || camera.work.z != 0u) { return; }
    let output = area.limits.w + id.x;
    let seed = gatherHash(id.x ^ (gather.state.x * 0x9e3779b9u) ^ area.shape.w);
    let triangle = gatherHash(seed ^ 0xa511e9b3u) % area.shape.y;
    let instance = gatherHash(seed ^ 0x632be59bu) % area.shape.z;
    let offset = instance * area.shape.x;
    let a = vertices[offset + indices[triangle * 3u]].position.xyz;
    let b = vertices[offset + indices[triangle * 3u + 1u]].position.xyz;
    let c = vertices[offset + indices[triangle * 3u + 2u]].position.xyz;
    let crossEdges = cross(b - a, c - a);
    let twiceArea = length(crossEdges);
    if (twiceArea == 0.0) { return; }
    let root = sqrt(gatherRandom(seed ^ 0x63d83595u));
    let v = gatherRandom(seed ^ 0x91e10da5u);
    let point = a * (1.0 - root) + b * (root * (1.0 - v)) + c * (root * v);
    let origin = camera.position.xyz + camera.normal.xyz * gather.sampling.x;
    let delta = point - origin;
    let distance = length(delta);
    if (distance < 0.00001 || distance > gi.spacing.w) { return; }
    let direction = delta / distance;
    let receiverCosine = max(0.0, dot(camera.normal.xyz, direction));
    let emitterCosine = abs(dot(crossEdges / twiceArea, -direction));
    let weight = receiverCosine * emitterCosine * twiceArea * 0.5 * f32(area.shape.y) * f32(area.shape.z)
        / (distance * distance);
    if (!(twiceArea > 0.0 && twiceArea <= 1e30 && weight >= 0.0 && weight <= 1e30)) {
        areaSamples[output].direction.w = -1.0;
        return;
    }
    if (weight == 0.0) { return; }
    let endpointTolerance = max(0.0001, distance * 0.00001);
    let identity = identities[instance];
    areaSamples[output] = FinalGatherAreaSample(vec4u(area.shape.w, identity.x, identity.y, triangle),
        vec4f(distance, weight, endpointTolerance, 0.0), vec4f(direction, 1.0));
    rays[output] = Ray(vec4f(origin, 0.00001), vec4f(direction, distance + endpointTolerance), vec4u(0u));
}

@compute @workgroup_size(64)
fn integrateArea(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y || anchors[id.x].identity.x == 0u) { return; }
    var emission = vec3f(0.0);
    var failures = 0u;
    for (var source = 0u; source < gather.emissive.x; source++) {
        let index = source * gather.settings.y + id.x;
        let sample = areaSamples[index];
        if (sample.direction.w < 0.0) { failures++; continue; }
        if (sample.expected.x == 0u) { continue; }
        let hit = hits[index];
        if (hit.work.z != 0u) { failures++; continue; }
        if (!all(hit.identity == sample.expected) || abs(hit.metric.x - sample.measure.x) > sample.measure.z) { continue; }
        let emitted = clamp(hit.emissive.xyz + select(vec3f(0.0), hit.baseColor.xyz, hit.emissive.w != 0.0),
            vec3f(0.0), vec3f(65504.0));
        let contribution = emitted * sample.measure.y;
        if (!all(contribution >= vec3f(0.0)) || !all(contribution <= vec3f(1e35))) { failures++; continue; }
        emission += contribution;
    }
    anchors[id.x].irradiance = vec4f(clamp(anchors[id.x].irradiance.xyz + emission, vec3f(0.0), vec3f(65504.0)), 1.0);
    anchors[id.x].state.z += failures;
    if (failures != 0u) { anchors[id.x].state.x = 0u; }
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y) { return; }
    var current = anchors[id.x];
    let previous = oldAnchors[id.x];
    let historyMatches = gather.state.z == 0u && current.state.x != 0u && previous.state.x != 0u
        && previous.state.y + 1u == gather.state.x && all(current.identity == previous.identity)
        && all(current.versions.yzw == previous.versions.yzw)
        && dot(current.normal.xyz, previous.normal.xyz) > 0.9999
        && distance(current.position.xyz, previous.position.xyz) < 0.0001;
    if (historyMatches) {
        let count = min(previous.irradiance.w + 1.0, f32(gather.settings.w));
        current.irradiance = vec4f(mix(previous.irradiance.xyz, current.irradiance.xyz, 1.0 / count), count);
    }
    anchors[id.x] = current;
}

// Read only the raw temporal field; the filtered output is never fed into that
// history. Surface identity and geometric checks keep adjacent walls and objects
// out of a receiver's irradiance filter, independently of its material albedo.
@compute @workgroup_size(64)
fn denoise(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y) { return; }
    var center = oldAnchors[id.x];
    if (center.state.x == 0u || center.state.y != gather.state.x) {
        anchors[id.x] = center;
        return;
    }
    let coordinate = vec2i(i32(id.x % gather.viewport.z), i32(id.x / gather.viewport.z));
    var sum = vec3f(0.0);
    var total = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let neighborCoordinate = coordinate + vec2i(x, y);
            if (any(neighborCoordinate < vec2i(0)) || any(neighborCoordinate >= vec2i(gather.viewport.zw))) { continue; }
            let index = u32(neighborCoordinate.x) + u32(neighborCoordinate.y) * gather.viewport.z;
            let neighbor = oldAnchors[index];
            if (neighbor.state.x == 0u || neighbor.state.y != gather.state.x
                || !all(neighbor.identity.xyz == center.identity.xyz)
                || !all(neighbor.versions.yzw == center.versions.yzw)
                || dot(neighbor.normal.xyz, center.normal.xyz) <= 0.98) { continue; }
            let delta = neighbor.position.xyz - center.position.xyz;
            if (length(delta) > 0.10 || abs(dot(delta, center.normal.xyz)) > 0.001
                || abs(dot(delta, neighbor.normal.xyz)) > 0.001) { continue; }
            let weight = exp(-f32(x * x + y * y) * 0.25);
            sum += neighbor.irradiance.xyz * weight;
            total += weight;
        }
    }
    // The valid center is always an eligible, positive-weight neighbor.
    center.irradiance = vec4f(clamp(sum / max(total, 0.000000000001), vec3f(0.0), vec3f(65504.0)), center.irradiance.w);
    anchors[id.x] = center;
}
