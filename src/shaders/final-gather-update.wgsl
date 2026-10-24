// GI_COMMON
struct FinalGatherParameters {
    inverseViewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    eye: vec4f,
    viewport: vec4u, // width, height, columns, rows
    settings: vec4u, // actual spacing, anchors, rays per anchor, maximum history
    state: vec4u, // frame, enabled, reset history, multiple diffuse bounces
    sampling: vec4f, // independent normal bias, reserved, plane tolerance, maximum filter distance
    emissive: vec4u, // active source prefix, area query count, material revision, camera moving
    rasterOrigin: vec4f,
    rasterDx: vec4f,
    rasterDy: vec4f,
    rasterDz: vec4f,
    raster: vec4u, // receiver mode, metadata rows, supplemental tile pitch, supplemental count
    coverage: vec4u, // map tile capacity, actual tiles, scan permutation stride, supplemental history age limit
}
struct FinalGatherAnchor {
    position: vec4f,
    normal: vec4f,
    identity: vec4u, // source, entity, generation, triangle
    versions: vec4u, // rendered frame, history, topology, material
    irradiance: vec4f, // RGB irradiance, accumulated sample updates
    state: vec4u, // valid, rendered frame, primary/area failures, shadow failures
    moments: vec4f, // raw variance, retained age, optional sequence count and 24-bit key
}
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
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
// Immutable per-dispatch path parameters; no queue overwrite between encoded segments.
@group(0) @binding(14) var<uniform> path: vec4u; // segment index, configured segment count, reserved
@group(0) @binding(15) var<storage, read_write> throughput: array<vec4f>;
@group(0) @binding(16) var receiverMotionDepth: texture_2d<f32>;
@group(0) @binding(17) var receiverIdentityNormals: texture_2d<u32>;
@group(0) @binding(18) var<storage, read> receiverMetadata: array<ReceiverMetadata>;
@group(0) @binding(19) var<storage, read_write> rasterCameraHits: array<Hit>;
@group(0) @binding(20) var<storage, read_write> coverageCandidates: array<Hit>;
@group(0) @binding(21) var<storage, read_write> coveragePrefix: array<u32>;
@group(0) @binding(22) var<storage, read_write> coverageMap: array<u32>;
@group(0) @binding(23) var<storage, read> historyCoverageMap: array<u32>;

struct ReceiverMetadata {
    identity: vec4u, // source, entity, generation, flags
    versions: vec4u, // frame, history, topology, material
}

// GI_INTERPOLATE

fn gatherHitGeometry(hit: Hit) -> vec3f {
    return surfaceOctDecode(vec2f(hit.position.w, hit.uvInfo.w));
}
fn gatherAnchorGeometry(anchor: FinalGatherAnchor) -> vec3f {
    return surfaceOctDecode(vec2f(anchor.position.w, anchor.normal.w));
}
fn gatherLuminance(irradiance: vec3f) -> f32 {
    return dot(irradiance, vec3f(0.2126, 0.7152, 0.0722));
}

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
    let seed = gatherHash(anchor ^ (gather.state.x * 0x9e3779b9u) ^ (path.x * 0x85ebca6bu));
    let u = (f32(sample) + gatherRandom(seed)) / f32(gather.settings.z);
    let v = gatherRandom(seed ^ (sample * 0x632be59bu) ^ 0xa511e9b3u);
    var uv = vec2f(u, v);
    // GATHER_PROGRESSIVE_DIRECTION
    let radius = sqrt(uv.x);
    let angle = 6.28318530718 * uv.y;
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle))
        + normal * sqrt(max(0.0, 1.0 - uv.x)));
}
fn gatherEmptyRay() -> Ray {
    return Ray(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(0.0), vec4u(0u));
}

fn gatherRasterHit(pixel: vec2f) -> Hit {
    let texel = vec2i(floor(pixel));
    let packed = textureLoad(receiverIdentityNormals, texel, 0);
    let motionDepth = textureLoad(receiverMotionDepth, texel, 0);
    var hit: Hit;
    hit.versions = vec4u(gather.state.x, 0u, 0u, gather.emissive.z);
    if (packed.x != 0u && packed.x <= gather.raster.y && motionDepth.w > 0.0 && motionDepth.w < 1e30
        && all(abs(motionDepth.xyz) < vec3f(1e30))) {
        let metadata = receiverMetadata[packed.x - 1u];
        let clip = vec2f(pixel.x / f32(gather.viewport.x) * 2.0 - 1.0,
            1.0 - pixel.y / f32(gather.viewport.y) * 2.0);
        let position = gather.rasterOrigin.xyz + motionDepth.w *
            (gather.rasterDx.xyz * clip.x + gather.rasterDy.xyz * clip.y + gather.rasterDz.xyz);
        let encodedNormal = unpack2x16snorm(packed.z);
        let encodedGeometry = unpack2x16snorm(packed.w);
        if (metadata.identity.x != 0u && metadata.versions.x == gather.state.x
            && metadata.versions.w == gather.emissive.z && all(abs(position) < vec3f(1e30))) {
            hit.identity = vec4u(metadata.identity.xyz, packed.y);
            hit.position = vec4f(position, encodedGeometry.x);
            hit.normal = vec4f(surfaceOctDecode(encodedNormal), 0.0);
            hit.previous = vec4f(motionDepth.xyz, 0.0);
            hit.uvInfo.w = encodedGeometry.y;
            hit.emissive.w = select(0.0, 1.0, (metadata.identity.w & 1u) != 0u);
            hit.versions = metadata.versions;
        }
    }
    return hit;
}

@compute @workgroup_size(64)
fn loadRasterReceivers(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.viewport.z * gather.viewport.w) { return; }
    let coordinate = vec2u(id.x % gather.viewport.z, id.x / gather.viewport.z);
    let pixelOffset = f32(gather.settings.x / 2u) + 0.5;
    let pixel = min(vec2f(coordinate) * f32(gather.settings.x) + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    rasterCameraHits[id.x] = gatherRasterHit(pixel);
}

// Match the production reconstruction's geometry gates and weights. This pass
// reads only completed primary receivers; it never reads another workgroup's
// supplemental output or uses an atomic race to choose a surface.
fn gatherReceiverCompatible(anchor: Hit, receiver: Hit) -> bool {
    let delta = receiver.position.xyz - anchor.position.xyz;
    return anchor.identity.x != 0u && anchor.versions.x == gather.state.x
        && anchor.versions.w == gather.emissive.z && all(anchor.identity.xy == receiver.identity.xy)
        && dot(anchor.normal.xyz, receiver.normal.xyz) >= 0.95
        && abs(dot(delta, gatherHitGeometry(receiver))) <= gather.sampling.z
        && abs(dot(delta, gatherHitGeometry(anchor))) <= gather.sampling.z
        && length(delta) <= gather.sampling.w;
}

fn gatherPrimaryCovers(pixel: vec2f, receiver: Hit) -> bool {
    let pitch = f32(gather.settings.x);
    let pixelOffset = f32(gather.settings.x / 2u) + 0.5;
    let lower = vec2i(floor((pixel - pixelOffset) / pitch));
    let lowerPixel = min(vec2f(lower) * pitch + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let upperPixel = min(vec2f(lower + vec2i(1)) * pitch + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let fraction = clamp((pixel - lowerPixel) / max(upperPixel - lowerPixel, vec2f(1.0)), vec2f(0.0), vec2f(1.0));
    var total = 0.0;
    for (var corner = 0u; corner < 4u; corner++) {
        let offset = vec2i(i32(corner & 1u), i32((corner >> 1u) & 1u));
        let coordinate = lower + offset;
        if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(gather.viewport.zw))) { continue; }
        if (!gatherReceiverCompatible(rasterCameraHits[u32(coordinate.x) + u32(coordinate.y) * gather.viewport.z], receiver)) { continue; }
        let weight = select(vec2f(1.0) - fraction, fraction, offset == vec2i(1));
        total += weight.x * weight.y;
    }
    if (total >= 0.001) { return true; }
    for (var y = -1; y <= 2; y++) { for (var x = -1; x <= 2; x++) {
        if (x >= 0 && x <= 1 && y >= 0 && y <= 1) { continue; }
        let coordinate = lower + vec2i(x, y);
        if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(gather.viewport.zw))) { continue; }
        if (!gatherReceiverCompatible(rasterCameraHits[u32(coordinate.x) + u32(coordinate.y) * gather.viewport.z], receiver)) { continue; }
        let anchorPixel = min(vec2f(coordinate) * pitch + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
        let delta = (pixel - anchorPixel) / pitch;
        total += exp(-dot(delta, delta));
    } }
    return total >= 0.001;
}

@compute @workgroup_size(64)
fn loadSupplementalReceivers(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.coverage.y || gather.raster.z == 0u) { return; }
    let pitch = gather.raster.z;
    let columns = (gather.viewport.x + pitch - 1u) / pitch;
    let begin = vec2u(id.x % columns, id.x / columns) * pitch;
    let end = min(begin + vec2u(pitch), gather.viewport.xy);
    var candidates: array<Hit, 8>;
    var support: array<u32, 8>;
    for (var y = begin.y; y < end.y; y++) {
        for (var x = begin.x; x < end.x; x++) {
            let pixel = vec2f(f32(x) + 0.5, f32(y) + 0.5);
            let hit = gatherRasterHit(pixel);
            if (hit.identity.x != 0u && hit.emissive.w == 0.0 && !gatherPrimaryCovers(pixel, hit)) {
                var matched = false;
                for (var i = 0u; i < 8u; i++) {
                    if (support[i] != 0u && gatherReceiverCompatible(candidates[i], hit)) {
                        support[i]++;
                        matched = true;
                    }
                }
                if (!matched) {
                    // Keep a bounded set of representative surfaces. Replace
                    // the least-supported candidate; ties are deterministic.
                    // Counts measure retained matches, not an optimal cover.
                    var slot = 0u;
                    for (var i = 1u; i < 8u; i++) { if (support[i] < support[slot]) { slot = i; } }
                    candidates[slot] = hit;
                    support[slot] = 1u;
                }
            }
        }
    }
    for (var rank = 0u; rank < 8u; rank++) {
        var best = 0u;
        for (var i = 1u; i < 8u; i++) { if (support[i] > support[best]) { best = i; } }
        if (support[best] != 0u) {
            coverageCandidates[id.x * 8u + rank] = candidates[best];
            support[best] = 0u;
        } else {
            // Publish an empty record for every exhausted rank; never reuse
            // a previously selected candidate in the remaining output slots.
            coverageCandidates[id.x * 8u + rank] = Hit();
        }
    }
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
    var hit = cameraHits[anchor];
    var weight = vec3f(1.0);
    if (path.x != 0u) {
        hit = primaryHits[id.x];
        weight = throughput[id.x].xyz * hit.baseColor.xyz * (1.0 - hit.previous.w) * 0.96;
        if (hit.emissive.w != 0.0) { weight = vec3f(0.0); }
    }
    var ray = gatherEmptyRay();
    var continues = false;
    if (hit.identity.x != 0u && hit.work.z == 0u && any(weight > vec3f(0.0))) {
        let direction = gatherDirection(anchor, id.x % gather.settings.z, hit.normal.xyz);
        let geometryNormal = gatherHitGeometry(hit);
        if (dot(direction, geometryNormal) > 0.0) {
            ray = Ray(vec4f(hit.position.xyz + geometryNormal * gather.sampling.x, 0.00001),
                vec4f(direction, gi.spacing.w), vec4u(0u));
            continues = true;
        }
    }
    rays[id.x] = ray;
    throughput[id.x] = vec4f(select(vec3f(0.0), weight, continues), select(0.0, 1.0, continues));
}

@compute @workgroup_size(64)
fn generateShadows(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y * gather.settings.z) { return; }
    let hit = primaryHits[id.x];
    var ray = gatherEmptyRay();
    if (throughput[id.x].w != 0.0 && hit.identity.x != 0u && hit.work.z == 0u && hit.emissive.w == 0.0
        && dot(hit.normal.xyz, gi.sunDirection.xyz) > 0.0
        && dot(gatherHitGeometry(hit), gi.sunDirection.xyz) > 0.0 && any(gi.sunColor.xyz > vec3f(0.0))) {
        ray = Ray(vec4f(hit.position.xyz + gatherHitGeometry(hit) * gather.sampling.x, 0.00001),
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
    // Keep material direction for irradiance matching and store oct geometry
    // in the two unused w lanes for plane checks, without growing the anchor.
    result.normal = vec4f(camera.normal.xyz, camera.uvInfo.w);
    result.identity = camera.identity;
    result.versions = camera.versions;
    let cameraSnapshotValid = camera.versions.x == gather.state.x && camera.versions.w == gather.emissive.z;
    var primaryFailures = select(0u, 1u, camera.work.z != 0u || !cameraSnapshotValid);
    var shadowFailures = 0u;
    var irradiance = vec3f(0.0);
    if (path.x != 0u) {
        result = anchors[id.x];
        primaryFailures = result.state.z;
        shadowFailures = result.state.w;
        irradiance = result.irradiance.xyz;
    }
    if (camera.identity.x != 0u && camera.work.z == 0u && cameraSnapshotValid) {
        for (var sample = 0u; sample < gather.settings.z; sample++) {
            let index = id.x * gather.settings.z + sample;
            let weight = throughput[index];
            if (weight.w == 0.0) { continue; }
            let hit = primaryHits[index];
            let shadow = hits[index];
            primaryFailures += select(0u, 1u, hit.work.z != 0u
                || hit.versions.x != gather.state.x || hit.versions.w != gather.emissive.z);
            shadowFailures += select(0u, 1u, shadow.work.z != 0u
                || shadow.versions.x != gather.state.x || shadow.versions.w != gather.emissive.z);
            var incoming = gi.sky.xyz;
            if (hit.identity.x != 0u) {
                // Only the first segment has a separate camera-to-emitter area estimate.
                let includeEmission = path.x != 0u || gather.emissive.x == 0u || hit.work.w == 0u;
                incoming = select(vec3f(0.0), hit.emissive.xyz, includeEmission);
                if (hit.emissive.w != 0.0) {
                    incoming += select(vec3f(0.0), hit.baseColor.xyz, includeEmission);
                } else {
                    let visible = select(0.0, 1.0, shadow.identity.x == 0u && shadow.work.z == 0u);
                    let cosine = select(0.0, max(0.0, dot(hit.normal.xyz, gi.sunDirection.xyz)),
                        dot(gatherHitGeometry(hit), gi.sunDirection.xyz) > 0.0);
                    var hitIrradiance = gi.sunColor.xyz * cosine * visible;
                    if (gather.state.w != 0u && path.x + 1u == path.y) {
                        hitIrradiance += giSampleSurfaceIrradiance(hit.position.xyz, hit.normal.xyz, gatherHitGeometry(hit), gi.state.x);
                    }
                    incoming += hit.baseColor.xyz * (1.0 - hit.previous.w) * (0.96 / 3.14159265359) * hitIrradiance;
                }
            }
            irradiance += weight.xyz * clamp(incoming, vec3f(0.0), vec3f(65504.0)) * (3.14159265359 / f32(gather.settings.z));
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
    let origin = camera.position.xyz + gatherHitGeometry(camera) * gather.sampling.x;
    let delta = point - origin;
    let distance = length(delta);
    if (distance < 0.00001 || distance > gi.spacing.w) { return; }
    let direction = delta / distance;
    if (dot(gatherHitGeometry(camera), direction) <= 0.0) { return; }
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

fn gatherStationaryHistoryMatches(current: FinalGatherAnchor, previous: FinalGatherAnchor) -> bool {
    return gather.state.z == 0u && current.state.x != 0u && previous.state.x != 0u
        && previous.state.y + 1u == gather.state.x && all(current.identity == previous.identity)
        && all(current.versions.yzw == previous.versions.yzw)
        && dot(current.normal.xyz, previous.normal.xyz) > 0.9999
        && distance(current.position.xyz, previous.position.xyz) < 0.0001;
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.settings.y) { return; }
    var current = anchors[id.x];
    // A fresh estimate carries no evidence of temporal variance. Clear this
    // explicitly, including moving reprojection's early rejection paths.
    current.moments = vec4f(0.0);
    anchors[id.x].moments = current.moments;
    if (gather.emissive.w == 0u) {
        // Preserve the stationary estimator exactly, including its strict
        // triangle identity and 0.1 mm positional checks.
        var previous = oldAnchors[id.x];
        var historyMatches = gatherStationaryHistoryMatches(current, previous);
        // Compaction changes storage addresses when another tile changes.
        // Recover the exact stationary receiver through the previous map;
        // physical slots are not surface identities.
        if (!historyMatches && gather.state.z == 0u && current.state.x != 0u
            && id.x >= gather.viewport.z * gather.viewport.w) {
            let clip = gather.previousViewProjection * vec4f(current.position.xyz, 1.0);
            let pixel = vec2f(clip.x, -clip.y) / clip.w * vec2f(gather.viewport.xy) * 0.5 + vec2f(gather.viewport.xy) * 0.5;
            let shape = (gather.viewport.xy + vec2u(gather.raster.z - 1u)) / gather.raster.z;
            let cell = vec2i(floor(pixel / f32(gather.raster.z)));
            for (var candidate = 0u; candidate < 72u && !historyMatches; candidate++) {
                let tileOffset = candidate / 8u;
                let coordinate = cell + vec2i(i32(tileOffset % 3u) - 1, i32(tileOffset / 3u) - 1);
                if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(shape))) { continue; }
                let tile = u32(coordinate.x) + u32(coordinate.y) * shape.x;
                let mapped = historyCoverageMap[tile * 8u + candidate % 8u];
                if (mapped == 0u) { continue; }
                let value = oldAnchors[mapped - 1u];
                if (gatherStationaryHistoryMatches(current, value)) { previous = value; historyMatches = true; }
            }
        }
        if (historyMatches) {
            let count = min(previous.irradiance.w + 1.0, f32(gather.settings.w));
            let alpha = 1.0 / count;
            let delta = gatherLuminance(current.irradiance.xyz) - gatherLuminance(previous.irradiance.xyz);
            current.moments.x = max(0.0, (1.0 - alpha) * (previous.moments.x + alpha * delta * delta));
            current.irradiance = vec4f(mix(previous.irradiance.xyz, current.irradiance.xyz, 1.0 / count), count);
        }
        anchors[id.x] = current;
        return;
    }
    if (gather.state.z != 0u || current.state.x == 0u) { return; }
    let previousPosition = cameraHits[id.x].previous.xyz;
    let clip = gather.previousViewProjection * vec4f(previousPosition, 1.0);
    if (!(clip.w > 0.0 && clip.w <= 1e30)) { return; }
    let ndc = clip.xyz / clip.w;
    if (!all(ndc >= vec3f(-1.0, -1.0, 0.0)) || !all(ndc <= vec3f(1.0))) { return; }
    let pixel = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * vec2f(gather.viewport.xy);
    if (any(pixel < vec2f(0.0)) || any(pixel >= vec2f(gather.viewport.xy))) { return; }
    let pixelOffset = f32(gather.settings.x / 2u) + 0.5;
    let lower = vec2i(floor((pixel - pixelOffset) / f32(gather.settings.x)));
    let lowerPixel = min(vec2f(lower) * f32(gather.settings.x) + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let upperPixel = min(vec2f(lower + vec2i(1)) * f32(gather.settings.x) + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let fraction = clamp((pixel - lowerPixel) / max(upperPixel - lowerPixel, vec2f(1.0)), vec2f(0.0), vec2f(1.0));
    let supplemental = id.x >= gather.viewport.z * gather.viewport.w;
    let extraPitch = max(gather.raster.z, 1u);
    let extraShape = (gather.viewport.xy + vec2u(extraPitch - 1u)) / extraPitch;
    let extraCell = vec2i(floor(pixel / f32(extraPitch)));
    var sum = vec3f(0.0);
    var total = 0.0;
    var history = 0.0;
    var mixtureMean = 0.0;
    var varianceMass = 0.0;
    for (var corner = 0u; corner < select(4u, 72u, supplemental); corner++) {
        let offset = vec2i(i32(corner & 1u), i32((corner >> 1u) & 1u));
        let tileOffset = corner / 8u;
        let coordinate = select(lower + offset, extraCell + vec2i(i32(tileOffset % 3u) - 1, i32(tileOffset / 3u) - 1), supplemental);
        let shape = select(gather.viewport.zw, extraShape, supplemental);
        if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(shape))) { continue; }
        var index = u32(coordinate.x) + u32(coordinate.y) * shape.x;
        if (supplemental) {
            let mapped = historyCoverageMap[index * 8u + corner % 8u];
            if (mapped == 0u) { continue; }
            index = mapped - 1u;
        }
        let previous = oldAnchors[index];
        if (previous.state.x == 0u || previous.state.y + 1u != gather.state.x
            || !all(previous.identity.xyz == current.identity.xyz)
            || !all(previous.versions.yzw == current.versions.yzw)
            || dot(previous.normal.xyz, current.normal.xyz) <= 0.98) { continue; }
        let delta = previous.position.xyz - previousPosition;
        if (length(delta) > 0.10 || abs(dot(delta, gatherAnchorGeometry(current))) > 0.001
            || abs(dot(delta, gatherAnchorGeometry(previous))) > 0.001) { continue; }
        let axisWeights = select(vec2f(1.0) - fraction, fraction, offset == vec2i(1));
        let weight = select(axisWeights.x * axisWeights.y, 1.0 / (dot(delta, delta) + 0.000001), supplemental);
        if (weight <= 0.0) { continue; }
        // Weighted mixture variance includes differences among reprojected
        // means, without subtracting two large second moments.
        let nextTotal = total + weight;
        let meanDelta = gatherLuminance(previous.irradiance.xyz) - mixtureMean;
        varianceMass += weight * previous.moments.x + meanDelta * meanDelta * total * weight / nextTotal;
        mixtureMean += meanDelta * weight / nextTotal;
        sum += previous.irradiance.xyz * weight;
        history += previous.irradiance.w * weight;
        total += weight;
    }
    if (total > 0.00000001) {
        let count = min(history / total + 1.0, f32(min(gather.settings.w, 8u)));
        let alpha = 1.0 / count;
        let delta = gatherLuminance(current.irradiance.xyz) - mixtureMean;
        current.moments.x = max(0.0, (1.0 - alpha) * (varianceMass / total + alpha * delta * delta));
        current.irradiance = vec4f(mix(sum / total, current.irradiance.xyz, 1.0 / count), count);
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
    let supplemental = id.x >= gather.viewport.z * gather.viewport.w;
    let extraPitch = max(gather.raster.z, 1u);
    let shape = select(gather.viewport.zw, (gather.viewport.xy + vec2u(extraPitch - 1u)) / extraPitch, supplemental);
    let indexBase = select(0u, gather.viewport.z * gather.viewport.w, supplemental);
    var tile = id.x;
    if (supplemental) { tile = historyCoverageMap[gather.coverage.x * 8u + id.x - indexBase]; }
    let coordinate = vec2i(i32(tile % shape.x), i32(tile / shape.x));
    let centerLuminance = gatherLuminance(center.irradiance.xyz);
    let centerMeanVariance = center.moments.x / max(center.irradiance.w - 1.0, 1.0);
    var sum = vec3f(0.0);
    var total = 0.0;
    let radius = select(2, 1, supplemental);
    for (var y = -radius; y <= radius; y++) {
        for (var x = -radius; x <= radius; x++) {
            let neighborCoordinate = coordinate + vec2i(x, y);
            if (any(neighborCoordinate < vec2i(0)) || any(neighborCoordinate >= vec2i(shape))) { continue; }
            for (var rank = 0u; rank < select(1u, 8u, supplemental); rank++) {
            var index = u32(neighborCoordinate.x) + u32(neighborCoordinate.y) * shape.x;
            if (supplemental) {
                let mapped = historyCoverageMap[index * 8u + rank];
                if (mapped == 0u) { continue; }
                index = mapped - 1u;
            }
            let neighbor = oldAnchors[index];
            if (neighbor.state.x == 0u || neighbor.state.y != gather.state.x
                || !all(neighbor.identity.xyz == center.identity.xyz)
                || !all(neighbor.versions.yzw == center.versions.yzw)
                || dot(neighbor.normal.xyz, center.normal.xyz) <= 0.98) { continue; }
            let delta = neighbor.position.xyz - center.position.xyz;
            if (length(delta) > 0.10 || abs(dot(delta, gatherAnchorGeometry(center))) > 0.001
                || abs(dot(delta, gatherAnchorGeometry(neighbor))) > 0.001) { continue; }
            let neighborLuminance = gatherLuminance(neighbor.irradiance.xyz);
            let meanVariance = centerMeanVariance + neighbor.moments.x / max(neighbor.irradiance.w - 1.0, 1.0);
            let epsilon = 1e-8 + 1e-6 * max(centerLuminance, neighborLuminance);
            let range = exp(-abs(centerLuminance - neighborLuminance) / (4.0 * sqrt(meanVariance) + epsilon));
            // One sample cannot establish a noiseless field. Introduce range
            // rejection over four updates; saturated history uses an effective
            // count estimate, not a formal independent-sample confidence bound.
            let reliability = clamp((min(center.irradiance.w, neighbor.irradiance.w) - 1.0) / 3.0, 0.0, 1.0);
            let spatial = select(exp(-f32(x * x + y * y) * 0.25), exp(-dot(delta, delta) / 0.0025), supplemental);
            let weight = spatial * mix(1.0, range, reliability);
            sum += neighbor.irradiance.xyz * weight;
            total += weight;
            }
        }
    }
    // The valid center is always an eligible, positive-weight neighbor.
    center.irradiance = vec4f(clamp(sum / max(total, 0.000000000001), vec3f(0.0), vec3f(65504.0)), center.irradiance.w);
    anchors[id.x] = center;
}
