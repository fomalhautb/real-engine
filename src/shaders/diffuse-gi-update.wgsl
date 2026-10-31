// GI_COMMON
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
struct Hit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
@group(0) @binding(0) var<uniform> gi: GiParameters;
@group(0) @binding(1) var<storage, read_write> rays: array<Ray>;
@group(0) @binding(2) var<storage, read> primary: array<Hit>;
@group(0) @binding(3) var<storage, read> shadows: array<Hit>;
@group(0) @binding(4) var<storage, read_write> probes: array<GiProbe>;
@group(0) @binding(5) var<storage, read> giProbes: array<GiProbe>;
@group(0) @binding(6) var<storage, read> distanceWeights: array<f32>;
// Probe captured by each slot of this update. A full update is the identity.
@group(0) @binding(7) var<storage, read> schedule: array<u32>;

// GI_INTERPOLATE

fn giHitGeometry(hit: Hit) -> vec3f {
    return surfaceOctDecode(vec2f(hit.position.w, hit.uvInfo.w));
}
fn giScheduledProbes() -> u32 { return u32(gi.shadow.y); }
// Feedback ignores captures older than the latest global light change.
fn giFeedbackCaptureFloor() -> u32 {
    let age = u32(gi.shadow.z);
    return select(0u, gi.state.x - age, gi.state.x > age);
}

// Scheduled updates first copy the previous field. Republish every carried
// probe under this field frame; captured slots overwrite their own records.
@compute @workgroup_size(64)
fn restamp(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gi.counts.x * gi.counts.y * gi.counts.z) { return; }
    probes[id.x].state.y = gi.state.x;
}

@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= giScheduledProbes() * gi.counts.w) { return; }
    let coordinate = giCoordinate(schedule[id.x / gi.counts.w], gi.counts.xyz);
    let origin = gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz;
    rays[id.x] = Ray(vec4f(origin, 0.0001), vec4f(giDirection(id.x % gi.counts.w, gi.counts.w), gi.spacing.w), vec4u(0u));
}

@compute @workgroup_size(64)
fn generateShadows(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= giScheduledProbes() * gi.counts.w) { return; }
    let hit = primary[id.x];
    var ray = Ray(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(gi.sunDirection.xyz, 0.0), vec4u(0u));
    if (hit.identity.x != 0u && hit.work.z == 0u && hit.emissive.w == 0.0
        && dot(hit.normal.xyz, gi.sunDirection.xyz) > 0.0
        && dot(giHitGeometry(hit), gi.sunDirection.xyz) > 0.0 && any(gi.sunColor.xyz > vec3f(0.0))) {
        // Keep other triangles of this entity eligible for self-shadowing.
        ray.origin = vec4f(hit.position.xyz + giHitGeometry(hit) * gi.shadow.x, 0.0001);
        ray.direction.w = gi.spacing.w;
    }
    rays[id.x] = ray;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= giScheduledProbes()) { return; }
    let probe = schedule[id.x];
    var coefficients: array<vec4f, 9>;
    var primaryOverflows = 0u;
    var shadowOverflows = 0u;
    var maximum = vec3f(0.0);
    for (var i = 0u; i < gi.counts.w; i++) {
        let index = id.x * gi.counts.w + i;
        let hit = primary[index];
        let shadow = shadows[index];
        primaryOverflows += select(0u, 1u, hit.work.z != 0u);
        shadowOverflows += select(0u, 1u, shadow.work.z != 0u);
        let direction = giDirection(i, gi.counts.w);
        var radiance = gi.sky.xyz;
        if (hit.identity.x != 0u) {
            // Explicit area sampling replaces only emission from opted-in
            // geometry. Its reflected sun/GI still belongs to the sphere rays.
            let sampleEmission = gi.state.w == 0u || hit.work.w == 0u;
            radiance = select(vec3f(0.0), hit.emissive.xyz, sampleEmission);
            if (hit.emissive.w != 0.0) {
                radiance += select(vec3f(0.0), hit.baseColor.xyz, sampleEmission);
            } else {
                let nl = select(0.0, max(0.0, dot(hit.normal.xyz, gi.sunDirection.xyz)), dot(giHitGeometry(hit), gi.sunDirection.xyz) > 0.0);
                let visible = select(0.0, 1.0, shadow.identity.x == 0u && shadow.work.z == 0u);
                // Diffuse-only capture: dielectric F0=.04, no specular bounce.
                radiance += hit.baseColor.xyz * (1.0 - hit.previous.w) * (0.96 / 3.14159265)
                    * gi.sunColor.xyz * nl * visible;
                if (gi.state.z != 0u) {
                    radiance += hit.baseColor.xyz * (1.0 - hit.previous.w) * (0.96 / 3.14159265)
                        * giSampleSurfaceIrradianceSince(hit.position.xyz, hit.normal.xyz, giHitGeometry(hit), gi.state.y,
                            giFeedbackCaptureFloor());
                }
            }
        }
        // Bounded linear HDR transport; bad/incomplete probes are never sampled.
        radiance = clamp(radiance, vec3f(0.0), vec3f(65504.0));
        maximum = max(maximum, radiance);
        let basis = giBasis(direction);
        for (var coefficient = 0u; coefficient < 9u; coefficient++) {
            coefficients[coefficient] += vec4f(radiance * basis[coefficient] * (12.56637061 / f32(gi.counts.w)), 0.0);
        }
    }
    coefficients[0].w = maximum.x;
    coefficients[1].w = maximum.y;
    coefficients[2].w = maximum.z;
    for (var coefficient = 0u; coefficient < 9u; coefficient++) {
        probes[probe].coefficients[coefficient] = coefficients[coefficient];
    }
    // Overflow counts are at most 256 rays each, so both fit one word.
    probes[probe].state = vec4u(select(0u, 1u, primaryOverflows + shadowOverflows == 0u), gi.state.x,
        primaryOverflows | (shadowOverflows << 16u), gi.state.x);
    probes[probe].moments[0].w = 0.0;
    probes[probe].moments[1].w = 0.0;
}

// One invocation per probe owns validity and the failure count. This pass is
// ordered before per-texel integration, so no texel races to publish validity.
@compute @workgroup_size(64)
fn validateVisibility(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= giScheduledProbes()) { return; }
    let probe = schedule[id.x];
    var failures = 0u;
    for (var i = 0u; i < gi.counts.w; i++) {
        failures += select(0u, 1u, shadows[id.x * gi.counts.w + i].work.z != 0u);
    }
    probes[probe].moments[1].w = f32(failures);
    if (failures != 0u) { probes[probe].state.x = 0u; }
}

// Separate texel invocations avoid a large private GiProbe copy, and all surface
// and feedback reads see this map only after the complete pass has finished.
@compute @workgroup_size(64)
fn integrateVisibility(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= giScheduledProbes() * GI_DISTANCE_CELLS) { return; }
    let slot = id.x / GI_DISTANCE_CELLS;
    let probeIndex = schedule[slot];
    let cell = id.x % GI_DISTANCE_CELLS;
    var moments = vec2f(0.0);
    var total = 0.0;
    for (var i = 0u; i < gi.counts.w; i++) {
        let index = slot * gi.counts.w + i;
        var distance = gi.spacing.w;
        if ((u32(gi.sunColor.w) & 1u) != 0u) {
            if (shadows[index].identity.x != 0u) { distance = shadows[index].metric.x; }
        } else if (primary[index].identity.x != 0u) { distance = primary[index].metric.x; }
        let weight = distanceWeights[cell * gi.counts.w + i];
        moments += vec2f(distance, distance * distance) * weight;
        total += weight;
    }
    var failures = 0.0;
    if (cell == 0u) { failures = probes[probeIndex].moments[0].w; }
    if (cell == 1u) { failures = probes[probeIndex].moments[1].w; }
    probes[probeIndex].moments[cell] = vec4f(moments / max(total, 0.00000001), total, failures);
}
