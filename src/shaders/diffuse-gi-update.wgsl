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

// GI_INTERPOLATE

fn giHitGeometry(hit: Hit) -> vec3f {
    return surfaceOctDecode(vec2f(hit.position.w, hit.uvInfo.w));
}

@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) id: vec3u) {
    let count = gi.counts.x * gi.counts.y * gi.counts.z;
    if (id.x >= count * gi.counts.w) { return; }
    let coordinate = giCoordinate(id.x / gi.counts.w, gi.counts.xyz);
    let origin = gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz;
    rays[id.x] = Ray(vec4f(origin, 0.0001), vec4f(giDirection(id.x % gi.counts.w, gi.counts.w), gi.spacing.w), vec4u(0u));
}

@compute @workgroup_size(64)
fn generateShadows(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gi.counts.x * gi.counts.y * gi.counts.z * gi.counts.w) { return; }
    let hit = primary[id.x];
    var ray = Ray(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(gi.sunDirection.xyz, 0.0), vec4u(0u));
    if (hit.identity.x != 0u && hit.work.z == 0u && hit.emissive.w == 0.0
        && dot(hit.normal.xyz, gi.sunDirection.xyz) > 0.0
        && dot(giHitGeometry(hit), gi.sunDirection.xyz) > 0.0 && any(gi.sunColor.xyz > vec3f(0.0))) {
        // Keep other triangles of this entity eligible for self-shadowing.
        ray.origin = vec4f(hit.position.xyz + giHitGeometry(hit) * gi.sunDirection.w, 0.0001);
        ray.direction.w = gi.spacing.w;
    }
    rays[id.x] = ray;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gi.counts.x * gi.counts.y * gi.counts.z) { return; }
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
                        * giSampleSurfaceIrradiance(hit.position.xyz, hit.normal.xyz, giHitGeometry(hit), gi.state.y);
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
        probes[id.x].coefficients[coefficient] = coefficients[coefficient];
    }
    probes[id.x].state = vec4u(select(0u, 1u, primaryOverflows + shadowOverflows == 0u), gi.state.x, primaryOverflows, shadowOverflows);
    probes[id.x].moments[0].w = 0.0;
}

// Separate texel invocations avoid a large private GiProbe copy, and all surface
// and feedback reads see this map only after the complete pass has finished.
@compute @workgroup_size(64)
fn integrateVisibility(@builtin(global_invocation_id) id: vec3u) {
    let probeCount = gi.counts.x * gi.counts.y * gi.counts.z;
    if (id.x >= probeCount * GI_DISTANCE_CELLS) { return; }
    let probeIndex = id.x / GI_DISTANCE_CELLS;
    let cell = id.x % GI_DISTANCE_CELLS;
    var moments = vec2f(0.0);
    var total = 0.0;
    for (var i = 0u; i < gi.counts.w; i++) {
        let index = probeIndex * gi.counts.w + i;
        var distance = gi.spacing.w;
        if (primary[index].identity.x != 0u) { distance = primary[index].metric.x; }
        let weight = distanceWeights[cell * gi.counts.w + i];
        moments += vec2f(distance, distance * distance) * weight;
        total += weight;
    }
    var emissiveFailures = 0.0;
    if (cell == 0u) { emissiveFailures = probes[probeIndex].moments[0].w; }
    probes[probeIndex].moments[cell] = vec4f(moments / max(total, 0.00000001), total, emissiveFailures);
}
