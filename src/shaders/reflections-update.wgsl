struct ReflectionRay { origin: vec4f, direction: vec4f, ignore: vec4u }
@group(0) @binding(0) var<uniform> reflection: ReflectionParameters;
@group(0) @binding(1) var<storage, read_write> rays: array<ReflectionRay>;
@group(0) @binding(2) var<storage, read> hits: array<ReflectionHit>;
@group(0) @binding(3) var<storage, read> cameras: array<ReflectionHit>;
@group(0) @binding(4) var<storage, read> secondary: array<ReflectionHit>;
@group(0) @binding(5) var<storage, read_write> samples: array<ReflectionSample>;
@group(0) @binding(6) var<storage, read_write> anchors: array<ReflectionAnchor>;
// REFLECTION_GI

fn reflectionHash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}
// Hashed per-anchor offset plus an R2 step per frame, as 0.32 fixed point.
fn reflectionRotation(frame: u32, anchor: u32) -> vec2u {
    let seed = reflectionHash((anchor * 0x9e3779b9u) ^ 0x5bd1e995u);
    let step = frame % 4096u;
    return vec2u(seed + step * 3242174889u, reflectionHash(seed) + step * 2447445414u);
}
fn emptyReflectionRay() -> ReflectionRay {
    return ReflectionRay(vec4f(0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4u(0u));
}
fn reflectionFailed(hit: ReflectionHit) -> bool {
    return hit.work.z != 0u || hit.versions.x != reflection.state.x || hit.versions.w != reflection.state.y;
}
@compute @workgroup_size(64)
fn generateCamera(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y) { return; }
    let cell = vec2u(id.x % reflection.viewport.z, id.x / reflection.viewport.z);
    let pixel = min(cell * reflection.settings.x + reflection.settings.x / 2u, reflection.viewport.xy - vec2u(1u));
    let uv = (vec2f(pixel) + 0.5) / vec2f(reflection.viewport.xy);
    let forwardRay = reflection.forward.xyz + reflection.right.xyz * (2.0 * uv.x - 1.0) + reflection.up.xyz * (1.0 - 2.0 * uv.y);
    let rayLength = length(forwardRay);
    rays[id.x] = ReflectionRay(vec4f(reflection.eye.xyz + forwardRay * reflection.right.w, 0.00001),
        vec4f(forwardRay / rayLength, (reflection.up.w - reflection.right.w) * rayLength), vec4u(0u));
}
@compute @workgroup_size(64)
fn generateReflections(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y * reflection.settings.z) { return; }
    let hit = cameras[id.x / reflection.settings.z];
    var ray = emptyReflectionRay();
    var sample: ReflectionSample;
    if (hit.identity.x != 0u && hit.emissive.w == 0.0 && !reflectionFailed(hit)) {
        let n = hit.normal.xyz;
        let v = normalize(reflection.eye.xyz - hit.position.xyz);
        let nv = dot(n, v);
        let roughness = hit.normal.w;
        let alpha = roughness * roughness;
        let a2 = alpha * alpha;
        let index = id.x % reflection.settings.z;
        var u = (f32(index) + 0.5) / f32(reflection.settings.z);
        var phi = 6.28318530718 * f32(reverseBits(index)) * 2.3283064365386963e-10;
        if (reflection.temporalState.x != 0u) {
            // Per-anchor Cranley-Patterson rotation of the same stratified set;
            // integer wrapping keeps the CPU replica exact (reflectionSampleCoordinates).
            let rotation = reflectionRotation(reflection.state.x, id.x / reflection.settings.z);
            phi = 6.28318530718 * f32((reverseBits(index) + rotation.x) >> 8u) * 5.9604644775390625e-8;
            u = min(fract(u + f32(rotation.y >> 8u) * 5.9604644775390625e-8), 0.99999994);
        }
        let cosine = sqrt((1.0 - u) / (1.0 + (a2 - 1.0) * u));
        let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
        let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
        let tangent = normalize(cross(helper, n));
        let half = normalize(tangent * (sine * cos(phi)) + cross(n, tangent) * (sine * sin(phi)) + n * cosine);
        let direction = reflect(-v, half);
        let nl = dot(n, direction);
        let vh = dot(v, half);
        if (nv > 0.0 && nl > 0.0 && vh > 0.0 && dot(direction, reflectionGeometry(hit)) > 0.0) {
            // NDF half-vector PDF cancels D in BRDF * NoL / pdf.
            // Match the direct path's Schlick-GGX geometry approximation.
            let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
            let shadingNv = max(nv, 0.001);
            let geometry = shadingNv / (shadingNv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
            let f0 = mix(vec3f(0.04), hit.baseColor.xyz, hit.previous.w);
            let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
            let weight = geometry * (4.0 * nl * vh) / max(cosine * max(4.0 * shadingNv * nl, 0.001), 1e-20);
            sample.weight = vec4f(fresnel * weight, 1.0);
            sample.direction = vec4f(direction, 0.0);
            ray = ReflectionRay(vec4f(hit.position.xyz + reflectionGeometry(hit) * reflection.limits.x, 0.00001),
                vec4f(direction, reflection.eye.w), vec4u(0u));
        }
    }
    samples[id.x] = sample; rays[id.x] = ray;
}
@compute @workgroup_size(64)
fn generateShadows(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y * reflection.settings.z) { return; }
    let hit = secondary[id.x];
    var ray = emptyReflectionRay();
    if (samples[id.x].weight.w > 0.0 && hit.identity.x != 0u && hit.emissive.w == 0.0 && !reflectionFailed(hit)
        && dot(hit.normal.xyz, reflection.sunDirection.xyz) > 0.0 && dot(reflectionGeometry(hit), reflection.sunDirection.xyz) > 0.0) {
        ray = ReflectionRay(vec4f(hit.position.xyz + reflectionGeometry(hit) * reflection.limits.x, 0.00001),
            vec4f(reflection.sunDirection.xyz, reflection.eye.w), vec4u(0u));
    }
    rays[id.x] = ray;
}
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y) { return; }
    let camera = cameras[id.x];
    var result: ReflectionAnchor;
    result.hit = camera;
    var failures = select(0u, 1u, reflectionFailed(camera));
    var sum = vec3f(0.0);
    var weightSum = 0.0; var distanceSum = 0.0; var motionSum = vec3f(0.0); var dominant = 0.0;
    result.reflected = vec4u(0u, 0u, 0u, 0xffffffffu);
    for (var sampleIndex = 0u; sampleIndex < reflection.settings.z; sampleIndex++) {
        let index = id.x * reflection.settings.z + sampleIndex;
        let sample = samples[index];
        if (sample.weight.w == 0.0) { continue; }
        let hit = secondary[index]; let shadow = hits[index];
        failures += select(0u, 1u, reflectionFailed(hit) || reflectionFailed(shadow));
        var incoming = reflection.sky.xyz;
        if (hit.identity.x != 0u) {
            incoming = hit.emissive.xyz;
            if (hit.emissive.w != 0.0) { incoming += hit.baseColor.xyz; }
            else {
                let visibility = select(0.0, 1.0, shadow.identity.x == 0u && shadow.work.z == 0u);
                incoming += surfaceDirectBrdf(hit.baseColor.xyz, hit.previous.w, hit.normal.w, hit.normal.xyz,
                    reflectionGeometry(hit), -sample.direction.xyz, reflection.sunDirection.xyz) * reflection.sunColor.xyz * visibility;
                incoming += reflectionDiffuse(hit);
            }
        }
        let distance = select(reflection.eye.w, hit.metric.x, hit.identity.x != 0u);
        incoming = reflectionApplyFog(camera.position.xyz + reflectionGeometry(camera) * reflection.limits.x,
            sample.direction.xyz, distance, incoming);
        sum += clamp(incoming, vec3f(0.0), vec3f(65504.0)) * sample.weight.xyz / f32(reflection.settings.z);
        // Reflected-hit statistics drive parallax reprojection and history checks.
        let lobe = dot(sample.weight.xyz, vec3f(0.2126, 0.7152, 0.0722));
        weightSum += lobe; distanceSum += lobe * distance;
        if (hit.identity.x != 0u) { motionSum += lobe * (hit.previous.xyz - hit.position.xyz); }
        if (lobe > dominant) { dominant = lobe; result.reflected = vec4u(hit.identity.xyz, sampleIndex); }
    }
    let raw = clamp(sum, vec3f(0.0), vec3f(65504.0));
    let meanDistance = select(reflection.eye.w, distanceSum / weightSum, weightSum > 0.0);
    result.raw = vec4f(raw, meanDistance);
    result.radiance = vec4f(raw, 1.0);
    result.motion = vec4f(select(vec3f(0.0), motionSum / weightSum, weightSum > 0.0), weightSum);
    result.state = vec4u(select(0u, 1u, camera.identity.x != 0u && camera.emissive.w == 0.0 && failures == 0u), reflection.state.x, failures, 0u);
    anchors[id.x] = result;
}
