// Local-light single scattering in the shared world media. Uses the fog frame,
// volumes and fogPhase from volumetric-fog-common/world, and a deterministic
// candidate mask provider: fogLocalCandidateWord(word) -> u32.
fn fogLocalLightsScattering(world: vec3f, direction: vec3f) -> vec3f {
    if (localLights.header.x == 0u) { return vec3f(0.0); }
    var globalDensity = 0.0;
    if (frame.viewport.z > 0.0) {
        globalDensity = exp(-frame.globalOptics.w * max(world.y - frame.globalOptics.z, 0.0));
    }
    // Media membership is evaluated once; per-light work only adds phase terms.
    var media = array<u32, 2>(0u, 0u);
    var present = globalDensity > 0.0;
    for (var i = 0u; i < frame.grid.w; i++) {
        let volume = volumes[i];
        let delta = world - volume.centerKind.xyz;
        var inside = all(abs(delta) <= volume.extent.xyz);
        if (volume.centerKind.w > 0.5) { inside = dot(delta, delta) <= volume.extent.x * volume.extent.x; }
        if (inside) { media[i / 32u] |= 1u << (i % 32u); present = true; }
    }
    if (!present) { return vec3f(0.0); }
    let ray = normalize(direction);
    var result = vec3f(0.0);
    for (var word = 0u; word < LOCAL_LIGHT_MASK_WORDS; word++) {
        var bits = fogLocalCandidateWord(word);
        loop {
            if (bits == 0u) { break; }
            let index = word * 32u + firstTrailingBit(bits);
            bits &= bits - 1u;
            let light = localLights.lights[index];
            let sample = localLightUnshadowed(light, world);
            if (!any(sample.irradiance > vec3f(0.0))) { continue; }
            let incident = sample.irradiance * localLightVisibility(light, world, vec3f(0.0));
            if (!any(incident > vec3f(0.0))) { continue; }
            // Incoming propagation is -direction-to-light; outgoing to the ray origin is -ray.
            let cosine = clamp(dot(ray, sample.direction), -1.0, 1.0);
            var scattering = vec3f(0.0);
            if (globalDensity > 0.0) {
                scattering += globalDensity * frame.globalOptics.x * frame.globalAlbedo.xyz * fogPhase(cosine, frame.globalOptics.y);
            }
            for (var i = 0u; i < frame.grid.w; i++) {
                if ((media[i / 32u] & (1u << (i % 32u))) == 0u) { continue; }
                let volume = volumes[i];
                scattering += volume.optics.x * volume.albedo.xyz * fogPhase(cosine, volume.optics.y);
            }
            result += scattering * incident;
        }
    }
    return result;
}
