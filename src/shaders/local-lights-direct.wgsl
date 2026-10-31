// PBR direct response to every enabled local light; requires surface-brdf.wgsl.
fn localLightsDirect(world: vec3f, n: vec3f, geometricNormal: vec3f, view: vec3f,
                     base: vec3f, metallic: f32, roughness: f32) -> vec3f {
    var sum = vec3f(0.0);
    for (var i = 0u; i < localLights.header.x; i++) {
        let light = localLights.lights[i];
        let sample = localLightUnshadowed(light, world);
        if (!any(sample.irradiance > vec3f(0.0)) || dot(geometricNormal, sample.direction) <= 0.0) { continue; }
        let response = surfaceDirectBrdf(base, metallic, roughness, n, geometricNormal, view, sample.direction);
        if (!any(response > vec3f(0.0))) { continue; }
        sum += response * sample.irradiance * localLightVisibility(light, world, geometricNormal);
    }
    return sum;
}
