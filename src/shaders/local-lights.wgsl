// Engine-owned bounded point/spot lights, shared by surfaces, GI capture, Final
// Gather, reflections and media. Radiometry and packing: src/local-lights.ts.
struct LocalLight {
    position: vec4f, // xyz world metres, w finite range
    axis: vec4f,     // xyz unit spot propagation axis, w kind: 0 point, 1 spot
    radiance: vec4f, // xyz linear colour × radiant intensity, w first atlas view or -1
    cone: vec4f,     // x cos outer, y 1 / (cos inner - cos outer), z shadow near, w shadow far
    shadow: vec4f,   // x tan(half view angle incl. PCF border), y normal bias texels, z depth bias texels
}
struct LocalLightBlock {
    header: vec4u, // enabled lights, shadowed lights, set revision, atlas resolution
    lights: array<LocalLight, LOCAL_LIGHT_CAPACITY>,
    views: array<mat4x4f, LOCAL_SHADOW_VIEW_CAPACITY>,
}
// LOCAL_LIGHT_BINDINGS

struct LocalLightSample { direction: vec3f, irradiance: vec3f }

// E⊥ = I × W(d) × C / max(d², 0.01²), W(d) = saturate(1 - (d/r)^4)^2. Exactly
// zero at and beyond the range, so every light has a bounded influence volume.
fn localLightUnshadowed(light: LocalLight, world: vec3f) -> LocalLightSample {
    var result = LocalLightSample(vec3f(0.0, 1.0, 0.0), vec3f(0.0));
    let delta = light.position.xyz - world;
    let distanceSquared = dot(delta, delta);
    let rangeSquared = light.position.w * light.position.w;
    if (!(distanceSquared < rangeSquared)) { return result; }
    let distance = sqrt(distanceSquared);
    if (distance > 0.0) { result.direction = delta / distance; }
    let ratio = distanceSquared / rangeSquared;
    let window = clamp(1.0 - ratio * ratio, 0.0, 1.0);
    var factor = window * window / max(distanceSquared, 0.0001);
    if (light.axis.w > 0.5) {
        let t = clamp((dot(-result.direction, light.axis.xyz) - light.cone.x) * light.cone.y, 0.0, 1.0);
        factor *= t * t;
    }
    result.irradiance = light.radiance.xyz * factor;
    return result;
}

// Point lights select the cube face by the dominant axis; the enlarged face
// frustum keeps the whole PCF footprint inside the selected view.
fn localLightShadowView(light: LocalLight, world: vec3f) -> u32 {
    let first = u32(light.radiance.w);
    if (light.axis.w > 0.5) { return first; }
    let d = world - light.position.xyz;
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) { return first + select(0u, 1u, d.x < 0.0); }
    if (a.y >= a.z) { return first + 2u + select(0u, 1u, d.y < 0.0); }
    return first + 4u + select(0u, 1u, d.z < 0.0);
}

// Normal-offset 3×3 tent PCF in the light's own perspective view. The offsets
// scale with the shadow texel footprint at the receiver, so the same rule works
// in fragment and compute stages. `offsetNormal` is zero for media samples.
fn localLightVisibility(light: LocalLight, world: vec3f, offsetNormal: vec3f) -> f32 {
    if (light.radiance.w < 0.0) { return 1.0; }
    let resolution = f32(localLights.header.w);
    let layer = localLightShadowView(light, world);
    let view = localLights.views[layer];
    let texel = 2.0 * max((view * vec4f(world, 1.0)).w, 0.0) * light.shadow.x / resolution;
    let clip = view * vec4f(world + offsetNormal * (light.shadow.y * texel), 1.0);
    let near = light.cone.z;
    let far = light.cone.w;
    if (!(clip.w > near)) { return 1.0; }
    let uv = clip.xy / clip.w * vec2f(0.5, -0.5) + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
    let depth = max(clip.w - light.shadow.z * texel, near);
    // Hardware depth of the biased linear depth for this perspective projection.
    let reference = far * (depth - near) / (depth * (far - near));
    let step = 0.5 / resolution;
    var lit = textureSampleCompareLevel(localShadowAtlas, localShadowSampler, uv + vec2f(-step, -step), layer, reference);
    lit += textureSampleCompareLevel(localShadowAtlas, localShadowSampler, uv + vec2f(step, -step), layer, reference);
    lit += textureSampleCompareLevel(localShadowAtlas, localShadowSampler, uv + vec2f(-step, step), layer, reference);
    lit += textureSampleCompareLevel(localShadowAtlas, localShadowSampler, uv + vec2f(step, step), layer, reference);
    return lit * 0.25;
}

// Σ E⊥ max(n·l, 0) for diffuse capture and gathering. Light from behind the
// geometric surface is rejected exactly as the sun path does.
fn localLightsIrradiance(world: vec3f, n: vec3f, geometricNormal: vec3f) -> vec3f {
    var sum = vec3f(0.0);
    for (var i = 0u; i < localLights.header.x; i++) {
        let light = localLights.lights[i];
        let sample = localLightUnshadowed(light, world);
        let nl = max(dot(n, sample.direction), 0.0);
        if (!any(sample.irradiance > vec3f(0.0)) || dot(geometricNormal, sample.direction) <= 0.0 || nl <= 0.0) { continue; }
        sum += sample.irradiance * nl * localLightVisibility(light, world, geometricNormal);
    }
    return sum;
}
