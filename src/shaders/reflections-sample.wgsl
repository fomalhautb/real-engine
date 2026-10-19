@group(3) @binding(4) var<uniform> reflection: ReflectionParameters;
@group(3) @binding(5) var<storage, read> reflectionAnchors: array<ReflectionAnchor>;
fn reflectionRadiance(pixel: vec2f, position: vec3f, normal: vec3f, geometricNormal: vec3f,
    base: vec3f, metallic: f32, roughness: f32, identity: vec2u) -> vec3f {
    if (reflection.settings.w == 0u) { return vec3f(0.0); }
    let offset = f32(reflection.settings.x / 2u) + 0.5;
    let coordinate = (pixel - offset) / f32(reflection.settings.x);
    let lower = vec2i(floor(coordinate));
    let lowPixel = min(vec2f(lower) * f32(reflection.settings.x) + offset, vec2f(reflection.viewport.xy) - 0.5);
    let highPixel = min(vec2f(lower + vec2i(1)) * f32(reflection.settings.x) + offset, vec2f(reflection.viewport.xy) - 0.5);
    let fraction = clamp((pixel - lowPixel) / max(highPixel - lowPixel, vec2f(1.0)), vec2f(0.0), vec2f(1.0));
    var sum = vec3f(0.0); var total = 0.0;
    for (var corner = 0u; corner < 4u; corner++) {
        let delta = vec2i(i32(corner & 1u), i32(corner >> 1u));
        let cell = lower + delta;
        if (any(cell < vec2i(0)) || any(cell >= vec2i(reflection.viewport.zw))) { continue; }
        let anchor = reflectionAnchors[u32(cell.y) * reflection.viewport.z + u32(cell.x)];
        let hit = anchor.hit;
        if (anchor.state.x == 0u || anchor.state.y != reflection.state.x || hit.versions.w != reflection.state.y
            || any(hit.identity.xy != identity)) { continue; }
        let distance = position - hit.position.xyz;
        if (dot(normal, hit.normal.xyz) < 0.995 || dot(geometricNormal, reflectionGeometry(hit)) < 0.995
            || abs(dot(distance, geometricNormal)) > reflection.limits.w
            || length(distance) > reflection.limits.z || any(abs(base - hit.baseColor.xyz) > vec3f(0.02))
            || abs(metallic - hit.previous.w) > 0.01 || abs(roughness - hit.normal.w) > 0.01) { continue; }
        let weights = select(vec2f(1.0) - fraction, fraction, delta == vec2i(1));
        let weight = weights.x * weights.y;
        sum += anchor.radiance.xyz * weight; total += weight;
    }
    if (total < 0.001) { return vec3f(0.0); }
    return sum / total;
}
