struct FinalGatherParameters {
    inverseViewProjection: mat4x4f,
    eye: vec4f,
    viewport: vec4u,
    settings: vec4u,
    state: vec4u,
    sampling: vec4f,
    emissive: vec4u,
}
struct FinalGatherAnchor {
    position: vec4f,
    normal: vec4f,
    identity: vec4u,
    versions: vec4u,
    irradiance: vec4f,
    state: vec4u,
}
@group(3) @binding(2) var<uniform> gather: FinalGatherParameters;
@group(3) @binding(3) var<storage, read> finalGatherAnchors: array<FinalGatherAnchor>;

// Irradiance is filtered independently of the receiver's albedo. Uncovered,
// incomplete or incompatible geometry explicitly uses the completed world field.
fn finalGatherIrradiance(pixel: vec2f, position: vec3f, normal: vec3f) -> vec3f {
    if (gather.state.y == 0u || gather.state.x != gi.state.x) {
        return diffuseGiIrradiance(position, normal);
    }
    let pixelOffset = f32(gather.settings.x / 2u) + 0.5;
    let grid = (pixel - pixelOffset) / f32(gather.settings.x);
    let lower = vec2i(floor(grid));
    // A partial last cell clamps its camera anchor to the final pixel center.
    // Use those actual positions rather than pretending its pitch is unchanged.
    let lowerPixel = min(vec2f(lower) * f32(gather.settings.x) + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let upperPixel = min(vec2f(lower + vec2i(1)) * f32(gather.settings.x) + pixelOffset, vec2f(gather.viewport.xy) - 0.5);
    let fraction = clamp((pixel - lowerPixel) / max(upperPixel - lowerPixel, vec2f(1.0)), vec2f(0.0), vec2f(1.0));
    var sum = vec3f(0.0);
    var total = 0.0;
    for (var corner = 0u; corner < 4u; corner++) {
        let offset = vec2i(i32(corner & 1u), i32((corner >> 1u) & 1u));
        let coordinate = lower + offset;
        if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(gather.viewport.zw))) { continue; }
        let index = u32(coordinate.x) + u32(coordinate.y) * gather.viewport.z;
        let anchor = finalGatherAnchors[index];
        if (anchor.state.x == 0u || anchor.state.y != gather.state.x || anchor.versions.w != gather.emissive.z) { continue; }
        let delta = position - anchor.position.xyz;
        let normalDot = dot(normal, anchor.normal.xyz);
        if (normalDot < 0.95 || abs(dot(delta, normal)) > gather.sampling.z
            || abs(dot(delta, anchor.normal.xyz)) > gather.sampling.z || length(delta) > gather.sampling.w) { continue; }
        let axisWeights = select(vec2f(1.0) - fraction, fraction, offset == vec2i(1));
        let weight = axisWeights.x * axisWeights.y;
        sum += anchor.irradiance.xyz * weight;
        total += weight;
    }
    if (total < 0.001) { return diffuseGiIrradiance(position, normal); }
    return clamp(sum / total, vec3f(0.0), vec3f(65504.0));
}
