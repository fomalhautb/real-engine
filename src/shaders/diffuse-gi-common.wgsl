// GI_DISTANCE_CONSTANTS
struct GiParameters {
    origin: vec4f, // xyz world origin, w enabled
    spacing: vec4f, // xyz spacing, w maximum ray distance
    counts: vec4u, // xyz grid dimensions, w rays per probe
    sunDirection: vec4f, // xyz unit vector, w receiver offset for spatial probe interpolation
    sunColor: vec4f, // xyz radiance; w exact integer flags: bit 0 receiver-directed support, bit 1 per-tap Chebyshev
    sky: vec4f, // xyz radiance, w low-confidence visibility weight threshold
    state: vec4u, // rendered frame, usable history frame, feedback enabled, emissive area sampling enabled
    shadow: vec4f, // x capture-to-sun geometric-normal bias; yzw reserved
}
struct GiProbe {
    coefficients: array<vec4f, 9>, // RGB incoming radiance SH; first three w values store sampled RGB maxima
    moments: array<vec4f, GI_DISTANCE_CELLS>, // mean distance, squared distance, weight; 0.w emissive failures, 1.w support failures
    state: vec4u, // valid, frame, primary overflows, shadow overflows
}
fn giDirection(index: u32, count: u32) -> vec3f {
    let y = 1.0 - 2.0 * (f32(index) + 0.5) / f32(count);
    let radius = sqrt(max(0.0, 1.0 - y * y));
    let phi = f32(index) * 2.399963229728653;
    return vec3f(cos(phi) * radius, y, sin(phi) * radius);
}
fn giBasis(d: vec3f) -> array<f32, 9> {
    return array<f32, 9>(0.28209479, 0.48860251 * d.y, 0.48860251 * d.z, 0.48860251 * d.x,
        1.09254843 * d.x * d.y, 1.09254843 * d.y * d.z, 0.31539157 * (3.0 * d.z * d.z - 1.0),
        1.09254843 * d.x * d.z, 0.546274215 * (d.x * d.x - d.y * d.y));
}
fn giOctaEncode(d: vec3f) -> vec2f {
    let projected = d / max(dot(abs(d), vec3f(1.0)), 0.00000001);
    var uv = projected.xy;
    if (projected.z < 0.0) {
        let signs = select(vec2f(-1.0), vec2f(1.0), uv >= vec2f(0.0));
        uv = (vec2f(1.0) - abs(uv.yx)) * signs;
    }
    return uv * 0.5 + 0.5;
}
// Reflect one-texel bilinear padding across each octahedral fold.
fn giDistanceTexel(coordinate: vec2i) -> u32 {
    let size = i32(GI_DISTANCE_RESOLUTION);
    var x = coordinate.x;
    var y = coordinate.y;
    if (x < 0) { x = 0; y = size - 1 - y; }
    else if (x >= size) { x = size - 1; y = size - 1 - y; }
    if (y < 0) { y = 0; x = size - 1 - x; }
    else if (y >= size) { y = size - 1; x = size - 1 - x; }
    return u32(y) * GI_DISTANCE_RESOLUTION + u32(x);
}
fn giCoordinate(index: u32, counts: vec3u) -> vec3u {
    return vec3u(index % counts.x, (index / counts.x) % counts.y, index / (counts.x * counts.y));
}
