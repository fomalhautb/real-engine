struct GiParameters {
    origin: vec4f, // xyz world origin, w enabled
    spacing: vec4f, // xyz spacing, w maximum ray distance
    counts: vec4u, // xyz grid dimensions, w rays per probe
    sunDirection: vec4f, // xyz unit vector, w world-space normal bias
    sunColor: vec4f,
    sky: vec4f,
    state: vec4u, // rendered frame
}
struct GiProbe {
    coefficients: array<vec4f, 9>, // incoming radiance SH, not surface color
    moments: array<vec4f, 6>, // mean distance, mean squared distance, sample count
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
fn giFace(d: vec3f) -> u32 {
    let a = abs(d);
    var axis = 0u;
    if (a.y > a.x) { axis = 1u; }
    if (a.z > a[axis]) { axis = 2u; }
    return axis * 2u + select(1u, 0u, d[axis] >= 0.0);
}
fn giCoordinate(index: u32, counts: vec3u) -> vec3u {
    return vec3u(index % counts.x, (index / counts.x) % counts.y, index / (counts.x * counts.y));
}
