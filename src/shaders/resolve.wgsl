@group(0) @binding(0) var sceneColor: texture_2d<f32>;
@group(0) @binding(1) var<uniform> exposure: vec4f;

@vertex
fn vertexMain(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(positions[vertex], 0.0, 1.0);
}
fn linearToSrgb(color: vec3f) -> vec3f {
    return select(12.92 * color, 1.055 * pow(color, vec3f(1.0 / 2.4)) - 0.055, color > vec3f(0.0031308));
}
@fragment
fn fragmentMain(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
    let radiance = max(textureLoad(sceneColor, vec2i(pixel.xy), 0).rgb, vec3f(0.0)) * exposure.x;
    // Existing filmic fit, applied exactly once after linear HDR composition.
    let mapped = clamp((radiance * (2.51 * radiance + 0.03)) /
        (radiance * (2.43 * radiance + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
    return vec4f(linearToSrgb(mapped), 1.0);
}
