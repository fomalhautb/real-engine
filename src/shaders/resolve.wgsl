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
fn displayAt(coordinate: vec2i, last: vec2i) -> vec3f {
    let radiance = max(textureLoad(sceneColor, clamp(coordinate, vec2i(0), last), 0).rgb,
        vec3f(0.0)) * exposure.x;
    let mapped = clamp((radiance * (2.51 * radiance + 0.03)) /
        (radiance * (2.43 * radiance + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
    return linearToSrgb(mapped);
}
fn displayBilinear(position: vec2f, last: vec2i) -> vec3f {
    // FXAA blends in perceptually encoded display space. The HDR attachment is
    // kept single-sample, so convert each texel before manual interpolation.
    let base = position - vec2f(0.5);
    let corner = vec2i(floor(base));
    let weight = fract(base);
    let top = mix(displayAt(corner, last), displayAt(corner + vec2i(1, 0), last), weight.x);
    let bottom = mix(displayAt(corner + vec2i(0, 1), last),
        displayAt(corner + vec2i(1, 1), last), weight.x);
    return mix(top, bottom, weight.y);
}
fn displayLuma(color: vec3f) -> f32 {
    return dot(color, vec3f(0.299, 0.587, 0.114));
}
fn fxaa(position: vec2f, center: vec3f, last: vec2i) -> vec3f {
    let coordinate = vec2i(position);
    let lumaM = displayLuma(center);
    let lumaN = displayLuma(displayAt(coordinate + vec2i(0, -1), last));
    let lumaS = displayLuma(displayAt(coordinate + vec2i(0, 1), last));
    let lumaW = displayLuma(displayAt(coordinate + vec2i(-1, 0), last));
    let lumaE = displayLuma(displayAt(coordinate + vec2i(1, 0), last));
    let lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    let lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    let lumaRange = lumaMax - lumaMin;
    // FXAA quality early exit: return the center exactly in flat regions.
    if (lumaRange < max(0.063, lumaMax * 0.125)) { return center; }

    let lumaNW = displayLuma(displayAt(coordinate + vec2i(-1, -1), last));
    let lumaNE = displayLuma(displayAt(coordinate + vec2i(1, -1), last));
    let lumaSW = displayLuma(displayAt(coordinate + vec2i(-1, 1), last));
    let lumaSE = displayLuma(displayAt(coordinate + vec2i(1, 1), last));
    // X curvature is strongest across a near-vertical edge; Y curvature is
    // strongest across a near-horizontal edge. The chosen normal must follow
    // that curvature, while the endpoint search follows the edge tangent.
    let acrossX = abs(lumaNW + lumaNE - 2.0 * lumaN) +
        2.0 * abs(lumaW + lumaE - 2.0 * lumaM) +
        abs(lumaSW + lumaSE - 2.0 * lumaS);
    let acrossY = abs(lumaNW + lumaSW - 2.0 * lumaW) +
        2.0 * abs(lumaN + lumaS - 2.0 * lumaM) +
        abs(lumaNE + lumaSE - 2.0 * lumaE);
    let horizontalEdge = acrossY >= acrossX;
    let gradientNegative = select(abs(lumaW - lumaM), abs(lumaN - lumaM), horizontalEdge);
    let gradientPositive = select(abs(lumaE - lumaM), abs(lumaS - lumaM), horizontalEdge);
    let negativeSide = gradientNegative >= gradientPositive;
    let side = select(1.0, -1.0, negativeSide);
    let adjacent = select(select(lumaE, lumaW, negativeSide),
        select(lumaS, lumaN, negativeSide), horizontalEdge);
    let localAverage = 0.5 * (lumaM + adjacent);
    let gradientThreshold = 0.25 * max(gradientNegative, gradientPositive);
    let normal = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), horizontalEdge);
    let tangent = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), horizontalEdge);
    let edgePosition = position + normal * side * 0.5;

    // Fixed four-step end-of-edge search. The loop never scales with scene size.
    let steps = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
    var foundNegative = false;
    var foundPositive = false;
    var endNegative = 0.0;
    var endPositive = 0.0;
    var distanceNegative = 4.0;
    var distancePositive = 4.0;
    for (var i = 0; i < 4; i++) {
        let distance = steps[i];
        if (!foundNegative) {
            endNegative = displayLuma(displayBilinear(edgePosition - tangent * distance, last)) - localAverage;
            distanceNegative = distance;
            foundNegative = abs(endNegative) >= gradientThreshold;
        }
        if (!foundPositive) {
            endPositive = displayLuma(displayBilinear(edgePosition + tangent * distance, last)) - localAverage;
            distancePositive = distance;
            foundPositive = abs(endPositive) >= gradientThreshold;
        }
        if (foundNegative && foundPositive) { break; }
    }
    var edgeOffset = 0.0;
    if (foundNegative && foundPositive) {
        let nearestNegative = distanceNegative <= distancePositive;
        let endLuma = select(endPositive, endNegative, nearestNegative);
        // Reject a span whose nearer endpoint has the same sign as this pixel.
        if ((endLuma < 0.0) != (lumaM - localAverage < 0.0)) {
            edgeOffset = 0.5 - min(distanceNegative, distancePositive) /
                (distanceNegative + distancePositive);
        }
    }

    let neighborhood = (2.0 * (lumaN + lumaS + lumaW + lumaE) +
        lumaNW + lumaNE + lumaSW + lumaSE) / 12.0;
    var subpixel = clamp(abs(neighborhood - lumaM) / lumaRange, 0.0, 1.0);
    subpixel = subpixel * subpixel * (3.0 - 2.0 * subpixel);
    subpixel = subpixel * subpixel * 0.75;
    let offset = max(edgeOffset, subpixel);
    if (offset == 0.0) { return center; }
    return displayBilinear(position + normal * side * offset, last);
}
@fragment
fn fragmentMain(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
    let radiance = max(textureLoad(sceneColor, vec2i(pixel.xy), 0).rgb, vec3f(0.0)) * exposure.x;
    // Existing filmic fit, applied exactly once after linear HDR composition.
    let mapped = clamp((radiance * (2.51 * radiance + 0.03)) /
        (radiance * (2.43 * radiance + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
    let center = linearToSrgb(mapped);
    if (exposure.y < 0.5) { return vec4f(center, 1.0); }
    return vec4f(fxaa(pixel.xy, center, vec2i(textureDimensions(sceneColor)) - vec2i(1)), 1.0);
}
