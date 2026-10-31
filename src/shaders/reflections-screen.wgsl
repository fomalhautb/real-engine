// Optional short screen-space supplement for near-contact reflected detail.
// It reads the previous frame's scene-linear HDR and depth through the previous
// view-projection and replaces a world-query sample only where it finds the
// same surface at the same reflected distance. Every other case keeps the
// world-query radiance (off-screen, beyond range, sky, metallic or moving hits,
// unresolved or disagreeing marches): screen tracing never removes content.
@group(0) @binding(11) var reflectionScreenColor: texture_2d<f32>;
@group(0) @binding(12) var reflectionScreenDepth: texture_depth_2d;

const REFLECTION_SCREEN_ACCEPTED = 0u;
const REFLECTION_SCREEN_OFFSCREEN = 1u;
const REFLECTION_SCREEN_UNRESOLVED = 2u;
const REFLECTION_SCREEN_DISAGREE = 3u;
const REFLECTION_SCREEN_SKIPPED = 4u;

struct ReflectionScreenSample { color: vec3f, confidence: f32, reason: u32 }

fn reflectionScreenLinearDepth(depth: f32) -> f32 {
    let near = reflection.right.w; let far = reflection.up.w;
    return far * near / (far - depth * (far - near));
}

struct ReflectionScreenPoint { pixel: vec2u, ndc: vec2f, depth: f32, valid: bool }

fn reflectionScreenProject(world: vec3f) -> ReflectionScreenPoint {
    var point = ReflectionScreenPoint(vec2u(0u), vec2f(0.0), 0.0, false);
    let clip = reflection.previousViewProjection * vec4f(world, 1.0);
    // clip.w is the previous view depth for the engine's perspective projection.
    if (!(clip.w > reflection.right.w) || clip.w > reflection.up.w) { return point; }
    let ndc = clip.xy / clip.w;
    if (any(abs(ndc) >= vec2f(1.0))) { return point; }
    let size = vec2f(reflection.screenState.zw);
    let texel = min(vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size, size - vec2f(1.0));
    point.pixel = vec2u(texel); point.ndc = ndc; point.depth = clip.w; point.valid = true;
    return point;
}

/** Linear march of at most screenState.y steps over [0, world distance + tolerance]. */
fn reflectionScreenTrace(origin: vec3f, direction: vec3f, hit: ReflectionHit, distance: f32) -> ReflectionScreenSample {
    var result = ReflectionScreenSample(vec3f(0.0), 0.0, REFLECTION_SCREEN_SKIPPED);
    // screen: maximum distance, thickness, distance tolerance, edge fade.
    let limits = reflection.screen;
    if (reflection.screenState.x == 0u || hit.identity.x == 0u || distance > limits.x || hit.previous.w > 0.5
        || length(hit.previous.xyz - hit.position.xyz) > limits.z) { return result; }
    let steps = reflection.screenState.y;
    let range = distance + limits.z;
    let stride = range / f32(steps);
    var before = 0.0;
    for (var step = 1u; step <= steps; step++) {
        let t = stride * f32(step);
        let point = reflectionScreenProject(origin + direction * t);
        if (!point.valid) { result.reason = REFLECTION_SCREEN_OFFSCREEN; return result; }
        let scene = reflectionScreenLinearDepth(textureLoad(reflectionScreenDepth, point.pixel, 0));
        if (point.depth < scene) { before = t; continue; }
        // Crossed behind the previous visible surface: refine the crossing.
        var low = before; var high = t; var found = point;
        for (var refine = 0u; refine < 4u; refine++) {
            let middle = 0.5 * (low + high);
            let probe = reflectionScreenProject(origin + direction * middle);
            if (!probe.valid) { break; }
            let depth = reflectionScreenLinearDepth(textureLoad(reflectionScreenDepth, probe.pixel, 0));
            if (probe.depth < depth) { low = middle; } else { high = middle; found = probe; }
        }
        let behind = found.depth - reflectionScreenLinearDepth(textureLoad(reflectionScreenDepth, found.pixel, 0));
        if (behind > limits.y + stride) { result.reason = REFLECTION_SCREEN_UNRESOLVED; return result; }
        if (abs(high - distance) > limits.z + stride / 16.0) { result.reason = REFLECTION_SCREEN_DISAGREE; return result; }
        let edge = clamp((1.0 - max(abs(found.ndc.x), abs(found.ndc.y))) / limits.w, 0.0, 1.0);
        let reach = 1.0 - smoothstep(0.6 * limits.x, limits.x, high);
        result.color = textureLoad(reflectionScreenColor, found.pixel, 0).xyz;
        result.confidence = edge * reach;
        result.reason = REFLECTION_SCREEN_ACCEPTED;
        return result;
    }
    result.reason = REFLECTION_SCREEN_UNRESOLVED;
    return result;
}
