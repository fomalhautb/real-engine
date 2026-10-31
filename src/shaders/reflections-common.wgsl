struct ReflectionParameters {
    eye: vec4f, // xyz camera, w maximum secondary distance
    right: vec4f, // xyz scaled basis, w near clip
    up: vec4f, // xyz scaled basis, w far clip
    forward: vec4f,
    sunDirection: vec4f,
    sunColor: vec4f,
    sky: vec4f,
    viewport: vec4u, // width, height, columns, rows
    settings: vec4u, // spacing, anchors, samples per anchor, enabled
    state: vec4u, // frame, material revision, fog steps, reserved
    limits: vec4f, // normal bias, texture LOD, maximum gather distance, plane tolerance
    previousViewProjection: mat4x4f,
    temporal: vec4f, // minimum history, maximum history, clamp sigma, parallax roughness
    temporalState: vec4u, // active, history valid, camera moved, scene motion
}
struct ReflectionHit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
// radiance: resolved rgb and effective history count. raw: this frame's rgb and
// weighted mean reflected distance. motion: weighted reflected-hit motion
// (previous - current) and total BSDF weight. reflected: dominant sample's
// batch/entity/generation/sample. temporal: mode, rejection bits, clamp flag.
struct ReflectionAnchor {
    hit: ReflectionHit, radiance: vec4f, state: vec4u,
    raw: vec4f, motion: vec4f, reflected: vec4u, temporal: vec4u,
}
struct ReflectionSample { direction: vec4f, weight: vec4f }
fn reflectionGeometry(hit: ReflectionHit) -> vec3f {
    return surfaceOctDecode(vec2f(hit.position.w, hit.uvInfo.w));
}
// World-query hits report the current position parametrically (origin + t*d)
// and the previous one barycentrically, so a static surface differs only by
// rounding. Motion within this relative tolerance is treated as none.
fn reflectionHitMotion(hit: ReflectionHit) -> vec3f {
    let delta = hit.previous.xyz - hit.position.xyz;
    let extent = max(max(abs(hit.position.x), abs(hit.position.y)), max(abs(hit.position.z), hit.metric.x));
    return select(delta, vec3f(0.0), all(abs(delta) <= vec3f(1e-5 * (1.0 + extent))));
}
