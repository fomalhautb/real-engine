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
    state: vec4u, // frame, material revision, reserved, reserved
    limits: vec4f, // normal bias, texture LOD, maximum gather distance, plane tolerance
}
struct ReflectionHit {
    identity: vec4u, metric: vec4f, position: vec4f, normal: vec4f,
    previous: vec4f, baseColor: vec4f, emissive: vec4f, versions: vec4u,
    uvInfo: vec4f, work: vec4u,
}
struct ReflectionAnchor { hit: ReflectionHit, radiance: vec4f, state: vec4u }
struct ReflectionSample { direction: vec4f, weight: vec4f }
fn reflectionGeometry(hit: ReflectionHit) -> vec3f {
    return surfaceOctDecode(vec2f(hit.position.w, hit.uvInfo.w));
}
