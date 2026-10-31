// Surface guide shared by the lobe-motion reconstruction and the spatial filter.
// Two anchors are compatible only on the same receiver identity and generation,
// with the reconstruction's material tolerances, normals within half the
// reflected lobe angle and a consistent receiving plane; the weight is a compact
// biweight of world distance over the center's footprint radius.
struct ReflectionGuide {
    position: vec4f, // receiver xyz, w world kernel radius (< 0: not a valid anchor)
    normal: vec4f, // shading normal xyz, w roughness
    geometry: vec4f, // geometric normal xyz, w metallic
    identity: vec4u, // batch, entity, generation, base color at 8 bits per channel
}

fn reflectionPackBase(color: vec3f) -> u32 {
    let q = vec3u(round(clamp(color, vec3f(0.0), vec3f(1.0)) * 255.0));
    return q.x | (q.y << 8u) | (q.z << 16u);
}

fn reflectionUnpackBase(word: u32) -> vec3f {
    return vec3f(f32(word & 255u), f32((word >> 8u) & 255u), f32((word >> 16u) & 255u)) / 255.0;
}

/** World distance over which the receiver can move while its lobe footprint on
 * content at `reflectedDistance` shifts by `scale` of its radius (CPU reflectionFilterRadius). */
fn reflectionFootprintRadius(roughness: f32, reflectedDistance: f32, viewDistance: f32, scale: f32) -> f32 {
    let sum = reflectedDistance + viewDistance;
    return select(0.0, scale * reflectionLobeTangent(roughness) * reflectedDistance * viewDistance / sum, sum > 0.0);
}

/** Guide of anchor `index`. The footprint radius uses the anchor's reflected-hit
 * distance (unfiltered.w before the filter) and is only read for scale > 0: a
 * neighbour's guide (scale 0) needs only its validity, so no pass reads a field
 * that a neighbouring invocation is writing. */
fn reflectionAnchorGuide(index: u32, scale: f32) -> ReflectionGuide {
    let state = anchors[index].state;
    let position = anchors[index].hit.position;
    let normal = anchors[index].hit.normal;
    var radius = -1.0;
    if (state.x != 0u && state.y == reflection.state.x) {
        radius = 0.0;
        if (scale > 0.0) {
            radius = reflectionFootprintRadius(normal.w, anchors[index].unfiltered.w,
                length(position.xyz - reflection.eye.xyz), scale);
        }
    }
    return ReflectionGuide(vec4f(position.xyz, radius), normal,
        vec4f(surfaceOctDecode(vec2f(position.w, anchors[index].hit.uvInfo.w)), anchors[index].hit.previous.w),
        vec4u(anchors[index].hit.identity.xyz, reflectionPackBase(anchors[index].hit.baseColor.xyz)));
}

fn reflectionGuideWeight(center: ReflectionGuide, other: ReflectionGuide) -> f32 {
    if (other.position.w < 0.0 || any(other.identity.xyz != center.identity.xyz)
        || abs(other.normal.w - center.normal.w) > 0.01 || abs(other.geometry.w - center.geometry.w) > 0.01
        || any(abs(reflectionUnpackBase(other.identity.w) - reflectionUnpackBase(center.identity.w)) > vec3f(0.02))) {
        return 0.0;
    }
    let offset = other.position.xyz - center.position.xyz;
    let distance = length(offset);
    if (distance >= center.position.w && distance > 0.0) { return 0.0; }
    // Normals within half the lobe angle reflect into the same lobe.
    let tolerance = cos(clamp(0.5 * atan(reflectionLobeTangent(center.normal.w)), 0.02, 0.45));
    let agreement = dot(center.geometry.xyz, other.geometry.xyz);
    if (agreement < tolerance || dot(center.normal.xyz, other.normal.xyz) < tolerance) { return 0.0; }
    let bend = distance * sqrt(max(1.0 - agreement * agreement, 0.0));
    if (abs(dot(offset, center.geometry.xyz)) > reflection.limits.w + bend) { return 0.0; }
    let u = select(0.0, distance / center.position.w, center.position.w > 0.0);
    let falloff = 1.0 - u * u;
    return falloff * falloff;
}
