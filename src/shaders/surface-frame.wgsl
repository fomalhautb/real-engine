// Shared geometry/shading frame. Geometry determines visibility and offsets;
// tangent-space detail changes only the material's scattering direction.
fn surfaceUnit(value: vec3f, fallback: vec3f) -> vec3f {
    let scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (scale > 0.0 && all(abs(value) <= vec3f(3.402823e38f))) { return normalize(value / scale); }
    return fallback;
}
fn surfaceOctEncode(normal: vec3f) -> vec2f {
    let n = normal / dot(abs(normal), vec3f(1.0));
    var result = n.xy;
    if (n.z < 0.0) { result = (vec2f(1.0) - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)); }
    return result;
}
fn surfaceOctDecode(encoded: vec2f) -> vec3f {
    var n = vec3f(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (n.z < 0.0) { n = vec3f((vec2f(1.0) - abs(n.yx)) * select(vec2f(-1.0), vec2f(1.0), n.xy >= vec2f(0.0)), n.z); }
    return surfaceUnit(n, vec3f(0.0, 0.0, 1.0));
}
fn surfaceMappedNormal(vertexNormal: vec3f, geometryNormal: vec3f, authoredTangent: vec4f,
    edge1: vec3f, edge2: vec3f, uv1: vec2f, uv2: vec2f, tangentNormal: vec3f) -> vec3f {
    let n = surfaceUnit(vertexNormal, geometryNormal);
    var tangent = authoredTangent.xyz;
    var handedness = authoredTangent.w;
    if (handedness == 0.0) {
        // Missing authored tangents use the actual deformed triangle UV frame.
        // This fallback is explicit; it does not reproduce authored MikkTSpace.
        let determinant = uv1.x * uv2.y - uv1.y * uv2.x;
        tangent = vec3f(0.0);
        handedness = 1.0;
        if (abs(determinant) > max(1e-20, length(uv1) * length(uv2) * 1e-7)) {
            tangent = (edge1 * uv2.y - edge2 * uv1.y) / determinant;
            let bitangent = (edge2 * uv1.x - edge1 * uv2.x) / determinant;
            handedness = select(-1.0, 1.0, dot(cross(n, tangent), bitangent) >= 0.0);
        }
    }
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = surfaceUnit(tangent - n * dot(n, tangent), normalize(cross(helper, n)));
    let b = cross(n, t) * select(-1.0, 1.0, handedness >= 0.0);
    let face = select(-1.0, 1.0, dot(n, geometryNormal) >= 0.0);
    var mapped = surfaceUnit((t * tangentNormal.x + b * tangentNormal.y + n * tangentNormal.z) * face, geometryNormal);
    // Shading normals must remain on the visible side of the real surface.
    let cosine = dot(mapped, geometryNormal);
    if (cosine < 0.0001) { mapped = surfaceUnit(mapped + geometryNormal * (0.0001 - cosine), geometryNormal); }
    return mapped;
}
