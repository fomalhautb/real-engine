struct FinalGatherParameters {
    inverseViewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    eye: vec4f,
    viewport: vec4u,
    settings: vec4u,
    state: vec4u,
    sampling: vec4f,
    emissive: vec4u,
    rasterOrigin: vec4f,
    rasterDx: vec4f,
    rasterDy: vec4f,
    rasterDz: vec4f,
    raster: vec4u,
}
struct FinalGatherAnchor {
    position: vec4f,
    normal: vec4f,
    identity: vec4u,
    versions: vec4u,
    irradiance: vec4f,
    state: vec4u,
    moments: vec4f,
}
@group(3) @binding(2) var<uniform> gather: FinalGatherParameters;
@group(3) @binding(3) var<storage, read> finalGatherAnchors: array<FinalGatherAnchor>;

struct FinalGatherReconstruction {
    irradiance: vec3f,
    total: f32,
    branch: u32, // 0 disabled, 1 primary, 2 recovery ring, 3 insufficient weight
    primaryAccepted: u32,
    ringAccepted: u32,
    supplementalAccepted: u32,
    rejections: array<u32, 16>, // four corners, then twelve ring cells
}

fn finalGatherRejectionMask(anchor: FinalGatherAnchor, position: vec3f,
    normal: vec3f, geometricNormal: vec3f, expectedIdentity: vec2u) -> u32 {
    var mask = 0u;
    if (anchor.state.x == 0u || anchor.state.y != gather.state.x || anchor.versions.w != gather.emissive.z) {
        mask |= 1u;
    }
    if (expectedIdentity.x != 0u && !all(anchor.identity.xy == expectedIdentity)) { mask |= 2u; }
    let delta = position - anchor.position.xyz;
    let anchorGeometry = surfaceOctDecode(vec2f(anchor.position.w, anchor.normal.w));
    if (!(dot(normal, anchor.normal.xyz) >= 0.95)) { mask |= 4u; }
    if (!(abs(dot(delta, geometricNormal)) <= gather.sampling.z)) { mask |= 8u; }
    if (!(abs(dot(delta, anchorGeometry)) <= gather.sampling.z)) { mask |= 16u; }
    if (!(length(delta) <= gather.sampling.w)) { mask |= 32u; }
    return mask;
}

fn finalGatherCompatible(anchor: FinalGatherAnchor, position: vec3f,
    normal: vec3f, geometricNormal: vec3f, expectedIdentity: vec2u) -> bool {
    return finalGatherRejectionMask(anchor, position, normal, geometricNormal, expectedIdentity) == 0u;
}

// Irradiance is filtered independently of the receiver's albedo. Uncovered,
// incomplete or incompatible geometry explicitly uses the completed world field.
fn finalGatherReconstruct(pixel: vec2f, position: vec3f, normal: vec3f, geometricNormal: vec3f,
    expectedIdentity: vec2u) -> FinalGatherReconstruction {
    var result: FinalGatherReconstruction;
    for (var slot = 0u; slot < 16u; slot++) { result.rejections[slot] = 128u; }
    if (gather.state.y == 0u || gather.state.x != gi.state.x) {
        result.irradiance = diffuseGiSurfaceIrradiance(position, normal, geometricNormal);
        return result;
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
        if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(gather.viewport.zw))) {
            result.rejections[corner] = 64u;
            continue;
        }
        let index = u32(coordinate.x) + u32(coordinate.y) * gather.viewport.z;
        let anchor = finalGatherAnchors[index];
        let rejection = finalGatherRejectionMask(anchor, position, normal, geometricNormal, expectedIdentity);
        result.rejections[corner] = rejection;
        if (rejection != 0u) { continue; }
        result.primaryAccepted++;
        let axisWeights = select(vec2f(1.0) - fraction, fraction, offset == vec2i(1));
        let weight = axisWeights.x * axisWeights.y;
        sum += anchor.irradiance.xyz * weight;
        total += weight;
    }
    if (total < 0.001) {
        // Thin details and normal maps can reject every immediate anchor while
        // the same surface is covered one cell away. Search a bounded outer
        // ring before falling back, with exactly the same geometric rejection.
        // This does not bridge parallel walls or relax normal compatibility.
        var ringSlot = 4u;
        for (var y = -1; y <= 2; y++) {
            for (var x = -1; x <= 2; x++) {
                if (x >= 0 && x <= 1 && y >= 0 && y <= 1) { continue; }
                let coordinate = lower + vec2i(x, y);
                if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(gather.viewport.zw))) {
                    result.rejections[ringSlot] = 64u;
                    ringSlot++;
                    continue;
                }
                let anchor = finalGatherAnchors[u32(coordinate.x) + u32(coordinate.y) * gather.viewport.z];
                let rejection = finalGatherRejectionMask(anchor, position, normal, geometricNormal, expectedIdentity);
                result.rejections[ringSlot] = rejection;
                ringSlot++;
                if (rejection != 0u) { continue; }
                result.ringAccepted++;
                let anchorPixel = min(vec2f(coordinate) * f32(gather.settings.x) + pixelOffset,
                    vec2f(gather.viewport.xy) - 0.5);
                let screenDistance = (pixel - anchorPixel) / f32(gather.settings.x);
                let weight = exp(-dot(screenDistance, screenDistance));
                sum += anchor.irradiance.xyz * weight;
                total += weight;
            }
        }
    }
    result.total = total;
    if (total < 0.001 && gather.raster.w != 0u && gather.raster.z != 0u) {
        let pitch = gather.raster.z;
        let shape = (gather.viewport.xy + vec2u(pitch - 1u)) / pitch;
        let cell = vec2i(floor(pixel / f32(pitch)));
        // A bounded second set of actual raster receivers. Primary and ring
        // success are unchanged; thin walls retain exactly the same gates.
        var extraSum = vec3f(0.0);
        var extraTotal = 0.0;
        for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
            let coordinate = cell + vec2i(x, y);
            if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(shape))) { continue; }
            let index = gather.viewport.z * gather.viewport.w + u32(coordinate.x) + u32(coordinate.y) * shape.x;
            let anchor = finalGatherAnchors[index];
            if (!finalGatherCompatible(anchor, position, normal, geometricNormal, expectedIdentity)) { continue; }
            let delta = position - anchor.position.xyz;
            let weight = 1.0 / (dot(delta, delta) + 0.000001);
            extraSum += anchor.irradiance.xyz * weight;
            extraTotal += weight;
            result.supplementalAccepted++;
        } }
        if (extraTotal > 0.0) {
            result.branch = 6u;
            result.total = extraTotal;
            result.irradiance = clamp(extraSum / extraTotal, vec3f(0.0), vec3f(65504.0));
            return result;
        }
    }
    if (total < 0.001) {
        result.branch = 3u;
        result.irradiance = diffuseGiSurfaceIrradiance(position, normal, geometricNormal);
        return result;
    }
    result.branch = select(1u, 2u, result.ringAccepted != 0u);
    result.irradiance = clamp(sum / total, vec3f(0.0), vec3f(65504.0));
    return result;
}

fn finalGatherIrradiance(pixel: vec2f, position: vec3f, normal: vec3f, geometricNormal: vec3f,
    expectedIdentity: vec2u) -> vec3f {
    return finalGatherReconstruct(pixel, position, normal, geometricNormal, expectedIdentity).irradiance;
}
