// Optional supplemental history separates retained surface coverage from the
// bounded current-frame trace list. Raw history is never spatially filtered.
// CACHE_CONSTANTS
@group(0) @binding(24) var<storage, read> cachePrevious: array<FinalGatherAnchor>;
@group(0) @binding(25) var<storage, read_write> cacheNext: array<FinalGatherAnchor>;
@group(0) @binding(26) var<storage, read_write> cacheFiltered: array<FinalGatherAnchor>;
@group(0) @binding(27) var<storage, read_write> cacheSurfaceMap: array<u32>;

fn cacheHistoryMatches(current: FinalGatherAnchor, previous: FinalGatherAnchor, previousPosition: vec3f) -> bool {
    if (gather.state.z != 0u || previous.state.x == 0u || previous.state.y + 1u != gather.state.x
        || previous.moments.y + 1.0 > f32(gather.coverage.w)
        || !all(previous.versions.yzw == current.versions.yzw)) { return false; }
    if (gather.emissive.w == 0u) {
        return all(current.identity == previous.identity)
            && dot(current.normal.xyz, previous.normal.xyz) > 0.9999
            && distance(current.position.xyz, previous.position.xyz) < 0.0001;
    }
    let delta = previous.position.xyz - previousPosition;
    return all(current.identity.xyz == previous.identity.xyz)
        && dot(current.normal.xyz, previous.normal.xyz) > 0.98 && length(delta) <= 0.10
        && abs(dot(delta, gatherAnchorGeometry(current))) <= 0.001
        && abs(dot(delta, gatherAnchorGeometry(previous))) <= 0.001;
}

@compute @workgroup_size(64)
fn updateSupplementalHistory(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.coverage.y * 8u) { return; }
    let hit = coverageCandidates[id.x];
    var current: FinalGatherAnchor;
    current.position = hit.position;
    current.normal = vec4f(hit.normal.xyz, hit.uvInfo.w);
    current.identity = hit.identity;
    current.versions = vec4u(gather.state.x, hit.versions.yz, gather.emissive.z);
    current.state.y = gather.state.x;
    current.irradiance.w = 1.0;
    if (hit.identity.x == 0u || hit.emissive.w != 0.0 || hit.versions.x != gather.state.x
        || hit.versions.w != gather.emissive.z) { cacheNext[id.x] = current; return; }

    var previous = cachePrevious[id.x];
    var found = cacheHistoryMatches(current, previous, hit.previous.xyz);
    if (!found && gather.state.z == 0u) {
        // Compacted trace slots and candidate ranks are not surface identities.
        let clip = gather.previousViewProjection * vec4f(hit.previous.xyz, 1.0);
        if (clip.w > 0.0 && clip.w <= 1e30) {
            let ndc = clip.xyz / clip.w;
            let pixel = (vec2f(ndc.x, -ndc.y) * 0.5 + 0.5) * vec2f(gather.viewport.xy);
            let shape = (gather.viewport.xy + vec2u(gather.raster.z - 1u)) / gather.raster.z;
            let cell = vec2i(floor(pixel / f32(gather.raster.z)));
            var nearest = 1e30;
            if (all(ndc >= vec3f(-1.0, -1.0, 0.0)) && all(ndc <= vec3f(1.0))) {
                for (var i = 0u; i < 72u; i++) {
                    let offset = i / 8u;
                    let coordinate = cell + vec2i(i32(offset % 3u) - 1, i32(offset / 3u) - 1);
                    if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(shape))) { continue; }
                    let index = (u32(coordinate.x) + u32(coordinate.y) * shape.x) * 8u + i % 8u;
                    let candidate = cachePrevious[index];
                    if (!cacheHistoryMatches(current, candidate, hit.previous.xyz)) { continue; }
                    let delta = candidate.position.xyz - hit.previous.xyz;
                    let score = dot(delta, delta);
                    if (score < nearest) { nearest = score; previous = candidate; found = true; }
                }
            }
        }
    }

    let mapped = coverageMap[id.x];
    if (mapped != 0u) {
        // Consume the new estimate before the ordinary accumulate pass. Never
        // merge an already accumulated estimate and count its history twice.
        current = anchors[mapped - 1u];
        current.moments = vec4f(0.0);
        if (current.state.x != 0u && found) {
            let cap = select(gather.settings.w, min(gather.settings.w, 8u), gather.emissive.w != 0u);
            let count = min(previous.irradiance.w + 1.0, f32(cap));
            let alpha = 1.0 / count;
            let delta = gatherLuminance(current.irradiance.xyz) - gatherLuminance(previous.irradiance.xyz);
            current.moments.x = max(0.0, (1.0 - alpha) * (previous.moments.x + alpha * delta * delta));
            current.irradiance = vec4f(mix(previous.irradiance.xyz, current.irradiance.xyz, alpha), count);
        }
        // A failed fresh query invalidates this entry; it is not hidden by old light.
    } else if (found) {
        current.irradiance = previous.irradiance;
        if (gather.emissive.w != 0u) { current.irradiance.w = min(current.irradiance.w, f32(min(gather.settings.w, 8u))); }
        current.moments = vec4f(previous.moments.x, previous.moments.y + 1.0, 0.0, 0.0);
        current.state.x = 1u;
    }
    cacheNext[id.x] = current;
}

@compute @workgroup_size(64)
fn filterSupplementalHistory(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.coverage.y * 8u) { return; }
    var center = cacheNext[id.x];
    let output = CACHE_PRIMARY_CAPACITY + id.x;
    cacheSurfaceMap[id.x] = select(0u, output + 1u, center.state.x != 0u);
    if (center.state.x == 0u) { cacheFiltered[output] = center; return; }
    let shape = (gather.viewport.xy + vec2u(gather.raster.z - 1u)) / gather.raster.z;
    let tile = id.x / 8u;
    let coordinate = vec2i(i32(tile % shape.x), i32(tile / shape.x));
    let centerLuminance = gatherLuminance(center.irradiance.xyz);
    let centerVariance = center.moments.x / max(center.irradiance.w - 1.0, 1.0);
    var sum = vec3f(0.0);
    var total = 0.0;
    for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
        let cell = coordinate + vec2i(x, y);
        if (any(cell < vec2i(0)) || any(cell >= vec2i(shape))) { continue; }
        for (var rank = 0u; rank < 8u; rank++) {
            let index = (u32(cell.x) + u32(cell.y) * shape.x) * 8u + rank;
            let neighbor = cacheNext[index];
            if (neighbor.state.x == 0u || neighbor.state.y != gather.state.x
                || !all(neighbor.identity.xyz == center.identity.xyz)
                || !all(neighbor.versions.yzw == center.versions.yzw)
                || dot(neighbor.normal.xyz, center.normal.xyz) <= 0.98) { continue; }
            let delta = neighbor.position.xyz - center.position.xyz;
            if (length(delta) > 0.10 || abs(dot(delta, gatherAnchorGeometry(center))) > 0.001
                || abs(dot(delta, gatherAnchorGeometry(neighbor))) > 0.001) { continue; }
            let neighborLuminance = gatherLuminance(neighbor.irradiance.xyz);
            let variance = centerVariance + neighbor.moments.x / max(neighbor.irradiance.w - 1.0, 1.0);
            let epsilon = 1e-8 + 1e-6 * max(centerLuminance, neighborLuminance);
            let range = exp(-abs(centerLuminance - neighborLuminance) / (4.0 * sqrt(variance) + epsilon));
            let reliability = clamp((min(center.irradiance.w, neighbor.irradiance.w) - 1.0) / 3.0, 0.0, 1.0);
            let weight = exp(-dot(delta, delta) / 0.0025) * mix(1.0, range, reliability);
            sum += neighbor.irradiance.xyz * weight;
            total += weight;
        }
    } }
    center.irradiance = vec4f(clamp(sum / max(total, 0.000000000001), vec3f(0.0), vec3f(65504.0)), center.irradiance.w);
    cacheFiltered[output] = center;
}
