// Optional supplemental history separates retained surface coverage from the
// bounded current-frame trace list. Raw history is never spatially filtered.
// CACHE_CONSTANTS
const CACHE_PROGRESSIVE_SAMPLING: bool = false;
const CACHE_SAMPLING_SEED: u32 = 0u;
// Counts remain exact in f32 storage and never use the capped history weight.
const CACHE_SEQUENCE_LIMIT: u32 = 1048576u;
const CACHE_HALTON_BASES: array<vec2u, 4> = array<vec2u, 4>(vec2u(2u, 3u), vec2u(5u, 7u), vec2u(11u, 13u), vec2u(17u, 19u));
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

fn cacheCurrentSurface(hit: Hit) -> FinalGatherAnchor {
    var current: FinalGatherAnchor;
    current.position = hit.position;
    current.normal = vec4f(hit.normal.xyz, hit.uvInfo.w);
    current.identity = hit.identity;
    current.versions = vec4u(gather.state.x, hit.versions.yz, gather.emissive.z);
    current.state.y = gather.state.x;
    current.irradiance.w = 1.0;
    return current;
}

struct CacheHistoryResolution {
    previous: FinalGatherAnchor,
    found: bool,
}

// Sampling and accumulation must resolve the same previous surface, including
// rank changes and bounded moving reprojection. At most 73 records are checked.
fn cacheResolveHistory(index: u32, current: FinalGatherAnchor, previousPosition: vec3f) -> CacheHistoryResolution {
    var previous = cachePrevious[index];
    var found = cacheHistoryMatches(current, previous, previousPosition);
    if (!found && gather.state.z == 0u) {
        // Compacted trace slots and candidate ranks are not surface identities.
        let clip = gather.previousViewProjection * vec4f(previousPosition, 1.0);
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
                    if (!cacheHistoryMatches(current, candidate, previousPosition)) { continue; }
                    let delta = candidate.position.xyz - previousPosition;
                    let score = dot(delta, delta);
                    if (score < nearest) { nearest = score; previous = candidate; found = true; }
                }
            }
        }
    }
    return CacheHistoryResolution(previous, found);
}

// The optional preparation pass runs after coverage and before every transport
// segment. The unpublished next-cache buffer temporarily holds resolved OLD
// records. Only updateSupplementalHistory replaces them with completed values;
// no pass may read a neighbor from this buffer while that replacement runs.
@compute @workgroup_size(64)
fn prepareSupplementalHistory(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.coverage.y * 8u) { return; }
    let hit = coverageCandidates[id.x];
    if (hit.identity.x == 0u || hit.emissive.w != 0.0 || hit.versions.x != gather.state.x
        || hit.versions.w != gather.emissive.z) { cacheNext[id.x] = FinalGatherAnchor(); return; }
    let resolved = cacheResolveHistory(id.x, cacheCurrentSurface(hit), hit.previous.xyz);
    if (resolved.found) { cacheNext[id.x] = resolved.previous; }
    else { cacheNext[id.x] = FinalGatherAnchor(); }
}

fn cachePreparedHistory(index: u32) -> CacheHistoryResolution {
    let previous = cacheNext[index];
    return CacheHistoryResolution(previous, previous.state.x != 0u);
}

fn cacheInitialSequenceKey(current: FinalGatherAnchor) -> u32 {
    // A failed trace invalidates lighting history. Starting its replacement at
    // a frame-independent direction could retry the same over-budget ray forever.
    // Valid retained histories keep their original key via cacheSequenceState.
    var key = gatherHash(CACHE_SAMPLING_SEED ^ current.versions.x);
    for (var i = 0u; i < 4u; i++) { key = gatherHash(key ^ current.identity[i]); }
    for (var i = 0u; i < 3u; i++) {
        key = gatherHash(key ^ bitcast<u32>(current.position[i]));
        key = gatherHash(key ^ bitcast<u32>(current.normal[i]));
        key = gatherHash(key ^ current.versions[i + 1u]);
    }
    // A numeric 24-bit key is exactly representable, unlike NaN-prone f32 bitcasts.
    return key & 0x00ffffffu;
}

fn cacheSequenceState(current: FinalGatherAnchor, history: CacheHistoryResolution) -> vec2u {
    if (history.found) {
        let count = history.previous.moments.z;
        let key = history.previous.moments.w;
        if (count >= 1.0 && count <= f32(CACHE_SEQUENCE_LIMIT) && floor(count) == count
            && key >= 0.0 && key <= 16777215.0 && floor(key) == key) {
            if (count < f32(CACHE_SEQUENCE_LIMIT)) { return vec2u(u32(count), u32(key)); }
            // At the declared wrap limit begin a new random shift. History
            // weighting remains independent of this sampling-state transition.
            let nextKey = gatherHash(u32(key) ^ 0x9e3779b9u) & 0x00ffffffu;
            return vec2u(0u, select(nextKey, (u32(key) + 1u) & 0x00ffffffu, nextKey == u32(key)));
        }
    }
    return vec2u(0u, cacheInitialSequenceKey(current));
}

fn cacheRadicalInverse(index: u32, base: u32) -> f32 {
    var remaining = index;
    var weight = 1.0 / f32(base);
    var result = 0.0;
    // index <= 1048576 * 16 - 1, so at most 24 base-2 or 16 base-3 digits.
    for (var i = 0u; i < 24u && remaining != 0u; i++) {
        result += f32(remaining % base) * weight;
        remaining /= base;
        weight /= f32(base);
    }
    return min(result, 0.9999999403953552);
}

fn cacheProgressiveUv(anchor: u32, sample: u32, fallback: vec2f) -> vec2f {
    let primary = gather.viewport.z * gather.viewport.w;
    if (anchor < primary || anchor >= primary + gather.raster.w || gather.coverage.w == 0u) { return fallback; }
    let tile = coverageMap[gather.coverage.x * 8u + anchor - primary];
    if (tile >= gather.coverage.y) { return fallback; }
    // Invert only this tile's eight entries; physical trace slots can rotate.
    for (var rank = 0u; rank < 8u; rank++) {
        let index = tile * 8u + rank;
        if (coverageMap[index] != anchor + 1u) { continue; }
        let hit = cameraHits[anchor];
        let current = cacheCurrentSurface(hit);
        let history = cachePreparedHistory(index);
        let sequence = cacheSequenceState(current, history);
        let indexInSequence = sequence.x * gather.settings.z + sample;
        // Distinct prime dimensions cover the joint path space. Merely shifting
        // the same base pair would keep every segment on a 2D diagonal manifold.
        let key = gatherHash(sequence.y ^ (path.x * 0x85ebca6bu));
        let shift = vec2f(gatherRandom(key ^ 0x68bc21ebu), gatherRandom(key ^ 0x02e5be93u));
        let bases = CACHE_HALTON_BASES[path.x];
        return fract(vec2f(cacheRadicalInverse(indexInSequence, bases.x), cacheRadicalInverse(indexInSequence, bases.y)) + shift);
    }
    return fallback;
}

@compute @workgroup_size(64)
fn updateSupplementalHistory(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.coverage.y * 8u) { return; }
    let hit = coverageCandidates[id.x];
    var current = cacheCurrentSurface(hit);
    if (hit.identity.x == 0u || hit.emissive.w != 0.0 || hit.versions.x != gather.state.x
        || hit.versions.w != gather.emissive.z) { cacheNext[id.x] = current; return; }
    let history = cacheResolveHistory(id.x, current, hit.previous.xyz);
    let previous = history.previous;
    let found = history.found;

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
        if (CACHE_PROGRESSIVE_SAMPLING && current.state.x != 0u) {
            let sequence = cacheSequenceState(cacheCurrentSurface(hit), history);
            current.moments.z = f32(sequence.x + 1u);
            current.moments.w = f32(sequence.y);
        }
        // A failed fresh query invalidates this entry; it is not hidden by old light.
    } else if (found) {
        current.irradiance = previous.irradiance;
        if (gather.emissive.w != 0u) { current.irradiance.w = min(current.irradiance.w, f32(min(gather.settings.w, 8u))); }
        current.moments = vec4f(previous.moments.x, previous.moments.y + 1.0, 0.0, 0.0);
        if (CACHE_PROGRESSIVE_SAMPLING) {
            current.moments.z = previous.moments.z;
            current.moments.w = previous.moments.w;
        }
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
