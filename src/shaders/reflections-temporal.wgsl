// Reprojected reflection history. A record belongs to one receiver surface
// identity/version; radiance is never reused across receivers, versions,
// material parameters or inconsistent reflected-hit distances.
struct ReflectionHistory {
    identity: vec4u, // batch, entity, generation, frame
    versions: vec4u, // history version, topology version, valid, rebuilt frames left
    position: vec4f, // receiver xyz, w mean reflected distance
    normal: vec4f, // receiver geometric normal, w roughness
    material: vec4f, // base color, metallic
    radiance: vec4f, // accumulated rgb, w effective history count
}
@group(0) @binding(9) var<storage, read> historyIn: array<ReflectionHistory>;
@group(0) @binding(10) var<storage, read_write> historyOut: array<ReflectionHistory>;

// Frames, including the response frame, in which a history rebuilt after a lobe
// change adds the spatial reconstruction instead of the anchor's own estimate.
const REFLECTION_REBUILD_FRAMES = 4u;
const REFLECTION_MODE_NONE = 0u;
const REFLECTION_MODE_STATIONARY = 1u;
const REFLECTION_MODE_SURFACE = 2u;
const REFLECTION_MODE_PARALLAX = 3u;
const REFLECTION_MODE_REJECTED = 4u;
// Rejection bits: identity/version, geometry, material, reflected distance, off-screen.
const REFLECTION_REJECT_IDENTITY = 1u;
const REFLECTION_REJECT_GEOMETRY = 2u;
const REFLECTION_REJECT_MATERIAL = 4u;
const REFLECTION_REJECT_DISTANCE = 8u;
const REFLECTION_REJECT_OFFSCREEN = 16u;

struct ReflectionCandidate { radiance: vec3f, count: f32, weight: f32, rejected: u32 }

fn reflectionHistoryLimit(roughness: f32) -> f32 {
    let t = clamp((roughness - 0.1) / 0.4, 0.0, 1.0);
    // floor(x + 0.5) matches the CPU Math.round contract for positive caps.
    return max(1.0, floor(reflection.temporal.x + (reflection.temporal.y - reflection.temporal.x) * t * t * (3.0 - 2.0 * t) + 0.5));
}

// This frame's estimate rebuilt from compatible anchors of the 5x5 neighbourhood
// within the reconstruction footprint, its effective anchor count, and the
// footprint-weighted fraction of their lobe weight whose reflected hit moved.
struct ReflectionRebuild { radiance: vec3f, count: f32, moved: f32 }

fn reflectionRebuild(index: u32) -> ReflectionRebuild {
    let center = reflectionAnchorGuide(index, reflection.spatial.z);
    let cell = vec2i(i32(index % reflection.viewport.z), i32(index / reflection.viewport.z));
    var sum = vec3f(0.0); var total = 0.0; var square = 0.0; var moved = 0.0;
    for (var tap = 0u; tap < 25u; tap++) {
        let coordinate = cell + vec2i(i32(tap % 5u) - 2, i32(tap / 5u) - 2);
        if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(reflection.viewport.zw))) { continue; }
        let other = u32(coordinate.y) * reflection.viewport.z + u32(coordinate.x);
        let weight = reflectionGuideWeight(center, reflectionAnchorGuide(other, 0.0));
        if (weight <= 0.0) { continue; }
        sum += anchors[other].raw.xyz * weight; total += weight; square += weight * weight;
        moved += anchors[other].motion.w * weight;
    }
    // The anchor itself always contributes with weight one.
    return ReflectionRebuild(sum / total, total * total / square, moved / total);
}

fn reflectionPreviousPixel(world: vec3f) -> vec3f {
    let clip = reflection.previousViewProjection * vec4f(world, 1.0);
    if (!(clip.w > 1e-6) || clip.w > 1e30) { return vec3f(0.0); }
    let ndc = clip.xy / clip.w;
    if (any(abs(ndc) > vec2f(1.0))) { return vec3f(0.0); }
    return vec3f((ndc.x * 0.5 + 0.5) * f32(reflection.viewport.x), (0.5 - ndc.y * 0.5) * f32(reflection.viewport.y), 1.0);
}

fn reflectionHistoryMatches(record: ReflectionHistory, current: ReflectionAnchor) -> u32 {
    let hit = current.hit;
    if (record.versions.z == 0u || record.identity.w + 1u != reflection.state.x
        || any(record.identity.xyz != hit.identity.xyz) || any(record.versions.xy != hit.versions.yz)) {
        return REFLECTION_REJECT_IDENTITY;
    }
    if (abs(record.normal.w - hit.normal.w) > 0.02 || abs(record.material.w - hit.previous.w) > 0.02
        || any(abs(record.material.xyz - hit.baseColor.xyz) > vec3f(0.03))) { return REFLECTION_REJECT_MATERIAL; }
    return 0u;
}

/** Bilinear previous-grid footprint at `pixel`; every corner is validated alone. */
fn reflectionGatherHistory(pixel: vec2f, current: ReflectionAnchor, receiverPrevious: vec3f,
    virtualPrevious: vec3f, parallax: bool) -> ReflectionCandidate {
    var result = ReflectionCandidate(vec3f(0.0), 0.0, 0.0, 0u);
    let spacing = f32(reflection.settings.x);
    let offset = f32(reflection.settings.x / 2u) + 0.5;
    let lower = vec2i(floor((pixel - offset) / spacing));
    let lowPixel = min(vec2f(lower) * spacing + offset, vec2f(reflection.viewport.xy) - 0.5);
    let highPixel = min(vec2f(lower + vec2i(1)) * spacing + offset, vec2f(reflection.viewport.xy) - 0.5);
    let fraction = clamp((pixel - lowPixel) / max(highPixel - lowPixel, vec2f(1.0)), vec2f(0.0), vec2f(1.0));
    let geometry = reflectionGeometry(current.hit);
    let roughness = current.hit.normal.w;
    let separation = reflection.limits.z * select(1.0, 8.0, parallax);
    for (var corner = 0u; corner < 4u; corner++) {
        let delta = vec2i(i32(corner & 1u), i32(corner >> 1u));
        let cell = lower + delta;
        if (any(cell < vec2i(0)) || any(cell >= vec2i(reflection.viewport.zw))) { continue; }
        let weights = select(vec2f(1.0) - fraction, fraction, delta == vec2i(1));
        let weight = weights.x * weights.y;
        if (weight <= 0.0) { continue; }
        let record = historyIn[u32(cell.y) * reflection.viewport.z + u32(cell.x)];
        var rejected = reflectionHistoryMatches(record, current);
        let toRecord = record.position.xyz - receiverPrevious;
        if (rejected == 0u && (dot(record.normal.xyz, geometry) < 0.98
            || abs(dot(toRecord, geometry)) > reflection.limits.w * 2.0 || length(toRecord) > separation)) {
            rejected = REFLECTION_REJECT_GEOMETRY;
        }
        // The same reflected content keeps a consistent distance: at the record
        // it equals its distance to the previous virtual image point.
        let expected = select(current.raw.w, length(virtualPrevious - record.position.xyz), parallax);
        if (rejected == 0u && abs(record.position.w - expected) > 0.05 + expected * (0.1 + roughness)) {
            rejected = REFLECTION_REJECT_DISTANCE;
        }
        if (rejected != 0u) { result.rejected |= rejected; continue; }
        result.radiance += record.radiance.xyz * weight;
        result.count += record.radiance.w * weight;
        result.weight += weight;
    }
    if (result.weight > 0.0) { result.radiance /= result.weight; result.count /= result.weight; }
    return result;
}

@compute @workgroup_size(64)
fn temporal(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y) { return; }
    let current = anchors[id.x];
    let hit = current.hit;
    if (current.state.x == 0u) {
        historyOut[id.x] = ReflectionHistory();
        anchors[id.x].temporal = vec4u(REFLECTION_MODE_NONE, 0u, 0u, 0u);
        return;
    }
    let raw = current.raw.xyz;
    let roughness = hit.normal.w;
    let geometry = reflectionGeometry(hit);
    var chosen = ReflectionCandidate(vec3f(0.0), 0.0, 0.0, 0u);
    var mode = REFLECTION_MODE_NONE;
    var rejected = 0u;
    // Frames left in which a history rebuilt after a lobe change still adds the
    // spatial reconstruction (stationary records only).
    var young = 0u;
    if (reflection.temporalState.y != 0u) {
        let receiverMotion = reflectionHitMotion(hit);
        let receiverPrevious = hit.position.xyz + receiverMotion;
        if (reflection.temporalState.z == 0u && all(receiverMotion == vec3f(0.0))) {
            // Stationary camera and receiver: the same anchor, without resampling.
            let record = historyIn[id.x];
            var reason = reflectionHistoryMatches(record, current);
            if (reason == 0u && any(abs(record.position.xyz - hit.position.xyz) > vec3f(1e-5 * (1.0 + hit.metric.x)))) {
                reason = REFLECTION_REJECT_GEOMETRY;
            }
            // Only scene motion can change static-view reflected content; rotated
            // samples straddling a static depth edge must not reset its history.
            if (reason == 0u && reflection.temporalState.w != 0u
                && abs(record.position.w - current.raw.w) > 0.05 + current.raw.w * (0.1 + roughness)) {
                reason = REFLECTION_REJECT_DISTANCE;
            }
            if (reason == 0u) {
                chosen = ReflectionCandidate(record.radiance.xyz, record.radiance.w, 1.0, 0u);
                mode = REFLECTION_MODE_STATIONARY; young = record.versions.w;
            }
            rejected = reason;
        } else {
            // Surface motion: where this receiver point was. Reflected-hit
            // parallax: where the previous virtual image of the reflected hit
            // projected, using the hit's own previous position for moving content.
            let view = normalize(hit.position.xyz - reflection.eye.xyz);
            let mirrored = reflect(view, geometry);
            let hitPrevious = hit.position.xyz + mirrored * current.raw.w + current.motion.xyz;
            let virtualPrevious = hitPrevious - 2.0 * dot(hitPrevious - receiverPrevious, geometry) * geometry;
            let surfacePixel = reflectionPreviousPixel(receiverPrevious);
            // Glossy lobes below the threshold try parallax first, then surface
            // motion; rougher lobes have no single virtual image and use surface motion.
            let useParallax = roughness < reflection.temporal.w;
            var surface = ReflectionCandidate(vec3f(0.0), 0.0, 0.0, REFLECTION_REJECT_OFFSCREEN);
            var parallax = ReflectionCandidate(vec3f(0.0), 0.0, 0.0, 0u);
            if (surfacePixel.z != 0.0) {
                surface = reflectionGatherHistory(surfacePixel.xy, current, receiverPrevious, virtualPrevious, false);
            }
            if (useParallax) {
                let parallaxPixel = reflectionPreviousPixel(virtualPrevious);
                parallax.rejected = REFLECTION_REJECT_OFFSCREEN;
                if (parallaxPixel.z != 0.0) {
                    parallax = reflectionGatherHistory(parallaxPixel.xy, current, receiverPrevious, virtualPrevious, true);
                }
            }
            if (parallax.weight > 0.0) { chosen = parallax; mode = REFLECTION_MODE_PARALLAX; }
            else if (surface.weight > 0.0) { chosen = surface; mode = REFLECTION_MODE_SURFACE; }
            rejected = surface.rejected | parallax.rejected;
        }
    }
    var count = 1.0;
    var resolved = raw;
    var clamped = 0u;
    var response = 0.0;
    // This frame's sample and its weight in frames: the anchor's own estimate, or
    // the reconstruction from compatible neighbours, worth their effective count.
    var sample = raw;
    var sampleWeight = 1.0;
    // A stationary or reprojected history exists; the response below may blend
    // into an empty one, which then has nothing to clamp.
    let found = chosen.weight > 0.0;
    if (reflection.spatial.y > 0.0 && reflection.temporalState.y != 0u) {
        let rebuilt = reflectionRebuild(id.x);
        response = smoothstep(0.0, reflection.spatial.y, rebuilt.moved);
        if (response > 0.0) {
            // Reflected content moved inside the lobe: the history keeps 1 - response
            // of its weight (none when rejected or missing) and the sample moves
            // toward the reconstruction; a mostly replaced lobe keeps adding the
            // reconstruction for REFLECTION_REBUILD_FRAMES frames in all.
            chosen.count = select(0.0, chosen.count * (1.0 - response), chosen.weight > 0.0);
            chosen.weight = 1.0;
            sample = mix(raw, rebuilt.radiance, response);
            sampleWeight = mix(1.0, rebuilt.count, response);
            young = select(0u, REFLECTION_REBUILD_FRAMES - 1u, response >= 0.5);
        } else if (young > 0u) {
            sample = rebuilt.radiance; sampleWeight = rebuilt.count; young -= 1u;
        }
    } else { young = 0u; }
    if (chosen.weight > 0.0) {
        var history = chosen.radiance;
        // Motion anywhere, a moving camera or a partially rejected footprint can
        // expose stale reflected content: clamp history to this frame's
        // compatible 3x3 anchor neighborhood before blending.
        if (found && (reflection.temporalState.z != 0u || reflection.temporalState.w != 0u || chosen.rejected != 0u
            || any(current.motion.xyz != vec3f(0.0)))) {
            let cell = vec2i(i32(id.x % reflection.viewport.z), i32(id.x / reflection.viewport.z));
            var mean = vec3f(0.0); var square = vec3f(0.0); var samples = 0.0;
            for (var neighbor = 0u; neighbor < 9u; neighbor++) {
                let coordinate = cell + vec2i(i32(neighbor % 3u) - 1, i32(neighbor / 3u) - 1);
                if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(reflection.viewport.zw))) { continue; }
                let index = u32(coordinate.y) * reflection.viewport.z + u32(coordinate.x);
                let other = anchors[index];
                if (other.state.x == 0u || other.state.y != reflection.state.x
                    || any(other.hit.identity.xy != hit.identity.xy)
                    || dot(reflectionGeometry(other.hit), geometry) < 0.9) { continue; }
                mean += other.raw.xyz; square += other.raw.xyz * other.raw.xyz; samples += 1.0;
            }
            mean /= max(samples, 1.0);
            let sigma = sqrt(max(square / max(samples, 1.0) - mean * mean, vec3f(0.0)));
            let t = clamp((roughness - 0.1) / 0.4, 0.0, 1.0);
            let width = reflection.temporal.z * (1.0 + t * t * (3.0 - 2.0 * t)) * sigma;
            let low = min(mean - width, raw);
            let high = max(mean + width, raw);
            let bounded = clamp(history, low, high);
            clamped = select(0u, 1u, any(bounded != history));
            history = bounded;
        }
        // Verified-static anchors of a static scene may use the stationary cap at any roughness.
        let stationaryCap = reflection.spatialState.z != 0u && mode == REFLECTION_MODE_STATIONARY
            && reflection.temporalState.w == 0u;
        count = min(chosen.count + sampleWeight, select(reflectionHistoryLimit(roughness), f32(reflection.spatialState.z), stationaryCap));
        resolved = mix(history, sample, min(sampleWeight / count, 1.0));
    }
    if (mode == REFLECTION_MODE_NONE && reflection.temporalState.y != 0u) { mode = REFLECTION_MODE_REJECTED; }
    resolved = clamp(resolved, vec3f(0.0), vec3f(65504.0));
    historyOut[id.x] = ReflectionHistory(vec4u(hit.identity.xyz, reflection.state.x), vec4u(hit.versions.yz, 1u, young),
        vec4f(hit.position.xyz, current.raw.w), vec4f(geometry, roughness), vec4f(hit.baseColor.xyz, hit.previous.w),
        vec4f(resolved, count));
    anchors[id.x].radiance = vec4f(resolved, count);
    anchors[id.x].unfiltered = vec4f(resolved, current.unfiltered.w);
    anchors[id.x].temporal = vec4u(mode, rejected, clamped, bitcast<u32>(response));
}
