// Shared by surface shading and feedback integration; both read a completed field.
fn giDistanceMoments(index: u32, direction: vec3f) -> vec2f {
    let pixel = giOctaEncode(direction) * f32(GI_DISTANCE_RESOLUTION) - 0.5;
    let lower = vec2i(floor(pixel));
    let fraction = fract(pixel);
    let bottom = mix(giProbes[index].moments[giDistanceTexel(lower)].xy,
        giProbes[index].moments[giDistanceTexel(lower + vec2i(1, 0))].xy, fraction.x);
    let top = mix(giProbes[index].moments[giDistanceTexel(lower + vec2i(0, 1))].xy,
        giProbes[index].moments[giDistanceTexel(lower + vec2i(1, 1))].xy, fraction.x);
    return mix(bottom, top, fraction.y);
}

fn giMomentVisibility(moment: vec2f, distance: f32) -> f32 {
    let variance = max(moment.y - moment.x * moment.x, 0.0001);
    let beyond = max(0.0, distance - moment.x);
    return variance / (variance + beyond * beyond);
}

fn giPerTapVisibility(index: u32, direction: vec3f, distance: f32) -> f32 {
    let pixel = giOctaEncode(direction) * f32(GI_DISTANCE_RESOLUTION) - 0.5;
    let lower = vec2i(floor(pixel));
    let fraction = fract(pixel);
    // Evaluate each directional distribution before interpolating confidence.
    // Mixing its moments first adds variance between unrelated near/far faces
    // and can turn a partially blocked footprint into full visibility.
    let bottom = mix(
        giMomentVisibility(giProbes[index].moments[giDistanceTexel(lower)].xy, distance),
        giMomentVisibility(giProbes[index].moments[giDistanceTexel(lower + vec2i(1, 0))].xy, distance), fraction.x);
    let top = mix(
        giMomentVisibility(giProbes[index].moments[giDistanceTexel(lower + vec2i(0, 1))].xy, distance),
        giMomentVisibility(giProbes[index].moments[giDistanceTexel(lower + vec2i(1, 1))].xy, distance), fraction.x);
    return mix(bottom, top, fraction.y);
}

fn giVisibility(index: u32, direction: vec3f, distance: f32) -> f32 {
    // Filter and single-sided support direction are independent options.
    if ((u32(gi.sunColor.w) & 2u) != 0u) {
        return giPerTapVisibility(index, direction, distance);
    }
    return giMomentVisibility(giDistanceMoments(index, direction), distance);
}

// Opt-in grid-axis-neighbor refinement (flag bit 2). Lanes match gi-visibility.ts.
const GI_AXIAL_LANE: u32 = 2u;
const GI_NEIGHBOR_LANE: u32 = 14u;

// The receiver offset along each grid axis must lie in front of the first
// surface the probe sees along that axis. Axial depth is invariant across a
// lobe for walls perpendicular to the axis, so thin walls stay decisive where
// the direction lobe straddles a silhouette or meets the wall at a grazing angle.
fn giAxisVisibility(index: u32, delta: vec3f) -> f32 {
    var result = 1.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let lane = GI_AXIAL_LANE + 2u * (2u * axis + select(1u, 0u, delta[axis] >= 0.0));
        let moment = vec2f(giProbes[index].moments[lane].w, giProbes[index].moments[lane + 1u].w);
        result = min(result, giMomentVisibility(moment, abs(delta[axis])));
    }
    return result;
}

fn giNeighborLane(offset: vec3i) -> u32 {
    return GI_NEIGHBOR_LANE + u32(offset.x + 1 + 3 * (offset.y + 1) + 9 * (offset.z + 1));
}

fn giInterpolationConfidence(facing: f32, visibility: f32) -> f32 {
    var confidence = facing * facing * visibility * visibility * visibility;
    if (confidence < gi.sky.w) {
        let fractionOfThreshold = confidence / gi.sky.w;
        confidence *= fractionOfThreshold * fractionOfThreshold;
    }
    return confidence;
}

fn giProbeIrradiance(index: u32, basis: array<f32, 9>) -> vec3f {
    var irradiance = vec3f(0.0);
    for (var coefficient = 0u; coefficient < 9u; coefficient++) {
        var convolution = 0.785398163;
        if (coefficient == 0u) { convolution = 3.14159265; }
        else if (coefficient < 4u) { convolution = 2.094395102; }
        irradiance += giProbes[index].coefficients[coefficient].xyz * basis[coefficient] * convolution;
    }
    let maximum = vec3f(giProbes[index].coefficients[0].w, giProbes[index].coefficients[1].w, giProbes[index].coefficients[2].w);
    return clamp(irradiance, vec3f(0.0), maximum * 3.14159265);
}

// Same weights as the default path, with each corner's visibility limited by
// the axial test and then by its mutual visibility with the strongest corner,
// so a cell split by a wall interpolates only the side the receiver is on.
fn giSampleRefinedIrradiance(biased: vec3f, normal: vec3f, geometricNormal: vec3f, lower: vec3u, fraction: vec3f,
    expectedFrame: u32, captureFloor: u32) -> vec3f {
    var indices: array<u32, 8>;
    var spatial: array<f32, 8>;
    var visibility: array<f32, 8>;
    var facing: array<f32, 8>;
    var valid: array<bool, 8>;
    var peakSpatialWeight = 0.0;
    var anchor = 0u;
    var anchorWeight = -1.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let offset = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
        let coordinate = lower + offset;
        let index = coordinate.x + gi.counts.x * (coordinate.y + gi.counts.y * coordinate.z);
        let trilinear = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        indices[corner] = index;
        spatial[corner] = trilinear.x * trilinear.y * trilinear.z;
        peakSpatialWeight = max(peakSpatialWeight, spatial[corner]);
        valid[corner] = !(giProbes[index].state.x == 0u || giProbes[index].state.y != expectedFrame
            || giProbes[index].state.w < captureFloor);
        if (!valid[corner]) { continue; }
        let delta = biased - (gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz);
        let distance = length(delta);
        let direction = delta / max(distance, 0.00001);
        visibility[corner] = min(giVisibility(index, direction, distance), giAxisVisibility(index, delta));
        facing[corner] = max(0.05, (1.0 - dot(direction, geometricNormal)) * 0.5);
        let weight = spatial[corner] * giInterpolationConfidence(facing[corner], visibility[corner]);
        if (weight > anchorWeight) { anchorWeight = weight; anchor = corner; }
    }
    let basis = giBasis(normal);
    let anchorOffset = vec3i(i32(anchor & 1u), i32((anchor >> 1u) & 1u), i32((anchor >> 2u) & 1u));
    var sum = vec3f(0.0);
    var total = 0.0;
    var visibleSupport = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        if (!valid[corner]) { continue; }
        var cornerVisibility = visibility[corner];
        if (corner != anchor) {
            let offset = vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u), i32((corner >> 2u) & 1u)) - anchorOffset;
            cornerVisibility *= min(giProbes[indices[anchor]].moments[giNeighborLane(offset)].w,
                giProbes[indices[corner]].moments[giNeighborLane(-offset)].w);
        }
        let weight = spatial[corner] * giInterpolationConfidence(facing[corner], cornerVisibility);
        if (weight > 0.0) {
            visibleSupport = max(visibleSupport, spatial[corner] * cornerVisibility * cornerVisibility * cornerVisibility);
        }
        sum += giProbeIrradiance(indices[corner], basis) * weight;
        total += weight;
    }
    let support = visibleSupport / peakSpatialWeight;
    var supportWeight = 1.0;
    if (support < gi.sky.w) {
        let fractionOfThreshold = support / gi.sky.w;
        supportWeight = fractionOfThreshold * fractionOfThreshold;
    }
    return sum / max(total, 0.000000000001) * supportWeight;
}

fn giSampleSurfaceIrradiance(position: vec3f, normal: vec3f, geometricNormal: vec3f, expectedFrame: u32) -> vec3f {
    return giSampleSurfaceIrradianceSince(position, normal, geometricNormal, expectedFrame, 0u);
}

// captureFloor additionally rejects probes captured before that frame; feedback
// uses it so a global light change never re-injects energy from older captures.
fn giSampleSurfaceIrradianceSince(position: vec3f, normal: vec3f, geometricNormal: vec3f, expectedFrame: u32,
    captureFloor: u32) -> vec3f {
    if (gi.origin.w == 0.0) { return vec3f(0.0); }
    let biased = position + geometricNormal * gi.sunDirection.w;
    let grid = (biased - gi.origin.xyz) / gi.spacing.xyz;
    // A half-cell fringe allows the first free-space probes to light boundaries.
    if (any(grid < vec3f(-0.5)) || any(grid > vec3f(gi.counts.xyz - vec3u(1u)) + 0.5)) { return vec3f(0.0); }
    let clamped = clamp(grid, vec3f(0.0), vec3f(gi.counts.xyz - vec3u(1u)));
    let lower = min(vec3u(floor(clamped)), gi.counts.xyz - vec3u(2u));
    let fraction = clamped - vec3f(lower);
    if ((u32(gi.sunColor.w) & 4u) != 0u) {
        return giSampleRefinedIrradiance(biased, normal, geometricNormal, lower, fraction, expectedFrame, captureFloor);
    }
    let basis = giBasis(normal);
    var sum = vec3f(0.0);
    var total = 0.0;
    var visibleSupport = 0.0;
    var peakSpatialWeight = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let offset = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
        let coordinate = lower + offset;
        let index = coordinate.x + gi.counts.x * (coordinate.y + gi.counts.y * coordinate.z);
        let trilinear = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        let spatialWeight = trilinear.x * trilinear.y * trilinear.z;
        peakSpatialWeight = max(peakSpatialWeight, spatialWeight);
        if (giProbes[index].state.x == 0u || giProbes[index].state.y != expectedFrame
            || giProbes[index].state.w < captureFloor) { continue; }
        let probePosition = gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz;
        let delta = biased - probePosition;
        let distance = length(delta);
        let direction = delta / max(distance, 0.00001);
        let visibility = giVisibility(index, direction, distance);
        let facing = max(0.05, (1.0 - dot(direction, geometricNormal)) * 0.5);
        var confidence = facing * facing * visibility * visibility * visibility;
        // A continuous curve suppresses weak occluded contributors before
        // spatial weighting, without introducing a hard visibility cutoff.
        if (confidence < gi.sky.w) {
            let fractionOfThreshold = confidence / gi.sky.w;
            confidence *= fractionOfThreshold * fractionOfThreshold;
        }
        let weight = spatialWeight * confidence;
        // Normalization must not turn a set of occluded probes into an
        // unoccluded field. Keep the best participating visibility separately
        // from facing: a visible probe with a grazing normal is still valid.
        // Spatial support vanishes continuously at cell boundaries; merely
        // entering a cell must not let a zero-weight corner restore full light.
        if (weight > 0.0) {
            visibleSupport = max(visibleSupport, spatialWeight * visibility * visibility * visibility);
        }
        var irradiance = vec3f(0.0);
        for (var coefficient = 0u; coefficient < 9u; coefficient++) {
            var convolution = 0.785398163;
            if (coefficient == 0u) { convolution = 3.14159265; }
            else if (coefficient < 4u) { convolution = 2.094395102; }
            irradiance += giProbes[index].coefficients[coefficient].xyz * basis[coefficient] * convolution;
        }
        // Positive SH ringing must not amplify energy on each feedback update.
        // Integral of a bounded nonnegative radiance is at most pi * its maximum.
        let maximum = vec3f(giProbes[index].coefficients[0].w, giProbes[index].coefficients[1].w, giProbes[index].coefficients[2].w);
        sum += clamp(irradiance, vec3f(0.0), maximum * 3.14159265) * weight;
        total += weight;
    }
    // Distance moments provide interpolation confidence, not physical
    // transmittance. Do not attenuate a well-supported field: this would lose
    // energy repeatedly in an otherwise uniform diffuse enclosure. Retain the
    // existing low-confidence curve only where every spatial contributor is
    // weak, so normalization cannot restore an unsupported bright field.
    let support = visibleSupport / peakSpatialWeight;
    var supportWeight = 1.0;
    if (support < gi.sky.w) {
        let fractionOfThreshold = support / gi.sky.w;
        supportWeight = fractionOfThreshold * fractionOfThreshold;
    }
    return sum / max(total, 0.000000000001) * supportWeight;
}

// Compatibility for diagnostics that intentionally use one surface direction.
fn giSampleIrradiance(position: vec3f, normal: vec3f, expectedFrame: u32) -> vec3f {
    return giSampleSurfaceIrradiance(position, normal, normal, expectedFrame);
}
