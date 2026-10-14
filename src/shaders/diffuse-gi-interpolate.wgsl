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

fn giSampleIrradiance(position: vec3f, normal: vec3f, expectedFrame: u32) -> vec3f {
    if (gi.origin.w == 0.0) { return vec3f(0.0); }
    let biased = position + normal * gi.sunDirection.w;
    let grid = (biased - gi.origin.xyz) / gi.spacing.xyz;
    // A half-cell fringe allows the first free-space probes to light boundaries.
    if (any(grid < vec3f(-0.5)) || any(grid > vec3f(gi.counts.xyz - vec3u(1u)) + 0.5)) { return vec3f(0.0); }
    let clamped = clamp(grid, vec3f(0.0), vec3f(gi.counts.xyz - vec3u(1u)));
    let lower = min(vec3u(floor(clamped)), gi.counts.xyz - vec3u(2u));
    let fraction = clamped - vec3f(lower);
    let basis = giBasis(normal);
    var sum = vec3f(0.0);
    var total = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let offset = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
        let coordinate = lower + offset;
        let index = coordinate.x + gi.counts.x * (coordinate.y + gi.counts.y * coordinate.z);
        if (giProbes[index].state.x == 0u || giProbes[index].state.y != expectedFrame) { continue; }
        let probePosition = gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz;
        let delta = biased - probePosition;
        let distance = length(delta);
        let direction = delta / max(distance, 0.00001);
        let moment = giDistanceMoments(index, direction);
        let variance = max(moment.y - moment.x * moment.x, 0.0001);
        let beyond = max(0.0, distance - moment.x);
        let visibility = variance / (variance + beyond * beyond);
        let trilinear = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        let facing = max(0.05, (1.0 - dot(direction, normal)) * 0.5);
        var confidence = facing * facing * visibility * visibility * visibility;
        // A continuous curve suppresses weak occluded contributors before
        // spatial weighting, without introducing a hard visibility cutoff.
        if (confidence < gi.sky.w) {
            let fractionOfThreshold = confidence / gi.sky.w;
            confidence *= fractionOfThreshold * fractionOfThreshold;
        }
        let weight = trilinear.x * trilinear.y * trilinear.z * confidence;
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
    // Visible probes can legitimately have very small facing weights. Preserve
    // a uniform field instead of darkening it with a large normalization floor.
    return sum / max(total, 0.000000000001);
}
