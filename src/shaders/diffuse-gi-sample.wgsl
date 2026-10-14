// GI_COMMON
@group(3) @binding(0) var<uniform> gi: GiParameters;
@group(3) @binding(1) var<storage, read> giProbes: array<GiProbe>;

fn diffuseGiIrradiance(position: vec3f, normal: vec3f) -> vec3f {
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
        let probe = giProbes[index];
        if (probe.state.x == 0u || probe.state.y != gi.state.x) { continue; }
        let probePosition = gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz;
        let delta = biased - probePosition;
        let distance = length(delta);
        let direction = delta / max(distance, 0.00001);
        let moment = probe.moments[giFace(direction)];
        if (moment.z == 0.0) { continue; }
        let variance = max(moment.y - moment.x * moment.x, 0.0001);
        let beyond = max(0.0, distance - moment.x);
        let visibility = variance / (variance + beyond * beyond);
        let trilinear = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        let facing = max(0.05, (1.0 - dot(direction, normal)) * 0.5);
        let weight = trilinear.x * trilinear.y * trilinear.z * facing * facing * visibility * visibility * visibility;
        var irradiance = vec3f(0.0);
        for (var coefficient = 0u; coefficient < 9u; coefficient++) {
            var convolution = 0.785398163;
            if (coefficient == 0u) { convolution = 3.14159265; }
            else if (coefficient < 4u) { convolution = 2.094395102; }
            irradiance += probe.coefficients[coefficient].xyz * basis[coefficient] * convolution;
        }
        sum += max(irradiance, vec3f(0.0)) * weight;
        total += weight;
    }
    return sum / max(total, 0.00001);
}
