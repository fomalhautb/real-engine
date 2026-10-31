// Low-frequency light in the air from the world GI probe field.
// Requires the host to declare: frame (FogFrame), volumes, gi (GiParameters),
// giProbes, giBasis and giVisibility. The probes store incoming radiance SH9
// (miss radiance is the GI sky, so geometric sky occlusion is part of the field);
// this is never a surface's outgoing color or normal-dependent irradiance.
// frame.padding: y = GI field enabled, z = author sky visibility beyond the field.

// Per-channel scattering moments: sum sigma_s, sum sigma_s * g, sum sigma_s * g^2.
// HG's zonal SH coefficients are g^l, so these three sums express every medium's
// phase convolution of the shared L2 radiance with one probe interpolation.
fn fogAirScattering(world: vec3f) -> array<vec3f, 3> {
  var moments = array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  if (frame.viewport.z > 0.0) {
    let density = exp(-frame.globalOptics.w * max(world.y - frame.globalOptics.z, 0.0));
    let sigma = density * frame.globalOptics.x * frame.globalAlbedo.xyz;
    let g = frame.globalOptics.y;
    moments[0] += sigma; moments[1] += sigma * g; moments[2] += sigma * g * g;
  }
  for (var i = 0u; i < frame.grid.w; i++) {
    let volume = volumes[i];
    let delta = world - volume.centerKind.xyz;
    var inside = all(abs(delta) <= volume.extent.xyz);
    if (volume.centerKind.w > 0.5) { inside = dot(delta, delta) <= volume.extent.x * volume.extent.x; }
    if (inside) {
      let sigma = volume.optics.x * volume.albedo.xyz;
      let g = volume.optics.y;
      moments[0] += sigma; moments[1] += sigma * g; moments[2] += sigma * g * g;
    }
  }
  return moments;
}

// Visibility-weighted trilinear probe interpolation without a surface normal. The
// same moment confidence and low-support curve as surfaces keep a probe behind a
// wall from lighting air on the other side.
fn fogAirField(world: vec3f, grid: vec3f, last: vec3f, view: vec3f, moments: array<vec3f, 3>) -> vec3f {
  let clamped = clamp(grid, vec3f(0.0), last);
  let lower = min(vec3u(floor(clamped)), gi.counts.xyz - vec3u(2u));
  let fraction = clamped - vec3f(lower);
  var sh = array<vec3f, 9>();
  var maximum = vec3f(0.0);
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
    if (giProbes[index].state.x == 0u || giProbes[index].state.y != gi.state.x) { continue; }
    let delta = world - (gi.origin.xyz + vec3f(coordinate) * gi.spacing.xyz);
    let distance = length(delta);
    let visibility = giVisibility(index, delta / max(distance, 0.00001), distance);
    let visible = visibility * visibility * visibility;
    var confidence = visible;
    if (confidence < gi.sky.w) {
      let fractionOfThreshold = confidence / gi.sky.w;
      confidence *= fractionOfThreshold * fractionOfThreshold;
    }
    let weight = spatialWeight * confidence;
    if (weight <= 0.0) { continue; }
    visibleSupport = max(visibleSupport, spatialWeight * visible);
    for (var coefficient = 0u; coefficient < 9u; coefficient++) {
      sh[coefficient] += giProbes[index].coefficients[coefficient].xyz * weight;
    }
    maximum = max(maximum, vec3f(giProbes[index].coefficients[0].w,
      giProbes[index].coefficients[1].w, giProbes[index].coefficients[2].w));
    total += weight;
  }
  if (total <= 0.0) { return vec3f(0.0); }
  let basis = giBasis(view);
  let band0 = sh[0] * basis[0];
  let band1 = sh[1] * basis[1] + sh[2] * basis[2] + sh[3] * basis[3];
  let band2 = sh[4] * basis[4] + sh[5] * basis[5] + sh[6] * basis[6] + sh[7] * basis[7] + sh[8] * basis[8];
  let radiance = (moments[0] * band0 + moments[1] * band1 + moments[2] * band2) / total;
  let support = visibleSupport / max(peakSpatialWeight, 0.000000000001);
  var supportWeight = 1.0;
  if (support < gi.sky.w) {
    let fractionOfThreshold = support / gi.sky.w;
    supportWeight = fractionOfThreshold * fractionOfThreshold;
  }
  // The phase integral of nonnegative radiance never exceeds its sampled maximum.
  return clamp(radiance, vec3f(0.0), moments[0] * maximum) * supportWeight;
}

// In-scattered source radiance per metre from GI and sky; add to the sun source.
fn fogAirSource(world: vec3f, direction: vec3f) -> vec3f {
  if (gi.origin.w == 0.0) { return vec3f(0.0); }
  let moments = fogAirScattering(world);
  if (all(moments[0] <= vec3f(0.0))) { return vec3f(0.0); }
  let grid = (world - gi.origin.xyz) / gi.spacing.xyz;
  let last = vec3f(gi.counts.xyz - vec3u(1u));
  let outside = max(max(-grid, grid - last), vec3f(0.0));
  // Field support fades over the same half-cell fringe that surfaces may sample.
  let fieldWeight = clamp(1.0 - 2.0 * max(outside.x, max(outside.y, outside.z)), 0.0, 1.0);
  var field = vec3f(0.0);
  if (fieldWeight > 0.0 && frame.padding.y > 0.5) {
    field = fogAirField(world, grid, last, normalize(direction), moments);
  }
  // A constant sky is isotropic under any normalized phase function.
  return fieldWeight * field + (1.0 - fieldWeight) * moments[0] * gi.sky.xyz * frame.padding.z;
}
