struct FogVolume {
  centerKind: vec4f,
  extent: vec4f,
  optics: vec4f,
  albedo: vec4f,
  emission: vec4f,
}
@group(0) @binding(1) var<storage, read> volumes: array<FogVolume>;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;

fn fogPhase(cosTheta: f32, g: f32) -> f32 {
  let denominator = max(1.0 + g * g - 2.0 * g * cosTheta, 0.000001);
  return (1.0 - g * g) / (12.5663706144 * denominator * sqrt(denominator));
}
fn fogVisibility(world: vec3f) -> f32 {
  let clip = frame.lightMatrix * vec4f(world, 1.0);
  let p = clip.xyz / clip.w;
  let uv = p.xy * vec2f(0.5, -0.5) + vec2f(0.5);
  // Same finite directional map as surfaces. Outside its coverage is unshadowed.
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z < 0.0 || p.z > 1.0) { return 1.0; }
  return textureSampleCompareLevel(shadowMap, shadowSampler, uv, p.z - frame.sunDirection.w);
}
fn fogMediumSource(optics: vec2f, albedo: vec3f, emission: vec3f, incident: vec3f, cosine: f32) -> vec4f {
  return vec4f(emission + optics.x * albedo * fogPhase(cosine, optics.y) * incident, optics.x);
}
fn fogWorldSource(world: vec3f, direction: vec3f) -> vec4f {
  let incident = frame.sunColor.xyz * fogVisibility(world);
  // Incoming propagation is -sunDirection; outgoing to the ray origin is -direction.
  let cosine = clamp(dot(normalize(direction), frame.sunDirection.xyz), -1.0, 1.0);
  var source = vec4f(0.0);
  if (frame.viewport.z > 0.0) {
    let density = exp(-frame.globalOptics.w * max(world.y - frame.globalOptics.z, 0.0));
    source = density * fogMediumSource(frame.globalOptics.xy, frame.globalAlbedo.xyz, frame.globalEmission.xyz, incident, cosine);
  }
  for (var i = 0u; i < frame.grid.w; i++) {
    let volume = volumes[i];
    let delta = world - volume.centerKind.xyz;
    var inside = all(abs(delta) <= volume.extent.xyz);
    if (volume.centerKind.w > 0.5) { inside = dot(delta, delta) <= volume.extent.x * volume.extent.x; }
    if (inside) { source += fogMediumSource(volume.optics.xy, volume.albedo.xyz, volume.emission.xyz, incident, cosine); }
  }
  return source;
}
