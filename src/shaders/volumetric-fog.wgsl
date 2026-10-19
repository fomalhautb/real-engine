struct FogVolume {
  centerKind: vec4f,
  extent: vec4f,
  optics: vec4f,
  albedo: vec4f,
  emission: vec4f,
}
@group(0) @binding(1) var<storage, read> volumes: array<FogVolume>;
@group(0) @binding(2) var<storage, read_write> sources: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> integrated: array<vec4f>;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
// FOG_RESOLVED_SOURCE

fn phase(cosTheta: f32, g: f32) -> f32 {
  let denominator = max(1.0 + g * g - 2.0 * g * cosTheta, 0.000001);
  return (1.0 - g * g) / (12.5663706144 * denominator * sqrt(denominator));
}
fn visibility(world: vec3f) -> f32 {
  let clip = frame.lightMatrix * vec4f(world, 1.0);
  let p = clip.xyz / clip.w;
  let uv = p.xy * vec2f(0.5, -0.5) + vec2f(0.5);
  // Same finite directional map as surfaces. Outside its coverage is unshadowed.
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z < 0.0 || p.z > 1.0) { return 1.0; }
  return textureSampleCompareLevel(shadowMap, shadowSampler, uv, p.z - frame.sunDirection.w);
}
fn mediumSource(optics: vec2f, albedo: vec3f, emission: vec3f, incident: vec3f, cosine: f32) -> vec4f {
  return vec4f(emission + optics.x * albedo * phase(cosine, optics.y) * incident, optics.x);
}
@compute @workgroup_size(4, 4, 4)
fn inject(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= frame.grid.xyz)) { return; }
  let ray = fogRay((vec2f(id.xy) + 0.5) / vec2f(frame.grid.xy));
  let sampleRay = fogRay((vec2f(id.xy) + 0.5 + frame.reserved.xy) / vec2f(frame.grid.xy));
  let sampleDepth = mix(sliceDepth(id.z), sliceDepth(id.z + 1u), 0.5 + frame.reserved.z);
  let world = frame.eye.xyz + sampleRay * sampleDepth;
  let incident = frame.sunColor.xyz * visibility(world);
  // Incoming propagation is -sunDirection; outgoing to camera is -ray.
  let cosine = clamp(dot(normalize(ray), frame.sunDirection.xyz), -1.0, 1.0);
  var source = vec4f(0.0);
  if (frame.viewport.z > 0.0) {
    let density = exp(-frame.globalOptics.w * max(world.y - frame.globalOptics.z, 0.0));
    source = density * mediumSource(frame.globalOptics.xy, frame.globalAlbedo.xyz, frame.globalEmission.xyz, incident, cosine);
  }
  for (var i = 0u; i < frame.grid.w; i++) {
    let volume = volumes[i];
    let delta = world - volume.centerKind.xyz;
    var inside = all(abs(delta) <= volume.extent.xyz);
    if (volume.centerKind.w > 0.5) { inside = dot(delta, delta) <= volume.extent.x * volume.extent.x; }
    if (inside) { source += mediumSource(volume.optics.xy, volume.albedo.xyz, volume.emission.xyz, incident, cosine); }
  }
  sources[cellIndex(id.xy, id.z)] = source;
}
@compute @workgroup_size(8, 8)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= frame.grid.xy)) { return; }
  let rayLength = length(fogRay((vec2f(id.xy) + 0.5) / vec2f(frame.grid.xy)));
  var cumulative = vec4f(0.0, 0.0, 0.0, 1.0);
  for (var z = 0u; z < frame.grid.z; z++) {
    let index = cellIndex(id.xy, z);
    cumulative = fogOver(cumulative, fogSegment(resolvedSource(index), (sliceDepth(z + 1u) - sliceDepth(z)) * rayLength));
    integrated[index] = cumulative;
  }
}
