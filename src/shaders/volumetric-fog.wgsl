@group(0) @binding(2) var<storage, read_write> sources: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> integrated: array<vec4f>;
// FOG_RESOLVED_SOURCE

@compute @workgroup_size(4, 4, 4)
fn inject(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= frame.grid.xyz)) { return; }
  let ray = fogRay((vec2f(id.xy) + 0.5) / vec2f(frame.grid.xy));
  let sampleRay = fogRay((vec2f(id.xy) + 0.5 + frame.reserved.xy) / vec2f(frame.grid.xy));
  let sampleDepth = mix(sliceDepth(id.z), sliceDepth(id.z + 1u), 0.5 + frame.reserved.z);
  let world = frame.eye.xyz + sampleRay * sampleDepth;
  sources[cellIndex(id.xy, id.z)] = fogWorldSource(world, ray);
}
// Air lighting is evaluated every frame at the unjittered cell centre and added after
// the temporal resolve, so fog history never holds GI or sky light: a changed GI field
// is seen on its first frame and needs no fog history invalidation. Cell-centre
// sampling suffices because the probe field is far coarser than a froxel.
@compute @workgroup_size(4, 4, 4)
fn air(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= frame.grid.xyz)) { return; }
  let index = cellIndex(id.xy, id.z);
  let ray = fogRay((vec2f(id.xy) + 0.5) / vec2f(frame.grid.xy));
  let world = frame.eye.xyz + ray * ((sliceDepth(id.z) + sliceDepth(id.z + 1u)) * 0.5);
  sources[index] = resolvedSource(index) + vec4f(fogAirSource(world, ray), 0.0);
}
fn integrationSource(index: u32) -> vec4f {
  // With air lighting, the air pass has already resolved this frame's source in place.
  if (FOG_AIR_PASS) { return sources[index]; }
  return resolvedSource(index);
}
@compute @workgroup_size(8, 8)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= frame.grid.xy)) { return; }
  let rayLength = length(fogRay((vec2f(id.xy) + 0.5) / vec2f(frame.grid.xy)));
  var cumulative = vec4f(0.0, 0.0, 0.0, 1.0);
  for (var z = 0u; z < frame.grid.z; z++) {
    let index = cellIndex(id.xy, z);
    cumulative = fogOver(cumulative, fogSegment(integrationSource(index), (sliceDepth(z + 1u) - sliceDepth(z)) * rayLength));
    integrated[index] = cumulative;
  }
}
