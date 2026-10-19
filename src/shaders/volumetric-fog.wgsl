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
