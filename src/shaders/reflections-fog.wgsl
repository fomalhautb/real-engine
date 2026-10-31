fn reflectionApplyFog(origin: vec3f, direction: vec3f, distance: f32, incoming: vec3f) -> vec3f {
  if (frame.forward.w < 0.5 || distance <= 0.0) { return incoming; }
  let steps = reflection.state.z;
  let segmentLength = distance / f32(steps);
  let ray = normalize(direction);
  var cumulative = vec4f(0.0, 0.0, 0.0, 1.0);
  for (var step = 0u; step < steps; step++) {
    let world = origin + ray * ((f32(step) + 0.5) * segmentLength);
    // Same world media and air lighting as the camera grid, sampled on this segment.
    cumulative = fogOver(cumulative, fogSegment(fogWorldSource(world, ray) + vec4f(fogAirSource(world, ray), 0.0), segmentLength));
  }
  return cumulative.xyz + cumulative.w * incoming;
}
