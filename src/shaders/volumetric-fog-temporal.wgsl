@group(0) @binding(1) var<storage, read> currentSource: array<vec4f>;
@group(0) @binding(2) var<storage, read> previousSource: array<vec4f>;
@group(0) @binding(3) var<storage, read> previousCount: array<u32>;
@group(0) @binding(4) var<storage, read_write> resolved: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> resolvedCount: array<u32>;
// FOG_OCCLUDER_REJECTION

fn phaseShape(cosine: f32, g: f32) -> f32 {
  let denominator = 1.0 + g * g - 2.0 * g * cosine;
  return 1.0 / (denominator * sqrt(denominator));
}
// Source radiance includes a view-dependent HG factor. Do not keep an old
// direction when its change could consume the entire bounded history error.
fn directionCompatible(world: vec3f, direction: vec3f) -> bool {
  let g = frame.temporal.z;
  if (g == 0.0) { return true; }
  let delta = world - frame.previousEye.xyz;
  if (dot(delta, delta) < 0.00000001) { return false; }
  let oldCosine = clamp(dot(normalize(delta), frame.sunDirection.xyz), -1.0, 1.0);
  let cosine = clamp(dot(direction, frame.sunDirection.xyz), -1.0, 1.0);
  let tolerance = 1.0 + 0.02 / frame.temporal.y;
  for (var side = 0u; side < 2u; side++) {
    let signedG = select(-g, g, side == 1u);
    let ratio = phaseShape(cosine, signedG) / phaseShape(oldCosine, signedG);
    if (ratio > tolerance || ratio < 1.0 / tolerance) { return false; }
  }
  return true;
}

@compute @workgroup_size(4, 4, 4)
fn temporalMain(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= frame.grid.xyz)) { return; }
  let index = cellIndex(id.xy, id.z);
  let value = currentSource[index];
  var result = value;
  var count = 1u;
  let ray = fogRay((vec2f(id.xy) + 0.5) / vec2f(frame.grid.xy));
  // History represents cell averages, so reproject the unjittered cell centre.
  let world = frame.eye.xyz + ray * ((sliceDepth(id.z) + sliceDepth(id.z + 1u)) * 0.5);
  let cellRadius = fogHistoryCellRadius(id, world);
  var support = cellRadius;
  if (frame.temporal.x > 0.5) {
    let clip = frame.previousViewProjection * vec4f(world, 1.0);
    let oldDepth = dot(world - frame.previousEye.xyz, frame.previousForward.xyz);
    if (clip.w > 0.0 && oldDepth >= 0.0 && oldDepth <= frame.eye.w && directionCompatible(world, normalize(ray))) {
      let uv = clip.xy / clip.w * vec2f(0.5, -0.5) + 0.5;
      if (all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
        let depthSquare = oldDepth / frame.eye.w * f32(frame.grid.z * frame.grid.z);
        let coordinate = vec3f(uv * vec2f(frame.grid.xy) - 0.5, sqrt(max(depthSquare - 0.25, 0.0)) - 0.5);
        let base = vec3i(floor(coordinate));
        let fraction = fract(coordinate);
        var history = vec4f(0.0);
        var weightSum = 0.0;
        var historyCount = u32(frame.temporal.y);
        var historySupport = 0.0;
        for (var corner = 0u; corner < 8u; corner++) {
          let offset = vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
          let weights = mix(vec3f(1.0) - fraction, fraction, vec3f(offset));
          let weight = weights.x * weights.y * weights.z;
          if (weight > FOG_HISTORY_TAP_THRESHOLD) {
            let cell = vec3u(clamp(base + vec3i(offset), vec3i(0), vec3i(frame.grid.xyz) - 1));
            let previousIndex = cellIndex(cell.xy, cell.z);
            let state = previousCount[previousIndex];
            history += weight * previousSource[previousIndex];
            weightSum += weight;
            historyCount = min(historyCount, fogHistoryCount(state));
            historySupport = max(historySupport, fogHistoryTapReach(id, cell, uv, oldDepth) + fogHistorySupport(state));
          }
        }
        let radius = max(cellRadius, historySupport);
        // Local rejection: a changed caster discards only histories whose support can see it.
        if (historyCount > 0u && !fogHistoryRejected(world, radius)) {
          count = min(historyCount + 1u, u32(frame.temporal.y));
          result = mix(fogHistoryNormalize(history, weightSum), value, 1.0 / f32(count));
          support = radius;
        }
      }
    }
  }
  resolved[index] = result;
  resolvedCount[index] = fogPackHistory(count, support);
}
