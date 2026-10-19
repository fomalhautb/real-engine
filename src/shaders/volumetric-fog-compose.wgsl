@group(0) @binding(2) var<storage, read> sources: array<vec4f>;
@group(0) @binding(3) var<storage, read> integrated: array<vec4f>;
@group(0) @binding(4) var sceneDepth: texture_depth_2d;
@group(0) @binding(5) var sceneColor: texture_2d<f32>;

@vertex fn vertexMain(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vertex << 1u) & 2u), f32(vertex & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn sampleColumn(xy: vec2i, depth: f32) -> vec4f {
  let cell = vec2u(clamp(xy, vec2i(0), vec2i(frame.grid.xy) - 1));
  let z = min(u32(sqrt(clamp(depth / frame.eye.w, 0.0, 1.0)) * f32(frame.grid.z)), frame.grid.z - 1u);
  var front = vec4f(0.0, 0.0, 0.0, 1.0);
  if (z > 0u) { front = integrated[cellIndex(cell, z - 1u)]; }
  let rayLength = length(fogRay((vec2f(cell) + 0.5) / vec2f(frame.grid.xy)));
  let partial = fogSegment(sources[cellIndex(cell, z)], max(0.0, depth - sliceDepth(z)) * rayLength);
  return fogOver(front, partial);
}
@fragment fn fragmentMain(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let surface = textureLoad(sceneColor, vec2i(pixel.xy), 0);
  if (frame.forward.w < 0.5) { return surface; }
  let depth = textureLoad(sceneDepth, vec2i(pixel.xy), 0);
  var viewDepth = frame.eye.w;
  if (depth < 1.0) { viewDepth = min(frame.eye.w, frame.right.w * frame.up.w / (frame.up.w - depth * (frame.up.w - frame.right.w))); }
  let xy = pixel.xy / frame.viewport.xy * vec2f(frame.grid.xy) - 0.5;
  let base = vec2i(floor(xy));
  let fraction = fract(xy);
  // All four columns stop at this full-resolution surface depth. No low-res
  // surface depth participates in filtering, so background fog cannot bleed over it.
  let a = mix(sampleColumn(base, viewDepth), sampleColumn(base + vec2i(1, 0), viewDepth), fraction.x);
  let b = mix(sampleColumn(base + vec2i(0, 1), viewDepth), sampleColumn(base + vec2i(1, 1), viewDepth), fraction.x);
  let medium = mix(a, b, fraction.y);
  return vec4f(min(medium.xyz + medium.w * surface.xyz, vec3f(65504.0)), surface.w);
}
