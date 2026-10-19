struct FogFrame {
  grid: vec4u,
  eye: vec4f,
  right: vec4f,
  up: vec4f,
  forward: vec4f,
  sunDirection: vec4f,
  sunColor: vec4f,
  globalOptics: vec4f,
  globalAlbedo: vec4f,
  globalEmission: vec4f,
  viewport: vec4f,
  reserved: vec4f,
  lightMatrix: mat4x4f,
  previousEye: vec4f,
  previousForward: vec4f,
  previousViewProjection: mat4x4f,
  temporal: vec4f,
  padding: vec4f,
}
@group(0) @binding(0) var<uniform> frame: FogFrame;

// Unnormalized ray: advancing by one unit changes camera-forward depth by one metre.
fn fogRay(uv: vec2f) -> vec3f {
  return frame.forward.xyz + frame.right.xyz * (uv.x * 2.0 - 1.0) + frame.up.xyz * (1.0 - uv.y * 2.0);
}
fn sliceDepth(z: u32) -> f32 {
  let t = f32(z) / f32(frame.grid.z);
  return frame.eye.w * t * t;
}
fn cellIndex(xy: vec2u, z: u32) -> u32 {
  return (z * frame.grid.y + xy.y) * frame.grid.x + xy.x;
}
// Analytic Beer-Lambert integration for the piecewise-constant cell. The series
// avoids subtracting almost equal floats in vacuum and optically thin segments.
fn fogSegment(source: vec4f, distance: f32) -> vec4f {
  let tau = source.w * distance;
  let transmission = exp(-tau);
  var integral = distance * (1.0 - tau * 0.5 + tau * tau / 6.0);
  if (tau >= 0.001) { integral = (1.0 - transmission) / source.w; }
  return vec4f(source.xyz * integral, transmission);
}
fn fogOver(front: vec4f, back: vec4f) -> vec4f {
  return vec4f(front.xyz + front.w * back.xyz, front.w * back.w);
}
