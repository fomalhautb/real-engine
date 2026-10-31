// Opt-in local history rejection for changed directional-shadow casters.
//
// Invariant: every sample contributing to a cell's history lies in the world-space
// ball (unjittered cell centre, stored support radius). A fresh cell holds only its
// jittered current sample, which stays inside the cell's eight corners. A blend adds
// the distance from this centre to each previous tap centre to that tap's radius
// (triangle inequality); radii are rounded up when stored as f16.
//
// The CPU uploads, for each changed caster, its pre- and post-change light-NDC
// rectangles expanded by the 2x2 comparison footprint and its nearest light depth.
// Sun visibility can only change inside such a rectangle at or behind that depth.
// A cell keeps history only if its ball misses every region, so by induction no
// retained sample has seen its sun visibility change.
struct FogOccluderRegions {
  header: vec4f, // x region count, y regions merged after capacity
  regions: array<vec4f, 64>, // [2i] light NDC min.xy/max.xy, [2i + 1].x nearest light depth
}
@group(0) @binding(6) var<uniform> occluders: FogOccluderRegions;

// Taps below this weight are dropped and the rest renormalized, so float noise in
// a static reprojection cannot widen the support through near-zero weights.
const FOG_HISTORY_TAP_THRESHOLD = 0.001;
// Saturated radius: such a cell is rejected by any caster change.
const FOG_HISTORY_UNBOUNDED = 60000.0;

// History state: low 16 bits frame count, high 16 bits f16 support radius in metres.
fn fogHistoryCount(state: u32) -> u32 { return state & 0xffffu; }
fn fogHistorySupport(state: u32) -> f32 { return unpack2x16float(state).y; }
fn fogPackHistory(count: u32, support: f32) -> u32 {
  // A floor above the binary16 normal range keeps the round-up exact.
  let radius = clamp(support, 0.0001, FOG_HISTORY_UNBOUNDED);
  var bits = pack2x16float(vec2f(radius, 0.0)) & 0xffffu;
  // Round up to the next binary16 value so the stored ball still contains the support.
  if (unpack2x16float(bits).x < radius) { bits += 1u; }
  return min(count, 0xffffu) | (bits << 16u);
}
fn fogHistoryNormalize(history: vec4f, weightSum: f32) -> vec4f { return history / weightSum; }

// Jittered samples stay inside the cell; the farthest corner bounds their distance.
fn fogHistoryCellRadius(id: vec3u, center: vec3f) -> f32 {
  var radius = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let uv = (vec2f(id.xy) + vec2f(f32(corner & 1u), f32((corner >> 1u) & 1u))) / vec2f(frame.grid.xy);
    let position = frame.eye.xyz + fogRay(uv) * sliceDepth(id.z + ((corner >> 2u) & 1u));
    radius = max(radius, distance(position, center));
  }
  return radius;
}

// Distance bound from this cell centre to a previous cell centre. In the previous
// view the centre sits at the reprojected uv and planar depth; the tap centre uses the
// same symmetric projection, whose tangents are the inverse row norms of the previous
// view-projection. An unchanged view (temporal.w) maps a cell exactly onto itself.
fn fogHistoryTapReach(id: vec3u, cell: vec3u, uv: vec2f, depth: f32) -> f32 {
  if (frame.temporal.w > 0.5 && all(cell == id)) { return 0.0; }
  let m = frame.previousViewProjection;
  let tangent = vec2f(1.0 / length(vec3f(m[0].x, m[1].x, m[2].x)), 1.0 / length(vec3f(m[0].y, m[1].y, m[2].y)));
  let delta = (vec2f(cell.xy) + 0.5) / vec2f(frame.grid.xy) - uv;
  let s = f32(cell.z) + 0.5;
  let tapDepth = frame.eye.w * (s * s + 0.25) / f32(frame.grid.z * frame.grid.z);
  let lateral = 2.0 * tapDepth * length(delta * tangent);
  let slant = length(vec3f(1.0, (uv.x * 2.0 - 1.0) * tangent.x, (1.0 - uv.y * 2.0) * tangent.y));
  // Relative and absolute allowances cover f32 rounding of both centres.
  return (lateral + slant * abs(tapDepth - depth)) * 1.001 + 0.00001 * (tapDepth + depth);
}

fn fogHistoryRejected(world: vec3f, radius: f32) -> bool {
  let count = min(u32(occluders.header.x), 32u);
  if (count == 0u) { return false; }
  if (radius >= FOG_HISTORY_UNBOUNDED) { return true; }
  // The sun matrix is affine (validated on the CPU), so the ball maps to a light box.
  let m = frame.lightMatrix;
  let center = (m * vec4f(world, 1.0)).xyz;
  let extent = radius * vec3f(length(vec3f(m[0].x, m[1].x, m[2].x)), length(vec3f(m[0].y, m[1].y, m[2].y)),
    length(vec3f(m[0].z, m[1].z, m[2].z)));
  let lower = center - extent;
  let upper = center + extent;
  for (var i = 0u; i < count; i++) {
    let rectangle = occluders.regions[i * 2u];
    if (upper.x >= rectangle.x && lower.x <= rectangle.z && upper.y >= rectangle.y && lower.y <= rectangle.w
      && upper.z >= occluders.regions[i * 2u + 1u].x) { return true; }
  }
  return false;
}
