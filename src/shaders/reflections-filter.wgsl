// Optional edge-aware à-trous filter over the anchor grid. It runs after the
// temporal pass on the resolved radiance; the history keeps the unfiltered
// value. Each level reads 3x3 taps at a step of 1, 2, 4, ... cells, weighted
// by the B1 spline and reflectionGuideWeight, whose radius is a fraction of the
// center's lobe footprint: a mirror-like anchor (radius below one cell) keeps
// its own value exactly. The last level writes the radiance surfaces read.
struct ReflectionFilterLevel { step: u32, last: u32, reserved0: u32, reserved1: u32 }
@group(0) @binding(13) var<storage, read_write> filterGuides: array<ReflectionGuide>;
@group(0) @binding(14) var<storage, read> filterInput: array<vec4f>;
@group(0) @binding(15) var<storage, read_write> filterOutput: array<vec4f>;
@group(0) @binding(16) var<uniform> filterLevel: ReflectionFilterLevel;

@compute @workgroup_size(64)
fn filterPrepare(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y) { return; }
    let guide = reflectionAnchorGuide(id.x, reflection.spatial.x);
    let radiance = anchors[id.x].radiance;
    filterGuides[id.x] = guide;
    filterOutput[id.x] = radiance;
    anchors[id.x].unfiltered = vec4f(radiance.xyz, max(guide.position.w, 0.0));
}

@compute @workgroup_size(64)
fn filterLevelPass(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= reflection.settings.y) { return; }
    let center = filterGuides[id.x];
    var result = filterInput[id.x];
    if (center.position.w > 0.0) {
        let cell = vec2i(i32(id.x % reflection.viewport.z), i32(id.x / reflection.viewport.z));
        let step = i32(filterLevel.step);
        var sum = vec3f(0.0); var total = 0.0;
        for (var tap = 0u; tap < 9u; tap++) {
            let offset = vec2i(i32(tap % 3u) - 1, i32(tap / 3u) - 1);
            let coordinate = cell + offset * step;
            if (any(coordinate < vec2i(0)) || any(coordinate >= vec2i(reflection.viewport.zw))) { continue; }
            let index = u32(coordinate.y) * reflection.viewport.z + u32(coordinate.x);
            let spline = select(0.25, 0.5, offset.x == 0) * select(0.25, 0.5, offset.y == 0);
            let weight = spline * reflectionGuideWeight(center, filterGuides[index]);
            if (weight <= 0.0) { continue; }
            sum += filterInput[index].xyz * weight; total += weight;
        }
        // The center tap always contributes, so total > 0.
        result = vec4f(sum / total, result.w);
    }
    if (filterLevel.last != 0u) { anchors[id.x].radiance = result; } else { filterOutput[id.x] = result; }
}
