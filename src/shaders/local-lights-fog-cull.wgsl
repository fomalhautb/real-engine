// Bounded per-block light list for the view froxel grid: each 4×4×4 inject
// workgroup tests every enabled light's range sphere once against the world
// AABB of its froxel block (including temporal jitter), then all 64 froxels
// iterate only the surviving lights in ascending, deterministic order.
var<workgroup> fogLocalMask: array<atomic<u32>, LOCAL_LIGHT_MASK_WORDS>;

fn fogLocalCandidateWord(word: u32) -> u32 { return atomicLoad(&fogLocalMask[word]); }

// Must be called before any divergent return, by every invocation of the workgroup.
fn fogCullLocalLights(workgroup: vec3u, lane: u32) {
    if (lane < LOCAL_LIGHT_MASK_WORDS) { atomicStore(&fogLocalMask[lane], 0u); }
    workgroupBarrier();
    let low = workgroup * 4u;
    let high = min(low + vec3u(4u), frame.grid.xyz);
    let uv0 = vec2f(low.xy) / vec2f(frame.grid.xy);
    let uv1 = vec2f(high.xy) / vec2f(frame.grid.xy);
    let depth0 = sliceDepth(low.z);
    let depth1 = sliceDepth(high.z);
    var boxMin = vec3f(3.402823e38);
    var boxMax = vec3f(-3.402823e38);
    for (var corner = 0u; corner < 8u; corner++) {
        let uv = vec2f(select(uv0.x, uv1.x, (corner & 1u) != 0u), select(uv0.y, uv1.y, (corner & 2u) != 0u));
        let point = frame.eye.xyz + fogRay(uv) * select(depth0, depth1, (corner & 4u) != 0u);
        boxMin = min(boxMin, point);
        boxMax = max(boxMax, point);
    }
    let count = localLights.header.x;
    for (var base = 0u; base < count; base += 64u) {
        let index = base + lane;
        if (index < count) {
            let light = localLights.lights[index];
            let gap = clamp(light.position.xyz, boxMin, boxMax) - light.position.xyz;
            if (dot(gap, gap) < light.position.w * light.position.w) {
                atomicOr(&fogLocalMask[index / 32u], 1u << (index % 32u));
            }
        }
    }
    workgroupBarrier();
}
