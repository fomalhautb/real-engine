// Bottom-up refit of the fixed-topology TLAS over pooled (batch, instance)
// leaves. Each depth is a separate dispatch so children are complete before
// parents read them. Inactive or unused leaves are explicitly invalid.
struct RefitParams {
    level: vec4u, // first word of this depth's node list, node count, Morton order word offset, 0
}
@group(0) @binding(0) var<uniform> refit: RefitParams;
// TLAS nodes [left|order index, right|1, escape, leaf], then order, then depth lists.
@group(0) @binding(1) var<storage, read> topology: array<u32>;
@group(0) @binding(2) var<storage, read> leaves: array<u32>;
@group(0) @binding(3) var<storage, read_write> nodes: array<vec4u>;

@compute @workgroup_size(64)
fn refitTlas(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= refit.level.y) { return; }
    let node = topology[refit.level.x + id.x];
    let t = node * 4u;
    var low = vec3f(3.402823466e+38);
    var high = -low;
    var info = 0u;
    if (topology[t + 3u] == 1u) {
        let leaf = topology[refit.level.z + topology[t]];
        if (leaf != 0xffffffffu && leaves[leaf * 8u + 7u] != 0u) {
            let root = leaves[leaf * 8u + 4u] * 2u;
            low = bitcast<vec3f>(nodes[root].xyz); high = bitcast<vec3f>(nodes[root + 1u].xyz);
            info = 0x80000000u | (leaf + 1u);
        }
    } else {
        for (var side = 0u; side < 2u; side++) {
            let child = topology[t + side] * 2u;
            if ((nodes[child + 1u].w & 0x80000000u) != 0u) {
                low = min(low, bitcast<vec3f>(nodes[child].xyz));
                high = max(high, bitcast<vec3f>(nodes[child + 1u].xyz));
                info = 0x80000000u;
            }
        }
    }
    if (info == 0u) { low = vec3f(0.0); high = vec3f(0.0); }
    nodes[node * 2u] = vec4u(bitcast<vec3u>(low), topology[t + 2u]);
    nodes[node * 2u + 1u] = vec4u(bitcast<vec3u>(high), info);
}
