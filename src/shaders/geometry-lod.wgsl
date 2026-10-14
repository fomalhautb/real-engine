struct View {
    viewProjection: mat4x4f,
    eye: vec4f,
    // projectionScale, near, maxErrorPixels, frustumCulling
    lod: vec4f,
}
struct Instance {
    model: mat4x4f, previousModel: mat4x4f, normalMatrix: mat4x4f,
    color: vec4f, material: vec4f,
}
struct Cluster {
    minimum: vec4f, maximum: vec4f,
    // source index offset/count, containing group, refined group
    data: vec4u,
    // packed vertex base/count, level, stable cluster ID
    extra: vec4u,
}
struct Group { minimumError: vec4f, maximumTerminal: vec4f }
struct Parameters {
    clusterCount: u32, clusterVertexCount: u32, sourceVertexCount: u32, outputCapacity: u32,
    instanceCount: u32, padding0: u32, padding1: u32, padding2: u32,
}
struct Arguments {
    indexCount: atomic<u32>, instanceCount: u32, firstIndex: u32, baseVertex: i32, firstInstance: u32,
    selectedCount: atomic<u32>, overflow: atomic<u32>, padding: u32,
}
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> groups: array<Group>;
@group(0) @binding(3) var<storage, read> sourceIndices: array<u32>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<uniform> params: Parameters;
@group(0) @binding(6) var<storage, read_write> outputIndices: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: Arguments;
@group(0) @binding(8) var<storage, read_write> selected: array<u32>;

struct WorldBounds { center: vec3f, extent: vec3f }
fn worldBounds(minimum: vec3f, maximum: vec3f, model: mat4x4f) -> WorldBounds {
    let center = (minimum + maximum) * 0.5;
    let extent = (maximum - minimum) * 0.5;
    var result: WorldBounds;
    result.center = (model * vec4f(center, 1.0)).xyz;
    result.extent = abs(model[0].xyz) * extent.x + abs(model[1].xyz) * extent.y + abs(model[2].xyz) * extent.z;
    // Keep roundoff from changing a shared boundary into a false exclusion.
    result.extent += (abs(result.center) + result.extent + vec3f(1.0)) * 0.000002;
    return result;
}
// sqrt(||M||1 * ||M||infinity) bounds the largest singular value, including shear.
fn scaleBound(model: mat4x4f) -> f32 {
    let x = abs(model[0].xyz); let y = abs(model[1].xyz); let z = abs(model[2].xyz);
    let columns = max(max(x.x + x.y + x.z, y.x + y.y + y.z), z.x + z.y + z.z);
    let rows = x + y + z;
    return sqrt(columns * max(max(rows.x, rows.y), rows.z));
}
fn coarsen(id: u32, model: mat4x4f, scale: f32) -> bool {
    let group = groups[id];
    if (group.maximumTerminal.w > 0.5) { return false; }
    let bounds = worldBounds(group.minimumError.xyz, group.maximumTerminal.xyz, model);
    let distance = length(max(abs(view.eye.xyz - bounds.center) - bounds.extent, vec3f(0.0)));
    let pixels = group.minimumError.w * scale * view.lod.x / max(view.lod.y, distance);
    return pixels <= view.lod.z;
}
fn visible(cluster: Cluster, model: mat4x4f) -> bool {
    if (view.lod.w < 0.5) { return true; }
    let bounds = worldBounds(cluster.minimum.xyz, cluster.maximum.xyz, model);
    let center = view.viewProjection * vec4f(bounds.center, 1.0);
    let radius = abs(view.viewProjection[0]) * bounds.extent.x
        + abs(view.viewProjection[1]) * bounds.extent.y + abs(view.viewProjection[2]) * bounds.extent.z;
    return !(center.x + center.w + radius.x + radius.w < 0.0
        || center.w - center.x + radius.w + radius.x < 0.0
        || center.y + center.w + radius.y + radius.w < 0.0
        || center.w - center.y + radius.w + radius.y < 0.0
        || center.z + radius.z < 0.0
        || center.w - center.z + radius.w + radius.z < 0.0);
}
@compute @workgroup_size(64)
fn selectClusters(@builtin(global_invocation_id) invocation: vec3u) {
    let index = invocation.x;
    if (index >= params.clusterCount * params.instanceCount) { return; }
    if (index == 0u) { args.instanceCount = 1u; }
    selected[index] = 0u;
    let instanceIndex = index / params.clusterCount;
    let cluster = clusters[index % params.clusterCount];
    let model = instances[instanceIndex].model;
    let scale = scaleBound(model);
    if (coarsen(cluster.data.z, model, scale)) { return; }
    if (cluster.data.w != 0xffffffffu && !coarsen(cluster.data.w, model, scale)) { return; }
    if (!visible(cluster, model)) { return; }
    let count = cluster.data.y;
    let offset = atomicAdd(&args.indexCount, count);
    if (offset > params.outputCapacity || count > params.outputCapacity - min(offset, params.outputCapacity)) {
        atomicStore(&args.overflow, 1u);
        return;
    }
    selected[index] = 1u;
    atomicAdd(&args.selectedCount, 1u);
    let instanceBase = instanceIndex * params.clusterVertexCount;
    for (var i = 0u; i < count; i++) {
        outputIndices[offset + i] = sourceIndices[cluster.data.x + i] + instanceBase;
    }
}
// Separate dispatch/pass: no reliance on cross-workgroup synchronization.
@compute @workgroup_size(1)
fn finalize() {
    if (atomicLoad(&args.overflow) != 0u) {
        atomicStore(&args.indexCount, 0u);
        args.instanceCount = 0u;
    }
}
