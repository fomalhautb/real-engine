// Candidate rank rounds first give every nonempty tile one representative,
// then share the remaining slots. A stable tile permutation avoids a top-left
// bias when the final round is partial. No atomic allocation or CPU readback.
fn coverageTile(order: u32) -> u32 { return (order * gather.coverage.z) % gather.coverage.y; }
fn coverageBlockBase() -> u32 { return gather.coverage.x * 8u; }
fn coverageTotalBase() -> u32 { return coverageBlockBase() + ((gather.coverage.x + 127u) / 128u) * 8u; }
var<workgroup> coverageLow: array<vec4u, 128>;
var<workgroup> coverageHigh: array<vec4u, 128>;

// Inclusive workgroup scan; every lane participates, including zero padding.
fn coverageScan(lane: u32) {
    workgroupBarrier();
    for (var offset = 1u; offset < 128u; offset *= 2u) {
        var low = vec4u(0u); var high = vec4u(0u);
        if (lane >= offset) { low = coverageLow[lane - offset]; high = coverageHigh[lane - offset]; }
        workgroupBarrier();
        coverageLow[lane] += low; coverageHigh[lane] += high;
        workgroupBarrier();
    }
}

@compute @workgroup_size(64)
fn clearSupplementalReceivers(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.raster.w) { return; }
    var hit: Hit;
    hit.versions = vec4u(gather.state.x, 0u, 0u, gather.emissive.z);
    rasterCameraHits[gather.viewport.z * gather.viewport.w + id.x] = hit;
}

@compute @workgroup_size(128)
fn scanSupplementalTiles(@builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
    var low = vec4u(0u); var high = vec4u(0u);
    if (id.x < gather.coverage.y) {
        let tile = coverageTile(id.x);
        for (var i = 0u; i < 4u; i++) {
            low[i] = select(0u, 1u, coverageCandidates[tile * 8u + i].identity.x != 0u);
            high[i] = select(0u, 1u, coverageCandidates[tile * 8u + i + 4u].identity.x != 0u);
        }
    }
    coverageLow[lane] = low; coverageHigh[lane] = high;
    coverageScan(lane);
    if (id.x < gather.coverage.y) {
        for (var i = 0u; i < 4u; i++) {
            coveragePrefix[id.x * 8u + i] = coverageLow[lane][i] - low[i];
            coveragePrefix[id.x * 8u + i + 4u] = coverageHigh[lane][i] - high[i];
        }
    }
    if (lane == 127u) {
        for (var i = 0u; i < 4u; i++) {
            coveragePrefix[coverageBlockBase() + group.x * 8u + i] = coverageLow[lane][i];
            coveragePrefix[coverageBlockBase() + group.x * 8u + i + 4u] = coverageHigh[lane][i];
        }
    }
}

@compute @workgroup_size(128)
fn scanSupplementalBlocks(@builtin(local_invocation_index) lane: u32) {
    let blocks = (gather.coverage.y + 127u) / 128u;
    var low = vec4u(0u); var high = vec4u(0u);
    if (lane < blocks) { for (var i = 0u; i < 4u; i++) {
        low[i] = coveragePrefix[coverageBlockBase() + lane * 8u + i];
        high[i] = coveragePrefix[coverageBlockBase() + lane * 8u + i + 4u];
    } }
    coverageLow[lane] = low; coverageHigh[lane] = high;
    coverageScan(lane);
    if (lane < blocks) { for (var i = 0u; i < 4u; i++) {
        coveragePrefix[coverageBlockBase() + lane * 8u + i] = coverageLow[lane][i] - low[i];
        coveragePrefix[coverageBlockBase() + lane * 8u + i + 4u] = coverageHigh[lane][i] - high[i];
    } }
    if (lane == 127u) { for (var i = 0u; i < 4u; i++) {
        coveragePrefix[coverageTotalBase() + i] = coverageLow[lane][i];
        coveragePrefix[coverageTotalBase() + i + 4u] = coverageHigh[lane][i];
    } }
}

@compute @workgroup_size(64)
fn scatterSupplementalReceivers(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gather.coverage.y) { return; }
    let tile = coverageTile(id.x);
    var roundBase = 0u;
    for (var rank = 0u; rank < 8u; rank++) {
        let candidate = coverageCandidates[tile * 8u + rank];
        let selected = roundBase + coveragePrefix[id.x * 8u + rank]
            + coveragePrefix[coverageBlockBase() + (id.x / 128u) * 8u + rank];
        var mapped = 0u;
        if (candidate.identity.x != 0u && selected < gather.raster.w) {
            let index = gather.viewport.z * gather.viewport.w + selected;
            rasterCameraHits[index] = candidate;
            coverageMap[gather.coverage.x * 8u + selected] = tile;
            mapped = index + 1u;
        }
        coverageMap[tile * 8u + rank] = mapped;
        roundBase += coveragePrefix[coverageTotalBase() + rank];
    }
}
