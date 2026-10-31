// Explicit diagnostic redraw only; attachments preserve the nearest visible
// fragment through the same depth test, avoiding fragment-storage write races.
struct GatherDiagnosticOutput {
    @location(0) values: vec4u,
    @location(1) rejections: vec4u,
}
@fragment
fn gatherDiagnosticFragment(input: Varyings, @builtin(front_facing) front: bool) -> GatherDiagnosticOutput {
    let sample = rasterSurface(input, front);
    if (!sample.covered) { discard; }
    if (MATERIAL_UNLIT) {
        return GatherDiagnosticOutput(vec4u(5u, 0u, 0u, 0u), vec4u(0x80808080u));
    }
    let result = finalGatherReconstruct(input.position.xy, input.world, sample.shadingNormal,
        sample.geometryNormal, input.surfaceIdentity);
    let world = diffuseGiSurfaceIrradiance(input.world, sample.shadingNormal, sample.geometryNormal);
    let luminance = vec3f(0.2126, 0.7152, 0.0722);
    var packed = vec4u(0u);
    for (var i = 0u; i < 16u; i++) { packed[i / 4u] |= result.rejections[i] << ((i % 4u) * 8u); }
    return GatherDiagnosticOutput(vec4u(result.branch | (result.primaryAccepted << 8u) | (result.ringAccepted << 16u)
        | (result.supplementalAccepted << 24u),
        bitcast<u32>(result.total), bitcast<u32>(dot(result.irradiance, luminance)), bitcast<u32>(dot(world, luminance))), packed);
}

// A second explicit redraw reuses the same 32-byte attachments. This records
// the real fragment origin and normals for independent irradiance integration;
// it does not substitute the camera ray's potentially different query surface.
@fragment
fn gatherDiagnosticSurfaceFragment(input: Varyings, @builtin(front_facing) front: bool) -> GatherDiagnosticOutput {
    let sample = rasterSurface(input, front);
    if (!sample.covered) { discard; }
    return GatherDiagnosticOutput(vec4u(bitcast<vec3u>(input.world), input.surfaceIdentity.x),
        vec4u(surfacePackSnorm2x16(surfaceOctEncode(sample.shadingNormal)),
            surfacePackSnorm2x16(surfaceOctEncode(sample.geometryNormal)), input.surfaceIdentity.y, input.representationToken));
}
