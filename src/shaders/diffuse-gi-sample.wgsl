// GI_COMMON
@group(3) @binding(0) var<uniform> gi: GiParameters;
@group(3) @binding(1) var<storage, read> giProbes: array<GiProbe>;

// GI_INTERPOLATE

fn diffuseGiIrradiance(position: vec3f, normal: vec3f) -> vec3f {
    return giSampleIrradiance(position, normal, gi.state.x);
}
fn diffuseGiSurfaceIrradiance(position: vec3f, normal: vec3f, geometricNormal: vec3f) -> vec3f {
    return giSampleSurfaceIrradiance(position, normal, geometricNormal, gi.state.x);
}
