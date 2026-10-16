struct Frame {
    viewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    lightViewProjection: mat4x4f,
    camera: vec4f,
    sunDirection: vec4f,
    sunColor: vec4f,
    debug: vec4u,
}
struct Instance {
    model: mat4x4f,
    previousModel: mat4x4f,
    normalMatrix: mat4x4f,
    color: vec4f,
    material: vec4f,
}
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
struct Parameters { vertexCount: u32, instanceCount: u32, jointCount: u32, padding: u32 }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<storage, read> vertices: array<Deformed>;
@group(1) @binding(1) var<storage, read> instances: array<Instance>;
@group(1) @binding(2) var<uniform> params: Parameters;
@group(1) @binding(3) var<storage, read> uvs: array<vec2f>;
@group(1) @binding(4) var<storage, read> clusterVertices: array<vec4u>;
struct MaterialParameters { values: array<vec4f, 16> }
@group(2) @binding(0) var<uniform> materialParameters: MaterialParameters;
@group(2) @binding(1) var materialImage0: texture_2d<f32>;
@group(2) @binding(2) var materialImage1: texture_2d<f32>;
@group(2) @binding(3) var materialImage2: texture_2d<f32>;
@group(2) @binding(4) var materialImage3: texture_2d<f32>;
@group(2) @binding(5) var materialSampler0: sampler;
@group(2) @binding(6) var materialSampler1: sampler;
@group(2) @binding(7) var materialSampler2: sampler;
@group(2) @binding(8) var materialSampler3: sampler;
// Raster uses implicit derivatives; world-space backends supply explicit-LOD helpers.
fn materialTexture0(uv: vec2f, lod: f32) -> vec4f { return textureSample(materialImage0, materialSampler0, uv); }
fn materialTexture1(uv: vec2f, lod: f32) -> vec4f { return textureSample(materialImage1, materialSampler1, uv); }
fn materialTexture2(uv: vec2f, lod: f32) -> vec4f { return textureSample(materialImage2, materialSampler2, uv); }
fn materialTexture3(uv: vec2f, lod: f32) -> vec4f { return textureSample(materialImage3, materialSampler3, uv); }
// MATERIAL_PROGRAM
// DIFFUSE_GI_PROGRAM

struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) material: vec2f,
    @location(4) light: vec4f,
    @location(5) previous: vec4f,
    @location(6) current: vec4f,
    @location(7) uv: vec2f,
    @location(8) @interpolate(flat) clusterDebug: vec3u,
    @location(9) tangent: vec4f,
}
fn geometryVaryings(vertex: Deformed, instanceIndex: u32, uv: vec2f) -> Varyings {
    let instance = instances[instanceIndex];
    var result: Varyings;
    result.position = frame.viewProjection * vertex.position;
    result.current = result.position;
    result.previous = frame.previousViewProjection * vertex.previous;
    result.world = vertex.position.xyz;
    result.normal = vertex.normal.xyz;
    result.tangent = vertex.tangent;
    result.color = instance.color;
    result.material = instance.material.xy;
    result.light = frame.lightViewProjection * vertex.position;
    result.uv = uv;
    result.clusterDebug = vec3u(0u);
    return result;
}
fn makeVaryings(index: u32, instanceIndex: u32) -> Varyings {
    return geometryVaryings(vertices[instanceIndex * params.vertexCount + index], instanceIndex, uvs[index]);
}
@vertex
fn vertexMain(@builtin(vertex_index) index: u32, @builtin(instance_index) instanceIndex: u32) -> Varyings {
    return makeVaryings(index, instanceIndex);
}
@vertex
fn clusterVertexMain(@builtin(vertex_index) index: u32) -> Varyings {
    let count = arrayLength(&clusterVertices);
    let identity = clusterVertices[index % count];
    var result = makeVaryings(identity.x, index / count);
    result.clusterDebug = vec3u(identity.w, identity.z, 1u);
    return result;
}
@vertex
fn pagedClusterVertexMain(@builtin(vertex_index) index: u32) -> Varyings {
    let physical = index % params.vertexCount;
    // The same storage binding holds the physical 128-byte records in this
    // entry point: UV at byte 80 and identity at byte 112. No full source lookup.
    let identity = clusterVertices[physical * 8u + 7u];
    let uv = bitcast<vec2f>(clusterVertices[physical * 8u + 5u].xy);
    var result = geometryVaryings(vertices[index], index / params.vertexCount, uv);
    result.clusterDebug = vec3u(identity.w, identity.z, 1u);
    return result;
}
fn debugColor(seed: u32) -> vec3f {
    // A fixed integer avalanche, independent of frame, dispatch and draw order.
    var value = seed ^ (seed >> 16u);
    value *= 0x7feb352du; value ^= value >> 15u;
    value *= 0x846ca68bu; value ^= value >> 16u;
    return vec3f(0.08) + vec3f(f32(value & 255u), f32((value >> 8u) & 255u), f32((value >> 16u) & 255u)) / 255.0 * 0.65;
}
struct ShadowVaryings {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) material: vec2f,
}
@vertex
fn shadowMain(@builtin(vertex_index) index: u32, @builtin(instance_index) instanceIndex: u32) -> ShadowVaryings {
    var result: ShadowVaryings;
    result.position = frame.lightViewProjection * vertices[instanceIndex * params.vertexCount + index].position;
    result.uv = uvs[index];
    result.color = instances[instanceIndex].color;
    result.material = instances[instanceIndex].material.xy;
    return result;
}
@fragment
fn shadowFragment(input: ShadowVaryings) {
    if (!evaluateCoverage(input.uv, 0.0, input.color, input.material)) { discard; }
}

fn visibility(light: vec4f) -> f32 {
    let projected = light.xyz / light.w;
    let uv = projected.xy * vec2f(0.5, -0.5) + 0.5;
    // Receiver-plane depth gradients: each PCF tap compares the receiver at
    // that tap, not the center depth of a sloping surface. Derivatives precede
    // the divergent bounds branch so every fragment in the quad participates.
    let du = dpdx(uv);
    let dv = dpdy(uv);
    let dz = vec2f(dpdx(projected.z), dpdy(projected.z));
    let determinant = du.x * dv.y - du.y * dv.x;
    var gradient = vec2f(0.0);
    if (abs(determinant) > 1e-12) {
        gradient = vec2f(dz.x * dv.y - dz.y * du.y, du.x * dz.y - dv.x * dz.x) / determinant;
    }
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || projected.z < 0.0 || projected.z > 1.0) {
        return 1.0;
    }
    let size = vec2i(textureDimensions(shadowMap));
    let texel = uv * vec2f(size) - 0.5;
    let base = vec2i(floor(texel));
    let fraction = fract(texel);
    // Rasterization snaps projected vertices to a subpixel grid. Its depth
    // plane can differ from the unsnapped receiver even when both are planar.
    // Reserve 1/16 of a shadow texel for that error, not whole PCF texels.
    let residualBias = 0.000002 + dot(abs(gradient), 1.0 / vec2f(size)) * 0.0625;
    var sum = 0.0;
    // Equivalent weights to 3x3 bilinear PCF, using the 16 unique texels.
    // Compare at each texel center: hardware linear compare uses a single
    // reference for four different receiver depths and causes slope acne.
    for (var y = -1; y <= 2; y++) {
        for (var x = -1; x <= 2; x++) {
            let coordinate = base + vec2i(x, y);
            var weight = vec2f(1.0);
            if (x == -1) { weight.x = 1.0 - fraction.x; }
            if (x == 2) { weight.x = fraction.x; }
            if (y == -1) { weight.y = 1.0 - fraction.y; }
            if (y == 2) { weight.y = fraction.y; }
            let sampleUv = (vec2f(coordinate) + 0.5) / vec2f(size);
            let receiverDepth = projected.z + dot(gradient, sampleUv - uv);
            var lit = 1.0;
            if (all(coordinate >= vec2i(0)) && all(coordinate < size)) {
                let depth = textureLoad(shadowMap, coordinate, 0);
                lit = select(0.0, 1.0, receiverDepth - residualBias <= depth);
            }
            sum += lit * weight.x * weight.y;
        }
    }
    return sum / 9.0;
}
struct FragmentOutput { @location(0) color: vec4f, @location(1) velocity: vec2f }
@fragment
fn fragmentMain(input: Varyings, @builtin(front_facing) front: bool) -> FragmentOutput {
    let shadowVisibility = visibility(input.light);
    let edge1 = dpdx(input.world);
    let edge2 = dpdy(input.world);
    let uv1 = dpdx(input.uv);
    let uv2 = dpdy(input.uv);
    let geometricNormal = surfaceUnit(cross(edge2, edge1), surfaceUnit(input.normal, vec3f(0.0, 1.0, 0.0)) * select(-1.0, 1.0, front));
    let surface = evaluateMaterial(input.uv, 0.0, input.color, input.material);
    let covered = evaluateCoverage(input.uv, 0.0, input.color, input.material);
    if (!covered) { discard; }
    var n = surfaceUnit(input.normal, geometricNormal);
    n *= select(-1.0, 1.0, dot(n, geometricNormal) >= 0.0);
    if (MATERIAL_NORMAL_MAPPED) {
        n = surfaceMappedNormal(input.normal, geometricNormal, input.tangent, edge1, edge2, uv1, uv2, surface.tangentNormal);
    }
    let v = normalize(frame.camera.xyz - input.world);
    let l = normalize(frame.sunDirection.xyz);
    let h = (l + v) / max(length(l + v), 0.00001);
    let nv = max(dot(n, v), 0.001);
    let nl = select(0.0, max(dot(n, l), 0.0), dot(geometricNormal, l) > 0.0);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let metalness = surface.metallic;
    let roughness = surface.roughness;
    let base = surface.baseColor;
    let alpha = roughness * roughness;
    let a2 = alpha * alpha;
    // This form avoids subtracting nearly equal values for smooth surfaces at normal incidence.
    let denominator = (1.0 - nh * nh) + nh * nh * a2;
    let distribution = a2 / max(3.14159265 * denominator * denominator, 1e-20);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let geometry = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
    let f0 = mix(vec3f(0.04), base, metalness);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    let specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
    let diffuse = (1.0 - fresnel) * (1.0 - metalness) * base / 3.14159265;
    let direct = (diffuse + specular) * frame.sunColor.rgb * nl * shadowVisibility;
    // Compiled world-space irradiance, or the legacy fill when GI is unconfigured.
    // INDIRECT_LIGHTING
    let lighting = select(direct + ambient + surface.emissive, base + surface.emissive, MATERIAL_UNLIT);
    var output: FragmentOutput;
    // rgba16float is the bounded scene-linear transport buffer. Exposure and
    // display transfer belong to the final resolve, after lighting and media.
    output.color = vec4f(clamp(lighting, vec3f(0.0), vec3f(65504.0)), 1.0);
    if (input.clusterDebug.z != 0u && frame.debug.x != 0u) {
        let seed = select(input.clusterDebug.x, input.clusterDebug.y + 17u, frame.debug.x == 2u);
        output.color = vec4f(debugColor(seed), 1.0);
    }
    output.velocity = vec2f(0.0);
    if (input.current.w > 0.0001 && input.previous.w > 0.0001) {
        output.velocity = clamp((input.current.xy / input.current.w - input.previous.xy / input.previous.w) * vec2f(0.5, -0.5), vec2f(-2.0), vec2f(2.0));
    }
    return output;
}
