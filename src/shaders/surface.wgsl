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
    material: vec2f,
    surfaceIdentity: vec2u,
}
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f, tangent: vec4f }
struct Parameters { vertexCount: u32, instanceCount: u32, jointCount: u32, padding: u32 }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<storage, read> vertices: array<Deformed>;
@group(1) @binding(1) var<storage, read> instances: array<Instance>;
@group(1) @binding(2) var<uniform> params: Parameters;
@group(1) @binding(3) var<storage, read> attributes: array<f32>;
@group(1) @binding(4) var<storage, read> clusterVertices: array<vec4u>;
struct MaterialParameters { values: array<vec4f, 16>, textureInfo: array<vec4f, 4> }
@group(2) @binding(0) var<uniform> materialParameters: MaterialParameters;
@group(2) @binding(1) var materialImage0: texture_2d<f32>;
@group(2) @binding(2) var materialImage1: texture_2d<f32>;
@group(2) @binding(3) var materialImage2: texture_2d<f32>;
@group(2) @binding(4) var materialImage3: texture_2d<f32>;
@group(2) @binding(5) var materialSampler0: sampler;
@group(2) @binding(6) var materialSampler1: sampler;
@group(2) @binding(7) var materialSampler2: sampler;
@group(2) @binding(8) var materialSampler3: sampler;
// A resident mip tail keeps source-space gradients, including odd dimensions.
fn materialTexture0(uv: vec2f, lod: f32) -> vec4f {
  let scale = materialParameters.textureInfo[0].yz;
  return textureSampleGrad(materialImage0, materialSampler0, uv, dpdx(uv) * scale, dpdy(uv) * scale);
}
fn materialTexture1(uv: vec2f, lod: f32) -> vec4f {
  let scale = materialParameters.textureInfo[1].yz;
  return textureSampleGrad(materialImage1, materialSampler1, uv, dpdx(uv) * scale, dpdy(uv) * scale);
}
fn materialTexture2(uv: vec2f, lod: f32) -> vec4f {
  let scale = materialParameters.textureInfo[2].yz;
  return textureSampleGrad(materialImage2, materialSampler2, uv, dpdx(uv) * scale, dpdy(uv) * scale);
}
fn materialTexture3(uv: vec2f, lod: f32) -> vec4f {
  let scale = materialParameters.textureInfo[3].yz;
  return textureSampleGrad(materialImage3, materialSampler3, uv, dpdx(uv) * scale, dpdy(uv) * scale);
}
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
    @location(10) @interpolate(flat) surfaceIdentity: vec2u,
    @location(11) vertexColor: vec4f,
}
fn surfaceUvAt(index: u32) -> vec2f {
    return vec2f(attributes[index * 2u], attributes[index * 2u + 1u]);
}
fn surfaceColorAt(index: u32) -> vec4f {
    let start = params.vertexCount * 2u + index * 4u;
    if (arrayLength(&attributes) <= start) { return vec4f(1.0); }
    return vec4f(attributes[start], attributes[start + 1u], attributes[start + 2u], attributes[start + 3u]);
}
fn geometryVaryings(vertex: Deformed, instanceIndex: u32, uv: vec2f, color: vec4f) -> Varyings {
    let instance = instances[instanceIndex];
    var result: Varyings;
    result.position = frame.viewProjection * vertex.position;
    result.current = result.position;
    result.previous = frame.previousViewProjection * vertex.previous;
    result.world = vertex.position.xyz;
    result.normal = vertex.normal.xyz;
    result.tangent = vertex.tangent;
    result.color = instance.color;
    result.vertexColor = color;
    result.material = instance.material.xy;
    result.surfaceIdentity = instance.surfaceIdentity;
    result.light = frame.lightViewProjection * vertex.position;
    result.uv = uv;
    result.clusterDebug = vec3u(0u);
    return result;
}
fn makeVaryings(index: u32, instanceIndex: u32) -> Varyings {
    return geometryVaryings(vertices[instanceIndex * params.vertexCount + index], instanceIndex, surfaceUvAt(index), surfaceColorAt(index));
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
    // Physical 144-byte record: UV at 80, identity at 112, linear RGBA at 128.
    let identity = clusterVertices[physical * 9u + 7u];
    let uv = bitcast<vec2f>(clusterVertices[physical * 9u + 5u].xy);
    let color = bitcast<vec4f>(clusterVertices[physical * 9u + 8u]);
    var result = geometryVaryings(vertices[index], index / params.vertexCount, uv, color);
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
    @location(3) vertexColor: vec4f,
}
@vertex
fn shadowMain(@builtin(vertex_index) index: u32, @builtin(instance_index) instanceIndex: u32) -> ShadowVaryings {
    var result: ShadowVaryings;
    result.position = frame.lightViewProjection * vertices[instanceIndex * params.vertexCount + index].position;
    result.uv = surfaceUvAt(index);
    result.vertexColor = surfaceColorAt(index);
    result.color = instances[instanceIndex].color;
    result.material = instances[instanceIndex].material.xy;
    return result;
}
@fragment
fn shadowFragment(input: ShadowVaryings) {
    if (!evaluateCoverage(input.uv, 0.0, input.color, input.material, input.vertexColor)) { discard; }
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
// REFLECTION_PROGRAM
struct FragmentOutput { @location(0) color: vec4f, @location(1) velocity: vec2f }
@fragment
fn fragmentMain(input: Varyings, @builtin(front_facing) front: bool) -> FragmentOutput {
    let shadowVisibility = visibility(input.light);
    let edge1 = dpdx(input.world);
    let edge2 = dpdy(input.world);
    let uv1 = dpdx(input.uv);
    let uv2 = dpdy(input.uv);
    let geometricNormal = surfaceUnit(cross(edge2, edge1), surfaceUnit(input.normal, vec3f(0.0, 1.0, 0.0)) * select(-1.0, 1.0, front));
    let surface = evaluateMaterial(input.uv, 0.0, input.color, input.material, input.vertexColor);
    let covered = evaluateCoverage(input.uv, 0.0, input.color, input.material, input.vertexColor);
    if (!covered) { discard; }
    var n = surfaceUnit(input.normal, geometricNormal);
    n *= select(-1.0, 1.0, dot(n, geometricNormal) >= 0.0);
    if (MATERIAL_NORMAL_MAPPED) {
        n = surfaceMappedNormal(input.normal, geometricNormal, input.tangent, edge1, edge2, uv1, uv2, surface.tangentNormal);
    }
    let v = normalize(frame.camera.xyz - input.world);
    let l = normalize(frame.sunDirection.xyz);
    let metalness = surface.metallic;
    let roughness = surface.roughness;
    let base = surface.baseColor;
    let direct = surfaceDirectBrdf(base, metalness, roughness, n, geometricNormal, v, l)
        * frame.sunColor.rgb * shadowVisibility;
    // Compiled world-space irradiance, or the legacy fill when GI is unconfigured.
    // INDIRECT_LIGHTING
    // REFLECTED_LIGHTING
    let lighting = select(direct + ambient + reflected + surface.emissive, base + surface.emissive, MATERIAL_UNLIT);
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
