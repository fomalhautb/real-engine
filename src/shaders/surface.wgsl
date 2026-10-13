struct Frame {
    viewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    lightViewProjection: mat4x4f,
    camera: vec4f,
    sunDirection: vec4f,
    sunColor: vec4f,
}
struct Instance {
    model: mat4x4f,
    previousModel: mat4x4f,
    normalMatrix: mat4x4f,
    color: vec4f,
    material: vec4f,
}
struct Deformed { position: vec4f, normal: vec4f, previous: vec4f }
struct Parameters { vertexCount: u32, instanceCount: u32, jointCount: u32, padding: u32 }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(1) @binding(0) var<storage, read> vertices: array<Deformed>;
@group(1) @binding(1) var<storage, read> instances: array<Instance>;
@group(1) @binding(2) var<uniform> params: Parameters;

struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) material: vec2f,
    @location(4) light: vec4f,
    @location(5) previous: vec4f,
    @location(6) current: vec4f,
}
@vertex
fn vertexMain(@builtin(vertex_index) index: u32, @builtin(instance_index) instanceIndex: u32) -> Varyings {
    let vertex = vertices[instanceIndex * params.vertexCount + index];
    let instance = instances[instanceIndex];
    var result: Varyings;
    result.position = frame.viewProjection * vertex.position;
    result.current = result.position;
    result.previous = frame.previousViewProjection * vertex.previous;
    result.world = vertex.position.xyz;
    result.normal = vertex.normal.xyz;
    result.color = instance.color;
    result.material = instance.material.xy;
    result.light = frame.lightViewProjection * vertex.position;
    return result;
}
@vertex
fn shadowMain(@builtin(vertex_index) index: u32, @builtin(instance_index) instanceIndex: u32) -> @builtin(position) vec4f {
    return frame.lightViewProjection * vertices[instanceIndex * params.vertexCount + index].position;
}

fn visibility(light: vec4f, ndotl: f32) -> f32 {
    let projected = light.xyz / light.w;
    let uv = projected.xy * vec2f(0.5, -0.5) + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || projected.z < 0.0 || projected.z > 1.0) {
        return 1.0;
    }
    let pixel = 1.0 / vec2f(textureDimensions(shadowMap));
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2f(f32(x), f32(y)) * pixel, projected.z - 0.0004 - (1.0 - ndotl) * 0.0008);
        }
    }
    return sum / 9.0;
}
fn tonemap(color: vec3f) -> vec3f {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
struct FragmentOutput { @location(0) color: vec4f, @location(1) velocity: vec2f }
@fragment
fn fragmentMain(input: Varyings) -> FragmentOutput {
    let n = normalize(input.normal);
    let v = normalize(frame.camera.xyz - input.world);
    let l = normalize(frame.sunDirection.xyz);
    let h = normalize(l + v);
    let nv = max(dot(n, v), 0.001);
    let nl = max(dot(n, l), 0.0);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let metalness = clamp(input.material.x, 0.0, 1.0);
    let roughness = clamp(input.material.y, 0.08, 1.0);
    var base = input.color.rgb;
    if (input.color.a < 0.5) {
        let grid = abs(fract(input.world.xz * 0.5 - 0.5) - 0.5);
        let line = 1.0 - smoothstep(0.01, 0.035, min(grid.x, grid.y));
        base *= 1.0 - line * 0.24;
    }
    let alpha = roughness * roughness;
    let a2 = alpha * alpha;
    let denominator = nh * nh * (a2 - 1.0) + 1.0;
    let distribution = a2 / max(3.14159265 * denominator * denominator, 0.00001);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let geometry = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
    let f0 = mix(vec3f(0.04), base, metalness);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    let specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
    let diffuse = (1.0 - fresnel) * (1.0 - metalness) * base / 3.14159265;
    let direct = (diffuse + specular) * frame.sunColor.rgb * nl * visibility(input.light, nl);
    // Constant ambient fill is explicitly not world-space GI.
    let ambient = base * (0.045 + 0.065 * max(n.y, 0.0)) * (1.0 - metalness * 0.6);
    let distanceFog = 1.0 - exp(-distance(frame.camera.xyz, input.world) * 0.008);
    let radiance = mix(direct + ambient, vec3f(0.055, 0.085, 0.10), distanceFog);
    var output: FragmentOutput;
    output.color = vec4f(pow(tonemap(radiance), vec3f(1.0 / 2.2)), 1.0);
    output.velocity = vec2f(0.0);
    if (input.current.w > 0.0001 && input.previous.w > 0.0001) {
        output.velocity = clamp((input.current.xy / input.current.w - input.previous.xy / input.previous.w) * vec2f(0.5, -0.5), vec2f(-2.0), vec2f(2.0));
    }
    return output;
}
