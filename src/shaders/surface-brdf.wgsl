// Shared raster and world-space direct-light evaluation. `view` and `light`
// are unit directions; radiance and visibility are applied by the caller.
fn surfaceDirectBrdf(base: vec3f, metallic: f32, roughness: f32, n: vec3f,
                     geometricNormal: vec3f, view: vec3f, light: vec3f) -> vec3f {
    let h = (light + view) / max(length(light + view), 0.00001);
    let nv = max(dot(n, view), 0.001);
    let nl = select(0.0, max(dot(n, light), 0.0), dot(geometricNormal, light) > 0.0);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(view, h), 0.0);
    let alpha = roughness * roughness;
    let a2 = alpha * alpha;
    // This form avoids subtracting nearly equal values for smooth surfaces at normal incidence.
    let denominator = (1.0 - nh * nh) + nh * nh * a2;
    let distribution = a2 / max(3.14159265 * denominator * denominator, 1e-20);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let geometry = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
    let f0 = mix(vec3f(0.04), base, metallic);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    let specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
    let diffuse = (1.0 - fresnel) * (1.0 - metallic) * base / 3.14159265;
    return (diffuse + specular) * nl;
}
