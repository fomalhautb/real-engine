struct OcclusionView { viewProjection:mat4x4f, viewport:vec4u, settings:vec4f }
@group(1) @binding(0) var<uniform> occlusion:OcclusionView;
@group(1) @binding(1) var occlusionDepth:texture_2d<f32>;

// Max depth and a conservative projected box: a hole/sky texel prevents rejection.
// Previous depth is only a prediction; every rejection is retested this frame.
fn geometryOccluded(center:vec3f,extent:vec3f)->bool {
    if(occlusion.settings.x<0.5){return false;}
    var minimum=vec2f(1e30); var maximum=vec2f(-1e30); var nearest=1.0;
    for(var corner=0u;corner<8u;corner++){
        let sign=vec3f(select(-1.0,1.0,(corner&1u)!=0u),select(-1.0,1.0,(corner&2u)!=0u),select(-1.0,1.0,(corner&4u)!=0u));
        let clip=occlusion.viewProjection*vec4f(center+extent*sign,1.0);
        // Near-plane intersections and unrepresentable projections remain visible.
        if(!(all(abs(clip)<vec4f(1e30))) || clip.w<=0.000001 || clip.z<=0.0){return false;}
        let ndc=clip.xyz/clip.w;
        let uv=vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5);
        minimum=min(minimum,uv); maximum=max(maximum,uv); nearest=min(nearest,ndc.z);
    }
    if(any(maximum<vec2f(0.0)) || any(minimum>vec2f(1.0))){return false;}
    // One source pixel guards projection/raster roundoff at the rectangle edge.
    let low=max(vec2f(0.0),floor(minimum*vec2f(occlusion.viewport.xy))-vec2f(1.0))/occlusion.settings.z;
    let high=min(vec2f(occlusion.viewport.xy)-vec2f(1.0),ceil(maximum*vec2f(occlusion.viewport.xy))+vec2f(1.0))/occlusion.settings.z;
    let span=max(high-low+vec2f(1.0),vec2f(1.0));
    let level=u32(clamp(ceil(log2(max(span.x,span.y))),0.0,occlusion.settings.w));
    let scale=exp2(f32(level));
    let limit=vec2i(textureDimensions(occlusionDepth,level))-vec2i(1);
    let a=clamp(vec2i(floor(low/scale)),vec2i(0),limit);
    let b=clamp(vec2i(floor(high/scale)),vec2i(0),limit);
    let farthest=max(max(textureLoad(occlusionDepth,a,i32(level)).x,textureLoad(occlusionDepth,b,i32(level)).x),
        max(textureLoad(occlusionDepth,vec2i(a.x,b.y),i32(level)).x,textureLoad(occlusionDepth,vec2i(b.x,a.y),i32(level)).x));
    return nearest>farthest+occlusion.settings.y;
}
