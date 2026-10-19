@group(0) @binding(0) var inputDepth:texture_2d<f32>;
@group(0) @binding(1) var outputDepth:texture_storage_2d<r32float,write>;
@compute @workgroup_size(8,8)
fn reduce(@builtin(global_invocation_id) id:vec3u){
    if(any(id.xy>=textureDimensions(outputDepth))){return;}
    let p=vec2i(id.xy*2u); let limit=vec2i(textureDimensions(inputDepth))-vec2i(1);
    let maximum=max(max(textureLoad(inputDepth,min(p,limit),0).x,textureLoad(inputDepth,min(p+vec2i(1,0),limit),0).x),
        max(textureLoad(inputDepth,min(p+vec2i(0,1),limit),0).x,textureLoad(inputDepth,min(p+vec2i(1),limit),0).x));
    textureStore(outputDepth,vec2i(id.xy),vec4f(maximum));
}
