struct BuildParameters { width:u32,height:u32,pixelSize:u32,padding:u32 }
@group(0) @binding(0) var sourceDepth:texture_depth_2d;
@group(0) @binding(1) var outputDepth:texture_storage_2d<r32float,write>;
@group(0) @binding(2) var<uniform> build:BuildParameters;
@compute @workgroup_size(8,8)
fn base(@builtin(global_invocation_id) id:vec3u){
    if(any(id.xy>=textureDimensions(outputDepth))){return;}
    let origin=id.xy*build.pixelSize;
    var farthest=0.0;
    for(var y=0u;y<build.pixelSize;y++){for(var x=0u;x<build.pixelSize;x++){
        let p=origin+vec2u(x,y);
        var depth=1.0;
        if(p.x<build.width && p.y<build.height){depth=textureLoad(sourceDepth,vec2i(p),0);}
        farthest=max(farthest,depth);
    }}
    textureStore(outputDepth,vec2i(id.xy),vec4f(farthest));
}
