struct View { viewProjection:mat4x4f, eye:vec4f, lod:vec4f }
struct Instance { model:mat4x4f, previousModel:mat4x4f, normalMatrix:mat4x4f, color:vec4f, material:vec4f }
struct Parameters {
    clusterCount:u32, physicalVertexCount:u32, instanceCount:u32, outputCapacity:u32,
    firstGroup:u32, groupCount:u32, clusterOffset:u32, groupOffset:u32,
    edgeOffset:u32, pageCount:u32, level:u32, padding:u32,
}
struct Arguments {
    indexCount:atomic<u32>, instanceCount:u32, firstIndex:u32, baseVertex:i32, firstInstance:u32,
    selectedCount:atomic<u32>, overflow:atomic<u32>, padding:atomic<u32>,
}
struct DrawArguments { first:Arguments, recovery:Arguments }
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var<uniform> params:Parameters;
@group(0) @binding(2) var<storage,read> metadata:array<u32>;
@group(0) @binding(3) var<storage,read> instances:array<Instance>;
@group(0) @binding(4) var<storage,read> residency:array<u32>;
// 0 = outside cut, 1 = complete cut but culled, 2 = drawn, 3 = occlusion pending.
@group(0) @binding(5) var<storage,read_write> selected:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> requests:array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> locations:array<vec4u>;
@group(0) @binding(8) var<storage,read> pageIndices:array<u32>;
@group(0) @binding(9) var<storage,read_write> outputIndices:array<u32>;
@group(0) @binding(10) var<storage,read_write> args:DrawArguments;

fn vector(at:u32)->vec3f { return bitcast<vec3f>(vec3u(metadata[at],metadata[at+1u],metadata[at+2u])); }
struct WorldBounds { center:vec3f, extent:vec3f }
fn worldBounds(minimum:vec3f,maximum:vec3f,model:mat4x4f)->WorldBounds {
    let localCenter=(minimum+maximum)*0.5;
    let localExtent=(maximum-minimum)*0.5;
    let center=(model*vec4f(localCenter,1)).xyz;
    var extent=abs(model[0].xyz)*localExtent.x+abs(model[1].xyz)*localExtent.y+abs(model[2].xyz)*localExtent.z;
    extent+=(abs(center)+extent+vec3f(1))*0.000002;
    return WorldBounds(center,extent);
}
fn scaleBound(model:mat4x4f)->f32 {
    let x=abs(model[0].xyz); let y=abs(model[1].xyz); let z=abs(model[2].xyz);
    let columns=max(max(x.x+x.y+x.z,y.x+y.y+y.z),z.x+z.y+z.z);
    let rows=x+y+z; return sqrt(columns*max(max(rows.x,rows.y),rows.z));
}
fn visible(at:u32,model:mat4x4f)->bool {
    if(view.lod.w<0.5){return true;}
    let bounds=worldBounds(vector(at),vector(at+4u),model);
    let center=view.viewProjection*vec4f(bounds.center,1);
    let radius=abs(view.viewProjection[0])*bounds.extent.x+abs(view.viewProjection[1])*bounds.extent.y+abs(view.viewProjection[2])*bounds.extent.z;
    return !(center.x+center.w+radius.x+radius.w<0 || center.w-center.x+radius.w+radius.x<0
        || center.y+center.w+radius.y+radius.w<0 || center.w-center.y+radius.w+radius.y<0
        || center.z+radius.z<0 || center.w-center.z+radius.w+radius.z<0);
}
// Demand is independent of the current cut. Otherwise a missing ancestor hides
// descendant demand and an LRU scheduler can repeatedly throw away its own path.
// Positive f32 bits are ordered as u32, allowing one maximum across instances.
@compute @workgroup_size(64)
fn demand(@builtin(global_invocation_id) invocation:vec3u) {
    let index=invocation.x;
    if(index>=params.groupCount*params.instanceCount){return;}
    let group=index%params.groupCount; let at=params.groupOffset+group*12u;
    if(metadata[at+7u]!=0u){return;}
    let model=instances[index/params.groupCount].model;
    if(!visible(at,model)){return;}
    let bounds=worldBounds(vector(at),vector(at+4u),model);
    let distance=length(max(abs(view.eye.xyz-bounds.center)-bounds.extent,vec3f(0)));
    let pixels=bitcast<f32>(metadata[at+3u])*scaleBound(model)*view.lod.x/max(view.lod.y,distance);
    if(pixels>view.lod.z){atomicMax(&requests[params.pageCount+group],bitcast<u32>(min(pixels,1e30)));}
}
@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) invocation:vec3u) {
    let index=invocation.x;
    if(index>=params.clusterCount*params.instanceCount){return;}
    let at=params.clusterOffset+(index%params.clusterCount)*16u;
    atomicStore(&selected[index],metadata[at+3u]);
}
// One dispatch per hierarchy level, descending. At the same level, validated
// groups have disjoint parents and children; all cross-level dependencies are
// separated by compute passes, never by an assumed cross-workgroup barrier.
@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) invocation:vec3u) {
    let index=invocation.x;
    if(index>=params.groupCount*params.instanceCount){return;}
    let instanceIndex=index/params.groupCount;
    let at=params.groupOffset+(params.firstGroup+index%params.groupCount)*12u;
    if(metadata[at+7u]!=0u){return;}
    let base=instanceIndex*params.clusterCount;
    let parentOffset=params.edgeOffset+metadata[at+8u]; let parentCount=metadata[at+9u];
    for(var i=0u;i<parentCount;i++){
        if(atomicLoad(&selected[base+metadata[parentOffset+i]])!=1u){return;}
    }
    let model=instances[instanceIndex].model;
    let bounds=worldBounds(vector(at),vector(at+4u),model);
    let distance=length(max(abs(view.eye.xyz-bounds.center)-bounds.extent,vec3f(0)));
    let pixels=bitcast<f32>(metadata[at+3u])*scaleBound(model)*view.lod.x/max(view.lod.y,distance);
    if(pixels<=view.lod.z){return;}
    let childOffset=params.edgeOffset+metadata[at+10u]; let childCount=metadata[at+11u];
    var ready=true;
    for(var i=0u;i<childCount;i++){
        let child=metadata[childOffset+i]; let page=metadata[params.clusterOffset+child*16u+7u];
        if(residency[page]==0u){atomicMax(&requests[page],params.level+1u);ready=false;}
    }
    if(!ready){return;}
    for(var i=0u;i<parentCount;i++){atomicStore(&selected[base+metadata[parentOffset+i]],0u);}
    for(var i=0u;i<childCount;i++){atomicStore(&selected[base+metadata[childOffset+i]],1u);}
}
@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) invocation:vec3u) {
    let index=invocation.x;
    if(index>=params.clusterCount*params.instanceCount){return;}
    if(index==0u){args.first.instanceCount=1u;}
    if(atomicLoad(&selected[index])==0u){return;}
    let cluster=index%params.clusterCount; let instanceIndex=index/params.clusterCount;
    let at=params.clusterOffset+cluster*16u;
    let model=instances[instanceIndex].model;
    if(!visible(at,model)){return;}
    let bounds=worldBounds(vector(at),vector(at+4u),model);
    if(geometryOccluded(bounds.center,bounds.extent)){
        atomicStore(&selected[index],3u); atomicAdd(&args.first.padding,1u); return;
    }
    let page=metadata[at+7u];
    // Even on a broken caller contract, do not draw a partially reused slot.
    let location0=locations[cluster*2u]; let location1=locations[cluster*2u+1u];
    if(residency[page]==0u || location1.y!=page){atomicStore(&args.first.overflow,1u);return;}
    let count=location1.x; let offset=atomicAdd(&args.first.indexCount,count);
    if(offset>params.outputCapacity || count>params.outputCapacity-min(offset,params.outputCapacity)){
        atomicStore(&args.first.overflow,1u);return;
    }
    let vertexBase=instanceIndex*params.physicalVertexCount+location0.x;
    for(var i=0u;i<count;i++){outputIndices[offset+i]=pageIndices[location0.w+i]+vertexBase;}
    atomicStore(&selected[index],2u); atomicAdd(&args.first.selectedCount,1u);
}
@compute @workgroup_size(1)
fn finalize(){
    if(atomicLoad(&args.first.overflow)!=0u){atomicStore(&args.first.indexCount,0u);args.first.instanceCount=0u;}
    args.recovery.firstIndex=atomicLoad(&args.first.indexCount);
    args.recovery.instanceCount=1u;
}
@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) invocation:vec3u) {
    let index=invocation.x;
    if(index>=params.clusterCount*params.instanceCount || atomicLoad(&args.first.overflow)!=0u){return;}
    if(atomicLoad(&selected[index])!=3u){return;}
    let cluster=index%params.clusterCount; let instanceIndex=index/params.clusterCount;
    let at=params.clusterOffset+cluster*16u;
    let bounds=worldBounds(vector(at),vector(at+4u),instances[instanceIndex].model);
    if(geometryOccluded(bounds.center,bounds.extent)){
        atomicStore(&selected[index],1u); atomicAdd(&args.recovery.padding,1u); return;
    }
    let page=metadata[at+7u];
    let location0=locations[cluster*2u]; let location1=locations[cluster*2u+1u];
    if(residency[page]==0u || location1.y!=page){atomicStore(&args.recovery.overflow,1u);return;}
    let count=location1.x; let offset=atomicAdd(&args.recovery.indexCount,count);
    let base=args.recovery.firstIndex;
    if(base>params.outputCapacity || offset>params.outputCapacity-base
        || count>params.outputCapacity-base-offset){atomicStore(&args.recovery.overflow,1u);return;}
    let vertexBase=instanceIndex*params.physicalVertexCount+location0.x;
    for(var i=0u;i<count;i++){outputIndices[base+offset+i]=pageIndices[location0.w+i]+vertexBase;}
    atomicStore(&selected[index],2u); atomicAdd(&args.recovery.selectedCount,1u);
}
@compute @workgroup_size(1)
fn finalizeRecovery(){
    if(atomicLoad(&args.recovery.overflow)!=0u || atomicLoad(&args.first.overflow)!=0u){
        atomicStore(&args.recovery.indexCount,0u); args.recovery.instanceCount=0u;
    }
}
