import type { Renderer } from '../../../src/renderer';
import type { WorldRay } from '../../../src/world-query';
type Vec3 = [number, number, number];
const dot = (a: readonly number[], b: readonly number[]) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const add = (a: readonly number[], b: readonly number[], s = 1): Vec3 => [a[0]+b[0]*s,a[1]+b[1]*s,a[2]+b[2]*s];
const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = (v: readonly number[]): Vec3 => { const l=Math.hypot(...v); return [v[0]/l,v[1]/l,v[2]/l]; };
const hash = (v: number): number => { v=Math.imul(v^(v>>>16),0x7feb352d);v=Math.imul(v^(v>>>15),0x846ca68b);return(v^(v>>>16))>>>0; };
export async function intelSponzaReference(renderer: Renderer, pixels: readonly {x:number;y:number}[], eye: Vec3, target: Vec3,
  sky: Vec3, sunColor: Vec3, sunDirection: Vec3, maxDistance: number) {
  const forward=unit(add(target,eye,-1)), right=unit(cross(forward,[0,1,0])), up=cross(right,forward);
  const {width,height}=renderer.stats, focal=Math.tan(Math.PI/6.4), sun=unit(sunDirection), samples=2048;
  const cameraRays: WorldRay[]=pixels.map(p=>({origin:eye,direction:unit(add(add(forward,right,
    (2*(p.x+.5)/width-1)*width/height*focal),up,(1-2*(p.y+.5)/height)*focal)),tMax:150}));
  const surfaces=(await renderer.traceRays(cameraRays)).hits;
  const rows=[];
  for(let point=0;point<surfaces.length;point++) {
    const source=surfaces[point];if(!source){rows.push({pixel:pixels[point],hit:null});continue;}
    const n=unit(source.normal),g=unit(source.geometricNormal),t=unit(cross(Math.abs(n[1])<.9?[0,1,0]:[1,0,0],n)),b=cross(n,t);
    const rays: WorldRay[]=[], sampleIds:number[]=[];
    for(let sample=0;sample<samples;sample++) {
      const seed=hash(sample^Math.imul(point+1,0x9e3779b9)),u=(hash(seed^0x68bc21eb)+.5)/0x100000000;
      const angle=2*Math.PI*(hash(seed^0x02e5be93)+.5)/0x100000000;
      const direction=unit(add(add(add([0,0,0],t,Math.sqrt(u)*Math.cos(angle)),b,Math.sqrt(u)*Math.sin(angle)),n,Math.sqrt(1-u)));
      if(dot(direction,g)<=0)continue;
      sampleIds.push(sample);rays.push({origin:add(source.position,g,.0001),direction,tMax:maxDistance});
    }
    const hits=(await renderer.traceRays(rays,{textureLod:0})).hits;
    const shadowRays:WorldRay[]=[],shadowIds:number[]=[];
    hits.forEach((hit,i)=>{if(hit&&hit.shading==='pbr'&&dot(hit.normal,sun)>0&&dot(hit.geometricNormal,sun)>0){
      shadowIds.push(i);shadowRays.push({origin:add(hit.position,hit.geometricNormal,.03),direction:sun,tMax:maxDistance});}});
    const shadowHits=(await renderer.traceRays(shadowRays,{textureLod:0})).hits;
    const visible=new Set(shadowIds.filter((_,i)=>!shadowHits[i]));
    const values:Vec3[]=Array.from({length:samples},()=>[0,0,0]),sources:Record<string,number>={};
    hits.forEach((hit,i)=>{
      const category=hit?.batch??'sky';sources[category]=(sources[category]??0)+1;
      const incoming:Vec3=hit?[...hit.emissive]:[...sky];
      if(hit?.shading==='unlit')for(let c=0;c<3;c++)incoming[c]+=hit.baseColor[c];
      else if(hit&&visible.has(i))for(let c=0;c<3;c++)incoming[c]+=.96*hit.baseColor[c]*(1-hit.metallic)/Math.PI*sunColor[c]*Math.max(0,dot(hit.normal,sun));
      for(let c=0;c<3;c++)values[sampleIds[i]][c]=incoming[c]*source.baseColor[c]*(1-source.metallic)*.96;
    });
    const mean:Vec3=[0,0,0],variance:Vec3=[0,0,0];
    for(const v of values)for(let c=0;c<3;c++)mean[c]+=v[c]/samples;
    for(const v of values)for(let c=0;c<3;c++)variance[c]+=(v[c]-mean[c])**2/(samples-1);
    rows.push({pixel:pixels[point],hit:source,samples,traced:rays.length,oneBounce:mean,
      standardError:variance.map(v=>Math.sqrt(v/samples)),sources});
  }
  return {method:'Independent CPU cosine-hemisphere Monte Carlo integrator and sun visibility; shares GPU world-query geometry/materials, not an independent geometry correctness reference. 2048 deterministic random samples per point; diffuse-only transport. Query textures use explicit source mip 0; raster derivative filtering can differ. Do not interpret HDR differences as transport error without matching receiver texture footprints.',rows};
}
