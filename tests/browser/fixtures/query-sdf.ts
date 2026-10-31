/**
 * Smallest distance-field experiment for the roadmap's BVH-versus-distance-field
 * decision. Test-only: this is not a production query path and has no engine
 * contract. A dense unsigned distance grid (float bits in u32) is splatted from
 * world-space triangles with exact point-triangle distances clamped to a band
 * (atomicMin), re-splatted per region after an edit, and sphere traced with
 * manual trilinear sampling. A hit is the first sample below `threshold`; it has
 * no triangle, material or entity identity. Every dispatch is bounded: splat work
 * is the clipped voxel box of each triangle, tracing stops at maxSteps or tMax.
 */
type Vec3 = [number, number, number];
export interface DenseUdfSettings { voxel: number; bandVoxels: number; thresholdVoxels: number; maxSteps: number }
export interface VoxelRegion { low: Vec3; high: Vec3 } // Voxel index box, high exclusive.

const SHADER = /* wgsl */ `
struct Grid { origin: vec3f, voxel: f32, dims: vec3u, count: u32, low: vec3u, first: u32, high: vec3u, rowStride: u32,
    band: f32, threshold: f32, maxSteps: u32, pad: u32 }
@group(0) @binding(0) var<uniform> grid: Grid;

fn cell(v: vec3u) -> u32 { return (v.z * grid.dims.y + v.y) * grid.dims.x + v.x; }

// Ericson, Real-Time Collision Detection 5.1.5: closest point on a triangle.
fn triangleDistance(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
    let ab = b - a; let ac = c - a; let ap = p - a;
    let d1 = dot(ab, ap); let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return length(ap); }
    let bp = p - b; let d3 = dot(ab, bp); let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return length(bp); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return length(ap - ab * (d1 / (d1 - d3))); }
    let cp = p - c; let d5 = dot(ab, cp); let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return length(cp); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return length(ap - ac * (d2 / (d2 - d6))); }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return length(bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let sum = va + vb + vc;
    if (!(sum > 0.0)) { return min(length(ap), min(length(bp), length(cp))); }
    return length(ap - ab * (vb / sum) - ac * (vc / sum));
}

@group(0) @binding(1) var<storage, read_write> cells: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> triangles: array<vec4f>;

@compute @workgroup_size(64)
fn clear(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    let extent = grid.high - grid.low;
    let index = (group.y * grid.rowStride + group.x) * 64u + lane;
    if (index >= extent.x * extent.y * extent.z) { return; }
    let voxel = grid.low + vec3u(index % extent.x, (index / extent.x) % extent.y, index / (extent.x * extent.y));
    atomicStore(&cells[cell(voxel)], bitcast<u32>(grid.band));
}

// One workgroup per triangle; lanes stride over its band box clipped to the region.
@compute @workgroup_size(64)
fn splat(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    let local = group.y * grid.rowStride + group.x;
    if (local >= grid.count) { return; }
    let t = (grid.first + local) * 3u;
    let a = triangles[t].xyz; let b = triangles[t + 1u].xyz; let c = triangles[t + 2u].xyz;
    let lowWorld = (min(min(a, b), c) - grid.band - grid.origin) / grid.voxel - 0.5;
    let highWorld = (max(max(a, b), c) + grid.band - grid.origin) / grid.voxel - 0.5;
    let low = max(vec3i(ceil(lowWorld)), vec3i(grid.low));
    let high = min(vec3i(floor(highWorld)) + 1, vec3i(grid.high));
    if (any(high <= low)) { return; }
    let extent = vec3u(high - low);
    let total = extent.x * extent.y * extent.z;
    for (var k = lane; k < total; k += 64u) {
        let voxel = vec3u(low) + vec3u(k % extent.x, (k / extent.x) % extent.y, k / (extent.x * extent.y));
        let distance = triangleDistance(grid.origin + (vec3f(voxel) + 0.5) * grid.voxel, a, b, c);
        if (distance < grid.band) { atomicMin(&cells[cell(voxel)], bitcast<u32>(distance)); }
    }
}
`;

const TRACE = /* wgsl */ `
struct Grid { origin: vec3f, voxel: f32, dims: vec3u, count: u32, low: vec3u, first: u32, high: vec3u, rowStride: u32,
    band: f32, threshold: f32, maxSteps: u32, pad: u32 }
struct Ray { origin: vec4f, direction: vec4f, ignore: vec4u }
@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> rays: array<Ray>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4f>;

fn fetch(v: vec3i) -> f32 {
    if (any(v < vec3i(0)) || any(v >= vec3i(grid.dims))) { return grid.band; }
    return field[(u32(v.z) * grid.dims.y + u32(v.y)) * grid.dims.x + u32(v.x)];
}
fn sampleField(p: vec3f) -> f32 {
    let g = (p - grid.origin) / grid.voxel - 0.5;
    let base = floor(g); let w = g - base; let i = vec3i(base);
    let x00 = mix(fetch(i), fetch(i + vec3i(1, 0, 0)), w.x);
    let x10 = mix(fetch(i + vec3i(0, 1, 0)), fetch(i + vec3i(1, 1, 0)), w.x);
    let x01 = mix(fetch(i + vec3i(0, 0, 1)), fetch(i + vec3i(1, 0, 1)), w.x);
    let x11 = mix(fetch(i + vec3i(0, 1, 1)), fetch(i + vec3i(1, 1, 1)), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}
// (hit distance or -1, steps, first sample, 1 when maxSteps ended an unresolved ray)
@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= grid.count) { return; }
    let ray = rays[id.x];
    var t = ray.origin.w;
    var d = sampleField(ray.origin.xyz + ray.direction.xyz * t);
    let first = d;
    var steps = 0u;
    var hit = -1.0;
    loop {
        if (d < grid.threshold) { hit = t; break; }
        t += d;
        if (t > ray.direction.w || steps >= grid.maxSteps) { break; }
        steps++;
        d = sampleField(ray.origin.xyz + ray.direction.xyz * t);
    }
    results[id.x] = vec4f(hit, f32(steps), first, select(0.0, 1.0, steps >= grid.maxSteps && hit < 0.0));
}
`;

const PARAM_BYTES = 256, PARAM_SLOTS = 64, MAX_ROW = 32_768;

export class DenseUdf {
  readonly dims: Vec3;
  readonly origin: Vec3;
  readonly band: number;
  readonly threshold: number;
  readonly cellBytes: number;
  private readonly cells: GPUBuffer;
  private readonly triangleBuffer: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly results: GPUBuffer;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly splatPipeline: GPUComputePipeline;
  private readonly tracePipeline: GPUComputePipeline;
  private readonly writeGroups: GPUBindGroup[] = [];
  private readonly traceGroups = new Map<GPUBuffer, GPUBindGroup[]>();
  private slot = 0;

  static async create(device: GPUDevice, bounds: { min: Vec3; max: Vec3 }, settings: DenseUdfSettings,
    triangles: Float32Array<ArrayBuffer>, maxRays: number): Promise<DenseUdf> {
    const module = device.createShaderModule({ code: SHADER, label: 'udf experiment build' });
    const traceModule = device.createShaderModule({ code: TRACE, label: 'udf experiment trace' });
    const [clear, splat, trace] = await Promise.all([
      device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'clear' } }),
      device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'splat' } }),
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: traceModule, entryPoint: 'trace' } })]);
    return new DenseUdf(device, bounds, settings, triangles, maxRays, clear, splat, trace);
  }

  private constructor(private readonly device: GPUDevice, bounds: { min: Vec3; max: Vec3 }, private readonly settings: DenseUdfSettings,
    triangles: Float32Array<ArrayBuffer>, maxRays: number, clear: GPUComputePipeline, splat: GPUComputePipeline, trace: GPUComputePipeline) {
    this.band = settings.voxel * settings.bandVoxels;
    this.threshold = settings.voxel * settings.thresholdVoxels;
    this.origin = bounds.min.map(value => value - this.band) as Vec3;
    this.dims = bounds.max.map((value, axis) => Math.ceil((value + this.band - this.origin[axis]) / settings.voxel)) as Vec3;
    this.cellBytes = this.dims[0] * this.dims[1] * this.dims[2] * 4;
    const limits = device.limits;
    if (this.cellBytes > limits.maxStorageBufferBindingSize || this.cellBytes > limits.maxBufferSize) {
      throw new RangeError(`UDF_GRID_BUDGET: ${this.dims.join('x')} cells need ${this.cellBytes} bytes`);
    }
    this.clearPipeline = clear; this.splatPipeline = splat; this.tracePipeline = trace;
    this.cells = device.createBuffer({ label: 'udf cells', size: this.cellBytes, usage: GPUBufferUsage.STORAGE });
    this.triangleBuffer = device.createBuffer({ label: 'udf triangles', size: triangles.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.triangleBuffer, 0, triangles);
    this.params = device.createBuffer({ label: 'udf params', size: PARAM_BYTES * PARAM_SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.results = device.createBuffer({ label: 'udf results', size: maxRays * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    for (let slot = 0; slot < PARAM_SLOTS; slot++) {
      this.writeGroups.push(device.createBindGroup({ layout: splat.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.params, offset: slot * PARAM_BYTES, size: 80 } },
        { binding: 1, resource: { buffer: this.cells } }, { binding: 2, resource: { buffer: this.triangleBuffer } }] }));
    }
  }

  get ownedBytes(): number { return this.cellBytes + this.triangleBuffer.size + this.params.size + this.results.size; }
  get resultBuffer(): GPUBuffer { return this.results; }
  get all(): VoxelRegion { return { low: [0, 0, 0], high: [...this.dims] }; }

  /** Voxel box whose centres can lie within the band of a world box. */
  region(bounds: { min: Vec3; max: Vec3 }): VoxelRegion {
    const low = bounds.min.map((value, axis) => Math.max(0, Math.ceil((value - this.band - this.origin[axis]) / this.settings.voxel - 0.5))) as Vec3;
    const high = bounds.max.map((value, axis) => Math.min(this.dims[axis], Math.floor((value + this.band - this.origin[axis]) / this.settings.voxel - 0.5) + 1)) as Vec3;
    return { low, high: high.map((value, axis) => Math.max(value, low[axis])) as Vec3 };
  }

  /** Hardware-independent splat work: voxel-triangle distance evaluations. */
  evaluations(triangles: Float32Array, first: number, count: number, region: VoxelRegion): number {
    let total = 0;
    for (let t = first; t < first + count; t++) {
      let product = 1;
      for (let axis = 0; axis < 3; axis++) {
        const values = [0, 1, 2].map(v => triangles[(t * 3 + v) * 4 + axis]);
        const low = Math.max(Math.ceil((Math.min(...values) - this.band - this.origin[axis]) / this.settings.voxel - 0.5), region.low[axis]);
        const high = Math.min(Math.floor((Math.max(...values) + this.band - this.origin[axis]) / this.settings.voxel - 0.5) + 1, region.high[axis]);
        product *= Math.max(0, high - low);
      }
      total += product;
    }
    return total;
  }

  writeTriangles(first: number, data: Float32Array<ArrayBuffer>): void { this.device.queue.writeBuffer(this.triangleBuffer, first * 48, data); }

  /** Call before encoding into a new command encoder; slots are reused per submission. */
  beginEncoder(): void { this.slot = 0; }

  private writeParams(region: VoxelRegion, first: number, count: number, rowStride: number, threshold = this.threshold): number {
    if (this.slot >= PARAM_SLOTS) throw new RangeError('UDF_PARAM_SLOTS');
    const data = new ArrayBuffer(80), floats = new Float32Array(data), words = new Uint32Array(data);
    floats.set(this.origin, 0); floats[3] = this.settings.voxel;
    words.set(this.dims, 4); words[7] = count;
    words.set(region.low, 8); words[11] = first;
    words.set(region.high, 12); words[15] = rowStride;
    floats[16] = this.band; floats[17] = threshold; words[18] = this.settings.maxSteps;
    this.device.queue.writeBuffer(this.params, this.slot * PARAM_BYTES, data);
    return this.slot++;
  }

  encodeClear(pass: GPUComputePassEncoder, region: VoxelRegion): void {
    const cells = (region.high[0] - region.low[0]) * (region.high[1] - region.low[1]) * (region.high[2] - region.low[2]);
    if (!cells) return;
    const groups = Math.ceil(cells / 64), x = Math.min(groups, MAX_ROW);
    pass.setPipeline(this.clearPipeline);
    pass.setBindGroup(0, this.groupFor(this.clearPipeline, this.writeParams(region, 0, 0, x)));
    pass.dispatchWorkgroups(x, Math.ceil(groups / x));
  }

  encodeSplat(pass: GPUComputePassEncoder, first: number, count: number, region: VoxelRegion): void {
    if (!count) return;
    const x = Math.min(count, MAX_ROW);
    pass.setPipeline(this.splatPipeline);
    pass.setBindGroup(0, this.groupFor(this.splatPipeline, this.writeParams(region, first, count, x)));
    pass.dispatchWorkgroups(x, Math.ceil(count / x));
  }

  /** `threshold` overrides the hit distance (metres) for an accuracy sweep. */
  encodeTrace(pass: GPUComputePassEncoder, rays: GPUBuffer, count: number, threshold = this.threshold): void {
    const slot = this.writeParams(this.all, 0, count, 0, threshold);
    let groups = this.traceGroups.get(rays);
    if (!groups) {
      groups = Array.from({ length: PARAM_SLOTS }, (_, index) => this.device.createBindGroup({ layout: this.tracePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.params, offset: index * PARAM_BYTES, size: 80 } }, { binding: 1, resource: { buffer: this.cells } },
        { binding: 2, resource: { buffer: rays } }, { binding: 3, resource: { buffer: this.results } }] }));
      this.traceGroups.set(rays, groups);
    }
    pass.setPipeline(this.tracePipeline);
    pass.setBindGroup(0, groups[slot]);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
  }

  // Clear and splat share one auto layout shape; bind groups are created against each pipeline.
  private readonly clearGroups: GPUBindGroup[] = [];
  private groupFor(pipeline: GPUComputePipeline, slot: number): GPUBindGroup {
    if (pipeline === this.splatPipeline) return this.writeGroups[slot];
    if (!this.clearGroups[slot]) {
      // The clear entry point does not read triangles, so its auto layout has no binding 2.
      this.clearGroups[slot] = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.params, offset: slot * PARAM_BYTES, size: 80 } }, { binding: 1, resource: { buffer: this.cells } }] });
    }
    return this.clearGroups[slot];
  }

  dispose(): void {
    this.cells.destroy(); this.triangleBuffer.destroy(); this.params.destroy(); this.results.destroy();
  }
}
