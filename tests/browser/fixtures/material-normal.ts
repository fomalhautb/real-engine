import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { buildGeometryHierarchy } from '../../../src/geometry-hierarchy';
import { identity, lookAt, multiply, normalMatrix, perspective, scaling, transformPoint, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { TextureAsset } from '../../../src/texture-assets';
import type { WorldHit, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const size = { width: 512, height: 384 };
const eye: Vec3 = [0, 0, 4], target: Vec3 = [0, 0, 0];
const sunDirection: Vec3 = [0.8, 0, 0.6], sunColor: Vec3 = [2, 2, 2];
const positions: Vec3[] = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
const sampleWorld: Vec3[] = [[-0.5, 0.18, 0], [0.5, 0.18, 0]];
const pixelRays = (pixels: { x: number; y: number }[]): WorldRay[] => pixels.map(pixel => {
  const ndcX = (pixel.x + 0.5) / size.width * 2 - 1;
  const ndcY = 1 - (pixel.y + 0.5) / size.height * 2;
  const halfHeight = 4 * Math.tan(Math.PI / 6.4);
  const direction = [ndcX * halfHeight * size.width / size.height,
    ndcY * halfHeight, -4] as Vec3;
  const length = Math.hypot(...direction);
  return ray(0, 0, 4, direction.map(value => value / length) as Vec3);
});

function quad(handedness = 1): MeshData {
  const vertices = new ArrayBuffer(positions.length * 80), view = new DataView(vertices);
  for (let i = 0; i < positions.length; i++) {
    const at = i * 80;
    positions[i].forEach((value, axis) => view.setFloat32(at + axis * 4, value, true));
    view.setFloat32(at + 12, 1, true);
    view.setFloat32(at + 24, 1, true);
    view.setUint32(at + 32, 0, true);
    view.setFloat32(at + 48, 1, true);
  }
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
    tangents: new Float32Array([1, 0, 0, handedness, 1, 0, 0, handedness,
      1, 0, 0, handedness, 1, 0, 0, handedness]),
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function instance(entityId: number, model = identity(), bones: Float32Array[] = [],
  deformScalars: number[] = []): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones, deformScalars,
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

function ray(x: number, y: number, z: number, direction: Vec3): WorldRay {
  return { origin: [x, y, z], direction, tMax: 10 };
}

function project(point: Vec3): { x: number; y: number } {
  const projection = perspective(Math.PI / 3.2, size.width / size.height, 0.1, 150);
  const ndc = transformPoint(multiply(projection, lookAt(eye, target, [0, 1, 0])), point);
  return { x: Math.floor((ndc[0] * 0.5 + 0.5) * size.width),
    y: Math.floor((1 - (ndc[1] * 0.5 + 0.5)) * size.height) };
}

function hits(result: Awaited<ReturnType<Renderer['traceRays']>>): WorldHit[] {
  return result.hits.map((hit, index) => {
    if (!hit) throw new Error(`NORMAL_MAP_MISSING_HIT_${index}`);
    return hit;
  });
}

/** Authored tangent and UV fixture: two texels carry opposite X slopes. */
export async function runMaterialNormalFixture() {
  const checker: TextureAsset = { width: 2, height: 1, colorSpace: 'linear',
    pixels: new Uint8Array([191, 128, 238, 255, 64, 128, 238, 255]) };
  const material = { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
    baseColor: m.constant([0.65, 0.45, 0.3, 1]), metallic: 0, roughness: 1,
    normal: m.texture(0), normalScale: m.component(m.parameter(0), 0) },
    parameters: new Float32Array([0, 0, 0, 0]), textures: [checker],
    samplers: [{ minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', lodMaxClamp: 0 } as const] };
  const plain = (name: string, mesh: MeshData, state: InstanceState, primitiveId: number): SceneBatch => ({
    name, mesh, capacity: 1, inputMode: 'committed', instances: [state],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'material-normal-quad', primitiveId, topologyVersion: 1 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
      baseColor: m.constant([0.65, 0.45, 0.3, 1]), metallic: 0, roughness: 1,
      normal: m.constant([128 / 255, 191 / 255, 238 / 255, 1]) } },
  });
  const staticBatch: SceneBatch = { ...plain('normal-checker', quad(), instance(601), 0), material };
  const mirrored = plain('normal-mirrored', quad(-1),
    instance(602, multiply(translation(-4, 0, 0), scaling(-1, 1, 1))), 1);
  const uvFallback = plain('normal-uv-fallback',
    { ...quad(), tangents: undefined }, instance(604, translation(7, 0, 0)), 3);
  // z = 0.4*x after Morph; one affine bone and a nonuniform model transform
  // must transport the authored tangent and normal through the same graph.
  const morphPositions = new Float32Array(12), morphNormals = new Float32Array(12);
  const morphTangents = new Float32Array(12);
  positions.forEach((position, i) => { morphPositions[i * 3 + 2] = position[0] * 0.4;
    morphNormals[i * 3] = -0.4; morphTangents[i * 3 + 2] = 0.4; });
  const dynamic: SceneBatch = { ...plain('normal-affine-morph', quad(),
    instance(603, multiply(translation(4, 0, 0), scaling(1.5, 0.8, 0.7)),
      [scaling(1.2, 1, 0.8)], [1]), 2),
    deformation: { jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 },
      { kind: 'skin', normalMode: 'affine' }] },
    morphTargets: [{ positions: morphPositions, normals: morphNormals, tangents: morphTangents }] };
  const waveMesh = quad();
  waveMesh.tangents = new Float32Array([0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1]);
  const wave = { ...plain('normal-wave', waveMesh,
    instance(606, translation(10, 0, 0), [], [0]), 4),
    deformation: { jointCount: 0, nodes: [{ kind: 'wave' as const,
      amplitude: 0.25, frequency: Math.PI / 2, phase: 0 }] },
  } satisfies SceneBatch;
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${size.width}px;height:${size.height}px;z-index:1000`;
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [staticBatch, mirrored, dynamic, uvFallback, wave], {
      worldQueries: { maxRays: 128, maxTrianglesPerMesh: 4, maxInstancesPerBatch: 1 },
      diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance: 8, skyRadiance: [0, 0, 0] },
      antiAliasing: 'none',
    });
    renderer.setDiffuseGiEnabled(false);
    const pixels = sampleWorld.map(project);
    const depthPoints = sampleWorld.map(point => [...point] as Vec3);
    const rays: WorldRay[] = [ray(-0.5, 0.18, 5, [0, 0, -1]),
      ray(0.5, 0.18, 5, [0, 0, -1]), ray(-0.5, 0.18, -5, [0, 0, 1]),
      ray(-4, 0, 5, [0, 0, -1]), ray(4, 0, 5, [0, 0, -1]),
      ray(7, 0, 5, [0, 0, -1]), ...pixelRays(pixels)];
    const capture = async (light: Vec3) => {
      renderer!.setSun({ direction: sunDirection, color: light });
      renderer!.render(eye, target);
      if (renderer!.stats.width !== size.width || renderer!.stats.height !== size.height) {
        throw new Error(`NORMAL_MAP_CANVAS_SIZE_${renderer!.stats.width}x${renderer!.stats.height}`);
      }
      // The PNG is taken immediately from the presented frame; readbacks follow.
      const png = canvas.toDataURL('image/png');
      if (!png.startsWith('data:image/png;base64,')) throw new Error('NORMAL_MAP_PNG_CAPTURE');
      return { png, hdr: await renderer!.probeHdr(pixels), depth: await renderer!.probeDepth(depthPoints),
        hits: hits(await renderer!.traceRays(rays, { textureLod: 0 })),
        variants: renderer!.diagnostics().materialVariants };
    };
    const neutral = await capture(sunColor);
    renderer.setMaterialParameters(staticBatch.name, new Float32Array([1, 0, 0, 0]));
    const tilted = await capture(sunColor);
    const noSun = await capture([0, 0, 0]);
    renderer.setDiffuseGiEnabled(true);
    renderer.render(eye, target);
    const giHits = hits(await renderer.traceRays(rays, { textureLod: 0 }));
    const waveRay = ray(10, 0.5, 5, [0, 0, -1]);
    const waveBefore = hits(await renderer.traceRays([waveRay]))[0];
    wave.instances = [{ ...instance(606, translation(10, 0, 0), [], [Math.PI / 2]),
      historyVersion: 1 }];
    renderer.render(eye, target);
    const waveAfter = hits(await renderer.traceRays([waveRay]))[0];
    const verification = await renderer.verifyDeformation({ allInstances: true });
    const authoredTangents = staticBatch.mesh.tangents;
    staticBatch.mesh.tangents = new Float32Array(authoredTangents!);
    let tangentReplacementError = '';
    try { await renderer.traceRays([rays[0]]); }
    catch (error) { tangentReplacementError = String(error); }
    finally { staticBatch.mesh.tangents = authoredTangents; }
    const restoredTangentHit = hits(await renderer.traceRays([rays[0]]))[0];
    const errors = [...renderer.errors];
    renderer.dispose();
    return { size, pixels, sampleWorld, sunDirection, sunColor,
      neutral, tilted, noSun, giHits, waveBefore, waveAfter, verification, tangentReplacementError, restoredTangentHit, errors,
      ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** A top face may have a tilted shading normal pointing toward a sun below it.
 * Direct light, world GI, and final gathering must honor the real face. */
export async function runMaterialNormalBacklightFixture() {
  const batch: SceneBatch = { name: 'normal-backlight-barrier', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [instance(605)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'normal-backlight-barrier', primitiveId: 0, topologyVersion: 1 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
      baseColor: m.constant([0.7, 0.7, 0.7, 1]), metallic: 0, roughness: 1,
      normal: m.constant([1, 0.5, 0.57, 1]) } } };
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${size.width}px;height:${size.height}px;z-index:1000`;
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  const backSun: Vec3 = [0.99 / Math.hypot(0.99, 0.1), 0, -0.1 / Math.hypot(0.99, 0.1)];
  try {
    renderer = await Renderer.create(canvas, [batch], {
      worldQueries: { maxRays: 4096, maxTrianglesPerMesh: 4, maxInstancesPerBatch: 1 },
      diffuseGi: { origin: [-0.5, -0.5, 0.25], spacing: [1, 1, 0.5],
        counts: [2, 2, 2], raysPerProbe: 16, maxDistance: 4,
        skyRadiance: [0, 0, 0], finalGather: { pixelSpacing: 8,
          raysPerAnchor: 2, maxAnchors: 1024, maxHistory: 4 } },
      antiAliasing: 'none',
    });
    const center = { x: size.width / 2, y: size.height / 2 };
    const render = async (frames: number) => {
      for (let i = 0; i < frames; i++) renderer!.render(eye, target);
      return (await renderer!.probeHdr([center]))[0];
    };
    renderer.setDiffuseGiEnabled(false);
    renderer.setSun({ direction: backSun, color: [0, 0, 0] });
    const dark = await render(1);
    renderer.setSun({ direction: backSun, color: [3, 3, 3] });
    const directOnly = await render(1);
    const hit = hits(await renderer.traceRays([ray(0, 0, 5, [0, 0, -1])]))[0];
    renderer.setDiffuseGiFinalGatherEnabled(false);
    renderer.setDiffuseGiEnabled(true);
    const worldGi = await render(4);
    renderer.setDiffuseGiFinalGatherEnabled(true);
    const finalGather = await render(4);
    const errors = [...renderer.errors];
    renderer.dispose();
    return { backSun, hit, dark, directOnly, worldGi, finalGather,
      errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Exercises a finite, representable f32 triangle whose cross-product length
 * would underflow if a normalizer squared it before scaling. */
export async function runMaterialNormalTinyTriangleFixture() {
  const extent = 1e-14;
  const tiny: Vec3[] = [[-extent, -extent, 0], [extent, -extent, 0], [0, extent, 0]];
  const vertices = new ArrayBuffer(3 * 80), view = new DataView(vertices);
  tiny.forEach((position, vertex) => {
    const at = vertex * 80;
    position.forEach((value, axis) => view.setFloat32(at + axis * 4, value, true));
    view.setFloat32(at + 12, 1, true);
    view.setFloat32(at + 24, 1, true);
    view.setUint32(at + 32, 0, true);
    view.setFloat32(at + 48, 1, true);
  });
  const mesh: MeshData = { vertices, vertexCount: 3, indices: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0.5, 1]),
    tangents: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
    bounds: { min: [-extent, -extent, 0], max: [extent, extent, 0] } };
  const batch: SceneBatch = { name: 'normal-tiny-triangle', mesh, capacity: 1,
    inputMode: 'committed', instances: [instance(607)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'normal-tiny-triangle', primitiveId: 0, topologyVersion: 1 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      normal: m.constant([0.5, 0.5, 1, 1]) } } };
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], {
      worldQueries: { maxRays: 1, maxTrianglesPerMesh: 1, maxInstancesPerBatch: 1 },
    });
    renderer.render([0, 0, 2], [0, 0, 0]);
    const hit = hits(await renderer.traceRays([ray(0, 0, 1, [0, 0, -1])]))[0];
    const errors = [...renderer.errors];
    renderer.dispose();
    return { extent, hit, errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** Uniform sky isolates the exact cosine-hemisphere visibility fraction:
 * normal·geometricNormal = 1/2 gives (1 + 1/2)/2 = 3/4. */
export async function runMaterialNormalGatherHemisphereFixture() {
  const sky: Vec3 = [0.2, 0.3, 0.4], base: Vec3 = [0.6, 0.5, 0.4];
  const batch: SceneBatch = { name: 'normal-hemisphere-receiver', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [instance(608)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'normal-hemisphere-receiver', primitiveId: 0, topologyVersion: 1 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      baseColor: m.constant([...base, 1]), metallic: 0, roughness: 1,
      normal: m.constant([0.5 + Math.sqrt(3) / 4, 0.5, 0.75, 1]) } } };
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  const points = [{ x: 62, y: 62 }, { x: 66, y: 62 }, { x: 64, y: 64 },
    { x: 62, y: 66 }, { x: 66, y: 66 }];
  try {
    renderer = await Renderer.create(canvas, [batch], {
      worldQueries: { maxRays: 8192, maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1 },
      diffuseGi: { origin: [-1, -1, 0.25], spacing: [2, 2, 1], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance: 8, skyRadiance: sky, multiBounce: false,
        finalGather: { pixelSpacing: 4, maxAnchors: 1024,
          raysPerAnchor: 8, maxHistory: 128 } },
      antiAliasing: 'none',
    });
    renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
    for (let frame = 0; frame < 160; frame++) renderer.render(eye, target);
    const hdr = await renderer.probeHdr(points);
    const hit = hits(await renderer.traceRays([ray(0, 0, 1, [0, 0, -1])]))[0];
    const diagnostics = renderer.diagnostics().diffuseGi?.finalGather;
    if (!diagnostics) throw new Error('NORMAL_HEMISPHERE_NO_GATHER_DIAGNOSTIC');
    const anchorIndices = [15 + 15 * 32, 16 + 15 * 32, 15 + 16 * 32, 16 + 16 * 32];
    const anchors = (await renderer.probeDiffuseGiFinalGather(anchorIndices)).anchors;
    const errors = [...renderer.errors];
    renderer.dispose();
    return { sky, base, points, hdr, hit, diagnostics, anchors, errors,
      ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** A real meshoptimizer hierarchy selects fewer raster triangles while world
 * queries retain the authored source UV/tangent frame. */
export async function runMaterialNormalLodFixture() {
  const cells = 16, side = cells + 1, half = 2;
  const vertices = new ArrayBuffer(side * side * 80), view = new DataView(vertices);
  const uvs = new Float32Array(side * side * 2), tangents = new Float32Array(side * side * 4);
  const indices = new Uint32Array(cells * cells * 6);
  for (let y = 0; y <= cells; y++) for (let x = 0; x <= cells; x++) {
    const vertex = y * side + x, at = vertex * 80;
    view.setFloat32(at, x * 2 * half / cells - half, true);
    view.setFloat32(at + 4, y * 2 * half / cells - half, true);
    view.setFloat32(at + 12, 1, true);
    view.setFloat32(at + 24, 1, true);
    view.setUint32(at + 32, 0, true);
    view.setFloat32(at + 48, 1, true);
    uvs.set([x / cells, y / cells], vertex * 2);
    tangents.set([1, 0, 0, 1], vertex * 4);
  }
  for (let y = 0, at = 0; y < cells; y++) for (let x = 0; x < cells; x++) {
    const a = y * side + x, c = a + side;
    indices.set([a, a + 1, c + 1, a, c + 1, c], at);
    at += 6;
  }
  const mesh: MeshData = { vertices, indices, vertexCount: side * side, uvs, tangents,
    bounds: { min: [-half, -half, 0], max: [half, half, 0] } };
  const hierarchy = await buildGeometryHierarchy({ primitiveKey: 'normal-lod-grid', mesh,
    coverage: 'opaque' }, { maxClusterTriangles: 24, maxClusterVertices: 48,
    groupSize: 4, simplifyRatio: 0.5, maxLevels: 8 });
  const rootTriangles = hierarchy.roots.reduce((sum, id) =>
    sum + hierarchy.clusters[id].triangles.length / 3, 0);
  if (rootTriangles >= hierarchy.sourceTriangleCount) {
    throw new Error(`NORMAL_LOD_HIERARCHY_NOT_REDUCED ${rootTriangles}/${hierarchy.sourceTriangleCount}`);
  }
  const checker: TextureAsset = { width: 2, height: 1, colorSpace: 'linear',
    pixels: new Uint8Array([191, 128, 238, 255, 64, 128, 238, 255]) };
  const batch: SceneBatch = { name: 'normal-lod-grid', mesh: hierarchy.source,
    geometryLod: hierarchy, capacity: 1, inputMode: 'committed',
    instances: [instance(609)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    queryIdentity: { assetId: 'normal-lod-grid', primitiveId: 0, topologyVersion: 1 },
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: false,
      baseColor: m.constant([0.65, 0.45, 0.3, 1]), metallic: 0, roughness: 1,
      normal: m.texture(0), normalScale: m.component(m.parameter(0), 0) },
      parameters: new Float32Array([0, 0, 0, 0]), textures: [checker],
      samplers: [{ minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest',
        addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', lodMaxClamp: 0 }] } };
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:384px;height:384px;';
  document.body.appendChild(canvas);
  const lodEye: Vec3 = [0, 0, 12], lodTarget: Vec3 = [0, 0, 0];
  const world: Vec3[] = [[-0.8, 0.17, 0], [0.8, 0.17, 0]];
  const projection = multiply(perspective(Math.PI / 3.2, 1, 0.1, 150),
    lookAt(lodEye, lodTarget, [0, 1, 0]));
  const pixels = world.map(point => {
    const ndc = transformPoint(projection, point);
    return { x: Math.floor((ndc[0] * 0.5 + 0.5) * 384),
      y: Math.floor((1 - ndc[1] * 0.5 - 0.5) * 384) };
  });
  const pixelRaysLod: WorldRay[] = pixels.map(pixel => {
    const halfHeight = 12 * Math.tan(Math.PI / 6.4);
    const direction: Vec3 = [((pixel.x + 0.5) / 384 * 2 - 1) * halfHeight,
      (1 - (pixel.y + 0.5) / 384 * 2) * halfHeight, -12];
    const length = Math.hypot(...direction);
    return { ...ray(0, 0, 12, direction.map(value => value / length) as Vec3), tMax: 20 };
  });
  const rays: WorldRay[] = [{ ...ray(-0.8, 0.17, 12, [0, 0, -1]), tMax: 20 },
    { ...ray(0.8, 0.17, 12, [0, 0, -1]), tMax: 20 }, ...pixelRaysLod];
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], {
      worldQueries: { maxRays: 4, maxTrianglesPerMesh: hierarchy.sourceTriangleCount,
        maxInstancesPerBatch: 1 }, antiAliasing: 'none',
    });
    renderer.setGeometryLod({ maxErrorPixels: 10_000, frustumCulling: false,
      frozen: false, debug: 'material' });
    const capture = async (sun: Vec3) => {
      renderer!.setSun({ direction: sunDirection, color: sun });
      renderer!.render(lodEye, lodTarget);
      return { lod: await renderer!.probeGeometryLod(batch.name),
        hdr: await renderer!.probeHdr(pixels),
        depth: await renderer!.probeDepth(world),
        hits: hits(await renderer!.traceRays(rays, { textureLod: 0 })),
        variants: renderer!.diagnostics().materialVariants };
    };
    const neutral = await capture(sunColor), neutralDark = await capture([0, 0, 0]);
    renderer.setMaterialParameters(batch.name, new Float32Array([1, 0, 0, 0]));
    const tilted = await capture(sunColor), tiltedDark = await capture([0, 0, 0]);
    const errors = [...renderer.errors];
    renderer.dispose();
    return { sourceTriangles: hierarchy.sourceTriangleCount, rootTriangles, pixels,
      sunDirection, sunColor, neutral, neutralDark, tilted, tiltedDark,
      errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
