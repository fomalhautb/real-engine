import type { SceneBatch } from '../../../src/contracts';
import { importGltf, prepareGltfTangents } from '../../../src/gltf-import';
import { identity, lookAt, multiply, perspective, scaling, transformPoint } from '../../../src/math';
import { Renderer } from '../../../src/renderer';
import type { WorldHit, WorldRay } from '../../../src/world-query';

type Vec3 = [number, number, number];
const size = { width: 512, height: 384 };
const eye: Vec3 = [0, 0, 4];
const sunDirection: Vec3 = [0.8, 0, 0.6];
const sunColor: Vec3 = [2, 2, 2];
const mapPixel = [230, 153, 209, 255];

/** A real, in-memory glTF 2.0 primitive with indexed mirrored UVs and all
 * relevant side streams. No TANGENT accessor is supplied. */
function mirroredGltf() {
  const chunks: Uint8Array[] = [];
  const bufferViews: { buffer: number; byteOffset: number; byteLength: number }[] = [];
  const accessors: { bufferView: number; componentType: number; count: number; type: string }[] = [];
  let length = 0;
  const add = (data: Float32Array | Uint16Array | Uint8Array, type: string, count: number,
    componentType = data instanceof Float32Array ? 5126 : data instanceof Uint16Array ? 5123 : 5121) => {
    const aligned = (length + 3) & ~3;
    if (aligned !== length) chunks.push(new Uint8Array(aligned - length));
    length = aligned;
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    chunks.push(bytes);
    bufferViews.push({ buffer: 0, byteOffset: length, byteLength: bytes.byteLength });
    accessors.push({ bufferView: bufferViews.length - 1, componentType, count, type });
    length += bytes.byteLength;
    return accessors.length - 1;
  };
  const position = add(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 'VEC3', 4);
  const normal = add(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), 'VEC3', 4);
  // The second triangle swaps U and V, so the same shared corners need
  // incompatible per-triangle tangent frames and handedness.
  const uv = add(new Float32Array([0, 0, 1, 0, 0, 1, 1, 0]), 'VEC2', 4);
  const joints = add(new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]), 'VEC4', 4);
  const weights = add(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0]), 'VEC4', 4);
  const indices = add(new Uint16Array([0, 1, 2, 0, 2, 3]), 'SCALAR', 6);
  const morphPosition = add(new Float32Array([-0, -0, -0.3, 0, 0, 0.1,
    0, 0, 0.3, 0, 0, -0.1]), 'VEC3', 4);
  const morphNormal = add(new Float32Array(Array.from({ length: 4 }, () => {
    const magnitude = Math.hypot(0.2, 0.1, 1);
    return [-0.2 / magnitude, -0.1 / magnitude, 1 / magnitude - 1];
  }).flat()), 'VEC3', 4);
  const buffer = new Uint8Array(length);
  let offset = 0;
  for (const bytes of chunks) { buffer.set(bytes, offset); offset += bytes.length; }
  const json = {
    asset: { version: '2.0' }, buffers: [{ byteLength: length }], bufferViews, accessors,
    images: [{ uri: 'fixture-normal.png' }], textures: [{ source: 0 }],
    materials: [{ name: 'generated-normal', doubleSided: true,
      pbrMetallicRoughness: { baseColorFactor: [0.65, 0.45, 0.3, 1], metallicFactor: 0, roughnessFactor: 1 },
      normalTexture: { index: 0, scale: 0 } }],
    meshes: [{ name: 'mirrored-uv', primitives: [{
      attributes: { POSITION: position, NORMAL: normal, TEXCOORD_0: uv,
        JOINTS_0: joints, WEIGHTS_0: weights }, indices, material: 0,
      targets: [{ POSITION: morphPosition, NORMAL: morphNormal }],
    }] }],
    nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0,
  };
  return { json, buffer: buffer.buffer };
}

function hitList(hits: (WorldHit | null)[]): WorldHit[] {
  return hits.map((hit, i) => {
    if (!hit) throw new Error(`GLTF_TANGENTS_MISSING_HIT_${i}`);
    return hit;
  });
}

const rays: WorldRay[] = [
  { origin: [0.55, -0.315, 4], direction: [0, 0, -1], tMax: 10 },
  { origin: [-0.55, 0.315, 4], direction: [0, 0, -1], tMax: 10 },
];

function project(point: Vec3): { x: number; y: number } {
  const projection = perspective(Math.PI / 3.2, size.width / size.height, 0.1, 150);
  const ndc = transformPoint(multiply(projection, lookAt(eye, [0, 0, 0], [0, 1, 0])), point);
  return { x: Math.floor((ndc[0] * 0.5 + 0.5) * size.width),
    y: Math.floor((1 - (ndc[1] * 0.5 + 0.5)) * size.height) };
}

function pixelRay(pixel: { x: number; y: number }): WorldRay {
  const ndcX = (pixel.x + 0.5) / size.width * 2 - 1;
  const ndcY = 1 - (pixel.y + 0.5) / size.height * 2;
  const halfHeight = 4 * Math.tan(Math.PI / 6.4);
  const direction = [ndcX * halfHeight * size.width / size.height, ndcY * halfHeight, -4];
  const length = Math.hypot(...direction);
  return { origin: eye, direction: direction.map(value => value / length) as Vec3, tMax: 10 };
}

export async function runGltfTangentsFixture() {
  await prepareGltfTangents();
  const source = mirroredGltf();
  const asset = importGltf(source.json, [source.buffer], {
    images: new Map([[0, { width: 1, height: 1, colorSpace: 'linear' as const,
      pixels: new Uint8Array(mapPixel) }]]),
  });
  const primitive = asset.meshes[0].primitives[0];
  const mesh = primitive.mesh;
  const view = new DataView(mesh.vertices);
  const pixels = [project([0.55, -0.315, 0.078]), project([-0.55, 0.315, -0.078])];
  const allRays = [...rays, ...pixels.map(pixelRay)];
  const vertices = Array.from({ length: mesh.vertexCount }, (_, i) => ({
    position: [0, 1, 2].map(axis => view.getFloat32(i * 80 + axis * 4, true)),
    normal: [0, 1, 2].map(axis => view.getFloat32(i * 80 + 16 + axis * 4, true)),
    joints: [0, 1, 2, 3].map(axis => view.getUint32(i * 80 + 32 + axis * 4, true)),
    weights: [0, 1, 2, 3].map(axis => view.getFloat32(i * 80 + 48 + axis * 4, true)),
    uv: Array.from(mesh.uvs!.subarray(i * 2, i * 2 + 2)),
    tangent: Array.from(mesh.tangents!.subarray(i * 4, i * 4 + 4)),
    morphPosition: Array.from(primitive.morphTargets[0].positions.subarray(i * 3, i * 3 + 3)),
    morphNormal: Array.from(primitive.morphTargets[0].normals.subarray(i * 3, i * 3 + 3)),
    morphTangent: Array.from(primitive.morphTargets[0].tangents!.subarray(i * 3, i * 3 + 3)),
  }));
  const instance = { entityId: 1701, model: identity(), normalMatrix: identity(),
    bones: [scaling(1.1, 0.9, 1.2)], deformScalars: [1],
    color: [1, 1, 1, 1] as [number, number, number, number],
    metalness: 1, roughness: 1, historyVersion: 0 };
  const batch: SceneBatch = { name: 'gltf-generated-mirrored', mesh, capacity: 1,
    inputMode: 'committed', instances: [instance], material: primitive.material,
    deformation: { jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 },
      { kind: 'skin', normalMode: 'affine' }] }, morphTargets: primitive.morphTargets,
    queryIdentity: { assetId: 'gltf-generated-mirrored', primitiveId: 0, topologyVersion: 1 } };
  const canvas = document.createElement('canvas');
  canvas.width = size.width; canvas.height = size.height;
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${size.width}px;height:${size.height}px;z-index:1000`;
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], {
      worldQueries: { maxRays: 8, maxTrianglesPerMesh: 2, maxInstancesPerBatch: 1 },
      antiAliasing: 'none',
    });
    renderer.setSun({ direction: sunDirection, color: sunColor });
    const capture = async () => {
      renderer!.render(eye, [0, 0, 0]);
      const png = canvas.toDataURL('image/png');
      const hits = hitList((await renderer!.traceRays(allRays, { textureLod: 0 })).hits);
      const hdr = await renderer!.probeHdr(pixels);
      return { png, hits, hdr, variants: renderer!.diagnostics().materialVariants };
    };
    const neutral = await capture();
    const mappedParameters = new Float32Array(primitive.material.parameters!);
    mappedParameters[12] = 1;
    renderer.setMaterialParameters(batch.name, mappedParameters);
    const mapped = await capture();
    renderer.setSun({ direction: sunDirection, color: [0, 0, 0] });
    const dark = await capture();
    const verification = await renderer.verifyDeformation({ allInstances: true });
    const errors = [...renderer.errors];
    renderer.dispose();
    return { size, pixels, vertices, indices: Array.from(mesh.indices), warnings: asset.warnings,
      neutral, mapped, dark, verification, sunDirection, sunColor, mapPixel, errors,
      ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
