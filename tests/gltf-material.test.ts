import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { importGltf, prepareGltfTangents } from '../src/gltf-import';
import { compileMaterial, evaluateMaterial } from '../src/material';
import type { TextureAsset } from '../src/texture-assets';
import type { GltfBinaryDocument } from '../src/gltf-binary';
import { createGltfMaterialLibrary } from '../src/gltf-material';
import type { PreparedTextureAsset } from '../src/texture-storage';

const directory = new URL('../public/assets/gltf/AnimatedMorphCube/glTF/', import.meta.url);
const source = readFileSync(new URL('AnimatedMorphCube.gltf', directory), 'utf8');
const binary = Uint8Array.from(readFileSync(new URL('AnimatedMorphCube.bin', directory))).buffer;
const image: TextureAsset = { width: 1, height: 1, pixels: new Uint8Array([128, 128, 128, 128]), colorSpace: 'srgb' };

function texturedFixture() {
  const json = JSON.parse(source) as {
    materials: Array<Record<string, unknown>>;
    images?: unknown[];
    textures?: unknown[];
    samplers?: unknown[];
  };
  json.images = [{ uri: 'same.png' }];
  json.textures = [{ source: 0, sampler: 0 }, { source: 0, sampler: 1 }, { source: 0 }];
  json.samplers = [
    { wrapS: 33071, wrapT: 33648, minFilter: 9728, magFilter: 9728 },
    {}, // glTF defaults: repeat wrapping and linear mip filtering.
  ];
  json.materials[0] = {
    pbrMetallicRoughness: {
      baseColorFactor: [0.5, 0.25, 1, 0.8], baseColorTexture: { index: 0 },
      roughnessFactor: 0.5, metallicFactor: 0.6, metallicRoughnessTexture: { index: 1 },
    },
    emissiveFactor: [0.2, 0.4, 0.6], emissiveTexture: { index: 2 },
    alphaMode: 'MASK', alphaCutoff: 0.35, doubleSided: true,
  };
  return json;
}

function withQuadUvs<T extends GltfBinaryDocument & {
  meshes: Array<{ primitives: Array<{ attributes: Record<string, number> }> }>;
}>(json: T): { json: T; buffers: ArrayBuffer[] } {
  // AnimatedMorphCube has four distinct vertices per face but no TEXCOORD_0.
  const uv = new Float32Array(24 * 2), quad = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (let vertex = 0; vertex < 24; vertex++) uv.set(quad[vertex % 4], vertex * 2);
  const buffer = json.buffers!.length, bufferView = json.bufferViews!.length, accessor = json.accessors!.length;
  json.buffers!.push({ byteLength: uv.byteLength });
  json.bufferViews!.push({ buffer, byteLength: uv.byteLength });
  json.accessors!.push({ bufferView, componentType: 5126, count: 24, type: 'VEC2' });
  json.meshes[0].primitives[0].attributes.TEXCOORD_0 = accessor;
  return { json, buffers: [binary, uv.buffer] };
}

describe('glTF material import', () => {
  it('maps color spaces, sampler defaults, PBR channels, alpha coverage, and double-sided state', () => {
    const material = importGltf(texturedFixture(), [binary], { images: new Map([[0, image]]) })
      .meshes[0].primitives[0].material;
    expect(material.definition).toMatchObject({ shading: 'pbr', alphaMode: 'mask', doubleSided: true });
    expect(material.textures?.map(texture => texture?.colorSpace)).toEqual(['srgb', 'linear', 'srgb']);
    expect(material.textures?.[0]).toBe(material.textures?.[2]); // Same decoded image and color-space view.
    expect(material.textures?.[0]).not.toBe(material.textures?.[1]);
    expect(image.colorSpace).toBe('srgb');
    expect(material.samplers?.[0]).toMatchObject({
      addressModeU: 'clamp-to-edge', addressModeV: 'mirror-repeat',
      minFilter: 'nearest', magFilter: 'nearest', mipmapFilter: 'nearest', lodMaxClamp: 0,
    });
    expect(material.samplers?.[1]).toMatchObject({
      addressModeU: 'repeat', addressModeV: 'repeat',
      minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear',
    });
    expect(material.samplers?.[2]).toBeUndefined();

    const program = compileMaterial(material.definition);
    const texture = (slot: number): [number, number, number, number] => {
      if (slot === 0) return [0.8, 0.4, 0.2, 0.5];
      if (slot === 1) return [0.9, 0.7, 0.3, 1];
      if (slot === 2) return [0.5, 0.5, 0.5, 1];
      throw new Error('unexpected texture slot');
    };
    const evaluated = evaluateMaterial(program, {
      parameters: material.parameters, textures: texture, instanceColor: [1, 1, 1, 1], instanceMetalRough: [1, 1],
    });
    for (const [actual, expected] of [
      [evaluated.baseColor[0], 0.4], [evaluated.baseColor[1], 0.1], [evaluated.baseColor[2], 0.2],
      [evaluated.opacity, 0.4], [evaluated.roughness, 0.35], [evaluated.metallic, 0.18],
      [evaluated.emissive[0], 0.1], [evaluated.emissive[1], 0.2], [evaluated.emissive[2], 0.3],
      [evaluated.alphaCutoff, 0.35],
    ]) expect(actual).toBeCloseTo(expected, 5);
    expect(evaluated.covered).toBe(true);
    expect(evaluateMaterial(program, {
      parameters: material.parameters,
      textures: slot => slot === 0 ? [0.8, 0.4, 0.2, 0.25] : texture(slot),
      instanceColor: [1, 1, 1, 1], instanceMetalRough: [1, 1],
    }).covered).toBe(false);
  });

  it('multiplies glTF baseColor factor and texture by interpolated vertex RGBA for color and coverage', () => {
    const material = importGltf(texturedFixture(), [binary], { images: new Map([[0, image]]) })
      .meshes[0].primitives[0].material;
    const program = compileMaterial(material.definition);
    const inputs = { parameters: material.parameters,
      textures: (slot: number): [number, number, number, number] => slot === 0 ? [0.8, 0.4, 0.2, 0.5] : [1, 1, 1, 1],
      instanceColor: [1, 1, 1, 1] as [number, number, number, number],
      instanceMetalRough: [1, 1] as [number, number] };
    const white = evaluateMaterial(program, inputs);
    const tinted = evaluateMaterial(program, { ...inputs, vertexColor: [0.25, 0.5, 1, 0.5] });
    expect(white.covered).toBe(true);
    expect(tinted.covered).toBe(false);
    for (const [actual, expected] of [
      [tinted.baseColor[0], 0.1], [tinted.baseColor[1], 0.05], [tinted.baseColor[2], 0.2],
      [tinted.opacity, 0.2],
    ]) expect(actual).toBeCloseTo(expected);
  });

  it('imports normalTexture as a linear slot-3 surface input with a live signed scale', async () => {
    await prepareGltfTangents();
    const { json, buffers: texturedBuffers } = withQuadUvs(texturedFixture() as ReturnType<typeof texturedFixture> & GltfBinaryDocument & {
      meshes: Array<{ primitives: Array<{ attributes: Record<string, number> }> }>;
    });
    json.materials[0].normalTexture = { index: 0, scale: -0.5 };
    const asset = importGltf(json, texturedBuffers, { images: new Map([[0, image]]) });
    const material = asset.meshes[0].primitives[0].material;
    const program = compileMaterial(material.definition);
    expect(program.normalMapped).toBe(true);
    expect(program.textureSlots).toEqual([0, 1, 2, 3]);
    expect(program.coverageTextureSlots).toEqual([0]);
    expect(program.parameterCount).toBe(4);
    expect(program.coverageParameterCount).toBe(2);
    expect(material.parameters).toHaveLength(16);
    expect(material.parameters?.[12]).toBe(-0.5);
    expect(material.textures?.[3]?.colorSpace).toBe('linear');
    expect(material.textures?.[3]).toBe(material.textures?.[1]);
    expect(material.samplers?.[3]).toMatchObject({
      addressModeU: 'clamp-to-edge', addressModeV: 'mirror-repeat',
      minFilter: 'nearest', magFilter: 'nearest', lodMaxClamp: 0,
    });
    expect(asset.warnings.some(warning => /triangle UV-derivative frame fallback/.test(warning))).toBe(false);
    const withoutTangents = texturedFixture() as ReturnType<typeof texturedFixture> & {
      meshes: Array<{ primitives: Array<{ attributes: Record<string, unknown>; targets?: Array<Record<string, unknown>> }> }>;
    };
    withoutTangents.materials[0].normalTexture = { index: 0 };
    delete withoutTangents.meshes[0].primitives[0].attributes.TANGENT;
    for (const target of withoutTangents.meshes[0].primitives[0].targets ?? []) delete target.TANGENT;
    const uvFixture = withQuadUvs(withoutTangents as typeof withoutTangents & GltfBinaryDocument & {
      meshes: Array<{ primitives: Array<{ attributes: Record<string, number> }> }>;
    });
    const generated = importGltf(uvFixture.json, uvFixture.buffers, { images: new Map([[0, image]]) });
    const generatedPrimitive = generated.meshes[0].primitives[0];
    expect(generatedPrimitive.mesh.tangents).toHaveLength(generatedPrimitive.mesh.vertexCount * 4);
    expect(generatedPrimitive.morphTargets.every(target => target.tangents?.length === generatedPrimitive.mesh.vertexCount * 3)).toBe(true);
    expect(generated.warnings.some(warning => /triangle UV-derivative frame fallback/.test(warning))).toBe(false);
    const textures = (slot: number): [number, number, number, number] => slot === 3
      ? [0.75, 0.5, 1, 1] : [1, 1, 1, 1];
    const evaluate = () => evaluateMaterial(program, { parameters: material.parameters, textures,
      instanceColor: [1, 1, 1, 1], instanceMetalRough: [1, 1] });
    expect(evaluate().tangentNormal[0]).toBeLessThan(0);
    material.parameters![12] = 0;
    expect(evaluate().tangentNormal).toEqual([0, 0, 1]);
  });

  it('rejects unsupported occlusion and normal texture coordinates', () => {
    const json = texturedFixture();
    json.materials[0].occlusionTexture = { index: 0 };
    expect(() => importGltf(json, [binary], { images: new Map([[0, image]]) }))
      .toThrow(/GLTF_UNSUPPORTED_MATERIAL: occlusion texture/);
    delete json.materials[0].occlusionTexture;
    json.materials[0].normalTexture = { index: 0, texCoord: 1 };
    expect(() => importGltf(json, [binary], { images: new Map([[0, image]]) }))
      .toThrow(/GLTF_UNSUPPORTED_TEXTURE_COORDINATES/);
  });

  it('requires UV0 for normal maps and enforces the tangent work budget', async () => {
    const missingUv = texturedFixture();
    missingUv.materials[0].normalTexture = { index: 0 };
    expect(() => importGltf(missingUv, [binary], { images: new Map([[0, image]]) }))
      .toThrow(/GLTF_NORMAL_TEXTURE_REQUIRES_UV0/);

    await prepareGltfTangents();
    const { json, buffers } = withQuadUvs(texturedFixture() as ReturnType<typeof texturedFixture> & GltfBinaryDocument & {
      meshes: Array<{ primitives: Array<{ attributes: Record<string, number>; targets?: Array<Record<string, number>> }> }>;
    });
    json.materials[0].normalTexture = { index: 0 };
    delete json.meshes[0].primitives[0].attributes.TANGENT;
    for (const target of json.meshes[0].primitives[0].targets ?? []) delete target.TANGENT;
    expect(() => importGltf(json, buffers, { images: new Map([[0, image]]), maxTangentBytes: 1 }))
      .toThrow(/GLTF_TANGENT_WORK_BUDGET/);
  });
});

const prepared = (colorSpace: 'srgb' | 'linear'): PreparedTextureAsset => ({
  kind: 'prepared', width: 2, height: 2, colorSpace, format: 'rgba8', baseMipLevel: 0,
  levels: [{ width: 2, height: 2, data: new Uint8Array(16) },
    { width: 1, height: 1, data: new Uint8Array(4) }],
});

const materialDocument = () => ({
  asset: { version: '2.0' }, images: [{ uri: 'shared.png' }],
  samplers: [
    { wrapS: 33071, wrapT: 33648, minFilter: 9728, magFilter: 9728 },
    { wrapS: 10497, wrapT: 10497, minFilter: 9987, magFilter: 9729 },
  ],
  textures: [{ source: 0, sampler: 0 }, { source: 0, sampler: 1 }],
  materials: [
    { pbrMetallicRoughness: { baseColorTexture: { index: 0 } } },
    { pbrMetallicRoughness: { baseColorTexture: { index: 1 }, metallicRoughnessTexture: { index: 0 } },
      normalTexture: { index: 0 } },
    { emissiveTexture: { index: 1 } },
  ],
});

describe('glTF material library without geometry', () => {
  it('shares an exact stream-fallback identity across materials while keeping samplers per texture', () => {
    const doc = materialDocument();
    const snapshot = JSON.stringify(doc);
    const srgb = prepared('srgb');
    const linear = prepared('linear');
    const library = createGltfMaterialLibrary(doc, { resolveImage: (_imageIndex, role) => role === 'srgb' ? srgb : linear });
    expect(library.materialCount).toBe(3);
    const first = library.get(0), second = library.get(1), third = library.get(2);
    expect(library.get(0)).toBe(first);
    expect(first.textures?.[0]).toBe(srgb);
    expect(second.textures?.[0]).toBe(srgb);
    expect(third.textures?.[2]).toBe(srgb);
    expect(second.textures?.[1]).toBe(linear);
    expect(second.textures?.[3]).toBe(linear);
    expect(first.samplers?.[0]).toMatchObject({ addressModeU: 'clamp-to-edge',
      addressModeV: 'mirror-repeat', minFilter: 'nearest', magFilter: 'nearest', lodMaxClamp: 0 });
    expect(second.samplers?.[0]).toMatchObject({ addressModeU: 'repeat',
      addressModeV: 'repeat', minFilter: 'linear', mipmapFilter: 'linear' });
    expect(JSON.stringify(doc)).toBe(snapshot);
    expect(srgb.levels[0].data).toHaveLength(16);
    expect(linear.levels[0].data).toHaveLength(16);
  });

  it('calls a resolver once per image and color role and preserves each returned asset exactly', () => {
    const doc = materialDocument();
    const srgb = prepared('srgb'), linear = prepared('linear');
    const calls: string[] = [];
    const library = createGltfMaterialLibrary(doc, { resolveImage: (index, role) => {
      calls.push(`${index}:${role}`);
      return role === 'srgb' ? srgb : linear;
    } });
    library.get(2);
    library.get(1);
    library.get(0);
    library.get(1);
    expect(calls).toEqual(['0:srgb', '0:linear']);
    expect(library.get(1).textures?.[1]).toBe(linear);
    expect(library.get(0).textures?.[0]).toBe(srgb);
  });

  it('passes prepared image resolution through the ordinary glTF importer', () => {
    const srgb = prepared('srgb'), linear = prepared('linear');
    const calls: string[] = [];
    const asset = importGltf(texturedFixture(), [binary], { resolveImage: (index, role) => {
      calls.push(`${index}:${role}`);
      return role === 'srgb' ? srgb : linear;
    } });
    const binding = asset.meshes[0].primitives[0].material;
    expect(binding.textures?.[0]).toBe(srgb);
    expect(binding.textures?.[1]).toBe(linear);
    expect(binding.textures?.[2]).toBe(srgb);
    expect(calls).toEqual(['0:srgb', '0:linear']);
  });

  it('validates prepared storage and color role without cloning or repairing it', () => {
    const doc = materialDocument();
    const mismatched = prepared('linear');
    expect(() => createGltfMaterialLibrary(doc, { images: new Map([[0, mismatched]]) }).get(0))
      .toThrow(/GLTF_IMAGE_COLOR_SPACE/);
    const malformed = prepared('srgb');
    (malformed.levels[1] as { width: number; height: number; data: Uint8Array }).data = new Uint8Array(3);
    expect(() => createGltfMaterialLibrary(doc, { images: new Map([[0, malformed]]) }).get(0))
      .toThrow(/TEXTURE_STORAGE/);
    expect(() => createGltfMaterialLibrary(doc).get(0)).toThrow(/GLTF_IMAGE_NOT_DECODED/);
  });

  it('does not fall back to the image map or reinterpret a resolver result with the wrong color space', () => {
    const doc = materialDocument();
    expect(() => createGltfMaterialLibrary(doc, { images: new Map([[0, image]]), resolveImage: () => undefined }).get(0))
      .toThrow(/GLTF_IMAGE_NOT_DECODED/);
    const decodedLinear: TextureAsset = { ...image, colorSpace: 'linear' };
    expect(() => createGltfMaterialLibrary(doc, { resolveImage: () => decodedLinear }).get(0))
      .toThrow(/GLTF_IMAGE_COLOR_SPACE/);
    expect(() => createGltfMaterialLibrary(doc, { resolveImage: () => prepared('linear') }).get(0))
      .toThrow(/GLTF_IMAGE_COLOR_SPACE/);
  });

  it('bounds metadata and leaves scene-only extension validation to the full importer', () => {
    expect(() => createGltfMaterialLibrary({ asset: { version: '1.0' } })).toThrow(/GLTF_VERSION/);
    expect(() => createGltfMaterialLibrary({ asset: { version: '2.0' }, images: new Array(65537) }))
      .toThrow(/GLTF_ARRAY_LIMIT/);
    const doc = { ...materialDocument(), extensionsRequired: ['KHR_lights_punctual'] };
    expect(createGltfMaterialLibrary(doc, { images: new Map([[0, image]]) }).get(0).definition.shading).toBe('pbr');
    const full = { ...texturedFixture(), extensionsRequired: ['KHR_lights_punctual'] };
    expect(() => importGltf(full, [binary], { images: new Map([[0, image]]) })).toThrow(/GLTF_REQUIRED_EXTENSION/);
    const invalidUnused = texturedFixture();
    invalidUnused.materials.push({ alphaMode: 'BLEND' });
    expect(() => importGltf(invalidUnused, [binary], { images: new Map([[0, image]]) })).toThrow(/GLTF_UNSUPPORTED_ALPHA_MODE/);
  });

  it('keeps decoded legacy pixels shared across roles and evaluates factors, normal scale, vertex color, and mask coverage', () => {
    const doc = materialDocument() as any;
    doc.materials[0] = {
      ...doc.materials[0], alphaMode: 'MASK', alphaCutoff: 0.3, doubleSided: true,
      pbrMetallicRoughness: { ...doc.materials[0].pbrMetallicRoughness,
        baseColorFactor: [0.5, 0.25, 1, 0.8], roughnessFactor: 0.5, metallicFactor: 0.6 },
      emissiveFactor: [0.2, 0.4, 0.6], emissiveTexture: { index: 1 },
      normalTexture: { index: 0, scale: -0.5 },
    };
    const decoded: TextureAsset = { width: 1, height: 1, colorSpace: 'srgb', pixels: new Uint8Array([255, 128, 64, 255]) };
    const library = createGltfMaterialLibrary(doc, { images: new Map([[0, decoded]]) });
    const material = library.get(0), other = library.get(1);
    expect(material.textures?.[0]).toBe(decoded);
    expect(material.textures?.[2]).toBe(decoded);
    expect(material.textures?.[3]).toBe(other.textures?.[1]);
    expect(material.textures?.[3]).not.toBe(decoded);
    expect((material.textures?.[3] as TextureAsset).pixels).toBe(decoded.pixels);
    expect(decoded.colorSpace).toBe('srgb');
    const program = compileMaterial(material.definition);
    expect(program).toMatchObject({ shading: 'pbr', alphaMode: 'mask', doubleSided: true, normalMapped: true });
    const inputs = { parameters: material.parameters,
      textures: (slot: number): [number, number, number, number] => slot === 0 ? [0.8, 0.4, 0.2, 0.5]
        : slot === 3 ? [0.75, 0.5, 1, 1] : [1, 1, 1, 1],
      instanceColor: [1, 1, 1, 1] as [number, number, number, number],
      instanceMetalRough: [1, 1] as [number, number] };
    const evaluated = evaluateMaterial(program, inputs);
    expect(evaluated.baseColor[0]).toBeCloseTo(0.4);
    expect(evaluated.roughness).toBeCloseTo(0.5);
    expect(evaluated.metallic).toBeCloseTo(0.6);
    evaluated.emissive.forEach((value, index) => expect(value).toBeCloseTo([0.2, 0.4, 0.6][index]));
    expect(evaluated.covered).toBe(true);
    expect(evaluated.tangentNormal[0]).toBeLessThan(0);
    expect(evaluateMaterial(program, { ...inputs, vertexColor: [1, 1, 1, 0.5] }).covered).toBe(false);
  });

  it('uses a null default and validates requested material, texture, and image indices lazily', () => {
    const doc = materialDocument();
    const library = createGltfMaterialLibrary(doc, { images: new Map([[0, image]]) });
    expect(library.get(null).definition).toMatchObject({ shading: 'pbr', alphaMode: 'opaque' });
    expect(library.get(null)).toBe(library.get(null));
    for (const index of [-1, 3, 0.5, NaN]) expect(() => library.get(index)).toThrow();
    const badTexture = materialDocument();
    badTexture.materials[0] = { pbrMetallicRoughness: { baseColorTexture: { index: 99 } } };
    expect(() => createGltfMaterialLibrary(badTexture, { images: new Map([[0, image]]) }).get(0)).toThrow();
    const badImage = materialDocument();
    badImage.textures[0].source = 99;
    expect(() => createGltfMaterialLibrary(badImage, { images: new Map([[0, image]]) }).get(0)).toThrow();
    const blendDoc = materialDocument();
    const blend = { ...blendDoc, materials: [...blendDoc.materials.slice(0, 2), { alphaMode: 'BLEND' }] };
    const lazy = createGltfMaterialLibrary(blend, { images: new Map([[0, image]]) });
    expect(lazy.get(0).definition.alphaMode).toBe('opaque');
    expect(() => lazy.get(2)).toThrow(/GLTF_UNSUPPORTED_ALPHA_MODE/);
  });
});
