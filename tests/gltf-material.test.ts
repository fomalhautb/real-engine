import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { importGltf } from '../src/gltf-import';
import { compileMaterial, evaluateMaterial } from '../src/material';
import type { TextureAsset } from '../src/texture-assets';

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

  it('rejects an unsupported normal texture instead of losing it silently', () => {
    const json = texturedFixture();
    json.materials[0].normalTexture = { index: 0 };
    expect(() => importGltf(json, [binary], { images: new Map([[0, image]]) }))
      .toThrow(/GLTF_UNSUPPORTED_MATERIAL: normal\/occlusion texture/);
  });
});
