import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

const script = resolve('scripts/apply-fbx-material-map.mjs');
const tools = await import(pathToFileURL(script).href) as {
  textureRole: (file: string, convention: string) => string;
  alphaDecision: (pixels: Uint8Array) => { alphaMode: string; fullyCut: boolean; minAlpha: number; cutTexels: number };
};

/** 4x4 legacy DDS made of one block. */
function dds(fourCC: string, block: number[]): Buffer {
  const header = Buffer.alloc(128);
  header.write('DDS ', 0, 'ascii'); header.writeUInt32LE(124, 4); header.writeUInt32LE(0x81007, 8);
  header.writeUInt32LE(4, 12); header.writeUInt32LE(4, 16); header.writeUInt32LE(block.length, 20); header.writeUInt32LE(1, 28);
  header.writeUInt32LE(32, 76); header.writeUInt32LE(4, 80); header.write(fourCC, 84, 'ascii'); header.writeUInt32LE(0x1000, 108);
  return Buffer.concat([header, Buffer.from(block)]);
}
const OPAQUE_DXT1 = [0x00, 0xf8, 0x1f, 0x00, 0, 0, 0, 0];
function alphaBlock(a0: number, a1: number, indices: number[]): number[] {
  let bits = 0n;
  indices.forEach((index, i) => { bits |= BigInt(index) << BigInt(3 * i); });
  return [a0, a1, ...Array.from({ length: 6 }, (_, k) => Number((bits >> BigInt(8 * k)) & 255n))];
}
// DXT5: alpha endpoints 255/0 with indices 0 and 1 alternating (8 opaque, 8 cut); color block red.
const HALF_ALPHA_DXT5 = [...alphaBlock(255, 0, Array.from({ length: 16 }, (_, i) => i & 1)), ...OPAQUE_DXT1];

describe('FBX material map', () => {
  it('classifies texture roles and alpha by the declared convention', () => {
    expect(tools.textureRole('/a/Wall_Brick_BaseColor.dds', 'orca-metal-rough')).toBe('baseColor');
    expect(tools.textureRole('/a/x_Specular.dds', 'orca-metal-rough')).toBe('occlusionRoughnessMetalness');
    expect(() => tools.textureRole('/a/x_Height.dds', 'orca-metal-rough')).toThrow('FBX_MATERIAL_TEXTURE_ROLE_UNKNOWN: x_Height.dds');
    expect(tools.alphaDecision(new Uint8Array([1, 2, 3, 255, 4, 5, 6, 255]))).toMatchObject({ alphaMode: 'OPAQUE', fullyCut: false });
    expect(tools.alphaDecision(new Uint8Array([0, 0, 0, 10, 0, 0, 0, 100]))).toMatchObject({ alphaMode: 'MASK', fullyCut: true, minAlpha: 10, cutTexels: 2 });
  });

  it('rewrites exported materials and publishes a self-contained directory with hard-linked textures', async () => {
    const root = await mkdtemp(join(tmpdir(), 'real-engine-fbx-map-'));
    try {
      const source = join(root, 'source'), blender = join(root, 'blender');
      await mkdir(join(source, 'Textures'), { recursive: true }); await mkdir(blender);
      const textures: Record<string, Buffer> = {
        'Stone_BaseColor.dds': dds('DXT1', OPAQUE_DXT1), 'Stone_Specular.dds': dds('DXT1', OPAQUE_DXT1),
        'Stone_Normal.dds': dds('ATI2', new Array(16).fill(128)), 'Leaf_BaseColor.dds': dds('DXT5', HALF_ALPHA_DXT5),
        'Leaf_Emissive.dds': dds('DXT1', OPAQUE_DXT1), 'Leaf_Height.dds': dds('DXT1', OPAQUE_DXT1),
      };
      for (const [name, bytes] of Object.entries(textures)) await writeFile(join(source, 'Textures', name), bytes);
      await writeFile(join(source, 'LICENSE.txt'), 'license text\n');
      const file = (name: string) => join(source, 'Textures', name);
      const report = (leafTextures: string[]) => ({ version: 1, format: 'real-engine-blender-fbx-report', blender: 'test', source: { file: 'scene.fbx', bytes: 1, sha256: '0' },
        objects: { MESH: 1 }, meshCount: 1, uniqueTriangles: 1, instanceExpandedTriangles: 1, sharedMeshes: 0, units: {}, boundsGltf: {}, lights: [],
        cameras: [{ name: 'Camera', fps: 24, frameRange: [1, 2], verticalFovRadians: 0.75, clip: [0.1, 100],
          path: [{ frame: 1, time: 0, position: [0, 1, 0], forward: [0, 0, -1], up: [0, 1, 0] }] }],
        materials: {
          Stone: { blendMethod: 'HASHED', textures: ['Stone_BaseColor.dds', 'Stone_BaseColor.dds', 'Stone_Specular.dds', 'Stone_Normal.dds'].map(n => ({ file: file(n) })), principled: {} },
          'Leaf.DoubleSided': { blendMethod: 'HASHED', textures: leafTextures.map(n => ({ file: file(n) })), principled: { Metallic: { value: 0.25 }, Roughness: { value: 0.75 } } },
        } });
      await writeFile(join(blender, 'scene.bin'), Buffer.alloc(12));
      await writeFile(join(blender, 'scene.gltf'), JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri: 'scene.bin', byteLength: 12 }],
        extensionsUsed: ['KHR_materials_specular', 'KHR_lights_punctual'], extensionsRequired: ['KHR_lights_punctual'],
        materials: [{ name: 'Stone', alphaMode: 'BLEND', doubleSided: true }, { name: 'Leaf.DoubleSided', alphaMode: 'BLEND', doubleSided: true }],
        meshes: [{ primitives: [{ attributes: {}, material: 0 }, { attributes: {}, material: 1 }] }], nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0 }));
      const run = (output: string) => execFileSync(process.execPath, [script, '--input', blender, '--source-root', source, '--output', output,
        '--license', join(source, 'LICENSE.txt')], { cwd: resolve('.'), stdio: ['ignore', 'pipe', 'pipe'], timeout: 60_000 }).toString();

      await writeFile(join(blender, 'blender-report.json'), JSON.stringify(report(['Leaf_BaseColor.dds', 'Leaf_Height.dds'])));
      expect(() => run(join(root, 'failed'))).toThrow(/FBX_MATERIAL_TEXTURE_ROLE_UNKNOWN: Leaf_Height.dds/);
      expect((await readdir(root)).sort()).toEqual(['blender', 'source']);

      await writeFile(join(blender, 'blender-report.json'), JSON.stringify(report(['Leaf_BaseColor.dds', 'Leaf_Emissive.dds'])));
      const output = join(root, 'out');
      run(output);
      const gltf = JSON.parse(await readFile(join(output, 'scene.gltf'), 'utf8'));
      expect(gltf.extensionsUsed).toEqual(['KHR_lights_punctual']);
      expect(gltf.extensionsRequired).toBeUndefined();
      expect(gltf.images.map((image: { uri: string }) => image.uri)).toEqual(['Textures/Stone_BaseColor.dds', 'Textures/Stone_Specular.dds',
        'Textures/Stone_Normal.dds', 'Textures/Leaf_BaseColor.dds', 'Textures/Leaf_Emissive.dds']);
      expect(gltf.materials[0]).toEqual({ name: 'Stone', alphaMode: 'OPAQUE', doubleSided: false, normalTexture: { index: 2, scale: 1 },
        occlusionTexture: { index: 1, strength: 1 }, pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], baseColorTexture: { index: 0 },
          metallicFactor: 1, roughnessFactor: 1, metallicRoughnessTexture: { index: 1 } } });
      expect(gltf.materials[1]).toEqual({ name: 'Leaf.DoubleSided', alphaMode: 'MASK', alphaCutoff: 0.5, doubleSided: true,
        emissiveTexture: { index: 4 }, emissiveFactor: [1, 1, 1], pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1],
          baseColorTexture: { index: 3 }, metallicFactor: 0.25, roughnessFactor: 0.75 } });
      expect((await stat(join(output, 'Textures', 'Stone_Normal.dds'))).ino).toBe((await stat(file('Stone_Normal.dds'))).ino);
      const conversion = JSON.parse(await readFile(join(output, 'conversion.json'), 'utf8'));
      expect(conversion.counts).toMatchObject({ materials: 2, images: 5, alphaModes: { OPAQUE: 1, MASK: 1 }, primitives: 2 });
      expect(conversion.materials[1].alpha).toMatchObject({ texels: 16, cutTexels: 8, opaqueTexels: 8, alphaMode: 'MASK' });
      expect(conversion.licenses[0]).toMatchObject({ file: 'LICENSE.txt', bytes: 13 });
      const path = JSON.parse(await readFile(join(output, 'camera-path.json'), 'utf8'));
      expect(path).toMatchObject({ format: 'real-engine-camera-path', fps: 24, samples: [{ frame: 1 }] });
      expect((await readdir(output)).sort()).toEqual(['LICENSE.txt', 'Textures', 'blender-report.json', 'camera-path.json', 'conversion.json',
        'manifest.json', 'scene.bin', 'scene.gltf']);
    } finally { await rm(root, { recursive: true, force: true }); }
  }, 120_000);
});
