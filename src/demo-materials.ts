import type { MaterialBinding } from './contracts';
import { m, type MaterialDefinition } from './material';
import type { TextureAsset } from './texture-assets';

function image(width: number, height: number, pixel: (x: number, y: number) => readonly [number, number, number, number]): TextureAsset {
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) pixels.set(pixel(x, y), (y * width + x) * 4);
  }
  return { width, height, pixels, colorSpace: 'srgb' };
}

/** One repeatable two-tone tile with thin grid lines, authored in code. */
export function createGroundTexture(): TextureAsset {
  return image(64, 64, (x, y) => {
    const line = x < 2 || y < 2 || x >= 62 || y >= 62;
    if (line) return [82, 119, 128, 255];
    const checker = (x < 32) !== (y < 32);
    return checker ? [67, 93, 102, 255] : [94, 117, 122, 255];
  });
}

/** Original cutout ring. Alpha is binary so the same mask works in shadow and surface passes. */
export function createCardTexture(): TextureAsset {
  return image(64, 64, (x, y) => {
    const dx = (x + 0.5 - 32) / 32, dy = (y + 0.5 - 32) / 32;
    const radius = Math.hypot(dx, dy);
    const mark = radius >= 0.34 && radius <= 0.79;
    if (!mark) return [0, 0, 0, 0];
    const spoke = Math.abs(dx) < 0.07 || Math.abs(dy) < 0.07;
    return spoke ? [255, 231, 151, 255] : [90, 221, 230, 255];
  });
}

const texturedColor = m.multiply(m.multiply(m.parameter(0), m.instanceColor), m.texture(0));
const black = m.constant([0, 0, 0, 0]);
const metalness = m.component(m.instanceMetalRough, 0);
const roughness = m.component(m.instanceMetalRough, 1);

export const GROUND_MATERIAL_DEFINITION: MaterialDefinition = {
  shading: 'pbr', alphaMode: 'opaque', doubleSided: false,
  baseColor: texturedColor, metallic: metalness, roughness,
  emissive: black, alphaCutoff: 0.5,
};

export const CARD_MATERIAL_DEFINITION: MaterialDefinition = {
  shading: 'unlit', alphaMode: 'mask', doubleSided: true,
  baseColor: texturedColor, metallic: 0, roughness: 1,
  emissive: black, alphaCutoff: 0.5,
};

export function createGroundMaterial(): MaterialBinding {
  return {
    definition: GROUND_MATERIAL_DEFINITION,
    parameters: new Float32Array([1, 1, 1, 1]),
    textures: [createGroundTexture()],
  };
}

export function createCardMaterial(): MaterialBinding {
  return {
    definition: CARD_MATERIAL_DEFINITION,
    parameters: new Float32Array([1, 1, 1, 1]),
    textures: [createCardTexture()],
  };
}
