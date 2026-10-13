import type { MaterialBinding, SceneBatch } from '../../../src/contracts';
import { identity, translation } from '../../../src/math';
import { m, type MaterialDefinition } from '../../../src/material';
import { parseObj } from '../../../src/mesh-import';
import { Renderer } from '../../../src/renderer';
import type { TextureAsset } from '../../../src/texture-assets';

const FRONT: [number, number, number] = [0, 0, 4];
const BACK: [number, number, number] = [0, 0, -4];
const TARGET: [number, number, number] = [0, 0, 0];
const PROBES: readonly [number, number, number][] = [[-0.5, 0, 0], [0.5, 0, 0]];
const WAVE_AMPLITUDE = 0.25;
const WAVE_FREQUENCY = Math.PI / 2;

const QUAD_OBJ = `
v -1 -1 0
v  1 -1 0
v  1  1 0
v -1  1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
`;

function texturedDefinition(alphaMode: 'opaque' | 'mask'): MaterialDefinition {
  return {
    shading: 'unlit', alphaMode, doubleSided: false,
    baseColor: m.multiply(m.texture(0), m.parameter(0)),
    metallic: 0, roughness: 0.5, emissive: m.constant([0, 0, 0, 0]), alphaCutoff: 0.5,
  };
}

function plainDefinition(shading: 'unlit' | 'pbr', doubleSided: boolean): MaterialDefinition {
  return {
    shading, alphaMode: 'opaque', doubleSided,
    baseColor: m.parameter(0), metallic: 0, roughness: 0.6,
    emissive: m.constant([0, 0, 0, 0]), alphaCutoff: 0.5,
  };
}

export async function createMaterialFixture() {
  const imported = parseObj(QUAD_OBJ);
  if (imported.primitives.length !== 1) throw new Error('Expected one imported quad');
  const mesh = imported.primitives[0].mesh;
  const batch: SceneBatch = {
    name: 'material-test-quad', mesh, capacity: 1, inputMode: 'committed',
    deformation: { jointCount: 0, nodes: [{ kind: 'wave', amplitude: WAVE_AMPLITUDE, frequency: WAVE_FREQUENCY, phase: 0 }] }, morphTargets: [],
    instances: [{
      entityId: 1, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [0],
      color: [1, 1, 1, 1], metalness: 0, roughness: 0.6, historyVersion: 1,
    }],
  };
  const cutout: TextureAsset = {
    width: 2, height: 2, colorSpace: 'linear',
    pixels: new Uint8Array([
      255, 80, 40, 255, 255, 80, 40, 0,
      255, 80, 40, 255, 255, 80, 40, 0,
    ]),
  };
  const canvas = document.createElement('canvas');
  canvas.id = 'material-fixture';
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;z-index:1000';
  document.body.appendChild(canvas);
  let renderer: Renderer;
  try {
    renderer = await Renderer.create(canvas, [batch], { materialVariantLimit: 4, textureBudgetBytes: 40 });
  } catch (error) {
    canvas.remove();
    throw error;
  }

  const capture = async (eye: [number, number, number] = FRONT, points = PROBES) => {
    renderer.render(eye, TARGET);
    const depth = await renderer.probeDepth(points);
    return {
      depth,
      diagnostics: renderer.diagnostics(),
      ownedGpuBytes: renderer.stats.ownedGpuBytes,
      errors: [...renderer.errors],
    };
  };
  const textureBinding = (alphaMode: 'opaque' | 'mask', withTexture = true): MaterialBinding => ({
    definition: texturedDefinition(alphaMode),
    parameters: new Float32Array([1, 1, 1, 1]),
    textures: withTexture ? [cutout] : [],
  });

  return {
    async textured(alphaMode: 'opaque' | 'mask') {
      await renderer.setMaterial(batch.name, textureBinding(alphaMode));
      renderer.setSun({ direction: [0, 0, 1], color: [2, 2, 2] });
      return capture();
    },
    async replacementRace() {
      const first = renderer.setMaterial(batch.name, textureBinding('opaque'));
      const last = renderer.setMaterial(batch.name, textureBinding('mask'));
      await Promise.all([first, last]);
      renderer.setSun({ direction: [0, 0, 1], color: [2, 2, 2] });
      return capture();
    },
    async deformedMask() {
      const phase = 0.2;
      const y = 0.5;
      // The four imported vertices are at y = ±1. Rasterized deformation is
      // linear across them, so interpolate their WGSL wave offsets here.
      const t = (y + 1) / 2;
      const waveOffset = WAVE_AMPLITUDE * ((1 - t) * Math.sin(-WAVE_FREQUENCY + phase) +
        t * Math.sin(WAVE_FREQUENCY + phase));
      const translationX = 0.35;
      batch.instances = [{ ...batch.instances[0], model: translation(translationX, 0, 0),
        deformScalars: [phase], historyVersion: 2 }];
      const points: readonly [number, number, number][] = [
        [-0.5 + translationX + waveOffset, y, 0],
        [0.5 + translationX + waveOffset, y, 0],
      ];
      const result = await capture(FRONT, points);
      return { ...result, points, waveOffset };
    },
    async rejectedTextureSwap() {
      let rejection = '';
      try { await renderer.setMaterial(batch.name, textureBinding('mask', false)); }
      catch (error) { rejection = String(error); }
      return { rejection, after: await capture() };
    },
    async rejectedBudgetSwap() {
      const larger: TextureAsset = { width: 3, height: 3, colorSpace: 'linear', pixels: new Uint8Array(36).fill(255) };
      let rejection = '';
      try { await renderer.setMaterial(batch.name, {
        definition: texturedDefinition('mask'),
        parameters: new Float32Array([1, 1, 1, 1]),
        textures: [larger],
      }); } catch (error) { rejection = String(error); }
      return { rejection, after: await capture() };
    },
    async tint(red: number) {
      await renderer.setMaterialParameters(batch.name, new Float32Array([red, 1, 1, 1]));
      return capture();
    },
    async plain(shading: 'unlit' | 'pbr', doubleSided: boolean, view: 'front' | 'back', sunColor: [number, number, number]) {
      await renderer.setMaterial(batch.name, {
        definition: plainDefinition(shading, doubleSided),
        parameters: new Float32Array([0.75, 0.2, 0.1, 1]),
      });
      renderer.setSun({ direction: view === 'front' ? [0, 0, 1] : [0, 0, -1], color: sunColor });
      return capture(view === 'front' ? FRONT : BACK);
    },
    async sun(color: [number, number, number]) {
      renderer.setSun({ direction: [0, 0, 1], color });
      return capture();
    },
    async sweepVariants() {
      const results = [];
      for (let i = 0; i < 8; i++) {
        await renderer.setMaterial(batch.name, {
          definition: { ...plainDefinition('unlit', false), baseColor: m.constant([0.1 + i / 20, 0.2, 0.3, 1]) },
        });
        results.push(await capture());
      }
      return results;
    },
    async disposeDuringCachedReplacement() {
      const binding = textureBinding('mask');
      await renderer.setMaterial(batch.name, binding);
      const pending = renderer.setMaterial(batch.name, binding);
      // Prepare can finish its cached await before the caller publishes it.
      await Promise.resolve();
      renderer.dispose(); canvas.remove();
      let outcome = 'resolved';
      try { await pending; } catch (error) { outcome = String(error); }
      return { outcome, errors: [...renderer.errors], ownedGpuBytes: renderer.stats.ownedGpuBytes };
    },
    async dispose() {
      const errors = [...renderer.errors];
      renderer.dispose();
      canvas.remove();
      return { errors, ownedGpuBytes: renderer.stats.ownedGpuBytes };
    },
    async disposeDuringPending() {
      // This variant has not been compiled in this fixture. Disposing in the
      // same task runs before its asynchronous pipeline preparation settles.
      const pending = renderer.setMaterial(batch.name, {
        definition: plainDefinition('pbr', true),
        parameters: new Float32Array([0.2, 0.4, 0.8, 1]),
      });
      renderer.dispose();
      canvas.remove();
      let outcome = 'resolved';
      try { await pending; } catch (error) { outcome = String(error); }
      return { outcome, errors: [...renderer.errors], ownedGpuBytes: renderer.stats.ownedGpuBytes };
    },
  };
}
