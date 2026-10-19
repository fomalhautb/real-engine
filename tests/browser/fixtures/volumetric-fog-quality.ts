import { Renderer } from '../../../src/renderer';
import { createCube } from '../../../src/geometry';
import { multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION } from '../../../src/material';
import type { SceneBatch } from '../../../src/contracts';

/** Independent analytic shadow intervals from horizontal slats above the view.
 * The reference never reads the fog grid or its history, or samples shadow maps. */
export async function runFogQualityFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:96px;height:64px'; document.body.appendChild(canvas);
  const slats = Array.from({ length: 8 }, (_, i) => ({ z: 0.21 + i * 0.73, width: 0.22 }));
  const mesh = createCube();
  const batch: SceneBatch = { name: 'analytic-shadow-slats', mesh, capacity: slats.length, inputMode: 'committed',
    instances: slats.map((slat, i) => {
      const model = multiply(translation(0, 3, slat.z), scaling(20, 0.1, slat.width));
      return { entityId: i + 1, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
        color: [1, 1, 1, 1] as [number, number, number, number], metalness: 0, roughness: 1, historyVersion: 0 };
    }), deformation: { jointCount: 0, nodes: [] }, morphTargets: [], material: { definition: DEFAULT_MATERIAL_DEFINITION } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], {
      antiAliasing: 'none', sunShadow: { resolution: 2048, focusBounds: { min: [-6, -4, -1], max: [6, 4, 8] } },
      volumetricFog: { pixelSize: 4, slices: 16, maxViewDepth: 6, temporal: { maxHistoryFrames: 32 },
        global: { coefficients: { extinction: 0.12, albedo: [0.8, 0.6, 0.4], emission: [0, 0, 0] } } },
    });
    const active = renderer;
    active.setSun({ direction: [0, 1, 0], color: [8, 6, 4] });
    const pixels = Array.from({ length: 8 }, (_, y) => Array.from({ length: 12 }, (_, x) => ({ x: 4 + x * 8, y: 4 + y * 8 }))).flat();
    const draw = (z: number) => active.render([0, 0, z], [0, 0, z - 6]);
    active.setVolumetricFogEnabled(false); draw(6); draw(6);
    const sky = await active.probeSurfaceHdr(pixels);
    active.setVolumetricFogEnabled(true);
    const reference = (eyeZ: number) => pixels.map((p, index) => {
      const tangent = Math.tan(Math.PI / 6.4);
      const length = Math.hypot((2 * (p.x + 0.5) / 96 - 1) * tangent * 1.5, (1 - 2 * (p.y + 0.5) / 64) * tangent, 1);
      const boundaries = [0, 6, ...slats.flatMap(s => [eyeZ - s.z - s.width / 2, eyeZ - s.z + s.width / 2])]
        .filter(t => t >= 0 && t <= 6).sort((a, b) => a - b);
      const scatter = [0, 0, 0]; let transmission = 1;
      for (let i = 0; i < boundaries.length - 1; i++) {
        const z = eyeZ - (boundaries[i] + boundaries[i + 1]) * 0.5;
        const visibility = slats.some(s => Math.abs(z - s.z) < s.width / 2) ? 0 : 1;
        const next = Math.exp(-0.12 * (boundaries[i + 1] - boundaries[i]) * length);
        for (let channel = 0; channel < 3; channel++) scatter[channel] += transmission * (1 - next)
          * [0.8, 0.6, 0.4][channel] * [8, 6, 4][channel] * visibility / (4 * Math.PI);
        transmission *= next;
      }
      return scatter.map((value, channel) => value + transmission * sky[index][channel]);
    });
    active.setVolumetricFogTemporalEnabled(false); draw(6);
    const centered = await active.probeHdr(pixels);
    active.setVolumetricFogTemporalEnabled(true);
    for (let i = 0; i < 96; i++) draw(6);
    const accumulated = await active.probeHdr(pixels), staticReference = reference(6);
    const moving: { z: number; centered: number[][]; temporal: number[][]; reference: number[][] }[] = [];
    // Record temporal motion before the centered control, so the control cannot
    // clear history in the sequence being assessed.
    for (let i = 0; i < 24; i++) {
      const z = 6 + (i + 1) * 0.008; draw(z);
      moving.push({ z, temporal: await active.probeHdr(pixels), centered: [], reference: reference(z) });
    }
    active.setVolumetricFogTemporalEnabled(false);
    for (const sample of moving) { draw(sample.z); sample.centered = await active.probeHdr(pixels); }
    const info = active.info, errors = [...active.errors], diagnostics = active.diagnostics().volumetricFog;
    active.dispose();
    return { centered, accumulated, staticReference, moving, info, errors, diagnostics, released: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
