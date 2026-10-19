import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import type { MediumCoefficients } from '../../../src/participating-media';

const EYE: [number, number, number] = [0, 0, 6];
const TARGET: [number, number, number] = [0, 0, 0];
const POINTS = [{ x: 32, y: 32 }, { x: 24, y: 29 }] as const;
const SUN = { direction: [0, 0, 1] as [number, number, number],
  color: [40, 24, 12] as [number, number, number] };
const MAX_VIEW_DEPTH = 12;
const MAX_FROXELS = 270_000;
const MAX_BYTES = 10 * 1024 * 1024;

function quad(): MeshData {
  const positions = [[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  for (let i = 0; i < 4; i++) {
    const offset = i * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, positions[i][axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setFloat32(offset + 48, 1, true);
  }
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-4, -4, 0], max: [4, 4, 0] } };
}

function instance(z: number): InstanceState {
  return { entityId: 1201, model: translation(0, 0, z), normalMatrix: identity(),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 };
}

/** Real render and GPU readback; numeric expectations live in the browser spec. */
export async function runVolumetricFogFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px;';
  document.body.appendChild(canvas);
  const batch: SceneBatch = { name: 'fog-unlit-plane', mesh: quad(), capacity: 1, inputMode: 'committed',
    instances: [instance(0)], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.constant([2, 1, 0.5, 1]) } } };
  const vacuum: MediumCoefficients = { extinction: 0, albedo: [0, 0, 0], emission: [0, 0, 0] };
  const absorption: MediumCoefficients = { extinction: 0.22, albedo: [0, 0, 0], emission: [0, 0, 0] };
  const emission: MediumCoefficients = { extinction: 0, albedo: [0, 0, 0], emission: [0.04, 0.02, 0.01] };
  const scattering: MediumCoefficients = { extinction: 0.2, albedo: [0.8, 0.6, 0.4],
    emission: [0, 0, 0], anisotropy: 0.5 };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], { antiAliasing: 'none',
      volumetricFog: { pixelSize: 1, slices: 64, maxViewDepth: MAX_VIEW_DEPTH,
        maxFroxels: MAX_FROXELS, maxBytes: MAX_BYTES } });
    renderer.setSun(SUN);
    const capture = async (planeZ: number | null) => {
      batch.instances = planeZ === null ? [] : [instance(planeZ)];
      renderer!.render(EYE, TARGET);
      const [surface, fog] = await Promise.all([
        renderer!.probeSurfaceHdr(POINTS), renderer!.probeHdr(POINTS),
      ]);
      return { planeZ, surface, fog };
    };
    renderer.setVolumetricFogEnabled(false);
    const disabled = await capture(0);
    renderer.setVolumetricFogEnabled(true);
    renderer.setVolumetricFogMedia({ global: { coefficients: vacuum } });
    const empty = await capture(0);
    renderer.setVolumetricFogMedia({ global: { coefficients: absorption } });
    const nearClip = await capture(5.8);
    const near = await capture(2);
    const far = await capture(-2);
    const beyondLimit = await capture(-130);
    const sky = await capture(null);
    renderer.setVolumetricFogMedia({ global: { coefficients: emission } });
    const emitted = await capture(0);
    renderer.setVolumetricFogMedia({ global: { coefficients: scattering } });
    const scattered = await capture(0);
    renderer.setVolumetricFogEnabled(false);
    const toggledOff = await capture(0);
    renderer.setVolumetricFogEnabled(true);
    renderer.setVolumetricFogMedia({});
    const replacedEmpty = await capture(0);
    const initialSize = { width: renderer.stats.width, height: renderer.stats.height };
    const initialFog = renderer.diagnostics().volumetricFog;
    const initialOwnedBytes = renderer.stats.ownedGpuBytes;

    canvas.style.width = '80px';
    renderer.render(EYE, TARGET);
    const resizedSize = { width: renderer.stats.width, height: renderer.stats.height };
    const resizedFog = renderer.diagnostics().volumetricFog;
    const resizedOwnedBytes = renderer.stats.ownedGpuBytes;
    const errors = [...renderer.errors], frames = renderer.stats.frames;
    renderer.dispose();
    return { points: POINTS, eye: EYE, sun: SUN, maxViewDepth: MAX_VIEW_DEPTH,
      maxFroxels: MAX_FROXELS, maxBytes: MAX_BYTES, vacuum, absorption, emission, scattering,
      disabled, empty, nearClip, near, far, beyondLimit, sky, emitted, scattered, toggledOff, replacedEmpty,
      initialSize, resizedSize, initialFog, resizedFog, initialOwnedBytes, resizedOwnedBytes,
      errors, frames, ownedBytesAfterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
