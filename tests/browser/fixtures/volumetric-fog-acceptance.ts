import type { SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { identity } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';
import { FOG_ANALYTIC_CASES } from './volumetric-fog-acceptance-cases';

type Vec3 = [number, number, number];
const SIZE = 64;
const EYE: Vec3 = [0, 0, 6];
const TARGET: Vec3 = [0, 0, 0];
/** Signal regions of interest (pixel centres) and an 8x8 lattice for finiteness. */
const ROI = [{ x: 32, y: 32 }, { x: 20, y: 24 }, { x: 44, y: 40 }, { x: 12, y: 52 }, { x: 54, y: 9 }];
const LATTICE = Array.from({ length: 64 }, (_, i) => ({ x: 4 + 8 * (i % 8), y: 4 + 8 * Math.floor(i / 8) }));

function plane(): MeshData {
  const positions = [[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  for (let index = 0; index < 4; index++) {
    const base = index * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(base + axis * 4, positions[index][axis], true);
    view.setFloat32(base + 12, 1, true);
    view.setFloat32(base + 24, 1, true);
    view.setFloat32(base + 48, 1, true);
  }
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-4, -4, 0], max: [4, 4, 0] } };
}

/** F01: separate L_fog and T read back from the explicit split view, raw and accumulated. */
export async function runFogAnalyticAcceptanceFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px`;
  document.body.appendChild(canvas);
  const batch: SceneBatch = { name: 'fog-acceptance-receiver', mesh: plane(), capacity: 1, inputMode: 'committed',
    instances: [{ entityId: 1, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
      color: [1, 1, 1, 1], metalness: 0, roughness: 0.5, historyVersion: 0 }],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', baseColor: m.parameter(0) },
      parameters: new Float32Array([2, 1, 0.5, 1]) } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [batch], { antiAliasing: 'none',
      volumetricFog: { pixelSize: 2, slices: 64, maxViewDepth: 8, maxFroxels: 32 * 32 * 64, maxBytes: 8 * 1024 * 1024,
        temporal: { maxHistoryFrames: 8 } } });
    const active = renderer;
    const split = async (pixels: readonly { x: number; y: number }[]) => {
      active.setVolumetricFogDebugView('scattering-transmittance');
      active.render(EYE, TARGET);
      const result = await active.probeHdr(pixels);
      active.setVolumetricFogDebugView('composite');
      return result;
    };
    const cases = [];
    for (const entry of FOG_ANALYTIC_CASES) {
      active.setSun(entry.sun);
      active.setVolumetricFogMedia({ global: entry.global, volumes: entry.volumes ?? [] });
      active.setVolumetricFogTemporalEnabled(false);
      const raw = await split(ROI);
      const rawLattice = await split(LATTICE);
      active.render(EYE, TARGET);
      const composite = await active.probeHdr(ROI);
      const surface = await active.probeSurfaceHdr(ROI);
      active.setVolumetricFogTemporalEnabled(true);
      for (let frame = 0; frame < 12; frame++) active.render(EYE, TARGET);
      const accumulated = await split(ROI);
      const accumulatedLattice = await split(LATTICE);
      cases.push({ label: entry.label, raw, rawLattice, composite, surface, accumulated, accumulatedLattice,
        history: active.diagnostics().volumetricFog?.history });
    }
    active.setVolumetricFogEnabled(false);
    const disabledSplit = await split(ROI);
    active.render(EYE, TARGET);
    const disabled = { hdr: await active.probeHdr(ROI), surface: await active.probeSurfaceHdr(ROI) };
    const errors = [...active.errors], info = active.info;
    active.dispose();
    return { size: SIZE, eye: EYE, roi: ROI, lattice: LATTICE, maxViewDepth: 8, cases, disabledSplit, disabled,
      errors, adapter: info.adapterInfo, ownedBytesAfterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
