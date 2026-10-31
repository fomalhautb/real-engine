import type { SceneBatch } from '../../../src/contracts';
import type { FogHistoryCell } from '../../../src/gpu-volumetric-fog';
import type { VolumetricFogOptions } from '../../../src/volumetric-fog';
import type { LocalLightDescriptor } from '../../../src/local-lights';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { identity } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];

/** Horizontal double-sided square in the plane y = plate.y. */
function plateMesh(plate: { x: number; y: number; z: number; half: number }) {
  const { x, y, z, half } = plate, vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  const points = [[x - half, y, z - half], [x - half, y, z + half], [x + half, y, z + half], [x + half, y, z - half]];
  points.forEach((point, index) => {
    for (let axis = 0; axis < 3; axis++) view.setFloat32(index * 80 + axis * 4, point[axis], true);
    view.setFloat32(index * 80 + 12, 1, true);
    view.setFloat32(index * 80 + 20, 1, true);
    view.setFloat32(index * 80 + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [x - half, y, z - half] as Vec3, max: [x + half, y, z + half] as Vec3 } };
}

export interface LocalLightFogPlan {
  size: number;
  eye: Vec3;
  target: Vec3;
  fog: VolumetricFogOptions;
  pointLight: LocalLightDescriptor;
  spotLight: LocalLightDescriptor;
  plate: { x: number; y: number; z: number; half: number };
  pixels: { x: number; y: number }[];
  grid: { columns: number; rows: number; slices: number };
}

/** Froxel sources (history right after each reset, so unjittered cell centres), HDR and diagnostics. */
export async function runLocalLightFogFixture(plan: LocalLightFogPlan) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = `position:fixed;left:0;top:0;width:${plan.size}px;height:${plan.size}px`;
  document.body.appendChild(canvas);
  const plate: SceneBatch = { name: 'local-light-fog-plate', mesh: plateMesh(plan.plate), capacity: 1, inputMode: 'committed',
    instances: [], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.constant([0, 0, 0, 1]) } } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [plate], { antiAliasing: 'none', volumetricFog: plan.fog,
      localLights: { maxLights: 4, maxShadowedLights: 1, shadowResolution: 512 } });
    renderer.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    // History is probed right after each reset, when the froxel samples sit unjittered at cell centres.
    const cells: FogHistoryCell[] = [];
    for (let z = 0; z < plan.grid.slices; z++) for (let y = 0; y < plan.grid.rows; y++) for (let x = 0; x < plan.grid.columns; x++) {
      if ((x + y + z) % 2 === 0) cells.push({ x, y, z });
    }
    const fogState = () => {
      const fog = renderer!.diagnostics().volumetricFog!;
      return { resetReason: fog.history.lastResetReason, framesSinceReset: fog.history.framesSinceReset, localLightScattering: fog.localLightScattering };
    };
    const timings: number[] = [];
    const capture = async () => {
      const start = performance.now();
      renderer!.render(plan.eye, plan.target);
      const grid = renderer!.diagnostics().volumetricFog!.grid!;
      if (grid.columns !== plan.grid.columns || grid.rows !== plan.grid.rows || grid.slices !== plan.grid.slices) {
        throw new Error('LOCAL_LIGHT_FOG_GRID: ' + JSON.stringify(grid));
      }
      const sources = (await renderer!.probeVolumetricFogHistory(cells)).map(cell => ({ x: cell.x, y: cell.y, z: cell.z, source: cell.source, count: cell.count }));
      const [hdr, surface] = await Promise.all([renderer!.probeHdr(plan.pixels), renderer!.probeSurfaceHdr(plan.pixels)]);
      timings.push(performance.now() - start);
      return { sources, hdr, surface, fog: fogState() };
    };
    const dark = await capture();
    const point = renderer.addLocalLight({ ...plan.pointLight, castShadows: true });
    const spot = renderer.addLocalLight(plan.spotLight);
    const lit = await capture();
    renderer.render(plan.eye, plan.target);
    const settled = fogState();
    plate.instances = [{ entityId: 1, model: identity(), normalMatrix: identity(), bones: [], deformScalars: [],
      color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 }];
    const shadowed = await capture();
    // A second reset frame with the plate already in the scene: the spot changes, the shadowed point light does not.
    renderer.updateLocalLight(spot, { intensity: plan.spotLight.intensity * 1.5 });
    const shadowedAgain = await capture();
    renderer.setLocalLightEnabled(point, false);
    const pointOff = await capture();
    renderer.updateLocalLight(spot, { intensity: plan.spotLight.intensity * 2 });
    const spotDouble = await capture();
    const diagnostics = { localLights: renderer.diagnostics().localLights, grid: renderer.diagnostics().volumetricFog!.grid, timings };
    const errors = [...renderer.errors], adapter = renderer.info.adapterInfo;
    renderer.dispose();
    return { dark, lit, settled, shadowed, shadowedAgain, pointOff, spotDouble, diagnostics, errors, adapter, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
