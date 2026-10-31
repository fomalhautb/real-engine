import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import type { LocalLightDescriptor } from '../../../src/local-lights';
import { multiply, normalMatrix, rotationY, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];

/** Unit quad facing +Z. */
function quad(): MeshData {
  const points = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]];
  const vertices = new ArrayBuffer(4 * 80), view = new DataView(vertices);
  points.forEach((point, index) => {
    const offset = index * 80;
    for (let axis = 0; axis < 3; axis++) view.setFloat32(offset + axis * 4, point[axis], true);
    view.setFloat32(offset + 12, 1, true);
    view.setFloat32(offset + 24, 1, true);
    view.setFloat32(offset + 48, 1, true);
  });
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function instance(entityId: number, model: Float32Array, metalness: number, roughness: number): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [], color: [1, 1, 1, 1],
    metalness, roughness, historyVersion: 0 };
}

function batch(name: string, instances: InstanceState[], definition: SceneBatch['material']): SceneBatch {
  return { name, mesh: quad(), capacity: 1, inputMode: 'committed', instances, deformation: { jointCount: 0, nodes: [] },
    morphTargets: [], material: definition };
}

export interface LocalLightReflectionPlan {
  eye: Vec3;
  pixels: { x: number; y: number }[];
  base: Vec3;
  light: LocalLightDescriptor;
  /** Unlit black double-sided strip between the light and the reflected wall hits. */
  blocker: { center: Vec3; halfX: number; halfY: number };
  /** A range that leaves every reflected wall hit outside the light. */
  shortRange: number;
}

/** A mirror reflects an off-screen PBR wall lit only by a point light (sun off, no GI). */
export async function runLocalLightReflectionFixture(plan: LocalLightReflectionPlan) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:32px;height:32px';
  document.body.appendChild(canvas);
  const anchors = plan.pixels.map(pixel => pixel.y * 32 + pixel.x);
  const mirror = batch('local-light-mirror', [instance(1, scaling(3, 3, 1), 1, 0.045)],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([1, 1, 1, 1]) } });
  const wallModel = multiply(translation(0, 0, 5), multiply(rotationY(Math.PI), scaling(20, 20, 1)));
  const wall = batch('local-light-reflected-wall', [instance(2, wallModel, 0.2, 0.5)],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([...plan.base, 1]) } });
  const blocker = batch('local-light-reflection-blocker', [],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true, baseColor: m.constant([0, 0, 0, 1]) } });
  const blockerModel = multiply(translation(...plan.blocker.center), scaling(plan.blocker.halfX, plan.blocker.halfY, 1));
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [mirror, wall, blocker], { antiAliasing: 'none', worldQueries: { maxRays: 4096 },
      reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4, maxDistance: 20, maxBytes: 16 * 1024 * 1024,
        skyRadiance: [0, 0, 0] },
      localLights: { maxLights: 2, maxShadowedLights: 1, shadowResolution: 512 } });
    renderer.setSun({ direction: [1, 0, -1], color: [0, 0, 0] });
    const capture = async (probe = true) => {
      renderer!.render(plan.eye, [0, 0, 0]);
      const [surface, reflection] = await Promise.all([renderer!.probeSurfaceHdr(plan.pixels),
        probe ? renderer!.probeReflections(anchors) : Promise.resolve(null)]);
      return { surface, reflection };
    };
    const dark = await capture();
    const handle = renderer.addLocalLight({ ...plan.light, castShadows: false });
    const lit = await capture();
    renderer.updateLocalLight(handle, { castShadows: true });
    const shadowOpen = await capture();
    blocker.instances = [instance(3, blockerModel, 0, 1)];
    const shadowBlocked = await capture();
    blocker.instances = [];
    const shadowRestored = await capture();
    renderer.setLocalLightEnabled(handle, false);
    const disabled = await capture();
    renderer.setLocalLightEnabled(handle, true);
    renderer.updateLocalLight(handle, { range: plan.shortRange });
    const outOfRange = await capture();
    renderer.updateLocalLight(handle, { range: plan.light.range });
    const relit = await capture();
    renderer.setReflectionsEnabled(false);
    const noReflection = await capture(false);
    const diagnostics = { reflections: renderer.diagnostics().reflections, localLights: renderer.diagnostics().localLights };
    const errors = [...renderer.errors], adapter = renderer.info.adapterInfo;
    renderer.dispose();
    return { anchors, dark, lit, shadowOpen, shadowBlocked, shadowRestored, disabled, outOfRange, relit, noReflection,
      diagnostics, errors, adapter, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
