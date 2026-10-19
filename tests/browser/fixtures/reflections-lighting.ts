import type { InstanceState, SceneBatch } from '../../../src/contracts';
import type { MeshData } from '../../../src/geometry';
import { multiply, normalMatrix, rotationY, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

const EYE: [number, number, number] = [0, 0, 3];
const TARGET: [number, number, number] = [0, 0, 0];
const PIXELS = [{ x: 16, y: 16 }, { x: 15, y: 16 }, { x: 17, y: 16 }];
const ANCHORS = PIXELS.map(pixel => pixel.y * 32 + pixel.x);
const BASE: [number, number, number] = [0.6, 0.3, 0.15];
const SUN: [number, number, number] = [4, 3, 2];

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
  return { vertices, vertexCount: 4, indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    bounds: { min: [-1, -1, 0], max: [1, 1, 0] } };
}

function instance(entityId: number, model: Float32Array, metalness: number, roughness: number): InstanceState {
  return { entityId, model, normalMatrix: normalMatrix(model), bones: [], deformScalars: [],
    color: [1, 1, 1, 1], metalness, roughness, historyVersion: 0 };
}

/** Lit PBR secondary hit: the blocker intersects its sun ray but misses mirror rays. */
export async function runReflectionsLightingFixture() {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:32px;height:32px';
  document.body.appendChild(canvas);
  const mirror: SceneBatch = { name: 'lighting-mirror', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [instance(1, scaling(3, 3, 1), 1, 0.045)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      baseColor: m.constant([1, 1, 1, 1]) } } };
  const receiverModel = multiply(translation(0, 0, 5), multiply(rotationY(Math.PI), scaling(20, 20, 1)));
  const receiver: SceneBatch = { name: 'offscreen-pbr-receiver', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [instance(2, receiverModel, 0.2, 0.5)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION,
      baseColor: m.constant([...BASE, 1]) } } };
  const blocker: SceneBatch = { name: 'secondary-sun-blocker', mesh: quad(), capacity: 1,
    inputMode: 'committed', instances: [], deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit',
      baseColor: m.constant([0, 0, 0, 1]) } } };
  const blockerModel = multiply(translation(2, 0, 3), scaling(0.8, 20, 1));
  const scene = [mirror, receiver, blocker];
  const options = { antiAliasing: 'none' as const, worldQueries: { maxRays: 4096 },
    reflections: { pixelSpacing: 1, maxAnchors: 1024, samplesPerAnchor: 4,
      maxDistance: 20, maxBytes: 16 * 1024 * 1024, skyRadiance: [0, 0, 0] as [number, number, number] } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, scene, options);
    const first = renderer;
    first.setSun({ direction: [1, 0, -1], color: SUN });
    const capture = async (active: Renderer, probe = true) => {
      active.render(EYE, TARGET);
      const [surface, reflection] = await Promise.all([
        active.probeSurfaceHdr(PIXELS), probe ? active.probeReflections(ANCHORS) : Promise.resolve(null),
      ]);
      return { surface, reflection, diagnostics: active.diagnostics().reflections };
    };
    const lit = await capture(first);
    first.setSun({ direction: [1, 0, -1], color: [0, 0, 0] });
    const noSun = await capture(first);
    first.setSun({ direction: [1, 0, -1], color: SUN });
    blocker.instances = [instance(3, blockerModel, 0, 1)];
    const shadowed = await capture(first);
    blocker.instances = [];
    const restored = await capture(first);
    first.setReflectionsEnabled(false);
    const disabled = await capture(first, false);
    const directErrors = [...first.errors];
    first.dispose();
    const directOwnedAfterDispose = first.stats.ownedGpuBytes;

    renderer = await Renderer.create(canvas, scene, { ...options,
      diffuseGi: { origin: [-2, -2, -1], spacing: [4, 4, 8], counts: [2, 2, 2],
        raysPerProbe: 16, maxDistance: 20, skyRadiance: [4, 2, 1] } });
    const withGi = renderer;
    withGi.setSun({ direction: [1, 0, -1], color: [0, 0, 0] });
    // Allow the probe field to complete before measuring the reflected hit.
    for (let frame = 0; frame < 3; frame++) withGi.render(EYE, TARGET);
    const giEnabled = await capture(withGi);
    withGi.setDiffuseGiEnabled(false);
    const giDisabled = await capture(withGi);
    withGi.setReflectionsEnabled(false);
    const giNoReflection = await capture(withGi, false);
    const giDiagnostics = withGi.diagnostics().diffuseGi;
    const giErrors = [...withGi.errors];
    withGi.dispose();
    return { adapters: [first.info.adapterInfo, withGi.info.adapterInfo], eye: EYE, pixels: PIXELS, anchorIndices: ANCHORS, base: BASE, sun: SUN,
      lit, noSun, shadowed, restored, disabled,
      giEnabled, giDisabled, giNoReflection, giDiagnostics,
      directErrors, giErrors, directOwnedAfterDispose,
      ownedBytesAfterDispose: withGi.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
