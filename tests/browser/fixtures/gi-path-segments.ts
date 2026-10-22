import type { InstanceState, SceneBatch } from '../../../src/contracts';
import { createCube, createPlane } from '../../../src/geometry';
import { identity, multiply, normalMatrix, scaling, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const sky: Vec3 = [0.03, 0.05, 0.08];
const emission: Vec3 = [0.7, 0.35, 0.14];
const base: Vec3 = [0.6, 0.5, 0.4];
const eye: Vec3 = [0, 1, 3], target: Vec3 = [0, 0, 0];
const anchorIndex = 8 + 8 * 16;
const centerPixel = { x: 34, y: 34 };

function state(id: number, model: Float32Array, deformScalars: number[] = []): InstanceState {
  return { entityId: id, entityGeneration: 1, model, normalMatrix: normalMatrix(model),
    bones: deformScalars.length ? [identity()] : [], deformScalars,
    color: [1, 1, 1, 1], metalness: 0, roughness: 1, historyVersion: 0 };
}

function cosineFraction(position: Vec3): number {
  // Independent midpoint integration of a 2m × 2m horizontal emitter. Its
  // solid-angle cosine fraction is ∫h²/(πr⁴)dA over the rectangle.
  const resolution = 256, step = 2 / resolution, h = 1.8 - position[1];
  let fraction = 0;
  for (let y = 0; y < resolution; y++) for (let x = 0; x < resolution; x++) {
    const dx = -1 + (x + 0.5) * step - position[0];
    const dz = -1 + (y + 0.5) * step - position[2];
    const r2 = dx * dx + dz * dz + h * h;
    fraction += h * h / (Math.PI * r2 * r2) * step * step;
  }
  return fraction;
}

function irradianceReference(position: Vec3, color: Vec3): Vec3 {
  const fraction = cosineFraction(position);
  return sky.map((value, c) => Math.PI * (value * (1 - fraction) + color[c] * fraction)) as Vec3;
}

export async function runGiPathSegmentsFixture(segments: 2 | 3 | 4, supplemental = false) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px';
  document.body.append(canvas);
  const floorModel = multiply(translation(0, 0, 0), scaling(20, 1, 20));
  const floor: SceneBatch = { name: 'path-floor', mesh: createPlane(), capacity: 1,
    inputMode: 'committed', instances: [state(1, floorModel)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, baseColor: m.constant([...base, 1]),
      metallic: 0, roughness: 1 } } };
  const roofMesh = createPlane();
  const vertices = new DataView(roofMesh.vertices), morph = new Float32Array(roofMesh.vertexCount * 3);
  for (let i = 0; i < roofMesh.vertexCount; i++) {
    for (let j = 0; j < 4; j++) vertices.setUint32(i * 80 + 32 + j * 4, 0, true);
    morph[i * 3 + 1] = 10;
  }
  const roofModel = multiply(translation(0, 1.8, 0), scaling(2, 1, 2));
  const roof: SceneBatch = { name: 'path-emitter', mesh: roofMesh, capacity: 1,
    inputMode: 'committed', instances: [], emissiveSampling: 'triangle-area',
    deformation: { jointCount: 1, nodes: [{ kind: 'morph', target: 0, weight: 0 }, { kind: 'skin' }] },
    morphTargets: [{ positions: morph, normals: new Float32Array(morph.length) }],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.constant([0, 0, 0, 1]), emissive: m.parameter(0) },
      parameters: new Float32Array([...emission, 1]) } };
  // Two adjacent half rectangles retain the independent full-rectangle truth
  // while exercising source-major area-ray packing for supplemental queries.
  const second: SceneBatch | null = supplemental ? { ...roof, name: 'path-emitter-second', instances: [] } : null;
  const roofPose = supplemental ? multiply(translation(-0.5, 1.8, 0), scaling(1, 1, 2)) : roofModel;
  const secondPose = multiply(translation(0.5, 1.8, 0), scaling(1, 1, 2));
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [floor, roof, ...(second ? [second] : [])], {
      worldQueries: { maxRays: 1024 },
      diffuseGi: { origin: [-1, 0.25, -1], spacing: [2, 1, 2], counts: [2, 2, 2],
        raysPerProbe: 32, maxDistance: 8, skyRadiance: sky, multiBounce: true,
        finalGather: { pixelSpacing: 4, raysPerAnchor: 1, pathSegments: segments,
          ...(supplemental ? { receiver: 'raster' as const, maxSupplementalAnchors: 128, maxSupplementalWorkUnits: 1_048_576 } : {}),
          maxAnchors: 1024, maxHistory: 128 } },
    });
    const active = renderer;
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const capture = async (frames: number) => {
      for (let i = 0; i < frames; i++) active.render(eye, target);
      const [hdr, probe] = await Promise.all([
        active.probeHdr([centerPixel]), active.probeDiffuseGiFinalGather([anchorIndex]),
      ]);
      const diagnostics = active.diagnostics().diffuseGi!.finalGather!;
      const supplementalAnchors = supplemental ? (await active.probeDiffuseGiFinalGather(Array.from(
        { length: diagnostics.supplementalAnchorsPerUpdate }, (_, i) => diagnostics.primaryAnchorsPerUpdate + i))).anchors : [];
      return { hdr: hdr[0], anchor: probe.anchors[0], supplementalAnchors, diagnostics };
    };
    active.setDiffuseGiMultiBounce(false);
    const missSingle = await capture(2);
    active.setDiffuseGiMultiBounce(true);
    const missMulti = await capture(2);
    roof.instances = [state(2, roofPose, [0])];
    if (second) second.instances = [state(3, secondPose, [0])];
    active.setDiffuseGiMultiBounce(false);
    const unlitSingle = await capture(128);
    active.setDiffuseGiMultiBounce(true);
    const unlitMulti = await capture(128);
    await active.setMaterial('path-emitter', { ...roof.material!,
      parameters: new Float32Array([emission[0] * 2, emission[1] * 2, emission[2] * 2, 1]) });
    if (second) await active.setMaterial(second.name, { ...second.material!,
      parameters: new Float32Array([emission[0] * 2, emission[1] * 2, emission[2] * 2, 1]) });
    const doubled = await capture(128);
    roof.instances[0] = { ...roof.instances[0], deformScalars: [1] };
    if (second) second.instances[0] = { ...second.instances[0], deformScalars: [1] };
    const morphedAway = await capture(2);
    const reference = irradianceReference(unlitMulti.anchor.position, emission);
    const doubledReference = irradianceReference(doubled.anchor.position,
      emission.map(value => value * 2) as Vec3);
    const supplementalAreaChecks = [unlitSingle, unlitMulti, doubled].flatMap((stage, stageIndex) =>
      stage.supplementalAnchors.filter(anchor => anchor.valid && anchor.identity[1] === 1)
        .sort((a, b) => Math.hypot(a.position[0], a.position[2]) - Math.hypot(b.position[0], b.position[2]))
        .slice(0, 3).map(anchor => ({ stage: stageIndex, anchor,
          reference: irradianceReference(anchor.position, emission.map(value => value * (stageIndex === 2 ? 2 : 1)) as Vec3) })));
    const adapter = active.info.adapterInfo;
    const errors = [...active.errors]; active.dispose();
    return { segments, sky, emission, base, missSingle, missMulti, unlitSingle,
      unlitMulti, doubled, morphedAway, reference, doubledReference, supplementalAreaChecks,
      adapter, errors, afterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** The first segment meets a nonemissive PBR ceiling; the next hemisphere
 * reaches a broad Unlit floor. This has no world-field or area-estimator term. */
export async function runGiLaterEmissionFixture(segments: 2 | 3 | 4) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px'; document.body.append(canvas);
  const floorModel = scaling(100, 1, 100), roofModel = multiply(translation(0, 1.8, 0), scaling(2, 1, 2));
  const bounceAlbedo: Vec3 = [0.5, 0.6, 0.7];
  const floor: SceneBatch = { name: 'later-emission-floor', mesh: createPlane(), capacity: 1,
    inputMode: 'committed', instances: [state(11, floorModel)], emissiveSampling: 'triangle-area',
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.constant([0, 0, 0, 1]), emissive: m.constant([...emission, 1]) } } };
  const roof: SceneBatch = { name: 'later-emission-ceiling', mesh: createPlane(), capacity: 1,
    inputMode: 'committed', instances: [state(12, roofModel)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
      baseColor: m.constant([...bounceAlbedo, 1]), metallic: 0, roughness: 1 } } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [floor, roof], {
      worldQueries: { maxRays: 1024 },
      diffuseGi: { origin: [-1, 0.25, -1], spacing: [2, 1, 2], counts: [2, 2, 2],
        raysPerProbe: 32, maxDistance: 30, skyRadiance: [0, 0, 0], multiBounce: true,
        finalGather: { pixelSpacing: 4, raysPerAnchor: 1, pathSegments: segments,
          maxAnchors: 1024, maxHistory: 128 } },
    });
    const active = renderer;
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    for (let i = 0; i < 128; i++) active.render(eye, target);
    const first = (await active.probeDiffuseGiFinalGather([anchorIndex])).anchors[0];
    const diagnostics = active.diagnostics().diffuseGi!.finalGather!;
    const fraction = cosineFraction(first.position);
    // The camera-hit receiver lies on the emitting floor. Its area sample
    // points below the +Y geometric normal and has exactly zero receiver
    // cosine. A later hit from the ceiling must still retain this registered
    // source's emission. The 30m ray cap and 100m floor omit <0.5% of a
    // cosine-weighted second hemisphere, so the infinite-floor expression is
    // a close bound rather than a mathematically exact finite-scene value.
    const expected = emission.map((value, c) => Math.PI * fraction * bounceAlbedo[c] * 0.96 * value) as Vec3;
    const adapter = active.info.adapterInfo;
    const errors = [...active.errors]; active.dispose();
    return { segments, bounceAlbedo, emission, fraction, finiteSecondSegmentLossBound: 0.005,
      expected, anchor: first,
      diagnostics, adapter, errors, afterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

/** A uniform closed room has the fixed-point radiance Le/(1-rho) in every
 * direction. It isolates the world-field terminal term from edge visibility,
 * area emitters and direct light. */
export async function runGiRoomCacheFixture(segments: 2 | 3 | 4) {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:64px;height:64px'; document.body.append(canvas);
  const le: Vec3 = [0.08, 0.12, 0.2], albedo = 0.5, rho = albedo * 0.96;
  const roomModel = scaling(4, 4, 4); // createCube spans ±0.5, so room spans ±2m.
  const room: SceneBatch = { name: 'closed-emissive-room', mesh: createCube(), capacity: 1,
    inputMode: 'committed', instances: [state(21, roomModel)],
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [],
    material: { definition: { ...DEFAULT_MATERIAL_DEFINITION, doubleSided: true,
      baseColor: m.constant([albedo, albedo, albedo, 1]), metallic: 0, roughness: 1,
      emissive: m.constant([...le, 1]) } } };
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, [room], {
      worldQueries: { maxRays: 1024 },
      diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
        raysPerProbe: 128, maxDistance: 20, normalBias: 0.001,
        skyRadiance: [0, 0, 0], multiBounce: true,
        finalGather: { pixelSpacing: 4, raysPerAnchor: 1, pathSegments: segments,
          maxAnchors: 1024, maxHistory: 128 } },
    });
    const active = renderer;
    active.setSun({ direction: [0, 1, 0], color: [0, 0, 0] });
    const roomEye: Vec3 = [0, 0, 0], roomTarget: Vec3 = [0, 0, -1];
    for (let i = 0; i < 160; i++) active.render(roomEye, roomTarget);
    const [probe, hdr, world] = await Promise.all([
      active.probeDiffuseGiFinalGather([anchorIndex]), active.probeHdr([centerPixel]), active.probeDiffuseGi(),
    ]);
    const expectedRadiance = le.map(value => value / (1 - rho)) as Vec3;
    const expectedIrradiance = expectedRadiance.map(value => Math.PI * value) as Vec3;
    const diagnostics = active.diagnostics().diffuseGi!.finalGather!;
    const adapter = active.info.adapterInfo;
    const errors = [...active.errors]; active.dispose();
    return { segments, le, albedo, rho, expectedRadiance, expectedIrradiance,
      anchor: probe.anchors[0], hdr: hdr[0], world, diagnostics, adapter,
      errors, afterDispose: active.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
