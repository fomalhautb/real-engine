import type { InstanceState, MaterialBinding, SceneBatch } from '../../../src/contracts';
import { createPlane } from '../../../src/geometry';
import { lookAt, multiply, normalMatrix, perspective, rotationX, scaling, transformPoint, translation } from '../../../src/math';
import { DEFAULT_MATERIAL_DEFINITION, m } from '../../../src/material';
import { Renderer } from '../../../src/renderer';

type Vec3 = [number, number, number];
const base: Vec3 = [0.6, 0.4, 0.2];
const sky: Vec3 = [0.1, 0.2, 0.3];
const receiverName = 'history-receiver';
const centerIndex = 16 + 16 * 32;
const allIndices = Array.from({ length: 32 * 32 }, (_, index) => index);

function state(id: number, z: number, size: number, historyVersion = 0, generation = 1): InstanceState {
  const model = multiply(translation(0, 0, z), multiply(rotationX(Math.PI / 2), scaling(size, 1, size)));
  return { entityId: id, entityGeneration: generation, model, normalMatrix: normalMatrix(model),
    bones: [], deformScalars: [], color: [1, 1, 1, 1], metalness: 0,
    roughness: 1, historyVersion };
}
function batch(name: string, id: number, instances: InstanceState[], material: MaterialBinding): SceneBatch {
  return { name, mesh: createPlane(), capacity: 1, instances, inputMode: 'committed', material,
    queryIdentity: { assetId: 'final-gather-history/' + name, primitiveId: id, topologyVersion: 1 },
    deformation: { jointCount: 0, nodes: [] }, morphTargets: [] };
}
function scene(withBlocker: boolean) {
  const receiver = batch(receiverName, 1, [state(1, 0, 20)],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'pbr', doubleSided: true,
      baseColor: m.constant([...base, 1]), metallic: 0, roughness: 1 } });
  const blocker = batch('history-black-blocker', 2,
    withBlocker ? [state(2, 1, 1.36)] : [],
    { definition: { ...DEFAULT_MATERIAL_DEFINITION, shading: 'unlit', doubleSided: true,
      baseColor: m.constant([0, 0, 0, 1]), emissive: m.constant([0, 0, 0, 1]) } });
  return { receiver, blocker };
}
async function create(canvas: HTMLCanvasElement, batches: SceneBatch[]) {
  const renderer = await Renderer.create(canvas, batches, {
    worldQueries: { maxRays: 4096 },
    diffuseGi: { origin: [-1, -1, -1], spacing: [2, 2, 2], counts: [2, 2, 2],
      raysPerProbe: 64, maxDistance: 12, normalBias: 0.03, skyRadiance: sky,
      multiBounce: false, finalGather: { pixelSpacing: 4, maxAnchors: 1024,
        raysPerAnchor: 4, maxHistory: 32 } },
  });
  renderer.setSun({ direction: [0, 0, 1], color: [0, 0, 0] });
  return renderer;
}
function canvas128(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:128px;height:128px;';
  document.body.appendChild(canvas);
  return canvas;
}

export async function runFinalGatherHistoryFixture() {
  const canvas = canvas128();
  const { receiver, blocker } = scene(false);
  let renderer: Renderer | undefined;
  const eye: Vec3 = [0, 0, 5], target: Vec3 = [0, 0, 0];
  try {
    renderer = await create(canvas, [receiver, blocker]);
    const rejection = async (action: () => Promise<unknown>) => {
      try { await action(); return ''; } catch (error) { return String(error); }
    };
    const beforeFrame = await rejection(() => renderer!.probeDiffuseGiFinalGather([centerIndex]));
    for (let frame = 0; frame < 64; frame++) renderer.render(eye, target);
    const staticAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    const staticHdr = (await renderer.probeHdr([{ x: 64, y: 64 }]))[0];
    for (let frame = 0; frame < 12; frame++) {
      eye[0] += 0.002; target[0] += 0.002;
      renderer.render(eye, target);
    }
    const movedAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    const movedHdr = (await renderer.probeHdr([{ x: 64, y: 64 }]))[0];
    renderer.invalidateHistory();
    renderer.render(eye, target);
    const cutAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    renderer.render(eye, target);
    const resumedAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    renderer.invalidateDiffuseGiHistory();
    renderer.render(eye, target);
    const explicitGiResetAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    renderer.setSun({ direction: [0, 0, 1], color: [1, 1, 1] });
    renderer.render(eye, target);
    const lightChangedAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    receiver.instances = [state(1, 0, 20, 1, 2)];
    renderer.render(eye, target);
    const identityChangedAnchor = (await renderer.probeDiffuseGiFinalGather([centerIndex])).anchors[0];
    const duplicates = await rejection(() => renderer!.probeDiffuseGiFinalGather([centerIndex, centerIndex]));
    const outsideGrid = await rejection(() => renderer!.probeDiffuseGiFinalGather([1024]));
    const overBudget = await rejection(() => renderer!.probeDiffuseGiFinalGather(Array.from({ length: 1025 }, (_, i) => i)));
    const snapshotFrame = renderer.stats.frames;
    const pending = renderer.probeDiffuseGiFinalGather([centerIndex]);
    const concurrent = await rejection(() => renderer!.probeDiffuseGiFinalGather([centerIndex]));
    renderer.render(eye, target);
    const captured = await pending;
    renderer.setDiffuseGiFinalGatherEnabled(false); renderer.render(eye, target);
    const disabled = await rejection(() => renderer!.probeDiffuseGiFinalGather([centerIndex]));
    const errors = [...renderer.errors];
    renderer.dispose();
    return { base, sky, staticAnchor, staticHdr, movedAnchor, movedHdr, cutAnchor,
      resumedAnchor, explicitGiResetAnchor, lightChangedAnchor, identityChangedAnchor,
      probeErrors: { beforeFrame, duplicates, outsideGrid, overBudget, concurrent, disabled }, errors,
      snapshot: { requestedFrame: snapshotFrame, capturedFrame: captured.frame,
        anchorFrame: captured.anchors[0].state[1] },
      afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}

export async function runFinalGatherDisocclusionFixture() {
  const canvas = canvas128();
  const { receiver, blocker } = scene(true);
  let renderer: Renderer | undefined;
  const oldEye: Vec3 = [0, 0, 5], oldTarget: Vec3 = [0, 0, 0];
  try {
    renderer = await create(canvas, [receiver, blocker]);
    for (let frame = 0; frame < 64; frame++) renderer.render(oldEye, oldTarget);
    const before = (await renderer.probeDiffuseGiFinalGather(allIndices)).anchors;
    const oldViewProjection = multiply(perspective(Math.PI / 3.2, 1, 0.1, 150),
      lookAt(oldEye, oldTarget, [0, 1, 0]));
    renderer.render([1.2, 0, 5], [1.2, 0, 0]);
    const after = (await renderer.probeDiffuseGiFinalGather(allIndices)).anchors;
    const newlyExposed = after.flatMap(anchor => {
      if (!anchor.valid || Math.abs(anchor.position[2]) > 0.01) return [];
      const clip = transformPoint(oldViewProjection, anchor.position);
      const x = (clip[0] + 1) * 64, y = (1 - clip[1]) * 64;
      const column = Math.floor((x - 2.5) / 4), row = Math.floor((y - 2.5) / 4);
      if (column < 0 || column + 1 >= 32 || row < 0 || row + 1 >= 32) return [];
      const previous = [before[column + row * 32], before[column + 1 + row * 32],
        before[column + (row + 1) * 32], before[column + 1 + (row + 1) * 32]];
      if (previous.some(sample => !sample.valid || sample.identity[1] !== 2
        || Math.abs(sample.position[2] - 1) > 0.01)) return [];
      return [{ currentIndex: anchor.index, previousIndices: previous.map(sample => sample.index),
        currentHistory: anchor.historyUpdates, previousHistory: Math.min(...previous.map(sample => sample.historyUpdates)),
        currentPosition: anchor.position, previousPositions: previous.map(sample => sample.position) }];
    });
    const carriedReceiver = after.find(anchor => anchor.valid && Math.abs(anchor.position[2]) < 0.01
      && anchor.historyUpdates > 1);
    const errors = [...renderer.errors];
    renderer.dispose();
    return { newlyExposed, carriedReceiver: carriedReceiver && {
      index: carriedReceiver.index, historyUpdates: carriedReceiver.historyUpdates },
    errors, afterDispose: renderer.stats.ownedGpuBytes };
  } finally { renderer?.dispose(); canvas.remove(); }
}
