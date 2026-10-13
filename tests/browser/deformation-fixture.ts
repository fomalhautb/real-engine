import type { InstanceState, SceneBatch } from '../../src/contracts';
import { compileDeformation, evaluateDeformation, type DeformationGraph, type DeformationVertex } from '../../src/deformation';
import { createCube } from '../../src/geometry';
import { identity, multiply, rotationY, rotationZ, transformPoint, translation } from '../../src/math';
import { buildMorphTarget } from '../../src/morph-assets';
import { Renderer, type DeformationVerification } from '../../src/renderer';

/** Runs real compiled WGSL through the production renderer, independent of the demo's four bones. */
export async function runDeformationFixture() {
  const mesh = createCube();
  const source = new DataView(mesh.vertices);
  for (let i = 0; i < mesh.vertexCount; i++) {
    for (let influence = 0; influence < 4; influence++) {
      source.setUint32(i * 80 + 32 + influence * 4, influence === 0 ? 5 : 0, true);
      source.setFloat32(i * 80 + 48 + influence * 4, influence === 0 ? 1 : 0, true);
    }
  }
  const morphTargets = [
    buildMorphTarget(mesh, ([, y, z]) => [0.23 * y + 0.11 * z, 0.04 * z, 0]),
    buildMorphTarget(mesh, ([x, , z]) => [0, 0.16 * x, 0.18 * x + 0.05 * z]),
  ];
  const morphs = [{ kind: 'morph' as const, target: 0, weight: 0 }, { kind: 'morph' as const, target: 1, weight: 1 }];
  const wave = { kind: 'wave' as const, amplitude: 0.35, frequency: 1.4, phase: 2 };
  const graphs: DeformationGraph[] = [
    { jointCount: 6, nodes: [...morphs, { kind: 'skin' }, wave] },
    { jointCount: 6, nodes: [wave, { kind: 'skin' }, ...morphs] },
  ];
  const pose = (entityId: number, time: number): InstanceState => {
    const instance = entityId % 100;
    const normalMatrix = rotationY(time * 0.15 + instance * 0.25);
    const bones = Array.from({ length: 6 }, () => identity());
    bones[5] = multiply(translation(0.3 + time * 0.22, 0.5 - time * 0.08, instance * 0.4), rotationZ(0.55 + time * 0.45 + instance * 0.3));
    return {
      entityId, model: multiply(translation(instance * 2.5 - 1.5 + time * 0.3, 1, 0), normalMatrix), normalMatrix,
      bones, deformScalars: [0.4 + time * 0.35 + instance * 0.2, -0.75 + time * 0.17 - instance * 0.1, time * 1.3 + instance],
      color: [0.3, 0.7, 0.9, 1], metalness: 0.2, roughness: 0.5, historyVersion: 1,
    };
  };
  const batches: SceneBatch[] = graphs.map((deformation, i) => ({
    name: `six-bone-ordered-graph-${i}`, mesh, capacity: 2, inputMode: 'committed', deformation, morphTargets,
    instances: [pose(100 * (i + 1), 0), pose(100 * (i + 1) + 1, 0)],
  }));
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  const renderer = await Renderer.create(canvas, batches);
  const referenceVertex: DeformationVertex = {
    position: [0, 1, 2].map(axis => source.getFloat32(axis * 4, true)) as [number, number, number],
    normal: [0, 1, 2].map(axis => source.getFloat32(16 + axis * 4, true)) as [number, number, number],
    joints: [5, 0, 0, 0], weights: [1, 0, 0, 0],
  };
  const localPoint = (batch: SceneBatch, state: InstanceState) => evaluateDeformation(
    compileDeformation(batch.deformation), referenceVertex, 0, morphTargets, state.bones, state.deformScalars,
  ).position;
  const worldPoint = (batch: SceneBatch, state: InstanceState) => transformPoint(state.model, localPoint(batch, state));
  const distance = (a: readonly number[], b: readonly number[]) => Math.hypot(...a.map((value, axis) => value - b[axis]));
  // This oracle uses independently retained submitted states, never the renderer's
  // packed current/previous arrays. Slot-based history or a broken upload must fail.
  const verifyHistory = (verification: DeformationVerification, currentStates: InstanceState[][], previousStates: InstanceState[][]) => {
    let currentMaxError = 0, historyMaxError = 0;
    const seen = new Set<string>();
    for (const sample of verification.samples) {
      const batchIndex = batches.findIndex(batch => batch.name === sample.batch);
      if (batchIndex < 0) throw new Error('GPU sample has an unknown batch');
      const current = currentStates[batchIndex].find(state => state.entityId === sample.entityId);
      const previous = previousStates[batchIndex].find(state => state.entityId === sample.entityId);
      const key = `${sample.batch}:${sample.entityId}`;
      if (!current || !previous || seen.has(key)) throw new Error('GPU samples have missing or duplicate entity identities');
      seen.add(key);
      const expectedCurrent = worldPoint(batches[batchIndex], current);
      const expectedPrevious = worldPoint(batches[batchIndex], previous);
      if (sample.current.length !== 3 || sample.previous.length !== 3) throw new Error('GPU samples must contain xyz coordinates');
      for (let axis = 0; axis < 3; axis++) {
        currentMaxError = Math.max(currentMaxError, Math.abs(sample.current[axis] - expectedCurrent[axis]));
        historyMaxError = Math.max(historyMaxError, Math.abs(sample.previous[axis] - expectedPrevious[axis]));
      }
    }
    if (seen.size !== currentStates.reduce((sum, states) => sum + states.length, 0)) throw new Error('GPU samples do not cover every instance');
    return { currentMaxError, historyMaxError };
  };
  try {
    const firstStates = batches.map(batch => structuredClone(batch.instances));
    renderer.render([5, 4, 8], [0, 1, 0]);
    const first = await renderer.verifyDeformation({ allInstances: true });
    const firstHistory = verifyHistory(first, firstStates, firstStates);

    for (const batch of batches) batch.instances = batch.instances.map(state => pose(state.entityId, 1));
    const secondStates = batches.map(batch => structuredClone(batch.instances));
    const historyMovement = batches.map((batch, i) => distance(worldPoint(batch, firstStates[i][0]), worldPoint(batch, secondStates[i][0])));
    renderer.render([5, 4, 8], [0, 1, 0]);
    const second = await renderer.verifyDeformation({ allInstances: true });
    const secondHistory = verifyHistory(second, secondStates, firstStates);

    // Reordering must preserve each entity's previous frame despite changing its slot.
    for (const batch of batches) batch.instances = [...batch.instances].reverse().map(state => pose(state.entityId, 2));
    const reorderedStates = batches.map(batch => structuredClone(batch.instances));
    const reorderedHistoryMovement = batches.map((batch, i) => distance(worldPoint(batch, secondStates[i][1]), worldPoint(batch, batch.instances[0])));
    renderer.render([5, 4, 8], [0, 1, 0]);
    const reordered = await renderer.verifyDeformation({ allInstances: true });
    const reorderedHistory = verifyHistory(reordered, reorderedStates, secondStates);
    const historyMaxError = Math.max(firstHistory.historyMaxError, secondHistory.historyMaxError, reorderedHistory.historyMaxError);
    const currentMaxError = Math.max(firstHistory.currentMaxError, secondHistory.currentMaxError, reorderedHistory.currentMaxError);
    const probePose = pose(100, 1);
    const orderDifference = distance(localPoint(batches[0], probePose), localPoint(batches[1], probePose));
    const diagnostics = renderer.diagnostics();
    const frames = renderer.stats.frames;
    const errors = [...renderer.errors];
    renderer.dispose();
    return {
      first, second, reordered, historyMovement, reorderedHistoryMovement, orderDifference,
      firstHistory, secondHistory, reorderedHistory, historyMaxError, currentMaxError,
      diagnostics, frames, errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes,
      expectedVerticesPerVerification: mesh.vertexCount * batches.reduce((sum, batch) => sum + batch.instances.length, 0),
    };
  } finally {
    renderer.dispose(); canvas.remove();
  }
}
