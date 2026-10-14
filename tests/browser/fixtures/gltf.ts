import type { DeformationVertex } from '../../../src/deformation';
import { compileDeformation, evaluateDeformation } from '../../../src/deformation';
import type { SceneBatch } from '../../../src/contracts';
import { loadGltf } from '../../../src/gltf-load';
import { GltfSceneInstance } from '../../../src/gltf-scene';
import { identity, transformPoint, translation } from '../../../src/math';
import { Renderer } from '../../../src/renderer';

const findVertex = (batch: SceneBatch, x: number, y: number, z: number): number => {
  const data = new DataView(batch.mesh.vertices);
  for (let i = 0; i < batch.mesh.vertexCount; i++) {
    if (data.getFloat32(i * 80, true) === x && data.getFloat32(i * 80 + 4, true) === y && data.getFloat32(i * 80 + 8, true) === z) return i;
  }
  throw new Error(`fixture source vertex [${x}, ${y}, ${z}] missing`);
};

function sourceVertex(batch: SceneBatch, index: number): DeformationVertex {
  const data = new DataView(batch.mesh.vertices), offset = index * 80;
  return {
    position: [data.getFloat32(offset, true), data.getFloat32(offset + 4, true), data.getFloat32(offset + 8, true)],
    normal: [data.getFloat32(offset + 16, true), data.getFloat32(offset + 20, true), data.getFloat32(offset + 24, true)],
    joints: [0, 1, 2, 3].map(i => data.getUint32(offset + 32 + i * 4, true)) as [number, number, number, number],
    weights: [0, 1, 2, 3].map(i => data.getFloat32(offset + 48 + i * 4, true)) as [number, number, number, number],
  };
}

/** GPU regression using the pinned Khronos glTF assets and the production renderer. */
export async function runGltfFixture() {
  const [skinAsset, morphAsset] = await Promise.all([
    loadGltf('/assets/gltf/SimpleSkin/glTF/SimpleSkin.gltf'),
    loadGltf('/assets/gltf/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf'),
  ]);
  const firstSkin = new GltfSceneInstance(skinAsset, { name: 'skin-left', firstEntityId: 701, transform: translation(-4, 0, 0) });
  const secondSkin = new GltfSceneInstance(skinAsset, { name: 'skin-right', firstEntityId: 702, transform: translation(4, 0, 0) });
  const morph = new GltfSceneInstance(morphAsset, { name: 'morph', firstEntityId: 703, transform: translation(0, 0, -3) });
  firstSkin.sample(0);
  secondSkin.sample(1);
  morph.sample(0);

  const topVertex = findVertex(firstSkin.batches[0], 0.5, 2, 0);
  const blendVertex = findVertex(firstSkin.batches[0], 0.5, 1, 0);
  const top = sourceVertex(firstSkin.batches[0], topVertex);
  if (top.joints[1] !== 1 || top.weights[1] !== 1) throw new Error('SimpleSkin top vertex lost its source joint');
  const blended = sourceVertex(firstSkin.batches[0], blendVertex);
  if (blended.weights[0] !== 0.5 || blended.weights[1] !== 0.5) throw new Error('SimpleSkin blended vertex lost its source weights');

  const morphBatch = morph.batches[0];
  morph.sample(2);
  const animatedWeights = [...morphBatch.instances[0].deformScalars];
  morph.sample(0);
  const bindWeights = [...morphBatch.instances[0].deformScalars];
  const morphPosition = (index: number, weights: readonly number[]): number[] => {
    const source = sourceVertex(morphBatch, index).position;
    const local = source.map((value, axis) => value + morphBatch.morphTargets.reduce(
      (sum, target, i) => sum + target.positions[index * 3 + axis] * weights[i], 0,
    ));
    return transformPoint(morphBatch.instances[0].model, local);
  };
  let morphVertex = 0, greatestMotion = -1;
  for (let i = 0; i < morphBatch.mesh.vertexCount; i++) {
    const before = morphPosition(i, bindWeights), after = morphPosition(i, animatedWeights);
    const motion = Math.hypot(...before.map((value, axis) => after[axis] - value));
    if (motion > greatestMotion) { morphVertex = i; greatestMotion = motion; }
  }
  if (greatestMotion < 0.1) throw new Error('AnimatedMorphCube sample did not move a source vertex');
  const morphFirst = morphPosition(morphVertex, bindWeights), morphSecond = morphPosition(morphVertex, animatedWeights);

  // Both nonuniform/sheared matrices are invertible. The selected source vertex
  // blends them equally, so its exact normal direction has a simple independent oracle.
  const affineBones = [
    new Float32Array([2, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1]),
    new Float32Array([1, 0, -0.5, 0, 0, 2, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1]),
  ];
  const affineBatch: SceneBatch = {
    ...firstSkin.batches[0], name: 'affine-skin', capacity: 1,
    instances: [{ ...firstSkin.batches[0].instances[0], entityId: 704, model: translation(0, 0, -5),
      normalMatrix: identity(), bones: affineBones, deformScalars: [] }],
  };
  const graph = compileDeformation(affineBatch.deformation);
  const affineCpuNormal = evaluateDeformation(graph, blended, blendVertex, [], affineBones, []).normal;
  const sign = Math.sign(blended.normal[2]);
  if (Math.abs(blended.normal[2]) < 0.99) throw new Error('SimpleSkin generated source normal is not planar');
  const affineExpectedNormal = [sign * -1 / Math.sqrt(37), 0, sign * 6 / Math.sqrt(37)];

  const batches = [...firstSkin.batches, ...secondSkin.batches, ...morph.batches, affineBatch];
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;left:0;top:0;width:256px;height:256px;';
  document.body.appendChild(canvas);
  let renderer: Renderer | undefined;
  try {
    renderer = await Renderer.create(canvas, batches);
    const expectedVertices = batches.reduce((sum, batch) => sum + batch.mesh.vertexCount, 0);
    const sampleVertices = [topVertex, blendVertex, morphVertex];
    renderer.render([5, 4, 10], [0, 1, 0]);
    const first = await renderer.verifyDeformation({ allInstances: true, sampleVertices });

    firstSkin.sample(1);
    secondSkin.sample(0);
    morph.sample(2);
    renderer.render([5, 4, 10], [0, 1, 0]);
    const second = await renderer.verifyDeformation({ allInstances: true, sampleVertices });
    const errors = [...renderer.errors];
    const frames = renderer.stats.frames;
    renderer.dispose();
    return {
      first, second, expectedVertices, topVertex, blendVertex, morphVertex,
      names: { firstSkin: firstSkin.batches[0].name, secondSkin: secondSkin.batches[0].name,
        morph: morphBatch.name, affine: affineBatch.name },
      morphFirst, morphSecond, greatestMotion, affineCpuNormal, affineExpectedNormal,
      frames, errors, ownedGpuBytesAfterDispose: renderer.stats.ownedGpuBytes,
    };
  } finally { renderer?.dispose(); canvas.remove(); }
}
