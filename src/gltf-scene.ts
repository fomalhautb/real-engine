import type { GltfAsset } from './gltf-import';
import type { InstanceState, SceneBatch } from './contracts';
import { compileDeformation, validateDeformationState, validateDeformationVertex } from './deformation';
import { identity, inverseAffine, multiply, normalMatrix, type Mat4 } from './math';

export interface GltfSceneOptions {
  firstEntityId?: number;
  allocateEntityId?: () => number;
  transform?: Mat4;
  scene?: number;
  name?: string;
}

/** Converts asset poses into the same committed render input used by other producers.
 * The renderer alone owns previous-frame history. Sampling never overwrites it. */
export class GltfSceneInstance {
  readonly batches: SceneBatch[] = [];
  private readonly bindings: { node: number; skin?: number; entityId: number; program: ReturnType<typeof compileDeformation> }[] = [];
  private readonly transform: Mat4;
  private readonly staticPose: boolean;
  private historyVersion = 0;
  private hasCommittedPose = false;

  constructor(private readonly asset: GltfAsset, options: GltfSceneOptions = {}) {
    const sceneIndex = options.scene ?? asset.scene;
    if (!Number.isInteger(sceneIndex) || sceneIndex < 0 || sceneIndex >= asset.scenes.length) throw new Error('GLTF_SCENE_INDEX');
    this.transform = new Float32Array(options.transform ?? identity());
    inverseAffine(this.transform);
    const roots = new Set(asset.scenes[sceneIndex].roots), selected = new Set<number>();
    for (let i = 0; i < asset.nodes.length; i++) {
      let cursor: number | null = i;
      while (cursor !== null) {
        if (roots.has(cursor)) { selected.add(i); break; }
        cursor = asset.nodes[cursor].parent;
      }
    }
    let nextId = options.firstEntityId ?? 1;
    const ids = new Set<number>();
    for (const nodeIndex of selected) {
      const node = asset.nodes[nodeIndex];
      if (node.mesh === undefined) continue;
      const skin = node.skin === undefined ? undefined : asset.skins[node.skin];
      for (const [primitiveIndex, primitive] of asset.meshes[node.mesh].primitives.entries()) {
        if (skin && !primitive.hasSkin) throw new Error('GLTF_SKIN_ATTRIBUTES_MISSING');
        const graph: SceneBatch['deformation'] = { jointCount: skin?.joints.length ?? 0,
          nodes: [...primitive.morphTargets.map((_, i) => ({ kind: 'morph' as const, target: i, weight: i })), ...(skin ? [{ kind: 'skin' as const, normalMode: 'affine' as const }] : [])] };
        const program = compileDeformation(graph), data = new DataView(primitive.mesh.vertices);
        for (let i = 0; i < primitive.mesh.vertexCount; i++) {
          const read = (offset: number, count: number) => Array.from({ length: count }, (_, k) => data.getFloat32(i * 80 + offset + k * 4, true));
          validateDeformationVertex(program, { position: read(0, 3) as [number, number, number], normal: read(16, 3) as [number, number, number],
            weights: read(48, 4) as [number, number, number, number], joints: Array.from({ length: 4 }, (_, k) => data.getUint32(i * 80 + 32 + k * 4, true)) as [number, number, number, number] });
        }
        const entityId = options.allocateEntityId ? options.allocateEntityId() : nextId++;
        if (!Number.isSafeInteger(entityId) || entityId < 1 || ids.has(entityId)) throw new Error('GLTF_ENTITY_ID');
        ids.add(entityId);
        this.bindings.push({ node: nodeIndex, skin: node.skin, entityId, program });
        this.batches.push({ name: `${options.name ?? 'gltf'}:${nodeIndex}:${primitiveIndex}`, mesh: primitive.mesh,
          material: primitive.material, morphTargets: primitive.morphTargets, deformation: graph,
          capacity: 1, inputMode: 'committed', instances: [] });
      }
    }
    // AnimationRig.sample defaults an omitted clip index to clip 0. Only assets
    // without clips have a pose that is independent of time and loop mode.
    this.staticPose = asset.rig.clipCount === 0;
    this.sample(0);
  }

  sample(time: number, clipIndex?: number, loop = false, invalidateHistory = false): void {
    if (this.staticPose && this.hasCommittedPose && !invalidateHistory) {
      // Match AnimationRig.sample validation even when the committed state is reused.
      if (!Number.isFinite(time)) throw new RangeError('sample time must be finite');
      if (clipIndex !== undefined && (!Number.isInteger(clipIndex) || clipIndex !== 0)) {
        throw new RangeError('invalid animation clip index');
      }
      return;
    }
    const pose = this.asset.rig.sample(time, clipIndex, loop);
    const version = this.historyVersion + Number(invalidateHistory);
    // Validate all new inputs before publishing any batch, so bad poses cannot leave a partial frame.
    const states = this.bindings.map((binding): InstanceState => {
      const meshWorld = pose.world[binding.node], model = multiply(this.transform, meshWorld);
      const determinant = model[0] * (model[5] * model[10] - model[9] * model[6]) - model[4] * (model[1] * model[10] - model[9] * model[2]) + model[8] * (model[1] * model[6] - model[5] * model[2]);
      if (determinant <= 0) throw new Error('GLTF_UNSUPPORTED_REFLECTED_OR_SINGULAR_MESH_TRANSFORM');
      const skin = binding.skin === undefined ? undefined : this.asset.skins[binding.skin];
      const inverseMesh = skin ? inverseAffine(meshWorld) : undefined;
      const bones = skin ? skin.joints.map((joint, i) => multiply(inverseMesh!, multiply(pose.world[joint], skin.inverseBindMatrices[i]))) : [];
      const scalars = pose.weights[binding.node];
      try { validateDeformationState(binding.program, bones, scalars); }
      catch (error) { throw new Error('GLTF_UNSUPPORTED_SKIN_POSE: ' + String(error)); }
      return { entityId: binding.entityId, model, normalMatrix: normalMatrix(model), bones, deformScalars: [...scalars],
        color: [1, 1, 1, 1], metalness: 1, roughness: 1, historyVersion: version };
    });
    this.historyVersion = version;
    for (let i = 0; i < states.length; i++) this.batches[i].instances = [states[i]];
    this.hasCommittedPose = true;
  }
}
