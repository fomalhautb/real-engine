import type { Mat4 } from './math';
import type { MeshData } from './geometry';
import type { DeformationGraph, MorphTarget } from './deformation';
import type { MaterialDefinition } from './material';
import type { TextureAsset } from './texture-assets';

export interface MaterialBinding {
  definition: MaterialDefinition;
  /** Runtime vec4 parameters, packed in slot order. Updating them never compiles a pipeline. */
  parameters?: Float32Array;
  /** Immutable image assets; slots referenced by the definition must be present. */
  textures?: readonly (TextureAsset | undefined)[];
  /** Per-texture sampling, independent of immutable image identity. */
  samplers?: readonly (Omit<GPUSamplerDescriptor, 'compare'> | undefined)[];
}

/** Render input independent of the gameplay category or pose producer. */
export interface InstanceState {
  /** Unsigned 32-bit identity, unique among the active instances of a batch. */
  entityId: number;
  /** Identity generation; omitted means zero. Reusing an ID with a new generation rejects previous-frame history. */
  entityGeneration?: number;
  model: Mat4;
  normalMatrix: Mat4;
  bones: Mat4[];
  deformScalars: number[];
  color: [number, number, number, number];
  metalness: number;
  roughness: number;
  /** Increment on teleport or identity-changing edits to reject temporal history. */
  historyVersion: number;
}
export interface SceneBatch {
  material?: MaterialBinding;
  /** Stable asset/primitive identity for optional world queries; bump on topology replacement. */
  queryIdentity?: { assetId: string; primitiveId: number; topologyVersion: number };
  /** Committed inputs replace instance objects on every change; mutable inputs
   * are re-evaluated on every render. Assets are immutable after GPU upload. */
  inputMode?: 'committed' | 'mutable';
  name: string;
  mesh: MeshData;
  capacity: number;
  instances: InstanceState[];
  deformation: DeformationGraph;
  morphTargets: MorphTarget[];
}
