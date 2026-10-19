import type { Mat4 } from './math';
import type { MeshData } from './geometry';
import type { DeformationGraph, MorphTarget } from './deformation';
import type { MaterialDefinition } from './material';
import type { MaterialTextureAsset } from './texture-storage';
import type { GeometryHierarchy } from './geometry-hierarchy-types';
import type { GeometryStreamSource } from './gpu-geometry-stream';
import type { GeometryQueryProxy } from './geometry-query-proxy';

export interface MaterialBinding {
  definition: MaterialDefinition;
  /** Runtime vec4 parameters, packed in slot order. Updating them never compiles a pipeline. */
  parameters?: Float32Array;
  /** Immutable image assets; slots referenced by the definition must be present. */
  textures?: readonly (MaterialTextureAsset | undefined)[];
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
  /** Immutable opt-in for GI area importance sampling. Samples the shared
   * deformed triangles and evaluates emission/coverage through world queries.
   * This is a sampling hint, not a separate light or material representation. */
  emissiveSampling?: 'triangle-area';
  /** Optional fully resident LOD for a static primitive. mesh must be hierarchy.source.
   * Opaque materials use GPU selection; Masked keeps the full-detail path. */
  geometryLod?: GeometryHierarchy;
  /** Paged static Opaque surface geometry. mesh remains the exact full-detail
   * shadow/query and Masked compatibility representation; not a collision LOD.
   * Mutually exclusive with geometryLod; dynamic graphs use the full path. */
  geometryStream?: GeometryStreamSource;
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

/** Source-free streamed surface with an explicit, fixed shadow/query proxy.
 * The proxy shares instance/material state and never follows camera LOD. */
export interface StreamedSceneBatch extends Omit<SceneBatch, 'mesh' | 'geometryLod' | 'geometryStream'> {
  mesh?: never;
  geometryLod?: never;
  geometryStream: GeometryStreamSource & { queryProxy: GeometryQueryProxy };
}

export type RenderBatch = SceneBatch | StreamedSceneBatch;

/** Resident representation for shared deformation, shadows and world queries.
 * A paged main surface reads its own physical page pool instead. */
export function residentMesh(scene: RenderBatch): MeshData {
  if (scene.mesh) return scene.mesh;
  const proxy = scene.geometryStream?.queryProxy;
  if (!proxy) throw new Error('GEOMETRY_STREAM_QUERY_PROXY_REQUIRED');
  return proxy.mesh;
}
