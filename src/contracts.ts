import type { Mat4 } from './math';
import type { MeshData } from './geometry';
import type { DeformationGraph, MorphTarget } from './deformation';

/** Render input independent of the gameplay category or pose producer. */
export interface InstanceState {
  entityId: number;
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
