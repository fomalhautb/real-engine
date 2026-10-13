import type { Mat4 } from './math';
import type { MeshData } from './geometry';

/** Stage 1.1 capabilities: four rigid joints, one position morph, opaque metal/rough. */
export interface InstanceState {
  entityId: number;
  model: Mat4;
  normalMatrix: Mat4;
  bones: Mat4[];
  morph: number;
  color: [number, number, number, number];
  metalness: number;
  roughness: number;
  /** Increment on teleport or identity-changing edits to reject temporal history. */
  historyVersion: number;
}
export interface SceneBatch {
  name: string;
  mesh: MeshData;
  capacity: number;
  instances: InstanceState[];
}
