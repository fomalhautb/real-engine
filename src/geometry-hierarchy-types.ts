import type { MeshData } from './geometry';
import type { Bounds3, DeformationGraph } from './deformation';

/** Original attributes are retained; cluster indices address this owned source mesh. */
export interface GeometryCluster {
  level: number;
  vertices: Uint32Array;
  triangles: Uint8Array;
  /** Present only on LOD0, in local triangle order. */
  sourceTriangles?: Uint32Array;
  bounds: Bounds3;
  /** Inherited union bounds used for monotonic LOD decisions, not tight culling. */
  lodBounds: Bounds3;
  /** Accumulated approximate simplification metric, in source mesh units. */
  error: number;
  /** Group whose detailed geometry this cluster helps replace; null on LOD0. */
  refinedGroup: number | null;
  /** Unique group containing this cluster as a child, including terminal groups. */
  group: number;
}

export interface GeometryGroup {
  level: number;
  children: number[];
  parents: number[];
  bounds: Bounds3;
  localError: number;
  error: number;
  /** Terminal groups retain their children as roots and have no coarser replacement. */
  terminal: boolean;
}

export interface GeometryBuildOptions {
  maxClusterVertices?: number;
  maxClusterTriangles?: number;
  groupSize?: number;
  simplifyRatio?: number;
  /** Absolute approximate metric allowed for one simplification step. */
  maxSimplificationError?: number;
  maxLevels?: number;
  maxVertices?: number;
  maxTriangles?: number;
  maxOutputTriangles?: number;
  maxClusters?: number;
  maxGroups?: number;
}

export interface GeometryBuildInput {
  primitiveKey: string;
  mesh: MeshData;
  coverage: 'opaque' | 'mask';
  /** Nonempty deformation graphs currently require the existing full-detail path. */
  deformation?: DeformationGraph;
}

/** Versioned build output; regrouping coarse clusters can form a DAG. */
export interface GeometryHierarchy {
  version: 1;
  builder: 'meshoptimizer-1.3.0';
  primitiveKey: string;
  source: MeshData;
  options: Required<GeometryBuildOptions>;
  clusters: GeometryCluster[];
  groups: GeometryGroup[];
  /** Cluster IDs in all terminal groups. All roots are required for full coverage. */
  roots: number[];
  sourceTriangleCount: number;
}
